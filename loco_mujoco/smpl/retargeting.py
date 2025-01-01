import os
import glob
from dataclasses import replace
import logging

import yaml
import joblib
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import jax.numpy as jnp
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import torch
from torch.autograd import Variable

import loco_mujoco
from loco_mujoco.smpl import SMPLH_Parser, SMPLH_BONE_ORDER_NAMES
from loco_mujoco.smpl.torch_fk_humanoid import ForwardKinematicsHumanoidTorch
from loco_mujoco.smpl.utils import torch_utils
from loco_mujoco.smpl.utils.smoothing import gaussian_filter_1d_batch
from loco_mujoco.environments import LocoEnv
from loco_mujoco.core.utils.math import quat_scalarlast2scalarfirst
from loco_mujoco.trajectory import (
    Trajectory,
    TrajectoryInfo,
    TrajectoryModel,
    TrajectoryData,
    interpolate_trajectories,
)
from loco_mujoco.datasets.data_generation import ExtendTrajData
from loco_mujoco import PATH_TO_SMPL_ROBOT_CONF
from loco_mujoco.utils import setup_logger


OPTIMIZED_SHAPE_FILE_NAME = "shape_optimized.pkl"


def fit_smpl_shape(robot_conf: DictConfig,
                   path_to_smpl_model: str,
                   save_path_new_smpl_shape: str,
                   logger: logging.Logger) -> None:

    humanoid_fk = ForwardKinematicsHumanoidTorch(robot_conf) # load forward kinematics model

    #### Define corresonpdances between h1 and smpl joints
    robot_joint_names_augment = humanoid_fk.joint_names_augment
    robot_joint_pick = [i[0] for i in robot_conf.joint_matches]
    smpl_joint_pick = [i[1] for i in robot_conf.joint_matches]

    robot_joint_pick_idx = [ robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPLH_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    #### Preparing fitting varialbes
    device = torch.device("cpu")
    pose_aa_robot = np.repeat \
        (np.repeat(sRot.identity().as_rotvec()[None, None, None, ], humanoid_fk.num_extend_dof , axis = 2), 1, axis = 1)
    pose_aa_robot = torch.from_numpy(pose_aa_robot).float()

    ###### prepare SMPL default pose for H1
    pose_aa_stand = np.zeros((1, 156))
    pose_aa_stand = pose_aa_stand.reshape(-1, 52, 3)

    for modifiers in robot_conf.smpl_pose_modifier:
        modifier_key = list(modifiers.keys())[0]
        modifier_value = list(modifiers.values())[0]
        pose_aa_stand[:, SMPLH_BONE_ORDER_NAMES.index(modifier_key)] = sRot.from_euler("xyz", eval(modifier_value),  degrees = False).as_rotvec()

    pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 156)).requires_grad_(False)
    smpl_parser_n = SMPLH_Parser(model_path=path_to_smpl_model, gender="neutral")

    ###### Shape fitting
    trans = torch.zeros([1, 3]).requires_grad_(False)
    beta = torch.zeros([1, 16]).requires_grad_(False)
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta, trans)
    offset = joints[:, 0] - trans
    root_trans_offset = trans + offset

    fk_return = humanoid_fk.fk_batch(pose_aa_robot[None, ], root_trans_offset[None, 0:1])

    shape_new = Variable(torch.zeros([1, 16]).to(device), requires_grad=True)
    scale = Variable(torch.ones([1]).to(device), requires_grad=True)
    optimizer_shape = torch.optim.Adam([shape_new, scale] ,lr=0.1)

    pbar = tqdm(range(1000))
    for iteration in pbar:
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])
        root_pos = joints[:, 0]
        joints = (joints - joints[:, 0]) * scale + root_pos
        if len(robot_conf.extend_config) > 0:
            diff = fk_return.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
        else:
            diff = fk_return.global_translation[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]

        loss_g = diff.norm(dim=-1).mean()
        loss = loss_g
        pbar.set_description_str(f"{iteration} - Loss: {loss.item() * 1000}")

        optimizer_shape.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_shape.step()

    joblib.dump((shape_new.detach(), scale), save_path_new_smpl_shape)
    logger.info(f"Shape parameters saved at {save_path_new_smpl_shape}" )


def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not 'mocap_framerate' in entry_data:
        return
    framerate = entry_data['mocap_framerate']

    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate
    }


def fit_smpl_motion(robot_conf: DictConfig,
                    path_to_smpl_model: str,
                    path_to_amass_datasets: str,
                    path_to_converted_amass_datasets: str,
                    motion_file_name: str,
                    logger: logging.Logger) -> Trajectory:

    device = torch.device("cpu")

    # path to loco-mujoco robot models
    path_to_all_amass_files = path_to_amass_datasets + "/**/*.npz"

    humanoid_fk = ForwardKinematicsHumanoidTorch(robot_conf)  # load forward kinematics model
    num_augment_joint = len(robot_conf.extend_config)

    #### Define corresonpdances between h1 and smpl joints
    robot_joint_names_augment = humanoid_fk.joint_names_augment
    robot_joint_pick = [i[0] for i in robot_conf.joint_matches]
    smpl_joint_pick = [i[1] for i in robot_conf.joint_matches]
    robot_joint_pick_idx = [robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPLH_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]
    all_pkls = glob.glob(str(path_to_all_amass_files), recursive=True)
    key_names = ["/".join(data_path.split("/")[-3:]).replace(".npz", "") for data_path in all_pkls]

    smpl_parser_n = SMPLH_Parser(model_path=path_to_smpl_model, gender="neutral")

    amass_data = load_amass_data(all_pkls[key_names.index(motion_file_name)])
    skip = int(amass_data['fps'] // 30)
    trans = torch.from_numpy(amass_data['trans'][::skip])
    N = trans.shape[0]
    pose_aa_walk = torch.from_numpy(amass_data['pose_aa'][::skip]).float()
    pose_aa_walk = torch.cat([pose_aa_walk, torch.zeros((pose_aa_walk.shape[0], 156 - pose_aa_walk.shape[1]))],
                             axis=-1)  # Setting the hand pose to zero.
    shape_new, scale = joblib.load(path_to_converted_amass_datasets + f"/{robot_conf.name}/shape_optimized.pkl")

    with torch.no_grad():
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk.reshape(N, -1, 3), shape_new.repeat(N, 1), trans)
        root_pos = joints[:, 0:1]
        joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos

    offset = joints[:, 0] - trans
    root_trans_offset = (trans + offset).clone()

    if root_trans_offset[..., 2].min() < 0.6:
        print("too low!!!!")

    gt_root_rot_quat = torch.from_numpy((sRot.from_rotvec(pose_aa_walk[:, :3]) * sRot.from_quat(
        [0.5, 0.5, 0.5, 0.5]).inv()).as_quat()).float()  # can't directly use this
    gt_root_rot = torch.from_numpy(
        sRot.from_quat(torch_utils.calc_heading_quat(gt_root_rot_quat)).as_rotvec()).float()  # so only use the heading.

    # def dof_to_pose_aa(dof_pos):
    dof_pos = torch.zeros((1, N, humanoid_fk.num_dof, 1))

    dof_pos_new = Variable(dof_pos.clone(), requires_grad=True)
    root_rot_new = Variable(gt_root_rot.clone(), requires_grad=True)
    root_pos_offset = Variable(torch.zeros(1, 3), requires_grad=True)
    optimizer_pose = torch.optim.Adadelta([dof_pos_new], lr=100)
    optimizer_root = torch.optim.Adam([root_rot_new, root_pos_offset], lr=0.01)

    kernel_size = 5  # Size of the Gaussian kernel
    sigma = 0.75  # Standard deviation of the Gaussian kernel

    pbar = tqdm(range(500))
    for iteration in pbar:
        pose_aa_h1_new = torch.cat([root_rot_new[None, :, None], humanoid_fk.dof_axis * dof_pos_new,
                                    torch.zeros((1, N, num_augment_joint, 3)).to(device)], axis=2)
        fk_return = humanoid_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None,] + root_pos_offset)

        if num_augment_joint > 0:
            diff = fk_return.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
        else:
            diff = fk_return.global_translation[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]

        loss_g = diff.norm(dim=-1).mean()
        loss = loss_g

        optimizer_pose.zero_grad()
        optimizer_root.zero_grad()
        loss.backward()
        optimizer_pose.step()
        optimizer_root.step()

        dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])

        pbar.set_description_str(f"Iter: {iteration} \t {loss.item() * 1000:.3f}")
        dof_pos_new.data = \
        gaussian_filter_1d_batch(dof_pos_new.squeeze().transpose(1, 0)[None,], kernel_size, sigma).transpose(2, 1)[
            ..., None]

    dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])

    height_diff = fk_return.global_translation[..., 2].min().item() - 0.05
    root_trans_offset_dump = (root_trans_offset + root_pos_offset).clone()
    root_trans_offset_dump[..., 2] -= height_diff

    dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])

    height_diff = fk_return.global_translation[..., 2].min().item() - 0.05
    root_trans_offset_dump = (root_trans_offset + root_pos_offset).clone()

    root_trans_offset = root_trans_offset_dump.squeeze().detach().numpy()
    root_trans_offset_dump[..., 2] -= height_diff
    root_rot_rotvec = root_rot_new.detach().numpy()
    root_rot_quat = sRot.from_rotvec(root_rot_rotvec).as_quat()

    # positions
    free_joint_pos = root_trans_offset
    free_joint_quat = quat_scalarlast2scalarfirst(root_rot_quat)
    joints = dof_pos_new.squeeze().detach().numpy()

    # calculate velocities using finite difference, remove first and last frame
    fps = 30
    free_joint_vel = (free_joint_pos[2:] - free_joint_pos[:-2]) / (2 * (1 / fps))
    free_joint_vel_rot = (root_rot_rotvec[2:] - root_rot_rotvec[:-2]) / (2 * (1 / fps))
    joints_vel = (joints[2:] - joints[:-2]) / (2 * (1 / fps))

    # remove first and last frame
    free_joint_pos = free_joint_pos[1:-1]
    free_joint_quat = free_joint_quat[1:-1]
    joints = joints[1:-1]

    # get qpos and qvel
    qpos = np.concatenate([free_joint_pos, free_joint_quat, joints], axis=1)
    qvel = np.concatenate([free_joint_vel, free_joint_vel_rot, joints_vel], axis=1)

    # store them in trajectory
    # create a trajectory info -- this stores basic information about the trajectory
    robot_cls = LocoEnv.registered_envs[robot_conf.name]
    spec = mujoco.MjSpec.from_file(robot_cls.get_default_xml_file_path())
    model = spec.compile()

    njnt = model.njnt
    jnt_type = model.jnt_type
    jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]
    traj_info = TrajectoryInfo(jnt_names, model=TrajectoryModel(njnt, jnp.array(jnt_type)), frequency=fps)

    # create a trajectory data -- this stores the actual trajectory data
    traj_data = TrajectoryData(jnp.array(qpos), jnp.array(qvel), split_points=jnp.array([0, len(qpos)]))

    # combine them to a trajectory
    return Trajectory(traj_info, traj_data)


def extend_motion(robot_conf: DictConfig,
                  traj: Trajectory,
                  logger: logging.Logger):

    # get environment class
    env_cls = LocoEnv.registered_envs[robot_conf.name]

    # create environment instance and run it on the trajectories frequency
    env = env_cls(**robot_conf.env_params, random_start=False, fixed_start_conf=(0, 0))

    # interpolate trajectory to environment frequency
    traj_data, traj_info = interpolate_trajectories(traj.data, traj.info, 1.0 / env.dt)
    traj = Trajectory(info=traj_info, data=traj_data)

    # load_trajectory
    env.load_trajectory(traj, warn=False)
    traj_data, traj_info = env.th.traj.data, env.th.traj.info

    callback = ExtendTrajData(env, model=env._model, n_samples=traj_data.n_samples)
    env.play_trajectory(n_episodes=env.th.n_trajectories,
                        render=False, callback_class=callback)
    traj_data, traj_info = callback.extend_trajectory_data(traj_data, traj_info)
    traj = replace(traj, data=traj_data, info=traj_info)

    return traj


def load_retargeted_amass_trajectory(env_name, dataset_name):

    logger = setup_logger("amass", identifier="[LocoMuJoCo's AMASS Retargeting Pipeline]")

    if "Mjx" in env_name:
        env_name.repalce("Mjx", "")

    # setup paths
    path_to_conf = loco_mujoco.PATH_TO_SMPL_CONF

    # read paths from amass conf file
    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        path_to_smpl_model = data["LOCOMUJOCO_SMPL_MODEL_PATH"]
        path_to_amass_datasets = data["LOCOMUJOCO_AMASS_PATH"]
        path_to_converted_amass_datasets = data["LOCOMUJOCO_CONVERTED_AMASS_PATH"]

    assert path_to_smpl_model is not None, "Please set the environment variable LOCOMUJOCO_SMPL_MODEL_PATH."
    assert path_to_amass_datasets is not None, "Please set the environment variable LOCOMUJOCO_AMASS_PATH."
    assert path_to_converted_amass_datasets is not None, ("Please set the environment variable "
                                                          "LOCOMUJOCO_CONVERTED_AMASS_PATH.")

    # load robot conf
    filename = f"{env_name}.yaml"
    filepath = os.path.join(PATH_TO_SMPL_ROBOT_CONF, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"YAML file '{filename}' not found in path: {PATH_TO_SMPL_ROBOT_CONF}")
    robot_conf = OmegaConf.load(filepath)

    # create robot data dir if it does not exist
    path_robot_smpl_data = path_to_converted_amass_datasets + f"/{robot_conf.name}"
    if not os.path.exists(path_robot_smpl_data):
        os.makedirs(path_robot_smpl_data, exist_ok=True)

    # check if smpl shape file for robot already exists, if not fit a new one
    path_to_robot_smpl_shape = path_robot_smpl_data + "/" + OPTIMIZED_SHAPE_FILE_NAME
    if not os.path.exists(path_to_robot_smpl_shape):
        logger.info("Robot shape file not found, fitting new one ...")
        fit_smpl_shape(robot_conf, path_to_smpl_model, path_to_robot_smpl_shape, logger)
    else:
        logger.info(f"Found existing robot shape file at {path_to_robot_smpl_shape}")

    # check if the retargeted motion file already exists, if not fit new one
    path_retargeted_motion_file = path_robot_smpl_data + f"/{dataset_name}.npz"
    if not os.path.exists(path_retargeted_motion_file):
        # create a trajectory containing only qpos and qvel, from the amass dataset
        logger.info("Retargeting AMASS motion file using optimized body shape ...")
        trajectory = fit_smpl_motion(robot_conf, path_to_smpl_model, path_to_amass_datasets,
                                     path_to_converted_amass_datasets, dataset_name, logger)
        logger.info("Using Mujoco's forward kinematics to calculate other model-specific entities ...")
        trajectory = extend_motion(robot_conf, trajectory, logger)
        # save trajectory
        trajectory.save(path_retargeted_motion_file)
    else:
        logger.info(f"Found existing retargeted motion file at {path_retargeted_motion_file}.Loading ...")
        # load trajectory
        trajectory = Trajectory.load(path_retargeted_motion_file)

    logger.info("Trajectory data loaded!")

    return trajectory


if __name__ == "__main__":
    load_retargeted_amass_trajectory("UnitreeH1", "KIT/674/dry_head04_poses")
