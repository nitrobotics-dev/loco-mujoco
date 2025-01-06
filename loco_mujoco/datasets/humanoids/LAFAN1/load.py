import os
import glob
from pathlib import Path
from dataclasses import replace
import logging
from typing import Union, List
import hashlib

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


def extend_motion(
        env_name: str,
        robot_conf: DictConfig,
        traj: Trajectory,
) -> Trajectory:
    """
    Extend a motion trajectory to include more model-specific entities
    like body xpos, site positions, etc. and to match the environment's frequency.

    Args:
        env_name (str): Name of the environment.
        robot_conf (DictConfig): Configuration of the robot.
        traj (Trajectory): The original trajectory data.

    Returns:
        Trajectory: The extended trajectory.

    """
    env_cls = LocoEnv.registered_envs[env_name]
    env = env_cls(**robot_conf.env_params, th_params=dict(random_start=False, fixed_start_conf=(0, 0)))

    traj_data, traj_info = interpolate_trajectories(traj.data, traj.info, 1.0 / env.dt)
    traj = Trajectory(info=traj_info, data=traj_data)

    env.load_trajectory(traj, warn=False)
    traj_data, traj_info = env.th.traj.data, env.th.traj.info

    callback = ExtendTrajData(env, model=env._model, n_samples=traj_data.n_samples)
    env.play_trajectory(
        n_episodes=env.th.n_trajectories,
        render=False,
        callback_class=callback
    )
    traj_data, traj_info = callback.extend_trajectory_data(traj_data, traj_info)
    traj = replace(traj, data=traj_data, info=traj_info)

    return traj


def load_lafan1_trajectory(
    env_name: str, dataset_name: Union[str, List[str]]
) -> Trajectory:
    """
    Load a trajectory from the LAFAN1 dataset.

    Args:
        env_name (str): The name of the environment.
        dataset_name (Union[str, List[str]]): The name of the dataset(s) to load.

    Returns:
        Trajectory: The loaded trajectory.

    """
    logger = setup_logger("lafan1", identifier="[LocoMuJoCo's LAFAN1 Retargeting Pipeline]")

    if "Mjx" in env_name:
        env_name = env_name.replace("Mjx", "")

    path_to_conf = loco_mujoco.PATH_TO_VARIABLES

    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        path_to_lafan1_datasets = data["LOCOMUJOCO_LAFAN1_PATH"]

    assert path_to_lafan1_datasets, ("Please use the command 'loco-mujoco-set-lafan1-path' to set the "
                                     "path to the LAFAN1 datasets.")

    assert env_name in ["UnitreeH1", "UnitreeG1"], ("Only UnitreeH1 and UnitreeG1 environments are "
                                                    "supported for the LAFAN1 dataset.")

    # load robot_conf
    robot_conf_path = os.path.join(Path(__file__).resolve().parent / "conf.yaml")
    all_robot_confs = OmegaConf.load(robot_conf_path)
    robot_conf = all_robot_confs[env_name]

    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    all_trajectories = []
    for d_name in dataset_name:

        # load the csv file
        d_name = d_name if d_name.endswith(".csv") else f"{d_name}.csv"
        file_path = os.path.join(path_to_lafan1_datasets, robot_conf.dir_name, f"{d_name}")

        # load csv
        data = np.loadtxt(file_path, delimiter=",")

        logger.info(f"Loaded LAFAN1 dataset: {d_name}.")

        # get the joint data
        free_joint_pos = data[:, :3]
        free_joint_quat_scalarlast = data[:, 3:7]
        free_joint_quat = quat_scalarlast2scalarfirst(free_joint_quat_scalarlast)
        free_joint_rotvec = sRot.from_quat(free_joint_quat).as_rotvec()
        joints = data[:, 7:]

        free_joint_vel = (free_joint_pos[2:] - free_joint_pos[:-2]) / (2 * (1 / robot_conf.fps))
        free_joint_vel_rot = (free_joint_rotvec[2:] - free_joint_rotvec[:-2]) / (2 * (1 / robot_conf.fps))
        joints_vel = (joints[2:] - joints[:-2]) / (2 * (1 / robot_conf.fps))

        free_joint_pos = free_joint_pos[1:-1]
        free_joint_quat = free_joint_quat[1:-1]
        joints = joints[1:-1]

        qpos = jnp.concatenate([free_joint_pos, free_joint_quat, joints], axis=1)
        qvel = jnp.concatenate([free_joint_vel, free_joint_vel_rot, joints_vel], axis=1)

        # put into Trajectory
        njnt = len(robot_conf.jnt_names)
        jnt_type = np.array(
            [int(mujoco.mjtJoint.mjJNT_FREE)] + [int(mujoco.mjtJoint.mjJNT_HINGE)] * (njnt-1))
        traj_info = TrajectoryInfo(joint_names=robot_conf.jnt_names,
                                   model=TrajectoryModel(njnt, jnt_type), frequency=robot_conf.fps)
        traj_data = TrajectoryData(qpos=qpos, qvel=qvel, split_points=jnp.array([0, len(qpos)]))
        traj = Trajectory(info=traj_info, data=traj_data)

        # order and extend the motion
        logger.info("Using Mujoco's forward kinematics to calculate other model-specific entities ...")
        traj = extend_motion(env_name, robot_conf, traj)

        all_trajectories.append(traj)

    # concatenate trajectories
    if len(all_trajectories) == 1:
        trajectory = all_trajectories[0]
    else:
        logger.info("Concatenating trajectories ...")
        traj_datas = [t.data for t in all_trajectories]
        traj_infos = [t.info for t in all_trajectories]
        traj_data, traj_info = TrajectoryData.concatenate(traj_datas, traj_infos)
        trajectory = Trajectory(traj_info, traj_data)

    logger.info("Trajectory data loaded!")

    return trajectory


if __name__ == "__main__":
    from loco_mujoco.datasets.humanoids.LAFAN1 import LAFAN1_ALL_DATASETS

    traj = load_lafan1_trajectory("UnitreeH1", LAFAN1_ALL_DATASETS)
    print("Done!")
