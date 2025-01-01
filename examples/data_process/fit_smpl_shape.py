import os
from pathlib import Path
import joblib
import yaml

from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import torch
from torch.autograd import Variable

import loco_mujoco
from loco_mujoco.smpl import SMPLH_Parser, SMPLH_BONE_ORDER_NAMES
from loco_mujoco.smpl.torch_fk_humanoid import ForwardKinematicsHumanoidTorch


@hydra.main(version_base=None, config_path="../../loco_mujoco/smpl/robot_confs", config_name="UnitreeH1")
def main(robot_conf: DictConfig) -> None:

    path_to_conf = loco_mujoco.PATH_TO_SMPL_CONF

    # read paths from yaml file
    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        path_to_smpl_model = data["LOCOMUJOCO_SMPL_MODEL_PATH"]
        path_to_amass_datasets = data["LOCOMUJOCO_AMASS_PATH"]
        path_to_converted_amass_datasets = data["LOCOMUJOCO_CONVERTED_AMASS_PATH"]

    assert path_to_smpl_model is not None, "Please set the environment variable LOCOMUJOCO_SMPL_MODEL_PATH."
    assert path_to_amass_datasets is not None, "Please set the environment variable LOCOMUJOCO_AMASS_PATH."
    assert path_to_converted_amass_datasets is not None, ("Please set the environment "
                                                          "variable LOCOMUJOCO_SMPL_MODEL_PATH.")

    humanoid_fk = ForwardKinematicsHumanoidTorch(robot_conf) # load forward kinematics model

    #### Define corresonpdances between h1 and smpl joints
    robot_joint_names_augment = humanoid_fk.joint_names_augment 
    robot_joint_pick = [i[0] for i in robot_conf.joint_matches]
    smpl_joint_pick = [i[1] for i in robot_conf.joint_matches]
    
    robot_joint_pick_idx = [ robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPLH_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    #### Preparing fitting varialbes
    device = torch.device("cpu")
    pose_aa_robot = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], humanoid_fk.num_extend_dof , axis = 2), 1, axis = 1)
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
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta , trans)
    offset = joints[:, 0] - trans
    root_trans_offset = trans + offset

    fk_return = humanoid_fk.fk_batch(pose_aa_robot[None, ], root_trans_offset[None, 0:1])

    shape_new = Variable(torch.zeros([1, 16]).to(device), requires_grad=True)
    scale = Variable(torch.ones([1]).to(device), requires_grad=True)
    optimizer_shape = torch.optim.Adam([shape_new, scale],lr=0.1)
    
    pbar = tqdm(range(1000))
    for iteration in pbar:
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])
        root_pos = joints[:, 0]
        joints = (joints - joints[:, 0]) * scale + root_pos
        if len(robot_conf.extend_config) > 0:
            diff = fk_return.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
        else:
            diff = fk_return.global_translation[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]

        loss_g = diff.norm(dim = -1).mean() 
        loss = loss_g
        pbar.set_description_str(f"{iteration} - Loss: {loss.item() * 1000}")

        optimizer_shape.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_shape.step()
        
    save_path = path_to_converted_amass_datasets + f"/{robot_conf.name}"
    os.makedirs(save_path, exist_ok=True)
    save_path += "/shape_optimized_v1.pkl"
    joblib.dump((shape_new.detach(), scale), save_path)
    print("Shape parameters saved at: ", save_path)


if __name__ == "__main__":
    main()
