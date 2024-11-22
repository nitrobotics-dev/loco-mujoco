import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from smpl_sim.utils import torch_utils
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser, 
)

import joblib
import torch
import torch.nn.functional as F
import math
from smpl_sim.utils.pytorch3d_transforms import axis_angle_to_matrix
from torch.autograd import Variable
from tqdm.notebook import tqdm
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from loco_mujoco.smpl.torch_humanoid_batch import ForwardKinematicsHumanoidTorch
from easydict import EasyDict
import hydra
from omegaconf import DictConfig, OmegaConf
import mujoco

@hydra.main(version_base=None, config_path="../../loco_mujoco/cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    
    robot_name = "h1"
    humanoid_fk = ForwardKinematicsHumanoidTorch(cfg.robot) # load forward kinematics model

    robot_joint_names = humanoid_fk.joint_names
    
    
    mj_model = mujoco.MjModel.from_xml_path(humanoid_fk.mjcf_file)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    dof_pos_new = torch.zeros((1, 2, len(humanoid_fk.joint_names) - 1, 1))
    dof_pos_new[0, 0, :, 0] = torch.randn(len(humanoid_fk.joint_names) - 1)
    mujoco.mj_forward(mj_model, mj_data)
    if humanoid_fk.has_freejoint:
        mj_data.qpos[:3] = np.zeros(3)
        mj_data.qpos[3:7] = np.array([1, 0, 0, 0])
        mj_data.qpos[7:] = dof_pos_new[0, 0, :, 0].numpy()
    else:
        mj_data.qpos[:3] = np.zeros(3)
        mj_data.qpos[3:6] = np.array([0, 0, 0])
        mj_data.qpos[6:] = dof_pos_new[0, 0, :, 0].numpy()
    mujoco.mj_forward(mj_model, mj_data)

    pose_aa_h1_new = torch.cat([torch.zeros((1, 2, 1, 3)), humanoid_fk.dof_axis * dof_pos_new], axis = 2)

    fk_return = humanoid_fk.fk_batch(pose_aa_h1_new, torch.zeros(1, 2, 3), return_full= False)
    diff = mj_data.xpos[1:] - fk_return.global_translation.squeeze().numpy()[0]
    print(np.abs(diff).sum())


    
if __name__ == "__main__":
    main()
