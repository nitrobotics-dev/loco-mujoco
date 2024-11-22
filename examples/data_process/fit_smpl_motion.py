import glob
from pathlib import Path

import joblib
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import torch
from torch.autograd import Variable

import loco_mujoco
from loco_mujoco.smpl import SMPL_Parser
from loco_mujoco.smpl.utils import torch_utils
from loco_mujoco.smpl import SMPL_BONE_ORDER_NAMES
from loco_mujoco.smpl.torch_fk_humanoid import ForwardKinematicsHumanoidTorch
from loco_mujoco.smpl.utils.smoothing import gaussian_filter_1d_batch


def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not 'mocap_framerate' in  entry_data:
        return 
    framerate = entry_data['mocap_framerate']


    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
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


@hydra.main(version_base=None, config_path="../../loco_mujoco/cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    device = torch.device("cpu")

    # path to loco-mujoco robot models
    loco_mujoco_path = Path(loco_mujoco.__file__).parent
    path_to_models = loco_mujoco_path / "environments" / "data"
    path_to_amass_datasets = loco_mujoco_path.parent / "data"
    path_to_all_amass_files = path_to_amass_datasets / "AMASS/SMPL+H G/**/*.npz"
    cfg.robot.mjcf_file = str(path_to_models / cfg.robot.mjcf_file)

    humanoid_fk = ForwardKinematicsHumanoidTorch(cfg.robot) # load forward kinematics model
    num_augment_joint = len(cfg.robot.extend_config)

    #### Define corresonpdances between h1 and smpl joints
    robot_joint_names_augment = humanoid_fk.joint_names_augment 
    robot_joint_pick = [i[0] for i in cfg.robot.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.robot.joint_matches]
    robot_joint_pick_idx = [ robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    all_pkls = glob.glob(str(path_to_all_amass_files), recursive=True)
    key_names = ["0-" + "_".join(data_path.split("/")[-3:]).replace(".npz", "") for data_path in all_pkls]

    #data_key = "0-KIT_3_walking_slow08_poses"
    #data_key = "0-Transitions_mocap_mazen_c3d_dance_stand_poses"
    data_key = "0-Transitions_mocap_mazen_c3d_airkick_jumpinplace_poses"

    smpl_parser_n = SMPL_Parser(model_path=loco_mujoco_path.parent / "data" / "smpl", gender="neutral")

    amass_data = load_amass_data(all_pkls[key_names.index(data_key)])
    skip = int(amass_data['fps']//30)
    trans = torch.from_numpy(amass_data['trans'][::skip])
    N = trans.shape[0]
    pose_aa_walk = torch.from_numpy(amass_data['pose_aa'][::skip]).float()
    shape_new, scale = joblib.load(loco_mujoco_path.parent / "data" / cfg.robot.name / "shape_optimized_v1.pkl")

    with torch.no_grad():
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
        root_pos = joints[:, 0:1]
        joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos
        
    offset = joints[:, 0] - trans
    root_trans_offset = (trans + offset).clone()



    if root_trans_offset[..., 2].min() < 0.6:
        print("too low!!!!")

    gt_root_rot_quat = torch.from_numpy((sRot.from_rotvec(pose_aa_walk[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat()).float() # can't directly use this 
    gt_root_rot = torch.from_numpy(sRot.from_quat(torch_utils.calc_heading_quat(gt_root_rot_quat)).as_rotvec()).float() # so only use the heading.

    # def dof_to_pose_aa(dof_pos):
    dof_pos = torch.zeros((1, N, humanoid_fk.num_dof, 1))

    dof_pos_new = Variable(dof_pos.clone(), requires_grad=True)
    root_rot_new = Variable(gt_root_rot.clone(), requires_grad=True)
    root_pos_offset = Variable(torch.zeros(1, 3), requires_grad=True)
    optimizer_pose = torch.optim.Adadelta([dof_pos_new],lr=100)
    optimizer_root = torch.optim.Adam([root_rot_new, root_pos_offset],lr=0.01)


    kernel_size = 5  # Size of the Gaussian kernel
    sigma = 0.75  # Standard deviation of the Gaussian kernel
    B, T, J, D = dof_pos_new.shape    

    pbar = tqdm(range(500))
    for iteration in pbar:
        pose_aa_h1_new = torch.cat([root_rot_new[None, :, None], humanoid_fk.dof_axis * dof_pos_new, torch.zeros((1, N, num_augment_joint, 3)).to(device)], axis = 2)
        fk_return = humanoid_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ] + root_pos_offset )
        
        if num_augment_joint > 0:
            diff = fk_return.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
        else:
            diff = fk_return.global_translation[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            
        loss_g = diff.norm(dim = -1).mean() 
        loss = loss_g
        
        optimizer_pose.zero_grad()
        optimizer_root.zero_grad()
        loss.backward()
        optimizer_pose.step()
        optimizer_root.step()
        
        dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])

        pbar.set_description_str(f"Iter: {iteration} \t {loss.item() * 1000:.3f}")
        dof_pos_new.data = gaussian_filter_1d_batch(dof_pos_new.squeeze().transpose(1, 0)[None, ], kernel_size, sigma).transpose(2, 1)[..., None]
        
        # from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        # import matplotlib.pyplot as plt
        
        # j3d = fk_return.global_translation[0, :, :, :].detach().numpy()
        # j3d_joints = joints.detach().numpy()
        # idx = 0
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.view_init(90, 0)
        # ax.scatter(j3d[idx, :,0], j3d[idx, :,1], j3d[idx, :,2])
        # ax.scatter(j3d_joints[idx, :,0], j3d_joints[idx, :,1], j3d_joints[idx, :,2])

        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # drange = 1
        # ax.set_xlim(-drange, drange)
        # ax.set_ylim(-drange, drange)
        # ax.set_zlim(-drange, drange)
        # plt.show()
        
        
    dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])
    pose_aa_h1_new = torch.cat([root_rot_new[None, :, None], humanoid_fk.dof_axis * dof_pos_new, torch.zeros((1, N, 2, 3))], axis = 2)

    height_diff = fk_return.global_translation[..., 2].min().item() - 0.05
    root_trans_offset_dump = (root_trans_offset + root_pos_offset ).clone()
    root_trans_offset_dump[..., 2] -= height_diff

    joints_dump = joints.numpy().copy()
    joints_dump[..., 2] -= height_diff
    #import ipdb; ipdb.set_trace()
    dumped_file = path_to_amass_datasets / f"{cfg.robot.name}/{data_key}.pkl"
    print(dumped_file)
    joblib.dump(
        {
            data_key:{
            "root_trans_offset": root_trans_offset_dump.squeeze().detach().numpy(),
            "pose_aa": pose_aa_h1_new.squeeze().detach().numpy(),   
            "dof": dof_pos_new.squeeze().detach().numpy(), 
            "root_rot": sRot.from_rotvec(root_rot_new.detach().numpy()).as_quat(),
            "smpl_joints": joints_dump, 
            "fps": 30
            }, 
            }, dumped_file)


if __name__ == "__main__":
    main()
