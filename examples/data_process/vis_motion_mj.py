import time
from pathlib import Path

import torch
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
import joblib
import yaml

import loco_mujoco
from loco_mujoco import PATH_TO_MODELS


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_connector(scene.geoms[scene.ngeom-1], mujoco.mjtGeom.mjGEOM_CAPSULE, radius, point1, point2)


def key_call_back( keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    else:
        print("not mapped", chr(keycode))
    

if __name__ == "__main__":

    path_to_conf = loco_mujoco.PATH_TO_SMPL_CONF

    # read paths from yaml file
    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        path_to_smpl_model = data["LOCOMUJOCO_SMPL_MODEL_PATH"]
        path_to_amass_datasets = data["LOCOMUJOCO_AMASS_PATH"]

    assert path_to_smpl_model is not None, "Please set the environment variable LOCOMUJOCO_SMPL_MODEL_PATH."
    assert path_to_amass_datasets is not None, "Please set the environment variable LOCOMUJOCO_AMASS_PATH."

    loco_mujoco_path = Path(loco_mujoco.__file__).parent

    device = torch.device("cpu")
    # humanoid_xml = "loco_mujoco/environments/data/talos/talos.xml"
    # humanoid_xml = loco_mujoco_path / "environments/data/unitree_g1/g1.xml"
    # humanoid_xml = "loco_mujoco/environments/data/stompy_old/stompy.xml"
    # humanoid_xml = "loco_mujoco/environments/data/stompymini/robot.xml"
    humanoid_xml = PATH_TO_MODELS / "unitree_h1/h1.xml"
    #sk_tree = SkeletonTree.from_mjcf(humanoid_xml)
    
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused = 0, 1, 0, set(), 0, 1/30, False
    
    # motion_file = "data/talos/0-KIT_3_walking_slow08_poses.pkl"
    # motion_file = "data/stompy/0-KIT_3_walking_slow08_poses.pkl"
    motion_file = path_to_amass_datasets + "/unitree_h1/0-Transitions_mocap_mazen_c3d_airkick_jumpinplace_poses.pkl"
    motion_data = joblib.load(motion_file)
    motion_data_keys = list(motion_data.keys())
    curr_motion_key = motion_data_keys[motion_id]
    
    mj_model = mujoco.MjModel.from_xml_path(str(humanoid_xml))
    mj_data = mujoco.MjData(mj_model)
    
    
    RECORDING = False
    
    mj_model.opt.timestep = dt
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        for _ in range(50):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
        # for _ in range(24):
        #     add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([0, 1, 0, 1]))
        # Close the viewer automatically after 30 wall-seconds.
        while viewer.is_running():
            step_start = time.time()
            curr_motion = motion_data[curr_motion_key]
            curr_time = int(time_step/dt) % curr_motion['dof'].shape[0]
            
            # mj_data.qpos[:3] = curr_motion['root_trans_offset'][curr_time]
            # mj_data.qpos[3:6] = sRot.from_quat(curr_motion['root_rot'][curr_time]).as_euler("XYZ")
            # mj_data.qpos[6:] = curr_motion['dof'][curr_time]

            mj_data.qpos[:3] = curr_motion['root_trans_offset'][curr_time]
            mj_data.qpos[3:7] = curr_motion['root_rot'][curr_time][[3, 0, 1, 2]]
            mj_data.qpos[7:] = curr_motion['dof'][curr_time]
                
            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                time_step += dt

            # pose_aa_h1_new = torch.cat([torch.from_numpy(sRot.from_quat(root_rot).as_rotvec()[None, :, None]), H1_ROTATION_AXIS * dof_pos[None, ..., None], torch.zeros((1, 1, 2, 3))], axis = 2).float()
            # for i in range(rg_pos_t.shape[1]):
            #     if not i in [20, 21, 22]:
            #         continue
            #     viewer.user_scn.geoms[i].pos = rg_pos_t[0, i]
                # viewer.user_scn.geoms[i].pos = mj_data.xpos[:][i]
            
            joint_gt = motion_data[curr_motion_key]['smpl_joints']
            
            for i in range(joint_gt.shape[1]):
                viewer.user_scn.geoms[i].pos = joint_gt[curr_time, i]
            # joints_opt = kps_dict['j3d_h1'].squeeze()#[h1_joint_pick_idx]
            # for i in range(len(joints_opt)):
            #     viewer.user_scn.geoms[i].pos = joints_opt[i]
                

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            # if RECORDING and time_step > motion_len:
            #     curr_start += num_motions
            #     motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
            #     time_step = 0