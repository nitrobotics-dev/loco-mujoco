import os
from warnings import warn
from pathlib import Path
from dataclasses import replace
from scipy.spatial.transform import Rotation as R

import mujoco
import numpy as np
import jax.numpy as jnp
import loco_mujoco
#from loco_mujoco import LocoEnv
from loco_mujoco.datasets.data_generation import load_robot_conf, load_dataset_conf, ExtendTrajData
from loco_mujoco.utils.dataset import adapt_mocap

from loco_mujoco.trajectory import *


PATH_RAW_DATASET = Path(loco_mujoco.__file__).parent / "datasets/data_generation/00_raw_mocap_data"
PATH_GENERATED_DATASET = Path(loco_mujoco.__file__).parent / "datasets/real"
PATH_ROBOT_CONFS = Path(loco_mujoco.__file__).parent / "datasets/data_generation/confs/robots"
PATH_DATASET_CONFS = Path(loco_mujoco.__file__).parent / "datasets/data_generation/confs/datasets"


def create_traj_data(dataset: dict):

    joint_names_in_qpos = []
    joint_names_in_qvel = []
    joint_ids = []
    qpos = []
    qvel = []
    frequency = None
    split_points = None

    for key, value in dataset.items():
        if key.startswith("q_"):
            j_name_in_xml = "_".join(key.split("_")[1:])
            joint_names_in_qpos.append(j_name_in_xml)
            qpos.append(value)
        elif key.startswith("dq_"):
            j_name_in_xml = "_".join(key.split("_")[1:])
            joint_names_in_qvel.append(j_name_in_xml)
            qvel.append(value)
        elif key == "frequency":
            frequency = value
        elif key == "split_points":
            split_points = value

    assert len(qpos) == len(qvel)

    # reorder qvel to match the order of qpos
    qvel = [qvel[joint_names_in_qvel.index(j_name)] for j_name in joint_names_in_qpos]

    qpos = np.array(qpos).T
    qvel = np.array(qvel).T

    # convert to free joint
    q_root_pos = qpos[:, :3]
    q_root_euler = qpos[:, 3:6]
    q_root_euler[:, 2] += np.pi
    #q_root_euler[:, 2] = np.pi
    q_root_euler = q_root_euler[:, [2, 0, 1]]
    #q_root_euler = np.zeros_like(q_root_euler)
    q_root_quat = R.from_euler("XYZ", q_root_euler).as_quat()
    q_root = np.concatenate([q_root_pos, q_root_quat], axis=1)
    qpos = jnp.concatenate([q_root, qpos[:, 6:]], axis=1)
    joint_names_in_qpos = ["root"] + [j_name for j_name in joint_names_in_qpos if "pelvis" not in j_name]

    split_points = jnp.array([0, qpos.shape[0]]) if split_points is None else jnp.array(split_points)

    joint_types = [int(mujoco.mjtJoint.mjJNT_FREE)] + [int(mujoco.mjtJoint.mjJNT_HINGE) for j in joint_names_in_qpos[1:]]
    traj_model = TrajectoryModel(njnt=len(joint_types), jnt_type=jnp.array(joint_types))
    traj_info = TrajectoryInfo(joint_names=joint_names_in_qpos, frequency=frequency, model=traj_model)
    traj_data = TrajectoryData(qpos=qpos, qvel=qvel, split_points=split_points)
    return traj_data, traj_info


def convert_single_dataset_of_env(env_name, file_path):

    registered_envs = loco_mujoco.get_registered_envs()

    robot_conf = load_robot_conf(PATH_ROBOT_CONFS / f"{env_name}.yaml")

    all_task_names = loco_mujoco.get_all_task_names()
    all_task_names = [name.split(".")[0] for name in all_task_names]

    print(f"Converting {file_path.stem} for robot {env_name}...")

    # specify the path to the target directory and file
    dir_target_path = PATH_GENERATED_DATASET / env_name
    dir_target_path.mkdir(parents=True, exist_ok=True)
    file_target_path = dir_target_path / f"{file_path.stem}.npz"

    # load the dataset configuration if it exists
    try:
        dataset_conf = load_dataset_conf(PATH_DATASET_CONFS / f"{file_path.stem}.yaml")
    except FileNotFoundError:
        dataset_conf = {}
    discard_first = dataset_conf.get("discard_first", 0)
    discard_last = dataset_conf.get("discard_last", 0)

    # load and convert the dataset
    dataset = adapt_mocap(str(file_path), discard_first=discard_first, discard_last=discard_last,
                          **robot_conf["traj_conf"])

    # create and save the initial trajectory datastructure
    traj_data, traj_info = create_traj_data(dataset)

    # get environment class
    env_cls = registered_envs[env_name]

    # create environment instance and run it on the trajectories frequency
    env = env_cls(**robot_conf["env_params"], random_start=False, fixed_start_conf=(0, 0))

    # interpolate trajectory to environment frequency
    traj_data, traj_info = interpolate_trajectories(traj_data, traj_info, 1.0 / env.dt)
    traj = Trajectory(info=traj_info, data=traj_data)

    # load_trajectory
    env.load_trajectory(traj, warn=False)
    traj_data, traj_info = env.th.traj.data, env.th.traj.info

    callback = ExtendTrajData(env, model=env._model, n_samples=traj_data.n_samples)
    env.play_trajectory(n_episodes=env.n_trajectories(traj_data),
                        render=False, callback_class=callback)
    traj_data, traj_info = callback.extend_trajectory_data(traj_data, traj_info)
    traj = replace(traj, data=traj_data, info=traj_info)

    # save the trajectory data
    traj.save(file_target_path)

    # if there is an Mjx version of that environment, create a symbolic link
    mjx_env_name = f"Mjx{env_name}"
    if mjx_env_name in all_task_names:
        link_path = dir_target_path.parent / mjx_env_name
        if not os.path.exists(link_path):
            os.symlink(dir_target_path, link_path)


def convert_all_datasets_of_env(env_name):

    for file_path in PATH_RAW_DATASET.iterdir():
        if file_path.suffix == ".mat":
            convert_single_dataset_of_env(env_name, file_path)


def convert_all_datasets():
    # get all env names
    all_task_names = loco_mujoco.get_all_task_names()

    converted_environments = []
    for task in all_task_names:
        if "Mjx" not in task:
            task = task.split(".")[0]
            try:
                if task not in converted_environments:
                    convert_all_datasets_of_env(task)
                converted_environments.append(task)
            except FileNotFoundError:
                warn(f"File {task}.yaml not found.")

    print("Done.")


if __name__ == "__main__":
    convert_all_datasets()
