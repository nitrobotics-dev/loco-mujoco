import os
from warnings import warn
from pathlib import Path

import mujoco
import numpy as np
import jax.numpy as jnp
import loco_mujoco
from loco_mujoco import LocoEnv
from loco_mujoco.datasets.data_generation import load_robot_conf, load_dataset_conf, ExtendTrajData
from loco_mujoco.utils.dataset import adapt_mocap

from loco_mujoco.core.utils.math import calc_rel_positions, calc_rel_quaternions, calc_rel_body_velocities
from loco_mujoco.trajectory import *


PATH_RAW_DATASET = Path("00_raw_mocap_data")
PATH_GENERATED_DATASET = Path(loco_mujoco.__file__).parent / "datasets/humanoids/real"
PATH_ROBOT_CONFS = Path("confs/robots")
PATH_DATASET_CONFS = Path("confs/datasets")


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

    qpos = jnp.array(qpos).T
    qvel = jnp.array(qvel).T

    split_points = jnp.array([0, qpos.shape[0]]) if split_points is None else jnp.array(split_points)

    joint_types = [int(mujoco.mjtJoint.mjJNT_HINGE) for j in joint_names_in_qpos]    # todo: this is not correct
    traj_model = TrajectoryModel(njnt=len(joint_types), jnt_type=jnp.array(joint_types))
    traj_info = TrajectoryInfo(joint_names=joint_names_in_qpos, frequency=frequency, model=traj_model)
    traj_data = TrajectoryData(qpos=qpos, qvel=qvel, split_points=split_points)
    return traj_data, traj_info


def convert_all_datasets_of_env(env_name, robot_conf):

    for file_path in PATH_RAW_DATASET.iterdir():
        if file_path.suffix == ".mat":

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
            env_cls = LocoEnv.registered_envs[env_name]

            # create environment instance and run it on the trajectories frequency
            env = env_cls(**robot_conf["env_params"], random_start=False, fixed_start_conf=(0, 0))

            # interpolate trajectory to environment frequency
            traj_data, traj_info = interpolate_trajectories(traj_data, traj_info, 1.0/env.dt)

            # load_trajectory
            traj_params = dict(traj_data=traj_data, traj_info=traj_info,
                               control_dt=env.dt)
            env.load_trajectory(traj_params)
            traj_data, traj_info = env.th.traj_data, env.th.traj_info

            callback = ExtendTrajData(env, model=env._model, n_samples=traj_data.n_samples)
            env.play_trajectory(n_episodes=env.n_trajectories(traj_data),
                                render=False, callback_class=callback)
            traj_data, traj_info = callback.extend_trajectory_data(traj_data, traj_info)

            # save the trajectory data
            save_trajectory_to_npz(file_target_path, traj_data, traj_info)


if __name__ == "__main__":

    # get all env names
    all_task_names = loco_mujoco.get_all_task_names()

    converted_environments = []
    for task in all_task_names:
        if "Mjx" not in task:
            task = task.split(".")[0]
            try:
                if task not in converted_environments:
                    robot_conf = load_robot_conf(PATH_ROBOT_CONFS / f"{task}.yaml")
                    convert_all_datasets_of_env(task, robot_conf)
                converted_environments.append(task)
            except FileNotFoundError:
                warn(f"File {task}.yaml not found.")

    print("Done.")
