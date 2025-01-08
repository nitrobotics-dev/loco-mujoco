import os
from pathlib import Path
from dataclasses import replace
from typing import Union, List

import jax.numpy as jnp
import mujoco
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as sRot

import loco_mujoco
from loco_mujoco.core.utils.math import quat_scalarlast2scalarfirst
from loco_mujoco.datasets.data_generation import ExtendTrajData, optimize_for_collisions
from loco_mujoco.environments import LocoEnv
from loco_mujoco.trajectory import (
    Trajectory,
    TrajectoryInfo,
    TrajectoryModel,
    TrajectoryData,
    interpolate_trajectories)
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
        env_name: str,
        dataset_name: Union[str, List[str]],
        max_steps: int = 100
) -> Trajectory:
    """
    Load a trajectory from the LAFAN1 dataset.

    Args:
        env_name (str): The name of the environment.
        dataset_name (Union[str, List[str]]): The name of the dataset(s) to load.
        max_steps (int, optional): The maximum number of steps to optimize for collisions. Defaults to 100.

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
        path_to_convert_lafan1_datasets = data["LOCOMUJOCO_CONVERTED_LAFAN1_PATH"]

    assert path_to_lafan1_datasets, ("Please use the command 'loco-mujoco-set-lafan1-path' to set the "
                                     "path to the LAFAN1 datasets.")
    assert path_to_convert_lafan1_datasets, ("Please use the command 'loco-mujoco-set-conv-lafan1-path' to set the "
                                            "path to the converted LAFAN1 datasets.")

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
        target_path_dataset = os.path.join(path_to_convert_lafan1_datasets, robot_conf.dir_name, f"{d_name}.npz")
        d_name = d_name if d_name.endswith(".csv") else f"{d_name}.csv"

        # check if the dataset exists
        if not os.path.exists(target_path_dataset):
            file_path = os.path.join(path_to_lafan1_datasets, robot_conf.dir_name, f"{d_name}")

            # load csv
            qpos = np.loadtxt(file_path, delimiter=",")

            logger.info(f"Loaded LAFAN1 dataset: {d_name}.")

            # convert free joint quaternion from scalar last to scalar first
            qpos[:, 3:7] = quat_scalarlast2scalarfirst(qpos[:, 3:7])

            # put into Trajectory
            njnt = len(robot_conf.jnt_names)
            jnt_type = np.array(
                [int(mujoco.mjtJoint.mjJNT_FREE)] + [int(mujoco.mjtJoint.mjJNT_HINGE)] * (njnt-1))
            traj_info = TrajectoryInfo(joint_names=robot_conf.jnt_names,
                                       model=TrajectoryModel(njnt, jnt_type), frequency=robot_conf.fps)
            # set qvel to 0 for now (will be calculated by optimize_for_collisions later)
            qvel = np.zeros_like(qpos)[:, 1:]
            traj_data = TrajectoryData(qpos=qpos, qvel=qvel, split_points=jnp.array([0, len(qpos)]))
            traj = Trajectory(info=traj_info, data=traj_data)

            # order and extend the motion
            logger.info("Using Mujoco to optimize dataset for collisions ...")
            traj = optimize_for_collisions(env_name, robot_conf, traj, max_steps=max_steps)
            traj.save(target_path_dataset)

        else:
            logger.info(f"Found converted dataset at: {target_path_dataset}.")
            traj = Trajectory.load(target_path_dataset)

        logger.info("Using Mujoco's kinematics to calculate other model-specific entities ...")
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
