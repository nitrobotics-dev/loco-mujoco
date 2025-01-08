import math
from copy import deepcopy
from dataclasses import replace
from typing import List

import jax.numpy as jnp
import mujoco
import numpy as np
import yaml
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as sRot
from omegaconf import DictConfig

from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast
from loco_mujoco.core.utils.mujoco import (
    mj_jntid2qposid,
    mj_jntid2qvelid,
    mj_jntname2qposid,
    mj_jntname2qvelid)
from loco_mujoco.trajectory import (
    Trajectory,
    TrajectoryInfo,
    TrajectoryModel,
    TrajectoryData,
    interpolate_trajectories)


class ReplayCallback:

    """Base class that can be used to do things while replaying a trajectory."""

    @staticmethod
    def __call__(env, model, data, traj_sample, carry):
        data = env.set_sim_state_from_traj_data(data, traj_sample, carry)
        model, data, carry = env._simulation_pre_step(model, data, carry)
        mujoco.mj_forward(model, data)
        data, carry = env._simulation_post_step(model, data, carry)
        return model, data, carry


class ExtendTrajData(ReplayCallback):

    def __init__(self, env, n_samples, model, body_names=None, site_names=None):
        self.b_names, self.b_ids = self.get_body_names_and_ids(env._model, body_names)
        self.s_names, self.s_ids = self.get_site_names_and_ids(env._model, site_names)
        dim_qpos, dim_qvel = 0, 0
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                dim_qpos += 7
                dim_qvel += 6
            else:
                dim_qpos += 1
                dim_qvel += 1
        self.recorder = dict(xpos=np.empty((n_samples, model.nbody, 3)),
                             xquat=np.empty((n_samples, model.nbody, 4)),
                             cvel=np.empty((n_samples, model.nbody, 6)),
                             subtree_com=np.empty((n_samples, model.nbody, 3)),
                             site_xpos=np.empty((n_samples, model.nsite, 3)),
                             site_xmat=np.empty((n_samples, model.nsite, 9)),
                             qpos=np.empty((n_samples, dim_qpos)),
                             qvel=np.empty((n_samples, dim_qvel)))
        self.traj_model = TrajectoryModel(njnt=model.njnt,
                                          jnt_type=jnp.array(model.jnt_type),
                                          nbody=model.nbody,
                                          body_rootid=jnp.array(model.body_rootid),
                                          body_weldid=jnp.array(model.body_weldid),
                                          body_mocapid=jnp.array(model.body_mocapid),
                                          body_pos=jnp.array(model.body_pos),
                                          body_quat=jnp.array(model.body_quat),
                                          body_ipos=jnp.array(model.body_ipos),
                                          body_iquat=jnp.array(model.body_iquat),
                                          nsite=model.nsite,
                                          site_bodyid=jnp.array(model.site_bodyid),
                                          site_pos=jnp.array(model.site_pos),
                                          site_quat=jnp.array(model.site_quat))
        self.current_length = 0

    def __call__(self, env, model, data, traj_sample, carry):
        model, data, carry = super().__call__(env, model, data, traj_sample, carry)

        self.recorder["xpos"][self.current_length] = data.xpos[self.b_ids]
        self.recorder["xquat"][self.current_length] = data.xquat[self.b_ids]
        self.recorder["cvel"][self.current_length] = data.cvel[self.b_ids]
        self.recorder["subtree_com"][self.current_length] = data.subtree_com[self.b_ids]
        self.recorder["site_xpos"][self.current_length] = data.site_xpos[self.s_ids]
        self.recorder["site_xmat"][self.current_length] = data.site_xmat[self.s_ids]

        # add joint properties
        self.recorder["qpos"][self.current_length] = data.qpos
        self.recorder["qvel"][self.current_length] = data.qvel

        self.current_length += 1

        return model, data, carry

    def extend_trajectory_data(self, traj_data: TrajectoryData, traj_info: TrajectoryInfo):
        assert self.current_length == traj_data.qpos.shape[0]
        assert traj_info.model.njnt == self.traj_model.njnt
        converted_data = {}
        for key, value in self.recorder.items():
            converted_data[key] = jnp.array(value)
        return (traj_data.replace(**converted_data),
                replace(traj_info, body_names=self.b_names if len(self.b_names) > 0 else None,
                        site_names=self.s_names if len(self.s_names) > 0 else None,
                        model=self.traj_model))

    @staticmethod
    def get_body_names_and_ids(model, keys=None):
        """
        Get the names of the bodies in the model. If keys is not None, only return the names of the bodies
        that are in keys, otherwise return all body names.

        Args:
            model: mujoco model
            keys: list of body names

        Returns:
            List of body names and list of body ids.
        """
        keys = deepcopy(keys)
        body_names = []
        ids = range(model.nbody)
        for i in ids:
            b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if (keys is not None) and (b_name in keys):
                keys.remove(b_name)
            body_names.append(b_name)
        assert keys is None or len(keys) == 0, f"Could not find the following body names: {keys}"
        return body_names, list(ids)

    @staticmethod
    def get_site_names_and_ids(model, keys=None):
        """
        Get the names of the sites in the model. If keys is not None, only return the names of the sites
        that are in keys, otherwise return all site names.

        Args:
            model: mujoco model
            keys: list of site names

        Returns:
            List of site names and list of site ids.
        """
        keys = deepcopy(keys)
        site_names = []
        ids = range(model.nsite)
        for i in ids:
            s_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            if (keys is not None) and (s_name in keys):
                keys.remove(s_name)
            site_names.append(s_name)
        assert keys is None or len(keys) == 0, f"Could not find the following site names: {keys}"
        return site_names, list(ids)


def add_mocap_bodies(mjspec: MjSpec,
                     sites_for_mimic: List[str],
                     mocap_bodies: List[str]):
    """
    Add mocap bodies to the model specification.

    Args:
        mjspec (MjSpec): The model specification.
        sites_for_mimic (List[str]): The sites to mimic.
        mocap_bodies (List[str]): The names of the mocap bodies to be added to the model specification.
        mocap_bodies_init_pos: The initial positions of the mocap bodies.

    """

    for mb_name in mocap_bodies:
        b_handle = mjspec.worldbody.add_body(name=mb_name, mocap=True)
        b_handle.add_site(name=mb_name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.05, 0.01],
                          rgba=[1.0, 0.0, 0.0, 0.5], group=1)

    for b1, b2 in zip(sites_for_mimic, mocap_bodies):
        mjspec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1=b1, name2=b2, objtype=mujoco.mjtObj.mjOBJ_SITE)

    return mjspec


class CollisionExtender(ExtendTrajData):
    """
    This class takes a trajectory of target qpos and calculates new qpos, which respect collisions.
    Joint velocities are calculated using finite difference.

    """

    def __init__(self, env, n_samples, model, target_qpos, body_names=None, site_names=None, max_steps=50):
        super().__init__(env, n_samples, model, body_names, site_names)
        self.target_qpos = target_qpos
        self.data_for_sites = env.get_data()
        self.sites_for_mimic = env.sites_for_mimic
        self.site_for_mimic_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
                                            for s in self.sites_for_mimic])
        self.free_joint_name = env.root_free_joint_xml_name
        self.free_joint_qpos_ids = np.array(mj_jntname2qposid(self.free_joint_name, model))
        self.free_joint_qvel_ids = np.array(mj_jntname2qvelid(self.free_joint_name, model))
        self.joints_ids = [i for i, t in enumerate(model.jnt_type) if t != mujoco.mjtJoint.mjJNT_FREE]
        self.joints_qpos_ids = np.squeeze([mj_jntid2qposid(i, model) for i in self.joints_ids])
        self.joints_qvel_ids = np.squeeze([mj_jntid2qvelid(i, model) for i in self.joints_ids])

        self.max_steps = max_steps

    def __call__(self, env, model, data, traj_sample, carry):
        # calculate the site position by setting the simulation state from the trajectory and forward kinematics
        self.data_for_sites.qpos = self.target_qpos[self.current_length]
        mujoco.mj_kinematics(model, self.data_for_sites)

        # set the required mocap positions and quaternions and simulate
        data.mocap_pos = self.data_for_sites.site_xpos[self.site_for_mimic_ids]
        data.mocap_quat = sRot.from_matrix(
            self.data_for_sites.site_xmat[self.site_for_mimic_ids].reshape(-1, 3, 3)).as_quat(scalar_first=True)
        mujoco.mj_step(model, data, self.max_steps)

        # add joint properties
        self.recorder["qpos"][self.current_length] = data.qpos
        self.recorder["qvel"][self.current_length] = np.zeros_like(data.qvel)  # not useful, but kept for dimensionality

        self.current_length += 1

        return model, data, carry

    def extend_trajectory_data(self, traj_data: TrajectoryData, traj_info: TrajectoryInfo):
        traj_data, traj_info = super().extend_trajectory_data(traj_data, traj_info)

        # calculate qvel using finite difference.
        free_joint_pos = traj_data.qpos[:, self.free_joint_qpos_ids[:3]]
        free_joint_quat_scalarfirst = traj_data.qpos[:, self.free_joint_qpos_ids[3:]]
        free_joint_quat = quat_scalarfirst2scalarlast(free_joint_quat_scalarfirst)
        free_joint_rotvec = sRot.from_quat(free_joint_quat).as_rotvec()
        joint_qpos = traj_data.qpos[:, self.joints_qpos_ids]

        free_joint_vel = (free_joint_pos[2:] - free_joint_pos[:-2]) / (2 * (1 / traj_info.frequency))
        free_joint_vel_rot = (free_joint_rotvec[2:] - free_joint_rotvec[:-2]) / (2 * (1 / traj_info.frequency))
        joints_vel = (joint_qpos[2:] - joint_qpos[:-2]) / (2 * (1 / traj_info.frequency))

        qvel = traj_data.qvel
        qvel = qvel.at[1:-1, self.free_joint_qvel_ids].set(jnp.concatenate([free_joint_vel, free_joint_vel_rot], axis=1))
        qvel = qvel.at[1:-1, self.joints_qvel_ids].set(joints_vel)

        qpos = traj_data.qpos
        free_joint_pose = jnp.concatenate([free_joint_pos, free_joint_quat_scalarfirst], axis=1)[1:-1]
        joint_qpos = joint_qpos[1:-1]
        qpos = qpos.at[1:-1, self.free_joint_qpos_ids].set(free_joint_pose)
        qpos = qpos.at[1:-1, self.joints_qpos_ids].set(joint_qpos)

        # create new traj_info including only joint properties
        traj_info = TrajectoryInfo(joint_names=traj_info.joint_names,
                                   model=TrajectoryModel(njnt=traj_info.model.njnt, jnt_type=traj_info.model.jnt_type),
                                   frequency=traj_info.frequency)

        return TrajectoryData(qpos=qpos, qvel=qvel, split_points=jnp.array([0, len(qpos)])), traj_info


def optimize_for_collisions(
        env_name: str,
        robot_conf: DictConfig,
        traj: Trajectory,
        max_steps: int = 100
) -> Trajectory:
    """
    Optimize a motion trajectory to consider collisions with the environment.

    Args:
        env_name (str): The name of the environment.
        robot_conf (DictConfig): The robot configuration.
        traj (Trajectory): The trajectory to optimize.
        max_steps (int): The maximum number of steps to optimize the trajectory for collisions.

    """
    from loco_mujoco import LocoEnv
    # add mocap bodies for all 'site_for_mimic' instances of an environment
    env_cls = LocoEnv.registered_envs[env_name]
    env = env_cls(**robot_conf.env_params, th_params=dict(random_start=False, fixed_start_conf=(0, 0)))
    mjspec = env.mjspec
    sites_for_mimic = env.sites_for_mimic
    target_mocap_bodies = ["target_mocap_body_" + s for s in sites_for_mimic]
    mjspec = add_mocap_bodies(mjspec, sites_for_mimic, target_mocap_bodies)
    env = LocoEnv.registered_envs[env_name](**robot_conf.env_params, xml_path=mjspec,
                                            th_params=dict(random_start=False, fixed_start_conf=(0, 0)))

    traj_data, traj_info = interpolate_trajectories(traj.data, traj.info, 1.0 / env.dt)
    traj = Trajectory(info=traj_info, data=traj_data)

    env.load_trajectory(traj, warn=False)
    traj_data, traj_info = env.th.traj.data, env.th.traj.info

    callback = CollisionExtender(env, model=env._model, target_qpos=traj_data.qpos,
                                 n_samples=traj_data.n_samples, max_steps=max_steps)
    env.play_trajectory(
        n_episodes=env.th.n_trajectories,
        render=False,
        callback_class=callback
    )
    traj_data, traj_info = callback.extend_trajectory_data(traj_data, traj_info)

    traj = replace(traj, data=traj_data, info=traj_info)

    return traj


def expression_constructor(loader, node):
    # Get the scalar value (the expression in string form)
    value = loader.construct_scalar(node)
    try:
        # Safely evaluate the expression
        return eval(value, {"__builtins__": None}, {"pi": math.pi, "np": math, "math": math})
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {value}, {e}")


def load_robot_conf(file_path):
    # Register the custom constructor for the tag '!expr'
    yaml.SafeLoader.add_constructor('!expr', expression_constructor)
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def load_dataset_conf(file_path):
    # simple yaml loader
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
