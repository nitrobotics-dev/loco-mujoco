from types import ModuleType
from typing import Any, Dict, Tuple, Union

import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model
import numpy as np
import jax.numpy as jnp
from jax._src.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.observations.base import ObservationType
from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid, mj_jntid2qposid
from loco_mujoco.core.utils.math import calculate_relative_site_quatities, quaternion_angular_distance
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast


def check_traj_provided(method):
    """
    Decorator to check if trajectory handler is None. Raises ValueError if not provided.
    """
    def wrapper(self, *args, **kwargs):
        env = kwargs.get('env', None) if 'env' in kwargs else args[5]  # Assumes 'env' is the 6th positional argument
        if getattr(env, "th") is None:
            raise ValueError("TrajectoryHandler not provided, but required for trajectory-based rewards.")
        return method(self, *args, **kwargs)
    return wrapper


class TrajectoryBasedReward(Reward):

    @property
    def requires_trajectory(self) -> bool:
        return True


class TargetVelocityTrajReward(TrajectoryBasedReward):
    """
    Reward function that computes the reward based on the deviation from the trajectory velocity. The trajectory
    velocity is provided as an observation in the environment. The reward is computed as the negative exponential
    of the squared difference between the current velocity and the goal velocity. The reward is computed for the
    x, y, and yaw velocities of the root.

    """

    def __init__(self, env: Any,
                 w_exp=10.0,
                 **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): Environment instance.
            w_exp (float, optional): Exponential weight for the reward. Defaults to 10.0.
            **kwargs (Any): Additional keyword arguments.
        """

        super().__init__(env, **kwargs)
        self._free_jnt_name = self._info_props["root_free_joint_xml_name"]
        self._free_joint_qpos_idx = np.array(mj_jntname2qposid(self._free_jnt_name, env._model))
        self._free_joint_qvel_idx = np.array(mj_jntname2qvelid(self._free_jnt_name, env._model))
        self._w_exp = w_exp

    @check_traj_provided
    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:
        """
        Computes a tracking reward based on the deviation from the trajectory velocity.
        Tracking is done on the x, y, and yaw velocities of the root.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.

        Raises:
            ValueError: If trajectory handler is not provided.

        """
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        def calc_local_vel(_d):
            _lin_vel_global = backend.squeeze(_d.qvel[self._free_joint_qvel_idx])[:3]
            _ang_vel_global = backend.squeeze(_d.qvel[self._free_joint_qvel_idx])[3:]
            _root_quat = R.from_quat(quat_scalarfirst2scalarlast(backend.squeeze(_d.qpos[self._free_joint_qpos_idx])[3:7]))
            _lin_vel_local = _root_quat.as_matrix().T @ _lin_vel_global
            # construct vel, x, y and yaw
            return backend.concatenate([_lin_vel_local[:2], backend.atleast_1d(_ang_vel_global[2])])

        # get root velocity from data
        vel_local = calc_local_vel(data)

        # calculate the same for the trajectory
        traj_data = env.th.traj.data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no, backend)
        traj_vel_local = calc_local_vel(traj_data)

        # calculate tracking reward
        tracking_reward = backend.exp(-self._w_exp*backend.mean(backend.square(vel_local - traj_vel_local)))

        return tracking_reward, carry


class MimicReward(TrajectoryBasedReward):
    """
    DeepMimic reward function that computes the reward based on the deviation from the trajectory. The reward is
    computed as the negative exponential of the squared difference between the current state and the trajectory state.
    The reward is computed for the joint positions, joint velocities, relative site positions,
    relative site orientations, and relative site velocities. These sites are specified in the environment properties
    and are placed at key points on the body to mimic the motion of the body.

    """

    def __init__(self, env: Any,
                 sites_for_mimic=None,
                 **kwargs):
        """
        Initialize the DeepMimic reward function.

        Args:
            env (Any): Environment instance.
            sites_for_mimic (List[str], optional): List of site names to mimic. Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        """

        super().__init__(env, **kwargs)

        # reward coefficients
        self._qpos_w_exp = kwargs.get("qpos_w_exp", 10.0)
        self._qvel_w_exp = kwargs.get("qvel_w_exp", 2.0)
        self._rpos_w_exp = kwargs.get("rpos_w_exp", 100.0)
        self._rquat_w_exp = kwargs.get("rquat_w_exp", 10.0)
        self._rvel_w_exp = kwargs.get("rvel_w_exp", 0.1)
        self._qpos_w_sum = kwargs.get("qpos_w_sum", 0.0)
        self._qvel_w_sum = kwargs.get("qvel_w_sum", 0.0)
        self._rpos_w_sum = kwargs.get("rpos_w_sum", 0.5)
        self._rquat_w_sum = kwargs.get("rquat_w_sum", 0.3)
        self._rvel_w_sum = kwargs.get("rvel_w_sum", 0.1)

        # get main body name of the environment
        self.main_body_name = self._info_props["upper_body_xml_name"]
        model = env._model
        self.main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.main_body_name)
        rel_site_names = self._info_props["sites_for_mimic"] if sites_for_mimic is None else sites_for_mimic
        self._rel_site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                                       for name in rel_site_names])
        self._rel_body_ids = np.array([model.site_bodyid[site_id] for site_id in self._rel_site_ids])

        # focus on joints in the observation space
        self._qpos_ind = np.concatenate([obs.data_type_ind for obs in self._obs_container.entries()
                                         if (type(obs) is ObservationType.JointPos) or
                                         (type(obs) is ObservationType.FreeJointPos) or
                                         (type(obs) is ObservationType.EntryFromFreeJointPos) or
                                         (type(obs) is ObservationType.FreeJointPosNoXY)])

        self._qvel_ind = np.concatenate([obs.data_type_ind for obs in self._obs_container.entries()
                                         if (type(obs) is ObservationType.JointVel) or
                                         (type(obs) is ObservationType.EntryFromFreeJointVel) or
                                         (type(obs) is ObservationType.FreeJointVel)])

        # determine the quaternions in qpos.
        quat_in_qpos = []
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                quat = mj_jntid2qposid(i, model)[3:]
                quat_in_qpos.append(quat)
        quat_in_qpos = np.concatenate(quat_in_qpos)
        self._quat_in_qpos = np.array([True if q in quat_in_qpos else False for q in self._qpos_ind])

    @check_traj_provided
    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:
        """
        Computes a deep mimic tracking reward based on the deviation from the trajectory. The reward is computed as the
        negative exponential of the squared difference between the current state and the trajectory state. The reward
        is computed for the joint positions, joint velocities, relative site positions, relative site orientations, and
        relative site velocities.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.

        Raises:
            ValueError: If trajectory handler is not provided.

        """

        # get trajectory data
        traj_data = env.th.traj.data

        # get all quantities from trajectory
        traj_data_single = traj_data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no, backend)
        qpos_traj, qvel_traj = traj_data_single.qpos[self._qpos_ind], traj_data_single.qvel[self._qvel_ind]
        qpos_quat_traj = qpos_traj[self._quat_in_qpos].reshape(-1, 4)
        site_rpos_traj, site_rangles_traj, site_rvel_traj =\
            calculate_relative_site_quatities(traj_data_single, self._rel_site_ids,
                                              self._rel_body_ids, model.body_rootid, backend)

        # get all quantities from the current data
        qpos, qvel = data.qpos[self._qpos_ind], data.qvel[self._qvel_ind]
        qpos_quat = qpos[self._quat_in_qpos].reshape(-1, 4)
        site_rpos, site_rangles, site_rvel = (
            calculate_relative_site_quatities(data, self._rel_site_ids, self._rel_body_ids,
                                              model.body_rootid, backend))

        # calculate distances
        qpos_dist = backend.mean(backend.square(qpos[~self._quat_in_qpos] - qpos_traj[~self._quat_in_qpos]))
        qpos_dist += backend.mean(quaternion_angular_distance(qpos_quat, qpos_quat_traj, backend))
        qvel_dist = backend.mean(backend.square(qvel - qvel_traj))
        rpos_dist = backend.mean(backend.square(site_rpos - site_rpos_traj))
        rquat_dist = backend.mean(backend.square(site_rangles - site_rangles_traj))
        rvel_rot_dist = backend.mean(backend.square(site_rvel[:3] - site_rvel_traj[:3]))
        rvel_lin_dist = backend.mean(backend.square(site_rvel[3:] - site_rvel_traj[3:]))

        # calculate rewards
        qpos_reward = backend.exp(-self._qpos_w_exp*qpos_dist)
        qvel_reward = backend.exp(-self._qvel_w_exp*qvel_dist)
        rpos_reward = backend.exp(-self._rpos_w_exp*rpos_dist)
        rquat_reward = backend.exp(-self._rquat_w_exp*rquat_dist)
        rvel_rot_reward = backend.exp(-self._rvel_w_exp*rvel_rot_dist)
        rvel_lin_reward = backend.exp(-self._rvel_w_exp*rvel_lin_dist)

        total_reward = (self._qpos_w_sum * qpos_reward + self._qvel_w_sum * qvel_reward
                        + self._rpos_w_sum * rpos_reward + self._rquat_w_sum * rquat_reward
                        + self._rvel_w_sum * rvel_rot_reward + self._rvel_w_sum * rvel_lin_reward)

        return total_reward, carry
