from typing import Union
import numpy as np
import jax
import mujoco
from flax import struct

from jax.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.core.observations import ObservationType
from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.core.utils.math import (calculate_relative_site_quatities,
                                         quaternion_angular_distance, quat_scalarfirst2scalarlast)
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid, mj_jntname2qvelid, mj_check_collisions


class Reward(StatefulObject):
    """
    Interface to specify a reward function.

    """

    registered = dict()

    def __init__(self, obs_container, info_props, model, data):
        self._obs_container = obs_container
        self._info_props = info_props
        self.initialized = False

    @classmethod
    def get_name(cls):
        return cls.__name__

    def init_from_traj(self, traj_handler=None):
        pass

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend):
        """
        Compute the reward.

        Args:
            state (np.ndarray): last state;
            action (np.ndarray): applied action;
            next_state (np.ndarray): current state.

        Returns:
            The reward for the current transition and the carry.

        """
        raise NotImplementedError

    @classmethod
    def register(cls):
        """
        Register a reward in the reward list.

        """
        env_name = cls.get_name()

        if env_name not in Reward.registered:
            Reward.registered[env_name] = cls

    @staticmethod
    def list_registered():
        """
        List registered rewards.

        Returns:
             The list of the registered rewards.

        """
        return list(Reward.registered.keys())

    @property
    def requires_trajectory(self):
        return False


class NoReward(Reward):
    """
    A reward function that returns always 0.

    """

    def __call__(self, state, action, next_state, absorbing, info, model, env, data, carry, backend):
        return 0, carry


class PosReward(Reward):

    def __init__(self, obs_container, pos_idx, **kwargs):
        self._pos_idx = pos_idx
        super().__init__(obs_container, **kwargs)

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend):
        pos = state[self._pos_idx]
        return pos, carry


class CustomReward(Reward):

    def __init__(self, obs_container, reward_callback=None, **kwargs):
        self._reward_callback = reward_callback
        super().__init__(obs_container, **kwargs)

    @staticmethod
    def get_name():
        return "custom"

    def reset(self, data, backend, traj_data, traj_state):
        pass

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend):
        if self._reward_callback is not None:
            return self._reward_callback(state, action, next_state), carry
        else:
            return 0, carry


class TargetXVelocityReward(Reward):

    def __init__(self, obs_container, target_velocity, free_jnt_name="dq_root", **kwargs):
        self._target_vel = target_velocity
        self._free_jnt_name = free_jnt_name
        self._x_vel_idx = obs_container[free_jnt_name].obs_ind[0]
        super().__init__(obs_container, **kwargs)

    def reset(self, data, backend, trajectory, traj_state):
        pass

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend):
        x_vel = backend.squeeze(state[self._x_vel_idx])
        return backend.exp(-backend.square(x_vel - self._target_vel)), carry


class TargetVelocityTrajReward(Reward):

    def __init__(self, obs_container, free_jnt_name="dq_root", **kwargs):
        self._free_jnt_name = free_jnt_name
        self._vel_idx = obs_container[free_jnt_name].data_type_ind
        super().__init__(obs_container, **kwargs)

    def reset(self, data, backend, trajectory, traj_state):
        pass

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend):

        # get trajectory data
        traj_data = env.th.traj.data

        # vel in data
        x_vel = backend.squeeze(data.qvel[self._vel_idx])

        # vel in trajectory
        traj_data_sample = traj_data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no)
        target_vel = backend.squeeze(traj_data_sample.qvel[self._vel_idx])

        return backend.exp(-backend.square(x_vel - target_vel)), carry


class TargetVelocityGoalReward(Reward):

    def __init__(self, obs_container, free_jnt_vel_name="dq_root", w_exp=10.0, **kwargs):
        free_jnt_vel = obs_container[free_jnt_vel_name]
        self._w_exp = w_exp

        assert type(free_jnt_vel) is ObservationType.FreeJointVel, (f"FreeJointVel observation is required "
                                                                    f"for the reward{self.__class__.__name__}.")

        self._vel_idx = obs_container[free_jnt_vel_name].data_type_ind

        # find the goal velocity observation
        try:
            goal_vel_obs = obs_container["GoalRandomRootVelocity"]
        except KeyError:
            raise ValueError(f"GoalRandomRootVelocity is the required goal for the reward {self.__class__.__name__}")

        self._goal_vel_idx = goal_vel_obs.data_type_ind
        super().__init__(obs_container, **kwargs)
        self._free_jnt_name = self._info_props["root_free_joint_xml_name"]

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend):
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        goal_state = getattr(carry.observation_states, "GoalRandomRootVelocity")

        # get root orientation
        root_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self._free_jnt_name)
        assert root_jnt_id != -1, f"Joint {self._free_jnt_name} not found in the model."
        root_jnt_qpos_start_id = model.jnt_qposadr[root_jnt_id]
        root_qpos = backend.squeeze(data.qpos[root_jnt_qpos_start_id:root_jnt_qpos_start_id+7])
        root_quat = R.from_quat(root_qpos[3:7])

        # get current local vel of root
        x_vel_global = backend.squeeze(data.qvel[self._vel_idx])[:3]
        x_vel_local = root_quat.as_matrix().T @ x_vel_global

        # vel in goal
        goal_vel = backend.squeeze(goal_state.goal_vel)

        tracking_reward = backend.exp(-self._w_exp*backend.mean(backend.square(x_vel_local[:2] - goal_vel)))

        action_penalty = 0.25*backend.mean(backend.abs(action))

        return tracking_reward - action_penalty, carry


class MimicReward(Reward):

    def __init__(self, obs_container, sites_for_mimic=None, **kwargs):

        super().__init__(obs_container, **kwargs)

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
        model = kwargs["model"]
        info_props = kwargs["info_props"]
        self.main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.main_body_name)
        rel_site_names = info_props["sites_for_mimic"] if sites_for_mimic is None else sites_for_mimic
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

        # determine the quaternions in qpos. Note: ObservationType.EntryFromFreeJointPos for quaternions not supported
        quat_in_qpos = []
        for obs in self._obs_container.entries():
            if type(obs) is ObservationType.FreeJointPos:
                quat = obs.data_type_ind[3:]
                assert len(quat) == 4, "Quaternions must have 4 elements"
                quat_in_qpos.append(quat)
            elif type(obs) is ObservationType.FreeJointPosNoXY:
                quat = obs.data_type_ind[1:]
                assert len(quat) == 4, "Quaternions must have 4 elements"
                quat_in_qpos.append(quat)

        quat_in_qpos = np.concatenate(quat_in_qpos)
        self._quat_in_qpos = np.array([True if q in quat_in_qpos else False for q in self._qpos_ind])

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend):

        # get trajectory data
        traj_data = env.th.traj.data

        # get all quantities from trajectory
        traj_data_single = traj_data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no)
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

    @property
    def requires_trajectory(self):
        return True


@struct.dataclass
class LocomotionRewardState:
    last_qvel: Union[np.ndarray, jax.Array]
    last_action: Union[np.ndarray, jax.Array]
    time_since_last_touchdown: Union[np.ndarray, jax.Array]


class LocomotionReward(TargetVelocityGoalReward):

    def __init__(self, obs_container, **kwargs):
        super().__init__(obs_container, **kwargs)
        model = kwargs["model"]
        self._free_joint_qpos_ind = np.array(mj_jntname2qposid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qvel_ind = np.array(mj_jntname2qvelid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qpos_mask = np.zeros(model.nq, dtype=bool)
        self._free_joint_qpos_mask[self._free_joint_qpos_ind] = True
        self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
        self._free_joint_qvel_mask[self._free_joint_qvel_ind] = True

        self._floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        foot_names = ["RL_foot", "RR_foot", "FL_foot", "FR_foot"]
        self._foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in foot_names]

        # reward coefficients
        self._tracking_rew_coeff = kwargs.get("tracking_rew_coeff", 1.0)
        self._z_vel_coeff = kwargs.get("z_vel_coeff", 2.0)
        self._roll_pitch_vel_coeff = kwargs.get("roll_pitch_vel_coeff", 5e-2)
        self._roll_pitch_pos_coeff = kwargs.get("roll_pitch_pos_coeff", 2e-1)
        self._nominal_joint_pos_coeff = kwargs.get("nominal_joint_pos_coeff", 2e-1)
        self._nominal_joint_pos = kwargs.get("nominal_joint_pos", np.zeros(1))
        self._joint_vel_coeff = kwargs.get("joint_vel_coeff", 0.0)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 2e-7)
        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 2e-5)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 1e-2)
        self._air_time_max = kwargs.get("air_time_max", 0.0)
        self._air_time_coeff = kwargs.get("air_time_coeff", 0.0)

    def init_state(self, env, key, model, data, backend):
        return LocomotionRewardState(last_qvel=data.qvel, last_action=backend.zeros(env.info.action_space.shape[0]),
                                     time_since_last_touchdown=backend.zeros(len(self._foot_ids)))

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend):

        if backend == np:
            R = np_R
        else:
            R = jnp_R

        # get current reward state
        reward_state = carry.reward_state

        # get global pose quantities
        global_pose_root = data.qpos[self._free_joint_qpos_ind]
        global_pos_root = global_pose_root[:3]
        global_quat_root = global_pose_root[3:]
        global_rot = R.from_quat(quat_scalarfirst2scalarlast(global_quat_root, backend))

        # get global velocity quantities
        global_vel_root = data.qvel[self._free_joint_qvel_ind]
        global_vel_root_lin = global_vel_root[:3]
        global_vel_root_ang = global_vel_root[3:]

        # get local velocity quantities
        local_vel_root_lin = global_rot.inv().apply(global_vel_root[:3])
        local_vel_root_ang = global_rot.inv().apply(global_vel_root[3:])

        # velocity reward
        z_vel_reward = self._z_vel_coeff * -(backend.square(local_vel_root_lin[2]))
        roll_pitch_vel_reward = self._roll_pitch_vel_coeff * -backend.square(local_vel_root_ang[:2]).sum()

        # position reward
        euler = global_rot.as_euler("xyz")
        roll_pitch_reward = self._roll_pitch_pos_coeff * -backend.square(euler[:2]).sum()

        # nominal joint pos reward
        joint_qpos = data.qpos[~self._free_joint_qpos_mask]
        joint_qpos_reward = self._nominal_joint_pos_coeff * -backend.square(joint_qpos - self._nominal_joint_pos).sum()

        # joint velocity reward
        joint_vel = data.qvel[~self._free_joint_qvel_mask]
        joint_vel_reward = self._joint_vel_coeff * -backend.square(joint_vel).sum()

        # joint acceleration reward
        last_joint_vel = reward_state.last_qvel[~self._free_joint_qvel_mask]
        joint_vel = data.qvel[~self._free_joint_qvel_mask]
        acceleration_norm = backend.sum(backend.square(joint_vel - last_joint_vel) / env.dt)
        acceleration_reward = self._joint_acc_coeff * -acceleration_norm

        # joint torque reward
        torque_norm = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))
        torque_reward = self._joint_torque_coeff * -torque_norm

        # action rate reward
        action_rate_norm = backend.sum(backend.square(action - reward_state.last_action))
        action_rate_reward = self._action_rate_coeff * -action_rate_norm

        # air time reward
        air_time_reward = 0.0
        foots_on_ground = backend.zeros(len(self._foot_ids))
        tslt = reward_state.time_since_last_touchdown.copy()
        for i, f_id in enumerate(self._foot_ids):
            foot_on_ground = mj_check_collisions(f_id, self._floor_id, data, backend)
            if backend == np:
                foots_on_ground[i] = foot_on_ground
            else:
                foots_on_ground = foots_on_ground.at[i].set(foot_on_ground)

            if backend == np:
                if foot_on_ground:
                    air_time_reward += (tslt[i] - self._air_time_max)
                    tslt[i] = 0.0
                else:
                    tslt[i] += env.dt
            else:
                tslt_i, air_time_reward = jax.lax.cond(foot_on_ground,
                                                       lambda: (0.0, air_time_reward + tslt[i] - self._air_time_max),
                                                       lambda: (tslt[i] + env.dt, air_time_reward))
                tslt = tslt.at[i].set(tslt_i)

        air_time_reward = self._air_time_coeff * air_time_reward

        # total reward
        tracking_reward, _ = super().__call__(state, action, next_state, absorbing, info,
                                              env, model, data, carry, backend)
        penality_rewards = (z_vel_reward + roll_pitch_vel_reward + roll_pitch_reward + joint_qpos_reward
                            + joint_vel_reward + acceleration_reward + torque_reward + action_rate_reward
                            + air_time_reward)
        total_reward = tracking_reward + penality_rewards
        total_reward = backend.maximum(total_reward, 0.0)

        # update reward state
        reward_state = reward_state.replace(last_qvel=data.qvel, last_action=action, time_since_last_touchdown=tslt)
        carry = carry.replace(reward_state=reward_state)

        return total_reward, carry
