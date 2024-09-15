import numpy as np

from loco_mujoco.core import ObservationType
from loco_mujoco.utils.math import mat2angle_xy


class Reward:
    """
    Interface to specify a reward function.

    """

    registered = dict()

    def __init__(self, obs_dict, info_props):
        self._obs_dict = obs_dict
        self._info_props = info_props
        self.initialized = False

    @staticmethod
    def get_name():
        raise NotImplementedError

    def init_from_traj(self, trajectories=None):
        pass

    def __call__(self, state, action, next_state, absorbing, info, model, data, carry, backend, trajectory=None):
        """
        Compute the reward.

        Args:
            state (np.ndarray): last state;
            action (np.ndarray): applied action;
            next_state (np.ndarray): current state.

        Returns:
            The reward for the current transition.

        """
        pass

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


class NoReward(Reward):
    """
    A reward function that returns always 0.

    """

    @staticmethod
    def get_name():
        return "no_reward"

    def __call__(self, state, action, next_state, absorbing, info, model, data, carry, backend, trajectory=None):
        return 0


class PosReward(Reward):

    def __init__(self, obs_dict, pos_idx, **kwargs):
        self._pos_idx = pos_idx
        super().__init__(obs_dict, **kwargs)

    @staticmethod
    def get_name():
        return "x_pos"

    def __call__(self, state, action, next_state, absorbing, info, model, data, carry, backend, trajectory=None):
        pos = state[self._pos_idx]
        return pos


class CustomReward(Reward):

    def __init__(self, obs_dict, reward_callback=None, **kwargs):
        self._reward_callback = reward_callback
        super().__init__(obs_dict, **kwargs)

    @staticmethod
    def get_name():
        return "custom"

    def reset(self, data, backend, trajectory, traj_state):
        pass

    def __call__(self, state, action, next_state, absorbing, info, model, data, carry, backend, trajectory=None):
        if self._reward_callback is not None:
            return self._reward_callback(state, action, next_state)
        else:
            return 0


class TargetVelocityReward(Reward):

    def __init__(self, obs_dict, target_velocity, vel_obs_dict_key, **kwargs):
        self._target_vel = target_velocity
        self._vel_obs_dict_key = vel_obs_dict_key
        self._x_vel_idx = obs_dict[vel_obs_dict_key].obs_ind
        super().__init__(obs_dict, **kwargs)

    @staticmethod
    def get_name():
        return "target_velocity"

    def reset(self, data, backend, trajectory, traj_state):
        pass

    def __call__(self, state, action, next_state, absorbing, info, model, data, carry, backend, trajectory=None):
        x_vel = backend.squeeze(state[self._x_vel_idx])
        return backend.exp(-backend.square(x_vel - self._target_vel))


class MultiTargetVelocityReward(Reward):

    def __init__(self, target_velocity, x_vel_idx, env_id_len, scalings, **kwargs):
        self._target_vel = target_velocity
        self._env_id_len = env_id_len
        self._scalings = scalings
        self._x_vel_idx = x_vel_idx

    def __call__(self, state, action, next_state, absorbing, info, model, data, carry, backend, trajectory=None):
        x_vel = state[self._x_vel_idx]
        env_id = state[-self._env_id_len:]

        # convert binary array to index
        # todo: evaluate if this working fine with jax backend
        ind = backend.packbits(env_id.astype(int), bitorder='big') >> (8 - env_id.shape[0])
        ind = ind[0]
        scaling = self._scalings[ind]

        # calculate target vel
        target_vel = self._target_vel * scaling

        return backend.exp(- backend.square(x_vel - target_vel))


class VelocityVectorReward(Reward):

    def __init__(self, x_vel_idx, y_vel_idx, angle_idx, goal_vel_idx, **kwargs):
        self._x_vel_idx = x_vel_idx
        self._y_vel_idx = y_vel_idx
        self._angle_idx = angle_idx
        self._goal_vel_idx = goal_vel_idx

    @staticmethod
    def get_name():
        return "velocity_vector"

    def __call__(self, state, action, next_state, absorbing, info, model, data, carry, backend, trajectory=None):

        # get current velocity vector in x-y-plane
        curr_velocity_xy = backend.array([state[self._x_vel_idx], state[self._y_vel_idx]])

        # get desired velocity vector in x-y-plane
        cos_sine = state[self._angle_idx]
        des_vel = state[self._goal_vel_idx] * cos_sine

        return backend.exp(-5.0*backend.linalg.norm(curr_velocity_xy - des_vel))


class MimicQPosReward(Reward):

    def __init__(self, obs_dict, **kwargs):
        super().__init__(obs_dict, **kwargs)

        # to be defined in init_from_traj
        self._obs_qpos_ind = None
        self._obs_qvel_ind = None
        self._traj_qpos_ind = None
        self._traj_qvel_ind = None

    @staticmethod
    def get_name():
        return "mimic_qpos"

    def init_from_traj(self, trajectories=None):
        # todo: remove shift by 2 once the observation space does not include xy anymore
        self._obs_qpos_ind = np.concatenate([obs.obs_ind for obs in self._obs_dict.values()
                                             if isinstance(obs, ObservationType.JointPos)])[2:] - 2
        self._obs_qvel_ind = np.concatenate([obs.obs_ind for obs in self._obs_dict.values()
                                             if isinstance(obs, ObservationType.JointVel)]) - 2
        self._traj_qpos_ind = trajectories.qpos_ind[2:]
        self._traj_qvel_ind = trajectories.qvel_ind

    def __call__(self, state, action, next_state, absorbing, info, model, data, carry, backend, trajectory=None):
        traj_sample = backend.ravel(trajectory[:, carry.traj_state.traj_no, carry.traj_state.subtraj_step_no])
        qpos_obs = state[self._obs_qpos_ind]
        qvel_obs = state[self._obs_qvel_ind]
        qpos_traj = backend.squeeze(traj_sample[self._traj_qpos_ind])
        qvel_traj = backend.squeeze(traj_sample[self._traj_qvel_ind])
        qpos_dist = backend.exp(-5.0*backend.linalg.norm(qpos_obs - qpos_traj))
        qvel_dist = backend.exp(-0.1*backend.linalg.norm(qvel_obs - qvel_traj))
        return qpos_dist + qvel_dist

