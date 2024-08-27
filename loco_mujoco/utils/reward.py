import numpy as np
from abc import ABC, abstractmethod

from loco_mujoco.utils.math import mat2angle_xy


class RewardInterface(ABC):
    """
    Interface to specify a reward function.

    """

    registered_rewards = dict()

    def __init__(self, obs_dict, goal_dict):
        self._obs_dict = obs_dict
        self._goal_dict = goal_dict
        self.initialized = False

    @staticmethod
    @abstractmethod
    def get_name():
        pass

    @abstractmethod
    def reset(self, data, backend, trajectory, traj_state):
        pass

    @abstractmethod
    def __call__(self, state, action, next_state, absorbing, info, model, data, backend):
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

        if env_name not in RewardInterface.registered_rewards:
            RewardInterface.registered_rewards[env_name] = cls

    @staticmethod
    def list_registered():
        """
        List registered rewards.

        Returns:
             The list of the registered rewards.

        """
        return list(RewardInterface.registered_rewards.keys())


class NoReward(RewardInterface):
    """
    A reward function that returns always 0.

    """

    @staticmethod
    def get_name():
        return "no_reward"

    def reset(self, data, backend, trajectory, traj_state):
        pass

    def __call__(self, state, action, next_state, absorbing, info, model, data, backend):
        return 0


class PosReward(RewardInterface):

    def __init__(self, obs_dict, goal_dict, pos_idx):
        self._pos_idx = pos_idx
        super().__init__(obs_dict, goal_dict)

    @staticmethod
    def get_name():
        return "x_pos"

    def reset(self, data, backend, trajectory, traj_state):
        pass

    def __call__(self, state, action, next_state, absorbing, info, model, data, backend):
        pos = state[self._pos_idx]
        return pos


class CustomReward(RewardInterface):

    def __init__(self, obs_dict, goal_dict, reward_callback=None):
        self._reward_callback = reward_callback
        super().__init__(obs_dict, goal_dict)

    @staticmethod
    def get_name():
        return "custom"

    def reset(self, data, backend, trajectory, traj_state):
        pass

    def __call__(self, state, action, next_state, absorbing, info, model, data, backend):
        if self._reward_callback is not None:
            return self._reward_callback(state, action, next_state)
        else:
            return 0


class TargetVelocityReward(RewardInterface):

    def __init__(self, obs_dict, goal_dict, target_velocity, vel_obs_dict_key):
        self._target_vel = target_velocity
        self._vel_obs_dict_key = vel_obs_dict_key
        self._x_vel_idx = obs_dict[vel_obs_dict_key].obs_ind
        super().__init__(obs_dict, goal_dict)

    @staticmethod
    def get_name():
        return "target_velocity"

    def reset(self, data, backend, trajectory, traj_state):
        pass

    def __call__(self, state, action, next_state, absorbing, info, model, data, backend):
        x_vel = backend.squeeze(state[self._x_vel_idx])
        return backend.exp(-backend.square(x_vel - self._target_vel))


class MultiTargetVelocityReward(RewardInterface):

    def __init__(self, target_velocity, x_vel_idx, env_id_len, scalings):
        self._target_vel = target_velocity
        self._env_id_len = env_id_len
        self._scalings = scalings
        self._x_vel_idx = x_vel_idx

    def __call__(self, state, action, next_state, absorbing, info, model, data, backend):
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


class VelocityVectorReward(RewardInterface):

    def __init__(self, x_vel_idx, y_vel_idx, angle_idx, goal_vel_idx):
        self._x_vel_idx = x_vel_idx
        self._y_vel_idx = y_vel_idx
        self._angle_idx = angle_idx
        self._goal_vel_idx = goal_vel_idx

    @staticmethod
    def get_name():
        return "velocity_vector"

    def reset(self, data, backend, trajectory, traj_state):
        pass

    def __call__(self, state, action, next_state, absorbing, info, model, data, backend):

        # get current velocity vector in x-y-plane
        curr_velocity_xy = backend.array([state[self._x_vel_idx], state[self._y_vel_idx]])

        # get desired velocity vector in x-y-plane
        cos_sine = state[self._angle_idx]
        des_vel = state[self._goal_vel_idx] * cos_sine

        return backend.exp(-5.0*backend.linalg.norm(curr_velocity_xy - des_vel))
