import numpy as np
import mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.core.utils.math import calc_rel_positions, calc_rel_quaternions, calc_rel_body_velocities, quat2angle, calculate_relative_site_quatities


class Reward:
    """
    Interface to specify a reward function.

    """

    registered = dict()

    def __init__(self, obs_container, info_props, model, data):
        self._obs_container = obs_container
        self._info_props = info_props
        self.initialized = False

    @staticmethod
    def get_name():
        raise NotImplementedError

    def init_from_traj(self, traj_handler=None):
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

    @property
    def requires_trajectory(self):
        return False


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

    def __init__(self, obs_container, pos_idx, **kwargs):
        self._pos_idx = pos_idx
        super().__init__(obs_container, **kwargs)

    @staticmethod
    def get_name():
        return "x_pos"

    def __call__(self, state, action, next_state, absorbing, info, model, data, carry, backend, trajectory=None):
        pos = state[self._pos_idx]
        return pos


class CustomReward(Reward):

    def __init__(self, obs_container, reward_callback=None, **kwargs):
        self._reward_callback = reward_callback
        super().__init__(obs_container, **kwargs)

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

    def __init__(self, obs_container, target_velocity, vel_obs_dict_key, **kwargs):
        self._target_vel = target_velocity
        self._vel_obs_dict_key = vel_obs_dict_key
        self._x_vel_idx = obs_container[vel_obs_dict_key].obs_ind
        super().__init__(obs_container, **kwargs)

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


class MimicReward(Reward):

    def __init__(self, obs_dict, qpos_w_exp=10.0, qvel_w_exp=2.0, rpos_w_exp=100.0,
                 rquat_w_exp=10.0, rvel_w_exp=0.1, qpos_w_sum=0.0, qvel_w_sum=0.0, rpos_w_sum=0.5,
                 rquat_w_sum=0.3, rvel_w_sum=0.1, standardize=False, sites_for_mimic=None, **kwargs):

        super().__init__(obs_dict, **kwargs)

        self._qpos_w_exp = qpos_w_exp
        self._qvel_w_exp = qvel_w_exp
        self._rpos_w_exp = rpos_w_exp
        self._rquat_w_exp = rquat_w_exp
        self._rvel_w_exp = rvel_w_exp
        self._qpos_w_sum = qpos_w_sum
        self._qvel_w_sum = qvel_w_sum
        self._rpos_w_sum = rpos_w_sum
        self._rquat_w_sum = rquat_w_sum
        self._rvel_w_sum = rvel_w_sum
        self._standardize = standardize


        # to be defined in init_from_traj
        self._obs_qpos_ind = None
        self._obs_qvel_ind = None
        self._traj_qpos_ind = None
        self._traj_qvel_ind = None
        self._joints_in_obs = None
        self._mean_qpos = None
        self._mean_qvel = None
        self._std_qpos = None
        self._std_qvel = None
        self._std_rpos = None
        self._std_rquat = None
        self._std_rvel = None

        # get main body name of the environment
        self.main_body_name = self._info_props["upper_body_xml_name"]
        model = kwargs["model"]
        info_props = kwargs["info_props"]
        self.main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.main_body_name)
        rel_site_names = info_props["sites_for_mimic"] if sites_for_mimic is None else sites_for_mimic
        self._rel_site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                                       for name in rel_site_names])
        self._rel_body_ids = np.array([model.site_bodyid[site_id] for site_id in self._rel_site_ids])

    @staticmethod
    def get_name():
        return "mimic"

    def init_from_traj(self, traj_handler=None):
        # todo: remove shift by 2 once the observation space does not include xy anymore
        self._obs_qpos_ind = np.concatenate([obs.obs_ind for obs in self._obs_container.entries()
                                             if isinstance(obs, ObservationType.JointPos)])[2:] - 2
        self._obs_qvel_ind = np.concatenate([obs.obs_ind for obs in self._obs_container.entries()
                                             if isinstance(obs, ObservationType.JointVel)])
        self._traj_qpos_ind = np.concatenate([obs.traj_data_type_ind for obs in self._obs_container.entries()
                                             if isinstance(obs, ObservationType.JointPos)])[2:]
        self._traj_qvel_ind = np.concatenate([obs.traj_data_type_ind for obs in self._obs_container.entries()
                                             if isinstance(obs, ObservationType.JointVel)])

        # setup standardizer
        qpos = traj_handler.traj_data.qpos
        qvel = traj_handler.traj_data.qvel

        # site_rpos = calc_rel_positions(traj_handler.traj_data.site_xpos, traj_handler.traj_data.site_xpos[self.main_body_id])
        # site_xmat = traj_handler.traj_data.site_xmat.reshape(-1, 9)
        # orig_shape = site_xmat.shape
        # from scipy.spatial.transform import Rotation as R
        # site_xquat = R.from_matrix(site_xmat).as_quat().reshape(orig_shape[0], orig_shape[1], 4)
        # site_rquat = calc_rel_quaternions(site_xquat, site_xquat[:, self.main_body_id], np)
        #
        #
        #
        # rpos = traj_handler.traj_data.rpos
        # rquat = traj_handler.traj_data.rquat
        # rvel = traj_handler.traj_data.rvel
        #
        # if self._standardize:
        #     self._std_qpos = np.std(qpos[:, self._traj_qpos_ind], axis=(0,))
        #     self._std_qvel = np.std(qvel[:, self._traj_qvel_ind], axis=(0,))
        #     self._std_rpos = np.std(rpos, axis=(0,))
        #     self._std_rquat = np.std(rquat, axis=(0,))
        #     self._std_rvel = np.std(rvel, axis=(0,))
        #
        #     # avoid division by zero in dims where std is zero
        #     self._std_qpos = np.where(self._std_qpos == 0.0, 1.0, self._std_qpos)
        #     self._std_qvel = np.where(self._std_qvel == 0.0, 1.0, self._std_qvel)
        #     self._std_rpos = np.where(self._std_rpos == 0.0, 1.0, self._std_rpos)
        #     self._std_rvel = np.where(self._std_rvel == 0.0, 1.0, self._std_rvel)
        #
        # else:
        #     self._std_qpos = 1.0
        #     self._std_qvel = 1.0
        #     self._std_rpos = 1.0
        #     self._std_rquat = 1.0
        #     self._std_rvel = 1.0

    def __call__(self, state, action, next_state, absorbing, info, model, data, carry, backend, traj_data=None):

        # todo: check that this is correct and code the part where the obs_container init_from_traj is called
        # get qpos and qvel from trajectory
        t = carry.traj_state.subtraj_step_no - carry.traj_state.subtraj_step_no_init
        traj_data_single = traj_data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no)
        qpos_traj, qvel_traj = traj_data_single.qpos[self._traj_qpos_ind], traj_data_single.qvel[self._traj_qvel_ind]
        site_rpos_traj, site_rangles_traj, site_rvel_traj =\
            calculate_relative_site_quatities(traj_data_single, self._rel_site_ids,
                                              self._rel_body_ids, model.body_rootid, backend)

        # get the same quantities from the current data
        qpos, qvel = data.qpos[self._traj_qpos_ind], data.qvel[self._traj_qvel_ind]
        site_rpos, site_rangles, site_rvel = (
            calculate_relative_site_quatities(data, self._rel_site_ids, self._rel_body_ids,
                                              model.body_rootid, backend))

        # calculate distances
        qpos_dist = backend.mean(backend.square(qpos - qpos_traj))
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

        #
        # # get qpos and qvel from observation
        # qpos_obs = next_state[self._obs_qpos_ind]
        # qvel_obs = next_state[self._obs_qvel_ind]
        #
        # # get body information of current state
        # rpos = calc_rel_positions(data.xpos, data.xpos[self.main_body_id], backend)
        # rquat = calc_rel_quaternions(data.xquat, data.xquat[self.main_body_id], backend)
        # rvel = calc_rel_body_velocities(data.cvel, data.xmat[self.main_body_id], backend)
        #
        # # standardize
        # qpos_obs = qpos_obs / self._std_qpos
        # qvel_obs = qvel_obs / self._std_qvel
        # qpos_traj = qpos_traj / self._std_qpos
        # qvel_traj = qvel_traj / self._std_qvel
        # rpos = rpos / self._std_rpos
        # rvel = rvel / self._std_rvel
        # rpos_traj = rpos_traj / self._std_rpos
        # rvel_traj = rvel_traj / self._std_rvel
        #
        # # calculate distances
        # qpos_dist = backend.mean(backend.square(qpos_obs - qpos_traj))
        # qvel_dist = backend.mean(backend.square(qvel_obs - qvel_traj))
        # rpos_dist = backend.mean(backend.square(rpos - rpos_traj))
        # rquat_dist = backend.mean(backend.square(quat2angle(rquat) - quat2angle(rquat_traj)))
        # rvel_dist = backend.mean(backend.square(rvel - rvel_traj))
        #
        # # calculate rewards
        # qpos_reward = backend.exp(-self._qpos_w_exp*qpos_dist)
        # qvel_reward = backend.exp(-self._qvel_w_exp*qvel_dist)
        # rpos_reward = backend.exp(-self._rpos_w_exp*rpos_dist)
        # rquat_reward = backend.exp(-self._rquat_w_exp*rquat_dist)
        # rvel_reward = backend.exp(-self._rvel_w_exp*rvel_dist)

        return (self._qpos_w_sum * qpos_reward + self._qvel_w_sum * qvel_reward + self._rpos_w_sum * rpos_reward
                + self._rquat_w_sum * rquat_reward + self._rvel_w_sum * rvel_rot_reward + self._rvel_w_sum * rvel_lin_reward)

    @property
    def requires_trajectory(self):
        return True

