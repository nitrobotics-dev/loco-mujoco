import functools
from typing import Any, Dict
from functools import partial
import mujoco
from mujoco import mjx
from flax import struct
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from loco_mujoco.core.mujoco_base import Mujoco, ObservationType, AdditionalCarry
from loco_mujoco.core.utils import MujocoViewer, VideoRecorder


@struct.dataclass
class MjxState:
    data: mjx.Data
    observation: jax.Array
    reward: float
    absorbing: bool
    done: bool
    additional_carry: Any
    info: Dict[str, Any] = struct.field(default_factory=dict)


@struct.dataclass
class MjxAdditionalCarry(AdditionalCarry):
    final_observation: jax.Array
    first_data: mjx.Data
    final_info: Dict[str, Any]


class Mjx(Mujoco):

    def __init__(self, n_envs, **kwargs):

        # call base mujoco env
        super().__init__(**kwargs)

        assert n_envs > 0, "Setting the number of environments smaller than or equal to 0 is not allowed."
        self._n_envs = n_envs

        self.sys = mjx.put_model(self._model)

        # add information to mdp_info
        # todo: remove the n_envs attribute
        self._mdp_info.mjx_env, self._mdp_info.n_envs = True, n_envs

    def mjx_step(self, state: MjxState, action: jax.Array):

        data = state.data
        carry = state.additional_carry

        # reset dones
        state = state.replace(done=jnp.zeros_like(state.done, dtype=bool))

        # modify the observation and the data if needed (does nothing by default)
        cur_obs, data, cur_info, carry = self._step_init(state.observation, data, state.info, carry)

        # preprocess action
        action = self._mjx_preprocess_action(action, data, carry)

        # modify data *before* step if needed (does nothing by default)
        data, carry = self._mjx_simulation_pre_step(data, carry)

        # step in the environment using the action
        ctrl = data.ctrl.at[jnp.array(self._action_indices)].set(action)
        data = data.replace(ctrl=ctrl)
        step_fn = lambda _, x: mjx.step(self.sys, x)
        data = jax.lax.fori_loop(0, self._n_substeps, step_fn, data)

        # modify data *after* step if needed (does nothing by default)
        data, carry = self._mjx_simulation_post_step(data, carry)

        # create the observation
        cur_obs = self._mjx_create_observation(data, carry)

        # modify the observation and the data if needed (does nothing by default)
        cur_obs, data, cur_info, carry = self._mjx_step_finalize(cur_obs, data, cur_info, carry)

        # create info
        cur_info = self._mjx_update_info_dictionary(cur_info, cur_obs, data, carry)

        # check if the next obs is an absorbing state
        absorbing = self._mjx_is_absorbing(cur_obs, cur_info, data, carry)

        # calculate the reward
        reward = self._mjx_reward(state.observation, action, cur_obs, absorbing, cur_info, self._model, data, carry)

        # check if done
        done = self._mjx_is_done(cur_obs, absorbing, cur_info, data, carry)

        state = state.replace(data=data, observation=cur_obs, reward=reward,
                              absorbing=absorbing, done=done, info=cur_info, additional_carry=carry)

        # reset states that are done
        state = self._mjx_reset_in_step(state)

        return state

    def mjx_reset(self, key):
        key, subkey = jax.random.split(key)

        mujoco.mj_resetData(self._model, self._data)
        data = mjx.put_data(self._model, self._data)

        carry = self._init_additional_carry(key, data)

        obs = self._mjx_create_observation(data, carry)
        reward = 0.0
        absorbing = jnp.array(False, dtype=bool)
        done = jnp.array(False, dtype=bool)
        info = self._mjx_reset_info_dictionary(obs, data, subkey)

        return MjxState(data=data, observation=obs, reward=reward, absorbing=absorbing, done=done,
                        info=info, additional_carry=carry)

    def _mjx_reset_in_step(self, state: MjxState):
        carry = state.additional_carry
        data = jax.lax.cond(state.done, lambda: carry.first_data, lambda: state.data)
        final_obs = jnp.where(state.done, state.observation, jnp.zeros_like(state.observation))
        carry = carry.replace(cur_step_in_episode=jnp.where(state.done, 1, carry.cur_step_in_episode + 1),
                              final_observation=final_obs)
        new_obs = self._mjx_create_observation(data, carry)

        return state.replace(data=data, observation=new_obs, additional_carry=carry)

    def _mjx_create_observation(self, data, carry):
        # get the base observation defined in observation_spec and the goal
        obs = self._create_observation_compat(data, jnp)
        return self._mjx_order_observation(obs)

    def _mjx_order_observation(self, obs):
        obs = obs.at[self._obs_indices.concatenated_indices].set(obs)
        return obs

    def _mjx_reset_info_dictionary(self, obs, data, key):
        return {}

    def _mjx_update_info_dictionary(self, info, obs, data, carry):
        return info

    @partial(jax.jit, static_argnums=(0, 6))
    def _mjx_reward(self, obs, action, next_obs, absorbing, info, model, data):
        return 0.0

    def _mjx_is_absorbing(self, obs, info, data, carry):
        return False

    def _mjx_is_done(self, obs, absorbing, info, data, carry):
        done = jnp.greater_equal(carry.cur_step_in_episode, self.info.horizon)
        done = jnp.logical_or(done, absorbing)
        return done

    def _mjx_simulation_pre_step(self, data, carry):
        return data, carry

    def _mjx_simulation_post_step(self, data, carry):
        return data, carry

    def _mjx_preprocess_action(self, action, data, carry):
        return action

    def _mjx_step_init(self, obs, data, info, carry):
        """
        Allows information to be accessed at the end of a step.
        """
        return obs, data, info, carry

    def _mjx_step_finalize(self, obs, data, info, carry):
        """
        Allows information to be accessed at the end of a step.
        """
        return obs, data, info, carry

    def _mjx_set_sim_state(self, data, sample):

        # set the free joint data (can not set qpos twice in replace)
        data = data.replace(
            qpos=data.qpos.at[self._data_indices.free_joint_qpos].set(sample[self._obs_indices.free_joint_qpos]),
            qvel=data.qvel.at[self._data_indices.free_joint_qvel].set(sample[self._obs_indices.free_joint_qvel]))

        return data.replace(
            xpos=data.xpos.at[self._data_indices.body_xpos].set(sample[self._obs_indices.body_xpos].reshape(-1, 3)),
            xquat=data.xquat.at[self._data_indices.body_xquat].set(sample[self._obs_indices.body_xquat].reshape(-1, 4)),
            cvel=data.cvel.at[self._data_indices.body_cvel].set(sample[self._obs_indices.body_cvel].reshape(-1, 6)),
            qpos=data.qpos.at[self._data_indices.joint_qpos].set(sample[self._obs_indices.joint_qpos]),
            qvel=data.qvel.at[self._data_indices.joint_qvel].set(sample[self._obs_indices.joint_qvel]),
            site_xpos=data.site_xpos.at[self._data_indices.site_xpos].set(sample[self._obs_indices.site_xpos].reshape(-1, 3)),
            site_xmat=data.site_xmat.at[self._data_indices.site_xmat].set(sample[self._obs_indices.site_xmat].reshape(-1, 9)))

    def mjx_render(self, state, record=False):
        """
        Renders all environments in parallel.
        """
        if self._viewer is None:
            self._viewer = MujocoViewer(self._model, self.dt, record=record, **self._viewer_params)

        return self._viewer.parallel_render(state, record)

    def mjx_render_trajectory(self, trajectory, record=False, **recorder_params):

        assert len(trajectory) > 0, "Mjx render got provided with an empty trajectory."

        if record:
            recorder = VideoRecorder(**recorder_params)
        else:
            recorder = None

        if self._viewer is None:
            self._viewer = MujocoViewer(self._model, self.dt, record=record, **self._viewer_params)

        # get number of environment per state in trajectory
        n_envs = trajectory[0].data.qpos.shape[0]

        # render each environment
        for i in range(n_envs):
            # for each environment, render all trajectories
            for state in trajectory:
                self._data.qpos, self._data.qvel = state.data.qpos[i, :], state.data.qvel[i, :]
                mujoco.mj_forward(self._model, self._data)
                im = self._viewer.render(self._data, record)
                if recorder:
                    recorder(im)

        if recorder:
            recorder.stop()

    def _init_additional_carry(self, key, data):
        return MjxAdditionalCarry(key=key, cur_step_in_episode=1,
                                  final_observation=jnp.zeros(self.info.observation_space.shape),
                                  first_data=data, final_info={})

    def _process_collision_groups(self, collision_groups):
        if collision_groups is not None and len(collision_groups) > 0:
            raise ValueError("Collisions groups are currently not supported in Mjx.")

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def mjx_env(self):
        return True
