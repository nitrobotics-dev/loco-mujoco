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

from loco_mujoco.core.mujoco_base import Mujoco, ObservationType
from loco_mujoco.core.utils import  MujocoViewer


@struct.dataclass
class MjxState:
    data: mjx.Data
    observation: jax.Array
    reward: float
    absorbing: bool
    done: bool
    final_observation: jax.Array
    first_data: mjx.Data
    info: Dict[str, Any] = struct.field(default_factory=dict)


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

        # reset dones
        state = state.replace(done=jnp.zeros_like(state.done, dtype=bool))

        # preprocess action
        action = self._mjx_preprocess_action(action, data)

        # modify data *before* step if needed
        data = self._mjx_simulation_pre_step(data)

        # step in the environment using the action
        ctrl = data.ctrl.at[jnp.array(self._action_indices)].set(action)
        data = data.replace(ctrl=ctrl)
        step_fn = lambda _, x: mjx.step(self.sys, x)
        data = jax.lax.fori_loop(0, self._n_substeps, step_fn, data)

        # modify data *after* step if needed
        data = self._mjx_simulation_post_step(data)

        # create the observation
        cur_obs = self._mjx_create_observation(data)

        # modify the observation and the data if needed
        cur_obs, data = self._mjx_step_finalize(cur_obs, data)

        # create info
        cur_info = self._mjx_update_info_dictionary(state.info, cur_obs, data)

        # check if the next obs is an absorbing state
        absorbing = self._mjx_is_absorbing(cur_obs, cur_info, data)

        # calculate the reward
        reward = self._mjx_reward(state.observation, action, cur_obs, absorbing, cur_info, self._model, data)

        # check if done
        done = jnp.greater_equal(state.info["cur_step_in_episode"], self.info.horizon)
        done = jnp.logical_or(done, absorbing)

        state = state.replace(data=data, observation=cur_obs, reward=reward,
                              absorbing=absorbing, done=done, info=cur_info)

        # reset states that are done
        state = self._mjx_reset_in_step(state)

        return state

    def mjx_reset(self, key):
        key, subkey = jax.random.split(key)

        mujoco.mj_resetData(self._model, self._data)
        data = mjx.put_data(self._model, self._data)

        obs = self._mjx_create_observation(data)
        reward = 0.0
        absorbing = jnp.array(False, dtype=bool)
        done = jnp.array(False, dtype=bool)
        info = self._mjx_reset_info_dictionary(obs, data, subkey)
        info["key"] = key

        return MjxState(data=data, observation=obs, reward=reward, absorbing=absorbing, done=done,
                        first_data=data, final_observation=jnp.zeros_like(obs), info=info)

    def _mjx_reset_in_step(self, state: MjxState):

        data = jax.lax.cond(state.done, lambda: state.first_data, lambda: state.data)
        final_obs = jnp.where(state.done, state.observation, jnp.zeros_like(state.observation))
        state.info["cur_step_in_episode"] = jnp.where(state.done, 1, state.info["cur_step_in_episode"] + 1)
        new_obs = self._mjx_create_observation(data)

        return state.replace(data=data, observation=new_obs, final_observation=final_obs)

    def _mjx_create_observation(self, data):
        return self._create_observation_compat(data, jnp)

    def _mjx_reset_info_dictionary(self, obs, data, key):
        info = {"cur_step_in_episode": 1,
                "final_observation": jnp.zeros_like(obs),
                "final_info": {"cur_step_in_episode": 1},
                }
        return info

    def _mjx_update_info_dictionary(self, info, obs, data):
        return info

    @partial(jax.jit, static_argnums=(0, 6))
    def _mjx_reward(self, obs, action, next_obs, absorbing, info, model, data):
        return 0.0

    def _mjx_is_absorbing(self, obs, info, data):
        return False

    def _mjx_simulation_pre_step(self, data):
        return data

    def _mjx_simulation_post_step(self, data):
        return data

    def _mjx_preprocess_action(self, action, data):
        return action

    def _mjx_step_finalize(self, obs, data):
        """
        Allows information to be accessed at the end of a step.
        """
        return obs, data

    def _mjx_set_sim_state(self, data, sample):

        return data.replace(
            xpos=data.xpos.at[self._data_body_xpos_ind, :].set(sample[self._body_xpos_range].reshape(-1, 3)),
            xquat=data.xquat.at[self._data_body_xquat_ind, :].set(sample[self._body_xquat_range].reshape(-1, 4)),
            cvel=data.cvel.at[self._data_body_cvel_ind, :].set(sample[self._body_cvel_range].reshape(-1, 6)),
            qpos=data.qpos.at[self._data_joint_qpos_ind].set(sample[self._joint_qpos_range]),
            qvel=data.qvel.at[self._data_joint_qvel_ind].set(sample[self._joint_qvel_range]),
            site_xpos=data.site_xpos.at[self._data_site_xpos_ind, :].set(sample[self._site_xpos_range].reshape(-1, 3)),
            site_xmat=data.site_xmat.at[self._data_site_xmat_ind, :].set(sample[self._site_xmat_range].reshape(-1, 9)))


        # # set the body pos
        # xpos = data.xpos.at[self._data_body_xpos_ind, :].set(sample[self._body_xpos_range].reshape(-1, 3))
        # # set the body orientation
        # xquat = data.xquat.at[self._data_body_xquat_ind, :].set(sample[self._body_xquat_range].reshape(-1, 4))
        # # set the body velocity
        # cvel = data.cvel.at[self._data_body_cvel_ind, :].set(sample[self._body_cvel_range].reshape(-1, 6))
        # # set the joint positions
        # qpos = data.qpos.at[self._data_joint_qpos_ind].set(sample[self._joint_qpos_range])
        # # set the joint velocities
        # qvel = data.qvel.at[self._data_joint_qvel_ind].set(sample[self._joint_qvel_range])
        # # set the site positions
        # site_xpos = data.site_xpos.at[self._data_site_xpos_ind, :].set(sample[self._site_xpos_range].reshape(-1, 3))
        # # set the site rotation
        # site_xmat = data.site_xmat.at[self._data_site_xmat_ind, :].set(sample[self._site_xmat_range].reshape(-1, 9))
        #
        # return data.replace(xpos=xpos, xquat=xquat, cvel=cvel, qpos=qpos,
        #                     qvel=qvel, site_xpos=site_xpos, site_xmat=site_xmat)

    def mjx_render_trajectory(self, trajectory, record=False):

        assert len(trajectory) > 0, "Mjx render got provided with an empty trajectory."

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
                self._viewer.render(self._data, record)

    def _process_collision_groups(self, collision_groups):
        if collision_groups is not None and len(collision_groups) > 0:
            raise ValueError("Collisions groups are currently not supported in Mjx.")

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def mjx_env(self):
        return True


# if __name__ == "__main__":
#
#
#     observation_spec = [("b_pelvis", "pelvis", ObservationType.BODY_POS),
#                         ("q_pelvis_tx", "pelvis_tx", ObservationType.JOINT_POS),
#                         ("q_l_arm_shy", "l_arm_shy", ObservationType.JOINT_POS),
#                         ("q_l_arm_shx", "l_arm_shx", ObservationType.JOINT_POS),
#                         ("q_l_arm_shz", "l_arm_shz", ObservationType.JOINT_POS),
#                         ("q_left_elbow", "left_elbow", ObservationType.JOINT_POS),
#                         ("q_r_arm_shy", "r_arm_shy", ObservationType.JOINT_POS)]
#
#     action_spec = ["back_bkz_actuator", "l_arm_shy_actuator", "l_arm_shx_actuator",
#                    "l_arm_shz_actuator", "left_elbow_actuator", "r_arm_shy_actuator", "r_arm_shx_actuator",
#                    "r_arm_shz_actuator", "right_elbow_actuator", "hip_flexion_r_actuator",
#                    "hip_adduction_r_actuator", "hip_rotation_r_actuator", "knee_angle_r_actuator",
#                    "ankle_angle_r_actuator", "hip_flexion_l_actuator", "hip_adduction_l_actuator",
#                    "hip_rotation_l_actuator", "knee_angle_l_actuator", "ankle_angle_l_actuator"]
#
#     env = Mjx(xml_file="/home/moore/PycharmProjects/MjxTest/data/unitree_h1/h1.xml",
#               actuation_spec=action_spec,
#               observation_spec=observation_spec,
#               horizon=1000,
#               n_envs=4000,
#               gamma=0.99)
#
#     action_dim = env.info.action_space.shape[0]
#
#     # env.reset()
#     # env.render()
#     #
#     # while True:
#     #     for i in range(500):
#     #         env.step(np.random.randn(action_dim))
#     #         env.render()
#     #     env.reset()
#     #
#     # exit()
#
#     LOGGING_FREQUENCY = 100000
#
#     def sample():
#         key = random.PRNGKey(758493)  # Random seed is explicit in JAX
#         action = random.uniform(key, shape=(env.info.n_envs, env.info.action_space.shape[0]))
#         return action
#
#
#     sample_X = jax.jit(sample)
#     ###
#     import time
#
#     previous_time = time.time()
#     step = 0
#     ###
#
#     keys = jax.random.PRNGKey(165416)  # Random seed is explicit in JAX
#     keys = jax.random.split(keys, env.info.n_envs + 1)
#     keys, env_keys = keys[0], keys[1:]
#
#     state = env.mjx_reset(env_keys)
#     i = 0
#     #rollout = []
#     while True:
#         # action = np.random.uniform(env.action_space.low, env.action_space.high, size=(env.nr_envs, env.action_space.shape[0]))
#         key = random.PRNGKey(758493)  # Random seed is explicit in JAX
#         action = random.uniform(key, shape=(env.n_envs, env.info.action_space.shape[0]))
#         #action = sample_X()
#         state = env.mjx_step(state, action)
#
#         #rollout.append(state)
#
#         #print(i)
#         i += 1
#         # print(observation)
#
#         ###
#         step += env.info.n_envs
#         if step % LOGGING_FREQUENCY == 0:
#             current_time = time.time()
#             print(f"{int(LOGGING_FREQUENCY / (current_time - previous_time))} steps per second")
#             previous_time = current_time
        ###

    # # Simulate and display video.
    # frames = env.render(rollout, camera="track")
    #
    # import cv2
    # fps = 1.0/env.dt  # Set frames per second
    # delay = int(1000 / fps)  # Calculate delay between frames in milliseconds
    #
    # # Display the video
    # for frame in frames:
    #     cv2.imshow('Video', frame)
    #     if cv2.waitKey(delay) & 0xFF == ord('q'):
    #         break
    #
    # cv2.destroyAllWindows()
