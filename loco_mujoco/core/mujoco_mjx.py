from typing import Any, Dict
import mujoco
from mujoco import mjx
from flax import struct
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from loco_mujoco.core.mujoco_base import Mujoco, ObservationType


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

        self._n_envs = n_envs
        self.sys = mjx.put_model(self._model)
        self.mjx_reset = jax.jit(jax.vmap(self._mjx_reset))
        self.mjx_step = jax.jit(jax.vmap(self._mjx_step))

        # add information to mdp_info
        self._mdp_info.mjx_env, self._mdp_info.n_envs = True, n_envs

    def reset(self):
        return self._reset(backend=np)

    def step(self, action):
        return self._step(action, backend=np)

    def _mjx_step(self, state: MjxState, action: jax.Array):

        data = state.data

        # reset dones
        state = state.replace(done=jnp.zeros_like(state.done, dtype=bool))

        # preprocess action
        action = self._preprocess_action(action, data, self.backend)

        # modify data *before* step if needed
        data = self._simulation_pre_step(data, self.backend)

        # step in the environment using the action
        # todo: what is xs?
        data, _ = jax.lax.scan(f=lambda data, _: (mjx.step(self.sys, data.replace(ctrl=action)), None),
                               init=data, xs=(), length=self._n_substeps)

        # modify data *after* step if needed
        data = self._simulation_post_step(data, self.backend)

        # create the observation
        next_obs = self._create_observation(data, self.backend)

        # modify the observation and the data if needed
        next_obs, data = self._step_finalize(next_obs, data, self.backend)

        # check if the next obs is an absorbing state
        absorbing = self._is_absorbing(next_obs, data, self.backend)

        # calculate the reward
        reward = self._reward(state.observation, action, next_obs, absorbing, data, self.backend)

        # check if done
        done = jnp.greater_equal(state.info["cur_step_in_episode"], self.info.horizon)
        done = jnp.logical_or(done, absorbing)

        #info = self._create_info_dictionary(next_obs)
        state.info["cur_step_in_episode"] += 1

        state = state.replace(data=data, observation=next_obs, reward=reward, absorbing=absorbing, done=done)

        # reset states that are done
        state = self._mjx_reset_in_step(state)

        return state

    def _mjx_reset(self, key):
        key, subkey = jax.random.split(key)

        # self._data is not modified, no reset need
        #mujoco.mj_resetData(self._model, self._data)
        # todo: double check that resetting
        qpos = self.sys.qpos0
        qvel = jnp.zeros(self.sys.nv)
        data = mjx.put_data(self._model, self._data)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.sys.nu))
        data = mjx.forward(self.sys, data)

        obs = self._create_observation(data, self.backend)
        # todo: activate this and add traj resetting
        #obs = self._modify_observation(obs)
        #obs, data = self.setup(obs, data)

        reward = 0.0
        absorbing = False
        done = False
        info = self._mjx_reset_info_dictionary(obs, data, self.backend, subkey=subkey)

        return MjxState(data=data, observation=obs, reward=reward, absorbing=absorbing, done=done,
                        first_data=data, final_observation=jnp.zeros_like(obs), info=info)

    def _mjx_reset_in_step(self, state: MjxState):

        def where_done(x, y):
            done = state.done
            return jnp.where(done, x, y)

        data = jax.tree.map(where_done, state.first_data, state.data)
        final_obs = where_done(state.observation, jnp.zeros_like(state.observation))
        state.info["cur_step_in_episode"] = where_done(0, state.info["cur_step_in_episode"])
        new_obs = self._create_observation(data, self.backend)

        return state.replace(data=data, observation=new_obs, final_observation=final_obs)

    def _mjx_reset_info_dictionary(self, obs, data, backend, **kwargs):
        info = {"cur_step_in_episode": 0,
                "final_observation": jnp.zeros_like(obs),
                "final_info": {"cur_step_in_episode": 0},
                "key": kwargs["subkey"]
                }
        return info

    def _mjx_modify_info_dictionary(self, info, obs, data, backend):
        return info

    def mjx_render_trajectory(self, trajectory, height: int = 480, width: int = 640, camera=None):

        # todo: include cv2 viewer, and add optional saving (using the same saving mechanisms as the orig viewer)
        renderer = mujoco.Renderer(self._model, height=height, width=width)
        camera = camera or -1

        def get_image(state: MjxState):
            d = mujoco.MjData(self._model)
            d.qpos, d.qvel = state.data.qpos, state.data.qvel
            mujoco.mj_forward(self._model, d)
            renderer.update_scene(d, camera=camera)
            return renderer.render()

        if isinstance(trajectory, list):
            return [get_image(s) for s in trajectory]

        return get_image(trajectory)

    def _process_collision_groups(self, collision_groups):
        if collision_groups is not None:
            raise ValueError("Collisions groups are currently not supported in Mjx.")

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def backend(self):
        return jnp

    @property
    def mjx_env(self):
        return True


if __name__ == "__main__":


    observation_spec = [("b_pelvis", "pelvis", ObservationType.BODY_POS),
                        ("q_pelvis_tx", "pelvis_tx", ObservationType.JOINT_POS),
                        ("q_l_arm_shy", "l_arm_shy", ObservationType.JOINT_POS),
                        ("q_l_arm_shx", "l_arm_shx", ObservationType.JOINT_POS),
                        ("q_l_arm_shz", "l_arm_shz", ObservationType.JOINT_POS),
                        ("q_left_elbow", "left_elbow", ObservationType.JOINT_POS),
                        ("q_r_arm_shy", "r_arm_shy", ObservationType.JOINT_POS)]

    action_spec = ["back_bkz_actuator", "l_arm_shy_actuator", "l_arm_shx_actuator",
                   "l_arm_shz_actuator", "left_elbow_actuator", "r_arm_shy_actuator", "r_arm_shx_actuator",
                   "r_arm_shz_actuator", "right_elbow_actuator", "hip_flexion_r_actuator",
                   "hip_adduction_r_actuator", "hip_rotation_r_actuator", "knee_angle_r_actuator",
                   "ankle_angle_r_actuator", "hip_flexion_l_actuator", "hip_adduction_l_actuator",
                   "hip_rotation_l_actuator", "knee_angle_l_actuator", "ankle_angle_l_actuator"]

    env = Mjx(xml_file="/home/moore/PycharmProjects/MjxTest/data/unitree_h1/h1.xml",
              actuation_spec=action_spec,
              observation_spec=observation_spec,
              horizon=1000,
              n_envs=4000,
              gamma=0.99)

    action_dim = env.info.action_space.shape[0]

    env.reset()
    env.render()

    while True:
        for i in range(500):
            env.step(np.random.randn(action_dim))
            env.render()
        env.reset()

    exit()

    LOGGING_FREQUENCY = 100000

    def sample():
        key = random.PRNGKey(758493)  # Random seed is explicit in JAX
        action = random.uniform(key, shape=(env.info.n_envs, env.info.action_space.shape[0]))
        return action


    sample_X = jax.jit(sample)
    ###
    import time

    previous_time = time.time()
    step = 0
    ###

    keys = jax.random.PRNGKey(165416)  # Random seed is explicit in JAX
    keys = jax.random.split(keys, env.info.n_envs + 1)
    keys, env_keys = keys[0], keys[1:]

    state = env.mjx_reset(env_keys)
    i = 0
    #rollout = []
    while True:
        # action = np.random.uniform(env.action_space.low, env.action_space.high, size=(env.nr_envs, env.action_space.shape[0]))
        key = random.PRNGKey(758493)  # Random seed is explicit in JAX
        action = random.uniform(key, shape=(env.n_envs, env.info.action_space.shape[0]))
        #action = sample_X()
        state = env.mjx_step(state, action)

        #rollout.append(state)

        #print(i)
        i += 1
        # print(observation)

        ###
        step += env.info.n_envs
        if step % LOGGING_FREQUENCY == 0:
            current_time = time.time()
            print(f"{int(LOGGING_FREQUENCY / (current_time - previous_time))} steps per second")
            previous_time = current_time
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
