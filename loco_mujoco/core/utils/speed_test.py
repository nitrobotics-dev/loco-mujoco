import time
import jax
import numpy as np
from loco_mujoco.core import Mujoco, Mjx, ObservationType, TestMjx

# just some randomly chosen observations
observation_spec = [("b_pelvis", "pelvis", ObservationType.BODY_POS),
                    ("q_pelvis_tx", "pelvis_tx", ObservationType.JOINT_POS),
                    ("q_l_arm_shy", "l_arm_shy", ObservationType.JOINT_POS),
                    ("q_l_arm_shx", "l_arm_shx", ObservationType.JOINT_POS),
                    ("q_l_arm_shz", "l_arm_shz", ObservationType.JOINT_POS),
                    ("q_left_elbow", "left_elbow", ObservationType.JOINT_VEL),
                    ("q_r_arm_shy", "r_arm_shy", ObservationType.JOINT_VEL)]

# all actuators that are in the xml
action_spec = ["back_bkz_actuator", "l_arm_shy_actuator", "l_arm_shx_actuator",
               "l_arm_shz_actuator", "left_elbow_actuator", "r_arm_shy_actuator", "r_arm_shx_actuator",
               "r_arm_shz_actuator", "right_elbow_actuator", "hip_flexion_r_actuator",
               "hip_adduction_r_actuator", "hip_rotation_r_actuator", "knee_angle_r_actuator",
               "ankle_angle_r_actuator", "hip_flexion_l_actuator", "hip_adduction_l_actuator",
               "hip_rotation_l_actuator", "knee_angle_l_actuator", "ankle_angle_l_actuator"]

# create env
env = TestMjx(xml_file="/home/moore/PycharmProjects/MjxTest/data/unitree_h1/h1.xml",
          actuation_spec=action_spec,
          observation_spec=observation_spec,
          horizon=1000,
          n_envs=4000,
          gamma=0.99)

action_dim = env.info.action_space.shape[0]
global_key = jax.random.PRNGKey(165416)  # Random seed is explicit in JAX
global_key, new_key = jax.random.split(global_key)
env.reset(new_key)
env.render()

while True:
    for i in range(500):
        env.step(np.random.randn(action_dim))
        env.render()
    global_key, new_key = jax.random.split(global_key)
    env.reset(new_key)



LOGGING_FREQUENCY = 100000
keys = jax.random.split(global_key, env.info.n_envs + 1)
global_key, env_keys = keys[0], keys[1:]


def sample():
    global global_key
    global_key, subkey = jax.random.split(global_key)
    action = jax.random.uniform(subkey, minval=env.info.action_space.low, maxval=env.info.action_space.high,
                                shape=(env.info.n_envs, action_dim))
    return action


sample_X = jax.jit(sample)
state = env.mjx_reset(env_keys)
step = 0
previous_time = time.time()

while True:

    action = sample_X()
    state = env.mjx_step(state, action)

    step += env.info.n_envs
    if step % LOGGING_FREQUENCY == 0:
        current_time = time.time()
        print(f"{int(LOGGING_FREQUENCY / (current_time - previous_time))} steps per second.")
        previous_time = current_time