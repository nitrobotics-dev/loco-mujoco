import os
import jax
import time
import numpy as np
from loco_mujoco import LocoEnv
from loco_mujoco.environments.humanoids.unitreeH1_mjx import MjxUnitreeH1

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# load yaml as dict:
path = parent_dir+"/jax_rl/config/domain_randomization/default_dom_rand_conf.yaml"
with open(path, 'r') as file:
    import yaml
    default_dom_rand_conf = yaml.load(file, Loader=yaml.FullLoader)

control_params = {
    "p_gain": np.array([200, 200, 200, 300, 40, 200, 200, 200, 300, 40, 300, 100, 100, 100, 100, 100, 100, 100, 100]),
    "d_gain": np.array([5, 5, 5, 6, 2, 5, 5, 5, 6, 2, 6, 2, 2, 2, 2, 2, 2, 2, 2]),
    "nominal_joint_positions": np.array([0.0]*19)
}


# env = LocoEnv.make("MjxUnitreeH1.walk",
#                    reward_type="LocomotionReward",
#                    terrain_type="RoughTerrain",
#                    terminal_state_type="HeightBasedTerminalStateHandler",
#                    # domain_randomization_type="DefaultRandomizer",
#                    goal_type="GoalRandomRootVelocity", goal_params=dict(visualize_goal=True),
#                    n_envs=50, disable_arms=False)

env = MjxUnitreeH1(reward_type="LocomotionReward",
                   terrain_type="RoughTerrain",
                   terminal_state_type="HeightBasedTerminalStateHandler",
                   domain_randomization_type="DefaultRandomizer",
                   domain_randomization_params=default_dom_rand_conf,
                   goal_type="GoalRandomRootVelocity", goal_params=dict(visualize_goal=False),
                   control_type="PDControl", control_params=control_params,
                   n_envs=2, disable_arms=False)

key = jax.random.key(0)
keys = jax.random.split(key, env.info.n_envs + 1)
key, env_keys = keys[0], keys[1:]

# jit and vmap all functions needed
rng_reset = jax.jit(jax.vmap(env.mjx_reset))
rng_step = jax.jit(jax.vmap(env.mjx_step))
rng_sample_uni_action = jax.jit(jax.vmap(env.sample_action_space))

# reset env
state = rng_reset(env_keys)
print(state.data.qpos.shape[0])
step = 0
previous_time = time.time()
LOGGING_FREQUENCY = 100000
i = 0
while i < 100000:

    keys = jax.random.split(key, env.info.n_envs + 1)
    key, action_keys = keys[0], keys[1:]
    action = rng_sample_uni_action(action_keys)
    print('before rng_step')
    state = rng_step(state, action)
    print('after rng_step')
    #print(state)
    env.mjx_render(state)

    step += env.info.n_envs
    if step % LOGGING_FREQUENCY == 0:
        current_time = time.time()
        print(f"{int(LOGGING_FREQUENCY / (current_time - previous_time))} steps per second.")
        previous_time = current_time

    i+=1

