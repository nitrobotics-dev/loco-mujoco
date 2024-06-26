import jax
import time
from loco_mujoco import LocoEnv


env = LocoEnv.make("MjxUnitreeG1.walk")

# optionally replay trajectory
#env.play_trajectory(n_episodes=10)

NENVS = 4000

key = jax.random.key(0)
keys = jax.random.split(key, NENVS + 1)
key, env_keys = keys[0], keys[1:]

# jit and vmap all functions needed
rng_reset = jax.jit(jax.vmap(env.mjx_reset))
rng_step = jax.jit(jax.vmap(env.mjx_step))
rng_sample_uni_action = jax.jit(jax.vmap(env.sample_action_space))

# reset env
state = rng_reset(env_keys)

# optionally collect rollouts for rendering
rollout = []

step = 0
previous_time = time.time()
LOGGING_FREQUENCY = 100000
while True:

    keys = jax.random.split(key, NENVS + 1)
    key, action_keys = keys[0], keys[1:]
    action = rng_sample_uni_action(action_keys)
    state = rng_step(state, action)

    #rollout.append(state)

    step += NENVS
    if step % LOGGING_FREQUENCY == 0:
        current_time = time.time()
        print(f"{int(LOGGING_FREQUENCY / (current_time - previous_time))} steps per second.")
        previous_time = current_time

# Simulate and display video.
env.mjx_render_trajectory(rollout)
