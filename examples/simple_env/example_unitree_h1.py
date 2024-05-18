import jax
import numpy as np
from loco_mujoco import LocoEnv

# create the environment and task
MODEL_OPTION = dict(iterations=100, ls_iterations=50)
env = LocoEnv.make("MjxUnitreeH1.walk", use_absorbing_states=False, random_start=False, model_option_conf=MODEL_OPTION)

# get the dataset for the chosen environment and task
expert_data = env.create_dataset()

action_dim = env.info.action_space.shape[0]

rng = jax.random.key(0)
rng, _rng = jax.random.split(rng)
env.reset(_rng)
env.render()
absorbing = False
i = 0

while True:
    if i == 1000 or absorbing:
        rng, _rng = jax.random.split(rng)
        env.reset(_rng)
        i = 0
    action = np.random.randn(action_dim) * 3
    nstate, reward, absorbing, done, info = env.step(action)

    env.render()
    i += 1
