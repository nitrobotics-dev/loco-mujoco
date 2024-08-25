import numpy as np
import jax
from copy import deepcopy
from loco_mujoco import LocoEnv


# create the environment and task
env = LocoEnv.make("MjxUnitreeH1.run", goal_type="goal_traj_arrow", goal_params=dict(visualize_goal=False))

# get the dataset for the chosen environment and task
expert_data = env.create_dataset()

action_dim = env.info.action_space.shape[0]

key = jax.random.key(0)
key, _rng = jax.random.split(key)

env.reset(_rng)

env.render()
absorbing = False
i = 0

while True:
    if i == 1000 or absorbing:
        key, _rng = jax.random.split(key)
        env.reset(_rng)
        i = 0
    action = np.random.randn(action_dim)
    nstate, reward, absorbing, done, info = env.step(action)

    env.render()
    i += 1
