import numpy as np
import jax
from loco_mujoco import LocoEnv

# create the environment and task
env = LocoEnv.make("UnitreeGo2",
                    terminal_state_type="HeightBasedTerminalStateHandler",
                   goal_type="GoalRandomRootVelocity",
                   goal_params=dict(visualize_goal=True),
                   reward_type="TargetVelocityGoalReward")

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
    action = np.random.randn(action_dim) * 0.0
    nstate, reward, absorbing, done, info = env.step(action)

    env.render()
    i += 1