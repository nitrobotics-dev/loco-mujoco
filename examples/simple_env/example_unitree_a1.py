import numpy as np
import jax
from loco_mujoco import LocoEnv

# load yaml as dict:
path = "/home/moore/PycharmProjects/loco-mujoco/examples/jax_rl/confs/domain_randomization/default_dom_rand_conf.yaml"
with open(path, 'r') as file:
    import yaml
    default_dom_rand_conf = yaml.load(file, Loader=yaml.FullLoader)

# create the environment and task
env = LocoEnv.make("MjxUnitreeA1",
                   terminal_state_type="HeightBasedTerminalStateHandler",
                   goal_type="GoalRandomRootVelocity", goal_params=dict(visualize_goal=True),
                   domain_randomization_type="DefaultRandomizer",
                   domain_randomization_params=default_dom_rand_conf,
                   terrain_type="RoughTerrain",
                   reward_type="LocomotionReward")

# get the dataset for the chosen environment and task
#expert_data = env.create_dataset()

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
    action = np.random.randn(action_dim)*0.0
    nstate, reward, absorbing, done, info = env.step(action)

    env.render()
    i += 1
