import numpy as np
from loco_mujoco import LocoEnv
import gymnasium as gym

from loco_mujoco.task_factories import DefaultDatasetConf, LAFAN1DatasetConf

# create the environment and task
env = gym.make("LocoMujoco", env_name="SkeletonTorque", render_mode="human",
               default_dataset_conf=DefaultDatasetConf("walk"),
               lafan1_dataset_conf=LAFAN1DatasetConf("walk1_subject1"),
               goal_type="GoalTrajMimicv2", goal_params=dict(visualize_goal=True),
               reward_type="MimicReward")

# get the dataset for the chosen environment and task --> useful for GAIL or AMP
expert_data = env.unwrapped.create_dataset()

action_dim = env.action_space.shape[0]

env.reset()
env.render()
terminated = False
i = 0

while True:
    if i == 1000 or terminated:
        env.reset()
        i = 0
    action = np.random.randn(action_dim)
    nstate, reward, terminated, truncated, info = env.step(action)

    env.render()
    i += 1
