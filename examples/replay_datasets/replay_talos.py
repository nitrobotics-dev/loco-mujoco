import numpy as np
from loco_mujoco import LocoEnv

np.random.seed(0)
mdp = LocoEnv.make("Talos.walk")

mdp.play_trajectory(n_episodes=30, n_steps_per_episode=500)
