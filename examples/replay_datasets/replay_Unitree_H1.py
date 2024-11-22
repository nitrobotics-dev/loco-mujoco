import numpy as np

from loco_mujoco import LocoEnv


def experiment(seed=0):

    np.random.seed(seed)

    mdp = LocoEnv.make("MjxUnitreeH1.random_walk.real", disable_arms=False, goal_type="NoGoal")

    mdp.play_trajectory(n_episodes=3, n_steps_per_episode=1000, render=True, from_velocity=True)


if __name__ == '__main__':
    experiment()
