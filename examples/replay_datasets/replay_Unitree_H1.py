import numpy as np

from loco_mujoco import LocoEnv


def experiment(seed=0):

    np.random.seed(seed)

    mdp = LocoEnv.make("MjxUnitreeH1.walk.real", disable_arms=False, goal_type="NoGoal")

    mdp.play_trajectory_from_velocity(n_episodes=3, n_steps_per_episode=500, render=True, record=True)


if __name__ == '__main__':
    experiment()
