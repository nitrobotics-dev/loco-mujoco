import numpy as np

from loco_mujoco import LocoEnv


def experiment(seed=0):

    np.random.seed(seed)

    mdp = LocoEnv.make("UnitreeH1.kitty_walk", disable_arms=False, goal_type="NoGoal")

    mdp.play_trajectory(n_episodes=3, n_steps_per_episode=1000, render=True, from_velocity=False)


if __name__ == '__main__':
    experiment()
