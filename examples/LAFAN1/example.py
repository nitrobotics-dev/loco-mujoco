import numpy as np
from loco_mujoco import LAFAN1ImitationFactory
from loco_mujoco.datasets.humanoids.LAFAN1 import LAFAN1_LOCOMOTION_DATASETS


def experiment(seed=0):

    np.random.seed(seed)

    # example: load many different locomotion datasets from AMASS
    # env = LAFAN1ImitationFactory.make("UnitreeH1", LAFAN1_LOCOMOTION_DATASETS)

    # # example: load just two datasets
    env = LAFAN1ImitationFactory.make("UnitreeH1", ["fight1_subject2.csv"],
                                      goal_type="GoalTrajMimic", goal_params=dict(visualize_goal=True))

    env.play_trajectory(n_episodes=3, n_steps_per_episode=1000, render=True, from_velocity=False)


if __name__ == '__main__':
    experiment()
