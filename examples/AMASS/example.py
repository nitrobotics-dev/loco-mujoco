import numpy as np
from loco_mujoco import AMASSImitationFactory
from loco_mujoco.smpl import AMASS_LOCOMOTION_DATASETS


def experiment(seed=0):

    np.random.seed(seed)

    # example: load many different locomotion datasets from AMASS
    # env = AMASSImitationFactory.make("UnitreeH1", AMASS_LOCOMOTION_DATASETS)

    # example: load just two datasets
    env = AMASSImitationFactory.make("UnitreeH1",
                                     ['KIT/3/walking_slow08_poses', 'KIT/167/walking_run02_poses'])

    env.play_trajectory(n_episodes=3, n_steps_per_episode=1000, render=True, from_velocity=False)


if __name__ == '__main__':
    experiment()
