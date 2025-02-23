import numpy as np
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf
from loco_mujoco.datasets.humanoids.LAFAN1 import LAFAN1_LOCOMOTION_DATASETS


def experiment(seed=0):

    np.random.seed(seed)


    # # example: load just two datasets
    env = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf("dance2_subject4"), n_substeps=20)

    env.play_trajectory(n_episodes=3, n_steps_per_episode=1000, render=True, from_velocity=False)


if __name__ == '__main__':
    experiment()
