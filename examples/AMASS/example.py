import numpy as np
from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf, AMASSDatasetConf
from loco_mujoco.smpl import AMASS_LOCOMOTION_DATASETS


def experiment(seed=0):

    np.random.seed(seed)

    env = ImitationFactory.make("UnitreeH1",
                                amass_dataset_conf=AMASSDatasetConf([
                                    "DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses",
                                    "KIT/12/WalkInClockwiseCircle11_poses",
                                    "HUMAN4D/HUMAN4D/Subject3_Medhi/INF_JumpingJack_S3_01_poses",
                                    'KIT/359/walking_fast05_poses']))


    env.play_trajectory(n_episodes=10, n_steps_per_episode=10000, render=True)


if __name__ == '__main__':
    experiment()
