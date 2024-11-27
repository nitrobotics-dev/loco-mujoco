from loco_mujoco.environments import LocoEnv
from loco_mujoco.utils import info_property


class BaseRobotQuadruped(LocoEnv):
    """
    Base Class for the Quadrupeds.

    """

    @info_property
    def sites_for_mimic(self):
        return []
