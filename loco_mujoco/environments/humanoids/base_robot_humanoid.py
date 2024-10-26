from loco_mujoco.environments import LocoEnv
from loco_mujoco.utils import info_property


class BaseRobotHumanoid(LocoEnv):
    """
    Base Class for the Humanoids.

    """

    @info_property
    def sites_for_mimic(self):
        return ["upper_body_mimic", "head_mimic", "pelvis_mimic",
                "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic",
                "left_hip_mimic", "left_knee_mimic", "left_foot_mimic",
                "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
                "right_hip_mimic", "right_knee_mimic", "right_foot_mimic"]
