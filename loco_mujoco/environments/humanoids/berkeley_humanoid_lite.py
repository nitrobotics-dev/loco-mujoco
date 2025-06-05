# loco_mujoco/environments/humanoids/berkeley_humanoid_lite.py

from typing import List, Tuple, Union
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property

class BerkeleyHumanoidLite(BaseRobotHumanoid):
    """
    LocoMuJoCo wrapper for the Berkeley Humanoid Lite MuJoCo model.

    Default Observation Space
    -------------------------
    Index  Name                         ObservationType     xml_name
    -----  ---------------------------  -----------------   --------------------------
    0–5    q_root                        FreeJointPosNoXY   "base_freejoint"
    6      q_arm_left_shoulder_pitch     JointPos           "arm_left_shoulder_pitch_joint"
    7      q_arm_left_shoulder_roll      JointPos           "arm_left_shoulder_roll_joint"
    8      q_arm_left_shoulder_yaw       JointPos           "arm_left_shoulder_yaw_joint"
    9      q_arm_left_elbow_pitch        JointPos           "arm_left_elbow_pitch_joint"
    10     q_arm_left_elbow_roll         JointPos           "arm_left_elbow_roll_joint"
    11     q_arm_right_shoulder_pitch    JointPos           "arm_right_shoulder_pitch_joint"
    12     q_arm_right_shoulder_roll     JointPos           "arm_right_shoulder_roll_joint"
    13     q_arm_right_shoulder_yaw      JointPos           "arm_right_shoulder_yaw_joint"
    14     q_arm_right_elbow_pitch       JointPos           "arm_right_elbow_pitch_joint"
    15     q_arm_right_elbow_roll        JointPos           "arm_right_elbow_roll_joint"
    16     q_leg_left_hip_roll            JointPos           "leg_left_hip_roll_joint"
    17     q_leg_left_hip_yaw             JointPos           "leg_left_hip_yaw_joint"
    18     q_leg_left_hip_pitch           JointPos           "leg_left_hip_pitch_joint"
    19     q_leg_left_knee_pitch          JointPos           "leg_left_knee_pitch_joint"
    20     q_leg_left_ankle_pitch         JointPos           "leg_left_ankle_pitch_joint"
    21     q_leg_left_ankle_roll          JointPos           "leg_left_ankle_roll_joint"
    22     q_leg_right_hip_roll           JointPos           "leg_right_hip_roll_joint"
    23     q_leg_right_hip_yaw            JointPos           "leg_right_hip_yaw_joint"
    24     q_leg_right_hip_pitch          JointPos           "leg_right_hip_pitch_joint"
    25     q_leg_right_knee_pitch         JointPos           "leg_right_knee_pitch_joint"
    26     q_leg_right_ankle_pitch        JointPos           "leg_right_ankle_pitch_joint"
    27     q_leg_right_ankle_roll         JointPos           "leg_right_ankle_roll_joint"

    28–33  dq_root                       FreeJointVel        "base_freejoint"
    34     dq_arm_left_shoulder_pitch    JointVel            "arm_left_shoulder_pitch_joint"
    35     dq_arm_left_shoulder_roll     JointVel            "arm_left_shoulder_roll_joint"
    36     dq_arm_left_shoulder_yaw      JointVel            "arm_left_shoulder_yaw_joint"
    37     dq_arm_left_elbow_pitch       JointVel            "arm_left_elbow_pitch_joint"
    38     dq_arm_left_elbow_roll        JointVel            "arm_left_elbow_roll_joint"
    39     dq_arm_right_shoulder_pitch   JointVel            "arm_right_shoulder_pitch_joint"
    40     dq_arm_right_shoulder_roll    JointVel            "arm_right_shoulder_roll_joint"
    41     dq_arm_right_shoulder_yaw     JointVel            "arm_right_shoulder_yaw_joint"
    42     dq_arm_right_elbow_pitch      JointVel            "arm_right_elbow_pitch_joint"
    43     dq_arm_right_elbow_roll       JointVel            "arm_right_elbow_roll_joint"
    44     dq_leg_left_hip_roll          JointVel            "leg_left_hip_roll_joint"
    45     dq_leg_left_hip_yaw           JointVel            "leg_left_hip_yaw_joint"
    46     dq_leg_left_hip_pitch         JointVel            "leg_left_hip_pitch_joint"
    47     dq_leg_left_knee_pitch        JointVel            "leg_left_knee_pitch_joint"
    48     dq_leg_left_ankle_pitch       JointVel            "leg_left_ankle_pitch_joint"
    49     dq_leg_left_ankle_roll        JointVel            "leg_left_ankle_roll_joint"
    50     dq_leg_right_hip_roll         JointVel            "leg_right_hip_roll_joint"
    51     dq_leg_right_hip_yaw          JointVel            "leg_right_hip_yaw_joint"
    52     dq_leg_right_hip_pitch        JointVel            "leg_right_hip_pitch_joint"
    53     dq_leg_right_knee_pitch       JointVel            "leg_right_knee_pitch_joint"
    54     dq_leg_right_ankle_pitch      JointVel            "leg_right_ankle_pitch_joint"
    55     dq_leg_right_ankle_roll       JointVel            "leg_right_ankle_roll_joint"

    Default Action Space
    --------------------
    0   arm_left_shoulder_pitch_joint
    1   arm_left_shoulder_roll_joint
    2   arm_left_shoulder_yaw_joint
    3   arm_left_elbow_pitch_joint
    4   arm_left_elbow_roll_joint
    5   arm_right_shoulder_pitch_joint
    6   arm_right_shoulder_roll_joint
    7   arm_right_shoulder_yaw_joint
    8   arm_right_elbow_pitch_joint
    9   arm_right_elbow_roll_joint
    10  leg_left_hip_roll_joint
    11  leg_left_hip_yaw_joint
    12  leg_left_hip_pitch_joint
    13  leg_left_knee_pitch_joint
    14  leg_left_ankle_pitch_joint
    15  leg_left_ankle_roll_joint
    16  leg_right_hip_roll_joint
    17  leg_right_hip_yaw_joint
    18  leg_right_hip_pitch_joint
    19  leg_right_knee_pitch_joint
    20  leg_right_ankle_pitch_joint
    21  leg_right_ankle_roll_joint
    """

    mjx_enabled = False

    def __init__(self,
                 spec: Union[str, MjSpec] = None,
                 observation_spec: List[Observation] = None,
                 actuation_spec: List[str] = None,
                 **kwargs) -> None:
        # If no spec is passed, use our default XML file path
        if spec is None:
            spec = self.get_default_xml_file_path()

        # Load the MuJoCo spec (either from file or MjSpec directly)
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        # Build observations (joint positions + velocities)
        if observation_spec is None:
            observation_spec = self._get_observation_specification(spec)
        else:
            observation_spec = self.parse_observation_spec(observation_spec)

        # Build actions (actuator names)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)
        
        # If using MJX, apply necessary spec modifications here
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(
            spec=spec,
            actuation_spec=actuation_spec,
            observation_spec=observation_spec,
            **kwargs
        )

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """
        Return a list of actuator names exactly as they appear in the XML.
        """
        return [
            "arm_left_shoulder_pitch_joint",
            "arm_left_shoulder_roll_joint",
            "arm_left_shoulder_yaw_joint",
            "arm_left_elbow_pitch_joint",
            "arm_left_elbow_roll_joint",
            "arm_right_shoulder_pitch_joint",
            "arm_right_shoulder_roll_joint",
            "arm_right_shoulder_yaw_joint",
            "arm_right_elbow_pitch_joint",
            "arm_right_elbow_roll_joint",
            "leg_left_hip_roll_joint",
            "leg_left_hip_yaw_joint",
            "leg_left_hip_pitch_joint",
            "leg_left_knee_pitch_joint",
            "leg_left_ankle_pitch_joint",
            "leg_left_ankle_roll_joint",
            "leg_right_hip_roll_joint",
            "leg_right_hip_yaw_joint",
            "leg_right_hip_pitch_joint",
            "leg_right_knee_pitch_joint",
            "leg_right_ankle_pitch_joint",
            "leg_right_ankle_roll_joint",
        ]

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        """
        Return a list of ObservationType instances (q and dq for each joint, plus freejoint).
        """
        obs = []

        # ------------- FREE JOINT (root) -------------
        # xml <freejoint name="base_freejoint"/>
        obs.append(ObservationType.FreeJointPosNoXY("q_root", xml_name="base_freejoint"))
        obs.append(ObservationType.FreeJointVel("dq_root", xml_name="base_freejoint"))

        # ------------- JOINT POS -------------
        joint_list = [
            "arm_left_shoulder_pitch",
            "arm_left_shoulder_roll",
            "arm_left_shoulder_yaw",
            "arm_left_elbow_pitch",
            "arm_left_elbow_roll",
            "arm_right_shoulder_pitch",
            "arm_right_shoulder_roll",
            "arm_right_shoulder_yaw",
            "arm_right_elbow_pitch",
            "arm_right_elbow_roll",
            "leg_left_hip_roll",
            "leg_left_hip_yaw",
            "leg_left_hip_pitch",
            "leg_left_knee_pitch",
            "leg_left_ankle_pitch",
            "leg_left_ankle_roll",
            "leg_right_hip_roll",
            "leg_right_hip_yaw",
            "leg_right_hip_pitch",
            "leg_right_knee_pitch",
            "leg_right_ankle_pitch",
            "leg_right_ankle_roll",
        ]

        for name in joint_list:
            xml_name = f"{name}_joint"      # matches <joint name="..._joint"/>
            obs.append(ObservationType.JointPos(f"q_{name}", xml_name=xml_name))
        # ------------- JOINT VEL -------------
        for name in joint_list:
            xml_name = f"{name}_joint"
            obs.append(ObservationType.JointVel(f"dq_{name}", xml_name=xml_name))

        return obs

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default path to the XML file within the LocoMuJoCo models directory.
        """
        # loco_mujoco.PATH_TO_MODELS points to "loco_mujoco/models"
        return (loco_mujoco.PATH_TO_MODELS / "berkeley_humanoid_lite" / "berkeley_humanoid_lite.xml").as_posix()

    @info_property
    def root_body_name(self) -> str:
        # In the XML, the top-level root body is named "base"
        return "base"

    @info_property
    def upper_body_xml_name(self) -> str:
        # This is used by some terminal/healthy-range handlers; pick the torso link if defined
        # In your XML, there is no explicit "torso" name; we’ll return “base” so it always stays above ground.
        return "base"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        # Must match <freejoint name="base_freejoint"/>
        return "base_freejoint"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        # Example healthy range so training doesn’t terminate if the root dips below 0.2 m
        return (0.4, 1.2)
