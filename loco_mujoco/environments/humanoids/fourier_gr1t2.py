import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class FourierGR1T2(BaseRobotHumanoid):

    """

    Description
    ------------


    Tasks
    -------------


    Dataset Types
    -----------------


    Observation Space
    -----------------


    Action Space
    ------------


    Rewards
    --------


    Initial States
    ---------------


    Terminal States
    ----------------

    Methods
    ---------

    """

    mjx_enabled = False

    def __init__(self, spec=None, observation_spec=None, action_spec=None, **kwargs):
        """
        Constructor.

        """

        if spec is None:
            spec = self.get_default_xml_file_path()

        # load the model specification
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        # get the observation and action specification
        if observation_spec is None:
            # get default
            observation_spec = self._get_observation_specification(spec)
        else:
            # parse
            observation_spec = self.parse_observation_spec(observation_spec)
        if action_spec is None:
            action_spec = self._get_action_specification(spec)

        # modify the specification if needed
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(spec, action_spec, observation_spec, enable_mjx=self.mjx_enabled,
                         **kwargs)

    @staticmethod
    def _get_observation_specification(spec: MjSpec):
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            A list of observation types.
        """
        observation_spec = [
            # ------------- JOINT POS -------------
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
            ObservationType.JointPos("q_joint_left_hip_roll", xml_name="joint_left_hip_roll"),
            ObservationType.JointPos("q_joint_left_hip_yaw", xml_name="joint_left_hip_yaw"),
            ObservationType.JointPos("q_joint_left_hip_pitch", xml_name="joint_left_hip_pitch"),
            ObservationType.JointPos("q_joint_left_knee_pitch", xml_name="joint_left_knee_pitch"),
            ObservationType.JointPos("q_joint_left_ankle_pitch", xml_name="joint_left_ankle_pitch"),
            ObservationType.JointPos("q_joint_left_ankle_roll", xml_name="joint_left_ankle_roll"),
            ObservationType.JointPos("q_joint_right_hip_roll", xml_name="joint_right_hip_roll"),
            ObservationType.JointPos("q_joint_right_hip_yaw", xml_name="joint_right_hip_yaw"),
            ObservationType.JointPos("q_joint_right_hip_pitch", xml_name="joint_right_hip_pitch"),
            ObservationType.JointPos("q_joint_right_knee_pitch", xml_name="joint_right_knee_pitch"),
            ObservationType.JointPos("q_joint_right_ankle_pitch", xml_name="joint_right_ankle_pitch"),
            ObservationType.JointPos("q_joint_right_ankle_roll", xml_name="joint_right_ankle_roll"),
            ObservationType.JointPos("q_joint_waist_yaw", xml_name="joint_waist_yaw"),
            ObservationType.JointPos("q_joint_waist_pitch", xml_name="joint_waist_pitch"),
            ObservationType.JointPos("q_joint_waist_roll", xml_name="joint_waist_roll"),
            ObservationType.JointPos("q_joint_head_pitch", xml_name="joint_head_pitch"),
            ObservationType.JointPos("q_joint_head_roll", xml_name="joint_head_roll"),
            ObservationType.JointPos("q_joint_head_yaw", xml_name="joint_head_yaw"),
            ObservationType.JointPos("q_joint_left_shoulder_pitch", xml_name="joint_left_shoulder_pitch"),
            ObservationType.JointPos("q_joint_left_shoulder_roll", xml_name="joint_left_shoulder_roll"),
            ObservationType.JointPos("q_joint_left_shoulder_yaw", xml_name="joint_left_shoulder_yaw"),
            ObservationType.JointPos("q_joint_left_elbow_pitch", xml_name="joint_left_elbow_pitch"),
            ObservationType.JointPos("q_joint_left_wrist_yaw", xml_name="joint_left_wrist_yaw"),
            ObservationType.JointPos("q_joint_left_wrist_roll", xml_name="joint_left_wrist_roll"),
            ObservationType.JointPos("q_joint_left_wrist_pitch", xml_name="joint_left_wrist_pitch"),
            ObservationType.JointPos("q_joint_right_shoulder_pitch", xml_name="joint_right_shoulder_pitch"),
            ObservationType.JointPos("q_joint_right_shoulder_roll", xml_name="joint_right_shoulder_roll"),
            ObservationType.JointPos("q_joint_right_shoulder_yaw", xml_name="joint_right_shoulder_yaw"),
            ObservationType.JointPos("q_joint_right_elbow_pitch", xml_name="joint_right_elbow_pitch"),
            ObservationType.JointPos("q_joint_right_wrist_yaw", xml_name="joint_right_wrist_yaw"),
            ObservationType.JointPos("q_joint_right_wrist_roll", xml_name="joint_right_wrist_roll"),
            ObservationType.JointPos("q_joint_right_wrist_pitch", xml_name="joint_right_wrist_pitch"),

            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            ObservationType.JointVel("dq_joint_left_hip_roll", xml_name="joint_left_hip_roll"),
            ObservationType.JointVel("dq_joint_left_hip_yaw", xml_name="joint_left_hip_yaw"),
            ObservationType.JointVel("dq_joint_left_hip_pitch", xml_name="joint_left_hip_pitch"),
            ObservationType.JointVel("dq_joint_left_knee_pitch", xml_name="joint_left_knee_pitch"),
            ObservationType.JointVel("dq_joint_left_ankle_pitch", xml_name="joint_left_ankle_pitch"),
            ObservationType.JointVel("dq_joint_left_ankle_roll", xml_name="joint_left_ankle_roll"),
            ObservationType.JointVel("dq_joint_right_hip_roll", xml_name="joint_right_hip_roll"),
            ObservationType.JointVel("dq_joint_right_hip_yaw", xml_name="joint_right_hip_yaw"),
            ObservationType.JointVel("dq_joint_right_hip_pitch", xml_name="joint_right_hip_pitch"),
            ObservationType.JointVel("dq_joint_right_knee_pitch", xml_name="joint_right_knee_pitch"),
            ObservationType.JointVel("dq_joint_right_ankle_pitch", xml_name="joint_right_ankle_pitch"),
            ObservationType.JointVel("dq_joint_right_ankle_roll", xml_name="joint_right_ankle_roll"),
            ObservationType.JointVel("dq_joint_waist_yaw", xml_name="joint_waist_yaw"),
            ObservationType.JointVel("dq_joint_waist_pitch", xml_name="joint_waist_pitch"),
            ObservationType.JointVel("dq_joint_waist_roll", xml_name="joint_waist_roll"),
            ObservationType.JointVel("dq_joint_head_pitch", xml_name="joint_head_pitch"),
            ObservationType.JointVel("dq_joint_head_roll", xml_name="joint_head_roll"),
            ObservationType.JointVel("dq_joint_head_yaw", xml_name="joint_head_yaw"),
            ObservationType.JointVel("dq_joint_left_shoulder_pitch", xml_name="joint_left_shoulder_pitch"),
            ObservationType.JointVel("dq_joint_left_shoulder_roll", xml_name="joint_left_shoulder_roll"),
            ObservationType.JointVel("dq_joint_left_shoulder_yaw", xml_name="joint_left_shoulder_yaw"),
            ObservationType.JointVel("dq_joint_left_elbow_pitch", xml_name="joint_left_elbow_pitch"),
            ObservationType.JointVel("dq_joint_left_wrist_yaw", xml_name="joint_left_wrist_yaw"),
            ObservationType.JointVel("dq_joint_left_wrist_roll", xml_name="joint_left_wrist_roll"),
            ObservationType.JointVel("dq_joint_left_wrist_pitch", xml_name="joint_left_wrist_pitch"),
            ObservationType.JointVel("dq_joint_right_shoulder_pitch", xml_name="joint_right_shoulder_pitch"),
            ObservationType.JointVel("dq_joint_right_shoulder_roll", xml_name="joint_right_shoulder_roll"),
            ObservationType.JointVel("dq_joint_right_shoulder_yaw", xml_name="joint_right_shoulder_yaw"),
            ObservationType.JointVel("dq_joint_right_elbow_pitch", xml_name="joint_right_elbow_pitch"),
            ObservationType.JointVel("dq_joint_right_wrist_yaw", xml_name="joint_right_wrist_yaw"),
            ObservationType.JointVel("dq_joint_right_wrist_roll", xml_name="joint_right_wrist_roll"),
            ObservationType.JointVel("dq_joint_right_wrist_pitch", xml_name="joint_right_wrist_pitch"),
        ]

        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec):
        """
        Getter for the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            A list of actuator names.
        """

        action_spec = [
            "link_left_hip_roll",
            "link_left_hip_yaw",
            "link_left_hip_pitch",
            "link_left_knee_pitch",
            "link_left_ankle_pitch",
            "link_left_ankle_roll",
            "link_right_hip_roll",
            "link_right_hip_yaw",
            "link_right_hip_pitch",
            "link_right_knee_pitch",
            "link_right_ankle_pitch",
            "link_right_ankle_roll",
            "link_waist_yaw",
            "link_waist_pitch",
            "link_waist_roll",
            "link_head_yaw",
            "link_head_roll",
            "link_head_pitch",
            "link_left_shoulder_pitch",
            "link_left_shoulder_roll",
            "link_left_shoulder_yaw",
            "link_left_elbow_pitch",
            "link_left_wrist_yaw",
            "link_left_wrist_roll",
            "link_left_wrist_pitch",
            "link_right_shoulder_pitch",
            "link_right_shoulder_roll",
            "link_right_shoulder_yaw",
            "link_right_elbow_pitch",
            "link_right_wrist_yaw",
            "link_right_wrist_roll",
            "link_right_wrist_pitch"
        ]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls):
        """
        Returns the default path to the xml file of the environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "fourier_gr1t2" / "gr1t2.xml").as_posix()

    @info_property
    def grf_size(self):
        """
        Returns the size of the ground force vector.

        """

        return 6

    @info_property
    def root_body_name(self):
        return "base"

    @info_property
    def upper_body_xml_name(self):
        return "link_torso"

    @info_property
    def root_free_joint_xml_name(self):
        return "root"

    @info_property
    def root_height_healthy_range(self):
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        return (0.6, 1.5)
