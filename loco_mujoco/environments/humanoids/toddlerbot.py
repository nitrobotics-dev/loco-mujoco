import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class ToddlerBot(BaseRobotHumanoid):

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

        super().__init__(spec, action_spec, observation_spec, **kwargs)

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
            ObservationType.JointPos("q_neck_yaw_drive", xml_name="neck_yaw_drive"),
            ObservationType.JointPos("q_neck_yaw_driven", xml_name="neck_yaw_driven"),
            ObservationType.JointPos("q_neck_pitch", xml_name="neck_pitch"),
            ObservationType.JointPos("q_neck_pitch_act", xml_name="neck_pitch_act"),
            ObservationType.JointPos("q_waist_yaw", xml_name="waist_yaw"),
            ObservationType.JointPos("q_waist_roll", xml_name="waist_roll"),
            ObservationType.JointPos("q_waist_act_1", xml_name="waist_act_1"),
            ObservationType.JointPos("q_waist_act_2", xml_name="waist_act_2"),
            ObservationType.JointPos("q_left_hip_pitch", xml_name="left_hip_pitch"),
            ObservationType.JointPos("q_left_hip_roll", xml_name="left_hip_roll"),
            ObservationType.JointPos("q_left_hip_yaw_driven", xml_name="left_hip_yaw_driven"),
            ObservationType.JointPos("q_left_hip_yaw_drive", xml_name="left_hip_yaw_drive"),
            ObservationType.JointPos("q_left_knee", xml_name="left_knee"),
            ObservationType.JointPos("q_left_ank_pitch", xml_name="left_ank_pitch"),
            ObservationType.JointPos("q_left_ank_roll", xml_name="left_ank_roll"),
            ObservationType.JointPos("q_left_knee_act", xml_name="left_knee_act"),
            ObservationType.JointPos("q_right_hip_pitch", xml_name="right_hip_pitch"),
            ObservationType.JointPos("q_right_hip_roll", xml_name="right_hip_roll"),
            ObservationType.JointPos("q_right_hip_yaw_driven", xml_name="right_hip_yaw_driven"),
            ObservationType.JointPos("q_right_hip_yaw_drive", xml_name="right_hip_yaw_drive"),
            ObservationType.JointPos("q_right_knee", xml_name="right_knee"),
            ObservationType.JointPos("q_right_ank_pitch", xml_name="right_ank_pitch"),
            ObservationType.JointPos("q_right_ank_roll", xml_name="right_ank_roll"),
            ObservationType.JointPos("q_right_knee_act", xml_name="right_knee_act"),
            ObservationType.JointPos("q_left_sho_pitch", xml_name="left_sho_pitch"),
            ObservationType.JointPos("q_left_sho_roll", xml_name="left_sho_roll"),
            ObservationType.JointPos("q_left_sho_yaw_drive", xml_name="left_sho_yaw_drive"),
            ObservationType.JointPos("q_left_elbow_roll", xml_name="left_elbow_roll"),
            ObservationType.JointPos("q_left_elbow_yaw_drive", xml_name="left_elbow_yaw_drive"),
            ObservationType.JointPos("q_left_wrist_pitch_drive", xml_name="left_wrist_pitch_drive"),
            ObservationType.JointPos("q_left_wrist_roll", xml_name="left_wrist_roll"),
            ObservationType.JointPos("q_right_sho_pitch", xml_name="right_sho_pitch"),
            ObservationType.JointPos("q_right_sho_roll", xml_name="right_sho_roll"),
            ObservationType.JointPos("q_right_sho_yaw_drive", xml_name="right_sho_yaw_drive"),
            ObservationType.JointPos("q_right_elbow_roll", xml_name="right_elbow_roll"),
            ObservationType.JointPos("q_right_elbow_yaw_drive", xml_name="right_elbow_yaw_drive"),
            ObservationType.JointPos("q_right_wrist_pitch_drive", xml_name="right_wrist_pitch_drive"),
            ObservationType.JointPos("q_right_wrist_roll", xml_name="right_wrist_roll"),

            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            ObservationType.JointVel("dq_neck_yaw_drive", xml_name="neck_yaw_drive"),
            ObservationType.JointVel("dq_neck_yaw_driven", xml_name="neck_yaw_driven"),
            ObservationType.JointVel("dq_neck_pitch", xml_name="neck_pitch"),
            ObservationType.JointVel("dq_neck_pitch_act", xml_name="neck_pitch_act"),
            ObservationType.JointVel("dq_waist_yaw", xml_name="waist_yaw"),
            ObservationType.JointVel("dq_waist_roll", xml_name="waist_roll"),
            ObservationType.JointVel("dq_waist_act_1", xml_name="waist_act_1"),
            ObservationType.JointVel("dq_waist_act_2", xml_name="waist_act_2"),
            ObservationType.JointVel("dq_left_hip_pitch", xml_name="left_hip_pitch"),
            ObservationType.JointVel("dq_left_hip_roll", xml_name="left_hip_roll"),
            ObservationType.JointVel("dq_left_hip_yaw_driven", xml_name="left_hip_yaw_driven"),
            ObservationType.JointVel("dq_left_hip_yaw_drive", xml_name="left_hip_yaw_drive"),
            ObservationType.JointVel("dq_left_knee", xml_name="left_knee"),
            ObservationType.JointVel("dq_left_ank_pitch", xml_name="left_ank_pitch"),
            ObservationType.JointVel("dq_left_ank_roll", xml_name="left_ank_roll"),
            ObservationType.JointVel("dq_left_knee_act", xml_name="left_knee_act"),
            ObservationType.JointVel("dq_right_hip_pitch", xml_name="right_hip_pitch"),
            ObservationType.JointVel("dq_right_hip_roll", xml_name="right_hip_roll"),
            ObservationType.JointVel("dq_right_hip_yaw_driven", xml_name="right_hip_yaw_driven"),
            ObservationType.JointVel("dq_right_hip_yaw_drive", xml_name="right_hip_yaw_drive"),
            ObservationType.JointVel("dq_right_knee", xml_name="right_knee"),
            ObservationType.JointVel("dq_right_ank_pitch", xml_name="right_ank_pitch"),
            ObservationType.JointVel("dq_right_ank_roll", xml_name="right_ank_roll"),
            ObservationType.JointVel("dq_right_knee_act", xml_name="right_knee_act"),
            ObservationType.JointVel("dq_left_sho_pitch", xml_name="left_sho_pitch"),
            ObservationType.JointVel("dq_left_sho_roll", xml_name="left_sho_roll"),
            ObservationType.JointVel("dq_left_sho_yaw_drive", xml_name="left_sho_yaw_drive"),
            ObservationType.JointVel("dq_left_elbow_roll", xml_name="left_elbow_roll"),
            ObservationType.JointVel("dq_left_elbow_yaw_drive", xml_name="left_elbow_yaw_drive"),
            ObservationType.JointVel("dq_left_wrist_pitch_drive", xml_name="left_wrist_pitch_drive"),
            ObservationType.JointVel("dq_left_wrist_roll", xml_name="left_wrist_roll"),
            ObservationType.JointVel("dq_right_sho_pitch", xml_name="right_sho_pitch"),
            ObservationType.JointVel("dq_right_sho_roll", xml_name="right_sho_roll"),
            ObservationType.JointVel("dq_right_sho_yaw_drive", xml_name="right_sho_yaw_drive"),
            ObservationType.JointVel("dq_right_elbow_roll", xml_name="right_elbow_roll"),
            ObservationType.JointVel("dq_right_elbow_yaw_drive", xml_name="right_elbow_yaw_drive"),
            ObservationType.JointVel("dq_right_wrist_pitch_drive", xml_name="right_wrist_pitch_drive"),
            ObservationType.JointVel("dq_right_wrist_roll", xml_name="right_wrist_roll"),
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
            "neck_yaw_drive",
            "neck_pitch_act",
            "waist_act_1",
            "waist_act_2",
            "left_hip_pitch",
            "left_hip_roll",
            "left_hip_yaw_drive",
            "left_knee_act",
            "left_ank_roll",
            "left_ank_pitch",
            "right_hip_pitch",
            "right_hip_roll",
            "right_hip_yaw_drive",
            "right_knee_act",
            "right_ank_roll",
            "right_ank_pitch",
            "left_sho_pitch",
            "left_sho_roll",
            "left_sho_yaw_drive",
            "left_elbow_roll",
            "left_elbow_yaw_drive",
            "left_wrist_pitch_drive",
            "left_wrist_roll",
            "right_sho_pitch",
            "right_sho_roll",
            "right_sho_yaw_drive",
            "right_elbow_roll",
            "right_elbow_yaw_drive",
            "right_wrist_pitch_drive",
            "right_wrist_roll"
        ]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls):
        """
        Returns the default path to the xml file of the environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "toddlerbot" / "toddlerbot.xml").as_posix()

    @info_property
    def grf_size(self):
        """
        Returns the size of the ground force vector.

        """

        return 6

    @info_property
    def root_body_name(self):
        return "torso"

    @info_property
    def upper_body_xml_name(self):
        return "spur_1m_20t"

    @info_property
    def root_free_joint_xml_name(self):
        return "root"

    @info_property
    def root_height_healthy_range(self):
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        return (0.2, 0.5)
