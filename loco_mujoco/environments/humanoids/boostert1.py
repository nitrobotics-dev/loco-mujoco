import mujoco
from mujoco import MjSpec
import numpy as np

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class BoosterT1(BaseRobotHumanoid):

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

        # uses PD control by default
        if "control_type" not in kwargs.keys():
            kwargs["control_type"] = "PDControl"
            kwargs["control_params"] = dict(p_gain=self.p_gains, d_gain=self.d_gains, scale_action_to_jnt_limits=False)

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
            ObservationType.JointPos("q_AAHead_yaw", xml_name="AAHead_yaw"),
            ObservationType.JointPos("q_Head_pitch", xml_name="Head_pitch"),
            ObservationType.JointPos("q_Left_Shoulder_Pitch", xml_name="Left_Shoulder_Pitch"),
            ObservationType.JointPos("q_Left_Shoulder_Roll", xml_name="Left_Shoulder_Roll"),
            ObservationType.JointPos("q_Left_Elbow_Pitch", xml_name="Left_Elbow_Pitch"),
            ObservationType.JointPos("q_Left_Elbow_Yaw", xml_name="Left_Elbow_Yaw"),
            ObservationType.JointPos("q_Right_Shoulder_Pitch", xml_name="Right_Shoulder_Pitch"),
            ObservationType.JointPos("q_Right_Shoulder_Roll", xml_name="Right_Shoulder_Roll"),
            ObservationType.JointPos("q_Right_Elbow_Pitch", xml_name="Right_Elbow_Pitch"),
            ObservationType.JointPos("q_Right_Elbow_Yaw", xml_name="Right_Elbow_Yaw"),
            ObservationType.JointPos("q_Waist", xml_name="Waist"),
            ObservationType.JointPos("q_Left_Hip_Pitch", xml_name="Left_Hip_Pitch"),
            ObservationType.JointPos("q_Left_Hip_Roll", xml_name="Left_Hip_Roll"),
            ObservationType.JointPos("q_Left_Hip_Yaw", xml_name="Left_Hip_Yaw"),
            ObservationType.JointPos("q_Left_Knee_Pitch", xml_name="Left_Knee_Pitch"),
            ObservationType.JointPos("q_Left_Ankle_Pitch", xml_name="Left_Ankle_Pitch"),
            ObservationType.JointPos("q_Left_Ankle_Roll", xml_name="Left_Ankle_Roll"),
            ObservationType.JointPos("q_Right_Hip_Pitch", xml_name="Right_Hip_Pitch"),
            ObservationType.JointPos("q_Right_Hip_Roll", xml_name="Right_Hip_Roll"),
            ObservationType.JointPos("q_Right_Hip_Yaw", xml_name="Right_Hip_Yaw"),
            ObservationType.JointPos("q_Right_Knee_Pitch", xml_name="Right_Knee_Pitch"),
            ObservationType.JointPos("q_Right_Ankle_Pitch", xml_name="Right_Ankle_Pitch"),
            ObservationType.JointPos("q_Right_Ankle_Roll", xml_name="Right_Ankle_Roll"),

            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            ObservationType.JointVel("dq_AAHead_yaw", xml_name="AAHead_yaw"),
            ObservationType.JointVel("dq_Head_pitch", xml_name="Head_pitch"),
            ObservationType.JointVel("dq_Left_Shoulder_Pitch", xml_name="Left_Shoulder_Pitch"),
            ObservationType.JointVel("dq_Left_Shoulder_Roll", xml_name="Left_Shoulder_Roll"),
            ObservationType.JointVel("dq_Left_Elbow_Pitch", xml_name="Left_Elbow_Pitch"),
            ObservationType.JointVel("dq_Left_Elbow_Yaw", xml_name="Left_Elbow_Yaw"),
            ObservationType.JointVel("dq_Right_Shoulder_Pitch", xml_name="Right_Shoulder_Pitch"),
            ObservationType.JointVel("dq_Right_Shoulder_Roll", xml_name="Right_Shoulder_Roll"),
            ObservationType.JointVel("dq_Right_Elbow_Pitch", xml_name="Right_Elbow_Pitch"),
            ObservationType.JointVel("dq_Right_Elbow_Yaw", xml_name="Right_Elbow_Yaw"),
            ObservationType.JointVel("dq_Waist", xml_name="Waist"),
            ObservationType.JointVel("dq_Left_Hip_Pitch", xml_name="Left_Hip_Pitch"),
            ObservationType.JointVel("dq_Left_Hip_Roll", xml_name="Left_Hip_Roll"),
            ObservationType.JointVel("dq_Left_Hip_Yaw", xml_name="Left_Hip_Yaw"),
            ObservationType.JointVel("dq_Left_Knee_Pitch", xml_name="Left_Knee_Pitch"),
            ObservationType.JointVel("dq_Left_Ankle_Pitch", xml_name="Left_Ankle_Pitch"),
            ObservationType.JointVel("dq_Left_Ankle_Roll", xml_name="Left_Ankle_Roll"),
            ObservationType.JointVel("dq_Right_Hip_Pitch", xml_name="Right_Hip_Pitch"),
            ObservationType.JointVel("dq_Right_Hip_Roll", xml_name="Right_Hip_Roll"),
            ObservationType.JointVel("dq_Right_Hip_Yaw", xml_name="Right_Hip_Yaw"),
            ObservationType.JointVel("dq_Right_Knee_Pitch", xml_name="Right_Knee_Pitch"),
            ObservationType.JointVel("dq_Right_Ankle_Pitch", xml_name="Right_Ankle_Pitch"),
            ObservationType.JointVel("dq_Right_Ankle_Roll", xml_name="Right_Ankle_Roll"),

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
        action_spec = ["AAHead_yaw", "Head_pitch", "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch",
                       "Left_Elbow_Yaw", "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch",
                       "Right_Elbow_Yaw", "Waist", "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
                       "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll", "Right_Hip_Pitch", "Right_Hip_Roll",
                       "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll"]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls):
        """
        Returns the default path to the xml file of the environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "booster_t1" / "booster_t1.xml").as_posix()

    @info_property
    def p_gains(self):
        """
        Returns the proportional gains for the default PD controller.
        """
        return 75

    @info_property
    def d_gains(self):
        """
        Returns the derivative gains for the default PD controller.
        """
        return 0.0

    @info_property
    def grf_size(self):
        """
        Returns the size of the ground force vector.

        """
        return 6

    @info_property
    def upper_body_xml_name(self):
        return self.root_body_name

    @info_property
    def root_body_name(self):
        return "Trunk"

    @info_property
    def root_free_joint_xml_name(self):
        return "root"

    @info_property
    def root_height_healthy_range(self):
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        return (0.3, 1.0)
