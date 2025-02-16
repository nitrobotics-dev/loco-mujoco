import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class Apollo(BaseRobotHumanoid):

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
            ObservationType.FreeJointPosNoXY("q_floating_base", xml_name="floating_base"),
            ObservationType.JointPos("q_neck_yaw", xml_name="neck_yaw"),
            ObservationType.JointPos("q_neck_roll", xml_name="neck_roll"),
            ObservationType.JointPos("q_neck_pitch", xml_name="neck_pitch"),
            ObservationType.JointPos("q_torso_pitch", xml_name="torso_pitch"),
            ObservationType.JointPos("q_torso_roll", xml_name="torso_roll"),
            ObservationType.JointPos("q_torso_yaw", xml_name="torso_yaw"),
            ObservationType.JointPos("q_l_hip_ie", xml_name="l_hip_ie"),
            ObservationType.JointPos("q_l_hip_aa", xml_name="l_hip_aa"),
            ObservationType.JointPos("q_l_hip_fe", xml_name="l_hip_fe"),
            ObservationType.JointPos("q_l_knee_fe", xml_name="l_knee_fe"),
            ObservationType.JointPos("q_l_ankle_ie", xml_name="l_ankle_ie"),
            ObservationType.JointPos("q_l_ankle_pd", xml_name="l_ankle_pd"),
            ObservationType.JointPos("q_r_hip_ie", xml_name="r_hip_ie"),
            ObservationType.JointPos("q_r_hip_aa", xml_name="r_hip_aa"),
            ObservationType.JointPos("q_r_hip_fe", xml_name="r_hip_fe"),
            ObservationType.JointPos("q_r_knee_fe", xml_name="r_knee_fe"),
            ObservationType.JointPos("q_r_ankle_ie", xml_name="r_ankle_ie"),
            ObservationType.JointPos("q_r_ankle_pd", xml_name="r_ankle_pd"),
            ObservationType.JointPos("q_l_shoulder_aa", xml_name="l_shoulder_aa"),
            ObservationType.JointPos("q_l_shoulder_ie", xml_name="l_shoulder_ie"),
            ObservationType.JointPos("q_l_shoulder_fe", xml_name="l_shoulder_fe"),
            ObservationType.JointPos("q_l_elbow_fe", xml_name="l_elbow_fe"),
            ObservationType.JointPos("q_l_wrist_roll", xml_name="l_wrist_roll"),
            ObservationType.JointPos("q_l_wrist_yaw", xml_name="l_wrist_yaw"),
            ObservationType.JointPos("q_l_wrist_pitch", xml_name="l_wrist_pitch"),
            ObservationType.JointPos("q_r_shoulder_aa", xml_name="r_shoulder_aa"),
            ObservationType.JointPos("q_r_shoulder_ie", xml_name="r_shoulder_ie"),
            ObservationType.JointPos("q_r_shoulder_fe", xml_name="r_shoulder_fe"),
            ObservationType.JointPos("q_r_elbow_fe", xml_name="r_elbow_fe"),
            ObservationType.JointPos("q_r_wrist_roll", xml_name="r_wrist_roll"),
            ObservationType.JointPos("q_r_wrist_yaw", xml_name="r_wrist_yaw"),
            ObservationType.JointPos("q_r_wrist_pitch", xml_name="r_wrist_pitch"),

            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_floating_base", xml_name="floating_base"),
            ObservationType.JointVel("dq_neck_yaw", xml_name="neck_yaw"),
            ObservationType.JointVel("dq_neck_roll", xml_name="neck_roll"),
            ObservationType.JointVel("dq_neck_pitch", xml_name="neck_pitch"),
            ObservationType.JointVel("dq_torso_pitch", xml_name="torso_pitch"),
            ObservationType.JointVel("dq_torso_roll", xml_name="torso_roll"),
            ObservationType.JointVel("dq_torso_yaw", xml_name="torso_yaw"),
            ObservationType.JointVel("dq_l_hip_ie", xml_name="l_hip_ie"),
            ObservationType.JointVel("dq_l_hip_aa", xml_name="l_hip_aa"),
            ObservationType.JointVel("dq_l_hip_fe", xml_name="l_hip_fe"),
            ObservationType.JointVel("dq_l_knee_fe", xml_name="l_knee_fe"),
            ObservationType.JointVel("dq_l_ankle_ie", xml_name="l_ankle_ie"),
            ObservationType.JointVel("dq_l_ankle_pd", xml_name="l_ankle_pd"),
            ObservationType.JointVel("dq_r_hip_ie", xml_name="r_hip_ie"),
            ObservationType.JointVel("dq_r_hip_aa", xml_name="r_hip_aa"),
            ObservationType.JointVel("dq_r_hip_fe", xml_name="r_hip_fe"),
            ObservationType.JointVel("dq_r_knee_fe", xml_name="r_knee_fe"),
            ObservationType.JointVel("dq_r_ankle_ie", xml_name="r_ankle_ie"),
            ObservationType.JointVel("dq_r_ankle_pd", xml_name="r_ankle_pd"),
            ObservationType.JointVel("dq_l_shoulder_aa", xml_name="l_shoulder_aa"),
            ObservationType.JointVel("dq_l_shoulder_ie", xml_name="l_shoulder_ie"),
            ObservationType.JointVel("dq_l_shoulder_fe", xml_name="l_shoulder_fe"),
            ObservationType.JointVel("dq_l_elbow_fe", xml_name="l_elbow_fe"),
            ObservationType.JointVel("dq_l_wrist_roll", xml_name="l_wrist_roll"),
            ObservationType.JointVel("dq_l_wrist_yaw", xml_name="l_wrist_yaw"),
            ObservationType.JointVel("dq_l_wrist_pitch", xml_name="l_wrist_pitch"),
            ObservationType.JointVel("dq_r_shoulder_aa", xml_name="r_shoulder_aa"),
            ObservationType.JointVel("dq_r_shoulder_ie", xml_name="r_shoulder_ie"),
            ObservationType.JointVel("dq_r_shoulder_fe", xml_name="r_shoulder_fe"),
            ObservationType.JointVel("dq_r_elbow_fe", xml_name="r_elbow_fe"),
            ObservationType.JointVel("dq_r_wrist_roll", xml_name="r_wrist_roll"),
            ObservationType.JointVel("dq_r_wrist_yaw", xml_name="r_wrist_yaw"),
            ObservationType.JointVel("dq_r_wrist_pitch", xml_name="r_wrist_pitch"),
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

        action_spec = ["neck_yaw", "neck_roll", "neck_pitch", "torso_pitch", "torso_roll", "torso_yaw", "l_hip_ie",
                       "l_hip_aa", "l_hip_fe", "l_knee_fe", "l_ankle_ie", "l_ankle_pd", "r_hip_ie", "r_hip_aa",
                       "r_hip_fe", "r_knee_fe", "r_ankle_ie", "r_ankle_pd", "l_shoulder_aa", "l_shoulder_ie",
                       "l_shoulder_fe", "l_elbow_fe", "l_wrist_roll", "l_wrist_yaw", "l_wrist_pitch", "r_shoulder_aa",
                       "r_shoulder_ie", "r_shoulder_fe", "r_elbow_fe", "r_wrist_roll", "r_wrist_yaw", "r_wrist_pitch"
                       ]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls):
        """
        Returns the default path to the xml file of the environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "apptronik_apollo" / "apptronik_apollo.xml").as_posix()

    @info_property
    def p_gains(self):
        """
        Returns the proportional gains for the default PD controller.
        """
        return [28, 9, 8, 1525, 2052, 600, 595, 1880, 1047, 606, 420, 882, 595, 1880, 1047, 606, 420, 882, 395, 530,
                277, 312, 47, 20, 18, 395, 530, 277, 312, 47, 20, 18]

    @info_property
    def d_gains(self):
        """
        Returns the derivative gains for the default PD controller.
        """
        return [15, 3, 3, 142, 165, 60, 171, 153, 92, 46, 11, 21, 171, 153, 92, 46, 11, 21, 26, 45, 21, 24, 15, 3, 3,
                26, 45, 21, 24, 15, 3, 3]

    @info_property
    def grf_size(self):
        """
        Returns the size of the ground force vector.

        """

        return 6

    @info_property
    def upper_body_xml_name(self):
        return "torso_link"

    @info_property
    def root_free_joint_xml_name(self):
        return "floating_base"

    @info_property
    def root_height_healthy_range(self):
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        return (0.6, 1.5)
