import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.quadrupeds.base_robot_quadruped import BaseRobotQuadruped
from loco_mujoco.core.utils import info_property
from loco_mujoco.environments import ValidTaskConf


class UnitreeGo2(BaseRobotQuadruped):

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

    valid_task_confs = ValidTaskConf(tasks=["simple", "hard"],
                                     data_types=["real", "perfect"])

    mjx_enabled = False

    def __init__(self, action_mode="torque", xml_path=None, camera_params=None, **kwargs):
        """
        Constructor.

        Args:
            action_mode (str): Either "torque", or "position". Defines the action controller.
            camera_params (dict): Dictionary defining some of the camera parameters for the visualization.

        """

        if xml_path is None and action_mode == "torque":
            xml_path = self.get_torque_xml_file_path()

        # load the model specification
        spec = mujoco.MjSpec.from_file(xml_path)

        # get the observation and action space
        observation_spec = self._get_observation_specification(spec)
        action_spec = self._get_action_specification(spec)

        # modify the specification if needed
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        if camera_params is None:
            # make the camera by default a bit higher
            camera_params = dict(follow=dict(distance=3.5, elevation=-20.0, azimuth=90.0))

        super().__init__(spec, action_spec, observation_spec,
                         camera_params=camera_params, enable_mjx=self.mjx_enabled, **kwargs)

    @staticmethod
    def _get_observation_specification(spec: MjSpec):
        """
        Getter for the observation space specification.

        Returns:
            A list of tuples containing the specification of each observation
            space entry.
        """

        observation_spec = [
            # ------------------- JOINT POS -------------------
            # --- Trunk ---
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
            # --- Front ---
            ObservationType.JointPos("q_FR_hip_joint", xml_name="FR_hip_joint"),
            ObservationType.JointPos("q_FR_thigh_joint", xml_name="FR_thigh_joint"),
            ObservationType.JointPos("q_FR_calf_joint", xml_name="FR_calf_joint"),
            ObservationType.JointPos("q_FL_hip_joint", xml_name="FL_hip_joint"),
            ObservationType.JointPos("q_FL_thigh_joint", xml_name="FL_thigh_joint"),
            ObservationType.JointPos("q_FL_calf_joint", xml_name="FL_calf_joint"),
            # --- Rear ---
            ObservationType.JointPos("q_RR_hip_joint", xml_name="RR_hip_joint"),
            ObservationType.JointPos("q_RR_thigh_joint", xml_name="RR_thigh_joint"),
            ObservationType.JointPos("q_RR_calf_joint", xml_name="RR_calf_joint"),
            ObservationType.JointPos("q_RL_hip_joint", xml_name="RL_hip_joint"),
            ObservationType.JointPos("q_RL_thigh_joint", xml_name="RL_thigh_joint"),
            ObservationType.JointPos("q_RL_calf_joint", xml_name="RL_calf_joint"),

            # ------------------- JOINT VEL -------------------
            # --- Trunk ---
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            # --- Front ---
            ObservationType.JointVel("dq_FR_hip_joint", xml_name="FR_hip_joint"),
            ObservationType.JointVel("dq_FR_thigh_joint", xml_name="FR_thigh_joint"),
            ObservationType.JointVel("dq_FR_calf_joint", xml_name="FR_calf_joint"),
            ObservationType.JointVel("dq_FL_hip_joint", xml_name="FL_hip_joint"),
            ObservationType.JointVel("dq_FL_thigh_joint", xml_name="FL_thigh_joint"),
            ObservationType.JointVel("dq_FL_calf_joint", xml_name="FL_calf_joint"),
            # --- Rear ---
            ObservationType.JointVel("dq_RR_hip_joint", xml_name="RR_hip_joint"),
            ObservationType.JointVel("dq_RR_thigh_joint", xml_name="RR_thigh_joint"),
            ObservationType.JointVel("dq_RR_calf_joint", xml_name="RR_calf_joint"),
            ObservationType.JointVel("dq_RL_hip_joint", xml_name="RL_hip_joint"),
            ObservationType.JointVel("dq_RL_thigh_joint", xml_name="RL_thigh_joint"),
            ObservationType.JointVel("dq_RL_calf_joint", xml_name="RL_calf_joint")]

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
            "FR_hip", "FR_thigh", "FR_calf",
            "FL_hip", "FL_thigh", "FL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
            "RL_hip", "RL_thigh", "RL_calf"
        ]

        return action_spec

    @classmethod
    def get_torque_xml_file_path(cls):
        """
        Returns the torque path to the xml file of the environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "unitree_go2" / "go2_torque.xml").as_posix()

    @info_property
    def grf_size(self):
        """
        Returns the size of the ground force vector.

        """

        return 12

    @info_property
    def upper_body_xml_name(self):
        return "base"

    @info_property
    def root_free_joint_xml_name(self):
        return "root"

    @info_property
    def root_height_healthy_range(self):
        """
        Return the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        return (0.25, 1.0)