import numpy as np
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

    def __init__(self, spec=None, camera_params=None,
                 observation_spec=None, action_spec=None, **kwargs):
        """
        Constructor.

        Args:
            camera_params (dict): Dictionary defining some of the camera parameters for the visualization.

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

        # set init position
        if "init_state_handler" not in kwargs.keys():
            kwargs["init_state_type"] = "DefaultInitialStateHandler"
            kwargs["init_state_params"] = (dict(qpos_init=self.init_qpos, qvel_init=self.init_qvel))

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
    def get_default_xml_file_path(cls):
        """
        Returns the torque path to the xml file of the environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "unitree_go2" / "go2.xml").as_posix()

    @info_property
    def grf_size(self):
        """
        Returns the size of the ground force vector.

        """

        return 12

    @info_property
    def upper_body_xml_name(self):
        return self.root_body_name

    @info_property
    def root_body_name(self):
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

    @info_property
    def foot_geom_names(self):
        """
        Returns the names of the foot geometries.

        """
        return ["RL_foot", "RR_foot", "FL_foot", "FR_foot"]

    @info_property
    def init_qpos(self):
        return np.array([0.0, 0.0, 0.27, 1.0, 0.0, 0.0, 0.0, 0.0, 0.9, -1.8, 0.0,
                         0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8])

    @info_property
    def init_qvel(self):
        return np.zeros(18)
