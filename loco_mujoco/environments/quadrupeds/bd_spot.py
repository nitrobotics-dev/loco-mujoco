import numpy as np
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.quadrupeds.base_robot_quadruped import BaseRobotQuadruped
from loco_mujoco.core.utils import info_property
from loco_mujoco.environments import ValidTaskConf


class BDSpot(BaseRobotQuadruped):

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

        # uses PD control by default
        if "control_type" not in kwargs.keys():
            kwargs["control_type"] = "PDControl"
            kwargs["control_params"] = dict(p_gain=500.0, d_gain=10.0, scale_action_to_jnt_limits=False,
                                            nominal_joint_positions=self.init_qpos[7:])

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
            ObservationType.FreeJointPosNoXY("q_root", xml_name="freejoint"),
            # --- Front Left ---
            ObservationType.JointPos("q_fl_hx", xml_name="fl_hx"),
            ObservationType.JointPos("q_fl_hy", xml_name="fl_hy"),
            ObservationType.JointPos("q_fl_kn", xml_name="fl_kn"),
            # --- Front Right ---
            ObservationType.JointPos("q_fr_hx", xml_name="fr_hx"),
            ObservationType.JointPos("q_fr_hy", xml_name="fr_hy"),
            ObservationType.JointPos("q_fr_kn", xml_name="fr_kn"),
            # --- Rear Left ---
            ObservationType.JointPos("q_hl_hx", xml_name="hl_hx"),
            ObservationType.JointPos("q_hl_hy", xml_name="hl_hy"),
            ObservationType.JointPos("q_hl_kn", xml_name="hl_kn"),
            # --- Rear Right ---
            ObservationType.JointPos("q_hr_hx", xml_name="hr_hx"),
            ObservationType.JointPos("q_hr_hy", xml_name="hr_hy"),
            ObservationType.JointPos("q_hr_kn", xml_name="hr_kn"),

            # ------------------- JOINT VEL -------------------
            # --- Trunk ---
            ObservationType.FreeJointVel("dq_root", xml_name="freejoint"),
            # --- Front Left ---
            ObservationType.JointVel("dq_fl_hx", xml_name="fl_hx"),
            ObservationType.JointVel("dq_fl_hy", xml_name="fl_hy"),
            ObservationType.JointVel("dq_fl_kn", xml_name="fl_kn"),
            # --- Front Right ---
            ObservationType.JointVel("dq_fr_hx", xml_name="fr_hx"),
            ObservationType.JointVel("dq_fr_hy", xml_name="fr_hy"),
            ObservationType.JointVel("dq_fr_kn", xml_name="fr_kn"),
            # --- Rear Left ---
            ObservationType.JointVel("dq_hl_hx", xml_name="hl_hx"),
            ObservationType.JointVel("dq_hl_hy", xml_name="hl_hy"),
            ObservationType.JointVel("dq_hl_kn", xml_name="hl_kn"),
            # --- Rear Right ---
            ObservationType.JointVel("dq_hr_hx", xml_name="hr_hx"),
            ObservationType.JointVel("dq_hr_hy", xml_name="hr_hy"),
            ObservationType.JointVel("dq_hr_kn", xml_name="hr_kn"),
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

        action_spec = ["fl_hx", "fl_hy", "fl_kn", "fr_hx", "fr_hy", "fr_kn",
                       "hl_hx", "hl_hy", "hl_kn", "hr_hx", "hr_hy", "hr_kn"]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls):
        """
        Returns the torque path to the xml file of the environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "bd_spot" / "spot.xml").as_posix()

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
        return "body"

    @info_property
    def root_free_joint_xml_name(self):
        return "freejoint"

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
        return ["HL", "HR", "FL", "FR"]

    @info_property
    def init_qpos(self):
        return np.array([0.0, 0.0, 0.46, 1.0, 0.0, 0.0, 0.0, 0.0, 1.04, -1.8, 0.0,
                         1.04, -1.8, 0.0, 1.04, -1.8, 0.0, 1.04, -1.8])

    @info_property
    def init_qvel(self):
        return np.zeros(18)
