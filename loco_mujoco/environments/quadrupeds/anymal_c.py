import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.quadrupeds.base_robot_quadruped import BaseRobotQuadruped
from loco_mujoco.core.utils import info_property
from loco_mujoco.environments import ValidTaskConf


class AnymalC(BaseRobotQuadruped):

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
            kwargs["control_params"] = dict(p_gain=100.0, d_gain=0.0)

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
            # --- Front Left ---
            ObservationType.JointPos("q_lf_haa", xml_name="LF_HAA"),
            ObservationType.JointPos("q_lf_hfe", xml_name="LF_HFE"),
            ObservationType.JointPos("q_lf_kfe", xml_name="LF_KFE"),
            # --- Front Right ---
            ObservationType.JointPos("q_rf_haa", xml_name="RF_HAA"),
            ObservationType.JointPos("q_rf_hfe", xml_name="RF_HFE"),
            ObservationType.JointPos("q_rf_kfe", xml_name="RF_KFE"),
            # --- Rear Left ---
            ObservationType.JointPos("q_lh_haa", xml_name="LH_HAA"),
            ObservationType.JointPos("q_lh_hfe", xml_name="LH_HFE"),
            ObservationType.JointPos("q_lh_kfe", xml_name="LH_KFE"),
            # --- Rear Right ---
            ObservationType.JointPos("q_rh_haa", xml_name="RH_HAA"),
            ObservationType.JointPos("q_rh_hfe", xml_name="RH_HFE"),
            ObservationType.JointPos("q_rh_kfe", xml_name="RH_KFE"),

            # ------------------- JOINT VEL -------------------
            # --- Trunk ---
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            # --- Front Left ---
            ObservationType.JointVel("dq_lf_haa", xml_name="LF_HAA"),
            ObservationType.JointVel("dq_lf_hfe", xml_name="LF_HFE"),
            ObservationType.JointVel("dq_lf_kfe", xml_name="LF_KFE"),
            # --- Front Right ---
            ObservationType.JointVel("dq_rf_haa", xml_name="RF_HAA"),
            ObservationType.JointVel("dq_rf_hfe", xml_name="RF_HFE"),
            ObservationType.JointVel("dq_rf_kfe", xml_name="RF_KFE"),
            # --- Rear Left ---
            ObservationType.JointVel("dq_lh_haa", xml_name="LH_HAA"),
            ObservationType.JointVel("dq_lh_hfe", xml_name="LH_HFE"),
            ObservationType.JointVel("dq_lh_kfe", xml_name="LH_KFE"),
            # --- Rear Right ---
            ObservationType.JointVel("dq_rh_haa", xml_name="RH_HAA"),
            ObservationType.JointVel("dq_rh_hfe", xml_name="RH_HFE"),
            ObservationType.JointVel("dq_rh_kfe", xml_name="RH_KFE"),
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

        action_spec = ["LF_HAA", "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE",
                       "LH_HAA", "LH_HFE",  "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE"]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls):
        """
        Returns the torque path to the xml file of the environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "anybotics_anymal_c" / "anymal_c.xml").as_posix()

    @info_property
    def grf_size(self):
        """
        Returns the size of the ground force vector.

        """

        return 12

    @info_property
    def upper_body_xml_name(self):
        return self.root_body_name

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
        return (0.30, 1.0)
