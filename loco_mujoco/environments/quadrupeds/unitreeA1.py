import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments import ValidTaskConf
from loco_mujoco.environments.quadrupeds.base_robot_quadruped import BaseRobotQuadruped
from loco_mujoco.core.utils import info_property


class UnitreeA1(BaseRobotQuadruped):

    """
    Description
    ------------

    Mujoco environment of Unitree A1 model.

    Tasks
    -----------------
    * **Simple**: The robot has to walk forward with a fixed speed of 0.5 m/s.
    * **Hard**: The robot has to walk in 8 different directions with a fixed speed of 0.5 m/s.


    Dataset Types
    -----------------
    The available dataset types for this environment can be found at: :ref:`env-label`.


    Observation Space
    -----------------

    The observation space has the following properties *by default* (i.e., only obs with Disabled == False):

    | For simple task: :code:`(min=-inf, max=inf, dim=37, dtype=float32)`
    | For hard task: :code:`(min=-inf, max=inf, dim=37, dtype=float32)`

    Some observations are **disabled by default**, but can be turned on. The detailed observation space is:

    ===== ========================================================= ========= ========= ======== === ========================
    0     Position of Joint trunk_tz                                -inf      inf       False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    1     Position of Joint trunk_list                              -inf      inf       False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    2     Position of Joint trunk_tilt                              -inf      inf       False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    3     Position of Joint trunk_rotation                          -inf      inf       False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    4     Position of Joint FR_hip_joint                            -0.802851 0.802851  False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    5     Position of Joint FR_thigh_joint                          -1.0472   4.18879   False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    6     Position of Joint FR_calf_joint                           -2.69653  -0.916298 False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    7     Position of Joint FL_hip_joint                            -0.802851 0.802851  False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    8     Position of Joint FL_thigh_joint                          -1.0472   4.18879   False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    9     Position of Joint FL_calf_joint                           -2.69653  -0.916298 False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    10    Position of Joint RR_hip_joint                            -0.802851 0.802851  False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    11    Position of Joint RR_thigh_joint                          -1.0472   4.18879   False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    12    Position of Joint RR_calf_joint                           -2.69653  -0.916298 False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    13    Position of Joint RL_hip_joint                            -0.802851 0.802851  False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    14    Position of Joint RL_thigh_joint                          -1.0472   4.18879   False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    15    Position of Joint RL_calf_joint                           -2.69653  -0.916298 False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    16    Velocity of Joint trunk_tx                                -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    17    Velocity of Joint trunk_ty                                -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    18    Velocity of Joint trunk_tz                                -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    19    Velocity of Joint trunk_list                              -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    20    Velocity of Joint trunk_tilt                              -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    21    Velocity of Joint trunk_rotation                          -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    22    Velocity of Joint FR_hip_joint                            -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    23    Velocity of Joint FR_thigh_joint                          -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    24    Velocity of Joint FR_calf_joint                           -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    25    Velocity of Joint FL_hip_joint                            -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    26    Velocity of Joint FL_thigh_joint                          -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    27    Velocity of Joint FL_calf_joint                           -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    28    Velocity of Joint RR_hip_joint                            -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    29    Velocity of Joint RR_thigh_joint                          -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    30    Velocity of Joint RR_calf_joint                           -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    31    Velocity of Joint RL_hip_joint                            -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    32    Velocity of Joint RL_thigh_joint                          -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    33    Velocity of Joint RL_calf_joint                           -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    34    Desired Velocity Angle represented as Sine-Cosine Feature 0.0       1         False    2   None
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    36    Desired Velocity                                          0.0       inf       False    1   Velocity [m/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    37    3D linear Forces between Front Left Foot and Floor        0.0       inf       True     3   Force [N]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    40    3D linear Forces between Front Right Foot and Floor       0.0       inf       True     3   Force [N]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    43    3D linear Forces between Back Left Foot and Floor         0.0       inf       True     3   Force [N]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    46    3D linear Forces between Back Right Foot and Floor        0.0       inf       True     3   Force [N]
    ===== ========================================================= ========= ========= ======== === ========================

    Action Space
    ------------

    | The action space has the following properties *by default* (i.e., only actions with Disabled == False):
    | :code:`(min=-1, max=1, dim=12, dtype=float32)`

    ===== =========== =========== =========== ========
    Index Name in XML Control Min Control Max Disabled
    ===== =========== =========== =========== ========
    0     FR_hip      -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    1     FR_thigh    -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    2     FR_calf     -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    3     FL_hip      -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    4     FL_thigh    -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    5     FL_calf     -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    6     RR_hip      -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    7     RR_thigh    -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    8     RR_calf     -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    9     RL_hip      -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    10    RL_thigh    -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    11    RL_calf     -1.0        1.0         False
    ===== =========== =========== =========== ========


    Rewards
    --------

    Reward function based on the difference between the desired velocity vector and the actual center of mass velocity
    vector in horizontal plane. The desired velocity vector is given by the dataset to imitate.

    **Class**: :class:`loco_mujoco.utils.reward.VelocityVectorReward`


    Initial States
    ---------------

    The initial state is sampled by default from the dataset to imitate.

    Terminal States
    ----------------

    The terminal state is reached when the robot falls, or rather starts falling. The condition to check if the robot
    is falling is based on the orientation of the robot and the height of the center of mass. More details can be found
    in the  :code:`_has_fallen` method of the environment.

    Methods
    ------------

    """

    valid_task_confs = ValidTaskConf(tasks=["simple", "hard"],
                                     data_types=["real", "perfect"])

    mjx_enabled = False

    def __init__(self, spec=None, camera_params=None,
                 observation_spec=None, action_spec=None, **kwargs):
        """
        Constructor.

        Args:
            action_mode (str): Either "torque", "position", or "position_difference". Defines the action controller.
            camera_params (dict): Dictionary defining some of the camera parameters for visualization.

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
            "RL_hip", "RL_thigh", "RL_calf"]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls):
        """
        Returns the default path to the xml file of the environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "unitree_a1" / "unitree_a1.xml").as_posix()

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
    def root_free_joint_xml_name(self):
        return "root"

    @info_property
    def root_height_healthy_range(self):
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        return (0.15, 1.0)

    @info_property
    def foot_geom_names(self):
        """
        Returns the names of the foot geometries.

        """
        return ["RL_foot", "RR_foot", "FL_foot", "FR_foot"]
