import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.environments import ValidTaskConf
from loco_mujoco.core.utils import info_property


class UnitreeH1(BaseRobotHumanoid):

    """
    Description
    ------------

    Mujoco environment of the Unitree H1 robot. Optionally, the H1 can carry
    a weight. This environment can be partially observable by hiding
    some of the state space entries from the policy using a state mask.
    Hidable entries are "positions", "velocities", "foot_forces",
    or "weight".

    Tasks
    -----------------
    * **Walking**: The robot has to walk forward with a fixed speed of 1.25 m/s.
    * **Running**: Run forward with a fixed speed of 2.5 m/s.
    * **Carry**: The robot has to walk forward with a fixed speed of 1.25 m/s while carrying a weight.
      The mass is either specified by the user or sampled from a uniformly from [0.1 kg, 1 kg, 5 kg, 10 kg].


    Dataset Types
    -----------------
    The available dataset types for this environment can be found at: :ref:`env-label`.


    Observation Space
    -----------------

    The observation space has the following properties *by default* (i.e., only obs with Disabled == False):

    | For walking task: :code:`(min=-inf, max=inf, dim=32, dtype=float32)`
    | For running task: :code:`(min=-inf, max=inf, dim=32, dtype=float32)`
    | For carry task: :code:`(min=-inf, max=inf, dim=33, dtype=float32)`

    Some observations are **disabled by default**, but can be turned on. The detailed observation space is:

    ===== ============================================= ===== ==== =========================== === ========================
    Index Description                                   Min   Max  Disabled                    Dim Units
    ===== ============================================= ===== ==== =========================== === ========================
    0     Position of Joint pelvis_ty                   -inf  inf  False                       1   Position [m]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    1     Position of Joint pelvis_tilt                 -inf  inf  False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    2     Position of Joint pelvis_list                 -inf  inf  False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    3     Position of Joint pelvis_rotation             -inf  inf  False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    4     Position of Joint back_bkz                    -2.35 2.35 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    5     Position of Joint l_arm_shy                   -2.87 2.87 True                        1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    6     Position of Joint l_arm_shx                   -0.34 3.11 True                        1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    7     Position of Joint l_arm_shz                   -1.3  4.45 True                        1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    8     Position of Joint left_elbow                  -1.25 2.61 True                        1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    9     Position of Joint r_arm_shy                   -2.87 2.87 True                        1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    10    Position of Joint r_arm_shx                   -3.11 0.34 True                        1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    11    Position of Joint r_arm_shz                   -4.45 1.3  True                        1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    12    Position of Joint right_elbow                 -1.25 2.61 True                        1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    13    Position of Joint hip_flexion_r               -1.57 1.57 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    14    Position of Joint hip_adduction_r             -0.43 0.43 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    15    Position of Joint hip_rotation_r              -0.43 0.43 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    16    Position of Joint knee_angle_r                -0.26 2.05 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    17    Position of Joint ankle_angle_r               -0.87 0.52 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    18    Position of Joint hip_flexion_l               -1.57 1.57 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    19    Position of Joint hip_adduction_l             -0.43 0.43 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    20    Position of Joint hip_rotation_l              -0.43 0.43 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    21    Position of Joint knee_angle_l                -0.26 2.05 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    22    Position of Joint ankle_angle_l               -0.87 0.52 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    23    Velocity of Joint pelvis_tx                   -inf  inf  False                       1   Velocity [m/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    24    Velocity of Joint pelvis_tz                   -inf  inf  False                       1   Velocity [m/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    25    Velocity of Joint pelvis_ty                   -inf  inf  False                       1   Velocity [m/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    26    Velocity of Joint pelvis_tilt                 -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    27    Velocity of Joint pelvis_list                 -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    28    Velocity of Joint pelvis_rotation             -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    29    Velocity of Joint back_bkz                    -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    30    Velocity of Joint l_arm_shy                   -inf  inf  True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    31    Velocity of Joint l_arm_shx                   -inf  inf  True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    32    Velocity of Joint l_arm_shz                   -inf  inf  True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    33    Velocity of Joint left_elbow                  -inf  inf  True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    34    Velocity of Joint r_arm_shy                   -inf  inf  True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    35    Velocity of Joint r_arm_shx                   -inf  inf  True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    36    Velocity of Joint r_arm_shz                   -inf  inf  True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    37    Velocity of Joint right_elbow                 -inf  inf  True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    38    Velocity of Joint hip_flexion_r               -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    39    Velocity of Joint hip_adduction_r             -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    40    Velocity of Joint hip_rotation_r              -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    41    Velocity of Joint knee_angle_r                -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    42    Velocity of Joint ankle_angle_r               -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    43    Velocity of Joint hip_flexion_l               -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    44    Velocity of Joint hip_adduction_l             -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    45    Velocity of Joint hip_rotation_l              -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    46    Velocity of Joint knee_angle_l                -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    47    Velocity of Joint ankle_angle_l               -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    48    Mass of the Weight                            0.0   inf  Only Enabled for Carry Task 1   Mass [kg]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    49    3D linear Forces between Right Foot and Floor 0.0   inf  True                        3   Force [N]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    52    3D linear Forces between Left Foot and Floor  0.0   inf  True                        3   Force [N]
    ===== ============================================= ===== ==== =========================== === ========================

    Action Space
    ------------

    | The action space has the following properties *by default* (i.e., only actions with Disabled == False):
    | :code:`(min=-1, max=1, dim=11, dtype=float32)`

    Some actions are **disabled by default**, but can be turned on. The detailed action space is:

    ===== ======================== =========== =========== ========
    Index Name in XML              Control Min Control Max Disabled
    ===== ======================== =========== =========== ========
    0     back_bkz_actuator        -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    1     l_arm_shy_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    2     l_arm_shx_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    3     l_arm_shz_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    4     left_elbow_actuator      -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    5     r_arm_shy_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    6     r_arm_shx_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    7     r_arm_shz_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    8     right_elbow_actuator     -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    9     hip_flexion_r_actuator   -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    10    hip_adduction_r_actuator -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    11    hip_rotation_r_actuator  -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    12    knee_angle_r_actuator    -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    13    ankle_angle_r_actuator   -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    14    hip_flexion_l_actuator   -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    15    hip_adduction_l_actuator -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    16    hip_rotation_l_actuator  -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    17    knee_angle_l_actuator    -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    18    ankle_angle_l_actuator   -1.0        1.0         False
    ===== ======================== =========== =========== ========


    Rewards
    --------

    The default reward function is based on the distance between the current center of mass velocity and the
    desired velocity in the x-axis. The desired velocity is given by the dataset to imitate.

    **Class**: :class:`loco_mujoco.utils.reward.TargetVelocityReward`

    Initial States
    ---------------

    The initial state is sampled by default from the dataset to imitate.

    Terminal States
    ----------------

    The terminal state is reached when the robot falls, or rather starts falling. The condition to check if the robot
    is falling is based on the orientation of the robot, the height of the center of mass, and the orientation of the
    back joint. More details can be found in the  :code:`_has_fallen` method of the environment.

    Methods
    ------------

    """

    valid_task_confs = ValidTaskConf(tasks=["walk", "run"],
                                     data_types=["real", "perfect"])

    mjx_enabled = False

    def __init__(self, disable_arms=False, disable_back_joint=False, spec=None,
                 observation_spec=None, action_spec=None, **kwargs):
        """
        Constructor.

        """

        self._disable_arms = disable_arms
        self._disable_back_joint = disable_back_joint

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
        if disable_arms or disable_back_joint:
            joints_to_remove, actuators_to_remove, equ_constraints_to_remove = self._get_spec_modifications()
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
            action_spec = [ac for ac in action_spec if ac not in actuators_to_remove]
            spec = self._delete_from_spec(spec, joints_to_remove,
                                          actuators_to_remove, equ_constraints_to_remove)
            if disable_arms:
                spec = self._reorient_arms(spec)

        super().__init__(spec, action_spec, observation_spec, enable_mjx=self.mjx_enabled,
                         **kwargs)

    def _get_spec_modifications(self):
        """
        Function that specifies which joints, motors and equality constraints
        should be removed from the Mujoco specification.

        Returns:
            A tuple of lists consisting of names of joints to remove, names of actuators to remove,
             and names of equality constraints to remove.

        """

        joints_to_remove = []
        actuators_to_remove = []
        equ_constr_to_remove = []

        if self._disable_arms:
            joints_to_remove += ["l_arm_shy", "l_arm_shx", "l_arm_shz", "left_elbow", "r_arm_shy",
                                 "r_arm_shx", "r_arm_shz", "right_elbow"]
            actuators_to_remove += ["l_arm_shy_actuator", "l_arm_shx_actuator", "l_arm_shz_actuator",
                                    "left_elbow_actuator", "r_arm_shy_actuator", "r_arm_shx_actuator",
                                    "r_arm_shz_actuator", "right_elbow_actuator"]

        if self._disable_back_joint:
            joints_to_remove += ["back_bkz"]
            actuators_to_remove += ["back_bkz_actuator"]

        return joints_to_remove, actuators_to_remove, equ_constr_to_remove

    @staticmethod
    def _reorient_arms(spec: MjSpec):
        """
        Reorients the elbow to not collide with the hip. Is used when disable_arms is set to True.

        Args:
            spec (MjSpec): Mujoco specification.

        Returns:
            Modified Mujoco specification.

        """
        # modify the arm orientation
        left_shoulder_pitch_link = spec.find_body("left_shoulder_pitch_link")
        left_shoulder_pitch_link.quat = [1.0, 0.25, 0.1, 0.0]
        right_elbow_link = spec.find_body("right_elbow_link")
        right_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]
        right_shoulder_pitch_link = spec.find_body("right_shoulder_pitch_link")
        right_shoulder_pitch_link.quat = [1.0, -0.25, 0.1, 0.0]
        left_elbow_link = spec.find_body("left_elbow_link")
        left_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]

        return spec

    @staticmethod
    def _get_observation_specification(spec: MjSpec):
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            A list of observation types.
        """
        observation_spec = [# ------------- JOINT POS -------------
                            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
                            ObservationType.JointPos("q_back_bkz", xml_name="back_bkz"),
                            ObservationType.JointPos("q_l_arm_shy", xml_name="l_arm_shy"),
                            ObservationType.JointPos("q_l_arm_shx", xml_name="l_arm_shx"),
                            ObservationType.JointPos("q_l_arm_shz", xml_name="l_arm_shz"),
                            ObservationType.JointPos("q_left_elbow", xml_name="left_elbow"),
                            ObservationType.JointPos("q_r_arm_shy", xml_name="r_arm_shy"),
                            ObservationType.JointPos("q_r_arm_shx", xml_name="r_arm_shx"),
                            ObservationType.JointPos("q_r_arm_shz", xml_name="r_arm_shz"),
                            ObservationType.JointPos("q_right_elbow", xml_name="right_elbow"),
                            ObservationType.JointPos("q_hip_flexion_r", xml_name="hip_flexion_r"),
                            ObservationType.JointPos("q_hip_adduction_r", xml_name="hip_adduction_r"),
                            ObservationType.JointPos("q_hip_rotation_r", xml_name="hip_rotation_r"),
                            ObservationType.JointPos("q_knee_angle_r", xml_name="knee_angle_r"),
                            ObservationType.JointPos("q_ankle_angle_r", xml_name="ankle_angle_r"),
                            ObservationType.JointPos("q_hip_flexion_l", xml_name="hip_flexion_l"),
                            ObservationType.JointPos("q_hip_adduction_l", xml_name="hip_adduction_l"),
                            ObservationType.JointPos("q_hip_rotation_l", xml_name="hip_rotation_l"),
                            ObservationType.JointPos("q_knee_angle_l", xml_name="knee_angle_l"),
                            ObservationType.JointPos("q_ankle_angle_l", xml_name="ankle_angle_l"),

                            # ------------- JOINT VEL -------------
                            ObservationType.FreeJointVel("dq_root", xml_name="root"),
                            ObservationType.JointVel("dq_back_bkz", xml_name="back_bkz"),
                            ObservationType.JointVel("dq_l_arm_shy", xml_name="l_arm_shy"),
                            ObservationType.JointVel("dq_l_arm_shx", xml_name="l_arm_shx"),
                            ObservationType.JointVel("dq_l_arm_shz", xml_name="l_arm_shz"),
                            ObservationType.JointVel("dq_left_elbow", xml_name="left_elbow"),
                            ObservationType.JointVel("dq_r_arm_shy", xml_name="r_arm_shy"),
                            ObservationType.JointVel("dq_r_arm_shx", xml_name="r_arm_shx"),
                            ObservationType.JointVel("dq_r_arm_shz", xml_name="r_arm_shz"),
                            ObservationType.JointVel("dq_right_elbow", xml_name="right_elbow"),
                            ObservationType.JointVel("dq_hip_flexion_r", xml_name="hip_flexion_r"),
                            ObservationType.JointVel("dq_hip_adduction_r", xml_name="hip_adduction_r"),
                            ObservationType.JointVel("dq_hip_rotation_r", xml_name="hip_rotation_r"),
                            ObservationType.JointVel("dq_knee_angle_r", xml_name="knee_angle_r"),
                            ObservationType.JointVel("dq_ankle_angle_r", xml_name="ankle_angle_r"),
                            ObservationType.JointVel("dq_hip_flexion_l", xml_name="hip_flexion_l"),
                            ObservationType.JointVel("dq_hip_adduction_l", xml_name="hip_adduction_l"),
                            ObservationType.JointVel("dq_hip_rotation_l", xml_name="hip_rotation_l"),
                            ObservationType.JointVel("dq_knee_angle_l", xml_name="knee_angle_l"),
                            ObservationType.JointVel("dq_ankle_angle_l", xml_name="ankle_angle_l")]

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
        action_spec = ["back_bkz_actuator", "l_arm_shy_actuator", "l_arm_shx_actuator",
                       "l_arm_shz_actuator", "left_elbow_actuator", "r_arm_shy_actuator", "r_arm_shx_actuator",
                       "r_arm_shz_actuator", "right_elbow_actuator", "hip_flexion_r_actuator",
                       "hip_adduction_r_actuator", "hip_rotation_r_actuator", "knee_angle_r_actuator",
                       "ankle_angle_r_actuator", "hip_flexion_l_actuator", "hip_adduction_l_actuator",
                       "hip_rotation_l_actuator", "knee_angle_l_actuator", "ankle_angle_l_actuator"]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls):
        """
        Returns the default path to the xml file of the environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "unitree_h1" / "h1.xml").as_posix()

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
        return "root"

    @info_property
    def root_height_healthy_range(self):
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        return (0.6, 1.5)
