import mujoco
from mujoco import MjSpec
import numpy as np

import loco_mujoco
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.environments import ValidTaskConf
from loco_mujoco.core import ObservationType
from loco_mujoco.core.utils import info_property


class Talos(BaseRobotHumanoid):

    """
    Description
    ------------

    Mujoco environment of the Talos robot. Optionally, Talos can carry
    a weight. This environment can be partially observable by hiding
    some of the state space entries from the policy using a state mask.
    Hidable entries are "positions", "velocities", "foot_forces",
    or "weight".

    Tasks
    -----------------
    * **Walking**: The robot has to walk forward with a fixed speed of 1.25 m/s.
    * **Carry**: The robot has to walk forward with a fixed speed of 1.25 m/s while carrying a weight.
      The mass is either specified by the user or sampled from a uniformly from [0.1 kg, 1 kg, 5 kg, 10 kg].


    Dataset Types
    -----------------
    The available dataset types for this environment can be found at: :ref:`env-label`.


    Observation Space
    -----------------

    The observation space has the following properties *by default* (i.e., only obs with Disabled == False):

    | For walking task: :code:`(min=-inf, max=inf, dim=34, dtype=float32)`
    | For carry task: :code:`(min=-inf, max=inf, dim=35, dtype=float32)`

    Some observations are **disabled by default**, but can be turned on. The detailed observation space is:

    ===== ============================================= ========== =========== =========================== === ========================
    Index Description                                   Min        Max         Disabled                    Dim Units
    ===== ============================================= ========== =========== =========================== === ========================
    0     Position of Joint pelvis_ty                   -inf       inf         False                       1   Position [m]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    1     Position of Joint pelvis_tilt                 -inf       inf         False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    2     Position of Joint pelvis_list                 -inf       inf         False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    3     Position of Joint pelvis_rotation             -inf       inf         False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    4     Position of Joint back_bkz                    -1.25664   1.25664     False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    5     Position of Joint back_bky                    -0.226893  0.733038    False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    6     Position of Joint l_arm_shz                   -1.5708    0.785398    True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    7     Position of Joint l_arm_shx                   0.00872665 2.87107     True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    8     Position of Joint l_arm_ely                   -2.42601   2.42601     True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    9     Position of Joint l_arm_elx                   -2.23402   0.00349066  True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    10    Position of Joint l_arm_wry                   -2.51327   2.51327     True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    11    Position of Joint l_arm_wrx                   -1.37008   1.37008     True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    12    Position of Joint r_arm_shz                   -0.785398  1.5708      True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    13    Position of Joint r_arm_shx                   -2.87107   -0.00872665 True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    14    Position of Joint r_arm_ely                   -2.42601   2.42601     True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    15    Position of Joint r_arm_elx                   -2.23402   0.00349066  True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    16    Position of Joint r_arm_wry                   -2.51327   2.51327     True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    17    Position of Joint r_arm_wrx                   -1.37008   1.37008     True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    18    Position of Joint hip_flexion_r               -2.095     0.7         False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    19    Position of Joint hip_adduction_r             -0.5236    0.5236      False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    20    Position of Joint hip_rotation_r              -1.5708    0.349066    False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    21    Position of Joint knee_angle_r                0.0        2.618       False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    22    Position of Joint ankle_angle_r               -1.27      0.68        False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    23    Position of Joint hip_flexion_l               -2.095     0.7         False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    24    Position of Joint hip_adduction_l             -0.5236    0.5236      False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    25    Position of Joint hip_rotation_l              -0.349066  1.5708      False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    26    Position of Joint knee_angle_l                0.0        2.618       False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    27    Position of Joint ankle_angle_l               -1.27      0.68        False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    28    Velocity of Joint pelvis_tx                   -inf       inf         False                       1   Velocity [m/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    29    Velocity of Joint pelvis_tz                   -inf       inf         False                       1   Velocity [m/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    30    Velocity of Joint pelvis_ty                   -inf       inf         False                       1   Velocity [m/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    31    Velocity of Joint pelvis_tilt                 -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    32    Velocity of Joint pelvis_list                 -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    33    Velocity of Joint pelvis_rotation             -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    34    Velocity of Joint back_bkz                    -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    35    Velocity of Joint back_bky                    -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    36    Velocity of Joint l_arm_shz                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    37    Velocity of Joint l_arm_shx                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    38    Velocity of Joint l_arm_ely                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    39    Velocity of Joint l_arm_elx                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    40    Velocity of Joint l_arm_wry                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    41    Velocity of Joint l_arm_wrx                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    42    Velocity of Joint r_arm_shz                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    43    Velocity of Joint r_arm_shx                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    44    Velocity of Joint r_arm_ely                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    45    Velocity of Joint r_arm_elx                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    46    Velocity of Joint r_arm_wry                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    47    Velocity of Joint r_arm_wrx                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    48    Velocity of Joint hip_flexion_r               -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    49    Velocity of Joint hip_adduction_r             -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    50    Velocity of Joint hip_rotation_r              -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    51    Velocity of Joint knee_angle_r                -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    52    Velocity of Joint ankle_angle_r               -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    53    Velocity of Joint hip_flexion_l               -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    54    Velocity of Joint hip_adduction_l             -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    55    Velocity of Joint hip_rotation_l              -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    56    Velocity of Joint knee_angle_l                -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    57    Velocity of Joint ankle_angle_l               -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    58    Mass of the Weight                            0.0        inf         Only Enabled for Carry Task 1   Mass [kg]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    59    3D linear Forces between Right Foot and Floor 0.0        inf         True                        3   Force [N]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    62    3D linear Forces between Left Foot and Floor  0.0        inf         True                        3   Force [N]
    ===== ============================================= ========== =========== =========================== === ========================

    Action Space
    ------------

    | The action space has the following properties *by default* (i.e., only actions with Disabled == False):
    | :code:`(min=-1, max=1, dim=12, dtype=float32)`

    The action range in LocoMuJoCo is always standardized, i.e. in [-1.0, 1.0].
    The XML of the environment specifies for each actuator a *gearing* ratio, which is used to scale the
    the action to the actual control range of the actuator.

    Some actions are **disabled by default**, but can be turned on. The detailed action space is:

    ===== ======================== =========== =========== ========
    Index Name in XML              Control Min Control Max Disabled
    ===== ======================== =========== =========== ========
    0     back_bkz_actuator        -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    1     back_bky_actuator        -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    2     l_arm_shz_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    3     l_arm_shx_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    4     l_arm_ely_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    5     l_arm_elx_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    6     l_arm_wry_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    7     l_arm_wrx_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    8     r_arm_shz_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    9     r_arm_shx_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    10    r_arm_ely_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    11    r_arm_elx_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    12    r_arm_wry_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    13    r_arm_wrx_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    14    hip_flexion_r_actuator   -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    15    hip_adduction_r_actuator -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    16    hip_rotation_r_actuator  -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    17    knee_angle_r_actuator    -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    18    ankle_angle_r_actuator   -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    19    hip_flexion_l_actuator   -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    20    hip_adduction_l_actuator -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    21    hip_rotation_l_actuator  -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    22    knee_angle_l_actuator    -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    23    ankle_angle_l_actuator   -1.0        1.0         False
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

    def __init__(self, disable_gripper=True, spec=None, observation_spec=None, action_spec=None, **kwargs):
        """
        Constructor.

        """

        self._disable_gripper = disable_gripper

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

        if disable_gripper:
            joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_spec_modifications()
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
            action_spec = [ac for ac in action_spec if ac not in motors_to_remove]

            spec = self._delete_from_spec(spec, joints_to_remove, motors_to_remove, equ_constr_to_remove)

        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(spec, action_spec, observation_spec, enable_mjx=self.mjx_enabled, **kwargs)

    def _get_spec_modifications(self):
        """
        Function that specifies which joints, motors and equality constraints
        should be removed from the Mujoco spec.

        Returns:
            A tuple of lists consisting of names of joints to remove, names of motors to remove,
             and names of equality constraints to remove.

        """

        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []

        if self._disable_gripper:
            joints_to_remove += [
                "gripper_left_joint",
                "gripper_left_inner_double_joint",
                "gripper_left_motor_single_joint",
                "gripper_left_inner_single_joint",
                "gripper_left_fingertip_1_joint",
                "gripper_left_fingertip_2_joint",
                "gripper_left_fingertip_3_joint",
                "gripper_right_joint",
                "gripper_right_inner_double_joint",
                "gripper_right_motor_single_joint",
                "gripper_right_inner_single_joint",
                "gripper_right_fingertip_1_joint",
                "gripper_right_fingertip_2_joint",
                "gripper_right_fingertip_3_joint"]

            motors_to_remove += [
                "gripper_left_joint_torque",
                "gripper_right_joint_torque"]

            equ_constr_to_remove += ["eq1", "eq2", "eq3", "eq4", "eq5", "eq6"]

        return joints_to_remove, motors_to_remove, equ_constr_to_remove

    @classmethod
    def get_default_xml_file_path(cls):
        """
        Returns the default path to the xml file of the environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "talos" / "talos.xml").as_posix()

    @info_property
    def grf_size(self):
        """
        Returns the size of the ground force vector.

        """

        return 6

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
            ObservationType.FreeJointPosNoXY("q_reference", xml_name="reference"),
            ObservationType.JointPos("q_torso_1_joint", xml_name="torso_1_joint"),
            ObservationType.JointPos("q_torso_2_joint", xml_name="torso_2_joint"),
            ObservationType.JointPos("q_head_1_joint", xml_name="head_1_joint"),
            ObservationType.JointPos("q_head_2_joint", xml_name="head_2_joint"),
            ObservationType.JointPos("q_arm_left_1_joint", xml_name="arm_left_1_joint"),
            ObservationType.JointPos("q_arm_left_2_joint", xml_name="arm_left_2_joint"),
            ObservationType.JointPos("q_arm_left_3_joint", xml_name="arm_left_3_joint"),
            ObservationType.JointPos("q_arm_left_4_joint", xml_name="arm_left_4_joint"),
            ObservationType.JointPos("q_arm_left_5_joint", xml_name="arm_left_5_joint"),
            ObservationType.JointPos("q_arm_left_6_joint", xml_name="arm_left_6_joint"),
            ObservationType.JointPos("q_arm_left_7_joint", xml_name="arm_left_7_joint"),
            ObservationType.JointPos("q_gripper_left_joint", xml_name="gripper_left_joint"),
            ObservationType.JointPos("q_gripper_left_inner_double_joint",
                                     xml_name="gripper_left_inner_double_joint"),
            ObservationType.JointPos("q_gripper_left_motor_single_joint",
                                     xml_name="gripper_left_motor_single_joint"),
            ObservationType.JointPos("q_gripper_left_inner_single_joint",
                                     xml_name="gripper_left_inner_single_joint"),
            ObservationType.JointPos("q_gripper_left_fingertip_1_joint", xml_name="gripper_left_fingertip_1_joint"),
            ObservationType.JointPos("q_gripper_left_fingertip_2_joint", xml_name="gripper_left_fingertip_2_joint"),
            ObservationType.JointPos("q_gripper_left_fingertip_3_joint", xml_name="gripper_left_fingertip_3_joint"),
            ObservationType.JointPos("q_arm_right_1_joint", xml_name="arm_right_1_joint"),
            ObservationType.JointPos("q_arm_right_2_joint", xml_name="arm_right_2_joint"),
            ObservationType.JointPos("q_arm_right_3_joint", xml_name="arm_right_3_joint"),
            ObservationType.JointPos("q_arm_right_4_joint", xml_name="arm_right_4_joint"),
            ObservationType.JointPos("q_arm_right_5_joint", xml_name="arm_right_5_joint"),
            ObservationType.JointPos("q_arm_right_6_joint", xml_name="arm_right_6_joint"),
            ObservationType.JointPos("q_arm_right_7_joint", xml_name="arm_right_7_joint"),
            ObservationType.JointPos("q_gripper_right_joint", xml_name="gripper_right_joint"),
            ObservationType.JointPos("q_gripper_right_inner_double_joint",
                                     xml_name="gripper_right_inner_double_joint"),
            ObservationType.JointPos("q_gripper_right_motor_single_joint",
                                     xml_name="gripper_right_motor_single_joint"),
            ObservationType.JointPos("q_gripper_right_inner_single_joint",
                                     xml_name="gripper_right_inner_single_joint"),
            ObservationType.JointPos("q_gripper_right_fingertip_1_joint",
                                     xml_name="gripper_right_fingertip_1_joint"),
            ObservationType.JointPos("q_gripper_right_fingertip_2_joint",
                                     xml_name="gripper_right_fingertip_2_joint"),
            ObservationType.JointPos("q_gripper_right_fingertip_3_joint",
                                     xml_name="gripper_right_fingertip_3_joint"),
            ObservationType.JointPos("q_leg_left_1_joint", xml_name="leg_left_1_joint"),
            ObservationType.JointPos("q_leg_left_2_joint", xml_name="leg_left_2_joint"),
            ObservationType.JointPos("q_leg_left_3_joint", xml_name="leg_left_3_joint"),
            ObservationType.JointPos("q_leg_left_4_joint", xml_name="leg_left_4_joint"),
            ObservationType.JointPos("q_leg_left_5_joint", xml_name="leg_left_5_joint"),
            ObservationType.JointPos("q_leg_left_6_joint", xml_name="leg_left_6_joint"),
            ObservationType.JointPos("q_leg_right_1_joint", xml_name="leg_right_1_joint"),
            ObservationType.JointPos("q_leg_right_2_joint", xml_name="leg_right_2_joint"),
            ObservationType.JointPos("q_leg_right_3_joint", xml_name="leg_right_3_joint"),
            ObservationType.JointPos("q_leg_right_4_joint", xml_name="leg_right_4_joint"),
            ObservationType.JointPos("q_leg_right_5_joint", xml_name="leg_right_5_joint"),
            ObservationType.JointPos("q_leg_right_6_joint", xml_name="leg_right_6_joint"),

            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_reference", xml_name="reference"),
            ObservationType.JointVel("dq_torso_1_joint", xml_name="torso_1_joint"),
            ObservationType.JointVel("dq_torso_2_joint", xml_name="torso_2_joint"),
            ObservationType.JointVel("dq_head_1_joint", xml_name="head_1_joint"),
            ObservationType.JointVel("dq_head_2_joint", xml_name="head_2_joint"),
            ObservationType.JointVel("dq_arm_left_1_joint", xml_name="arm_left_1_joint"),
            ObservationType.JointVel("dq_arm_left_2_joint", xml_name="arm_left_2_joint"),
            ObservationType.JointVel("dq_arm_left_3_joint", xml_name="arm_left_3_joint"),
            ObservationType.JointVel("dq_arm_left_4_joint", xml_name="arm_left_4_joint"),
            ObservationType.JointVel("dq_arm_left_5_joint", xml_name="arm_left_5_joint"),
            ObservationType.JointVel("dq_arm_left_6_joint", xml_name="arm_left_6_joint"),
            ObservationType.JointVel("dq_arm_left_7_joint", xml_name="arm_left_7_joint"),
            ObservationType.JointVel("dq_gripper_left_joint", xml_name="gripper_left_joint"),
            ObservationType.JointVel("dq_gripper_left_inner_double_joint",
                                     xml_name="gripper_left_inner_double_joint"),
            ObservationType.JointVel("dq_gripper_left_motor_single_joint",
                                     xml_name="gripper_left_motor_single_joint"),
            ObservationType.JointVel("dq_gripper_left_inner_single_joint",
                                     xml_name="gripper_left_inner_single_joint"),
            ObservationType.JointVel("dq_gripper_left_fingertip_1_joint",
                                     xml_name="gripper_left_fingertip_1_joint"),
            ObservationType.JointVel("dq_gripper_left_fingertip_2_joint",
                                     xml_name="gripper_left_fingertip_2_joint"),
            ObservationType.JointVel("dq_gripper_left_fingertip_3_joint",
                                     xml_name="gripper_left_fingertip_3_joint"),
            ObservationType.JointVel("dq_arm_right_1_joint", xml_name="arm_right_1_joint"),
            ObservationType.JointVel("dq_arm_right_2_joint", xml_name="arm_right_2_joint"),
            ObservationType.JointVel("dq_arm_right_3_joint", xml_name="arm_right_3_joint"),
            ObservationType.JointVel("dq_arm_right_4_joint", xml_name="arm_right_4_joint"),
            ObservationType.JointVel("dq_arm_right_5_joint", xml_name="arm_right_5_joint"),
            ObservationType.JointVel("dq_arm_right_6_joint", xml_name="arm_right_6_joint"),
            ObservationType.JointVel("dq_arm_right_7_joint", xml_name="arm_right_7_joint"),
            ObservationType.JointVel("dq_gripper_right_joint", xml_name="gripper_right_joint"),
            ObservationType.JointVel("dq_gripper_right_inner_double_joint",
                                     xml_name="gripper_right_inner_double_joint"),
            ObservationType.JointVel("dq_gripper_right_motor_single_joint",
                                     xml_name="gripper_right_motor_single_joint"),
            ObservationType.JointVel("dq_gripper_right_inner_single_joint",
                                     xml_name="gripper_right_inner_single_joint"),
            ObservationType.JointVel("dq_gripper_right_fingertip_1_joint",
                                     xml_name="gripper_right_fingertip_1_joint"),
            ObservationType.JointVel("dq_gripper_right_fingertip_2_joint",
                                     xml_name="gripper_right_fingertip_2_joint"),
            ObservationType.JointVel("dq_gripper_right_fingertip_3_joint",
                                     xml_name="gripper_right_fingertip_3_joint"),
            ObservationType.JointVel("dq_leg_left_1_joint", xml_name="leg_left_1_joint"),
            ObservationType.JointVel("dq_leg_left_2_joint", xml_name="leg_left_2_joint"),
            ObservationType.JointVel("dq_leg_left_3_joint", xml_name="leg_left_3_joint"),
            ObservationType.JointVel("dq_leg_left_4_joint", xml_name="leg_left_4_joint"),
            ObservationType.JointVel("dq_leg_left_5_joint", xml_name="leg_left_5_joint"),
            ObservationType.JointVel("dq_leg_left_6_joint", xml_name="leg_left_6_joint"),
            ObservationType.JointVel("dq_leg_right_1_joint", xml_name="leg_right_1_joint"),
            ObservationType.JointVel("dq_leg_right_2_joint", xml_name="leg_right_2_joint"),
            ObservationType.JointVel("dq_leg_right_3_joint", xml_name="leg_right_3_joint"),
            ObservationType.JointVel("dq_leg_right_4_joint", xml_name="leg_right_4_joint"),
            ObservationType.JointVel("dq_leg_right_5_joint", xml_name="leg_right_5_joint"),
            ObservationType.JointVel("dq_leg_right_6_joint", xml_name="leg_right_6_joint")
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
            "torso_1_joint_torque",
            "torso_2_joint_torque",
            "head_1_joint_torque",
            "head_2_joint_torque",
            "arm_left_1_joint_torque",
            "arm_left_2_joint_torque",
            "arm_left_3_joint_torque",
            "arm_left_4_joint_torque",
            "arm_left_5_joint_torque",
            "arm_left_6_joint_torque",
            "arm_left_7_joint_torque",
            "gripper_left_joint_torque",
            "arm_right_1_joint_torque",
            "arm_right_2_joint_torque",
            "arm_right_3_joint_torque",
            "arm_right_4_joint_torque",
            "arm_right_5_joint_torque",
            "arm_right_6_joint_torque",
            "arm_right_7_joint_torque",
            "gripper_right_joint_torque",
            "leg_left_1_joint_torque",
            "leg_left_2_joint_torque",
            "leg_left_3_joint_torque",
            "leg_left_4_joint_torque",
            "leg_left_5_joint_torque",
            "leg_left_6_joint_torque",
            "leg_right_1_joint_torque",
            "leg_right_2_joint_torque",
            "leg_right_3_joint_torque",
            "leg_right_4_joint_torque",
            "leg_right_5_joint_torque",
            "leg_right_6_joint_torque"
        ]

        return action_spec

    @info_property
    def upper_body_xml_name(self):
        return "torso_2_link"

    @info_property
    def root_body_name(self):
        return "base_link"

    @info_property
    def root_free_joint_xml_name(self):
        return "reference"

    @info_property
    def init_qpos(self):
        return np.array([0.0, 0.0, 1.08, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, -0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @info_property
    def init_qvel(self):
        return np.zeros(49)

    @info_property
    def root_height_healthy_range(self):
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        return (-10000, 100000)
