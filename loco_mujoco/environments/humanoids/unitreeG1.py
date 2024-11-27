import mujoco
from mujoco import MjSpec
import numpy as np

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.utils import check_validity_task_mode_dataset
from loco_mujoco.environments import ValidTaskConf
from loco_mujoco.utils import info_property


class UnitreeG1(BaseRobotHumanoid):

    """
    Description
    ------------

    Mujoco environment of the Unitree G1 robot.

    Tasks
    -----------------
    * **Walking**: The robot has to walk forward with a fixed speed of 1.25 m/s.
    * **Running**: Run forward with a fixed speed of 2.5 m/s.


    Dataset Types
    -----------------
    The available dataset types for this environment can be found at: :ref:`env-label`.


    Observation Space
    -----------------

    The observation space has the following properties *by default* (i.e., only obs with Disabled == False):

    | For walking task: :code:`(min=-inf, max=inf, dim=56, dtype=float32)`
    | For running task: :code:`(min=-inf, max=inf, dim=56, dtype=float32)`

    Some observations are **disabled by default**, but can be turned on. The detailed observation space is:

    ===== ============================================= ======== ====== ======== === ========================
    Index Description                                   Min      Max    Disabled Dim Units
    ===== ============================================= ======== ====== ======== === ========================
    0     Position of Joint pelvis_ty                   -inf     inf    False    1   Position [m]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    1     Position of Joint pelvis_tilt                 -inf     inf    False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    2     Position of Joint pelvis_list                 -inf     inf    False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    3     Position of Joint pelvis_rotation             -inf     inf    False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    4     Position of Joint left_hip_pitch_joint        -2.35    3.05   False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    5     Position of Joint left_hip_roll_joint         -0.26    2.53   False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    6     Position of Joint left_hip_yaw_joint          -2.75    2.75   False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    7     Position of Joint left_knee_joint             -0.33489 2.5449 False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    8     Position of Joint left_ankle_pitch_joint      -0.68    0.73   False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    9     Position of Joint left_ankle_roll_joint       -0.2618  0.2618 False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    10    Position of Joint right_hip_pitch_joint       -2.35    3.05   False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    11    Position of Joint right_hip_roll_joint        -2.53    0.26   False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    12    Position of Joint right_hip_yaw_joint         -2.75    2.75   False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    13    Position of Joint right_knee_joint            -0.33489 2.5449 False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    14    Position of Joint right_ankle_pitch_joint     -0.68    0.73   False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    15    Position of Joint right_ankle_roll_joint      -0.2618  0.2618 False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    16    Position of Joint torso_joint                 -2.618   2.618  False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    17    Position of Joint left_shoulder_pitch_joint   -2.9671  2.7925 False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    18    Position of Joint left_shoulder_roll_joint    -1.5882  2.2515 False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    19    Position of Joint left_shoulder_yaw_joint     -2.618   2.618  False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    20    Position of Joint left_elbow_pitch_joint      -0.2268  3.4208 False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    21    Position of Joint left_elbow_roll_joint       -2.0943  2.0943 False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    22    Position of Joint right_shoulder_pitch_joint  -2.9671  2.7925 False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    23    Position of Joint right_shoulder_roll_joint   -2.2515  1.5882 False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    24    Position of Joint right_shoulder_yaw_joint    -2.618   2.618  False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    25    Position of Joint right_elbow_pitch_joint     -0.2268  3.4208 False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    26    Position of Joint right_elbow_roll_joint      -2.0943  2.0943 False    1   Angle [rad]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    27    Velocity of Joint pelvis_tx                   -inf     inf    False    1   Velocity [m/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    28    Velocity of Joint pelvis_tz                   -inf     inf    False    1   Velocity [m/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    29    Velocity of Joint pelvis_ty                   -inf     inf    False    1   Velocity [m/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    30    Velocity of Joint pelvis_tilt                 -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    31    Velocity of Joint pelvis_list                 -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    32    Velocity of Joint pelvis_rotation             -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    33    Velocity of Joint left_hip_pitch_joint        -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    34    Velocity of Joint left_hip_roll_joint         -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    35    Velocity of Joint left_hip_yaw_joint          -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    36    Velocity of Joint left_knee_joint             -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    37    Velocity of Joint left_ankle_pitch_joint      -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    38    Velocity of Joint left_ankle_roll_joint       -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    39    Velocity of Joint right_hip_pitch_joint       -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    40    Velocity of Joint right_hip_roll_joint        -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    41    Velocity of Joint right_hip_yaw_joint         -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    42    Velocity of Joint right_knee_joint            -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    43    Velocity of Joint right_ankle_pitch_joint     -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    44    Velocity of Joint right_ankle_roll_joint      -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    45    Velocity of Joint torso_joint                 -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    46    Velocity of Joint left_shoulder_pitch_joint   -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    47    Velocity of Joint left_shoulder_roll_joint    -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    48    Velocity of Joint left_shoulder_yaw_joint     -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    49    Velocity of Joint left_elbow_pitch_joint      -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    50    Velocity of Joint left_elbow_roll_joint       -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    51    Velocity of Joint right_shoulder_pitch_joint  -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    52    Velocity of Joint right_shoulder_roll_joint   -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    53    Velocity of Joint right_shoulder_yaw_joint    -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    54    Velocity of Joint right_elbow_pitch_joint     -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    55    Velocity of Joint right_elbow_roll_joint      -inf     inf    False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    56    3D linear Forces between Right Foot and Floor 0.0      inf    True     3   Force [N]
    ----- --------------------------------------------- -------- ------ -------- --- ------------------------
    59    3D linear Forces between Left Foot and Floor  0.0      inf    True     3   Force [N]
    ===== ============================================= ======== ====== ======== === ========================

    Action Space
    ------------

    | The action space has the following properties *by default* (i.e., only actions with Disabled == False):
    | :code:`(min=-1, max=1, dim=23, dtype=float32)`

    ===== ========================== =========== =========== ========
    Index Name in XML                Control Min Control Max Disabled
    ===== ========================== =========== =========== ========
    0     left_hip_pitch_joint       -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    1     left_hip_roll_joint        -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    2     left_hip_yaw_joint         -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    3     left_knee_joint            -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    4     left_ankle_pitch_joint     -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    5     left_ankle_roll_joint      -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    6     right_hip_pitch_joint      -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    7     right_hip_roll_joint       -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    8     right_hip_yaw_joint        -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    9     right_knee_joint           -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    10    right_ankle_pitch_joint    -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    11    right_ankle_roll_joint     -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    12    torso_joint                -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    13    left_shoulder_pitch_joint  -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    14    left_shoulder_roll_joint   -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    15    left_shoulder_yaw_joint    -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    16    left_elbow_pitch_joint     -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    17    left_elbow_roll_joint      -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    18    right_shoulder_pitch_joint -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    19    right_shoulder_roll_joint  -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    20    right_shoulder_yaw_joint   -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    21    right_elbow_pitch_joint    -1.0        1.0         False
    ----- -------------------------- ----------- ----------- --------
    22    right_elbow_roll_joint     -1.0        1.0         False
    ===== ========================== =========== =========== ========

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
                                     data_types=["real"])
    mjx_enabled = False

    def __init__(self, disable_arms=False, disable_back_joint=False, xml_path=None, **kwargs):
        """
        Constructor.

        """

        self._disable_arms = disable_arms
        self._disable_back_joint = disable_back_joint

        if xml_path is None:
            xml_path = self.get_default_xml_file_path()

        # load the model specification
        spec = mujoco.MjSpec.from_file(xml_path)

        # get the observation and action space
        observation_spec = self._get_observation_specification(spec)
        action_spec = self._get_action_specification(spec)

        # modify the specification if needed
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)
        if disable_arms or disable_back_joint:
            joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_xml_modifications()
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
            action_spec = [ac for ac in action_spec if ac not in motors_to_remove]
            spec = self._delete_from_spec(spec, joints_to_remove,
                                          motors_to_remove, equ_constr_to_remove)
            if disable_arms:
                spec = self._reorient_arms(spec)

        super().__init__(spec, action_spec, observation_spec, enable_mjx=self.mjx_enabled,
                         **kwargs)

    def _get_ground_forces(self):
        """
        Calculates the ground forces (np.array). Per foot, the ground reaction force (linear --> 3D) is measured at
        4 points resulting in a 4*3*2=24 dimensional force vector.

        Returns:
            The ground forces (np.array) vector of all foots.

        """

        grf = np.concatenate([self._get_collision_force("floor", "right_foot_1")[:3],
                              self._get_collision_force("floor", "right_foot_2")[:3],
                              self._get_collision_force("floor", "right_foot_3")[:3],
                              self._get_collision_force("floor", "right_foot_4")[:3],
                              self._get_collision_force("floor", "left_foot_1")[:3],
                              self._get_collision_force("floor", "left_foot_2")[:3],
                              self._get_collision_force("floor", "left_foot_3")[:3],
                              self._get_collision_force("floor", "left_foot_4")[:3]])

        return grf

    @info_property
    def grf_size(self):
        """
        Returns:
            The size of the ground force vector.

        """

        return 24

    def _get_xml_modifications(self):
        """
        Function that specifies which joints, motors and equality constraints
        should be removed from the Mujoco xml.

        Returns:
            A tuple of lists consisting of names of joints to remove, names of motors to remove,
             and names of equality constraints to remove.

        """

        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []

        if self._disable_arms:
            joints_to_remove += ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                                 "right_elbow_pitch_joint", "right_elbow_roll_joint",
                                 "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                                 "left_elbow_pitch_joint", "left_elbow_roll_joint"]
            motors_to_remove += ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                                 "right_elbow_pitch_joint", "right_elbow_roll_joint",
                                 "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                                 "left_elbow_pitch_joint", "left_elbow_roll_joint"]

        if self._disable_back_joint:
            joints_to_remove += ["torso_joint"]
            motors_to_remove += ["torso_joint"]

        return joints_to_remove, motors_to_remove, equ_constr_to_remove

    @staticmethod
    def _get_observation_specification(spec):
        """
        Getter for the observation space specification.

        Returns:
            A list of tuples containing the specification of each observation
            space entry.

        """

        observation_spec = [# ------------- JOINT POS -------------
                            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
                            ObservationType.JointPos("q_left_hip_pitch_joint", xml_name="left_hip_pitch_joint"),
                            ObservationType.JointPos("q_left_hip_roll_joint", xml_name="left_hip_roll_joint"),
                            ObservationType.JointPos("q_left_hip_yaw_joint", xml_name="left_hip_yaw_joint"),
                            ObservationType.JointPos("q_left_knee_joint", xml_name="left_knee_joint"),
                            ObservationType.JointPos("q_left_ankle_pitch_joint", xml_name="left_ankle_pitch_joint"),
                            ObservationType.JointPos("q_left_ankle_roll_joint", xml_name="left_ankle_roll_joint"),
                            ObservationType.JointPos("q_right_hip_pitch_joint", xml_name="right_hip_pitch_joint"),
                            ObservationType.JointPos("q_right_hip_roll_joint", xml_name="right_hip_roll_joint"),
                            ObservationType.JointPos("q_right_hip_yaw_joint", xml_name="right_hip_yaw_joint"),
                            ObservationType.JointPos("q_right_knee_joint", xml_name="right_knee_joint"),
                            ObservationType.JointPos("q_right_ankle_pitch_joint", xml_name="right_ankle_pitch_joint"),
                            ObservationType.JointPos("q_right_ankle_roll_joint", xml_name="right_ankle_roll_joint"),
                            ObservationType.JointPos("q_waist_yaw_joint", xml_name="waist_yaw_joint"),
                            ObservationType.JointPos("q_left_shoulder_pitch_joint", xml_name="left_shoulder_pitch_joint"),
                            ObservationType.JointPos("q_left_shoulder_roll_joint", xml_name="left_shoulder_roll_joint"),
                            ObservationType.JointPos("q_left_shoulder_yaw_joint", xml_name="left_shoulder_yaw_joint"),
                            ObservationType.JointPos("q_left_elbow_joint", xml_name="left_elbow_joint"),
                            ObservationType.JointPos("q_left_wrist_roll_joint", xml_name="left_wrist_roll_joint"),
                            ObservationType.JointPos("q_right_shoulder_pitch_joint", xml_name="right_shoulder_pitch_joint"),
                            ObservationType.JointPos("q_right_shoulder_roll_joint", xml_name="right_shoulder_roll_joint"),
                            ObservationType.JointPos("q_right_shoulder_yaw_joint", xml_name="right_shoulder_yaw_joint"),
                            ObservationType.JointPos("q_right_elbow_joint", xml_name="right_elbow_joint"),
                            ObservationType.JointPos("q_right_wrist_roll_joint", xml_name="right_wrist_roll_joint"),

                            # ------------- JOINT VEL -------------
                            ObservationType.FreeJointVel("dq_root", xml_name="root"),
                            ObservationType.JointVel("dq_left_hip_pitch_joint", xml_name="left_hip_pitch_joint"),
                            ObservationType.JointVel("dq_left_hip_roll_joint", xml_name="left_hip_roll_joint"),
                            ObservationType.JointVel("dq_left_hip_yaw_joint", xml_name="left_hip_yaw_joint"),
                            ObservationType.JointVel("dq_left_knee_joint", xml_name="left_knee_joint"),
                            ObservationType.JointVel("dq_left_ankle_pitch_joint", xml_name="left_ankle_pitch_joint"),
                            ObservationType.JointVel("dq_left_ankle_roll_joint", xml_name="left_ankle_roll_joint"),
                            ObservationType.JointVel("dq_right_hip_pitch_joint", xml_name="right_hip_pitch_joint"),
                            ObservationType.JointVel("dq_right_hip_roll_joint", xml_name="right_hip_roll_joint"),
                            ObservationType.JointVel("dq_right_hip_yaw_joint", xml_name="right_hip_yaw_joint"),
                            ObservationType.JointVel("dq_right_knee_joint", xml_name="right_knee_joint"),
                            ObservationType.JointVel("dq_right_ankle_pitch_joint", xml_name="right_ankle_pitch_joint"),
                            ObservationType.JointVel("dq_right_ankle_roll_joint", xml_name="right_ankle_roll_joint"),
                            ObservationType.JointVel("dq_waist_yaw_joint", xml_name="waist_yaw_joint"),
                            ObservationType.JointVel("dq_left_shoulder_pitch_joint", xml_name="left_shoulder_pitch_joint"),
                            ObservationType.JointVel("dq_left_shoulder_roll_joint", xml_name="left_shoulder_roll_joint"),
                            ObservationType.JointVel("dq_left_shoulder_yaw_joint", xml_name="left_shoulder_yaw_joint"),
                            ObservationType.JointVel("dq_left_elbow_joint", xml_name="left_elbow_joint"),
                            ObservationType.JointVel("dq_left_wrist_roll_joint", xml_name="left_wrist_roll_joint"),
                            ObservationType.JointVel("dq_right_shoulder_pitch_joint", xml_name="right_shoulder_pitch_joint"),
                            ObservationType.JointVel("dq_right_shoulder_roll_joint", xml_name="right_shoulder_roll_joint"),
                            ObservationType.JointVel("dq_right_shoulder_yaw_joint", xml_name="right_shoulder_yaw_joint"),
                            ObservationType.JointVel("dq_right_elbow_joint", xml_name="right_elbow_joint"),
                            ObservationType.JointVel("dq_right_wrist_roll_joint", xml_name="right_wrist_roll_joint"),]

        return observation_spec

    @staticmethod
    def _get_action_specification(spec):
        """
        Getter for the action space specification.

        Returns:
            A list of actuator names.

        """
        action_spec = []
        for actuator in spec.actuators:
            action_spec.append(actuator.name)
        return action_spec

    @staticmethod
    def _reorient_arms(spec: MjSpec):
        """
        Reorients the elbow to not collide with the hip.

        Args:
            spec (MjSpec): to Mujoco specification.

        Returns:
            Modified Mujoco specification.

        """

        # modify the arm orientation
        left_shoulder_pitch_link = spec.find_body("left_shoulder_pitch_link")
        left_shoulder_pitch_link.quat = [1.0, 0.25, 0.1, 0.0]
        right_elbow_link = spec.find_body("right_elbow_pitch_link")
        right_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]
        right_shoulder_pitch_link = spec.find_body("right_shoulder_pitch_link")
        right_shoulder_pitch_link.quat = [1.0, -0.25, 0.1, 0.0]
        left_elbow_link = spec.find_body("left_elbow_pitch_link")
        left_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]

        return spec

    def _modify_spec_for_mjx(self, spec: MjSpec):
        raise NotImplementedError

    @classmethod
    def get_default_xml_file_path(cls):
        """
        Returns the default path to the xml file of the environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "unitree_g1" / "g1_23dof.xml").as_posix()

    @info_property
    def upper_body_xml_name(self):
        return "torso_link"

    @info_property
    def root_height_healthy_range(self):
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        # todo: check the range
        return (0.0, 1.0)
