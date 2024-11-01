from pathlib import Path
from copy import deepcopy

import numpy as np
from dm_control import mjcf

# from mushroom_rl.utils.running_stats import *

from loco_mujoco.core import ObservationType
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.utils import check_validity_task_mode_dataset
from loco_mujoco.environments import ValidTaskConf
from loco_mujoco.utils import info_property


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

    def __init__(self, disable_arms=False, disable_back_joint=False, **kwargs):
        """
        Constructor.

        """

        xml_path = (Path(__file__).resolve().parent.parent.parent / "models" / "unitree_h1" / "h1.xml").as_posix()

        action_spec = self._get_action_specification()

        observation_spec = self._get_observation_specification()

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._disable_arms = disable_arms
        self._disable_back_joint = disable_back_joint

        xml_handles = []
        xml_handle = mjcf.from_path(xml_path)

        if self.mjx_enabled:
            xml_handle = self._modify_xml_for_mjx(xml_handle)

        if disable_arms or disable_back_joint:

            if disable_arms or disable_back_joint:
                joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_xml_modifications()
                obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
                observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
                action_spec = [ac for ac in action_spec if ac not in motors_to_remove]

                xml_handle = self._delete_from_xml_handle(xml_handle, joints_to_remove,
                                                          motors_to_remove, equ_constr_to_remove)

                xml_handle = self._reorient_arms(xml_handle)

                xml_handles.append(xml_handle)
        else:
            xml_handles.append(xml_handle)

        super().__init__(xml_handles, action_spec, observation_spec, enable_mjx=self.mjx_enabled,
                         **kwargs)

    @info_property
    def grf_size(self):
        """
        Returns the size of the ground force vector.

        """

        return 6

    @info_property
    def upper_body_xml_name(self):
        return "torso_link"

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
            joints_to_remove += ["l_arm_shy", "l_arm_shx", "l_arm_shz", "left_elbow", "r_arm_shy",
                                 "r_arm_shx", "r_arm_shz", "right_elbow"]
            motors_to_remove += ["l_arm_shy_actuator", "l_arm_shx_actuator", "l_arm_shz_actuator",
                                 "left_elbow_actuator", "r_arm_shy_actuator", "r_arm_shx_actuator",
                                 "r_arm_shz_actuator", "right_elbow_actuator"]

        if self._disable_back_joint:
            joints_to_remove += ["back_bkz"]
            motors_to_remove += ["back_bkz_actuator"]

        return joints_to_remove, motors_to_remove, equ_constr_to_remove

    def _has_fallen(self, obs, info, data, return_err_msg=False):
        """
        Checks if a model has fallen.

        Args:
            obs (np.array): Current observation.
            return_err_msg (bool): If True, an error message with violations is returned.

        Returns:
            True, if the model has fallen for the current observation, False otherwise.
            Optionally an error message is returned.
        """

        pelvis_cond, pelvis_y_cond, pelvis_tilt_cond, pelvis_list_cond, pelvis_rotation_cond = (
            self._has_fallen_compat(obs, info, data, np))

        if return_err_msg:
            error_msg = ""
            if pelvis_y_cond:
                error_msg += "pelvis_y_condition violated.\n"
            elif pelvis_tilt_cond:
                error_msg += "pelvis_tilt_condition violated.\n"
            elif pelvis_list_cond:
                error_msg += "pelvis_list_condition violated.\n"
            elif pelvis_rotation_cond:
                error_msg += "pelvis_rotation_condition violated.\n"
            return pelvis_cond, error_msg
        else:

            return pelvis_cond

    def _has_fallen_compat(self, obs, info, data, backend):

        q_pelvis_y = self._get_from_obs(obs, "q_pelvis_ty")
        q_pelvis_tilt = self._get_from_obs(obs, "q_pelvis_tilt")
        q_pelvis_list = self._get_from_obs(obs, "q_pelvis_list")
        q_pelvis_rotation = self._get_from_obs(obs, "q_pelvis_rotation")

        pelvis_y_cond = backend.logical_or(backend.less(q_pelvis_y, -0.3),
                                           backend.greater(q_pelvis_y, 0.1))
        pelvis_tilt_cond = backend.logical_or(backend.less(q_pelvis_tilt, -backend.pi / 4.5),
                                              backend.greater(q_pelvis_tilt, backend.pi / 12))
        pelvis_list_cond = backend.logical_or(backend.less(q_pelvis_list, -backend.pi / 12),
                                              backend.greater(q_pelvis_list, backend.pi / 8))
        pelvis_rotation_cond = backend.logical_or(backend.less(q_pelvis_rotation, -backend.pi / 8),
                                                  backend.greater(q_pelvis_rotation, backend.pi / 8))

        pelvis_cond = backend.logical_or(backend.logical_or(pelvis_y_cond, pelvis_tilt_cond),
                                         backend.logical_or(pelvis_list_cond, pelvis_rotation_cond))

        pelvis_cond = backend.squeeze(pelvis_cond)
        pelvis_y_cond = backend.squeeze(pelvis_y_cond)
        pelvis_tilt_cond = backend.squeeze(pelvis_tilt_cond)
        pelvis_list_cond = backend.squeeze(pelvis_list_cond)
        pelvis_rotation_cond = backend.squeeze(pelvis_rotation_cond)

        return pelvis_cond, pelvis_y_cond, pelvis_tilt_cond, pelvis_list_cond, pelvis_rotation_cond

    @classmethod
    def generate(cls, task="walk", dataset_type="real", **kwargs):
        """
        Returns an environment corresponding to the specified task.

        Args:
        task (str): Main task to solve.
        dataset_type (str): "real" or "perfect". "real" uses real motion capture data as the
        reference trajectory. This data does not perfectly match the kinematics
        and dynamics of this environment, hence it is more challenging. "perfect" uses
        a perfect dataset.

        """
        check_validity_task_mode_dataset(UnitreeH1.__name__, task, None, dataset_type,
                                         *UnitreeH1.valid_task_confs.get_all())

        return BaseRobotHumanoid.generate(cls, task, dataset_type,
                                          clip_trajectory_to_joint_ranges=True, **kwargs)

    @staticmethod
    def _reorient_arms(xml_handle):
        """
        Reorients the elbow to not collide with the hip. Is used when disable_arms is set to True.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Modified Mujoco XML handle.

        """
        # modify the arm orientation
        left_shoulder_pitch_link = xml_handle.find("body", "left_shoulder_pitch_link")
        left_shoulder_pitch_link.quat = [1.0, 0.25, 0.1, 0.0]
        right_elbow_link = xml_handle.find("body", "right_elbow_link")
        right_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]
        right_shoulder_pitch_link = xml_handle.find("body", "right_shoulder_pitch_link")
        right_shoulder_pitch_link.quat = [1.0, -0.25, 0.1, 0.0]
        left_elbow_link = xml_handle.find("body", "left_elbow_link")
        left_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]

        return xml_handle

    @staticmethod
    def _get_observation_specification():
        """
        Getter for the observation space specification.

        Returns:
            A list of tuples containing the specification of each observation
            space entry.

        """

        observation_spec = [# ------------- JOINT POS -------------
                            ObservationType.JointPos("q_pelvis_tx", xml_name="pelvis_tx"),
                            ObservationType.JointPos("q_pelvis_tz", xml_name="pelvis_tz"),
                            ObservationType.JointPos("q_pelvis_ty", xml_name="pelvis_ty"),
                            ObservationType.JointPos("q_pelvis_tilt", xml_name="pelvis_tilt"),
                            ObservationType.JointPos("q_pelvis_list", xml_name="pelvis_list"),
                            ObservationType.JointPos("q_pelvis_rotation", xml_name="pelvis_rotation"),
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
                            ObservationType.JointVel("dq_pelvis_tx", xml_name="pelvis_tx"),
                            ObservationType.JointVel("dq_pelvis_tz", xml_name="pelvis_tz"),
                            ObservationType.JointVel("dq_pelvis_ty", xml_name="pelvis_ty"),
                            ObservationType.JointVel("dq_pelvis_tilt", xml_name="pelvis_tilt"),
                            ObservationType.JointVel("dq_pelvis_list", xml_name="pelvis_list"),
                            ObservationType.JointVel("dq_pelvis_rotation", xml_name="pelvis_rotation"),
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
    def _get_action_specification():
        """
        Getter for the action space specification.

        Returns:
            A list of tuples containing the specification of each action
            space entry.

        """

        action_spec = ["back_bkz_actuator", "l_arm_shy_actuator", "l_arm_shx_actuator",
                       "l_arm_shz_actuator", "left_elbow_actuator", "r_arm_shy_actuator", "r_arm_shx_actuator",
                       "r_arm_shz_actuator", "right_elbow_actuator", "hip_flexion_r_actuator",
                       "hip_adduction_r_actuator", "hip_rotation_r_actuator", "knee_angle_r_actuator",
                       "ankle_angle_r_actuator", "hip_flexion_l_actuator", "hip_adduction_l_actuator",
                       "hip_rotation_l_actuator", "knee_angle_l_actuator", "ankle_angle_l_actuator"]

        return action_spec
