import os
import warnings
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from dm_control import mjcf
from jax.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments import ValidTaskConf
from loco_mujoco.environments import LocoEnv
from loco_mujoco.utils import info_property


class BaseSkeleton(LocoEnv):
    """
    Mujoco environment of a base skeleton model.

    """

    valid_task_confs = ValidTaskConf(tasks=["walk", "run"],
                                     data_types=["real", "perfect"])

    mjx_enabled = False

    def __init__(self, use_muscles=False, use_box_feet=True, disable_arms=False, alpha_box_feet=0.5, **kwargs):
        """
        Constructor.

        Args:
            use_muscles (bool): If True, muscle actuators will be used, else torque actuators will be used.
            use_box_feet (bool): If True, boxes are used as feet (for simplification).
            disable_arms (bool): If True, all arm joints are removed and the respective
                actuators are removed from the action specification.
            alpha_box_feet (float): Alpha parameter of the boxes, which might be added as feet.

        """
        if use_muscles:
            xml_path = (Path(__file__).resolve().parent.parent.parent / "models" / "humanoid" /
                        "humanoid_muscle.xml").as_posix()
        else:
            xml_path = (Path(__file__).resolve().parent.parent.parent / "models" / "humanoid" /
                        "humanoid_torque.xml").as_posix()

        action_spec = self._get_action_specification(use_muscles)

        observation_spec = self._get_observation_specification()

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._use_muscles = use_muscles
        self._use_box_feet = use_box_feet
        self._disable_arms = disable_arms
        joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_xml_modifications()

        xml_handle = mjcf.from_path(xml_path)
        if self._use_box_feet or self._disable_arms:
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
            action_spec = [ac for ac in action_spec if ac not in motors_to_remove]

            xml_handle = self._delete_from_xml_handle(xml_handle, joints_to_remove,
                                                      motors_to_remove, equ_constr_to_remove)
            if self._use_box_feet:
                xml_handle = self._add_box_feet_to_xml_handle(xml_handle, alpha_box_feet)

            if self._disable_arms:
                xml_handle = self._reorient_arms(xml_handle)

        if self.mjx_enabled:
            assert use_box_feet
            xml_handle = self._modify_xml_for_mjx(xml_handle)

        super().__init__(xml_handle, action_spec, observation_spec, enable_mjx=self.mjx_enabled, **kwargs)

    def _get_xml_modifications(self):
        """
        Function that specifies which joints, motors and equality constraints
        should be removed from the Mujoco xml. Also the required collision
        groups will be returned.

        Returns:
            A tuple of lists consisting of names of joints to remove, names of motors to remove,
             names of equality constraints to remove, and names of collision groups to be used.

        """

        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []
        if self._use_box_feet:
            joints_to_remove += ["subtalar_angle_l", "mtp_angle_l", "subtalar_angle_r", "mtp_angle_r"]
            if not self._use_muscles:
                motors_to_remove += ["mot_subtalar_angle_l", "mot_mtp_angle_l", "mot_subtalar_angle_r", "mot_mtp_angle_r"]
            equ_constr_to_remove += [j + "_constraint" for j in joints_to_remove]

        if self._disable_arms:
            joints_to_remove += ["arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r", "wrist_flex_r",
                                 "wrist_dev_r", "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l",
                                 "wrist_flex_l", "wrist_dev_l"]
            motors_to_remove += ["mot_shoulder_flex_r", "mot_shoulder_add_r", "mot_shoulder_rot_r", "mot_elbow_flex_r",
                                 "mot_pro_sup_r", "mot_wrist_flex_r", "mot_wrist_dev_r", "mot_shoulder_flex_l",
                                 "mot_shoulder_add_l", "mot_shoulder_rot_l", "mot_elbow_flex_l", "mot_pro_sup_l",
                                 "mot_wrist_flex_l", "mot_wrist_dev_l"]
            equ_constr_to_remove += ["wrist_flex_r_constraint", "wrist_dev_r_constraint",
                                     "wrist_flex_l_constraint", "wrist_dev_l_constraint"]

        return joints_to_remove, motors_to_remove, equ_constr_to_remove

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

        return LocoEnv.generate(cls, task, dataset_type,
                                clip_trajectory_to_joint_ranges=True, **kwargs)


    @staticmethod
    def _get_observation_specification():
        """
        Getter for the observation space specification.

        Returns:
            A list of tuples containing the specification of each observation
            space entry.

        """

        observation_spec = [  # ------------- JOINT POS -------------
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),

            # ObservationType.JointPos("q_pelvis_tx", xml_name="pelvis_tx"),
            # ObservationType.JointPos("q_pelvis_tz", xml_name="pelvis_tz"),
            # ObservationType.JointPos("q_pelvis_ty", xml_name="pelvis_ty"),
            # ObservationType.JointPos("q_pelvis_tilt", xml_name="pelvis_tilt"),
            # ObservationType.JointPos("q_pelvis_list", xml_name="pelvis_list"),
            # ObservationType.JointPos("q_pelvis_rotation", xml_name="pelvis_rotation"),
            # --- lower limb right ---
            ObservationType.JointPos("q_hip_flexion_r", xml_name="hip_flexion_r"),
            ObservationType.JointPos("q_hip_adduction_r", xml_name="hip_adduction_r"),
            ObservationType.JointPos("q_hip_rotation_r", xml_name="hip_rotation_r"),
            ObservationType.JointPos("q_knee_angle_r", xml_name="knee_angle_r"),
            ObservationType.JointPos("q_ankle_angle_r", xml_name="ankle_angle_r"),
            ObservationType.JointPos("q_subtalar_angle_r", xml_name="subtalar_angle_r"),
            ObservationType.JointPos("q_mtp_angle_r", xml_name="mtp_angle_r"),
            # --- lower limb left ---
            ObservationType.JointPos("q_hip_flexion_l", xml_name="hip_flexion_l"),
            ObservationType.JointPos("q_hip_adduction_l", xml_name="hip_adduction_l"),
            ObservationType.JointPos("q_hip_rotation_l", xml_name="hip_rotation_l"),
            ObservationType.JointPos("q_knee_angle_l", xml_name="knee_angle_l"),
            ObservationType.JointPos("q_ankle_angle_l", xml_name="ankle_angle_l"),
            ObservationType.JointPos("q_subtalar_angle_l", xml_name="subtalar_angle_l"),
            ObservationType.JointPos("q_mtp_angle_l", xml_name="mtp_angle_l"),
            # --- lumbar ---
            ObservationType.JointPos("q_lumbar_extension", xml_name="lumbar_extension"),
            ObservationType.JointPos("q_lumbar_bending", xml_name="lumbar_bending"),
            ObservationType.JointPos("q_lumbar_rotation", xml_name="lumbar_rotation"),
            # --- upper body right ---
            ObservationType.JointPos("q_arm_flex_r", xml_name="arm_flex_r"),
            ObservationType.JointPos("q_arm_add_r", xml_name="arm_add_r"),
            ObservationType.JointPos("q_arm_rot_r", xml_name="arm_rot_r"),
            ObservationType.JointPos("q_elbow_flex_r", xml_name="elbow_flex_r"),
            ObservationType.JointPos("q_pro_sup_r", xml_name="pro_sup_r"),
            ObservationType.JointPos("q_wrist_flex_r", xml_name="wrist_flex_r"),
            ObservationType.JointPos("q_wrist_dev_r", xml_name="wrist_dev_r"),
            # --- upper body left ---
            ObservationType.JointPos("q_arm_flex_l", xml_name="arm_flex_l"),
            ObservationType.JointPos("q_arm_add_l", xml_name="arm_add_l"),
            ObservationType.JointPos("q_arm_rot_l", xml_name="arm_rot_l"),
            ObservationType.JointPos("q_elbow_flex_l", xml_name="elbow_flex_l"),
            ObservationType.JointPos("q_pro_sup_l", xml_name="pro_sup_l"),
            ObservationType.JointPos("q_wrist_flex_l", xml_name="wrist_flex_l"),
            ObservationType.JointPos("q_wrist_dev_l", xml_name="wrist_dev_l"),

            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_root", xml_name="root"),

            # ObservationType.JointVel("dq_pelvis_tx", xml_name="pelvis_tx"),
            # ObservationType.JointVel("dq_pelvis_tz", xml_name="pelvis_tz"),
            # ObservationType.JointVel("dq_pelvis_ty", xml_name="pelvis_ty"),
            # ObservationType.JointVel("dq_pelvis_tilt", xml_name="pelvis_tilt"),
            # ObservationType.JointVel("dq_pelvis_list", xml_name="pelvis_list"),
            # ObservationType.JointVel("dq_pelvis_rotation", xml_name="pelvis_rotation"),
            # --- lower limb right ---
            ObservationType.JointVel("dq_hip_flexion_r", xml_name="hip_flexion_r"),
            ObservationType.JointVel("dq_hip_adduction_r", xml_name="hip_adduction_r"),
            ObservationType.JointVel("dq_hip_rotation_r", xml_name="hip_rotation_r"),
            ObservationType.JointVel("dq_knee_angle_r", xml_name="knee_angle_r"),
            ObservationType.JointVel("dq_ankle_angle_r", xml_name="ankle_angle_r"),
            ObservationType.JointVel("dq_subtalar_angle_r", xml_name="subtalar_angle_r"),
            ObservationType.JointVel("dq_mtp_angle_r", xml_name="mtp_angle_r"),
            # --- lower limb left ---
            ObservationType.JointVel("dq_hip_flexion_l", xml_name="hip_flexion_l"),
            ObservationType.JointVel("dq_hip_adduction_l", xml_name="hip_adduction_l"),
            ObservationType.JointVel("dq_hip_rotation_l", xml_name="hip_rotation_l"),
            ObservationType.JointVel("dq_knee_angle_l", xml_name="knee_angle_l"),
            ObservationType.JointVel("dq_ankle_angle_l", xml_name="ankle_angle_l"),
            ObservationType.JointVel("dq_subtalar_angle_l", xml_name="subtalar_angle_l"),
            ObservationType.JointVel("dq_mtp_angle_l", xml_name="mtp_angle_l"),
            # --- lumbar ---
            ObservationType.JointVel("dq_lumbar_extension", xml_name="lumbar_extension"),
            ObservationType.JointVel("dq_lumbar_bending", xml_name="lumbar_bending"),
            ObservationType.JointVel("dq_lumbar_rotation", xml_name="lumbar_rotation"),
            # --- upper body right ---
            ObservationType.JointVel("dq_arm_flex_r", xml_name="arm_flex_r"),
            ObservationType.JointVel("dq_arm_add_r", xml_name="arm_add_r"),
            ObservationType.JointVel("dq_arm_rot_r", xml_name="arm_rot_r"),
            ObservationType.JointVel("dq_elbow_flex_r", xml_name="elbow_flex_r"),
            ObservationType.JointVel("dq_pro_sup_r", xml_name="pro_sup_r"),
            ObservationType.JointVel("dq_wrist_flex_r", xml_name="wrist_flex_r"),
            ObservationType.JointVel("dq_wrist_dev_r", xml_name="wrist_dev_r"),
            # --- upper body left ---
            ObservationType.JointVel("dq_arm_flex_l", xml_name="arm_flex_l"),
            ObservationType.JointVel("dq_arm_add_l", xml_name="arm_add_l"),
            ObservationType.JointVel("dq_arm_rot_l", xml_name="arm_rot_l"),
            ObservationType.JointVel("dq_elbow_flex_l", xml_name="elbow_flex_l"),
            ObservationType.JointVel("dq_pro_sup_l", xml_name="pro_sup_l"),
            ObservationType.JointVel("dq_wrist_flex_l", xml_name="wrist_flex_l"),
            ObservationType.JointVel("dq_wrist_dev_l", xml_name="wrist_dev_l")]

        return observation_spec

    @staticmethod
    def _get_action_specification(use_muscles):
        """
        Getter for the action space specification.

        Returns:
            A list of tuples containing the specification of each action
            space entry.

        """
        if use_muscles:
            action_spec = ["mot_shoulder_flex_r", "mot_shoulder_add_r", "mot_shoulder_rot_r", "mot_elbow_flex_r",
                           "mot_pro_sup_r", "mot_wrist_flex_r", "mot_wrist_dev_r", "mot_shoulder_flex_l",
                           "mot_shoulder_add_l", "mot_shoulder_rot_l", "mot_elbow_flex_l", "mot_pro_sup_l",
                           "mot_wrist_flex_l", "mot_wrist_dev_l", "glut_med1_r", "glut_med2_r",
                           "glut_med3_r", "glut_min1_r", "glut_min2_r", "glut_min3_r", "semimem_r", "semiten_r",
                           "bifemlh_r", "bifemsh_r", "sar_r", "add_long_r", "add_brev_r", "add_mag1_r", "add_mag2_r",
                           "add_mag3_r", "tfl_r", "pect_r", "grac_r", "glut_max1_r", "glut_max2_r", "glut_max3_r",
                           "iliacus_r", "psoas_r", "quad_fem_r", "gem_r", "peri_r", "rect_fem_r", "vas_med_r",
                           "vas_int_r", "vas_lat_r", "med_gas_r", "lat_gas_r", "soleus_r", "tib_post_r",
                           "flex_dig_r", "flex_hal_r", "tib_ant_r", "per_brev_r", "per_long_r", "per_tert_r",
                           "ext_dig_r", "ext_hal_r", "glut_med1_l", "glut_med2_l", "glut_med3_l", "glut_min1_l",
                           "glut_min2_l", "glut_min3_l", "semimem_l", "semiten_l", "bifemlh_l", "bifemsh_l",
                           "sar_l", "add_long_l", "add_brev_l", "add_mag1_l", "add_mag2_l", "add_mag3_l",
                           "tfl_l", "pect_l", "grac_l", "glut_max1_l", "glut_max2_l", "glut_max3_l",
                           "iliacus_l", "psoas_l", "quad_fem_l", "gem_l", "peri_l", "rect_fem_l",
                           "vas_med_l", "vas_int_l", "vas_lat_l", "med_gas_l", "lat_gas_l", "soleus_l",
                           "tib_post_l", "flex_dig_l", "flex_hal_l", "tib_ant_l", "per_brev_l", "per_long_l",
                           "per_tert_l", "ext_dig_l", "ext_hal_l", "ercspn_r", "ercspn_l", "intobl_r",
                           "intobl_l", "extobl_r", "extobl_l"]
        else:
            action_spec = ["mot_lumbar_ext", "mot_lumbar_bend", "mot_lumbar_rot", "mot_shoulder_flex_r",
                           "mot_shoulder_add_r", "mot_shoulder_rot_r", "mot_elbow_flex_r", "mot_pro_sup_r",
                           "mot_wrist_flex_r", "mot_wrist_dev_r", "mot_shoulder_flex_l", "mot_shoulder_add_l",
                           "mot_shoulder_rot_l", "mot_elbow_flex_l", "mot_pro_sup_l", "mot_wrist_flex_l",
                           "mot_wrist_dev_l", "mot_hip_flexion_r", "mot_hip_adduction_r", "mot_hip_rotation_r",
                           "mot_knee_angle_r", "mot_ankle_angle_r", "mot_subtalar_angle_r", "mot_mtp_angle_r",
                           "mot_hip_flexion_l", "mot_hip_adduction_l", "mot_hip_rotation_l", "mot_knee_angle_l",
                           "mot_ankle_angle_l", "mot_subtalar_angle_l", "mot_mtp_angle_l"]

        return action_spec

    @staticmethod
    def _add_box_feet_to_xml_handle(xml_handle, alpha_box_feet, scaling=1.0):
        """
        Adds box feet to Mujoco XML handle and makes old feet non-collidable.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Modified Mujoco XML handle.

        """

        # find foot and attach box
        toe_l = xml_handle.find("body", "toes_l")
        size = np.array([0.112, 0.03, 0.05]) * scaling
        pos = np.array([-0.09, 0.019, 0.0]) * scaling
        toe_l.add("geom", name="foot_box_l", type="box", size=size.tolist(), pos=pos.tolist(),
                  rgba=[0.5, 0.5, 0.5, alpha_box_feet], euler=[0.0, 0.15, 0.0])
        toe_r = xml_handle.find("body", "toes_r")
        toe_r.add("geom", name="foot_box_r", type="box", size=size.tolist(), pos=pos.tolist(),
                  rgba=[0.5, 0.5, 0.5, alpha_box_feet], euler=[0.0, -0.15, 0.0])

        # make true foot uncollidable
        foot_r = xml_handle.find("geom", "r_foot")
        bofoot_r = xml_handle.find("geom", "r_bofoot")
        foot_l = xml_handle.find("geom", "l_foot")
        bofoot_l = xml_handle.find("geom", "l_bofoot")
        foot_r.contype = 0
        foot_r.conaffinity = 0
        bofoot_r.contype = 0
        bofoot_r.conaffinity = 0
        foot_l.contype = 0
        foot_l.conaffinity = 0
        bofoot_l.contype = 0
        bofoot_l.conaffinity = 0

        return xml_handle

    @staticmethod
    def _reorient_arms(xml_handle):
        """
        Reorients the arm of a humanoid model given its Mujoco XML handle.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Modified Mujoco XML handle.

        """

        h = xml_handle.find("body", "humerus_l")
        h.quat = [1.0, -0.1, -1.0, -0.1]
        h = xml_handle.find("body", "ulna_l")
        h.quat = [1.0, 0.6, 0.0, 0.0]
        h = xml_handle.find("body", "humerus_r")
        h.quat = [1.0, 0.1, 1.0, -0.1]
        h = xml_handle.find("body", "ulna_r")
        h.quat = [1.0, -0.6, 0.0, 0.0]

        return xml_handle

    @info_property
    def root_height_healthy_range(self):
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        return (0.8, 1.1)

    @info_property
    def upper_body_xml_name(self):
        return "torso"

    def _modify_xml_for_mjx(self, xml_handle):
        """
        Mjx is bad in handling many complex contacts. To speed-up simulation significantly we apply
        some changes to the XML:
            1. Disable all contacts except the ones between feet and the floor.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Mujoco XML handle.

        """

        # --- disable all contacts in geom except foot ---
        geoms = xml_handle.find_all("geom")
        for g in geoms:
            if "foot_box" not in g.name and "floor" not in g.name:
                g.contype = 0
                g.conaffinity = 0

        return xml_handle

    @info_property
    def sites_for_mimic(self):
        return ["upper_body_mimic", "head_mimic", "pelvis_mimic",
                "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic",
                "left_hip_mimic", "left_knee_mimic", "left_foot_mimic",
                "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
                "right_hip_mimic", "right_knee_mimic", "right_foot_mimic"]