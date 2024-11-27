import numpy as np
import mujoco
from mujoco import MjSpec

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

    def __init__(self, use_muscles=False, use_box_feet=True, disable_arms=False,
                 alpha_box_feet=0.5, xml_path=None, **kwargs):
        """
        Constructor.

        Args:
            use_muscles (bool): If True, muscle actuators will be used, else torque actuators will be used.
            use_box_feet (bool): If True, boxes are used as feet (for simplification).
            disable_arms (bool): If True, all arm joints are removed and the respective
                actuators are removed from the action specification.
            alpha_box_feet (float): Alpha parameter of the boxes, which might be added as feet.

        """

        if xml_path is None:
            xml_path = self.get_default_xml_file_path()

        # load the model specification
        spec = mujoco.MjSpec.from_file(xml_path)

        # get the observation and action space
        observation_spec = self._get_observation_specification(spec)
        action_spec = self._get_action_specification(spec)

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._use_muscles = use_muscles
        self._use_box_feet = use_box_feet
        self._disable_arms = disable_arms
        joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_spec_modifications()

        if self._use_box_feet or self._disable_arms:
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
            action_spec = [ac for ac in action_spec if ac not in motors_to_remove]

            spec = self._delete_from_spec(spec, joints_to_remove,
                                          motors_to_remove, equ_constr_to_remove)
            if self._use_box_feet:
                spec = self._add_box_feet_to_spec(spec, alpha_box_feet)

            if self._disable_arms:
                spec = self._reorient_arms(spec)

        if self.mjx_enabled:
            assert use_box_feet
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(spec, action_spec, observation_spec, enable_mjx=self.mjx_enabled, **kwargs)

    def _get_spec_modifications(self):
        """
        Function that specifies which joints, motors and equality constraints
        should be removed from the Mujoco specification. Also, the required collision
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

    @staticmethod
    def _get_observation_specification(spec: MjSpec):
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            A list of observation types.
        """

        observation_spec = [  # ------------- JOINT POS -------------
                            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
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
    def _add_box_feet_to_spec(spec, alpha_box_feet, scaling=1.0):
        """
        Adds box feet to Mujoco spec and makes old feet non-collidable.

        Args:
            spec: Mujoco specification.

        Returns:
            Modified Mujoco spec.

        """

        # find foot and attach box
        toe_l = spec.find_body("toes_l")
        size = np.array([0.112, 0.03, 0.05]) * scaling
        pos = np.array([-0.09, 0.019, 0.0]) * scaling
        toe_l.add_geom(name="foot_box_l", type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos,
                       rgba=[0.5, 0.5, 0.5, alpha_box_feet], euler=[0.0, 0.15, 0.0])
        toe_r = spec.find_body("toes_r")
        toe_r.add_geom(name="foot_box_r", type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos,
                       rgba=[0.5, 0.5, 0.5, alpha_box_feet], euler=[0.0, -0.15, 0.0])

        # make true foot uncollidable
        foot_geoms = ["r_foot", "r_bofoot", "l_foot", "l_bofoot"]
        for g in spec.geoms:
            if g.name in foot_geoms:
                g.contype = 0
                g.conaffinity = 0

        return spec

    @staticmethod
    def _reorient_arms(spec: MjSpec):
        """
        Reorients the arm of a humanoid model given its Mujoco specification.

        Args:
            spec: MjSpec: Mujoco specification.

        Returns:
            Modified Mujoco specification.

        """

        h = spec.find_body("humerus_l")
        h.quat = [1.0, -0.1, -1.0, -0.1]
        h = spec.find_body("ulna_l")
        h.quat = [1.0, 0.6, 0.0, 0.0]
        h = spec.find_body("humerus_r")
        h.quat = [1.0, 0.1, 1.0, -0.1]
        h = spec.find_body("ulna_r")
        h.quat = [1.0, -0.6, 0.0, 0.0]

        return spec

    @info_property
    def root_height_healthy_range(self):
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        return (0.8, 1.1)

    @info_property
    def upper_body_xml_name(self):
        return "torso"

    def _modify_spec_for_mjx(self, spec):
        """
        Mjx is bad in handling many complex contacts. To speed-up simulation significantly we apply
        some changes to the Mujoco specification:
            1. Disable all contacts except the ones between feet and the floor.

        Args:
            spec: Handle to Mujoco specification.

        Returns:
            Mujoco specification.

        """

        # --- disable all contacts in geom ---
        for g in spec.geoms:
            g.contype = 0
            g.conaffinity = 0

        # --- define contacts between feet and floor --
        spec.add_pair(geomname1="floor", geomname2="foot_box_r")
        spec.add_pair(geomname1="floor", geomname2="foot_box_l")

        return spec

    @info_property
    def sites_for_mimic(self):
        return ["upper_body_mimic", "head_mimic", "pelvis_mimic",
                "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic",
                "left_hip_mimic", "left_knee_mimic", "left_foot_mimic",
                "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
                "right_hip_mimic", "right_knee_mimic", "right_foot_mimic"]
