import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class UnitreeH1v2(BaseRobotHumanoid):

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

    mjx_enabled = False

    def __init__(self, disable_hands=True, spec=None,
                 observation_spec=None, action_spec=None, **kwargs):
        """
        Constructor.

        """

        self._disable_handss = disable_hands

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
        if disable_hands:
            joints_to_remove, actuators_to_remove, equ_constraints_to_remove = self._get_spec_modifications()
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
            action_spec = [ac for ac in action_spec if ac not in actuators_to_remove]
            spec = self._delete_from_spec(spec, joints_to_remove,
                                          actuators_to_remove, equ_constraints_to_remove)

        # uses PD control by default
        if "control_type" not in kwargs.keys():
            kwargs["control_type"] = "PDControl"
            kwargs["control_params"] = dict(p_gain=[self.p_gains[act.name] for act in spec.actuators],
                                            d_gain=[self.d_gains[act.name] for act in spec.actuators],
                                            scale_action_to_jnt_limits=False)

        super().__init__(spec, action_spec, observation_spec, **kwargs)

    def _get_spec_modifications(self):
        """
        Function that specifies which joints, motors and equality constraints
        should be removed from the Mujoco specification.

        Returns:
            A tuple of lists consisting of names of joints to remove, names of actuators to remove,
             and names of equality constraints to remove.

        """

        joints_to_remove = [
            # Left Hand
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "L_thumb_distal_joint",
            "L_index_proximal_joint",
            "L_index_intermediate_joint",
            "L_middle_proximal_joint",
            "L_middle_intermediate_joint",
            "L_ring_proximal_joint",
            "L_ring_intermediate_joint",
            "L_pinky_proximal_joint",
            "L_pinky_intermediate_joint",
            # Right Hand
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_intermediate_joint",
            "R_thumb_distal_joint",
            "R_index_proximal_joint",
            "R_index_intermediate_joint",
            "R_middle_proximal_joint",
            "R_middle_intermediate_joint",
            "R_ring_proximal_joint",
            "R_ring_intermediate_joint",
            "R_pinky_proximal_joint",
            "R_pinky_intermediate_joint",
        ]

        actuators_to_remove = [
            # Left Hand
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "L_thumb_distal_joint",
            "L_index_proximal_joint",
            "L_index_intermediate_joint",
            "L_middle_proximal_joint",
            "L_middle_intermediate_joint",
            "L_ring_proximal_joint",
            "L_ring_intermediate_joint",
            "L_pinky_proximal_joint",
            "L_pinky_intermediate_joint",
            # Right Hand
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_intermediate_joint",
            "R_thumb_distal_joint",
            "R_index_proximal_joint",
            "R_index_intermediate_joint",
            "R_middle_proximal_joint",
            "R_middle_intermediate_joint",
            "R_ring_proximal_joint",
            "R_ring_intermediate_joint",
            "R_pinky_proximal_joint",
            "R_pinky_intermediate_joint",
        ]

        equ_constr_to_remove = []

        return joints_to_remove, actuators_to_remove, equ_constr_to_remove

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
            ObservationType.FreeJointPosNoXY("q_floating_base_joint", xml_name="floating_base_joint"),
            ObservationType.JointPos("q_left_hip_yaw_joint", xml_name="left_hip_yaw_joint"),
            ObservationType.JointPos("q_left_hip_pitch_joint", xml_name="left_hip_pitch_joint"),
            ObservationType.JointPos("q_left_hip_roll_joint", xml_name="left_hip_roll_joint"),
            ObservationType.JointPos("q_left_knee_joint", xml_name="left_knee_joint"),
            ObservationType.JointPos("q_left_ankle_pitch_joint", xml_name="left_ankle_pitch_joint"),
            ObservationType.JointPos("q_left_ankle_roll_joint", xml_name="left_ankle_roll_joint"),
            ObservationType.JointPos("q_right_hip_yaw_joint", xml_name="right_hip_yaw_joint"),
            ObservationType.JointPos("q_right_hip_pitch_joint", xml_name="right_hip_pitch_joint"),
            ObservationType.JointPos("q_right_hip_roll_joint", xml_name="right_hip_roll_joint"),
            ObservationType.JointPos("q_right_knee_joint", xml_name="right_knee_joint"),
            ObservationType.JointPos("q_right_ankle_pitch_joint", xml_name="right_ankle_pitch_joint"),
            ObservationType.JointPos("q_right_ankle_roll_joint", xml_name="right_ankle_roll_joint"),
            ObservationType.JointPos("q_torso_joint", xml_name="torso_joint"),
            ObservationType.JointPos("q_left_shoulder_pitch_joint", xml_name="left_shoulder_pitch_joint"),
            ObservationType.JointPos("q_left_shoulder_roll_joint", xml_name="left_shoulder_roll_joint"),
            ObservationType.JointPos("q_left_shoulder_yaw_joint", xml_name="left_shoulder_yaw_joint"),
            ObservationType.JointPos("q_left_elbow_joint", xml_name="left_elbow_joint"),
            ObservationType.JointPos("q_left_wrist_roll_joint", xml_name="left_wrist_roll_joint"),
            ObservationType.JointPos("q_left_wrist_pitch_joint", xml_name="left_wrist_pitch_joint"),
            ObservationType.JointPos("q_left_wrist_yaw_joint", xml_name="left_wrist_yaw_joint"),
            ObservationType.JointPos("q_right_shoulder_pitch_joint", xml_name="right_shoulder_pitch_joint"),
            ObservationType.JointPos("q_right_shoulder_roll_joint", xml_name="right_shoulder_roll_joint"),
            ObservationType.JointPos("q_right_shoulder_yaw_joint", xml_name="right_shoulder_yaw_joint"),
            ObservationType.JointPos("q_right_elbow_joint", xml_name="right_elbow_joint"),
            ObservationType.JointPos("q_right_wrist_roll_joint", xml_name="right_wrist_roll_joint"),
            ObservationType.JointPos("q_right_wrist_pitch_joint", xml_name="right_wrist_pitch_joint"),
            ObservationType.JointPos("q_right_wrist_yaw_joint", xml_name="right_wrist_yaw_joint"),
            ObservationType.JointPos("q_L_index_proximal_joint", xml_name="L_index_proximal_joint"),
            ObservationType.JointPos("q_L_index_intermediate_joint", xml_name="L_index_intermediate_joint"),
            ObservationType.JointPos("q_L_middle_proximal_joint", xml_name="L_middle_proximal_joint"),
            ObservationType.JointPos("q_L_middle_intermediate_joint", xml_name="L_middle_intermediate_joint"),
            ObservationType.JointPos("q_L_ring_proximal_joint", xml_name="L_ring_proximal_joint"),
            ObservationType.JointPos("q_L_ring_intermediate_joint", xml_name="L_ring_intermediate_joint"),
            ObservationType.JointPos("q_L_pinky_proximal_joint", xml_name="L_pinky_proximal_joint"),
            ObservationType.JointPos("q_L_pinky_intermediate_joint", xml_name="L_pinky_intermediate_joint"),
            ObservationType.JointPos("q_L_thumb_proximal_yaw_joint", xml_name="L_thumb_proximal_yaw_joint"),
            ObservationType.JointPos("q_L_thumb_proximal_pitch_joint", xml_name="L_thumb_proximal_pitch_joint"),
            ObservationType.JointPos("q_L_thumb_intermediate_joint", xml_name="L_thumb_intermediate_joint"),
            ObservationType.JointPos("q_L_thumb_distal_joint", xml_name="L_thumb_distal_joint"),
            ObservationType.JointPos("q_R_index_proximal_joint", xml_name="R_index_proximal_joint"),
            ObservationType.JointPos("q_R_index_intermediate_joint", xml_name="R_index_intermediate_joint"),
            ObservationType.JointPos("q_R_middle_proximal_joint", xml_name="R_middle_proximal_joint"),
            ObservationType.JointPos("q_R_middle_intermediate_joint", xml_name="R_middle_intermediate_joint"),
            ObservationType.JointPos("q_R_ring_proximal_joint", xml_name="R_ring_proximal_joint"),
            ObservationType.JointPos("q_R_ring_intermediate_joint", xml_name="R_ring_intermediate_joint"),
            ObservationType.JointPos("q_R_pinky_proximal_joint", xml_name="R_pinky_proximal_joint"),
            ObservationType.JointPos("q_R_pinky_intermediate_joint", xml_name="R_pinky_intermediate_joint"),
            ObservationType.JointPos("q_R_thumb_proximal_yaw_joint", xml_name="R_thumb_proximal_yaw_joint"),
            ObservationType.JointPos("q_R_thumb_proximal_pitch_joint", xml_name="R_thumb_proximal_pitch_joint"),
            ObservationType.JointPos("q_R_thumb_intermediate_joint", xml_name="R_thumb_intermediate_joint"),
            ObservationType.JointPos("q_R_thumb_distal_joint", xml_name="R_thumb_distal_joint"),

            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_floating_base_joint", xml_name="floating_base_joint"),
            ObservationType.JointVel("dq_left_hip_yaw_joint", xml_name="left_hip_yaw_joint"),
            ObservationType.JointVel("dq_left_hip_pitch_joint", xml_name="left_hip_pitch_joint"),
            ObservationType.JointVel("dq_left_hip_roll_joint", xml_name="left_hip_roll_joint"),
            ObservationType.JointVel("dq_left_knee_joint", xml_name="left_knee_joint"),
            ObservationType.JointVel("dq_left_ankle_pitch_joint", xml_name="left_ankle_pitch_joint"),
            ObservationType.JointVel("dq_left_ankle_roll_joint", xml_name="left_ankle_roll_joint"),
            ObservationType.JointVel("dq_right_hip_yaw_joint", xml_name="right_hip_yaw_joint"),
            ObservationType.JointVel("dq_right_hip_pitch_joint", xml_name="right_hip_pitch_joint"),
            ObservationType.JointVel("dq_right_hip_roll_joint", xml_name="right_hip_roll_joint"),
            ObservationType.JointVel("dq_right_knee_joint", xml_name="right_knee_joint"),
            ObservationType.JointVel("dq_right_ankle_pitch_joint", xml_name="right_ankle_pitch_joint"),
            ObservationType.JointVel("dq_right_ankle_roll_joint", xml_name="right_ankle_roll_joint"),
            ObservationType.JointVel("dq_torso_joint", xml_name="torso_joint"),
            ObservationType.JointVel("dq_left_shoulder_pitch_joint", xml_name="left_shoulder_pitch_joint"),
            ObservationType.JointVel("dq_left_shoulder_roll_joint", xml_name="left_shoulder_roll_joint"),
            ObservationType.JointVel("dq_left_shoulder_yaw_joint", xml_name="left_shoulder_yaw_joint"),
            ObservationType.JointVel("dq_left_elbow_joint", xml_name="left_elbow_joint"),
            ObservationType.JointVel("dq_left_wrist_roll_joint", xml_name="left_wrist_roll_joint"),
            ObservationType.JointVel("dq_left_wrist_pitch_joint", xml_name="left_wrist_pitch_joint"),
            ObservationType.JointVel("dq_left_wrist_yaw_joint", xml_name="left_wrist_yaw_joint"),
            ObservationType.JointVel("dq_right_shoulder_pitch_joint", xml_name="right_shoulder_pitch_joint"),
            ObservationType.JointVel("dq_right_shoulder_roll_joint", xml_name="right_shoulder_roll_joint"),
            ObservationType.JointVel("dq_right_shoulder_yaw_joint", xml_name="right_shoulder_yaw_joint"),
            ObservationType.JointVel("dq_right_elbow_joint", xml_name="right_elbow_joint"),
            ObservationType.JointVel("dq_right_wrist_roll_joint", xml_name="right_wrist_roll_joint"),
            ObservationType.JointVel("dq_right_wrist_pitch_joint", xml_name="right_wrist_pitch_joint"),
            ObservationType.JointVel("dq_right_wrist_yaw_joint", xml_name="right_wrist_yaw_joint"),
            ObservationType.JointVel("dq_L_index_proximal_joint", xml_name="L_index_proximal_joint"),
            ObservationType.JointVel("dq_L_index_intermediate_joint", xml_name="L_index_intermediate_joint"),
            ObservationType.JointVel("dq_L_middle_proximal_joint", xml_name="L_middle_proximal_joint"),
            ObservationType.JointVel("dq_L_middle_intermediate_joint", xml_name="L_middle_intermediate_joint"),
            ObservationType.JointVel("dq_L_ring_proximal_joint", xml_name="L_ring_proximal_joint"),
            ObservationType.JointVel("dq_L_ring_intermediate_joint", xml_name="L_ring_intermediate_joint"),
            ObservationType.JointVel("dq_L_pinky_proximal_joint", xml_name="L_pinky_proximal_joint"),
            ObservationType.JointVel("dq_L_pinky_intermediate_joint", xml_name="L_pinky_intermediate_joint"),
            ObservationType.JointVel("dq_L_thumb_proximal_yaw_joint", xml_name="L_thumb_proximal_yaw_joint"),
            ObservationType.JointVel("dq_L_thumb_proximal_pitch_joint", xml_name="L_thumb_proximal_pitch_joint"),
            ObservationType.JointVel("dq_L_thumb_intermediate_joint", xml_name="L_thumb_intermediate_joint"),
            ObservationType.JointVel("dq_L_thumb_distal_joint", xml_name="L_thumb_distal_joint"),
            ObservationType.JointVel("dq_R_index_proximal_joint", xml_name="R_index_proximal_joint"),
            ObservationType.JointVel("dq_R_index_intermediate_joint", xml_name="R_index_intermediate_joint"),
            ObservationType.JointVel("dq_R_middle_proximal_joint", xml_name="R_middle_proximal_joint"),
            ObservationType.JointVel("dq_R_middle_intermediate_joint", xml_name="R_middle_intermediate_joint"),
            ObservationType.JointVel("dq_R_ring_proximal_joint", xml_name="R_ring_proximal_joint"),
            ObservationType.JointVel("dq_R_ring_intermediate_joint", xml_name="R_ring_intermediate_joint"),
            ObservationType.JointVel("dq_R_pinky_proximal_joint", xml_name="R_pinky_proximal_joint"),
            ObservationType.JointVel("dq_R_pinky_intermediate_joint", xml_name="R_pinky_intermediate_joint"),
            ObservationType.JointVel("dq_R_thumb_proximal_yaw_joint", xml_name="R_thumb_proximal_yaw_joint"),
            ObservationType.JointVel("dq_R_thumb_proximal_pitch_joint", xml_name="R_thumb_proximal_pitch_joint"),
            ObservationType.JointVel("dq_R_thumb_intermediate_joint", xml_name="R_thumb_intermediate_joint"),
            ObservationType.JointVel("dq_R_thumb_distal_joint", xml_name="R_thumb_distal_joint"),
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
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "torso_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
            "L_index_proximal_joint",
            "L_index_intermediate_joint",
            "L_middle_proximal_joint",
            "L_middle_intermediate_joint",
            "L_ring_proximal_joint",
            "L_ring_intermediate_joint",
            "L_pinky_proximal_joint",
            "L_pinky_intermediate_joint",
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "L_thumb_distal_joint",
            "R_index_proximal_joint",
            "R_index_intermediate_joint",
            "R_middle_proximal_joint",
            "R_middle_intermediate_joint",
            "R_ring_proximal_joint",
            "R_ring_intermediate_joint",
            "R_pinky_proximal_joint",
            "R_pinky_intermediate_joint",
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_intermediate_joint",
            "R_thumb_distal_joint"
        ]

        return action_spec

    @property
    def p_gains(self):
        p_gains = {
            'left_hip_yaw_joint': 200.0,
            'left_hip_pitch_joint': 200.0,
            'left_hip_roll_joint': 200.0,
            'left_knee_joint': 300.0,
            'left_ankle_pitch_joint': 40.0,
            'left_ankle_roll_joint': 40.0,
            'right_hip_yaw_joint': 200.0,
            'right_hip_pitch_joint': 200.0,
            'right_hip_roll_joint': 200.0,
            'right_knee_joint': 300.0,
            'right_ankle_pitch_joint': 40.0,
            'right_ankle_roll_joint': 40.0,
            'torso_joint': 200.0,
            'left_shoulder_pitch_joint': 40.0,
            'left_shoulder_roll_joint': 40.0,
            'left_shoulder_yaw_joint': 18.0,
            'left_elbow_joint': 18.0,
            'left_wrist_roll_joint': 19.0,
            'left_wrist_pitch_joint': 19.0,
            'left_wrist_yaw_joint': 19.0,
            'right_shoulder_pitch_joint': 40.0,
            'right_shoulder_roll_joint': 40.0,
            'right_shoulder_yaw_joint': 18.0,
            'right_elbow_joint': 18.0,
            'right_wrist_roll_joint': 19.0,
            'right_wrist_pitch_joint': 19.0,
            'right_wrist_yaw_joint': 19.0,
            'L_index_proximal_joint': 1.0,
            'L_index_intermediate_joint': 1.0,
            'L_middle_proximal_joint': 1.0,
            'L_middle_intermediate_joint': 1.0,
            'L_ring_proximal_joint': 1.0,
            'L_ring_intermediate_joint': 1.0,
            'L_pinky_proximal_joint': 1.0,
            'L_pinky_intermediate_joint': 1.0,
            'L_thumb_proximal_yaw_joint': 1.0,
            'L_thumb_proximal_pitch_joint': 1.0,
            'L_thumb_intermediate_joint': 1.0,
            'L_thumb_distal_joint': 1.0,
            'R_index_proximal_joint': 1.0,
            'R_index_intermediate_joint': 1.0,
            'R_middle_proximal_joint': 1.0,
            'R_middle_intermediate_joint': 1.0,
            'R_ring_proximal_joint': 1.0,
            'R_ring_intermediate_joint': 1.0,
            'R_pinky_proximal_joint': 1.0,
            'R_pinky_intermediate_joint': 1.0,
            'R_thumb_proximal_yaw_joint': 1.0,
            'R_thumb_proximal_pitch_joint': 1.0,
            'R_thumb_intermediate_joint': 1.0,
            'R_thumb_distal_joint': 1.0,
        }

        return p_gains

    @property
    def d_gains(self):
        d_gains = {
            'left_hip_yaw_joint': 2.5,
            'left_hip_pitch_joint': 2.5,
            'left_hip_roll_joint': 2.5,
            'left_knee_joint': 4.0,
            'left_ankle_pitch_joint': 2.0,
            'left_ankle_roll_joint': 2.0,
            'right_hip_yaw_joint': 2.5,
            'right_hip_pitch_joint': 2.5,
            'right_hip_roll_joint': 2.5,
            'right_knee_joint': 4.0,
            'right_ankle_pitch_joint': 2.0,
            'right_ankle_roll_joint': 2.0,
            'torso_joint': 2.5,
            'left_shoulder_pitch_joint': 2.0,
            'left_shoulder_roll_joint': 2.0,
            'left_shoulder_yaw_joint': 1.8,
            'left_elbow_joint': 1.8,
            'left_wrist_roll_joint': 1.9,
            'left_wrist_pitch_joint': 1.9,
            'left_wrist_yaw_joint': 1.9,
            'right_shoulder_pitch_joint': 2.0,
            'right_shoulder_roll_joint': 2.0,
            'right_shoulder_yaw_joint': 1.8,
            'right_elbow_joint': 1.8,
            'right_wrist_roll_joint': 1.9,
            'right_wrist_pitch_joint': 1.9,
            'right_wrist_yaw_joint': 1.9,
            'L_index_proximal_joint': 0.1,
            'L_index_intermediate_joint': 0.1,
            'L_middle_proximal_joint': 0.1,
            'L_middle_intermediate_joint': 0.1,
            'L_ring_proximal_joint': 0.1,
            'L_ring_intermediate_joint': 0.1,
            'L_pinky_proximal_joint': 0.1,
            'L_pinky_intermediate_joint': 0.1,
            'L_thumb_proximal_yaw_joint': 0.1,
            'L_thumb_proximal_pitch_joint': 0.1,
            'L_thumb_intermediate_joint': 0.1,
            'L_thumb_distal_joint': 0.1,
            'R_index_proximal_joint': 0.1,
            'R_index_intermediate_joint': 0.1,
            'R_middle_proximal_joint': 0.1,
            'R_middle_intermediate_joint': 0.1,
            'R_ring_proximal_joint': 0.1,
            'R_ring_intermediate_joint': 0.1,
            'R_pinky_proximal_joint': 0.1,
            'R_pinky_intermediate_joint': 0.1,
            'R_thumb_proximal_yaw_joint': 0.1,
            'R_thumb_proximal_pitch_joint': 0.1,
            'R_thumb_intermediate_joint': 0.1,
            'R_thumb_distal_joint': 0.1,
        }

        return d_gains

    @classmethod
    def get_default_xml_file_path(cls):
        """
        Returns the default path to the xml file of the environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "unitree_h1_2" / "h1_2.xml").as_posix()

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
        return "floating_base_joint"

    @info_property
    def root_height_healthy_range(self):
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        return (0.6, 1.5)
