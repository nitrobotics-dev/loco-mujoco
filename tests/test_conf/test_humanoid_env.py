from pathlib import Path
import mujoco
from mujoco import MjSpec

from loco_mujoco.environments import LocoEnv
from loco_mujoco.core.observations import ObservationType
from loco_mujoco.core.utils import info_property



class TestHumamoidEnv(LocoEnv):


    def __init__(self, observation_spec=None, actuation_spec=None, spec=None, **kwargs):

        # load the model specification
        if spec is None:
            spec = self.get_default_xml_file_path()

        spec = mujoco.MjSpec.from_file(spec)

        # get the observation and action specification
        if observation_spec is None:
            # get default
            observation_spec = self._get_observation_specification(spec)
        else:
            # parse
            observation_spec = self.parse_observation_spec(observation_spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        super(TestHumamoidEnv, self).__init__(spec, actuation_spec, observation_spec, **kwargs)

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
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
            ObservationType.JointPos("q_abdomen_z", xml_name="abdomen_z"),

            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            ObservationType.JointVel("dq_abdomen_z", xml_name="abdomen_z"),

            # ------------- BODIES -------------
            ObservationType.BodyPos("left_thigh_obs_pos", xml_name="left_thigh"),
            ObservationType.BodyVel("left_thigh_obs_vel", xml_name="left_thigh"),

        ]

        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec):
        action_spec = ["abdomen_y", "right_knee"]
        return action_spec

    @classmethod
    def get_default_xml_file_path(cls):
        return (Path(__file__).resolve().parent / "humanoid_test.xml").as_posix()

    @info_property
    def sites_for_mimic(self):
        return ["torso_site", "pelvis_site", "right_thigh_site", "right_foot_site", "left_thigh_site",
                "left_foot_site"]

    @info_property
    def root_body_name(self):
        return "torso"

    @info_property
    def goal_visualization_arrow_offset(self):
        return [0, 0, 0.6]

    @info_property
    def grf_size(self):
        return 6

    @info_property
    def upper_body_xml_name(self):
        return "torso_link"

    @info_property
    def root_free_joint_xml_name(self):
        return "root"

    @info_property
    def root_height_healthy_range(self):
        return (0.6, 1.5)
