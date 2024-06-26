import jax.numpy as jnp
from .unitreeG1 import UnitreeG1
from loco_mujoco.environments import ValidTaskConf


class MjxUnitreeG1(UnitreeG1):
    valid_task_confs = ValidTaskConf(tasks=["walk", "run"],
                                     data_types=["real"],
                                     non_combinable=[("carry", None, "perfect")])
    mjx_enabled = True

    def __init__(self, timestep=0.002, n_substeps=5, **kwargs):
        if "model_option_conf" not in kwargs.keys():
            model_option_conf = dict(iterations=2, ls_iterations=4)
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]
        super().__init__(timestep=timestep, n_substeps=n_substeps, model_option_conf=model_option_conf, **kwargs)

    @staticmethod
    def _modify_xml_for_mjx(xml_handle):
        """
        Mjx is bad in handling many complex contacts. To speed-up simulation significantly we apply
        some changes to the XML:
            1. Replace the complex foot meshes with primitive shapes. Here, one foot mesh is replaced with
               two capsules.
            2. Disable all contacts except the ones between feet and the floor.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Mujoco XML handle.

        """

        # --- 1. disable all contacts in the collision geom class ---
        default = xml_handle.find_all("default")
        for d in default:
            if d.dclass == "collision":
                d.geom.contype = 0
                d.geom.conaffinity = 0

        # --- 2. frictionloss not yet implemented in Mjx ---
        joint_def = default[0].joint
        joint_def.frictionloss = 0.0

        # --- 3. enable collision for foot geometries ---
        # remove original foot meshes
        foot_geoms = ["right_foot_1_col", "right_foot_2_col", "right_foot_3_col", "right_foot_4_col",
                      "left_foot_1_col", "left_foot_2_col", "left_foot_3_col", "left_foot_4_col"]
        for fg in foot_geoms:
            g_handle = xml_handle.find("geom", fg)
            g_handle.contype = 1
            g_handle.conaffinity = 1

        return xml_handle

    def _get_collision_groups(self):
        return []

    def _mjx_has_fallen(self, obs, info, data):
        pelvis_cond, _, _, _, _ = self._has_fallen_compat(obs, info, data, jnp)
        return pelvis_cond
