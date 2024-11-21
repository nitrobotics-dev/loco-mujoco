from loco_mujoco.environments.humanoids.talos import Talos
from loco_mujoco.environments import ValidTaskConf


class MjxTalos(Talos):

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
        contact_geoms = ["left_foot", "right_foot", "floor"]

        # --- disable all contacts in geom except foot and floor ---
        geoms = xml_handle.find_all("geom")
        for g in geoms:
            if g.name not in contact_geoms:
                g.contype = 0
                g.conaffinity = 0

        return xml_handle
