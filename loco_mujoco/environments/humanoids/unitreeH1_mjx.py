import jax
import jax.numpy as jnp
from .unitreeH1 import UnitreeH1
from loco_mujoco.environments import ValidTaskConf


class MjxUnitreeH1(UnitreeH1):

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

        # --- 1. remove old feet and add new ones ---
        # remove original foot meshes
        rf_handle = xml_handle.find("geom", "right_foot")
        lf_handle = xml_handle.find("geom", "left_foot")
        rf_handle.remove()
        lf_handle.remove()

        # add primitive foot shapes
        back_foot_attr = dict(type="capsule", quat=[1.0, 0.0, 1.0, 0.0], pos=[-0.03, 0.0, -0.05],
                              size=[0.015, 0.025], rgba=[1.0, 1.0, 1.0, 0.2], contype=1, conaffinity=1)
        front_foot_attr = dict(type="capsule", quat=[1.0, 1.0, 0.0, 0.0], pos=[0.15, 0.0, -0.054],
                               size=[0.02, 0.025], rgba=[1.0, 1.0, 1.0, 0.2], contype=1, conaffinity=1)

        r_foot_b = xml_handle.find("body", "right_ankle_link")
        r_foot_b.add("geom", name="right_foot1", **back_foot_attr)
        r_foot_b.add("geom", name="right_foot2", **front_foot_attr)

        l_foot_b = xml_handle.find("body", "left_ankle_link")
        l_foot_b.add("geom", name="left_foot1", **back_foot_attr)
        l_foot_b.add("geom", name="left_foot2", **front_foot_attr)

        # back_foot_attr_1 = dict(type="sphere", pos=[-0.03, 0.01, -0.06], size=[0.001],
        #                         rgba=[1.0, 1.0, 1.0, 0.2], contype=1, conaffinity=1)
        # back_foot_attr_2 = dict(type="sphere", pos=[-0.03, -0.01, -0.06], size=[0.001],
        #                         rgba=[1.0, 1.0, 1.0, 0.2], contype=1, conaffinity=1)
        # front_foot_attr_1 = dict(type="sphere", pos=[0.14, 0.03, -0.06], size=[0.001],
        #                          rgba=[1.0, 1.0, 1.0, 0.2], contype=1, conaffinity=1)
        # front_foot_attr_2 = dict(type="sphere", pos=[0.14, -0.03, -0.06], size=[0.001],
        #                          rgba=[1.0, 1.0, 1.0, 0.2], contype=1, conaffinity=1)
        #
        # r_foot_b = xml_handle.find("body", "right_ankle_link")
        # r_foot_b.add("geom", name="right_foot1", **back_foot_attr_1)
        # r_foot_b.add("geom", name="right_foot2", **back_foot_attr_2)
        # r_foot_b.add("geom", name="right_foot3", **front_foot_attr_1)
        # r_foot_b.add("geom", name="right_foot4", **front_foot_attr_2)
        #
        # r_foot_b = xml_handle.find("body", "left_ankle_link")
        # r_foot_b.add("geom", name="left_foot1", **back_foot_attr_1)
        # r_foot_b.add("geom", name="left_foot2", **back_foot_attr_2)
        # r_foot_b.add("geom", name="left_foot3", **front_foot_attr_1)
        # r_foot_b.add("geom", name="left_foot4", **front_foot_attr_2)

        # --- 2. disable all contacts in the collision geom class ---
        default = xml_handle.find_all("default")
        for d in default:
            if d.dclass == "collision":
                d.geom.contype = 0
                d.geom.conaffinity = 0

        return xml_handle

    def _get_collision_groups(self):
        return []

    def _mjx_has_fallen(self, obs, info, data):
        pelvis_cond, _, _, _, _ = self._has_fallen_compat(obs, info, data, jnp)
        return pelvis_cond

    # def _mjx_is_done(self, obs, absorbing, info, data, carry):
    #
    #     done = super()._mjx_is_done(obs, absorbing, info, data, carry)
    #
    #     import mujoco
    #     body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self.upper_body_xml_name)
    #
    #     # current position main body
    #     pos = data.xpos[body_id]
    #
    #     # traj body position
    #     traj_state = carry.traj_state
    #     current_pos = self.th.traj_data.get_xpos(traj_state.traj_no, traj_state.subtraj_step_no)[body_id]
    #     init_pos = self.th.traj_data.get_xpos(traj_state.traj_no, traj_state.subtraj_step_no_init)[body_id]
    #     virtual_pos = current_pos - init_pos
    #
    #     dist = jnp.linalg.norm(virtual_pos - pos)
    #
    #     current_qpos = self.th.traj_data.get_qpos(traj_state.traj_no, traj_state.subtraj_step_no)[0]
    #     init_qpos = self.th.traj_data.get_qpos(traj_state.traj_no, traj_state.subtraj_step_no_init)[0]
    #     virtual_qpos = current_qpos - init_qpos
    #
    #     dist2 = jnp.mean(jnp.square(virtual_qpos - data.qpos[0]))
    #
    #     #
    #     # jax.debug.print("dist {x}",x= dist)
    #     # jax.debug.print("dist qpos {x}",x= dist2)
    #
    #     done = jnp.logical_or(done, dist2 > 0.5)
    #
    #     return done
