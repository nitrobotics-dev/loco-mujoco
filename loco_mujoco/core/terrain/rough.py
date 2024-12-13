from typing import Union
import jax
import jax.numpy as jnp
from flax import struct
import mujoco
from mujoco import MjSpec
from pathlib import Path
import numpy as np
import scipy as np_scipy
import jax.scipy as jnp_scipy

from loco_mujoco.core.terrain import DynamicTerrain
from loco_mujoco.core.utils import mj_jntname2qposid


@struct.dataclass
class RoughTerrainState:
    height_field_raw: Union[np.ndarray, jax.Array]


class RoughTerrain(DynamicTerrain):

    # this is a dynamic terrain, so the viewer needs to update the hfield
    viewer_needs_to_update_hfield = False

    def __init__(self, env, inner_platform_size_in_meters=1.0, random_min_height=-0.05,
                 random_max_height=0.05, random_step=0.005, random_downsampled_scale=0.4, **kwargs):
        super().__init__(env, **kwargs)

        self.inner_platform_size_in_meters = inner_platform_size_in_meters
        self.random_min_height = random_min_height
        self.random_max_height = random_max_height
        self.random_step = random_step
        self.random_downsampled_scale = random_downsampled_scale

        self.hfield_size = (4, 4, 30.0, 0.125)
        self.hfield_length = 80
        self.hfield_half_length_in_meters = self.hfield_size[0]
        self.max_possible_height = self.hfield_size[2]

        self.one_meter_length = int(self.hfield_length / (self.hfield_half_length_in_meters * 2))
        self.hfield_half_length = self.hfield_length // 2
        self.mujoco_height_scaling = self.max_possible_height

        self.heights_range = jnp.arange(self.random_min_height, self.random_max_height + self.random_step,
                                        self.random_step)

        self.x = jnp.linspace(0, self.hfield_length, int(self.hfield_length * self.random_downsampled_scale))
        self.y = jnp.linspace(0, self.hfield_length, int(self.hfield_length * self.random_downsampled_scale))
        x_upsampled = jnp.linspace(0, self.hfield_length, self.hfield_length)
        y_upsampled = jnp.linspace(0, self.hfield_length, self.hfield_length)
        x_upsampled_grid, y_upsampled_grid = jnp.meshgrid(x_upsampled, y_upsampled, indexing='ij')
        self.points = jnp.stack([x_upsampled_grid.ravel(), y_upsampled_grid.ravel()], axis=1)
        platform_size = int(self.inner_platform_size_in_meters * self.one_meter_length)
        self.x1 = self.hfield_half_length - (platform_size // 2)
        self.y1 = self.hfield_half_length - (platform_size // 2)
        self.x2 = self.hfield_half_length + (platform_size // 2)
        self.y2 = self.hfield_half_length + (platform_size // 2)

        # get id of the free joint in xml
        root_free_joint_xml_name = env.root_free_joint_xml_name
        self._free_jnt_qpos_id = np.array(mj_jntname2qposid(root_free_joint_xml_name, env._model))

    def init_state(self, env, key, model, data, backend):
        return RoughTerrainState(backend.zeros((self.hfield_length, self.hfield_length)))

    def modify_spec(self, spec: MjSpec):

        # Add hfield to the spec
        file_name = Path(__file__).resolve().parent.parent.parent / "models" / "common" / "default_hfield_80.png"
        spec.add_hfield(name='rough_terrain', size=self.hfield_size, file=str(file_name))
        for i, field in enumerate(spec.hfields):
            if field.name == 'rough_terrain':
                self.hfield_id = i
                break

        # remove floor geom
        for g in spec.geoms:
            if g.name == 'floor':
                g.delete()
                break

        # add new hfield floor
        wb = spec.worldbody
        wb.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname='rough_terrain', group=2,
                    pos=(0, 0, 0), material="MatPlane", rgba=(0.8, 0.9, 0.8, 1))

        return spec

    def reset(self, env, model, data, carry, backend):
        """ Should be called in _mjx_reset_init_data. """

        terrain_state = carry.terrain_state

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            height_field_raw = self.isaac_hf_to_mujoco_hf(self._jnp_random_uniform_terrain(_k), backend)
            carry = carry.replace(key=key)
        else:
            height_field_raw = self.isaac_hf_to_mujoco_hf(self._np_random_uniform_terrain(), backend)

        terrain_state = terrain_state.replace(height_field_raw=height_field_raw)
        carry = carry.replace(terrain_state=terrain_state)

        return data, carry

    def update(self, env, model, data, carry, backend):
        """ should be called in simulation_pre_step."""

        # update hfield data
        terrain_state = carry.terrain_state
        model = self._set_attribute_in_model(model, "hfield_data", terrain_state.height_field_raw, backend)

        # reset the robot if it is on the edge
        data = self._reset_on_edge(data, backend)

        return model, data, carry

    def isaac_hf_to_mujoco_hf(self, isaac_hf, backend):
        hf = isaac_hf + backend.abs(backend.min(isaac_hf))
        hf /= self.mujoco_height_scaling
        return hf.reshape(-1)

    def _np_random_uniform_terrain(self):
        add_height_field_downsampled = np.random.choice(self.heights_range, size=(int(self.hfield_length * self.random_downsampled_scale), int(self.hfield_length * self.random_downsampled_scale)))
        interpolator = np_scipy.interpolate.RegularGridInterpolator((self.x, self.y), add_height_field_downsampled, method='linear')
        add_height_field = interpolator(self.points).reshape((self.hfield_length, self.hfield_length))
        add_height_field[self.x1:self.x2, self.y1:self.y2] = 0
        height_field_raw = add_height_field
        return height_field_raw

    def _jnp_random_uniform_terrain(self, key):
        add_height_field_downsampled = jax.random.choice(key, self.heights_range, shape=(int(self.hfield_length * self.random_downsampled_scale), int(self.hfield_length * self.random_downsampled_scale)))
        interpolator = jnp_scipy.interpolate.RegularGridInterpolator((self.x, self.y), add_height_field_downsampled, method='linear')
        add_height_field = interpolator(self.points).reshape((self.hfield_length, self.hfield_length))
        add_height_field = add_height_field.at[self.x1:self.x2, self.y1:self.y2].set(0)
        height_field_raw = add_height_field
        return height_field_raw

    def _reset_on_edge(self, data, backend):
        # if the com of the robot is close to boundary, reset the robot to the center
        min_edge = self.hfield_half_length_in_meters - 0.5
        max_edge = self.hfield_half_length_in_meters
        com_pos = data.qpos[self._free_jnt_qpos_id][:2]
        reached_edge = jnp.array(((min_edge < jnp.abs(com_pos[0])) & (jnp.abs(com_pos[0]) < max_edge)) | (
                    (min_edge < jnp.abs(com_pos[1])) & (jnp.abs(com_pos[1]) < max_edge)))
        if backend == jnp:
            init_data = data.replace(qpos=data.qpos.at[self._free_jnt_qpos_id].set(0.0))
            data = jax.lax.cond(reached_edge, lambda _: init_data, lambda _: data, None)
        else:
            if reached_edge:
                data.qpos[self._free_jnt_qpos_id] = 0.0

        return data
