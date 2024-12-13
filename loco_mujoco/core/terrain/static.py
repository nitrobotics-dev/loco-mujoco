from loco_mujoco.core.terrain import Terrain


class StaticTerrain(Terrain):

    def reset(self, env, model, data, carry, backend):
        """ Should be called in _mjx_reset_init_data. """
        return data, carry

    def update(self, env, model, data, carry, backend):
        """ should be called in simulation_pre_step."""
        return model, data, carry

    def modify_spec(self, spec):
        return spec

    def is_dynamic(self):
        return False
