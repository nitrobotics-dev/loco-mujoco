from loco_mujoco.core.domain_randomizer import DomainRandomizer


class NoDomainRandomization(DomainRandomizer):

    def reset(self, env, model, data, carry, backend):
        """ Should be called in _mjx_reset_init_data. """
        return data, carry

    def update(self, env, model, data, carry, backend):
        """ Should be called in simulation_prestep. """
        return model, data, carry

    def update_observation(self, env, obs, model, data, carry, backend):
        """ Should be called in step_finalize. """
        return obs, carry

    def update_action(self, env, action, model, data, carry, backend):
        """ should be called in simulation_pre_step."""
        return action, carry
