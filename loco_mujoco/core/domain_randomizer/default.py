from typing import Union

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from mujoco import mjx

from loco_mujoco.core.domain_randomizer import DomainRandomizer


@struct.dataclass
class DefaultRandomizerState:
    geom_friction: Union[np.ndarray, jax.Array]


class DefaultRandomizer(DomainRandomizer):

    def init_state(self, env, key, model, data, backend):
        return DefaultRandomizerState(geom_friction=backend.array([0.0, 0.0, 0.0]))

    def reset(self, env, model, data, carry, backend):
        """ Should be called in _mjx_reset_init_data. """

        domain_randomizer_state = carry.domain_randomizer_state

        fric_tan_min, fric_tan_max = self.rand_conf["geom_friction_tangential_range"]
        fric_tor_min, fric_tor_max = self.rand_conf["geom_friction_torsional_range"]
        fric_roll_min, fric_roll_max = self.rand_conf["geom_friction_rolling_range"]

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(len(model.geom_friction),))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=(len(model.geom_friction),))

        if self.rand_conf["randomize_geom_friction_tangential"]:
            sampled_friction_tangential = fric_tan_min + (fric_tan_max - fric_tan_min) * interpolation
        else:
            sampled_friction_tangential = model.geom_friction[:, 0]
        if self.rand_conf["randomize_geom_friction_torsional"]:
            sampled_friction_torsional = fric_tor_min + (fric_tor_max - fric_tor_min) * interpolation
        else:
            sampled_friction_torsional = model.geom_friction[:, 1]
        if self.rand_conf["randomize_geom_friction_rolling"]:
            sampled_friction_rolling = fric_roll_min + (fric_roll_max - fric_roll_min) * interpolation
        else:
            sampled_friction_rolling = model.geom_friction[:, 2]
        geom_friction = jnp.array([sampled_friction_tangential,
                                  sampled_friction_torsional, sampled_friction_rolling]).T

        # add new geom_friction to carry
        carry = carry.replace(domain_randomizer_state=domain_randomizer_state.replace(geom_friction=geom_friction))

        return data, carry

    def update(self, env, model, data, carry, backend):
        """ Should be called in simulation_prestep. """

        domrand_state = carry.domain_randomizer_state
        model = self._set_attribute_in_model(model, "geom_friction", domrand_state.geom_friction, backend)

        return model, data, carry

    def update_observation(self, env, obs, model, data, carry, backend):
        """ Should be called in step_finalize. """
        return obs, carry

    def update_action(self, env, action, model, data, carry, backend):
        """ should be called in simulation_pre_step."""
        return action, carry

