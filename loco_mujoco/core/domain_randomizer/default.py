from typing import Any, Union

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.domain_randomizer import DomainRandomizer


@struct.dataclass
class DefaultRandomizerState:
    """
    Represents the state of the default randomizer.

    Attributes:
        geom_friction (Union[np.ndarray, jax.Array]): Friction parameters for geometry.
    """
    geom_friction: Union[np.ndarray, jax.Array]


class DefaultRandomizer(DomainRandomizer):
    """
    A domain randomizer that modifies simulation parameters like geometry friction.
    """

    def init_state(self, env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: Union[np, jnp]) -> DefaultRandomizerState:
        """
        Initialize the randomizer state.

        Args:
            env (Any): The environment instance.
            key (Any): Random seed key.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (Union[np, jnp]): Backend used for simulation (e.g., JAX or NumPy).

        Returns:
            DefaultRandomizerState: The initialized randomizer state.
        """
        return DefaultRandomizerState(geom_friction=backend.array([0.0, 0.0, 0.0]))

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: Union[np, jnp]) -> tuple:
        """
        Reset the randomizer, applying domain randomization.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (Union[np, jnp]): Backend used for simulation (e.g., JAX or NumPy).

        Returns:
            tuple: Updated data and carry.
        """
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

        sampled_friction_tangential = (
            fric_tan_min + (fric_tan_max - fric_tan_min) * interpolation
            if self.rand_conf["randomize_geom_friction_tangential"]
            else model.geom_friction[:, 0]
        )
        sampled_friction_torsional = (
            fric_tor_min + (fric_tor_max - fric_tor_min) * interpolation
            if self.rand_conf["randomize_geom_friction_torsional"]
            else model.geom_friction[:, 1]
        )
        sampled_friction_rolling = (
            fric_roll_min + (fric_roll_max - fric_roll_min) * interpolation
            if self.rand_conf["randomize_geom_friction_rolling"]
            else model.geom_friction[:, 2]
        )
        geom_friction = jnp.array([
            sampled_friction_tangential,
            sampled_friction_torsional,
            sampled_friction_rolling,
        ]).T

        carry = carry.replace(domain_randomizer_state=domain_randomizer_state.replace(geom_friction=geom_friction))

        return data, carry

    def update(self, env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: Union[np, jnp]) -> tuple:
        """
        Update the randomizer by applying the state changes to the model.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (Union[np, jnp]): Backend used for simulation (e.g., JAX or NumPy).

        Returns:
            tuple: Updated model, data, and carry.
        """
        domrand_state = carry.domain_randomizer_state
        model = self._set_attribute_in_model(model, "geom_friction", domrand_state.geom_friction, backend)

        return model, data, carry

    def update_observation(self, env: Any,
                           obs: Union[np.ndarray, jnp.ndarray],
                           model: Union[MjModel, Model],
                           data: Union[MjData, Data],
                           carry: Any,
                           backend: Union[np, jnp]) -> tuple:
        """
        Update the observation with randomization effects.

        Args:
            env (Any): The environment instance.
            obs (Union[np.ndarray, jnp.ndarray]): The observation to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (Union[np, jnp]): Backend used for simulation (e.g., JAX or NumPy).

        Returns:
            tuple: The updated observation and carry.
        """
        return obs, carry

    def update_action(self,
                      env: Any,
                      action: Union[np.ndarray, jnp.ndarray],
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      carry: Any,
                      backend: Union[np, jnp]) -> tuple:
        """
        Update the action with randomization effects.

        Args:
            env (Any): The environment instance.
            action (Union[np.ndarray, jnp.ndarray]): The action to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (Union[np, jnp]): Backend used for simulation (e.g., JAX or NumPy).

        Returns:
            tuple: The updated action and carry.
        """
        return action, carry
