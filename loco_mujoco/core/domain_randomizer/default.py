from typing import Any, Union, Tuple
from types import ModuleType

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.domain_randomizer import DomainRandomizer
from loco_mujoco.core.utils.backend import assert_backend_is_supported


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
                   backend: ModuleType) -> DefaultRandomizerState:
        """
        Initialize the randomizer state.

        Args:
            env (Any): The environment instance.
            key (Any): Random seed key.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            DefaultRandomizerState: The initialized randomizer state.
        """
        assert_backend_is_supported(backend)
        return DefaultRandomizerState(geom_friction=backend.array([0.0, 0.0, 0.0]))

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the randomizer, applying domain randomization.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.
        """
        assert_backend_is_supported(backend)
        domain_randomizer_state = carry.domain_randomizer_state

        # update different randomization parameters
        geom_friction, carry = self._sample_geom_friction(model, carry, backend)

        carry = carry.replace(domain_randomizer_state=domain_randomizer_state.replace(geom_friction=geom_friction))

        return data, carry

    def update(self, env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        """
        Update the randomizer by applying the state changes to the model.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjModel, Model], Union[MjData, Data], Any]: The updated simulation model, data, and carry.
        """
        assert_backend_is_supported(backend)
        domrand_state = carry.domain_randomizer_state
        model = self._set_attribute_in_model(model, "geom_friction", domrand_state.geom_friction, backend)

        return model, data, carry

    def update_observation(self, env: Any,
                           obs: Union[np.ndarray, jnp.ndarray],
                           model: Union[MjModel, Model],
                           data: Union[MjData, Data],
                           carry: Any,
                           backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Update the observation with randomization effects.

        Args:
            env (Any): The environment instance.
            obs (Union[np.ndarray, jnp.ndarray]): The observation to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The updated observation and carry.
        """
        assert_backend_is_supported(backend)
        return obs, carry

    def update_action(self,
                      env: Any,
                      action: Union[np.ndarray, jnp.ndarray],
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      carry: Any,
                      backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Update the action with randomization effects.

        Args:
            env (Any): The environment instance.
            action (Union[np.ndarray, jnp.ndarray]): The action to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The updated action and carry.
        """
        assert_backend_is_supported(backend)
        return action, carry

    def _sample_geom_friction(self, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the geometry friction parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized geometry friction parameters and carry.
        """
        assert_backend_is_supported(backend)

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
        geom_friction = backend.array([
            sampled_friction_tangential,
            sampled_friction_torsional,
            sampled_friction_rolling,
        ]).T

        return geom_friction, carry

    def _sample_geom_damping_and_stiffness(self, model: Union[MjModel, Model],
                                           carry: Any,
                                           backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the geometry damping and stiffness parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray], Any]: The randomized geometry damping
            and stiffness parameters and carry.
        """
        assert_backend_is_supported(backend)

        damping_min, damping_max = self.rand_conf["geom_damping_range"]
        n_geoms = model.ngeom
        stiffness_min, stiffness_max = self.rand_conf["geom_stiffness_range"]

        if backend == jnp:
            key = carry.key
            key, _k_damp, _k_stiff = jax.random.split(key, 3)
            interpolation_damping = jax.random.uniform(_k_damp, shape=(len(n_geoms),))
            interpolation_stiff = jax.random.uniform(_k_stiff, shape=(len(n_geoms),))
            carry = carry.replace(key=key)
        else:
            interpolation_damping = np.random.uniform(size=(len(n_geoms),))
            interpolation_stiff = np.random.uniform(size=(len(n_geoms),))

        sampled_damping = (
            damping_min + (damping_max - damping_min) * interpolation_damping
            if self.rand_conf["randomize_geom_damping"]
            else model.geom_solref[:, 1]
        )
        sampled_stiffness = (
            stiffness_min + (stiffness_max - stiffness_min) * interpolation_stiff
            if self.rand_conf["randomize_geom_stiffness"]
            else model.geom_solref[:, 0]
        )

        return sampled_damping, sampled_stiffness, carry
