from typing import Any, Union

import numpy as np
import jax.numpy as jnp
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.domain_randomizer import DomainRandomizer


class NoDomainRandomization(DomainRandomizer):
    """
    A domain randomizer that performs no randomization.
    """

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: Union[np, jnp]) -> tuple:
        """
        Reset with no randomization applied.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (Union[np, jnp]): Backend used for simulation (e.g., JAX or NumPy).

        Returns:
            tuple: The unchanged data and carry.
        """
        return data, carry

    def update(self, env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: Union[np, jnp]) -> tuple:
        """
        Update with no randomization applied.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (Union[np, jnp]): Backend used for simulation (e.g., JAX or NumPy).

        Returns:
            tuple: The unchanged model, data, and carry.
        """
        return model, data, carry

    def update_observation(self, obs: Union[np.ndarray, jnp.ndarray],
                           env: Any,
                           model: Union[MjModel, Model],
                           data: Union[MjData, Data],
                           carry: Any,
                           backend: Union[np, jnp]) -> tuple:
        """
        Update the observation with no randomization applied.

        Args:
            obs (Union[np.ndarray, jnp.ndarray]): The observation to be updated.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (Union[np, jnp]): Backend used for simulation (e.g., JAX or NumPy).

        Returns:
            tuple: The unchanged observation and carry.
        """
        return obs, carry

    def update_action(self, action: Union[np.ndarray, jnp.ndarray],
                      env: Any,
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      carry: Any,
                      backend: Union[np, jnp]) -> tuple:
        """
        Update the action with no randomization applied.

        Args:
            action (Union[np.ndarray, jnp.ndarray]): The action to be updated.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (Union[np, jnp]): Backend used for simulation (e.g., JAX or NumPy).

        Returns:
            tuple: The unchanged action and carry.
        """
        return action, carry
