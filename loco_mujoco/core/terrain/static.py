from typing import Any, Union

import numpy as np
import jax.numpy as jnp
from mujoco import MjData, MjModel, MjSpec
from mujoco.mjx import Data, Model

from loco_mujoco.core.terrain import Terrain


class StaticTerrain(Terrain):
    """
    Static terrain class inheriting from Terrain. This class is used for terrains that do not change over time
    (e.g., flat terrain).

    """

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: Union[np, jnp]) -> tuple:
        """
        Reset the terrain.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (Union[np, jnp]): Backend used for simulation (e.g., JAX or NumPy).

        Returns:
            tuple: Updated data and carry.
        """
        return data, carry

    def update(self, env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: Union[np, jnp]) -> tuple:
        """
        Update the terrain.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (Union[np, jnp]): Backend used for simulation (e.g., JAX or NumPy).

        Returns:
            tuple: Updated model, data, and carry.
        """
        return model, data, carry

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        """
        Modify the simulation specification.

        Args:
            spec (MjSpec): The simulation specification.

        Returns:
            MjSpec: The unmodified simulation specification.
        """
        return spec

    @property
    def is_dynamic(self) -> bool:
        """
        Check if the terrain is dynamic.

        Returns:
            bool: False, as this terrain is static.
        """
        return False
