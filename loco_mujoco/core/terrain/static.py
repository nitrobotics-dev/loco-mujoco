from typing import Any, Union
from types import ModuleType

from mujoco import MjData, MjModel, MjSpec
from mujoco.mjx import Data, Model

from loco_mujoco.core.terrain import Terrain
from loco_mujoco.core.utils.backend import assert_backend_is_supported


class StaticTerrain(Terrain):
    """
    Static terrain class inheriting from Terrain. This class is used for terrains that do not change over time
    (e.g., flat terrain).

    """

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType) -> tuple:
        """
        Reset the terrain.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            tuple: Updated data and carry.
        """
        assert_backend_is_supported(backend)
        return data, carry

    def update(self, env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType) -> tuple:
        """
        Update the terrain.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            tuple: Updated model, data, and carry.
        """
        assert_backend_is_supported(backend)
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
