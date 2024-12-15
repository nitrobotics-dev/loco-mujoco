from typing import Dict, Any, List, Union
from types import ModuleType

import jax.numpy as jnp
from mujoco import MjData, MjModel, MjSpec
from mujoco.mjx import Data, Model

from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.core.utils.backend import assert_backend_is_supported


class Terrain(StatefulObject):
    """
    Base interface for all terrain classes.
    """

    registered: Dict[str, type] = dict()

    # If the hfield is generated dynamically, this should be set to True to update the hfield in the viewer.
    viewer_needs_to_update_hfield: bool = False

    def __init__(self, env: Any, **terrain_config: Any):
        """
        Initialize the Terrain class.

        Args:
            env (Any): The environment instance.
            **terrain_config (Any): Configuration parameters for the terrain.
        """
        self.terrain_conf = terrain_config

        # To be specified from spec
        self.hfield_id: Union[int, None] = None

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType):
        """
        Reset the terrain.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError

    def update(self, env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType):
        """
        Update the terrain.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        """
        Modify the simulation specification.

        Args:
            spec (MjSpec): The simulation specification.

        Returns:
            MjSpec: The modified simulation specification.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError

    def get_height_matrix(self,
                          matrix_config: Dict[str, Any],
                          env: Any, model: Union[MjModel, Model],
                          data: Union[MjData, Data],
                          carry: Any,
                          backend: ModuleType) -> ModuleType:
        """
        Get the height matrix for the terrain.

        Args:
            matrix_config (Dict[str, Any]): Configuration for the height matrix.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            ModuleType: The height matrix.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError

    @property
    def is_dynamic(self) -> bool:
        """
        Check if the terrain is dynamic.

        Returns:
            bool: True if the terrain is dynamic, False otherwise.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError

    @property
    def requires_spec_modification(self) -> bool:
        """
        Check if the terrain requires modification of the simulation specification.

        Returns:
            bool: True if the terrain requires specification modification, False otherwise.
        """
        return self.__class__.modify_spec != Terrain.modify_spec

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of the terrain class.

        Returns:
            str: The name of the terrain class.
        """
        return cls.__name__

    @classmethod
    def register(cls):
        """
        Register a terrain class.

        Raises:
            ValueError: If a terrain with the same name is already registered.
        """
        env_name = cls.get_name()

        if env_name in Terrain.registered:
            raise ValueError(f"Terrain '{env_name}' is already registered.")

        Terrain.registered[env_name] = cls

    @staticmethod
    def list_registered() -> List[str]:
        """
        List registered terrain classes.

        Returns:
            List[str]: A list of registered terrain class names.
        """
        return list(Terrain.registered.keys())

    @staticmethod
    def _set_attribute_in_model(model: Union[MjModel, Model],
                                attribute: str,
                                value: Any,
                                backend: ModuleType) -> Union[MjModel, Model]:
        """
        Set an attribute in the model. This works for both NumPy and JAX backends.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            attribute (str): The attribute to set.
            value (Any): The value to assign to the attribute.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Union[MjModel, Model]: The updated model.
        """
        assert_backend_is_supported(backend)

        if backend == jnp:
            model = model.tree_replace({attribute: value})
        else:
            setattr(model, attribute, value)
        return model
