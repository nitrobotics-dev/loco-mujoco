from typing import Any, Dict, List, Union
from types import ModuleType

import numpy as np
import jax
import jax.numpy as jnp
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.utils.backend import assert_backend_is_supported
from loco_mujoco.core.stateful_object import StatefulObject


class DomainRandomizer(StatefulObject):
    """
    Base interface for all domain randomization classes.

    Attributes:
        registered (Dict[str, type]): Dictionary to store registered domain randomizer classes.
    """

    registered: Dict[str, type] = dict()

    def __init__(self, **randomization_config: Any):
        """
        Initialize the DomainRandomizer class.

        Args:
            **randomization_config (Any): Configuration parameters for domain randomization.
        """
        self.rand_conf: Dict[str, Any] = randomization_config

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType):
        """
        Reset the domain randomizer.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Raises:
            ValueError: If the backend module is not supported.
            NotImplementedError: If the method is not implemented in a subclass.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError

    def update(self, env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType):
        """
        Update the domain randomizer.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Raises:
            ValueError: If the backend module is not supported.
            NotImplementedError: If the method is not implemented in a subclass.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError

    def apply_on_model(self,
                       env: Any,
                       model: Union[MjModel, Model],
                       data: Union[MjData, Data],
                       carry: Any,
                       backend: ModuleType):
        """
        Apply domain randomization to the model.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Raises:
            ValueError: If the backend module is not supported.
            NotImplementedError: If the method is not implemented in a subclass.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError

    def update_observation(self, env: Any,
                           obs: Union[np.ndarray, jax.Array],
                           model: Union[MjModel, Model],
                           data: Union[MjData, Data],
                           carry: Any,
                           backend: ModuleType):
        """
        Update the observation with domain randomization effects.

        Args:
            env (Any): The environment instance.
            obs (Union[np.ndarray, jax.Array]): The observation to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Raises:
            ValueError: If the backend module is not supported.
            NotImplementedError: If the method is not implemented in a subclass.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError

    def update_action(self, env: Any,
                      action: Union[np.ndarray, jax.Array],
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      carry: Any,
                      backend: ModuleType):
        """
        Update the action with domain randomization effects.

        Args:
            env (Any): The environment instance.
            action (Union[np.ndarray, jax.Array]): The action to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Raises:
            ValueError: If the backend module is not supported.
            NotImplementedError: If the method is not implemented in a subclass.
        """
        if not assert_backend_is_supported(backend):
            raise ValueError(f"Unsupported backend module: {backend.__name__}")
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of the domain randomizer class.

        Returns:
            str: The name of the domain randomizer class.
        """
        return cls.__name__

    @classmethod
    def register(cls):
        """
        Register a domain randomizer class.

        Raises:
            ValueError: If the domain randomizer is already registered.
        """
        env_name = cls.get_name()

        if env_name in DomainRandomizer.registered:
            raise ValueError(f"DomainRandomizer '{env_name}' is already registered.")

        DomainRandomizer.registered[env_name] = cls

    @staticmethod
    def list_registered() -> List[str]:
        """
        List registered domain randomizers.

        Returns:
            List[str]: A list of registered domain randomizer class names.
        """
        return list(DomainRandomizer.registered.keys())

    @staticmethod
    def _set_attribute_in_model(model: Union[MjModel, Model],
                                attribute: str,
                                value: Any,
                                backend: ModuleType) -> Union[MjModel, Model]:
        """
        Set an attribute in the model. Works for both NumPy and JAX backends.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            attribute (str): The attribute to set.
            value (Any): The value to assign to the attribute.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Union[MjModel, Model]: The updated model.

        Raises:
            ValueError: If the backend module is not supported.
        """
        assert_backend_is_supported(backend)

        if backend == jnp:
            model = model.tree_replace({attribute: value})
        else:
            setattr(model, attribute, value)
        return model
