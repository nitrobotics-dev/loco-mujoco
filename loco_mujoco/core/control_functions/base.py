from typing import Any, Dict, List, Union, Tuple
from types import ModuleType

import numpy as np
import jax
import jax.numpy as jnp
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model


from loco_mujoco.core.utils.backend import assert_backend_is_supported
from loco_mujoco.core.stateful_object import StatefulObject


class ControlFunction(StatefulObject):
    """
    Base class for all control functions.
    """

    registered: Dict[str, type] = dict()

    def __init__(self, env: any, **kwargs: Dict):
        """
        Initialize the control function class.

        Args:
            env (Any): The environment instance.
            **kwargs (Dict): Additional keyword arguments.
        """
        pass

    def generate_action(self, env: Any,
                        action: Union[np.ndarray, jax.Array],
                        model: Union[MjModel, Model],
                        data: Union[MjData, Data],
                        carry: Any,
                        backend: ModuleType) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """
        Call the action with control function.

        Args:
            env (Any): The environment instance.
            action (Union[np.ndarray, jax.Array]): The action to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jax.Array], Any]: The generated action and carry.

        Raises:
            ValueError: If the backend module is not supported.
            NotImplementedError: If the method is not implemented in a subclass.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of the control function class.

        Returns:
            str: The name of the control function class.
        """
        return cls.__name__

    @classmethod
    def register(cls):
        """
        Register a control function class.

        Raises:
            ValueError: If the control function is already registered.
        """
        cls_name = cls.get_name()

        if cls_name in ControlFunction.registered:
            raise ValueError(f"ControlFunction '{cls_name}' is already registered.")

        ControlFunction.registered[cls_name] = cls

    @staticmethod
    def list_registered() -> List[str]:
        """
        List registered control functions.

        Returns:
            List[str]: A list of registered control function class names.
        """
        return list(ControlFunction.registered.keys())
