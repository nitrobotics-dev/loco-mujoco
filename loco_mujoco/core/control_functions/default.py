from typing import Any, Union, Tuple
from types import ModuleType

import numpy as np
import jax
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.control_functions import ControlFunction


class DefaultControl(ControlFunction):

    """
    Uses the default actuator from the environment.
    """

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
            Tuple[Union[np.ndarray, jax.Array], Any]: The action and carry.

        """
        return action, carry
