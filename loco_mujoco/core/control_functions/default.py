from typing import Any, Union, Tuple, Dict
from types import ModuleType

import numpy as np
import jax
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.control_functions import ControlFunction


class DefaultControl(ControlFunction):

    """
    Uses the default actuator from the environment. This controller internally normalizes the action space to [-1, 1]
    for the agent but uses the original action space for the environment.
    """

    def __init__(self, env: any, **kwargs: Dict):
        """
        Initialize the control function class.

        Args:
            env (Any): The environment instance.
            **kwargs (Dict): Additional keyword arguments.
        """
        # get the limits of the action space
        self._actuator_low, self._actuator_high = self._get_actuator_limits(env._action_indices, env._model)

        # calculate mean and delta
        self.norm_act_mean = (self._actuator_high + self._actuator_low) / 2.0
        self.norm_act_delta = (self._actuator_high - self._actuator_low) / 2.0

        # set the action space limits for the agent to -1 and 1
        low = -np.ones_like(self.norm_act_mean)
        high = np.ones_like(self.norm_act_mean)

        super(DefaultControl, self).__init__(env, low, high, **kwargs)

    def generate_action(self, env: Any,
                        action: Union[np.ndarray, jax.Array],
                        model: Union[MjModel, Model],
                        data: Union[MjData, Data],
                        carry: Any,
                        backend: ModuleType) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """
        Calculates the action. This function scales the action from [-1, 1] to the original action space.

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
        unnormalized_action = self._unnormalize_action(action)
        return unnormalized_action, carry

    @staticmethod
    def get_action_indices(model, data, actuation_spec):
        """
        Returns the action indices given the MuJoCo model, data, and actuation_spec.

        Args:
            model: MuJoCo model.
            data: MuJoCo data structure.
             actuation_spec (list): A list specifying the names of the joints
                which should be controllable by the agent. Can be left empty
                when all actuators should be used;

        Returns:
            A list of actuator indices.

        """
        if len(actuation_spec) == 0:
            action_indices = [i for i in range(0, len(data.actuator_force))]
        else:
            action_indices = []
            for name in actuation_spec:
                action_indices.append(model.actuator(name).id)
        return action_indices

    def _unnormalize_action(self, action: Union[np.ndarray, jax.Array]) -> Union[np.ndarray, jax.Array]:
        """
        Rescale the action from [-1, 1] to the desired action space.

        Args:
            action (Union[np.ndarray, jax.Array]): The action to be unnormalized.

        Returns:
            Union[np.ndarray, jax.Array]: The unnormalized action

        """
        unnormalized_action = ((action * self.norm_act_delta) + self.norm_act_mean)
        return unnormalized_action
