import os.path
import warnings
from pathlib import Path
from copy import deepcopy
import numpy as np

# from mushroom_rl.utils.running_stats import *

import loco_mujoco
from loco_mujoco.environments import LocoEnv


class BaseRobotHumanoid(LocoEnv):
    """
    Base Class for the Mujoco environment of Atlas, UnitreeH1/G1 and Talos.

    """

    def get_mask(self, obs_to_hide):
        """
        This function returns a boolean mask to hide observations from a fully observable state.

        Args:
            obs_to_hide (tuple): A tuple of strings with names of objects to hide.
            Hidable objects are "positions", "velocities", "foot_forces", and "env_type".

        Returns:
            Mask in form of a np.array of booleans. True means that that the obs should be
            included, and False means that it should be discarded.

        """

        if type(obs_to_hide) == str:
            obs_to_hide = (obs_to_hide,)
        assert all(x in self._hidable_obs for x in obs_to_hide), "Some of the observations you want to hide are not" \
                                                                 "supported. Valid observations to hide are %s." \
                                                                 % (self._hidable_obs,)

        pos_dim, vel_dim = self._len_qpos_qvel()
        force_dim = self.grf_size

        mask = []
        if "positions" not in obs_to_hide:
            mask += [np.ones(pos_dim-2, dtype=bool)]
        else:
            mask += [np.zeros(pos_dim-2, dtype=bool)]

        if "velocities" not in obs_to_hide:
            mask += [np.ones(vel_dim, dtype=bool)]
        else:
            mask += [np.zeros(vel_dim, dtype=bool)]

        if self._use_foot_forces:
            if "foot_forces" not in obs_to_hide:
                mask += [np.ones(force_dim, dtype=bool)]
            else:
                mask += [np.zeros(force_dim, dtype=bool)]
        else:
            assert "foot_forces" not in obs_to_hide, "Creating a mask to hide foot forces without activating " \
                                                     "the latter is not allowed."

        if self._hold_weight:
            if "weight" not in obs_to_hide:
                mask += [np.ones(1, dtype=bool)]
            else:
                mask += [np.zeros(1, dtype=bool)]
        else:
            assert "weight" not in obs_to_hide, "Creating a mask to hide the carried weight without activating " \
                                                "the latter is not allowed."

        return np.concatenate(mask).ravel()

    def _get_observation_space(self):
        """
        Returns a tuple of the lows and highs (np.array) of the observation space.

        """

        low, high = super(BaseRobotHumanoid, self)._get_observation_space()
        if self._hold_weight:
            low = np.concatenate([low, [self._valid_weights[0]]])
            high = np.concatenate([high, [self._valid_weights[-1]]])

        return low, high

    def _create_observation(self, obs, carry):
        """
        Creates a full vector of observations.

        Args:
            obs (np.array): Observation vector to be modified or extended;
            return_err_msg (bool): If True, an error message with violations is returned.

        Returns:
            New observation vector (np.array).

        """

        obs = super(BaseRobotHumanoid, self)._create_observation(obs, carry)
        if self._hold_weight:
            weight_mass = deepcopy(self._model.body("weight").mass)
            obs = np.concatenate([obs, weight_mass])

        return obs

    def _get_box_color(self, ind):
        """
        Calculates the rgba color based on the index of the environment.

        Args:
            ind (int): Current index of the environment.

        Returns:
            rgba np.array.

        """

        red_rgba = np.array([1.0, 0.0, 0.0, 1.0])
        blue_rgba = np.array([0.2, 0.0, 1.0, 1.0])
        interpolation_var = ind / (len(self._valid_weights) - 1)
        color = blue_rgba + ((red_rgba - blue_rgba) * interpolation_var)

        return color
