import atexit
from copy import deepcopy
import types
from enum import Enum
from typing import List, Optional, Any, Dict
from dataclasses import dataclass
from functools import partial
import mujoco
import numpy as np
from dm_control import mjcf

from mujoco import mjx
from flax import struct
import jax
import jax.numpy as jnp

from loco_mujoco.core.utils import (Box, MDPInfo, MujocoViewer, ObservationType,
                                    ObservationIndexContainer, ObservationContainer)


@struct.dataclass
class AdditionalCarry:
    key: jax.Array
    cur_step_in_episode: int


class Mujoco:
    """
        This is the base class for all Mujoco environments, CPU and MjX.

    """

    registered_envs = dict()

    def __init__(self, xml_file, actuation_spec, observation_spec, gamma, horizon, goal=None,
                 n_environments=1, timestep=None, n_substeps=1, n_intermediate_steps=1,
                 model_option_conf=None, **viewer_params):

        # load the model and data
        self._model = self.load_model(xml_file)
        self._modify_model(self._model, model_option_conf)
        self._data = mujoco.MjData(self._model)

        # set the timestep if provided, else read it from model
        if timestep is not None:
            self._model.opt.timestep = timestep
            self._timestep = timestep
        else:
            self._timestep = self._model.opt.timestep

        # set some attributes
        self._n_intermediate_steps = n_intermediate_steps
        self._n_substeps = n_substeps
        self._viewer_params = viewer_params
        self._viewer = None
        self._obs = None
        self._info = None
        self._additional_carry = None
        self._cur_step_in_episode = 0

        # set goal
        self._goal = goal

        # read the observation space, create a dictionary of observations and goals containing information
        # about each observation's type, indices, min and max values, etc. Additionally, create two dataclasses
        # containing the indices in the datastructure for each observation type (data_indices) and the indices for
        # each observation type in the observation array (obs_indices).
        self.obs_container, self._data_indices, self._obs_indices = (
            self._setup_observations(observation_spec, goal, self._model, self._data))

        # define observation space bounding box
        observation_space = Box(*self._get_obs_limits())

        # read the actuation spec and build the mapping between actions and ids
        self._action_indices = self.get_action_indices(self._model, self._data, actuation_spec)

        # define action space bounding box
        action_space = Box(*self._get_action_limits(self._action_indices, self._model))

        # finally, create the MDP information
        self._mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, n_environments, self.dt)

        # set the warning callback to stop the simulation when a mujoco warning occurs
        mujoco.set_mju_user_warning(self.user_warning_raise_exception)

        # check whether the function compute_action was overridden or not. If yes, we want to compute
        # the action at simulation frequency, if not we do it at control frequency. (not supported for Mjx)
        if type(self)._compute_action == Mujoco._compute_action:
            self._recompute_action_per_step = False
        else:
            self._recompute_action_per_step = True

        atexit.register(self.stop)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, key):
        key, subkey = jax.random.split(key)
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        # todo: replace all cur_step_in_episode to use additional info!
        self._additional_carry = self._init_additional_carry(key, self._data)
        self._data = self._reset_init_data(self._data, self._additional_carry)
        self._obs = self._create_observation(self._data, self._additional_carry)
        self._info = self._reset_info_dictionary(self._obs, self._data, subkey)
        return self._obs

    def step(self, action):
        cur_obs = self._obs.copy()
        cur_info = self._info.copy()
        carry = self._additional_carry

        # preprocess action (does nothing by default)
        action = self._preprocess_action(action, self._data, carry)

        # modify obs and data, before stepping in the env (does nothing by default)
        cur_obs, self._data, cur_info, carry = self._step_init(cur_obs, self._data, cur_info, carry)

        ctrl_action = None

        for i in range(self._n_intermediate_steps):

            if self._recompute_action_per_step or ctrl_action is None:
                ctrl_action = self._compute_action(cur_obs, action)
                self._data.ctrl[self._action_indices] = ctrl_action

            # modify data during simulation, before main step (does nothing by default)
            self._data, carry = self._simulation_pre_step(self._data, carry)

            # main mujoco step, runs the sim for n_substeps
            mujoco.mj_step(self._model, self._data, self._n_substeps)

            # modify data during simulation, after main step (does nothing by default)
            self._data, carry = self._simulation_post_step(self._data, carry)

            # recompute thef action at each intermediate step (not executed by default)
            if self._recompute_action_per_step:
                cur_obs = self._create_observation(self._data, carry)

        # create the final observation
        if not self._recompute_action_per_step:
            cur_obs = self._create_observation(self._data, carry)

        # modify obs and data, before stepping in the env (does nothing by default)
        cur_obs, self._data, cur_info, carry = self._step_finalize(cur_obs, self._data, cur_info, carry)

        # update info (does nothing by default)
        cur_info = self._update_info_dictionary(cur_info, cur_obs, self._data, carry)

        # check if the current state is an absorbing state
        absorbing = self._is_absorbing(cur_obs, cur_info, self._data, carry)

        # calculate the reward
        reward = self._reward(self._obs, action, cur_obs, absorbing, cur_info, self._model, self._data, carry)

        # calculate flag indicating whether this is the last obs before resetting
        done = self._is_done(cur_obs, absorbing, cur_info, self._data, carry)

        self._obs = cur_obs
        self._cur_step_in_episode += 1
        self._additional_carry = carry

        return np.asarray(cur_obs), reward, absorbing, done, cur_info

    def render(self, record=False):
        if self._viewer is None:
            self._viewer = MujocoViewer(self._model, self.dt, record=record, **self._viewer_params)

        return self._viewer.render(self._data, record)

    def stop(self):
        if self._viewer is not None:
            self._viewer.stop()
            del self._viewer
            self._viewer = None

    @partial(jax.jit, static_argnums=(0,))
    def sample_action_space(self, key):
        action_dim = self.info.action_space.shape[0]
        action = jax.random.uniform(key, minval=self.info.action_space.low, maxval=self.info.action_space.high,
                                    shape=(action_dim,))
        return action

    def get_all_observation_keys(self):
        from loco_mujoco.core.utils import Goal
        return [k for k, elem in self.obs_container.items() if not issubclass(elem.__class__, Goal)]

    def _is_absorbing(self, obs, info, data, carry):
        """
        Check whether the given state is an absorbing state or not.

        Args:
            obs (np.array): the state of the system.

        Returns:
            A boolean flag indicating whether this state is absorbing or not.

        """
        return False

    def _is_done(self, obs, absorbing, info, data, carry):
        done = absorbing or (self._cur_step_in_episode >= self.info.horizon)
        return done

    def _reset_init_data(self, data, carry):
        return data

    def _step_init(self, obs, data, info, carry):
        return obs, data, info, carry

    def _step_finalize(self, obs, data, info, carry):
        """
        Allows information to be accessed at the end of a step.
        """
        return obs, data, info, carry

    def _reset_info_dictionary(self, obs, data, key):
        return {}

    def _update_info_dictionary(self, info, obs, data, carry):
        return info

    def _preprocess_action(self, action, data, carry):
        """
        Compute a transformation of the action provided to the
        environment.

        Args:
            action (np.ndarray or jax.Array): numpy array with the actions
                provided to the environment.

        Returns:
            The action to be used for the current step
        """
        return action

    def _compute_action(self, obs, action):
        """
        Compute a transformation of the action at every intermediate step.
        Useful to add control signals simulated directly in python.

        Args:
            obs (np.ndarray): numpy array with the current state of teh simulation;
            action (np.ndarray): numpy array with the actions, provided at every step.

        Returns:
            The action to be set in the actual pybullet simulation.

        """
        return action

    def _simulation_pre_step(self, data, carry):
        """
        Allows information to be accessed and changed at every intermediate step
        before taking a step in the mujoco simulation.
        Can be useful to apply an external force/torque to the specified bodies.

        ex: apply a force over X to the torso:
        force = [200, 0, 0]
        torque = [0, 0, 0]
        self.sim.data.xfrc_applied[self.sim.model._body_name2id["torso"],:] = force + torque

        """
        return data, carry

    def _simulation_post_step(self, data, carry):
        """
        Allows information to be accessed at every intermediate step
        after taking a step in the mujoco simulation.
        Can be useful to average forces over all intermediate steps.

        """
        return data, carry

    def load_model(self, xml_file):
        """
        Takes an xml_file and compiles and loads the model.

        Args:
            xml_file (str/xml handle): A string with a path to the xml or an Mujoco xml handle.

        Returns:
            Mujoco model.

        """
        if type(xml_file) == mjcf.element.RootElement:
            # load from xml handle
            model = mujoco.MjModel.from_xml_string(xml=xml_file.to_xml_string(),
                                                   assets=xml_file.get_assets())
            # todo: activate this
            #self._xml_handles = xml_file
        elif type(xml_file) == str:
            # load from path
            model = mujoco.MjModel.from_xml_path(xml_file)
            # todo: activate this
            #self._xml_handles = mjcf.from_path(xml_file)
        else:
            raise ValueError(f"Unsupported type for xml_file {type(xml_file)}.")

        return model

    def set_actuation_spec(self, actuation_spec):
        """
        Sets the actuation of the environment to overwrite the default one.

        Args:
            actuation_spec (list): A list of actuator names.

        """
        self._action_indices = self.get_action_indices(self._model, self._data, actuation_spec)
        self._mdp_info.action_space = Box(*self._get_action_limits(self._action_indices, self._model))

    def set_observation_spec(self, observation_spec):
        """
        Sets the observation of the environment to overwrite the default one.

        Args:
            observation_spec (list): A list of observation types.

        """
        # update the obs_container and the data_indices and obs_indices
        self.obs_container, self._data_indices, self._obs_indices = (
            self._setup_observations(observation_spec, self._goal, self._model, self._data))

        # update the observation space
        self._mdp_info.observation_space = Box(*self._get_obs_limits())

    @staticmethod
    def _setup_observations(observation_spec, goal, model, data):
        """
        Sets up the observation space for the environment. It generates a dictionary containing all the observation
        types and their corresponding information, as well as two dataclasses containing the indices in the
        Mujoco datastructure for each observation type (data_indices) and the indices for each observation type
        in the observation array (obs_indices). Goals are equally treated as observation types.

        Args:
            observation_spec (list): A list of observation types.
            goal (Goal): A goal class.
            model: Mujoco model.
            data: Mujoco data structure.

        Returns:
            A dictionary containing all the observation types and their corresponding information, as well as two
            dataclasses containing the indices in the Mujoco datastructure for each observation type (data_indices)
            and the indices for each observation type in the observation array (obs_indices).

        """

        # this dict will contain all the observation types and their corresponding information
        obs_container = ObservationContainer()

        # these containers will be used to store the indices of the different observation
        # types in the data structure and in the observation array.
        data_ind = ObservationIndexContainer()
        obs_ind = ObservationIndexContainer()

        i = 0
        # calculate the indices for the different observation types
        for obs in observation_spec:
            # initialize the observation type and get all relevant data indices
            d_ind, o_ind = obs.init_from_mj(model, data, i)
            # add the indices to the corresponding observation type data indices for fast observation retrieval
            # during the create_observation function
            if isinstance(obs, ObservationType.BodyPos):
                data_ind.body_xpos.extend(d_ind)
                obs_ind.body_xpos.extend(o_ind)
            elif isinstance(obs, ObservationType.BodyVel):
                data_ind.body_cvel.extend(d_ind)
                obs_ind.body_cvel.extend(o_ind)
            elif isinstance(obs, ObservationType.BodyRot):
                data_ind.body_xquat.extend(d_ind)
                obs_ind.body_xquat.extend(o_ind)
            elif isinstance(obs, ObservationType.JointPos):
                data_ind.joint_qpos.extend(d_ind)
                obs_ind.joint_qpos.extend(o_ind)
            elif isinstance(obs, ObservationType.JointVel):
                data_ind.joint_qvel.extend(d_ind)
                obs_ind.joint_qvel.extend(o_ind)
            elif isinstance(obs, ObservationType.FreeJointPos):
                data_ind.free_joint_qpos.extend(d_ind)
                obs_ind.free_joint_qpos.extend(o_ind)
            elif isinstance(obs, ObservationType.FreeJointVel):
                data_ind.free_joint_qvel.extend(d_ind)
                obs_ind.free_joint_qvel.extend(o_ind)
            elif isinstance(obs, ObservationType.SitePos):
                data_ind.site_xpos.extend(d_ind)
                obs_ind.site_xpos.extend(o_ind)
            elif isinstance(obs, ObservationType.SiteRot):
                data_ind.site_xmat.extend(d_ind)
                obs_ind.site_xmat.extend(o_ind)
            elif isinstance(obs, ObservationType.Force):
                data_ind.forces.extend(d_ind)
                obs_ind.forces.extend(o_ind)
            else:
                raise ValueError

            i += obs.dim
            if obs.name in obs_container.keys():
                raise KeyError("Duplicate keys are not allowed. Key: ", obs.name)

            obs_container[obs.name] = obs

        # add goal class to observation dict
        # todo: only single goals are supported, maybe change this in future
        if goal is not None:
            d_ind, o_ind = goal.init_from_mj(model, data, i)
            data_ind.goal.extend(d_ind)
            obs_ind.forces.extend(o_ind)

        if goal.name in obs_container.keys():
            raise KeyError("Duplicate keys are not allowed. Key: ", goal.name)

        obs_container[goal.name] = goal

        # convert all lists to numpy arrays
        data_ind.convert_to_numpy()
        obs_ind.convert_to_numpy()

        return obs_container, data_ind, obs_ind

    def _reward(self, obs, action, next_obs, absorbing, info, model, data, carry):
        return 0.0

    def _create_observation(self, data, carry):
        # get the base observation defined in observation_spec and the goal
        obs = self._create_observation_compat(data, np)
        return self._order_observation(obs)

    def _order_observation(self, obs):
        """
        order the indices to match the order in observation_spec + goal
        """
        obs[self._obs_indices.concatenated_indices] = obs
        return obs

    def _create_observation_compat(self, data, backend):
        """
        Creates the observation array by concatenating the observation extracted from all observation types.
        """

        # extract the observations from all observation types in the Mujoco Datastructure
        obs = backend.concatenate(
            [ObservationType.BodyPos.get_obs(None, data, self._data_indices.body_xpos, backend),
             ObservationType.BodyRot.get_obs(None, data, self._data_indices.body_xquat, backend),
             ObservationType.BodyVel.get_obs(None, data, self._data_indices.body_cvel, backend),
             ObservationType.FreeJointPos.get_obs(None, data, self._data_indices.free_joint_qpos, backend),
             ObservationType.FreeJointVel.get_obs(None, data, self._data_indices.free_joint_qvel, backend),
             ObservationType.JointPos.get_obs(None, data, self._data_indices.joint_qpos, backend),
             ObservationType.JointVel.get_obs(None, data, self._data_indices.joint_qvel, backend),
             ObservationType.SitePos.get_obs(None, data, self._data_indices.site_xpos, backend),
             ObservationType.SiteRot.get_obs(None, data, self._data_indices.site_xmat, backend),
             self._goal.get_obs(None, data, self._data_indices.goal, backend)])

        return obs

    @staticmethod
    def _set_sim_state_from_traj_data(data, traj_data):
        """
        Sets the Mujoco datastructure to the state specified in the trajectory data.

        Args:
            data: Mujoco data structure.
            traj_data: Trajectory data.

        """
        # set the body pos
        if traj_data.xpos.size > 0:
            data.xpos = traj_data.xpos
        # set the body orientation
        if traj_data.xquat.size > 0:
            data.xquat = traj_data.xquat
        # set the body velocity
        if traj_data.cvel.size > 0:
            data.cvel = traj_data.cvel
        # set the joint positions
        if traj_data.qpos.size > 0:
            data.qpos = traj_data.qpos
        # set the joint velocities
        if traj_data.qvel.size > 0:
            data.qvel = traj_data.qvel

        return data

    def _set_sim_state_from_obs(self, data, obs):
        """
        Sets the Mujoco datastructure to the state specified in the observation.

        Args:
            data: Mujoco data structure.
            obs: Observation array.

        """

        # set the body pos
        data.xpos[self._data_indices.body_xpos, :] = obs[self._obs_indices.body_xpos].reshape(-1, 3)
        # set the body orientation
        data.xquat[self._data_indices.body_xquat, :] = obs[self._obs_indices.body_xquat].reshape(-1, 4)
        # set the body velocity
        data.cvel[self._data_indices.body_cvel, :] = obs[self._obs_indices.body_cvel].reshape(-1, 6)
        # set the free joint positions
        data.qpos[self._data_indices.free_joint_qpos] = obs[self._obs_indices.free_joint_qpos]
        # set the free joint velocities
        data.qvel[self._data_indices.free_joint_qvel] = obs[self._obs_indices.free_joint_qvel]
        # set the joint positions
        data.qpos[self._data_indices.joint_qpos] = obs[self._obs_indices.joint_qpos]
        # set the joint velocities
        data.qvel[self._data_indices.joint_qvel] = obs[self._obs_indices.joint_qvel]
        # set the site positions
        data.site_xpos[self._data_indices.site_xpos, :] = obs[self._obs_indices.site_xpos].reshape(-1, 3)
        # set the site rotation
        data.site_xmat[self._data_indices.site_xmat, :] = obs[self._obs_indices.site_xmat].reshape(-1, 9)

        return data

    def _get_obs_limits(self):
        obs_min = np.concatenate([np.array(entry.min) for entry in self.obs_container.values()])
        obs_max = np.concatenate([np.array(entry.max) for entry in self.obs_container.values()])
        return obs_min, obs_max

    def _init_additional_carry(self, key, data):
        return AdditionalCarry(key=key, cur_step_in_episode=1)

    def get_model(self):
        return deepcopy(self._model)

    def get_data(self):
        return deepcopy(self._data)

    @staticmethod
    def _modify_model(model, option_config):
        if option_config is not None:
            for key, value in option_config.items():
                setattr(model.opt, key, value)

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

    @staticmethod
    def _get_action_limits(action_indices, model):
        """
        Returns the action space bounding box given the action_indices and the model.

         Args:
             action_indices (list): A list of actuator indices.
             model: MuJoCo model.

         Returns:
             Two nd.arrays defining the action space limits.

         """
        low = []
        high = []
        for index in action_indices:
            if model.actuator_ctrllimited[index]:
                low.append(model.actuator_ctrlrange[index][0])
                high.append(model.actuator_ctrlrange[index][1])
            else:
                low.append(-np.inf)
                high.append(np.inf)

        return np.array(low), np.array(high)

    def _get_collision_force(self, group1, group2):
        """
        Returns the collision force and torques between the specified groups.

        Args:
            group1 (string): A name referring to an entry contained in the
                collision_groups list handed to the constructor;
            group2 (string): A name referring to an entry contained in the
                collision_groups list handed to the constructor.

        Returns:
            A 6D vector specifying the collision forces/torques[3D force + 3D torque]
            between the given groups. Vector of 0's git statusin case there was no collision.
            http://mujoco.org/book/programming.html#siContact

        """
        # todo: implement this
        c_array = np.zeros(6, dtype=np.float64)
        return c_array

    @property
    def xml_handle(self):
        """ Returns the XML handle of the environment. This will raise an error if the environment contains more
            than one xml_handle. """

        if len(self._xml_handles) > 1:
            raise ValueError("This environment contains multiple models and hence multiple xml_handles. Use the "
                             "property \"xml_handles\" instead.")
        return self._xml_handles[0]

    @property
    def xml_handles(self):
        """ Returns all XML handles of the environment. """

        return self._xml_handles

    @property
    def cur_step_in_episode(self):
        return self._cur_step_in_episode

    @property
    def mjx_env(self):
        return False

    @property
    def mdp_info(self):
        return self._mdp_info

    @staticmethod
    def user_warning_raise_exception(warning):
        """
        Detects warnings in Mujoco and raises the respective exception.

        Args:
            warning: Mujoco warning.

        """
        if 'Pre-allocated constraint buffer is full' in warning:
            raise RuntimeError(warning + 'Increase njmax in mujoco XML')
        elif 'Pre-allocated contact buffer is full' in warning:
            raise RuntimeError(warning + 'Increase njconmax in mujoco XML')
        elif 'Unknown warning type' in warning:
            raise RuntimeError(warning + 'Check for NaN in simulation.')
        else:
            raise RuntimeError('Got MuJoCo Warning: ' + warning)

    @property
    def info(self):
        """
        Returns:
             An object containing the info of the environment.

        """
        return self._mdp_info

    @property
    def dt(self):
        return self._timestep * self._n_intermediate_steps * self._n_substeps

    @classmethod
    def register(cls):
        """
        Register an environment in the environment list.

        """
        env_name = cls.__name__

        if env_name not in Mujoco.registered_envs:
            Mujoco.registered_envs[env_name] = cls

    @staticmethod
    def list_registered():
        """
        List registered environments.

        Returns:
             The list of the registered environments.

        """
        return list(Mujoco.registered_envs.keys())

    @staticmethod
    def make(env_name, *args, **kwargs):
        """
        Generate an environment given an environment name and parameters.
        The environment is created using the generate method, if available. Otherwise, the constructor is used.
        The generate method has a simpler interface than the constructor, making it easier to generate a standard
        version of the environment. If the environment name contains a '.' separator, the string is splitted, the first
        element is used to select the environment and the other elements are passed as positional parameters.

        Args:
            env_name (str): Name of the environment,
            *args: positional arguments to be provided to the environment generator;
            **kwargs: keyword arguments to be provided to the environment generator.

        Returns:
            An instance of the constructed environment.

        """

        if '.' in env_name:
            env_data = env_name.split('.')
            env_name = env_data[0]
            args = env_data[1:] + list(args)

        env = Mujoco.registered_envs[env_name]

        if hasattr(env, 'generate'):
            return env.generate(*args, **kwargs)
        else:
            return env(*args, **kwargs)
