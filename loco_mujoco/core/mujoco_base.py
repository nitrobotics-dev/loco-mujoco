import atexit
from copy import deepcopy
from typing import Union
from enum import Enum
from typing import List, Optional, Any, Dict
from dataclasses import dataclass
from functools import partial
import mujoco
from mujoco import MjSpec
import numpy as np
from dm_control import mjcf

from mujoco import mjx
from flax import struct
import jax
import jax.numpy as jnp

from loco_mujoco.core.utils import Box, MDPInfo, MujocoViewer, Reward, info_property
from loco_mujoco.core.observations import ObservationType, ObservationIndexContainer, ObservationContainer, Goal
from loco_mujoco.core.domain_randomizer import DomainRandomizer
from loco_mujoco.core.terrain import Terrain
from loco_mujoco.core.utils import TerminalStateHandler


@struct.dataclass
class AdditionalCarry:
    key: jax.Array
    cur_step_in_episode: int
    observation_states: Union[np.ndarray, jax.Array]
    reward_state: Union[np.ndarray, jax.Array]
    domain_randomizer_state: Union[np.ndarray, jax.Array]
    terrain_state: Union[np.ndarray, jax.Array]


class Mujoco:
    """
        This is the base class for all Mujoco environments, CPU and MjX.

    """

    registered_envs = dict()

    def __init__(self, spec, actuation_spec, observation_spec, gamma, horizon,
                 n_environments=1, timestep=None, n_substeps=1, n_intermediate_steps=1, model_option_conf=None,
                 reward_type="NoReward", reward_params=None,
                 goal_type="NoGoal", goal_params=None,
                 terminal_state_type="RootPoseTrajTerminalStateHandler", terminal_state_params=None,
                 domain_randomization_type="NoDomainRandomization", domain_randomization_params=None,
                 terrain_type="StaticTerrain", terrain_params=None,
                 **viewer_params):

        # set the timestep if provided, else read it from model
        if timestep is not None:
            if model_option_conf is None:
                model_option_conf = {"timestep": timestep}
            else:
                model_option_conf["timestep"] = timestep

        # load the model, spec and data
        self._init_model, self._model, self._data, self._mjspec = self.load_mujoco(spec, model_option_conf)

        # set some attributes
        self._n_intermediate_steps = n_intermediate_steps
        self._n_substeps = n_substeps
        self._viewer_params = viewer_params
        self._viewer = None
        self._obs = None
        self._info = None
        self._additional_carry = None
        self._cur_step_in_episode = 0

        # setup goal
        spec, self._goal = self._setup_goal(spec, goal_type, goal_params)
        if self._goal.requires_spec_modification:
            self._init_model, self._model, self._data, self._mjspec = self.load_mujoco(spec)
        observation_spec.append(self._goal)

        # read the observation space, create a dictionary of observations and goals containing information
        # about each observation's type, indices, min and max values, etc. Additionally, create two dataclasses
        # containing the indices in the datastructure for each observation type (data_indices) and the indices for
        # each observation type in the observation array (obs_indices).
        self.obs_container, self._data_indices, self._obs_indices = (
            self._setup_observations(observation_spec, self._model, self._data))

        # define observation space bounding box
        observation_space = Box(*self._get_obs_limits())

        # read the actuation spec and build the mapping between actions and ids
        self._action_indices = self.get_action_indices(self._model, self._data, actuation_spec)

        # define action space bounding box
        action_space = Box(*self._get_action_limits(self._action_indices, self._model))

        # setup reward function
        self._reward_function = self._setup_reward(reward_type, reward_params)

        # setup terrain
        terrain_params = {} if terrain_params is None else terrain_params
        self._terrain = Terrain.registered[terrain_type](self, **terrain_params)
        if self._terrain.requires_spec_modification:
            spec = self._terrain.modify_spec(spec)
            self._init_model, self._model, self._data, self._mjspec = self.load_mujoco(spec)

        # setup domain randomization
        domain_randomization_params = {} if domain_randomization_params is None else domain_randomization_params
        self._domain_randomizer = DomainRandomizer.registered[domain_randomization_type](**domain_randomization_params)

        # setup terminal state handler
        if terminal_state_params is None:
            terminal_state_params = {}
        self._terminal_state_handler = TerminalStateHandler.make(terminal_state_type, self._model,
                                                                 self._get_all_info_properties(),
                                                                 **terminal_state_params)

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
        self._model = deepcopy(self._init_model)
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        # todo: replace all cur_step_in_episode to use additional info!
        self._additional_carry = self._init_additional_carry(key, self._model, self._data, np)
        self._data, self._additional_carry =\
            self._reset_init_data_and_model(self._model, self._data, self._additional_carry)

        # reset all stateful entities
        self._data, self._additional_carry = self.obs_container.reset_state(self, self._model, self._data,
                                                                            self._additional_carry, jnp)
        self._obs, self._additional_carry = self._create_observation(self._model, self._data, self._additional_carry)
        self._info = self._reset_info_dictionary(self._obs, self._data, subkey)
        return self._obs

    def step(self, action):
        cur_obs = self._obs.copy()
        cur_info = self._info.copy()
        carry = self._additional_carry

        # preprocess action
        action, carry = self._preprocess_action(action, self._model, self._data, carry)

        # modify obs and data, before stepping in the env (does nothing by default)
        cur_obs, self._data, cur_info, carry = self._step_init(cur_obs, self._data, cur_info, carry)

        ctrl_action = None

        for i in range(self._n_intermediate_steps):

            if self._recompute_action_per_step or ctrl_action is None:
                ctrl_action = self._compute_action(cur_obs, action)
                self._data.ctrl[self._action_indices] = ctrl_action

            # modify data during simulation, before main step
            self._model, self._data, carry = self._simulation_pre_step(self._model, self._data, carry)

            # main mujoco step, runs the sim for n_substeps
            mujoco.mj_step(self._model, self._data, self._n_substeps)

            # modify data during simulation, after main step (does nothing by default)
            self._data, carry = self._simulation_post_step(self._model, self._data, carry)

            # recompute thef action at each intermediate step (not executed by default)
            if self._recompute_action_per_step:
                cur_obs, carry = self._create_observation(self._model, self._data, carry)

        # create the final observation
        if not self._recompute_action_per_step:
            cur_obs, carry = self._create_observation(self._model, self._data, carry)

        # modify obs and data, before stepping in the env (does nothing by default)
        cur_obs, self._data, cur_info, carry = self._step_finalize(cur_obs, self._model, self._data, cur_info, carry)

        # update info (does nothing by default)
        cur_info = self._update_info_dictionary(cur_info, cur_obs, self._data, carry)

        # check if the current state is an absorbing state
        absorbing = self._is_absorbing(cur_obs, cur_info, self._data, carry)

        # calculate the reward
        reward, carry = self._reward(self._obs, action, cur_obs, absorbing, cur_info, self._model, self._data, carry)

        # calculate flag indicating whether this is the last obs before resetting
        done = self._is_done(cur_obs, absorbing, cur_info, self._data, carry)

        self._obs = cur_obs
        self._cur_step_in_episode += 1
        self._additional_carry = carry

        return np.asarray(cur_obs), reward, absorbing, done, cur_info

    def render(self, record=False):
        if self._viewer is None:
            self._viewer = MujocoViewer(self._model, self.dt, record=record, **self._viewer_params)

        if self._terrain.is_dynamic:
            terrain_state = self._additional_carry.terrain_state
            assert hasattr(terrain_state, "height_field_raw"), "Terrain state does not have height_field_raw."
            assert self._terrain.hfield_id is not None, "Terrain hfield id is not set."
            # todo: updating hfield buffer at every render is not efficient, rendering will be slow
            hfield_data = np.array(terrain_state.height_field_raw)
            self._model.hfield_data = hfield_data
            self._viewer.upload_hfield(self._model, hfield_id=self._terrain.hfield_id)

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

    def _reset_init_data_and_model(self, model, data, carry):
        """
        Initializes the data and model at the beginning of the reset.

        Args:
            model: Mujoco model.
            data: Mujoco data structure.
            carry: Additional carry information.

        Returns:
            The updated model, data and carry.
        """
        data, carry = self._terrain.reset(self, model, data, carry, np)
        data, carry = self._domain_randomizer.reset(self, model, data, carry, np)
        return data, carry

    def _step_init(self, obs, data, info, carry):
        return obs, data, info, carry

    def _step_finalize(self, obs, model, data, info, carry):
        """
        Allows information to be accessed at the end of a step.
        """
        obs, carry = self._domain_randomizer.update_observation(self, obs, model, data, carry, np)
        return obs, data, info, carry

    def _reset_info_dictionary(self, obs, data, key):
        return {}

    def _update_info_dictionary(self, info, obs, data, carry):
        return info

    def _preprocess_action(self, action, model, data, carry):
        """
        Compute a transformation of the action provided to the
        environment.

        Args:
            action (np.ndarray): numpy array with the actions
                provided to the environment.
            model: Mujoco model.
            data: Mujoco data structure.
            carry: Additional carry information.

        Returns:
            The action to be used for the current step and the updated carry.
        """
        action, carry = self._domain_randomizer.update_action(self, action, model, data, carry, np)
        return action, carry

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

    def _simulation_pre_step(self, model, data, carry):
        """
        Allows information to be accessed and changed at every intermediate step
        before taking a step in the mujoco simulation.

        """
        model, data, carry = self._terrain.update(self, model, data, carry, np)
        model, data, carry = self._domain_randomizer.update(self, model, data, carry, np)
        return model, data, carry

    def _simulation_post_step(self, model, data, carry):
        """
        Allows information to be accessed at every intermediate step
        after taking a step in the mujoco simulation.
        Can be useful to average forces over all intermediate steps.

        """
        return data, carry

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
            self._setup_observations(observation_spec, self._model, self._data))

        # update the observation space
        self._mdp_info.observation_space = Box(*self._get_obs_limits())

    @staticmethod
    def _setup_observations(observation_spec, model, data):
        """
        Sets up the observation space for the environment. It generates a dictionary containing all the observation
        types and their corresponding information, as well as two dataclasses containing the indices in the
        Mujoco datastructure for each observation type (data_indices) and the indices for each observation type
        in the observation array (obs_indices).

        Args:
            observation_spec (list): A list of observation types.
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
            obs.init_from_mj(model, data, i, data_ind, obs_ind)
            i += obs.dim
            obs_container[obs.name] = obs

        # lock container to avoid unwanted modifications
        obs_container.lock()

        # convert all lists to numpy arrays
        data_ind.convert_to_numpy()
        obs_ind.convert_to_numpy()

        return obs_container, data_ind, obs_ind

    def _setup_reward(self, reward_type, reward_params):
        """
        Constructs a reward function.

        Args:
            reward_type (string): Name of the reward.
            reward_params (dict): Parameters of the reward function.

        Returns:
            Reward function.

        """
        # collect all info properties of the env (dict all @info_properties decorated function returns)
        info_props = self._get_all_info_properties()

        reward_cls = Reward.registered[reward_type]
        reward = reward_cls(self.obs_container, info_props=info_props, model=self._model, data=self._data)\
            if reward_params is None else reward_cls(self.obs_container, info_props=info_props, model=self._model,
                                                     data=self._data, **reward_params)

        return reward

    def _setup_goal(self, spec, goal_type, goal_params):
        """
        Setup the goal.

        Args:
            spec (MjSpec): Specification of the environment.
            goal_type (str): Type of the goal.
            goal_params (dict): Parameters of the goal.

        Returns:
            MjSpec: Modified specification.
            Goal: Goal
        """
        # collect all info properties of the env (dict all @info_properties decorated function returns)
        info_props = self._get_all_info_properties()

        # get the goal
        goal_cls = Goal.registered[goal_type]
        goal = goal_cls(info_props=info_props) if goal_params is None \
            else goal_cls(info_props=info_props, **goal_params)

        # apply the modification to the spec if needed
        spec = goal.apply_spec_modifications(spec, info_props)

        return spec, goal

    def _reward(self, obs, action, next_obs, absorbing, info, model, data, carry):
        return 0.0, carry

    def _create_observation(self, model, data, carry):
        """
        Creates the observation array by concatenating the observation extracted from all observation types.
        """
        # fast getter for all simple non-stateful observations
        obs_not_stateful = np.concatenate([obs_type.get_all_obs_of_type(self, model, data, self._data_indices, np)
                                   for obs_type in ObservationType.list_all_non_stateful()])
        # order non-stateful obs the way they were in obs_spec
        obs_not_stateful[self._obs_indices.concatenated_indices] = obs_not_stateful

        # get all stateful observations
        obs_stateful = []
        for obs in self.obs_container.list_all_stateful():
            obs_s, carry = obs.get_obs_and_update_state(self, model, data, carry, np)
            obs_stateful.append(obs_s)

        return np.concatenate([obs_not_stateful, *obs_stateful]), carry

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

    def _init_additional_carry(self, key, model, data, backend):
        return AdditionalCarry(key=key,
                               cur_step_in_episode=1,
                               observation_states=self.obs_container.init_state(self, key, model, data, backend))

    def get_model(self):
        return deepcopy(self._model)

    def get_data(self):
        return deepcopy(self._data)

    def load_mujoco(self, xml_file, model_option_conf=None):
        """
        Takes a xml_file and compiles and loads the model.

        Args:
            xml_file (str/MjSpec): A string with a path to the xml or a Mujoco specification.

        Returns:
            Mujoco model and the specification.

        """
        if type(xml_file) == MjSpec:
            # compile from specification
            if model_option_conf is not None:
                xml_file = self._modify_option_spec(xml_file, model_option_conf)
            model = xml_file.compile()
            spec = xml_file
        elif type(xml_file) == str:
            # load from path
            spec = mujoco.MjSpec.from_file(xml_file)
            if model_option_conf is not None:
                spec = self._modify_option_spec(spec, model_option_conf)
            model = spec.compile()
        else:
            raise ValueError(f"Unsupported type for xml_file {type(xml_file)}.")

        # create data
        data = mujoco.MjData(model)

        return model, deepcopy(model), data, spec

    @staticmethod
    def _modify_option_spec(spec, option_config):
        if option_config is not None:
            for key, value in option_config.items():
                setattr(spec.option, key, value)
        return spec

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

    def _get_all_info_properties(self):
        """
        Returns all info properties of the environment. (decorated with @info_property)

        """
        info_props = {}
        for attr_name in dir(self):
            attr_value = getattr(self.__class__, attr_name, None)
            if isinstance(attr_value, property) and getattr(attr_value.fget, '_is_info_property', False):
                info_props[attr_name] = getattr(self, attr_name)
        return info_props

    @property
    def mjspec(self):
        return self._mjspec

    @property
    def cur_step_in_episode(self):
        return self._cur_step_in_episode

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

    @info_property
    def simulation_dt(self):
        return self._model.opt.timestep

    @info_property
    def dt(self):
        return self.simulation_dt * self._n_intermediate_steps * self._n_substeps

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
