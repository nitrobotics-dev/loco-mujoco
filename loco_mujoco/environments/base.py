import os

import warnings
from pathlib import Path
from copy import deepcopy
from tempfile import mkdtemp
from itertools import product
from functools import partial

import jax.random
import jax.numpy as jnp
import numpy as np
from flax import struct
import mujoco
from dm_control import mjcf

import loco_mujoco
from loco_mujoco.core.mujoco_mjx import Mjx, MjxState
from loco_mujoco.core.utils import Box, VideoRecorder
from loco_mujoco.utils import Trajectory
from loco_mujoco.utils import NoReward, CustomReward,\
    TargetVelocityReward, PosReward, DomainRandomizationHandler


@struct.dataclass
class TrajState:
    traj_no: int
    subtraj_step_no: int


class LocoEnv(Mjx):
    """
    Base class for all kinds of locomotion environments.

    """

    def __init__(self, xml_handles, action_spec, observation_spec, collision_groups=None, enable_mjx=False,
                 n_envs=1, gamma=0.99, horizon=1000, n_substeps=10,  reward_type=None, reward_params=None,
                 traj_params=None, random_start=True, fixed_start_conf=None, timestep=0.001,
                 use_foot_forces=False, default_camera_mode="follow", use_absorbing_states=True,
                 domain_randomization_config=None, parallel_dom_rand=True, N_worker_per_xml_dom_rand=4,
                 model_option_conf=None, **viewer_params):
        """
        Constructor.

        Args:
            xml_handles : MuJoCo xml handles.
            actuation_spec (list): A list specifying the names of the joints
                which should be controllable by the agent. Can be left empty
                when all actuators should be used;
            observation_spec (list): A list containing the names of data that
                should be made available to the agent as an observation and
                their type (ObservationType). They are combined with a key,
                which is used to access the data. An entry in the list
                is given by: (key, name, type). The name can later be used
                to retrieve specific observations;
            collision_groups (list, None): A list containing groups of geoms for
                which collisions should be checked during simulation via
                ``check_collision``. The entries are given as:
                ``(key, geom_names)``, where key is a string for later
                referencing in the "check_collision" method, and geom_names is
                a list of geom names in the XML specification;
            enable_mjx (bool): Flag specifying wheter Mjx simulation is enabled or not.
            n_envs (int): Number of environment to run in parallel when using Mjx.
            gamma (float): The discounting factor of the environment;
            horizon (int): The maximum horizon for the environment;
            n_substeps (int): The number of substeps to use by the MuJoCo
                simulator. An action given by the agent will be applied for
                n_substeps before the agent receives the next observation and
                can act accordingly;
            reward_type (string): Type of reward function to be used.
            reward_params (dict): Dictionary of parameters corresponding to
                the chosen reward function;
            traj_params (dict): Dictionrary of parameters to construct trajectories.
            random_start (bool): If True, a random sample from the trajectories
                is chosen at the beginning of each time step and initializes the
                simulation according to that. This requires traj_params to be passed!
            fixed_start_conf (tuple(int)): Tuple of two ints defining the trajectory index and within that
                trajectory the sample to be used for initializing the simulation. Default is None.
                If both, random_start is False and fixed_start_conf is None, then the default initial starting
                position (all qpos=0) is used.
            the If set, the respective sample from the trajectories
                is taken to initialize the simulation;
            timestep (float): The timestep used by the MuJoCo simulator. If None, the
                default timestep specified in the XML will be used;
            use_foot_forces (bool): If True, foot forces are computed and added to
                the observation space;
            default_camera_mode (str): String defining the default camera mode. Available modes are "static",
                "follow", and "top_static".
            use_absorbing_states (bool): If True, absorbing states are defined for each environment. This means
                that episodes can terminate earlier.
            domain_randomization_config (str): Path to the domain/dynamics randomization config file.
            parallel_dom_rand (bool): If True and a domain_randomization_config file is passed, the domain
                randomization will run in parallel to speed up simulation run-time.
            N_worker_per_xml_dom_rand (int): Number of workers used per xml-file for parallel domain randomization.
                If parallel is set to True, this number has to be greater 1.

        """

        if type(xml_handles) != list:
            xml_handles = [xml_handles]
        self._xml_handles = xml_handles

        if collision_groups is None:
            collision_groups = list()

        if use_foot_forces:
            n_intermediate_steps = n_substeps
            n_substeps = 1
        else:
            n_intermediate_steps = 1

        if "geom_group_visualization_on_startup" not in viewer_params.keys():
            viewer_params["geom_group_visualization_on_startup"] = [0, 2]   # enable robot geom [0] and floor visual [2]

        if domain_randomization_config is not None:
            self._domain_rand = DomainRandomizationHandler(xml_handles, domain_randomization_config, parallel_dom_rand,
                                                           N_worker_per_xml_dom_rand)
        else:
            self._domain_rand = None

        # todo: xml_handles are currently not supported to be lists
        xml_handles = xml_handles[0]

        if enable_mjx:
            # call parent (Mjx) constructor
            super(LocoEnv, self).__init__(n_envs, xml_file=xml_handles, actuation_spec=action_spec,
                                          observation_spec=observation_spec, gamma=gamma,
                                          horizon=horizon, n_substeps=n_substeps,
                                          n_intermediate_steps=n_intermediate_steps,
                                          timestep=timestep, collision_groups=collision_groups,
                                          default_camera_mode=default_camera_mode,
                                          model_option_conf=model_option_conf, **viewer_params)
        else:
            assert n_envs == 1, "Mjx not enabled, setting the number of environments > 1 is not allowed."
            # call grandparent constructor (Mujoco (CPU) environment)
            super(Mjx, self).__init__(xml_file=xml_handles, actuation_spec=action_spec,
                                      observation_spec=observation_spec, gamma=gamma,
                                      horizon=horizon, n_substeps=n_substeps, n_intermediate_steps=n_intermediate_steps,
                                      timestep=timestep, collision_groups=collision_groups,
                                      default_camera_mode=default_camera_mode,
                                      model_option_conf=model_option_conf, **viewer_params)

        # specify reward function
        self._reward_function = self._get_reward_function(reward_type, reward_params)

        # optionally use foot forces in the observation space
        self._use_foot_forces = use_foot_forces

        self.info.observation_space = Box(*self._get_observation_space())

        # the action space is supposed to be between -1 and 1, so we normalize it
        low, high = self.info.action_space.low.copy(), self.info.action_space.high.copy()
        self.norm_act_mean = (high + low) / 2.0
        self.norm_act_delta = (high - low) / 2.0
        self.info.action_space.low[:] = -1.0
        self.info.action_space.high[:] = 1.0

        # todo: implement running average and reactivate here
        # setup a running average window for the mean ground forces
        #self.mean_grf = self._setup_ground_force_statistics()

        # dataset dummy
        self._dataset = None

        if traj_params:
            self.trajectories = None
            self.load_trajectory(traj_params)
            self._trajectory_loaded = True
            self._jax_trajectory = self.trajectories.get_jax_trajectory()
        else:
            self.trajectories = None
            self._trajectory_loaded = True

        self._random_start = random_start
        self._fixed_start_conf = fixed_start_conf
        self._use_fixed_start = True if fixed_start_conf is not None else False

        self._use_absorbing_states = use_absorbing_states

    def load_trajectory(self, traj_params, warn=True):
        """
        Loads trajectories. If there were trajectories loaded already, this function overrides the latter.

        Args:
            traj_params (dict): Dictionary of parameters needed to load trajectories.
            warn (bool): If True, a warning will be raised if the
                trajectory ranges are violated.

        """

        if self.trajectories is not None:
            warnings.warn("New trajectories loaded, which overrides the old ones.", RuntimeWarning)

        self.trajectories = Trajectory(keys=self.get_all_observation_keys(),
                                       low=self.info.observation_space.low,
                                       high=self.info.observation_space.high,
                                       joint_pos_idx=self._joint_qpos_range,
                                       interpolate_map=self._interpolate_map,
                                       interpolate_remap=self._interpolate_remap,
                                       interpolate_map_params=self._get_interpolate_map_params(),
                                       interpolate_remap_params=self._get_interpolate_remap_params(),
                                       warn=warn,
                                       **traj_params)
        self._jax_trajectory = self.trajectories.get_jax_trajectory()
        self._trajectory_loaded = True

    def reward(self, state, action, next_state, absorbing):
        """
        Calls the reward function of the environment.

        """

        return self._reward_function(state, action, next_state, absorbing)

    def setup(self, data, key):
        """
        Function to setup the initial state of the simulation. Initialization can be done either
        randomly, from a certain initial, or from the default initial state of the model.

        Args:
            obs (np.array): Observation to initialize the environment from;

        """

        # todo: think about how to handle reward resets
        #self._reward_function.reset_state()


        if self.trajectories is not None:
            if self._random_start:
                sample = self.trajectories.reset_trajectory()
            elif self._init_step_no:
                traj_len = self.trajectories.trajectory_length
                n_traj = self.trajectories.number_of_trajectories
                assert self._init_step_no <= traj_len * n_traj
                substep_no = int(self._init_step_no % traj_len)
                traj_no = int(self._init_step_no / traj_len)
                sample = self.trajectories.reset_trajectory(substep_no, traj_no)
            else:
                # sample random trajectory and use the first sample
                sample = self.trajectories.reset_trajectory(substep_no=0)

            self.set_sim_state(sample)

    def _is_absorbing(self, obs, info, data):
        """
        Checks if an observation is an absorbing state or not.

        Args:
            obs (np.array): Current observation;

        Returns:
            True, if the observation is an absorbing state; otherwise False;

        """
        return self._has_fallen(obs, info, data) if self._use_absorbing_states else False

    def _mjx_is_absorbing(self, obs, info, data):
        return jax.lax.cond(self._use_absorbing_states, lambda o, i, d: self._mjx_has_fallen(o, i, d),
                            lambda o, i, d: jnp.array(False), obs, info, data)

    def get_kinematic_obs_mask(self):
        """
        Returns a mask (np.array) for the observation specified in observation_spec (or part of it).

        """

        return np.arange(len(self.obs_helper.observation_spec) - 2)

    def get_obs_idx(self, key):
        """
        Returns a list of indices corresponding to the respective key.

        """

        # shift by 2 to account for deleted x and y
        idx = self._obs_dict[key].obs_ind
        idx = [i-2 for i in idx]

        return idx

    def create_dataset(self, ignore_keys=None):
        """
        Creates a dataset from the specified trajectories.

        Args:
            ignore_keys (list): List of keys to ignore in the dataset.

        Returns:
            Dictionary containing states, next_states and absorbing flags. For the states the shape is
            (N_traj x N_samples_per_traj, dim_state), while the absorbing flag has the shape is
            (N_traj x N_samples_per_traj). For perfect and preference datasets, the actions are also provided.

        """

        if self._dataset is None:
            if self.trajectories is not None:
                dataset = self.trajectories.create_dataset(ignore_keys=ignore_keys)
                # check that all state in the dataset satisfy the has fallen method.
                for state in dataset["states"]:
                    has_fallen, msg = self._has_fallen(state, return_err_msg=True)
                    if has_fallen:
                        err_msg = "Some of the states in the created dataset are terminal states. " \
                                  "This should not happen.\n\nViolations:\n"
                        err_msg += msg
                        raise ValueError(err_msg)

            else:
                raise ValueError("No trajectory was passed to the environment. "
                                 "To create a dataset pass a trajectory first.")

            self._dataset = deepcopy(dataset)

            return dataset
        else:
            return deepcopy(self._dataset)

    def play_trajectory(self, n_episodes=None, n_steps_per_episode=None, render=True,
                        record=False, recorder_params=None, key=None):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the trajectories at every step.

        Args:
            n_episodes (int): Number of episode to replay.
            n_steps_per_episode (int): Number of steps to replay per episode.
            render (bool): If True, trajectory will be rendered.
            record (bool): If True, the rendered trajectory will be recorded.
            recorder_params (dict): Dictionary containing the recorder parameters.

        """

        assert self.trajectories is not None

        if key is None:
            key = jax.random.key(0)

        if record:
            assert render
            fps = 1/self.dt
            recorder = VideoRecorder(fps=fps, **recorder_params) if recorder_params is not None else\
                VideoRecorder(fps=fps)
        else:
            recorder = None

        key, subkey = jax.random.split(key)
        self.reset(subkey)
        traj_no = 0
        subtraj_step_no = 0
        sample = self.get_traj_current_sample(self._jax_trajectory, traj_no, subtraj_step_no)
        self._set_sim_state(self._data, np.array(sample))

        if render:
            frame = self.render(record)
        else:
            frame = None

        if record:
            recorder(frame)

        highest_int = np.iinfo(np.int32).max
        if n_steps_per_episode is None:
            n_steps_per_episode = highest_int
        if n_episodes is None:
            n_episodes = highest_int
        for i in range(n_episodes):
            for j in range(n_steps_per_episode):

                self._set_sim_state(self._data, np.array(sample))

                self._simulation_pre_step(self._data)
                mujoco.mj_forward(self._model, self._data)
                self._simulation_post_step(self._data)

                sample, traj_no, subtraj_step_no = self.get_traj_next_sample(self._jax_trajectory,
                                                                             traj_no, subtraj_step_no)

                obs = self._create_observation(self._data)

                if self._has_fallen(obs, {}, self._data):
                    print("Has fallen!")

                if render:
                    frame = self.render(record)
                else:
                    frame = None

                if record:
                    recorder(frame)

            key, subkey = jax.random.split(key)
            self.reset(subkey)

        self.stop()
        if record:
            recorder.stop()

    def play_trajectory_from_velocity(self, n_episodes=None, n_steps_per_episode=None, render=True,
                                      record=False, recorder_params=None):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones calculated from the joint velocities
        in the trajectories at every step. Therefore, the joint positions
        are set from the trajectory in the first step. Afterward, numerical
        integration is used to calculate the next joint positions using
        the joint velocities in the trajectory.

        Args:
            n_episodes (int): Number of episode to replay.
            n_steps_per_episode (int): Number of steps to replay per episode.
            render (bool): If True, trajectory will be rendered.
            record (bool): If True, the replay will be recorded.
            recorder_params (dict): Dictionary containing the recorder parameters.

        """

        assert self.trajectories is not None

        if record:
            assert render
            fps = 1/self.dt
            recorder = VideoRecorder(fps=fps, **recorder_params) if recorder_params is not None else\
                VideoRecorder(fps=fps)
        else:
            recorder = None

        self.reset()
        sample = self.trajectories.get_current_sample()
        self.set_sim_state(sample)
        if render:
            frame = self.render(record)
        else:
            frame = None

        if record:
            recorder(frame)

        highest_int = np.iinfo(np.int32).max
        if n_steps_per_episode is None:
            n_steps_per_episode = highest_int
        if n_episodes is None:
            n_episodes = highest_int
        len_qpos, len_qvel = self._len_qpos_qvel()
        curr_qpos = sample[0:len_qpos]
        for i in range(n_episodes):
            for j in range(n_steps_per_episode):

                qvel = sample[len_qpos:len_qpos + len_qvel]
                qpos = [qp + self.dt * qv for qp, qv in zip(curr_qpos, qvel)]
                sample[:len(qpos)] = qpos

                self.set_sim_state(sample)

                self._simulation_pre_step()
                mujoco.mj_forward(self._model, self._data)
                self._simulation_post_step()

                # get current qpos
                curr_qpos = self._get_joint_pos()

                sample = self.trajectories.get_next_sample()
                if sample is None:
                    self.reset()
                    sample = self.trajectories.get_current_sample()
                    curr_qpos = sample[0:len_qpos]

                obs = self._create_observation(np.concatenate(sample))
                if self._has_fallen(obs):
                    print("Has fallen!")

                if render:
                    frame = self.render(record)
                else:
                    frame = None

                if record:
                    recorder(frame)

            self.reset()

            # get current qpos
            curr_qpos = self._get_joint_pos()

        self.stop()
        if record:
            recorder.stop()

    def load_dataset_and_get_traj_files(self, dataset_path, freq=None):
        """
        Calculates a dictionary containing the kinematics given a dataset. If freq is provided,
        the x and z positions are calculated based on the velocity.

        Args:
            dataset_path (str): Path to the dataset.
            freq (float): Frequency of the data in obs.

        Returns:
            Dictionary containing the keys specified in observation_spec with the corresponding
            values from the dataset.

        """

        dataset = np.load(str(Path(loco_mujoco.__file__).resolve().parent / dataset_path))
        self._dataset = deepcopy({k: d for k, d in dataset.items()})

        states = dataset["states"]
        last = dataset["last"]

        states = np.atleast_2d(states)
        rel_keys = [obs_spec[0] for obs_spec in self.obs_helper.observation_spec]
        num_data = len(states)
        trajectories = dict()
        for i, key in enumerate(rel_keys):
            if i < 2:
                if freq is None:
                    # fill with zeros for x and y position
                    data = np.zeros(num_data)
                else:
                    # compute positions from velocities
                    dt = 1 / float(freq)
                    assert len(states) > 2
                    vel_idx = rel_keys.index("d" + key) - 2
                    data = [0.0]
                    for j, o in enumerate(states[:-1, vel_idx], 1):
                        if last is not None and last[j - 1] == 1:
                            data.append(0.0)
                        else:
                            data.append(data[-1] + dt * o)
                    data = np.array(data)
            else:
                data = states[:, i - 2]
            trajectories[key] = data

        # add split points
        if len(states) > 2:
            trajectories["split_points"] = np.concatenate([[0], np.squeeze(np.argwhere(last == 1) + 1)])

        return trajectories

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

    def _get_observation_space(self):
        """
        Returns a tuple of the lows and highs (np.array) of the observation space.

        """

        sim_low, sim_high = (self.info.observation_space.low[2:],
                             self.info.observation_space.high[2:])

        if self._use_foot_forces:
            grf_low, grf_high = (-np.ones((self._get_grf_size(),)) * np.inf,
                                 np.ones((self._get_grf_size(),)) * np.inf)
            return (np.concatenate([sim_low, grf_low]),
                    np.concatenate([sim_high, grf_high]))
        else:
            return sim_low, sim_high

    def _create_observation(self, data):
        obs = super()._create_observation(data)

        if self._use_foot_forces:
            obs = np.concatenate([obs[2:],
                                  self.mean_grf.mean / 1000.,
                                  ]).flatten()
        else:
            obs = np.concatenate([obs[2:],
                                  ]).flatten()

        return obs

    def _mjx_create_observation(self, data):
        obs = super()._mjx_create_observation(data)

        # remove the first two entries and add foot forces if needed
        # todo: foot forces not added yet for mjx
        obs = jax.lax.cond(self._use_foot_forces, lambda o: obs[2:].flatten(),
                           lambda o: obs[2:].flatten(), obs)
        return obs

    def reset(self, key):
        key, subkey = jax.random.split(key)
        obs = super().reset(key)

        # some sanity checks
        self._check_reset_configuration()

        # reset trajectory state
        traj_state = self._reset_trajectory_state(subkey)
        self._info["TrajState"] = traj_state

        if self._random_start or self._use_fixed_start:
            # init simulation from trajectory state
            sample = self._get_init_state_from_trajectory(traj_state)      # get sample from trajectory state
            self._set_sim_state(self._data, sample)

        obs = self._create_observation(self._data)
        return obs

    def mjx_reset(self, key):
        key, subkey = jax.random.split(key)
        mjx_state = super().mjx_reset(key)

        # some sanity checks
        jax.debug.callback(self._check_reset_configuration)

        # reset trajectory state
        traj_state = self._reset_trajectory_state(subkey)
        mjx_state.info["TrajState"] = traj_state

        if self._random_start or self._use_fixed_start:
            # init simulation from trajectory state
            sample = self._get_init_state_from_trajectory(traj_state)      # get sample from trajectory state
            data = self._mjx_set_sim_state(mjx_state.data, sample)
            mjx_state = mjx_state.replace(data=data, observation=self._mjx_create_observation(data))

        return mjx_state

    def _mjx_reset_in_step(self, state: MjxState):
        def where_done(x, y):
            done = state.done
            return jnp.where(done, x, y)

        # reset trajectory state
        key = state.info["key"]
        key, subkey = jax.random.split(key)
        traj_state = self._reset_trajectory_state(subkey)
        # jax.debug.print("traj_no {x} and subtraj_no: {y}", x=traj_state.traj_no, y=traj_state.subtraj_step_no)
        state.info["TrajState"] = traj_state

        if self._random_start or self._use_fixed_start:
            # init simulation from trajectory state
            sample = self._get_init_state_from_trajectory(traj_state)      # get sample from trajectory state
            data = jax.tree.map(where_done, self._mjx_set_sim_state(state.data, sample), state.data)
            #data = jax.lax.cond(state.done[0], lambda d, s: d, lambda d, s: self._mjx_set_sim_state(d, s), state.data, sample)

        else:
            # init simulation from default state
            data = jax.tree.map(where_done, state.first_data, state.data)

        final_obs = where_done(state.observation, jnp.zeros_like(state.observation))
        state.info["cur_step_in_episode"] = where_done(0, state.info["cur_step_in_episode"])
        new_obs = self._mjx_create_observation(data)

        state.info["key"] = key

        return state.replace(data=data, observation=new_obs, final_observation=final_obs)

    def _update_info_dictionary(self, info, obs, data):
        info = super()._update_info_dictionary(info, obs, data)
        self._update_trajectory_info(info)
        return info

    def _mjx_update_info_dictionary(self, info, obs, data):
        super()._mjx_update_info_dictionary(info, obs, data)
        self._update_trajectory_info(info)
        return info

    @partial(jax.jit, static_argnums=(0,))
    def _reset_trajectory_state(self, key):

        n_trajs = self.n_trajectories(self._jax_trajectory)
        len_traj = self.len_trajectory(self._jax_trajectory)

        if self._random_start:
            idx = jax.random.randint(key, shape=(2,), minval=jnp.array([0, 0]),
                                     maxval=jnp.array([n_trajs, len_traj]))
        elif self._use_fixed_start:
            idx = self._fixed_start_conf
        else:
            idx = [0, 0]

        new_traj_no, new_subtraj_step_no = idx
        traj_state = TrajState(new_traj_no, new_subtraj_step_no)
        return traj_state

    @partial(jax.jit, static_argnums=(0,))
    def _get_init_state_from_trajectory(self, traj_state):
        sample = self.get_traj_current_sample(self._jax_trajectory, traj_state.traj_no, traj_state.subtraj_step_no)
        sample = sample.at[0:2].set(0.0)
        return sample

    def _update_trajectory_info(self, info):
        traj_state = info["TrajState"]
        next_traj_no, next_subtraj_step_no = self.increment_traj_counter(self._jax_trajectory,
                                                                         traj_state.traj_no,
                                                                         traj_state.subtraj_step_no)
        traj_state = traj_state.replace(traj_no=next_traj_no, subtraj_step_no=next_subtraj_step_no)
        info["TrajState"] = traj_state

    def _preprocess_action(self, action, data):
        """
        This function preprocesses all actions. All actions in this environment expected to be between -1 and 1.
        Hence, we need to unnormalize the action to send to correct action to the simulation.
        Note: If the action is not in [-1, 1], the unnormalized version will be clipped in Mujoco.

        Args:
            action (np.array): Action to be sent to the environment;

        Returns:
            Unnormalized action (np.array) that is sent to the environment;

        """
        return self._preprocess_action_compat(action, data)

    def _mjx_preprocess_action(self, action, data):
        return self._preprocess_action_compat(action, data)

    def _preprocess_action_compat(self, action, data):
        unnormalized_action = ((action * self.norm_act_delta) + self.norm_act_mean)
        return unnormalized_action

    def _simulation_post_step(self, data):
        """
        Update the ground forces statistics if needed.

        """

        if self._use_foot_forces:
            grf = self._get_ground_forces()
            self.mean_grf.update_stats(grf)

        return data

    def _init_sim_from_obs(self, obs):
        """
        Initializes the simulation from an observation.

        Args:
            obs (np.array): The observation to set the simulation state to.

        """

        assert len(obs.shape) == 1

        # append x and y pos
        obs = np.concatenate([[0.0, 0.0], obs])

        obs_spec = self.obs_helper.observation_spec
        assert len(obs) >= len(obs_spec)

        # remove anything added to obs that is not in obs_spec
        obs = obs[:len(obs_spec)]

        # set state
        self.set_sim_state(obs)

    def _setup_ground_force_statistics(self):
        """
        Returns a running average method for the mean ground forces.  By default, 4 ground force sensors are used.
        Environments that use more or less have to override this function.

        """

        mean_grf = RunningAveragedWindow(shape=(self._get_grf_size(),), window_size=self._n_intermediate_steps)

        return mean_grf

    def _get_ground_forces(self):
        """
        Returns the ground forces (np.array). By default, 4 ground force sensors are used.
        Environments that use more or less have to override this function.

        """

        grf = np.concatenate([self._get_collision_force("floor", "foot_r")[:3],
                              self._get_collision_force("floor", "front_foot_r")[:3],
                              self._get_collision_force("floor", "foot_l")[:3],
                              self._get_collision_force("floor", "front_foot_l")[:3]])

        return grf

    def _get_reward_function(self, reward_type, reward_params):
        """
        Constructs a reward function.

        Args:
            reward_type (string): Name of the reward.
            reward_params (dict): Parameters of the reward function.

        Returns:
            Reward function.

        """

        if reward_type == "custom":
            reward_func = CustomReward(**reward_params)
        elif reward_type == "target_velocity":
            x_vel_idx = self.get_obs_idx("dq_pelvis_tx")
            assert len(x_vel_idx) == 1
            x_vel_idx = x_vel_idx[0]
            reward_func = TargetVelocityReward(x_vel_idx=x_vel_idx, **reward_params)
        elif reward_type == "x_pos":
            x_idx = self.get_obs_idx("q_pelvis_tx")
            assert len(x_idx) == 1
            x_idx = x_idx[0]
            reward_func = PosReward(pos_idx=x_idx)
        elif reward_type is None:
            reward_func = NoReward()
        else:
            raise NotImplementedError("The specified reward has not been implemented: %s" % reward_type)

        return reward_func

    def _get_joint_pos(self):
        """
        Returns a vector (np.array) containing the current joint position of the model in the simulation.

        """

        return self.obs_helper.get_joint_pos_from_obs(self.obs_helper._build_obs(self._data))

    def _get_joint_vel(self):
        """
        Returns a vector (np.array) containing the current joint velocities of the model in the simulation.

        """

        return self.obs_helper.get_joint_vel_from_obs(self.obs_helper._build_obs(self._data))

    @partial(jax.jit, static_argnames=['self', 'key'])
    def _get_from_obs(self, obs, key):
        """
        Returns a part of the observation based on the specified keys.

        Args:
            obs (np.array or jnp.array): Observation array.
            key str: Key which are used to extract entries from the observation.

        Returns:
            jnp.array including the parts of the original observation whose
            keys were specified.

        """

        # account for removed x, y
        idx = self._obs_dict[key].obs_ind - 2
        return obs[idx]

    def _get_idx(self, keys):
        """
        Returns the indices of the specified keys.
        Args:
            keys (list or str): List of keys or just one key which are
                used to get the indices from the observation space.

        Returns:
             np.array including the indices of the specified keys.

        """
        if type(keys) != list:
            assert type(keys) == str
            keys = [keys]

        entries = []
        for key in keys:
            entries.append(self.obs_helper.obs_idx_map[key])

        return np.concatenate(entries) - 2

    def _len_qpos_qvel(self):
        """
        Returns the lengths of the joint position vector and the joint velocity vector, including x and y.

        """

        keys = self.get_all_observation_keys()
        len_qpos = len([key for key in keys if key.startswith("q_")])
        len_qvel = len([key for key in keys if key.startswith("dq_")])

        return len_qpos, len_qvel

    def _has_fallen(self, obs, info, data, return_err_msg=False):
        """
        Checks if a model has fallen. This has to be implemented for each environment.
        
        Args:
            obs (np.array): Current observation.
            return_err_msg (bool): If True, an error message with violations is returned.

        Returns:
            True, if the model has fallen for the current observation, False otherwise.

        """
        
        raise NotImplementedError

    def _get_interpolate_map_params(self):
        """
        Returns all parameters needed to do the interpolation mapping for the respective environment.

        """

        pass

    def _get_interpolate_remap_params(self):
        """
        Returns all parameters needed to do the interpolation remapping for the respective environment.

        """

        pass

    def _check_reset_configuration(self):

        if not self.trajectories and self._random_start:
            raise ValueError("Random start not possible without trajectory data.")
        elif not self.trajectories and self._use_fixed_start:
            raise ValueError("Setting an initial start is not possible without trajectory data.")
        elif self._use_fixed_start and self._random_start:
            raise ValueError("Either use a random start or set a fixed initial start, but not both.")

    @staticmethod
    def _get_grf_size():
        """
        Returns the size of the ground force vector.

        """

        return 12

    @classmethod
    def sample_from_trajectories(cls, key, trajectories, traj_no, subtraj_step_no):
        key, subkey = jax.random.split(key)

        n_trajectories = cls.n_trajectories(trajectories)
        length_trajectory = cls.len_trajectory(trajectories)

        new_sample_idx = jax.random.randint(key, (2,),
                                            minval=jnp.array([0, n_trajectories]),
                                            maxval=jnp.array([0, length_trajectory]))
        new_traj_no = new_sample_idx[0]
        new_subtraj_step_no = new_sample_idx[1]

        return key, jnp.ravel(trajectories[:, new_traj_no, new_subtraj_step_no]), new_traj_no, new_subtraj_step_no

    @staticmethod
    def increment_traj_counter(trajectories, traj_no, subtraj_step_no):
        n_trajectories = LocoEnv.n_trajectories(trajectories)
        length_trajectory = LocoEnv.len_trajectory(trajectories)

        subtraj_step_no += 1

        # set to zero once exceeded
        next_subtraj_step_no = jnp.mod(subtraj_step_no, length_trajectory)

        # check whether to go to the next trajectory
        next_traj_no = jax.lax.cond(next_subtraj_step_no == 0, lambda t, nt: jnp.mod(t+1, nt),
                                    lambda t, nt: t, traj_no, n_trajectories)

        return next_traj_no, next_subtraj_step_no

    @staticmethod
    @jax.jit
    def get_traj_next_sample(trajectories, traj_no, subtraj_step_no):
        next_traj_no, next_subtraj_step_no = LocoEnv.increment_traj_counter(trajectories, traj_no, subtraj_step_no)
        return jnp.ravel(trajectories[:, next_traj_no, next_subtraj_step_no]), next_traj_no, next_subtraj_step_no

    @staticmethod
    @jax.jit
    def get_traj_current_sample(trajectories, traj_no, subtraj_step_no):
        return jnp.ravel(trajectories[:, traj_no, subtraj_step_no])

    @staticmethod
    def dim_obs_trajectory(trajectories):
        return jnp.shape(trajectories)[0]

    @staticmethod
    def n_trajectories(trajectories):
        return jnp.shape(trajectories)[1]

    @staticmethod
    def len_trajectory(trajectories):
        return jnp.shape(trajectories)[2]

    @staticmethod
    def list_registered_loco_mujoco():
        """
        List registered loco_mujoco environments.

        Returns:
             The list of the registered loco_mujoco environments.

        """
        return list(LocoEnv._registered_envs.keys())

    @staticmethod
    def _interpolate_map(traj, **interpolate_map_params):
        """
        A mapping that is supposed to transform a trajectory into a space where interpolation is
        allowed. E.g., maps a rotation matrix to a set of angles. If this function is not
        overwritten, it just converts the list of np.arrays to a np.array.

        Args:
            traj (list): List of np.arrays containing each observations. Each np.array
                has the shape (n_trajectories, n_samples, (dim_observation)). If dim_observation
                is one the shape of the array is just (n_trajectories, n_samples).
            interpolate_map_params: Set of parameters needed by the individual environments.

        Returns:
            A np.array with shape (n_observations, n_trajectories, n_samples). dim_observation
            has to be one.

        """

        return np.array(traj)

    @staticmethod
    def _interpolate_remap(traj, **interpolate_remap_params):
        """
        The corresponding backwards transformation to _interpolation_map. If this function is
        not overwritten, it just converts the np.array to a list of np.arrays.

        Args:
            traj (np.array): Trajectory as np.array with shape (n_observations, n_trajectories, n_samples).
            dim_observation is one.
            interpolate_remap_params: Set of parameters needed by the individual environments.

        Returns:
            List of np.arrays containing each observations. Each np.array has the shape
            (n_trajectories, n_samples, (dim_observation)). If dim_observation
            is one the shape of the array is just (n_trajectories, n_samples).

        """

        return [obs for obs in traj]

    @staticmethod
    def _delete_from_xml_handle(xml_handle, joints_to_remove, motors_to_remove, equ_constraints):
        """
        Deletes certain joints, motors and equality constraints from a Mujoco XML handle.

        Args:
            xml_handle: Handle to Mujoco XML.
            joints_to_remove (list): List of joint names to remove.
            motors_to_remove (list): List of motor names to remove.
            equ_constraints (list): List of equality constraint names to remove.

        Returns:
            Modified Mujoco XML handle.

        """

        for j in joints_to_remove:
            j_handle = xml_handle.find("joint", j)
            j_handle.remove()
        for m in motors_to_remove:
            m_handle = xml_handle.find("actuator", m)
            m_handle.remove()
        for e in equ_constraints:
            e_handle = xml_handle.find("equality", e)
            e_handle.remove()

        return xml_handle

    @staticmethod
    def _save_xml_handle(xml_handle, tmp_dir_name, file_name="tmp_model.xml"):
        """
        Save the Mujoco XML handle to a file at tmp_dir_name. If tmp_dir_name is None,
        a temporary directory is created at /tmp.

        Args:
            xml_handle: Mujoco XML handle.
            tmp_dir_name (str): Path to temporary directory. If None, a
            temporary directory is created at /tmp.

        Returns:
            String of the save path.

        """

        if tmp_dir_name is not None:
            assert os.path.exists(tmp_dir_name), "specified directory (\"%s\") does not exist." % tmp_dir_name

        dir = mkdtemp(dir=tmp_dir_name)
        file_path = os.path.join(dir, file_name)

        # dump data
        mjcf.export_with_assets(xml_handle, dir, file_name)

        return file_path

    @staticmethod
    def raise_mjx_not_enabled_error(*args, **kwargs):
        return ValueError("Mjx not enabled in this environment")

    @classmethod
    def get_all_task_names(cls):
        """
        Returns a list of all available tasks in LocoMujoco.

        """

        task_names = []
        for e in cls.list_registered_loco_mujoco():
            env = cls._registered_envs[e]
            confs = env.valid_task_confs.get_all_combinations()
            for conf in confs:
                task_name = list(conf.values())
                task_name.insert(0, env.__name__, )
                task_name = ".".join(task_name)
                task_names.append(task_name)

        return task_names


class ValidTaskConf:

    """ Simple class that holds all valid configurations of an environment. """

    def __init__(self, tasks=None, modes=None, data_types=None, non_combinable=None):
        """

        Args:
            tasks (list): List of valid tasks.
            modes (list): List of valid modes.
            data_types (list): List of valid data_types.
            non_combinable (list): List of tuples ("task", "mode", "dataset_type"),
                which are NOT allowed to be combined. If one of them is None, it is neglected.

        """

        self.tasks = tasks
        self.modes = modes
        self.data_types = data_types
        self.non_combinable = non_combinable
        if non_combinable is not None:
            for nc in non_combinable:
                assert len(nc) == 3

    def get_all(self):
        return deepcopy(self.tasks), deepcopy(self.modes),\
               deepcopy(self.data_types), deepcopy(self.non_combinable)

    def get_all_combinations(self):
        """
        Returns all possible combinations of configurations.

        """

        confs = []

        if self.tasks is not None:
            tasks = self.tasks
        else:
            tasks = [None]
        if self.modes is not None:
            modes = self.modes
        else:
            modes = [None]
        if self.data_types is not None:
            data_types = self.data_types
        else:
            data_types = [None]

        for t, m, dt in product(tasks, modes, data_types):
            conf = dict()
            if t is not None:
                conf["task"] = t
            if m is not None:
                conf["mode"] = m
            if dt is not None:
                conf["data_type"] = dt

            # check for non-combinable
            if self.non_combinable is not None:
                for nc in self.non_combinable:
                    bad_t, bad_m, bad_dt = nc
                    if not((t == bad_t or bad_t is None) and
                           (m == bad_m or bad_m is None) and
                           (dt == bad_dt or bad_dt is None)):
                        confs.append(conf)
            else:
                confs.append(conf)

        return confs
