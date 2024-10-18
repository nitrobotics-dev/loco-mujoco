import os
import yaml
import warnings
from pathlib import Path
from copy import deepcopy
from tempfile import mkdtemp
from itertools import product
from functools import partial
from tqdm import tqdm

import jax.random
import jax.numpy as jnp
import numpy as np
from flax import struct
import mujoco
from dm_control import mjcf

import loco_mujoco
from loco_mujoco.core.mujoco_mjx import Mjx, MjxState, MjxAdditionalCarry
from loco_mujoco.core.utils import Box, VideoRecorder, Goal
from loco_mujoco.trajectory import TrajectoryHandler
from loco_mujoco.utils import Reward, DomainRandomizationHandler
from loco_mujoco.utils import info_property, RunningAveragedWindow, RunningAverageWindowState


@struct.dataclass
class TrajState:
    traj_no: int
    subtraj_step_no: int
    subtraj_step_no_init: int


@struct.dataclass
class LocoCarry(MjxAdditionalCarry):
    traj_state: TrajState
    running_avg_state: RunningAverageWindowState


class LocoEnv(Mjx):
    """
    Base class for all kinds of locomotion environments.

    """

    def __init__(self, xml_handles, action_spec, observation_spec, collision_groups=None, enable_mjx=False,
                 n_envs=1, gamma=0.99, horizon=1000, n_substeps=10,  reward_type="no_reward", reward_params=None,
                 goal_type="NoGoal", goal_params=None, traj_params=None, random_start=True, fixed_start_conf=None,
                 timestep=0.001, use_foot_forces=False, default_camera_mode="follow", use_absorbing_states=True,
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
            enable_mjx (bool): Flag specifying whether Mjx simulation is enabled or not.
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
            traj_params (dict): Dictionary of parameters to construct trajectories.
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

        # add sites for goal to xml_handle
        xml_handles, goal = self._setup_goal(xml_handles, goal_type, goal_params)

        if enable_mjx:
            # call parent (Mjx) constructor
            super(LocoEnv, self).__init__(n_envs, xml_file=xml_handles, actuation_spec=action_spec,
                                          observation_spec=observation_spec, goal=goal, gamma=gamma,
                                          horizon=horizon, n_substeps=n_substeps,
                                          n_intermediate_steps=n_intermediate_steps,
                                          timestep=timestep, collision_groups=collision_groups,
                                          default_camera_mode=default_camera_mode,
                                          model_option_conf=model_option_conf, **viewer_params)
        else:
            assert n_envs == 1, "Mjx not enabled, setting the number of environments > 1 is not allowed."
            # call grandparent constructor (Mujoco (CPU) environment)
            super(Mjx, self).__init__(xml_file=xml_handles, actuation_spec=action_spec,
                                      observation_spec=observation_spec, goal=goal, gamma=gamma,
                                      horizon=horizon, n_substeps=n_substeps, n_intermediate_steps=n_intermediate_steps,
                                      timestep=timestep, collision_groups=collision_groups,
                                      default_camera_mode=default_camera_mode,
                                      model_option_conf=model_option_conf, **viewer_params)

        # specify reward function
        self._reward_function = self._setup_reward(reward_type, reward_params)

        # optionally use foot forces in the observation space
        self._use_foot_forces = use_foot_forces

        # todo: delete this function once the grf are setup and -2 is not in observation
        self.info.observation_space = Box(*self._get_observation_space())

        # the action space is supposed to be between -1 and 1, so we normalize it
        self._scale_action_space()

        # setup a running average window for the mean ground forces
        self.mean_grf = self._setup_ground_force_statistics()

        # dataset dummy
        self._dataset = None

        if traj_params:
            self.th = None
            self.load_trajectory(traj_params)
            self._trajectory_loaded = True
            self._jax_trajectory = self.th.get_jax_trajectory()
        else:
            self.th = None
            self._trajectory_loaded = True

        self._random_start = random_start
        self._fixed_start_conf = fixed_start_conf
        self._use_fixed_start = True if fixed_start_conf is not None else False

        self._use_absorbing_states = use_absorbing_states

    def set_actuation_spec(self, actuation_spec):
        """
        Sets the actuation of the environment to overwrite the default one.

        Args:
            actuation_spec (list): A list of actuator names.

        """
        super().set_actuation_spec(actuation_spec)

        # the action space is supposed to be between -1 and 1, so we normalize it
        self._scale_action_space()

    def load_trajectory(self, traj_params, warn=True):
        """
        Loads trajectories. If there were trajectories loaded already, this function overrides the latter.

        Args:
            traj_params (dict): Dictionary of parameters needed to load trajectories.
            warn (bool): If True, a warning will be raised if the
                trajectory ranges are violated.

        """

        if self.th is not None:
            warnings.warn("New trajectories loaded, which overrides the old ones.", RuntimeWarning)

        self.th = TrajectoryHandler(model=self._model, warn=warn, **traj_params)

        # setup trajectory information in observation_dict, goal and reward if needed
        for obs_entry in self.obs_container.entries():
            obs_entry.init_from_traj(self.th)
        self._goal.init_from_traj(self.th)
        self._reward_function.init_from_traj(self.th)

    def _scale_action_space(self):
        """
        Scale the action space to be between -1 and 1. The action mean and delta are used to rescale the actions
        before sending them back to the environment.

        """
        low, high = self.info.action_space.low.copy(), self.info.action_space.high.copy()
        self.norm_act_mean = (high + low) / 2.0
        self.norm_act_delta = (high - low) / 2.0
        self.info.action_space.low[:] = -1.0
        self.info.action_space.high[:] = 1.0

    def _reward(self, state, action, next_state, absorbing, info, model, data, carry):
        """
        Calls the reward function of the environment.

        """
        return self._reward_function(state, action, next_state, absorbing, info, model, data,
                                     carry, np, self.th.traj_data)

    @partial(jax.jit, static_argnums=(0, 6))
    def _mjx_reward(self, state, action, next_state, absorbing, info, model, data, carry):
        """
        Calls the reward function of the environment.

        """
        return self._reward_function(state, action, next_state, absorbing, info, model, data,
                                     carry, jnp, self.th.traj_data)

    def setup(self, data, key):
        """
        Function to set up the initial state of the simulation. Initialization can be done either
        randomly, from a certain initial, or from the default initial state of the model.

        Args:
            obs (np.array): Observation to initialize the environment from;

        """

        if self.th is not None:
            if self._random_start:
                sample = self.th.reset_trajectory()
            elif self._init_step_no:
                traj_len = self.th.trajectory_length
                n_traj = self.th.n_trajectories
                assert self._init_step_no <= traj_len * n_traj
                substep_no = int(self._init_step_no % traj_len)
                traj_no = int(self._init_step_no / traj_len)
                sample = self.th.reset_trajectory(substep_no, traj_no)
            else:
                # sample random trajectory and use the first sample
                sample = self.th.reset_trajectory(substep_no=0)

            self.set_sim_state(sample)

    def _is_absorbing(self, obs, info, data, carry):
        """
        Checks if an observation is an absorbing state or not.

        Args:
            obs (np.array): Current observation;

        Returns:
            True, if the observation is an absorbing state; otherwise False;

        """
        return self._has_fallen(obs, info, data) if self._use_absorbing_states else False

    def _mjx_is_absorbing(self, obs, info, data, carry):
        return jax.lax.cond(self._use_absorbing_states, lambda o, i, d: self._mjx_has_fallen(o, i, d),
                            lambda o, i, d: jnp.array(False), obs, info, data)

    def _mjx_is_done(self, obs, absorbing, info, data, carry):
        done = super()._mjx_is_done(obs, absorbing, info, data, carry)

        if self._goal.requires_trajectory or self._reward_function.requires_trajectory:
            # either the goal or the reward function requires the trajectory at each step, so we need to check
            # if the end of the trajectory is reached, if so, we set done to True
            traj_state = carry.traj_state
            len_traj = self.len_trajectory(self.th.traj_data, traj_state.traj_no)
            reached_end_of_traj = jax.lax.cond(jnp.greater_equal(traj_state.subtraj_step_no, len_traj - 1),
                                               lambda: True, lambda: False)
            done = jnp.logical_or(done, reached_end_of_traj)

        return done

    def _is_done(self, obs, absorbing, info, data, carry):
        done = super()._is_done(obs, absorbing, info, data, carry)

        if self._goal.requires_trajectory or self._reward_function.requires_trajectory:
            # either the goal or the reward function requires the trajectory at each step, so we need to check
            # if the end of the trajectory is reached, if so, we set done to True
            traj_state = carry.traj_state
            if traj_state.subtraj_step_no >= self.th.len_trajectory(traj_state.traj_no) - 1:
                done != True
            else:
                done != False

        return done

    def _simulation_post_step(self, data, carry):
        """
        Update trajectory state and goal in Mujoco data structure if needed.

        """
        carry = self._update_trajectory_state(carry)
        data = self._goal.set_data(data, backend=np, traj_data=self.th.traj_data, traj_state=carry.traj_state)

        # if self._use_foot_forces:
        #     grf = self._get_ground_forces()
        #     self.mean_grf.update_state(jnp.array(grf), carry.running_avg_state)

        return data, carry

    def _mjx_simulation_post_step(self, data, carry):
        """
        Update trajectory state and goal in Mujoco data structure if needed.

        """
        carry = self._update_trajectory_state(carry)
        data = self._goal.set_data(data, backend=jnp, traj_data=self.th.traj_data, traj_state=carry.traj_state)
        return data, carry

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
        idx = self.obs_container[key].obs_ind
        idx = [i-2 for i in idx]

        return idx

    def create_dataset(self, rng_key=None):
        """
        Creates a dataset from the specified trajectories.

        Args:
            rng_key (jax.random.PRNGKey): Random key to use for creating the dataset.

        Returns:
            Dictionary containing states, next_states and absorbing flags. For the states the shape is
            (N_traj x N_samples_per_traj, dim_state), while the absorbing flag has the shape is
            (N_traj x N_samples_per_traj). For perfect and preference datasets, the actions are also provided.

        """

        if self._dataset is None:

            if self.th is not None:

                # save the original starting configuration
                orig_random_start = self._random_start
                orig_fix = self._use_fixed_start
                orig_fix_conf = self._fixed_start_conf
                orig_traj_data = deepcopy(self.th.traj_data)

                # set it to a fixed start configuration
                self._random_start = False
                self._use_fixed_start = True
                self._fixed_start_conf = (0, 0)

                # setup containers for the dataset
                all_states, all_next_states, all_lasts = [], [], []

                if rng_key is None:
                    rng_key = jax.random.key(0)

                for i in tqdm(range(self.n_trajectories(self.th.traj_data)), desc="Creating Dataset"):

                    # set configuration to the first state of the current trajectory
                    self._fixed_start_conf = (i, 0)

                    self.th.traj_data = self.th.traj_data.to_numpy()
                    traj_data_single = self.th.traj_data.get(i, 0, np)
                    carry = self._init_additional_carry(rng_key, traj_data_single)
                    traj_data_single = self._reset_init_data(traj_data_single, carry)

                    states = [self._create_observation(traj_data_single, carry)]
                    for j in tqdm(range(1, self.len_trajectory(self.th.traj_data, i)),
                                  desc=f"Converted Samples in Trajectory {i}", leave=False):
                        traj_data_single = self.th.traj_data.get(i, j, np)
                        states.append(self._create_observation(traj_data_single, carry))
                        traj_data_single, carry = self._simulation_post_step(traj_data_single, carry)

                        # check if the current state is an absorbing state
                        has_fallen, msg = self._has_fallen(states[-1], None, traj_data_single,
                                                           return_err_msg=True)
                        if has_fallen:
                            err_msg = "Some of the states in the created dataset are terminal states. " \
                                      "This should not happen.\n\nViolations:\n"
                            err_msg += msg
                            raise ValueError(err_msg)

                    states = np.vstack(states)
                    all_states.append(states[:-1])
                    all_next_states.append(states[1:])
                    lasts = np.zeros(states.shape[0]-1)
                    lasts[-1] = 1
                    all_lasts.append(lasts)

                all_states = np.concatenate(all_states)
                all_next_states = np.concatenate(all_next_states)
                all_lasts = np.concatenate(all_lasts)
                all_absorbing = np.zeros_like(all_lasts)    # assume no absorbing states

                dataset = dict(states=all_states, next_states=all_next_states, absorbing=all_absorbing, last=all_lasts)

                # reset the original configuration
                self._random_start = orig_random_start
                self._use_fixed_start = orig_fix
                self._fixed_start_conf = orig_fix_conf
                self.th.traj_data = orig_traj_data

            else:
                raise ValueError("No trajectory was passed to the environment. "
                                 "To create a dataset pass a trajectory first.")

            self._dataset = deepcopy(dataset)

            return dataset
        else:
            return deepcopy(self._dataset)

    def play_trajectory(self, n_episodes=None, n_steps_per_episode=None, render=True,
                        record=False, recorder_params=None, callback_class=None, key=None):
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

        assert self.th is not None

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
        traj_data_sample = self.get_traj_current_sample(self.th.traj_data, traj_no, subtraj_step_no)
        self._set_sim_state_from_traj_data(self._data, traj_data_sample)

        if render:
            frame = self.render(record)
        else:
            frame = None

        if record:
            recorder(frame)

        highest_int = np.iinfo(np.int32).max
        if n_steps_per_episode is None:
            n_steps_per_episode = self.len_trajectory(self.th.traj_data, traj_no)
        if n_episodes is None:
            n_episodes = highest_int
        for i in range(n_episodes):
            for j in tqdm(range(n_steps_per_episode)):

                if callback_class is None:
                    self._set_sim_state_from_traj_data(self._data, traj_data_sample)
                    self._simulation_pre_step(self._data, self._additional_carry)
                    mujoco.mj_forward(self._model, self._data)
                    self._simulation_post_step(self._data, self._additional_carry)
                else:
                    callback_class(self, self._model, self._data, traj_data_sample, self._additional_carry)

                traj_data_sample, traj_no, subtraj_step_no = self.get_traj_next_sample(self.th.traj_data,
                                                                             traj_no, subtraj_step_no)

                obs = self._create_observation(self._data, self._additional_carry)

                if self._has_fallen(obs, {}, self._data):
                    pass
                    #print("Has fallen!")

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

        assert self.th is not None

        if record:
            assert render
            fps = 1/self.dt
            recorder = VideoRecorder(fps=fps, **recorder_params) if recorder_params is not None else\
                VideoRecorder(fps=fps)
        else:
            recorder = None

        self.reset()
        sample = self.th.get_current_sample()
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

                sample = self.th.get_next_sample()
                if sample is None:
                    self.reset()
                    sample = self.th.get_current_sample()
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
            grf_low, grf_high = (-np.ones((self.grf_size,)) * np.inf,
                                 np.ones((self.grf_size,)) * np.inf)
            return (np.concatenate([sim_low, grf_low]),
                    np.concatenate([sim_high, grf_high]))
        else:
            return sim_low, sim_high

    def _create_observation(self, data, carry):
        obs = super()._create_observation(data, carry)

        if self._use_foot_forces:
            obs = np.concatenate([obs[2:],
                                  self.mean_grf.mean / 1000.,
                                  ]).flatten()
        else:
            obs = np.concatenate([obs[2:],
                                  ]).flatten()

        return obs

    def _mjx_create_observation(self, data, carry):
        obs = super()._mjx_create_observation(data, carry)

        # remove the first two entries and add foot forces if needed
        # todo: foot forces not added yet for mjx
        obs = jax.lax.cond(self._use_foot_forces, lambda o: obs[2:].flatten(),
                           lambda o: obs[2:].flatten(), obs)
        return obs

    def reset(self, key):
        key, subkey = jax.random.split(key)
        obs = super().reset(key)
        carry = self._additional_carry

        # some sanity checks
        self._check_reset_configuration()

        if self._random_start or self._use_fixed_start:
            # init simulation from trajectory state (traj state has been reset in init_additional_carry)
            traj_sample = self._get_init_state_from_trajectory(carry.traj_state)      # get sample from trajectory state
            self._set_sim_state_from_traj_data(self._data, traj_sample)

        self._obs = self._create_observation(self._data, carry)

        # todo: think whether a reset of the reward function is needed in future, right now it is not.
        # self._reward_function.reset(self._data, carry, np, self._jax_trajectory)
        return self._obs

    def mjx_reset(self, key):
        key, subkey = jax.random.split(key)
        mjx_state = super().mjx_reset(key)
        carry = mjx_state.additional_carry

        # some sanity checks
        jax.debug.callback(self._check_reset_configuration)

        if self._random_start or self._use_fixed_start:
            # init simulation from trajectory state (traj state has been reset in init_additional_carry)
            traj_data_sample = self._get_init_state_from_trajectory(carry.traj_state)
            data = self._mjx_set_sim_state_from_traj_data(mjx_state.data, traj_data_sample)
            mjx_state = mjx_state.replace(data=data, observation=self._mjx_create_observation(data, carry),
                                          additional_carry=carry)

        # todo: think whether a reset of the reward function is needed in future, right now it is not.
        # self._reward_function.reset(data, carry, jnp, self._jax_trajectory)

        return mjx_state

    def _mjx_reset_in_step(self, state: MjxState):
        def where_done(x, y):
            done = state.done
            return jnp.where(done, x, y)

        # reset trajectory state
        carry = state.additional_carry
        key = carry.key
        key, subkey = jax.random.split(key)
        traj_state = self._reset_trajectory_state(subkey)

        if self._random_start or self._use_fixed_start:
            # init simulation from trajectory state
            traj_data_sample = self._get_init_state_from_trajectory(traj_state)      # get sample from trajectory state
            data = jax.tree.map(where_done,
                                self._mjx_set_sim_state_from_traj_data(state.data, traj_data_sample),
                                state.data)

        else:
            # init simulation from default state
            data = jax.tree.map(where_done, state.additional_carry.first_data, state.data)

        final_obs = where_done(state.observation, jnp.ones_like(state.observation))
        cur_step_in_episode = where_done(1, carry.cur_step_in_episode + 1)
        carry = carry.replace(key=key, cur_step_in_episode=cur_step_in_episode,
                              traj_state=traj_state, final_observation=final_obs)
        data = self._mjx_reset_init_data(data, carry)
        new_obs = self._mjx_create_observation(data, carry)

        return state.replace(data=data, observation=new_obs, additional_carry=carry)

    @partial(jax.jit, static_argnums=(0,))
    def _reset_trajectory_state(self, key):

        if self._random_start:
            k1, k2 = jax.random.split(key)
            traj_idx = jax.random.randint(k1, shape=(1,), minval=0, maxval=self.n_trajectories(self.th.traj_data))
            subtraj_step_idx = jax.random.randint(k2, shape=(1,), minval=0, maxval=self.th.len_trajectory(traj_idx))
            idx = [traj_idx[0], subtraj_step_idx[0]]

        elif self._use_fixed_start:
            idx = self._fixed_start_conf
        else:
            idx = [0, 0]

        new_traj_no, new_subtraj_step_no = idx
        new_subtraj_step_no_init = new_subtraj_step_no
        traj_state = TrajState(new_traj_no, new_subtraj_step_no, new_subtraj_step_no_init)
        return traj_state

    @partial(jax.jit, static_argnums=(0,))
    def _get_init_state_from_trajectory(self, traj_state):
        traj_data = self.get_traj_current_sample(self.th.traj_data, traj_state.traj_no, traj_state.subtraj_step_no)
        traj_data = traj_data.replace(qpos=traj_data.qpos.at[0:2].set(0.0))
        return traj_data

    def _update_trajectory_state(self, carry):
        traj_state = carry.traj_state
        next_traj_no, next_subtraj_step_no = self.increment_traj_counter(self.th.traj_data,
                                                                         traj_state.traj_no,
                                                                         traj_state.subtraj_step_no)
        traj_state = traj_state.replace(traj_no=next_traj_no, subtraj_step_no=next_subtraj_step_no)
        return carry.replace(traj_state=traj_state)

    def _preprocess_action(self, action, data, carry):
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

    def _mjx_preprocess_action(self, action, data, carry):
        return self._preprocess_action_compat(action, data)

    def _preprocess_action_compat(self, action, data):
        """
        Rescale the action from [-1, 1] to the desired action space.
        """
        unnormalized_action = ((action * self.norm_act_delta) + self.norm_act_mean)
        return unnormalized_action

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

    def _init_additional_carry(self, key, data):

        key, subkey = jax.random.split(key)

        # reset trajectory state
        traj_state = self._reset_trajectory_state(subkey)

        # reset running average window for ground forces
        running_avg_state = RunningAverageWindowState(storage=jnp.zeros((self._n_substeps,
                                                                        self.grf_size)))

        # create additional carry
        carry = LocoCarry(key=key,
                          traj_state=traj_state,
                          running_avg_state=running_avg_state,
                          cur_step_in_episode=1,
                          first_data=data,
                          final_info={},
                          final_observation=jnp.zeros(self.info.observation_space.shape)
                          )

        return carry

    def _reset_init_data(self, data, carry):
        data = self._goal.set_data(data, backend=np, traj_data=self.th.traj_data, traj_state=carry.traj_state)
        return data

    def _mjx_reset_init_data(self, data, carry):
        data = self._goal.set_data(data, backend=jnp, traj_data=self.th.traj_data, traj_state=carry.traj_state)
        return data

    def _setup_ground_force_statistics(self):
        """
        Returns a running average method for the mean ground forces.  By default, 4 ground force sensors are used.
        Environments that use more or less have to override this function.

        """

        mean_grf = RunningAveragedWindow(shape=(self.grf_size,), window_size=self._n_intermediate_steps)

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

    def _setup_goal(self, xml_handle, goal_type, goal_params):

        # collect all info properties of the env (dict all @info_properties decorated function returns)
        info_props = self._get_all_info_properties()

        # get the goal
        goal_cls = Goal.registered[goal_type]
        goal = goal_cls(info_props=info_props) if goal_params is None \
            else goal_cls(info_props=info_props, **goal_params)

        # apply the modification to the xml needed
        xml_handle = goal.apply_xml_modifications(xml_handle, self.root_body_name)

        return xml_handle, goal

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

    # def _get_goal(self, goal_type, goal_params):
    #     """
    #     Constructs a goal function.
    #
    #     Args:
    #         goal_type (string): Name of the goal.
    #         goal_params (dict): Parameters of the goal function.
    #
    #     Returns:
    #         Goal function.
    #
    #     """
    #     if goal_type == "NoGoal" or goal_type is None:
    #         goal = NoGoal()
    #     elif goal_type == "goal_arrow":
    #         data_goal_site_ind = [value.data_type_ind for value in self._goal_dict.values()]
    #         goal = GoalTrajArrow(data_goal_site_ind)
    #     elif goal_type == "goal_traj":
    #         # todo: access the correct goal_params
    #         goal = GoalTrajArrow(**goal_params)
    #     else:
    #         raise NotImplementedError("The specified goal is not supported: %s" % goal_type)
    #
    #     return goal

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

    def _get_from_obs(self, obs, key):
        """
        Returns a part of the observation based on the specified keys.

        Args:
            obs (np.array or jnp.array): Observation array.
            key str: Key which are used to extract entries from the observation.
            backend: Backend to use.

        Returns:
            np or jnp array including the parts of the original observation whose
            keys were specified.

        """

        idx = self.obs_container[key].obs_ind - 2
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

        if not self.th and self._random_start:
            raise ValueError("Random start not possible without trajectory data.")
        elif not self.th and self._use_fixed_start:
            raise ValueError("Setting an initial start is not possible without trajectory data.")
        elif self._use_fixed_start and self._random_start:
            raise ValueError("Either use a random start or set a fixed initial start, but not both.")

    @staticmethod
    def generate(env_cls, task="walk", dataset_type="real", debug=False,
                 clip_trajectory_to_joint_ranges=False, **kwargs):
        """
        Returns an environment corresponding to the specified task.

        Args:
            env_cls: Class of the chosen environment.
            task (str): Main task to solve. Either "walk" or "carry". The latter is walking while carrying
            an unknown weight, which makes the task partially observable.
            dataset_type (str): "real" or "perfect". "real" uses real motion capture data as the
            reference trajectory. This data does not perfectly match the kinematics
            and dynamics of this environment, hence it is more challenging. "perfect" uses
            a perfect dataset.
            debug (bool): If True, the smaller test datasets are used for debugging purposes.
            clip_trajectory_to_joint_ranges (bool): If True, trajectory is clipped to joint ranges.

        Returns:
            An environment of specified class and task.

        """

        # load correct task configuration
        config_key = ".".join((env_cls.__name__, task, dataset_type))
        task_config, dataset_path = env_cls.get_task_config_and_dataset_path(config_key, **kwargs)

        # create the environment
        env = env_cls(**task_config)

        # load the trajectory
        traj_params = env.get_traj_params(env, dataset_path, dataset_type, debug, clip_trajectory_to_joint_ranges)
        env.load_trajectory(traj_params, warn=False)

        return env

    @info_property
    def grf_size(self):
        """
        Returns the size of the ground force vector.

        """

        return 12

    @info_property
    def upper_body_xml_name(self):
        raise NotImplementedError(f"Please implement the upper_body_xml_name property "
                                  f"in the {type(self).__name__} environment.")

    @staticmethod
    def get_task_config_and_dataset_path(config_key, **kwargs):

        def load_all_yaml_at_path(path):
            all_files = os.listdir(path)
            yaml_files = [path  / f for f in all_files if f.endswith(".yaml")]
            all_configs = dict()
            for config_file_path in yaml_files:
                with open(config_file_path, 'r') as file:
                    one_config = yaml.safe_load(file)
                    assert list(one_config.keys()) not in list(all_configs.keys())
                    all_configs |= one_config
            return all_configs

        config_file_path = Path(loco_mujoco.__file__).resolve().parent / "tasks/"

        # load all configurations
        all_configs = load_all_yaml_at_path(config_file_path)

        # get task-specific configuration
        try:
            task_config = all_configs[config_key]
        except KeyError:
            raise KeyError("The specified task configuration could not be found: %s" % config_key)

        if "reward_type" in kwargs.keys() and "reward_params" not in kwargs.keys():
            try:
                del task_config["reward_params"]
            except:
                pass

        # overwrite the task_config with the kwargs
        for key, value in task_config.items():
            if key in kwargs.keys():
                task_config[key] = kwargs[key]
                del kwargs[key]  # delete from kwargs to avoid passing a kwarg twice

        # add rest of kwargs to task_config
        task_config |= kwargs

        # get the dataset path and delete it from the task_config
        dataset_path = task_config["dataset_path"]
        del task_config["dataset_path"]

        return task_config, dataset_path

    @staticmethod
    def get_traj_params(mdp, path, dataset_type, debug, clip_trajectory_to_joint_ranges):

        # Load the trajectory
        env_freq = 1 / mdp._timestep  # hz
        desired_contr_freq = 1 / mdp.dt  # hz
        n_substeps = env_freq // desired_contr_freq

        if dataset_type == "real":
            traj_data_freq = 500  # hz
            use_mini_dataset = not os.path.exists(Path(loco_mujoco.__file__).resolve().parent / path)
            if debug or use_mini_dataset:
                if use_mini_dataset:
                    warnings.warn("Datasets not found, falling back to test datasets. Please download and install "
                                  "the datasets to use this environment for imitation learning!")
                path = path.split("/")
                path.insert(3, "mini_datasets")
                path = "/".join(path)

            path = Path(loco_mujoco.__file__).resolve().parent / path
            traj_params = dict(traj_path=Path(loco_mujoco.__file__).resolve().parent / path,
                               control_dt=(1 / desired_contr_freq),
                               clip_trajectory_to_joint_ranges=clip_trajectory_to_joint_ranges)

        elif dataset_type == "perfect":
            # todo: this needs to be adapted to new traj_data
            traj_files = mdp.load_dataset_and_get_traj_files(path, traj_data_freq)
            traj_params = dict(traj_files=traj_files,
                               control_dt=(1 / desired_contr_freq),
                               clip_trajectory_to_joint_ranges=clip_trajectory_to_joint_ranges)

        # elif dataset_type == "preference":
        #     traj_data_freq = 100  # hz
        #     infos = []
        #     all_paths = next(os.walk(Path(loco_mujoco.__file__).resolve().parent / path), (None, None, []))[2]
        #     for i, p in enumerate(all_paths):
        #         traj_files = mdp.load_dataset_and_get_traj_files(path + p, traj_data_freq)
        #         if i == 0:
        #             all_traj_files = traj_files
        #         else:
        #             for key in traj_files.keys():
        #                 if key == "split_points":
        #                     all_traj_files[key] = np.concatenate([all_traj_files[key],
        #                                                           traj_files[key][1:] + all_traj_files[key][-1]])
        #                 else:
        #                     all_traj_files[key] = np.concatenate([all_traj_files[key], traj_files[key]])
        #         info = p.split(".")[0]
        #         info = info.split("_")[-2]
        #         n_traj = len(traj_files["split_points"]) - 1
        #         infos += [info] * n_traj
        #
        #     traj_params = dict(traj_files=all_traj_files,
        #                        traj_dt=(1 / traj_data_freq),
        #                        traj_info = infos,
        #                        control_dt=(1 / desired_contr_freq),
        #                        clip_trajectory_to_joint_ranges=clip_trajectory_to_joint_ranges)

        return traj_params

    # @classmethod
    # def sample_from_trajectories(cls, key, trajectories, traj_no, subtraj_step_no):
    #     key, subkey = jax.random.split(key)
    #
    #     n_trajectories = cls.n_trajectories(trajectories)
    #     length_trajectory = cls.len_trajectory(trajectories, traj_no)
    #
    #     new_sample_idx = jax.random.randint(key, (2,),
    #                                         minval=jnp.array([0, n_trajectories]),
    #                                         maxval=jnp.array([0, length_trajectory]))
    #     new_traj_no = new_sample_idx[0]
    #     new_subtraj_step_no = new_sample_idx[1]
    #
    #     return key, jnp.ravel(trajectories[:, new_traj_no, new_subtraj_step_no]), new_traj_no, new_subtraj_step_no

    @staticmethod
    @jax.jit
    def increment_traj_counter(traj_data, traj_no, subtraj_step_no):
        n_trajectories = LocoEnv.n_trajectories(traj_data)
        length_trajectory = LocoEnv.len_trajectory(traj_data, traj_no)

        subtraj_step_no += 1

        # set to zero once exceeded
        next_subtraj_step_no = jnp.mod(subtraj_step_no, length_trajectory)

        # check whether to go to the next trajectory
        next_traj_no = jax.lax.cond(next_subtraj_step_no == 0, lambda t, nt: jnp.mod(t+1, nt),
                                    lambda t, nt: t, traj_no, n_trajectories)

        return next_traj_no, next_subtraj_step_no

    @staticmethod
    @jax.jit
    def get_traj_next_sample(traj_data, traj_no, subtraj_step_no):
        next_traj_no, next_subtraj_step_no = LocoEnv.increment_traj_counter(traj_data, traj_no, subtraj_step_no)
        return traj_data.get(next_traj_no, next_subtraj_step_no), next_traj_no, next_subtraj_step_no

    @staticmethod
    @jax.jit
    def get_traj_current_sample(traj_data, traj_no, subtraj_step_no):
        return traj_data.get(traj_no, subtraj_step_no)

    @staticmethod
    def n_trajectories(traj_data):
        return len(traj_data.split_points) - 1

    @staticmethod
    def len_trajectory(traj_data, traj_ind):
        return traj_data.split_points[traj_ind+1] - traj_data.split_points[traj_ind]

    @staticmethod
    def list_registered_loco_mujoco():
        """
        List registered loco_mujoco environments.

        Returns:
             The list of the registered loco_mujoco environments.

        """
        return list(LocoEnv.registered_envs.keys())

    @staticmethod
    def _interpolate_map(traj, **interpolate_map_params):
        """
        A mapping that is supposed to transform a trajectory into a space where interpolation is
        allowed. E.g., maps a rotation matrix to a set of angles. If this function is not
        overwritten, it just converts the list of np.arrays to a np.array.

        Args:
            traj (list): List of np.arrays containing each observation. Each np.array
                has the shape (n_trajectories, n_samples, dim_observation).
            interpolate_map_params: Set of parameters needed by the individual environments.

        Returns:
            A np.array with shape (n_observations, n_samples) and a list of shapes of the original
            array for backwards conversion.

        """
        orig_shape = [t.shape for t in traj]
        return np.concatenate(traj, axis=-1).T, orig_shape

    @staticmethod
    def _interpolate_remap(traj, orig_shape, **interpolate_remap_params):
        """
        The corresponding backwards transformation to _interpolation_map. If this function is
        not overwritten, it just converts the np.array to a list of np.arrays.

        Args:
            traj (np.array): Trajectory as np.array with shape (D, n_samples).
                D is the dimensionality of the concatenated elements at a certain time in the trajectory.
            orig_shape: List of shapes of the original arrays before concatenation.
                The shapes are (n_samples, dim_observation). Note that the sum of all dim_observation in the list
                equals D.
            interpolate_remap_params: Set of parameters needed by the individual environments.

        Returns:
            List of np.arrays containing each observation. Each np.array has the shape
            (n_trajectories, n_samples, (dim_observation)). If dim_observation
            is one the shape of the array is just (n_trajectories, n_samples).

        """
        out = []
        start_ind = 0
        for i, shape in enumerate(orig_shape):
            end_ind = start_ind + shape[1]
            out.append(traj[start_ind:end_ind, :].T)
            start_ind = end_ind

        return out

    @property
    def root_body_name(self):
        return "pelvis"

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
            env = cls.registered_envs[e]
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
                    if not ((t == bad_t or bad_t is None) and
                            (m == bad_m or bad_m is None) and
                            (dt == bad_dt or bad_dt is None)):
                        confs.append(conf)
            else:
                confs.append(conf)

        return confs
