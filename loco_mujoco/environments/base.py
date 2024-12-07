import os
import yaml
import warnings
from pathlib import Path
from copy import deepcopy
from tempfile import mkdtemp
from itertools import product
from functools import partial
from tqdm import tqdm
from dataclasses import replace

import jax.random
import jax.numpy as jnp
import numpy as np
from flax import struct
import mujoco
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as np_R

import loco_mujoco
from loco_mujoco.core.stateful_object import EmptyState
from loco_mujoco.core.observations import Goal
from loco_mujoco.core.mujoco_mjx import Mjx, MjxState, MjxAdditionalCarry
from loco_mujoco.core.utils import VideoRecorder, TerminalStateHandler, Reward
from loco_mujoco.trajectory import TrajectoryHandler
from loco_mujoco.utils import DomainRandomizationHandler
from loco_mujoco.utils import info_property, RunningAveragedWindow
from loco_mujoco.trajectory import Trajectory, TrajState, TrajectoryTransitions
from loco_mujoco.datasets.data_generation import convert_single_dataset_of_env, PATH_RAW_DATASET


@struct.dataclass
class LocoCarry(MjxAdditionalCarry):
    traj_state: TrajState


class LocoEnv(Mjx):
    """
    Base class for all kinds of locomotion environments.

    """

    def __init__(self, spec, action_spec, observation_spec, enable_mjx=False,
                 n_envs=1, gamma=0.99, horizon=1000, n_substeps=10, reward_type="NoReward", reward_params=None,
                 goal_type="NoGoal", goal_params=None, terminal_state_type=None,
                 terminal_state_params=None, traj_params=None, random_start=True, fixed_start_conf=None,
                 timestep=0.001, use_foot_forces=False, default_camera_mode="follow",
                 domain_randomization_config=None, parallel_dom_rand=True, N_worker_per_xml_dom_rand=4,
                 model_option_conf=None, **viewer_params):
        """
        Constructor.

        Args:
            spec (MjSpec): MuJoCo specification.
            actuation_spec (list): A list specifying the names of the joints
                which should be controllable by the agent. Can be left empty
                when all actuators should be used;
            observation_spec (list): A list containing the names of data that
                should be made available to the agent as an observation and
                their type (ObservationType). They are combined with a key,
                which is used to access the data. An entry in the list
                is given by: (key, name, type). The name can later be used
                to retrieve specific observations;
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

        if use_foot_forces:
            n_intermediate_steps = n_substeps
            n_substeps = 1
        else:
            n_intermediate_steps = 1

        if "geom_group_visualization_on_startup" not in viewer_params.keys():
            viewer_params["geom_group_visualization_on_startup"] = [0, 2]   # enable robot geom [0] and floor visual [2]

        if domain_randomization_config is not None:
            # todo: this is deprecated
            self._domain_rand = DomainRandomizationHandler(spec, domain_randomization_config, parallel_dom_rand,
                                                           N_worker_per_xml_dom_rand)
        else:
            self._domain_rand = None

        # add sites for goal to spec
        spec, goal = self._setup_goal(spec, goal_type, goal_params)

        if enable_mjx:
            # call parent (Mjx) constructor
            super(LocoEnv, self).__init__(n_envs, xml_file=spec, actuation_spec=action_spec,
                                          observation_spec=observation_spec, goal=goal, gamma=gamma,
                                          horizon=horizon, n_substeps=n_substeps,
                                          n_intermediate_steps=n_intermediate_steps,
                                          timestep=timestep,
                                          default_camera_mode=default_camera_mode,
                                          model_option_conf=model_option_conf, **viewer_params)
        else:
            assert n_envs == 1, "Mjx not enabled, setting the number of environments > 1 is not allowed."
            # call grandparent constructor (Mujoco (CPU) environment)
            super(Mjx, self).__init__(xml_file=spec, actuation_spec=action_spec,
                                      observation_spec=observation_spec, goal=goal, gamma=gamma,
                                      horizon=horizon, n_substeps=n_substeps, n_intermediate_steps=n_intermediate_steps,
                                      timestep=timestep,
                                      default_camera_mode=default_camera_mode,
                                      model_option_conf=model_option_conf, **viewer_params)

        # specify reward function
        self._reward_function = self._setup_reward(reward_type, reward_params)

        # optionally use foot forces in the observation space
        self._use_foot_forces = use_foot_forces

        # the action space is supposed to be between -1 and 1, so we normalize it
        self._scale_action_space()

        # setup terminal state handler
        if terminal_state_type is None:
            self._terminal_state_handler = TerminalStateHandler.make("RootPoseTrajTerminalStateHandler",
                                                                     self._model, self._get_all_info_properties())
        elif terminal_state_type is not None and terminal_state_params is None:
            self._terminal_state_handler = TerminalStateHandler.make(terminal_state_type,
                                                                     self._model, self._get_all_info_properties())
        else:
            self._terminal_state_handler = TerminalStateHandler.make(terminal_state_type,
                                                                     self._model, self._get_all_info_properties(),
                                                                     **terminal_state_params)

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

    def set_actuation_spec(self, actuation_spec):
        """
        Sets the actuation of the environment to overwrite the default one.

        Args:
            actuation_spec (list): A list of actuator names.

        """
        super().set_actuation_spec(actuation_spec)

        # the action space is supposed to be between -1 and 1, so we normalize it
        self._scale_action_space()

    def load_trajectory(self,  traj: Trajectory = None, traj_path=None, warn=True):
        """
        Loads trajectories. If there were trajectories loaded already, this function overrides the latter.

        Args:
            traj (Trajectory): Datastructure containing all trajectory files. If traj_path is specified, this
                should be None.
            traj_path (string): path with the trajectory for the model to follow. Should be a numpy zipped file (.npz)
                with a 'traj_data' array and possibly a 'split_points' array inside. The 'traj_data'
                should be in the shape (joints x observations). If traj_files is specified, this should be None.
            warn (bool): If True, a warning will be raised if the
                trajectory ranges are violated.

        """

        if self.th is not None:
            warnings.warn("New trajectories loaded, which overrides the old ones.", RuntimeWarning)

        if traj_path is not None:
            if not os.path.exists(traj_path):
                # trajectories for this environment do not exist yet, so convert and save them
                self.convert_dataset_for_env(traj_path)

        self.th = TrajectoryHandler(model=self._model, warn=warn, traj_path=traj_path,
                                    traj=traj, control_dt=self.dt, random_start=self._random_start,
                                    fixed_start_conf=self._fixed_start_conf)

        if self.th.traj.obs_container is not None:
            assert self.obs_container == self.th.traj.obs_container, \
                ("Observation containers of trajectory and environment do not match. \n"
                 "Please, either load a trajectory with the same observation container or "
                 "set the observation container of the environment to the one of the trajectory.")

        # setup trajectory information in observation_dict, goal and reward if needed
        for obs_entry in self.obs_container.entries():
            obs_entry.init_from_traj(self.th)
        self._goal.init_from_traj(self.th)
        self._reward_function.init_from_traj(self.th)
        self._terminal_state_handler.init_from_traj(self.th)

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
        return self._reward_function(state, action, next_state, absorbing, info, self, model, data, carry, np)

    @partial(jax.jit, static_argnums=(0, 6))
    def _mjx_reward(self, state, action, next_state, absorbing, info, model, data, carry):
        """
        Calls the reward function of the environment.

        """
        return self._reward_function(state, action, next_state, absorbing, info, self, model, data, carry, jnp)

    def _is_absorbing(self, obs, info, data, carry):
        """
        Checks if an observation is an absorbing state or not.

        Args:
            obs (np.array): Current observation;

        Returns:
            True, if the observation is an absorbing state; otherwise False;

        """
        return self._terminal_state_handler.is_absorbing(obs, info, data, carry)

    def _mjx_is_absorbing(self, obs, info, data, carry):
        return self._terminal_state_handler.mjx_is_absorbing(obs, info, data, carry)

    def _mjx_is_done(self, obs, absorbing, info, data, carry):
        done = super()._mjx_is_done(obs, absorbing, info, data, carry)

        if self._goal.requires_trajectory or self._reward_function.requires_trajectory:
            # either the goal or the reward function requires the trajectory at each step, so we need to check
            # if the end of the trajectory is reached, if so, we set done to True
            traj_state = carry.traj_state
            len_traj = self.th.len_trajectory(traj_state.traj_no)
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
        if self.th is not None:
            carry = self.th.update_state(self, self._model, data, carry, np)
        data = self._goal.set_data(self, self._model, data, carry, backend=np)

        return data, carry

    def _mjx_simulation_post_step(self, data, carry):
        """
        Update trajectory state and goal in Mujoco data structure if needed.

        """
        if self.th is not None:
            carry = self.th.update_state(self, self._model, data, carry, jnp)
        data = self._goal.set_data(self, self._model, data, carry, backend=jnp)

        return data, carry

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

        if self.th is not None:

            if self.th.traj.transitions is None:
                # save the original starting configuration
                orig_random_start = self._random_start
                orig_fix = self.th.use_fixed_start
                orig_fix_conf = self._fixed_start_conf
                orig_traj_data = deepcopy(self.th.traj.data)

                # set it to a fixed start configuration
                self._random_start = False
                self.th.use_fixed_start = True
                self._fixed_start_conf = (0, 0)

                # setup containers for the dataset
                all_observations, all_next_observations, all_dones = [], [], []

                if rng_key is None:
                    rng_key = jax.random.key(0)

                for i in tqdm(range(self.n_trajectories(self.th.traj.data)), desc="Creating Transition Dataset"):

                    # set configuration to the first state of the current trajectory
                    self._fixed_start_conf = (i, 0)

                    self.th.traj = replace(self.th.traj, data=self.th.traj.data.to_numpy())
                    traj_data_single = self.th.traj.data.get(i, 0, np)
                    carry = self._init_additional_carry(rng_key, traj_data_single, np)
                    traj_data_single = self._reset_init_data(traj_data_single, carry)

                    observations = [self._create_observation(self._model, traj_data_single, carry)]
                    for j in range(1, self.len_trajectory(self.th.traj.data, i)):
                        traj_data_single = self.th.traj.data.get(i, j, np)
                        observations.append(self._create_observation(self._model, traj_data_single, carry))
                        traj_data_single, carry = self._simulation_post_step(traj_data_single, carry)

                        # check if the current state is an absorbing state
                        has_fallen, msg = self._has_fallen(observations[-1], None, traj_data_single,
                                                           return_err_msg=True)
                        if has_fallen:
                            err_msg = "Some of the states in the created dataset are terminal states. " \
                                      "This should not happen.\n\nViolations:\n"
                            err_msg += msg
                            raise ValueError(err_msg)

                    observations = np.vstack(observations)
                    all_observations.append(observations[:-1])
                    all_next_observations.append(observations[1:])
                    dones = np.zeros(observations.shape[0]-1)
                    dones[-1] = 1
                    all_dones.append(dones)

                all_observations = np.concatenate(all_observations).astype(np.float32)
                all_next_observations = np.concatenate(all_next_observations).astype(np.float32)
                all_dones = np.concatenate(all_dones).astype(np.float32)
                all_absorbing = np.zeros_like(all_dones).astype(np.float32)    # assume no absorbing states

                # reset the original configuration
                self._random_start = orig_random_start
                self.th.use_fixed_start = orig_fix
                self._fixed_start_conf = orig_fix_conf
                self.th.traj = replace(self.th.traj, data=orig_traj_data)

                transitions = TrajectoryTransitions(jnp.array(all_observations),
                                                    jnp.array(all_next_observations),
                                                    jnp.array(all_absorbing),
                                                    jnp.array(all_dones))
                self.th.traj = replace(self.th.traj, transitions=transitions)

            return self.th.traj.transitions

        else:
            raise ValueError("No trajectory was passed to the environment. "
                             "To create a dataset pass a trajectory first.")

    def play_trajectory(self, n_episodes=None, n_steps_per_episode=None, from_velocity=False, render=True,
                        record=False, recorder_params=None, callback_class=None, key=None):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the trajectories at every step.

        Args:
            n_episodes (int): Number of episode to replay.
            n_steps_per_episode (int): Number of steps to replay per episode.
            from_velocity (bool): If True, the joint positions are calculated from the joint
                velocities in the trajectory.
            render (bool): If True, trajectory will be rendered.
            record (bool): If True, the rendered trajectory will be recorded.
            recorder_params (dict): Dictionary containing the recorder parameters.
            callback_class (class): Class to be called at each step of the simulation.
            key (jax.random.PRNGKey): Random key to use for the simulation.

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

        is_free_joint_qpos_quat, is_free_joint_qvel_rotvec = [], []
        for i in range(self._model.njnt):
            if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                is_free_joint_qpos_quat.extend([False, False, False, True, True, True, True])
                is_free_joint_qvel_rotvec.extend([False, False, False, True, True, True])
            else:
                is_free_joint_qpos_quat.append(False)
                is_free_joint_qvel_rotvec.append(False)

        is_free_joint_qpos_quat = np.array(is_free_joint_qpos_quat)
        is_free_joint_qvel_rotvec = np.array(is_free_joint_qvel_rotvec)

        key, subkey = jax.random.split(key)
        self.reset(subkey)
        traj_no = 0
        subtraj_step_no = 0
        traj_data_sample = self.get_traj_current_sample(self.th.traj.data, traj_no, subtraj_step_no)
        self._set_sim_state_from_traj_data(self._data, traj_data_sample)

        if render:
            frame = self.render(record)
        else:
            frame = None

        if record:
            recorder(frame)

        highest_int = np.iinfo(np.int32).max
        if n_steps_per_episode is None:
            n_steps_per_episode = self.len_trajectory(self.th.traj.data, traj_no)
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

                traj_data_sample, traj_no, subtraj_step_no = self.get_traj_next_sample(self.th.traj.data,
                                                                                       traj_no, subtraj_step_no)


                if subtraj_step_no == 0:
                    print("here")

                if from_velocity and subtraj_step_no != 0:
                    qpos = self._data.qpos
                    qvel = np.array(traj_data_sample.qvel)

                    qpos_quat = self._data.qpos[is_free_joint_qpos_quat]

                    # Integrate angular velocity using rotation vector approach
                    delta_q = np_R.from_rotvec(self.dt * qvel[is_free_joint_qvel_rotvec])

                    # Apply the incremental rotation to the current quaternion orientation
                    new_qpos = delta_q * np_R.from_quat(qpos_quat)

                    #new_qpos = np_R.from_euler("xyz", self.dt * qvel[is_free_joint_qvel_rotvec]) * np_R.from_quat(qpos_quat)
                    qpos_quat = new_qpos.as_quat()

                    # qpos_quat = np_R.from_euler("xyz", qpos_rotvec).as_quat()

                    # todo: implement for more than one free joint
                    assert len(qpos_quat) <= 4, "currently only one free joints per scene is supported for replay."

                    qpos[~is_free_joint_qpos_quat] = [qp + self.dt * qv for qp, qv in zip(self._data.qpos[~is_free_joint_qpos_quat],
                                                                                          qvel[~is_free_joint_qvel_rotvec])]
                    qpos[is_free_joint_qpos_quat] = qpos_quat
                    traj_data_sample = traj_data_sample.replace(qpos=jnp.array(qpos))

                obs = self._create_observation(self._model, self._data, self._additional_carry)

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
                                      record=False, recorder_params=None, callback_class=None, key=None):
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
            record (bool): If True, the rendered trajectory will be recorded.
            recorder_params (dict): Dictionary containing the recorder parameters.
            callback_class (class): Class to be called at each step of the simulation.
            key (jax.random.PRNGKey): Random key to use for the simulation.

        """
        warnings.warn(
            "play_trajectory_from_velocity() is deprecated and will be removed in future. "
            "Use play_trajectory() and set from_velocity=True instead.",
            category=DeprecationWarning,
            stacklevel=2
        )
        self.play_trajectory(n_episodes, n_steps_per_episode, True, render,
                             record, recorder_params, callback_class, key)

    def reset(self, key):
        if self.th is not None:
            self.th.to_numpy()
        key, subkey = jax.random.split(key)
        obs = super().reset(key)
        carry = self._additional_carry

        # reset data
        data = carry.first_data

        # reset trajectory state
        data, carry = self.th.reset_state(self, self._model, data, carry, jnp) if self.th is not None else (data, carry)

        if self.th and (self._random_start or self.th.use_fixed_start):
            assert self.th is not None, "If random_start or fixed_start is set to True, a trajectory has to be loaded."
            # init simulation from trajectory
            curr_traj_data = self.th.get_current_traj_data(carry, np)
            data = self._set_sim_state_from_traj_data(data, curr_traj_data)

        # apply general modifications on reset
        data = self._reset_init_data(data, carry)

        # reset all stateful entities
        data, carry = self.obs_container.reset_state(self, self._model, data, carry, jnp)

        self._obs, self._additional_carry = self._create_observation(self._model, data, carry)
        self._data = data

        return self._obs

    def mjx_reset(self, key):
        if self.th is not None:
            self.th.to_jax()
        key, subkey = jax.random.split(key)
        mjx_state = super().mjx_reset(key)
        carry = mjx_state.additional_carry

        # reset data
        data = carry.first_data

        # reset trajectory state
        data, carry = self.th.reset_state(self, self._model, data, carry, jnp) if self.th is not None else (data, carry)

        if self.th and (self._random_start or self.th.use_fixed_start):
            assert self.th is not None, "If random_start or fixed_start is set to True, a trajectory has to be loaded."
            # init simulation from trajectory
            curr_traj_data = self.th.get_current_traj_data(carry, jnp)
            data = self._mjx_set_sim_state_from_traj_data(data, curr_traj_data)

        # apply general modifications on reset
        data = self._mjx_reset_init_data(data, carry)

        # reset all stateful entities
        data, carry = self.obs_container.reset_state(self, self._model, data, carry, jnp)

        obs, carry = self._mjx_create_observation(self._model, data, carry)
        mjx_state = mjx_state.replace(data=data, observation=obs, additional_carry=carry)

        return mjx_state

    def _mjx_reset_in_step(self, state: MjxState):

        carry = state.additional_carry
        key = carry.key
        key, _k1 = jax.random.split(key)

        # reset data
        data = carry.first_data

        # reset trajectory state
        if self.th is not None:
            data, carry = self.th.reset_state(self, self._model, data, carry, jnp)

        if self.th and (self._random_start or self.th.use_fixed_start):
            assert self.th is not None, "If random_start or fixed_start is set to True, a trajectory has to be loaded."
            # init simulation from trajectory
            curr_traj_data = self.th.get_current_traj_data(carry, jnp)
            data = self._mjx_set_sim_state_from_traj_data(data, curr_traj_data)

        # apply general modifications on reset
        data = self._mjx_reset_init_data(data, carry)

        # reset carry
        carry = carry.replace(key=key,
                              cur_step_in_episode=1,
                              final_observation=state.observation,
                              final_info=state.info)

        # update all stateful entities
        data, carry = self.obs_container.reset_state(self, self._model, data, carry, jnp)

        # create new observation
        obs, carry = self._mjx_create_observation(self._model, data, carry)

        return state.replace(data=data, observation=obs, additional_carry=carry)

    @staticmethod
    def _set_sim_state_from_traj_data(data, traj_data):
        # set x and y to 0
        traj_data.qpos[0:2] = 0.0
        return Mjx._set_sim_state_from_traj_data(data, traj_data)

    @staticmethod
    def _mjx_set_sim_state_from_traj_data(data, traj_data):
        # set x and y to 0
        traj_data = traj_data.replace(qpos=traj_data.qpos.at[0:2].set(0.0))
        return Mjx._mjx_set_sim_state_from_traj_data(data, traj_data)

    @partial(jax.jit, static_argnums=(0,))
    def _reset_trajectory_state(self, key):

        # check that th is not None if random_start or fixed_start is set to true
        if self._random_start or self.th.use_fixed_start:
            assert self.th is not None, "If random_start or fixed_start is set to True, a trajectory has to be loaded."

        if self._random_start:
            k1, k2 = jax.random.split(key)
            traj_idx = jax.random.randint(k1, shape=(1,), minval=0, maxval=self.n_trajectories(self.th.traj.data))
            subtraj_step_idx = jax.random.randint(k2, shape=(1,), minval=0, maxval=self.th.len_trajectory(traj_idx))
            idx = [traj_idx[0], subtraj_step_idx[0]]

        elif self.th.use_fixed_start:
            idx = self._fixed_start_conf
        else:
            idx = [0, 0]

        new_traj_no, new_subtraj_step_no = idx
        new_subtraj_step_no_init = new_subtraj_step_no
        traj_state = TrajState(new_traj_no, new_subtraj_step_no, new_subtraj_step_no_init)
        return traj_state

    @partial(jax.jit, static_argnums=(0,))
    def _get_init_state_from_trajectory(self, traj_state):
        traj_data = self.get_traj_current_sample(self.th.traj.data, traj_state.traj_no, traj_state.subtraj_step_no)
        traj_data = traj_data.replace(qpos=traj_data.qpos.at[0:2].set(0.0))
        return traj_data

    def _update_trajectory_state(self, carry):
        traj_state = carry.traj_state
        next_traj_no, next_subtraj_step_no = self.increment_traj_counter(self.th.traj.data,
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

    def _init_additional_carry(self, key, model, data, backend):

        key, _k1, _k2, _k3 = jax.random.split(key, 4)

        # create additional carry
        carry = LocoCarry(key=key,
                          traj_state=self.th.init_state(self, _k1, model, data, backend) if self.th is not None else EmptyState(),
                          cur_step_in_episode=1,
                          first_data=data,
                          final_info={},
                          final_observation=backend.zeros(self.info.observation_space.shape),
                          observation_states=self.obs_container.init_state(self, _k2, model, data, backend),
                          reward_state=self._reward_function.init_state(self, _k3, model, data, backend),
                          )

        return carry

    def _reset_init_data(self, data, carry):
        data = self._goal.set_data(self, self._model, data, carry, backend=np)
        return data

    def _mjx_reset_init_data(self, data, carry):
        data = self._goal.set_data(self, self._model, data, carry, backend=jnp)
        return data

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
        spec = goal.apply_spec_modifications(spec, self.root_body_name)

        return spec, goal

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

        idx = self.obs_container[key].obs_ind
        return obs[idx]

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

    def _mjx_has_fallen(self, obs, info, data, carry):
        raise NotImplementedError

    def _modify_spec_for_mjx(self, spec: MjSpec):
        raise NotImplementedError

    @classmethod
    def generate(cls, task=None, dataset_type="real", debug=False,
                 clip_trajectory_to_joint_ranges=False, **kwargs):
        """
        Returns an environment corresponding to the specified task.

        Args:
            task (str): Main task to solve.
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
        config_key = ".".join((cls.__name__, task, dataset_type)) if task is not None\
            else ".".join((cls.__name__, "DEFAULT", dataset_type))
        task_config = cls.get_task_config(config_key, **kwargs)

        # create the environment
        env = cls(**task_config)

        # load the trajectory
        if task is not None:
            traj_path = env.get_traj_path(cls, dataset_type, task, debug)
            env.load_trajectory(traj_path=traj_path, warn=False)

        return env

    @classmethod
    def get_default_xml_file_path(cls):
        """
        Returns the default path to the xml file of the environment.
        """
        raise NotImplementedError(f"Please implement the default_xml_file_path property "
                                  f"in the {type(cls).__name__} environment.")

    @info_property
    def grf_size(self):
        """
        Returns the size of the ground force vector.

        """

        return 12

    @info_property
    def root_free_joint_xml_name(self):
        return "root"

    @info_property
    def upper_body_xml_name(self):
        raise NotImplementedError(f"Please implement the upper_body_xml_name property "
                                  f"in the {type(self).__name__} environment.")

    @info_property
    def root_height_healthy_range(self):
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        raise NotImplementedError(f"Please implement the root_height_healthy_range property "
                                  f"in the {type(self).__name__} environment.")

    @info_property
    def foot_geom_names(self):
        """
        Returns the names of the foot geometries.

        """
        # todo: raise NotImplementedError, once added to all envs
        raise []

    @staticmethod
    def _get_observation_specification(spec: MjSpec):
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            A list of observation types.
        """
        raise NotImplementedError(f"Please implement the _get_observation_specification method "
                                  f"in the {type(spec).__name__} environment.")

    @staticmethod
    def _get_action_specification(spec: MjSpec):
        """
        Getter for the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            A list of actuator names.

        """
        raise NotImplementedError(f"Please implement the _get_action_specification method "
                                  f"in the {type(spec).__name__} environment.")

    @staticmethod
    def get_task_config(config_key, **kwargs):

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
            if config_key in all_configs.keys():
                task_config = all_configs[config_key]
            else:
                # load default config
                def_config_key = config_key.split(".")
                def_config_key[1] = "DEFAULT"
                def_config_key = ".".join(def_config_key)
                task_config = all_configs[def_config_key]
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

        return task_config

    @staticmethod
    def get_traj_path(env_cls, dataset_type, task, debug):

        if dataset_type == "real":
            traj_path = str(env_cls.path_to_real_datasets() / (task+".npz"))
            if debug:
                traj_path = traj_path.split("/")
                traj_path.insert(3, "mini_datasets")
                traj_path = "/".join(traj_path)

        elif dataset_type == "perfect":
            # todo: this needs to be adapted to new traj_data
            traj_path = None

        return traj_path

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
    def list_registered_loco_mujoco():
        """
        List registered loco_mujoco environments.

        Returns:
             The list of the registered loco_mujoco environments.

        """
        return list(LocoEnv.registered_envs.keys())

    @property
    def root_body_name(self):
        return "pelvis"

    @staticmethod
    def _delete_from_spec(spec, joints_to_remove, actuators_to_remove, equ_constraints_to_remove):
        """
        Deletes certain joints, motors and equality constraints from a Mujoco specification.

        Args:
            spec (MjSpec): Mujoco specification.
            joints_to_remove (list): List of joint names to remove.
            actuators_to_remove (list): List of actuator names to remove.
            equ_constraints_to_remove (list): List of equality constraint names to remove.

        Returns:
            Modified Mujoco specification.

        """

        for joint in spec.joints:
            if joint.name in joints_to_remove:
                joint.delete()
        for actuator in spec.actuators:
            if actuator.name in actuators_to_remove:
                actuator.delete()
        for equality in spec.equalities:
            if equality.name in equ_constraints_to_remove:
                equality.delete()

        return spec

    @staticmethod
    def raise_mjx_not_enabled_error(*args, **kwargs):
        return ValueError("Mjx not enabled in this environment")

    @classmethod
    def path_to_real_datasets(cls):
        """
        Returns the path to the real datasets.

        """

        return Path(loco_mujoco.__file__).resolve().parent / "datasets" / "real" / cls.__name__

    @classmethod
    def path_to_perfect_datasets(cls):
        """
        Returns the path to the perfect datasets.

        """

        return Path(loco_mujoco.__file__).resolve().parent / "datasets" / "perfect" / cls.__name__

    @classmethod
    def convert_dataset_for_env(cls, traj_path):
        try:
            env_name = cls.__name__.split(".")[0]
            if "Mjx" in env_name:
                env_name = env_name[3:]
            file_path = PATH_RAW_DATASET / (traj_path.split("/")[-1].split(".")[0] + ".mat")
            print(f"Trajectory files for env \"{env_name}\" and task \"{file_path.stem}\" not yet converted."
                  f" Trying to convert.\n")
            convert_single_dataset_of_env(env_name, file_path)
        except Exception:
            print(f"Trajectory file {traj_path} not found and could not be created.")
            raise

        print(f"Conversion successful. Saving dataset to: {traj_path}.\n")

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
