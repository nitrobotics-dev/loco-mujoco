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
from loco_mujoco.core.mujoco_mjx import Mjx, MjxAdditionalCarry
from loco_mujoco.core.visuals import VideoRecorder
from loco_mujoco.trajectory import TrajectoryHandler
from loco_mujoco.core.utils import info_property
from loco_mujoco.trajectory import Trajectory, TrajState, TrajectoryTransitions


@struct.dataclass
class LocoCarry(MjxAdditionalCarry):
    traj_state: TrajState


class LocoEnv(Mjx):
    """
    Base class for all kinds of locomotion environments.

    """

    mjx_enabled = False

    def __init__(self, spec, action_spec, observation_spec, enable_mjx=False,
                 n_envs=1, gamma=0.99, horizon=1000, n_substeps=10, th_params=None,
                 traj_params=None, timestep=0.001, default_camera_mode="follow",
                 model_option_conf=None, **core_params):
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

        if "geom_group_visualization_on_startup" not in core_params.keys():
            core_params["geom_group_visualization_on_startup"] = [0, 2]   # enable robot geom [0] and floor visual [2]

        if enable_mjx:
            # call parent (Mjx) constructor
            super(LocoEnv, self).__init__(n_envs, spec=spec, actuation_spec=action_spec,
                                          observation_spec=observation_spec, gamma=gamma,
                                          horizon=horizon, n_substeps=n_substeps,
                                          timestep=timestep,
                                          default_camera_mode=default_camera_mode,
                                          model_option_conf=model_option_conf, **core_params)
        else:
            assert n_envs == 1, "Mjx not enabled, setting the number of environments > 1 is not allowed."
            # call grandparent constructor (Mujoco (CPU) environment)
            super(Mjx, self).__init__(spec=spec, actuation_spec=action_spec,
                                      observation_spec=observation_spec, gamma=gamma,
                                      horizon=horizon, n_substeps=n_substeps,
                                      timestep=timestep,
                                      default_camera_mode=default_camera_mode,
                                      model_option_conf=model_option_conf, **core_params)

        # the action space is supposed to be between -1 and 1, so we normalize it
        self._scale_action_space()

        # dataset dummy
        self._dataset = None

        # setup trajectory
        if traj_params:
            self.th = None
            self.load_trajectory(**traj_params)
        else:
            self.th = None

        self._th_params = th_params

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
            warn (bool): If True, a warning will be raised.

        """

        if self.th is not None and warn:
            warnings.warn("New trajectories loaded, which overrides the old ones.", RuntimeWarning)

        th_params = self._th_params if self._th_params is not None else {}
        self.th = TrajectoryHandler(model=self._model, warn=warn, traj_path=traj_path,
                                    traj=traj, control_dt=self.dt, **th_params)

        if self.th.traj.obs_container is not None:
            assert self.obs_container == self.th.traj.obs_container, \
                ("Observation containers of trajectory and environment do not match. \n"
                 "Please, either load a trajectory with the same observation container or "
                 "set the observation container of the environment to the one of the trajectory.")

        # setup trajectory information in observation_dict, goal and reward if needed
        for obs_entry in self.obs_container.entries():
            obs_entry.init_from_traj(self.th)
        self._goal.init_from_traj(self.th)
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
            # goals can terminate an episode
            done = jnp.logical_or(done, self._goal.mjx_is_done(self, self._model, data, carry, jnp))

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

            # goals can terminate an episode
            done != self._goal.is_done(self, self._model, data, carry, np)

        return done

    def _simulation_post_step(self, model, data, carry):
        """
        Update trajectory state if needed.

        """
        # call parent to update domain randomization and terrain
        data, carry = super()._simulation_post_step(model, data, carry)

        # update trajectory state
        if self.th is not None:
            carry = self.th.update_state(self, model, data, carry, np)

        return data, carry

    def _mjx_simulation_post_step(self, model, data, carry):
        """
        Update trajectory state if needed.

        """
        # call parent to update domain randomization and terrain
        data, carry = super()._mjx_simulation_post_step(model, data, carry)

        # update trajectory state
        if self.th is not None:
            carry = self.th.update_state(self, self._model, data, carry, jnp)

        return data, carry

    def create_dataset(self, rng_key=None):
        """
        Generates a dataset from the specified trajectories without including actions.

        Notes:
        - Observations are created by following steps similar to the `reset()` and `step()` methods.
        - TrajectoryData is used instead of MjData to reduce memory usage. Forward dynamics are applied
          to compute additional entities.
        - Since TrajectoryData only contains very few kinematic properties (to save memory), Mujoco's
          forward dynamics are used to calculate other entities.
        - Kinematic entities derived from mocap data are generally reliable, while dynamics-related
          properties may be less accurate.
        - Observations based on kinematic entities are recommended to ensure realistic datasets.
        - The dataset is built iteratively to compute stateful observations consistently with the
          `reset()` and `step()` methods.

        Args:
        rng_key (jax.random.PRNGKey, optional): A random key for reproducibility. Defaults to None.

        Returns:
        TrajectoryTransitions: A dictionary containing the following:
            - observations (array): An array of shape (N_traj x (N_samples_per_traj-1), dim_state).
            - next_observations (array): An array of shape (N_traj x (N_samples_per_traj-1), dim_state).
            - absorbing (array): A flag array of shape (N_traj x (N_samples_per_traj-1)), indicating absorbing states.
            - For non-mocap datasets, actions may also be included.

        Raises:
        ValueError: If no trajectory is provided to the environment.

        """

        if self.th is not None:

            if self.th.traj.transitions is None:

                # create new trajectory and trajectory handler
                info, data = deepcopy(self.th.traj.info), deepcopy(self.th.traj.data)
                info.model = info.model.to_numpy()
                data = data.to_numpy()
                traj = Trajectory(info, data)
                th = TrajectoryHandler(model=self._model, traj=traj, control_dt=self.dt,
                                       random_start=False, fixed_start_conf=(0, 0))

                # set trajectory handler and store old one for later
                orig_th = self.th
                self.th = th

                # get a new model and data
                model = self.mjspec.compile()
                data = mujoco.MjData(model)
                mujoco.mj_resetData(model, data)

                # setup containers for the dataset
                all_observations, all_next_observations, all_dones = [], [], []

                if rng_key is None:
                    rng_key = jax.random.key(0)

                for i in tqdm(range(self.th.n_trajectories), desc="Creating Transition Dataset"):

                    # set configuration to the first state of the current trajectory
                    self.th.fixed_start_conf = (i, 0)

                    # do a reset
                    key, subkey = jax.random.split(rng_key)
                    traj_data_single = self.th.traj.data.get(i, 0, np)  # get first sample
                    carry = self._init_additional_carry(key, model, data, np)

                    # set data from traj_data (qpos and qvel) and forward to calculate other kinematic entities.
                    mujoco.mj_resetData(model, data)
                    data = self.set_sim_state_from_traj_data(data, traj_data_single, carry)
                    mujoco.mj_forward(model, data)

                    data, carry = self._reset_init_data_and_model(model, data, carry)
                    data, carry = (
                        self.obs_container.reset_state(self, model, data, carry, np))
                    obs, carry = self._create_observation(model, data, carry)
                    info = self._reset_info_dictionary(obs, data, subkey)

                    # initiate obs container
                    observations = [obs]
                    for j in range(1, self.th.len_trajectory(i)):
                        # get next sample and calculate forward dynamics
                        traj_data_single = self.th.traj.data.get(i, j, np)  # get next sample
                        data = self.set_sim_state_from_traj_data(data, traj_data_single, carry)
                        mujoco.mj_forward(model, data)

                        data, carry = self._simulation_post_step(model, data, carry)
                        obs, carry = self._create_observation(model, data, carry)
                        obs, data, info, carry = (
                            self._step_finalize(obs, model, data, info, carry))
                        observations.append(obs)

                        # check if the current state is an absorbing state
                        is_absorbing, carry = self._is_absorbing(obs, info, data, carry)
                        if is_absorbing:
                            warnings.warn("Some of the states in the created dataset are terminal states. "
                                          "This should not happen.")

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

                if orig_th.is_numpy:
                    backend = np
                else:
                    backend = jnp

                transitions = TrajectoryTransitions(backend.array(all_observations),
                                                    backend.array(all_next_observations),
                                                    backend.array(all_absorbing),
                                                    backend.array(all_dones))

                self.th = orig_th
                self.th.traj = replace(self.th.traj, transitions=transitions)

            return self.th.traj.transitions

        else:
            raise ValueError("No trajectory was passed to the environment. "
                             "To create a dataset pass a trajectory first.")

    def play_trajectory(self, n_episodes=None, n_steps_per_episode=None, from_velocity=False, render=True,
                        record=False, recorder_params=None, callback_class=None, quiet=False, key=None):
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
            quiet (bool): If True, disable tqdm
            key (jax.random.PRNGKey): Random key to use for the simulation.

        """

        assert self.th is not None

        if not self.th.is_numpy:
            was_jax = True
            self.th.to_numpy()
        else:
            was_jax = False

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
        subtraj_step_no = 0
        traj_data_sample = self.th.get_current_traj_data(self._additional_carry, np)

        if render:
            frame = self.render(record)
        else:
            frame = None

        if record:
            recorder(frame)

        highest_int = np.iinfo(np.int32).max
        if n_episodes is None:
            n_episodes = highest_int
        for i in range(n_episodes):
            if n_steps_per_episode is None:
                nspe = (self.th.len_trajectory(self._additional_carry.traj_state.traj_no) -
                        self._additional_carry.traj_state.subtraj_step_no)
            else:
                nspe = n_steps_per_episode

            for j in tqdm(range(nspe), disable=quiet):
                if callback_class is None:
                    self._data = self.set_sim_state_from_traj_data(self._data, traj_data_sample, self._additional_carry)
                    self._model, self._data, self._additional_carry = (
                        self._simulation_pre_step(self._model, self._data, self._additional_carry))
                    mujoco.mj_forward(self._model, self._data)
                    self._data, self._additional_carry = (
                        self._simulation_post_step(self._model, self._data, self._additional_carry))
                else:
                    self._model, self._data, self._additional_carry = (
                        callback_class(self, self._model, self._data, traj_data_sample, self._additional_carry))

                traj_data_sample = self.th.get_current_traj_data(self._additional_carry, np)

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

                obs, self._additional_carry = self._create_observation(self._model, self._data, self._additional_carry)

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

        if was_jax:
            self.th.to_jax()

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
            stacklevel=3
        )
        self.play_trajectory(n_episodes, n_steps_per_episode, True, render,
                             record, recorder_params, callback_class, key)

    def set_sim_state_from_traj_data(self, data, traj_data, carry):
        traj_state = carry.traj_state
        # get the initial state of the current trajectory
        traj_data_init = self.th.traj.data.get(traj_state.traj_no, traj_state.subtraj_step_no_init, np)
        # subtract the initial state from the current state
        traj_data.qpos[0:2] -= traj_data_init.qpos[0:2]
        return Mjx.set_sim_state_from_traj_data(data, traj_data, carry)

    def mjx_set_sim_state_from_traj_data(self, data, traj_data, carry):
        traj_state = carry.traj_state
        # get the initial state of the current trajectory
        traj_data_init = self.th.traj.data.get(traj_state.traj_no, traj_state.subtraj_step_no_init, jnp)
        # subtract the initial state from the current state
        traj_data = traj_data.replace(qpos=traj_data.qpos.at[0:2].add(-traj_data_init.qpos[0:2]))
        return Mjx.mjx_set_sim_state_from_traj_data(data, traj_data, carry)

    def _init_additional_carry(self, key, model, data, backend):

        carry = super()._init_additional_carry(key, model, data, backend)

        key = carry.key
        key, _k = jax.random.split(key)

        # create additional carry
        carry = LocoCarry(
            traj_state=self.th.init_state(self, _k, model, data, backend) if self.th is not None else EmptyState(),
            **vars(carry.replace(key=key)))

        return carry

    def reset(self, key):
        if self.th is not None and not self.th.is_numpy:
            self.th.to_numpy()
        return super().reset(key)

    def mjx_reset(self, key):
        if self.th is not None and self.th.is_numpy:
            raise ValueError("Trajectory is in numpy format, but your attempting to run the MJX backend. "
                             "Please call the <your_env_name>.th.to_jax() function on your environment first.")
        return super().mjx_reset(key)

    def _reset_init_data_and_model(self, model, data, carry):

        # reset trajectory state
        if self.th is not None:
            data, carry = self.th.reset_state(self, self._model, data, carry, np) \
                if self.th is not None else (data, carry)

        # call parent to apply domain randomization and terrain
        data, carry = super()._reset_init_data_and_model(model, data, carry)

        return data, carry

    def _mjx_reset_init_data_and_model(self, model, data, carry):

        # reset trajectory state
        if self.th is not None:
            data, carry = self.th.reset_state(self, self._model, data, carry, jnp) \
                if self.th is not None else (data, carry)

        # call parent to apply domain randomization and terrain
        data, carry = super()._mjx_reset_init_data_and_model(model, data, carry)

        return data, carry

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
    def generate(cls, task=None, dataset_type="mocap", debug=False,
                 clip_trajectory_to_joint_ranges=False, **kwargs):
        """
        Returns an environment corresponding to the specified task.

        Args:
            task (str): Main task to solve.
            dataset_type (str): "mocap" or "pretrained". "real" uses real motion capture data as the
            reference trajectory. This data does not perfectly match the kinematics
            and dynamics of this environment, hence it is more challenging. "perfect" uses
            a perfect dataset.
            debug (bool): If True, the smaller test datasets are used for debugging purposes.
            clip_trajectory_to_joint_ranges (bool): If True, trajectory is clipped to joint ranges.

        Returns:
            An environment of specified class and task.

        """

        warnings.warn(
            "The methods `LocoEnv.make()` and `LocoEnv.generate()` are deprecated and will be "
            "removed in a future release.\nPlease use the task factory classes instead: "
            "`ImitationFactory`, `RLFactory`, or `AMASSImitationFactory`.",
            category=DeprecationWarning,
            stacklevel=3
        )

        # import here to avoid circular dependency
        from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf
        from loco_mujoco.task_factories import RLFactory


        if task is None:
            return RLFactory.make(cls.__name__, **kwargs)
        else:
            return ImitationFactory.make(cls.__name__, DefaultDatasetConf(task, dataset_type, debug), **kwargs)

    @classmethod
    def get_default_xml_file_path(cls):
        """
        Returns the default path to the xml file of the environment.
        """
        raise NotImplementedError(f"Please implement the default_xml_file_path property "
                                  f"in the {type(cls).__name__} environment.")

    @info_property
    def root_body_name(self):
        raise NotImplementedError(f"Please implement the root_body_name "
                                  f"info property in the {type(self).__name__} environment.")

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
        return []

    @info_property
    def goal_visualization_arrow_offset(self):
        """
        Returns the offset of the goal visualization arrow.

        """
        return [0, 0, 0.0]

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
