from pathlib import Path
import mujoco
from mujoco import MjSpec
from mujoco import mjx

import jax
import jax.numpy as jnp
import numpy as np

from loco_mujoco.environments import LocoEnv
from loco_mujoco.core.observations import ObservationType
from loco_mujoco.core.utils import info_property

from loco_mujoco.trajectory import TrajectoryTransitions

from copy import deepcopy
from dataclasses import replace



class DummyHumamoidEnv(LocoEnv):


    def __init__(self, observation_spec=None, actuation_spec=None, spec=None, **kwargs):

        # load the model specification
        if spec is None:
            spec = self.get_default_xml_file_path()

        spec = mujoco.MjSpec.from_file(spec)

        # get the observation and action specification
        if observation_spec is None:
            # get default
            observation_spec = self._get_observation_specification(spec)
        else:
            # parse
            observation_spec = self.parse_observation_spec(observation_spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        self._using_jax = kwargs["enable_mjx"]

        super(DummyHumamoidEnv, self).__init__(spec, actuation_spec, observation_spec, **kwargs)

    @staticmethod
    def _get_observation_specification(spec: MjSpec):
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            A list of observation types.
        """
        observation_spec = [
            # ------------- JOINT POS -------------
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
            ObservationType.JointPos("q_abdomen_z", xml_name="abdomen_z"),

            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            ObservationType.JointVel("dq_abdomen_z", xml_name="abdomen_z"),

            # ------------- BODIES -------------
            ObservationType.BodyPos("left_thigh_obs_pos", xml_name="left_thigh"),
            ObservationType.BodyVel("left_thigh_obs_vel", xml_name="left_thigh"),

        ]

        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec):
        action_spec = ["abdomen_y", "right_knee"]
        return action_spec

    @classmethod
    def get_default_xml_file_path(cls):
        return (Path(__file__).resolve().parent / "humanoid_test.xml").as_posix()

    @info_property
    def sites_for_mimic(self):
        return ["torso_site", "pelvis_site", "right_thigh_site", "right_foot_site", "left_thigh_site",
                "left_foot_site"]

    @info_property
    def root_body_name(self):
        return "torso"

    @info_property
    def goal_visualization_arrow_offset(self):
        return [0, 0, 0.6]

    @info_property
    def grf_size(self):
        return 6

    @info_property
    def upper_body_xml_name(self):
        return "torso_link"

    @info_property
    def root_free_joint_xml_name(self):
        return "root"

    @info_property
    def root_height_healthy_range(self):
        return (0.6, 1.5)


    def generate_trajectory_from_nominal(self, nominal_traj, rng_key=None):
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
nominal_traj          `reset()` and `step()` methods.

        Args:
        rng_key (jax.random.PRNGKey, optional): A random key for reproducibility. Defaults to None.
        nominal_traj (Trajectory, optional): The nominal trajectory, to generate the dataset. Defaults to the trajectory
        loaded by the environment.

        Returns:
        TrajectoryTransitions: A dictionary containing the following:
            - observations (array): An array of shape (N_traj x (N_samples_per_traj-1), dim_state).
            - next_observations (array): An array of shape (N_traj x (N_samples_per_traj-1), dim_state).
            - absorbing (array): A flag array of shape (N_traj x (N_samples_per_traj-1)), indicating absorbing states.
            - For non-mocap datasets, actions may also be included.

        Raises:
        ValueError: If no trajectory is provided to the environment.

        """
        if self.th is None:
            raise ValueError("No trajectory was passed to the environment. "
                             "To create a dataset pass a trajectory first.")

        if self.th.traj.transitions is None:
            # create new trajectory and trajectory handler
            th = deepcopy(self.th)

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

            for i in range(self.th.n_trajectories):

                # set configuration to the first state of the current trajectory
                self.th.fixed_start_conf = (i, 0)

                # do a reset
                key, subkey = jax.random.split(rng_key)
                traj_data_single = nominal_traj.data.get(i, 0, np)  # get first sample
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
                for j in range(1, nominal_traj.data.len_trajectory(i)):
                    # get next sample and calculate forward dynamics
                    traj_data_single = nominal_traj.data.get(i, j, np)  # get next sample
                    data = self.set_sim_state_from_traj_data(data, traj_data_single, carry)
                    mujoco.mj_forward(model, data)

                    data, carry = self._simulation_post_step(model, data, carry)
                    obs, carry = self._create_observation(model, data, carry)
                    obs, data, info, carry = (
                        self._step_finalize(obs, model, data, info, carry))
                    observations.append(obs)

                    # check if the current state is an absorbing state
                    is_absorbing, carry = self._is_absorbing(obs, info, data, carry)

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

            transitions = TrajectoryTransitions(np.array(all_observations),
                                                np.array(all_next_observations),
                                                np.array(all_absorbing),
                                                np.array(all_dones))

            self.th = orig_th
            self.th.traj = replace(self.th.traj, transitions=transitions)

        return self.th.traj.transitions

    def mjx_generate_trajectory_from_nominal(self, nominal_traj, rng_key=None):
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
nominal_traj          `reset()` and `step()` methods.

        Args:
        rng_key (jax.random.PRNGKey, optional): A random key for reproducibility. Defaults to None.
        nominal_traj (Trajectory, optional): The nominal trajectory, to generate the dataset. Defaults to the trajectory
        loaded by the environment.

        Returns:
        TrajectoryTransitions: A dictionary containing the following:
            - observations (array): An array of shape (N_traj x (N_samples_per_traj-1), dim_state).
            - next_observations (array): An array of shape (N_traj x (N_samples_per_traj-1), dim_state).
            - absorbing (array): A flag array of shape (N_traj x (N_samples_per_traj-1)), indicating absorbing states.
            - For non-mocap datasets, actions may also be included.

        Raises:
        ValueError: If no trajectory is provided to the environment.

        """
        if self.th is None:
            raise ValueError("No trajectory was passed to the environment. "
                             "To create a dataset pass a trajectory first.")

        if self.th.traj.transitions is None:
            # create new trajectory and trajectory handler
            th = deepcopy(self.th)

            # set trajectory handler and store old one for later
            orig_th = self.th
            self.th = th

            # get a new model and data
            model = self.mjspec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            sys = mjx.put_model(model)
            data = mjx.put_data(model, data)
            first_data = mjx.forward(sys, data)



            # setup containers for the dataset
            all_observations, all_next_observations, all_dones = [], [], []

            if rng_key is None:
                rng_key = jax.random.key(0)

            for i in range(self.th.n_trajectories):

                # set configuration to the first state of the current trajectory
                self.th.fixed_start_conf = (i, 0)

                # do a reset
                data = first_data

                key, subkey = jax.random.split(rng_key)
                traj_data_single = nominal_traj.data.get(i, 0, jnp)  # get first sample
                carry = self._init_additional_carry(key, model, data, jnp)

                # set data from traj_data (qpos and qvel) and forward to calculate other kinematic entities.
                data = self.mjx_set_sim_state_from_traj_data(data, traj_data_single, carry)
                mjx.forward(sys, data)

                data, carry = self._mjx_reset_init_data_and_model(model, data, carry)
                data, carry = (
                    self.obs_container.reset_state(self, model, data, carry, jnp))
                obs, carry = self._mjx_create_observation(model, data, carry)
                info = self._reset_info_dictionary(obs, data, subkey)

                # initiate obs container
                observations = [obs]
                for j in range(1, nominal_traj.data.len_trajectory(i)):
                    # get next sample and calculate forward dynamics
                    traj_data_single = nominal_traj.data.get(i, j, jnp)  # get next sample
                    data = self.mjx_set_sim_state_from_traj_data(data, traj_data_single, carry)
                    mjx.forward(sys, data)

                    data, carry = self._mjx_simulation_post_step(model, data, carry)
                    obs, carry = self._mjx_create_observation(model, data, carry)
                    obs, data, info, carry = (
                        self._mjx_step_finalize(obs, model, data, info, carry))
                    observations.append(obs)

                    # check if the current state is an absorbing state
                    is_absorbing, carry = self._is_absorbing(obs, info, data, carry)

                observations = jnp.vstack(observations)
                all_observations.append(observations[:-1])
                all_next_observations.append(observations[1:])
                dones = jnp.zeros(observations.shape[0]-1)
                x = x.at[-1].set(1)
                all_dones.append(dones)

            all_observations = jnp.concatenate(all_observations).astype(np.float32)
            all_next_observations = jnp.concatenate(all_next_observations).astype(np.float32)
            all_dones = jnp.concatenate(all_dones).astype(np.float32)
            all_absorbing = jnp.zeros_like(all_dones).astype(np.float32)    # assume no absorbing states

            transitions = TrajectoryTransitions(jnp.array(all_observations),
                                                jnp.array(all_next_observations),
                                                jnp.array(all_absorbing),
                                                jnp.array(all_dones))

            self.th = orig_th
            self.th.traj = replace(self.th.traj, transitions=transitions)

        return self.th.traj.transitions