from enum import Enum
from typing import List, Optional, Any, Dict
from dataclasses import dataclass
import mujoco
import numpy as np
from dm_control import mjcf

from mujoco import mjx
from flax import struct
import jax
import jax.numpy as jnp

from loco_mujoco.core.utils import Box, MDPInfo, MujocoViewer


class ObservationType(Enum):
    """
    An enum indicating the type of data that should be added to the observation
    of the environment, can be joint/body/site positions, rotations, and velocities.
    The Observation have the following returns:
        BODY_POS: (3,) x, y, z position of the body
        BODY_ROT: (4,) quaternion of the body
        BODY_VEL: (6,) first angular velocity around x, y, z. Then linear velocity for x, y, z
        JOINT_POS: (1,) rotation of the joint OR (7,) position, quaternion of a free joint
        JOINT_VEL: (1,) velocity of the joint OR (6,) FIRST linear then angular velocity !different to BODY_VEL!
        SITE_POS: (3,) x, y, z position of the body
        SITE_ROT: (9,) rotation matrix of the site
    """
    __order__ = "BODY_POS BODY_ROT BODY_VEL JOINT_POS JOINT_VEL SITE_POS SITE_ROT"
    BODY_POS = 0
    BODY_ROT = 1
    BODY_VEL = 2
    JOINT_POS = 3
    JOINT_VEL = 4
    SITE_POS = 5
    SITE_ROT = 6


@dataclass
class ObservationEntry:
    obs_ind: Optional[List[int]]
    xml_name: str
    obs_type_ind: int
    dim: int
    obs_min: Optional[List[float]]
    obs_max: Optional[List[float]]
    obs_type: ObservationType
    mj_type: mujoco.mjtObj


class Mujoco:
    """
        This is the base class for all Mujoco environments, CPU and MjX.

    """

    def __init__(self, xml_file, actuation_spec, observation_spec, gamma, horizon, n_environments=1,
                 timestep=None, n_substeps=1, n_intermediate_steps=1,  collision_groups=None,
                 **viewer_params):

        # load the model and data
        self._model = self.load_model(xml_file)
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
        self._cur_step_in_episode = 0

        # create a dictionary of ObservationEntry for each observation specified in observation_spec
        self._build_obs_dict(observation_spec, self._model, self._data)

        # define observation space bounding box
        observation_space = Box(*self._get_obs_limits())

        # read the actuation spec and build the mapping between actions and ids
        self._action_indices = self.get_action_indices(self._model, self._data, actuation_spec)

        # define action space bounding box
        action_space = Box(*self._get_action_limits(self._action_indices, self._model))

        # pre-process the collision groups for faster detection of contacts
        self.collision_groups = self._process_collision_groups(collision_groups)

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

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        return self._reset(self.backend)

    def _reset(self, backend):
        mujoco.mj_resetData(self._model, self._data)
        self._data = self._setup(self._data, self.backend)
        self._obs = self._create_observation(self._data, backend)
        self._info = self._reset_info_dictionary(self._obs, self._data, backend)
        self._cur_step_in_episode = 0
        return self._obs

    def step(self, action):
        return self._step(action, self.backend)

    def _step(self, action, backend):
        cur_obs = self._obs.copy()
        cur_info = self._info.copy()

        # preprocess action (does nothing by default)
        action = self._preprocess_action(action, self._data, self.backend)

        # modify obs and data, before stepping in the env (does nothing by default)
        cur_obs, self._data = self._step_init(cur_obs, self._data, self.backend)

        ctrl_action = None

        for i in range(self._n_intermediate_steps):

            if self._recompute_action_per_step or ctrl_action is None:
                ctrl_action = self._compute_action(cur_obs, action)
                self._data.ctrl[self._action_indices] = ctrl_action

            # modify data during simulation, before main step (does nothing by default)
            self._data = self._simulation_pre_step(self._data, self.backend)

            # main mujoco step, runs the sim for n_substeps
            mujoco.mj_step(self._model, self._data, self._n_substeps)

            # modify data during simulation, after main step (does nothing by default)
            self._data = self._simulation_pre_step(self._data, self.backend)

            # recompute the action at each intermediate step (not executed by default)
            if self._recompute_action_per_step:
                cur_obs = self._create_observation(self._data, backend)

        # create the final observation
        if not self._recompute_action_per_step:
            cur_obs = self._create_observation(self._data, backend)

        # modify obs and data, before stepping in the env (does nothing by default)
        cur_obs, self._data = self._step_finalize(cur_obs, self._data, self.backend)

        # check if the current state is an absorbing state
        absorbing = self._is_absorbing(cur_obs, self._data, self.backend)

        # calculate the reward
        reward = self._reward(self._obs, action, cur_obs, absorbing, self._data, backend)

        # create info (does nothing by default)
        info = self._modify_info_dictionary(cur_info, cur_obs, self._data, backend)

        # calculate flag indicating whether this is the last obs before resetting
        done = absorbing or (self._cur_step_in_episode >= self.info.horizon)

        self._obs = cur_obs
        self._cur_step_in_episode += 1

        return cur_obs, reward, absorbing, done, info

    def render(self, record=False):
        if self._viewer is None:
            self._viewer = MujocoViewer(self._model, self.dt, record=record, **self._viewer_params)

        return self._viewer.render(self._data, record)

    def stop(self):
        if self._viewer is not None:
            self._viewer.stop()
            del self._viewer
            self._viewer = None

    def _setup(self, data, backend):
        return data

    def _step_init(self, obs, data, backend):
        return obs, data

    def _is_absorbing(self, obs, data, backend):
        """
        Check whether the given state is an absorbing state or not.

        Args:
            obs (np.array): the state of the system.

        Returns:
            A boolean flag indicating whether this state is absorbing or not.

        """
        return False

    def _step_finalize(self, obs, data, backend):
        """
        Allows information to be accesed at the end of a step.
        """
        return obs, data

    def _reset_info_dictionary(self, obs, data, backend, **kwargs):
        return {}

    def _modify_info_dictionary(self, info, obs, data, backend):
        return info

    def _preprocess_action(self, action, data, backend):
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

    def _simulation_pre_step(self, data, backend):
        """
        Allows information to be accesed and changed at every intermediate step
        before taking a step in the mujoco simulation.
        Can be usefull to apply an external force/torque to the specified bodies.

        ex: apply a force over X to the torso:
        force = [200, 0, 0]
        torque = [0, 0, 0]
        self.sim.data.xfrc_applied[self.sim.model._body_name2id["torso"],:] = force + torque

        """
        return data

    def _simulation_post_step(self, data, backend):
        """
        Allows information to be accesed at every intermediate step
        after taking a step in the mujoco simulation.
        Can be usefull to average forces over all intermediate steps.

        """
        return data

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

    def _build_obs_dict(self, observation_spec, model, data):

        self._obs_dict = dict()
        self._obs_body_xpos_ind = []
        self._obs_body_xquat_ind = []
        self._obs_body_cvel_ind = []
        self._obs_joint_qpos_ind = []
        self._obs_joint_qvel_ind = []
        self._obs_site_xpos_ind = []
        self._obs_site_xmat_ind = []

        jnt_name2id = dict()
        jnt_name2type = dict()
        for i in range(model.njnt):
            j = model.joint(i)
            jnt_name2id[j.name] = i
            jnt_name2type[j.name] = j.type[0]   # right now not used

        i = 0
        for obs in observation_spec:
            observation_name, xml_name, obs_type = obs
            if obs_type == ObservationType.BODY_POS:
                mj_type = mujoco.mjtObj.mjOBJ_BODY
                dim = len(data.body(xml_name).xpos)
                obs_min, obs_max = [-np.inf] * dim, [np.inf] * dim
                obs_type_ind = data.body(xml_name).id
                self._obs_body_xpos_ind.append(obs_type_ind)
            elif obs_type == ObservationType.BODY_VEL:
                mj_type = mujoco.mjtObj.mjOBJ_BODY
                dim = len(data.body(xml_name).cvel)
                obs_min, obs_max = [-np.inf] * dim, [np.inf] * dim
                obs_type_ind = data.body(xml_name).id
                self._obs_body_cvel_ind.append(obs_type_ind)
            elif obs_type == ObservationType.BODY_ROT:
                mj_type = mujoco.mjtObj.mjOBJ_BODY
                dim = len(data.body(xml_name).xquat)
                obs_min, obs_max = [-np.inf] * dim, [np.inf] * dim
                obs_type_ind = data.body(xml_name).id
                self._obs_body_xquat_ind.append(obs_type_ind)
            elif obs_type == ObservationType.JOINT_POS:
                mj_type = mujoco.mjtObj.mjOBJ_JOINT
                dim = len(data.joint(xml_name).qpos)
                jh = model.joint(jnt_name2id[xml_name])
                if dim == 1 and jh.limited:
                    obs_min, obs_max = [jh.range[0]], [jh.range[1]]
                else:
                    # note: free joints do not have limits
                    obs_min, obs_max = [-np.inf] * dim, [np.inf] * dim
                obs_type_ind = data.joint(xml_name).id
                self._obs_joint_qpos_ind.append(obs_type_ind)
            elif obs_type == ObservationType.JOINT_VEL:
                mj_type = mujoco.mjtObj.mjOBJ_JOINT
                dim = len(data.joint(xml_name).qvel)
                obs_min, obs_max = [-np.inf] * dim, [np.inf] * dim
                obs_type_ind = data.joint(xml_name).id
                self._obs_joint_qvel_ind.append(obs_type_ind)
            elif obs_type == ObservationType.SITE_POS:
                mj_type = mujoco.mjtObj.mjOBJ_SITE
                dim = len(data.site(xml_name).xpos)
                obs_min, obs_max = [-np.inf] * dim, [np.inf] * dim
                obs_type_ind = data.site(xml_name).id
                self._obs_site_xpos_ind.append(obs_type_ind)
            elif obs_type == ObservationType.SITE_ROT:
                # Sites don't have rotation quaternion for some reason...
                # x_mat is rotation matrix with shape (9,)
                mj_type = mujoco.mjtObj.mjOBJ_SITE
                dim = len(data.site(xml_name).xmat)
                obs_min, obs_max = [-np.inf] * dim, [np.inf] * dim
                obs_type_ind = data.site(xml_name).id
                self._obs_site_xmat_ind.append(obs_type_ind)
            else:
                raise ValueError

            obs_ind = [j for j in range(i, i + dim)]
            i += dim
            if observation_name in self._obs_dict.keys():
                raise KeyError("Duplicate keys are not allowed. Key: ", observation_name)

            self._obs_dict[observation_name] = ObservationEntry(obs_ind, xml_name, obs_type_ind,
                                                                dim, obs_min, obs_max, obs_type, mj_type)

        self._obs_body_xpos_ind = self.backend.array(self._obs_body_xpos_ind, dtype=int)
        self._obs_body_xquat_ind = self.backend.array(self._obs_body_xquat_ind, dtype=int)
        self._obs_body_cvel_ind = self.backend.array(self._obs_body_cvel_ind, dtype=int)
        self._obs_joint_qpos_ind = self.backend.array(self._obs_joint_qpos_ind, dtype=int)
        self._obs_joint_qvel_ind = self.backend.array(self._obs_joint_qvel_ind, dtype=int)
        self._obs_site_xpos_ind = self.backend.array(self._obs_site_xpos_ind, dtype=int)
        self._obs_site_xmat_ind = self.backend.array(self._obs_site_xmat_ind, dtype=int)

    def _reward(self, obs, action, next_obs, absorbing, data, backend):
        return 0.0

    def _create_observation(self, data, backend):

        obs = backend.concatenate(
            [backend.ravel(data.xpos[self._obs_body_xpos_ind]),
             backend.ravel(data.xquat[self._obs_body_xquat_ind]),
             backend.ravel(data.cvel[self._obs_body_cvel_ind]),
             backend.ravel(data.qpos[self._obs_joint_qpos_ind]),
             backend.ravel(data.qvel[self._obs_joint_qvel_ind]),
             backend.ravel(data.site_xpos[self._obs_site_xpos_ind]),
             backend.ravel(data.site_xmat[self._obs_site_xmat_ind])])

        return obs

    def _process_collision_groups(self, collision_groups):
        processed_collision_groups = {}
        if collision_groups is not None:
            for name, geom_names in collision_groups:
                col_group = list()
                for geom_name in geom_names:
                    mj_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                    assert mj_id != -1, f"geom \"{geom_name}\" not found! Can't be used for collision-checking."
                    col_group.append(mj_id)
                processed_collision_groups[name] = set(col_group)

        return processed_collision_groups

    def _get_obs_limits(self):
        obs_min = np.concatenate([np.array(entry.obs_min) for entry in self._obs_dict.values()])
        obs_max = np.concatenate([np.array(entry.obs_max) for entry in self._obs_dict.values()])
        return obs_min, obs_max

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
    def backend(self):
        return np

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