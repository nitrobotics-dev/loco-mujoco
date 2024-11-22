import numpy as np
import jax.numpy as jnp
import mujoco
from mujoco import MjSpec
from jax.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.core.utils.observations import Observation, ObservationType
from loco_mujoco.core.utils.math import calculate_relative_site_quatities


class Goal(Observation):

    def __init__(self, info_props, visualize_goal=False):

        self._initialized_from_traj = False
        self._info_props = info_props
        if visualize_goal:
            assert self.has_visual, (f"{self.__class__.__name__} does not support visualization. "
                                     f"Please set visualize_goal to False.")
        self.visualize_goal = visualize_goal
        super().__init__(obs_name=self.__class__.__name__)

    @property
    def has_visual(self):
        raise NotImplementedError

    @property
    def requires_trajectory(self):
        return False

    @classmethod
    def data_type(cls):
        return "userdata"

    def reset(self, data, backend, trajectory=None, traj_state=None):
        assert self.initialized
        return data

    def set_data(self, data, backend, traj_data=None, traj_state=None):
        assert self.initialized
        return data

    def apply_spec_modifications(self, spec: MjSpec, root_body_name: str):
        """
        Apply modifications to the Mujoco XML specification to include the goal.

        Args:
            spec (MjSpec): Mujoco specification.
            root_body_name (str): Name of the body to which the goal is attached.

        Returns:
            MjSpec: Modified Mujoco specification.
        """
        return spec

    def set_attr_data_compat(self, data, backend, attr, arr, ind):
        """
        Setter for different attributes in data that is compatible with Mujoco and MJX data structures.


        Args:
            data: The data structure to modify.
            backend (str): The backend to use for the modification (either numpy or jax-numpy).
            attr (str): The attribute to modify.
            arr (object): The array to set.
            ind (list(ind)): The index to set.


        Returns:
            The modified data structure.

        """
        if backend == np:
            getattr(data, attr)[ind] = arr
        elif backend == jnp:
            data = data.replace(**{attr: getattr(data, attr).at[ind].set(arr)})
        else:
            raise NotImplementedError
        return data

    @property
    def initialized(self):
        return self._initialized_from_mj and self._initialized_from_traj

    @property
    def spec(self):
        return []

    def allocate_user_data(self, spec):
        if self.size_user_data > 0:
            spec.nuserdata = self.size_user_data
        return spec

    @property
    def size_user_data(self):
        raise NotImplementedError

    @property
    def dim(self):
        return self.size_user_data

    @classmethod
    def list_goals(cls):
        return [goal for goal in Goal.__subclasses__()]


class NoGoal(Goal):

    def _init_from_mj(self, model, data, current_obs_size):
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.data_type_ind = np.array([])
        self.obs_ind = np.array([])
        self._initialized_from_mj = True

    def init_from_traj(self, traj_handler):
        self._initialized_from_traj = True

    @classmethod
    def get_obs(cls, model, data, ind, backend):
        return backend.array([])

    @property
    def has_visual(self):
        return False

    def reset(self, data, backend, trajectory=None, traj_state=None):
        return data

    @property
    def size_user_data(self):
        return 0


class GoalTrajArrow(Goal):
    # todo: update this class to work with new traj_data!

    _site_name_keypoint_1 = "goal_visual_k1"
    _site_name_keypoint_2 = "goal_visual_k2"
    _site_name_capsule = "goal_visual_cap"
    _name_goal_dict = "VEL_2D"
    _arrow_to_goal_ratio = 0.3

    def __init__(self, info_props, **kwargs):
        self._traj_goal_ind = None
        self.upper_body_xml_name = info_props["upper_body_xml_name"]

        super().__init__(info_props, **kwargs)

    def _init_from_mj(self, model, data, current_obs_size):
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        # todo: This will only work if userdata contains only a single goal and no other info.
        self.data_type_ind = np.array([i for i in range(data.userdata.size)])
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._initialized_from_mj = True

    def init_from_traj(self, traj_handler):
        self._initialized_from_traj = True

    @classmethod
    def get_obs(cls, model, data, ind, backend):
        return backend.ravel(data.userdata[ind.GoalTrajArrow])

    @property
    def has_visual(self):
        return True

    @property
    def requires_trajectory(self):
        return True

    def reset(self, data, backend, traj_data=None, traj_state=None):
        return self.set_data(data, backend, trajectory, traj_state)

    def set_data(self, data, backend, traj_data=None, traj_state=None):
        assert self._traj_goal_ind is not None
        # get the goal from the trajectory
        traj_goal = traj_data[self._traj_goal_ind, traj_state.traj_no, traj_state.subtraj_step_no]
        # set the goal in the userdata
        data = self.set_attr_data_compat(data, backend, "userdata", traj_goal, self.data_type_ind)
        # set the visual data
        if self.visualize_goal:
            data = self.set_visual_data(data, backend, traj_goal)
        return data

    def set_visual_data(self, data, backend, traj_goal):
        # get the relative desired position of the arrow (keypoint 2) and scale it by the ratio (for visual purposes)
        rel_target_arrow_pos = backend.concatenate([traj_goal * self._arrow_to_goal_ratio, jnp.ones(1)])
        # calculate and set the absolute position of the arrow
        abs_target_arrow_pos = ((data.body(self.upper_body_xml_name).xmat.reshape(3, 3) @ rel_target_arrow_pos) +
                                data.body(self.upper_body_xml_name).xpos)
        data.site(self._site_name_keypoint_2).xpos = abs_target_arrow_pos
        return data

    def apply_spec_modifications(self, spec, root_body_name):
        # apply the default modifications needed to store the goal in data
        self.allocate_user_data(spec)
        # optionally add sites for visualization
        if self.visualize_goal:
            # add sites for visualization
            root_body_name = self.upper_body_xml_name
            root = spec.find("body", root_body_name)
            # use two spheres to represent an arrow
            point_properties = dict(type="sphere", group=0, rgba=[1.0, 0.0, 0.0, 1.0])
            k1_pos, k2_pos = [0.0, 0.0, 1], [0.5, 0.0, 1]
            root.add("site", name=self._site_name_keypoint_1, pos=k1_pos, size=[0.05], **point_properties)
            root.add("site", name=self._site_name_keypoint_2, pos=k2_pos, size=[0.03], **point_properties)
        return spec

    @property
    def size_user_data(self):
        return 2


class GoalTrajMimic(Goal):

    main_body_id = None
    main_site_id = None
    _relevant_body_names = []
    _relevant_body_ids = []
    _rel_site_ids = []
    _body_rootid = None
    _site_bodyid = None

    def __init__(self, info_props, **kwargs):

        # todo: implement n_step_lookahead (requires dynamic slicing in jax)
        self.n_step_lookahead = 1
        super().__init__(info_props, **kwargs)

        # get main body name of the environment
        self.main_body_name = self._info_props["upper_body_xml_name"]

        # these will be calculated during initialization
        self._qpos_ind = None
        self._qvel_ind = None
        self._size_user_data = None
        self._size_additional_observation = None

    def _init_from_mj(self, model, data, current_obs_size):
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.data_type_ind = np.array([i for i in range(data.userdata.size)])
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        for body_name in self._relevant_body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            self._relevant_body_ids.append(body_id)
        for name in self._info_props["sites_for_mimic"]:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            self._rel_site_ids.append(site_id)
        self.__class__._rel_site_ids = np.array(self._rel_site_ids)
        self.__class__.main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.main_body_name)
        self.__class__._body_rootid = model.body_rootid
        self.__class__._site_bodyid = model.site_bodyid
        self._initialized_from_traj = True

    def apply_spec_modifications(self, spec: MjSpec, root_body_name: str):
        """
        Apply modifications to the Mujoco XML specification to include the goal.

        Args:
            spec (MjSpec): Mujoco specification.
            root_body_name (str): Name of the body to which the goal is attached.

        Returns:
            MjSpec: Modified Mujoco specification.
        """
        joints = spec.joints
        n_joints = len(joints)
        # for j in joints:
        #     body = j.parent
        #     if body.name not in self._relevant_body_names and body.name != self.main_body_name:
        #         self._relevant_body_names.append(body.name)
        for body in spec.bodies:
            if body.name not in self._relevant_body_names and body.name != self.main_body_name and body.name != "world":
                self._relevant_body_names.append(body.name)
        n_sites = len(self._info_props["sites_for_mimic"]) - 1   # number of sites to be considered, -1 because all quantities are relative to the main site
        size_for_joint_pos = (5 + (n_joints-1)) * self.n_step_lookahead     # free_joint has 7 dim -2 to not incl. x and y
        size_for_joint_vel = (6 + (n_joints-1)) * self.n_step_lookahead     # free_joint has 6 dim
        size_for_sites = (3 + 3 + 6) * n_sites * self.n_step_lookahead    # 3 for rpos, 3 for raxis_angle, 6 for rvel
        self._size_user_data = (size_for_joint_pos + size_for_joint_vel + size_for_sites) * self.n_step_lookahead
        self._size_additional_observation = size_for_sites
        self.allocate_user_data(spec)
        if self.visualize_goal:
            wb = spec.worldbody
            mimic_site_defaults = None
            # visualize mimic sites todo: the red sites of the robot should also be visible
            for site in spec.sites:
                if "mimic" in site.name:
                    site.group = 0
                    if mimic_site_defaults is None:
                        mimic_site_defaults = site.default()

            for body_name in self._info_props["sites_for_mimic"]:
                name = "visual_goal_" + body_name
                visual_goal = wb.add_body(name=name, mocap=True)
                visual_goal.set_default(mimic_site_defaults)
                visual_goal.add_site(name=name, rgba=[0.0, 1.0, 0.0, 0.5], group=0)

        return spec

    def init_from_traj(self, traj_handler):
        # focus on joints in the observation space
        self._qpos_ind = np.concatenate([obs.data_type_ind for obs in self.obs_container.entries()
                                         if (type(obs) is ObservationType.JointPos) or
                                         (type(obs) is ObservationType.FreeJointPos) or
                                         (type(obs) is ObservationType.EntryFromFreeJointPos) or
                                         (type(obs) is ObservationType.FreeJointPosNoXY)])
        self._qvel_ind = np.concatenate([obs.data_type_ind for obs in self.obs_container.entries()
                                         if (type(obs) is ObservationType.JointVel) or
                                         (type(obs) is ObservationType.EntryFromFreeJointVel) or
                                         (type(obs) is ObservationType.FreeJointVel)])
        self._initialized_from_traj = True

    @classmethod
    def get_obs(cls, model, data, data_ind_cont, backend):

        # get trajectory goal
        goal_traj = data.userdata[data_ind_cont.GoalTrajMimic]

        if len(cls._rel_site_ids) > 0 and goal_traj.size > 0:

            # get relative site quantities for current data
            rel_site_ids = cls._rel_site_ids
            rel_body_ids = cls._site_bodyid[rel_site_ids]
            site_rpos, site_rangles, site_rvel = calculate_relative_site_quatities(data, rel_site_ids, rel_body_ids,
                                                                                   cls._body_rootid, backend)

            # concatenate all relevant information
            return backend.concatenate([backend.ravel(site_rpos),
                                        backend.ravel(site_rangles),
                                        backend.ravel(site_rvel),
                                        backend.ravel(goal_traj)])
        else:
            return backend.array([])

    @property
    def has_visual(self):
        return True

    @property
    def requires_trajectory(self):
        return True

    def reset(self, data, backend, trajectory=None, traj_state=None):
        return self.set_data(data, backend, trajectory, traj_state)

    def set_data(self, data, backend, traj_data=None, traj_state=None):

        # get traj_data for current time step
        traj_data_single = traj_data.get(traj_state.traj_no, traj_state.subtraj_step_no, backend)

        # get joint positions and velocities
        qpos_traj = traj_data_single.qpos
        qvel_traj = traj_data_single.qvel

        # get relative site quantities
        rel_site_ids = self._rel_site_ids
        rel_body_ids = self._site_bodyid[rel_site_ids]
        site_rpos, site_rangles, site_rvel = calculate_relative_site_quatities(traj_data_single, rel_site_ids, rel_body_ids,
                                                                               self._body_rootid, backend)

        # setup goal observation
        goal_obs = backend.concatenate([qpos_traj[self._qpos_ind],
                                        qvel_traj[self._qvel_ind],
                                        backend.ravel(site_rpos),
                                        backend.ravel(site_rangles),
                                        backend.ravel(site_rvel)])

        # set the goal in the userdata
        data = self.set_attr_data_compat(data, backend, "userdata", goal_obs, self.data_type_ind)
        if self.visualize_goal:
            # todo: this currently only works with normal Mujoco since Mjx does not support mocap bodies yet.
            # if backend == jnp:
            #     raise NotImplementedError("mocap_pos and mocap_quat are not supported in MJX yet.")
            data = self.set_visual_data(data, backend, traj_data, traj_state)
        return data

    def set_visual_data(self, data, backend, traj_data, traj_state):

        if backend == np:
            R = np_R
        else:
            R = jnp_R

        qpos_init = traj_data.get_qpos(traj_state.traj_no, traj_state.subtraj_step_no_init)
        site_xpos = traj_data.get_site_xpos(traj_state.traj_no, traj_state.subtraj_step_no)
        site_xmat = traj_data.get_site_xmat(traj_state.traj_no, traj_state.subtraj_step_no)
        site_xquat = R.from_matrix(site_xmat.reshape(-1, 3, 3)).as_quat()
        s_ids = jnp.array(self._rel_site_ids)
        site_xpos = site_xpos.at[:, :2].add(-qpos_init[:2]) # reset to the initial position
        data = self.set_attr_data_compat(data, backend, "mocap_pos", site_xpos[s_ids], jnp.arange(len(s_ids)))
        data = self.set_attr_data_compat(data, backend, "mocap_quat", site_xquat[s_ids], jnp.arange(len(s_ids)))

        return data

    @property
    def dim(self):
        return self.size_user_data + self._size_additional_observation

    @property
    def size_user_data(self):
        if self._size_user_data is None:
            raise NotImplementedError
        return self._size_user_data
