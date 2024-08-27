from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
import jax.numpy as jnp

from loco_mujoco.core.mujoco_base import GoalType


class GoalInterface(ABC):

    registered_goals = dict()

    def __init__(self, info_props, visualize_goal=False):
        self.initialized = False
        self._info_props = info_props
        if visualize_goal:
            assert self.has_visual, (f"{self.__name__} does not support visualization. "
                                     f"Please set visualize_goal to False.")
        self.visualize_goal = visualize_goal

    @staticmethod
    @abstractmethod
    def get_name():
        pass

    @property
    @abstractmethod
    def has_visual(self):
        pass

    @abstractmethod
    def initialize(self, goal_dict, trajectories=None):
        pass

    @property
    def requires_trajectory(self):
        return False

    @abstractmethod
    def reset(self, data, backend, trajectory=None, traj_state=None):
        assert self.initialized
        return data

    def set_data(self, data, backend, trajectory=None, traj_state=None):
        assert self.initialized
        return data

    def apply_xml_modifications(self, xml_handle, root_body_name):
        return xml_handle

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
    def spec(self):
        return []

    def allocate_user_data(self, xml_handle):
        if self.size_user_data > 0:
            xml_handle.size.nuserdata = 2
        return xml_handle

    @property
    @abstractmethod
    def size_user_data(self):
        pass

    @classmethod
    def register(cls):
        """
        Register a goal in the goal list.

        """
        env_name = cls.get_name()

        if env_name not in GoalInterface.registered_goals:
            GoalInterface.registered_goals[env_name] = cls

    @staticmethod
    def list_registered():
        """
        List registered goals.

        Returns:
             The list of the registered goals.

        """
        return list(GoalInterface.registered_goals.keys())


class NoGoal(GoalInterface):

    def initialize(self, goal_dict, trajectories=None):
        self.initialized = True

    @staticmethod
    def get_name():
        return "no_goal"

    @property
    def has_visual(self):
        return False

    def reset(self, data, backend, trajectory=None, traj_state=None):
        return data

    @property
    def size_user_data(self):
        return 0


class GoalTrajArrow(GoalInterface):

    _site_name_keypoint_1 = "goal_visual_k1"
    _site_name_keypoint_2 = "goal_visual_k2"
    _site_name_capsule = "goal_visual_cap"
    _name_goal_dict = "VEL_2D"
    _arrow_to_goal_ratio = 0.3

    def __init__(self, info_props, **kwargs):
        self._data_goal_userdata_ind = None
        self._traj_goal_ind = None
        self.upper_body_xml_name = info_props["upper_body_xml_name"]

        super().__init__(info_props, **kwargs)

    @staticmethod
    def get_name():
        return "goal_traj_arrow"

    @property
    def has_visual(self):
        return True

    def initialize(self, goal_dict, trajectories=None):
        assert trajectories is not None
        self._traj_goal_ind = np.concatenate([ind for k, ind in trajectories.keys2ind.items() if k == "VEL_2D"])
        self._data_goal_userdata_ind = goal_dict[self._name_goal_dict].data_type_ind
        self.initialized = True

    @property
    def requires_trajectory(self):
        return True

    def reset(self, data, backend, trajectory=None, traj_state=None):
        return self.set_data(data, backend, trajectory, traj_state)

    def set_data(self, data, backend, trajectory=None, traj_state=None):
        assert self._traj_goal_ind is not None
        # get the goal from the trajectory
        traj_goal = trajectory[self._traj_goal_ind, traj_state.traj_no, traj_state.subtraj_step_no]
        # set the goal in the userdata
        # if backend == np:
        #     data.userdata[self._data_goal_userdata_ind] = traj_goal
        # elif backend == jnp:
        #     data = data.replace(userdata=data.userdata.at[self._data_goal_userdata_ind].set(traj_goal))
        #
        data = self.set_attr_data_compat(data, backend, "userdata", traj_goal, self._data_goal_userdata_ind)
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

    def apply_xml_modifications(self, xml_handle, root_body_name):
        # apply the default modifications needed to store the goal in data
        self.allocate_user_data(xml_handle)
        # optionally add sites for visualization
        if self.visualize_goal:
            # add sites for visualization
            root_body_name = self.upper_body_xml_name
            root = xml_handle.find("body", root_body_name)
            # use two spheres to represent an arrow
            point_properties = dict(type="sphere", group=0, rgba=[1.0, 0.0, 0.0, 1.0])
            k1_pos, k2_pos = [0.0, 0.0, 1], [0.5, 0.0, 1]
            root.add("site", name=self._site_name_keypoint_1, pos=k1_pos, size=[0.05], **point_properties)
            root.add("site", name=self._site_name_keypoint_2, pos=k2_pos, size=[0.03], **point_properties)
        return xml_handle

    @property
    def spec(self):
        goal_spec = [(self._name_goal_dict, None, GoalType.VEL_2D, None, None)]     # userdata does not have xml_name
        return goal_spec

    @property
    def size_user_data(self):
        return 2


class GoalDirectionVelocity:

    def __init__(self):
        self._direction = None
        self._velocity = None

    def __call__(self):
        return self.get_goal()

    def get_goal(self):
        assert self._direction is not None
        assert self._velocity is not None
        return deepcopy(self._direction), deepcopy(self._velocity)

    def set_goal(self, direction, velocity):
        self._direction = direction
        self._velocity = velocity

    def get_direction(self):
        assert self._direction is not None
        return deepcopy(self._direction)

    def get_velocity(self):
        assert self._velocity is not None
        return deepcopy(self._velocity)
