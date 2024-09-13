from copy import deepcopy

import numpy as np
import mujoco
import jax.numpy as jnp


def jnt_name2id(name, model):
    """
    Get the joint ID (in the Mujoco datastructure) from the joint name.
    """
    for i in range(model.njnt):
        j = model.joint(i)
        if j.name == name:
            return i
    raise ValueError(f"Joint name {name} not found in model!")


class ObservationIndexContainer:
    """
    Container for indices of different observation types, used to store indices
    related to observations within Mujoco data structures or observations
    created in an environment.
    """

    def __init__(self):
        """
        Indices for the different observation types in the datastructure.
        """
        self.body_xpos = []
        self.body_xquat = []
        self.body_cvel = []
        self.free_joint_qpos = []
        self.free_joint_qvel = []
        self.joint_qpos = []
        self.joint_qvel = []
        self.site_xpos = []
        self.site_xmat = []
        self.forces = []
        self.goal = []

        self.concatenated_indices = None

    def convert_to_numpy(self):
        """
        Converts all list attributes of the class to NumPy arrays in place.
        """
        ind = []
        # Iterate through all attributes of the class
        for attr_name in vars(self):
            attr_value = getattr(self, attr_name)
            if attr_value is not None:
                ind += attr_value
            # Check if the attribute is a list before converting
            if isinstance(attr_value, list):
                setattr(self, attr_name, np.array(attr_value, dtype=int))

        # this array concatenates all indices in the order of this class
        self.concatenated_indices = np.array(ind)


class Obs:
    """
    Base class for all observation types.
    """
    def __init__(self, obs_name: str):
        self.name = obs_name

        # these attributes will be initialized from MjData
        self.obs_ind = None
        self.data_type_ind = None
        self.min, self.max = None, None

    def init_from_mj(self, model, data, current_obs_size):
        """
        Initialize the observation type from the Mujoco data structure and model.

        Args:
            model: The Mujoco model.
            data: The Mujoco data structure.
            current_obs_size: The current size of the observation space.

        Returns:
            The data type indices in the Mujoco data structure and the observation indices.

        """
        raise NotImplementedError

    @classmethod
    def get_obs(cls, model, data, ind, backend):
        """
        Getter for all the observations of this type from the Mujoco datastructure.

        Args:
            model: The Mujoco model.
            data: The Mujoco data structure.
            ind: The indices of *allÜ observations of this types in the datastructure.
            backend: The backend to use for the observation.

        Returns:
            The observation regarding this observation type.

        """
        raise NotImplementedError

    @staticmethod
    def to_list(val):
        """
        Convert the input to a list of integers.
        """
        if isinstance(val, int):
            return [val]
        elif isinstance(val, np.ndarray) and val.dtype == int:
            return val.tolist()
        else:
            raise ValueError("Input must be an integer or a numpy array of integers")


class SimpleObs(Obs):
    """
    See also:
        :class:`Obs` for the base observation class.
    """

    def __init__(self, obs_name: str, xml_name: str):
        self.xml_name = xml_name
        super().__init__(obs_name)


class BodyPos(SimpleObs):
    """
    Observation Type holding x, y, z position of the body.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 3

    def init_from_mj(self, model, data, current_obs_size):
        dim = len(data.body(self.xml_name).xpos)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        data_type_ind = self.to_list(data.body(self.xml_name).id)
        obs_ind = [j for j in range(current_obs_size, current_obs_size + dim)]
        self.data_type_ind = np.array(data_type_ind)
        self.obs_ind = np.array(obs_ind)
        return deepcopy(data_type_ind), deepcopy(obs_ind)

    @classmethod
    def get_obs(cls, model, data, ind, backend):
        return backend.ravel(data.xpos[ind])


class BodyRot(SimpleObs):
    """
    Observation Type holding the quaternion of the body.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 4

    def init_from_mj(self, model, data, current_obs_size):
        dim = len(data.body(self.xml_name).cvel)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        data_type_ind = self.to_list(data.body(self.xml_name).id)
        obs_ind = [j for j in range(current_obs_size, current_obs_size + dim)]
        self.data_type_ind = np.array(data_type_ind)
        self.obs_ind = np.array(obs_ind)
        return deepcopy(data_type_ind), deepcopy(obs_ind)

    @classmethod
    def get_obs(cls, model, data, ind, backend):
        return backend.ravel(data.xquat[ind])


class BodyVel(SimpleObs):
    """
    Observation Type holding the angular velocity around x, y, z and the linear velocity for x, y, z.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 6

    def init_from_mj(self, model, data, current_obs_size):
        dim = len(data.body(self.xml_name).xquat)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        data_type_ind = self.to_list(data.body(self.xml_name).id)
        obs_ind = [j for j in range(current_obs_size, current_obs_size + dim)]
        self.data_type_ind = np.array(data_type_ind)
        self.obs_ind = np.array(obs_ind)
        return deepcopy(data_type_ind), deepcopy(obs_ind)

    @classmethod
    def get_obs(cls, model, data, ind, backend):
        return backend.ravel(data.cvel[ind])


class FreeJointPos(SimpleObs):
    """
    Observation Type holding the 3D position and the 4D quaternion of a free joint.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 7

    def init_from_mj(self, model, data, current_obs_size):
        dim = len(data.joint(self.xml_name).qpos)
        assert dim == self.dim
        # note: free joints do not have limits
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        data_type_ind = self.to_list(data.joint(self.xml_name).id)
        obs_ind = [j for j in range(current_obs_size, current_obs_size + dim)]
        self.data_type_ind = np.array(data_type_ind)
        self.obs_ind = np.array(obs_ind)
        return deepcopy(data_type_ind), deepcopy(obs_ind)

    @classmethod
    def get_obs(cls, model, data, ind, backend):
        return backend.ravel(data.qpos[ind])


class JointPos(SimpleObs):
    """
    Observation Type holding the rotation of the joint.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 1

    def init_from_mj(self, model, data, current_obs_size):
        dim = len(data.joint(self.xml_name).qpos)
        assert dim == self.dim
        jh = model.joint(jnt_name2id(self.xml_name, model))
        if jh.limited:
            self.min, self.max = [jh.range[0]], [jh.range[1]]
        else:
            self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        data_type_ind = self.to_list(data.joint(self.xml_name).id)
        obs_ind = [j for j in range(current_obs_size, current_obs_size + dim)]
        self.data_type_ind = np.array(data_type_ind)
        self.obs_ind = np.array(obs_ind)
        return deepcopy(data_type_ind), deepcopy(obs_ind)

    @classmethod
    def get_obs(cls, model, data, ind, backend):
        return backend.ravel(data.qpos[ind])


class FreeJointVel(SimpleObs):
    """
    Observation Type holding the 3D linear velocity and the 3D angular velocity of a free joint.
    Note: Different to the BODY_VEL observation type!

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 6

    def init_from_mj(self, model, data, current_obs_size):
        dim = len(data.joint(self.xml_name).qvel)
        assert dim == self.dim
        # note: free joints do not have limits
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        data_type_ind = self.to_list(data.joint(self.xml_name).id)
        obs_ind = [j for j in range(current_obs_size, current_obs_size + dim)]
        self.data_type_ind = np.array(data_type_ind)
        self.obs_ind = np.array(obs_ind)
        return deepcopy(data_type_ind), deepcopy(obs_ind)

    @classmethod
    def get_obs(cls, model, data, ind, backend):
        return backend.ravel(data.qvel[ind])


class JointVel(SimpleObs):
    """
    Observation Type holding the velocity of the joint.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 1

    def init_from_mj(self, model, data, current_obs_size):
        dim = len(data.joint(self.xml_name).qvel)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        data_type_ind = self.to_list(data.joint(self.xml_name).id)
        obs_ind = [j for j in range(current_obs_size, current_obs_size + dim)]
        self.data_type_ind = np.array(data_type_ind)
        self.obs_ind = np.array(obs_ind)
        return deepcopy(data_type_ind), deepcopy(obs_ind)

    @classmethod
    def get_obs(cls, model, data, ind, backend):
        return backend.ravel(data.qvel[ind])


class SitePos(SimpleObs):
    """
    Observation Type holding the x, y, z position of the site.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 3

    def init_from_mj(self, model, data, current_obs_size):
        dim = len(data.site(self.xml_name).xpos)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        data_type_ind = self.to_list(data.site(self.xml_name).id)
        obs_ind = [j for j in range(current_obs_size, current_obs_size + dim)]
        self.data_type_ind = np.array(data_type_ind)
        self.obs_ind = np.array(obs_ind)
        return deepcopy(data_type_ind), deepcopy(obs_ind)

    @classmethod
    def get_obs(cls, model, data, ind, backend):
        return backend.ravel(data.site_xpos[ind])


class SiteRot(SimpleObs):
    """
    Observation Type holding the flattened rotation matrix of the site.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 9

    def init_from_mj(self, model, data, current_obs_size):
        # Sites don't have rotation quaternion for some reason...
        # x_mat is rotation matrix with shape (9, )
        dim = len(data.site(self.xml_name).xmat)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        data_type_ind = self.to_list(data.site(self.xml_name).id)
        obs_ind = [j for j in range(current_obs_size, current_obs_size + dim)]
        self.data_type_ind = np.array(data_type_ind)
        self.obs_ind = np.array(obs_ind)
        return deepcopy(data_type_ind), deepcopy(obs_ind)

    @classmethod
    def get_obs(cls, model, data, ind, backend):
        return backend.ravel(data.site_xmat[ind])


class Force(Obs):
    """
    Observation Type holding the collision forces/torques [3D force + 3D torque]
    between two geoms.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 6

    def __init__(self, obs_name: str, xml_name_geom1: str, xml_name_geom2: str):
        self.xml_name_geom1 = xml_name_geom1
        self.xml_name_geom2 = xml_name_geom2
        super().__init__(obs_name)

        self.mjx_contact_id = None
        self.data_geom_id1 = None
        self.data_geom_id2 = None

    def init_from_mj(self, model, data, current_obs_size):
        # get all required information from data
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.data_geom_id1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.xml_name_geom1)
        self.data_geom_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.xml_name_geom2)
        data_type_ind = [self.data_geom_id1, self.data_geom_id2]
        obs_ind = [j for j in range(current_obs_size, current_obs_size + self.dim)]
        self.data_type_ind = np.array(data_type_ind)
        self.obs_ind = np.array(obs_ind)
        return deepcopy(data_type_ind), deepcopy(obs_ind)

    @classmethod
    def get_obs(cls, model, data, ind, backend):
        if backend == np:
            return backend.ravel(cls.mj_collision_force(model, data, ind))
        elif backend == jnp:
            return backend.ravel(cls.mjx_collision_force(model, data, ind))
        else:
            raise ValueError(f"Unknown backend {backend}.")

    @staticmethod
    def mj_collision_force(model, data, ind):

        c_array = np.zeros((len(ind), 6), dtype=np.float64)
        for i, geom_ids in enumerate(ind):

            for con_i in range(0, data.ncon):
                con = data.contact[con_i]
                con_geom_ids = (con.geom1, con.geom2)

                if geom_ids == con_geom_ids:
                    mujoco.mj_contactForce(model, data,
                                           con_i, c_array[i])

        return c_array

    @staticmethod
    def mjx_collision_force(model, data, ind):
        # will be added once mjx adds the collision force function to the official release
        c_array = np.zeros((len(ind), 6), dtype=np.float64)
        return c_array


class ObservationType:
    """
    Namespace for all observation types for easy access.
    """
    BodyPos = BodyPos
    BodyRot = BodyRot
    BodyVel = BodyVel
    JointPos = JointPos
    JointVel = JointVel
    FreeJointPos = FreeJointPos
    FreeJointVel = FreeJointVel
    SitePos = SitePos
    SiteRot = SiteRot
    Force = Force
