import numpy as np
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.trajectory import TrajectoryHandler
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast


class TerminalStateHandler:

    registered = dict()

    def __init__(self, model,  info_props):
        self._info_props = info_props

    def is_absorbing(self, obs, info, data, carry):
        """
        Check if the current state is terminal. Function for CPU Mujoco.

        Args:
            obs (np.array): shape (n_samples, n_obs), the observations
            info (dict): the info dictionary
            data: Mujoco data structure
            carry: additional carry.

        Returns:
            Boolean indicating whether the current state is terminal or not.
        """
        raise NotImplementedError

    def mjx_is_absorbing(self, obs, info, data, carry):
        """
        Check if the current state is terminal. Function for Mjx.

        Args:
            obs (np.array): shape (n_samples, n_obs), the observations
            info (dict): the info dictionary
            data: Mujoco data structure
            carry: additional carry.

        Returns:
            Boolean indicating whether the current state is terminal or not.
        """
        raise NotImplementedError

    def _is_absorbing_compat(self, obs, info, data, carry, backend):
        """
        Check if the current state is terminal. Function for CPU Mujoco and Mjx.

        Args:
            obs (np.array): shape (n_samples, n_obs), the observations
            info (dict): the info dictionary
            data: Mujoco data structure
            carry: additional carry.

        Returns:
            Boolean indicating whether the current state is terminal or not.
        """
        raise NotImplementedError

    def init_from_traj(self, th: TrajectoryHandler):
        """
        Initialize the TerminalStateHandler from a Trajectory (Optional).

        Args:
            th (TrajectoryHandler): The trajectory handler containing the trajectory.

        """
        pass

    @classmethod
    def register(cls):
        """
        Register a TerminalStateHandler in the TerminalStateHandler list.

        """
        env_name = cls.__name__

        if env_name not in TerminalStateHandler.registered:
            TerminalStateHandler.registered[env_name] = cls

    @classmethod
    def make(cls, name, *args, **kwargs):
        """
        Create a TerminalStateHandler instance from the TerminalStateHandler list.

        """
        if name not in TerminalStateHandler.registered:
            raise ValueError(f"Unknown TerminalStateHandler: {name}")

        return TerminalStateHandler.registered[name](*args, **kwargs)


class HeightBasedTerminalStateHandler(TerminalStateHandler):
    """
    Check if the current state is terminal based on the height of the root.
    """
    def __init__(self, model, info_props):
        super().__init__(model, info_props)

        self.root_height_range = info_props["root_height_healthy_range"]
        self.root_free_joint_xml_ind = np.array(mj_jntname2qposid(info_props["root_free_joint_xml_name"], model))

    def is_absorbing(self, obs, info, data, carry):
        return self._is_absorbing_compat(obs, info, data, carry, backend=np)

    def mjx_is_absorbing(self, obs, info, data, carry):
        return self._is_absorbing_compat(obs, info, data, carry, backend=jnp)

    def _is_absorbing_compat(self, obs, info, data, carry, backend):
        root_pose = data.qpos[self.root_free_joint_xml_ind]
        height = root_pose[2]
        height_cond = backend.logical_or(backend.less(height, self.root_height_range[0]),
                                         backend.greater(height, self.root_height_range[1]))
        return height_cond


class RootPoseTrajTerminalStateHandler(TerminalStateHandler):

    def __init__(self, model,  info_props, root_height_margin=0.3, root_rot_margin_degrees=30.0):
        self._initialized = False

        self.root_joint_name = info_props["root_free_joint_xml_name"]

        self.root_height_margin = root_height_margin
        self.root_rot_margin_degrees = root_rot_margin_degrees

        # to be determined in init_from_traj
        self.root_height_ind = None
        self.root_quat_ind = None
        self.root_height_range = None
        self._centroid_quat = None
        self._valid_threshold = None

    def init_from_traj(self, th: TrajectoryHandler):
        """
        Initialize the TerminalStateHandler from a Trajectory.

        Args:
            th (TrajectoryHandler): The trajectory handler containing the trajectory.

        """
        assert th is not None, f"{self.__class__.__name__} requires a TrajectoryHandler to be initialized."

        traj = th.traj
        root_ind = traj.info.joint_name2ind_qpos[self.root_joint_name]
        self.root_height_ind = root_ind[2]
        self.root_quat_ind = root_ind[3:7]
        assert len(self.root_quat_ind) == 4

        # get the root quaternions
        root_quats = traj.data.qpos[:, self.root_quat_ind]

        # calculate the centroid of the root quaternions and the maximum angular distance from the centroid
        self._centroid_quat, self._valid_threshold = self._calc_root_rot_centroid_and_margin(
            quat_scalarfirst2scalarlast(root_quats))

        # calculate the range of the root height
        root_height_min = np.min(traj.data.qpos[:, self.root_height_ind])
        root_height_max = np.max(traj.data.qpos[:, self.root_height_ind])
        self.root_height_range = (root_height_min - self.root_height_margin, root_height_max + self.root_height_margin)

        self._initialized = True

    def is_absorbing(self, obs, info, data, carry):
        """
        Check if the current state is terminal. The state is terminal if the root height is outside the range or the
        root rotation is outside the valid threshold. Function for CPU Mujoco.

        Args:
            obs (np.array): shape (n_samples, n_obs), the observations
            info (dict): the info dictionary
            data: Mujoco data structure
            carry: additional carry.

        Returns:
            Boolean indicating whether the current state is terminal or not.

        """
        if self.initialized:
            return self._is_absorbing_compat(obs, info, data, carry, backend=np)
        else:
            return False

    def mjx_is_absorbing(self, obs, info, data, carry):
        """
        Check if the current state is terminal. The state is terminal if the root height is outside the range or the
        root rotation is outside the valid threshold. Function for Mjx.

        Args:
            obs (jnp.array): shape (n_samples, n_obs), the observations
            info (dict): the info dictionary
            data: Mjx data structure
            carry: additional carry.

        Returns:
            Boolean indicating whether the current state is terminal or not.

        """
        if self.initialized:
            return self._is_absorbing_compat(obs, info, data, carry, backend=jnp)
        else:
            return False

    def _is_absorbing_compat(self, obs, info, data, carry, backend):
        """
        Check if the current state is terminal. The state is terminal if the root height is outside the range or the
        root rotation is outside the valid threshold.

        Args:
            obs (np.array): shape (n_samples, n_obs), the observations
            info (dict): the info dictionary
            data: Mujoco data structure
            carry: additional carry.
            backend: the backend to use (np or jnp)

        Returns:
            Boolean indicating whether the current state is terminal or not.

        """
        # get height and rotation of the root joint
        height = data.qpos[self.root_height_ind]
        root_quat = quat_scalarfirst2scalarlast(data.qpos[self.root_quat_ind])

        # check if the root height is outside the range
        height_cond = backend.logical_or(backend.less(height, self.root_height_range[0]),
                                         backend.greater(height, self.root_height_range[1]))

        # check if the root rotation is outside the valid threshold
        root_quat = root_quat / backend.linalg.norm(root_quat)
        angular_distance = 2 * backend.arccos(backend.clip(backend.dot(self._centroid_quat, root_quat), -1, 1))
        root_rot_cond = backend.greater(angular_distance, self._valid_threshold)

        return backend.logical_or(height_cond, root_rot_cond)

    def _calc_root_rot_centroid_and_margin(self, root_quats):
        """
        Calculate the centroid of the root quaternions and the maximum angular distance from the centroid.

        Args:
            root_quats: np.array, shape (n_samples, 4), the root quaternions. (quaternions is expected to be scalar last)

        Returns:
            centroid_quat: np.array, shape (4,), the centroid of the quaternions, where the quaternions are scalar last.
            valid_threshold: float, the maximum angular distance from the centroid.
        """

        # normalize them
        norm_root_quats = root_quats / np.linalg.norm(root_quats, axis=1, keepdims=True)

        # compute centroid of the quaternions
        r = np_R.from_quat(norm_root_quats)
        centroid_quat = r.mean().as_quat()

        # Compute maximum deviation in angular distance
        dot_products = np.clip(np.einsum('ij,j->i', norm_root_quats,
                                         centroid_quat), -1, 1)
        angular_distances = 2 * np.arccos(dot_products)

        max_distance = np.max(angular_distances)

        # Add margin
        valid_threshold = max_distance + np.radians(self.root_rot_margin_degrees)

        return centroid_quat, valid_threshold

    @property
    def initialized(self):
        return self._initialized
