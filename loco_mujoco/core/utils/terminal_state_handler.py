import numpy as np
import jax.numpy as jnp
import mujoco
from jax.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.trajectory import TrajectoryHandler


class TerminalStateHandler:

    registered = dict()

    def __init__(self, model,  info_props):
        self._info_props = info_props

    def is_absorbing(self, obs, info, data, carry):
        """
        Check if the current state is terminal.
        """
        raise NotImplementedError

    def mjx_is_absorbing(self, obs, info, data, carry):
        """
        Check if the current state is terminal.
        """
        raise NotImplementedError

    def _is_absorbing_compat(self, obs, info, data, carry, backend):
        """
        Check if the current state is terminal.
        """
        raise NotImplementedError

    def init_from_traj(self, th: TrajectoryHandler):
        """
        Initialize the TerminalStateHandler from a Trajectory (optional).

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
    Check if the current state is terminal based on the height of the pelvis.
    """
    def __init__(self, model, info_props):
        super().__init__(model, info_props)

        self.root_height_range = info_props["root_height_healthy_range"]
        root_free_joint_xml_ind = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, info_props["root_free_joint_xml_name"])
        assert root_free_joint_xml_ind != -1
        self.root_free_joint_xml_ind = np.arange(root_free_joint_xml_ind, root_free_joint_xml_ind + 7)

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

    def __init__(self, model,  info_props, root_height_slack=0.2, root_rot_slack=1.0):
        self._initialized = False

        self.root_joint_name = info_props["root_free_joint_xml_name"]

        self.root_height_slack = root_height_slack
        self.root_rot_slack = root_rot_slack

        # to be determined in init_from_traj
        self.root_height_ind = None
        self.root_quat_ind = None
        self.root_height_range = None
        self.root_rotX_range = None
        self.root_rotY_range = None
        self.root_rotZ_range = None

    def init_from_traj(self, th: TrajectoryHandler):
        traj = th.traj
        root_ind = traj.info.joint_name2ind_qpos[self.root_joint_name]
        self.root_height_ind = root_ind[2]
        self.root_quat_ind = root_ind[3:7]
        assert len(self.root_quat_ind) == 4

        # convert quaternion to euler angles
        traj_root_quat = traj.data.qpos[:, self.root_quat_ind]
        traj_root_euler = np_R.from_quat(traj_root_quat).as_rotvec()

        # calculate the range of the root pose data
        root_height_min = np.min(traj.data.qpos[:, self.root_height_ind])
        root_height_max = np.max(traj.data.qpos[:, self.root_height_ind])
        root_rot_x_min = np.min(np.fmod(traj_root_euler[:, 0], np.pi))
        root_rot_x_max = np.max(np.fmod(traj_root_euler[:, 0], np.pi))
        root_rot_y_min = np.min(np.fmod(traj_root_euler[:, 1], np.pi))
        root_rot_y_max = np.max(np.fmod(traj_root_euler[:, 1], np.pi))
        root_rot_z_min = np.min(np.fmod(traj_root_euler[:, 2], np.pi))
        root_rot_z_max = np.max(np.fmod(traj_root_euler[:, 2], np.pi))

        # calc range and add slack to the range
        self.root_height_range = (root_height_min-self.root_height_slack, root_height_max+self.root_height_slack)
        self.root_rotX_range = (root_rot_x_min-self.root_rot_slack, root_rot_x_max+self.root_rot_slack)
        self.root_rotY_range = (root_rot_y_min-self.root_rot_slack, root_rot_y_max+self.root_rot_slack)
        self.root_rotZ_range = (root_rot_z_min-self.root_rot_slack, root_rot_z_max+self.root_rot_slack)

        self._initialized = True

    def is_absorbing(self, obs, info, data, carry):
        if self.initialized:
            return self._is_absorbing_compat(obs, info, data, carry, backend=np)
        else:
            return False

    def mjx_is_absorbing(self, obs, info, data, carry):
        if self.initialized:
            return self._is_absorbing_compat(obs, info, data, carry, backend=jnp)
        else:
            return False

    def _is_absorbing_compat(self, obs, info, data, carry, backend):

        # get rotation backend
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        # get height and rotation of the root joint
        height = data.qpos[self.root_height_ind]
        root_quat = data.qpos[self.root_quat_ind]
        root_euler = R.from_quat(root_quat).as_rotvec()
        root_rot_x = backend.fmod(root_euler[0], backend.pi)
        root_rot_y = backend.fmod(root_euler[1], backend.pi)
        root_rot_z = backend.fmod(root_euler[2], backend.pi)

        # check if the root pose is outside the range
        height_cond = backend.logical_or(backend.less(height, self.root_height_range[0]),
                                         backend.greater(height, self.root_height_range[1]))
        root_rot_x_cond = backend.logical_or(backend.less(root_rot_x, self.root_rotX_range[0]),
                                             backend.greater(root_rot_x, self.root_rotX_range[1]))
        root_rot_y_cond = backend.logical_or(backend.less(root_rot_y, self.root_rotY_range[0]),
                                             backend.greater(root_rot_y, self.root_rotY_range[1]))
        root_rot_z_cond = backend.logical_or(backend.less(root_rot_z, self.root_rotZ_range[0]),
                                             backend.greater(root_rot_z, self.root_rotZ_range[1]))
        
        return backend.logical_or(backend.logical_or(height_cond, root_rot_x_cond),
                                  backend.logical_or(root_rot_y_cond, root_rot_z_cond))

    @property
    def initialized(self):
        return self._initialized
