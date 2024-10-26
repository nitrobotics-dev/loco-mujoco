from dataclasses import replace
import mujoco
import jax.numpy as jnp

from loco_mujoco.trajectory.dataclasses import Trajectory, interpolate_trajectories


class TrajectoryHandler:
    """
    General class to handle Trajectory. It filters and extends the trajectory data to match
    the current model's joints, bodies and sites. The key idea is to ensure that TrajectoryData has the same
    dimensionality and order for all its attributes as in the Mujoco data structure. So TrajectoryData is a
    simplified version of the Mujoco data structure with fewer attributes. This class also automatically
    interpolates the trajectory to the desired control frequency.

    """
    def __init__(self, model, traj_path=None, traj: Trajectory = None, control_dt=0.01,
                 clip_trajectory_to_joint_ranges=False, warn=True):
        """
        Constructor.

        Args:
            model (mjModel): Current model.
            traj_path (string): path with the trajectory for the model to follow. Should be a numpy zipped file (.npz)
                with a 'traj_data' array and possibly a 'split_points' array inside. The 'traj_data'
                should be in the shape (joints x observations). If traj_files is specified, this should be None.
            traj (Trajectory): Datastructure containing all trajectory files. If traj_path is specified, this
                should be None.
            control_dt (float): Model control frequency used to interpolate the trajectory.
            clip_trajectory_to_joint_ranges (bool): If True, the joint positions in the trajectory are clipped
                between the low and high values in the trajectory. todo
            warn (bool): If True, a warning will be raised, if some trajectory ranges are violated. todo

        """

        assert (traj_path is not None) != (traj is not None), ("Please specify either traj_path or "
                                                               "trajectory, but not both.")

        # load data
        if traj_path is not None:
            traj = Trajectory.load(traj_path)

        # filter/extend the trajectory based on the model/data
        traj_data, traj_info = self.filter_and_extend(traj.data, traj.info, model)

        # todo: implement this in observation types in init_from_traj!
        #self.check_if_trajectory_is_in_range(low, high, keys, joint_pos_idx, warn, clip_trajectory_to_joint_ranges)

        self.traj_dt = 1 / traj_info.frequency
        self.control_dt = control_dt

        if self.traj_dt != self.control_dt:
            traj_data, traj_info = interpolate_trajectories(traj_data, traj_info, 1.0 / self.control_dt)

        self.traj = replace(traj, data=traj_data, info=traj_info)

    def len_trajectory(self, traj_ind):
        return self.traj.data.split_points[traj_ind + 1] - self.traj.data.split_points[traj_ind]

    @property
    def n_trajectories(self):
        return len(self.traj.data.split_points) - 1

    @staticmethod
    def filter_and_extend(traj_data, traj_info, model):
        """
        To ensure that the data structure of the current model and the trajectory data have the same dimensionality
        and order for all supported attributes, this function filters the elements present in the trajectory but not
        the current model and extends the trajectory data's joints, bodies and sites with elements present in
        the current model but not the trajectory. It is doing so by adding dummy joints, bodies and sites to the
        trajectory data if they are not present in the trajectory data but in the model. It also reorders the
        joints, bodies and sites based on the model.

        Args:
            traj_data (TrajectoryData): Trajectory data to be filtered and extended.
            traj_info (TrajectoryInfo): Trajectory info to be filtered and extended.
            model (mjModel): Current model.

        Returns:
            TrajectoryData, TrajectoryInfo: Filtered and extended trajectory data and trajectory info.

        """

        # --- filter the trajectory based on the model and data ---
        # get the joint names from current model
        joint_names = set()
        joint_ids = set()
        joint_name2id = dict()
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_names.add(name)
            if name in joint_name2id.keys():
                joint_name2id[name].append(i)   # free joints can have multiple ids
            else:
                joint_name2id[name] = [i]
            joint_ids.add(i)

        # get the body names from current model
        body_names = set()
        body_name2id = dict()
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            body_names.add(name)
            body_name2id[name] = i

        # get the site names from current model
        site_names = set()
        site_name2id = dict()
        for i in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            site_names.add(name)
            site_name2id[name] = i

        joint_to_be_removed = dict()
        for i, j_name in enumerate(traj_info.joint_names):
            if j_name not in joint_names:
                joint_to_be_removed[j_name] = i

        bodies_to_be_removed = dict()
        if traj_info.body_names is not None:
            for i, b_name in enumerate(traj_info.body_names):
                if b_name not in body_names:
                    bodies_to_be_removed[b_name] = i

        site_to_be_removed = dict()
        if traj_info.site_names is not None:
            for i, s_name in enumerate(traj_info.site_names):
                if s_name not in site_names:
                    site_to_be_removed[s_name] = i

        # create new traj_data and traj_info with removed joints, bodies and sites
        if joint_to_be_removed:
            traj_data = traj_data.remove_joints(jnp.array(list(joint_to_be_removed.values())))
            traj_info = traj_info.remove_joints(list(joint_to_be_removed.keys()))
        if bodies_to_be_removed:
            traj_data = traj_data.remove_bodies(jnp.array(list(bodies_to_be_removed.values())))
            traj_info = traj_info.remove_bodies(list(bodies_to_be_removed.keys()))
        if site_to_be_removed:
            traj_data = traj_data.remove_sites(jnp.array(list(site_to_be_removed.values())))
            traj_info = traj_info.remove_sites(list(site_to_be_removed.keys()))

        # --- extend the trajectory data's joints, bodies and sites using the current model and data ---
        for j_name, j_id in zip(joint_names, joint_ids):
            j_type = model.jnt_type[j_id]
            if j_name not in traj_info.joint_names:
                traj_info = traj_info.add_joint(j_name, j_type)
                traj_data = traj_data.add_joint()

        if traj_info.body_names is not None:
            for b_name in body_names:
                if b_name not in traj_info.body_names:
                    b_id = body_name2id[b_name]
                    traj_info = traj_info.add_body(b_name, model.body_rootid[b_id], model.body_weldid[b_id],
                                                   model.body_mocapid[b_id], model.body_pos[b_id],
                                                   model.body_quat[b_id], model.body_ipos[b_id],
                                                   model.body_iquat[b_id])
                    traj_data = traj_data.add_body()

        if traj_info.site_names is not None:
            for s_name in site_names:
                if s_name not in traj_info.site_names:
                    s_id = site_name2id[s_name]
                    traj_info = traj_info.add_site(s_name, model.site_pos[s_id], model.site_quat[s_id],
                                                   model.site_bodyid[s_id])
                    traj_data = traj_data.add_site()

        # --- reorder the joints and bodies based on the model ---
        new_joint_order_names = []
        new_joint_order_ids = []
        for j_name in joint_name2id.keys():
            new_joint_order_names.append(traj_info.joint_names.index(j_name))
            new_joint_order_ids.append(traj_info.joint_name2ind[j_name])

        if traj_info.body_names is not None:
            new_body_order = []
            for b_name in body_name2id.keys():
                new_body_order.append(traj_info.body_names.index(b_name))

        if traj_info.site_names is not None:
            new_site_order = []
            for s_name in site_name2id.keys():
                new_site_order.append(traj_info.site_names.index(s_name))

        traj_info = traj_info.reorder_joints(new_joint_order_names)
        traj_info = traj_info.reorder_bodies(new_body_order) if traj_info.body_names is not None else traj_info
        traj_info = traj_info.reorder_sites(new_site_order) if traj_info.site_names is not None else traj_info
        traj_data = traj_data.reorder_joints(jnp.concatenate(new_joint_order_ids))
        traj_data = traj_data.reorder_bodies(jnp.array(new_body_order)) \
            if traj_info.body_names is not None else traj_data
        traj_data = traj_data.reorder_sites(jnp.array(new_site_order)) \
            if traj_info.site_names is not None else traj_data

        # setup userdata
        traj_data = traj_data.set_userdata(model.nuserdata)

        return traj_data, traj_info

    # def check_if_trajectory_is_in_range(self, low, high, keys, j_idx, warn, clip_trajectory_to_joint_ranges):
    #
    #     if warn or clip_trajectory_to_joint_ranges:
    #
    #         # get q_pos indices
    #         j_idx = j_idx[2:]   # exclude x and y
    #         highs = dict(zip(keys[2:], high))
    #         lows = dict(zip(keys[2:], low))
    #
    #         # check if they are in range
    #         for i, item in enumerate(self._trajectory_files.items()):
    #             k, d = item
    #             if i in j_idx and k in keys:
    #                 if warn:
    #                     clip_message = "Clipping the trajectory into range!" if clip_trajectory_to_joint_ranges else ""
    #                     if np.max(d) > highs[k]:
    #                         warnings.warn("Trajectory violates joint range in %s. Maximum in trajectory is %f "
    #                                       "and maximum range is %f. %s"
    #                                       % (k, np.max(d), highs[k], clip_message), RuntimeWarning)
    #                     elif np.min(d) < lows[k]:
    #                         warnings.warn("Trajectory violates joint range in %s. Minimum in trajectory is %f "
    #                                       "and minimum range is %f. %s"
    #                                       % (k, np.min(d), lows[k], clip_message), RuntimeWarning)
    #
    #                 # clip trajectory to min & max
    #                 if clip_trajectory_to_joint_ranges:
    #                     self._trajectory_files[k] = np.clip(self._trajectory_files[k], lows[k], highs[k])
