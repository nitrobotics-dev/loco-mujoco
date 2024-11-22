from copy import deepcopy
import mujoco
import numpy as np
import jax
from loco_mujoco import LocoEnv
from scipy.spatial.transform import Rotation as R


class KeyPointTraj:

    def __init__(self, keys, pos=None, quat=None, correspondence_map=None):

        # some sanity checks
        assert type(keys) == list or type(keys) == tuple or type(keys) == set, "Unsupported type for keys"
        assert len(keys) == len(set(keys)), "Found duplicates in keys!"
        assert correspondence_map is None or (len(correspondence_map) == len(keys) and type(correspondence_map) == dict)
        if pos is None or quat is None:
            assert pos == quat
        else:
            assert len(pos) == len(quat)
            for p, q in zip(pos, quat):
                assert len(p) == len(q)

        self._keys = tuple(keys)
        self._pos = pos if pos is not None else [[] for k in keys]
        self._quat = quat if quat is not None else [[] for k in keys]
        self._correspondence_map = correspondence_map

    def append(self, key, pos, quat):
        assert type(pos) == np.ndarray and pos.shape == (3,)
        assert type(quat) == np.ndarray and quat.shape == (4,)
        ind = self._keys.index(key)
        self._pos[ind].append(pos)
        self._quat[ind].append(quat)

    def get_all_pos(self, key):
        ind = self._keys.index(key)
        return deepcopy(self._pos[ind])

    def get_all_quat(self, key):
        ind = self._keys.index(key)
        return deepcopy(self._quat[ind])

    def get_ith_pos(self, key, i):
        ind = self._keys.index(key)
        return deepcopy(self._pos[ind][i])

    def get_ith_quat(self, key, i):
        ind = self._keys.index(key)
        return deepcopy(self._quat[ind][i])

    def apply_affine_trans_to_pos(self, key, mult, offset):
        mult = np.array(mult)
        offset = np.array(offset)
        assert mult.shape == (3,) and offset.shape == (3,)
        ind = self._keys.index(key)
        self._pos[ind] = (self._pos[ind] * mult) + offset

    def apply_affine_trans_to_pos_all(self, mult, offset, except_keys=None):
        mult = np.array(mult)
        offset = np.array(offset)
        assert mult.shape == (3,) and offset.shape == (3,)
        assert except_keys is None or type(except_keys) == list
        except_keys_ind = [] if except_keys is None else [self._keys.index(k) for k in except_keys]
        for i in range(len(self._pos)):
            if i not in except_keys_ind:
                self._pos[i] = (self._pos[i] * mult) + offset

    def apply_rot_to_quat(self, key, rot, seq="xyz", degrees=False):
        rot = np.array(rot)
        assert rot.shape == (3,)
        ind = self._keys.index(key)
        old_quat = self._quat[ind]
        rot_old = R.from_quat(old_quat)
        rot_transformation = R.from_euler(angles=rot, seq=seq, degrees=degrees)
        combined_rotation = rot_transformation * rot_old
        self._quat[ind] = combined_rotation.as_quat()

    def add_correspondence_map(self, cmap):
        assert len(cmap) == len(self._keys)
        assert type(cmap) == dict
        self._correspondence_map = deepcopy(cmap)

    def get_ith_ckey(self, key):
        assert self._correspondence_map is not None
        return self._correspondence_map[key]

    @property
    def keys(self):
        return self._keys


class Base:

    """This class is just for demonstration what the callback does"""
    def __init__(self, env):
        self.env = env

    def __call__(self, model, data, sample):
        sample = [s for s in sample]
        self.env._set_sim_state(data, np.array(sample))

        mujoco.mj_forward(model, data)


class RecordKeyPoints(Base):

    def __init__(self, env, keys):
        super(RecordKeyPoints, self).__init__(env)
        self._key_point_traj = KeyPointTraj(keys)

    def __call__(self, model, data, sample):
        super(RecordKeyPoints, self).__call__(model, data, sample)
        for k in self._key_point_traj.keys:
            pos = deepcopy(data.body(k).xpos)
            quat = deepcopy(data.body(k).xquat)
            self._key_point_traj.append(k, pos, quat)

    def get_data(self):
        return deepcopy(self._key_point_traj)


class Retarget(Base):

    def __init__(self, env, target_data):
        super(Retarget, self).__init__(env)
        self._target_data = target_data
        self._counter = 0

    def __call__(self, model, data, sample):
        data.eq_active = 1
        for k in self._target_data.keys:
            mocap_body_name = self._target_data.get_ith_ckey(k)
            mocap_body_ind = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mocap_body_name) - (model.nbody - model.nmocap)
            data.mocap_pos[mocap_body_ind] = self._target_data.get_ith_pos(k, self._counter)
            data.mocap_quat[mocap_body_ind] = self._target_data.get_ith_quat(k, self._counter)
        mujoco.mj_step(model, data, 100)
        self._counter += 1


def add_mocap_bodies(xml_handle, mocap_ref_bodies, mocap_bodies, mocap_bodies_init_pos):

    for k in mocap_ref_bodies:
        b_handle = xml_handle.find("body", k)
        b_handle.add("site", name=k + "_mocap_ref", type="box", size=[0.1, 0.05, 0.01],
                     rgba=[0.0, 0.5, 1.0, 0.5], group=1)

    for k, p in zip(mocap_bodies, mocap_bodies_init_pos):
        b_handle = xml_handle.worldbody.add("body", name=k, mocap=True, pos=p)
        b_handle.add("site", type="box", size=[0.1, 0.05, 0.01], rgba=[1.0, 0.0, 0.0, 0.5], group=1)

    for b1, b2 in zip(mocap_ref_bodies, mocap_bodies):
        xml_handle.equality.add("weld", body1=b1, body2=b2)

    return xml_handle


if __name__ == "__main__":
    """
    ---- 1. Step: Extract keypoint positions and orientation from a given dataset ----
        
        Here we rollout one joint space trajectory from LocoMujoco and record the positions and orientation of some
        predefined bodies. 
        
        In future, this will be replaced with the AMASS dataset.
         
    """

    # define the bodies you want to track. Here, we use the HumanoidTorque environment.
    keys_orig = ["toes_l", "toes_r", "pelvis", "torso", "hand_l", "hand_r"]
    env = LocoEnv.make("HumanoidTorque.walk", disable_arms=False)
    data_recorder = RecordKeyPoints(env, keys_orig)     # callback class to record the keypoints
    env.play_trajectory(n_episodes=1, n_steps_per_episode=10000, render=False, callback_class=data_recorder)
    data = data_recorder.get_data()

    """
    ---- 2. Step: Apply some transformations to the dataset ----
        Do some simple transformation to adapt the dataset to the size of the G1 robot.
        Right now, this is done with global transformation, should be done on local positions in future. 

    """
    # apply affine transformations to match the G1 embodiment
    data.apply_affine_trans_to_pos("pelvis", mult=[1.0, 1.0, 0.75], offset=[0.0, 0.0, -0.10])
    data.apply_affine_trans_to_pos("torso", mult=[1.0, 1.0, 0.75], offset=[0.0, 0.0, -0.10])
    data.apply_affine_trans_to_pos("hand_l", mult=[1.0, 1.0, 0.75], offset=[0.2, 0.0, -0.2])
    data.apply_affine_trans_to_pos("hand_r", mult=[1.0, 1.0, 0.75], offset=[0.2, 0.0, -0.2])
    data.apply_affine_trans_to_pos_all(mult=[0.725, 1.0, 1.0], offset=[0.0, 0.0, 0.0])

    # apply rotations to quaternions to match the G1 embodiment
    data.apply_rot_to_quat("toes_l", [0.0, 0.0, -90.0], degrees=True)
    data.apply_rot_to_quat("toes_r", [0.0, 0.0, -90.0], degrees=True)
    data.apply_rot_to_quat("pelvis", [0.0, 0.0, -90.0], degrees=True)
    data.apply_rot_to_quat("torso", [0.0, 0.0, -90.0], degrees=True)
    data.apply_rot_to_quat("hand_l", [0.0, -90.0, 0.0], degrees=True)
    data.apply_rot_to_quat("hand_r", [0.0, 90.0, 0.0], degrees=True)

    """
        ---- 3. Step: Define correspondences between the two models ----

    """
    # define a correspondence map that defines what bodies from the HumanoidTorque model correspond to
    # the bodies from the G1 robot
    correspondence_map = dict(pelvis="pelvis", torso="torso_link", toes_l="left_ankle_roll_link",
                              toes_r="right_ankle_roll_link", hand_l="left_zero_link", hand_r="right_zero_link")

    # names of the reference bodies in G1
    mocap_ref_bodies = correspondence_map.values()

    # names of the mocap body names to be added to the G1 model
    mocap_bodies = [k+"_mocap" for k in correspondence_map.keys()]

    """
        ---- 4. Step: Retargeting. ----
            Add motion capture bodies and weld constraints to the G1 robot. The motion capture bodies initial positions
            are set to the initial positions of the reference bodies from G1. 

    """

    # create the G1 environment and reset
    env2 = LocoEnv.make("UnitreeG1.walk", random_start=False)
    rng = jax.random.key(0)
    rng, _rng = jax.random.split(rng)
    env2.reset(_rng)

    # get the global positions of all reference bodies from G1
    mocap_bodies_init_pos = [env2._data.body(k).xpos for k in mocap_ref_bodies]

    # add the mocap bodies and the weld constraints to the xml (they way it is added is not nice,
    # just a workaround for now)
    xml_handle = env2._xml_handles[0]
    xml_handle = add_mocap_bodies(xml_handle, mocap_ref_bodies, mocap_bodies, mocap_bodies_init_pos)
    env2._xml_handles = [xml_handle]
    env2.reload_model()

    # define the retargeting callback and replay the data
    mocap_correspondence_map = dict(zip(correspondence_map.keys(), mocap_bodies))
    data.add_correspondence_map(mocap_correspondence_map)
    retarget_vb = Retarget(env2, data)
    env2.play_trajectory(render=True, callback_class=retarget_vb, record=True)



