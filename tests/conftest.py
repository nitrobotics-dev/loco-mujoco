import pytest
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import MjModel, MjData

from loco_mujoco.trajectory import (
    TrajectoryInfo,
    TrajectoryModel,
    TrajectoryData,
    TrajectoryTransitions,
)


@pytest.fixture
def input_trajectory_info_data() -> TrajectoryInfo:
    def factory(backend):
        backend_type = jnp if backend == "jax" else np
        model_path = "test_datasets/humanoid_test.xml"
        mujoco_model = MjModel.from_xml_path(model_path)

        njnt = mujoco_model.njnt
        nbody = mujoco_model.nbody
        nsite = mujoco_model.nsite
        frequency = 100.0

        joint_names = [
            mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(njnt)
        ]
        jnt_type = backend_type.array(mujoco_model.jnt_type)

        body_names = [
            mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(nbody)
        ]
        body_pos = backend_type.array(mujoco_model.body_pos)
        body_quat = backend_type.array(mujoco_model.body_quat)
        body_ipos = backend_type.array(mujoco_model.body_ipos)
        body_iquat = backend_type.array(mujoco_model.body_iquat)
        body_rootid = backend_type.array(mujoco_model.body_rootid)
        body_weldid = backend_type.array(mujoco_model.body_weldid)
        body_mocapid = backend_type.array(mujoco_model.body_mocapid)

        site_names = [
            mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_SITE, i)
            for i in range(nsite)
        ]
        site_bodyid = backend_type.array(mujoco_model.site_bodyid)
        site_pos = backend_type.array(mujoco_model.site_pos)
        site_quat = backend_type.array(mujoco_model.site_quat)

        metadata = {
            "robot_name": "minimal bipedal robot",
            "description": "Test trajectory for a bipedal locomotion robot",
        }

        model = TrajectoryModel(
            njnt,
            jnt_type,
            nbody,
            body_rootid,
            body_weldid,
            body_mocapid,
            body_pos,
            body_quat,
            body_ipos,
            body_iquat,
            nsite,
            site_bodyid,
            site_pos,
            site_quat,
        )

        return TrajectoryInfo(
            joint_names, model, frequency, body_names, site_names, metadata
        )

    return factory


@pytest.fixture
def input_expected_joint_name2ind_qpos():
    def factory(backend):
        backend_type = jnp if backend == "jax" else np
        return {
            "root": backend_type.array([0, 1, 2, 3, 4, 5, 6]),
            "abdomen_z": backend_type.array([7]),
            "abdomen_y": backend_type.array([8]),
            "abdomen_x": backend_type.array([9]),
            "right_hip_x": backend_type.array([10]),
            "right_hip_z": backend_type.array([11]),
            "right_hip_y": backend_type.array([12]),
            "right_knee": backend_type.array([13]),
            "left_hip_x": backend_type.array([14]),
            "left_hip_z": backend_type.array([15]),
            "left_hip_y": backend_type.array([16]),
            "left_knee": backend_type.array([17]),
        }

    return factory


@pytest.fixture
def input_body_name2ind():
    def factory(backend):
        backend_type = jnp if backend == "jax" else np
        return {
            "world": backend_type.array([0]),
            "torso": backend_type.array([1]),
            "lwaist": backend_type.array([2]),
            "pelvis": backend_type.array([3]),
            "right_thigh": backend_type.array([4]),
            "right_shin": backend_type.array([5]),
            "right_foot": backend_type.array([6]),
            "left_thigh": backend_type.array([7]),
            "left_shin": backend_type.array([8]),
            "left_foot": backend_type.array([9]),
        }

    return factory


@pytest.fixture
def input_site_name2ind():
    def factory(backend):
        backend_type = jnp if backend == "jax" else np
        return {
            "torso_site": backend_type.array([0]),
            "pelvis_site": backend_type.array([1]),
            "right_thigh_site": backend_type.array([2]),
            "right_foot_site": backend_type.array([3]),
            "left_thigh_site": backend_type.array([4]),
            "left_foot_site": backend_type.array([5]),
        }

    return factory


@pytest.fixture
def input_trajectory_info_field_names():
    return ["joint_names", "model", "frequency", "body_names", "site_names", "metadata"]


@pytest.fixture
def input_trajectory_data() -> TrajectoryData:
    def factory(backend):
        backend_type = jnp if backend == "jax" else np
        model_path = "test_datasets/humanoid_test.xml"
        mujoco_model = MjModel.from_xml_path(model_path)

        data = MjData(mujoco_model)
        N_steps = 1000

        qpos = data.qpos
        qvel = data.qvel
        qpos = backend_type.tile(qpos, (N_steps, 1))
        qvel = backend_type.tile(qvel, (N_steps, 1))

        cvel = data.cvel
        cvel = backend_type.tile(cvel, (N_steps, 1, 1))

        xpos = data.xpos
        xpos = backend_type.tile(xpos, (N_steps, 1, 1))

        xquat = data.xquat
        xquat_norms = backend_type.linalg.norm(xquat, axis=-1)
        xquat = backend_type.where(
            xquat_norms[:, None] == 0, backend_type.array([1, 0, 0, 0]), xquat
        )
        xquat = backend_type.tile(xquat, (N_steps, 1, 1))

        subtree_com = data.subtree_com
        subtree_com = backend_type.tile(subtree_com, (N_steps, 1, 1))

        site_xpos = data.site_xpos
        site_xpos = backend_type.tile(site_xpos, (N_steps, 1, 1))
        site_xmat = data.site_xmat
        site_xmat = backend_type.tile(site_xmat, (N_steps, 1, 1))

        userdata = data.userdata
        userdata = backend_type.tile(userdata, (N_steps, 1))

        trajectoryData: TrajectoryData = TrajectoryData(
            qpos,
            qvel,
            xpos,
            xquat,
            cvel,
            subtree_com,
            site_xpos,
            site_xmat,
            userdata,
            split_points=backend_type.array([0, N_steps]),
        )

        return trajectoryData

    return factory


@pytest.fixture
def input_trajectory_transitions():
    def factory(backend):
        backend_type = jnp if backend == "jax" else np

        observations = backend_type.array([[1.0, 2.0], [3.0, 4.0]])
        next_observations = backend_type.array([[5.0, 6.0], [7.0, 8.0]])
        absorbings = backend_type.array([True, False])
        dones = backend_type.array([False, True])
        actions = backend_type.array([1, 2])
        rewards = backend_type.array([0.5, 1.0])

        return TrajectoryTransitions(
            observations=observations,
            next_observations=next_observations,
            absorbings=absorbings,
            dones=dones,
            actions=actions,
            rewards=rewards,
        )

    return factory
