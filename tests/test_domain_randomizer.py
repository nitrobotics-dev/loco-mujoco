from mujoco import mjx
import jax
import numpy as np
import numpy.random
import jax.numpy as jnp
import pytest
import os
from math import isclose

from test_conf import DummyHumamoidEnv
from loco_mujoco.core.domain_randomizer.default import DefaultRandomizerState
from copy import deepcopy

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# load yaml as dict:
path = parent_dir + "/tests/test_conf/default_dom_rand_conf.yaml"
with open(path, "r") as file:
    import yaml

    default_dom_rand_conf = yaml.load(file, Loader=yaml.FullLoader)

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_init_state(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    default_randomizer_state: DefaultRandomizerState = None
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        default_randomizer_state = mjx_env._domain_randomizer.init_state(
            mjx_env, key, mjx_env._model, mjx_env._data, backend
        )
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)

        default_randomizer_state = mjx_env._domain_randomizer.init_state(
            mjx_env, key, mjx_env._model, state.data, backend
        )

    np.testing.assert_allclose(
        default_randomizer_state.gravity, backend.array([0.0, 0.0, -9.81])
    )
    np.testing.assert_allclose(
        default_randomizer_state.geom_friction,
        backend.array(mjx_env._model.geom_friction.copy()),
    )
    np.testing.assert_allclose(
        default_randomizer_state.geom_stiffness, backend.zeros(mjx_env._model.ngeom)
    )
    np.testing.assert_allclose(
        default_randomizer_state.geom_damping, backend.zeros(mjx_env._model.ngeom)
    )
    np.testing.assert_allclose(
        default_randomizer_state.com_displacement, backend.array([0.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(
        default_randomizer_state.link_mass_multipliers,
        backend.array([1.0] * (mjx_env._model.nbody - 1)),
    )
    np.testing.assert_allclose(
        default_randomizer_state.joint_friction_loss,
        backend.array([0.0] * (mjx_env._model.nv - 6)),
    )
    np.testing.assert_allclose(
        default_randomizer_state.joint_damping,
        backend.array([0.0] * (mjx_env._model.nv - 6)),
    )
    np.testing.assert_allclose(
        default_randomizer_state.joint_armature,
        backend.array([0.0] * (mjx_env._model.nv - 6)),
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_update(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        initial_model = deepcopy(mjx_env._model)
        model, _, _ = mjx_env._domain_randomizer.update(
            mjx_env, mjx_env._model, mjx_env._data, mjx_env._additional_carry, backend
        )
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        initial_model = deepcopy(mjx_env._model)

        sys = mjx.put_model(mjx_env._model)

        model, _, _ = mjx_env._domain_randomizer.update(
            mjx_env, sys, state.data, state.additional_carry, backend
        )

    # Ensure that values have changed
    assert not np.allclose(
        model.geom_friction, initial_model.geom_friction, atol=1e-6
    ), "Geom friction should not match"
    assert not np.allclose(
        model.geom_solref, initial_model.geom_solref, atol=1e-6
    ), "Geom solref should not match"
    assert not np.allclose(
        model.body_ipos, initial_model.body_ipos, atol=1e-6
    ), "Body position should be modified"
    assert not np.allclose(
        model.body_mass, initial_model.body_mass, atol=1e-6
    ), "Body mass should be modified"
    assert not np.allclose(
        model.dof_frictionloss, initial_model.dof_frictionloss, atol=1e-6
    ), "Joint friction loss should be modified"
    assert not np.allclose(
        model.dof_damping, initial_model.dof_damping, atol=1e-6
    ), "Joint damping should be modified"
    assert not np.allclose(
        model.dof_armature, initial_model.dof_armature, atol=1e-6
    ), "Joint armature should be modified"


OBSERVATION_SPACE = [
    {"obs_name": "name_obs3", "type": "BodyVel", "xml_name": "left_thigh"},
    {"obs_name": "name_obs4", "type": "BodyVel", "xml_name": "right_shin"},
    {"obs_name": "name_obs5", "type": "ProjectedGravityVector", "xml_name": "root"},
    {"obs_name": "name_obs6", "type": "JointPos", "xml_name": "left_hip_y"},
    {"obs_name": "name_obs7", "type": "JointVel", "xml_name": "left_hip_y"},
    {"obs_name": "name_obs8", "type": "JointPos", "xml_name": "right_hip_y"},
    {"obs_name": "name_obs9", "type": "JointVel", "xml_name": "right_hip_y"},
    {"obs_name": "name_obs10", "type": "FreeJointVel", "xml_name": "root"},
]


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_update_observation(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    body_name1 = "left_shin"
    body_name2 = "right_thigh"

    # specify the observation space
    observation_spec = [
        {"obs_name": "name_obs1", "type": "BodyPos", "xml_name": body_name1},
        {"obs_name": "name_obs2", "type": "BodyPos", "xml_name": body_name2},
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        observation_spec=observation_spec,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        initial_carry = mjx_env._additional_carry

        obs_updated, carry = mjx_env._domain_randomizer.update_observation(
            mjx_env,
            obs,
            mjx_env._model,
            mjx_env._data,
            mjx_env._additional_carry,
            backend,
        )

        np.testing.assert_allclose(
            obs_updated,
            [
                -0.02511085,
                0.08873329,
                0.425,
                0.00726929,
                -0.09849757,
                0.828,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.00240988,
                -0.01288478,
                -1.00309631,
                0.0018594,
                0.01368004,
                0.00537351,
                -0.13805371,
                0.04884265,
                -0.0838331,
                0.03811637,
                -0.01981807,
                0.0203576,
                0.00683175,
            ],
            atol=1e-7,
        )
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs = state.observation
        initial_carry = state.additional_carry

        obs_updated, carry = mjx_env._domain_randomizer.update_observation(
            mjx_env, obs, mjx_env._model, state.data, state.additional_carry, backend
        )

        np.testing.assert_allclose(
            obs_updated,
            [
                -0.02511085,
                0.0887333,
                0.42499998,
                0.00726929,
                -0.09849758,
                0.82799995,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.00418337,
                -0.02806187,
                -0.99809885,
                -0.00272283,
                0.11321597,
                -0.00228598,
                -0.12698306,
                -0.01148276,
                0.05986609,
                0.19948058,
                0.04477494,
                0.02922314,
                0.04064021,
            ],
            atol=1e-7,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_update_action(backend):
    pass


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_geom_friction(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        geom_friction, _ = mjx_env._domain_randomizer._sample_geom_friction(
            mjx_env._model, mjx_env._additional_carry, backend
        )

        np.testing.assert_allclose(
            geom_friction,
            [
                [1.07698885e00, 5.76988848e-03, 1.07698885e-04],
                [1.02664058e00, 5.26640582e-03, 1.02664058e-04],
                [9.06155796e-01, 4.06155796e-03, 9.06155796e-05],
                [1.00929922e00, 5.09299221e-03, 1.00929922e-04],
                [8.37576204e-01, 3.37576204e-03, 8.37576204e-05],
                [1.03037860e00, 5.30378598e-03, 1.03037860e-04],
                [1.17171848e00, 6.71718479e-03, 1.17171848e-04],
                [9.27427581e-01, 4.27427581e-03, 9.27427581e-05],
                [1.06696415e00, 5.66964152e-03, 1.06696415e-04],
                [8.52719145e-01, 3.52719145e-03, 8.52719145e-05],
                [1.08653088e00, 5.86530882e-03, 1.08653088e-04],
                [9.15762437e-01, 4.15762437e-03, 9.15762437e-05],
            ],
            atol=1e-7,
        )
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        geom_friction, _ = mjx_env._domain_randomizer._sample_geom_friction(
            mjx_env._model, state.additional_carry, backend
        )

        np.testing.assert_allclose(
            geom_friction,
            [
                [1.14048123e00, 6.40481291e-03, 1.14048125e-04],
                [1.08886874e00, 5.88868745e-03, 1.08886867e-04],
                [8.38817060e-01, 3.38817085e-03, 8.38817068e-05],
                [1.11036968e00, 6.10369723e-03, 1.11036970e-04],
                [8.07135820e-01, 3.07135819e-03, 8.07135802e-05],
                [8.45426857e-01, 3.45426844e-03, 8.45426839e-05],
                [9.31010962e-01, 4.31010965e-03, 9.31010945e-05],
                [9.04455245e-01, 4.04455233e-03, 9.04455228e-05],
                [1.00076437e00, 5.00764325e-03, 1.00076431e-04],
                [1.15421045e00, 6.54210430e-03, 1.15421040e-04],
                [1.03211248e00, 5.32112457e-03, 1.03211241e-04],
                [1.14779615e00, 6.47796178e-03, 1.14779614e-04],
            ],
            atol=1e-7,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_geom_damping_and_stiffness(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        sampled_damping, sampled_stiffness, _ = (
            mjx_env._domain_randomizer._sample_geom_damping_and_stiffness(
                mjx_env._model, mjx_env._additional_carry, backend
            )
        )

        np.testing.assert_allclose(
            sampled_damping,
            [
                83.079554,
                81.065623,
                76.246232,
                80.371969,
                73.503048,
                81.215144,
                86.868739,
                77.097103,
                82.678566,
                74.108766,
                83.461235,
                76.630497,
            ],
            atol=1e-7,
        )

        np.testing.assert_allclose(
            sampled_stiffness,
            [
                936.638272,
                1017.302587,
                904.021509,
                1065.788006,
                900.939095,
                1035.563307,
                954.001595,
                1047.038804,
                1092.437709,
                949.750629,
                1015.231467,
                1018.408386,
            ],
            atol=1e-7,
        )
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        sampled_damping, sampled_stiffness, _ = (
            mjx_env._domain_randomizer._sample_geom_damping_and_stiffness(
                mjx_env._model, state.additional_carry, backend
            )
        )
        np.testing.assert_allclose(
            sampled_damping,
            [
                80.181335,
                81.81961,
                80.06223,
                81.34811,
                87.876854,
                76.93823,
                78.43573,
                82.175674,
                74.420654,
                72.32476,
                81.58818,
                74.52422,
            ],
            atol=1e-7,
        )

        np.testing.assert_allclose(
            sampled_stiffness,
            [
                1063.2867,
                1008.16724,
                903.2543,
                945.1598,
                1035.8601,
                948.56836,
                924.58514,
                922.1524,
                1025.7334,
                1044.9795,
                936.45764,
                908.6345,
            ],
            atol=1e-7,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_joint_friction_loss(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        sampled_friction_loss, _ = (
            mjx_env._domain_randomizer._sample_joint_friction_loss(
                mjx_env._model, mjx_env._additional_carry, backend
            )
        )

        np.testing.assert_allclose(
            sampled_friction_loss,
            [
                0.13849442,
                0.11332029,
                0.0530779,
                0.10464961,
                0.0187881,
                0.1151893,
                0.18585924,
                0.06371379,
                0.13348208,
                0.02635957,
                0.14326544,
            ],
            atol=1e-5,
        )

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        sampled_friction_loss, _ = (
            mjx_env._domain_randomizer._sample_joint_friction_loss(
                mjx_env._model, state.additional_carry, backend
            )
        )

        np.testing.assert_allclose(
            sampled_friction_loss,
            [
                0.17024064,
                0.14443436,
                0.01940854,
                0.15518486,
                0.00356791,
                0.10512004,
                0.06550548,
                0.05222762,
                0.10038216,
                0.17710522,
                0.11605623,
            ],
            atol=1e-7,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_joint_damping(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        sampled_damping, _ = mjx_env._domain_randomizer._sample_joint_damping(
            mjx_env._model, mjx_env._additional_carry, backend
        )
        np.testing.assert_allclose(
            sampled_damping,
            [
                1.13096654,
                0.97992175,
                0.61846739,
                0.92789766,
                0.41272861,
                0.99113579,
                1.41515544,
                0.68228274,
                1.10089246,
                0.45815743,
                1.15959264,
            ],
            atol=1e-7,
        )

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        sampled_damping, _ = mjx_env._domain_randomizer._sample_joint_damping(
            mjx_env._model, state.additional_carry, backend
        )
        np.testing.assert_allclose(
            sampled_damping,
            [
                1.3214438,
                1.1666062,
                0.41645122,
                1.2311093,
                0.32140747,
                0.93072027,
                0.6930329,
                0.6133657,
                0.902293,
                1.3626313,
                0.9963374,
            ],
            atol=1e-7,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_joint_armature(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        sampled_armature, _ = mjx_env._domain_randomizer._sample_joint_armature(
            mjx_env._model, mjx_env._additional_carry, backend
        )

        np.testing.assert_allclose(
            sampled_armature,
            [
                0.10769888,
                0.10266406,
                0.09061558,
                0.10092992,
                0.08375762,
                0.10303786,
                0.11717185,
                0.09274276,
                0.10669642,
                0.08527191,
                0.10865309,
            ],
            atol=1e-7,
        )

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        sampled_armature, _ = mjx_env._domain_randomizer._sample_joint_armature(
            mjx_env._model, state.additional_carry, backend
        )

        np.testing.assert_allclose(
            sampled_armature,
            [
                0.11404812,
                0.10888687,
                0.08388171,
                0.11103697,
                0.08071358,
                0.101024,
                0.09310109,
                0.09044552,
                0.10007643,
                0.11542104,
                0.10321124,
            ],
            atol=1e-7,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_gravity(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        sampled_gravity, _ = mjx_env._domain_randomizer._sample_gravity(
            mjx_env._model, mjx_env._additional_carry, backend
        )

        np.testing.assert_allclose(sampled_gravity, [0.0, 0.0, -9.92548327], atol=1e-7)

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        sampled_gravity, _ = mjx_env._domain_randomizer._sample_gravity(
            mjx_env._model, state.additional_carry, backend
        )

        np.testing.assert_allclose(
            sampled_gravity,
            [0.0, 0.0, -9.851224],
            atol=1e-7,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_base_mass(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        sampled_base_mass, _ = mjx_env._domain_randomizer._sample_base_mass(
            mjx_env._model, mjx_env._additional_carry, backend
        )

        assert isclose(sampled_base_mass, 0.7698884774800794)

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        sampled_base_mass, _ = mjx_env._domain_randomizer._sample_base_mass(
            mjx_env._model, state.additional_carry, backend
        )

        assert jnp.isclose(sampled_base_mass, 0.2748232)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_com_displacement(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        sampled_com_displacement, _ = (
            mjx_env._domain_randomizer._sample_com_displacement(
                mjx_env._model, mjx_env._additional_carry, backend
            )
        )

        np.testing.assert_allclose(
            sampled_com_displacement, [0.05774164, 0.01998044, -0.07038315], atol=1e-7
        )

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        sampled_com_displacement, _ = (
            mjx_env._domain_randomizer._sample_com_displacement(
                mjx_env._model, state.additional_carry, backend
            )
        )

        np.testing.assert_allclose(
            sampled_com_displacement,
            [-0.1474975, 0.00260535, -0.07673296],
            atol=1e-7,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_link_mass_multipliers(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        mass_multipliers, _ = mjx_env._domain_randomizer._sample_link_mass_multipliers(
            mjx_env._model, mjx_env._additional_carry, backend
        )

        np.testing.assert_allclose(
            mass_multipliers,
            [
                1.46946096,
                1.02664058,
                0.9061558,
                1.00929922,
                0.8375762,
                1.0303786,
                1.17171848,
                0.92742758,
                1.06696415,
            ],
            atol=1e-7,
        )

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        mass_multipliers, _ = mjx_env._domain_randomizer._sample_link_mass_multipliers(
            mjx_env._model, state.additional_carry, backend
        )

        np.testing.assert_allclose(
            mass_multipliers,
            [
                1.4263293,
                0.8988327,
                0.8013591,
                0.89718336,
                1.0634074,
                0.84470874,
                1.0975928,
                0.96769184,
                0.8068497,
            ],
            atol=1e-7,
        )


control_params_test = {
    "p_gain": [
        -300,  # back_bkz_actuator -> torso
        -200,  # hip_flexion_r_actuator -> hip_pitch
        -200,  # hip_adduction_r_actuator -> hip_roll
        -200,  # hip_rotation_r_actuator -> hip_yaw
        -300,  # knee_angle_r_actuator -> knee
        -40,  # ankle_angle_r_actuator -> ankle
        -200,  # hip_flexion_l_actuator -> hip_pitch
        -200,  # hip_adduction_l_actuator -> hip_roll
        -200,  # hip_rotation_l_actuator -> hip_yaw
        -300,  # knee_angle_l_actuator -> knee
        -40,  # ankle_angle_l_actuator -> ankle
    ],
    "d_gain": [
        -6,  # back_bkz_actuator -> torso
        -5,  # hip_flexion_r_actuator -> hip_pitch
        -5,  # hip_adduction_r_actuator -> hip_roll
        -5,  # hip_rotation_r_actuator -> hip_yaw
        -6,  # knee_angle_r_actuator -> knee
        -2,  # ankle_angle_r_actuator -> ankle
        -5,  # hip_flexion_l_actuator -> hip_pitch
        -5,  # hip_adduction_l_actuator -> hip_roll
        -5,  # hip_rotation_l_actuator -> hip_yaw
        -6,  # knee_angle_l_actuator -> knee
        -2,  # ankle_angle_l_actuator -> ankle
    ],
}


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_p_gains_noise(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        control_type="PDControl",
        control_params=control_params_test,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        p_noise, _ = mjx_env._domain_randomizer._sample_p_gains_noise(
            mjx_env, mjx_env._model, mjx_env._additional_carry, backend
        )

        np.testing.assert_allclose(
            p_noise,
            [
                -30.53640149,
                -29.76504388,
                -37.91778352,
                -23.57559142,
                5.39774507,
                4.28301049,
                -21.08903454,
                8.06353894,
                -24.44890141,
                -6.24824934,
                -3.90655615,
            ],
            atol=1e-7,
        )

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        p_noise, _ = mjx_env._domain_randomizer._sample_p_gains_noise(
            mjx_env, mjx_env._model, state.additional_carry, backend
        )

        np.testing.assert_allclose(
            p_noise,
            [
                -0.16460896,
                -9.782219,
                -16.564526,
                -8.960214,
                -2.0106053,
                1.9906788,
                -14.575982,
                2.425294,
                14.651022,
                20.975117,
                3.0969133,
            ],
            atol=1e-7,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_d_gains_noise(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        control_type="PDControl",
        control_params=control_params_test,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        d_noise, _ = mjx_env._domain_randomizer._sample_d_gains_noise(
            mjx_env, mjx_env._model, mjx_env._additional_carry, backend
        )

        np.testing.assert_allclose(
            d_noise,
            [
                -0.11045032,
                -0.07225191,
                0.27691837,
                -0.45274901,
                0.06344955,
                -0.13856347,
                -0.19947928,
                0.20256305,
                -0.31379782,
                0.12419311,
                -0.15244128,
            ],
            atol=1e-7,
        )

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        d_noise, _ = mjx_env._domain_randomizer._sample_d_gains_noise(
            mjx_env, mjx_env._model, state.additional_carry, backend
        )

        np.testing.assert_allclose(
            d_noise,
            [
                -0.00329218,
                -0.24455547,
                -0.41411316,
                -0.22400534,
                -0.04021211,
                0.09953394,
                -0.36439955,
                0.06063235,
                0.36627555,
                0.41950238,
                0.15484567,
            ],
            atol=1e-7,
        )
