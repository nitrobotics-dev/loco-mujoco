import numpy.random

from loco_mujoco.core.utils import mj_jntname2qposid

from test_conf import *


def _create_env(backend, init_state_type, init_state_params=None, trajectory=None):
    DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}

    mjx_env = DummyHumamoidEnv(enable_mjx=backend == "jax",
                               init_state_type=init_state_type,
                               init_state_params=init_state_params,
                               **DEFAULTS)

    if trajectory is not None:
        if backend == "numpy":
            trajectory.data = trajectory.data.to_numpy()

        mjx_env.load_trajectory(trajectory)

    return mjx_env

@pytest.mark.parametrize("backend", ["numpy"])
def test_DefaultInitialStateHandler(falling_trajectory, backend):
    mjx_env = _create_env(backend, "DefaultInitialStateHandler")

    seed = 0
    key = jax.random.PRNGKey(seed)

    state_0 = mjx_env.reset(key)

    qpos_init = np.zeros(18)
    qpos_init[:7] = np.array([1.5, 1.2, 0.3, 1., 0,  0., 0.])
    j_id = mj_jntname2qposid("abdomen_z", mjx_env._model)
    qpos_init[j_id] = 0.3

    qvel_init = 0.1*np.ones(17)

    init_state_params= dict(qpos_init=qpos_init, qvel_init=qvel_init)

    mjx_env = _create_env(backend, "DefaultInitialStateHandler", init_state_params=init_state_params)
    state_1 = mjx_env.reset(key)

    if backend == "numpy":
        print("\n", state_0)
        state_0_test = np.array([1.4, 1., 0., 0., -0.00902001,  0., 0.1, 0.9350026, 0., 0., 0., 0., 0., -0.00902001, 0.1,
                                  0.9350026, 0., 0., 0., 0., 0., 0.])

        assert np.allclose(state_0, state_0_test)

        state_1_test = np.array([0.3, 1.,  0., 0., -0.00902001,  0., .1, 0.9350026, 0., 0., 0., 0., 0.,
                                  -0.00902001,  0.1, 0.9350026, 0., 0., 0., 0., 0., 0.])
        print("\n", state_1)
        assert np.allclose(state_1, state_1_test)

        assert np.allclose(mjx_env._data.qpos, qpos_init)
        assert np.allclose(mjx_env._data.qvel, qvel_init)
    else:
        pass


@pytest.mark.parametrize("backend", ["numpy"])
def test_TrajInitialStateHandler(falling_trajectory, backend):
    mjx_env = _create_env(backend, "TrajInitialStateHandler", trajectory=falling_trajectory)

    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    state_0 = mjx_env.reset(key)
    state_1 = mjx_env.reset(key)
    state_2 = mjx_env.reset(key)

    if backend == "numpy":
        state_0_test = np.array([ 1.27379999e-01,  6.11406028e-01, -2.17664987e-01, -7.40720451e-01,
                                 -6.61719680e-01, -9.68485983e-05,  5.23620665e-01,  1.02717586e-01,
                                  6.59988774e-03, 1.23926671e-02,  7.23121702e-05,  4.86690697e-04,
                                 -1.52403500e-03, -6.61719680e-01,  5.23620665e-01,  1.02717586e-01,
                                  6.59988774e-03,  1.23926671e-02,  7.23121702e-05,  4.86690697e-04,
                                 -9.68485983e-05, -1.52403500e-03])
        state_1_test = np.array([ 1.25226304e-01,  6.15476191e-01, -2.16497868e-01, -7.37226367e-01,
                                 -6.62434101e-01, -1.58643743e-04,  5.23801744e-01,  1.03564799e-01,
                                  8.32138676e-03,  1.56304557e-02,  1.59624280e-04,  6.26335677e-04,
                                 -1.91622844e-03, -6.62434101e-01,  5.23801744e-01,  1.03564799e-01,
                                  8.32138676e-03,  1.56304557e-02,  1.59624280e-04,  6.26335677e-04,
                                 -1.58643743e-04, -1.91622844e-03])
        state_2_test = np.array( [ 1.26494944e-01,  6.13081455e-01, -2.17193708e-01, -7.39284277e-01,
                                  -6.62012160e-01, -1.22567348e-04,  5.23689568e-01,  1.03066877e-01,
                                   7.29704089e-03,  1.37028117e-02,  1.08488151e-04,  5.43690694e-04,
                                  -1.68306101e-03, -6.62012160e-01,  5.23689568e-01,  1.03066877e-01,
                                   7.29704089e-03,  1.37028117e-02,  1.08488151e-04,  5.43690694e-04,
                                  -1.22567348e-04, -1.68306101e-03])

        print("\n", state_0)
        assert np.allclose(state_0, state_0_test)

        print("\n", state_1)
        assert np.allclose(state_1, state_1_test)

        print("\n", state_2)
        assert np.allclose(state_2, state_2_test)
    else:
        pass
