from test_conf import *


def _create_env(backend, init_state_type, trajectory=None):
    DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}

    mjx_env = DummyHumamoidEnv(enable_mjx=backend == "jax",
    init_state_type=init_state_type)

    if trajectory is not None:
        if backend == "numpy":
            trajectory.data = trajectory.data.to_numpy()

        mjx_env.load_trajectory(trajectory)

    return mjx_env

@pytest.mark.parametrize("backend", ["numpy"])
def test_DefaultInitialStateHandler(backend):
    mjx_env = _create_env(backend, "DefaultInitialStateHandler")

    seed = 0
    key = jax.random.PRNGKey(seed)

    a_1 = mjx_env.reset(key)
    a_2 = mjx_env.reset(key)
    a_3 = mjx_env.reset(key)

    if backend == "numpy":
        a_1_test = np.array([ 1.4, 1., 0., 0., -0.00902001,  0., 0.1, 0.9350026, 0., 0., 0., 0., 0., -0.00902001, 0.1,
                              0.9350026, 0., 0., 0., 0., 0., 0.,])
        a_2_test = np.array([1.4, 1., 0., 0., -0.00902001, 0., 0.1, 0.9350026, 0., 0., 0., 0., 0., -0.00902001, 0.1,
                             0.9350026, 0., 0., 0., 0., 0., 0., ])
        a_3_test = np.array([1.4, 1., 0., 0., -0.00902001, 0., 0.1, 0.9350026, 0., 0., 0., 0., 0., -0.00902001, 0.1,
                             0.9350026, 0., 0., 0., 0., 0., 0., ])

        assert np.allclose(a_1, a_1_test)
        assert np.allclose(a_2, a_2_test)
        assert np.allclose(a_3, a_3_test)
    else:
        pass


@pytest.mark.parametrize("backend", ["numpy"])
def test_TrajInitialStateHandler(standing_trajectory, backend):
    mjx_env = _create_env(backend, "TrajInitialStateHandler", standing_trajectory)

    seed = 0
    key = jax.random.PRNGKey(seed)

    a_1 = mjx_env.reset(key)
    a_2 = mjx_env.reset(key)
    a_3 = mjx_env.reset(key)

    if backend == "numpy":
        a_1_test = np.array([1.4, 1., 0., 0., -0.00902001, 0., 0.1, 0.9350026, 0., 0., 0., 0., 0., -0.00902001, 0.1,
                             0.9350026, 0., 0., 0., 0., 0., 0., ])
        a_2_test = np.array([1.4, 1., 0., 0., -0.00902001, 0., 0.1, 0.9350026, 0., 0., 0., 0., 0., -0.00902001, 0.1,
                             0.9350026, 0., 0., 0., 0., 0., 0., ])
        a_3_test = np.array([1.4, 1., 0., 0., -0.00902001, 0., 0.1, 0.9350026, 0., 0., 0., 0., 0., -0.00902001, 0.1,
                             0.9350026, 0., 0., 0., 0., 0., 0., ])

        assert np.allclose(a_1, a_1_test)
        assert np.allclose(a_2, a_2_test)
        assert np.allclose(a_3, a_3_test)
    else:
        pass
