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

@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_DefaultInitialStateHandler(falling_trajectory, backend):
    seed = 0
    key = jax.random.PRNGKey(seed)

    mjx_env_1 = _create_env(backend, "DefaultInitialStateHandler")

    # create init_state_params
    qpos_init = np.zeros(18)
    qpos_init[:7] = np.array([1.5, 1.2, 0.3, 1., 0,  0., 0.])
    j_id = mj_jntname2qposid("abdomen_z", mjx_env_1._model)
    qpos_init[j_id] = 0.3

    qvel_init = 0.1 * np.ones(17)

    init_state_params = dict(qpos_init=qpos_init, qvel_init=qvel_init)

    mjx_env_2 = _create_env(backend, "DefaultInitialStateHandler", init_state_params=init_state_params)

    if backend == "numpy":
        state_0 = mjx_env_1.reset(key)
        state_1 = mjx_env_2.reset(key)

        print("\nstate_0", state_0)
        print("position: ", mjx_env_1._data.qpos[0:3])
        print("quaternion: ", mjx_env_1._data.qpos[3:7])
        print("quaternion[4]: ", mjx_env_1._data.qpos[6])
        state_0_test = np.array([ 1.4, 1., 0., 0., 0., # FreeJointPosNoXY
                                  0., # JointPos
                                  0., 0., 0., 0., 0., 0., # FreeJointVel
                                  0., # JointVel
                                  -0.00902001,  0.1, 0.9350026, # BodyPos
                                  0., 0., 0., 0., 0., 0. # BodyVel
                                ])

        assert np.allclose(state_0, state_0_test)

        state_1_test = np.array([0.3, 1., 0., 0., 0., # FreeJointPosNoXY
                                 0.3, # JointPos
                                 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, # FreeJointVel
                                 .1, # JointVel
                                 -0.00902001,  0.1, 0.9350026, # BodyPos
                                  0., 0., 0., 0., 0., 0. # BodyVel
                                 ])
        print("\nstate_1", state_1)
        assert np.allclose(state_1, state_1_test)

        assert np.allclose(mjx_env_2._data.qpos, qpos_init)
        assert np.allclose(mjx_env_2._data.qvel, qvel_init)
    else:
        state_0 = mjx_env_1.mjx_reset(key)
        state_1 = mjx_env_2.mjx_reset(key)

        print("\nstate_0", state_0.observation)
        state_0_test = jnp.array([1.4, 1., 0., 0., 0.,  # FreeJointPosNoXY
                                  0.,  # JointPos
                                  0., 0., 0., 0., 0., 0.,  # FreeJointVel
                                  0.,  # JointVel
                                  -0.00902001, 0.1, 0.9350026,  # BodyPos
                                  0., 0., 0., 0., 0., 0.  # BodyVel
                                 ])

        assert jnp.allclose(state_0.observation, state_0_test)

        state_1_test = jnp.array([0.3, 1., 0., 0., 0.,  # FreeJointPosNoXY
                                  0.3,  # JointPos
                                  0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # FreeJointVel
                                  .1,  # JointVel
                                  -0.00902001, 0.1, 0.9350026,  # BodyPos
                                  0., 0., 0., 0., 0., 0.  # BodyVel
                                  ])
        print("\nstate_1", state_1.observation)
        assert jnp.allclose(state_1.observation, state_1_test)

        assert jnp.allclose(state_1.data.qpos, qpos_init)
        assert jnp.allclose(state_1.data.qvel, qvel_init)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_TrajInitialStateHandler(falling_trajectory, backend):
    mjx_env = _create_env(backend, "TrajInitialStateHandler", trajectory=falling_trajectory)

    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    if backend == "numpy":
        state_0 = mjx_env.reset(key)
        state_1 = mjx_env.reset(key)
        state_2 = mjx_env.reset(key)

        state_0_test = np.array([ 1.27379999e-01,  6.11406028e-01, -2.17664987e-01, -7.40720451e-01,
                                 -1.73602626e-01,  3.92150600e-03,  1.23301230e-03, -7.03400816e-04,
                                  1.52836146e-03,  6.77359349e-04, -8.20835214e-03,  8.05158925e-04,
                                 -1.23359161e-04, -6.61719680e-01,  5.23620665e-01,  1.02717586e-01,
                                  6.59988774e-03,  1.23926671e-02,  7.23121702e-05,  4.86690697e-04,
                                 -9.68485983e-05, -1.52403500e-03])
        state_1_test = np.array([ 1.25226304e-01,  6.15476191e-01, -2.16497868e-01, -7.37226367e-01,
                                 -1.75542042e-01,  4.09621233e-03,  1.48461049e-03, -8.25593190e-04,
                                  1.93808251e-03,  7.41310418e-04, -1.03822229e-02,  1.07889611e-03,
                                 -1.56038834e-04, -6.62434101e-01,  5.23801744e-01,  1.03564799e-01,
                                  8.32138676e-03,  1.56304557e-02,  1.59624280e-04,  6.26335677e-04,
                                 -1.58643743e-04, -1.91622844e-03])
        state_2_test = np.array([ 1.26494944e-01,  6.13081455e-01, -2.17193708e-01, -7.39284277e-01,
                                 -1.74404174e-01,  3.99326440e-03,  1.33853650e-03, -7.54700217e-04,
                                  1.69381592e-03,  7.06091523e-04, -9.08721518e-03,  9.12379881e-04,
                                 -1.37748750e-04, -6.62012160e-01,  5.23689568e-01,  1.03066877e-01,
                                  7.29704089e-03,  1.37028117e-02,  1.08488151e-04,  5.43690694e-04,
                                 -1.22567348e-04, -1.68306101e-03])

        print("\nstate_0", state_0)
        assert np.allclose(state_0, state_0_test)

        print("\nstate_1", state_1)
        assert np.allclose(state_1, state_1_test)

        print("\nstate_2", state_2)
        assert np.allclose(state_2, state_2_test)
    else:
        state_0 = mjx_env.mjx_reset(key)
        key = jax.random.PRNGKey(seed+1)
        state_1 = mjx_env.mjx_reset(key)
        key = jax.random.PRNGKey(seed+2)
        state_2 = mjx_env.mjx_reset(key)

        state_0_test = np.array([ 1.1286861e-01,  6.3870221e-01, -2.0777164e-01, -7.1726537e-01,
                                 -1.8552853e-01,  4.8972582e-03,  3.5754945e-03, -2.0033114e-03,
                                  5.7487963e-03, -2.7495090e-04, -3.0445326e-02,  4.6008267e-03,
                                 -1.0005956e-03, -6.6718584e-01,  5.2557540e-01,  1.0827588e-01,
                                  3.3251036e-02,  4.0564332e-02, -8.1857044e-04,  2.0098409e-03,
                                 -7.3761720e-04, -6.3186735e-03])
        state_1_test = np.array([ 1.29183963e-01,  6.07940674e-01, -2.18539864e-01, -7.43687928e-01,
                                 -1.71978965e-01,  3.79295624e-03,  6.68634384e-05, -2.11837629e-04,
                                  1.63286415e-04,  3.77590361e-04, -1.10921788e-03, -6.82454556e-04,
                                  4.28220257e-04, -6.61141336e-01,  5.23520589e-01,  1.02023564e-01,
                                 -2.18771608e-03,  1.45972520e-03,  1.34152069e-04,  3.40628649e-05,
                                  1.23126301e-04,  2.05085627e-04])
        state_2_test = np.array([ 1.2080755e-01,  6.2382293e-01, -2.1366180e-01, -7.3010105e-01,
                                 -1.7929325e-01,  4.4326615e-03,  2.4164848e-03, -9.8634663e-04,
                                  3.1212410e-03, -1.0984721e-04, -1.6716938e-02,  2.2293206e-03,
                                 -2.1318589e-04, -6.6398108e-01,  5.2431017e-01,  1.0527880e-01,
                                  1.3378816e-02,  2.4976406e-02,  3.8249002e-04,  1.1231992e-03,
                                 -3.7994684e-04, -3.0530260e-03])

        print("\nstate_0", state_0.observation)
        assert np.allclose(state_0.observation, state_0_test)

        print("\nstate_1", state_1.observation)
        assert np.allclose(state_1.observation, state_1_test)

        print("\nstate_2", state_2.observation)
        assert np.allclose(state_2.observation, state_2_test)
