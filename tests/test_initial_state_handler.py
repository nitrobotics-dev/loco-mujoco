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
        state_0_test = np.array([ 1.293, 1., 0., 0., 0., # FreeJointPosNoXY
                                  0., # JointPos
                                  0., 0., 0., 0., 0., 0., # FreeJointVel
                                  0., # JointVel
                                  -0.02726929, 0.09849757, 0.828, # BodyPos
                                  0., 0., 0., 0., 0., 0. # BodyVel
                                ])

        assert np.allclose(state_0, state_0_test)

        state_1_test = np.array([0.3, 1., 0., 0., 0., # FreeJointPosNoXY
                                 0.3, # JointPos
                                 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, # FreeJointVel
                                 .1, # JointVel
                                 -0.02726929, 0.09849757, 0.828, # BodyPos
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
        state_0_test = jnp.array([1.293, 1., 0., 0., 0.,  # FreeJointPosNoXY
                                  0.,  # JointPos
                                  0., 0., 0., 0., 0., 0.,  # FreeJointVel
                                  0.,  # JointVel
                                  -0.02726929, 0.09849757, 0.828,  # BodyPos
                                  0., 0., 0., 0., 0., 0.  # BodyVel
                                 ])

        assert jnp.allclose(state_0.observation, state_0_test)

        state_1_test = jnp.array([0.3, 1., 0., 0., 0.,  # FreeJointPosNoXY
                                  0.3,  # JointPos
                                  0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # FreeJointVel
                                  .1,  # JointVel
                                  -0.02726929, 0.09849757, 0.828,  # BodyPos
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

        state_0_test = np.array([ 1.16946056e-01,  5.30123770e-01, -6.83287457e-02, -7.51042604e-01,
                                3.87601584e-01,  2.95905891e-04,  5.39141707e-04,  1.68639977e-04,
                                5.47435484e-04, -1.07494928e-03, -2.97159073e-03, -3.45852110e-03,
                                -9.20652226e-03, -1.75645009e-01,  2.62723386e-01,  7.68627971e-02,
                                3.15095822e-04,  1.20598695e-03, -2.67173303e-03, -5.65242954e-06,
                                -5.97170379e-04,  1.19107797e-04 ])
        state_1_test = np.array([ 1.16053157e-01,  5.34218609e-01, -6.17813431e-02, -7.50009537e-01,
                                3.85070413e-01,  2.22238284e-02,  1.67215895e-03,  1.36915594e-03,
                                3.53976939e-04, -5.14284382e-03,  1.58112578e-03,  1.57346632e-02,
                                -2.22745053e-02, -1.73618317e-01,  2.62186408e-01,  7.75714517e-02,
                                -1.48864184e-03, -4.70039481e-03, -9.58127249e-03, -3.51673691e-03,
                                4.39609401e-04, -4.01069090e-04 ] )
        state_2_test = np.array([ 1.16527945e-01,  5.31937778e-01, -6.59242198e-02, -7.50498652e-01,
                                3.86585057e-01,  9.83789936e-03,  1.19256822e-03,  2.16552499e-03,
                                7.92007602e-04, -4.06703772e-03, -2.17886455e-03,  1.85526609e-02,
                                -1.97520927e-02, -1.75115407e-01,  2.62619197e-01,  7.70714879e-02,
                                -8.25731026e-04, -4.81245574e-03, -8.68918840e-03, -2.91174022e-03,
                                5.95915946e-04, -6.62494451e-04 ])

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

        state_0_test = np.array([ 0.17086907,  0.5356333,  0.21885236, -0.8140784,  0.04976885,  0.01962737,
                                -0.0144955,  0.0083312,  0.0072153,  0.01165351, -0.04078958,  0.15574612,
                                0.0225309, -0.00567735,  0.19789234,  0.2348802, -0.270485, -0.18992859,
                                0.7182234,  0.03185572, -0.01060285, -0.00989754 ])
        state_1_test = np.array([ 1.1691184e-01,  5.3017032e-01, -6.8392314e-02, -7.5097591e-01,
                                3.8765591e-01,  9.2281483e-04,  4.4982339e-04,  3.7463877e-04,
                                5.9321593e-04, -1.3572660e-03, -3.0983721e-03, -6.7582639e-04,
                                -1.1808202e-02, -1.7565699e-01,  2.6275480e-01,  7.6853015e-02,
                                3.2749615e-04,  1.1764233e-03, -2.7255940e-03, -5.5725245e-06,
                                -5.8416842e-04,  1.0148822e-04 ])
        state_2_test = np.array([ 1.15985692e-01,  5.33005655e-01, -5.82038686e-02, -7.50167012e-01,
                                3.86997104e-01,  3.70270573e-02,  4.30292450e-03, -1.64990267e-03,
                                -9.37687757e-04, -6.48063049e-03,  1.19192265e-02, -1.36089688e-02,
                                1.57578848e-03, -1.69700474e-01,  2.61411369e-01,  7.81876817e-02,
                                -1.73241668e-03, -3.45178437e-03, -1.46292038e-02, -5.73879899e-03,
                                7.47278682e-04, -6.67195200e-05 ])

        print("\nstate_0", state_0.observation)
        assert np.allclose(state_0.observation, state_0_test)

        print("\nstate_1", state_1.observation)
        assert np.allclose(state_1.observation, state_1_test)

        print("\nstate_2", state_2.observation)
        assert np.allclose(state_2.observation, state_2_test)
