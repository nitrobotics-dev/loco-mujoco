import numpy
from math import isclose

from test_conf import *


def _create_env(backend, control_type="DefaultControl", control_params=None):
    DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}

    mjx_env = DummyHumamoidEnv(enable_mjx=backend == "jax",
                               control_type=control_type,
                               control_params=control_params,
                               **DEFAULTS)

    return mjx_env

@pytest.mark.parametrize("backend", ["numpy"])
def test_DefaultControl(backend):
    mjx_env = _create_env(backend, "DefaultControl")

    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    action = np.random.randn(mjx_env.info.action_space.shape[0])

    if backend == 'numpy':
        mjx_env.reset(key)
        mjx_env.step(action)
    else:
        state = mjx_env.mjx_reset(key)
        action = jnp.array(action)
        mjx_env.mjx_step(state, action)

    assert isclose(mjx_env._data.ctrl[0], 0.7056209383870656)
    assert isclose(mjx_env._data.ctrl[6], 0.16006288334688934)


@pytest.mark.parametrize("backend", ["numpy"])
def test_PDControl(falling_trajectory, backend):
    p_gain = 1.0
    d_gain = 0.1
    control_params=dict(p_gain=p_gain, d_gain=d_gain)
    mjx_env = _create_env(backend, "PDControl", control_params=control_params)

    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    action = np.random.randn(mjx_env.info.action_space.shape[0])

    if backend == 'numpy':
        mjx_env.reset(key)
        mjx_env.step(action)
    else:
        state = mjx_env.mjx_reset(key)
        action = jnp.array(action)
        mjx_env.mjx_step(state, action)

    assert isclose(mjx_env._data.ctrl[0], 0.4)
    assert isclose(mjx_env._data.ctrl[6], -0.4)

