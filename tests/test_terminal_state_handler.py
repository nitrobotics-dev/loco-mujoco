import mujoco.mjx
import jax
import numpy as np
from mujoco import MjSpec

from loco_mujoco.core.observations import ObservationType
from loco_mujoco.environments import LocoEnv
from loco_mujoco.trajectory import Trajectory

from test_conf import *


DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs":1}


@pytest.mark.parametrize("backend", ["jax"])
def test_RootPoseTrajTerminalStateHandler(input_trajectory, backend):

    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = TestHumamoidEnv(enable_mjx=True,
                              terminal_state_type="RootPoseTrajTerminalStateHandler",
                              **DEFAULTS)

    # load trajectory to env
    traj: Trajectory = input_trajectory(backend)
    backend_type = jnp if backend == "jax" else np
    mjx_env.load_trajectory(traj)

    # reset the environment in Mjx
    state = mjx_env.mjx_reset(key)
    obs_mjx = state.observation

    # todo: do what you have to do ...


    # do the same testing for Mujoco function
    mjx_env.th.to_numpy()

    # reset the environment in Mujoco
    obs = mjx_env.reset(key)

    # todo: do what you have to do ...





