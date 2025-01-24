import mujoco.mjx
import jax
import numpy as np
from mujoco import MjSpec

from loco_mujoco.core.observations import ObservationType
from loco_mujoco.environments import LocoEnv
from loco_mujoco.trajectory import Trajectory

from test_conf import *


DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs":1}


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_RootPoseTrajTerminalStateHandler(standing_trajectory, falling_trajectory, backend):

    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(enable_mjx=backend == "jax",
                               terminal_state_type="RootPoseTrajTerminalStateHandler",
                               **DEFAULTS)

    # load trajectory to env
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    if backend == "numpy":
        expert_traj.data = expert_traj.data.to_numpy()
        nominal_traj.data = nominal_traj.data.to_numpy()

    mjx_env.load_trajectory(expert_traj)

    # Create dataset of transitions using the nominal trajectory
    if backend == "numpy":
        transitions = mjx_env.generate_trajectory_from_nominal(falling_trajectory)
    else:
        transitions = mjx_env.mjx_generate_trajectory_from_nominal(falling_trajectory)

    print(len(transitions.absorbings))
    print(np.argwhere(transitions.absorbings))
