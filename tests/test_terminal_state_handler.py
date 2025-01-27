import mujoco.mjx
import jax
import numpy as np
from mujoco import MjSpec

from loco_mujoco.core.observations import ObservationType
from loco_mujoco.environments import LocoEnv
from loco_mujoco.trajectory import Trajectory

from test_conf import *


DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs":1}

def _generate_trajectories(expert_traj, nominal_traj, backend, terminal_state_type):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(enable_mjx=backend == "jax",
                               terminal_state_type=terminal_state_type,
                               **DEFAULTS)

    if backend == "numpy":
        expert_traj.data = expert_traj.data.to_numpy()
        nominal_traj.data = nominal_traj.data.to_numpy()

    mjx_env.load_trajectory(expert_traj)

    # Create dataset of transitions using the nominal trajectory
    if backend == "numpy":
        return mjx_env.generate_trajectory_from_nominal(nominal_traj)
    else:
        return mjx_env.mjx_generate_trajectory_from_nominal(nominal_traj)


@pytest.mark.parametrize("backend", ["numpy"])
def test_HeightBasedTerminalStateHandler(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory
    terminal_state_type = "HeightBasedTerminalStateHandler"

    transitions = _generate_trajectories(expert_traj, nominal_traj, backend, terminal_state_type)

    if backend == "numpy":
        assert len(transitions.absorbings) == 999
        assert np.sum(transitions.absorbings) == 922
        assert np.all(transitions.absorbings[0:77] == 0)
        assert np.all(transitions.absorbings[77:] == 1)
    else:
        assert False


@pytest.mark.parametrize("backend", ["numpy"])
def test_NoTerminalStateHandler(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory
    terminal_state_type = "NoTerminalStateHandler"

    transitions = _generate_trajectories(expert_traj, nominal_traj, backend, terminal_state_type)

    if backend == "numpy":
        assert len(transitions.absorbings) == 999
        assert np.sum(transitions.absorbings) == 0
        assert np.all(transitions.absorbings == 0)
    else:
        assert False


@pytest.mark.parametrize("backend", ["numpy"])
def test_RootPoseTrajTerminalStateHandler(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory
    terminal_state_type = "RootPoseTrajTerminalStateHandler"

    transitions = _generate_trajectories(expert_traj, nominal_traj, backend, terminal_state_type)

    if backend == "numpy":
        assert len(transitions.absorbings) == 999
        assert np.sum(transitions.absorbings) == 958
        assert np.all(transitions.absorbings[0:41] == 0)
        assert np.all(transitions.absorbings[41:] == 1)
    else:
        assert False