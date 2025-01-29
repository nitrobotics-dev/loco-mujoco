import jax
from .dummy_humanoid_env import DummyHumamoidEnv

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs":1}

def generate_test_trajectories(expert_traj, nominal_traj, backend, **kwargs):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(enable_mjx=backend == "jax",
                               **kwargs,
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