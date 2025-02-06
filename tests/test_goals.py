import pytest
import numpy as np
import jax.numpy as jnp
import jax

from test_conf import DummyHumamoidEnv
from loco_mujoco.core.observations.goals import Goal

from loco_mujoco.trajectory import Trajectory
from test_conf import *

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_NoGoal(backend):
    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        goal_type="NoGoal",
        reward_type="NoReward",
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np

    current_goal: Goal = mjx_env._goal
    dim = current_goal.dim
    assert dim == 0, "The dimension has to be 0"

    goal, carry = current_goal.get_obs_and_update_state(
        mjx_env, mjx_env._model, mjx_env._data, None, backend
    )

    assert goal.shape == (0,), "NoGoal should return an empty observation"
    assert carry is None, "Carry should remain unchanged"


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_GoalRandomRootVelocity(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        goal_type="GoalRandomRootVelocity",
        reward_type="NoReward",
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np

    current_goal: Goal = mjx_env._goal
    dim = current_goal.dim
    assert dim == 3, "The dimension has to be 3"

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        carry = mjx_env._additional_carry
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs = state.observation
        carry = state.additional_carry

    obs = obs[-dim:]

    assert (
        -current_goal.max_x_vel <= obs[0] <= current_goal.max_x_vel
    ), "X velocity out of bounds"
    assert (
        -current_goal.max_y_vel <= obs[1] <= current_goal.max_y_vel
    ), "Y velocity out of bounds"
    assert (
        -current_goal.max_yaw_vel <= obs[2] <= current_goal.max_yaw_vel
    ), "Yaw velocity out of bounds"

    goal, carry = current_goal.get_obs_and_update_state(
        mjx_env, mjx_env._model, mjx_env._data, carry, backend
    )
    # check the observation
    np.testing.assert_allclose(
        obs,
        goal,
        err_msg="Mismatch between Mujoco observation and goal",
    )

    data, carry = current_goal.reset_state(
        mjx_env, mjx_env._model, mjx_env._data, carry, backend
    )

    if backend == jnp:
        obs, carry = mjx_env._mjx_create_observation(mjx_env._model, data, carry)
    else:
        obs, carry = mjx_env._create_observation(mjx_env._model, data, carry)

    obs = obs[-dim:]
    goal, carry = current_goal.get_obs_and_update_state(
        mjx_env, mjx_env._model, mjx_env._data, carry, backend
    )
    # check the observation
    np.testing.assert_allclose(
        obs,
        goal,
        err_msg="Mismatch between Mujoco observation and goal",
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_GoalTrajRootVelocity(backend, standing_trajectory):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        goal_type="GoalTrajRootVelocity",
        reward_type="NoReward",
        **DEFAULTS,
    )

    trajectory: Trajectory = standing_trajectory
    mjx_env.load_trajectory(trajectory)

    backend = jnp if backend == "jax" else np

    goal: Goal = mjx_env._goal
    dim = goal.dim
    assert dim == 6, "The dimension has to be 6"
    assert goal.requires_trajectory == True
    assert goal.has_visual == True

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        carry = mjx_env._additional_carry
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs = state.observation
        carry = state.additional_carry
    obs = obs[-dim:]

    goal, _ = goal.get_obs_and_update_state(
        mjx_env, mjx_env._model, mjx_env._data, carry, backend
    )

    # check the observation
    np.testing.assert_allclose(
        obs,
        goal,
        err_msg="Mismatch between Mujoco observation and goal",
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_GoalTrajMimic(backend, standing_trajectory):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        goal_type="GoalTrajMimic",
        reward_type="NoReward",
        **DEFAULTS,
    )

    trajectory: Trajectory = standing_trajectory
    mjx_env.load_trajectory(trajectory)

    backend = jnp if backend == "jax" else np

    goal: Goal = mjx_env._goal
    dim = goal.dim

    assert goal.requires_trajectory == True
    assert goal.has_visual == True
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        carry = mjx_env._additional_carry
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs = state.observation
        carry = state.additional_carry

    obs = obs[-dim:]

    goal, carry = goal.get_obs_and_update_state(
        mjx_env, mjx_env._model, mjx_env._data, carry, backend
    )

    # check the observation
    np.testing.assert_allclose(
        obs, goal, err_msg="Mismatch between Mujoco observation and goal", atol=1e-7
    )
