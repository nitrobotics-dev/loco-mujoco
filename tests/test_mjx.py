import loco_mujoco
from loco_mujoco import LocoEnv
from loco_mujoco.core.wrappers import MjxRolloutWrapper, RolloutWrapper
import jax


def test_mjx_simto_mujoco():
    """
    Test whether the Mjx simulation is equal/similar to the Mujoco simulation.
    """

    # tolerance for difference between MuJoCo and MJX forward calculations - mostly
    # due to float precision
    _TOLERANCE = 1e-4

    # set Jax-backend to CPU
    jax.config.update('jax_platform_name', 'cpu')
    print(f"Jax backend device: {jax.default_backend()} \n")

    task_names = loco_mujoco.get_all_task_names()

    N_ENVS = 100
    # numerical errors compound quickly, so limit rollout horizon and substeps.
    MODEL_OPTION = dict(iterations=100, ls_iterations=50)

    def test_one_task(task_name,  n_steps, **task_params):

        key = jax.random.PRNGKey(94)
        keys = jax.random.split(key, N_ENVS)

        # Mujoco
        task_env = LocoEnv.make(task_name, debug=True, **task_params)
        rollout_env = RolloutWrapper(task_env)

        obs, action, reward, next_obs, absorbing, done, cum_return = (
            rollout_env.batch_rollout(rng_keys=keys, n_steps=n_steps, policy_params=None))

        # Mjx rollout
        mjx_task_env = LocoEnv.make(task_name, debug=True, **task_params)

        mjx_rollout_env = MjxRolloutWrapper(mjx_task_env)

        mjx_obs, mjx_action, mjx_reward, mjx_next_obs, mjx_absorbing, mjx_done, mjx_cum_return = (
            mjx_rollout_env.batch_rollout(rng_keys=keys, n_steps=n_steps, policy_params=None))

        qpos_idx = mjx_task_env._joint_qpos_range - 2
        qpos_idx = qpos_idx[2:]
        qvel_idx = mjx_task_env._joint_qvel_range - 2
        qvel_idx = qvel_idx[2:]

        # print("obs max diff: ", jax.numpy.max(obs - mjx_obs, axis=(0, 1)))
        # print("next obs max diff: ", jax.numpy.max(next_obs - mjx_next_obs, axis=(0, 1)))

        # check qpos
        assert jax.numpy.allclose(obs[:, :, qpos_idx], mjx_obs[:, :, qpos_idx],
                                  atol=_TOLERANCE, rtol=_TOLERANCE)
        assert jax.numpy.allclose(next_obs[:, :, qpos_idx], mjx_next_obs[:, :, qpos_idx],
                                  atol=_TOLERANCE, rtol=_TOLERANCE)

        # check qvel
        _QVEL_TOLERANCE = _TOLERANCE * 10   # velocities are higher in magnitude
        assert jax.numpy.allclose(obs[:, :, qvel_idx], mjx_obs[:, :, qvel_idx],
                                  atol=_QVEL_TOLERANCE, rtol=_QVEL_TOLERANCE)
        assert jax.numpy.allclose(next_obs[:, :, qvel_idx], mjx_next_obs[:, :, qvel_idx],
                                  atol=_QVEL_TOLERANCE, rtol=_QVEL_TOLERANCE)

        # check actions
        assert jax.numpy.allclose(action, mjx_action, atol=_TOLERANCE, rtol=_TOLERANCE)

        # check rewards
        assert jax.numpy.allclose(reward, mjx_reward, atol=_TOLERANCE, rtol=_TOLERANCE)

        # check dones
        assert jax.numpy.allclose(done, mjx_done, atol=_TOLERANCE, rtol=_TOLERANCE)

    for task_name in task_names:
        if "Mjx" in task_name:

            n_steps = 4
            n_substeps = 2
            print("Running ", task_name, " with default start.")
            test_one_task(task_name, random_start=False, n_steps=n_steps,
                          n_substeps=n_substeps, model_option_conf=MODEL_OPTION)

            # By setting short horizons we test the resetting mechanism.
            # Note: we do not test it with random resets from trajectory, as the rng_keys propagate
            # slightly different in Mujoco and Mjx (due to _mjx_reset_in_step), so the sampled init
            # states wouldn't be equivalent. However, the main reset function _reset(key) and _mjx_reset(key)
            # are equivalent, which is why the above works.

            short_horizon = 2

            print("Running ", task_name, " with default start.")
            test_one_task(task_name, random_start=False, n_steps=n_steps,
                          n_substeps=n_substeps, model_option_conf=MODEL_OPTION, horizon=short_horizon)

            print("Running ", task_name, " with fixed start.")
            test_one_task(task_name, random_start=False, n_steps=n_steps, fixed_start_conf=(0, 0),
                          n_substeps=n_substeps, model_option_conf=MODEL_OPTION, horizon=short_horizon)
