import os
import sys
import jax
import jax.numpy as jnp
import wandb
from loco_mujoco import LocoEnv
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.algorithms import save_ckpt

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import traceback


@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    try:

        os.environ['XLA_FLAGS'] = (
            '--xla_gpu_triton_gemm_any=True ')

        # Accessing the current sweep number
        result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        # setup wandb
        wandb.login()
        omega_conf_container = OmegaConf.to_container(config.experiment, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, config=omega_conf_container)

        env = LocoEnv.make(**config.experiment.env_params)

        # jit everything
        train_func = PPOJax.get_train_function(env, config.experiment)
        train_jit = jax.jit(jax.vmap(train_func)) if config.experiment.n_seeds > 1 else jax.jit(train_func)

        # run training
        rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds+1)]  # create rngs from seed
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
        out = train_jit(_rng)

        import time
        t_start = time.time()
        # get the metrics and log them
        if not config.experiment.debug:
            metrics = out["metrics"]

            # calculate mean and variance across seeds
            mean_ret = jnp.mean(jnp.atleast_2d(metrics.mean_episode_return), axis=0)
            mean_len = jnp.mean(jnp.atleast_2d(metrics.mean_episode_length), axis=0)
            step = jnp.mean(jnp.atleast_2d(metrics.max_timestep), axis=0)

            var_ret = jnp.var(jnp.atleast_2d(metrics.mean_episode_return), axis=0)
            var_len = jnp.var(jnp.atleast_2d(metrics.mean_episode_length), axis=0)

            for i in range(len(mean_ret)):
                run.log({"Mean Episode Return": mean_ret[i], "Mean Episode Length": mean_len[i],
                         "Var Episode Return": var_ret[i], "Var Episode Length": var_len[i]}, step=int(step[i]))

        print(f"Time taken to log metrics: {time.time() - t_start}s")

        runner_state = out["runner_state"]
        ckpt = dict(train_state=runner_state[0], disc_train_state=runner_state[1])

        # Automatically created log directory
        save_ckpt(ckpt, path=result_dir, path_is_local=False)
        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
