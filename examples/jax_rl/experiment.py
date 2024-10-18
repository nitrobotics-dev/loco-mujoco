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
        job_id = hydra.core.hydra_config.HydraConfig.get().job.num
        result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        # setup wandb
        wandb.login()
        omega_conf_container = OmegaConf.to_container(config.experiment, resolve=True)
        wandb_init_params = dict(project=config.wandb.project, group=f"exp_{job_id}", reinit=True)
        omega_conf_container["seed"] = 0

        run1 = wandb.init(**wandb_init_params, config=omega_conf_container)

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
            for i in range(config.experiment.n_seeds):
                if i > 0:
                    omega_conf_container["seed"] = i
                    run = wandb.init(**wandb_init_params, config=omega_conf_container)
                else:
                    run = run1

                mean_ret = jnp.atleast_2d(metrics.mean_episode_return)[i]
                mean_len = jnp.atleast_2d(metrics.mean_episode_length)[i]
                step = jnp.atleast_2d(metrics.max_timestep)[i]

                for j in range(len(mean_ret)):
                    run.log({"Mean Episode Return: ": mean_ret[j], "Mean Episode Length: ": mean_len[j]},
                            step=step[j])

                # data = [[x, y, z] for (x, y, z) in zip(jnp.atleast_2d(metrics.mean_episode_return)[i],
                #                                        jnp.atleast_2d(metrics.mean_episode_length[i]),
                #                                        jnp.atleast_2d(metrics.max_timestep[i]))]
                # table = wandb.Table(data=data, columns=["Mean Return", "Mean Length", "Training Step"])
                # run.log({"Mean Episode Return: ": jnp.atleast_2d(metrics.mean_episode_return)[i],
                #          "Mean Episode Length: ": jnp.atleast_2d(metrics.mean_episode_length[i]),
                #          "Timestep: ": jnp.atleast_2d(metrics.max_timestep[i])})
                # #run.log({"plot_return": wandb.plot.line(table, "Step", "Mean Return", title="Mean Return")})
                # #run.log({"plot_length": wandb.plot.line(table, "Step", "Mean Length", title="Mean Length")})

        print(f"Time taken to log metrics: {time.time() - t_start}s")

        runner_state = out["runner_state"]
        ckpt = dict(train_state=runner_state[0], disc_train_state=runner_state[1])

        # Automatically created log directory
        save_ckpt(ckpt, path=result_dir)
        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
