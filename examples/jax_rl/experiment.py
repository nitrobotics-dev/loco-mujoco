import os
import jax
import wandb
from loco_mujoco import LocoEnv
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.algorithms import save_ckpt

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):

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
    train_jit = jax.jit(jax.vmap(PPOJax.get_train_function(env, config.experiment)))

    # run training
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, config.experiment.n_seeds+1)
    rng, _rng = rngs[0], rngs[1:]
    out = train_jit(_rng)

    # get the metrics and log them
    if not config.experiment.debug:
        metrics = out["metrics"]
        for i in range(config.experiment.n_seeds):
            if i > 0:
                omega_conf_container["seed"] = i
                run = wandb.init(**wandb_init_params, config=omega_conf_container)
            else:
                run = run1
            data = [[x, y, z] for (x, y, z) in zip(metrics.mean_episode_return[i], metrics.mean_episode_length[i],
                                                   metrics.max_timestep[i])]
            table = wandb.Table(data=data, columns=["Mean Return", "Mean Length", "Step"])
            run.log({"plot_return": wandb.plot.line(table, "Step", "Mean Return", title="Mean Return")})
            run.log({"plot_length": wandb.plot.line(table, "Step", "Mean Length", title="Mean Length")})

    runner_state = out["runner_state"]
    ckpt = dict(train_state=runner_state[0], disc_train_state=runner_state[1])

    # Automatically created log directory
    save_ckpt(ckpt, path=result_dir)
    wandb.finish()


if __name__ == "__main__":
    experiment()
