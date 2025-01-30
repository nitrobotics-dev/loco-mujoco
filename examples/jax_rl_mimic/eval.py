import os
import sys

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax

import hydra
from omegaconf import DictConfig, OmegaConf
import traceback


@hydra.main(version_base=None, config_path="./config", config_name="conf_h1_real")
def experiment(config: DictConfig):

    try:

        os.environ['XLA_FLAGS'] = (
            '--xla_gpu_triton_gemm_any=True ')

        # get task factory
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

        # create env
        config.experiment.env_params["headless"] = False
        env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

        # load train state
        path = "/examples/jax_rl_mimic/outputs/2024-11-22/17-07-26/PPOJax_saved.pkl"
        agent_conf, agent_state = PPOJax.load_agent(path)

        # run eval
        PPOJax.play_policy(env, agent_conf, agent_state, deterministic=False, n_steps=10000, n_envs=1, record=True, train_state_seed=0)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
