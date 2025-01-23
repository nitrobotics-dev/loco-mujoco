import os
import sys

from loco_mujoco import LocoEnv
from loco_mujoco.algorithms import PPOJax

import hydra
from omegaconf import DictConfig, OmegaConf
import traceback


@hydra.main(version_base=None, config_path="./config", config_name="conf_h1")
def experiment(config: DictConfig):

    try:

        os.environ['XLA_FLAGS'] = (
            '--xla_gpu_triton_gemm_any=True ')

        config.experiment.env_params["headless"] = False
        env = LocoEnv.make(goal_params=dict(visualize_goal=True), **config.experiment.env_params)

        # load train state
        path = "/path/to/the/generated/pkl/file"
        agent_conf, agent_state = PPOJax.load_agent(path)

        # run eval
        PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=10000, n_envs=50, record=True, train_state_seed=1)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
