import os
import sys

from loco_mujoco import LocoEnv
from loco_mujoco.algorithms import PPOJax

import hydra
from omegaconf import DictConfig, OmegaConf
import traceback


@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):

    try:

        os.environ['XLA_FLAGS'] = (
            '--xla_gpu_triton_gemm_any=True ')

        env = LocoEnv.make(**config.experiment.env_params)

        # load train state
        path = "/examples/jax_rl_mimic/ckpts/PPOJax_saved.pkl"
        agent_conf, agent_state = PPOJax.load_agent(path)

        # run eval
        PPOJax.play_policy(env, agent_conf, agent_state, n_steps=1000, n_envs=20, record=True, train_state_seed=0)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
