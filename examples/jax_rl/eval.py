import os
import sys
import jax
import jax.numpy as jnp
import wandb
from dataclasses import fields
from loco_mujoco import LocoEnv
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.algorithms import save_ckpt
from loco_mujoco.utils.metrics import ValidationSummary, QuantityContainer
from loco_mujoco.algorithms import load_raw_checkpoint

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import traceback

import orbax
from loco_mujoco.algorithms import (BaseJaxRLAlgorithm, ActorCritic, FullyConnectedNet, Transition, TrainState,
                                    BestTrainStates, TrainStateBuffer, MetriXTransition)


@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):

    try:

        os.environ['XLA_FLAGS'] = (
            '--xla_gpu_triton_gemm_any=True ')

        env = LocoEnv.make(**config.experiment.env_params)

        # load train state
        checkpoint_path = "ckpts/20241022_013158"
        train_state = PPOJax.load_train_state(env, checkpoint_path, config.experiment, path_is_local=False)

        # run eval
        PPOJax.play_policy(train_state, env, config.experiment, n_steps=1000, n_envs=20, record=True)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
