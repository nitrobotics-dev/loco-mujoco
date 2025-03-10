import os
import sys

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import AMPJax

import hydra
from omegaconf import DictConfig, OmegaConf
import traceback


os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

# load train state
path = "/home/moore/PycharmProjects/loco-mujoco/examples/jax_amp/multirun/2025-02-14/23-23-51/0/AMPJax_saved.pkl"
agent_conf, agent_state = AMPJax.load_agent(path)
config = agent_conf.config

# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

# run eval mjx
AMPJax.play_policy(env, agent_conf, agent_state, deterministic=False, n_steps=10000, n_envs=1, record=True,
                   train_state_seed=0)

# run eval mujoco
# AMPJax.play_policy_mujoco(env, agent_conf, agent_state, deterministic=False, n_steps=10000, record=True,
#                           train_state_seed=0)
