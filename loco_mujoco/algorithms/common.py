import os
from typing import Sequence, NamedTuple, Any

import orbax
from flax.training import orbax_utils
import jax.numpy as jnp

from loco_mujoco.environments.base import TrajState
from loco_mujoco.core.wrappers.mjx import Metrics


class BaseJaxRLAlgorithm:

    @staticmethod
    def get_train_function(env, config):
        raise NotImplementedError

    @staticmethod
    def eval(path, env, config):
        raise NotImplementedError


class Transition(NamedTuple):
    # todo: add absorbing
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    traj_state: TrajState
    metrics: Metrics


def save_ckpt(ckpt, path="ckpts", tag=None, step=0):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    from datetime import datetime
    if tag is None:
        time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tag = time_stamp
    ckpt_dir = os.getcwd() + "/" + path + "/" + tag
    options = orbax.checkpoint.CheckpointManagerOptions(create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        ckpt_dir, orbax_checkpointer, options)
    checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})


def load_train_state(path):
    ckpt_dir = os.getcwd() + "/" + path
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        ckpt_dir, orbax_checkpointer, options)
    step = checkpoint_manager.latest_step()
    raw_restored = checkpoint_manager.restore(step)
    return raw_restored
