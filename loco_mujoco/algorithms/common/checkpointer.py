import os
import orbax
import jax
import jax.numpy as jnp
from flax.training import orbax_utils
from loco_mujoco.algorithms.common.dataclasses import BestTrainStates, TrainState


def save_ckpt(ckpt, path="ckpts", tag=None, step=0, path_is_local=True):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    from datetime import datetime
    if tag is None:
        time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tag = time_stamp
    if path_is_local:
        ckpt_dir = os.getcwd() + "/" + path + "/" + tag
    else:
        ckpt_dir = path + "/" + tag
    options = orbax.checkpoint.CheckpointManagerOptions(create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        ckpt_dir, orbax_checkpointer, options)
    checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})


def load_raw_checkpoint(path, path_is_local=True):
    if path_is_local:
        ckpt_dir = os.getcwd() + "/" + path
    else:
        ckpt_dir = path
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        ckpt_dir, orbax_checkpointer, options)
    step = checkpoint_manager.latest_step()
    raw_restored = checkpoint_manager.restore(step)
    return raw_restored


def add_trainstate(best_states: BestTrainStates, new_state: TrainState,
                   metric: float, time_step: int) -> BestTrainStates:
    def add(best_states, train_state, metric, time_step):
        # Combine new state and metric with the existing best states and metrics
        combined_states = jax.tree_map(lambda a, b: jnp.concatenate([a, b]), best_states.states, train_state)
        combined_metrics = jnp.concatenate([best_states.metrics, jnp.array([metric])])
        combined_iterations = jnp.concatenate([best_states.iterations, jnp.array([time_step])])

        # Get the indices of the N greatest metrics
        top_indices = jnp.argsort(combined_metrics, descending=True)[:best_states.n]

        # Select the top N states and metrics
        top_states = jax.tree_map(lambda x: x[top_indices], combined_states)
        top_metrics = combined_metrics[top_indices]
        top_iterations = combined_iterations[top_indices]

        size = jax.lax.cond(best_states.size < best_states.n, lambda x: x + 1, lambda x: x,
                            best_states.size)

        return BestTrainStates(states=top_states, metrics=top_metrics, n=best_states.n, size=size,
                               cur_worst_perf=top_metrics[best_states.n], iterations=top_iterations)

    best_states = jax.lax.cond(metric > best_states.cur_worst_perf, add, lambda w, x, y, z: w,
                               best_states, new_state, metric, time_step)

    return best_states
