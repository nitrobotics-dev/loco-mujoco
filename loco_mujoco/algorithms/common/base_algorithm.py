from functools import partial
import jax
from loco_mujoco.core import MjxRolloutWrapper
from loco_mujoco.core.wrappers import LogEnvState
from loco_mujoco.algorithms.common.dataclasses import MetriXTransition


class BaseJaxRLAlgorithm:

    @staticmethod
    def get_train_function(env, config):
        raise NotImplementedError

    @staticmethod
    def load_and_eval(path, env, config):
        raise NotImplementedError


