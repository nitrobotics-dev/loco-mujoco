__version__ = '0.3.0'

try:

    from .core import Mujoco, Mjx
    from .environments import LocoEnv

    def get_all_task_names():
        return LocoEnv.get_all_task_names()

    def get_registered_envs():
        return LocoEnv.registered_envs

except ImportError as e:
    print(e)
