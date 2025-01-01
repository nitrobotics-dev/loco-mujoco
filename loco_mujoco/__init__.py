from pathlib import Path

__version__ = '0.3.0'

try:

    from .core import Mujoco, Mjx
    from .environments import LocoEnv

    def get_all_task_names():
        return LocoEnv.get_all_task_names()

    def get_registered_envs():
        return LocoEnv.registered_envs

    PATH_TO_MODELS = Path(__file__).resolve().parent / "models"
    PATH_TO_SMPL_CONF = Path(__file__).resolve().parent / "smpl" / "conf_paths.yaml"
    PATH_TO_SMPL_ROBOT_CONF = Path(__file__).resolve().parent / "smpl" / "robot_confs"

except ImportError as e:
    print(e)
