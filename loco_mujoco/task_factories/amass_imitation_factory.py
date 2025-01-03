from typing import Union, List

from loco_mujoco.environments.base import LocoEnv
from loco_mujoco.smpl.retargeting import load_retargeted_amass_trajectory


class AMASSImitationFactory:
    """
    A factory class for creating imitation learning environments with preloaded AMASS trajectories.

    Methods:
        make(env_name: str, dataset_name: str, **kwargs) -> LocoEnv:
            Creates an environment, loads a retargeted AMASS trajectory, and returns the environment.
    """

    @staticmethod
    def make(env_name: str, rel_dataset_path: Union[str, List[str]], **kwargs) -> LocoEnv:
        """
        Creates and returns an imitation learning environment with a preloaded AMASS trajectory.

        Args:
            env_name (str): The name of the registered environment to create.
            rel_dataset_path (Union[str, List[str]]): The relative path of the AMASS dataset to load.
                The absolute path will be constructed by LOCOMUJOCO_CONVERTED_AMASS_PATH/rel_dataset_path.
                Extensions like ".npz" can be left out. LOCOMUJOCO_CONVERTED_AMASS_PATH must be set.
                As an example: rel_dataset_path="KIT/674/dry_head04_poses".
                Also, list of relative paths can be passed to load multiple trajectories.

            **kwargs: Additional keyword arguments to pass to the environment constructor.

        Returns:
            LocoEnv: An instance of the requested imitation learning environment with the AMASS trajectory preloaded.

        Raises:
            KeyError: If `env_name` is not found in `Mujoco.registered_envs`.
        """

        if env_name not in LocoEnv.registered_envs:
            raise KeyError(f"Environment '{env_name}' is not a registered LocoMuJoCo environment.")

        # Get environment class
        env_cls = LocoEnv.registered_envs[env_name]

        # Create and return the environment
        env = env_cls(**kwargs)

        # Load AMASS Trajectory
        traj = load_retargeted_amass_trajectory(env_name, rel_dataset_path)

        # Add it to the environment
        env.load_trajectory(traj=traj, warn=False)

        return env
