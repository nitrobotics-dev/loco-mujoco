from typing import Union, List, Optional
from .base import TaskFactory
from loco_mujoco.environments.base import LocoEnv
from loco_mujoco.smpl.retargeting import load_retargeted_amass_trajectory
from loco_mujoco.smpl.const import AMASS_LOCOMOTION_DATASETS


class AMASSImitationFactory(TaskFactory):
    """
    A factory class for creating imitation learning environments with preloaded AMASS trajectories.

    This class supports loading trajectories either from a specific relative dataset path
    or from a predefined dataset group.
    """

    @staticmethod
    def make(env_name: str,
             rel_dataset_path: Optional[Union[str, List[str]]] = None,
             dataset_group: Optional[str] = None,
             **kwargs) -> LocoEnv:
        """
        Creates and returns an imitation learning environment with a preloaded AMASS trajectory.

        Args:
            env_name (str): The name of the registered environment to create.
            rel_dataset_path (Optional[Union[str, List[str]]]): The relative path(s) of the AMASS dataset to load.
                The absolute path will be constructed by LOCOMUJOCO_CONVERTED_AMASS_PATH/rel_dataset_path.
                Extensions like ".npz" can be left out. LOCOMUJOCO_CONVERTED_AMASS_PATH must be set.
                For example: `rel_dataset_path="KIT/674/dry_head04_poses"`.
                A list of paths can also be passed to load multiple trajectories.
                Either this or `dataset_group` must be set, but not both.
            dataset_group (Optional[str]): A predefined group of datasets to load.
                For example: `AMASS_LOCOMOTION_DATASETS`, which is a collection of relative paths
                focusing on locomotion tasks. Either this or `rel_dataset_path` must be set, but not both.
            **kwargs: Additional keyword arguments to pass to the environment constructor.

        Returns:
            LocoEnv: An instance of the requested imitation learning environment with the AMASS trajectory preloaded.

        Raises:
            ValueError: If both or neither `rel_dataset_path` and `dataset_group` are provided.
            KeyError: If `env_name` is not found in `LocoEnv.registered_envs`.
        """

        if env_name not in LocoEnv.registered_envs:
            raise KeyError(f"Environment '{env_name}' is not a registered LocoMuJoCo environment.")

        if (rel_dataset_path is None and dataset_group is None) or (rel_dataset_path and dataset_group):
            raise ValueError("Either `rel_dataset_path` or `dataset_group` must be set, but not both.")

        # Get environment class
        env_cls = LocoEnv.registered_envs[env_name]

        # Create the environment
        env = env_cls(**kwargs)

        # Determine dataset paths
        if dataset_group:
            if dataset_group == "AMASS_LOCOMOTION_DATASETS":
                dataset_paths = AMASS_LOCOMOTION_DATASETS
            else:
                raise ValueError(f"Unknown dataset group: {dataset_group}")
        else:
            dataset_paths = rel_dataset_path if isinstance(rel_dataset_path, list) else [rel_dataset_path]

        # Load AMASS Trajectory
        traj = load_retargeted_amass_trajectory(env_name, dataset_paths)

        # Add it to the environment
        env.load_trajectory(traj=traj, warn=False)

        return env
