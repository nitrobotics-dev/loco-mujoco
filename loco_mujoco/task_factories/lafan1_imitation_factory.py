from typing import Union, List, Optional
from .base import TaskFactory
from loco_mujoco.environments.base import LocoEnv
from loco_mujoco.datasets.humanoids.LAFAN1 import load_lafan1_trajectory
from loco_mujoco.datasets.humanoids.LAFAN1 import (LAFAN1_LOCOMOTION_DATASETS,
                                                   LAFAN1_DANCE_DATASETS, LAFAN1_ALL_DATASETS)


class LAFAN1ImitationFactory(TaskFactory):
    """
    A factory class for creating imitation learning environments with preloaded LAFAN1 trajectories.

    This class supports loading trajectories either from a specific relative dataset path
    or from a predefined dataset group.
    """

    @staticmethod
    def make(env_name: str,
             dataset_name: Optional[Union[str, List[str]]] = None,
             dataset_group: Optional[str] = None,
             terminal_state_type: str = "RootPoseTrajTerminalStateHandler",
             init_state_type: str = "TrajInitialStateHandler",
             **kwargs) -> LocoEnv:
        """
        Creates and returns an imitation learning environment with a preloaded LAFAN1 trajectory.

        Args:
            env_name (str): The name of the registered environment to create.
            dataset_name (Optional[Union[str, List[str]]]): The name(s) of the LAFAN1 dataset to load.
                The absolute path will be constructed by LOCOMUJOCO_CONVERTED_LAFAN1_PATH/{robot_name}/dataset_name.
                Extensions like ".csv" can be left out. LOCOMUJOCO_CONVERTED_LAFAN1_PATH must be set.
                For example: `dataset_name="dance1_subject1"`.
                A list of paths can also be passed to load multiple trajectories.
                Either this or `dataset_group` must be set, but not both.
            dataset_group (Optional[str]): A predefined group of datasets to load.
                For example: `LAFAN1_LOCOMOTION_DATASETS`, which is a collection of relative paths
                focusing on locomotion tasks. Either this or `rel_dataset_path` must be set, but not both.
            terminal_state_type: str ="RootPoseTrajTerminalStateHandler",
            init_state_type: str = "TrajInitialStateHandler",
            **kwargs: Additional keyword arguments to pass to the environment constructor.

        Returns:
            LocoEnv: An instance of the requested imitation learning environment with the LAFAN1 trajectory preloaded.

        Raises:
            ValueError: If both or neither `rel_dataset_path` and `dataset_group` are provided.
            KeyError: If `env_name` is not found in `LocoEnv.registered_envs`.
        """

        if env_name not in LocoEnv.registered_envs:
            raise KeyError(f"Environment '{env_name}' is not a registered LocoMuJoCo environment.")

        if (dataset_name is None and dataset_group is None) or (dataset_name and dataset_group):
            raise ValueError("Either `dataset_name` or `dataset_group` must be set, but not both.")

        # Get environment class
        env_cls = LocoEnv.registered_envs[env_name]

        # Create the environment
        env = env_cls(init_state_type=init_state_type, terminal_state_type=terminal_state_type, **kwargs)

        # Determine dataset paths
        if dataset_group:
            if dataset_group == "LAFAN1_LOCOMOTION_DATASETS":
                dataset_paths = LAFAN1_LOCOMOTION_DATASETS
            elif dataset_group == "LAFAN1_DANCE_DATASETS":
                dataset_paths = LAFAN1_DANCE_DATASETS
            elif dataset_group == "LAFAN1_ALL_DATASETS":
                dataset_paths = LAFAN1_ALL_DATASETS
            else:
                raise ValueError(f"Unknown dataset group: {dataset_group}")
        else:
            dataset_paths = dataset_name if isinstance(dataset_name, list) else [dataset_name]

        # Load LAFAN1 Trajectory
        traj = load_lafan1_trajectory(env_name, dataset_paths)

        # Add it to the environment
        env.load_trajectory(traj=traj, warn=False)

        return env
