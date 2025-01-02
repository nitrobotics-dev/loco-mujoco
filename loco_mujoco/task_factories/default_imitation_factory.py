from loco_mujoco.environments.base import LocoEnv


class ImitationFactory:
    """
    A factory class for creating imitation learning environments with preloaded trajectories.

    Methods:
        make(env_name: str, task: str, dataset_type: str, debug: bool = False, **kwargs) -> LocoEnv:
            Creates an environment, loads a trajectory based on the task and dataset type, and returns the environment.

        get_traj_path(env_cls, dataset_type: str, task: str, debug: bool) -> str:
            Determines the path to the trajectory file based on the dataset type, task, and debug mode.
    """

    @classmethod
    def make(cls, env_name: str, task: str, dataset_type: str, debug: bool = False, **kwargs) -> LocoEnv:
        """
        Creates and returns an imitation learning environment with the specified parameters.

        Args:
            env_name (str): The name of the registered environment to create.
            task (str): The main task to solve, used to select the appropriate trajectory.
            dataset_type (str): The type of dataset to use, either "real" or "perfect".
            debug (bool, optional): Whether to use debug mode for smaller datasets. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the environment constructor.

        Returns:
            LocoEnv: An instance of the requested imitation learning environment with the trajectory preloaded.

        Raises:
            ValueError: If the `dataset_type` is unknown.
        """

        if env_name not in LocoEnv.registered_envs:
            raise KeyError(f"Environment '{env_name}' is not a registered LocoMuJoCo environment.")

        # Get environment class
        env_cls = LocoEnv.registered_envs[env_name]

        # Create and return the environment
        env = env_cls(**kwargs)

        # Load the trajectory
        traj_path = cls.get_traj_path(env_cls, dataset_type, task, debug)
        env.load_trajectory(traj_path=traj_path, warn=False)

        return env

    @staticmethod
    def get_traj_path(env_cls, dataset_type: str, task: str, debug: bool) -> str:
        """
        Determines the path to the trajectory file based on the dataset type, task, and debug mode.

        Args:
            env_cls: The class of the environment, which provides dataset paths.
            dataset_type (str): The type of dataset to use, either "real" or "perfect".
            task (str): The main task to solve, used to select the appropriate trajectory.
            debug (bool): Whether to use debug mode for smaller datasets.

        Returns:
            str: The path to the trajectory file.

        Raises:
            ValueError: If the `dataset_type` is unknown.
        """

        if dataset_type == "real":
            traj_path = str(env_cls.path_to_real_datasets() / (task + ".npz"))
            if debug:
                traj_path = traj_path.split("/")
                traj_path.insert(3, "mini_datasets")
                traj_path = "/".join(traj_path)

        elif dataset_type == "perfect":
            # TODO: Adapt this for the new trajectory data format
            traj_path = None
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        return traj_path
