import numpy as np
from huggingface_hub import list_repo_files
import gc
import jax
import os
from multiprocessing import Process

from loco_mujoco import ImitationFactory
from loco_mujoco.task_factories import DefaultDatasetConf, LAFAN1DatasetConf
from loco_mujoco.utils import video2gif

def run_experiment(dataset_type, env_name, dataset_name, dataset_source, make_gif, compress):
    """ Runs a single experiment in a separate process to avoid memory leaks. """

    os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Force JAX to use CPU
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable memory preallocation


    repo_id = "robfiras/loco-mujoco-datasets"
    save_path_video = f"./record_all/{dataset_type}"

    # Remove .npz extension
    dataset_name = dataset_name.split(".")[0]

    if dataset_type == "DefaultDatasets":
        mdp = ImitationFactory.make(env_name,
                                    default_dataset_conf=DefaultDatasetConf(dataset_name, dataset_source),
                                    headless=True,
                                    recorder_params=dict(tag=env_name, path=save_path_video,
                                                         video_name=dataset_name, compress=compress))
    elif dataset_type == "Lafan1":
        mdp = ImitationFactory.make(env_name,
                                    lafan1_dataset_conf=LAFAN1DatasetConf(dataset_name),
                                    headless=True,
                                    recorder_params=dict(tag=env_name, path=save_path_video,
                                                         video_name=dataset_name, compress=compress))
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    print(f"Recording {env_name} with dataset {dataset_name}...")

    replay_params = dict(n_episodes=1, n_steps_per_episode=1000, record=True)
    mdp.play_trajectory(recorder_params=dict(tag=env_name, path=save_path_video, compress=compress), **replay_params)

    path_video = f"{save_path_video}/{env_name}/{dataset_name}.mp4"

    if make_gif:
        duration = 6.0
        video2gif(path_video, duration=duration, fps=15, scale=540)

    mdp.stop()

    del mdp
    gc.collect()  # Force garbage collection
    jax.clear_caches()  # Free JAX backend memory (if using JAX)


def experiment(seed=0):
    np.random.seed(seed)

    dataset_types = ["Lafan1", "DefaultDatasets"]
    dataset_source = "mocap"
    make_gif = True
    compress = True

    repo_id = "robfiras/loco-mujoco-datasets"
    all_files = list_repo_files(repo_id, repo_type="dataset")
    N_workers = 1

    processes = []

    for dataset_type in dataset_types:
        directory = f"{dataset_type}/{dataset_source}"  # Adjust to your dataset structure
        env_names_and_datasets = [f.split("/")[2:] for f in all_files if f.startswith(directory)]

        for env_name, dataset_name in env_names_and_datasets:
            p = Process(target=run_experiment, args=(dataset_type, env_name, dataset_name, dataset_source, make_gif, compress))
            p.start()
            processes.append(p)

            # Optional: Limit the number of concurrent processes (adjust this based on system RAM)
            if len(processes) >= N_workers:  # Change 4 to the number of parallel processes you want
                for p in processes:
                    p.join()
                processes = []

    # Ensure all remaining processes are completed
    for p in processes:
        p.join()


if __name__ == '__main__':
    experiment()
