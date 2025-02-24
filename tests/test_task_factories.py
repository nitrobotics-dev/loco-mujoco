import pytest
import jax

from huggingface_hub import list_repo_files

from loco_mujoco.environments import LocoEnv
from loco_mujoco import RLFactory, ImitationFactory
from loco_mujoco.task_factories import DefaultDatasetConf, LAFAN1DatasetConf, CustomDatasetConf

from test_conf import *


# Set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_compilation_cache', False)
jax.config.update('jax_disable_jit', True)
print(f"Jax backend device: {jax.default_backend()} \n")


def get_numpy_env_names():
    """Generates a list of numpy env_names for testing."""
    np_env_list = []
    for env_name in LocoEnv.registered_envs.keys():
        if "Mjx" not in env_name:
            np_env_list.append(env_name)
    return np_env_list


def get_numpy_env_names_default_dataset_conf():
    """Generates a list of numpy env_names for testing (only for envs that have uploaded datasets). """
    repo_id = "robfiras/loco-mujoco-datasets"
    all_files = list_repo_files(repo_id, repo_type="dataset")
    directory = "DefaultDatasets/mocap"  # Adjust to your dataset structure
    np_env_list = set([f.split("/")[-2] for f in all_files if f.startswith(directory)])
    return np_env_list


def get_numpy_env_names_lafan1_dataset_conf():
    """Generates a list of numpy env_names for testing (only for envs that have uploaded datasets). """
    repo_id = "robfiras/loco-mujoco-datasets"
    all_files = list_repo_files(repo_id, repo_type="dataset")
    directory = "Lafan1/mocap"  # Adjust to your dataset structure
    np_env_list = set([f.split("/")[-2] for f in all_files if f.startswith(directory)])
    return np_env_list


def get_custom_traj(env_name):
    """ Generates a custom trajectory for testing. (all zeros) """
    N_steps = 100

    # create the environment
    env_cls = LocoEnv.registered_envs[env_name]
    env = env_cls()

    # reset the env
    key = jax.random.PRNGKey(0)
    env.reset(key)

    # get the model and data of the environment
    model = env.model
    data = env.data

    # get the initial qpos and qvel of the environment
    qpos = data.qpos
    qvel = data.qvel

    # stack qpos and qvel to a trajectory
    qpos = np.tile(qpos, (N_steps, 1))
    qvel = np.tile(qvel, (N_steps, 1))

    # create a trajectory info -- this stores basic information about the trajectory
    njnt = model.njnt
    jnt_type = model.jnt_type.copy()
    jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]
    traj_info = TrajectoryInfo(jnt_names, model=TrajectoryModel(njnt, jnp.array(jnt_type)), frequency=1 / env.dt)

    # create a trajectory data -- this stores the actual trajectory data
    traj_data = TrajectoryData(jnp.array(qpos), jnp.array(qvel), split_points=jnp.array([0, N_steps]))

    # combine them to a trajectory
    traj = Trajectory(traj_info, traj_data)

    return traj


@pytest.mark.parametrize("env_name", get_numpy_env_names())
def test_RLFactory(env_name):
    env = RLFactory.make(env_name)
    assert isinstance(env, LocoEnv.registered_envs[env_name])


@pytest.mark.parametrize("env_name", get_numpy_env_names_default_dataset_conf())
def test_ImitationFactoryDefaultDatasetConf(env_name):
    task = "balance"
    env = ImitationFactory.make(env_name, default_dataset_conf=DefaultDatasetConf(task))
    assert isinstance(env, LocoEnv.registered_envs[env_name])


@pytest.mark.parametrize("env_name", get_numpy_env_names_lafan1_dataset_conf())
def test_ImitationFactorLafan1(env_name):
    dataset_name = "walk1_subject1"
    env = ImitationFactory.make(env_name, lafan1_dataset_conf=LAFAN1DatasetConf(dataset_name))
    assert isinstance(env, LocoEnv.registered_envs[env_name])


@pytest.mark.parametrize("env_name", get_numpy_env_names())
def test_ImitationFactoryCustomDataset(env_name):
    traj = get_custom_traj(env_name)
    env = ImitationFactory.make(env_name, custom_dataset_conf=CustomDatasetConf(traj))
    assert isinstance(env, LocoEnv.registered_envs[env_name])

