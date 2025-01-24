import mujoco.mjx
import jax
import numpy as np
from mujoco import MjSpec

from loco_mujoco.core.observations import ObservationType
from loco_mujoco.environments import LocoEnv

from test_conf import DummyHumamoidEnv


DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs":1}


def test_BodyPos():

    seed = 0
    key = jax.random.PRNGKey(seed)
    body_name1 = "left_shin"
    body_name2 = "right_thigh"

    # specify the observation space
    observation_spec = [{"obs_name": "name_obs1", "type": "BodyPos", "xml_name": body_name1},
                        {"obs_name": "name_obs2", "type": "BodyPos", "xml_name": body_name2}]

    # specify the name of the actuators of the xml
    action_spec = ["abdomen_y"]  # --> use more motors if needed

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(enable_mjx=True,
                               actuation_spec=action_spec,
                               observation_spec=observation_spec,
                               **DEFAULTS)

    # reset the environment in Mujoco
    obs = mjx_env.reset(key)

    # reset the environment in Mjx
    state = mjx_env.mjx_reset(key)
    obs_mjx = state.observation

    # get the body position from data
    model = mjx_env.get_model()
    body_ids = []
    for name in [body_name1, body_name2]:
        body_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))

    # get the body position from Mujoco
    data = mjx_env.get_data()
    body_pos1 = np.array(data.xpos[body_ids[0]])
    body_pos2 = np.array(data.xpos[body_ids[1]])
    gt_obs_mujoco = np.concatenate([body_pos1, body_pos2])

    # get the body position from Mjx
    body_pos1 = np.array(state.data.xpos[body_ids[0]])
    body_pos2 = np.array(state.data.xpos[body_ids[1]])
    gt_obs_mjx = np.concatenate([body_pos1, body_pos2])

    # check the observation
    np.array_equal(obs, gt_obs_mujoco)
    np.array_equal(obs, gt_obs_mjx)





