# Simple Mujoco Interface for RL Environments

A simple **unifying** interface that supports both **Mujoco-CPU**
and **Mujoco-Mjx** environments.

## Example
Here is a simple example to run a Mujoco-CPU environment:

```python
from loco_mujoco.core import Mujoco, Mjx, ObservationType

# specify what observation you would like to retriev from the xml
# --> checkout ObservationType to see what observations are supported by default
observation_spec = [("name_obs_1", "joint_name_in_xml", ObservationType.JOINT_POS),
                    ("name_obs_1", "joint_name_in_xml", ObservationType.JOINT_VEL),
                    ("name_obs_2", "body_name_in_xml", ObservationType.BODY_POS)]

# specify the name of the actuators of the xml
action_spec = ["name_actuator_in_xml1", "name_actuator_in_xml2"]

# define a simple Mujoco environment (CPU)
env = Mujoco(spec="/path/to/xml_file.xml",
             actuation_spec=action_spec,
             observation_spec=observation_spec,
             horizon=1000,
             gamma=0.99)

# get action dimensionality
action_dim = env.info.action_space.shape[0]

env.reset()
env.render()

while True:
    for i in range(500):
        env.step(np.random.randn(action_dim))
        env.render()
    env.reset()
```

Similarily, we can run a Mujoco-Mjx environment:
```python
from loco_mujoco.core import Mujoco, Mjx, ObservationType

# specify what observation you would like to retriev from the xml
# --> checkout ObservationType to see what observations are supported by default
observation_spec = [("name_obs_1", "joint_name_in_xml", ObservationType.JOINT_POS),
                    ("name_obs_1", "joint_name_in_xml", ObservationType.JOINT_VEL),
                    ("name_obs_2", "body_name_in_xml", ObservationType.BODY_POS)]

# specify the name of the actuators of the xml
action_spec = ["name_actuator_in_xml1", "name_actuator_in_xml2"]

mjx_env = Mjx(xml_file="/home/moore/PycharmProjects/MjxTest/data/unitree_h1/h1.xml",
          actuation_spec=action_spec,
          observation_spec=observation_spec,
          horizon=1000,
          n_envs=4000,
          gamma=0.99)


action_dim = mjx_env.info.action_space.shape[0]

LOGGING_FREQUENCY = 100000
global_key = jax.random.PRNGKey(165416)  # Random seed is explicit in JAX
keys = jax.random.split(global_key, mjx_env.info.n_envs + 1)
global_key, env_keys = keys[0], keys[1:]

def sample():
    global global_key
    global_key, subkey = jax.random.split(global_key)
    action = jax.random.uniform(subkey, minval=mjx_env.info.action_space.low, maxval=mjx_env.info.action_space.high,
                                shape=(mjx_env.info.n_envs, action_dim))
    return action

sample_X = jax.jit(sample)
state = mjx_env.mjx_reset(env_keys)
step = 0
previous_time = time.time()

while True:

    action = sample_X()
    state = mjx_env.mjx_step(state, action)
```
Note that the Mjx environment can do everything the Mujoco environment can do, even running the simulation on CPU. Hence, 
we can still do a CPU rollout:
```python
mjx_env.reset()
mjx_env.render()

while True:
    for i in range(500):
        mjx_env.step(np.random.randn(action_dim))
        mjx_env.render()
    mjx_env.reset()
```
When using a Mjx environment, the `mjx_step` and `mjx_reset` functions are a jitted functions that run the simulation
on the GPU, while the `step` and `reset` functions are running the simulation on the CPU. It is important
to note that `mjx_step` does asyhronous resetting of each environment simiar to vector environments of Gymansium and 
stable-baselines3. `mjx_reset` resets all the environments at once.
