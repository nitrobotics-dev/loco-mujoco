import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any

from loco_mujoco import LocoEnv
from loco_mujoco.core import Box, MjxState
from loco_mujoco.core.wrappers import MjxRolloutWrapper

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training import train_state
import distrax

#from gymnax.environments import environment, spaces
# from gymnax.wrappers.purerl import GymnaxWrapper
#from brax import envs
#from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    def reset(self, key: chex.PRNGKey, params: Optional = None):
        return self._env.reset(key)

    def step(self, key: chex.PRNGKey, params: Optional = None):
        return self._env.step(key)

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def info(self):
        return self._env.info


@struct.dataclass
class LogEnvState:
    env_state: MjxState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional = None
    ) -> Tuple[chex.Array, MjxState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: MjxState,
        action: Union[int, float],
        params: Optional = None,
    ) -> Tuple[chex.Array, MjxState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info

class BraxGymnaxWrapper:
    def __init__(self, env_name):
        env = LocoEnv.make(env_name)
        # env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        # env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.info.action_space.shape[0]
        self.observation_size = env.info.observation_space.shape

    def reset(self, key, params=None):
        state = self._env.mjx_reset(key)
        return state.observation, state

    def step(self, key, state, action, params=None):
        next_state = self._env.mjx_step(state, action)
        next_obs = jnp.where(next_state.done, next_state.final_observation, next_state.observation)
        return next_obs, next_state, next_state.reward, next_state.done, {}

    def observation_space(self, params):
        # return spaces.Box(
        #     low=-jnp.inf,
        #     high=jnp.inf,
        #     shape=(self._env.observation_size,),
        # )
        return Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=self.observation_size,
        )

    def action_space(self, params):
        # return spaces.Box(
        #     low=-1.0,
        #     high=1.0,
        #     shape=(self._env.action_size,),
        # )
        return Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_size,),
        )

    def create_dataset(self):
        return self._env.create_dataset()

    @property
    def info(self):
        return self._env.info

    def mjx_render_trajectory(self, trajectory, record=False):
        return self._env.mjx_render_trajectory(trajectory, record)


class ClipAction(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)


class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: MjxState


class NormalizeVecObservation(GymnaxWrapper):
    def __init__(self, env, init_mean=None, init_var=None, init_count=None):
        super().__init__(env)
        # use init state tot set running statistics
        self._init_mean = init_mean
        self._init_var = init_var
        self._init_count = init_count

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = NormalizeVecObsEnvState(
            mean=jnp.zeros_like(obs) if self._init_mean is None else self._init_mean,
            var=jnp.ones_like(obs) if self._init_var is None else self._init_var,
            count=1e-4 if self._init_count is None else self._init_count,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state, reward, done, info


@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: MjxState


class NormalizeVecReward(GymnaxWrapper):

    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        return_val = (state.return_val * self.gamma * (1 - done) + reward)

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info


class RunningMeanStd(nn.Module):
    """Layer that maintains running mean and std for input normalization."""

    @nn.compact
    def __call__(self, x):
        # Initialize running mean, std, and count
        mean = self.variable('run_stats', 'mean', lambda: jnp.zeros(x.shape[-1]))
        std = self.variable('run_stats', 'std', lambda: jnp.ones(x.shape[-1]))
        count = self.variable('run_stats', 'count', lambda: jnp.array(1e-6))

        # Compute batch mean and variance
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)

        # Update the running mean and std using a simple moving average
        updated_count = count.value + 1
        new_mean = (count.value * mean.value + batch_mean) / updated_count
        new_std = jnp.sqrt((count.value * (
                    std.value ** 2 + mean.value ** 2) + batch_var + batch_mean ** 2) / updated_count - new_mean ** 2)

        # Normalize the input
        normalized_x = (x - new_mean) / (new_std + 1e-8)

        # Update the parameters
        mean.value = new_mean
        std.value = new_std
        count.value = updated_count

        return normalized_x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):

        x = RunningMeanStd()(x)
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Discriminator(nn.Module):
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        x = RunningMeanStd()(x)
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        z = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        z = activation(z)
        z = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(z)
        z = activation(z)
        z = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(z)
        return jnp.squeeze(z)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class TrainState(train_state.TrainState):
  run_stats: Any


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None

    expert_dataset = env.create_dataset()
    expert_states = jnp.array(expert_dataset["states"])

    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        #env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        discriminator = Discriminator(activation=config["ACTIVATION"])
        rng, _rng1, _rng2 = jax.random.split(rng, 3)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng1, init_x)
        discrim_params = discriminator.init(_rng2, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        disc_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["DISC_LR"], eps=1e-5),
        )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params["params"],
            run_stats=network_params["run_stats"],
            tx=tx,
        )

        disc_train_state = TrainState.create(
            apply_fn=discriminator.apply,
            params=discrim_params["params"],
            run_stats=discrim_params["run_stats"],
            tx=disc_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, disc_train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                y, updates = network.apply({'params': train_state.params,
                                                  'run_stats': train_state.run_stats},
                                                 last_obs, mutable=["run_stats"])
                pi, value = y
                train_state = train_state.replace(run_stats=updates['run_stats'])   # update stats
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, disc_train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, disc_train_state, env_state, last_obs, rng = runner_state
            y, _ = network.apply({'params': train_state.params,
                                              'run_stats': train_state.run_stats},
                                             last_obs, mutable=["run_stats"])
            pi, last_val = y

            def _calculate_gae(traj_batch, last_val, disc_train_state):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward, obs = (
                        transition.done,
                        transition.value,
                        transition.reward,
                        transition.obs
                    )

                    # predict reward with discriminator
                    logits, _ = discriminator.apply({'params': disc_train_state.params,
                                                     'run_stats': disc_train_state.run_stats},
                                                    obs, mutable=["run_stats"])

                    plcy_prob = nn.sigmoid(logits)
                    reward = jnp.squeeze(-jnp.log(1 - plcy_prob + 1e-8))

                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, disc_train_state)

            # UPDATE ACTOR & CRITIC NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        y, _ = network.apply({'params': params, 'run_stats': train_state.run_stats},
                                             traj_batch.obs, mutable=["run_stats"])
                        pi, value = y
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        alpha = config["DPO_ALPHA"]
                        beta = config["DPO_BETA"]
                        log_diff = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(log_diff)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        is_pos = (gae >= 0.0).astype("float32")
                        r1 = ratio - 1.0
                        drift1 = nn.relu(r1 * gae - alpha * nn.tanh(r1 * gae / alpha))
                        drift2 = nn.relu(
                            log_diff * gae - beta * nn.tanh(log_diff * gae / beta)
                        )
                        drift = drift1 * is_pos + drift2 * (1 - is_pos)
                        loss_actor = -(ratio * gae - drift).mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, disc_train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, disc_train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, disc_train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            def _update_discriminator(disc_train_state, traj_batch, rng):

                def _get_one_batch(data, batch_size, rng):
                    permutation = jax.random.permutation(rng, batch_size)
                    shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), data
                    )
                    return shuffled_batch

                def _discrim_loss(params, disc_train_state, inputs, targets):
                    logits, updates = discriminator.apply({'params': params,
                                                           'run_stats': disc_train_state.run_stats},
                                                          inputs, mutable=["run_stats"])

                    # update running statistics
                    disc_train_state.replace(run_stats=updates["run_stats"])

                    # binary cross entropy loss
                    bce_loss = jnp.maximum(logits, jnp.zeros_like(logits)) - logits * targets + jnp.log(
                        1 + jnp.exp(-jnp.abs(logits)))
                    bce_loss = jnp.mean(bce_loss)

                    # bernoulli entropy
                    bernoulli_ent = (config["DISC_ENT_COEF"] *
                                     jnp.mean((1. - nn.sigmoid(logits)) * logits - nn.log_sigmoid(logits)))

                    total_loss = bce_loss - bernoulli_ent

                    return total_loss, disc_train_state

                # Get one batch of policy and expert demonstrations
                rng, _rng1, _rng2 = jax.random.split(rng, 3)
                batch_size = config["DISC_MINIBATCH_SIZE"]

                obs = traj_batch.obs.reshape((-1, traj_batch.obs.shape[-1]))
                plcy_input = _get_one_batch(obs, batch_size, _rng1)
                demo_input = _get_one_batch(expert_states, batch_size, _rng2)

                # Create labels
                plcy_target = jnp.zeros(shape=(plcy_input.shape[0], 1))
                demo_target = jnp.ones(shape=(demo_input.shape[0], 1))

                # concatenate inputs and targets
                inputs = jnp.concatenate([plcy_input, demo_input], axis=0)
                targets = jnp.concatenate([plcy_target, demo_target], axis=0)

                # update discriminator
                grad_fn = jax.value_and_grad(_discrim_loss, has_aux=True)
                (total_loss, disc_train_state), grads = grad_fn(disc_train_state.params, disc_train_state, inputs, targets)
                disc_train_state = disc_train_state.apply_gradients(grads=grads)

                return disc_train_state, rng

            counter = ((train_state.step + 1) // config["NUM_MINIBATCHES"] ) // config["UPDATE_EPOCHS"]

            disc_train_state, rng = jax.lax.cond(counter % config["TRAIN_DISC_INTERVAL"] == 0,
                                                 lambda x, y, z: _update_discriminator(x, y, z),
                                                 lambda x, y, z: (x, z), disc_train_state, traj_batch, rng)

            metric = traj_batch.info
            if config.get("DEBUG"):
                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                jax.debug.callback(callback, metric)

            runner_state = (train_state, disc_train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, disc_train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train
#%%
config = {
    "LR": 1e-4,
    "DISC_LR": 5e-5,
    "NUM_ENVS": 2048,
    "NUM_STEPS": 1, #10
    "TOTAL_TIMESTEPS": 2e7,
    "UPDATE_EPOCHS": 10,    #4
    "TRAIN_DISC_INTERVAL": 3,
    "DISC_MINIBATCH_SIZE": 2048,
    "NUM_MINIBATCHES": 32,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "DPO_ALPHA": 2.0,
    "DPO_BETA": 0.6,
    "ENT_COEF": 1e-3, # 0.0
    "DISC_ENT_COEF": 0.0,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ENV_NAME": "MjxUnitreeH1.walk",
    "ANNEAL_LR": False,
    "NORMALIZE_ENV": True,
    "DEBUG": True,
}
rng = jax.random.PRNGKey(30)
train_jit = jax.jit(make_train(config))
out = train_jit(rng)

import time
# replay policy
runner_state = out["runner_state"]
train_state = runner_state[0]

#env = LocoEnv.make("MjxUnitreeH1.walk", n_envs=100)

# def action_model(policy_params, obs, rng_key):
#     pi, value = network.apply(policy_params, obs)
#     return pi.sample(seed=rng_key)
#
#
# mjx_rollout_env = MjxRolloutWrapper(env, action_model)
# keys = jax.random.split(rng, env.info.n_envs)
# mjx_obs, mjx_action, mjx_reward, mjx_next_obs, mjx_absorbing, mjx_done, mjx_cum_return = (
#     mjx_rollout_env.batch_rollout(rng_keys=keys, n_steps=1000, policy_params=train_state.params))
#
# print("mjx_cum_return: ", jnp.mean(mjx_cum_return[mjx_done]))


# optionally replay trajectory
#env.play_trajectory(n_episodes=10)

# add the wrappers
env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
env = LogWrapper(env)
env = ClipAction(env)
env = VecEnv(env)
if config["NORMALIZE_ENV"]:
    #env = NormalizeVecObservation(env)
    env = NormalizeVecReward(env, config["GAMMA"])

network = ActorCritic(
    env.info.action_space.shape[0], activation=config["ACTIVATION"]
)


key = jax.random.key(0)
keys = jax.random.split(key, env.info.n_envs + 1)
key, env_keys = keys[0], keys[1:]

# jit and vmap all functions needed
# rng_reset = jax.jit(jax.vmap(env.mjx_reset))
# rng_step = jax.jit(jax.vmap(env.mjx_step))

# reset env
obs, state = env.reset(env_keys, None)


# optionally collect rollouts for rendering
rollout = []

step = 0
previous_time = time.time()
LOGGING_FREQUENCY = 100000


for i in range(1000):

    keys = jax.random.split(key, env.info.n_envs + 1)
    key, action_keys = keys[0], keys[1:]

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    y, _ = network.apply({'params': train_state.params, 'run_stats': train_state.run_stats},
                         obs, mutable=["run_stats"])
    pi, _ = y
    action = pi.sample(seed=_rng)

    obs, state, reward, done, info = env.step(None, state, action)

    rollout.append(state.env_state.env_state)

    step += env.info.n_envs
    if step % LOGGING_FREQUENCY == 0:
        current_time = time.time()
        print(f"{int(LOGGING_FREQUENCY / (current_time - previous_time))} steps per second.")
        previous_time = current_time

# Simulate and display video.
env = env._env
env.mjx_render_trajectory(rollout, record=True)

