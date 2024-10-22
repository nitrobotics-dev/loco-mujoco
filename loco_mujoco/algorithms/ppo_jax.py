import wandb
from omegaconf import open_dict
import warnings
from functools import partial
from dataclasses import dataclass
from typing import NamedTuple
from typing import Any
from omegaconf import DictConfig, OmegaConf

import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax

from loco_mujoco.algorithms import (BaseJaxRLAlgorithm, ActorCritic, FullyConnectedNet, Transition, TrainState,
                                    BestTrainStates, TrainStateBuffer, MetriXTransition)
from loco_mujoco.core.wrappers import LogWrapper, LogEnvState, VecEnv, NormalizeVecReward, SummaryMetrics
from loco_mujoco.utils import MetricsHandler, ValidationSummary
from loco_mujoco.algorithms import load_raw_checkpoint


class PPOJax(BaseJaxRLAlgorithm):

    @dataclass(frozen=True)
    class AgentConf:
        config: DictConfig
        network: ActorCritic
        discriminator: FullyConnectedNet

        def serialize(self):
            """
            Serialize the agent configuration and network configuration.

            Returns:
                Serialized agent configuration as a dictionary.

            """
            serialized_network = flax.serialization.to_state_dict(self.network)
            serialized_discriminator = flax.serialization.to_state_dict(self.discriminator)
            conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
            return {"config": conf_dict, "network": serialized_network, "discriminator": serialized_discriminator}

        @classmethod
        def from_dict(cls, d):
            return cls(config=OmegaConf.create(d["config"]),
                       network=flax.serialization.from_state_dict(ActorCritic, d["network"]),
                       discriminator=flax.serialization.from_state_dict(FullyConnectedNet, d["discriminator"]))

    @classmethod
    def init_agent(cls, env, config):

        with (open_dict(config.experiment)):
            config.experiment.num_updates = (
                    config.experiment.total_timesteps // config.experiment.num_steps // config.experiment.num_envs)
            config.experiment.minibatch_size = (
                    config.experiment.num_envs * config.experiment.num_steps // config.experiment.num_minibatches)
            config.experiment.validation_interval = config.experiment.num_updates // config.experiment.validation.num
            config.experiment.validation.num = int(
                config.experiment.num_updates // config.experiment.validation_interval)

        #expert_dataset = env.create_dataset()
        #expert_states = jnp.array(expert_dataset["states"])

        # INIT NETWORK
        network = ActorCritic(
            env.info.action_space.shape[0],
            activation=config.experiment.activation,
            init_std=config.experiment.init_std,
            learnable_std=config.experiment.learnable_std,
            hidden_layer_dims=config.experiment.hidden_layers
        )
        discriminator = FullyConnectedNet(activation=config.experiment.activation,
                                          hidden_layer_dims=config.experiment.hidden_layers,
                                          output_dim=1, output_activation=None,
                                          use_running_mean_stand=True,
                                          squeeze_output=True)

        return cls.AgentConf(config, network, discriminator)

    @classmethod
    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def train_agent(cls, rng, env,
                    agent_conf: AgentConf,
                    mh: MetricsHandler = None,
                    train_state: TrainState = None,
                    disc_train_state: TrainState = None):

        # extract static agent info
        config, network, discriminator = agent_conf.config.experiment, agent_conf.network, agent_conf.discriminator

        assert (train_state is None) == (disc_train_state is None)

        if train_state is None:

            rng, _rng1, _rng2 = jax.random.split(rng, 3)
            init_x = jnp.zeros(env.info.observation_space.shape)
            network_params = network.init(_rng1, init_x)
            discrim_params = discriminator.init(_rng2, init_x)
            if config.anneal_lr:
                tx = optax.chain(
                    optax.clip_by_global_norm(config.max_grad_norm),
                    optax.adam(learning_rate=lambda count: cls.linear_lr_schedule(count, config.num_minibatches,
                                                                                  config.update_epochs, config.lr,
                                                                                  config.num_updates), eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config.max_grad_norm),
                    optax.adam(config.lr, eps=1e-5),
                )
            # disc_tx = optax.adam(config["DISC_LR"], eps=1e-5)

            disc_tx = optax.chain(
                optax.clip_by_global_norm(config.lr),
                optax.adam(config.disc_lr, eps=1e-5),
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

        env = cls.wrap_env(env, config)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = env.reset(reset_rng)

        train_state_buffer = TrainStateBuffer.create(train_state, config.validation.num)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, disc_train_state, env_state, last_obs, train_state_buffer, rng = runner_state

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
                obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                # GET METRICS
                log_env_state = env_state.find(LogEnvState)
                logged_metrics = log_env_state.metrics

                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info, env_state.additional_carry.traj_state,
                    logged_metrics
                )
                runner_state = (train_state, disc_train_state, env_state, obsv, train_state_buffer, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, disc_train_state, env_state, last_obs, train_state_buffer, rng = runner_state
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

                    # todo: deactivated discriminator for now to use mimic reward
                    # # predict reward with discriminator
                    # logits, _ = discriminator.apply({'params': disc_train_state.params,
                    #                                  'run_stats': disc_train_state.run_stats},
                    #                                 obs, mutable=["run_stats"])
                    #
                    # plcy_prob = nn.sigmoid(logits)
                    # reward = jnp.squeeze(-jnp.log(1 - plcy_prob + 1e-6))

                    delta = reward + config.gamma * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config.gamma * config.gae_lambda * (1 - done) * gae
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
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )


                        # CALCULATE PPO ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config.clip_eps,
                                    1.0 + config.clip_eps,
                                )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config.vf_coef * value_loss
                            - config.ent_coef * entropy
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
                batch_size = config.minibatch_size * config.num_minibatches
                assert (
                    batch_size == config.num_steps * config.num_envs
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
                        x, [config.num_minibatches, -1] + list(x.shape[1:])
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
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            rng = update_state[-1]

            def _update_discriminator(runner_state, unused):
                disc_train_state, traj_batch, rng = runner_state
                def _get_one_batch(data, batch_size, rng):
                    idx = jax.random.randint(rng, shape=(batch_size,), minval=0, maxval=data.shape[0])
                    return data[idx]

                def _discrim_loss(params, disc_train_state, inputs, targets):
                    logits, updates = discriminator.apply({'params': params,
                                                           'run_stats': disc_train_state.run_stats},
                                                          inputs, mutable=["run_stats"])

                    # update running statistics
                    disc_train_state.replace(run_stats=updates["run_stats"])

                    # binary cross entropy loss
                    # bce_loss = jnp.maximum(logits, jnp.zeros_like(logits)) - logits * targets + jnp.log(
                    #     1 + jnp.exp(-jnp.abs(logits)))
                    # bce_loss = jnp.mean(bce_loss)
                    #
                    log_p = jax.nn.log_sigmoid(logits)
                    log_not_p = jax.nn.log_sigmoid(-logits)
                    bce_loss = jnp.mean(-targets * log_p - (1. - targets) * log_not_p)

                    # bernoulli entropy
                    discrim_prob = nn.sigmoid(logits)
                    bernoulli_ent = (config.disc_ent_coef *
                                     jnp.mean((1. - discrim_prob) * logits - nn.log_sigmoid(logits)))

                    total_loss = bce_loss - bernoulli_ent

                    if config.debug:

                        def callback(discrim_probs_policy, discrim_probs_exp):
                            wandb.log({"Policy Discriminator Output": jnp.mean(discrim_probs_policy)})
                            wandb.log({"Expert Discriminator Output": jnp.mean(discrim_probs_exp)})

                        #plcy_idxs = jnp.arange(0, config["DISC_MINIBATCH_SIZE"])
                        #exp_idxs = jnp.arange(config["DISC_MINIBATCH_SIZE"], 2*config["DISC_MINIBATCH_SIZE"])
                        #discrim_probs_policy = discrim_prob[plcy_idxs]
                        #discrim_probs_exp = discrim_prob[exp_idxs]
                        #jax.debug.callback(callback, discrim_probs_policy, discrim_probs_exp)

                    return total_loss, disc_train_state

                # Get one batch of policy and expert demonstrations
                rng, _rng1, _rng2 = jax.random.split(rng, 3)
                batch_size = config.disc_minibatch_size

                obs = traj_batch.obs.reshape((-1, traj_batch.obs.shape[-1]))
                plcy_input = _get_one_batch(obs, batch_size, _rng1)
                demo_input = _get_one_batch(expert_states, batch_size, _rng2)

                # Create labels
                plcy_target = jnp.zeros(shape=(plcy_input.shape[0],))
                demo_target = jnp.ones(shape=(demo_input.shape[0],))

                # concatenate inputs and targets
                inputs = jnp.concatenate([plcy_input, demo_input], axis=0)
                targets = jnp.concatenate([plcy_target, demo_target], axis=0)

                # update discriminator
                grad_fn = jax.value_and_grad(_discrim_loss, has_aux=True)
                (total_loss, disc_train_state), grads =\
                    grad_fn(disc_train_state.params, disc_train_state, inputs, targets)

                # TODO: discabled DISCRIM TRAINING!
                #disc_train_state = disc_train_state.apply_gradients(grads=grads)

                runner_state = (disc_train_state, traj_batch, rng)
                return runner_state, None

            counter = ((train_state.step + 1) // config.num_minibatches) // config.update_epochs

            # (disc_train_state, traj_batch, rng), _ = jax.lax.scan(
            #     _update_discriminator, (disc_train_state, traj_batch, rng), xs=None, length=config["N_DISC_EPOCHS"]
            # )

            # disc_train_state, discrim_probs_plcy, discrim_probs_exp, rng =\
            #     jax.lax.cond(counter % config["TRAIN_DISC_INTERVAL"] == 0,
            #                  lambda x, y, z: _update_discriminator(x, y, z),
            #                  lambda x, y, z: (x, z), disc_train_state, traj_batch, rng)

            # disc_train_state, rng = _update_discriminator(disc_train_state,
            #                                               traj_batch,
            #                                               rng)

            #log_env_state = env_state.find(LogEnvState)
            #logged_metrics = log_env_state.metrics
            logged_metrics = traj_batch.metrics
            #jax.debug.breakpoint()
            metric = SummaryMetrics(
                mean_episode_return=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)) / jnp.sum(logged_metrics.done),
                mean_episode_length=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)) / jnp.sum(logged_metrics.done),
                max_timestep=jnp.max(logged_metrics.timestep * config.num_envs),
            )

            def _evaluation_step():

                def _eval_env(runner_state, unused):
                    train_state, disc_train_state, env_state, last_obs, train_state_buffer, rng = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    y, updates = train_state.apply_fn({'params': train_state.params,
                                                       'run_stats': train_state.run_stats},
                                                      last_obs, mutable=["run_stats"])
                    pi, value = y
                    train_state = train_state.replace(run_stats=updates['run_stats'])  # update stats
                    action = pi.sample(seed=_rng)

                    # STEP ENV
                    obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                    # GET METRICS
                    log_env_state = env_state.find(LogEnvState)
                    logged_metrics = log_env_state.metrics

                    transition = MetriXTransition(env_state, logged_metrics)

                    runner_state = (train_state, disc_train_state, env_state, obsv, train_state_buffer, rng)
                    return runner_state, transition

                rng = runner_state[-1]
                reset_rng = jax.random.split(rng, config.validation.num_envs)
                obsv, env_state = env.reset(reset_rng)
                runner_state_eval = (train_state, disc_train_state, env_state, obsv, train_state_buffer, rng)

                # do evaluation runs
                _, traj_batch_eval = jax.lax.scan(
                    _eval_env, runner_state_eval, None, config.validation.num_steps
                )

                env_states = traj_batch_eval.env_state

                validation_metrics = mh(env_states)

                return validation_metrics

            if mh is None:
                validation_metrics = ValidationSummary()
            else:
                validation_metrics = jax.lax.cond(counter % config.validation_interval == 0, _evaluation_step,
                                                   mh.get_zero_container)

            if config.debug:
                def callback(metrics):
                    return_values = metrics.returned_episode_returns[metrics.done]
                    timesteps = metrics.timestep[metrics.done] * config.num_envs

                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                    wandb.log({"Episodic Return": jnp.mean(return_values)})

                jax.debug.callback(callback, env_state.metrics)

            # add train state to buffer if needed
            train_state_buffer = jax.lax.cond(counter % config.validation_interval == 0,
                                              lambda x, y: TrainStateBuffer.add(x, y),
                                              lambda x, y: x, train_state_buffer, train_state)

            runner_state = (train_state, disc_train_state, env_state, last_obs, train_state_buffer, rng)
            return runner_state, (metric, validation_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, disc_train_state, env_state, obsv, train_state_buffer, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config.num_updates
        )
        return {"runner_state": runner_state,
                "training_metrics": metrics[0],
                "validation_metrics": metrics[1]}


    @classmethod
    def get_train_function(cls, env, config):
        num_updates = (
            config.total_timesteps // config.num_steps // config.num_envs
        )
        minibatch_size = (
            config.num_envs * config.num_steps // config.num_minibatches
        )

        with open_dict(config):
            config.validation_interval = num_updates // config.validation.num

        config.validation.num = int(num_updates // config.validation_interval)

        #expert_dataset = env.create_dataset()
        #expert_states = jnp.array(expert_dataset["states"])

        env = cls.wrap_env(env, config)

        # set up the validation if needed
        mh = MetricsHandler(config, env)

        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config.num_minibatches * config.update_epochs))
                / num_updates
            )
            return config.lr * frac

        def train(train_rng):
            # INIT NETWORK
            network = ActorCritic(
                env.info.action_space.shape[0], activation=config.activation, init_std=config.init_std,
                learnable_std=config.learnable_std, hidden_layer_dims=config.hidden_layers
            )
            discriminator = FullyConnectedNet(activation=config.activation, hidden_layer_dims=config.hidden_layers,
                                              output_dim=1, output_activation=None, use_running_mean_stand=True,
                                              squeeze_output=True)

            rng, _rng1, _rng2 = jax.random.split(train_rng, 3)
            init_x = jnp.zeros(env.info.observation_space.shape)
            network_params = network.init(_rng1, init_x)
            discrim_params = discriminator.init(_rng2, init_x)
            if config.anneal_lr:
                tx = optax.chain(
                    optax.clip_by_global_norm(config.max_grad_norm),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config.max_grad_norm),
                    optax.adam(config.lr, eps=1e-5),
                )
            # disc_tx = optax.adam(config["DISC_LR"], eps=1e-5)

            disc_tx = optax.chain(
                optax.clip_by_global_norm(config.lr),
                optax.adam(config.disc_lr, eps=1e-5),
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

            train_state_buffer = TrainStateBuffer.create(train_state, config.validation.num)

            # INIT ENV
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config.num_envs)
            obsv, env_state = env.reset(reset_rng)

            # TRAIN LOOP
            def _update_step(runner_state, unused):
                # COLLECT TRAJECTORIES
                def _env_step(runner_state, unused):
                    train_state, disc_train_state, env_state, last_obs, train_state_buffer, rng = runner_state

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
                    obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                    # GET METRICS
                    log_env_state = env_state.find(LogEnvState)
                    logged_metrics = log_env_state.metrics

                    transition = Transition(
                        done, action, value, reward, log_prob, last_obs, info, env_state.additional_carry.traj_state,
                        logged_metrics
                    )
                    runner_state = (train_state, disc_train_state, env_state, obsv, train_state_buffer, rng)
                    return runner_state, transition

                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config.num_steps
                )

                # CALCULATE ADVANTAGE
                train_state, disc_train_state, env_state, last_obs, train_state_buffer, rng = runner_state
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

                        # todo: deactivated discriminator for now to use mimic reward
                        # # predict reward with discriminator
                        # logits, _ = discriminator.apply({'params': disc_train_state.params,
                        #                                  'run_stats': disc_train_state.run_stats},
                        #                                 obs, mutable=["run_stats"])
                        #
                        # plcy_prob = nn.sigmoid(logits)
                        # reward = jnp.squeeze(-jnp.log(1 - plcy_prob + 1e-6))

                        delta = reward + config.gamma * next_value * (1 - done) - value
                        gae = (
                            delta
                            + config.gamma * config.gae_lambda * (1 - done) * gae
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
                            ).clip(-config.clip_eps, config.clip_eps)
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                            )


                            # CALCULATE PPO ACTOR LOSS
                            ratio = jnp.exp(log_prob - traj_batch.log_prob)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                    jnp.clip(
                                        ratio,
                                        1.0 - config.clip_eps,
                                        1.0 + config.clip_eps,
                                    )
                                    * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()
                            entropy = pi.entropy().mean()

                            total_loss = (
                                loss_actor
                                + config.vf_coef * value_loss
                                - config.ent_coef * entropy
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
                    batch_size = minibatch_size * config.num_minibatches
                    assert (
                        batch_size == config.num_steps * config.num_envs
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
                            x, [config.num_minibatches, -1] + list(x.shape[1:])
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
                    _update_epoch, update_state, None, config.update_epochs
                )
                train_state = update_state[0]
                rng = update_state[-1]

                def _update_discriminator(runner_state, unused):
                    disc_train_state, traj_batch, rng = runner_state
                    def _get_one_batch(data, batch_size, rng):
                        idx = jax.random.randint(rng, shape=(batch_size,), minval=0, maxval=data.shape[0])
                        return data[idx]

                    def _discrim_loss(params, disc_train_state, inputs, targets):
                        logits, updates = discriminator.apply({'params': params,
                                                               'run_stats': disc_train_state.run_stats},
                                                              inputs, mutable=["run_stats"])

                        # update running statistics
                        disc_train_state.replace(run_stats=updates["run_stats"])

                        # binary cross entropy loss
                        # bce_loss = jnp.maximum(logits, jnp.zeros_like(logits)) - logits * targets + jnp.log(
                        #     1 + jnp.exp(-jnp.abs(logits)))
                        # bce_loss = jnp.mean(bce_loss)
                        #
                        log_p = jax.nn.log_sigmoid(logits)
                        log_not_p = jax.nn.log_sigmoid(-logits)
                        bce_loss = jnp.mean(-targets * log_p - (1. - targets) * log_not_p)

                        # bernoulli entropy
                        discrim_prob = nn.sigmoid(logits)
                        bernoulli_ent = (config.disc_ent_coef *
                                         jnp.mean((1. - discrim_prob) * logits - nn.log_sigmoid(logits)))

                        total_loss = bce_loss - bernoulli_ent

                        if config.debug:

                            def callback(discrim_probs_policy, discrim_probs_exp):
                                wandb.log({"Policy Discriminator Output": jnp.mean(discrim_probs_policy)})
                                wandb.log({"Expert Discriminator Output": jnp.mean(discrim_probs_exp)})

                            #plcy_idxs = jnp.arange(0, config["DISC_MINIBATCH_SIZE"])
                            #exp_idxs = jnp.arange(config["DISC_MINIBATCH_SIZE"], 2*config["DISC_MINIBATCH_SIZE"])
                            #discrim_probs_policy = discrim_prob[plcy_idxs]
                            #discrim_probs_exp = discrim_prob[exp_idxs]
                            #jax.debug.callback(callback, discrim_probs_policy, discrim_probs_exp)

                        return total_loss, disc_train_state

                    # Get one batch of policy and expert demonstrations
                    rng, _rng1, _rng2 = jax.random.split(rng, 3)
                    batch_size = config.disc_minibatch_size

                    obs = traj_batch.obs.reshape((-1, traj_batch.obs.shape[-1]))
                    plcy_input = _get_one_batch(obs, batch_size, _rng1)
                    demo_input = _get_one_batch(expert_states, batch_size, _rng2)

                    # Create labels
                    plcy_target = jnp.zeros(shape=(plcy_input.shape[0],))
                    demo_target = jnp.ones(shape=(demo_input.shape[0],))

                    # concatenate inputs and targets
                    inputs = jnp.concatenate([plcy_input, demo_input], axis=0)
                    targets = jnp.concatenate([plcy_target, demo_target], axis=0)

                    # update discriminator
                    grad_fn = jax.value_and_grad(_discrim_loss, has_aux=True)
                    (total_loss, disc_train_state), grads =\
                        grad_fn(disc_train_state.params, disc_train_state, inputs, targets)

                    # TODO: discabled DISCRIM TRAINING!
                    #disc_train_state = disc_train_state.apply_gradients(grads=grads)

                    runner_state = (disc_train_state, traj_batch, rng)
                    return runner_state, None

                counter = ((train_state.step + 1) // config.num_minibatches) // config.update_epochs

                # (disc_train_state, traj_batch, rng), _ = jax.lax.scan(
                #     _update_discriminator, (disc_train_state, traj_batch, rng), xs=None, length=config["N_DISC_EPOCHS"]
                # )

                # disc_train_state, discrim_probs_plcy, discrim_probs_exp, rng =\
                #     jax.lax.cond(counter % config["TRAIN_DISC_INTERVAL"] == 0,
                #                  lambda x, y, z: _update_discriminator(x, y, z),
                #                  lambda x, y, z: (x, z), disc_train_state, traj_batch, rng)

                # disc_train_state, rng = _update_discriminator(disc_train_state,
                #                                               traj_batch,
                #                                               rng)

                #log_env_state = env_state.find(LogEnvState)
                #logged_metrics = log_env_state.metrics
                logged_metrics = traj_batch.metrics
                #jax.debug.breakpoint()
                metric = SummaryMetrics(
                    mean_episode_return=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)) / jnp.sum(logged_metrics.done),
                    mean_episode_length=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)) / jnp.sum(logged_metrics.done),
                    max_timestep=jnp.max(logged_metrics.timestep * config.num_envs),
                )

                def _evaluation_step():

                    def _eval_env(runner_state, unused):
                        train_state, disc_train_state, env_state, last_obs, train_state_buffer, rng = runner_state

                        # SELECT ACTION
                        rng, _rng = jax.random.split(rng)
                        y, updates = train_state.apply_fn({'params': train_state.params,
                                                           'run_stats': train_state.run_stats},
                                                          last_obs, mutable=["run_stats"])
                        pi, value = y
                        train_state = train_state.replace(run_stats=updates['run_stats'])  # update stats
                        action = pi.sample(seed=_rng)

                        # STEP ENV
                        obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                        # GET METRICS
                        log_env_state = env_state.find(LogEnvState)
                        logged_metrics = log_env_state.metrics

                        transition = MetriXTransition(env_state, logged_metrics)

                        runner_state = (train_state, disc_train_state, env_state, obsv, train_state_buffer, rng)
                        return runner_state, transition

                    rng = runner_state[-1]
                    reset_rng = jax.random.split(rng, config.validation.num_envs)
                    obsv, env_state = env.reset(reset_rng)
                    runner_state_eval = (train_state, disc_train_state, env_state, obsv, train_state_buffer, rng)

                    # do evaluation runs
                    _, traj_batch_eval = jax.lax.scan(
                        _eval_env, runner_state_eval, None, config.validation.num_steps
                    )

                    env_states = traj_batch_eval.env_state

                    validation_metrics = mh(env_states)

                    return validation_metrics

                validation_metrics = jax.lax.cond(counter % config.validation_interval == 0, _evaluation_step,
                                                  mh.get_zero_container)

                if config.debug:
                    def callback(metrics):
                        return_values = metrics.returned_episode_returns[metrics.done]
                        timesteps = metrics.timestep[metrics.done] * config.num_envs

                        for t in range(len(timesteps)):
                            print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                        wandb.log({"Episodic Return": jnp.mean(return_values)})

                    jax.debug.callback(callback, env_state.metrics)

                # add train state to buffer if needed
                train_state_buffer = jax.lax.cond(counter % config.validation_interval == 0,
                                                  lambda x, y: TrainStateBuffer.add(x, y),
                                                  lambda x, y: x, train_state_buffer, train_state)

                runner_state = (train_state, disc_train_state, env_state, last_obs, train_state_buffer, rng)
                return runner_state, (metric, validation_metrics)

            rng, _rng = jax.random.split(rng)
            runner_state = (train_state, disc_train_state, env_state, obsv, train_state_buffer, _rng)
            runner_state, metrics = jax.lax.scan(
                _update_step, runner_state, None, num_updates
            )
            return {"runner_state": runner_state,
                    "training_metrics": metrics[0],
                    "validation_metrics": metrics[1]}

        return train, config

    @classmethod
    def play_policy(cls, train_state, env, config, n_envs, n_steps=None, render=True,
                    record=False, rng=None, train_state_seed=0):

        def sample_actions(ts, obs, _rng):
            y, updates = ts.apply_fn({'params': ts.params,
                                      'run_stats': ts.run_stats},
                                     obs, mutable=["run_stats"])
            ts = ts.replace(run_stats=updates['run_stats'])  # update stats
            pi, _ = y
            action = pi.sample(seed=_rng)
            return action, ts

        if not render and n_steps is None and not record:
            warnings.warn("No rendering, no record, no n_steps specified. This will run forever with no effect.")

        # create env
        env = cls.wrap_env(env, config)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        plcy_call = jax.jit(sample_actions)

        # reset env
        obs, env_state = env.reset(env_keys)

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        # take the seed queried for evaluation
        train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

        for i in range(n_steps):

            # SAMPLE ACTION
            rng, _rng = jax.random.split(rng)
            action, train_state = plcy_call(train_state, obs, _rng)

            # STEP ENV
            obs, reward, absorbing, done, info, env_state = env.step(env_state, action)

            # RENDER
            env.mjx_render(env_state, record=record)

        env.stop()

    @classmethod
    def load_train_state(cls, env, path, config, path_is_local=False):
        raw_restored = load_raw_checkpoint(path, path_is_local)
        network = ActorCritic(
            env.info.action_space.shape[0], activation=config.activation, init_std=config.init_std,
            learnable_std=config.learnable_std, hidden_layer_dims=config.hidden_layers
        )
        train_state = TrainState(**raw_restored["train_state"], apply_fn=network.apply, tx=None)
        return train_state

    # @classmethod
    # def save(cls, config, runner_state, ):

    @staticmethod
    def wrap_env(env, config):

        env = LogWrapper(env)
        env = VecEnv(env)
        if config.normalize_env:
            env = NormalizeVecReward(env, config.gamma)
        return env
