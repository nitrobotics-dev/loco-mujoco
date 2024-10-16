import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax


def get_activation_fn(name: str):
    """ Get activation function by name from the flax.linen module."""
    try:
        # Use getattr to dynamically retrieve the activation function from jax.nn
        return getattr(nn, name)
    except AttributeError:
        raise ValueError(f"Activation function '{name}' not found. Name must be the same as in flax.linen!")


class FullyConnectedNet(nn.Module):

    hidden_layer_dims: Sequence[int]
    output_dim: int
    activation: str = "tanh"
    output_activation: str = None    # none means linear activation
    use_running_mean_stand: bool = True
    squeeze_output: bool = True

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x

    @nn.compact
    def __call__(self, x):

        if self.use_running_mean_stand:
            x = RunningMeanStd()(x)

        # build network
        for i, dim_layer in enumerate(self.hidden_layer_dims):
            x = nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = self.activation_fn(x)

        # add last layer
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        x = self.output_activation_fn(x)

        return jnp.squeeze(x) if self.squeeze_output else x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (1024, 512)

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, x):

        x = RunningMeanStd()(x)

        # build actor
        actor_mean = FullyConnectedNet(self.hidden_layer_dims, self.action_dim, self.activation,
                                       None, False, False)(x)
        actor_logtstd = self.param("log_std", nn.initializers.constant(jnp.log(self.init_std)),
                                   (self.action_dim,))
        if not self.learnable_std:
            actor_logtstd = jax.lax.stop_gradient(actor_logtstd)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # build critic
        critic = FullyConnectedNet(self.hidden_layer_dims, 1, self.activation, None, False, False)(x)

        return pi, jnp.squeeze(critic, axis=-1)


class RunningMeanStd(nn.Module):
    """Layer that maintains running mean and std for input normalization."""

    @nn.compact
    def __call__(self, x):

        x = jnp.atleast_2d(x)

        # Initialize running mean, std, and count
        mean = self.variable('run_stats', 'mean', lambda: jnp.zeros(x.shape[-1]))
        std = self.variable('run_stats', 'std', lambda: jnp.ones(x.shape[-1]))
        count = self.variable('run_stats', 'count', lambda: jnp.array(1e-6))

        # Compute batch mean and variance
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)

        batch_var = jnp.where(batch_var < 1e-3, jnp.ones_like(batch_var), batch_var) # dimensions which almost not change, won't be scaled

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

        return jnp.squeeze(normalized_x)