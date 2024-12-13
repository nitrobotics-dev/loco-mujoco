import jax.numpy as jnp
from loco_mujoco.core.stateful_object import StatefulObject


class DomainRandomizer(StatefulObject):
    """
    Base interface for all domain randomization classes.

    """

    registered = dict()

    def __init__(self, **randomization_config):
        self.rand_conf = randomization_config

    def reset(self, env, model, data, carry, backend):
        """ Should be called in _mjx_reset_init_data. """
        raise NotImplementedError

    def update(self, env, model, data, carry, backend):
        """ Should be called in simulation_prestep. """
        raise NotImplementedError

    def apply_on_model(self, env, model, data, carry, backend):
        raise NotImplementedError

    def update_observation(self, obs, env, model, data, carry, backend):
        """ Should be called in step_finalize. """
        raise NotImplementedError

    def update_action(self, action, env, model, data, carry, backend):
        """ should be called in preprocess action."""
        raise NotImplementedError

    @classmethod
    def get_name(cls):
        return cls.__name__

    @classmethod
    def register(cls):
        """
        Register a domain randomizer.

        """
        env_name = cls.get_name()

        if env_name not in DomainRandomizer.registered:
            DomainRandomizer.registered[env_name] = cls

    @staticmethod
    def list_registered():
        """
        List registered domain randomizers.

        Returns:
             The list of the registered domain randomizers.

        """
        return list(DomainRandomizer.registered.keys())

    @staticmethod
    def _set_attribute_in_model(model, attribute, value, backend):
        """
        Set an attribute in the model. This works for both, numpy and jax backends.
        """
        if backend == jnp:
            model = model.tree_replace({attribute: value})
        else:
            setattr(model, attribute, value)
        return model
