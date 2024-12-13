import jax.numpy as jnp
from loco_mujoco.core.stateful_object import StatefulObject


class Terrain(StatefulObject):
    """
        Base interface for all terrain classes.

    """

    registered = dict()

    # if the hfield is generated dynamically, this should be set to True to update the hfield in the viewer
    viewer_needs_to_update_hfield = False

    def __init__(self, env, **terrain_config):
        self.terrain_conf = terrain_config

        # to be specified from spec
        self.hfield_id = None

    def reset(self, env, model, data, carry, backend):
        """ Should be called in _mjx_reset_init_data. """
        raise NotImplementedError

    def update(self, env, model, data, carry, backend):
        """ should be called in simulation_pre_step."""
        raise NotImplementedError

    def modify_spec(self, spec):
        raise NotImplementedError

    @property
    def is_dynamic(self):
        raise NotImplementedError

    @property
    def requires_spec_modification(self):
        return self.__class__.modify_spec != Terrain.modify_spec

    @classmethod
    def get_name(cls):
        return cls.__name__

    @classmethod
    def register(cls):
        """
        Register a terrain.

        """
        env_name = cls.get_name()

        if env_name not in Terrain.registered:
            Terrain.registered[env_name] = cls

    @staticmethod
    def list_registered():
        """
        List registered terrains.

        Returns:
             The list of the registered terrains.

        """
        return list(Terrain.registered.keys())

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