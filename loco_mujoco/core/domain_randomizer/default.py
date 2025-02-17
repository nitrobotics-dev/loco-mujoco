from typing import Any, Union, Tuple
from types import ModuleType

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.domain_randomizer import DomainRandomizer
from loco_mujoco.core.utils.backend import assert_backend_is_supported


@struct.dataclass
class DefaultRandomizerState:
    """
    Represents the state of the default randomizer.

    Attributes:
        geom_friction (Union[np.ndarray, jax.Array]): Friction parameters for geometry.
    """
    geom_friction: Union[np.ndarray, jax.Array]
    com_displacement: Union[np.ndarray, jax.Array]
    link_mass_multipliers: Union[np.ndarray, jax.Array]
    joint_friction_loss: Union[np.ndarray, jax.Array]
    joint_damping: Union[np.ndarray, jax.Array]
    joint_armature: Union[np.ndarray, jax.Array]


class DefaultRandomizer(DomainRandomizer):
    """
    A domain randomizer that modifies simulation parameters like geometry friction.
    """

    def init_state(self, env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType) -> DefaultRandomizerState:
        """
        Initialize the randomizer state.

        Args:
            env (Any): The environment instance.
            key (Any): Random seed key.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            DefaultRandomizerState: The initialized randomizer state.
        """
        assert_backend_is_supported(backend)
        return DefaultRandomizerState(geom_friction=backend.array([0.0, 0.0, 0.0]), 
                                      com_displacement=backend.array([0.0, 0.0, 0.0]),
                                      link_mass_multipliers=backend.array([1.0] * (model.nbody-1)), #exclude worldbody
                                      joint_friction_loss=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
                                      joint_damping=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
                                      joint_armature=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
                                      )

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the randomizer, applying domain randomization.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.
        """
        assert_backend_is_supported(backend)
        domain_randomizer_state = carry.domain_randomizer_state

        # update different randomization parameters
        geom_friction, carry = self._sample_geom_friction(model, carry, backend)
        com_displacement, carry = self._sample_com_displacement(model, carry, backend)
        link_mass_multipliers, carry = self._sample_link_mass_multipliers(model, carry, backend)
        joint_friction_loss, carry = self._sample_joint_friction_loss(model, carry, backend)
        joint_damping, carry = self._sample_joint_damping(model, carry, backend)
        joint_armature, carry = self._sample_joint_armature(model, carry, backend)

        if "PDControl" in str(env._control_func):
            control_func_state = carry.control_func_state

            p_noise, carry = self._sample_p_gains_noise(env, model, carry, backend)
            d_noise, carry = self._sample_d_gains_noise(env, model, carry, backend)
            carry = carry.replace(control_func_state=control_func_state.replace(p_gain_noise=p_noise,
                                                                                d_gain_noise=d_noise,
                                                                                pos_offset=backend.zeros_like(env._control_func._nominal_joint_positions),
                                                                                ctrl_mult=backend.ones_like(env._control_func._nominal_joint_positions)))



        carry = carry.replace(domain_randomizer_state=domain_randomizer_state.replace(geom_friction=geom_friction, 
                                                                                      com_displacement=com_displacement,
                                                                                      link_mass_multipliers=link_mass_multipliers,
                                                                                      joint_friction_loss=joint_friction_loss,
                                                                                      joint_damping=joint_damping,
                                                                                      joint_armature=joint_armature,
                                                                                      ))


        return data, carry

    def update(self, env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        """
        Update the randomizer by applying the state changes to the model.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjModel, Model], Union[MjData, Data], Any]: The updated simulation model, data, and carry.
        """
        assert_backend_is_supported(backend)
        info_props = env._get_all_info_properties()
        root_body_name = info_props["root_body_name"]
        root_body_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)

        domrand_state = carry.domain_randomizer_state
        model = self._set_attribute_in_model(model, "geom_friction", domrand_state.geom_friction, backend)
        model = self._set_attribute_in_model(model, "body_ipos", model.body_ipos.at[root_body_id].set(model.body_ipos[root_body_id] + domrand_state.com_displacement), backend)
        model = self._set_attribute_in_model(model, "body_mass", model.body_mass.at[root_body_id:].set(model.body_mass[root_body_id:] * domrand_state.link_mass_multipliers), backend)
        model = self._set_attribute_in_model(model, "dof_frictionloss", model.dof_frictionloss.at[6:].set(domrand_state.joint_friction_loss), backend)
        model = self._set_attribute_in_model(model, "dof_damping", model.dof_damping.at[6:].set(domrand_state.joint_damping), backend)
        model = self._set_attribute_in_model(model, "dof_armature", model.dof_armature.at[6:].set(domrand_state.joint_armature), backend)

        return model, data, carry

    def update_observation(self, env: Any,
                           obs: Union[np.ndarray, jnp.ndarray],
                           model: Union[MjModel, Model],
                           data: Union[MjData, Data],
                           carry: Any,
                           backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Update the observation with randomization effects.

        Args:
            env (Any): The environment instance.
            obs (Union[np.ndarray, jnp.ndarray]): The observation to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The updated observation and carry.
        """
        assert_backend_is_supported(backend)

        ind_of_all_joint_pos = env._obs_indices.JointPos
        ind_of_all_joint_vel = env._obs_indices.JointVel
        ind_of_gravity_vec = env._obs_indices.ProjectedGravityVector
        ind_of_lin_vel = env._obs_indices.FreeJointVel[:3]
        ind_of_ang_vel = env._obs_indices.FreeJointVel[3:]

        joint_pos_noise_scale = self.rand_conf["joint_pos_noise_scale"]
        joint_vel_noise_scale = self.rand_conf["joint_vel_noise_scale"]
        gravity_noise_scale = self.rand_conf["gravity_noise_scale"]
        lin_vel_noise_scale = self.rand_conf["lin_vel_noise_scale"]
        ang_vel_noise_scale = self.rand_conf["ang_vel_noise_scale"]

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            noise = jax.random.normal(
                _k, 
                shape=(
                    len(ind_of_all_joint_pos) + 
                    len(ind_of_all_joint_vel) + 
                    len(ind_of_gravity_vec) + 
                    len(ind_of_lin_vel) + 
                    len(ind_of_ang_vel),
                )
            )

            # Add noise to joint positions
            if self.rand_conf["add_joint_pos_noise"]:
                obs = obs.at[ind_of_all_joint_pos].add(noise[:len(ind_of_all_joint_pos)] * joint_pos_noise_scale)
            
            # Add noise to joint velocities
            if self.rand_conf["add_joint_vel_noise"]:
                obs = obs.at[ind_of_all_joint_vel].add(
                    noise[len(ind_of_all_joint_pos):len(ind_of_all_joint_pos) + len(ind_of_all_joint_vel)] * joint_vel_noise_scale
                )
            
            # Add noise to gravity vector
            if self.rand_conf["add_gravity_noise"]:
                obs = obs.at[ind_of_gravity_vec].add(
                    noise[
                        len(ind_of_all_joint_pos) + len(ind_of_all_joint_vel):
                        len(ind_of_all_joint_pos) + len(ind_of_all_joint_vel) + len(ind_of_gravity_vec)
                    ] * gravity_noise_scale
                )
            
            # Add noise to linear velocities
            if self.rand_conf["add_free_joint_lin_vel_noise"]:
                obs = obs.at[ind_of_lin_vel].add(
                    noise[
                        len(ind_of_all_joint_pos) + len(ind_of_all_joint_vel) + len(ind_of_gravity_vec):
                        len(ind_of_all_joint_pos) + len(ind_of_all_joint_vel) + len(ind_of_gravity_vec) + len(ind_of_lin_vel)
                    ] * lin_vel_noise_scale
                )

            # Add noise to angular velocities
            if self.rand_conf["add_free_joint_ang_vel_noise"]:
                obs = obs.at[ind_of_ang_vel].add(
                    noise[
                        len(ind_of_all_joint_pos) + len(ind_of_all_joint_vel) + len(ind_of_gravity_vec) + len(ind_of_lin_vel):
                    ] * ang_vel_noise_scale
                )
            
            carry = carry.replace(key=key)

        else:
            noise = np.random.normal(
                size=(
                    len(ind_of_all_joint_pos) + 
                    len(ind_of_all_joint_vel) + 
                    len(ind_of_gravity_vec) + 
                    len(ind_of_lin_vel) + 
                    len(ind_of_ang_vel),
                )
            )

             # Add noise to joint positions
            if self.rand_conf["add_joint_pos_noise"]:
                obs[ind_of_all_joint_pos] += noise[:len(ind_of_all_joint_pos)] * joint_pos_noise_scale

            # Add noise to joint velocities
            if self.rand_conf["add_joint_vel_noise"]:
                obs[ind_of_all_joint_vel] += noise[
                    len(ind_of_all_joint_pos):len(ind_of_all_joint_pos) + len(ind_of_all_joint_vel)
                ] * joint_vel_noise_scale

            # Add noise to gravity vector
            if self.rand_conf["add_gravity_noise"]:
                obs[ind_of_gravity_vec] += noise[
                    len(ind_of_all_joint_pos) + len(ind_of_all_joint_vel):
                    len(ind_of_all_joint_pos) + len(ind_of_all_joint_vel) + len(ind_of_gravity_vec)
                ] * gravity_noise_scale

            # Add noise to linear velocities
            if self.rand_conf["add_free_joint_lin_vel_noise"]:
                obs[ind_of_lin_vel] += noise[
                    len(ind_of_all_joint_pos) + len(ind_of_all_joint_vel) + len(ind_of_gravity_vec):
                    len(ind_of_all_joint_pos) + len(ind_of_all_joint_vel) + len(ind_of_gravity_vec) + len(ind_of_lin_vel)
                ] * lin_vel_noise_scale

            # Add noise to angular velocities
            if self.rand_conf["add_free_joint_ang_vel_noise"]:
                obs[ind_of_ang_vel] += noise[
                    len(ind_of_all_joint_pos) + len(ind_of_all_joint_vel) + len(ind_of_gravity_vec) + len(ind_of_lin_vel):
                ] * ang_vel_noise_scale

        return obs, carry

    def update_action(self,
                      env: Any,
                      action: Union[np.ndarray, jnp.ndarray],
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      carry: Any,
                      backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Update the action with randomization effects.

        Args:
            env (Any): The environment instance.
            action (Union[np.ndarray, jnp.ndarray]): The action to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The updated action and carry.
        """
        assert_backend_is_supported(backend)
        return action, carry
    
    ########################Geoms related randomization##########################################

    def _sample_geom_friction(self, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the geometry friction parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized geometry friction parameters and carry.
        """
        assert_backend_is_supported(backend)

        fric_tan_min, fric_tan_max = self.rand_conf["geom_friction_tangential_range"]
        fric_tor_min, fric_tor_max = self.rand_conf["geom_friction_torsional_range"]
        fric_roll_min, fric_roll_max = self.rand_conf["geom_friction_rolling_range"]

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(len(model.geom_friction),))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=(len(model.geom_friction),))

        sampled_friction_tangential = (
            fric_tan_min + (fric_tan_max - fric_tan_min) * interpolation
            if self.rand_conf["randomize_geom_friction_tangential"]
            else model.geom_friction[:, 0]
        )
        sampled_friction_torsional = (
            fric_tor_min + (fric_tor_max - fric_tor_min) * interpolation
            if self.rand_conf["randomize_geom_friction_torsional"]
            else model.geom_friction[:, 1]
        )
        sampled_friction_rolling = (
            fric_roll_min + (fric_roll_max - fric_roll_min) * interpolation
            if self.rand_conf["randomize_geom_friction_rolling"]
            else model.geom_friction[:, 2]
        )
        geom_friction = backend.array([
            sampled_friction_tangential,
            sampled_friction_torsional,
            sampled_friction_rolling,
        ]).T

        return geom_friction, carry

    def _sample_geom_damping_and_stiffness(self, model: Union[MjModel, Model],
                                           carry: Any,
                                           backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the geometry damping and stiffness parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray], Any]: The randomized geometry damping
            and stiffness parameters and carry.
        """
        assert_backend_is_supported(backend)

        damping_min, damping_max = self.rand_conf["geom_damping_range"]
        n_geoms = model.ngeom
        stiffness_min, stiffness_max = self.rand_conf["geom_stiffness_range"]

        if backend == jnp:
            key = carry.key
            key, _k_damp, _k_stiff = jax.random.split(key, 3)
            interpolation_damping = jax.random.uniform(_k_damp, shape=(len(n_geoms),))
            interpolation_stiff = jax.random.uniform(_k_stiff, shape=(len(n_geoms),))
            carry = carry.replace(key=key)
        else:
            interpolation_damping = np.random.uniform(size=(len(n_geoms),))
            interpolation_stiff = np.random.uniform(size=(len(n_geoms),))

        sampled_damping = (
            damping_min + (damping_max - damping_min) * interpolation_damping
            if self.rand_conf["randomize_geom_damping"]
            else model.geom_solref[:, 1]
        )
        sampled_stiffness = (
            stiffness_min + (stiffness_max - stiffness_min) * interpolation_stiff
            if self.rand_conf["randomize_geom_stiffness"]
            else model.geom_solref[:, 0]
        )

        return sampled_damping, sampled_stiffness, carry
    
    def _sample_joint_friction_loss(self, model: Union[MjModel, Model],
                                           carry: Any,
                                           backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the joint friction and stiffness parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray], Any]: The randomized geometry damping
            and stiffness parameters and carry.
        """
        assert_backend_is_supported(backend)

        friction_min, friction_max = self.rand_conf["joint_friction_loss_range"]
        n_dofs = model.nv - 6 #exclude freejoint 6 degrees of freedom

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(n_dofs,))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=(n_dofs,))

        sampled_friction_loss = (
            friction_min + (friction_max - friction_min) * interpolation
            if self.rand_conf["randomize_joint_friction_loss"]
            else model.dof_frictionloss[6:]
        )

        return sampled_friction_loss, carry
    
    def _sample_joint_damping(self, model: Union[MjModel, Model],
                                           carry: Any,
                                           backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the joint friction and stiffness parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray], Any]: The randomized geometry damping
            and stiffness parameters and carry.
        """
        assert_backend_is_supported(backend)

        damping_min, damping_max = self.rand_conf["joint_damping_range"]
        n_dofs = model.nv - 6 #exclude freejoint 6 degrees of freedom

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(n_dofs,))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=(n_dofs,))

        sampled_damping = (
            damping_min + (damping_max - damping_min) * interpolation
            if self.rand_conf["randomize_joint_damping"]
            else model.dof_damping[6:]
        )

        return sampled_damping, carry
    
    def _sample_joint_armature(self, model: Union[MjModel, Model],
                                           carry: Any,
                                           backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the joint friction and stiffness parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray], Any]: The randomized geometry damping
            and stiffness parameters and carry.
        """
        assert_backend_is_supported(backend)

        armature_min, armature_max = self.rand_conf["joint_armature_range"]
        n_dofs = model.nv - 6 #exclude freejoint 6 degrees of freedom

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(n_dofs,))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=(n_dofs,))

        sampled_damping = (
            armature_min + (armature_max - armature_min) * interpolation
            if self.rand_conf["randomize_joint_armature"]
            else model.dof_armature[6:]
        )

        return sampled_damping, carry

    def _sample_com_displacement(self, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the geometry friction parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized geometry friction parameters and carry.
        """
        assert_backend_is_supported(backend)

        displ_min, displ_max = self.rand_conf["com_displacement_range"]

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(3,))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=3)

        sampled_com_displacement = (
            displ_min + (displ_max - displ_min) * interpolation
            if self.rand_conf["randomize_com_displacement"]
            else backend.array([0.0, 0.0, 0.0])
        )

        return sampled_com_displacement, carry
    
    def _sample_link_mass_multipliers(self, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the geometry friction parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized geometry friction parameters and carry.
        """
        assert_backend_is_supported(backend)

        multiplier_dict = self.rand_conf["link_mass_multiplier_range"]

        mult_base_min, mult_base_max = multiplier_dict["root_body"]

        mult_other_min, mult_other_max = multiplier_dict["other_bodies"]

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(model.nbody-1,)) #exclude worldbody 
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=1)

        sampled_base_mass_multiplier = (
            mult_base_min + (mult_base_max - mult_base_min) * interpolation[0]
            if self.rand_conf["randomize_link_mass"]
            else backend.array([1.0])
        )

        sampled_base_mass_multiplier = jnp.expand_dims(sampled_base_mass_multiplier, axis=0)

        sampled_other_bodies_mass_multipliers = (
            mult_other_min + (mult_other_max - mult_other_min) * interpolation[1:]
            if self.rand_conf["randomize_link_mass"]
            else backend.array([1.0]*(model.nbody-2))
        )

        mass_multipliers = backend.concatenate([
            sampled_base_mass_multiplier,
            sampled_other_bodies_mass_multipliers,
        ])

        return mass_multipliers, carry
    
    def _sample_p_gains_noise(self, env, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the p_gains_noise.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The p_gains_noise and carry.
        """
        assert_backend_is_supported(backend)

        init_p_gain = env._control_func._init_p_gain

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(len(init_p_gain),), minval=-1.0, maxval=1.0)
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.normal(size=len(init_p_gain))

        p_noise_scale = self.rand_conf["p_gains_noise_scale"]

        p_noise = interpolation * (p_noise_scale * init_p_gain)

        p_noise = (
            interpolation * (p_noise_scale * init_p_gain) 
            if self.rand_conf["add_p_gains_noise"]
            else backend.array([0.0]*len(init_p_gain))
            )

        return p_noise, carry
    
    def _sample_d_gains_noise(self, env, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples d_gains_noise.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The d_noise and carry.
        """
        assert_backend_is_supported(backend)

        init_d_gain = env._control_func._init_d_gain

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(len(init_d_gain),), minval=-1.0, maxval=1.0)
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=len(init_d_gain), low=-1.0, high=1.0)

        d_noise_scale = self.rand_conf["d_gains_noise_scale"]

        d_noise = (
            interpolation * (d_noise_scale * init_d_gain) 
            if self.rand_conf["add_d_gains_noise"]
            else backend.array([0.0]*len(init_d_gain))
            )

        return d_noise, carry