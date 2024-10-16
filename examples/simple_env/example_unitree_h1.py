import numpy as np
import jax
from copy import deepcopy
from loco_mujoco import LocoEnv
from loco_mujoco.core.utils.math import calc_rel_positions, calc_rel_quaternions, calc_rel_body_velocities, calc_site_velocities


# create the environment and task
env = LocoEnv.make("UnitreeH1.walk", disable_arms=False,
                   reward_type="mimic", goal_params=dict(visualize_goal=False))

# get the dataset for the chosen environment and task
expert_data = env.create_dataset()

action_dim = env.info.action_space.shape[0]

key = jax.random.key(0)
key, _rng = jax.random.split(key)

env.reset(_rng)

#env.render()
absorbing = False
i = 0

while True:
    if i == 1000 or absorbing:
        key, _rng = jax.random.split(key)
        env.reset(_rng)
        i = 0
    action = np.random.randn(action_dim)*0.0
    nstate, reward, absorbing, done, info = env.step(action)



    # import mujoco
    # local_frame = False
    # data, model = env._data, env._model
    # site_ids = np.arange(model.nsite)
    # parent_body_id = model.site_bodyid[site_ids]
    # root_body_id = model.body_rootid[parent_body_id]
    # site_xvel = calc_site_velocities(site_ids, data, parent_body_id, root_body_id, np, local_frame)
    #
    # for i in range(site_xvel.shape[0]):
    #     site_xvel_mj = np.empty(6)
    #     mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, i, site_xvel_mj, local_frame)
    #     site_xvel_jax = site_xvel[i]
    #     site_xvel_single = np.squeeze(calc_site_velocities(i, data, parent_body_id[i], root_body_id[i], np, local_frame))
    #     print("Difference mj vs jax: ", np.abs(site_xvel_jax - site_xvel_mj))
    #     print("Difference mj vs single: ", np.abs(site_xvel_single - site_xvel_mj))
    #
    # print("done")

    # import mujoco
    # data = env._data
    # model = env._model
    # # local_frame = True
    # # site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hip_mimic")
    # # body_id = model.site_bodyid[site_id]
    # # #xmat = data.xmat[body_id]
    # # #ipos = model.body_ipos[body_id]
    # # pelvis_id = model.body_rootid[body_id]
    # # body_pos_1 = data.xpos[pelvis_id] + data.xmat[pelvis_id].reshape(3, 3) @ model.body_ipos[pelvis_id]
    # # body_pos_2 = data.subtree_com[model.body_rootid[body_id]]
    # # print("Difference in position: ", np.abs(body_pos_1 - body_pos_2))
    # # print("Subtree position: ", body_pos_2)
    # # print("Pelvis position: ", data.xpos[pelvis_id])
    # # print("Pelvis I position: ", model.body_ipos[pelvis_id])
    # # print("rootid ID: ", model.body_rootid[body_id])
    # # site_cvel = calc_site_velocities(data.site_xpos[site_id],
    # #                                  data.site_xmat[site_id],
    # #                                  data.xpos[body_id] + data.xmat[body_id].reshape(3,3) @ model.body_ipos[body_id],
    # #                                  data.cvel[body_id], np, flg_local=local_frame)
    # #
    #
    #
    # pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    # pelvis_root_id = model.body_rootid[pelvis_id]
    # pelvis_cvel = data.cvel[pelvis_id]
    # pelvis_vel_local = np.empty(6)
    # pelvis_vel_global = np.empty(6)
    # mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis_id, pelvis_vel_local, True)
    # mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis_id, pelvis_vel_global, False)
    #
    #
    # site_id1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hip_mimic")
    # body_id1 = model.site_bodyid[site_id1]
    # site_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "upper_body_mimic")
    # body_id2 = model.site_bodyid[site_id2]
    # body_cvel1 = data.cvel[body_id1]
    # body_cvel2 = data.cvel[body_id2]
    # site_cvel_mj1 = np.empty(6)
    # local_frame = True
    # mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id1, site_cvel_mj1, local_frame)
    # site_cvel_mj2 = np.empty(6)
    # mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id2, site_cvel_mj2, local_frame)
    #
    # cpos_1 = data.subtree_com[body_id1]
    # cpos_2 = data.xipos[body_id1]
    #
    # print("cvel1: ", body_cvel1, "  cvel2: ", body_cvel2)
    # print("site_cvel1: ", site_cvel_mj1, "  site_cvel2: ", site_cvel_mj2)




    # print("Difference: ", np.abs(site_cvel - site_cvel_mj))
    # print("V: ", site_cvel)

    env.render()
    i += 1
