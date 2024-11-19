import mujoco


def mj_jntname2qposid(j_name, model):
    """
    Get qpos index of a joint in mujoco data structure.

    Args:
        j_name (str): joint name.
        model (mjModel): mujoco model.

    Returns:
        list of qpos index in MjData corresponding to that joint.
    """
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
    if j_id == -1:
        raise ValueError(f"Joint name {j_name} not found in model.")

    return mj_jntid2qposid(j_id, model)


def mj_jntname2qvelid(j_name, model):
    """
    Get qvel index of a joint in mujoco data structure.

    Args:
        j_name (str): joint name.
        model (mjModel): mujoco model.

    Returns:
        list of qvel index in MjData corresponding to that joint.

    """
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
    if j_id == -1:
        raise ValueError(f"Joint name {j_name} not found in model.")

    return mj_jntid2qvelid(j_id, model)


def mj_jntid2qposid(j_id, model):
    """
    Get qpos index of a joint in mujoco data structure.

    Args:
        j_id (int): joint id.
        model (mjModel): mujoco model.

    Returns:
        list of qpos index in MjData corresponding to that joint.
    """
    start_qpos_id = model.jnt_qposadr[j_id]
    jnt_type = model.jnt_type[j_id]

    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        qpos_id = [i for i in range(start_qpos_id, start_qpos_id+7)]
    else:
        qpos_id = [start_qpos_id]

    return qpos_id


def mj_jntid2qvelid(j_id, model):
    """
    Get qvel index of a joint in mujoco data structure.

    Args:
        j_id (int): joint id.
        model (mjModel): mujoco model.

    Returns:
        list of qvel index in MjData corresponding to that joint.

    """
    start_qvel_id = model.jnt_dofadr[j_id]
    jnt_type = model.jnt_type[j_id]

    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        qvel_id = [i for i in range(start_qvel_id, start_qvel_id+6)]
    else:
        qvel_id = [start_qvel_id]

    return qvel_id