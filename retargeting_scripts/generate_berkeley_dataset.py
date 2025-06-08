#!/usr/bin/env python3
"""
retarget_h1walk_to_berkeley_manual_ik_full.py

Loads UnitreeH1’s walk.npz, scales its site trajectories,
identifies the intersection of mimic sites present on both H1 and Berkeley,
then uses a Jacobian-based IK loop to retarget each frame
onto BerkeleyHumanoidLite. Writes out the four .npy files under:
    ~/.cache/loco-mujoco/datasets/BerkeleyHumanoidLite/Default/walk1/
for use by experiment.py.
"""
import os
import sys
import time
import traceback

import numpy as np
import mujoco

# LocoMuJoCo imports
from loco_mujoco.environments.humanoids import UnitreeH1, BerkeleyHumanoidLite

CACHE_DIR = os.path.expanduser("~/.loco-mujoco-caches/DEFAULT/mocap/UnitreeH1")
OUT_DIR   = os.path.expanduser(
    "~/.cache/loco-mujoco/datasets/BerkeleyHumanoidLite/Default/walk1"
)

# 1) Load H1's walk.npz
npz_path = os.path.join(CACHE_DIR, "walk.npz")
if not os.path.isfile(npz_path):
    raise FileNotFoundError(f"Missing H1 walk.npz at {npz_path}")
print(f"→ Loading H1 walk.npz from {npz_path}")
data = np.load(npz_path)
print("   Keys in walk.npz:", data.files)

# Required arrays
qpos_h1      = data['qpos']        # (T, H1_nq)
qvel_h1      = data['qvel']        # (T, H1_nv)
site_xpos_h1 = data['site_xpos']   # (T, nsite_h1, 3)
# site velocities may be named differently
if 'site_xvel' in data.files:
    site_xvel_h1 = data['site_xvel']
elif 'site_vel' in data.files:
    site_xvel_h1 = data['site_vel']
else:
    site_xvel_h1 = np.zeros_like(site_xpos_h1)

T, h1_nsite, _ = site_xpos_h1.shape
print(f"   → Found {T} frames, {h1_nsite} H1 sites.")

# 2) Build MuJoCo models + data
model_h1 = mujoco.MjModel.from_xml_path(UnitreeH1.get_default_xml_file_path())
data_h1  = mujoco.MjData(model_h1)
model_b  = mujoco.MjModel.from_xml_path(BerkeleyHumanoidLite.get_default_xml_file_path())
data_b   = mujoco.MjData(model_b)

# 3) Compute uniform scale s via upper_body height ratio
mujoco.mj_forward(model_h1, data_h1)
h1_upper_body_sid = mujoco.mj_name2id(model_h1, mujoco.mjtObj.mjOBJ_SITE, 'upper_body_mimic')
h1_upper_body_z   = data_h1.site_xpos[h1_upper_body_sid][2]

mujoco.mj_forward(model_b, data_b)
berk_upper_body_sid = mujoco.mj_name2id(model_b, mujoco.mjtObj.mjOBJ_SITE, 'upper_body_mimic')
berk_upper_body_z   = data_b.site_xpos[berk_upper_body_sid][2]
s = berk_upper_body_z / h1_upper_body_z
print(f"→ Scaling factor s = {s:.4f} (Berk_z / H1_z)")

dt = model_b.opt.timestep

# 4) Identify common mimic sites
# SITE_NAMES you want to retarget (mimic_sites.xml)
SITE_NAMES = [
    "left_hip_mimic",
    "left_knee_mimic",
    "left_foot_mimic",
    "right_hip_mimic",
    "right_knee_mimic",
    "right_foot_mimic",
    "upper_body_mimic",
    "left_shoulder_mimic",
    "left_elbow_mimic",
    "left_hand_mimic",
    "right_shoulder_mimic",
    "right_elbow_mimic",
    "right_hand_mimic",
]
# H1 NPZ site names
h1_names = [s.decode() if isinstance(s, bytes) else s for s in data['site_names']]

h1_idxs  = []  # indices into NPZ arrays
berk_ids = []  # MuJoCo site IDs
common   = []  # site names
for name in SITE_NAMES:
    if name in h1_names:
        sid_b = mujoco.mj_name2id(model_b, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid_b != -1:
            h1_idxs.append(h1_names.index(name))
            berk_ids.append(sid_b)
            common.append(name)

n_common = len(common)
print(f"→ Retargeting {n_common} common sites: {common}")
if n_common == 0:
    raise RuntimeError("No common mimic sites found between H1 and Berkeley.")

# 5) Allocate arrays for Berkeley
B_nq = model_b.nq
B_nv = model_b.nv
qpos_b_all   = np.zeros((T, B_nq))
qvel_b_all   = np.zeros((T, B_nv))
site_xpos_b  = np.zeros((T, n_common, 3))
site_xvel_b  = np.zeros((T, n_common, 3))

# 6) Manual Jacobian-based IK per frame
print("→ Running manual IK retarget...")
t0 = time.time()
for t in range(T):
    # init pose
    data_b.qpos[:] = qpos_b_all[t-1] if t>0 else 0
    mujoco.mj_forward(model_b, data_b)
    # targets
    target = s * site_xpos_h1[t, h1_idxs, :]
    # iterative solve
    for _ in range(15):
        cur = np.vstack([data_b.site_xpos[sid] for sid in berk_ids])
        e = (target - cur).reshape(-1)
        if np.linalg.norm(e) < 1e-4:
            break
        # 2) build Jacobian using mj_jacSite(jacp, jacr)
        Jblocks = []
        for sid in berk_ids:
            Jp = np.zeros((3, model_b.nv), dtype=np.float64)
            Jr = np.zeros((3, model_b.nv), dtype=np.float64)
            mujoco.mj_jacSite(model_b, data_b, Jp, Jr, sid)
            Jblocks.append(Jp)
        J = np.vstack(Jblocks)  # (3*n_common, nv)
        lam = 1e-4
        dq = np.linalg.solve(J.T@J + lam*np.eye(model_b.nv), J.T@e)
                # integrate only actuated joints (skip the free-base dofs)
        # MuJoCo's freejoint uses first 7 qpos dims (3 pos + 4 quat) and first 6 qvel dims (3 lin + 3 ang)
        actuated_dq = dq[6:]
        # update only the actuated joint positions (qpos[7:] corresponds to actuated joints)
        data_b.qpos[7:] += actuated_dq * dt
        mujoco.mj_forward(model_b, data_b)
    # record
    qpos_b_all[t] = data_b.qpos[:]
    for i, sid in enumerate(berk_ids):
        site_xpos_b[t,i] = data_b.site_xpos[sid]
                # site velocities will be computed by finite difference after IK loop

print(f"✔ IK retarget done in {time.time()-t0:.1f}s")

# 7) Finite-difference for actuated joint velocities
# Free-base velocities (first 6 dims) are set to zero
for t in range(T-1):
    # qvel has length model_b.nv = 6 (base) + actuated_n
    qvel_b_all[t][:6] = 0.0
    # actuated joint velocities: use difference of qpos (indices 7: for pos, 6: for vel)
    qvel_b_all[t][6:] = (qpos_b_all[t+1][7:] - qpos_b_all[t][7:]) / dt
# For the last frame, repeat the previous velocity
qvel_b_all[-1] = qvel_b_all[-2]

# 8) Compute site velocities by finite difference
for t in range(T-1):
    site_xvel_b[t] = (site_xpos_b[t+1] - site_xpos_b[t]) / dt
site_xvel_b[-1] = site_xvel_b[-2]

# 9) Save to Berkeley cache Compute site velocities by finite difference
for t in range(T-1):
    site_xvel_b[t] = (site_xpos_b[t+1] - site_xpos_b[t]) / dt
site_xvel_b[-1] = site_xvel_b[-2]

# 9) Save to Berkeley cache
os.makedirs(OUT_DIR, exist_ok=True)
print(f"→ Saving retargeted data to {OUT_DIR}")
np.save(os.path.join(OUT_DIR, "qpos.npy"),         qpos_b_all)
np.save(os.path.join(OUT_DIR, "qvel.npy"),         qvel_b_all)
np.save(os.path.join(OUT_DIR, "site_xpos.npy"),    site_xpos_b)
np.save(os.path.join(OUT_DIR, "site_xvel.npy"),    site_xvel_b)
print("✔ Done. Run: python experiment.py")



