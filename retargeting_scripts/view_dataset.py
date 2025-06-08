#!/usr/bin/env python3
import os
import time
import numpy as np
import mujoco
from mujoco.viewer import launch_passive
from loco_mujoco.environments.humanoids import BerkeleyHumanoidLite

# 1) Load the model & your dataset
xml_path = BerkeleyHumanoidLite.get_default_xml_file_path()
model    = mujoco.MjModel.from_xml_path(xml_path)
data     = mujoco.MjData(model)

# adjust path if your cache is elsewhere
cache_dir = os.path.expanduser(
    "~/.cache/loco-mujoco/datasets/BerkeleyHumanoidLite/Default/walk1"
)
qpos_all   = np.load(os.path.join(cache_dir, "qpos.npy"))
site_xpos  = np.load(os.path.join(cache_dir, "site_xpos.npy"))
n_frames   = qpos_all.shape[0]

# 2) (Optional) choose whether to draw site frames
#    mjFRAME_SITE will render little XYZ axes at each <site>.
draw_sites = True

with launch_passive(model, data) as viewer:
    if draw_sites:
        with viewer.lock():
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    for t in range(n_frames):
        # 3) Set joint positions & forward‐kinematics
        data.qpos[:] = qpos_all[t]
        mujoco.mj_forward(model, data)

        # 4) (Optional) print world-space site positions for frame t
        # Uncomment to debug numeric values
        # print(f"Frame {t:4d}:")
        # for i, sid in enumerate([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        #                           for name in BerkeleyHumanoidLite.get_site_names()]):
        #     pos = data.site_xpos[sid]
        #     print(f"  {i:02d} {pos}")

        # 5) Sync & render
        with viewer.lock():
            viewer.sync()

        # 6) delay to play at real time (or adjust sleep to slow down)
        time.sleep(model.opt.timestep)

        # 7) allow early exit if window closed
        if not viewer.is_running():
            break
