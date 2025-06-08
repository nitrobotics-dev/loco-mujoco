# from loco_mujoco.environments.humanoids import UnitreeH1, BerkeleyHumanoidLite

# import time

# import mujoco
# import mujoco.viewer
# xml_path = BerkeleyHumanoidLite.get_default_xml_file_path()

# m = mujoco.MjModel.from_xml_path(xml_path)
# d = mujoco.MjData(m)

# with mujoco.viewer.launch_passive(m, d) as viewer:
#   # Close the viewer automatically after 30 wall-seconds.
#   start = time.time()
#   while viewer.is_running():
#     step_start = time.time()

#     # mj_step can be replaced with code that also evaluates
#     # a policy and applies a control signal before stepping the physics.
#     mujoco.mj_step(m, d)

#     # Example modification of a viewer option: toggle contact points every two seconds.
#     with viewer.lock():
#       viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

#     # Pick up changes to the physics state, apply perturbations, update options from GUI.
#     viewer.sync()

#     # Rudimentary time keeping, will drift relative to wall clock.
#     time_until_next_step = m.opt.timestep - (time.time() - step_start)
#     if time_until_next_step > 0:
#       time.sleep(time_until_next_step)

import time
import numpy as np

import mujoco
from mujoco.viewer import launch_passive

from loco_mujoco.environments.humanoids import BerkeleyHumanoidLite, UnitreeH1

# 1) Load the Berkeley Humanoid Lite model & data
xml_path = UnitreeH1.get_default_xml_file_path()
model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

# 2) Pick the list of mimic‐site names you want to inspect
#    (must exactly match <site name="XXX_mimic"/> in your XML)
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

# 3) Convert each name → site ID using mj_name2id
site_ids = {}
for name in SITE_NAMES:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid == -1:
        raise RuntimeError(f"Site '{name}' not found in the model. Check spelling.")
    site_ids[name] = sid

# 4) Launch the passive viewer and enable site-frame rendering
with launch_passive(model, data) as viewer:
    # Tell the viewer to draw a little XYZ axis at every <site>
    with viewer.lock():
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    last_print = -1.0
    while viewer.is_running():
        frame_start = time.time()

        # Step the physics forward by one timestep
        # mujoco.mj_step(model, data)

        # Every 0.1 seconds, print out the world-space (x,y,z) of each site
        if data.time - last_print > 0.1:
            last_print = data.time
            xpos = { name: data.site_xpos[sid].copy() for name, sid in site_ids.items() }
            print(f"t = {data.time:6.3f} s  |  Site world-positions:")
            for name in SITE_NAMES:
                x, y, z = xpos[name]
                print(f"  {name:20s}: [ {x: .3f}, {y: .3f}, {z: .3f} ]")
            print("-" * 60)

        # Sync/render under the viewer lock
        with viewer.lock():
            viewer.sync()

        # Sleep so we run in real time
        wait = model.opt.timestep - (time.time() - frame_start)
        if wait > 0:
            time.sleep(wait)
