from .backend import *
from .env import MDPInfo, Box
from .viewer import MujocoViewer
from .video_recorder import VideoRecorder
from .reward import *
from .mujoco import *
from .decorators import info_property

# register all rewards
NoReward.register()
PosReward.register()
TargetVelocityGoalReward.register()
TargetXVelocityReward.register()
TargetVelocityTrajReward.register()
CustomReward.register()
MimicReward.register()
LocomotionReward.register()

