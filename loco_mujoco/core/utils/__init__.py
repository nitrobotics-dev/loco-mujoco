from .backend import *
from .env import MDPInfo, Box
from .viewer import MujocoViewer
from .video_recorder import VideoRecorder
from .reward import *
from .terminal_state_handler import (TerminalStateHandler, HeightBasedTerminalStateHandler,
                                     RootPoseTrajTerminalStateHandler)
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


# register all terminal state handlers
HeightBasedTerminalStateHandler.register()
RootPoseTrajTerminalStateHandler.register()

