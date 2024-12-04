from .env import MDPInfo, Box
from .viewer import MujocoViewer
from .video_recorder import VideoRecorder
from .terminal_state_handler import (TerminalStateHandler, HeightBasedTerminalStateHandler,
                                     RootPoseTrajTerminalStateHandler)
from .mujoco import *

# register all terminal state handlers
HeightBasedTerminalStateHandler.register()
RootPoseTrajTerminalStateHandler.register()

