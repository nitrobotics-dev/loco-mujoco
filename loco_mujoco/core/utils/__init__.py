from .env import MDPInfo, Box
from .viewer import MujocoViewer
from .video_recorder import VideoRecorder
from .observations import ObservationType, ObservationIndexContainer, ObservationContainer, Observation
from .goals import *
from .terminal_state_handler import (TerminalStateHandler, HeightBasedTerminalStateHandler,
                                     RootPoseTrajTerminalStateHandler)
from .mujoco import *

# register all goals
NoGoal.register()
GoalTrajArrow.register()
GoalTrajMimic.register()
