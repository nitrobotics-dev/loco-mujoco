from .reward import *
from loco_mujoco.core.utils.goals import *
from .trajectory import *
from .checks import *
from .video import video2gif
from .domain_randomization import *
from .decorators import *
from .running_stats import *
from .dataset import download_all_datasets, download_real_datasets, download_perfect_datasets
from .speed_test import mjx_speed_test

# register all rewards
NoReward.register()
PosReward.register()
TargetVelocityReward.register()
CustomReward.register()
VelocityVectorReward.register()
