from .base_robot_quadruped import BaseRobotQuadruped
from .unitreeA1 import UnitreeA1
from .unitreeA1_mjx import MjxUnitreeA1

# register environment
UnitreeA1.register()
MjxUnitreeA1.register()
