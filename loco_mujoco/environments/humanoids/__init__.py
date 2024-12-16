from .atlas import Atlas
from .atlas_mjx import MjxAtlas
from .talos import Talos
from .talos_mjx import MjxTalos
from .unitreeH1 import UnitreeH1
from .unitreeH1_mjx import MjxUnitreeH1
from .unitreeG1 import UnitreeG1
from .unitreeG1_mjx import MjxUnitreeG1
from .skeletons import (SkeletonTorque, MjxSkeletonTorque, HumanoidTorque, SkeletonMuscle, MjxSkeletonMuscle,
                        HumanoidMuscle, HumanoidTorque4Ages, HumanoidMuscle4Ages)


# register environments in mushroom
Atlas.register()
MjxAtlas.register()
Talos.register()
MjxTalos.register()
UnitreeH1.register()
MjxUnitreeH1.register()
UnitreeG1.register()
MjxUnitreeG1.register()
SkeletonTorque.register()
MjxSkeletonTorque.register()
SkeletonMuscle.register()
MjxSkeletonMuscle.register()

# compatability with old names
HumanoidTorque.register()
HumanoidMuscle.register()
HumanoidTorque4Ages.register()
HumanoidMuscle4Ages.register()


from gymnasium import register

# register gymnasium wrapper environment
register("LocoMujoco",
         entry_point="loco_mujoco.core.wrappers.gymnasium:GymnasiumWrapper"
         )
