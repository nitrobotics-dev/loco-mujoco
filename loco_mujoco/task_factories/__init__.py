from .base import TaskFactory
from .rl_factory import RLFactory
from .default_imitation_factory import ImitationFactory
from .dataset_confs import DefaultDatasetConf, AMASSDatasetConf, LAFAN1DatasetConf, CustomTrajectoryConf


# register factories
RLFactory.register()
ImitationFactory.register()


