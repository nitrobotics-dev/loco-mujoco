from .base import TaskFactory
from .rl_factory import RLFactory
from .default_imitation_factory import ImitationFactory
from .amass_imitation_factory import AMASSImitationFactory

# register all
RLFactory.register()
ImitationFactory.register()
AMASSImitationFactory.register()
