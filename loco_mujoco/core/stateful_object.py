from flax import struct


@struct.dataclass
class EmptyState:
    pass


class StatefulObject:

    def reset_state(self, env, model, data, carry, backend):
        return data, carry

    def init_state(self, env, key, model, data, backend):
        return EmptyState()
