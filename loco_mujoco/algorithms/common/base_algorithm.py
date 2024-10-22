

class BaseJaxRLAlgorithm:

    @staticmethod
    def get_train_function(env, config):
        raise NotImplementedError

    @classmethod
    def play_policy(cls, train_state, env, config, n_envs, n_steps=None, record=False, key=None):
        raise NotImplementedError

    @staticmethod
    def wrap_env(env, config):
        raise NotImplementedError

    @classmethod
    def linear_lr_schedule(cls, count, num_minibatches, update_epochs, lr, num_updates):
        frac = (
                1.0
                - (count // (num_minibatches * update_epochs))
                / num_updates
        )
        return lr * frac
