import numpy as np


class OptimizerAbstract:
    def __init__(self, *args, **kwargs):
        pass

    def pre_update_params(self):
        pass

    def update_params(self, layer):
        pass

    def post_update_params(self):
        pass


class OptimizerAdam(OptimizerAbstract):
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (
                np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (
                np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class OptimizerSGD(OptimizerAbstract):
    def __init__(self, learning_rate=0.001, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * \
                             layer.dweights
            bias_updates = -self.current_learning_rate * \
                           layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


class OptimizerAdagrad(OptimizerAbstract):
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
        layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class OptimizerRMSprop(OptimizerAbstract):
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class OptimizerCGF(OptimizerAbstract):
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, max_update=10):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.max_update = max_update

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        def calc_beta(dweights, dweights_prev, eps):
            return dweights.T.dot(dweights) / (dweights_prev.T.dot(dweights_prev) + eps)

        if not hasattr(layer, 'prev_dweights'):
            layer.prev_dweights = np.zeros_like(layer.dweights)
            layer.prev_dbiases = np.zeros_like(layer.dbiases)
            layer.p_weights = -layer.dweights
            layer.p_biases = -layer.dbiases

        # beta_weights = np.array([calc_beta(dweight, dweight_prev, self.epsilon) for (dweight, dweight_prev) in
        #                          zip(layer.dweights, layer.grad_prev_weights)])
        # beta_biases = np.array([calc_beta(dweight, dweight_prev, self.epsilon) for (dweight, dweight_prev) in
        #                         zip(layer.dbiases, layer.grad_prev_biases)])

        weight_update = self.current_learning_rate * layer.p_weights
        biases_update = self.current_learning_rate * layer.p_biases

        beta_weights = calc_beta(layer.dweights, layer.prev_dweights, self.epsilon)
        beta_biases = calc_beta(layer.dbiases, layer.prev_dbiases, self.epsilon)

        layer.p_weights = -layer.dweights + beta_weights * layer.p_weights
        layer.p_biases = -layer.dbiases + beta_biases * layer.p_biases

        layer.prev_dweights = layer.dweights
        layer.prev_dbiases = layer.dbiases

        layer.weights += weight_update  # np.clip(weight_update, -self.max_update, self.max_update)
        layer.biases += biases_update  # np.clip(biases_update, -self.max_update, self.max_update)
        # if not hasattr(layer, 'weights_p'):
        #     layer.weight_p = -layer.dweights
        #     layer.bias_p = -layer.dbiases
        #     layer.old_dweights = np.zeros_like(layer.dweights) + 1e-8
        #     layer.old_dbiases = np.zeros_like(layer.dbiases) + 1e-8
        #
        # updated_weights = layer.weights + self.current_learning_rate * layer.weight_p
        # updated_biases = layer.biases + self.current_learning_rate * layer.bias_p
        #
        # beta_weights = layer.dweights.T.dot(layer.dweights) / layer.old_dweights.T.dot(layer.old_dweights)
        # beta_biases = layer.dbiases.T.dot(layer.dbiases) / layer.old_dbiases.T.dot(layer.old_dbiases)
        #
        # layer.weight_p = -layer.dweights + beta_weights * layer.weight_p
        # layer.bias_p = -layer.dbiases + beta_biases * layer.bias_p
        #
        # layer.old_dbiases = layer.dbiases
        # layer.old_dweights = layer.dweights
        #
        # layer.weights = updated_weights
        # layer.biases = updated_biases

    def post_update_params(self):
        self.iterations += 1
