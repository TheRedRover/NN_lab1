import numpy as np

from ai.neural_network import NeuralNetwork
from config import LAYER_TYPE_COMPUTATIONAL

class OptimizerAbstract:
    def __init__(self, network: NeuralNetwork, learning_rate: float, *args, **kwargs):
        self.network = network
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.is_trained = False
        self.losses = {}

    def pre_update_params(self):
        pass

    def update_params(self, layer):
        pass

    def post_update_params(self):
        pass

    def update_weights(self):
        self.pre_update_params()
        for layer in self.network.layers:
            if layer.type == LAYER_TYPE_COMPUTATIONAL:
                self.update_params(layer)
        self.post_update_params()

    def fit(self, x, y, epochs=10000, loss_function=None):
        self._validate_fit(loss_function)

        for epoch in range(epochs):
            predictions = self.network.forward(x)

            loss = self.network.loss_function.calculate(predictions, y)
            if (epoch % 100) == 0:
                print(f'epoch: {epoch}, ' +
                      f'loss: {loss:.3f} ' +
                      f'lr: {self.current_learning_rate}')

                self.losses[epoch] = loss

            self.network.backward(predictions, y)
            self.update_weights()

        return self

    def _validate_fit(self, loss_function=None):
        if loss_function is not None:
            self.network.set_loss_function(loss_function)

        if len(self.network.layers) == 0 or self.network.layers is None:
            raise ValueError("No layers were provided for the neural network.")

        if self.network.loss_function is None:
            raise ValueError("Loss function was not provided.")


class OptimizerAdam(OptimizerAbstract):
    def __init__(self, network, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        super(OptimizerAdam, self).__init__(network, learning_rate)
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
    def __init__(self, network, learning_rate=0.001, decay=0., momentum=0.):
        super(OptimizerSGD, self).__init__(network, learning_rate)
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


class OptimizerAdagrad(OptimizerAbstract):
    def __init__(self, network, learning_rate=0.001, decay=0., epsilon=1e-7):
        super(OptimizerAdagrad, self).__init__(network, learning_rate)
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

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class OptimizerRMSprop(OptimizerAbstract):
    def __init__(self, network, learning_rate=0.001, decay=0., epsilon=1e-7,
                 rho=0.9):
        super(OptimizerRMSprop, self).__init__(network, learning_rate)
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

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class OptimizerCGF(OptimizerAbstract):
    def __init__(self, network, learning_rate=0.001, decay=0., epsilon=1e-7, max_update=10):
        super(OptimizerCGF, self).__init__(network, learning_rate)
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.max_update = max_update

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        def calc_beta(dweights, dweights_prev):
            assert dweights.shape[0] == dweights.size
            assert dweights_prev.shape[0] == dweights_prev.size
            return dweights.dot(dweights) / (dweights_prev.dot(dweights_prev) + 1)

        if not hasattr(layer, 'prev_dweights'):
            layer.prev_dweights = layer.dweights.copy()
            layer.prev_dbiases = layer.dbiases.copy()
            layer.weight_p = np.zeros_like(-layer.dweights)
            layer.bias_p = np.zeros_like(-layer.dbiases)

        weight_update = self.current_learning_rate * layer.weight_p
        biases_update = self.current_learning_rate * layer.bias_p

        beta_weights = np.array([calc_beta(dweight, dweight_prev) for (dweight, dweight_prev) in
                                 zip(layer.dweights.T, layer.prev_dweights.T)])
        beta_biases = np.array([calc_beta(dweight, dweight_prev) for (dweight, dweight_prev) in
                                zip(layer.dbiases.T, layer.prev_dbiases.T)])

        layer.weight_p = -layer.dweights + beta_weights * layer.weight_p
        layer.bias_p = -layer.dbiases + beta_biases * layer.bias_p

        layer.prev_dweights = layer.dweights.copy()
        layer.prev_dbiases = layer.dbiases.copy()

        layer.weights += weight_update  # np.clip(weight_update, -self.max_update, self.max_update)
        layer.biases += biases_update  # np.clip(biases_update, -self.max_update, self.max_update)

    def post_update_params(self):
        self.iterations += 1


class OptimizerGDM(OptimizerAbstract):
    def __init__(self, network, learning_rate=0.001, decay=0., momentum=0.):
        super(OptimizerGDM, self).__init__(network, learning_rate)
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        weight_updates = \
            self.momentum * layer.weight_momentums + \
            (1.0 - self.momentum) * self.current_learning_rate * layer.dweights
        layer.weight_momentums = weight_updates

        bias_updates = \
            self.momentum * layer.bias_momentums + \
            (1.0 - self.momentum) * self.current_learning_rate * layer.dbiases
        layer.bias_momentums = bias_updates

        layer.weights -= weight_updates
        layer.biases -= bias_updates

    def post_update_params(self):
        self.iterations += 1


class OptimizerBFGS(OptimizerAbstract):
    def __init__(self, network, learning_rate=0.001, decay=0., epsilon=1e-7):
        super(OptimizerBFGS, self).__init__(network, learning_rate)
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        flat_weights = layer.weights.flatten()
        flat_dweights = layer.dweights.flatten()

        I = np.eye(flat_weights.shape[0])
        if not hasattr(layer, f'prev_weights'):
            layer.H = I
            layer.prev_weights = np.zeros_like(flat_weights)
            layer.prev_dweights = np.zeros_like(flat_dweights)

        sk = flat_weights - layer.prev_weights
        yk = flat_dweights - layer.prev_dweights

        rho_inv = yk @ sk.T
        rho = 1 / (rho_inv + 0.1)

        A1 = (I - rho * (yk @ sk.T))
        A2 = (I - rho * (sk @ yk.T))
        left = A1 @ (layer.H @ A2)
        layer.H = left + rho * (yk @ yk.T)

        weight_update = self.current_learning_rate * (-layer.H @ flat_dweights)

        layer.weights += weight_update.reshape((layer.weights.shape[0], layer.weights.shape[1]))

        layer.prev_weights = layer.weights.flatten().copy()
        layer.prev_dweights = layer.dweights.flatten().copy()

    def post_update_params(self):
        self.iterations += 1
