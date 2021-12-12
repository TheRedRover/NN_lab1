import numpy as np

from ai.layers.layer_abstract import LayerAbstract
from config import LAYER_TYPE_COMPUTATIONAL


class LayerDense(LayerAbstract):
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0,
                 weight_regularizer_l2=0):

        self.type = LAYER_TYPE_COMPUTATIONAL

        self._initialize_weights(n_inputs, n_neurons)

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2

    def forward(self, inputs):
        bias = np.ones((inputs.shape[0], 1))
        self.inputs = np.concatenate((inputs, bias), axis=1)
        self.output = np.dot(self.inputs, self.weights)

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        self.dweights = self.dweights.clip(-100, 100)

        # Gradient on values
        w_no_biases = self.weights[:-1]
        self.dinputs = np.dot(dvalues, w_no_biases.T)

    def _initialize_weights(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.weights = np.concatenate((self.weights, np.zeros((1, n_neurons))), axis=0)
