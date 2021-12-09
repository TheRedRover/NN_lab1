import numpy as np

from ai.layers.layer_abstract import LayerAbstract
from config import LAYER_TYPE_COMPUTATIONAL


class LayerDense(LayerAbstract):
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0,
                 weight_regularizer_l2=0,
                 bias_regularizer_l1=0,
                 bias_regularizer_l2=0):

        self.type = LAYER_TYPE_COMPUTATIONAL

        self._initialize_weights(n_inputs, n_neurons)
        self._initialize_biases(n_neurons)

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                             self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                            self.biases

        self.dweights = self.dweights.clip(-100, 100)
        self.dbiases = self.dbiases.clip(-100, 100)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    def _initialize_weights(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)  # glorot

    def _initialize_biases(self, n_neurons):
        self.biases = np.zeros((1, n_neurons))
