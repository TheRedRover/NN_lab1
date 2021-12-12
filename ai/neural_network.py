from typing import List

from config import LAYER_TYPE_COMPUTATIONAL
from ai.layers.layer_abstract import LayerAbstract
from ai.losses import LossAbstract


class NeuralNetwork:
    def __init__(self, layers: List[LayerAbstract] = [], loss_function: LossAbstract = None,
                 optimizer=None):

        self.layers: List[LayerAbstract] = layers
        self.loss_function: LossAbstract = loss_function
        self.optimizer = optimizer

        self.is_trained = False
        self.losses = {}

    def predict(self, X):
        return self.forward(X)

    def forward(self, X):
        x = X
        for layer in self.layers:
            layer.forward(x)
            x = layer.output
        return x

    def backward(self, predictions, y):
        self.loss_function.backward(predictions, y)
        dinputs = self.loss_function.dinputs

        for layer in self.layers[::-1]:
            layer.backward(dinputs)
            dinputs = layer.dinputs

    def append_layer(self, layer):
        self.layers.append(layer)

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function
