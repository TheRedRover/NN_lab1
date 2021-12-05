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

    def fit(self, X, y, epochs=10000, loss_function=None, optimizer=None):
        self._validate_fit(loss_function, optimizer)

        self.losses = {}
        for epoch in range(epochs):
            predictions = self.forward(X)

            loss = self.loss_function.calculate(predictions, y)
            if (epoch % 100) == 0:
                print(f'epoch: {epoch}, ' +
                      f'loss: {loss:.3f} ' +
                      f'lr: {self.optimizer.current_learning_rate}')

                self.losses[epoch] = loss

            self.backward(predictions, y)
            self.update_weights()

        self.is_trained = True
        return self

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("The model is not trained")
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

    def update_weights(self):
        self.optimizer.pre_update_params()
        for layer in self.layers:
            if layer.type == LAYER_TYPE_COMPUTATIONAL:
                self.optimizer.update_params(layer)
        self.optimizer.post_update_params()

    def append_layer(self, layer):
        self.layers.append(layer)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def _validate_fit(self, loss_function=None, optimizer=None):
        if loss_function is not None:
            self.set_loss_function(loss_function)
        if optimizer is not None:
            self.set_optimizer(optimizer)

        if len(self.layers) == 0 or self.layers is None:
            raise ValueError("No layers were provided for the neural network.")

        if self.loss_function is None:
            raise ValueError("Loss function was not provided.")

        if self.optimizer is None:
            raise ValueError("Optimizer was not provided.")
