import numpy as np

from ai.utils.loss_functions import mse, mae


class LossAbstract:
    def regularization_loss(self, layer):

        regularization_loss = 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * \
                                   np.sum(np.abs(layer.weights))

        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * \
                                   np.sum(layer.weights * \
                                          layer.weights)

        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * \
                                   np.sum(np.abs(layer.biases))
        # L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * \
                                   np.sum(layer.biases * \
                                          layer.biases)
        return regularization_loss

    def calculate(self, y_pred, y_true):
        return np.mean(self.forward(y_pred, y_true))


class LossMeanSquaredError(LossAbstract):
    def forward(self, y_pred, y_true):
        return mse(y_pred, y_true)

    def backward(self, dvalues, y_true):
        self.dinputs = -2 * (y_true - dvalues) / len(dvalues[0])
        self.dinputs /= len(dvalues)


class LossMeanAbsoluteError(LossAbstract):
    def forward(self, y_pred, y_true):
        return mae(y_pred, y_true)

    def backward(self, dvalues, y_true):
        self.dinputs = -np.sign(y_true - dvalues) / len(dvalues[0])
        self.dinputs /= len(dvalues)