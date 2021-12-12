import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from ai.layers.layer_activations import ActivationLinear, ActivationReLU
from ai.layers.layer_dense import LayerDense
from ai.losses import LossMeanSquaredError
from ai.neural_network import NeuralNetwork
from ai.optimizers import OptimizerAbstract, OptimizerBFGS, OptimizerSGD, OptimizerGDM, OptimizerCGF
from utils.lab_01_get_data import create_dataset_lab_01

np.random.seed(42)


def test_optimizer(optimizer: Callable[[NeuralNetwork], OptimizerAbstract]):
    neural_network = NeuralNetwork()

    neural_network.append_layer(LayerDense(1, 32, weight_regularizer_l2=0.001))
    neural_network.append_layer(ActivationReLU())
    neural_network.append_layer(LayerDense(32, 32, weight_regularizer_l2=0.001))
    neural_network.append_layer(ActivationReLU())
    # neural_network.append_layer(LayerDense(16, 16, weight_regularizer_l2=0.001))
    # neural_network.append_layer(ActivationReLU())
    # neural_network.append_layer(LayerDense(16, 16, weight_regularizer_l2=0.001))
    # neural_network.append_layer(ActivationReLU())
    neural_network.append_layer(LayerDense(32, 1, weight_regularizer_l2=0.001))
    neural_network.append_layer(ActivationLinear())

    loss_function = LossMeanSquaredError()
    neural_network.set_loss_function(loss_function)

    X, y = create_dataset_lab_01(samples=1000)
    start_time = time.time()
    opter = optimizer(neural_network)
    opter.fit(X, y, epochs=20000)
    print("--- %s seconds ---" % (time.time() - start_time))
    losses = opter.losses

    plt.plot(losses.keys(), losses.values())
    plt.show()

    X_test, y_test = create_dataset_lab_01()
    predicted = neural_network.predict(X_test)

    plt.plot(X_test, y_test, label='test')
    plt.plot(X_test, predicted, label='predict')
    plt.legend()
    plt.show()


opts = []
# opts.append(lambda nn: OptimizerSGD(nn, learning_rate=0.01, decay=1e-5, momentum=0.2))
# opts.append(lambda nn: OptimizerRMSprop(nn, learning_rate=0.01, decay=1e-4))
# opts.append(lambda nn: OptimizerAdagrad(nn, learning_rate=0.005, decay=1e-4))
# opts.append(lambda nn: OptimizerAdam(nn, learning_rate=0.005, decay=1e-5))
# opts.append(lambda nn: OptimizerGDM(nn, learning_rate=.1, decay=1e-5, momentum=.9))
# opts.append(lambda nn: OptimizerCGF(nn, learning_rate=0.1, decay=0.0005))
opts.append(lambda nn: OptimizerBFGS(nn, learning_rate=1, decay=1e-4))
for opt in opts:
    test_optimizer(opt)
