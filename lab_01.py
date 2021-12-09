import time

import matplotlib.pyplot as plt

from ai.layers.layer_activations import ActivationLinear, ActivationReLU, ActivationLeakyReLU
from ai.layers.layer_dense import LayerDense
from ai.losses import LossMeanSquaredError
from ai.neural_network import NeuralNetwork
from ai.optimizers import OptimizerCGF, OptimizerSGD
from utils.lab_01_get_data import create_dataset_lab_01

X, y = create_dataset_lab_01(samples=1000)

loss_function = LossMeanSquaredError()

# Раскоментировать нужную

# optimizer = OptimizerSGD(learning_rate=0.1, decay=1e-5, momentum=0.2)
# optimizer = OptimizerRMSprop(learning_rate=0.01, decay=1e-4)
# optimizer = OptimizerAdagrad(learning_rate=0.005, decay=1e-4)
# optimizer = OptimizerAdam(learning_rate=0.005, decay=1e-5)
optimizer = OptimizerCGF(learning_rate=0.001, decay=1e-5)

neural_network = NeuralNetwork()

neural_network.append_layer(LayerDense(1, 64, weight_regularizer_l2=0.1, bias_regularizer_l2=0.1))
neural_network.append_layer(ActivationReLU())
neural_network.append_layer(LayerDense(64, 64, weight_regularizer_l2=0.1, bias_regularizer_l2=0.1))
neural_network.append_layer(ActivationReLU())
neural_network.append_layer(LayerDense(64, 64, weight_regularizer_l2=0.1, bias_regularizer_l2=0.1))
neural_network.append_layer(ActivationReLU())
neural_network.append_layer(LayerDense(64, 64, weight_regularizer_l2=0.1, bias_regularizer_l2=0.1))
neural_network.append_layer(ActivationReLU())
neural_network.append_layer(LayerDense(64, 1, weight_regularizer_l2=0.1, bias_regularizer_l2=0.1))
neural_network.append_layer(ActivationLinear())

neural_network.set_loss_function(loss_function)
neural_network.set_optimizer(optimizer)

start_time = time.time()
neural_network.fit(X, y, epochs=20000)
print("--- %s seconds ---" % (time.time() - start_time))
losses = neural_network.losses

plt.plot(losses.keys(), losses.values())
plt.show()

X_test, y_test = create_dataset_lab_01()
predicted = neural_network.predict(X_test)

plt.plot(X_test, y_test)
plt.plot(X_test, predicted)
plt.show()
