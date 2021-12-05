from config import LAYER_TYPE_ACTIVATION
from ai.utils.activation_functions import relu, linear, sigmoid
from ai.layers.layer_abstract import LayerAbstract


class ActivationAbstract(LayerAbstract):
    def __init__(self, *args, **kwargs):
        self.type = LAYER_TYPE_ACTIVATION

class ActivationReLU(ActivationAbstract):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = relu(inputs)

    def backward(self, d):
        self.dinputs = d.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class ActivationLinear(ActivationAbstract):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = linear(inputs)

    def backward(self, d):
        self.dinputs = d.copy()

class ActivationSigmoid(ActivationAbstract):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = sigmoid(inputs)

    def backward(self, d):
        self.dinputs = d * (1 - self.output) * self.output