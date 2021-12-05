import numpy as np


def relu(x):
    return np.maximum(0, x)


def linear(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def binary_step(x):
    return np.heaviside(x, 1)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
