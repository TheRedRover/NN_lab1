import numpy as np

def mse(y_pred, y_true):
    result = np.mean((y_true - y_pred)**2, axis=-1)
    return result

def mae(y_pred, y_true):
    return np.mean(np.abs(y_true - y_pred), axis=-1)
