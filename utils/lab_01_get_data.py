import numpy as np


# Мой номер в журнале - 17
# Вариант: 2
# Сигнал: y = cos(5x) - sin(4x) + 0.5cos(87x)

def create_dataset_lab_01(samples=1000):
    # y = cos(5x) - sin(4x) + 0.5cos(87x)

    X = np.arange(samples).reshape(-1, 1) / samples

    y = np.cos(5 * X) - np.sin(4 * X) + 0.5 * np.cos(87 * X)
    y = y.reshape(-1, 1)

    return X, y
