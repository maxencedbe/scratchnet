import numpy as np

class CrossEntropy:
    def forward(self, pred, y_true):
        n = y_true.shape[0]
        p = np.clip(pred, 1e-9, 1 - 1e-9)
        return -np.sum(y_true * np.log(p)) / n

    def backward(self, pred, y_true):
        return (pred - y_true) / y_true.shape[0]


class MSE:
    def forward(self, pred, y_true):
        return np.mean((pred - y_true) ** 2)

    def backward(self, pred, y_true):
        return 2 * (pred - y_true) / y_true.shape[0]