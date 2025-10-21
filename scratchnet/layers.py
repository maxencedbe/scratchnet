import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        limit = np.sqrt(6 / (input_size + output_size))
        self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        self.b = np.zeros((1, output_size))

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, d_out, lr, optimizer=None):
        dW = self.X.T @ d_out
        db = np.sum(d_out, axis=0, keepdims=True)
        dX = d_out @ self.W.T
        if optimizer is None:
            self.W -= lr * dW
            self.b -= lr * db
        else:
            self.W, self.b = optimizer.update(self.W, self.b, dW, db)
        return dX


class ReLU:
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask

    def backward(self, d_out):
        return d_out * self.mask


class Sigmoid:
    def forward(self, X):
        self.out = 1 / (1 + np.exp(-X))
        return self.out

    def backward(self, d_out):
        return d_out * (self.out * (1 - self.out))


class Softmax:
    def forward(self, X):
        exp = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.out = exp / np.sum(exp, axis=1, keepdims=True)
        return self.out

    def backward(self, d_out):
        return d_out