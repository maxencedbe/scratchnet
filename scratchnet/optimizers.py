import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, W, b, dW, db):
        W -= self.lr * dW
        b -= self.lr * db
        return W, b


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mW, self.vW, self.mb, self.vb = {}, {}, {}, {}
        self.t = 0

    def update(self, W, b, dW, db, key=None):
        if key is None:
            key = id(W)
        if key not in self.mW:
            self.mW[key] = np.zeros_like(W)
            self.vW[key] = np.zeros_like(W)
            self.mb[key] = np.zeros_like(b)
            self.vb[key] = np.zeros_like(b)

        self.t += 1
        self.mW[key] = self.beta1 * self.mW[key] + (1 - self.beta1) * dW
        self.vW[key] = self.beta2 * self.vW[key] + (1 - self.beta2) * (dW ** 2)
        self.mb[key] = self.beta1 * self.mb[key] + (1 - self.beta1) * db
        self.vb[key] = self.beta2 * self.vb[key] + (1 - self.beta2) * (db ** 2)

        mW_hat = self.mW[key] / (1 - self.beta1 ** self.t)
        vW_hat = self.vW[key] / (1 - self.beta2 ** self.t)
        mb_hat = self.mb[key] / (1 - self.beta1 ** self.t)
        vb_hat = self.vb[key] / (1 - self.beta2 ** self.t)

        W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
        b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

        return W, b