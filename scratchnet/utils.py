import numpy as np
import random

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

def normalize(X):
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

def one_hot(y, num_classes=None):
    if num_classes is None:
        num_classes = len(np.unique(y))
    onehot = np.zeros((y.size, num_classes))
    onehot[np.arange(y.size), y] = 1
    return onehot

def accuracy(y_pred, y_true):
    if y_true.ndim > 1:  # si one-hot
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_pred == y_true)

def make_batches(X, y, batch_size):
    n = X.shape[0]
    indices = np.random.permutation(n)
    for i in range(0, n, batch_size):
        batch_idx = indices[i:i + batch_size]
        yield X[batch_idx], y[batch_idx]