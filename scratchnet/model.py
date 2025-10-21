from .layers import Dense, ReLU, Sigmoid, Softmax

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', output_activation=None):
        self.layers = []
        prev = input_size

        act = ReLU if activation == 'relu' else (Sigmoid if activation == 'sigmoid' else None)
        if activation == 'tanh':
            # simple Tanh sans fichier séparé
            import numpy as np
            class Tanh:
                def forward(self, X):
                    self.out = np.tanh(X); return self.out
                def backward(self, d_out):
                    return d_out * (1 - self.out**2)
            act = Tanh

        for h in hidden_sizes:
            self.layers.append(Dense(prev, h))
            self.layers.append(act())
            prev = h

        # couche de sortie
        self.layers.append(Dense(prev, output_size))
        if output_activation == 'softmax':
            self.layers.append(Softmax())  # classification
        # sinon: régression => pas d'activation finale

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad, lr, optimizer):
        for layer in reversed(self.layers):
            if hasattr(layer, "backward") and layer.__class__.__name__ == "Dense":
                grad = layer.backward(grad, lr, optimizer)
            elif hasattr(layer, "backward"):
                grad = layer.backward(grad)