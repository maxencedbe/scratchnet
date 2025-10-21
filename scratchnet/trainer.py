import numpy as np
from .losses import CrossEntropy, MSE
from .optimizers import SGD, Adam
import pickle
import os


class Trainer:
    def __init__(self, model, loss='cross_entropy', optimizer='sgd', lr=0.01):
        self.model = model
        self.lr = lr
        self.loss_name = loss.lower()
        self.optimizer_name = optimizer
        self.loss_fn = CrossEntropy() if self.loss_name == 'cross_entropy' else MSE()
        self.optimizer = SGD(lr) if optimizer.lower() == 'sgd' else Adam(lr)

    # ============================================================
    # Model training with early stopping
    # ============================================================
    def fit(self, X, y, epochs=10, batch_size=32, patience=None, min_delta=1e-6):
        """
        Train the model on the provided dataset.
        If 'patience' is set, enables early stopping when the loss no longer improves.
        """
        n_samples = X.shape[0]

        # Prepare target according to task type
        if self.loss_name == "cross_entropy":
            y_train = self._to_one_hot(y)
        else:
            # Regression → keep column shape
            y_train = y.reshape(-1, 1) if y.ndim == 1 else y

        best_loss = float("inf")
        best_state = None
        wait = 0

        print("\nStarting training...")

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X, y_train = X[indices], y_train[indices]
            losses = []

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                Xb = X[i:i+batch_size]
                yb = y_train[i:i+batch_size]

                # Forward pass
                pred = self.model.forward(Xb)
                loss = self.loss_fn.forward(pred, yb)
                losses.append(loss)

                # Backward pass
                grad = self.loss_fn.backward(pred, yb)
                self.model.backward(grad, self.lr, self.optimizer)

            avg_loss = np.mean(losses)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

            # Early stopping
            if patience:
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    wait = 0
                    best_state = pickle.dumps(self.model)
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"\nEarly stopping triggered: no improvement for {patience} epochs.")
                        break

        # Restore best model if early stopping was triggered
        if patience and best_state is not None:
            self.model = pickle.loads(best_state)
            print(f"Best model restored with lowest loss: {best_loss:.6f}")

    # ============================================================
    # One-hot encoding helper
    # ============================================================
    def _to_one_hot(self, y):
        """Convert a vector of integer labels to one-hot encoding."""
        y = y.astype(int)
        n_classes = len(np.unique(y))
        one_hot = np.zeros((y.size, n_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    # ============================================================
    # Prediction
    # ============================================================
    def predict(self, X, y_scaler=None):
        """
        Predict model outputs for new samples.
        If a scaler is provided (e.g., for regression), the output is inverse-transformed.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        y_pred = self.model.forward(X)

        if self.loss_name == "cross_entropy":
            y_pred = np.argmax(y_pred, axis=1)
        elif self.loss_name == "mse" and y_scaler is not None:
            y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        return y_pred

    # ============================================================
    # Save model
    # ============================================================
    def save(self, path):
        """Save the model and training parameters to a pickle file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "loss_name": self.loss_name,
                "optimizer_name": self.optimizer_name,
                "lr": self.lr
            }, f)

    # ============================================================
    # Load model
    # ============================================================
    @staticmethod
    def load(path):
        """Load a previously saved model from a pickle file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)
        trainer = Trainer(
            model=data["model"],
            loss=data.get("loss_name", "cross_entropy"),
            optimizer=data.get("optimizer_name", "adam"),
            lr=data.get("lr", 0.01)
        )
        print(f"Model loaded from {path}")
        return trainer