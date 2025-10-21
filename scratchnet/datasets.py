import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# ============================================================
# Synthetic dataset generators
# ============================================================

def make_xor():
    """Simple 2D XOR dataset."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    return X, y


def make_spiral(n_points=100, n_classes=3, noise=0.2, seed=42):
    """Generate a 2D spiral dataset."""
    np.random.seed(seed)
    X = np.zeros((n_points * n_classes, 2))
    y = np.zeros(n_points * n_classes, dtype='uint8')
    for j in range(n_classes):
        ix = range(n_points * j, n_points * (j + 1))
        r = np.linspace(0.0, 1, n_points)
        t = np.linspace(j * 4, (j + 1) * 4, n_points) + np.random.randn(n_points) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y

# ============================================================
# Standard dataset loaders
# ============================================================

def load_iris_dataset():
    data = load_iris()
    return data.data, data.target


def load_wine_dataset():
    data = load_wine()
    return data.data, data.target


def load_mnist_dataset(limit=None):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X / 255.0
    y = y.astype(int)
    if limit:
        X, y = X[:limit], y[:limit]
    return X, y

# ============================================================
# Main dataset loader
# ============================================================

def get_dataset(name, test_split=0.2, seed=42, **kwargs):
    """
    Load a dataset by name.
    Supports both standard and custom CSV datasets.
    Returns:
        X_train, y_train, X_test, y_test, {"x_scaler", "y_scaler", "label_encoder", "task_type"}
    """
    name = name.lower()
    np.random.seed(seed)

    # --- Built-in or synthetic datasets
    if name == "xor":
        X, y = make_xor()
    elif name == "spiral":
        X, y = make_spiral(**kwargs)
    elif name == "iris":
        X, y = load_iris_dataset()
    elif name == "wine":
        X, y = load_wine_dataset()
    elif name == "mnist":
        X, y = load_mnist_dataset(limit=kwargs.get("limit", 5000))

    # --- Custom CSV dataset
    elif name == "custom":
        path = kwargs.get("path")
        target_col = kwargs.get("target_col")

        if path is None or target_col is None:
            raise ValueError("Custom dataset requires 'path' and 'target_col' parameters.")

        df = pd.read_csv(path)
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in dataset.")

        # Separate features and target
        X = df.drop(columns=[target_col]).values.astype(np.float32)
        y = df[target_col].values

        # Initialize scalers and encoders
        X_scaler = StandardScaler()
        y_scaler = None
        label_encoder = None

        print("Normalizing input features...")
        X = X_scaler.fit_transform(X)

        # ============================================================
        # Automatic task type detection
        # ============================================================
        if np.issubdtype(y.dtype, np.number):
            n_unique = len(np.unique(y))
            ratio_unique = n_unique / len(y)

            if n_unique > 20 or ratio_unique > 0.05:
                # Regression
                y_scaler = MinMaxScaler()
                y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
                task_type = "regression"
                print(f"Detected regression task ({n_unique} unique values).")
            else:
                # Numeric classification
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y.astype(int))
                task_type = "classification"
                print(f"Detected numeric classification ({n_unique} classes).")
        else:
            # Categorical classification
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            task_type = "classification"
            print(f"Detected categorical classification ({len(np.unique(y))} classes).")

        # Train/test split
        if test_split > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=seed
            )
        else:
            X_train, y_train, X_test, y_test = X, y, None, None

        return X_train, y_train, X_test, y_test, {
            "x_scaler": X_scaler,
            "y_scaler": y_scaler,
            "label_encoder": label_encoder,
            "task_type": task_type
        }

    else:
        raise ValueError(f"Unknown dataset name: '{name}'.")

    # --- For standard datasets
    if X.ndim > 1:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Train/test split
    if test_split > 0 and len(X) > 10:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=seed,
                stratify=y if len(np.unique(y)) > 1 else None
            )
        except ValueError:
            X_train, y_train, X_test, y_test = X, y, None, None
    else:
        X_train, y_train, X_test, y_test = X, y, None, None

    return X_train, y_train, X_test, y_test, {
        "x_scaler": None,
        "y_scaler": None,
        "label_encoder": None,
        "task_type": "classification"
    }