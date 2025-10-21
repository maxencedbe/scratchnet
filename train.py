from scratchnet import MLP, Trainer, get_dataset
import numpy as np
import pickle, os

print("\n=== Welcome to ScratchNet ===\n")

# ============================================================
# Utility functions for safe user input
# ============================================================

def ask_choice(prompt, options, default=None):
    """Prompt user for a valid choice; repeat until valid."""
    while True:
        ans = input(prompt).strip().lower()
        if not ans and default is not None:
            return default
        if ans in options:
            return ans
        print(f"Invalid choice. Valid options: {', '.join(options)}")

def ask_float(prompt, default=None):
    """Prompt user for a float value."""
    while True:
        val = input(prompt).strip()
        if not val and default is not None:
            return default
        try:
            return float(val)
        except ValueError:
            print("Please enter a valid numeric value.")

def ask_int(prompt, default=None):
    """Prompt user for an integer value."""
    while True:
        val = input(prompt).strip()
        if not val and default is not None:
            return default
        try:
            return int(val)
        except ValueError:
            print("Please enter a valid integer value.")

# ============================================================
# Dataset selection
# ============================================================

available_datasets = ["xor", "spiral", "iris", "wine", "mnist", "custom"]

def ask_dataset():
    print("Select a dataset:")
    for i, name in enumerate(available_datasets, 1):
        print(f"  {i}. {name}")

    while True:
        ans = input(f"Choice ({'/'.join(str(i) for i in range(1, len(available_datasets)+1))}): ").strip()
        if ans.isdigit() and 1 <= int(ans) <= len(available_datasets):
            dataset = available_datasets[int(ans) - 1]
            break
        elif ans.lower() in available_datasets:
            dataset = ans.lower()
            break
        else:
            print("Invalid selection. Please try again.")

    print(f"Dataset selected: {dataset}\n")
    return dataset

dataset = ask_dataset()

# ============================================================
# Dataset loading
# ============================================================

custom_args = {}
if dataset == "custom":
    import pandas as pd
    path = input("Path to your CSV file: ").strip()
    while not os.path.exists(path):
        print("File not found.")
        path = input("Path to your CSV file: ").strip()

    df = pd.read_csv(path)
    print(f"Dataset loaded ({df.shape[0]} rows, {df.shape[1]} columns)")
    print(f"Columns: {', '.join(df.columns)}")

    target_col = input("Target column name (y): ").strip()
    while target_col not in df.columns:
        print("Invalid column name.")
        target_col = input("Target column name (y): ").strip()

    custom_args = {"path": path, "target_col": target_col}

else:
    print(f"Loading dataset '{dataset}'...")

# ============================================================
# Preprocessing and split
# ============================================================

split = 0.0 if dataset in ["xor", "spiral"] else 0.2
dataset_kwargs = {"test_split": split}
dataset_kwargs.update(custom_args)

X_train, y_train, X_test, y_test, scalers = get_dataset(dataset, **dataset_kwargs)

x_scaler = scalers.get("x_scaler")
y_scaler = scalers.get("y_scaler")
label_encoder = scalers.get("label_encoder")
task_type = scalers.get("task_type", "classification")

# Normalization fallback
if x_scaler is None:
    from sklearn.preprocessing import StandardScaler
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    if X_test is not None:
        X_test = x_scaler.transform(X_test)

n_features = X_train.shape[1]
n_classes = len(np.unique(y_train)) if task_type == "classification" else 1

print("Dataset ready for training.\n")

# ============================================================
# Model configuration
# ============================================================

print("Model configuration:")

manual = ask_choice("Do you want to configure the model manually? (y/n) [n]: ", 
                    ["y", "yes", "n", "no", ""], default="n")

if manual in ["y", "yes"]:
    hidden_layers = input("Hidden layers (e.g. 64,32) [64,32]: ").strip() or "64,32"
    hidden_layers = [int(x) for x in hidden_layers.split(",")]
    activation = ask_choice("Activation function (relu/sigmoid/tanh) [relu]: ", ["relu", "sigmoid", "tanh", ""], default="relu")
    optimizer = ask_choice("Optimizer (adam/sgd) [adam]: ", ["adam", "sgd", ""], default="adam")
    lr = ask_float("Learning rate [auto=-1]: ", default=-1)
    epochs = ask_int("Number of epochs [50]: ", default=50)
    batch_size = ask_int("Batch size [32]: ", default=32)
else:
    # Automatic mode
    if n_features <= 2:
        hidden_layers = [8, 4]
    elif n_features <= 10:
        hidden_layers = [32, 16]
    elif n_features <= 100:
        hidden_layers = [64, 32]
    else:
        hidden_layers = [128, 64]

    activation = "tanh" if dataset in ["xor", "spiral"] else "relu"
    optimizer = "sgd" if dataset in ["xor", "spiral"] else "adam"

    if len(X_train) < 200:
        epochs, batch_size = 200, 8
    elif len(X_train) < 2000:
        epochs, batch_size = 100, 16
    elif len(X_train) < 10000:
        epochs, batch_size = 50, 32
    else:
        epochs, batch_size = 50, 64

    lr = -1  # auto

# ============================================================
# Automatic learning rate adjustment
# ============================================================

if lr == -1:
    if dataset in ["xor", "spiral"]:
        lr = 0.02
    elif dataset in ["iris", "wine"]:
        lr = 0.01
    elif dataset == "mnist":
        lr = 0.001
    elif task_type == "regression":
        lr = 0.001
    else:
        lr = 0.01

# ============================================================
# Task-dependent parameters
# ============================================================

loss = "mse" if task_type == "regression" else "cross_entropy"
output_activation = None if task_type == "regression" else "softmax"

# ============================================================
# Model initialization
# ============================================================

model = MLP(
    input_size=n_features,
    hidden_sizes=hidden_layers,
    output_size=n_classes,
    activation=activation,
    output_activation=output_activation
)
trainer = Trainer(model, loss=loss, optimizer=optimizer, lr=lr)

# ============================================================
# Confirmation before training
# ============================================================

print("\nPlease review your configuration before training:")
print(f"   • Dataset: {dataset}")
print(f"   • Task type: {task_type.capitalize()}")
print(f"   • Loss: {loss}")
print(f"   • Optimizer: {optimizer}")
print(f"   • Learning rate: {lr}")
print(f"   • Epochs: {epochs}")
print(f"   • Batch size: {batch_size}")
print(f"   • Hidden layers: {hidden_layers}\n")

confirm = ask_choice("Start training? (y/n) [y]: ", ["y", "yes", "n", "no", ""], default="y")
if confirm not in ["", "y", "yes"]:
    print("Training canceled.\n")
    exit()

# ============================================================
# Training with early stopping
# ============================================================

trainer.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, patience=5)

# ============================================================
# Model saving
# ============================================================

save_choice = ask_choice("\nSave this model? (y/n) [y]: ", ["y", "yes", "n", "no", ""], default="y")
if save_choice in ["", "y", "yes"]:
    os.makedirs("models", exist_ok=True)

    dataset_name = os.path.splitext(os.path.basename(custom_args.get("path", dataset)))[0]
    model_path = f"models/{dataset_name}_model.pkl"
    trainer.save(model_path)
    print(f"Model saved in '{model_path}'")

    if x_scaler is not None:
        with open(f"models/{dataset_name}_x_scaler.pkl", "wb") as f:
            pickle.dump(x_scaler, f)
    if y_scaler is not None:
        with open(f"models/{dataset_name}_y_scaler.pkl", "wb") as f:
            pickle.dump(y_scaler, f)

    if dataset == "custom":
        import pandas as pd
        df = pd.read_csv(custom_args["path"])
        target_col = custom_args["target_col"]
        feature_info = {
            "feature_names": list(df.drop(columns=[target_col]).columns),
            "target_name": target_col,
            "task_type": task_type
        }
        with open(f"models/{dataset_name}_info.pkl", "wb") as f:
            pickle.dump(feature_info, f)
        print(f"Feature info saved in 'models/{dataset_name}_info.pkl'")

    if label_encoder is not None:
        with open(f"models/{dataset_name}_label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)
        print(f"Label encoder saved in 'models/{dataset_name}_label_encoder.pkl'")

print("\nTraining completed successfully.\n")