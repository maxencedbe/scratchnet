import os
import pickle
import numpy as np
from scratchnet import Trainer

print("\n=== ScratchNet — Prediction Mode ===\n")

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
        print(f"Invalid input. Valid options: {', '.join(options)}")

def ask_float(prompt, default=0.0):
    """Prompt user for a float value."""
    while True:
        val = input(prompt).strip()
        if not val:
            return default
        try:
            return float(val)
        except ValueError:
            print("Please enter a valid numeric value.")

# ============================================================
# Model selection
# ============================================================

models_dir = "models"
if not os.path.exists(models_dir):
    raise FileNotFoundError("No 'models' directory found.")

available_models = [f for f in os.listdir(models_dir) if f.endswith("_model.pkl")]
if not available_models:
    raise FileNotFoundError("No model files found in the 'models' directory.")

print("Available models:")
for i, name in enumerate(available_models, 1):
    print(f"  {i}. {name}")

# Secure model choice
while True:
    choice = input(f"\nSelect a model (1-{len(available_models)}): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(available_models):
        model_filename = available_models[int(choice) - 1]
        break
    elif choice == "":
        model_filename = available_models[0]
        print(f"Default model selected: {model_filename}")
        break
    else:
        print("Invalid choice, please try again.")

dataset_name = model_filename.replace("_model.pkl", "")
model_path = os.path.join(models_dir, model_filename)
print(f"\nModel selected: {dataset_name}")

# ============================================================
# Load model and related objects
# ============================================================

trainer = Trainer.load(model_path)

def load_object(name):
    path = os.path.join(models_dir, f"{dataset_name}_{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"{name} loaded successfully.")
        return obj
    else:
        print(f"No {name} file found.")
    return None

x_scaler = load_object("x_scaler")
y_scaler = load_object("y_scaler")
feature_info = load_object("info")
label_encoder = load_object("label_encoder")

# ============================================================
# Determine task type and feature columns
# ============================================================

task_type = "classification"
target_name = "value"
feature_names = [f"x{i+1}" for i in range(trainer.model.layers[0].W.shape[0])]

if feature_info:
    task_type = feature_info.get("task_type", "classification")
    target_name = feature_info.get("target_name", "value")
    feature_names = feature_info.get("feature_names", feature_names)

print(f"\nTask type detected: {task_type.capitalize()}")
print(f"Target: {target_name}")

# ============================================================
# User input for prediction
# ============================================================

print("\nEnter feature values for prediction:")
values = []
for name in feature_names:
    val = ask_float(f"  → {name}: ", default=0.0)
    values.append(val)

example = np.array([values], dtype=np.float32)
if x_scaler:
    example = x_scaler.transform(example)

# ============================================================
# Model prediction
# ============================================================

y_pred = trainer.predict(example)

# ============================================================
# Display result
# ============================================================

print("\nPrediction result:")

# ----- Regression -----
if task_type == "regression":
    if y_scaler:
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    print(f"  {target_name}: {float(y_pred.squeeze()):,.2f}")

# ----- Classification -----
else:
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        cls = int(np.argmax(y_pred, axis=1)[0])
    else:
        cls = int(y_pred[0] > 0.5)

    if label_encoder:
        decoded = label_encoder.inverse_transform([cls])[0]
        print(f"  {target_name}: class {cls} (original label: {decoded})")
    else:
        print(f"  {target_name}: class {cls}")

print("\nPrediction completed successfully.\n")