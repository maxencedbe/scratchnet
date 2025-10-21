# 🧠 ScratchNet

**ScratchNet** is a simple **Multilayer Perceptron (MLP)** framework built entirely **from scratch using NumPy** — no PyTorch or TensorFlow required.

It includes a clean modular design with dataset management, model training (including early stopping), prediction utilities, and a friendly CLI for experimenting with neural networks on standard or custom datasets.

---

## 🚀 Features

- Fully implemented **MLP** (forward, backward propagation, gradients, etc.)
- Supports both **classification** and **regression**
- **Early stopping** during training
- **Automatic or manual** hyperparameter configuration
- Built-in datasets:
  - `xor`
  - `spiral`
  - `iris`
  - `wine`
  - `mnist`
  - `custom` (load your own CSV)
- Automatic task-type detection for custom datasets (classification vs regression)
- Automatic scaling and label encoding
- Model persistence  (`.pkl` files for model, scalers, encoders)
- Clear and safe CLI with user-friendly prompts

---

## 🧩 Project Structure

ScratchNet/
│
├── train.py # Main training script (CLI)
├── predict.py # Prediction script (CLI)
├── README.md # Project documentation
├── requirements.txt # Dependencies
│
├── custom_datasets/ # Custom CSV datasets and generators
│ ├── flowers.csv
│ ├── flowers_generate.py
│ ├── houses.csv
│ └── houses_generate.py
│
├── models/ # Saved models, scalers, and metadata
│
├── scratchnet/ # Core implementation
│   ├── init.py
│   ├── datasets.py # Dataset loaders and preprocessing
│   ├── layers.py # Core layer definitions (Dense, etc.)
│   ├── losses.py # Loss functions (CrossEntropy, MSE)
│   ├── model.py # MLP architecture
│   ├── optimizers.py # Optimizers (SGD, Adam)
│   ├── trainer.py # Training logic + early stopping
│   └── utils.py # Utilities (one-hot encoding, accuracy)
└

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/maxencedbe/scratchnet.git
cd ScratchNet
```

Then:

```bash
pip install -r "requirements.txt"
```

---

## Training a model

Launch the training CLI:

```bash
python train.py
```

You’ll be prompted to:

- Choose a dataset (e.g., mnist, iris, or custom for your own CSV)
- Configure the model manually or let ScratchNet auto-tune hyperparameters
- Confirm before training start

Example :

```bash
Select a dataset:
  1. xor
  2. spiral
  3. iris
  4. wine
  5. mnist
  6. custom
Choice (1/2/3/4/5/6): 3
Dataset selected: iris

Loading dataset 'iris'...
Dataset ready for training.

Model configuration:
Do you want to configure the model manually? (y/n) [n]: n

Please review your configuration before training:
   • Dataset: iris
   • Task type: Classification
   • Loss: cross_entropy
   • Optimizer: adam
   • Learning rate: 0.01
   • Epochs: 200
   • Batch size: 8
   • Hidden layers: [32, 16]

Start training? (y/n): y

...
Best model restored with lowest loss: 0.024720

Save this model? (y/n) [y]:
Model saved at models/iris_model.pkl

Training completed successfully.

```

--- 

## Predictions

Once a model is trained, you can load it and make predictions:

```bash
python predict.py
```

You’ll be asked to:
- Select a saved model (e.g., models/iris_model.pkl)
- Enter feature values manually

Example :

```bash
Available models:
  1. iris_model.pkl
Select a model (1-1): 1
Model selected: iris

Task type detected: Classification
Target: species

Enter feature values for prediction:
  → sepal_length : 5.1
  → sepal_width  : 3.5
  → petal_length : 1.4
  → petal_width  : 0.2

Prediction result:
  species: class 0 (original label: setosa)

Prediction completed successfully.
```

---

## Custom Datasets

You can train on your own CSV file:
- Choose custom as dataset.
- Provide the path to your CSV and target column name.
- ScratchNet automatically detects whether the task is classification or regression.

Example:

```bash
Path to your CSV: data/housing.csv
Target column (y): price_eur
```

---

## Model Saving

After training, ScratchNet automatically saves:
- The trained model (*_model.pkl)
- Feature scaler (*_x_scaler.pkl)
- Target scaler or label encoder (*_y_scaler.pkl, *_label_encoder.pkl)
- Dataset metadata (*_info.pkl)

All files are stored under the /models directory.

---

## Built with

- NumPy
- scikit-learn
- pandas

---

## License

This project is released under the MIT License — free to use, modify, and share.
