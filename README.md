
# Neural Network Library

This project implements a neural network framework **from scratch** using Python and NumPy. It is applied to the classic **Zoo dataset** to classify animals into one of seven predefined classes.

**Dataset**: [Zoo Animal Classification on Kaggle](https://www.kaggle.com/datasets/uciml/zoo-animal-classification)

The implementation includes:

- A modular dense neural network framework
- Forward and backward propagation
- ReLU and Softmax activation functions
- Categorical Crossentropy loss
- Batch training with support for different optimizers (Gradient Descent, Adam)
- Evaluation and prediction

---

## Dataset

We use the **Zoo dataset**, which includes 101 animals described by 16 boolean features (e.g., "has feathers", "has legs") and a label representing the class of the animal (1 to 7).

- Input shape: `(number of samples, 16 features)`
- Output shape: `(number of samples, 7 classes)` (one-hot encoded)

---

## Architecture

The neural network can be customized in terms of the number of layers and neurons. A sample configuration used in this project:

- **Input Layer**: 16 features
- **Hidden Layer 1**: 32 neurons, ReLU activation
- **Hidden Layer 2**: 16 neurons, ReLU activation
- **Hidden Layer 3**: 8 neurons, ReLU activation
- **Output Layer**: 7 neurons, Softmax activation

---

## Features

### 1. Dense Layers

Each dense layer implements the following operations:

**Forward Pass**:
```text
Z = W · A_prev + b
A = activation(Z)
```

**Backward Pass** (chain rule):
```text
dW = (1/m) · (dZ · A_prev.T)
db = (1/m) · sum(dZ)
dA_prev = W.T · dZ
```

Where:
- `Z` is the linear output
- `A` is the activated output
- `m` is the batch size
- `W`, `b` are the layer's parameters
- `dZ`, `dW`, `db` are gradients during backpropagation

---

### 2. Activation Functions

Included:
- **ReLU** and its derivative
- **Softmax** (numerically stable)
- Optional: **Sigmoid**, **Tanh**, **Leaky ReLU**

Softmax is used in the final output layer for multi-class classification.

---

### 3. Loss Function

**Categorical Crossentropy Loss**:
```text
L = -Σ (y * log(ŷ))
```

Where:
- `y` is the true one-hot label
- `ŷ` is the predicted softmax output

To ensure numerical stability:
- `log(0)` is clipped using small epsilon values.

---

### 4. Optimizers

Supported optimizers:

- **Gradient Descent** (vanilla update rule)
- **Adam** (adaptive method combining momentum and RMSProp)

---

### 5. Batch Processing

Data is split into mini-batches during training for better generalization and efficiency.

Benefits of batch training:
- Enables vectorized matrix operations
- Improves memory efficiency
- Reduces variance in gradient updates

---

## Training and Evaluation

Training involves:
- Mini-batch forward and backward propagation
- Parameter updates using the chosen optimizer
- Loss tracking per epoch

Evaluation:
- Calculates classification accuracy on a held-out test set

---

## Prediction

After training, the model can predict the class of a new animal based on its 16-feature vector. The predicted class is chosen by selecting the index of the highest probability in the softmax output.

---

## Project Structure

- `neural_network_framework.py` – Main training and evaluation logic
- `DenseLayer` – Defines forward/backward propagation per layer
- `NeuralNetwork` – Manages overall model, training loop, and evaluation
- `activations.py` – Implements activation functions
- `loss.py` – Implements crossentropy loss
- `optimizers.py` – Contains implementations for GradientDescent and Adam
- `data_loader.py` – Loads and preprocesses Zoo dataset

---

## Extending the Project

Bonus improvements to try:

- Dropout regularization
- Additional optimizers (e.g., RMSProp, SGD with momentum)
- Learning rate decay
- Training loss/accuracy visualization
- YAML/JSON config for defining model structure and training params

---

## How to Run

1. Download the dataset (`zoo.csv`) to your working directory.
2. Ensure the dataset path is set correctly in the script.
3. Run the Python script (`.py` file) in Jupyter Notebook, Google Colab, or any Python environment.
4. Customize the model architecture, learning rate, batch size, and number of epochs as needed.

---

## Key Learning Outcomes

- Understand forward and backward propagation mechanics
- Implement a neural network without high-level libraries
- Learn how gradient descent drives learning
- Gain experience with batch training and matrix computations
- Explore the influence of architecture and hyperparameters on model performance
