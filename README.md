# Neural Network Library

This project implements a neural network framework **from scratch** using Python and NumPy. It is applied to the classic **Zoo dataset** to classify animals into one of seven predefined classes.

Here is the dataset link: https://www.kaggle.com/datasets/uciml/zoo-animal-classification

The implementation includes:

- A modular dense neural network framework
- Forward and backward propagation
- ReLU and Softmax activation functions
- Categorical Crossentropy loss
- Batch training with support for different optimizers (Gradient Descent, Adam)
- Evaluation and prediction

---

## Dataset

We use the **Zoo dataset** which includes 101 animals described by 16 boolean features (e.g., "has feathers", "has legs") and a label representing the class of the animal (1 to 7).

- Input shape: (16 features × number of samples)
- Output: One-hot encoded class labels (7 classes)

---

## Architecture

The neural network can be customized in terms of the number of layers and neurons. A sample configuration used in this project:

- Input Layer: 16 features
- Hidden Layer 1: 32 neurons, ReLU
- Hidden Layer 2: 16 neurons, ReLU
- Hidden Layer 3: 8 neurons, ReLU
- Output Layer: 7 neurons, Softmax

---

## Features

### 1. Dense Layers

Each dense layer implements the following operations:

- **Forward Pass**:
  \[
  Z = W \cdot A\_{prev} + b
  \]
  \[
  A = ext{activation}(Z)
  \]

- **Backward Pass** (using chain rule):
  \[
  dW = rac{1}{m} \cdot (dZ \cdot A*{prev}^T)
  \]
  \[
  db = rac{1}{m} \cdot \sum dZ
  \]
  \[
  dA*{prev} = W^T \cdot dZ
  \]

### 2. Activation Functions

Includes:

- **ReLU** and its derivative
- **Softmax** (numerically stable)
- **Sigmoid**, **Tanh**, and **Leaky ReLU** for optional extension

Softmax is used in the output layer for multi-class classification.

### 3. Loss Function

- **Categorical Crossentropy Loss**:
  \[
  L = -\sum y \cdot \log(\hat{y})
  \]
  This is combined with softmax to simplify gradient calculation.

Clipping is used to prevent `log(0)` and ensure numerical stability.

### 4. Optimizers

Supports two optimizers:

- **Gradient Descent**: Basic weight update rule
- **Adam**: Combines momentum and RMSProp for adaptive learning

### 5. Batch Processing

Data is split into mini-batches during training. This improves generalization and training stability.

Batch gradient computation:

- Allows vectorized operations (faster)
- Requires transposing inputs to align dimensions for matrix multiplication
- Reduces memory consumption per step

---

## Training and Evaluation

Training involves:

- Mini-batch forward and backward propagation
- Updating parameters with the selected optimizer
- Loss monitoring per batch and epoch

Evaluation is done by computing classification accuracy on a test split.

---

## Prediction

The trained model is used to classify new animals based on their feature vectors. The highest softmax probability is selected as the predicted class.

---

## Project Structure

- `neural_network_framework.py`: Main training and evaluation logic
- `DenseLayer`: Defines layer behavior
- `NeuralNetwork`: Controls the training loop, forward and backward pass
- Activation and loss function modules
- Optimizer modules (Adam and GradientDescent)
- Zoo dataset loading and preprocessing
- Evaluation and prediction

---

## Extending the Project

Bonus improvements that can be implemented:

- Add support for Dropout regularization
- Implement additional optimizers (SGD with momentum, RMSProp)
- Add learning rate decay scheduler
- Visualize training loss and accuracy
- Create a more flexible configuration API for network architecture

---

## How to Run

1. Upload the dataset (`zoo.csv`) to your working directory or Google Drive
2. Make sure the dataset path in the script is correctly set
3. Run the script in a Python environment (e.g., Colab, Jupyter, or local)
4. Adjust layer configuration or learning parameters as needed

---

## Key Learning Outcomes

- Understanding the mathematics of forward and backpropagation
- Implementing a neural network without deep learning libraries
- Grasping how gradient descent and activations affect learning
- Handling batch-based training and data flow in matrix form
- Observing how model performance is influenced by architecture and learning configuration
