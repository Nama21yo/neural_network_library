import numpy as np

# Base Activation Function Class
class Activation:
    def forward(self, z):
        """Forward pass for activation"""
        raise NotImplementedError

    def backward(self, z):
        """Backward pass for activation derivative"""
        raise NotImplementedError

# ReLU Activation
class ReLU(Activation):
    def forward(self, z):
        """ReLU activation: f(z) = max(0, z)"""
        return np.maximum(0, z)

    def backward(self, z):
        """Derivative of ReLU: 1 if z > 0, else 0"""
        return (z > 0).astype(float)

# Sigmoid Activation
class Sigmoid(Activation):
    def forward(self, z):
        """Sigmoid activation: f(z) = 1 / (1 + exp(-z))"""
        return 1 / (1 + np.exp(-z))

    def backward(self, z):
        """Derivative of Sigmoid"""
        s = self.forward(z) # Calculate sigmoid output during backward pass
        return s * (1 - s)

# Tanh Activation
class Tanh(Activation):
    def forward(self, z):
        """Tanh activation"""
        return np.tanh(z)

    def backward(self, z):
        """Derivative of Tanh"""
        return 1 - np.tanh(z) ** 2

# Softmax Activation
class Softmax(Activation):
    def forward(self, z):
        """Softmax activation for multi-class output"""
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # Subtract max for numerical stability
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def backward(self, z):
        """Dummy derivative for softmax when used with crossentropy loss"""
        # The gradient for softmax is typically handled together with the cross-entropy loss
        # For backpropagation purposes with crossentropy, we return 1 here and the
        # combined gradient is calculated in the loss function's backward pass (or directly in the training loop).
        return np.ones_like(z)
