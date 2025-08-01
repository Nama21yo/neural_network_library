import numpy as np
# Base Loss Function Class
class Loss:
    def compute_loss(self, y_pred, y_true):
        """Compute the loss"""
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        """Compute the initial gradient for backpropagation"""
        raise NotImplementedError

# Mean Squared Error Loss
class MSE(Loss):
    def compute_loss(self, y_pred, y_true):
        """Mean Squared Error loss"""
        return np.mean((y_pred - y_true) ** 2) / 2

    def backward(self, y_pred, y_true):
        """Initial gradient for MSE"""
        return y_pred - y_true

# Categorical Crossentropy Loss
class CategoricalCrossentropy(Loss):
    def compute_loss(self, y_pred, y_true):
        """Categorical Crossentropy loss with clipping to avoid log(0)"""
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=0))

    def backward(self, y_pred, y_true):
        """Initial gradient for Categorical Crossentropy (often combined with Softmax)"""
        # For Softmax + Categorical Crossentropy, the combined gradient is simply y_pred - y_true
        return y_pred - y_true
