import numpy as np
class Optimizer:
    def __init__(self, learning_rate):
        """Base optimizer class with learning rate"""
        self.learning_rate = learning_rate

    def update(self, layers, grads):
        pass

class GradientDescent(Optimizer):
    """Gradient Descent optimizer (mini-batch GD; SGD with batch_size=1)"""
    def update(self, layers, grads):
        """Update weights and biases using gradient descent"""
        for layer, (dW, db) in zip(layers, grads):
            layer.weights -= self.learning_rate * dW
            layer.biases -= self.learning_rate * db

class Adam(Optimizer):
    """Adam optimizer with momentum and RMSprop components"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.v = {}  # First moment estimate
        self.s = {}  # Second moment estimate
        self.t = 0   # Time step

    def update(self, layers, grads):
        """Update weights and biases using Adam optimization"""
        self.t += 1
        for i, (layer, (dW, db)) in enumerate(zip(layers, grads)):
            if i not in self.v:
                self.v[i] = {'W': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.biases)}
                self.s[i] = {'W': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.biases)}
            # Update biased first moment estimate
            self.v[i]['W'] = self.beta1 * self.v[i]['W'] + (1 - self.beta1) * dW
            self.v[i]['b'] = self.beta1 * self.v[i]['b'] + (1 - self.beta1) * db
            # Update biased second raw moment estimate
            self.s[i]['W'] = self.beta2 * self.s[i]['W'] + (1 - self.beta2) * (dW ** 2)
            self.s[i]['b'] = self.beta2 * self.s[i]['b'] + (1 - self.beta2) * (db ** 2)
            # Compute bias-corrected estimates
            v_corr_W = self.v[i]['W'] / (1 - self.beta1 ** self.t)
            v_corr_b = self.v[i]['b'] / (1 - self.beta1 ** self.t)
            s_corr_W = self.s[i]['W'] / (1 - self.beta2 ** self.t)
            s_corr_b = self.s[i]['b'] / (1 - self.beta2 ** self.t)
            # Update parameters
            layer.weights -= self.learning_rate * v_corr_W / (np.sqrt(s_corr_W) + self.epsilon)
            layer.biases -= self.learning_rate * v_corr_b / (np.sqrt(s_corr_b) + self.epsilon)
