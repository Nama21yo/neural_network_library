import numpy as np
class DenseLayer:
    def __init__(self, input_size, output_size, activation):
        """Initialize a dense layer with weights, biases, and an activation object"""
        # Xavier/Glorot initialization
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(1.0 / input_size)
        self.biases = np.zeros((output_size, 1))
        self.activation = activation # Store an instance of an Activation class
        self.a_prev = None  # Previous layer's activation
        self.z = None       # Pre-activation value

    def forward(self, a_prev):
        """Forward propagation through the layer"""
        self.a_prev = a_prev
        self.z = self.weights @ a_prev + self.biases
        return self.activation.forward(self.z) # Use the activation object's forward method

    def backward(self, da):
        """Backward propagation to compute gradients"""
        batch_size = self.a_prev.shape[1] # type: ignore
        dz = da * self.activation.backward(self.z) # Use the activation object's backward method
        dW = (dz @ self.a_prev.T) / batch_size # type: ignore
        db = np.sum(dz, axis=1, keepdims=True) / batch_size
        da_prev = self.weights.T @ dz
        return da_prev, dW, db
