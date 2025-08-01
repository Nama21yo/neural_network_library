import numpy as np
from utils import BatchGenerator
class NeuralNetwork:
    def __init__(self, optimizer, loss_function, verbosity=1):
        """Initialize the network with an optimizer, loss function object, and verbosity level"""
        self.layers = []
        self.optimizer = optimizer
        self.loss_function = loss_function # Store an instance of a Loss class
        self.verbosity = verbosity # Add verbosity parameter

    def add_layer(self, layer):
        """Add a layer to the network"""
        self.layers.append(layer)

    def forward(self, X):
        """Forward propagation through all layers"""
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def compute_loss(self, y_pred, y_true):
        """Compute the loss using the loss function object"""
        return self.loss_function.compute_loss(y_pred, y_true)

    def train(self, X, y, epochs, batch_size):
        """Train the network using mini-batch processing"""
        n_batches = X.shape[1] // batch_size
        batch_generator = BatchGenerator(X, y, batch_size) # Create a BatchGenerator instance

        for epoch in range(epochs):
            # Process data in mini-batches
            batch_loss = 0
            for i, (X_batch, y_batch) in enumerate(batch_generator): # Iterate through the BatchGenerator
                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_pred, y_batch)
                batch_loss += loss

                # Print batch loss if verbosity is high enough
                if self.verbosity >= 2:
                    print(f"\nEpoch {epoch+1}, Batch {i+1}/{n_batches}")
                    print(f"  Loss: {loss:.6f}")
                    # Print y_pred for the first sample in the batch
                    if self.verbosity >= 2 and X_batch.shape[1] > 0:
                         print(f"  y_pred (first sample): {y_pred[:, 0]}")
                         print(f"  y_true (first sample): {y_batch[:, 0]}")


                # Initial gradient for output layer
                da = self.loss_function.backward(y_pred, y_batch) # Use loss object's backward method

                grads = []
                # Backpropagation through layers
                for layer in reversed(self.layers):
                    da, dW, db = layer.backward(da)
                    grads.append((dW, db))
                # Reverse the grads list to match the layer order
                grads.reverse()

                # Print summary of gradients if verbosity is high enough
                if self.verbosity >= 2:
                    print("  Gradients (per layer):")
                    for idx, (dW, db) in enumerate(grads):
                        print(f"    Layer {idx}: dW (mean={np.mean(dW):.6f}, std={np.std(dW):.6f}), "
                              f"db (mean={np.mean(db):.6f}, std={np.std(db):.6f})")


                # Update parameters
                self.optimizer.update(self.layers, grads)


            # Monitor progress (epoch loss)
            if self.verbosity >= 1 and (epoch % 10 == 0 or epoch == epochs - 1): # Print epoch loss every 10 epochs or on the last epoch
                y_pred_full = self.forward(X)
                loss_full = self.compute_loss(y_pred_full, y)
                print(f"\n=== Epoch {epoch+1} Summary ===")
                print(f"Average Batch Loss: {batch_loss / n_batches:.6f}")
                print(f"Full Dataset Loss: {loss_full:.6f}")
                if self.verbosity >= 2:
                    print(f"Final y_pred sample: {y_pred_full[:, 0]}")
                    print(f"Final y_true sample: {y[:, 0]}")
