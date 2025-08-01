import numpy as np
class BatchGenerator:
    def __init__(self, X, y, batch_size):
        """Initialize BatchGenerator with data and batch size"""
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_samples = X.shape[1]
        self.indices = np.arange(self.num_samples)
        self.current_index = 0

    def __iter__(self):
        """Shuffle indices and reset current index for a new epoch"""
        np.random.shuffle(self.indices)
        self.current_index = 0
        return self

    def __next__(self):
        """Generate the next mini-batch"""
        if self.current_index >= self.num_samples:
            raise StopIteration

        end_index = min(self.current_index + self.batch_size, self.num_samples)
        batch_indices = self.indices[self.current_index:end_index]

        self.current_index = end_index

        return self.X[:, batch_indices], self.y[:, batch_indices]
