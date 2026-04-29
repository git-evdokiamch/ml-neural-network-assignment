"""
Fully connected neural network with one hidden layer.
Only NumPy is used for the implementation of the model.
"""

import numpy as np
import matplotlib.pyplot as plt  # allowed by the assignment; not required by the class logic


class FullyConnectedNN:
    """Binary classifier: input -> hidden ReLU -> sigmoid output."""

    def __init__(self, input_size, hidden_size, output_size=1, lr=1e-2, seed=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.seed = seed

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None
        self.gradients = {}
        self.N = None

    def init_parameters(self):
        """Initialize parameters from N(0, 0.1)."""
        if self.seed is not None:
            np.random.seed(self.seed)
        self.W1 = 0.1 * np.random.randn(self.input_size, self.hidden_size)
        self.b1 = 0.1 * np.random.randn(1, self.hidden_size)
        self.W2 = 0.1 * np.random.randn(self.hidden_size, self.output_size)
        self.b2 = 0.1 * np.random.randn(1, self.output_size)

    @staticmethod
    def sigmoid(z):
        # Numerically stable clipping avoids overflow in exp.
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

    @staticmethod
    def _check_X_y(X, y):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (N, p).")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.ndim != 2 or y.shape[1] != 1:
            raise ValueError("y must have shape (N,) or (N, 1).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        return X.astype(float), y.astype(float)

    def forward(self, X):
        """Compute Z1, A1, Z2, A2 and store them in the object."""
        if self.W1 is None:
            raise ValueError("Parameters are not initialized. Call init_parameters() or fit().")
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def loss(self, X, y):
        X, y = self._check_X_y(X, y)
        probs = self.forward(X)
        eps = 1e-12
        probs = np.clip(probs, eps, 1.0 - eps)
        return float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))

    def backward(self, X, y):
        """Compute gradients for BCE loss with sigmoid output."""
        X, y = self._check_X_y(X, y)
        m = X.shape[0]

        # For BCE + sigmoid: dZ2 = A2 - y.
        dZ2 = self.A2 - y
        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return self.gradients

    def step(self):
        """Perform one Gradient Descent update."""
        if not self.gradients:
            raise ValueError("No gradients available. Call backward() first.")
        self.W1 -= self.lr * self.gradients["dW1"]
        self.b1 -= self.lr * self.gradients["db1"]
        self.W2 -= self.lr * self.gradients["dW2"]
        self.b2 -= self.lr * self.gradients["db2"]

    def fit(self, X, y, iterations=10000, batch_size=None, show_step=1000):
        X, y = self._check_X_y(X, y)
        if X.shape[1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} input features, got {X.shape[1]}.")

        self.N = X.shape[0]
        self.init_parameters()

        rng = np.random.default_rng(self.seed)
        order = rng.permutation(self.N)
        X_shuffled = X[order]
        y_shuffled = y[order]

        if batch_size is None:
            batch_size = self.N
        if batch_size <= 0:
            raise ValueError("batch_size must be positive or None.")

        start = 0
        for i in range(1, iterations + 1):
            end = min(start + batch_size, self.N)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            start = end

            if start >= self.N:
                order = rng.permutation(self.N)
                X_shuffled = X[order]
                y_shuffled = y[order]
                start = 0

            self.forward(X_batch)
            self.backward(X_batch, y_batch)
            self.step()

            if show_step is not None and show_step > 0 and i % show_step == 0:
                print(f"Iteration {i}: loss = {self.loss(X, y):.6f}")
