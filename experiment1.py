"""Experiment 1: synthetic blobs and flower datasets."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fully_connected_nn import FullyConnectedNN

try:
    from generate_datasets import generate_binary_problem, generate_flower_problem
except ImportError:
    def generate_binary_problem(N=1000, centers=None, std=1.0, random_state=42):
        rng = np.random.default_rng(random_state)
        if centers is None:
            centers = np.array([[0, 0], [8, 8]])
        n0 = N // 2
        n1 = N - n0
        X0 = rng.normal(loc=centers[0], scale=std, size=(n0, 2))
        X1 = rng.normal(loc=centers[1], scale=std, size=(n1, 2))
        X = np.vstack([X0, X1])
        y = np.hstack([np.zeros(n0), np.ones(n1)])
        idx = rng.permutation(N)
        return X[idx], y[idx]

    def generate_flower_problem(N=1000, noise=0.20, random_state=42):
        rng = np.random.default_rng(random_state)
        theta = rng.uniform(0, 2 * np.pi, N)
        y = (np.sin(4 * theta) > 0).astype(int)
        r = 2.0 + 0.8 * np.sin(4 * theta) + rng.normal(0, noise, N)
        X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        return X, y

try:
    from helpers import plot_decision_boundary
except ImportError:
    def plot_decision_boundary(predict_fn, X, y, title="Decision boundary"):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = predict_fn(grid).reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.25)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=15, edgecolors="k")
        plt.title(title)
        plt.xlabel("x1")
        plt.ylabel("x2")


def run_dataset(name, X, y, hidden_size=8, iterations=8000, lr=0.05, batch_size=None):
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    model = FullyConnectedNN(input_size=2, hidden_size=hidden_size, output_size=1, lr=lr, seed=42)
    model.fit(X_train, y_train, iterations=iterations, batch_size=batch_size, show_step=1000)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: train accuracy = {train_acc:.4f}, test accuracy = {test_acc:.4f}")

    plt.figure(figsize=(6, 5))
    plot_decision_boundary(lambda data: model.predict(data), X_test, y_test.ravel(), f"{name} boundary")
    out = f"{name.lower().replace(' ', '_')}_decision_boundary.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return model, test_acc


if __name__ == "__main__":
    X_blobs, y_blobs = generate_binary_problem(N=1000, centers=np.array([[0, 0], [8, 8]]))
    run_dataset("Easy blobs", X_blobs, y_blobs, hidden_size=6, iterations=5000, lr=0.05)

    X_hard, y_hard = generate_binary_problem(N=1000, centers=np.array([[0, 0], [2, 2]]), std=1.7)
    run_dataset("Hard blobs", X_hard, y_hard, hidden_size=8, iterations=8000, lr=0.03)

    X_flower, y_flower = generate_flower_problem(N=1000)
    run_dataset("Flower", X_flower, y_flower, hidden_size=16, iterations=12000, lr=0.05)
