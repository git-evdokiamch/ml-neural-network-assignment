"""Experiment 3: Breast Cancer and synthetic data with PyTorch."""

import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


class TorchBinaryNN(nn.Module):
    def __init__(self, input_size, hidden_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_torch_model(X_train, y_train, X_test, y_test, hidden_size=16, lr=0.01, epochs=250, batch_size=64, seed=0):
    torch.manual_seed(seed)
    model = TorchBinaryNN(X_train.shape[1], hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        probs = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    preds = (probs >= 0.5).astype(int)
    return model, accuracy_score(y_test, preds)


def breast_cancer_experiment():
    data = load_breast_cancer()
    X, y = data.data, data.target.reshape(-1, 1)
    accuracies = []
    start = time.perf_counter()

    for run in range(20):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=run, stratify=y
        )
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        _, acc = train_torch_model(X_train, y_train, X_test, y_test, seed=run)
        accuracies.append(acc)
        print(f"Run {run + 1:02d}: test accuracy = {acc:.4f}")

    accuracies = np.array(accuracies)
    print("\nPyTorch Breast Cancer Results")
    print(f"Mean accuracy: {accuracies.mean():.4f}")
    print(f"Std accuracy:  {accuracies.std():.4f}")
    print(f"Execution time: {time.perf_counter() - start:.2f} seconds")


def architecture_trials_breast_cancer():
    data = load_breast_cancer()
    X, y = data.data, data.target.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for hidden in [4, 8, 16, 32]:
        _, acc = train_torch_model(X_train, y_train, X_test, y_test, hidden_size=hidden, seed=42)
        print(f"Hidden neurons {hidden:02d}: accuracy = {acc:.4f}")


if __name__ == "__main__":
    breast_cancer_experiment()
    print("\nArchitecture trials")
    architecture_trials_breast_cancer()
