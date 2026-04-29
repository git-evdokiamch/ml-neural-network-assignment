"""Experiment 2: Breast Cancer with the NumPy model."""

import time
import platform
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from fully_connected_nn import FullyConnectedNN


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target.reshape(-1, 1)

    accuracies = []
    start_time = time.perf_counter()

    for run in range(20):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=run, stratify=y
        )

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = FullyConnectedNN(
            input_size=X_train.shape[1], hidden_size=16, output_size=1, lr=0.01, seed=run
        )
        model.fit(X_train, y_train, iterations=5000, batch_size=64, show_step=None)
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracies.append(acc)
        print(f"Run {run + 1:02d}: test accuracy = {acc:.4f}")

    total_time = time.perf_counter() - start_time
    accuracies = np.array(accuracies)
    print("\nResults")
    print(f"Mean accuracy: {accuracies.mean():.4f}")
    print(f"Std accuracy:  {accuracies.std():.4f}")
    print(f"Execution time: {total_time:.2f} seconds")
    print(f"CPU: {platform.processor() or platform.machine()}")
    print("Memory: fill in your machine RAM, e.g. 16 GB")


if __name__ == "__main__":
    main()
