# serial/run_serial.py

import numpy as np
import time
from sklearn.metrics import accuracy_score
from knn import predict_all


def main():
    print("Loading processed dataset...")

    data = np.load("../data/processed/fashion_mnist_processed.npz")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    X_test = X_test[:500]
    y_test = y_test[:500]

    print("Starting KNN (serial)...")

    start = time.time()

    y_pred = predict_all(X_test, X_train, y_train, k=3)

    end = time.time()

    acc = accuracy_score(y_test, y_pred)

    print("\n--- RESULTS ---")
    print("Accuracy:", acc)
    print("Time:", end - start, "seconds")


if __name__ == "__main__":
    main()