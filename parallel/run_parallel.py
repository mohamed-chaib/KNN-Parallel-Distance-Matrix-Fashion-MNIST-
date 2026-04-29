import numpy as np
import time
import ctypes
from sklearn.metrics import accuracy_score


# ================================
# Load C shared library
# ================================
lib = ctypes.CDLL("./libknn.so")


# ================================
# Define C function signature
# ================================
lib.knn_predict.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # X_test
    ctypes.c_int,                      # n_test
    ctypes.POINTER(ctypes.c_double),   # X_train
    ctypes.POINTER(ctypes.c_int),      # y_train
    ctypes.c_int,                      # n_train
    ctypes.c_int,                      # dim
    ctypes.c_int,                      # k
    ctypes.POINTER(ctypes.c_int)       # predictions
]


# ================================
# Python wrapper for C KNN
# ================================
def predict_all(X_test, X_train, y_train, k=3):
    X_test = np.ascontiguousarray(X_test, dtype=np.float64)
    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    y_train = np.ascontiguousarray(y_train, dtype=np.int32)

    n_test, dim = X_test.shape
    n_train = X_train.shape[0]

    y_pred = np.zeros(n_test, dtype=np.int32)

    lib.knn_predict(
        X_test.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_test,
        X_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_train.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        n_train,
        dim,
        k,
        y_pred.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )

    return y_pred


# ================================
# Main execution
# ================================
def main():
    print("Loading processed dataset...")

    data = np.load("../data/processed/fashion_mnist_processed.npz")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Small subset for testing
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    X_test = X_test[:500]
    y_test = y_test[:500]

    print("Dataset loaded:")
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    print("\nRunning C + OpenMP KNN...")

    start = time.time()

    y_pred = predict_all(X_test, X_train, y_train, k=3)

    end = time.time()

    acc = accuracy_score(y_test, y_pred)

    # ================================
    # Results
    # ================================
    print("\n--- RESULTS ---")
    print("Accuracy:", acc)
    print("Time:", end - start, "seconds")


if __name__ == "__main__":
    main()