import numpy as np
import time
import ctypes
import os
from sklearn.metrics import accuracy_score

# ================================
# Load CUDA shared library
# ================================
try:
    lib = ctypes.CDLL("./libknn_cuda.so")
except OSError:
    print("Error: libknn_cuda.so not found. Did you run 'make' in bonus/cuda/?")
    exit(1)

# ================================
# Define C function signature
# ================================
lib.knn_cuda_predict.argtypes = [
    ctypes.POINTER(ctypes.c_float), # X_test
    ctypes.c_int,                  # n_test
    ctypes.POINTER(ctypes.c_float), # X_train
    ctypes.c_int,                  # n_train
    ctypes.POINTER(ctypes.c_int),   # y_train
    ctypes.c_int,                  # dim
    ctypes.c_int,                  # k
    ctypes.POINTER(ctypes.c_int)    # predictions
]

def predict_all_cuda(X_test, X_train, y_train, k=3):
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    y_train = np.ascontiguousarray(y_train, dtype=np.int32)

    n_test, dim = X_test.shape
    n_train = X_train.shape[0]
    y_pred = np.zeros(n_test, dtype=np.int32)

    lib.knn_cuda_predict(
        X_test.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n_test,
        X_train.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n_train,
        y_train.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dim,
        k,
        y_pred.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )

    return y_pred

def main():
    print("=== Raw CUDA KNN Implementation ===")
    
    # Load dataset
    paths_to_check = [
        "../../data/processed/fashion_mnist_processed.npz",
        "../../data/data/processed/fashion_mnist_processed.npz"
    ]
    
    data = None
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"Loading dataset from: {path}")
            data = np.load(path)
            break
            
    if data is None:
        print("Error: Processed dataset not found.")
        return

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    X_train_sub = X_train[:20000]
    y_train_sub = y_train[:20000]
    X_test_sub = X_test[:2000]
    y_test_sub = y_test[:2000]

    print(f"Dataset subset: Train={X_train_sub.shape}, Test={X_test_sub.shape}")

    start = time.time()
    y_pred = predict_all_cuda(X_test_sub, X_train_sub, y_train_sub, k=3)
    end = time.time()

    acc = accuracy_score(y_test_sub, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Time:     {end - start:.4f} seconds")

if __name__ == "__main__":
    main()
