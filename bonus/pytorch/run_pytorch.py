import numpy as np
import time
import torch
import os
from sklearn.metrics import accuracy_score
from knn_pytorch import predict_all

def main():
    print("=== PyTorch GPU/CPU KNN Implementation ===")
    
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
        print("Error: Processed dataset not found. Run 'python data/prepare_dataset.py' from the project root.")
        return

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Use a larger subset for GPU to show off performance
    # GPUs typically handle larger datasets much better than CPUs
    X_train_sub = X_train[:20000]
    y_train_sub = y_train[:20000]
    X_test_sub = X_test[:2000]
    y_test_sub = y_test[:2000]

    print(f"Dataset subset: Train={X_train_sub.shape}, Test={X_test_sub.shape}")

    start = time.time()
    
    # Run prediction
    y_pred = predict_all(X_test_sub, X_train_sub, y_train_sub, k=3, batch_size=200)
    
    end = time.time()
    elapsed = end - start
    
    acc = accuracy_score(y_test_sub, y_pred)
    
    print("\n--- RESULTS ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Time:     {elapsed:.4f} seconds")
    
    if torch.cuda.is_available():
        print(f"Throughput: {len(X_test_sub)/elapsed:.2f} samples/sec (GPU)")
    else:
        print(f"Throughput: {len(X_test_sub)/elapsed:.2f} samples/sec (CPU-PyTorch)")

if __name__ == "__main__":
    main()
