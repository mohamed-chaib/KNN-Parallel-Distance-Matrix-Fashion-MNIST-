import torch

def predict_all(X_test, X_train, y_train, k=3, batch_size=100):
    """
    GPU-accelerated KNN prediction using PyTorch.
    """
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert to PyTorch tensors
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)

    n_test = X_test_t.shape[0]
    predictions = []

    # Process in batches to avoid OOM on GPUs with limited memory
    for i in range(0, n_test, batch_size):
        X_test_batch = X_test_t[i:i+batch_size]
        
        # Compute Euclidean distance using matrix operations:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b
        dist = torch.cdist(X_test_batch, X_train_t)
        
        # Get k nearest neighbors
        _, topk_indices = torch.topk(dist, k, largest=False)
        
        # Get labels and find most common (majority vote)
        topk_labels = y_train_t[topk_indices]
        
        # Use mode to get the most frequent label
        # torch.mode returns (values, indices)
        batch_preds, _ = torch.mode(topk_labels, dim=1)
        predictions.append(batch_preds.cpu())

        if i % (batch_size * 5) == 0:
            print(f"Processed {min(i + batch_size, n_test)}/{n_test} samples...")

    return torch.cat(predictions).numpy()
