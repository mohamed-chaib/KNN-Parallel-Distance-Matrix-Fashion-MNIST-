import numpy as np


def compute_distances(x, X_train):
    return np.sqrt(np.sum((X_train - x) ** 2, axis=1))


def predict_one(x, X_train, y_train, k=3):
    distances = compute_distances(x, X_train)
    k_idx = np.argsort(distances)[:k]
    k_labels = y_train[k_idx]
    return np.bincount(k_labels).argmax()


def predict_all(X_test, X_train, y_train, k=3):
    predictions = []

    for i, x in enumerate(X_test):
        pred = predict_one(x, X_train, y_train, k)
        predictions.append(pred)

        if i % 50 == 0:
            print(f"Processed {i} samples...")

    return np.array(predictions)