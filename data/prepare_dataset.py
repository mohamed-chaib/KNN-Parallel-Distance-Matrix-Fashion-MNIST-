import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from torchvision.datasets import FashionMNIST

OUTPUT_FILE = "fashion_mnist_processed.npz"


def load_data(data_dir="data"):
    """
    Download/load Fashion-MNIST and return train/test arrays.
    """
    print("Loading Fashion-MNIST dataset...")

    train_dataset = FashionMNIST(root=data_dir, train=True, download=True)
    test_dataset = FashionMNIST(root=data_dir, train=False, download=True)

    X_train = np.array(train_dataset.data)
    y_train = np.array(train_dataset.targets)

    X_test = np.array(test_dataset.data)
    y_test = np.array(test_dataset.targets)

    print("Dataset loaded.")
    return X_train, y_train, X_test, y_test


def preprocess_chunk(chunk):
    """
    Preprocess one chunk:
    - flatten 28x28 images to 784 features
    - normalize pixel values to [0, 1]
    """
    chunk = chunk.reshape(chunk.shape[0], 784)
    chunk = chunk.astype(np.float32) / 255.0
    return chunk


def preprocess_serial(data):
    """
    Serial preprocessing for comparison.
    """
    return preprocess_chunk(data)


def split_chunks(data, chunk_size):
    """
    Split dataset into chunks manually.
    """
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i : i + chunk_size])
    return chunks


def get_safe_process_count():
    """
    Pick a safer default number of processes for cross-platform use.
    Using all cores is often slower here because of multiprocessing overhead.
    """
    cpu_count = mp.cpu_count()
    return max(1, min(2, cpu_count))


def preprocess_parallel(data, chunk_size=5000, processes=None):
    """
    Parallel preprocessing using multiprocessing.
    Works on Windows and Linux when called under __main__.
    """
    if processes is None:
        processes = get_safe_process_count()

    chunks = split_chunks(data, chunk_size)
    print(f"Parallel preprocessing: {len(chunks)} chunks, {processes} process(es)...")

    with mp.Pool(processes=processes) as pool:
        processed_chunks = pool.map(preprocess_chunk, chunks)

    return np.vstack(processed_chunks)


def save_data(X_train, y_train, X_test, y_test, output_dir="data/processed"):
    """
    Save processed dataset into one .npz file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / OUTPUT_FILE
    np.savez(
        output_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    return output_path


def print_results(serial_time, parallel_time):
    """
    Print timing comparison and speedup.
    """
    print("\n--- Speed Comparison ---")
    print(f"Serial preprocessing time:   {serial_time:.4f} seconds")
    print(f"Parallel preprocessing time: {parallel_time:.4f} seconds")

    if parallel_time > 0:
        speedup_factor = serial_time / parallel_time
        speedup_percent = ((serial_time - parallel_time) / serial_time) * 100
        print(f"Speedup factor: {speedup_factor:.2f}x")
        print(f"Speedup percentage: {speedup_percent:.2f}%")
    else:
        print("Parallel time is too small to compute speedup.")

    if parallel_time < serial_time:
        print("Result: Parallel preprocessing is faster.")
    elif parallel_time > serial_time:
        print("Result: Serial preprocessing is faster on this machine.")
    else:
        print("Result: Serial and parallel preprocessing took the same time.")


def main():
    # 1) load dataset
    X_train, y_train, X_test, y_test = load_data()

    # 2) serial preprocessing
    print("\nStarting serial preprocessing...")
    start_serial = time.time()

    X_train_serial = preprocess_serial(X_train)
    X_test_serial = preprocess_serial(X_test)

    end_serial = time.time()
    serial_time = end_serial - start_serial
    print("Serial preprocessing done.")

    # 3) parallel preprocessing
    print("\nStarting parallel preprocessing...")
    start_parallel = time.time()

    X_train_parallel = preprocess_parallel(X_train, chunk_size=5000, processes=2)
    X_test_parallel = preprocess_parallel(X_test, chunk_size=5000, processes=2)

    end_parallel = time.time()
    parallel_time = end_parallel - start_parallel
    print("Parallel preprocessing done.")

    # 4) optional check to make sure both results are equal
    if np.array_equal(X_train_serial, X_train_parallel) and np.array_equal(
        X_test_serial, X_test_parallel
    ):
        print("\nCheck passed: serial and parallel results are identical.")
    else:
        print("\nWarning: serial and parallel results are different.")

    # 5) save processed dataset
    output_path = save_data(X_train_parallel, y_train, X_test_parallel, y_test)

    # 6) print final info
    print("\nDataset ready.")
    print(f"Saved file: {output_path}")
    print("X_train shape:", X_train_parallel.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test_parallel.shape)
    print("y_test shape:", y_test.shape)

    # 7) compare speed
    print_results(serial_time, parallel_time)


if __name__ == "__main__":
    mp.freeze_support()  # important for Windows
    main()
