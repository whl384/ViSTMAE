import os
import numpy as np
import torch

def load_and_preprocess_dataset(dataset_name: str, project_dir: str, window_size: int = 224):
    paths = [os.path.join(project_dir, d, dataset_name) for d in ["dataset", "data"]] + \
            [os.path.join(project_dir, d) for d in ["dataset", "data"]] + [project_dir]
    train_path = next((os.path.join(p, f"{dataset_name}_train.npy") for p in paths if os.path.exists(os.path.join(p, f"{dataset_name}_train.npy"))), None)

    if train_path is None:
        raise FileNotFoundError(f"Cannot locate {dataset_name}_train.npy under project directory!")

    test_path = os.path.join(os.path.dirname(train_path), f"{dataset_name}_test.npy")
    label_path = os.path.join(os.path.dirname(train_path), f"{dataset_name}_test_label.npy")

    test_data_raw = np.nan_to_num(np.load(test_path, allow_pickle=True).astype(np.float32), nan=0.0)
    true_labels_raw = np.load(label_path, allow_pickle=True)
    raw_channels = test_data_raw.shape[1]

    train_load_raw = np.load(train_path, allow_pickle=True)
    if isinstance(train_load_raw, np.ndarray) and train_load_raw.dtype == 'object':
        valid_arrays = [arr for arr in train_load_raw if isinstance(arr, np.ndarray) and arr.ndim == 2]
        train_data_raw = np.vstack(valid_arrays).astype(np.float32) if len(valid_arrays) > 0 else test_data_raw.copy()
    elif train_load_raw.ndim == 3:
        train_data_raw = train_load_raw.reshape(-1, train_load_raw.shape[-1]).astype(np.float32)
    else:
        train_data_raw = train_load_raw.astype(np.float32)
    if train_data_raw.shape[1] > raw_channels:
        train_data_raw = train_data_raw[:, :raw_channels]
    train_data_raw = np.nan_to_num(train_data_raw, nan=0.0)

    active_mask = np.std(train_data_raw, axis=0) > 1e-4
    if 0 < np.sum(active_mask) < raw_channels and np.sum(active_mask) >= 3:
        train_data_raw = train_data_raw[:, active_mask]
        test_data_raw = test_data_raw[:, active_mask]

    channels = train_data_raw.shape[1]

    g_mean = np.mean(train_data_raw, axis=0, keepdims=True)
    g_std = np.std(train_data_raw, axis=0, keepdims=True)
    g_std = np.where(g_std < 1e-4, 1.0, g_std)
    train_data = (train_data_raw - g_mean) / g_std
    test_data = (test_data_raw - g_mean) / g_std

    all_data_tensor = torch.stack([
        torch.tensor(train_data[i * window_size: (i + 1) * window_size].T, dtype=torch.float32)
        for i in range(len(train_data) // window_size)
    ])

    return train_data_raw, test_data, true_labels_raw, all_data_tensor, channels