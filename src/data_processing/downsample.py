import torch
import numpy as np
from scipy.signal import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils import to_t


def downsample(data, times, method, target_sps, original_sps, n_pca_components):
    if method not in ['mean', 'max', 'pca', 'resample', 'decimate', 'pca_overlap']:
        print("Not a valid downsampling method. Using 'mean' as default.")
        method = 'mean'
    
    n_features = data.shape[1]
    print(f"Downsampling {original_sps}Hz data to {target_sps}Hz using {method} method...")
    print(f"Original data has {data.shape[0]} samples and {n_features} features.")

    # Trim to length divisible by downsampling factor
    downsampling_factor = original_sps // target_sps
    trim_length = (len(data) // downsampling_factor) * downsampling_factor
    data = data[:trim_length]
    times = times[:trim_length]
    print(f"Trimmed data to {data.shape[0]} samples for downsampling by a factor of {downsampling_factor}.")

    # Downsample data
    if method == "decimate":
        data = data[::downsampling_factor]
        times = times[::downsampling_factor]

    elif method == "mean":
        data = data.reshape(-1, downsampling_factor, n_features)
        data = data.mean(axis=1)
        times = times.reshape(-1, downsampling_factor)[:, 0]

    elif method == "max":
        data = data.reshape(-1, downsampling_factor, n_features).max(axis=1)
        times = times.reshape(-1, downsampling_factor)[:, 0]

    elif method == "pca":
        data = run_blocked_pca(data, n_pca_components, downsampling_factor)
        times = times.reshape(-1, downsampling_factor)[:, 0]

    elif method == "resample":
        n_blocks = len(data) // downsampling_factor
        data = resample(data, n_blocks, axis=0)
        times = np.linspace(times[0].item(), times[-1].item(), n_blocks)
    
    elif method == "pca_overlap":
        data = run_blocked_pca_overlap(data, n_pca_components, downsampling_factor)
        times = times.reshape(-1, downsampling_factor)[:, 0]
    
    print(f"Downsampled data has {data.shape[0]} samples and {data.shape[1]} features.")
    return to_t(data), torch.tensor(times, dtype=torch.float64)


def run_blocked_pca(data, n_components, downsampling_factor):
    n_blocks = len(data) // downsampling_factor
    
    # Flatten data so each new row is a block of downsampling_factor samples
    flattened_data = data.reshape(n_blocks, -1) # (n_blocks, block size * num features)

    data_pca, _, _ = run_pca(flattened_data, n_components)
    return data_pca


def run_blocked_pca_overlap(data, n_components, downsampling_factor, overlap=20):
    print(f"Running blocked PCA with overlap of {overlap} samples")

    step_size = downsampling_factor
    window_size = step_size + overlap
    n_windows = int(len(data) / step_size)

    # Pad with zeros at end to handle last window
    padding = np.zeros((overlap, data.shape[1]))
    data = np.concatenate((data, padding), axis=0)

    # Generate and flatten overlapping windows
    flattened_windows = []
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        window = data[start:end, :]

        if window.shape[0] != window_size:
            continue # Skip if window is not full size (just in case)

        flattened = window.flatten()  # Flatten the window
        flattened_windows.append(flattened)
    
    # Stack into array
    flattened_windows = np.stack(flattened_windows)

    # Run PCA
    data_pca, _, _ = run_pca(flattened_windows, n_components)
    return data_pca


def run_pca(data, n_components):
    # Standardize
    scalar = StandardScaler()
    data = scalar.fit_transform(data)

    # Run PCA
    print(f"Running PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)

    return data_pca, scalar.mean_, scalar.scale_