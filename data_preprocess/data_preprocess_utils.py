import numpy as np
from scipy.stats import rankdata
import os
import torch
import julius
from typing import Tuple, List
import pickle

def rank_metrics(metrics: np.ndarray) -> np.ndarray:
    """
    Rank metrics based on their values.
    
    Parameters:
        metrics (numpy.ndarray): Array of shape (n_samples, n_metrics).
            Metrics should have low values for good data.
    
    Returns:
        numpy.ndarray: Array of shape (n_samples) with the indices of the best samples.
    """
    assert len(metrics.shape) == 2
    ranks = np.apply_along_axis(rankdata, 0, metrics, method="max")
    ranks = np.mean(ranks, axis=1)
    best_ind = np.argsort(ranks)
    return best_ind

def downsampling(data_t: np.ndarray, data_x: np.ndarray, data_y: np.ndarray, freq: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample recordings to 50Hz.
    
    Parameters:
        data_t (numpy.ndarray): Time array.
        data_x (numpy.ndarray): Sensor recordings.
        data_y (numpy.ndarray): Labels.
        freq (int): Original frequency.
    
    Returns:
        tuple: Downsampled input arrays (data_t, data_x, data_y).
    """
    idx = np.arange(0, data_t.shape[0], int(freq/50))
    return data_t[idx], data_x[idx], data_y[idx]

def normalize_samples(data: np.ndarray) -> np.ndarray:
    """
    Normalize input data.
    
    Parameters:
        data (numpy.ndarray): Input data.
    
    Returns:
        numpy.ndarray: Normalized data.
    """
    if len(data) != 0:
        return (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    else:
        return data
    
def generate_batch_sequences(x: List[np.ndarray], y: List[np.ndarray], pid: List[np.ndarray], time_diff: int = 8) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Generate batches of data.
    
    Parameters:
        x (list): Data.
        y (list): Labels.
        pid (list): Subject ids and starting times.
        time_diff (int): Time difference between samples.
    
    Returns:
        tuple: Batches of data (x, y, pid).
    """
    diffs = np.unique(np.diff(pid[:,1]), return_counts=True)
    most_frequent_diff = diffs[0][np.argmax(diffs[1])]

    indices = []
    time_diff = most_frequent_diff
    current_sub = None
    current_start_time = None
    current_indices = []
    for i in range(len(pid)):
        sub, start_time = pid[i][0], pid[i][1]
        if current_sub != sub or abs(start_time - current_start_time) != time_diff:
            if len(current_indices) > 0:
                indices.append(np.array(current_indices))
            current_indices = [i]
            current_sub = sub
            current_start_time = start_time
        else:
            current_indices.append(i)
            current_start_time = start_time

    x = [x[i] for i in indices]
    y = [y[i] for i in indices]
    pid = [pid[i] for i in indices]

    return x, y, pid
    
def discretize_hr(y: np.ndarray, hr_min: float, hr_max: float, n_bins: int = 64, sigma: float = 1.5) -> np.ndarray:
    """
    Discretize continuous heart rate values into a one-hot encoding.
    
    Parameters:
        y (numpy.ndarray): Continuous heart rate values.
        hr_min (float): Minimum heart rate value.
        hr_max (float): Maximum heart rate value.
        n_bins (int): Number of bins.
        sigma (float): Sigma value for Gaussian function.
    
    Returns:
        numpy.ndarray: One-hot encoded heart rate values.
    """
    def gaussian(x, mu, sig):
        if sig == 0:
            return (x == mu) * 1.0
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    hr_range = hr_max - hr_min
    y = np.clip(y,-sigma/hr_range*3, 1 + sigma/hr_range*3)

    bins = np.linspace(0, 1, n_bins-1)
    bin_values = np.concatenate([[bins[0]], (bins[1:] + bins[:-1])/2 , [bins[-1]]])
    y_discretized = np.array([gaussian(bin_values , x, sigma/hr_range) for x in y])
    y_discretized = y_discretized / y_discretized.sum(axis=1, keepdims=True)

    return y_discretized

def norm_hr(y: float, hr_min: float, hr_max: float) -> float:
    """
    Normalize continuous heart rate values.
    
    Parameters:
        y (float): Continuous heart rate value.
        hr_min (float): Minimum heart rate value.
        hr_max (float): Maximum heart rate value.
    
    Returns:
        float: Normalized heart rate value.
    """
    return (y - hr_min) / (hr_max - hr_min)

def load_pickle(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from a pickle file.
    
    Parameters:
        path (str): Path to the pickle file.
    
    Returns:
        tuple: Loaded data (x, y, pid, metrics).
    """
    if not os.path.exists(path):
        FileNotFoundError(f"File {path} does not exist")

    with open(path, 'rb') as f:
        data = pickle.load(f)

    if len(data) == 3:
        x_, y_, pid_ = data
        metrics_ = np.array([])
    elif len(data) == 4:
        x_, y_, pid_, metrics_ = data
    elif len(data) == 2:
        x_, pid_ = data
        y_ = np.array([])
        metrics_ = np.array([])
    else:
        raise ValueError(f"Invalid data shape {len(data)}")
    
    return x_, y_, pid_, metrics_

def resample_data(x: np.ndarray, fs: float, fs_new: float, cuda: int = -1) -> np.ndarray:
    """
    Resample data to a new frequency.
    
    Parameters:
        x (numpy.ndarray): Input data.
        fs (float): Original sampling frequency.
        fs_new (float): New sampling frequency.
        cuda (int): GPU device index.
    
    Returns:
        numpy.ndarray: Resampled data.
    """
    if cuda != -1:
        DEVICE = torch.device('cuda:' + str(cuda) if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = "cpu"
    resample = julius.ResampleFrac(fs, fs_new).to(DEVICE)
    x = np.stack([resample(torch.Tensor(x[:,:,i]).to(DEVICE).float()).cpu().numpy() for i in range(x.shape[2])], axis=-1)

    return x