import numpy as np
from scipy.stats import rankdata
import os
import torch
import julius



def rank_metrics(metrics):
    assert len(metrics.shape) == 2
    # Takes as input an np array of shape (n_samples, n_metrics)
    # The metric should have low values for good data
    # Returns an array of shape (n_samples) with the indices of the best samples
    ranks = np.apply_along_axis(rankdata, 0, metrics, method="max")
    ranks = np.mean(ranks, axis=1)
    best_ind = np.argsort(ranks)
    return best_ind



def downsampling(data_t, data_x, data_y, freq):
    """Recordings are downsamplied to 50Hz
    
    :param data_t: numpy integer array
        time array
    :param data_x: numpy integer array
        sensor recordings
    :param data_y: numpy integer array
        labels
    :return: numpy integer array
        Downsampled input
    """
    idx = np.arange(0, data_t.shape[0], int(freq/50))

    return data_t[idx], data_x[idx], data_y[idx]

def normalize_samples(data):
    if len(data) != 0:
        return (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    else:
        return data
    
def generate_batch_sequences(x, y, pid, time_diff=8):

    # Generates batches of the given data
    # Always puts consecutive samples from the same subject into the same batch
    # Uses the subject id and starting time from pid for batching
    # Because we don't know what's the general time difference between samples from the same subject, we infer it from the data

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

    
    
def discretize_hr(y, hr_min, hr_max, n_bins:int=64, sigma=1.5):
    """
    Discretizes a continuous heartrate value into a one-hot encoding.
    Assumes that alle y values are in the range [0,1]
    Values outside this range will be put into the first or last bin.
    Outputs a shape (n_samples, n_bins) array.
    """
    def gaussian(x, mu, sig):
        if sig == 0:
            return (x == mu) * 1.0
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    hr_range = hr_max - hr_min
    
    # clip values outside the range plus 3* sigma
    y = np.clip(y,-sigma/hr_range*3, 1 + sigma/hr_range*3)

    bins = np.linspace(0, 1, n_bins-1)
    bin_values = np.concatenate([[bins[0]], (bins[1:] + bins[:-1])/2 , [bins[-1]]])

    # creates discrete distributions of hr values
    y_discretized = np.array([gaussian(bin_values , x, sigma/hr_range) for x in y])

    y_discretized = y_discretized / y_discretized.sum(axis=1, keepdims=True)


    return y_discretized

def norm_hr(y, hr_min, hr_max):
    """
    Normalizes a continuous heartrate value into the range [0,1]
    """
    return (y - hr_min) / (hr_max - hr_min)


def load_pickle(path):

    if not os.path.exists(path):
        FileNotFoundError(f"File {path} does not exist")

    with open(path, 'rb') as f:
        data = cp.load(f)

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


def resample_data(x, fs, fs_new, cuda=-1):
    if cuda != -1:
        DEVICE = torch.device('cuda:' + str(cuda) if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = "cpu"
    resample = julius.ResampleFrac(fs, fs_new).to(DEVICE)

    # needs to apply this on all channels seperately
    x = np.stack([resample(torch.Tensor(x[:,:,i]).to(DEVICE).float()).cpu().numpy() for i in range(x.shape[2])], axis=-1)

    return x