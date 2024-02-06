'''
Data Pre-processing on HHAR dataset.

'''
import sys
import numpy as np
import pandas as pd
import sys
import os
from scipy.interpolate import interp1d
import torch
import pickle as cp
from torch.utils.data import Dataset, DataLoader
import config
import re
import json
from scipy.signal import resample
from scipy.interpolate import interp1d
from scipy.stats import rankdata

NUM_FEATURES = 1


def rank_metrics(metrics):
    assert len(metrics.shape) == 2
    # Takes as input an np array of shape (n_samples, n_metrics)
    # The metric should have low values for good data
    # Returns an array of shape (n_samples) with the indices of the best samples
    ranks = np.apply_along_axis(rankdata, 0, metrics, method="max")
    ranks = np.mean(ranks, axis=1)
    best_ind = np.argsort(ranks)
    return best_ind

class data_loader_hr(Dataset):
    def __init__(self, samples, labels, domains):

        self.samples = samples
        self.domains = domains
        self.labels = labels
    
    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain
    def __len__(self):
        return len(self.samples)


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
    

def preprocess():
    print('preprocessing the data...')
    print('preprocessing done!')

def subsample_by_index(x,y, d, random_indices, ratio):
    pids = np.unique(d[:,0])

    for pid in pids:
        indices = np.where(d[:,0] == pid)[0]
        indices = indices[random_indices[int(pid)][:int(len(indices)*ratio)]]
        if pid == pids[0]:
            indices_all = indices
        else:
            indices_all = np.concatenate((indices_all, indices))

    x = x[indices_all]
    y = y[indices_all]
    d = d[indices_all]
    return x,y,d
    
def load_random_indices(data_path):
    with open(os.path.join(data_path, "random_indices.pickle"), 'rb') as f:
        random_indices = cp.load(f)
    return random_indices
    
def load_data(data_path, split=0,  subsample=1.0, reconstruction=False, subsample_ranked_train=False, subsample_ranked_val=False):


    def load_pickle(path):
        with open(path, 'rb') as f:
            data = cp.load(f)
        return data
    
    def load_split(files, path):
        x, y, pid, mectrics = [], [], [], []
        for file in files:
            data_ = load_pickle(os.path.join(path, file))

            if len(data_) == 3:
                x_, y_, pid_ = data_
                metrics_ = np.array([])
            elif len(data_) == 4:
                x_, y_, pid_, metrics_ = data_
            elif len(data_) == 2:
                x_, pid_ = data_
                y_ = np.array([])
                metrics_ = np.array([])

            if len(x_) != 0:
                x.append(x_)
                y.append(y_)
                pid.append(pid_)
                mectrics.append(metrics_)

        if len(x) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        x = np.concatenate(x)
        y = np.concatenate(y)
        pid = np.concatenate(pid)
        mectrics = np.concatenate(mectrics)

        return x,y,pid, mectrics
    
    def load_reconstruction_signal(files, path):
        # Load reconstruction signal for autoencoder training

        y = []
        for file in files:
            file = file.replace('data', 'ecg')
            with open(os.path.join(path, file), 'rb') as f:
                y_ = cp.load(f)
            y.append(y_)
        if len(y) == 0:
            return np.array([])   

        # adds the channel dimension, should be 1
        y = np.concatenate(y)
        y = y.reshape(y.shape[0], y.shape[1], -1)

        return y
    

    split_path = os.path.join(data_path, "splits.json")
    with open(split_path, 'r') as f:
        splits = json.load(f)
    split_files = splits[str(split)]

    train = split_files["train"]
    test = split_files["test"]
    val = split_files["val"]

    x_train, y_train, d_train, metrics_train = load_split(train, data_path)
    x_val, y_val, d_val, metrics_val = load_split(val, data_path)
    x_test, y_test, d_test, metrics_test = load_split(test, data_path)

    if reconstruction:
        y_train = load_reconstruction_signal(train, data_path)
        y_val = load_reconstruction_signal(val, data_path)
        y_test = load_reconstruction_signal(test, data_path)

    if subsample_ranked_train != None and subsample_ranked_train != 0 and subsample_ranked_train != 1.0:
        best_ind_train = rank_metrics(metrics_train)
        num_values = int(len(x_train)*subsample_ranked_train)
        x_train = x_train[best_ind_train[:num_values]]
        y_train = y_train[best_ind_train[:num_values]]
        d_train = d_train[best_ind_train[:num_values]]

    if subsample_ranked_val != None and subsample_ranked_val != 0 and subsample_ranked_val != 1.0:
        best_ind_val = rank_metrics(metrics_val)
        num_values = int(len(x_val)*subsample_ranked_val)
        x_val = x_val[best_ind_val[:num_values]]
        y_val = y_val[best_ind_val[:num_values]]
        d_val = d_val[best_ind_val[:num_values]]

        best_ind_test = rank_metrics(metrics_test)
        num_values = int(len(x_test)*subsample_ranked_val)
        x_test = x_test[best_ind_test[:num_values]]
        y_test = y_test[best_ind_test[:num_values]]
        d_test = d_test[best_ind_test[:num_values]]


    d_train = d_train.astype(float)
    d_val = d_val.astype(float)
    d_test = d_test.astype(float)

    if subsample != 1.0:

        random_indices = load_random_indices(data_path)
        x_train, y_train, d_train = subsample_by_index(x_train, y_train, d_train, random_indices, subsample)

    return x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test 

def load_data_no_labels(data_path, split:int=0, subsample=1.0, reconstruction=False):
    # Load data without labels
    # Used for capture24 dataset
    # Substitutes labels with zeros

    if reconstruction:
        raise NotImplementedError("Reconstruction not possible for capture24 dataset")


    def load_pickle(path):
        with open(path, 'rb') as f:
            x,pid = cp.load(f)
        return x,pid
    
    def load_split(files, path):
        x,  pid = [], []
        for file in files:
            x_, pid_ = load_pickle(os.path.join(path, file))
            if len(x_) == 0:
                continue
            x.append(x_)
            pid.append(pid_)
        x = np.concatenate(x)
        pid = np.concatenate(pid)
        return x,pid


    split_path = os.path.join(data_path, "splits.json")
    with open(split_path, 'r') as f:
        splits = json.load(f)
    split = splits[str(split)]

    train = split["train"]
    test = split["test"]
    val = split["val"]

    x_train, d_train = load_split(train, data_path)
    x_val, d_val = load_split(val, data_path)
    x_test, d_test = load_split(test, data_path)
                

    y_train = np.ones((x_train.shape[0],1))
    y_val = np.ones((x_val.shape[0],1))
    y_test = np.ones((x_test.shape[0],1))

    
    if subsample != 1.0:

        random_indices = load_random_indices(data_path)
        x_train, y_train, d_train = subsample_by_index(x_train, y_train, d_train, random_indices, subsample)


    return x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test 

def prep_hr(args, dataset=None, split=None, resampling_rate=1, subsample_rate=1.0, reconstruction=False, sample_sequences:bool=False, discrete_hr:bool=False, sigma=None):
    if dataset is None:
        dataset = args.dataset
    if split is None:
        split = args.split

    if dataset == 'max':
        data_path = config.data_dir_Max_processed
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split,  subsample=subsample_rate, reconstruction=reconstruction, subsample_ranked_train=args.subsample_ranked_train, subsample_ranked_val=args.subsample_ranked_val)
    elif dataset == 'apple':
        data_path = config.data_dir_Apple_processed
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split,  subsample=subsample_rate, reconstruction=reconstruction, subsample_ranked_train=args.subsample_ranked_train, subsample_ranked_val=args.subsample_ranked_val)
    elif dataset == 'm2sleep':
        data_path = config.data_dir_M2Sleep_processed
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split,  subsample=subsample_rate, reconstruction=reconstruction, subsample_ranked_train=args.subsample_ranked_train, subsample_ranked_val=args.subsample_ranked_val)
    elif dataset == 'capture24':
        data_path = config.data_dir_Capture24_processed
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data_no_labels(data_path=data_path, split=split,  subsample=subsample_rate, reconstruction=reconstruction)
    elif dataset == 'apple100':
        # take the dataset, that has a couple of metrics for each window
        # the dataset with metrics is computed a little differently, so there might be differences
        if (args.subsample_ranked_val == None or args.subsample_ranked_val == 0.0) and (args.subsample_ranked_train == None or args.subsample_ranked_train == 0.0):
            data_path = config.data_dir_Apple_processed_100hz
        else:
            data_path = config.data_dir_Apple_processed_100hz_wmetrics
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, subsample=subsample_rate, reconstruction=reconstruction, subsample_ranked_train=args.subsample_ranked_train, subsample_ranked_val=args.subsample_ranked_val)
    elif dataset == 'm2sleep100':
        data_path = config.data_dir_M2Sleep_processed_100Hz
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split,  subsample=subsample_rate, reconstruction=reconstruction, subsample_ranked_train=args.subsample_ranked_train, subsample_ranked_val=args.subsample_ranked_val)
    elif dataset == "parkinson100":
        # take the dataset that has a couple of metrics for each window
        # will change results
        if (args.subsample_ranked_val == None or args.subsample_ranked_val == 0.0) and (args.subsample_ranked_train == None or args.subsample_ranked_train == 0.0):
            data_path = config.data_dir_Parkinson_processed_100Hz
        else:
            data_path = config.data_dir_Parkinson_processed_100Hz_wmetrics
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split,  subsample=subsample_rate, reconstruction=reconstruction, subsample_ranked_train=args.subsample_ranked_train, subsample_ranked_val=args.subsample_ranked_val)

    assert x_train.shape[0] == y_train.shape[0] == d_train.shape[0]

    if resampling_rate != 1:
        x_train, y_train, d_train = resample_data(x_train, y_train, d_train, resampling_rate)
        x_val, y_val, d_val = resample_data(x_val, y_val, d_val, resampling_rate)
        x_test, y_test, d_test = resample_data(x_test, y_test, d_test, resampling_rate)
        
    if args.normalize:
        x_train = normalize_samples(x_train)
        x_val = normalize_samples(x_val)
        x_test = normalize_samples(x_test)
    
    if args.normalize and reconstruction:
        y_train = normalize_samples(y_train)
        y_val = normalize_samples(y_val)
        y_test = normalize_samples(y_test)

    if not reconstruction:
        y_train = norm_hr(y_train, args.hr_min, args.hr_max)
        y_val = norm_hr(y_val, args.hr_min, args.hr_max)
        y_test = norm_hr(y_test, args.hr_min, args.hr_max)

    if discrete_hr and not reconstruction:
        sigma = args.label_sigma if sigma is None else sigma
        y_train = discretize_hr(y_train, hr_min = args.hr_min, hr_max = args.hr_max, n_bins=args.n_prob_class, sigma=sigma)
        y_val = discretize_hr(y_val, hr_min = args.hr_min, hr_max = args.hr_max, n_bins=args.n_prob_class, sigma=sigma)
        y_test = discretize_hr(y_test, hr_min = args.hr_min, hr_max = args.hr_max, n_bins=args.n_prob_class, sigma=sigma)

    if sample_sequences:
        x_train, y_train, d_train = generate_batch_sequences(x_train, y_train, d_train)
        x_val, y_val, d_val = generate_batch_sequences(x_val, y_val, d_val)
        x_test, y_test, d_test = generate_batch_sequences(x_test, y_test, d_test)

        train_set = data_loader_hr(x_train, y_train, d_train)
        train_loader = DataLoader(train_set, batch_size=None, drop_last=False, batch_sampler=None, num_workers=args.num_workers, pin_memory=True, shuffle=True)

        test_set = data_loader_hr(x_test, y_test, d_test)
        test_loader = DataLoader(test_set, batch_size=None, batch_sampler=None, shuffle=False)

        val_set = data_loader_hr(x_val, y_val, d_val)
        val_loader = DataLoader(val_set, batch_size=None, batch_sampler=None, shuffle=False)
        
    else:
        train_set = data_loader_hr(x_train, y_train, d_train)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers, pin_memory=True, shuffle=True)

        test_set = data_loader_hr(x_test, y_test, d_test)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        val_set = data_loader_hr(x_val, y_val, d_val)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")

    return train_loader, val_loader, test_loader

def resample_data(x, y, d, resampling_rate):
    x_resampled = resample(x, int(x.shape[0]/resampling_rate), axis=1)
    time_vector = np.arange(0, x.shape[1], 1)
    time_vector_resampled = np.arange(0, x.shape[1], 1/resampling_rate)
    f_y = interp1d(time_vector, y, axis=1)
    f_d = interp1d(time_vector, d, axis=1)
    y_resampled = f_y(time_vector_resampled, kind='linear', copy=False, fill_value="extrapolate")
    d_resampled = f_d(time_vector_resampled, kind='previous', copy=False, fill_value="extrapolate")
    return x_resampled, y_resampled, d_resampled



def test_prep_hr():
    # Define mock arguments
    class Args:
        data_path = config.data_dir_Max_processed
        split = 0
        batch_size = 32  # Adjust the batch size as needed

    # Call the prep_hr function with mock arguments
    train_loader, val_loader, test_loader = prep_hr(Args)

    # Print information about the loaded data
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")

    # Print the first batch in each loader for inspection
    print("\nFirst batch in train_loader:")
    for i, (samples, labels, domains) in enumerate(train_loader):
        print(f"Sample shape: {samples.shape}, Label shape: {labels.shape}, Domain shape: {domains.shape}")
        break  # Print only the first batch

    print("\nFirst batch in val_loader:")
    for i, (samples, labels, domains) in enumerate(val_loader):
        print(f"Sample shape: {samples.shape}, Label shape: {labels.shape}, Domain shape: {domains.shape}")
        break  # Print only the first batch

    print("\nFirst batch in test_loader:")
    for i, (samples, labels, domains) in enumerate(test_loader):
        print(f"Sample shape: {samples.shape}, Label shape: {labels.shape}, Domain shape: {domains.shape}")
        break  # Print only the first batch

    print("Testing prep_hr complete.")

# Call the test function
#test_prep_hr()