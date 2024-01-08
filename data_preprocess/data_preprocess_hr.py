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

NUM_FEATURES = 1



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
    
def discretize_hr(y, n_bins:int=64):
    """
    Discretizes a continuous heartrate value into a one-hot encoding.
    Assumes that alle y values are in the range [0,1]
    Values outside this range will be put into the first or last bin.
    Outputs a shape (n_samples, n_bins) array.
    """
    bins = np.linspace(0, 1, n_bins-1)
    # digitize values to get discrete values
    digitized = np.digitize(y, bins).reshape(-1)
    # creates one-hot encoding of discrete values
    y_onehot = np.eye(n_bins)[digitized]


    return y_onehot

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
    
def load_data(data_path, split=0,  subsample=1.0, reconstruction=False):


    def load_pickle(path):
        with open(path, 'rb') as f:
            x,y,pid = cp.load(f)
        return x,y,pid
    
    def load_split(files, path):
        x, y, pid = [], [], []
        for file in files:
            x_, y_, pid_ = load_pickle(os.path.join(path, file))
            x.append(x_)
            y.append(y_)
            pid.append(pid_)
        if len(x) == 0:
            return np.array([]), np.array([]), np.array([])
            
        x = np.concatenate(x)
        y = np.concatenate(y)
        pid = np.concatenate(pid)
        return x,y,pid
    
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

    x_train, y_train, d_train = load_split(train, data_path)
    x_val, y_val, d_val = load_split(val, data_path)
    x_test, y_test, d_test = load_split(test, data_path)

    if reconstruction:
        y_train = load_reconstruction_signal(train, data_path)
        y_val = load_reconstruction_signal(val, data_path)
        y_test = load_reconstruction_signal(test, data_path)


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

def prep_hr(args, dataset=None, split=None, resampling_rate=1, subsample_rate=1.0, reconstruction=False):
    if dataset is None:
        dataset = args.dataset
    if split is None:
        split = args.split

    if dataset == 'max':
        data_path = config.data_dir_Max_processed
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split,  subsample=subsample_rate, reconstruction=reconstruction)
    elif dataset == 'apple':
        data_path = config.data_dir_Apple_processed
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split,  subsample=subsample_rate, reconstruction=reconstruction)
    elif dataset == 'm2sleep':
        data_path = config.data_dir_M2Sleep_processed
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split,  subsample=subsample_rate, reconstruction=reconstruction)
    elif dataset == 'capture24':
        data_path = config.data_dir_Capture24_processed
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data_no_labels(data_path=data_path, split=split,  subsample=subsample_rate, reconstruction=reconstruction)
    elif dataset == 'apple100':
        data_path = config.data_dir_Apple_processed_100hz
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, subsample=subsample_rate, reconstruction=reconstruction)
    elif dataset == 'm2sleep100':
        data_path = config.data_dir_M2Sleep_processed_100Hz
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split,  subsample=subsample_rate, reconstruction=reconstruction)
    
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

    if args.discretize_hr and not reconstruction:
        y_train = discretize_hr(y_train, n_bins=args.n_class)
        y_val = discretize_hr(y_val, n_bins=args.n_class)
        y_test = discretize_hr(y_test, n_bins=args.n_class)



    train_set = data_loader_hr(x_train, y_train, d_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers, pin_memory=True, shuffle=True)

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