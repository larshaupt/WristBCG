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
import julius

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
    # Subsamples the data by the given indices
    # The indices are given in a dictionary, where the key is the subject id and the value is an array of indices
    # The ratio is the fraction of samples to keep
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
    
def load_data(data_path, split=0,  subsample=1.0, args=None, reconstruction=False, take_every_nth_train=1, take_every_nth_test=1):

    if args is not None: # taking the parameters from the args
        subsample_ranked_train = args.subsample_ranked_train
        subsample_ranked_val = args.subsample_ranked_val
        data_thr_avg = args.data_thr_avg
        data_thr_max = args.data_thr_max
        data_thr_angle = args.data_thr_angle
        data_thr_hr = args.data_thr_hr

    else: # if the args are not given, then the parameters are set to None, i.e. the data is not subsampled
        subsample_ranked_train = None
        subsample_ranked_val = None
        data_thr_avg = None
        data_thr_max = None
        data_thr_angle = None
        data_thr_hr = None


    def load_pickle(path):

        if not os.path.exists(path):
            FileNotFoundError(f"File {path} does not exist")

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
    
    def filter_by_metric(metrics, thr_avg, thr_max, thr_angle, thr_hr):
        mask = np.ones_like(metrics[:,0], dtype=bool)
        if thr_avg != None and thr_avg != 0:
            mask = mask* (metrics[:,0] < thr_avg)
        if thr_max != None and thr_max != 0:
            mask = mask* (metrics[:,1] < thr_max)
        if thr_angle != None and thr_angle != 0:
            mask = mask* (metrics[:,2] < thr_angle)
        if thr_hr != None and thr_hr != 0:
            mask = mask* (metrics[:,3] < thr_hr)
        return mask
    

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

    if take_every_nth_train != 1:
        x_train = x_train[::take_every_nth_train]
        y_train = y_train[::take_every_nth_train]
        d_train = d_train[::take_every_nth_train]
        metrics_train = metrics_train[::take_every_nth_train]

    if take_every_nth_test != 1:
        x_test = x_test[::take_every_nth_test]
        y_test = y_test[::take_every_nth_test]
        d_test = d_test[::take_every_nth_test]
        metrics_test = metrics_test[::take_every_nth_test]

        x_val = x_val[::take_every_nth_test]
        y_val = y_val[::take_every_nth_test]
        d_val = d_val[::take_every_nth_test]
        metrics_val = metrics_val[::take_every_nth_test]


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


    mask_train = filter_by_metric(metrics_train, data_thr_avg, data_thr_max, data_thr_angle, data_thr_hr)
    mask_val = filter_by_metric(metrics_val, data_thr_avg, data_thr_max, data_thr_angle, data_thr_hr)
    mask_test = filter_by_metric(metrics_test, data_thr_avg, data_thr_max, data_thr_angle, data_thr_hr)

    print(f"Discarded {mask_train.sum()/len(mask_train)*100:.2f}% of training data")
    print(f"Discarded {mask_val.sum()/len(mask_val)*100:.2f}% of validation data")
    print(f"Discarded {mask_test.sum()/len(mask_test)*100:.2f}% of test data")

    x_train = x_train[mask_train]
    y_train = y_train[mask_train]
    d_train = d_train[mask_train]

    x_val = x_val[mask_val]
    y_val = y_val[mask_val]
    d_val = d_val[mask_val]

    x_test = x_test[mask_test]
    y_test = y_test[mask_test]
    d_test = d_test[mask_test]

    d_train = d_train.astype(float)
    d_val = d_val.astype(float)
    d_test = d_test.astype(float)

    if subsample != 1.0:

        subsample_indices = load_random_indices(data_path)
        x_train, y_train, d_train = subsample_by_index(x_train, y_train, d_train, subsample_indices, subsample)

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

        subsample_indices = load_random_indices(data_path)

        x_train, y_train, d_train = subsample_by_index(x_train, y_train, d_train, subsample_indices, subsample)


    return x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test 

def prep_hr(args, dataset=None, split=None, subsample_rate=1.0, reconstruction=False, sample_sequences:bool=False, discrete_hr:bool=False, sigma=None):
    if dataset is None:
        dataset = args.dataset
    if split is None:
        split = args.split

    if dataset == 'max':
        sampling_rate = 100
        if args.step_size == 8 and args.window_size == 10:
            data_path = config.data_dir_Max_processed
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, args=args, subsample=subsample_rate, reconstruction=reconstruction)
        else:
            raise ValueError(f"Invalid step size {args.step_size} and window size {args.window_size} for dataset {dataset}")
        
    elif dataset == 'apple':
        sampling_rate = 50
        if args.step_size == 8 and args.window_size == 10:
            data_path = config.data_dir_Apple_processed
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, args=args, subsample=subsample_rate, reconstruction=reconstruction)
        else:
            raise ValueError(f"Invalid step size {args.step_size} and window size {args.window_size} for dataset {dataset}")
    
    elif dataset == 'm2sleep':
        sampling_rate = 32
        if args.step_size == 8 and args.window_size == 10:
            data_path = config.data_dir_M2Sleep_processed
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, args=args,  subsample=subsample_rate, reconstruction=reconstruction)
        else:
            raise ValueError(f"Invalid step size {args.step_size} and window size {args.window_size} for dataset {dataset}")
    
    elif dataset == 'capture24':
        if args.sampling_rate == 100 or args.sampling_rate == 0:
            sampling_rate = 100
            if args.step_size == 8 and args.window_size == 10:
                data_path = config.data_dir_Capture24_processed
                x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data_no_labels(data_path=data_path, split=split,  subsample=subsample_rate, reconstruction=reconstruction)
            
            else:
                raise ValueError(f"Invalid step size {args.step_size} and window size {args.window_size} for dataset {dataset}")
        elif args.sampling_rate == 125:
            sampling_rate = 125
            if args.step_size == 6 and args.window_size == 8:
                data_path = config.data_dir_Capture24_processed_125Hz_8w
                x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data_no_labels(data_path=data_path, split=split,  subsample=subsample_rate, reconstruction=reconstruction)
            else:
                raise ValueError(f"Invalid step size {args.step_size} and window size {args.window_size} for dataset {dataset}")
        else:
            raise ValueError(f"Invalid sampling rate {args.sampling_rate} for dataset {dataset}")
    
    elif dataset == 'apple100':
        sampling_rate = 100
        # take the dataset, that has a couple of metrics for each window
        # the dataset with metrics is computed a little differently, so there might be differences
        if (args.subsample_ranked_val == None or args.subsample_ranked_val == 0.0) and (args.subsample_ranked_train == None or args.subsample_ranked_train == 0.0):
            if args.step_size == 8 and args.window_size == 10:
                data_path = config.data_dir_Apple_processed_100hz
            elif args.step_size == 4 and args.window_size == 5:
                data_path = config.data_dir_Apple_processed_100hz_5w_4s
            elif args.step_size == 4 and args.window_size == 8:
                data_path = config.data_dir_Apple_processed_100hz_8w_4s
            elif args.step_size == 4 and args.window_size == 10:
                data_path = config.data_dir_Apple_processed_100hz_10w_4s
            elif args.step_size == 4 and args.window_size == 30:
                data_path = config.data_dir_Apple_processed_100hz_30w_4s
            elif args.step_size == 4 and args.window_size == 60:
                data_path = config.data_dir_Apple_processed_100hz_60w_4s
            elif args.step_size == 1 and args.window_size == 10:
                data_path = config.data_dir_Apple_processed_100hz_10w_1s
            else:
                raise ValueError(f"Invalid step size {args.step_size} for dataset {dataset}")
        else: # load the dataset with metrics
            if args.step_size == 8 and args.window_size == 10:
                data_path = config.data_dir_Apple_processed_100hz_wmetrics
            else:
                raise ValueError(f"Invalid step size {args.step_size} for dataset {dataset}")
            
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, args=args, subsample=subsample_rate, reconstruction=reconstruction, take_every_nth_train=args.take_every_nth_train, take_every_nth_test=args.take_every_nth_test)
    
    elif dataset == 'appleall':
        sampling_rate = 100
        if args.step_size == 8 and args.window_size == 10:
            data_path = config.data_dir_Apple_processed_all
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, args=args,  subsample=subsample_rate, reconstruction=reconstruction)
        else:
            raise ValueError(f"Invalid step size {args.step_size} and window size {args.window_size} for dataset {dataset}")
    
    elif dataset == 'm2sleep100':
        sampling_rate = 100
        if args.step_size == 8 and args.window_size == 10:
            data_path = config.data_dir_M2Sleep_processed_100Hz
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, args=args,  subsample=subsample_rate, reconstruction=reconstruction)
        else:
            raise ValueError(f"Invalid step size {args.step_size} and window size {args.window_size} for dataset {dataset}")
    
    elif dataset == "parkinson100":
        sampling_rate = 100
        if args.step_size == 8 and args.window_size == 10:
            # take the dataset that has a couple of metrics for each window
            # will change results
            if (args.subsample_ranked_val == None or args.subsample_ranked_val == 0.0) and (args.subsample_ranked_train == None or args.subsample_ranked_train == 0.0):
                data_path = config.data_dir_Parkinson_processed_100Hz
            else:
                data_path = config.data_dir_Parkinson_processed_100Hz_wmetrics
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, args=args,  subsample=subsample_rate, reconstruction=reconstruction)
        else:
            raise ValueError(f"Invalid step size {args.step_size} and window size {args.window_size} for dataset {dataset}")

    elif dataset == "IEEE":
        sampling_rate = 125
        if args.window_size == 8:
            data_path = config.data_dir_IEEE_processed
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, args=args,  subsample=subsample_rate, reconstruction=reconstruction)
        else:
            raise ValueError(f"Invalid step size {args.step_size} and window size {args.window_size} for dataset {dataset}")
    
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    assert x_train.shape[0] == y_train.shape[0] == d_train.shape[0]

    if args.sampling_rate != sampling_rate and args.sampling_rate != 0:
        x_train, y_train, d_train = resample_data(x_train, y_train, d_train, sampling_rate, args.sampling_rate)
        x_val, y_val, d_val = resample_data(x_val, y_val, d_val, sampling_rate, args.sampling_rate)
        x_test, y_test, d_test = resample_data(x_test, y_test, d_test, sampling_rate, args.sampling_rate)
        
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

def resample_data(x, y, d, fs, fs_new):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resample = julius.ResampleFrac(fs, fs_new).to(DEVICE)

    time_vector = np.arange(0, x.shape[1], 1)
    time_vector_resampled = np.arange(0, x.shape[1], fs/fs_new)
    f_d = interp1d(time_vector, d, axis=1)
    d = f_d(time_vector_resampled, kind='previous', copy=False, fill_value="extrapolate")

    x = resample(torch.Tensor(x).to(DEVICE).float()).cpu().numpy()
    y = resample(torch.Tensor(y).to(DEVICE).float()).cpu().numpy()
    d = resample(torch.Tensor(d).to(DEVICE).float()).cpu().numpy()


    return x, y, d



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