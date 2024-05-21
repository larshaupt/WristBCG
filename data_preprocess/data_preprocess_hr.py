'''
Data Pre-processing on HHAR dataset.

'''

import numpy as np
import pandas as pd
import os
import torch
import pickle as cp
from torch.utils.data import Dataset, DataLoader
import config
import json
from scipy.signal import periodogram, butter, sosfiltfilt
from .data_preprocess_utils import *



class data_loader_hr(Dataset):
    def __init__(self, samples, labels, domains, split = None, partition=None, dataset=None):

        self.samples = samples
        self.domains = domains
        self.labels = labels
        self.split = split
        self.partition = partition
        self.dataset = dataset
    
    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain
    def __len__(self):
        return len(self.samples)
    
def load_data(data_path:str, 
              split:int=0,  
              params=None, 
              reconstruction:bool=False, 
              take_every_nth_train:int=1, 
              take_every_nth_test:int=1, 
              sampling_rate:int = None):

    if sampling_rate == None or sampling_rate == 0:
            sampling_rate = 100 # since we are normalizing the spectrogram later anyway, we can use any sampling rate

    if params is not None: # taking the parameters from the params
        data_thr_avg = params.data_thr_avg
        data_thr_max = params.data_thr_max
        data_thr_angle = params.data_thr_angle
        data_thr_hr = params.data_thr_hr
        bandpass_freq_min = params.bandpass_freq_min
        bandpass_freq_max = params.bandpass_freq_max

    else: # if the params are not given, then the parameters are set to None, i.e. the data is not subsampled
        data_thr_avg = None
        data_thr_max = None
        data_thr_angle = None
        data_thr_hr = None
        bandpass_freq_min = None
        bandpass_freq_max = None

    def load_split(files:list[str], path:str):
        # Load data and split by subject
        # Files should contain the filenames of the data files in the given split
        x, y, pid, mectrics = [], [], [], []
        for file in files:
            x_, y_, pid_, metrics_ = load_pickle(os.path.join(path, file))

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

        return x, y, pid, mectrics
    
    def load_split_by_time(files:dict[str, list], path:str):
        # Load data and split by time
        # Files should contain a dict with the filenames and the time intervals for each file

        x, y, pid, mectrics = [], [], [], []
        for filename, times in files.items():
            x_, y_, pid_, metrics_ = load_pickle(os.path.join(path, filename))

            if len(x_) == 0:
                continue

            timestamps = [el[1] for el in pid_]
            times = np.array(times).reshape(-1,2)

            for (start_time, end_time) in times:
                start_index = np.searchsorted(timestamps, start_time)
                end_index = np.searchsorted(timestamps, end_time)
                x_ = x_[start_index:end_index]
                y_ = y_[start_index:end_index]
                pid_ = pid_[start_index:end_index]
                metrics_ = metrics_[start_index:end_index]

                if len(x_) == 0:
                    continue

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

        return x, y, pid, mectrics
    
    def load_reconstruction_signal(files:list[str], path:str):
        # Load reconstruction signal for autoencoder training, usually the ECG signal

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
    
    def filter_by_metric(metrics, data:list, thr_avg:float, thr_max:float, thr_angle:float, thr_hr:float, print_discarded:bool=False):
        # filter each data point by the metrics and the given thresholds

        if len(metrics) == 0:
            print("No metrics found, returning all data")
            return data
        mask = np.ones_like(metrics[:,0], dtype=bool)
        if thr_avg != None and thr_avg != 0:
            mask = mask* (metrics[:,0] < thr_avg)
        if thr_max != None and thr_max != 0:
            mask = mask* (metrics[:,1] < thr_max)
        if thr_angle != None and thr_angle != 0:
            mask = mask* (metrics[:,2] < thr_angle)
        if thr_hr != None and thr_hr != 0:
            mask = mask* (metrics[:,3] > thr_hr)

        if print_discarded:
            print(f"Discarded {100 - mask.sum()/len(mask)*100:.2f}% of data by thresholding")

        data = [d[mask] for d in data]
        return data

    if params.split_by == "time": 
        # includes all subjects in all splits
        # takes the last 20% for testing
        # and the rest for training, with a 5-fold cross validation scheme
        split_path = os.path.join(data_path, "splits_by_time.json")
        with open(split_path, 'r') as f:
            splits = json.load(f)
        split_files = splits[str(split)]

        train_split = split_files["train"]
        val_split = split_files["val"]
        test_split = split_files["test"]

        x_train, y_train, d_train, metrics_train = load_split_by_time(train_split, data_path)
        x_val, y_val, d_val, metrics_val = load_split_by_time(val_split, data_path)
        x_test, y_test, d_test, metrics_test = load_split_by_time(test_split, data_path)


    else: #params.split_by == "subject"
        split_path = os.path.join(data_path, "splits.json")
        with open(split_path, 'r') as f:
            splits = json.load(f)
        split_files = splits[str(split)]

        train_split = split_files["train"]
        val_split = split_files["val"]
        test_split = split_files["test"]

        x_train, y_train, d_train, metrics_train = load_split(train_split, data_path)
        x_val, y_val, d_val, metrics_val = load_split(val_split, data_path)
        x_test, y_test, d_test, metrics_test = load_split(test_split, data_path)

    if reconstruction:
        y_train = load_reconstruction_signal(train_split, data_path)
        y_val = load_reconstruction_signal(val_split, data_path)
        y_test = load_reconstruction_signal(test_split, data_path)


    if take_every_nth_train != 1:
        x_train = x_train[take_every_nth_train//2::take_every_nth_train]
        y_train = y_train[take_every_nth_train//2::take_every_nth_train]
        d_train = d_train[take_every_nth_train//2::take_every_nth_train]
        metrics_train = metrics_train[take_every_nth_train//2::take_every_nth_train]

    if take_every_nth_test != 1:
        x_test = x_test[take_every_nth_test//2::take_every_nth_test]
        y_test = y_test[take_every_nth_test//2::take_every_nth_test]
        d_test = d_test[take_every_nth_test//2::take_every_nth_test]
        metrics_test = metrics_test[take_every_nth_test//2::take_every_nth_test]

        x_val = x_val[take_every_nth_test//2::take_every_nth_test]
        y_val = y_val[take_every_nth_test//2::take_every_nth_test]
        d_val = d_val[take_every_nth_test//2::take_every_nth_test]
        metrics_val = metrics_val[take_every_nth_test//2::take_every_nth_test]

    
    (x_train, y_train, d_train) = filter_by_metric(metrics_train, [x_train, y_train, d_train], data_thr_avg, data_thr_max, data_thr_angle, data_thr_hr, print_discarded=True)
    (x_val, y_val, d_val) = filter_by_metric(metrics_val, [x_val, y_val, d_val], data_thr_avg, data_thr_max, data_thr_angle, data_thr_hr, print_discarded=False)
    (x_test, y_test, d_test) = filter_by_metric(metrics_test, [x_test, y_test, d_test], data_thr_avg, data_thr_max, data_thr_angle, data_thr_hr, print_discarded=False)


    if bandpass_freq_min not in [None, 0.1,0] or bandpass_freq_max not in [None, 18, 0]:
        # Caution: signal is by default already filtered to 0.1-18 Hz, so this is only for additional filtering

        if bandpass_freq_min in [None, 0.1,0]:
            sos = butter(4, bandpass_freq_max, btype='high', fs=sampling_rate, output='sos')
        elif bandpass_freq_max in [None, 18, 0]:
            sos = butter(4, bandpass_freq_min, btype='low', fs=sampling_rate, output='sos')
        else:
            sos = butter(4, [bandpass_freq_min, bandpass_freq_max], btype='band', fs=sampling_rate, output='sos')
        x_train = np.apply_along_axis(lambda x: sosfiltfilt(sos, x), 1, x_train)
        x_val = np.apply_along_axis(lambda x: sosfiltfilt(sos, x), 1, x_val)
        x_test = np.apply_along_axis(lambda x: sosfiltfilt(sos, x), 1, x_test)


    if params.add_frequency: # computes frequency on top of the signal and stack the two components

        freq_func = lambda x: periodogram(x, fs=sampling_rate, axis=0, scaling='density', nfft=(x.shape[0] - 1)*2)[1]
        x_train = np.concatenate([x_train, np.apply_along_axis(freq_func, 0, x_train)], axis=-1)
        x_val = np.concatenate([x_val, np.apply_along_axis(freq_func, 0, x_val)], axis=-1)
        x_test = np.concatenate([x_test, np.apply_along_axis(freq_func, 0, x_test)], axis=-1)



    d_train = d_train.astype(float)
    d_val = d_val.astype(float)
    d_test = d_test.astype(float)

    return x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test 

def load_data_no_labels(data_path:str, 
                        split:int = 0, 
                        reconstruction:bool = False,
                        take_every_nth_train:int = 1, 
                        take_every_nth_test:int = 1
                        ):


    # Load data without labels
    # Used for capture24 dataset
    # Substitutes labels with zeros

    if reconstruction:
        raise NotImplementedError(f"Reconstruction not possible for dataset {data_path}")

    
    def load_split(files, path):
        # loads the data from the files in the given split
        x,  pid = [], []
        for file in files:
            x_, _, pid_, _ = load_pickle(os.path.join(path, file))
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
                

    if take_every_nth_train != 1:
        x_train = x_train[take_every_nth_train//2::take_every_nth_train]
        d_train = d_train[take_every_nth_train//2::take_every_nth_train]

    if take_every_nth_test != 1:
        x_test = x_test[take_every_nth_test//2::take_every_nth_test]
        d_test = d_test[take_every_nth_test//2::take_every_nth_test]

        x_val = x_val[take_every_nth_test//2::take_every_nth_test]
        d_val = d_val[take_every_nth_test//2::take_every_nth_test]

        
    y_train = np.ones((x_train.shape[0],1))
    y_val = np.ones((x_val.shape[0],1))
    y_test = np.ones((x_test.shape[0],1))


    return x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test 

def prep_hr(params, dataset=None, split=None, reconstruction=False, sample_sequences:bool=False, discrete_hr:bool=False, sigma=None):


    if dataset is None:
        dataset = params.dataset
    if split is None:
        split = params.split

    #####################################################################################
    ### Change this section to load your own dataset ####################################
    #####################################################################################

    if dataset == 'max':
        sampling_rate = 100
        if params.step_size == 8 and params.window_size == 10:
            data_path = config.data_dir_Max_processed
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, params=params,  reconstruction=reconstruction, sampling_rate=sampling_rate)
        else:
            raise ValueError(f"Invalid step size {params.step_size} and window size {params.window_size} for dataset {dataset}")
    
    elif dataset == 'max_v2':
        sampling_rate = 100
        if params.step_size == 8 and params.window_size == 10:
            data_path = config.data_dir_Max_processed_v2
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, params=params,  reconstruction=reconstruction, sampling_rate=sampling_rate)
        else:
            raise ValueError(f"Invalid step size {params.step_size} and window size {params.window_size} for dataset {dataset}")
        
    elif dataset == 'max_hrv':
        sampling_rate = 100
        if params.step_size == 15 and params.window_size == 60:
            data_path = config.data_dir_Max_processed_hrv
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, params=params,  reconstruction=reconstruction, sampling_rate=sampling_rate)
        else:
            raise ValueError(f"Invalid step size {params.step_size} and window size {params.window_size} for dataset {dataset}")
        

    elif dataset == 'apple':
        sampling_rate = 50
        if params.step_size == 8 and params.window_size == 10:
            data_path = config.data_dir_Apple_processed
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, params=params,  reconstruction=reconstruction, sampling_rate=sampling_rate)
        else:
            raise ValueError(f"Invalid step size {params.step_size} and window size {params.window_size} for dataset {dataset}")
    
    elif dataset == 'm2sleep':
        sampling_rate = 32
        if params.step_size == 8 and params.window_size == 10:
            data_path = config.data_dir_M2Sleep_processed
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, params=params,   reconstruction=reconstruction, sampling_rate=sampling_rate)
        else:
            raise ValueError(f"Invalid step size {params.step_size} and window size {params.window_size} for dataset {dataset}")
    
    elif dataset == 'capture24':
        if params.sampling_rate == 100 or params.sampling_rate == 0:
            sampling_rate = 100
            if params.step_size == 8 and params.window_size == 10:
                data_path = config.data_dir_Capture24_processed
                breakpoint()
                x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data_no_labels(data_path=data_path, split=split,   reconstruction=reconstruction)
            
            else:
                raise ValueError(f"Invalid step size {params.step_size} and window size {params.window_size} for dataset {dataset}")
        elif params.sampling_rate == 125:
            sampling_rate = 125
            if params.step_size == 6 and params.window_size == 8:
                data_path = config.data_dir_Capture24_processed_125Hz_8w
                x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data_no_labels(data_path=data_path, split=split,   reconstruction=reconstruction)
            else:
                raise ValueError(f"Invalid step size {params.step_size} and window size {params.window_size} for dataset {dataset}")
        else:
            raise ValueError(f"Invalid sampling rate {params.sampling_rate} for dataset {dataset}")
    elif dataset == 'capture24all':
        if (params.sampling_rate == 100 or params.sampling_rate == 0) and params.step_size == 8 and params.window_size == 10:
            sampling_rate = 100

            data_path = config.data_dir_Capture24_processed_all
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data_no_labels(data_path=data_path, split=split,   reconstruction=reconstruction)
        else:
            raise ValueError(f"Invalid step size {params.step_size} and window size {params.window_size} or sampling rate {params.sampling_rate} for dataset {dataset}")

    elif dataset == 'apple100':
        sampling_rate = 100
        # take the dataset, that has a couple of metrics for each window
        # the dataset with metrics is computed a little differently, so there might be differences
       
        if params.step_size == 8 and params.window_size == 10:
            data_path = params.data_dir_Apple_processed_100hz
        elif params.step_size == 4 and params.window_size == 5:
            data_path = params.data_dir_Apple_processed_100hz_5w_4s
        elif params.step_size == 4 and params.window_size == 8:
            data_path = params.data_dir_Apple_processed_100hz_8w_4s
        elif params.step_size == 4 and params.window_size == 10:
            data_path = params.data_dir_Apple_processed_100hz_10w_4s
        elif params.step_size == 4 and params.window_size == 30:
            data_path = params.data_dir_Apple_processed_100hz_30w_4s
        elif params.step_size == 4 and params.window_size == 60:
            data_path = params.data_dir_Apple_processed_100hz_60w_4s
        elif params.step_size == 1 and params.window_size == 10:
            data_path = params.data_dir_Apple_processed_100hz_10w_1s
        else:
            raise ValueError(f"Invalid step size {params.step_size} for dataset {dataset}")

            
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, params=params,  reconstruction=reconstruction, take_every_nth_train=params.take_every_nth_train, take_every_nth_test=params.take_every_nth_test, sampling_rate=sampling_rate)
    
    elif dataset == 'appleall':
        sampling_rate = 100
        if params.step_size == 8 and params.window_size == 10:
            data_path = config.data_dir_Apple_processed_all
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, params=params,   reconstruction=reconstruction, sampling_rate=sampling_rate)
        else:
            raise ValueError(f"Invalid step size {params.step_size} and window size {params.window_size} for dataset {dataset}")
    
    elif dataset == 'm2sleep100':
        sampling_rate = 100
        if params.step_size == 8 and params.window_size == 10:
            data_path = config.data_dir_M2Sleep_processed_100Hz
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, params=params,   reconstruction=reconstruction, sampling_rate=sampling_rate)
        else:
            raise ValueError(f"Invalid step size {params.step_size} and window size {params.window_size} for dataset {dataset}")
    
    elif dataset == "parkinson100":
        sampling_rate = 100
        if params.step_size == 8 and params.window_size == 10:
            data_path = config.data_dir_Parkinson_processed_100Hz_wmetrics
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, params=params,   reconstruction=reconstruction, sampling_rate=sampling_rate)
        else:
            raise ValueError(f"Invalid step size {params.step_size} and window size {params.window_size} for dataset {dataset}")

    elif dataset == "IEEE":
        sampling_rate = 125
        if params.window_size == 8:
            data_path = config.data_dir_IEEE_processed
            x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split, params=params,   reconstruction=reconstruction, sampling_rate=sampling_rate)
        else:
            raise ValueError(f"Invalid step size {params.step_size} and window size {params.window_size} for dataset {dataset}")
    
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    #####################################################################################

    assert x_train.shape[0] == y_train.shape[0] == d_train.shape[0]

    if params.sampling_rate != sampling_rate and params.sampling_rate != 0:
        x_train = resample_data(x_train, sampling_rate, params.sampling_rate, params.cuda)
        x_val = resample_data(x_val, sampling_rate, params.sampling_rate, params.cuda)
        x_test = resample_data(x_test, sampling_rate, params.sampling_rate, params.cuda)
    else:
        params.sampling_rate = sampling_rate

    params.input_length = params.window_size * params.sampling_rate

    if params.normalize:
        # normalizes the data samples using z-score normalization
        x_train = normalize_samples(x_train)
        x_val = normalize_samples(x_val)
        x_test = normalize_samples(x_test)
    
    if params.normalize and reconstruction:
        # also normlizes the target samples (ECG) using z-score normalization
        y_train = normalize_samples(y_train)
        y_val = normalize_samples(y_val)
        y_test = normalize_samples(y_test)

    if not reconstruction:
        # normliazes the domain labels with min-max normalization
        y_train = norm_hr(y_train, params.hr_min, params.hr_max)
        y_val = norm_hr(y_val, params.hr_min, params.hr_max)
        y_test = norm_hr(y_test, params.hr_min, params.hr_max)
    else:
        # normalize ecg
        ecg_min = -1
        ecg_max = 6
        y_train = (y_train - ecg_min) / (ecg_max - ecg_min)
        y_val = (y_val - ecg_min) / (ecg_max - ecg_min)
        y_test = (y_test - ecg_min) / (ecg_max - ecg_min)

    if discrete_hr and not reconstruction:
        # discretizes the heart rate labels, for example for classification
        # smoothes the labels with a gaussian kernel
        sigma = params.label_sigma if sigma is None else sigma
        y_train = discretize_hr(y_train, hr_min = params.hr_min, hr_max = params.hr_max, n_bins=params.n_prob_class, sigma=sigma)
        y_val = discretize_hr(y_val, hr_min = params.hr_min, hr_max = params.hr_max, n_bins=params.n_prob_class, sigma=sigma)
        y_test = discretize_hr(y_test, hr_min = params.hr_min, hr_max = params.hr_max, n_bins=params.n_prob_class, sigma=sigma)

    if sample_sequences:
        # generates sequences of samples for the training, validation and test sets
        # each batch only contains subsequent samples
        x_train, y_train, d_train = generate_batch_sequences(x_train, y_train, d_train)
        x_val, y_val, d_val = generate_batch_sequences(x_val, y_val, d_val)
        x_test, y_test, d_test = generate_batch_sequences(x_test, y_test, d_test)

        train_set = data_loader_hr(x_train, y_train, d_train, split=split, partition="train", dataset=dataset)
        train_loader = DataLoader(train_set, batch_size=None, drop_last=False, batch_sampler=None, num_workers=params.num_workers, pin_memory=True, shuffle=True)

        test_set = data_loader_hr(x_test, y_test, d_test, split=split, partition="test", dataset=dataset)
        test_loader = DataLoader(test_set, batch_size=None, batch_sampler=None, shuffle=False)

        val_set = data_loader_hr(x_val, y_val, d_val, split=split, partition="val", dataset=dataset)
        val_loader = DataLoader(val_set, batch_size=None, batch_sampler=None, shuffle=False)
        
    else:
        # generates batches of samples for the training, validation and test sets
        train_set = data_loader_hr(x_train, y_train, d_train, split=split, partition="train", dataset=dataset)
        train_loader = DataLoader(train_set, batch_size=params.batch_size, drop_last=False, num_workers=params.num_workers, pin_memory=True, shuffle=True)

        test_set = data_loader_hr(x_test, y_test, d_test, split=split, partition="test", dataset=dataset)
        test_loader = DataLoader(test_set, batch_size=params.batch_size, shuffle=False)

        val_set = data_loader_hr(x_val, y_val, d_val, split=split, partition="val", dataset=dataset)
        val_loader = DataLoader(val_set, batch_size=params.batch_size, shuffle=False)

    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")

    return train_loader, val_loader, test_loader

