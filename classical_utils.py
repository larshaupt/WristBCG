import os
import json
import pandas as pd
import numpy as np
import pickle
import sys
import heartpy as hp
import ssa_hr
from scipy.signal import find_peaks, find_peaks_cwt

sys.path.append("/local/home/lhauptmann/thesis/t-mt-2023-WristBCG-LarsHauptmann/source")
import data_utils

def load_subjects(dataset_dir, subject_paths):
    X, Y, pid, metrics = [], [], [], []
    for sub in subject_paths:
        with open(os.path.join(dataset_dir, sub), "rb") as f:
            data = pickle.load(f)
            if len(data) == 4:
                X_sub, Y_sub, pid_sub, metrics_sub = data
            else:
                X_sub, Y_sub, pid_sub = data
                metrics_sub = np.zeros_like(Y_sub)

        X.append(X_sub)
        Y.append(Y_sub)
        pid.append(pid_sub)
        metrics.append(metrics_sub)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    pid = np.concatenate(pid, axis=0)
    metrics = np.concatenate(metrics, axis=0)

    return X, Y, pid, metrics

def load_dataset(dataset_dir, split, load_train=False):
    # load split
    split_file = os.path.join(dataset_dir, f'splits.json')

    if not os.path.exists(split_file):
        raise ValueError(f"Split file {split_file} does not exist")
    
    with open(split_file) as f:
        splits = json.load(f)

    split = splits[str(split)]


    # load data
    if load_train:
        X_train, Y_train, pid_train, metrics_train = load_subjects(dataset_dir, split["train"])
    X_val, Y_val, pid_val, metrics_val = load_subjects(dataset_dir, split["val"])
    X_test, Y_test, pid_test, metrics_test = load_subjects(dataset_dir, split["test"])

    if load_train:
        return X_train, Y_train, pid_train, metrics_train, X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test
    else:
        return X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test

def compute_hr_bioglass(X, fs = None, peak_distance = 0.5, peak_prominence = 0.3):
    

    if hasattr(X, "fs"):
        fs = X.fs
    else:
        if fs is None:
            raise ValueError("fs is not defined")

    if not isinstance (X, pd.DataFrame):
        X = pd.DataFrame(X, columns=["acc_x", "acc_y", "acc_z"])
    
    df_snip_processed = data_utils.full_a_processing(X, fs)
    filtered_signal = df_snip_processed["mag_filtered"]

    filtered_signal = hp.scale_data(filtered_signal)
    filtered_signal = hp.remove_baseline_wander(filtered_signal, fs)
    filtered_signal = hp.filter_signal(filtered_signal, 0.05, fs, filtertype='notch')
    #ecg = hp.enhance_ecg_peaks(hp.scale_data(ecg), fs, aggregation='median', iterations=5)

    hr_rr = extract_hr_peaks(filtered_signal, fs)
    return hr_rr


def compute_hr_ssa(X, fs = None, lagged_window_size = 501, first_n_components=20, peak_distance = 0.5, peak_prominence = 0.3):
    if hasattr(X, "fs"):
        fs = X.fs
    else:
        if fs is None:
            raise ValueError("fs is not defined")

    if not isinstance(X, np.ndarray): 
        X = X.to_numpy()
    
    filtered_signal = ssa_hr.do_ssa_firstn(X, lagged_window_size=lagged_window_size, first_n_components=first_n_components)
    filtered_signal = data_utils.butterworth_bandpass(filtered_signal, fs, 0.5, 2)

    #filtered_signal = hp.scale_data(filtered_signal)
    #filtered_signal = hp.remove_baseline_wander(filtered_signal, fs)
    #filtered_signal = hp.filter_signal(filtered_signal, 0.05, fs, filtertype='notch')
    #filtered_signal = hp.enhance_ecg_peaks(filtered_signal, fs, aggregation='median', iterations=5)
    hr_rr = extract_hr_peaks(filtered_signal, fs)

    return hr_rr


def extract_hr_peaks(signal, fs):
    peaks = find_peaks_cwt(signal, np.arange(5,80))
    #peaks = find_peaks(signal, distance=0.5*fs, prominence=0.3 * np.quantile(signal, 0.9))[0]
    rr = np.diff(peaks)
    rr = rr[(rr > 0.5*fs) & (rr < 1.5*fs)]
    if len(rr) < 2:
        return np.nan
    hr = 60*fs/np.mean(rr)
    return hr