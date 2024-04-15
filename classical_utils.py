#%%
import os
import json
import pandas as pd
import numpy as np
import pickle
import sys
import heartpy as hp
import ssa_hr
from scipy.signal import find_peaks, find_peaks_cwt
import scipy
from scipy.signal import butter, sosfiltfilt, hilbert, correlate
import TROIKA_bcg
import data_utils
import matplotlib.pyplot as plt

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

def compute_hr_bioglass(X, fs = None, **kwargs):

    if hasattr(X, "fs"):
        fs = X.fs
    else:
        if fs is None:
            raise ValueError("fs is not defined")

    if not isinstance (X, pd.DataFrame):
        X = pd.DataFrame(X, columns=["acc_x", "acc_y", "acc_z"])
    
    df_snip_processed = data_utils.full_a_processing(X, fs, cutoff_frequencies=(0.5, 2))
    filtered_signal = df_snip_processed["mag_filtered"]

    hr_rr = extract_hr_peaks(filtered_signal, fs, method="cwt")
    return hr_rr

def compute_hr_bioglass_original(X, fs = None, **kwargs):

    if hasattr(X, "fs"):
        fs = X.fs
    else:
        if fs is None:
            raise ValueError("fs is not defined")

    if not isinstance (X, pd.DataFrame):
        X = pd.DataFrame(X, columns=["acc_x", "acc_y", "acc_z"])
    
    df_snip_processed = data_utils.full_a_processing(X, fs, cutoff_frequencies=(0.75, 2.5))
    filtered_signal = df_snip_processed["mag_filtered"]

    hr = extract_hr_spectrum(filtered_signal, fs, cutoff_frequencies=(0.75, 2.5))
    return hr


def compute_hr_ssa(X, fs = None, lagged_window_size = 501, first_n_components=20, **kwargs):
    if hasattr(X, "fs"):
        fs = X.fs
    else:
        if fs is None:
            raise ValueError("fs is not defined")

    if not isinstance(X, np.ndarray): 
        X = X.to_numpy()
    
    filtered_signal = ssa_hr.do_ssa_firstn(X, lagged_window_size=lagged_window_size, first_n_components=first_n_components)
    filtered_signal = data_utils.butterworth_bandpass(filtered_signal, fs, 0.5, 2)

    hr_rr = extract_hr_peaks(filtered_signal, fs)

    return hr_rr

def compute_hr_ssa_original(X, fs = None, lagged_window_size = 501, **kwargs):
    if hasattr(X, "fs"):
        fs = X.fs
    else:
        if fs is None:
            raise ValueError("fs is not defined")

    if not isinstance(X, np.ndarray): 
        X = X.to_numpy()
    
    filtered_signal = ssa_hr.do_ssa_axis(X, lagged_window_size=lagged_window_size)
    filtered_signal = data_utils.butterworth_bandpass(filtered_signal, fs, 0.5, 2)

    hr_rr = extract_hr_peaks(filtered_signal, fs, method="classical")

    return hr_rr


def compute_hr_median(X, Y_train, fs=100, **kwargs):
    # compute median HR from training data
    hr_median = np.median(Y_train)
    return [hr_median]*len(X)

def extract_hr_peaks(signal, fs, method="cwt"):
    if method == "cwt":
        peaks = find_peaks_cwt(signal, np.arange(5,80))
    elif method == "classical":
        peaks = find_peaks(signal, distance=0.5*fs, prominence=0.3 * np.quantile(signal, 0.9))[0]
    else:
        raise ValueError("Method not implemented")
    rr = np.diff(peaks)
    rr = rr[(rr > 0.5*fs) & (rr < 1.5*fs)]
    if len(rr) < 2:
        return np.nan
    hr = 60*fs/np.mean(rr)
    return hr

def extract_hr_spectrum(signal, fs, cutoff_frequencies=(0.5, 2)):
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)

    hr_frequenies = (freqs > cutoff_frequencies[0]) & (freqs < cutoff_frequencies[1])
    hr_periodogram = np.abs(fft[hr_frequenies])**2
    max_frequency = freqs[np.argmax(hr_periodogram)]
    hr = 60*max_frequency
    return hr





def single_channel_butterworth_bandpass(signal, fs, low, high, order=4):
    """
    Apply Butterworth bandpass filter to a single channel of signal data.

    Args:
    - signal (array-like): Single-channel signal data.
    - fs (float): Sampling frequency of the signal.
    - low (float): Low cutoff frequency of the bandpass filter in Hz.
    - high (float): High cutoff frequency of the bandpass filter in Hz.
    - order (int, optional): Order of the Butterworth filter. Default is 4.

    Returns:
    - filtered_signal (array-like): Filtered single-channel signal data.
    """
    if fs/2 <= high:
        print("Warning: Sampling frequency is too low for the specified high cutoff frequency. Consider increasing the sampling frequency or decreasing the high cutoff frequency.")
        sos = butter(order, low, btype='highpass', fs=fs, output='sos')
    else:
        sos = butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
    filtered = sosfiltfilt(sos, signal)
    return filtered




def wcorr(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    weighted correlation of ts1 and ts2.
    w is precomputed for reuse.
    """
    L = 500
    N = 1000
    K = N - L + 1
    w = np.concatenate((np.arange(1, L+1), np.full((K-L,), L), np.arange(L-1, 0, -1)))
    w_covar = (w * ts1 * ts2).sum()
    ts1_w_norm = np.sqrt((w * ts1 * ts1).sum())
    ts2_w_norm = np.sqrt((w * ts2 * ts2).sum())
    
    return w_covar / (ts1_w_norm * ts2_w_norm)


def select_components(acc_groups, threshold=0.1):
    selected_indices = []
    for i in range(acc_groups.shape[0]):
        _, periodogram = scipy.signal.periodogram(acc_groups[i,:], nfft=4096 * 2 - 1)
        frequencies = np.linspace(0,100, 4096)
        max_amplitude = np.max(np.abs(periodogram))
        hr_frequenies = (frequencies > 0.5) & (frequencies < 4)

        if np.any(periodogram[hr_frequenies] > max_amplitude*threshold):
            selected_indices = np.append(selected_indices, i)

    selected_indices = np.array(selected_indices, dtype=int)
    acc_reconstructed = acc_groups[selected_indices,:].sum(axis=0)
    return acc_reconstructed

def compute_hr_troika_w_tracking(X, fs=100, f_low=0.5, f_high=4, **kwargs):
    troika = TROIKA_bcg.Troika(window_duration=10, acc_sampling_freq=fs, cutoff_freqs=(f_low, f_high))
    hr = []
    for i, hr_i in enumerate(troika.transform(X)):
        hr.append(hr_i)
    return hr

def compute_hr_troika(X, fs=100, n_freq=4096, f_low=0.5, f_high=4, **kwargs):

    acc_x = data_utils.butterworth_bandpass(X[:,0], low=f_low, high=f_high, fs=fs)
    acc_y = data_utils.butterworth_bandpass(X[:,1], low=f_low, high=f_high, fs=fs)
    acc_z = data_utils.butterworth_bandpass(X[:,2], low=f_low, high=f_high, fs=fs)
    acc_groups_x, wcorr_x = TROIKA_bcg.ssa(acc_x, 500, perform_grouping=True, ret_Wcorr=True)
    acc_groups_y, wcorr_y = TROIKA_bcg.ssa(acc_y, 500, perform_grouping=True, ret_Wcorr=True)
    acc_groups_z, wcorr_z = TROIKA_bcg.ssa(acc_z, 500, perform_grouping=True, ret_Wcorr=True)


    def select_components(acc_groups, threshold=0.1):
        selected_indices = []
        for i in range(acc_groups.shape[0]):
            frequencies, periodogram = scipy.signal.periodogram(acc_groups[i,:], nfft=n_freq * 2 - 1, fs=fs)
            max_amplitude = np.max(np.abs(periodogram))
            hr_frequenies = (frequencies > 0.5) & (frequencies < 2)

            if np.any(periodogram[hr_frequenies] > max_amplitude*threshold):
                selected_indices = np.append(selected_indices, i)
        #print(selected_indices)
        selected_indices = np.array(selected_indices, dtype=int)
        acc_reconstructed = acc_groups[selected_indices,:].sum(axis=0)
        return acc_reconstructed


    acc_reconstructed_x = select_components(acc_groups_x)
    acc_reconstructed_y = select_components(acc_groups_y)
    acc_reconstructed_z = select_components(acc_groups_z)

    acc_reconstructed = np.sqrt(acc_reconstructed_x**2 + acc_reconstructed_y**2 + acc_reconstructed_z**2)


    # differentiating for more robustness
    acc_reconstructed = np.diff(acc_reconstructed)

    frequencies, periodogram = scipy.signal.periodogram(acc_reconstructed, nfft=n_freq * 2 - 1, fs=fs)


    hr_frequenies_ind = (frequencies > 0.5) & (frequencies < 2)
    hr_periodogram, hr_frequenies = periodogram[hr_frequenies_ind], frequencies[hr_frequenies_ind]
    hr_peak_ind = scipy.signal.find_peaks(hr_periodogram)[0]
    highest_hr_peak_ind = np.argsort(hr_periodogram[hr_peak_ind])[::-1][:10]

    highest_hr_peaks = hr_frequenies[hr_peak_ind[highest_hr_peak_ind]] * 60
    highest_hr_peak = highest_hr_peaks[0]
    return highest_hr_peak

#%%

def compute_hr_kantelhardt(X, fs=100, peak_detection = "cwt", **kwargs): 

    X = data_utils.butterworth_bandpass(X, low=5, high=14, fs=fs)
    X = np.apply_along_axis(lambda x: np.abs(hilbert(x)), 0, X)

    if peak_detection == "classical":
        peaks = [find_peaks(x, distance=0.5*fs, height=0.3 * np.quantile(x, 0.9))[0] for x in X.T]
    elif peak_detection == "cwt":
        peaks = [find_peaks_cwt(x, np.arange(5,80)) for x in X.T]
    peaks_diff = [np.diff(p) for p in peaks]
    peaks_diff = [p[(p > 0.5*fs) & (p < 1.5*fs)] for p in peaks_diff]
    hr_canditates = np.array([60*fs/np.mean(p) for p in peaks_diff])

    possible_axes = (hr_canditates>40) & (hr_canditates<200)

    if np.sum(possible_axes) > 1:
        # compute autocorrelation function of hilbert output
        X_autocorr = np.apply_along_axis(lambda x: np.correlate(x, x, mode='full'), 0, X)
        X_autocorr_selected = X_autocorr[int(len(X_autocorr)//2 + 0.4*fs): int(len(X_autocorr)//2 + 1.5*fs)]
        selected_axis = np.argmax(np.max(X_autocorr_selected, axis=0))
    elif np.sum(possible_axes) == 1:
        selected_axis = np.argmax(possible_axes)
    else: # no axis found, take x axis
        selected_axis = 0

    hr = hr_canditates[selected_axis]
    return hr


def compute_hr_kantelhardt_original(X, fs=100, **kwargs): 

    X = data_utils.butterworth_bandpass(X, low=5, high=14, fs=fs)
    X = np.apply_along_axis(lambda x: np.abs(x + 1j*np.abs(hilbert(x))), 0, X)


    peaks = [find_peaks(x, distance=0.5*fs, height=0.3 * np.quantile(x, 0.9))[0] for x in X.T]

    peaks_diff = [np.diff(p) for p in peaks]
    peaks_diff = [p[(p > 0.5*fs) & (p < 1.5*fs)] for p in peaks_diff]
    hr_canditates = np.array([60*fs/np.mean(p) for p in peaks_diff])

    possible_axes = (hr_canditates>40) & (hr_canditates<200)

    if np.sum(possible_axes) > 1:
        # compute autocorrelation function of hilbert output
        X_autocorr = np.apply_along_axis(lambda x: np.correlate(x, x, mode='full'), 0, X)
        X_autocorr_selected = X_autocorr[int(len(X_autocorr)//2 + 0.4*fs): int(len(X_autocorr)//2 + 1.5*fs)]
        selected_axis = np.argmax(np.max(X_autocorr_selected, axis=0))
    elif np.sum(possible_axes) == 1:
        selected_axis = np.argmax(possible_axes)
    else: # no axis found, take x axis
        selected_axis = 0

    hr = hr_canditates[selected_axis]
    return hr

def plot_true_pred(hr_true, hr_pred, x_lim=[20, 120], y_lim=[20, 120]):
    corr = lambda a, b: pd.DataFrame({'a':a, 'b':b}).corr().iloc[0,1]
    figure = plt.figure(figsize=(8, 8))
    hr_true, hr_pred = np.array(hr_true), np.array(hr_pred)
    mae = np.round(np.abs(hr_true - hr_pred).mean(), 2)
    correlation_coefficient = corr(hr_true, hr_pred)

    plt.scatter(x = hr_true, y = hr_pred, alpha=0.2, label=f"MAE: {mae:.2f}, Corr: {correlation_coefficient:.2f}")

    plt.plot(x_lim, y_lim, color='k', linestyle='-', linewidth=2)
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.xlabel('True HR (bpm)')
    plt.ylabel('Predicted HR (bpm)')
    plt.legend()
    return figure