import numpy as np
import os
import pandas as pd
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
from scipy.signal import convolve, butter, sosfilt, find_peaks, sosfiltfilt, resample
from scipy.ndimage import maximum_filter1d
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
from sklearn.model_selection import GroupShuffleSplit, KFold, GroupKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
import heartpy as hp
import h5py
import torch
import julius
import pickle
import json

from scipy.optimize import linear_sum_assignment

fig_width = 6
fig_height = 6
cmap = plt.get_cmap('tab10')


def read_acc_M2Sleep(recording_path, compute_mag=False):
    """
    Read accelerometer data from an M2Sleep recording.

    Parameters:
    recording_path (str): Path to the directory containing the recording files.

    Returns:
    pd.DataFrame: A DataFrame containing accelerometer data with columns 'timestamp', 'acc_x', 'acc_y', 'acc_z', 'mag', 'time', and 'fs'.
    """
    acc_file = os.path.join(recording_path, 'ACC.csv')
    try:
        df = pd.read_csv(acc_file, dtype='int', header=None)
        timestamp_init = df.iloc[0]
        sampling_rate = df.iloc[1]

        # Ensure all initial timestamps and sampling rates are the same
        assert timestamp_init.eq(timestamp_init.iloc[0]).all()
        assert sampling_rate.eq(sampling_rate.iloc[0]).all()
        timestamp_init = timestamp_init.iloc[0]
        sampling_rate = sampling_rate.iloc[0]
        
        # Process data
        df = df.tail(-2)
        num_samples = len(df)
        df.index = pd.Series([timestamp_init + i/sampling_rate for i in range(num_samples)], name='timestamp')
        df.columns = pd.Series(['acc_x', 'acc_y', 'acc_z'])
        if compute_mag:
            df['mag'] = df.apply(lambda x: np.sqrt(x['acc_x']**2 + x['acc_y']**2 + x['acc_z']**2), axis=1)
        df['time'] = df.index.map(lambda x: datetime.fromtimestamp(x))
        df.fs = sampling_rate
        return df
    
    except Exception as e:
        print(f'Error reading {acc_file}: {e}')

def read_hr_M2Sleep(recording_path):
    """
    Read heart rate data from an M2Sleep recording.

    Parameters:
    recording_path (str): Path to the directory containing the recording files.

    Returns:
    pd.DataFrame: A DataFrame containing heart rate data with columns 'timestamp', 'hr', 'time', and 'fs'.
    """
    hr_file = os.path.join(recording_path, 'HR.csv')

    df = pd.read_csv(hr_file, dtype='float', header=None)
    timestamp_init = df.iloc[0].astype('int')
    sampling_rate = df.iloc[1]

    # Ensure all initial timestamps and sampling rates are the same
    assert timestamp_init.eq(timestamp_init.iloc[0]).all()
    assert sampling_rate.eq(sampling_rate.iloc[0]).all()
    timestamp_init = timestamp_init.iloc[0]
    sampling_rate = sampling_rate.iloc[0]

    # Process data
    df = df.tail(-2)
    num_samples = len(df)
    df.index = pd.Series([timestamp_init + i/sampling_rate for i in range(num_samples)], name='timestamp')
    df.columns = pd.Series(['hr'])
    df['time'] = df.index.map(lambda x: datetime.fromtimestamp(x))
    df.fs = sampling_rate
    return df

def read_bvp_M2Sleep(recording_path):
    """
    Read blood volume pulse data from an M2Sleep recording.

    Parameters:
    recording_path (str): Path to the directory containing the recording files.

    Returns:
    pd.DataFrame: A DataFrame containing BVP data with columns 'timestamp', 'bvp', 'time', and 'fs'.
    """
    bvp_file = os.path.join(recording_path, 'BVP.csv')
    try:
        df = pd.read_csv(bvp_file, dtype='float', header=None)
        timestamp_init = df.iloc[0].astype('int')
        sampling_rate = df.iloc[1]

        # Ensure all initial timestamps and sampling rates are the same
        assert timestamp_init.eq(timestamp_init.iloc[0]).all()
        assert sampling_rate.eq(sampling_rate.iloc[0]).all()
        timestamp_init = timestamp_init.iloc[0]
        sampling_rate = sampling_rate.iloc[0]

        # Process data
        df = df.tail(-2)
        num_samples = len(df)
        df.index = pd.Series([timestamp_init + i/sampling_rate for i in range(num_samples)], name='timestamp')
        df.columns = pd.Series(['bvp'])
        df['time'] = df.index.map(lambda x: datetime.fromtimestamp(x))
        df.fs = sampling_rate
        return df
    except Exception as e:
        print(f'Error reading {bvp_file}: {e}')

def read_M2Sleep(recording_dir, compute_mag=False, sampling_rate=0):
    """
    Reads and processes data from M2Sleep recordings.

    Args:
        recording_dir (str): The directory containing M2Sleep recording data in CSV format.

    Returns:
        pandas.DataFrame: A DataFrame containing the processed data, with columns for acceleration (acc),
        heart rate (hr), and blood volume pulse (bvp).

    This function reads data from the specified recording directory, including accelerometer (acc), heart rate (hr),
    and blood volume pulse (bvp) measurements. It combines the data into a single DataFrame, interpolates missing values,
    and aligns the data with the accelerometer's timestamp index. The resulting DataFrame includes the interpolated
    heart rate and blood volume pulse values, with the accelerometer's sampling frequency preserved.

    Example:
    >>> recording_data = read_M2Sleep('/path/to/recording_directory')
    >>> print(recording_data.head())
    """

    # Use try-except to handle potential errors while reading the CSV files
    print(f'Read {os.path.basename(recording_dir)}')
    df_acc = read_acc_M2Sleep(recording_dir,compute_mag=compute_mag)
    df_hr = read_hr_M2Sleep(recording_dir)
    df_bvp = read_bvp_M2Sleep(recording_dir)

    # resamples if wanted
    if sampling_rate != 0 and sampling_rate != df_acc.fs:
        df_acc = df_resample(df_acc, sampling_rate)


    # Joins the dataframes on the timestamp index, interpolating missing values
    df_all = df_acc.join(df_hr['hr'], how='outer')
    df_all = df_all.join(df_bvp['bvp'], how='outer')
    df_all['hr'] = df_all['hr'].interpolate('linear', limit_direction="both", fill_value="extrapolate")
    df_all['bvp'] = df_all['bvp'].interpolate('linear')
    df_all = df_all.loc[df_acc.index]

    df_all.fs = df_acc.fs

    return df_all

def df_resample(df, fs, channels=["acc_x", "acc_y", "acc_z", "mag"]):
    """
    Resamples a DataFrame to a specified sampling frequency.

    Args:
        df (pandas.DataFrame): The DataFrame to resample.
        fs (int): The sampling frequency to resample to.
        channels (list of str, optional): The channels to resample (default is ["acc_x", "acc_y", "acc_z", "mag"]).

    Returns:
        pandas.DataFrame: The resampled DataFrame.

    This function resamples the specified channels of a DataFrame to a specified sampling frequency.
    The resampling is performed using linear interpolation.
    Assumes that the data is continuous, i.e. there are no time gaps in the data.
    """
    df_resampled = {}
    num_resampled = int(len(df) * fs / df.fs)

    for ch in channels:
        if ch in df.columns:
            df_resampled[ch] = resample(df[ch], num_resampled, domain="time")

    new_time = resample(df.index, num_resampled, domain="time")

    df_resampled = pd.DataFrame(df_resampled)
    df_resampled.fs = fs
    df_resampled.index = new_time
    df_resampled['time'] = df_resampled.index.map(lambda x: datetime.fromtimestamp(x))

    return df_resampled

def compute_fourier(signal, fs):
    """
    Compute the Fourier transform of a signal.

    Parameters:
    signal (array-like): Input signal.
    fs (float): Sampling frequency of the signal.

    Returns:
    tuple: A tuple containing the Fourier transform results (fft_result) and corresponding frequencies (frequencies).
    """
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_result), 1 / fs)
    return fft_result, frequencies

def plot_fourier(fft_result, frequencies, ax=None, color=None, title=''):
    """
    Plot the magnitude of the Fourier transform.

    Parameters:
    fft_result (array-like): The Fourier transform results.
    frequencies (array-like): Corresponding frequencies.
    ax (matplotlib.Axes, optional): The axes to plot on (if None, a new figure is created).
    color (str, optional): Line color.
    title (str, optional): Plot title.
    """
    if ax is None:
        fig, axes = plt.subplots(1)
        ax = axes

    ax.plot(frequencies, np.abs(fft_result), color=color, label=title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(left=0)
    ax.set_title("Fourier Transform")
    ax.grid(True)

    if ax is None:
        plt.show()

def compute_and_plot_fourier(signal, fs, ax=None, color=None, title=''):
    """
    Compute and plot the magnitude of the Fourier transform of a signal.

    Parameters:
    signal (array-like): Input signal.
    fs (float): Sampling frequency of the signal.
    ax (matplotlib.Axes, optional): The axes to plot on (if None, a new figure is created).
    color (str, optional): Line color.
    title (str, optional): Plot title.
    """
    fft_result, frequencies = compute_fourier(signal, fs)
    plot_fourier(fft_result, frequencies, ax=ax, color=None, title=title)

def plot_signal(signal, time, peaks=[], ax=None, title='', lines=[], color=None, signal_ratio=1.0):
    """
    Plot a signal with optional peaks and vertical lines.

    Parameters:
    signal (array-like): The signal to plot.
    time (array-like): Time values for the signal.
    peaks (array-like, optional): Positions of peaks.
    ax (matplotlib.Axes, optional): The axes to plot on (if None, a new figure is created).
    title (str, optional): Plot title.
    lines (array-like, optional): Vertical lines to draw.
    color (str, optional): Line color.
    signal_ratio (float, optional): Ratio of the signal to plot (0.0 to 1.0).

    Returns:
    None
    """
    time = np.array(time)
    signal = np.array(signal)

    if ax is None:
        fig, axes = plt.subplots(1, figsize=(fig_width, fig_height))
        ax = axes

    if signal_ratio < 1.0:
        signal = signal[:int(signal_ratio * len(signal))]
        time = time[:int(signal_ratio * len(time))]

    ax.plot(time, signal, color=color)
    ax.set_xlabel('Time')
    ax.set_ylabel('Signal')
    ax.set_title(title)
    ax.tick_params(labelrotation=45)

    if len(peaks) != 0:
        peaks = peaks[peaks < len(signal)]
        ax.plot(time[peaks], signal[peaks], 'x', color=color, label=title)

    if len(lines) != 0:
        lines = lines[lines < len(signal)]
        for line in lines:
            ax.axvline(x=line, color='r')

    if ax is None:
        plt.show()

def plot_signal_and_fourier(signal, time, fs, peaks=[]):
    """
    Plot a signal and its Fourier transform side by side.

    Parameters:
    signal (array-like): The signal to plot.
    time (array-like): Time values for the signal.
    fs (float): Sampling frequency of the signal.
    peaks (array-like, optional): Positions of peaks.

    Returns:
    None
    """
    fig, axes = plt.subplots(2, figsize=(fig_width, fig_height))
    plot_signal(signal, time, peaks=peaks, ax=axes[0])
    compute_and_plot_fourier(signal, fs, ax=axes[1])
    fig.tight_layout()

def plot_signals_and_fourier(signals, time, fss, titles=[], peaks=[], signal_ratio=1.0):
    """
    Plot multiple signals and their Fourier transforms in separate subplots.

    Parameters:
    signals (list of array-like): List of signals to plot.
    time (array-like): Time values for the signals.
    fss (float or list of float): Sampling frequencies for the signals.
    titles (str or list of str, optional): Titles for the subplots.
    peaks (list of array-like, optional): Positions of peaks for each signal.
    signal_ratio (float, optional): Ratio of the signals to plot (0.0 to 1.0).

    Returns:
    None
    """
    num_signals = len(signals)

    if isinstance(titles, list) and len(titles) == 1:
        titles = titles * num_signals
    elif isinstance(titles, str):
        titles = [titles] * num_signals
    elif isinstance(titles, list) and len(titles) == num_signals:
        pass
    else:
        titles = ['Signal'] * num_signals

    if not isinstance(fss, list):
        fss = [fss] * num_signals
    assert len(fss) == num_signals

    fig, axes = plt.subplots(2, num_signals, figsize=(fig_width * num_signals, fig_height))
    for i in range(num_signals):
        if num_signals == 1:
            ax0 = axes[0]
            ax1 = axes[1]
        else:
            ax0 = axes[0, i]
            ax1 = axes[1, i]
        peak = peaks[i] if len(peaks) > i else []
        plot_signal(signals[i], time, ax=ax0, title=titles[i], peaks=peak, signal_ratio=signal_ratio)
        compute_and_plot_fourier(signals[i], fss[i], ax=ax1)
    fig.tight_layout()

def plot_signals_and_fourier_together(signals, time, fss, titles='Signal', peaks=[], signal_ratio=1.0):
    """
    Plot multiple signals and their Fourier transforms together in a single subplot.

    Parameters:
    signals (list of array-like): List of signals to plot.
    time (array-like): Time values for the signals.
    fss (float or list of float): Sampling frequencies for the signals.
    titles (str or list of str, optional): Titles for the signals.
    peaks (list of array-like, optional): Positions of peaks for each signal.
    signal_ratio (float, optional): Ratio of the signals to plot (0.0 to 1.0).

    Returns:
    None
    """
    num_signals = len(signals)

    if isinstance(titles, list) and len(titles) == 1:
        titles = titles * num_signals
    elif isinstance(titles, str):
        titles = [titles] * num_signals
    elif isinstance(titles, list) and len(titles) == num_signals:
        pass
    else:
        titles = ['Signal'] * num_signals

    if not isinstance(fss, list):
        fss = [fss] * num_signals
    assert len(fss) == num_signals

    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_width))
    for i in range(num_signals):
        peak = peaks[i] if len(peaks) > i else []
        plot_signal(signals[i], time, ax=axes[0], title=titles[i], peaks=peak, color=cmap(i), signal_ratio=signal_ratio)
        compute_and_plot_fourier(signals[i], fss[i], ax=axes[1], color=cmap(i), title=titles[i])
    axes[0].legend()
    axes[1].legend()

    fig.tight_layout()

def match_pairs(a, b):
    """
    Match pairs of elements between two arrays using linear sum assignment.

    Parameters:
    a (array-like): First array.
    b (array-like): Second array.

    Returns:
    tuple: A tuple containing the assignment result, average distance, and average absolute distance.
    """
    dist_matrix = np.zeros([len(a), len(b)])
    for i, el in enumerate(a):
        dist_matrix[i, :] = el - b
    c_matrix = np.abs(dist_matrix)
    assignment = linear_sum_assignment(c_matrix)
    energy = c_matrix[assignment[0], assignment[1]].sum()
    average_absolute_distance = c_matrix[assignment[0], assignment[1]].mean()
    average_distance = dist_matrix[assignment[0], assignment[1]].mean()
    return assignment, average_distance, average_absolute_distance

def match_pairs_neighbouring(a, b):
    """
    Match pairs of neighboring elements between two arrays.

    Parameters:
    a (array-like): First array.
    b (array-like): Second array.

    Returns:
    tuple: A tuple containing the assignment result, average distance, and average absolute distance.
    """
    assignments = []
    for i, el in enumerate(a[:-1]):
        next_el = a[i + 1]
        matching_b = [x for x in b if el < x < next_el]
        if len(matching_b) > 0:
            assignments.append((el, np.min(matching_b)))
    assignments = np.array(assignments).reshape(-1, 2)
    average_distance = np.abs(assignments[:, 0] - assignments[:, 1]).mean()
    average_absolute_distance = np.abs(assignments[:, 0] - assignments[:, 1]).mean()
    return assignments, average_distance, average_absolute_distance

def z_normalization(signal, signal_mean=None, signal_std=None):
    """
    Perform z-score normalization on a signal.

    Parameters:
    signal (array-like): Input signal.

    Returns:
    array-like: Normalized signal.
    """
    if not isinstance(signal_mean, np.ndarray) or len(signal_mean) != signal.shape[1]:
        signal_mean = signal.mean(axis=0)
    if not isinstance(signal_std, np.ndarray) or len(signal.std) != signal.shape[1]:
        signal_std = signal.std(axis=0)
    if signal_std != 0:
        signal_norm = (signal - signal_mean) / signal_std
    else:
        signal_norm = signal - signal_mean
    return signal_norm

def minmax_normalization(signal):
    """
    Perform min-max normalization on a signal.

    Parameters:
    signal (array-like): Input signal.

    Returns:
    array-like: Normalized signal.
    """
    signal_min = signal.min()
    signal_max = signal.max()
    if signal_max != signal_min:
        signal_norm = (signal - signal_min) / (signal_max - signal_min)
    else:
        signal_norm = signal - signal_min
    return signal_norm

def quantile_normalization(signal, quantile=0.001):
    """
    Perform quantile normalization on a signal.

    Parameters:
    signal (array-like): Input signal.
    quantile (float, optional): Quantile value to determine the normalization range.

    Returns:
    array-like: Normalized signal.
    """
    signal_min = np.quantile(signal, quantile)
    signal_max = np.quantile(signal, 1 - quantile)
    if signal_max != signal_min:
        signal_norm = (signal - signal_min) / (signal_max - signal_min)
    else:
        signal_norm = signal - signal_min
    return signal_norm

def df_normalization(df, channels=['acc_x','acc_y', 'acc_z'], method='z-score'):
    df_norm = df.copy()
    if method == 'z-score':
        df_norm[channels] = df_norm[channels].apply(z_normalization)
    elif method == 'min-max':
        df_norm[channels] = df_norm[channels].apply(minmax_normalization)
    elif method == 'quantile':
        df_norm[channels] = df_norm[channels].apply(quantile_normalization)
    else:
        raise NotImplementedError(f'Normalization method {method} not implemented')
    return df_norm

def split_into_windows(df, fs, window_size=10, window_overlap=0.5):
    """
    Split a DataFrame into overlapping windows.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    fs (float): Sampling frequency.
    window_size (int): Size of the window in seconds.
    window_overlap (float): Overlap between windows as a fraction (0.0 to 1.0).

    Yields:
    pd.DataFrame: Windows of the input DataFrame.
    """
    window_size = int(window_size * fs)
    for start_index in np.arange(0, len(df), int(window_size * (1 - window_overlap))):
        end_index = start_index + window_size
        if end_index > len(df):
            break
        yield df.iloc[start_index:end_index]

def extract_bcg_peaks(acc, bvp, peak_distance=10, interval_length=30, peak_prominence=0.6, max_peak_distance=30):
    """
    Extract BCG peaks from accelerometer and BVP signals.

    Parameters:
    acc (array-like): Accelerometer signal.
    bvp (array-like): BVP signal.
    peak_distance (int): Minimum distance between peaks.
    interval_length (int): Length of the interval for searching peaks.
    peak_prominence (float): Minimum peak prominence.
    max_peak_distance (int): Maximum peak distance.

    Returns:
    list: List of BCG peaks.
    """
    peaks_acc, _ = find_peaks(acc, prominence=peak_prominence * np.percentile(np.abs(acc), 90), distance=peak_distance)
    peaks_bvp, _ = find_peaks(bvp, prominence=peak_prominence * np.percentile(np.abs(bvp), 90), distance=peak_distance)
    bcg_peaks = []
    for i, bvp_peak in enumerate(peaks_bvp[:-1]):
        next_bvg_peak = peaks_bvp[i + 1]
        corresponding_acc_peaks = [acc_peak for acc_peak in peaks_acc if bvp_peak <= acc_peak < next_bvg_peak and max_peak_distance >= abs(acc_peak - bvp_peak)]
        if len(corresponding_acc_peaks) == 0:
            continue
        corresponding_acc_peaks.sort()
        corresponding_acc_peak = corresponding_acc_peaks[0]
        start_time = max(int(corresponding_acc_peak - interval_length), 0)
        start_time = bvp_peak
        end_time = min(int(corresponding_acc_peak + interval_length), len(acc))
        cooresponding_acc_period = acc[start_time:end_time]
        padding_end = max(max_peak_distance + interval_length - len(cooresponding_acc_period), 0)
        bcg_peaks.append(cooresponding_acc_period)
    return bcg_peaks

def extract_bvp_peaks(bvp, fs, interval_length=30, peak_distance = 0.5, peak_prominence = 0.6):
    """
    Extract BVP peaks from a BVP signal.

    Parameters:
    bvp (array-like): BVP signal.
    fs (float): Sampling frequency.
    interval_length (int): Length of the interval for extracting peaks.
    peak_distance (int): Minimum distance between peaks.
    peak_prominence (float): Minimum peak prominence.
    peak_distance (float): Minimum distance between peaks in seconds

    Returns:
    list: List of BVP peaks.
    """
    peak_distance = peak_distance * fs
    peaks_bvp, _ = find_peaks(bvp, prominence=peak_prominence * np.percentile(np.abs(bvp), 90), distance=peak_distance)

    bvp_peaks = []

    for bvp_peak in peaks_bvp: 
        start_time = max(int(bvp_peak - interval_length), 0)
        padding_start = max(int(interval_length - bvp_peak), 0)
        end_time = min(int(bvp_peak + interval_length), len(bvp))
        padding_end = max(int(interval_length - (len(bvp) - bvp_peak)), 0)
        
        bvp_period = bvp[start_time:end_time]

        bvp_period = np.pad(bvp_period, (padding_start, padding_end), 'constant')
        bvp_peaks.append(bvp_period)

    return bvp_peaks

def a_processing(signal, fs, detrend_window_size=None):
    """
    Apply signal processing to remove noise and artifacts.

    Parameters:
    signal (array-like): Input signal.
    fs (float): Sampling frequency.
    detrend_window_size (int, optional): Size of the window for detrending.

    Returns:
    array-like: Processed signal.
    """
    if not detrend_window_size:
        detrend_window_size = int(3 * fs)

    signal_norm = z_normalization(signal)

    signal_detrend = signal_norm - convolve(signal_norm, np.ones(detrend_window_size) / detrend_window_size, 'same')
    sos = butter(4, [4, 11], 'bandpass', fs=fs, output='sos')
    signal_bandpass = sosfiltfilt(sos, signal_detrend)
    return signal_bandpass

def full_a_processing(df_snip, fs, channels=[], detrend_window_size=None):
    """
    Apply signal processing to a DataFrame.

    Parameters:
    df_snip (pd.DataFrame): Input DataFrame.
    fs (float): Sampling frequency.
    channels (list of str, optional): List of channels to process.
    detrend_window_size (int, optional): Size of the window for detrending.

    Returns:
    pd.DataFrame: Processed DataFrame.
    """
    if channels == []:
        channels = ['acc_x', 'acc_y', 'acc_z']
    
    channels_filtered = [a_processing(df_snip[channel], fs, detrend_window_size=detrend_window_size) for channel in channels]

    mag_filtered = np.linalg.norm(np.array(channels_filtered), axis=0)

    filter_sos = butter(2, [0.5, 2.0], 'bandpass', fs=fs, output='sos')
    mag_filtered = sosfiltfilt(filter_sos, mag_filtered)

    df_mag_filtered = pd.Series(mag_filtered, index=df_snip.index, name='mag_filtered')
    df_snip = df_snip.join(df_mag_filtered)

    return df_snip

def generate_processed_windows(df_snip, fs, window_size=10, window_overlap=0.5, channels=[]):
    """
    Generate processed windows from a DataFrame.

    Parameters:
    df_snip (pd.DataFrame): Input DataFrame.
    fs (float): Sampling frequency.
    window_size (int): Size of the window in seconds.
    window_overlap (float): Overlap between windows as a fraction (0.0 to 1.0).
    channels (list of str, optional): List of channels to process.

    Yields:
    pd.DataFrame: Processed windows of the input DataFrame.
    """
    df_gen = split_into_windows(df_snip, fs, window_size=window_size, window_overlap=window_overlap)

    for df_snip_window in df_gen:
        df_snip_window_filtered = pd.Series(full_a_processing(df_snip_window, fs, channels=channels), index=df_snip_window.index, name='mag_filtered')
        df_snip_window = df_snip_window.join(df_snip_window_filtered)
        
        yield df_snip_window

def extract_hr_from_spectrum(signal, fs):

    """
    Extract heart rate (HR) from the spectrum of a signal.

    Parameters:
    signal (array-like): Input signal.
    fs (float): Sampling frequency.

    Returns:
    int: Heart rate in beats per minute (BPM).
    """

    fft_result, freq = compute_fourier(signal, fs)
    frequency_res = fs/len(fft_result)

    # For more robustness, runs a rolling window with 0.1Hz length over the positive axis
    rolling_window_size = int(0.1*len(fft_result)/fs)
    fft_result = np.abs(fft_result)
    fft_result[1:] = np.convolve(fft_result[1:], np.ones(rolling_window_size)/rolling_window_size, mode='same')

    highest_frequencies_index = np.argsort(fft_result)[::-1]

    # filter all frequencies that are lower than (including) 0Hz and higher than 2.5Hz
    highest_frequencies_index = highest_frequencies_index[(0 < highest_frequencies_index) & (highest_frequencies_index < 2.5/frequency_res)]

    highest_frequency = freq[highest_frequencies_index[0]]

    return int(highest_frequency *60)  

def plot_dist_hist(ass, fs, title='',bins = 500, ax=None):
    """
    Plot a histogram of distances between peaks.

    Parameters:
    - ass (numpy.ndarray): An array containing the peak assignments.
    - title (str, optional): A title for the histogram plot (default is an empty string).
    - bins (int, optional): The number of bins to use in the histogram (default is 500).

    Returns:
    - None

    This function calculates the distances between peaks in the input data and creates
    a histogram to visualize their distribution. It also displays the mean distance (MD)
    and the mode of the distribution as text on the plot.

    Example usage:
    >>> plot_dist_hist(assignments, title='Peak Distance Histogram', bins=100)
    """
    if ax == None:
        fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))

    dist = (ass[:,1] - ass[:,0])/fs
    av_dist = np.abs(dist).mean()
    ax.hist(dist, bins=bins)
    ax.set_ylabel('Count')
    ax.set_xlabel('Distance between peaks (s)')
    ax.set_title(title)
    #ax.set_xlim(-0.1, 2.0)

    dist_hist = np.histogram(dist, bins=bins)
    mode_index = np.argmax(dist_hist[0])
    mode = (dist_hist[1][mode_index:mode_index+1]).mean()

    props = dict(boxstyle='round', alpha=0.5)
    textstr = f'mean: {np.round(av_dist,3)} s \nmode: {np.round(np.max(mode),3)} s'
    plt.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=props, horizontalalignment='left')
    
def read_Max_ecg(recording_dir, fs=100, person_id = None):
    """
    Read ECG data from a specified directory.

    Args:
        recording_dir (str): Directory containing the ECG data file.
        fs (int, optional): Sampling frequency in Hz. Defaults to 100.

    Returns:
        pandas.DataFrame: ECG data with columns 'time' and 'bvp', and specified sampling frequency.
    """
    ecg_dir = os.path.join(recording_dir, f'p{person_id}_ecg_movisense.npy')
    # Read wrist accelerometer and gyroscope data from CSV file
    ecg = np.load(ecg_dir, allow_pickle=True).reshape(-1,1)

    df_ecg = pd.DataFrame(data = ecg, columns = ['ecg'])

    # No time given :(
    # sets time index determined by the sampling frequency
    df_ecg['time'] = pd.to_timedelta(np.arange(0,len(df_ecg))/fs, unit='s')
    df_ecg.set_index('time', inplace=True)

    # Set the sampling frequency
    df_ecg.fs = fs

    return df_ecg

def read_Max_chest(recording_dir, fs=100, person_id = None):
    """
    Read chest accelerometer data from a specified directory.

    Args:
        recording_dir (str): Directory containing the chest accelerometer data file.
        fs (int, optional): Sampling frequency in Hz. Defaults to 100.

    Returns:
        pandas.DataFrame: Chest accelerometer data with columns 'time', 'acc_x', 'acc_y', 'acc_z', 'mag',
        and specified sampling frequency.
    """
    acc_chest_dir = os.path.join(recording_dir, f'p{person_id}_acc_movisense.npy')
    # Read wrist accelerometer and gyroscope data from CSV file
    acc_chest = np.load(acc_chest_dir, allow_pickle=True)

    df_acc_chest = pd.DataFrame(data=acc_chest.swapaxes(0,1), columns = ['acc_x', 'acc_y', 'acc_z', 'mag'])
    # No time given :(
    # sets time index determined by the sampling frequency
    df_acc_chest['time'] = pd.to_timedelta(np.arange(0,len(df_acc_chest))/fs, unit='s')
    df_acc_chest.set_index('time', inplace=True)

    # Set the sampling frequency
    df_acc_chest.fs = fs

    return df_acc_chest

def read_Max_wrist(recording_dir, fs=100,person_id=''):
    """
    Read wrist accelerometer and gyroscope data from a specified directory.

    Args:
        recording_dir (str): Directory containing the wrist accelerometer and gyroscope data file.
        fs (int, optional): Sampling frequency in Hz. Defaults to 100.

    Returns:
        pandas.DataFrame: Wrist accelerometer and gyroscope data with columns 'time', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag'.
    """
    acc_wrist_dir = os.path.join(recording_dir, f'p{person_id}_acc_axivity.npy')
    # Read wrist accelerometer and gyroscope data from CSV file
    acc_wrist = np.load(acc_wrist_dir, allow_pickle=True)

    df_acc_wrist = pd.DataFrame(data=acc_wrist.swapaxes(0,1), columns = ['time', 'acc_x', 'acc_y', 'acc_z', 'mag'])
    # samplign frequency should be 100 HZ. Hence, the time should be in microseconds
    df_acc_wrist.drop('time', axis=1, inplace=True)

    # replaces original time index with a new one determined by the sampling frequency
    df_acc_wrist['time'] = pd.to_timedelta(np.arange(0,len(df_acc_wrist))/fs, unit='s')
    df_acc_wrist.set_index('time', inplace=True)

    

    # Set the sampling frequency
    df_acc_wrist.fs = fs
    return df_acc_wrist

def read_starting_time_Max(recording_dir, person_id):
    # Create an empty list to store datetime values
    datetime_list = []
    file_path = os.path.join(recording_dir, f'p{person_id}_start_time.txt')
    # Open the file for reading
    with open(file_path, "r") as file:
        # Iterate over each line in the file
        for line in file:
            # Strip leading/trailing whitespace and convert the line to a datetime object
            try:
                dt_value = datetime.strptime(line.strip(), "%H.%M.%S")  # Modify the format as needed
                datetime_list.append(dt_value)
            except ValueError:
                print(f"Skipping invalid line: {line.strip()}")
    
    assert len(datetime_list) == 1
    return datetime_list[0]

def read_Max(recording_dir, fs=100, person_id:str=None):
    """
    Read and align wrist accelerometer, chest accelerometer, and ECG data from a specified directory.

    Args:
        recording_dir (str): Directory containing the data files.
        fs (int, optional): Sampling frequency in Hz. Defaults to 100.

    Returns:
        tuple of pandas.DataFrames: Two DataFrames with aligned data, one for wrist data and one for chest data.
    """
    # Read wrist accelerometer data
    df_acc_wrist = read_Max_wrist(recording_dir, fs=fs, person_id=person_id)
    # Read chest accelerometer data
    df_acc_chest = read_Max_chest(recording_dir, fs=fs, person_id=person_id)
    # Read ECG data
    df_ecg = read_Max_ecg(recording_dir, fs=fs, person_id=person_id)

    starting_time = read_starting_time_Max(recording_dir, person_id=person_id)
    
    # Align wrist and chest accelerometer data with ECG data by setting the same index
    df_acc_wrist.index = df_acc_wrist.index + starting_time
    df_acc_chest.index = df_acc_chest.index + starting_time
    df_ecg.index = df_ecg.index + starting_time

    # Align wrist and chest accelerometer data with ECG data by setting the same index
    # only true when sampling frequencies are the same
    if len(df_ecg) == len(df_acc_wrist) == len(df_acc_chest):
        df_acc_wrist.index = df_ecg.index
        df_acc_chest.index = df_ecg.index
    
    else:
        raise NotImplementedError('Sampling frequencies are not the same')

    # Concatenate chest accelerometer and ECG data into a single DataFrame
    df_chest = pd.concat([df_acc_chest, df_ecg], axis=1)
    df_chest['time'] = df_chest.index
    df_chest.fs = df_acc_chest.fs

    # Concatenate wrist accelerometer and ECG data into a single DataFrame
    df_wrist = pd.concat([df_acc_wrist, df_ecg], axis=1)
    df_wrist['time'] = df_wrist.index 
    df_wrist.fs = df_acc_wrist.fs

    return df_wrist, df_chest


def read_Apple(file_number, data_dir_Apple, compute_mag=False, fs = None):

    """
    Read and process Apple Watch data for heart rate and motion.

    Args:
        file_number (str): The file number or identifier.
        data_dir_Apple (str): Directory containing Apple Watch data files.

    Returns:
        pandas.DataFrame: Combined and processed data with columns for time, heart rate (hr), accelerometer data,
        and magnitude of acceleration (mag).

    This function reads heart rate and accelerometer data from Apple Watch files, aligns and interpolates them to
    have consistent timestamps, calculates the magnitude of acceleration, and checks the sampling frequencies. The
    resulting DataFrame contains the processed data.

    Example:
    >>> apple_data = read_Apple('file123', '/path/to/apple_data_directory')
    >>> print(apple_data.head())
    """
    file_number = str(file_number)
    hr = pd.read_csv(os.path.join(data_dir_Apple, 'heart_rate', file_number + "_heartrate.txt"), header=None, dtype='float', delimiter=',')
    hr.columns = ['time', 'hr']
    acc = pd.read_csv(os.path.join(data_dir_Apple, 'motion', file_number + "_acceleration.txt"), header=None, dtype='float', delimiter=' ')
    acc.columns = ['time', 'acc_x', 'acc_y', 'acc_z']
    #hr['time'] = pd.to_datetime(hr['time'])
    #acc['time'] = pd.to_datetime(acc['time'])
    # get rid of all the data before the ppg actually starts recording as we dont have accurate labels there
    hr.set_index('time', inplace=True)
    acc.set_index('time', inplace=True)

    if compute_mag:
        acc['mag'] = acc.apply(lambda row: np.sqrt(row['acc_x']**2 + row['acc_y']**2 + row['acc_z']**2), axis=1)
    fs_acc_orig = round(np.median((1/np.diff(acc.index)))) # should be 50
    if fs_acc_orig != 50:
        print(f'Warning: Sampling Frequency of accelerometer is {fs_acc_orig} Hz instead of 50 Hz')
    fs_hr_orig = np.round(np.median((1/np.diff(hr.index))),1) # should be 1
    if fs_hr_orig != 0.2:
        print(f'Warning: Sampling Frquency of heart rate is {fs_hr_orig} Hz instead of 0.2 Hz')

    if fs == None:
        acc.fs = fs_acc_orig
    else:
        acc.fs = fs
    
    # HR data has duplicates in time column. I.e. for one timepoint we get multiple values. We takes the first values here
    hr = hr[~hr.index.duplicated(keep='first')]

    # Puts both HR and ACC data together, by interpolating HR data at ACC timepoints
    df_time = pd.Index(np.arange(acc.index[0], acc.index[-1], 1/acc.fs))
    df_all = acc.reindex(df_time.union(acc.index)).interpolate('linear')
    df_all = df_all.loc[df_time]
    # Interpolates HR data at ACC timepoints linearly
    hr = hr.reindex(df_time.union(hr.index)).interpolate('linear')
    df_all = df_all.join(hr['hr'], how='left')
    df_all = df_all.loc[df_time]


    df_all = df_all[df_all.index > 0]
    df_all.fs = acc.fs
    df_all['time'] = pd.to_datetime(df_all.index, unit='s')

    return df_all



def read_Apple_old(file_number, data_dir_Apple, compute_mag=False, fs = None):

    """
    Read and process Apple Watch data for heart rate and motion.

    Args:
        file_number (str): The file number or identifier.
        data_dir_Apple (str): Directory containing Apple Watch data files.

    Returns:
        pandas.DataFrame: Combined and processed data with columns for time, heart rate (hr), accelerometer data,
        and magnitude of acceleration (mag).

    This function reads heart rate and accelerometer data from Apple Watch files, aligns and interpolates them to
    have consistent timestamps, calculates the magnitude of acceleration, and checks the sampling frequencies. The
    resulting DataFrame contains the processed data.

    Example:
    >>> apple_data = read_Apple('file123', '/path/to/apple_data_directory')
    >>> print(apple_data.head())
    """
    file_number = str(file_number)
    hr = pd.read_csv(os.path.join(data_dir_Apple, 'heart_rate', file_number+ "_heartrate.txt"), header=None, dtype='float', delimiter=',')
    hr.columns = ['time', 'hr']
    acc = pd.read_csv(os.path.join(data_dir_Apple, 'motion', file_number+ "_acceleration.txt"), header=None, dtype='float', delimiter=' ')
    acc.columns = ['time', 'acc_x', 'acc_y', 'acc_z']
    #hr['time'] = pd.to_datetime(hr['time'])
    #acc['time'] = pd.to_datetime(acc['time'])
    # get rid of all the data before the ppg actually starts recording as we dont have accurate labels there

    
    fs_acc_orig = round(np.median((1/np.diff(acc["time"])))) # should be 50
    if fs_acc_orig != 50:
        print(f'Warning: Sampling Frequency of accelerometer is {fs_acc_orig} Hz instead of 50 Hz')
    fs_hr_orig = np.round(np.median((1/np.diff(hr["time"]))),1) # should be 1
    if fs_hr_orig != 0.2:
        print(f'Warning: Sampling Frquency of heart rate is {fs_hr_orig} Hz instead of 0.2 Hz')


    new_time = np.arange(acc["time"].iloc[0], acc["time"].iloc[-1], 1/fs)
    #mask_acc = get_gap_mask(acc["time"], new_time, gap_len=gap_thr)
    #mask_hr = get_gap_mask(hr["time"], new_time, gap_len=gap_thr)
    #mask = mask_acc + mask_hr

    x = np.interp(new_time, acc["time"], acc["acc_x"])
    y = np.interp(new_time, acc["time"], acc["acc_y"])
    z = np.interp(new_time, acc["time"], acc["acc_z"])
    hr = np.interp(new_time, hr["time"], hr["hr"])

    """ if fs != None and fs != fs_acc_orig:
        # we need to perform resampling with equally spaced values
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resample = julius.ResampleFrac(fs_acc_orig, fs).to(DEVICE)

        x = resample(torch.Tensor(x).to(DEVICE).float()).cpu().numpy()
        y = resample(torch.Tensor(y).to(DEVICE).float()).cpu().numpy()
        z = resample(torch.Tensor(z).to(DEVICE).float()).cpu().numpy()
        hr = resample(torch.Tensor(hr).to(DEVICE).float()).cpu().numpy()
        #new_time_r = np.arange(new_time[0], new_time[-1], 1000/fs)
        new_time_r = np.arange(new_time[0], new_time[0] + 1/fs * len(x), 1/fs)[:len(x)]
        #f_interp = interp1d(new_time, mask, kind='nearest',assume_sorted=True, fill_value="extrapolate")
        #mask = f_interp(new_time_r) > 0.5
        new_time = new_time_r """


    #x = x[~mask]
    #y = y[~mask]
    #z = z[~mask]
    #hr = hr[~mask]
    #new_time = new_time[~mask]

    df_all = pd.DataFrame({'time': new_time, 'acc_x': x, 'acc_y': y, 'acc_z': z, 'hr': hr})
    df_all.set_index('time', inplace=True)
    df_all.fs = fs

    df_all['time'] = pd.to_datetime(df_all.index, unit='s')

    if compute_mag:
        df_all['mag'] = df_all[["acc_x", "acc_y", "acc_z"]].pow(2).sum(axis=1).pow(1/2)

    return df_all

def plot_true_pred(hr_true, hr_pred, signal_std=[], signal_threshold=0.05, ax=None, title='', **kwargs):

    """
    Plot the true vs. predicted heart rate values with optional data filtering and statistics.

    Parameters:
        hr_pred (array-like): Predicted heart rate values.
        hr_true (array-like): True heart rate values.
        signal_std (array-like): Signal standard deviation values for data filtering.
        signal_threshold (float, optional): Threshold for signal filtering. Default is 0.05.
        ax (matplotlib Axes, optional): The Axes to draw the plot on. If not provided, a new figure is created.
        title (str, optional): Title for the plot.

    This function creates a scatter plot of true vs. predicted heart rate values, with points
    categorized based on signal standard deviation. It calculates and displays the Mean Absolute
    Error (MAE) and the correlation coefficient for the entire dataset and the low-signal subset.

    Returns:
        None

    Example:
    >>> plot_true_pred(hr_pred, hr_true, signal_std, signal_threshold=0.1, title='Heart Rate Prediction')
    """

    if ax == None:
        fig, ax = plt.subplots()

    split_by_std = len(signal_std) != 0

    if split_by_std:
        thr_h = signal_std > signal_std.max()*signal_threshold
        num_low = (thr_h).sum()/signal_std.count() * 100
    else:
        thr_h = np.array([False] * hr_pred, dtype='bool')
        num_low = len(hr_pred)

    
    hr_true_h = hr_true[thr_h]
    hr_pred_h = hr_pred[thr_h]
    hr_true_l = hr_true[~thr_h]
    hr_pred_l = hr_pred[~thr_h]


    h_args = {'x': hr_true_l, 'y': hr_pred_l, 'alpha': 0.2, 'label': f'std > {signal_threshold} std_max ({np.round(num_low, 2)}%))'}
    l_args = {'x': hr_true_h, 'y': hr_pred_h, 'alpha': 0.2, 'label': f'std < {signal_threshold} std_max ({np.round(100-num_low, 2)}%))'}
    # This allows us to pass some extra plot arguments into the function. Passed arguments overwrite the default ones
    h_args.update(kwargs)
    l_args.update(kwargs)

    ax.scatter(**h_args)
    ax.scatter(**l_args)

    ax.plot([40, 120], [40, 120], color='k', linestyle='-', linewidth=2)
    ax.set_xlabel('True HR (bpm)')
    ax.set_ylabel('Predicted HR (bpm)')
    ax.set_title(title)
    #ax.set_ylim([25, 110])
    #ax.set_xlim([35, 85])
    if split_by_std:
        ax.legend(loc='upper right')

    
    mae = np.round(np.abs(hr_true - hr_pred).mean(), 2)
    mae_l = np.round(np.abs(hr_true_l - hr_pred_l).mean(), 2)
    correlation_coefficient = np.round(np.corrcoef(hr_true, hr_pred)[0, 1],3)
    correlation_coefficient_l = np.round(np.corrcoef(hr_true_l, hr_pred_l)[0, 1],3)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    if split_by_std:
        textstr = f'MAE = {mae} \ncorr = {correlation_coefficient} \nMAE_low = {mae_l} \ncorr_low = {correlation_coefficient_l}'
    else:
        textstr = f'MAE = {mae} \ncorr = {correlation_coefficient}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)

def get_gap_mask(timestamp, new_time, gap_len=1000):
    """
    Generate a boolean mask indicating gaps in time series.

    Parameters:
    - timestamp (numpy.ndarray): Array of timestamps from the original time series.
    - new_time (numpy.ndarray): Array of timestamps for which the gap mask is generated.
    - gap_len (int, optional): Minimum gap length to be considered as a gap. Default is 1000.

    Returns:
    - numpy.ndarray: Boolean mask indicating gaps in the new_time array.
    """

    # Find indices where the time difference exceeds the specified gap length
    gaps = np.where(np.diff(timestamp) > gap_len)[0]

    # Extract start and end values for each identified gap
    gap_values = [(timestamp[gap], timestamp[gap+1]) for gap in gaps]

    # Create an empty boolean mask with the same shape as new_time
    mask = np.empty_like(new_time, dtype=bool)

    # Iterate through gap values and set the corresponding indices in the mask to True
    for i, el in enumerate(gap_values):
        start_i = np.searchsorted(new_time, el[0], side="right")
        end_i = np.searchsorted(new_time, el[1], side="left")
        mask[start_i:end_i] = True

    return mask

def read_parkinson(subject_id, data_file, compute_mag=False, fs=50, gap_thr = 1):
    """
    Read and process Parkinson dataset for accelerometer and heart rate.

    Args:
        subject_id (str): Subject identifier.
        data_file (str): Path to the Parkinson dataset file (HDF5 format).
        compute_mag (bool, optional): Whether to compute the magnitude of acceleration. Defaults to False.
        fs (int, optional): Sampling frequency in Hz. Defaults to 50.

    Returns:
        pandas.DataFrame: Combined and processed data with columns for time, accelerometer data,
        and heart rate.

    This function reads accelerometer and heart rate data from the Parkinson dataset file, aligns and interpolates them to
    have consistent timestamps, resamples it to fs, calculates the magnitude of acceleration (if specified), and checks the sampling frequencies.
    The resulting DataFrame contains the processed data.

    For resampling, the functions uses cuda if available

    Example:
    >>> parkinson_data = read_Parkinson('C_00b3', '/path/to/parkinson_data_file.h5')
    >>> print(parkinson_data.head())
    """
    print(f"Reading data for subject {subject_id}...")
    
    with h5py.File(data_file, "r") as f:
        trace = f[subject_id]

        timestamp_acc = trace['1']['timestamp'][()]

        # we assume that the data is recorded with 14 bit resolution ([-8191,8192]) with the maximum values representing 8g
        # hence, we need to take the values /8192 * 8 to get the acceleration in g
        x = trace['1']['x'][()]/ 8192 * 8
        y = trace['1']['y'][()]/ 8192 * 8
        z = trace['1']['z'][()]/ 8192 * 8

        hr = trace['2']["value"][()]
        hr_timestamp = trace['2']["timestamp"][()]


    fs_acc_orig = round(np.median((1000 / np.diff(timestamp_acc))))
    if fs_acc_orig != 50:
        print(f'Warning: Sampling Frequency of accelerometer is {fs_acc_orig} Hz instead of 50 Hz')

    fs_hr_orig = np.round(np.median((1000 / np.diff(hr_timestamp))), 1)
    if fs_hr_orig != 1:
        print(f'Warning: Sampling Frequency of heart rate is {fs_hr_orig} Hz instead of 1 Hz')

    #print(f"Reading data took {time.time() - current_time} seconds")
    
    #print("Finding gaps in data...")

    # we create a new time index with consistent sampling frequency
    # since the Parkinson dataset has many gaps in the data, we first find these gaps and exclude them from the new time index
    # hence, when we interpolate, we do not interpolate over the gaps
    # these gaps still need to be filtered out later on, when windows are generated
    new_time = np.arange(timestamp_acc[0], timestamp_acc[-1], 1000/fs_acc_orig)
    mask_acc = get_gap_mask(timestamp_acc, new_time, gap_len=gap_thr*1000)
    mask_hr = get_gap_mask(hr_timestamp, new_time, gap_len=gap_thr*10000)

    # the mask is True for all values that are in a gap, hence, there is no information from the recording
    mask = mask_acc + mask_hr

    #print(f"Finding gaps took {time.time() - current_time} seconds")
    
    #print("Interpolating data...")
    time
    # interpolate data to have consistent timestamps
    x = np.interp(new_time, timestamp_acc, x)
    y = np.interp(new_time, timestamp_acc, y)
    z = np.interp(new_time, timestamp_acc, z)
    hr = np.interp(new_time, hr_timestamp, hr)

    #print(f"Interpolating data took {time.time() - current_time} seconds")
    
    #print("Resampling data...")
    if fs != fs_acc_orig:
        # we need to perform resampling with equally spaced values
        #num = int(len(x) * fs / fs_acc_orig)
        #x = resample(x, num)
        #y = resample(y, num)
        #z = resample(z, num)
        #hr = resample(hr, num)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resample = julius.ResampleFrac(fs_acc_orig, fs).to(DEVICE)

        x = resample(torch.Tensor(x).to(DEVICE).float()).cpu().numpy()
        y = resample(torch.Tensor(y).to(DEVICE).float()).cpu().numpy()
        z = resample(torch.Tensor(z).to(DEVICE).float()).cpu().numpy()
        hr = resample(torch.Tensor(hr).to(DEVICE).float()).cpu().numpy()
        #new_time_r = np.arange(new_time[0], new_time[-1], 1000/fs)
        new_time_r = np.arange(timestamp_acc[0], timestamp_acc[0] + 1000/fs * len(hr), 1000/fs)
        f_interp = interp1d(new_time, mask, kind='nearest',assume_sorted=True, fill_value="extrapolate")
        mask = f_interp(new_time_r) > 0.5
        new_time = new_time_r
        
    x = x[~mask]
    y = y[~mask]
    z = z[~mask]
    hr = hr[~mask]
    new_time = new_time[~mask]
    #print(f"Resampling data took {time.time() - current_time} seconds")
    
    #print("Creating DataFrame...")

    df_all = pd.DataFrame({"time": new_time/1000, "acc_x": x, "acc_y": y, "acc_z": z, "hr": hr})
    df_all.fs = fs
    df_all.set_index("time", inplace=True)
    


    # Combine accelerometer and heart rate data
    #print(f"Creating DataFrame took {time.time() - current_time} seconds")
    
    #print("Computing magnitude...")
    if compute_mag:
        df_all['mag'] = df_all[["acc_x", "acc_y", "acc_z"]].pow(2).sum(axis=1).pow(1/2)
    df_all["time"] = pd.to_datetime(df_all.index, unit='s')

    #print(f"Computing magnitude took {time.time() - current_time} seconds")
    

    return df_all

def ensemble_predictions(predictions, signal_std, std_threshold=0.9, kind='linear'):
    """
    Ensembles predictions based on a signal's standard deviation.

    Parameters:
    - predictions (array-like): Array of prediction values.
    - signal_std (array-like): Array of signal standard deviations.
    - std_threshold (float): Threshold for standard deviation below which interpolation is performed.
    - kind (str, optional): Interpolation method ('linear' by default).

    Returns:
    - array-like: Ensembled predictions.
    """

    # Compute the mean of predictions
    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)

    # Ensure the same length for predictions and signal_std
    assert len(predictions) == len(signal_std)

    # Create a mask for interpolation based on signal_std
    interp_mask = signal_std < np.quantile(signal_std, std_threshold)

    if np.sum(interp_mask) != 0:
        xp = np.where(interp_mask)[0]
        fp = predictions[interp_mask]
        x = np.where(~interp_mask)[0]

        # Perform interpolation on selected values
        predictions_interp = interp1d(xp, fp, fill_value=0, kind=kind, bounds_error=False)

        predictions[x] = predictions_interp(x)

    return predictions

def split_train_val_test(X, Y, pid, test_split=0.2, n_val_splits=5, val_split=None, grouped=True, random_state=None):
    """
    Split the data into training, validation, and test sets, with the option of grouped splitting for the test set.

    Parameters:
        X (array-like): The feature data.
        Y (array-like): The target data.
        pid (array-like): An array of grouping labels.
        test_split (float, optional): The proportion of data to be allocated to the test set.
        n_val_splits (int, optional): The number of validation splits to be generated for the training set.
        grouped (bool, optional): If True, grouped splitting is performed based on 'pid' for the test set.
        random_state (int, None, or RandomState, optional): Seed for the random number generator.

    Yields:
        Tuple of numpy arrays: (X_train, Y_train, pid_train, X_val, Y_val, pid_val, X_test, Y_test, pid_test)
            - X_train, Y_train, pid_train: Feature data, target data, and grouping labels for the training set.
            - X_val, Y_val, pid_val: Feature data, target data, and grouping labels for the validation set.
            - X_test, Y_test, pid_test: Feature data, target data, and grouping labels for the test set (if grouped splitting is enabled).

    Notes:
        - If grouped=True, grouped splitting is performed for the test set, ensuring that the same test set is used for each validation split.
        - The number of validation splits is determined by the 'n_val_splits' parameter.

    Example:
        for X_train, Y_train, pid_train, X_val, Y_val, pid_val, X_test, Y_test, pid_test in split_train_val_test(X, Y, pid):
            # Use the data splits for training and evaluation
    """
    # Check if grouped splitting is enabled for the test set
    if grouped:
        groups = pid
        groups = np.squeeze(groups)
        if groups.ndim != 1:
            # if array contains more than information about subject
            # use only the first column
            groups = groups[:,0]
    else:   
        groups = None


    # Generate the test split using GroupShuffleSplit
    if test_split == 0:
        test_indices = []
        train_val_indices = np.arange(len(X))
    else:
        test_splitter = GroupKFold(n_splits=int(1/test_split))
        train_val_indices, test_indices = next(test_splitter.split(X, Y, groups=groups))

    if val_split == None:
        val_split = 1 / n_val_splits

    # Generate validation splits using KFold
    val_splitter = GroupShuffleSplit(n_splits=n_val_splits, test_size=val_split)

    # Retrieve the data for the test set
    X_test, Y_test, pid_test = X[test_indices], Y[test_indices], pid[test_indices]

    # Iterate through the validation splits
    for train_indices, val_indices in val_splitter.split(X[train_val_indices], Y[train_val_indices], groups[train_val_indices]):
        X_train, Y_train, pid_train = X[train_val_indices][train_indices], Y[train_val_indices][train_indices], pid[train_val_indices][train_indices]
        X_val, Y_val, pid_val = X[train_val_indices][val_indices], Y[train_val_indices][val_indices], pid[train_val_indices][val_indices]

        yield X_train, Y_train, pid_train, X_val, Y_val, pid_val, X_test, Y_test, pid_test

def extract_time_snips(df, time1:time, time2:time):
    """
    Extracts time snippets from a DataFrame based on the provided time range.

    Parameters:
    - df (DataFrame): The DataFrame containing a column named 'time' of type datetime.
    - time1 (datetime.time): The start time for the snippet extraction.
    - time2 (datetime.time): The end time for the snippet extraction.

    Returns:
    - DataFrame: A subset of the input DataFrame containing rows where the 'time' column
      falls within the specified time range. If time2 is greater than time1, the function
      extracts snippets within the same day. If time2 is less than time1, it spans across
      different days.
    """
    # If the start time and end time are the same, return the original DataFrame
    if time1 == time2:
        return df
    # only looks at time, not date
    df_time = df['time'].dt.time
    if time2 >= time1: # same day
        df_snip = df[(time1 <= df_time) & (df_time <= time2)]
    else: # different day
        df_snip = df[(time1 <= df_time) | (df_time <= time2)]

    #df_snip.fs = df

    return df_snip

def butterworth_bandpass(signal, fs: float, low, high, order=4, channels=["acc_x", "acc_y", "acc_z"]):
    """
    Apply Butterworth bandpass filter to signal data.

    Args:
    - signal (pd.DataFrame or array-like): Signal data or DataFrame with columns representing different channels.
    - fs (float): Sampling frequency of the signal.
    - low (float): Low cutoff frequency of the bandpass filter in Hz.
    - high (float): High cutoff frequency of the bandpass filter in Hz.
    - order (int, optional): Order of the Butterworth filter. Default is 4.
    - channels (list, optional): List of channel names if signal is a DataFrame. Default is ["acc_x", "acc_y", "acc_z"].

    Returns:
    - filtered_signal (pd.DataFrame or array-like): Filtered signal data or DataFrame with filtered channels.
    """
    # if the input is a dataframe with acceleration signals as columns, apply the filter to each column and return the dataframe
    if isinstance(signal, pd.DataFrame):
        assert all([ch in signal.columns for ch in channels]), "Channels not in signal"
        for ch in channels:
            signal.loc[:, ch] = single_channel_butterworth_bandpass(signal[ch], fs, low, high, order=order)

        return signal
    else:
        signal = single_channel_butterworth_bandpass(signal, fs, low, high, order=order)

        return signal

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

def compute_hr_from_ecg_peaks(ecg, fs, peak_distance=0.8, peak_prominence=0.3):
    """
    Compute heart rate (HR) from the electrocardiogram (ECG) signal based on detected peaks.

    Args:
    - ecg (array-like): Electrocardiogram signal.
    - fs (int): Sampling frequency of the ECG signal.
    - peak_distance (float, optional): Minimum distance between peaks in seconds (default: 0.8).
    - peak_prominence (float, optional): Minimum relative prominence of peaks (default: 0.3).

    Returns:
    - hr_ecg (float): Computed heart rate in beats per minute (bpm) using peak detection.
    """
    # Convert peak_distance to the number of samples
    peak_distance = peak_distance * fs

    # Find peaks in the ECG signal using specified parameters
    peaks_ecg, _ = find_peaks(z_normalization(ecg), prominence=peak_prominence * np.percentile(np.abs(ecg), 50), distance=peak_distance)

    # Calculate heart rate based on the number of peaks detected
    num_seconds = len(ecg) / fs
    hr_ecg = len(peaks_ecg) * 60 / num_seconds

    return hr_ecg

def compute_hr_from_ecg_rr(ecg, fs, hr_thr=200):
    """
    Compute heart rate (HR) from the electrocardiogram (ECG) signal using RR interval analysis.

    Args:
    - ecg (array-like): Electrocardiogram signal.
    - fs (int): Sampling frequency of the ECG signal.
    - hr_thr (int, optional): Threshold for acceptable heart rate values (default: 200).

    Returns:
    - hr_rr (float): Computed heart rate in beats per minute (bpm) using RR interval analysis.
    """
    # Scale and preprocess the ECG signal
    ecg = hp.scale_data(ecg)
    ecg = hp.remove_baseline_wander(ecg, fs)
    ecg = hp.filter_signal(ecg, 0.05, fs, filtertype='notch')
    ecg = resample(ecg, len(ecg) * 2)

    # Process the ECG signal to extract heart rate using HeartPy library
    _, measures = hp.process(ecg, fs * 2, calc_freq=False, hampel_correct=False, bpmmax=160, bpmmin=30, reject_segmentwise=False)

    hr_rr = measures['bpm']

    # Adjust HR if it exceeds the specified threshold
    if hr_rr > hr_thr:
        ecg = hp.enhance_ecg_peaks(hp.scale_data(ecg), fs, aggregation='median', iterations=5)
        _, measures = hp.process(ecg, fs * 2, calc_freq=False, hampel_correct=False, bpmmax=160, bpmmin=30, reject_segmentwise=False)
        hr_rr = measures['bpm']

    return hr_rr

def generate_XY(df_all, 
                fs=50, 
                window_size=10, 
                window_overlap=None,
                step_size=None, 
                channels=['acc_x', 'acc_y', 'acc_z'], 
                compute_hr = None, 
                peaks = None,
                normalize="", 
                hr_index = "mean",
                mask = None, 
                return_time=False,
                return_discarded=False,
                return_ecg=False,
                metrics = [],
                min_beats = 4):
    """
    Generate feature matrix (X) and target array (Y) from the provided dataframe incorporating different heart rate estimation methods.

    Args:
    - df_all (DataFrame): DataFrame containing sensor data including columns for acceleration (acc_x, acc_y, acc_z)
                         and electrocardiogram (ecg).
    - fs (int, optional): Sampling frequency (default: 50).
    - window_size (int, optional): Size of the sliding window in seconds (default: 10).
    - window_overlap (float, optional): Proportion of overlap between consecutive windows (default: 0.5).
    - channels (list, optional): List of sensor channels to consider (default: ['acc_x', 'acc_y', 'acc_z']).
    - sample_normalize (bool, optional): Flag indicating whether to perform sample-wise normalization (default: False).

    Returns:
    - X (ndarray): Feature matrix of shape (samples, window_size*fs, num_channels).
    - Y (ndarray): Target array of computed heart rates.
    """
    assert all([ch in df_all.columns for ch in channels]), 'Not all channels are in dataframe'

    if return_ecg:
        assert 'ecg' in df_all.columns, 'No ecg column in dataframe'


    if step_size is not None:
        window_overlap = 1 - (step_size / window_size)

    num_samples = df_all.shape[0]
    X, Y, start_times, ECG, metr = [], [], [], [], []
    discard_count = 0
    sample_count = 0

    for start_index in np.arange(0, num_samples, int(window_size * fs * (1 - window_overlap))):
        sample_count += 1
        end_index = int(start_index + window_size * fs)

        if end_index > num_samples:
            break

        start_time = df_all.index[start_index]
        end_time = df_all.index[end_index]
        
        if end_time - start_time > window_size * fs * 1.5:
            # discards all windows that have time jumps inside them, i.e. where the difference between start timestamp and end timestamp is much bigger than the window size
            breakpoint()
            discard_count += 1
            continue

        if mask is not None:
            # since mask is computed centrally for each window, we have to consider the central index here
            if mask[int((end_index+start_index)/2)] == False:
                discard_count += 1
                continue
         

        df_snip = df_all.iloc[start_index:start_index + window_size * fs]

        #### X ####
        x = np.array([df_snip[ch].to_numpy() for ch in channels]).swapaxes(0, 1)

        # performs sample-wise normalization if specified
        if normalize == "sample":
            x = StandardScaler(with_mean=True, with_std=True).fit_transform(x)

        #### Y ####
        
        if compute_hr == "ecg":
            # computing hr from ecg
            assert 'ecg' in df_all.columns, 'No ecg column in dataframe'
            ecg = df_snip['ecg'].to_numpy()

            # Calculate heart rate estimates using different methods
            hr_peaks = compute_hr_from_ecg_peaks(ecg, fs)
            hr_freq = extract_hr_from_spectrum(ecg, fs)
            try:
                hr_rr = compute_hr_from_ecg_rr(ecg, fs)
            except Exception as e:
                hr_rr = np.nan

            # Fusion of heart rate estimates and handling discarding criteria
            if hr_rr > 120 or np.isnan(hr_rr):
                discard_count += 1
                continue
            elif abs(hr_peaks - hr_rr) > 10 and abs(hr_freq - hr_rr) > 10:
                discard_count += 1
                continue
            else:
                y = hr_rr

        elif compute_hr == "peaks" or compute_hr == "peaks_median":
            # computing HR from peak indices
            assert peaks is not None, 'No beats provided for heart rate estimation'
            extra_peaks_window = 5 #how many peaks to consider outside the window in seconds
            start_index_hr = max(0, start_index - int(extra_peaks_window * fs))
            end_index_hr = min(num_samples, end_index + int(extra_peaks_window * fs))
            lower_index = np.searchsorted(peaks, start_index_hr, side='left')
            upper_index = np.searchsorted(peaks, end_index_hr, side='right')

            beats_i = peaks[lower_index:upper_index]

            hr_diff = np.diff(beats_i)/100
            hr_diff = hr_diff[(hr_diff < 2) & (hr_diff > 0.3)]
            if len(hr_diff) < min_beats:
                discard_count += 1
                continue
            else:                  
                if compute_hr == "peaks_median":
                    y = 60/np.nanmedian(hr_diff)
                else:
                    y = 60/np.nanmean(hr_diff)

        elif compute_hr == "peaks_hrv":
            # Computing the SDANN
            # See https://www.mdpi.com/1424-8220/21/12/3998 for more
            assert peaks is not None, 'No beats provided for heart rate estimation'
            lower_index = np.searchsorted(peaks, start_index, side='left')
            upper_index = np.searchsorted(peaks, end_index, side='right')

            beats_i = peaks[lower_index:upper_index]

            hr_diff = np.diff(beats_i)/100
            hr_diff = hr_diff[(hr_diff < 2) & (hr_diff > 0.2)]

            if len(hr_diff) < min_beats:
                discard_count += 1
                continue
            else:
                y = np.nanstd(hr_diff)

        else:
            assert 'hr' in df_all.columns, 'No heart rate column in dataframe'
            # computes the heart rate as the mean of all the heart rates in the window
            if hr_index == "mean":
                y = df_snip['hr'].mean()
            elif hr_index == "last":
                y = df_snip['hr'].iloc[-1]

        #### ECG ####
        if return_ecg:
            ecg = df_snip['ecg'].to_numpy()
            ECG.append(ecg)

        #### Timestamps ####
        if isinstance(start_time, pd._libs.tslibs.timestamps.Timestamp):
            start_time = start_time.timestamp()
        start_times.append(start_time)      

        X.append(x)
        Y.append(y)

        #### Metrics ####
        if len(metrics) > 0:
            windowed_metrics = [metric[int((end_index+start_index)/2)] for metric in metrics]
            metr.append(windowed_metrics)


    discard_rate = discard_count / sample_count
    print(f'Discarded {np.round(discard_rate * 100,2)}% of samples')
    X = np.array(X)
    Y = np.array(Y)
    ECG = np.array(ECG)
    metr = np.array(metr)

    assert len(X) == len(Y) == len(start_times)

    if normalize == "dataset" and len(X) != 0:
        scaler = StandardScaler()
        X[:,:,0] = scaler.fit_transform(X[:,:,0])
        X[:,:,1] = scaler.fit_transform(X[:,:,1])
        X[:,:,2] = scaler.fit_transform(X[:,:,2])

    start_times = np.array(start_times)

    return_vars = [X, Y]
    if return_time:
        return_vars.append(start_times)
    if return_discarded:
        return_vars.append(discard_count)
    if return_ecg:
        return_vars.append(ECG)
    if len(metrics) > 0:
        return_vars.append(metr)
    
    return return_vars
    
def generate_X(df_all, 
               fs=50,
                window_size=10, 
                window_overlap=0.5, 
                channels=['acc_x', 'acc_y', 'acc_z'],  
                normalize="", 
                mask=None, 
                return_time=False,
                return_discarded = False,
                metrics = []
                ):
    """
    Generate feature matrix (X) from the provided dataframe

    Args:
    - df_all (DataFrame): DataFrame containing sensor data including columns for acceleration (acc_x, acc_y, acc_z)
    - fs (int, optional): Sampling frequency (default: 50).
    - window_size (int, optional): Size of the sliding window in seconds (default: 10).
    - window_overlap (float, optional): Proportion of overlap between consecutive windows (default: 0.5).
    - channels (list, optional): List of sensor channels to consider (default: ['acc_x', 'acc_y', 'acc_z']).
    - sample_normalize (bool, optional): Flag indicating whether to perform sample-wise normalization (default: False).

    Returns:
    - X (ndarray): Feature matrix of shape (samples, window_size*fs, num_channels).
    """
    assert all([ch in df_all.columns for ch in channels]), 'Not all channels are in dataframe'

    num_samples = df_all.shape[0]
    X, Y, start_times, metr = [], [], [], []
    discard_count = 0
    sample_count = 0

    for start_index in np.arange(0, num_samples, int(window_size * fs * (1 - window_overlap))):

        sample_count += 1
        end_index = int(start_index + window_size * fs)
        if end_index > num_samples:
            break

        if mask is not None:
            # since mask is computed centrally for each window, we have to consider the central index here
            if not mask[int((end_index+start_index)/2)]:
                discard_count += 1
                continue

        start_time = df_all.index[start_index]
        start_times.append(start_time)
        
        df_snip = df_all.iloc[start_index:start_index + window_size * fs]
        x = np.array([df_snip[ch].to_numpy() for ch in channels]).swapaxes(0, 1)

        # performs sample-wise normalization if specified
        if normalize == "sample":
            x = (x - x.mean(axis=0)) / x.std(axis=0)

        X.append(x)

        #### Metrics ####
        if len(metrics) > 0:
            windowed_metrics = [metric[int((end_index+start_index)/2)] for metric in metrics]
            metr.append(windowed_metrics)

    discard_rate = discard_count / sample_count
    print(f'Discarded {np.round(discard_rate * 100,2)}% of samples')
    X = np.array(X)
    metr = np.array(metr)



    if normalize == "dataset" and len(X) != 0:
        scaler = StandardScaler()
        X[:,:,0] = scaler.fit_transform(X[:,:,0])
        X[:,:,1] = scaler.fit_transform(X[:,:,1])
        X[:,:,2] = scaler.fit_transform(X[:,:,2])
        
    start_times = np.array(start_times)

    return_vars = [X]
    if return_time:
        return_vars.append(start_times)
    if return_discarded:
        return_vars.append(discard_count)
    if len(metrics) > 0:
        return_vars.append(metr)
    
    return return_vars
    
def mask_hr_jumps(hr_signal, fs, window_size=10, thr=10, print_ratio=False, visualize=True, return_metric=False):
    """
    Generate a boolean mask indicating valid heart rate values and masking jumps.

    Parameters:
    - hr_signal (pandas.Series): Series containing the heart rate signal.
    - fs (int): Sampling frequency of the heart rate signal.
    - window_size (int, optional): Size of the rolling window for jump detection. Default is 10.
    - thr (int, optional): Threshold for considering a difference in heart rate as a jump. Default is 10.
    - print_ratio (bool, optional): If True, print the ratio of masked (valid) values. Default is False.
    - visualize (bool, optional): Not implemented yet. Placeholder for future visualization. Default is False.

    Returns:
    - np.array: Boolean mask indicating valid heart rate values, masking jumps.
    """

    # Calculate differences between consecutive heart rate values
    hr_diffs = hr_signal.rolling(1 * fs, center=True, step=1*fs, min_periods=1*fs).mean().diff().abs()


    # Use a rolling window to create a mask where True indicates valid heart rate values (no jumps)
    metric = hr_diffs.rolling(window_size, center=True, step=1, min_periods=window_size//2).max().dropna()
    mask = metric < thr

    mask = mask.reindex_like(hr_signal, method="nearest").to_numpy()

    # Print the ratio of masked (valid) values if requested
    if print_ratio:
        print(f"HR jumps: Ratio of masked (valid) values for: {mask.sum() / len(mask) * 100:.2f} %")

    if return_metric:
        return mask, metric.reindex_like(hr_signal, method="nearest").to_numpy()

    if visualize:
        plt.plot(hr_signal)

        # Change time axis for pandas DataFrame with time index
        if isinstance(hr_signal, pd.Series):
            plt.plot(hr_signal.index[np.where(mask)[0]], [hr_signal.mean()] * mask.sum(), '.', markersize=0.005, color='r')
            plt.xticks(rotation=45)
        else:
            plt.xticks(rotation=45)
            plt.plot(np.where(mask)[0], [hr_signal.mean()] * mask.sum(), '.', markersize=0.005, color='r')
        plt.show()

    return mask

def mask_angle_changes(signal, fs, avg_window_size=5, window_size=10, angle_threshold=1, visualize=False, print_ratio=False, return_metric=False):
    """
    Detects angle changes in a signal and creates a mask based on thresholds.
    Took inspiration from van Hess et al. Estimating sleep parameters using an accelerometer without sleep diary

    Args:
    - signal (pd.DataFrame or array-like): Input signal as a pandas DataFrame or array-like structure.
    - fs (float): Sampling frequency of the signal.
    - avg_window_size (int, optional): Size of the averaging window in seconds. Default is 5.
    - angle_threshold (int, optional): Threshold value for angle change detection. Default is 1.
    - visualize (bool, optional): Flag to visualize the masked signal. Default is False.
    - print_ratio (bool, optional): Flag to print the ratio of masked values. Default is False.

    Returns:
    - np.ndarray: Mask representing regions with small angle changes
    """

    # make sure signal is a pandas dataframe
    if not isinstance(signal, pd.DataFrame):
        signal = pd.DataFrame(signal, columns=['acc_x', 'acc_y', 'acc_z'])
    else:
        signal = signal[["acc_x", "acc_y", "acc_z"]].copy()

    def compute_angles(x, y, z):
        # Compute angles from accelerometer readings
        z_angle = np.arctan2(np.sqrt(x**2 + y**2), z) * 180 / np.pi
        y_angle = np.arctan2(np.sqrt(x**2 + z**2), y) * 180 / np.pi
        x_angle = np.arctan2(np.sqrt(y**2 + z**2), x) * 180 / np.pi
        return pd.Series([x_angle, y_angle, z_angle], index=['x_angle', 'y_angle', 'z_angle'])

    # Compute angles using rolling averaging and a step size of 1 second
    angles = signal.rolling(fs * avg_window_size, step=int(fs * 1), center=True, min_periods=1).mean().apply(
        lambda x: compute_angles(x['acc_x'], x['acc_y'], x['acc_z']), axis=1)
    
    # Computes the difference between consecutive angles and takes the absolute value
    # Then computes the rolling average of the absolute differences with a window size of 10 seconds
    # Finally, takes the maximum value of the rolling average for each axis (x,y,z)
    angles_diff = angles.diff().abs().rolling(window_size, step=1, center=True, min_periods=1).mean().max(axis=1)
    mask = angles_diff < angle_threshold

    # Compute mask and upsample it to the original size
    mask = mask.reindex_like(signal, method="nearest").to_numpy()

    # Visualization
    if visualize:
        plt.plot(signal)

        if isinstance(signal, pd.DataFrame):
            # Plot masked values against time for DataFrame inputs
            plt.plot(signal.index[np.where(mask)[0]], [0] * mask.sum(), '.', markersize=0.005, color='r')
            plt.xticks(rotation=45)
        else:
            # Plot masked values for other input types
            plt.plot(np.where(mask)[0], [0] * mask.sum(), '.', markersize=0.005, color='r')
        plt.show()

    # Print the ratio of masked values if requested
    if print_ratio:
        print(f"Angle Changes: Ratio of masked (valid) values: {mask.sum() / len(mask) * 100:.2f} %")

    if return_metric:
        metric = angles_diff.reindex_like(signal, method="nearest").to_numpy()
        return mask, metric
    
    return mask

def mask_valid_windows(signal, fs, avg_window_size=60*30, avg_window_thr=0.008, 
                       max_window_size=10, max_window_thr=0.05, visualize=False, print_ratio=False, return_metric=False):
    """
    Masks invalid windows in a signal based on specified thresholds.

    Args:
    - signal (array-like): Input signal data.
    - fs (float): Sampling frequency of the signal.
    - avg_window_size (int, optional): Size of the average window in seconds. Default is 60*30 (30 minutes).
    - avg_window_thr (float, optional): Threshold for the average window. Default is 0.008.
    - max_window_size (int, optional): Size of the maximum window in seconds. Default is 10.
    - max_window_thr (float, optional): Threshold for the maximum window. Default is 0.05.
    - visualize (bool, optional): Whether to visualize the masking. Default is False.

    Returns:
    - low_mask (array): Masked array indicating valid windows based on thresholds.

    """

    # Calculate window sizes based on sampling frequency
    window_size1 = fs * avg_window_size
    window_size2 = fs * max_window_size

    # Generate the average window and mask based on average threshold
    window1 = np.ones(window_size1) / window_size1
    low_values1 = convolve(np.abs(signal), window1, mode='same')
    low_mask1 = low_values1 < avg_window_thr

    # Generate the mask based on maximum threshold
    window2 = np.ones(window_size2) / window_size2
    #low_values2 = (maximum_filter1d(np.abs(signal), window_size2) - convolve(np.abs(signal), window2, mode='same'))
    low_values2 = maximum_filter1d(np.abs(signal), window_size2)
    low_mask2 = low_values2 < max_window_thr

    # Combine masks to determine valid windows
    low_mask = low_mask1 & low_mask2

    # Calculate ratio of filtered samples and print if requested
    
    if print_ratio:
        print(f"Signal std: Ratio of masked (valid) values: {low_mask.sum() / len(low_mask) * 100:.2f} %")

    # Visualize the masking if requested
    if visualize:
        plt.plot(signal)

        # we need to change the time axis if file is a pandas dataframe with time index
        if isinstance(signal, pd.Series):
            plt.plot(signal.index[np.where(low_mask)[0]], [0] * low_mask.sum(), '.', markersize=0.005, color='r')
            plt.xticks(rotation=45)
        else:
            plt.xticks(rotation=45)
            plt.plot(np.where(low_mask)[0], [0] * low_mask.sum(), '.', markersize=0.005, color='r')
        plt.show()

    if return_metric:
        return low_mask, (low_values1, low_values2)

    return low_mask

def mask_hr(Y, fs, std_threshold=0.001, std_window_size=5, visualize=False, print_ratio=False):
    """
    Masks out high variability regions in a signal based on its standard deviation.

    Args:
    - Y (array-like): Input signal.
    - fs (int or float): Sampling frequency of the signal.
    - std_threshold (float, optional): Threshold for standard deviation. Defaults to 0.001.
    - std_window_size (int, optional): Window size for calculating rolling standard deviation. Defaults to 5.
    - visualize (bool, optional): Whether to visualize the masked regions. Defaults to False.
    - print_ratio (bool, optional): Whether to print the ratio of masked values. Defaults to False.

    Returns:
    - mask (pd.Series): Boolean mask indicating high variability regions in the signal (e.g. valid windows).
    """

    # Calculate the rolling standard deviation
    Y_std = pd.Series(Y).rolling(window=int(std_window_size * fs), min_periods=1).std()
    mask = Y_std > std_threshold
    mask =  mask.to_numpy()

    # Print ratio of masked values if specified
    if print_ratio:
        print(f"HR std: Ratio of masked (valid) values: {mask.sum() / len(mask) * 100:.2f} %")

    # Visualize if specified
    if visualize:
        plt.plot(Y)

        # Change time axis for pandas DataFrame with time index
        if isinstance(Y, pd.Series):
            plt.plot(Y.index[np.where(mask)[0]], [Y.mean()] * mask.sum(), '.', markersize=0.005, color='r')
            plt.xticks(rotation=45)
        else:
            plt.xticks(rotation=45)
            plt.plot(np.where(mask)[0], [Y.mean()] * mask.sum(), '.', markersize=0.005, color='r')
        plt.show()

    return mask

def mask_gaps(df_signal, fs, window_size=10, thr=1, print_ratio=False, visualize=False):
    """
    Generate a boolean mask indicating valid values in a time series with gaps.

    Parameters:
    - df_signal (pandas.DataFrame): DataFrame containing the time series signal.
    - fs (int): Sampling frequency of the time series.
    - window_size (int, optional): Size of the rolling window for gap detection. Default is 10.
    - thr (int, optional): Threshold for considering a time difference as a gap. Default is 1.
    - print_ratio (bool, optional): If True, print the ratio of masked (valid) values. Default is False.
    - visualize (bool, optional): Not implemented yet. Placeholder for future visualization. Default is False.

    Returns:
    - np.array: Boolean mask indicating valid values in the time series.
    """

    # Calculate time differences between consecutive timestamps
    time_diffs = df_signal.index.to_series().diff()

    # Identify gaps based on the specified threshold
    gaps = time_diffs > thr

    # Use a rolling window to create a mask where True indicates valid values (not near a gap)
    mask = gaps.rolling(window_size * fs, center=True, step=1).max() == 0

    # Print the ratio of masked (valid) values if requested
    if print_ratio:
        print(f"Gaps: Ratio of masked (valid) values: {mask.sum() / len(mask) * 100:.2f} %")

    return mask.to_numpy()

def load_matching_recording_M2Sleep(user, session_id, sleep_labels_path:str, data_dir_M2Sleep, sampling_rate=0, compute_mag=True):
    """
    Loads a matching recording from the M2Sleep dataset based on user and session IDs, 
    aligned with specified sleep intervals.

    Args:
    - user (str): User identifier.
    - session_id (int): Session identifier.
    - sleep_labels_path (str): Path to the sleep labels file.
    - data_dir_M2Sleep (str): Directory path containing M2Sleep dataset.
    - sampling_rate (int, optional): Sampling rate of the recording. Defaults to 0.
    - compute_mag (bool, optional): Flag to compute magnitude. Defaults to True.

    Returns:
    - red_df (DataFrame or None): DataFrame containing the recording data within the specified sleep interval.
                                  Returns None if no matching recording is found or if data falls outside the interval.
    """

    # Read sleep labels file and convert timestamp columns to datetime objects
    sleep_labels = pd.read_csv(sleep_labels_path)
    sleep_labels["Start_Timestamp"] = sleep_labels["Start_Timestamp"].apply(lambda x: datetime.strptime(x, "%Y:%m:%d %H:%M:%S:%f"))
    sleep_labels["End_Timestamp"] = sleep_labels["End_Timestamp"].apply(lambda x: datetime.strptime(x, "%Y:%m:%d %H:%M:%S:%f"))

    # Filter sleep session based on user and session ID
    sleep_session = sleep_labels[(sleep_labels["User"] == user) & (sleep_labels["SessionID"] == session_id)]

    # Extract start and end timestamps for the session
    start_timestamp = sleep_session["Start_Timestamp"].iloc[0]
    end_timestamp = sleep_session["End_Timestamp"].iloc[0]

    # Generate recording name based on start timestamp and previous day's timestamp
    recording_name = start_timestamp.strftime("%y%m%d-")
    recording_name_prev = (start_timestamp - timedelta(1)).strftime("%y%m%d-")
    subjects_path = os.path.join(data_dir_M2Sleep, user)

    # Find matching recordings based on generated names in the subject's directory
    matching_recordings = [os.path.join(subjects_path, el) for el in os.listdir(subjects_path) if recording_name in el or recording_name_prev in el]

    if len(matching_recordings) == 0:
        return None

    # Function to read the starting time of a recording from its directory
    def read_start_time(recording_dir):
        starting_time = pd.read_csv(os.path.join(recording_dir, "ACC.csv"), usecols=[0], nrows=1, header=None, dtype="int").iloc[0,0]
        starting_time = datetime.fromtimestamp(starting_time)
        return starting_time

    # Retrieve starting times of matching recordings
    starting_times = [read_start_time(el) for el in matching_recordings]
    time_name_df = pd.DataFrame({"time": starting_times, "name": matching_recordings})    

    # Filter recordings that start before the sleep session and select the closest one
    time_name_df = time_name_df.loc[time_name_df["time"] < start_timestamp]
    time_name_df.sort_values("time", ascending=False, inplace=True)

    # Return None if no matching recordings are found
    if len(time_name_df) == 0:
        return None

    # Select the closest matching recording
    matching_recording = time_name_df.iloc[0]["name"]
    print(f"Selected recording: {matching_recording}")

    # Read the selected recording and perform additional checks based on timestamps
    red_df = read_M2Sleep(matching_recording, sampling_rate=sampling_rate, compute_mag=compute_mag)

    # Return None if recording doesn't cover the entire sleep session interval
    if red_df['time'].iloc[0] > start_timestamp or red_df['time'].iloc[-1] < end_timestamp:
        return None

    # Extract data within the specified sleep interval and return
    snip_df = red_df.loc[(red_df["time"] >= start_timestamp) & (red_df["time"] <= end_timestamp)]
    snip_df.fs = red_df.fs
    return snip_df

def discretize_hr(y, n_bins:int=64):
    """
    Discretizes a continuous heartrate value into a one-hot encoding.
    Assumes that alle y values are in the range [0,1]
    Outputs a shape (n_samples, n_bins) array.
    """

    # define bins
    bins = np.linspace(0, 1, n_bins+1)
    # digitize values to get discrete values
    digitized = np.digitize(y, bins).reshape(-1)
    # creates one-hot encoding of discrete values
    y_onehot = np.eye(n_bins)[digitized - 1]

    return y_onehot

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



def read_Max_v2(data_dir, fs, person_id, drop_seconds = (30,30), take_every_n = 2, compute_mag=False, return_ecg=False):

    assert fs == 100, "Only 100Hz is supported"

    start_index = 100*drop_seconds[0]
    end_index = 100*(120-drop_seconds[-1])

    df_in = pd.read_parquet(os.path.join(data_dir, f"{person_id}.parquet"))
    df_in = df_in.iloc[::take_every_n]

    acc_x = np.concatenate(df_in["acc_a_x"].apply(lambda x: x[start_index: end_index]).to_list())
    acc_y = np.concatenate(df_in["acc_a_y"].apply(lambda x: x[start_index: end_index]).to_list())
    acc_z = np.concatenate(df_in["acc_a_z"].apply(lambda x: x[start_index: end_index]).to_list())

    df_in["beats_m"] = df_in["beats_m"].apply(lambda x: x[(x > start_index*1024/100) & (x < end_index*1024/100)])

    beats_valid = np.concatenate(df_in.apply(lambda row: row["hrv_is_valid"][row["beats_m"]], axis="columns").to_list())

    df_in["beats_m"] = df_in["beats_m"].apply(lambda x : x/1024*100) #since the beats are sampled in 1024Hz, we need to convert it to 100Hz
    df_in["beats_m"] = df_in.apply(lambda row: row["beats_m"] + start_index + row["start_100Hz"], axis="columns")
    beats = np.concatenate(df_in["beats_m"].to_list())
    df_time = np.concatenate([np.arange(start_index, end_index) + s_time for s_time in df_in["start_100Hz"].to_list()])

    df_all = pd.DataFrame({"acc_x": acc_x, "acc_y": acc_y, "acc_z": acc_z, "time": df_time})
    df_all.set_index(df_all["time"].copy(), inplace=True)
    if compute_mag:
        df_all["mag"] = np.concatenate(df_in["acc_a_mag"].apply(lambda x: x[start_index: end_index]).to_list())
    if return_ecg:
        ecg = np.concatenate(df_in["ecg"].apply(lambda x: x[start_index: end_index]).to_list())
        df_all["ecg"] = ecg
    return df_all, beats, beats_valid