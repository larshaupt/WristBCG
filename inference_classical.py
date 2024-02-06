#%%
import os
import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime

import pickle
import config
import importlib

import sys
sys.path.append("/local/home/lhauptmann/thesis/t-mt-2023-WristBCG-LarsHauptmann/source")
import data_utils
import heartpy as hp
import wandb
import ssa_hr
from scipy.signal import find_peaks, find_peaks_cwt

#%%


def load_subjects(dataset_dir, subject_paths):
    X, Y, pid, metrics = [], [], [], []
    for sub in subject_paths:
        with open(os.path.join(dataset_dir, sub), "rb") as f:
            data = pickle.load(f)
            if len(data) == 4:
                X_sub, Y_sub, pid_sub, metrics_sub = data
            else:
                X_sub, Y_sub, pid_sub = data
                metrics_sub = None

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
# %%

results_dir = config.classical_results_dir

dataset = "max"


if dataset == "max":
    dataset_dir = config.data_dir_Max_processed
else:
    NotImplementedError



#%%


for split in range(1,5):

    X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = load_dataset(dataset_dir, split)

    config_dict = {"peak_distance": 0.5, 
                "peak_prominence": 0.3, 
                "framework": "SSA",
                "dataset": dataset,
                "split": split,}

    if config_dict["framework"] == 'Bioglass':
        compute_hr = compute_hr_bioglass
    elif config_dict["framework"] == 'SSA':
        compute_hr = compute_hr_ssa

    # Initialize WandB with your project name and optional configuration
    wandb.init(project="hr_results", config=config_dict, group=config_dict["framework"], mode="online")
    hr_rr_val = []
    for X in X_val:
        hr_rr = compute_hr(X, fs=100)
        hr_rr_val.append(hr_rr)
    results_df_val = pd.DataFrame({"y_true": Y_val, "hr_pred": hr_rr_val})
    corr_val = results_df_val.corr().iloc[0,1]
    results_df_val["diff_abs"] = (results_df_val["y_true"] - results_df_val["hr_pred"]).abs()
    mae_val = results_df_val.dropna()["diff_abs"].mean()

    wandb.log({"Val_Corr": corr_val, "Val_MAE": mae_val})

    hr_rr_test = []
    for X in X_test:
        hr_rr = compute_hr(X, fs=100)

        hr_rr_test.append(hr_rr)    
    results_df_test = pd.DataFrame({"y_true": Y_test, "hr_pred": hr_rr_test})
    corr_test = results_df_test.corr().iloc[0,1]
    results_df_test["diff_abs"] = (results_df_test["y_true"] - results_df_test["hr_pred"]).abs()
    mae_test = results_df_test.dropna()["diff_abs"].mean()

    wandb.log({"Test_Corr": corr_test, "Test_MAE": mae_test})

    # Finish the run
    wandb.finish()

# %%


""" X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = load_dataset(dataset_dir, 1)
fs = 100

X = X_val[0] 
df_snip_processed = data_utils.full_a_processing(pd.DataFrame(X, columns = ["acc_x", "acc_y", "acc_z"]), fs)
filtered_signal = df_snip_processed["mag_filtered"]

import heartpy as hp


filtered_signal = hp.scale_data(filtered_signal)
filtered_signal = hp.remove_baseline_wander(filtered_signal, fs)
filtered_signal = hp.filter_signal(filtered_signal, 0.05, fs, filtertype='notch')
#ecg = hp.enhance_ecg_peaks(hp.scale_data(ecg), fs, aggregation='median', iterations=5)

working_data, measures = hp.process(filtered_signal, fs, report_time=False)
hr_rr = measures["bpm"] """

# %%
""" import ssa_hr
# %%
import importlib
importlib.reload(ssa_hr)
signal = X_val[200]
#signal = np.apply_along_axis(data_utils.butterworth_bandpass, 0, signal, 100, 0.5, 10)
filtered_signal = ssa_hr.do_ssa(signal, lagged_window_size=501, tau=0.6, main_axis="y")
filtered_signal = data_utils.butterworth_bandpass(filtered_signal, 100, 0.5, 2)
plt.plot(filtered_signal)
# %%
filtered_signal_b = data_utils.butterworth_bandpass(filtered_signal, 100, 0.5, 3)
plt.plot(filtered_signal_b)
# %%
X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = load_dataset(dataset_dir, 1)
df_X = pd.DataFrame(X_val[300], columns = ["acc_x", "acc_y", "acc_z"])
signal_filtered = data_utils.full_a_processing(df_X, 100)
# %%
df_X.plot()
plt.xlabel("Time [ms]")
plt.ylabel("Acceleration")
# %%
signal_filtered["mag_filtered"].plot()
plt.xlabel("Time [ms]")
plt.ylabel("Acceleration")



# %%
working_data, measures = hp.process(signal_filtered["mag_filtered"], 100, report_time=False)
# plot peaks
peaks = working_data["peaklist"]
plt.plot(signal_filtered["mag_filtered"])
plt.plot(peaks, signal_filtered["mag_filtered"][peaks], "x")
# %%
hp.plotter(working_data, measures)
# %%
i = 1400
hr_ssa = compute_hr_ssa(X_val[i], fs=100, tau=0.6, lagged_window_size=501,first_n_components=10, main_axis="z")
print(hr_ssa, Y_val[i])
# %% """
#%%
""" X = pd.DataFrame(X_test[400], columns=["acc_x", "acc_y", "acc_z"])
fs|=100
df_snip_processed = data_utils.full_a_processing(X, fs)
filtered_signal = df_snip_processed["mag_filtered"]
# %%

peaks = find_peaks_cwt(filtered_signal, np.arange(5, 100))
plt.plot(filtered_signal)
plt.plot(peaks, filtered_signal[peaks], "x")
# %%
X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = load_dataset(dataset_dir, 0)


#%%
X_processed = []
for  i, (X, Y) in enumerate(zip(X_test, Y_test)):
    X = pd.DataFrame(X, columns=["acc_x", "acc_y", "acc_z"])
    df_snip_processed = data_utils.full_a_processing(X, 100)
    X_processed.append(df_snip_processed)

#%%
max_width = 80
min_steps = [2, 5,10]
from scipy.signal import find_peaks_cwt
fs = 100
results_4 = {}
min_width = 0



for  i, (X, Y) in enumerate(zip(X_processed, Y_test)):
    
    for min_step in min_steps:
        print(min_width)
        peaks = find_peaks_cwt(X["mag_filtered"], np.arange(min_width, max_width, min_step))
        rr = np.diff(peaks)
        hr = 60*fs/np.mean(rr)
        results_4[(i, min_step)] = (hr, Y)
# %%
results_df = pd.DataFrame(results_3).T
results_df.columns = ["hr", "y_true"]
results_df.index.names = ["i", "min_width"]
results_df = results_df.reset_index().set_index("i")
# compute correlation for each group between y_true and hr
corr = results_df.groupby("min_width").apply(lambda x: x["hr"].corr(x["y_true"]))
# compute mae
mae = results_df.groupby("min_width").apply(lambda x: np.abs(x["hr"]-x["y_true"]).mean())
corr,mae
# %%
"""