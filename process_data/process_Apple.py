

import numpy as np
import os
import pandas as pd
from datetime import datetime, time, date
import matplotlib.pyplot as plt
from scipy.signal import convolve, butter, sosfilt, find_peaks, sosfiltfilt
from scipy.ndimage import maximum_filter1d
import config
import pickle
import argparse

from .data_utils import *
from . import data_metrics

overwrite = False
            

def process_Apple(params):
    protocol = []
    params["data_dir"] = config.data_dir_Apple


    if params["step_size"] == 8 and params["window_size"] == 10:
        params["save_dir"] = config.data_dir_Apple_processed_all

    elif params["step_size"] == 1:
        if params["window_size"] == 10:
            params["save_dir"] = config.data_dir_Apple_processed_100hz_10w_1s
        else:
            raise ValueError("Window size not recognized")
        
    elif params["step_size"] == 2:
        if params["window_size"] == 10:
            params["save_dir"] = config.data_dir_Apple_processed_100hz_10w_2s
        else:
            raise ValueError("Window size not recognized")

    elif params["step_size"] == 4:
        if params["window_size"] == 5:
            params["save_dir"] = config.data_dir_Apple_processed_100hz_5w_4s
        elif params["window_size"] == 8:
            params["save_dir"] = config.data_dir_Apple_processed_100hz_8w_4s
        elif params["window_size"] == 10:
            params["save_dir"] = config.data_dir_Apple_processed_100hz_10w_4s
        elif params["window_size"] == 30:
            params["save_dir"] = config.data_dir_Apple_processed_100hz_30w_4s
        elif params["window_size"] == 60:
            params["save_dir"] = config.data_dir_Apple_processed_100hz_60w_4s
        else:
            raise ValueError("Window size not recognized")
        
    elif params["step_size"] == 8:
        if params["window_size"] == 10:
            params["save_dir"] = config.data_dir_Apple_processed_100hz_10w_8s
        else:
            raise ValueError("Window size not recognized")

        
    elif params["step_size"] == 10:
        if params["window_size"] == 10:
            params["save_dir"] = config.data_dir_Apple_processed_100hz_10w_10s
        else:
            raise ValueError("Window size not recognized")

    else: # step size anything else
        raise ValueError("Step size not recognized")

    print(params)

    filenames = os.listdir(os.path.join(params["data_dir"], 'motion'))
    file_numbers = [el.replace("_acceleration.txt","") for el in filenames]
    file_numbers.sort()

    os.makedirs(params["save_dir"], exist_ok=True)
    for sub in file_numbers:

        print(f'Processing {sub}')
        if os.path.exists(os.path.join(params["save_dir"], f'{sub}_data.pickle')) and not overwrite:
            print(f'{sub} already processed')
            continue

        df_all = read_Apple(sub, params["data_dir"], fs = params["fs"], compute_mag=True)
        fs = df_all.fs
        df_all = extract_time_snips(df_all, params["start_time"], params["end_time"])


        masking_signal = df_all["mag"].copy()
        masking_signal = butterworth_bandpass(masking_signal, fs, params["cut_off_freqs"][0], params["cut_off_freqs"][1], order=params["order"], channels=params["channels"])
        masking_signal = masking_signal - masking_signal.mean()

        mask_hr_low = mask_hr(df_all["hr"], fs, print_ratio=True, std_threshold=params["hr_std_thr"], std_window_size=100)
        mask = mask_hr_low 

        metric_funcs = [data_metrics.metric_angle_changes, data_metrics.metric_max, data_metrics.metric_std, data_metrics.metric_mean, data_metrics.metric_mean_amplitude_deviation]
        metrics = [metric_func(df_all, df_all["mag"], params["fs"], window_size=params["window_size"], step_size=1) for metric_func in metric_funcs]

        df_all = butterworth_bandpass(df_all, fs, params["cut_off_freqs"][0], params["cut_off_freqs"][1], order=params["order"], channels=["acc_x", "acc_y", "acc_z"])

        # take windows and appends them to X
        X_sub, Y_sub, starting_times, discard_count, metrics = generate_XY(
                    df_all, 
                    fs = fs, 
                    window_size = params["window_size"], 
                    step_size = params["step_size"],
                    channels = params["channels"],
                    compute_hr= params["compute_hr"],
                    normalize= params["normalize"], 
                    mask = mask, 
                    return_time=True, 
                    return_discarded=True,
                    metrics=metrics)

        print("Number of windows: ", len(X_sub))
        print("Number of discarded windows: ", discard_count)
        pid_sub = [(int(sub), int(starting_time)) for starting_time in starting_times]


        protocol.append([sub, df_all['time'].iloc[0], df_all['time'].iloc[-1], discard_count, len(X_sub)])
        
        with open(os.path.join(params["save_dir"], f'{sub}_data.pickle'), 'wb') as f:
            pickle.dump((X_sub, Y_sub, pid_sub, metrics), f)

    protocol_df = pd.DataFrame(protocol, columns=['subject', 'start_time', 'end_time', 'discard_count', 'num_windows'])

    with open(os.path.join(params["save_dir"], 'protocol.pickle'), 'wb') as f:
        pickle.dump((params, protocol_df), f)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your function")
    parser.add_argument("--save_dir", type=str, default="", help="Save directory")
    parser.add_argument("--data_dir", type=str, default=config.data_dir_Apple, help="Data directory")
    parser.add_argument("--fs", type=int, default=100, help="Sampling frequency")
    parser.add_argument("--window_size", type=int, default=10, help="Window size")
    parser.add_argument("--step_size", type=int, default=8, help="Step size")
    parser.add_argument("--start_time", type=lambda s: datetime.strptime(s, "%H:%M").time(), default=time(22, 0), help="Start time")
    parser.add_argument("--end_time", type=lambda s: datetime.strptime(s, "%H:%M").time(), default=time(22, 0), help="End time")
    parser.add_argument("--cut_off_freqs", type=float, nargs='+', default=[0.1, 18], help="Cutoff frequencies")
    parser.add_argument("--order", type=int, default=4, help="Order")
    parser.add_argument("--channels", type=str, nargs='+', default=['acc_x', 'acc_y', 'acc_z'], help="Channels")
    parser.add_argument("--compute_hr", action="store_true", help="Compute HR")
    parser.add_argument("--normalize", type=str, default="sample", help="Normalize")
    parser.add_argument("--hr_std_thr", type=float, default=0.0001, help="HR standard deviation threshold")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    process_Apple(vars(args))

