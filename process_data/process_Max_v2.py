
import numpy as np
import os
import pandas as pd
import argparse
from datetime import datetime, time, date, timedelta
import matplotlib.pyplot as plt
import pickle 
from sklearn.preprocessing import RobustScaler
from scipy.signal import resample
from sklearn.model_selection import StratifiedKFold
import json


import config
from .data_utils import *
from . import data_metrics


fig_width = 6
fig_height = 6
cmap = plt.get_cmap('tab10')
overwrite = False



def make_split_by_time(data_dir:str, 
                       save_dir:str, 
                       index_gap:int = 1, 
                       test_split:float = 0.2, 
                       n_folds:int = 5) -> dict:
    
    """
    Creates time-based splits for cross-validation from subject metadata and data files.

    Args:
        data_dir (str): Directory containing the "metadata.csv" file with subject metadata.
        save_dir (str): Directory where subject data pickle files are stored.
        index_gap (int, optional): Gap between indices to avoid overlap in splits. Defaults to 1.
        test_split (float, optional): Proportion of data for the test set. Defaults to 0.2.
        n_folds (int, optional): Number of folds for cross-validation. Defaults to 5.

    Returns:
        dict: Dictionary with keys for each fold (0 to n_folds-1). Each fold contains 
              'train', 'val', and 'test' sets mapping subject data file names to timestamp lists.
    """

    metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    metadata = metadata.dropna(subset=["start"])

    subjects = metadata["pid"].to_numpy()

    splits = {}

    for split in range(n_folds):
        splits[split] = {'train': {}, 'val': {}, 'test': {}}

    for sub in subjects:
        sub_file = f"{sub:.0f}_data.pickle"
        with open(os.path.join(save_dir, f"{sub}_data.pickle"), "rb") as f:
            X, Y, pid, metrics = pickle.load(f)
        timestamps = np.array([el[1] for el in pid])
        len_t = len(timestamps)
        if len_t < (1/test_split + n_folds) * (1+index_gap):
            continue
        n_t_folds = n_folds + 1
        folds = np.array([(int(len_t * i/n_t_folds) + index_gap, int(len_t * (i+1)/n_t_folds - 1)) for i in range(n_t_folds)])
        test_index = n_t_folds//2 # take middle fold for testing
        test_indices = folds[test_index] 
        val_train_indices = np.array([folds[i] for i in range(n_t_folds) if i != test_index])
        np.random.shuffle(val_train_indices)
        for split in range(n_folds):
            splits[split]["test"][sub_file] = timestamps[test_indices].tolist()
            splits[split]["val"][sub_file] = timestamps[val_train_indices[split]].tolist()
            train_indices = np.array([el for i, el in enumerate(val_train_indices) if i != split])
            splits[split]["train"][sub_file] = timestamps[train_indices].tolist()

    return splits


def make_split_by_subject(data_dir:str,
                            n_splits:int = 5,
                            random_seed = None,
                            test_ids = [2, 6, 15, 20],
                            ) -> dict:
    
    """
    Creates subject-based splits for cross-validation, ensuring stratification by sex.

    Args:
        data_dir (str): Directory containing the "metadata.csv" file.
        n_splits (int, optional): Number of folds for cross-validation. Defaults to 5.
        random_seed (int, optional): Seed for random number generator. Defaults to None.
        test_ids (list, optional): List of subject IDs to use for the test set. Defaults to [2, 6, 15, 20].

    Returns:
        dict: Dictionary with keys for each fold (0 to n_splits-1). Each fold contains 
              'train', 'val', and 'test' sets mapping subject data file names to subject IDs.
    """


    metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    metadata = metadata.dropna(subset=["start"])
    to_string = lambda x: [f"{el:.0f}_data.pickle" for el in x]
    train_val_inds = metadata[~metadata["pid"].isin(test_ids)].index

    train_val_groups = metadata["sex"].loc[train_val_inds].to_numpy()

    splits = {}

    for i, (train, val) in enumerate(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed).split(train_val_inds, train_val_groups)):
        splits[i] = {
            "train": to_string(metadata.iloc[train]["pid"].tolist()),
            "val": to_string(metadata.iloc[val]["pid"].tolist()),
            "test": to_string(metadata.loc[test_ids]["pid"].tolist())
        }

    return splits

def extract_hr(
        indices, 
        beats, 
        min_beats:int = 4
        ) -> float:

    """
    Extracts the heart rate (HR) from beat timestamps within specified indices.

    Args:
        indices (array-like): Indices specifying the time window for HR extraction.
        beats (array-like): Timestamps of heartbeats.
        min_beats (int, optional): Minimum number of beats required to calculate HR. Defaults to 4.

    Returns:
        float: Calculated heart rate (HR) in beats per minute, or NaN if insufficient data.
    """

    ind_min = np.min(indices)
    ind_max = np.max(indices)

    lower_index = np.searchsorted(beats, ind_min, side='left')
    upper_index = np.searchsorted(beats, ind_max, side='right')

    beats_i = beats[lower_index:upper_index]

    hr_i_diff = np.diff(beats_i)/100
    hr_i_diff = hr_i_diff[(hr_i_diff < 2) & (hr_i_diff > 0.2)]
    if len(hr_i_diff) < min_beats:
        return np.nan
    hr_i = 60/np.nanmean(hr_i_diff)

    return hr_i



def plot_signal_and_hr(df_all):

    import matplotlib.dates as mdates
    fig, ax = plt.subplots(1,1, figsize=(fig_width, fig_height))
    time_values = pd.to_datetime(df_all["time"]/100, unit="s")
    twin1 = ax.twinx()
    ax.plot(time_values, df_all["hr"], label="HR", color=cmap(0))
    twin1.plot(time_values, df_all["mag"], label="Magnitude", color=cmap(1))
    ax.set_xlabel("Time")
    ax.set_ylabel("HR [bpm]")
    twin1.set_ylabel("Magnitude")

    ax.set_xticklabels(time_values, rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # set rotation of x-axis labels
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = twin1.get_legend_handles_labels()

    # Setting up the legend
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

#plot_signal_and_hr(df_all)




def process_Max(params):

    os.makedirs(params["save_dir"], exist_ok=True)

    files = list(os.listdir(params["data_dir"]))
    files = [el.replace(".parquet", "") for el in files if el.endswith(".parquet")]
    files.sort()

    person_ids = np.unique(files)
    person_ids.sort()
    protocol = []

    for person_id in person_ids:
        print(f'Processing {person_id}')
        if os.path.exists(os.path.join(params["save_dir"], f'{person_id}_data.pickle')) and not overwrite:
            print(f'{person_id} already processed')
            continue

        # reads in the chest and wrist data, only uses wirst data here
        df_all, beats, beats_valid = read_Max_v2(params["data_dir"], fs = params["fs"], person_id=person_id, return_ecg=True, compute_mag=True)

        beats_masked = beats.copy()
        beats_masked[~beats_valid.astype(bool)] = np.nan

        # computes the final mask as intersection of the two masks
        # each mask has True for valid windows
        mask = np.ones(len(df_all), dtype=bool)
        metric_funcs = [data_metrics.metric_angle_changes, data_metrics.metric_max, data_metrics.metric_std, data_metrics.metric_mean, data_metrics.metric_mean_amplitude_deviation]
        metrics = [metric_func(df_all, df_all["mag"], params["fs"], window_size=params["window_size"], step_size=1) for metric_func in metric_funcs]
        # bandpass filters the signals and normalizes them
        df_all = butterworth_bandpass(df_all, params["fs"], 0.1, 18, order=4, channels=["acc_x", "acc_y", "acc_z"])
        #df_all = butterworth_bandpass(df_all, params["fs"], 0.1, 50, order=4, channels=["ecg"])
        df_all = df_normalization(df_all, channels=["acc_x", "acc_y", "acc_z", "ecg"], method='z-score')
        # take 10s windows and appends them to X
        X_sub, Y_sub, start_times, discard_count, ECG_sub, metrics = generate_XY(df_all, 
                                                            fs = params["fs"], 
                                                            window_size = params["window_size"], 
                                                            step_size = params["step_size"], 
                                                            channels = params["channels"],
                                                            normalize = params["normalize"], 
                                                            mask = mask, 
                                                            return_time = True, 
                                                            compute_hr = params["compute_hr"],
                                                            peaks = beats_masked, 
                                                            min_beats = 4,
                                                            return_discarded = True,
                                                            return_ecg = True,
                                                            metrics=metrics)
        # encodes the person id as int (using ascii values of the characters in the string), for torch datalaoder
        pid_sub = [(int(person_id),t) for t in start_times]
        assert len(X_sub) == len(Y_sub) == len(pid_sub)

        # saves the acc and hr data
        with open(os.path.join(params["save_dir"], f'{person_id}_data.pickle'), 'wb') as f:
            pickle.dump((X_sub, Y_sub, pid_sub, metrics), f)

        # saves the ecg data (only for reconstruction)
        with open(os.path.join(params["save_dir"], f'{person_id}_ecg.pickle'), 'wb') as f:
            pickle.dump(ECG_sub, f)

        protocol.append((person_id, df_all['time'].iloc[0], df_all['time'].iloc[-1], discard_count, len(X_sub)))
        
    # saves a protocol file of the data, and how many windows were discarded
    protocol_df = pd.DataFrame(protocol, columns=['person_id', 'z', 'end_time', 'discard_count', 'num_windows'])
    protocol_df.to_csv(os.path.join(params["save_dir"], 'protocol.csv'), index=False)

    print(f"Saved data to {params['save_dir']}")

    if params["make_split"]:

        splits_sub = make_split_by_subject(params["data_dir"], n_splits=5, random_seed=params["random_seed"])
        with open(os.path.join(params["save_dir"], "splits.json"), "w") as f:
            json.dump(splits_sub, f)


        splits_time = make_split_by_time(params["data_dir"], params["save_dir"])
        with open(os.path.join(params["save_dir"], "splits_by_time.json"), "w") as f:
            json.dump(splits_time, f) 
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your function")
    parser.add_argument("--save_dir", type=str, default=config.data_dir_Max_processed_v2, help="Save directory")
    parser.add_argument("--data_dir", type=str, default=config.data_dir_MaxDataset_v2, help="Data directory")
    parser.add_argument("--fs", type=int, default=100, help="Sampling frequency")
    parser.add_argument("--window_size", type=int, default=10, help="Window size")
    parser.add_argument("--step_size", type=int, default=8, help="Step size")
    parser.add_argument("--cut_off_freqs", type=float, nargs='+', default=[0.1, 18], help="Cutoff frequencies")
    parser.add_argument("--order", type=int, default=4, help="Order")
    parser.add_argument("--channels", type=str, nargs='+', default=['acc_x', 'acc_y', 'acc_z'], help="Channels")
    parser.add_argument("--normalize", type=str, default="sample", help="Normalize")
    parser.add_argument("--make_split", action="store_true", help="Make split")
    parser.add_argument("--random_seed", type=int, default=10, help="Random seed")
    parser.add_argument("--compute_hr", type=str, default="peaks", help="Compute HR", choices=["peaks", "peaks_hrv", "peaks_median"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    process_Max(vars(args))

