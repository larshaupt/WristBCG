
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import config
import .data_utils as data_utils
import pickle
import torch
import julius
from datetime import timedelta
from scipy.interpolate import interp1d
data_dir_Capture24 = config.data_dir_Capture24
overwrite = True

# %%
subjects = [el for el in  os.listdir(data_dir_Capture24) if el.endswith(".csv.gz") and el.startswith("P")]
subjects.sort()
fs_acc_orig = 100 # original sampling frequency
fs = 100 # sampling frequency 100 Hz
protocol = []
save_folder = config.data_dir_Capture24_processed_all
os.makedirs(save_folder, exist_ok=True)

for sub_name in subjects:
    sub_path = os.path.join(data_dir_Capture24, sub_name)
    sub_name = sub_name.split(".")[0]
    save_file = os.path.join(save_folder, f'{sub_name}_data.pickle')

    if os.path.exists(save_file) and not overwrite:
        print(f'{sub_name} already exists. Skipping...')
        continue

    print(sub_name)

    # read and process data
    df_all = pd.read_csv(sub_path, dtype={"time": str, "annotation": str})
    #df_all = df_all[df_all["annotation"].apply(lambda x: isinstance(x,str) and "sleeping" in x.lower())]
    df_all.drop(columns=["annotation"], inplace=True)
    df_all.index = pd.to_datetime(df_all["time"])
    df_all["time"] = pd.to_datetime(df_all["time"])
    df_all.rename({"x": "acc_x", "y": "acc_y", "z": "acc_z"}, axis=1, inplace=True)

    if fs != fs_acc_orig:
        # we need to perform resampling with equally spaced values
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resample = julius.ResampleFrac(fs_acc_orig, fs).to(DEVICE)

        x = df_all["acc_x"].values
        y = df_all["acc_y"].values
        z = df_all["acc_z"].values

        x = resample(torch.Tensor(x).to(DEVICE).float()).cpu().numpy()
        y = resample(torch.Tensor(y).to(DEVICE).float()).cpu().numpy()
        z = resample(torch.Tensor(z).to(DEVICE).float()).cpu().numpy()
        #new_time_r = np.arange(new_time[0], new_time[-1], 1000/fs)
        new_time_r = pd.Series([df_all["time"].iloc[0] + i* timedelta(seconds=1/fs) for i in range(len(x))])
        new_time = new_time_r

        df_all = pd.DataFrame({"acc_x": x, "acc_y": y, "acc_z": z}, index=new_time)

    # computes a mask to filter out invalid windows
    signal = df_all.apply(lambda x: np.sqrt(x["acc_x"]**2 + x["acc_y"]**2 + x["acc_z"]**2), axis=1)
    signal = data_utils.butterworth_bandpass(signal, fs, 0.1 ,18)
    signal_mask, (metric_avg, metric_max) = data_utils.mask_valid_windows(signal, fs, avg_window_size=60*30, avg_window_thr=1, 
                        max_window_size=10, max_window_thr=0.05, visualize=False, return_metric=True)
    
    # filters data
    df_all = data_utils.butterworth_bandpass(df_all, fs, 0.1, 18,  order = 4)
    # normalizes data
    #df_all = data_utils.df_normalization(df_all, ['acc_x', 'acc_y', 'acc_z'], method='z-score')
    # take 10s windows and appends them to X
    X, start_times, discard_count, metrics = data_utils.generate_X(df_all, 
                                            fs = fs, 
                                            window_size = 10, 
                                            window_overlap = 0.2, 
                                            channels = ['acc_x', 'acc_y', 'acc_z'], 
                                            normalize="sample", 
                                            mask=None, 
                                            return_time = True,
                                            return_discarded=True,
                                            metrics = [metric_avg, metric_max]
                                            )
    # creates metadata of sample: (subject_name, timestamp)
    pid = [(int(sub_name[1:]), t.timestamp()) for t in start_times]
    pid = np.array(pid)
    # appends metadata and discard count to protocol
    protocol.append([sub_name, df_all.index[0], df_all.index[-1], discard_count, len(X)])


    with open(save_file, 'wb') as f:
        pickle.dump((X, pid), f)
    print(f'Saved {sub_name} to {save_folder}')

# saves protocol
protocol = pd.DataFrame(protocol, columns=['subject', 'start', 'end', 'discard_count', 'n_samples'])
protocol.to_csv(os.path.join(save_folder, 'protocol.csv'), index=False)





