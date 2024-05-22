import numpy as np
import os
import pandas as pd
from datetime import datetime, time, date
import matplotlib.pyplot as plt
from scipy.signal import convolve, butter, sosfilt, find_peaks, sosfiltfilt
from scipy.ndimage import maximum_filter1d
from data_utils import *
import config
import pickle


data_dir_M2Sleep = config.data_dir_M2Sleep
analysis_dir = config.analysis_dir_M2Sleep
save_folder = config.data_dir_M2Sleep_processed_100Hz


fig_width = 6
fig_height = 6
cmap = plt.get_cmap('tab10')

overwrite = False


# Reads in the Apple Dataset, resamples it and saves it as npy file for ML training

subjects = [el for el in os.listdir(data_dir_M2Sleep) if 'S' in el]
subjects.sort()

protocol = []
sleep_label_path = os.path.join(data_dir_M2Sleep,"../" "Selfreports", "final_labels.csv")
sleep_labels = pd.read_csv(sleep_label_path)


for sub_name in subjects:
    subject_dir = os.path.join(data_dir_M2Sleep, sub_name)

    if os.path.exists(os.path.join(save_folder, f'{sub_name}_data.pickle')) and not overwrite:
        print(f'{sub_name} already processed')
        continue

    print(f'Processing {sub_name}')
    recordings = [el for el in list(os.listdir(subject_dir)) if el[0] != '.']
    recordings.sort()
    
    X = []
    Y = []
    pid = []

    sessionIDs = sleep_labels[sleep_labels["User"]==sub_name]["SessionID"].to_list()
    sessionIDs.sort()

    for sessionID in sessionIDs:
        
        print(f'Processing {sessionID} from {sub_name}')

        df_all = load_matching_recording_M2Sleep(sub_name, sessionID, sleep_label_path, data_dir_M2Sleep, compute_mag=True, sampling_rate=100)
        if df_all is None:
            continue
        fs = df_all.fs

        masking_signal = df_all['mag']
        masking_signal = butterworth_bandpass(masking_signal, fs, 0.1 ,18)
        masking_signal = masking_signal/64
        masking_signal = masking_signal - masking_signal.mean()

        low_mask = mask_valid_windows(masking_signal, fs, avg_window_size = 60*30, avg_window_thr = 0.008, max_window_size = 10, max_window_thr = 0.01, visualize=False)
        hr_mask = mask_hr(df_all["hr"], fs, std_threshold=0.0001, std_window_size=100, visualize=True, print_ratio=False)

        mask = np.logical_and(low_mask, hr_mask)

        f_all = butterworth_bandpass(df_all, fs, 0.1, 18,  order = 4)
        # take 10s windows and appends them to X
        #df_all = df_normalization(df_all, ['acc_x', 'acc_y', 'acc_z'], method='z-score')
        # take 10s windows and appends them to X
        X_sub, Y_sub, start_times, discard_count = generate_XY(df_all, 
                                                    fs = fs, 
                                                    window_size = 10, 
                                                    window_overlap = 0.2, 
                                                    channels = ['acc_x', 'acc_y', 'acc_z'], 
                                                    compute_hr=False, 
                                                    normalize="dataset", 
                                                    mask=mask, 
                                                    return_time = True,
                                                    return_discarded=True)
        if len(X_sub) == 0 or discard_count/(len(X_sub) + discard_count) > 0.99:
            # disacards recording if more than 99% of the windows are discarded
            print(f'Discarding {sessionID} from {sub_name}')
            continue
        X.append(X_sub)
        Y.append(Y_sub)
        pid.append([(int(sub_name[1:]), int(sessionID), int(float(t))) for t in start_times])
        protocol.append([sub_name, sessionID, df_all['time'].iloc[0], df_all['time'].iloc[-1], discard_count, len(X_sub)])

    """ except Exception as e:
        print(f'Error processing {sessionID} from {sub_name}')
        print(e)
        continue """

    if len(X) == 0:
        print(f'No valid windows found for {sub_name}')
        continue

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    pid = np.concatenate(pid)
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, f'{sub_name}_data.pickle'), 'wb') as f:
        pickle.dump((X, Y, pid), f)
    print(f'Saved {sub_name} to {save_folder}')


protocol_df = pd.DataFrame(protocol, columns=['subject', 'recording', 'start', 'end', 'discard_count', 'window_count'])
protocol_df.to_csv(os.path.join(analysis_dir, 'processing_protocol.csv'), index=False)
