#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
import wandb
import scipy.signal
import argparse
import tqdm
import pdb
from classical_utils import *


#%%

results_dir = config.classical_results_dir

def run_classical(config_dict):

    if config_dict["dataset"] == "max":
        dataset_dir = config.data_dir_Max_processed
    elif config_dict["dataset"] == "max_v2":
        dataset_dir = config.data_dir_Max_processed_v2
    elif config_dict["dataset"] == "Apple100" :
        dataset_dir = config.data_dir_Apple_processed_100Hz
    elif config_dict["dataset"] == "appleall" :
        dataset_dir = config.data_dir_Apple_processed_all
    elif config_dict["dataset"] == "M2Sleep":
        dataset_dir = config.data_dir_M2Sleep_processed_100Hz
    else:
        NotImplementedError

    fs = 100

    dataset = config_dict["dataset"]
    framework = config_dict["framework"]

    print(f"Processing {dataset} with {framework}")

    results_df_test = None
    split = config_dict["split"]
    print(f"Processing split {split}")
    X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = load_dataset(dataset_dir, split, args=config_dict)
    predictions_path_val = os.path.join(results_dir, f"predictions_{framework}_val_{dataset}_split{split}.pkl")
    predictions_path_test = os.path.join(results_dir, f"predictions_{framework}_test_{dataset}.pkl")
    config_dict.update({"peak_distance": 0.5, 
                "peak_prominence": 0.3, 
                "split": split,
                "peak_detection": "cwt",
                "predictions_path_val": predictions_path_val,
                "predictions_path_test": predictions_path_test,})
    print(f"Config: {config_dict}")
    if config_dict["framework"] == 'Bioglass':
        compute_hr = compute_hr_bioglass
    elif config_dict["framework"] == 'Bioglass_original':
        compute_hr = compute_hr_bioglass_original
    elif config_dict["framework"] == 'SSA':
        compute_hr = compute_hr_ssa
    elif config_dict["framework"] == 'SSA_original':
        compute_hr = compute_hr_ssa_original
    elif config_dict["framework"] == 'Troika':
        compute_hr = compute_hr_troika
    elif config_dict["framework"] == 'Troika_w_tracking':
        compute_hr = compute_hr_troika_w_tracking
    elif config_dict["framework"] == 'Kantelhardt':
        compute_hr = compute_hr_kantelhardt
    elif config_dict["framework"] == 'Kantelhardt_original':
        compute_hr = compute_hr_kantelhardt_original
    elif config_dict["framework"] == 'median':
        compute_hr = compute_hr_median
        # also need to load training set
        X_train, Y_train, pid_train, metrics_train, X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = load_dataset(dataset_dir, split, load_train=True)
    elif config_dict["framework"] == 'subject_median':
        compute_hr = compute_hr_subject_median
    else:
        raise NotImplementedError

    # Initialize WandB with your project name and optional configuration
    wandb.init(project="hr_results", config=config_dict, group=config_dict["framework"], mode=config_dict["wandb_mode"])
    #### Validation
    if not config_dict["test_only"]:
        if config_dict["framework"] == 'median': 
            hr_rr_val = compute_hr(X_val, Y_train)
        elif config_dict["framework"] in 'subject_median': 
            hr_rr_val = compute_hr(Y_val, pid_val, fs=fs)
        elif config_dict["framework"] == 'Troika_w_tracking':
            hr_rr_val = compute_hr(X_val, fs=fs)
        else:
            hr_rr_val = []
            print(f"Processing val set")
            progress_bar = tqdm.tqdm(total=len(X_val))
            for i, X in enumerate(X_val):

                if len(X) == 0:
                    hr_rr_val.append(np.nan)
                    continue

                progress_bar.update(1)
                hr_rr = compute_hr(X, fs=fs, **config_dict)
                hr_rr_val.append(hr_rr)
                
        results_df_val = pd.DataFrame({"y_true": Y_val, "hr_pred": hr_rr_val, "pid": [el for el in pid_val], "metrics": [el for el in metrics_val]})
        corr_val = results_df_val[["y_true", "hr_pred"]].corr().iloc[0,1]
        results_df_val["diff_abs"] = (results_df_val["y_true"] - results_df_val["hr_pred"]).abs()
        mae_val = results_df_val.dropna()["diff_abs"].mean()

        wandb.log({"Val_Corr": corr_val, "Val_MAE": mae_val})
        results_df_val.to_pickle(predictions_path_val)
        print(f"Saves results to {predictions_path_val}")

        figure = plot_true_pred(results_df_val["y_true"], results_df_val["hr_pred"])
        wandb.log({"true_pred_val": figure})
    
    if not isinstance(results_df_test, pd.DataFrame):
        if config_dict["framework"] == 'median':
            hr_rr_test = compute_hr_median(X_test, Y_train)
        elif config_dict["framework"] == 'subject_median': 
            hr_rr_test = compute_hr(Y_test, pid_test, fs=fs)
        elif config_dict["framework"] == 'Troika_w_tracking':
            hr_rr_test = compute_hr(X_test, fs=fs)
        else:
            hr_rr_test = []
            print(f"Processing test set")
            progress_bar = tqdm.tqdm(total=len(X_test))
            for i, X in enumerate(X_test):

                if len(X) == 0:
                    hr_rr_test.append(np.nan)
                    continue

                progress_bar.update(1)
                hr_rr = compute_hr(X, fs=fs)

                hr_rr_test.append(hr_rr)    
        results_df_test = pd.DataFrame({"y_true": Y_test, "hr_pred": hr_rr_test, "pid": [el for el in pid_test], "metrics": [el for el in metrics_test]})
        corr_test = results_df_test[["y_true", "hr_pred"]].corr().iloc[0,1]
        results_df_test["diff_abs"] = (results_df_test["y_true"] - results_df_test["hr_pred"]).abs()
        mae_test = results_df_test.dropna()["diff_abs"].mean()

        results_per_pid = []
        for pid in np.unique(pid_test[:,0]):
            hr_true_pid = np.array(Y_test)[np.where(pid_test[:,0] == pid)]
            hr_pred_pid = np.array(hr_rr_test)[np.where(pid_test[:,0] == pid)]
            mae_pid = np.abs(hr_true_pid - hr_pred_pid).mean()
            corr_pid = pd.Series(hr_true_pid).corr(pd.Series(hr_pred_pid))
            results_per_pid.append((pid, mae_pid, corr_pid))
            wandb.log({f"Test_MAE_{pid}": mae_pid, f"Test_Corr_{pid}": corr_pid})

    else:
        pass # reuse the previous split results, since no training is involved and test set stays the same

    wandb.log({"Test_Corr": corr_test, "Test_MAE": mae_test})
    print(f"Saves results to {predictions_path_test}")
    results_df_test.to_pickle(predictions_path_test)

    figure = plot_true_pred(results_df_test["y_true"], results_df_test["hr_pred"])
    wandb.log({"true_pred_test": figure})

    # Finish the run
    wandb.finish() 


def parse_args():   
    args = argparse.ArgumentParser(description="Description of your function")
    args.add_argument("--dataset", type=str, default="Apple100")
    args.add_argument("--framework", type=str, default="SSA")
    args.add_argument("--wandb_mode", type=str, default="online")
    args.add_argument('--data_thr_avg', type=float, default=0.5, help='threshold for input signal average')
    args.add_argument('--data_thr_max', type=float, default=0.08, help='threshold for input signal max')
    args.add_argument('--data_thr_angle', type=float, default=1, help='threshold for input signal angle')
    args.add_argument('--split', type=int, default=0, help='split number')
    args.add_argument('--test_only', type=bool, default=True, help='only test')

    #return args.parse_args("--framework Troika_w_tracking --wandb_mode online".split(" "))
    return args.parse_args()

#%%
if __name__ == "__main__":
    args = parse_args()
    run_classical(vars(args))
# %%
