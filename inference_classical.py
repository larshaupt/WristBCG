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
    elif config_dict["dataset"] == "Apple100":
        dataset_dir = config.data_dir_Apple_processed_100hz
    elif config_dict["dataset"] == "M2Sleep":
        dataset_dir = config.data_dir_M2Sleep_processed_100Hz
    else:
        NotImplementedError

    fs = 100

    dataset = config_dict["dataset"]
    framework = config_dict["framework"]

    print(f"Processing {dataset} with {framework}")



    results_df_test = None
    for split in range(5):
        print(f"Processing split {split}")
        X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = load_dataset(dataset_dir, split)
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
        if config_dict["framework"] == 'Bioglass_original':
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
            X_train, Y_train, pid_train, metrics_train, X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = load_dataset(dataset_dir, split, load_train=True)
        else:
            raise NotImplementedError

        # Initialize WandB with your project name and optional configuration
        #pdb.set_trace()
        wandb.init(project="hr_results", config=config_dict, group=config_dict["framework"], mode=config_dict["wandb_mode"])

        if config_dict["framework"] == 'median':
            hr_rr_val = compute_hr_median(X_val, Y_train)
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
    args.add_argument("--wandb_mode", type=str, default="disabled")
    #return args.parse_args("--framework Troika_w_tracking --wandb_mode online".split(" "))
    return args.parse_args()

#%%
if __name__ == "__main__":
    args = parse_args()
    run_classical(vars(args))
# %%
