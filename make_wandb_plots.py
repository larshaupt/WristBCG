#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import wandb
import pickle

import config
import seaborn as sns
from scipy.stats import norm
import torch
import warnings

save_dir = config.plot_dir
ssl_names = {"simsiam": "SimSiam", 
                            "byol": "BYOL", 
                            "simclr": "SimCLR", 
                            "tstcc":"TS-TCC", 
                            "nnclr": "NNCLR",
                            "supervised": "Supervised",
                            "reconstruction": "Reconstruction",
                            "oxford": "Multi-Task"}
dataset_names = {"max_v2": "In-House", 
                 "appleall": "Apple Watch", 
                 "Apple100": "Apple Watch",
                 "apple100": "Apple Watch",
                 "M2Sleep": "M2Sleep"}
metric_names = {"Val_MAE": "Validation MAE [bpm]",
                "Val_Corr": "Validation Correlation",
                "Test_MAE": "Test MAE [bpm]",
                "Test_Corr": "Test Correlation",
                "Corr": "Correlation",
                "MAE": "MAE [bpm]",
                "Test_NLL": "Test NLL",
                "Test_ECE": "Test ECE",
                "Val_NLL": "Validation NLL",
                "Val_ECE": "Validation ECE",
                "Post_Test_MAE": "Test MAE [bpm]",
                "Post_Test_Corr": "Test Correlation",
                "Post_Val_MAE": "Validation MAE [bpm]",
                "Post_Val_Corr": "Validation Correlation",
                "Test_MAE_diff": "∆ Test MAE [bpm]"}
uncertainty_names = {"NLE": "Maximum Likelihood Regression", 
                    "bnn_pretrained": "Bayesian Neural Network",
                    "bnn_pretrained_firstlast": "Pretrained BNN (First/Last Layer)",
                    "bnn": "Bayesian Neural Network",
                    "ensemble": "Deep Ensemble",
                    "mcdropout": "Monte Carlo Dropout",
                    "gaussian_classification": "Classification",
                    "none": "Baseline",
                    }
postprocessing_names = {"none": "Supervised",
                        "raw": "None",
                        "kalmansmoothing": "Kalman Smoothing",
                        "viterbi": "HMM Viterbi",
                        "sumprod": "HMM Belief Propagation"}
signal_processing_names = {"median": "Median",
                      "subject_median": "Subject Median",
                      "Bioglass": "BiogInsights (adapted)",
                      "Bioglass_original": "BiogInsights (original)",
                        "Kantelhardt": "Kantelhardt (adapted)",
                        "Kantelhardt_original": "Kantelhardt (original)",
                        "SSA": "SSA (adapted)", 
                        "SSA_original": "SSA (original)",
                        "Troika_w_tracking": "Troika with Tracking",
                        "Troika": "Troika"
                      }
#%% functions

def plot_predictions(df_run, plot_uncertainty=False, mode = "post"):
    assert mode in ["pre", "post"]
    df_run["time"] = df_run.pid.apply(lambda x:x[1]).astype(float) / 3600 #in hours
    df_run["time"] = df_run["time"] - df_run["time"].min()
    df_run.set_index("time", inplace=True)
    pids = df_run.pid.apply(lambda x:x[0]).astype(int)
    pids_unique = np.unique(pids)
    fig, axes = plt.subplots(len(pids_unique), 1, figsize=(10, 3*len(pids_unique)), sharey=False, tight_layout=True)
    for i, pid_unique in enumerate(pids_unique):
        if isinstance(axes, list):
            ax = axes[i]
        else:
            ax = axes
        df_pid = df_run[pids == pid_unique]
        
        ax.plot(df_pid.hr_true, label="True HR", alpha=1.0, markersize=1, linewidth=1.0)
        if mode == "pre":
            line1, = ax.plot( df_pid.hr_pred, label="Predicted HR", alpha=0.5, markersize=1, linewidth=1.0)
        elif "hr_pred_post" in df_pid.columns and mode == "post":
            line1, = ax.plot(df_pid.hr_pred_post, label="Predicted HR (Postprocessed)", alpha=0.8, markersize=1, linewidth=1.0)
        if plot_uncertainty and "uncertainty" in df_pid.columns:
            ax.fill_between(df_pid.index, df_pid.hr_pred - df_pid.uncertainty*90, df_pid.hr_pred + df_pid.uncertainty*90, alpha=0.5, label="Uncertainty", color=line1.get_color())
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("HR [bpm]")
        ax.set_title(f"Subject ID: {pid_unique}")
        ax.set_ylim(min(40,min(df_pid.hr_true.min(), df_pid.hr_pred.min()) - 0), max(80,max(df_pid.hr_true.max(), df_pid.hr_pred.max()) + 0))
        if i == 0:
            ax.legend(loc="upper right")


def ece_loss(y_true, y_pred_probs, bins=10):

    """
    Calculate the Expected Calibration Error (ECE) between predicted probabilities and true labels.

    ECE measures the discrepancy between predicted probabilities (y_pred) and empirical accuracy (y_true).

    Parameters:
    y_true (array-like): True labels. Each row corresponds to a sample, and each column corresponds to a class.
    y_pred_probs (array-like): Predicted probabilities. Each row corresponds to a sample, and each column corresponds to a class.
    bins (int, optional): Number of equally spaced bins for dividing the range of predicted probabilities. Default is 10.

    Returns:
    float: The Expected Calibration Error (ECE) value.

    Notes:
    - The best possible value of ECE is 0, indicating perfect calibration.
    - The worst possible value of ECE is 1, indicating complete miscalibration.
    """
        
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(y_pred_probs, axis=1)
    
    #accuracies = y_true[np.arange(len(y_true)), np.argmax(y_pred, axis=1)]
    if y_true.squeeze().ndim == 2: # if y_true is one-hot encoded
        y_true_max = np.argmax(y_true, axis=1)
    elif y_true.squeeze().ndim == 1:
        y_true_max = y_true
    else:
        raise ValueError("y_true must be one-hot encoded or a single column")
    accuracies = y_true_max == np.argmax(y_pred_probs, axis=1)
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return ece

def load_predictions(run, pred_dir = None):

    if pred_dir == None:
        if "pred_dir" in run.index:
            pred_dir = run["pred_dir"]
        
    if pred_dir != None:
        df_pred = pd.read_pickle(pred_dir)
        df_pred.rename(columns={"y_true": "hr_true"}, inplace=True)
        df_pred["dataset"] = run["dataset"]

        if run.dataset == "max_v2":
            df_pred["pid"] = df_pred["pid"].apply(lambda x : [x[0], x[1] / 100])

        return df_pred
    return None

def get_pid_mask(sweep_df, exclude_subs=[20, 8258170]):
    
    sweep_df["pred_dir"] = sweep_df.apply(find_prediction_dir, axis=1)

    pid_masks = {}
    for dataset in sweep_df["dataset"].unique():

        first_run = sweep_df[sweep_df["dataset"] == dataset].iloc[0]
        pred_df = load_predictions(first_run)
        pids = pred_df.pid
        pids = pids.apply(lambda x: x[0])
        pid_mask = ~pids.isin(exclude_subs)
        pid_mask = pid_mask.to_numpy()
        pid_masks[dataset] = pid_mask
    return pid_masks

def mask_metrics(run, pid_masks = None, metrics = ["Test_MAE", "Test_ECE"], subjects=[20, 8258170]):
    if pd.isna(run["Post_Test_MAE"]):
        post=False
    else:
        post=True
    run["pred_dir"] = find_prediction_dir(run)

    df_pred = load_predictions(run)

    if run["model_uncertainty"].lower() == "ensemble":
        pass

    if pid_masks is not None and len(pid_masks[run["dataset"]]) == len(df_pred):
        mask = pid_masks[run["dataset"]]
    else:
        mask = df_pred["pid"].apply(lambda x: x[0] not in subjects)

    
    df_pred = df_pred.loc[mask]

    if "Test_MAE" in metrics:
        mae = (df_pred.hr_true - df_pred.hr_pred).abs().mean()
        run["Test_MAE"] = mae
        if post:
            mae_post = (df_pred.hr_true - df_pred.hr_pred_post).abs().mean()
            run["Post_Test_MAE"] = mae_post
    if "Test_ECE" in metrics and "probs" in df_pred.columns:
        hr_true = (df_pred.hr_true.to_numpy() - run.hr_min) / (run.hr_max - run.hr_min)
        bins = np.concatenate([[-np.inf], np.linspace(0,1,int(run.n_prob_class)),[ np.inf]])
        hr_true_bins = np.digitize(hr_true, bins) - 1
        ece = ece_loss(hr_true_bins, np.stack(df_pred.probs))
        run["Test_ECE"] = ece
        if post:
            ece_post = ece_loss(df_pred.hr_true, np.stack(df_pred.probs_post))
            run["Post_Test_ECE"] = ece_post
        


    return run

def find_prediction_dir(run, partition="test"):
    partition = partition.lower()
    assert partition in ["test", "val"]
    pred_dir = ""
    if run.framework in signal_processing_names.keys():
        # Signal Processing
        if partition == "test":
            pred_dir = run["predictions_path_test"]
        elif partition == "val":
            pred_dir = run["predictions_path_val"]
        else: 
            raise ValueError("Partition must be 'test' or 'val'")
    
    elif "Post_Test_MAE" in run.index and not pd.isna(run["Post_Test_MAE"]):
        if partition == "test":
            # Deep Learning with Postprocessing
            if not pd.isna(run["predictions_dir_post_test"]):
                pred_dir = run["predictions_dir_post_test"]
            else:
                pred_dir = os.path.join(run.model_dir_name, run.lincl_model_file.replace(".pt", f"Test_{run.model_uncertainty}_{run.postprocessing}.pickle"))
        elif partition == "val":
            # Deep Learning with Postprocessing
            if not pd.isna(run["predictions_dir_post_val"]):
                pred_dir = run["predictions_dir_post_val"]
            else:
                pred_dir = os.path.join(run.model_dir_name, run.lincl_model_file.replace(".pt", f"Val_{run.model_uncertainty}_{run.postprocessing}.pickle"))
        else: 
            raise ValueError("Partition must be 'test' or 'val'")
    else:
        # Deep Learning without Postprocessing
        if partition == "test":
            if "predictions_dir_test" in run.index and not pd.isna(run["predictions_dir_test"]):
                pred_dir = run["predictions_dir_test"]
            else:
                pred_dir = os.path.join(run.model_dir_name, run.lincl_model_file.replace(".pt", f"_test.pickle"))
        elif partition == "val":
            if "predictions_dir_val" in run.index and not pd.isna(run["predictions_dir_val"]):
                pred_dir = run["predictions_dir_val"]
            else:
                pred_dir = os.path.join(run.model_dir_name, run.lincl_model_file.replace(".pt", f"_val.pickle"))
        else: 
            raise ValueError("Partition must be 'test' or 'val'")
    if os.path.exists(pred_dir):
        return pred_dir
    else:
        return None


def make_latex(sweep_df, metric="Test_MAE", variance=True,parameter="framework", digits=2):
    # Determine if minimum or maximum is best based on the metric
    if "mae" in metric.lower() or "ece" in metric.lower():
        min_best = True
    else:
        min_best = False

    caption = f"{metric_names[metric]} for different {parameter} on the In-House and Apple Watch dataset."
    # Group and format the data
    if variance:
        mean_std_string = lambda x: f"{x.mean():.{digits}f} ± {x.std():.{digits}f}"
    else:
        mean_std_string = lambda x: f"{x.mean():.{digits}f}"
    sweep_df_grouped = sweep_df.groupby([parameter, "dataset"]).agg(mean_std_string)
    sweep_df_table = sweep_df_grouped[metric].reset_index(level="dataset").pivot(columns="dataset", values=metric)
    
    # Apply bold formatting to best values
    sweep_df_formatted = sweep_df_table.apply(bold_best, min_best=min_best, axis=0, result_type="expand")
    columns = sweep_df_table.columns
    
    # Convert to LaTeX without column names
    sweep_df_latex = sweep_df_formatted.to_latex(bold_rows=False,
                                                 index_names = True,
                                                 caption=caption, 
                                                 label=f"tab:{parameter}_{metric}",
                                                 header = columns)
    print(sweep_df_latex)


def make_latex_table(df, n_digits = 2, best_bold=False, min_best=True):
    """
    Converts a pandas DataFrame with mean and std values into a LaTeX table,
    where each cell contains mean and std in a single cell in the format: mean (std).
    
    Args:
    df (pd.DataFrame): A DataFrame with MultiIndex columns where level 0 is the variable
                       (like deviation, error) and level 1 is the statistic (mean, std).
    
    Returns:
    str: A LaTeX formatted table as a string.
    """
    # Create a new DataFrame to format mean and std into one cell
    formatted_df = pd.DataFrame(index=df.index)

    
    # little workaround to keep it sorted and woring
    unique_vars = []
    for var in df.columns:
        if var[0] not in unique_vars:
            unique_vars.append(var[0])
    for var in unique_vars:  # For each variable like deviation, error
        # Concatenate mean and std into a single string in the format: 'mean (std)'
        if (var, "std") in df.columns and (df[(var, "std")].fillna(0) != 0).any() :
            formatted_df[var] = df[(var, 'mean')].apply(lambda x: f"{x:.{n_digits}f}") + " ± " + \
                                df[(var, 'std')].apply(lambda y: f"{y:.{n_digits}f}")
        else:
            formatted_df[var] = df[(var, 'mean')].apply(lambda x: f"{x:.{n_digits}f}")
    # Use to_latex to convert the formatted DataFrame to a LaTeX table

    if best_bold:
        find_min_df = df.copy()
        find_min_df[[el for el in find_min_df.columns.levels[0] if "median" in el.lower()]] = np.nan
        if min_best:
            best_approaches = find_min_df.xs("mean", level=1, axis=1).apply(lambda x:x.argmin(), axis=1) 
        else:
            best_approaches = find_min_df.xs("mean", level=1, axis=1).apply(lambda x:x.argmax(), axis=1)
    for index, pos in best_approaches.items():
        formatted_df.loc[index, formatted_df.columns[pos]] = "\\textbf{" + formatted_df.loc[index, formatted_df.columns[pos]] + "}"
    formatted_df.replace({"nan": "-"}, inplace=True)
    latex_str = formatted_df.to_latex(index=True, header=True)
    print(latex_str)


def bold_best(s, min_best=True):
    if isinstance(s[0], str):
        s_values = s.apply(lambda x: float(x.split("±")[0]))
    else:
        s_values = s
    if min_best:
        is_best = s_values == s_values.min()
    else:
        is_best = s_values == s_values.max()
    return ['\\textbf{' + str(v) + '}' if max_val else str(v) for v, max_val in zip(s, is_best)]

def confidence_plot(uncert, pred_expectation, Y, ax=None, name="", distribution="gaussian", num_points = 10):
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,5))


    def gaussian_confidence_interval(conf, uncert):
        return norm.ppf(0.5 + conf/2, loc=0, scale=uncert)

    def laplace_confidence_interval(conf, uncert):
        return np.array(- uncert/np.sqrt(2) * np.log(1-conf))
    if distribution == "gaussian":
        conf_function = gaussian_confidence_interval
    elif distribution == "laplace":
        conf_function = laplace_confidence_interval
    else:
        raise ValueError(f"Distribution {distribution} not supported")

    conf_scores = []
    for conf in np.linspace(0, 1, num_points + 1):
        conf_interval = conf_function(conf, uncert).reshape(-1)
        inside_confidence_interval = (Y > pred_expectation - conf_interval) & (Y < pred_expectation + conf_interval)
        conf_score = inside_confidence_interval.sum() / len(inside_confidence_interval)
        conf_scores.append((conf, conf_score))
    df_conf_scores = pd.DataFrame(conf_scores, columns=["Confidence", "Score"])
    ece = (df_conf_scores["Confidence"] - df_conf_scores["Score"]).abs().mean()
    
    df_conf_scores["Confidence"] = df_conf_scores["Confidence"].round(1)
    df_conf_scores.plot.bar(x="Confidence", y="Score", ax=ax, label=name)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Percentage of data inside confidence interval")
    #ax.set_title("Confidence Calibration")
    ax.set_ylim(0,1)
    ax.plot(np.arange(0,1.1,1/num_points), color="red")
    return ece
#confidence_plot(uncerts, preds, Y_concat, distribution="laplace", name="Raw Prediction", num_points=10)

def plot_excluion(HR_true, HR_pred, uncertainty, ax=None, metric="MAE", name=""):

    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,5))

    df_res = pd.DataFrame({"hr_true": HR_true, 
                            "hr_pred": HR_pred, 
                            "uncertainty": uncertainty})

    df_res.sort_values("uncertainty", inplace = True, ascending=True)

    p_keep = np.linspace(0,100,100)
    total_num = len(df_res)
    if metric=="MAE":
        df_res["mae"] = np.abs(df_res["hr_true"] - df_res["hr_pred"])
        mae = [df_res["mae"].iloc[:int(p/100*total_num)].mean() for p in p_keep]
        ax.plot(p_keep, mae, label=name)
        ax.set_ylabel("MAE [bpm]")
    
    elif metric=="Corr":
        corr = [df_res[["hr_true", "hr_pred"]].iloc[:int(p/100*total_num)].corr().iloc[0,1] for p in p_keep]
        ax.plot(p_keep, corr, label=name)
        ax.set_ylabel("Correlation")


    ax.set_xlabel("Percentage of data evaluated")
    #ax.set_title("Exclusion of uncertain predictions")
    ax.legend()
#plot_excluion(Y_concat, preds, uncerts, metric="MAE", name="Raw Prediction")

def plot_uncert_vs_mae(HR_true, HR_pred, uncertainty, ax=None,  name="", bins=10):

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,5))

    df_res = pd.DataFrame({"hr_true": HR_true, 
                            "hr_pred": HR_pred, 
                            "uncertainty_squared": uncertainty**2})



    df_res["bin"] = pd.cut(df_res["uncertainty_squared"], bins=bins)
    df_res["bin"] = df_res["bin"].apply(lambda x : np.round((x.left + x.right)/2, 4))
    df_res["mse"] = (df_res["hr_true"] - df_res["hr_pred"])**2
    groups = df_res.groupby("bin")["mse"].mean()
    groups.plot.bar(ax=ax, alpha=1.0, label=name)

    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("MSE (Mean)")
    #ax.set_title("Uncertainty vs MAE")
    ax.legend()
#plot_uncert_vs_mae(Y_concat, preds ,uncerts, name="Raw Prediction")

def plot_hit_rate(HR_true, HR_pred_probs, ax=None,  name="", hrmin=30, hrmax = 120, n_bins_divisor=8, n_bins_original=64, all_probs=True):
    assert np.log2(n_bins_divisor).is_integer(), "n_bins_divisor must be a power of 2"
    n_bins = n_bins_original // n_bins_divisor
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,5))

    if isinstance(HR_true, torch.Tensor) or isinstance(HR_true, np.ndarray):
        HR_pred_probs = [el for el in HR_pred_probs]

    df_res = pd.DataFrame({"hr_true": HR_true, 
                            "probs": HR_pred_probs})

    if all_probs:
        # takes all probabilities and compares them against the true accuarcy

        df_res["probs"] = df_res["probs"].apply(lambda x : x.reshape(-1, n_bins_divisor).sum(axis=1))
        df_res["bin_l"] = [np.concatenate([[-np.inf], np.linspace(0,1,n_bins-1)])]*len(df_res)
        df_res["bin_h"] = [np.concatenate([np.linspace(0,1,n_bins-1), [np.inf],])]*len(df_res)

        df_res = df_res.explode(["bin_l", "bin_h", "probs"])

        df_res["correct"] = df_res.apply(lambda row: (row["hr_true"] -hrmin)/(hrmax-hrmin) > row["bin_l"] and (row["hr_true"] -hrmin)/(hrmax-hrmin) < row["bin_h"], axis=1)
        df_res["bin"] = pd.cut(df_res["probs"], bins=np.arange(0,1.1,0.1))
    else:
        # only takes probability of the predicted class and compares them against the true accuarcy
        # For each instance, consider only the probability of the class with the highest probability (predicted class)
        bins_values = np.concatenate([[-np.inf], np.linspace(0,1,n_bins-1), [np.inf]])
        bins = [(bins_values[i], bins_values[i+1]) for i in range(len(bins_values)-1)]
        df_res["probs"] = df_res["probs"].apply(lambda x : x.reshape(-1, n_bins_divisor).sum(axis=1))
        df_res['prob_max'] = df_res['probs'].apply(np.argmax)
        df_res['probs'] = df_res['probs'].apply(np.max)
        df_res["bin_l"] = df_res['prob_max'].apply(lambda x: bins[x][0])
        df_res["bin_h"] = df_res['prob_max'].apply(lambda x: bins[x][1])

        df_res["correct"] = df_res.apply(lambda row: (row["hr_true"] -hrmin)/(hrmax-hrmin) > row["bin_l"] and (row["hr_true"] -hrmin)/(hrmax-hrmin) < row["bin_h"], axis=1)
        df_res["bin"] = pd.cut(df_res["probs"], bins=np.arange(0,1.1,0.1))


    accuracy = df_res.groupby("bin")["correct"].mean()
    accuracy.fillna(0, inplace=True)
    confidence = np.array([np.mean([float(x.left), float(x.right)]) for x in accuracy.index])
    gaps = confidence - accuracy.to_numpy()
    groups_color = df_res.groupby("bin")["correct"].count()
    norm = plt.Normalize(groups_color.min(), groups_color.max())
    colors = plt.cm.viridis(norm(groups_color))
    ax.bar(confidence, accuracy, width=1/10, edgecolor='black', label='Outputs')
    ax.bar(confidence, gaps, bottom=accuracy, color='red', alpha=0.5, width=1/10, label='Gap')
    ax.plot(np.arange(0.05,1.0,0.1), np.arange(0.05,1.0,0.1), "k--")
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    #ax.set_title(f"Confidence Calibration ({n_bins} bins)")
    ax.legend()
    return df_res
#%% load wandb data

def load_wandb_data(project="hr_results", entity = "lhauptm"):

    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(f"{entity}/{project}")

    summary_list, config_list, name_list = [], [], []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        sweep_name = run.sweep.name if run.sweep else None
        sweep_id = run.sweep.id if run.sweep else None
        name_list.append({"name": run.name,
                          "sweep_id": sweep_name,
                          "sweep_name": sweep_id,
                          "state": run.state,})

    runs_df = pd.concat([pd.DataFrame(summary_list), pd.DataFrame(config_list), pd.DataFrame(name_list)], axis=1)
    return runs_df
runs_results_df = load_wandb_data(project="hr_results").reset_index(drop=True)
runs_sweep_df = load_wandb_data(project="hr_sweep").reset_index(drop=True)
runs_ablation_df = load_wandb_data(project="hr_ablation").reset_index(drop=True)
common_columns = runs_results_df.columns.intersection(runs_sweep_df.columns).intersection(runs_ablation_df.columns)
runs_df = pd.concat([runs_results_df[common_columns], runs_sweep_df[common_columns]], ignore_index=True)
#%% baseline architecture

def baseline_architecture_plot(runs_df):
    metrics = ["Val_MAE", "Val_Corr", "Test_MAE", "Test_Corr", "Total_Params"]
    params = ["dataset",  "Architecture"]
    sweep_df = runs_df[runs_df["sweep_id"].isin(["qqh947aw", "pfzg4bmf", "twnux4xy"])]
    sweep_df = sweep_df[sweep_df["backbone"].isin(["Transformer", "LSTM", "FCN", "CorNET", "HRCTPNet"])]
    architecture_names = {"FCN": "Fully-CNN", "CorNET": "CorNET", "Transformer": "Transformer", "LSTM": "GRU", "HRCTPNet": "HRCTPNet"}
    architecture_names = {k : f"{v} ({sweep_df[sweep_df['backbone'] == k]['Total_Params'].iloc[0]/1000:.0f}k #p)" for k, v in architecture_names.items()}
    sweep_df.replace(architecture_names, inplace=True)
    sweep_df.replace(dataset_names, inplace=True)
    sweep_df.rename(columns={"backbone": "Architecture"}, inplace=True)
    sweep_df = sweep_df[metrics + params]
    for metric in ["Test_MAE", "Val_MAE"]:

        fig, ax = plt.subplots(1, 1)
        #sns.boxplot(data=sweep_df, x=params[0], hue=params[1], y=metric, showmeans=True)
        sns.barplot(data=sweep_df, x=params[0], hue=params[1], y=metric, ci="sd")
        if "Corr" in metric:
            ax.set_ylim(0,1)

        ax.set_ylabel(metric_names[metric])
        ax.set_xlabel("Dataset")

        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., 0),#p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize= 8)
        plt.savefig(os.path.join(save_dir, f"baseline_architecture_{metric}.pdf"))

    return fig, ax
baseline_architecture_plot(runs_results_df)

# %% signal processing

def signal_processing_plot(runs_df):
    metrics = ["Val_MAE", "Val_Corr", "Test_MAE", "Test_Corr"]
    params = ["dataset",  "Approach"]
    frameworks = ["Bioglass", 
                  "Bioglass_original",
                  "Kantelhardt", 
                  "Kantelhardt_original",
                  "SSA", 
                  "SSA_original",
                  "Troika", 
                  "Troika_w_tracking",
                  "median"
                  ]
    sweep_df = runs_df[runs_df["framework"].isin(frameworks)]
    sweep_df = sweep_df[sweep_df["dataset"].isin(["max_v2", "appleall", "Apple100"])]
    sweep_df.replace(dataset_names, inplace=True)
    sweep_df.rename(columns={"framework": "Approach"}, inplace=True)
    sweep_df.replace({"NaN": 0,
                      "median": "Median",
                      "Bioglass": "Bioglass (adapted)",
                      "Bioglass_original": "Bioglass (original)",
                        "Kantelhardt": "Kantelhardt (adapted)",
                        "Kantelhardt_original": "Kantelhardt (original)",
                        "SSA": "SSA (adapted)", 
                        "SSA_original": "SSA (original)",
                        "Troika_w_tracking": "Troika with Tracking",
                      }, inplace=True)
    sweep_df = sweep_df[metrics + params]
    #sweep_df["MAE"] = (sweep_df["Val_MAE"] + sweep_df["Test_MAE"])/2
    #sweep_df["Corr"] = (sweep_df["Val_Corr"] + sweep_df["Test_Corr"])/2
    print(sweep_df.groupby(params).mean()["Test_MAE"].unstack().transpose())
    for metric in ["Test_MAE", "Test_Corr"]:
        fig, ax = plt.subplots(1, 1)
        sns.barplot(data=sweep_df, x=params[0], hue=params[1], y=metric, errorbar=None)
        if "Corr" in metric:
            ax.set_ylim(0,1)
        ax.set_ylabel(metric_names[metric])
        ax.set_xlabel("Dataset")
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., 0),#p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize= 6)
        plt.savefig(os.path.join(save_dir, f"signal_processing_{metric}.pdf"))
    return sweep_df
sweep_df = signal_processing_plot(runs_results_df)

#%% signal processing 2

def signal_processing_plot(runs_df):
    metrics = ["Test_MAE", "Test_Corr"]
    params = ["dataset",  "framework"]
    sweeps = ["c5niepix"]
    sweep_df = runs_df[runs_df["sweep_id"].isin(sweeps)]
    sweep_df.replace(dataset_names, inplace=True)
    sweep_df.replace({"NaN": 0}, inplace=True)
    sweep_df.replace(signal_processing_names, inplace=True)
    sweep_df = sweep_df[metrics + params]
    print(sweep_df.groupby(params).mean()["Test_MAE"].unstack().transpose())
    for metric in ["Test_MAE", "Test_Corr"]:
        fig, ax = plt.subplots(1, 1)
        sns.barplot(data=sweep_df, x=params[0], hue=params[1], y=metric, errorbar=None)
        if "Corr" in metric:
            ax.set_ylim(0,1)
        ax.set_ylabel(metric_names[metric])
        ax.set_xlabel("Dataset")
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., 0),#p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize= 6)
        #plt.savefig(os.path.join(save_dir, f"signal_processing_{metric}.pdf"))
    return sweep_df
sweep_df = signal_processing_plot(runs_results_df)
make_latex(sweep_df, metric="Test_MAE", parameter="framework", digits=2, variance=False)

#%% baseline sweep

def baseline_sweep_plot(runs_df):
    sweeps = {
    'kernel_size' : "bxafrs5p",
    'dropout_rate' : "2qo37tsn",
    'lstm_units' : "xdlxqw5b",
    'num_kernels' : "p7o6mgod",
    'loss' : "zrkdjllj",
    'lr': "qxgpacin",
    'batch_size': "detvisbb"}

    parameter_names = {
    'kernel_size' : "Kernel Size",
    'dropout_rate' : "Dropout Rate",
    'lstm_units' : "Number of RNN Units",
    'num_kernels' : "Number of CNN Kernels",
    'loss' : "Loss Function",
    'lr': "Learning Rate",
    'batch_size': "Batch Size"}

    for name, sweep_id in sweeps.items():

        metrics = ["Val_MAE", "Val_Corr", "Test_MAE", "Test_Corr"]
        params = ["dataset",  name]
        sweep_df = runs_df[runs_df["sweep_id"] == sweep_id]
        sweep_df = sweep_df[params + metrics].dropna(axis=1, how='any')
        sweep_df["dataset"].replace(dataset_names, inplace=True)
        
        for metric in ["Test_MAE", "Test_Corr"]:
            fig, ax = plt.subplots(1, 1)
            #sns.boxplot(data=sweep_df, x="dataset", hue=sweep_df[name].astype(str), y=metric, showmeans=True)
            sns.barplot(data=sweep_df, x="dataset", hue=sweep_df[name].astype(str), y=metric, errorbar="sd", ax=ax)
            
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.2f'), 
                            (p.get_x() + p.get_width() / 2.,0), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points',
                            fontsize= 8)
            
            ax.legend(title = parameter_names[name], loc='upper left')
            if "Corr" in metric:
                ax.set_ylim(0,1)
            else:
                all_sweeps_df = runs_df[runs_df["sweep_id"].isin(sweeps.values())]
                ax.set_ylim(all_sweeps_df[metric].min()*0.8, all_sweeps_df[metric].max()*1.2)
            ax.set_ylabel(metric_names[metric])
            ax.set_xlabel("Dataset")

            

            fig.savefig(os.path.join(save_dir, f"sweep_{name}_{metric}.pdf"))



baseline_sweep_plot(runs_results_df)




#%% SSL frameworks

def ssl_frameworks_plot(runs_df, metrics = ["Val_MAE", "Val_Corr", "Test_MAE", "Test_Corr"]):
    sweeps = [
        "64oys3jc", #ssl
        "u7e7hc8c", #supervised
        #"detvisbb", #supervised
        "3382xzph", #reconstruction
        "3a7lgi17" #Oxford
        ]
    parameters = ["framework", "dataset"]

    sweep_df = runs_df[runs_df["sweep_id"].isin(sweeps)]
    sweep_df  = sweep_df[(sweep_df["random_seed"] == 10) & (sweep_df["batch_size"] == 512)]
    sweep_df = sweep_df[parameters + metrics]
    sweep_df["dataset"].replace(dataset_names, inplace=True)
    sweep_df["framework"].replace(ssl_names, inplace=True)
    for metric in metrics:
        fig, ax = plt.subplots(1, 1)
        sns.barplot(data=sweep_df, x=parameters[1], ax=ax, hue=parameters[0], y=metric)
        if "Corr" in metric:
            ax.set_ylim(0,1)
        ax.set_ylabel(metric_names[metric])
        plt.savefig(os.path.join(save_dir, f"ssl_frameworks_{metric}.pdf"))

    return sweep_df

sweep_df = ssl_frameworks_plot(runs_df)
make_latex(sweep_df, metric="Test_MAE", parameter="framework", digits=2, variance=True)
make_latex(sweep_df, metric="Val_MAE", parameter="framework", digits=2, variance=True)

# %% ssl augmentations
def ssl_augemtations_plot(runs_df, mectrics = ["Test_MAE", "Test_Corr", "Val_MAE", "Val_Corr"]):
    metrics = ["Val_Corr", "Val_MAE", "Test_Corr", "Test_MAE", "Train_Loss", "Val_Loss", "Test_Loss", "Train_MAE", "Train_Corr"]
    parameters = ["aug1", "aug2", "framework", "wandb_tag", "split", "pretrain", "loss"]
    params_groupby = ["aug1", "aug2"]

    runs_df_sel = runs_df[parameters + metrics]
    runs_df_sel = runs_df_sel.loc[(runs_df_sel["wandb_tag"] == "pretrain_augs")]
    runs_df_sel.replace({"bioglass": "bioinsights"}, inplace=True)
    
    all_augs = np.unique(runs_df_sel["aug1"].to_list() + runs_df_sel["aug2"].to_list()).tolist()
    #runs_df_sel = runs_df_sel.loc[(runs_df_sel["lr_finetune_backbone"] == 0) | (runs_df_sel["lr_finetune_backbone"] == 1e-05) | (runs_df_sel["lr_finetune_backbone"] == 0.0001)]
    #runs_df_sel = runs_df_sel.loc[(runs_df_sel["lr_finetune_lstm"] == 0) | (runs_df_sel["lr_finetune_lstm"] == 1e-05) | (runs_df_sel["lr_finetune_lstm"] == 0.0001) | (runs_df_sel["lr_finetune_lstm"] == 0.0005)]

    for metric in mectrics:

        param1_col = params_groupby[0]
        param2_col = params_groupby[1]
        mean_col = (metric, 'mean')  # Replace 'your_metric_column' with the column name of the metric
        std_col = (metric, 'std')    # Replace 'your_metric_column' with the column name of the std metric

        agg_df = runs_df_sel.groupby(params_groupby)[metrics].aggregate(["mean", "std", "count"])
        agg_df = agg_df.reset_index()
        agg_df = agg_df.fillna(0)
        #agg_df = agg_df.loc[agg_df[("Test_Corr", "count")] == 6]
        # Pivot the DataFrame to get it in the right format for the heatmap
        pivot_df_mean = agg_df.pivot(index=param2_col, columns=param1_col, values=mean_col)
        pivot_df_std = agg_df.pivot(index=param2_col, columns=param1_col, values=std_col)

        pivot_df_mean = pivot_df_mean.reindex(all_augs, axis=0)
        pivot_df_mean = pivot_df_mean.reindex(all_augs, axis=1).iloc[::-1]
        pivot_df_std = pivot_df_std.reindex(all_augs, axis=0)
        pivot_df_std = pivot_df_std.reindex(all_augs, axis=1).iloc[::-1]

        # Create a heatmap using seaborn
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=600)

        colormap = "viridis_r" if "MAE" in metric else "viridis"
        # Plotting the mean values with annotations of mean and std
        sns.heatmap(pivot_df_mean, annot=False, cmap=colormap, fmt=".3f", cbar=True,
                    annot_kws={"fontsize": 8, "fontweight": "bold", "color": "black"},
                    cbar_kws={'label': metric}, ax=ax)



        # Adding std values in annotations
        for i in range(pivot_df_mean.shape[0]):
            for j in range(pivot_df_mean.shape[1]):
                plt.text(j + 0.5, i + 0.35,f"{pivot_df_mean.iloc[i,j]:.2f}", #\n±{pivot_df_std.iloc[i, j]:.2f}",
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white', fontsize=7, fontweight='bold')
                plt.text(j + 0.5, i + 0.65,f"±{pivot_df_std.iloc[i, j]:.2f}",
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white', fontsize=7, fontweight="normal")

        #ax.set_title(f' Search Results for {metric}')
        cbar = ax.collections[0].colorbar
        cbar.set_label(metric_names[metric])
        ax.set_xlabel("Augmentation 1")
        ax.set_ylabel("Augmentation 2")
        plt.savefig(os.path.join(save_dir, f"ssl_augs_{metric}.pdf"), bbox_inches='tight')


ssl_augemtations_plot(runs_sweep_df)



# %% finetune lr

def plot_finetune_lr(runs_df, mask=False, metrics = ["Test_MAE", "Val_MAE"]):
    sweep_id = [
        #"bzis8i3k", #nnclr
        "s040lvms", #cross dataset
        ]

    for metric in metrics:
        for dataset in ["max_v2", "appleall"]:

            sweep_df = runs_df[runs_df["sweep_id"].isin(sweep_id)]
            sweep_df = sweep_df[sweep_df["dataset"] == dataset]

            if mask:
                pid_masks = get_pid_mask(sweep_df)
                sweep_df = sweep_df.apply(mask_metrics, pid_masks = pid_masks, metrics=["Test_MAE", "Test_Corr"], axis=1)

            params = ["lr_finetune_backbone", "lr_finetune_lstm"]
            
            sweep_df = sweep_df[metrics + params]
            agg_df = sweep_df.groupby(params)[metric].aggregate(["mean", "std", "count"])
            #agg_df.fillna(0, inplace=True)
            agg_df = agg_df.reset_index()

            pivot_df_mean = agg_df.pivot(index = params[0], columns = params[1], values="mean").iloc[::-1].astype(float)
            pivot_df_std = agg_df.pivot(index = params[0], columns = params[1], values="std").iloc[::-1].astype(float)

            fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=600)


            # Plotting the mean values with annotations of mean and std
            sns.heatmap(pivot_df_mean, annot=False, cmap='viridis_r', fmt=".3f", cbar=True,
                        annot_kws={"fontsize": 8, "fontweight": "bold", "color": "black"},
                        cbar_kws={'label': metric}, ax=ax)



            # Adding std values in annotations
            for i in range(pivot_df_mean.shape[0]):
                for j in range(pivot_df_mean.shape[1]):
                    plt.text(j + 0.5, i + 0.35,f"{pivot_df_mean.iloc[i,j]:.2f}", #\n±{pivot_df_std.iloc[i, j]:.2f}",
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='white', fontsize=12, fontweight='bold')
                    plt.text(j + 0.5, i + 0.65,f"±{pivot_df_std.iloc[i, j]:.2f}",
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='white', fontsize=12, fontweight="normal")

            cbar = ax.collections[0].colorbar
            cbar.set_label(metric_names[metric])
            ax.set_ylabel("CNN Learning Rate")
            ax.set_xlabel("RNN Learning Rate")
            ax.set_title(f"{dataset_names[dataset]}")
            plt.savefig(os.path.join(save_dir, f"finetune_lr_{dataset}_{metric}.pdf"), bbox_inches='tight')

plot_finetune_lr(runs_df, mask=False, metrics = ["Test_MAE", "Val_MAE", "Test_Corr", "Val_Corr"])



# %% ssl number of samples
#make number of samples plot
def plot_number_samples(runs_df, mask=False, metrics = ["Test_MAE", "Val_MAE", "Test_Corr", "Val_Corr"]):
    sweeps = [
            #"5jjq3nbu", #old, not optimal hyperaprams
            #"2zgfp6kh", #subsaples, best, simcrl apple
            #"u3q38e1c", #subsaples, best, simcrl max
            "ey2yrcwv", #supervised
            "dnq4szje", #capture24all
            "0xhhj3lg", #subsampled new best hyperparams, nnclr
            #"h7hwgfyv", #simclr, max, apple
            #"h0qa02ns", #simclr, max, apple, cross dataset
            ]
    params = ["dataset",  "pretrain_subsample"]
    sweep_df = runs_df[runs_df["sweep_id"].isin(sweeps)]
    sweep_df = sweep_df[(sweep_df["batch_size"] == 512) & (sweep_df["random_seed"] == 10)]
    sweep_df = sweep_df[sweep_df["framework"].isin(["supervised", "nnclr"])]
    if mask:
        pid_masks = get_pid_mask(sweep_df)
        sweep_df = sweep_df.apply(mask_metrics, pid_masks = pid_masks, metrics=["Test_MAE", "Test_Corr"], axis=1)

    sweep_df.loc[sweep_df["framework"] == "supervised", "pretrain_subsample"] = 0
    
    sweep_df.sort_values(by=params[1], inplace=True)


    sweep_df["pretrain_subsample"] = (sweep_df["pretrain_subsample"]* 100).astype(int).astype(str) + "%"
    sweep_df.loc[(sweep_df["pretrain_dataset"].isin(["appleall", "max_v2"])) & (sweep_df["pretrain_dataset"] == sweep_df["dataset"]) , "pretrain_subsample"] = "autodataset"
    sweep_df.loc[(sweep_df["pretrain_dataset"].isin(["appleall", "max_v2"])) & (sweep_df["pretrain_dataset"] != sweep_df["dataset"]) , "pretrain_subsample"] = "crossdataset"
    sweep_df.loc[sweep_df["pretrain_dataset"] == "capture24all", "pretrain_subsample"] = "100% all activities"
    sweep_df.replace(dataset_names, inplace=True)
    sweep_df = sweep_df[metrics + params]
    sweep_df["pretrain_subsample"] = sweep_df["pretrain_subsample"].astype(str)

    for metric in metrics:
        fig, ax = plt.subplots(1, 1)
        sns.barplot(data=sweep_df, x=params[0], hue=params[1], y=metric, errorbar='sd')
        if "Corr" in metric:
            ax.set_ylim(0,1)
        ax.set_ylabel(metric_names[metric])

        ax.set_xlabel("Dataset")
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., 0),#p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize= 8)
        plt.legend(title = 'Subsample Rate', loc='upper left')

        plt.savefig(os.path.join(save_dir, f"ssl_pretrain_subsample_{metric}.pdf"))
    return sweep_df

sweep_df = plot_number_samples(runs_results_df, mask=False)

make_latex(sweep_df, parameter="pretrain_subsample", metric="Test_MAE")

# %% classifier layers

# make classifier_layers plot

def plot_classifier_layers(runs_df, metrics = ["Val_MAE", "Val_Corr", "Test_MAE", "Test_Corr"]):
    sweeps = ["j8mgyqv4"]
    

    sweep_df = runs_df[runs_df["sweep_id"].isin(sweeps)]
    sweep_df.replace(dataset_names, inplace=True)
    sweep_df = sweep_df[sweep_df["framework"].isin(["nnclr", "simclr"])]
    sweep_df["Framework - Classifier Layers"] = sweep_df["framework"] + " - " + sweep_df["num_layers_classifier"].astype(int).astype(str)
    params = ["dataset",  "Framework - Classifier Layers"]
    sweep_df = sweep_df[metrics + params]
    sweep_df.sort_values(by=params[1], inplace=True)

    for metric in metrics:
        fig, ax = plt.subplots(1, 1)
        sns.barplot(data=sweep_df, x=params[0], hue=params[1], y=metric, ci="sd")
        if "Corr" in metric:
            
            ax.set_ylim(0,1)
        ax.set_ylabel(metric_names[metric])
        ax.set_xlabel("Dataset")

        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., 0),#p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')
        plt.savefig(os.path.join(save_dir, f"ssl_classifier_{metric}.pdf"))
    return sweep_df
sweep_df = plot_classifier_layers(runs_results_df)

make_latex(sweep_df, parameter="Framework - Classifier Layers", metric="Test_MAE")
make_latex(sweep_df, parameter="Framework - Classifier Layers", metric="Val_MAE")


# %% tsne plots

# find TSNE plots


def find_tsne_dir(run, format="png"):

    if pd.isna(run.tsne_dir):
        tsne_dir = run.model_dir_name.replace("bestmodel.pt", "tsne.png")
    else:
        tsne_dir = run["tsne_dir"]
    
    if format == "pickle":
        tsne_dir = tsne_dir.replace("png", "pickle")
    return tsne_dir

def get_tsne_plot(run):
    tsne_dir = find_tsne_dir(run)
    if os.path.exists(tsne_dir):
        tsne_image = plt.imread(tsne_dir)
        return tsne_image
    else:
        return None

def get_tsne_results(run):
    tsne_dir = find_tsne_dir(run, format="pickle")
    if os.path.exists(tsne_dir):
        with open(tsne_dir, "rb") as f:
            (tsne_results, y_true) = pickle.load(f)
        return (tsne_results, y_true)
    else:
        return None

# make tsne plots
sweeps = ["wyqx6gyi", #Finetuned
          "2b2hblic" #pretrained
        ]

name_dict = {("supervised", 60): "Supervised",
             ("nnclr", 60): "Finetuned",
             ("nnclr", 0): "Pretrained",}

fig, axes = plt.subplots(2, 3, figsize=(10*3,10*2))
for j, dataset in enumerate(["appleall", "max_v2"]):

    sweep_df = runs_results_df[runs_results_df["sweep_id"].isin(sweeps)]
    sweep_df = sweep_df[sweep_df["dataset"] == dataset]
    sweep_df["tsne_dir"] = sweep_df.apply(find_tsne_dir, axis=1)
    
    i = 0
    for framework in sweep_df["framework"].unique():
        for n_epoch in sweep_df["n_epoch"].unique():
            sweep_df_cat = sweep_df[(sweep_df["framework"] == framework) & (sweep_df["n_epoch"] == n_epoch)]
            if len(sweep_df_cat) == 0:
                continue
            
            run = sweep_df_cat.iloc[0]
            (tsne_results, y_true) = get_tsne_results(run)
            axes[j,i].scatter(x=tsne_results[:,0], y=tsne_results[:,1], c=y_true, cmap='viridis', alpha=0.5)
            if j == 0:
                axes[j,i].set_title(name_dict[(framework, n_epoch)])
            axes[j,i].axis('off')
            axes[j,i].title.set_size(50)
            i += 1

row_titles = ['Apple Watch', 'In-House']
for ax, row_title in zip(axes[:,0], row_titles):
    # This will align the text to the right, so it appears to be to the left of the y-axis of the left-most subplots
    fig.text(0.05, ax.get_position().y0 + ax.get_position().height / 2, row_title,
             va='center', ha='center', rotation='horizontal', transform=fig.transFigure, fontsize=50)
axes[0,0].colorbar(label='True HR [bpm]')
plt.savefig(os.path.join(save_dir, f"tsne_plot.png"), bbox_inches='tight')
    




tsne_results, y_true = get_tsne_results(sweep_df.iloc[0])
plt.scatter(x=tsne_results[:,0], y=tsne_results[:,1], c=y_true, cmap='viridis', alpha=0.2)

#%% number of finetuning samples
sweep_ids = [
    "u67ps4pz", #train subsample with ssl and supervised learning, 300 epochs
    #"ya8m5f7t", #train subsample with ssl and supervised learning, 300 epochs, full pretrain data
    # 0.2 pretrain_subsample performs better than 1
    "gul7s0el", #subsample, additional params

    "ey2yrcwv",# supervised full
    "0xhhj3lg", # nnclr full
    "z931vofl", #simclr1
    "ncmkhdzw", #simclr2

]


for metric in ["Test_MAE", "Val_MAE", "Test_Corr", "Val_Corr"]:
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    sweep_df = runs_df[runs_df["sweep_id"].isin(sweep_ids)]
    sweep_df = sweep_df[(sweep_df["framework"] != "nnclr") | (sweep_df["pretrain_subsample"].isin([1,0.2]))]
    sweep_df["subsample_share"] = sweep_df["subsample"]*100
    #sweep_df["subsample_share"] = sweep_df["subsample"]*(40560*(sweep_df["dataset"] == "max_v2") + 68975*((sweep_df["dataset"] == "appleall")))
    sweep_df["framework"] = sweep_df["framework"].replace(ssl_names)
    sweep_df["dataset"] = sweep_df["dataset"].replace(dataset_names)
    sweep_df["dataset_framework"] = sweep_df["framework"]

    sweep_df.rename(columns={"dataset": "Dataset", "framework": "Framework"}, inplace=True)

    sns.lineplot(data=sweep_df, x="subsample_share", y=metric, hue="Framework", marker="o", style = 'Dataset', ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("Training Samples [%]")
    ax.set_ylabel(metric_names[metric])
    fig.savefig(os.path.join(save_dir, f"ssl_finetune_samples_{metric}.pdf"), bbox_inches='tight')

# %% uncertainty performance


#Uncertainty

def plot_uncertainty_performance(runs_df, metrics=["Test_MAE", "Test_ECE", "Val_MAE", "Val_ECE"],  mask=False):
    sweeps = ["qu24hmkc", # uncertainty
              "sc724fuu", # uncertainty ensemble
            #"u7e7hc8c", # supervised
            "ey2yrcwv", #supervised
            ]
    params = ["dataset",  "model_uncertainty"]
    
    sweep_df = runs_df[runs_df["sweep_id"].isin(sweeps)]
    sweep_df.loc[sweep_df["model_uncertainty"] == "none", "postprocessing"] = "raw"
    sweep_df = sweep_df[sweep_df["random_seed"] == 10]
    
    sweep_df = sweep_df[sweep_df["postprocessing"] == "raw"]
    if mask:
        pid_masks = get_pid_mask(sweep_df, exclude_subs=[])
        sweep_df = sweep_df.apply(mask_metrics, pid_masks = pid_masks, metrics=["Test_MAE", "Test_ECE"],subjects=[], axis=1)
    sweep_df["model_uncertainty"] = sweep_df["model_uncertainty"].replace(uncertainty_names)
    sweep_df["dataset"] = sweep_df["dataset"].replace(dataset_names)
    sweep_df = sweep_df[metrics + params]

    sweep_df.sort_values(by=params[1], inplace=True)

    for metric in metrics:
        fig, ax = plt.subplots(1, 1)
        sns.barplot(data=sweep_df, x=params[0], hue=params[1], y=metric, errorbar="sd")
        if "Corr" in metric:
            ax.set_ylim(0,1)
        ax.set_ylabel(metric_names[metric])
        ax.set_xlabel("Dataset")
        ax.legend(title = "Uncertainty Model")

        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., 0),#p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')
        suffix = "_masked" if mask else ""
        plt.savefig(os.path.join(save_dir, f"uncertainty_performance_{metric}{suffix}.pdf"))
    return sweep_df
sweep_df = plot_uncertainty_performance(runs_results_df, mask=True)

make_latex(sweep_df, metric="Test_MAE", parameter="model_uncertainty")
make_latex(sweep_df, metric="Val_MAE", parameter="model_uncertainty")
make_latex(sweep_df, metric="Test_ECE", parameter="model_uncertainty", digits=3)

# %% Uncertainty calibration

sweeps = ["qu24hmkc", # uncertainty
          "sc724fuu", # uncertainty ensemble
          #"detvisbb", # supervised
          "ey2yrcwv", #supervised
          ]
partition = "test"
sweep_df = runs_results_df[runs_results_df["sweep_id"].isin(sweeps)]

sweep_df = sweep_df[sweep_df["batch_size"] == 512]
sweep_df = sweep_df[sweep_df["postprocessing"] == "raw"]
sweep_df = sweep_df[sweep_df["model_uncertainty"] != "bnn_pretrained_firstlast"]
sweep_df["pred_path"] = sweep_df.apply(find_prediction_dir, axis=1, partition=partition)
sweep_df.replace(dataset_names, inplace=True)
sweep_df.replace(uncertainty_names, inplace=True)
sweep_df_grouped = sweep_df.groupby(["model_uncertainty", "dataset"]).aggregate({"Test_MAE": ["mean", "std"], "Test_Corr": ["mean", "std"], "Test_NLL": ["mean", "std"], "Test_ECE": ["mean", "std"], "pred_path": lambda x: list(x)})

# uncertainty model
num_datasets = len(sweep_df_grouped.index.levels[1])
num_uncert = len(sweep_df_grouped.index.levels[0])

fig_ex, ax_ex = plt.subplots(1, num_datasets, figsize=(3*num_datasets,3), sharex=True, sharey=True,tight_layout=True)
fig_conf, ax_conf = plt.subplots(num_uncert, num_datasets, figsize=(3*num_datasets,3*num_uncert), sharex=True, sharey=True, tight_layout=True)
fig_conf_max, ax_conf_max = plt.subplots(num_uncert, num_datasets, figsize=(3*num_datasets,3*num_uncert), sharex=True, sharey=True,tight_layout=True)
fig_unc, ax_unc = plt.subplots(num_uncert, num_datasets, figsize=(3*num_datasets,3*num_uncert), sharex=True, sharey=True,tight_layout=True)
fig_reg, ax_reg = plt.subplots(num_uncert, num_datasets, figsize=(3*num_datasets,3*num_uncert), sharex=True, sharey=True,tight_layout=True)


for j, uncert in enumerate(sweep_df_grouped.index.levels[0]):

    for i, dataset in enumerate(sweep_df_grouped.index.levels[1]):
        row = sweep_df_grouped.loc[(uncert, dataset)]
        name = row.name[0]
        print(name)
        pred_paths = row["pred_path"].iloc[0]
        df_pred = pd.concat([pd.read_pickle(pred_path) for pred_path in pred_paths])
        plot_excluion(df_pred.hr_true, df_pred.hr_pred, df_pred.uncertainty, name=name, ax=ax_ex[i])
        confidence_plot(df_pred.uncertainty*90, df_pred.hr_pred,df_pred.hr_true,  name=name, distribution="laplace", ax=ax_reg[j,i])
        plot_hit_rate(df_pred.hr_true, df_pred.probs, name=name, hrmin=30, hrmax = 120, n_bins_divisor=4, n_bins_original=64, all_probs=True, ax=ax_conf[j,i])
        plot_hit_rate(df_pred.hr_true, df_pred.probs, name=name, hrmin=30, hrmax = 120, n_bins_divisor=4, n_bins_original=64, all_probs=False, ax=ax_conf_max[j,i])
        plot_uncert_vs_mae(df_pred.hr_true, df_pred.hr_pred, df_pred.uncertainty*90, name=name, bins=np.linspace(0,120, 12), ax=ax_unc[j,i])
        ax_unc[j,i].set_ylim(0,120)
        ax_unc[j,i].set_title(name)
        ax_conf[j,i].set_title(name)
        ax_conf_max[j,i].set_title(name)
        ax_reg[j,i].set_title(name)


fig_conf.savefig(os.path.join(save_dir, f"uncert_confidence_{partition}.pdf"))
fig_conf_max.savefig(os.path.join(save_dir, f"uncert_confidence_max_{partition}.pdf"))
fig_unc.savefig(os.path.join(save_dir, f"uncert_mae_{partition}.pdf"))
fig_reg.savefig(os.path.join(save_dir, f"uncert_reg_{partition}.pdf"))
fig_ex.savefig(os.path.join(save_dir, f"uncert_exclusion_{partition}.pdf"))
# %% postprocessing performance

#Postprocessing

def plot_postprocessing(runs_df):
    sweeps = ["qu24hmkc", # uncertainty
            "detvisbb" # supervised
            ]


    metrics = ["Post_Val_MAE", "Post_Val_Corr", "Post_Test_MAE", "Post_Test_Corr", "Post_Test_NLL", "Post_Test_ECE","Post_Val_NLL", "Post_Val_ECE"]
    params = ["dataset",  "postprocessing"]

    sweep_df = runs_df[runs_df["sweep_id"].isin(sweeps)]
    sweep_df = sweep_df[sweep_df["batch_size"] == 512]
    sweep_df.loc[sweep_df["model_uncertainty"] == "none", "Post_Test_MAE"] = sweep_df.loc[sweep_df["model_uncertainty"] == "none"]["Test_MAE"]
    sweep_df.loc[sweep_df["postprocessing"] == "none", "Post_Test_MAE"] = sweep_df.loc[sweep_df["postprocessing"] == "none"]["Test_MAE"]
    sweep_df = sweep_df[(sweep_df["model_uncertainty"] == "gaussian_classification")]
    #sweep_df["pred_path"] = sweep_df.apply(lambda x: os.path.join(x.model_dir_name, x.lincl_model_file.replace(".pt", f"Test_{x.model_uncertainty}_{x.postprocessing}.pickle")), axis=1)
    sweep_df.replace(dataset_names, inplace=True)
    sweep_df.replace(postprocessing_names, inplace=True)
    sweep_df = sweep_df[metrics + params]

    sweep_df.sort_values(by=params[1], inplace=True)
    for metric in ["Post_Test_MAE", "Post_Val_MAE"]:#, "Post_Test_NLL", "Post_Test_ECE","Post_Val_NLL", "Post_Val_ECE","Post_Val_MAE", "Post_Val_Corr"]:
        fig, ax = plt.subplots(1, 1)
        sns.barplot(data=sweep_df, x=params[0], hue=params[1], y=metric, ci="sd")
        if "Corr" in metric:
            ax.set_ylim(0,1)
        #ax.set_ylabel(metric_names[metric])
        ax.set_xlabel("Dataset")
        ax.legend(title = "Uncertainty Model")

        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., 0),#p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')
        plt.savefig(os.path.join(save_dir, f"postprocessing_{metric}.pdf"))
    
    return sweep_df
sweep_df = plot_postprocessing(runs_results_df)
make_latex(sweep_df, metric="Post_Test_MAE", parameter="postprocessing")

# %% postprocessing & uncertainty

#
sweeps = [
            "qu24hmkc", # uncertainty
            "detvisbb", # supervised
            "sc724fuu", # uncertainty ensemble
            #"yh3ys8wr" # uncertainty ssl
            ]


metrics = ["Post_Val_MAE", "Post_Val_Corr", "Post_Test_MAE", "Post_Test_Corr", "Post_Test_NLL", "Post_Test_ECE","Post_Val_NLL", "Post_Val_ECE"]
params = ["dataset",  "postprocessing", "model_uncertainty"]
for mask in [False]:
    fig, axes = plt.subplots(1, 2, figsize=(12,6), dpi=600, sharex=True, sharey=True, tight_layout=True)
    for k, dataset in enumerate(["appleall", "max_v2"]):
        ax = axes[k]
        sweep_df = runs_results_df[runs_results_df["sweep_id"].isin(sweeps)]
        sweep_df = sweep_df[sweep_df["batch_size"] == 512]
        if mask:
            pid_masks = get_pid_mask(sweep_df)
            sweep_df = sweep_df.apply(mask_metrics, pid_masks = pid_masks, axis=1)
        sweep_df.loc[sweep_df["model_uncertainty"] == "none", "Post_Test_MAE"] = sweep_df.loc[sweep_df["model_uncertainty"] == "none"]["Test_MAE"]
        sweep_df.loc[sweep_df["postprocessing"] == "none", "Post_Test_MAE"] = sweep_df.loc[sweep_df["postprocessing"] == "none"]["Test_MAE"]
        sweep_df.loc[sweep_df["model_uncertainty"] == "none", "Post_Val_MAE"] = sweep_df.loc[sweep_df["model_uncertainty"] == "none"]["Val_MAE"]
        sweep_df.loc[sweep_df["postprocessing"] == "none", "Post_Val_MAE"] = sweep_df.loc[sweep_df["postprocessing"] == "none"]["Val_MAE"]
        sweep_df = sweep_df[sweep_df["dataset"] == dataset]
        sweep_df = sweep_df[sweep_df["model_uncertainty"]!= "bnn_pretrained_firstlast"]
        sweep_df.replace(dataset_names, inplace=True)
        sweep_df.replace(postprocessing_names, inplace=True)
        sweep_df.replace(uncertainty_names, inplace=True)
        sweep_df["Test_MAE_diff"] = runs_results_df[runs_results_df["sweep_id"] == "detvisbb"]["Test_MAE"].mean() - sweep_df["Post_Test_MAE"]
        sweep_df["Test_MAE_diff"].clip(lower=0, inplace=True)
        metric = "Post_Val_MAE"
        sweep_df_table = sweep_df.pivot_table(index=params[1], values=metric, columns=params[2], aggfunc=["mean", "std"])
        #result_tables[(dataset, mask, metric)] = sweep_df_table
        #sweep_df_table.fillna(0, inplace=True)
        # Plotting the mean values with annotations of mean and std
        sns.heatmap(sweep_df_table["mean"], annot=False, cmap='viridis_r', fmt=".3f", cbar=True,
                    annot_kws={"fontsize": 8, "fontweight": "bold", "color": "black"},
                    cbar_kws={'label': metric}, ax=ax, vmin=2.7, vmax=6)

        for i in range(sweep_df_table["mean"].shape[0]):
            for j in range(sweep_df_table["std"].shape[1]):
                ax.text(j + 0.5, i + 0.35,f"{sweep_df_table['mean'].iloc[i,j]:.2f}", #\n±{pivot_df_std.iloc[i, j]:.2f}",
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white', fontsize=12, fontweight='bold')
                ax.text(j + 0.5, i + 0.65,f"±{sweep_df_table['std'].iloc[i, j]:.2f}",
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white', fontsize=12, fontweight="normal")

        ax.set_ylabel("Postprocessing")
        ax.set_xlabel("Uncertainty Model")
        cbar = ax.collections[0].colorbar
        # tilt x labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_horizontalalignment('right')
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)
        cbar.set_label(metric_names[metric])
        ax.set_title(f"{dataset_names[dataset]}")
    save_name = f"postprocessing_uncert_{metric}_mask" if mask else f"postprocessing_uncert_{metric}"
    plt.savefig(os.path.join(save_dir, f"{save_name}.pdf"), bbox_inches='tight')
    #plt.savefig(os.path.join(save_dir, f"{save_name}.png"), bbox_inches='tight')


#%%
res = []
for dataset in ["appleall", "max_v2"]:
    res_t = result_tables[(dataset, False, 'Post_Test_MAE')].iloc[:4,[1,3]]
    res.append((res_t - res_t.iloc[3,:]).mean(axis=1))
(res[0] + res[1])/2



# %% plot predictions
# look at some predictions

sweeps = [#"qu24hmkc", # uncertainty
            #"detvisbb", # supervised
            #"sc724fuu", # uncertainty ensemble
            "yrlpx4z9", # ssl uncertainty post
            ]

sweep_df = runs_results_df[runs_results_df["sweep_id"].isin(sweeps)]
sweep_df["pred_dir"] = sweep_df.apply(find_prediction_dir, axis=1, partition="test")
run = sweep_df[(sweep_df["model_uncertainty"] == "NLE") & (sweep_df["postprocessing"] == "sumprod")]
run = run[(run["split"] == 1)]
df_run = pd.concat(run.apply(load_predictions, axis=1).to_list(), axis=0)
df_run["subject"] = df_run["pid"].apply(lambda x: x[0])
df_run = df_run[df_run["subject"] == 844359]


#%%
for dataset in ["appleall", "max_v2"]:
    df_run_dataset = df_run[df_run["dataset"] == dataset]
    if len(df_run_dataset) == 0:
        continue
    plot_predictions(df_run_dataset, plot_uncertainty=True, mode="post")
    plt.savefig(os.path.join(save_dir, f"predictions_uncertainty_post_{dataset}_844359.pdf"), bbox_inches='tight')



#%%


def plot_scatter(df, ax=None, title=None, post=False, hr_lim = [30,120]):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    pids = df.pid.apply(lambda x:x[0]).astype(int).astype(str)
    pids.name = "Subject ID"

    if post:
        hr_pred = df["hr_pred_post"]
    else:
        hr_pred = df["hr_pred"]
    hr_true = df["hr_true"]

    sns.scatterplot(x=hr_pred, y=hr_true, hue=pids, ax=ax, palette = "viridis", alpha=0.5, s=4)
    ax.set_xlabel("predicted HR")
    ax.set_ylabel("true HR")
    ax.set_ylim(hr_lim)
    ax.set_xlim(hr_lim)
    ax.plot(hr_lim, hr_lim, 'k--')

    corr = hr_true.corr(hr_pred)
    mae = (hr_true - hr_pred).abs().mean()
    ax.text(0.68,  0.08, f'r = {corr:.2f} \nMAE = {mae:.2f} bpm', fontsize=12, color='black', transform=ax.transAxes,
         bbox=dict(facecolor='green', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.5'))
    ax.legend(loc='upper left', title="Subject ID")

    if title is not None:
        ax.set_title(title)
    return ax


fig, axes = plt.subplots(1, 2, figsize=(11,5), sharex=True, sharey=True, tight_layout=True)

for i, dataset in enumerate(["appleall", "max_v2"]):
    df_run_dataset = df_run[df_run["dataset"] == dataset]
    plot_scatter(df_run_dataset, post=True, ax = axes[i], title=dataset_names[dataset])

fig.savefig(os.path.join(save_dir, "results_scatter.pdf"), dpi=300)
fig.savefig(os.path.join(save_dir, "results_scatter.png"), dpi=300)


#%%

def plot_bland_altmann(df, ax=None, title=None, post=False):

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    
    if post:
        hr_pred = df["hr_pred_post"]
    else:
        hr_pred = df["hr_pred"]
    hr_true = df["hr_true"]

    pids = df.pid.apply(lambda x:x[0]).astype(int).astype(str)
    pids.name = "Subject ID"

    means = (hr_pred + hr_true) / 2
    differences = hr_true - hr_pred
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)

    #g = sns.JointGrid()
    sns.scatterplot(x=means, y=differences, ax=ax, hue=pids,  palette = "viridis", alpha=0.5, s=4)
    #sns.kdeplot(y=differences, ax=ax, hue=pids, palette = "viridis", alpha=0.5, bw_adjust=0.5, fill=True)
    sns.kdeplot(x=means, y=differences, levels=10, color="w", linewidths=1, ax=ax)
    #sns.jointplot(x=means, y=differences, ax=ax, hue=pids,  palette = "viridis", alpha=0.5, s=4)
    ax.axhline(mean_diff, color='red', linestyle='--')
    ax.axhline(mean_diff + 1.96*std_diff, color='green', linestyle='--')
    ax.axhline(mean_diff - 1.96*std_diff, color='green', linestyle='--')

    share_samples_inside_std = ((differences < mean_diff + 1.96*std_diff) & (differences > mean_diff - 1.96*std_diff)).sum() / len(differences)
    print(f"Share of samples inside 95% CI: {share_samples_inside_std:.2f}")
    ax.set_xlabel('Means of True HR and Predicted HR')
    ax.set_ylabel('True HR - Predicted HR')
    ax.legend(loc='upper left', title="Subject ID")
    ax.set_ylim(-30, 30)
    ax.set_xlim(40, 100)

    if title is not None:
        ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(11,5), sharex=True, sharey=True, tight_layout=True, dpi=900)

for i, dataset in enumerate(["appleall", "max_v2"]):
    df_run_dataset = df_run[df_run["dataset"] == dataset]
    plot_bland_altmann(df_run_dataset, post=True, ax = axes[i], title=dataset_names[dataset])

fig.savefig(os.path.join(save_dir, "results_bland_altmann.pdf"))
fig.savefig(os.path.join(save_dir, "results_bland_altmann.png"))

#%% ablations
sweeps = [["tkbeegcg"], # window size
          ["bueh0d7v"], # step size
          ["3924tcru", "uq1xh1vv"], #sampling rate
          ]
parameters = ["window_size", "take_every_nth_train", "sampling_rate"]
parameters_name = ["Window Size [s]", "Step Size [s]", "Sampling rate [Hz]"]
for i, (sweep, param, param_name) in enumerate(zip(sweeps, parameters, parameters_name)):

    sweep_df = runs_ablation_df[runs_ablation_df["sweep_id"].isin(sweep)]
    if len(sweep_df) == 0:
        sweep_df = runs_results_df[runs_results_df["sweep_id"].isin(sweep)]
    sweep_df = sweep_df[[param, "Val_MAE", "Test_MAE", "Val_Corr", "Test_Corr", "dataset"]]
    for metric in ["MAE", "Corr"]:

        sweep_df_m = sweep_df.melt(id_vars=sweep_df.columns.difference([f"Val_{metric}", f"Test_{metric}"]), var_name= "partition", value_name=metric)
        sweep_df_m.partition.replace({f"Test_{metric}": "Test", f"Val_{metric}": "Validation"}, inplace=True)
        sweep_df_m["dataset"].replace(dataset_names, inplace=True)
        sweep_df_m["part_dataset"] = sweep_df_m["partition"] + " - " + sweep_df_m["dataset"]

        fig, ax = plt.subplots(1, 1, figsize=(4,3.5), tight_layout=True, dpi=600)

        sns.lineplot(data=sweep_df_m, x=param, y=metric, hue='part_dataset', errorbar=("sd", 0.2), err_style="bars", ax=ax, marker="o")
        
        if metric == 'Corr':
            ax.set_ylim(0,1)
        else:

            ax.set_ylim(min(3, sweep_df_m[metric].min()), max(6, sweep_df_m[metric].max()))
        ax.set_ylabel(metric_names[metric])
        ax.set_xlabel(param_name)
        ax.legend()
        #plt.legend(title = "Ablation", loc='upper left')
        
        plt.savefig(os.path.join(save_dir, f"ablation_{param}_{metric}.pdf"))




#%%

sweep = "bueh0d7v"
metric = ["Test_MAE", 'Val_MAE']
parameters = ["take_every_nth_train"]
sweep_df = runs_ablation_df[runs_ablation_df["sweep_id"] == sweep]
sweep_df = sweep_df.melt(id_vars=runs_ablation_df.columns.difference(["Val_MAE", "Test_MAE"]), var_name= "partition", value_name="MAE")
sweep_df.partition.replace({"Test_MAE": "Test", "Val_MAE": "Val"}, inplace=True)

# %% AttentionNetwork

# AttentionNetwork

sweeps = [
    "3u8ez9r3", #attentionsupervised
    "detvisbb" #supervised
]

metrics = ["Val_MAE", "Test_MAE"]
params = ["dataset",  "backbone"]

sweep_df = runs_results_df[runs_results_df["sweep_id"].isin(sweeps)]
sweep_df = sweep_df[sweep_df["batch_size"] == 512]
for metric in metrics:
    fig, ax = plt.subplots(1, 1)
    sns.barplot(data=sweep_df, x=params[0], hue=params[1], y=metric, ci="sd", ax=ax)
    #plt.savefig(os.path.join(save_dir, f"attention_network_{metric}.pdf"))



# %%

# Final Results

sweeps = [
    "c5niepix", # signal processing
    #"u7e7hc8c", # supervised baseline old
    #"detvisbb", # supervised baseline #2
    "ey2yrcwv", # supervised baseline #3
    "0xhhj3lg", # ssl model (best on test set)
    #"54uzziuj", #ssl model (best on validation set)
    "qu24hmkc", # uncertainty
    "sc724fuu", # uncertainty ensemble
    "3u8ez9r3", #attentionsupervised
    #"yh3ys8wr", #ssl uncertainty + postprocessing old
    "yrlpx4z9", #ssl uncertainty + postprocessing

]

sweep_df = runs_results_df[runs_results_df["sweep_id"].isin(sweeps)]
sweep_df.loc[(~sweep_df["model_uncertainty"].isin(["none", np.nan])) & (sweep_df["postprocessing"] != "raw") & (sweep_df["framework"] != "supervised"), "framework" ] = "ssl + uncertainty + postprocessing"
sweep_df = sweep_df[(sweep_df["framework"] != "supervised") | (sweep_df["random_seed"] == 10)]
sweep_df = sweep_df[(sweep_df["framework"] != "supervised") | (sweep_df["batch_size"] == 512)]
sweep_df = sweep_df[(sweep_df["framework"] != "nnclr") | (sweep_df["pretrain_subsample"].isin([0.2]))]
sweep_df = sweep_df[(sweep_df["framework"] != "simclr") | (sweep_df["lr_finetune_lstm"] == 0.0005) & (sweep_df["lr_finetune_backbone"] == 0.0001)]
sweep_df = sweep_df[(sweep_df["model_uncertainty"].isin(["none", np.nan])) | 
                    ((sweep_df["framework"] == "supervised")& (sweep_df["model_uncertainty"] == "ensemble")& (sweep_df["dataset"] == "max_v2") & (sweep_df["postprocessing"] == "raw")) |
                    ((sweep_df["framework"] == "supervised")& (sweep_df["model_uncertainty"] == "ensemble")& (sweep_df["dataset"] == "appleall") & (sweep_df["postprocessing"] == "raw")) |
                    ((sweep_df["framework"] == "supervised")& (sweep_df["model_uncertainty"] == "NLE")& (sweep_df["dataset"] == "appleall") & (sweep_df["postprocessing"] == "sumprod")) |
                    ((sweep_df["framework"] == "supervised")& (sweep_df["model_uncertainty"] == "NLE")& (sweep_df["dataset"] == "max_v2") & (sweep_df["postprocessing"] == "sumprod")) |
                    ((sweep_df["framework"] != "supervised")& (sweep_df["model_uncertainty"] == "NLE")& (sweep_df["dataset"] == "appleall") & (sweep_df["postprocessing"] == "sumprod")) |
                    ((sweep_df["framework"] != "supervised")& (sweep_df["model_uncertainty"] == "NLE")& (sweep_df["dataset"] == "max_v2") & (sweep_df["postprocessing"] == "sumprod")) 
                    ]
#sweep_df.loc[(sweep_df["framework"] == "nnclr") & (sweep_df["pretrain_subsample"] == 0.1), "framework"] = "pretrained_new"
sweep_df.loc[(~sweep_df["model_uncertainty"].isin(["none", np.nan])) & (sweep_df["postprocessing"] != "raw") & (sweep_df["framework"] != "supervised"), "framework" ] = "ssl + uncertainty + postprocessing"
sweep_df.loc[(~sweep_df["model_uncertainty"].isin(["none", np.nan])) & (sweep_df["postprocessing"] == "raw")& (sweep_df["framework"] == "supervised"), "framework"] = "uncertainty"
sweep_df.loc[(~sweep_df["model_uncertainty"].isin(["none", np.nan])) & (sweep_df["postprocessing"] != "raw") & (sweep_df["framework"] == "supervised"), "framework"] = "uncertainty + postprocessing"
sweep_df.loc[sweep_df["backbone"] == "AttentionCorNET", "framework"] = "attention"
corr_function  = lambda a,b: pd.Series(a).corr(pd.Series(b))
sweep_df.groupby("framework").count()['Test_MAE']

#%%
warnings.filterwarnings('ignore', category=DeprecationWarning)
def sort_table(value):
    breakpoint()
    try:
        value = float(value)
        return value
    except:
        if "In-House" in value:
            return 30
        else:
            return 10000000
        
def compute_metric_per_subject(run, metrics=["Test_MAE", "Post_Test_MAE"]):
    res_subject = {}
    test_metrics = [el for el in metrics if "Test" in el]
    val_metrics = [el for el in metrics if "Val" in el]
    partition_subjects = {}

    partitions =  {
                    "Test": test_metrics, 
                    "Val": val_metrics
                   }
    for partition_name, partition_metrics in partitions.items():
    
        if len(partition_metrics) == 0:
            continue

        pred_dir = find_prediction_dir(run, partition_name)
        pred_df = load_predictions(run, pred_dir=pred_dir)

        if pred_df is None:
            print(f"Missing predictions for {run.name}, {pred_dir}")
            return None

        pred_df["subject"] = pred_df["pid"].apply(lambda x: f"{x[0]:.0f}")

        for metric in partition_metrics:
        
            if "MAE" in metric and "Post" not in metric:
                mae_subject = (pred_df.hr_true - pred_df.hr_pred).abs().groupby(pred_df["subject"]).mean()
                mae_subject[f"{run.dataset}_{partition_name}"] = (pred_df.hr_true - pred_df.hr_pred).abs().mean()
                res_subject[f"{partition_name}_MAE"] = mae_subject
                
            elif "Corr" in metric and "Post" not in metric:
                corr_subject = pred_df.groupby("subject").apply(lambda x: x.hr_true.corr(x.hr_pred))
                corr_subject[f"{run.dataset}_{partition_name}"] = pred_df.hr_true.corr(pred_df.hr_pred)
                res_subject[f"{partition_name}_Corr"] = corr_subject
        
            elif "MAE" in metric and "Post" in metric:
                if "hr_pred_post" in pred_df.columns:
                    post_mae_subject =(pred_df.hr_true - pred_df.hr_pred_post).abs().groupby(pred_df["subject"]).mean()
                    post_mae_subject[f"{run.dataset}_{partition_name}"] = (pred_df.hr_true - pred_df.hr_pred_post).abs().mean()
                    res_subject[f"Post_{partition_name}_MAE"] = post_mae_subject
                else: # postprocessing metrics not available, skip
                    continue

            elif "Corr" in metric and "Post" in metric:
                if "hr_pred_post" in pred_df.columns:
                    post_corr_subject = pred_df.groupby("subject").apply(lambda x: x.hr_true.corr(x.hr_pred_post))
                    post_corr_subject[f"{run.dataset}_{partition_name}"] = pred_df.hr_true.corr(pred_df.hr_pred_post)
                    res_subject[f"Post_{partition_name}_Corr"] = post_corr_subject
                else: # postprocessing metrics not available, skip
                    continue
            elif "quant90" in metric:
                quant95_subject = pred_df.groupby("subject").apply(lambda x: (x.hr_true - x.hr_pred).abs().quantile(0.90))
                quant95_subject[f"{run.dataset}_{partition_name}"] = (pred_df.hr_true - pred_df.hr_pred).abs().quantile(0.90)
                res_subject[f"{partition_name}_quant95"] = quant95_subject

            elif "quant" in metric:
                thr = 5
                pred_df["ae"] = (pred_df.hr_true - pred_df.hr_pred).abs()
                quant_subject = pred_df.groupby("subject")["ae"].apply(lambda x: len(x < thr) / len(x))
                quant_subject[f"{run.dataset}_{partition_name}"] = len(pred_df[pred_df["ae"] < thr]) / len(pred_df)
                res_subject[f"{partition_name}_quant"] = quant_subject

            
            else:
                raise ValueError(f"Metric {metric} not implemented")

        for sub in pred_df.subject.unique():
            partition_subjects[sub] = partition_name
        partition_subjects[f"{run.dataset}_{partition_name}"] = partition_name
        num_samples = pred_df.groupby("subject").size()
        num_samples[f"{run.dataset}_{partition_name}"] = pred_df.shape[0]
        res_subject[f"num_samples_{partition_name}"] = num_samples

            

    if len(res_subject) != 0:
        df_res = pd.DataFrame(res_subject)
        df_partition = pd.Series(partition_subjects)
        df_res["partition"] = df_partition
        return df_res
    return None

results = {}
for index, row in sweep_df.iterrows():
    mae_subjects = compute_metric_per_subject(row, metrics = ["Test_MAE", "Post_Test_MAE", "Test_Corr", "Post_Test_Corr", "Test_quant90", "Test_quant"])
    if mae_subjects is not None:
        results[index] = mae_subjects

# Constructing a single DataFrame from results
results_df = pd.concat(results, names=['id', 'subject'])
results_df = results_df.reset_index(level='subject', drop=False)  # Dropping subject level, adjust if needed
results_df["Test_MAE"] = results_df["Test_MAE"].fillna(0) + results_df["Val_MAE"].fillna(0) if "Val_MAE" in results_df.columns else results_df["Test_MAE"]
#results_df["MAE_post"] = results_df["Post_Test_MAE"].fillna(0) + results_df["Post_Val_MAE"].fillna(0)
results_df["MAE"] = results_df.apply(lambda row: row["Test_MAE"] if pd.isna(row["Post_Test_MAE"]) else row["Post_Test_MAE"], axis=1)
results_df["Corr"] = results_df.apply(lambda row: row["Test_Corr"] if pd.isna(row["Post_Test_Corr"]) else row["Post_Test_Corr"], axis=1)

# Joining results_df with sweep_df
final_df = results_df.join(sweep_df, how="inner", rsuffix="_sweep")
#%%
metric = "Corr"

final_df.sort_values(["dataset", "split"], inplace=True)
final_df_sel = final_df[final_df["partition"].isin(["Test"])]
final_df_sel_val = final_df_sel[final_df_sel["subject"].isin(["max_v2_Test", "appleall_Test"])].copy()
final_df_sel_val["MAE"] = final_df_sel_val['Val_MAE']
final_df_sel_val["Corr"] = final_df_sel_val['Val_Corr']
final_df_sel_val["split"] = "Validation"
final_df_sel_val["subject"] = final_df_sel_val["subject"].apply(lambda x: x.replace("Test", "Val"))
final_df_sel = pd.concat([final_df_sel, final_df_sel_val])
final_df_table = final_df_sel.pivot_table(index=["subject"], values=metric, columns="framework", aggfunc=["mean", "std"])
final_df_table = final_df_table.swaplevel(0,1,axis=1)

report_table = final_df_table[["subject_median", "median", "Bioglass", "supervised", "attention", "nnclr", "uncertainty", "uncertainty + postprocessing", "ssl + uncertainty + postprocessing"]]

report_table.rename(columns=signal_processing_names, inplace=True)
report_table.rename(columns={"uncertainty":"Uncertainty", "supervised": "Baseline", "nnclr": "Pretrained", "uncertainty + postprocessing" : "Uncertainty + Postprocessing", "ssl + uncertainty + postprocessing" : "SSL + Uncertainty + Postprocessing","attention": "Attention"}, inplace=True)
report_table.rename(index={"max_v2_Test": "In-House", "appleall_Test": "Apple Watch", "max_v2_Val": "In-House (Val)", "appleall_Val": "Apple Watch (Val)"}, inplace=True)
report_table.sort_index(inplace=True, key=lambda x: x.map(sort_table))


quant_per_dataset = final_df[(final_df["framework"] == "ssl + uncertainty + postprocessing") & (final_df["subject"].isin(["appleall_Test", "max_v2_Test"]))].groupby("dataset")["Test_quant"].mean()
print("Amount of samples within 5 bpm", quant_per_dataset)

make_latex_table(report_table, n_digits=2, best_bold=True, min_best=metric=="MAE")
report_table
# %%

for fram in final_df["framework"].unique():
    df_framework = final_df[(final_df["framework"] == fram) & (final_df["subject"].apply(lambda x : "Test" in x))]
    a = (df_framework["MAE"] * df_framework["num_samples_Test"]) / df_framework["num_samples_Test"].sum()
    row_to_insert = df_framework.iloc[0].copy()
    row_to_insert["MAE"] = a.sum()
    row_to_insert["subject"] = f"Test"
    row_to_insert["num_samples_Test"] = df_framework["num_samples_Test"].sum()
    row_to_insert["partition"] = "Test"
    row_to_insert["dataset"] = "all"
    row_to_insert.index = row_to_insert.index + "_all"

    final_df = pd.concat([final_df, row_to_insert], axis=0, ignore_index=True)
# %%
