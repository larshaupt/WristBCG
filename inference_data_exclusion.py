#%%
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle

from models.backbones import *
from models.loss import *
from trainer import *
from argparse import Namespace
import classical_utils
from main import get_parser
metric_names = ["Angle Changes", "Absolute Max", "STD", "Mean", "MAD"]


#%%

def make_latex_table(df, n_digits = 2, best_bold=False, min_best=True, best_per_row=True):
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
        elif (var, "mean") in df.columns:
            formatted_df[var] = df[(var, 'mean')].apply(lambda x: f"{x:.{n_digits}f}")
        else:
            formatted_df[var] = df[var].apply(lambda x: f"{x:.{n_digits}f}")
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

    latex_str = formatted_df.to_latex(index=True, header=True)
    print(latex_str)

def load_model(json_file, mode="finetune", extra_args={}):

    with open(json_file) as json_file:
        json_args = json.load(json_file)


    parser = get_parser()
    args = parser.parse_args([])
    args_dict = vars(args)
    args_dict.update(json_args)
    args_dict.update(extra_args)
    args = Namespace(**args_dict)


    if not hasattr(args, "model_name"):
        split = 0
        args.model_name = args.backbone + '_'+args.dataset + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) + '_split' + str(split)

    if mode == "finetune":
        model_weights_path = args.lincl_model_file
    elif mode == "pretrain":
        model_weights_path = args.pretrain_model_file
    else:
        raise ValueError(f"Mode {mode} not supported")

    # Testing
    #######################
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE, 'dataset:', args.dataset)


    # Initialize and load test model
    if args.backbone == 'FCN':
        model_test = FCN(n_channels=args.n_feature, n_classes=args.n_class, input_size=args.input_length, backbone=True)
    elif args.backbone == 'DCL':
        model_test = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, input_size=args.input_length, conv_kernels=64, kernel_size=5, LSTM_units=args.lstm_units, backbone=True)
    elif args.backbone == 'LSTM':
        model_test = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=args.lstm_units, backbone=True)
    elif args.backbone == 'AE':
        model_test = AE(n_channels=args.n_feature, input_size=args.input_length, n_classes=args.n_class, outdim=128, backbone=True)
    elif args.backbone == 'CNN_AE':
        model_test = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, embdedded_size=128, input_size=args.input_length, backbone=True, n_channels_out=args.n_channels_out)
    elif args.backbone == 'Transformer':
        model_test = Transformer(n_channels=args.n_feature, input_size=args.input_length, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True)
    elif args.backbone == "CorNET":
        model_test = CorNET(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=args.num_kernels, kernel_size=args.kernel_size, LSTM_units=args.lstm_units, backbone=True, input_size=args.input_length, rnn_type=args.rnn_type)
    elif args.backbone == "TCN":
        model_test = TemporalConvNet(num_channels=[32, 64, 128], n_classes=args.n_class,  num_inputs=args.n_feature, input_length = args.input_length, kernel_size=16, dropout=0.2, backbone=True)
    elif args.backbone == "AttentionCorNET":
        model_test = AttentionCorNET(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=args.num_kernels, kernel_size=args.kernel_size, LSTM_units=args.lstm_units, backbone=True, input_size=args.input_length, rnn_type=args.rnn_type)
    else:
        raise NotImplementedError

    # Testing
    model_weights_dict = torch.load(model_weights_path, map_location = DEVICE)

    if mode == "finetune":
        classifier = setup_linclf(args, DEVICE, model_test.out_dim)
        model_test.set_classification_head(classifier)
        model_weights = load_best_lincls(args, device=DEVICE)


    elif mode == "pretrain":
        model_weights = model_weights_dict["model_state_dict"]

    model_test.load_state_dict(model_weights, strict=True)


    return model_test, args, DEVICE


def run_inference(json_file, X_test):
    batch_size = 64
    P = []

    model_test, args, DEVICE = load_model(json_file, mode="finetune")
    model_test = model_test.to(DEVICE)
    for i in range(0,len(X_test), batch_size):
        print(i)
        X = torch.Tensor(X_test[i:i+batch_size]).to(DEVICE).float()
        p,_ = model_test(X)
        del X
        p = p.detach().cpu().numpy()
        P.append(p)

    P = np.concatenate(P,axis=0).squeeze()
    P = P*(args.hr_max - args.hr_min) + args.hr_min

    return P

def print_results(P, Y):
    corr = np.ma.corrcoef(Y,P)[0,1]
    mae = np.nanmean(np.abs(Y-P))
    rmse = np.sqrt(np.nanmean((Y-P)**2))
    print(f"Correlation: {corr}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

def run_inference_bioglass(json_file, X_test, fs=100):
    P_bioglass = []
    for i, x in enumerate(X_test):
        if i % 100 == 0:
            print(i)
        p  = classical_utils.compute_hr_bioglass(x, fs)
        P_bioglass.append(p)
    P_bioglass = np.array(P_bioglass)
    return P_bioglass

#%%
predictions = {}
data_dir_apple = "/local/home/lhauptmann/thesis/data/AppleDataset_22to7_100Hz_all_metrics"
data_dir_max = "/local/home/lhauptmann/thesis/data/MaxDataset_v2_metrics"

for split in range(5):

    json_file_apple = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_CorNET_dataset_appleall_split{split}_eps60_bs512_config.json"
    json_file_max = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_CorNET_dataset_max_v2_split{split}_eps60_bs512_config.json"

    for dataset, data_dir in zip(["apple", "max"], [data_dir_apple, data_dir_max]):
        
        X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = classical_utils.load_dataset(data_dir, split,normalize=True)
        assert len(X_val) == len(Y_val) == len(pid_val) == len(metrics_val)
        assert len(X_test) == len(Y_test) == len(pid_test) == len(metrics_test)
        if dataset == "apple":
            json_file = json_file_apple
        elif dataset == "max":
            json_file = json_file_max
        
        metrics_test, metrics_val = metrics_test.tolist(), metrics_val.tolist()
        
        for approach, approach_func in zip(["baseline", "bioglass"], [run_inference, run_inference_bioglass]):
            print(approach, dataset, split)
            if split == 0:
                # initialize the element in the dict

                P_test = approach_func(json_file, X_test)
                P_test = P_test.reshape(-1)
                print_results(P_test, Y_test)
                predictions[(approach, dataset)] = (P_test.tolist(), Y_test.tolist(), metrics_test) 
                
            P_val = approach_func(json_file, X_val)
            print_results(P_val, Y_val)
            predictions[(approach, dataset)][0].extend(P_val.tolist())
            predictions[(approach, dataset)][1].extend(Y_val.tolist())
            predictions[(approach, dataset)][2].extend(metrics_val)
            print("Val", split, len(Y_val), len(metrics_val))

""" with open("/local/home/lhauptmann/thesis/CL-HAR/data_exclusion_predictions.pickle", "wb") as f:
    pickle.dump(predictions, f)
 """

# %%
with open("/local/home/lhauptmann/thesis/CL-HAR/data_exclusion_predictions1.pickle", "rb") as f:
    predictions = pickle.load(f)
#%%

df_preds = pd.DataFrame(predictions).T
df_preds.columns = ["P", "Y", "M"]
df_preds.index.names = ["approach", "dataset"]
df_preds = df_preds.applymap(np.array)
df_preds["abs_diff"] = df_preds.apply(lambda x: np.abs(x["P"]-x["Y"]), axis=1)
#%%
def exclusion(P, Y, M):

    
    res_dict = {}
    sort_indices = np.argsort(M, axis=0)
    abs_diffs = np.abs(Y-P)
    for metric in range(M.shape[1]):  
        abs_diffs_sorted = abs_diffs[:, sort_indices[:,metric]]
        maes = []
        for p in np.linspace(0,1,100):
            abs_diffs_p = abs_diffs_sorted[:,:int(abs_diffs_sorted.shape[1]*p)]
            mae_p = np.nanmean(abs_diffs_p)
            maes.append(mae_p)
        maes = np.array(maes)
        res_dict[metric_names[metric]] = maes
    return res_dict

def metric_minimum(exclusion):

    min_mae_ind = np.nanargmin(exclusion[-30:]) + len(exclusion) - 30
    min_mae = exclusion[min_mae_ind]

    return min_mae


def metric_quantile(exclusion, quantile = 70):
    
    mae_ind = int(len(exclusion)*quantile/100)
    mae = exclusion[mae_ind]

    return mae

def plot_exclusion(res_dict, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for metric, maes in res_dict.items():
        ax.plot(np.linspace(0,1,len(maes)), maes, label=metric)
    plt.legend()
    
# %%
def make_exclusions(predictions):
    fig, axes = plt.subplots(2,2)
    res_dict = {}
    for i, dataset in enumerate(["apple", "max"]):
        for j, approach in enumerate(["baseline", "bioglass"]):
            P, Y, M = predictions[(approach, dataset)]
            P, Y, M = np.array(P), np.array(Y), np.array(M)
            if P.ndim == 1:
                P = P.reshape(1,-1)
            res = exclusion(P, Y, M)
            res_dict[(dataset, approach)] = res
            plot_exclusion(res, ax=axes[i,j])
            axes[i,j].set_title(f"{dataset} {approach}")

    fig.tight_layout()
    res_df = pd.DataFrame(res_dict).reset_index()
    res_df.set_index("index", inplace=True)
    return res_df

#%%

df_ex = make_exclusions(predictions)
df_ex.applymap(metric_minimum)
df_ex.applymap(metric_quantile, quantile=70)

# %%

df_corr = pd.DataFrame({i:df_preds.apply(lambda row : pd.Series(row.abs_diff).corr(pd.Series(row.M[:,i])), axis=1) for i in range(5)})
df_corr.columns = metric_names
df_corr.rename(index={"apple": "Apple Watch", "max": "In-House", "baseline": "Learning-based", "bioglass": "Signal Processing"}, inplace=True)

df_corr = df_corr.groupby(["dataset", "approach"]).mean()
fig, ax = plt.subplots()

df_corr.plot(kind="bar", ax=ax)
ax.xaxis.set_tick_params(rotation=20)

ax.set_ylabel("Correlation")
ax.set_xlabel("(Dataset, Approach)")
plt.savefig("/local/home/lhauptmann/thesis/images/thesis/datametrics_correlation.pdf")

std_df = df_corr.T.groupby(level=0, axis=1).std()
mean_df = df_corr.T.groupby(level=0, axis=1).mean()

# Merge the dataframes
result_df = pd.concat([std_df, mean_df], axis=1, keys=['std', 'mean']).swaplevel(0,1, axis=1)

make_latex_table(result_df, n_digits=2, best_bold=True, min_best=False)

# %%

std = df_ex.apply(lambda row : np.std(np.array(row)),axis=1)*0.3
mean = df_ex.apply(lambda row : np.mean(row),axis=1)
x = np.linspace(0,100,100)
fig, ax = plt.subplots()

for metric in mean.index:
    ax.plot(x, mean[metric], label=metric)
    ax.fill_between(x, mean[metric]-std[metric], mean[metric]+std[metric], alpha=0.1)


ax.set_xlim(50,100)
ax.legend()
ax.set_ylim(3.5,5)
ax.set_xlabel("Percentage of excluded data [%]")
ax.set_ylabel("MAE [bpm]")
plt.savefig("/local/home/lhauptmann/thesis/images/thesis/datametrics_exclusion.pdf")


# %%
