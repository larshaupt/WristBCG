#%%
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models.backbones import *
from models.loss import *
from trainer import *
from argparse import Namespace
import classical_utils
from main import get_parser
metric_names = ["Angle Changes", "Absolute Max", "STD", "Mean", "MAD"]


#%%
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
    model_test = model_test.to(DEVICE)


    return model_test, args, DEVICE


#%%
def run_inference(json_file, X_test):

    P = []
    for split in range(5):
        _P = []
        model_test, args, DEVICE = load_model(json_file, mode="finetune")

        for i, x in enumerate(X_test):
            print(i)
            x = torch.Tensor(x).unsqueeze(0).to(DEVICE).float()
            p,_ = model_test(x)
            p = p.detach().cpu().numpy()
            _P.append(p.reshape(1,-1))

        _P = np.concatenate(_P).squeeze()
        _P = _P*(args.hr_max - args.hr_min) + args.hr_min
        P.append(_P)

    P = np.stack(P, axis=0)
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
        print(i)
        p  = classical_utils.compute_hr_bioglass(x, fs)
        P_bioglass.append(p)
    P_bioglass = np.array(P_bioglass)
    return P_bioglass

#%%
split = 0
data_dir_apple = "/local/home/lhauptmann/thesis/data/AppleDataset_22to7_100Hz_all_metrics"
data_dir_max = "/local/home/lhauptmann/thesis/data/MaxDataset_v2_metrics"

json_file_apple = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_CorNET_dataset_appleall_split{split}_eps60_bs512_config.json"
json_file_max = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_CorNET_dataset_max_v2_split{split}_eps60_bs512_config.json"

predictions = {}

for dataset, data_dir in zip(["apple", "max"], [data_dir_apple, data_dir_max]):
    X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = classical_utils.load_dataset(data_dir, 0,normalize=True)
    if dataset == "apple":
        json_file = json_file_apple
    elif dataset == "max":
        json_file = json_file_max

    for approach, approach_func in zip(["baseline", "bioglass"], [run_inference, run_inference_bioglass]):
        P = approach_func(json_file, X_test)
        print_results(P, Y_test)

        predictions[(approach, dataset)] = (P, Y_test, metrics_test)

# %%
def exclusion(P, Y, M):

    
    res_dict = {}
    sort_indices = np.argsort(M, axis=0)
    abs_diffs = np.abs(Y-P)
    for metric in range(M.shape[1]):  
        abs_diffs_sorted = abs_diffs[:, sort_indices[:,metric]]
        maes = []
        for p in np.linspace(0,1,100):
            abs_diffs_p = abs_diffs_sorted[:,:int(abs_diffs_sorted.shape[1]*p)]
            mae_p = np.mean(abs_diffs_p)
            maes.append(mae_p)
        maes = np.array(maes)
        res_dict[metric_names[metric]] = maes
    return res_dict

def metric_minimum(res_dict):
    min_dict = {}
    for metric, maes in min_dict.items():
        min_mae_ind = np.nanargmin(maes[-30:]) + len(maes) - 30
        min_mae = maes[min_mae_ind]
    
        res_dict[metric] = (min_mae, min_mae_ind, metric)

    min_df = pd.DataFrame(data=min_dict).T
    min_df.columns = ["MAE", "MAE_index", "metric_name"]
    return min_df


def metric_quantile(res_dict, quantile = 70):
    quantile_dict = {}
    for metric, maes in res_dict.items():
        mae_ind = int(len(maes)*quantile/100)
        mae = maes[mae_ind]
    
        quantile_dict[metric] = (mae, mae_ind, metric)

    quantile_df = pd.DataFrame(data=quantile_dict).T
    quantile_df.columns = ["MAE", "MAE_index", "metric_name"]
    return quantile_df

def plot_exclusion(res_dict, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for metric, maes in res_dict.items():
        ax.plot(np.linspace(0,1,len(maes)), maes, label=metric)
    plt.legend()
    
# %%
fig, axes = plt.subplots(2,2)
res_dict = {}
for i, dataset in enumerate(["apple", "max"]):
    for j, approach in enumerate(["baseline", "bioglass"]):
        P, Y, M = predictions[(approach, dataset)]
        if P.ndim == 1:
            P = P.reshape(1,-1)
        res = exclusion(P, Y, M)
        res_dict[(dataset, approach)] = res
        plot_exclusion(res, ax=axes[i,j])
        axes[i,j].set_title(f"{dataset} {approach}")

fig.tight_layout()
res_df = pd.DataFrame(res_dict).reset_index()
res_df

# %%
quantile = 90
res_bioglass = exclusion(P_bioglass.reshape(1,-1), Y, M)
res_baseline = exclusion(P, Y, M)
plot_exclusion(res_bioglass)
res_df_bioglass = metric_quantile(res_bioglass, quantile=quantile)
res_df_baseline = metric_quantile(res_baseline, quantile=quantile)
res_df = pd.concat({"baseline":res_df_baseline, "bioglass":res_df_bioglass}, axis=0)
res_df
# %%
fig, axes = plt.subplots(2,5, sharex=False, sharey=True, figsize=(20,10))

for i, (dataset, data_dir) in enumerate(zip(["apple", "max"], [data_dir_apple, data_dir_max])):
    X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = classical_utils.load_dataset(data_dir, 0,normalize=True)

    for j in range(metrics_test.shape[1]):
        axes[i,j].hist(metrics_test[:,j], bins=100, log=True)
        axes[i,j].set_title(f"{dataset} {metric_names[j]}")


# %%
