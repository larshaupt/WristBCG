#%%
import os
import json
import pandas
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
import re
from utils import tsne, plot_true_pred

from models.backbones import *
from models.loss import *
from trainer import *
from argparse import Namespace
import classical_utils
from main import get_parser

from captum.attr import IntegratedGradients

split_res = {}

#%%
def predict_from_json(json_file, mode="finetune", extra_args={}, split_partition="test"):

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

    args.cuda = 1
    args.data_thr_hr = 0
    args.data_thr_max = 100
    args.data_thr_angle = 100
    args.data_thr_avg = 100

    # Testing
    #######################
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE, 'dataset:', args.dataset)

    # Load data
    train_loader, val_loader, test_loader = setup_dataloaders(args)
    if split_partition == "train":
        data_loader = train_loader
    elif split_partition == "val":
        data_loader = val_loader
    else:
        data_loader = test_loader


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


    X, Y, D, P = [], [], [], []
    model_test.eval()
    for i, (x, y, d) in enumerate(data_loader):

        x = x.to(DEVICE).float()
        y = y.to(DEVICE)
        d = d.to(DEVICE)
        p,_ = model_test(x)
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        d = d.detach().cpu().numpy()
        p = p.detach().cpu().numpy()
        X.append(x)
        Y.append(y)
        D.append(d)
        P.append(p)
        
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    D = np.concatenate(D)
    P = np.concatenate(P).squeeze()

    P = P*(args.hr_max - args.hr_min) + args.hr_min
    Y = Y*(args.hr_max - args.hr_min) + args.hr_min

    corr = np.corrcoef(Y, P)[0,1]
    mae = np.mean(np.abs(Y-P))
    rmse = np.sqrt(np.mean((Y-P)**2))
    print(f"Correlation: {corr}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

    return X, Y, D, P

X, Y, D, P = [], [], [], [] 

for split in [0]:


    #json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_stepsize_4/lincls_hrmin_20_hrmax_120_CorNET_dataset_apple100_split0_eps60_bs128_config.json"
    #json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm/lincls_CorNET_dataset_apple100_split{split}_eps60_bs128_config.json"
    #json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_CorNET_dataset_apple100_split{split}_eps60_bs512_config.json"
    #json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_CorNET_dataset_max_v2_split{split}_eps60_bs512_config.json"
    #json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_AttentionCorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_AttentionCorNET_dataset_apple100_split{split}_eps60_bs512_config.json"
    #json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm/lincls_hrmin_30_hrmax_120_CorNET_dataset_max_split{split}_eps60_bs512_config.json"
    #json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm/lincls_hrmin_30_hrmax_120_CorNET_dataset_max_split4_eps60_bs512_config.json"
    #json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_nnclr_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1perm_jit_aug2bioglass_dim-pdim128-128_EMA0.996_criterion_NTXent_lambda1_1.0_lambda2_1.0_tempunit_tsfm_pretrain_subsample_0.100/lincls__lr_0.000_lr_lstm_0.000CorNET_dataset_apple100_split{split}_eps60_bs128_config.json"
    #json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_windowsize_60_stepsize_15_gru/lincls_hrmin_0.0_hrmax_0.25_CorNET_dataset_max_hrv_split{split}_eps60_bs512_config.json"
    #json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_timesplit_hrmin_30_hrmax_120_CorNET_dataset_max_v2_split{split}_eps60_bs512_config.json"
    #json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_timesplit_hrmin_30_hrmax_120_CorNET_dataset_appleall_split{split}_eps60_bs512_config.json"
    #json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_reconstruction_backbone_CNN_AE_pretrain_max_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_MSE_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_lr_1.0E-05_lr_lstm_1.0E-04_hrmin_30_hrmax_120_CNN_AE_dataset_max_v2_split4_eps60_bs512_config.json"
    json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_AttentionCorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_AttentionCorNET_dataset_appleall_split{split}_eps60_bs512_config.json"

    #_X, _Y, _D, _P = predict_from_json(json_file, mode="finetune", split_partition = "val")
    #X.append(_X)
    #Y.append(_Y)
    #D.append(_D)
    #P.append(_P)

_X, _Y, _D, _P = predict_from_json(json_file, mode="finetune", split_partition = "test")
X.append(_X)
Y.append(_Y)
D.append(_D)
P.append(_P)
Y = np.concatenate(Y)
X = np.concatenate(X)
D = np.concatenate(D)
P = np.concatenate(P)

#%%
split = 1
mode = "finetuning"
split_partition = "test"
extra_args = {}
#json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_reconstruction_backbone_CNN_AE_pretrain_max_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_MSE_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_lr_1.0E-05_lr_lstm_1.0E-04_hrmin_30_hrmax_120_CNN_AE_dataset_max_v2_split{split}_eps60_bs512_config.json"
json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_CorNET_dataset_appleall_split{split}_eps60_bs512_config.json"
json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_nnclr_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1perm_jit_aug2bioglass_dim-pdim128-128_EMA0.996_criterion_NTXent_lambda1_1.0_lambda2_1.0_tempunit_tsfm_pretrain_subsample_0.100/lincls__lr_0.000_lr_lstm_0.000CorNET_dataset_apple100_split{split}_eps60_bs128_config.json"
with open(json_file) as json_file:
        json_args = json.load(json_file)

if "rnn_type" not in json_args.keys():
    json_args["rnn_type"] = "lstm"

# %%

parser = get_parser()
args = parser.parse_args([])
args_dict = vars(args)
args_dict.update(json_args)
args_dict.update(extra_args)
args = Namespace(**args_dict)


if not hasattr(args, "model_name"):
    split = 0
    args.model_name = args.backbone + '_'+args.dataset + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) + '_split' + str(split)

if mode == "finetuning":
    model_weights_path = args.lincl_model_file
elif mode == "pretraining":
    
    model_weights_path = args.pretrain_model_file
else:
    raise ValueError(f"Mode {mode} not supported")

args.cuda = 1
args.data_thr_hr = 0
args.data_thr_max = 100
args.data_thr_angle = 100
args.data_thr_avg = 100
args.dataset = "appleall"


# Testing
#######################
DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE, 'dataset:', args.dataset)
#%%
# Load data
train_loader, val_loader, test_loader = setup_dataloaders(args, mode=mode)
if split_partition == "train":
    data_loader = train_loader
elif split_partition == "val":
    data_loader = val_loader
else:
    data_loader = test_loader
#%%

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
    model_test = CNN_AE(n_channels=args.n_feature, 
                        n_classes=args.n_class, 
                        embdedded_size=128, 
                        input_size=args.input_length, 
                        backbone=True, 
                        n_channels_out=args.n_channels_out,
                        kernel_size=args.kernel_size,
                        conv_kernels=args.num_kernels,)
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

if mode == "finetuning":
    classifier = setup_linclf(args, DEVICE, model_test.out_dim)
    model_test.set_classification_head(classifier)
    model_weights = load_best_lincls(args, device=DEVICE)


elif mode == "pretraining":
    model_weights = model_weights_dict["model_state_dict"]

model_test.load_state_dict(model_weights, strict=False)
model_test = model_test.to(DEVICE)
#%%

X, Y, D, P = [], [], [], []
model_test.eval()
for i, (x, y, d) in enumerate(data_loader):

    x = x.to(DEVICE).float()
    y = y.to(DEVICE)
    d = d.to(DEVICE)
    p,_ = model_test(x)
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    d = d.detach().cpu().numpy()
    p = p.detach().cpu().numpy()
    X.append(x)
    Y.append(y)
    D.append(d)
    P.append(p)
    
X = np.concatenate(X)
Y = np.concatenate(Y)
D = np.concatenate(D)
P = np.concatenate(P).squeeze()

P = P*(args.hr_max - args.hr_min) + args.hr_min
Y = Y*(args.hr_max - args.hr_min) + args.hr_min

corr = np.corrcoef(Y, P)[0,1]
mae = np.mean(np.abs(Y-P))
rmse = np.sqrt(np.mean((Y-P)**2))
print(f"Correlation: {corr}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

#%%
df_bioglass = pd.read_pickle(os.path.join(config.classical_results_dir, "predictions_Bioglass_test_Apple100_split0.pkl"))
df_ssa = pd.read_pickle(os.path.join(config.classical_results_dir, "predictions_SSA_test_Apple100_split0.pkl"))
metadata = pd.read_csv("/local/home/lhauptmann/thesis/data/WristBCG/metadata.csv")
metadata.dropna(subset=["start"], inplace=True)
metadata.fillna({"separate matress":"y"}, inplace=True)
#%%


df_res = pandas.DataFrame({"Y": Y, 
                        "P": P, 
                        "sub": [el[0] for el in D],
                            "time": [el[1] for el in D], 
                        "X": [el for el in X]})
df_res.sort_values("time", inplace=True)
df_res["ae"] = np.abs(df_res["Y"] - df_res["P"])
df_res.set_index("time", inplace=True)
df_res["split"] = split

corr = np.corrcoef(df_res["Y"], df_res["P"])[0,1]
mae = np.mean(np.abs(df_res["Y"] - df_res["P"]))
rmse = np.sqrt(np.mean((df_res["Y"] - df_res["P"])**2))
print(f"Correlation: {corr}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

#%%

#df_res["sub"] = df_res["sub"].map(subjects_mapping)

subjects = df_res["sub"].unique()

df_grouped = pd.concat([df_res.groupby("sub").apply(lambda x: x['Y'].corr(x['P'])), df_res.groupby("sub").apply(lambda x: np.mean(np.abs(x['Y']-x['P'])))], axis=1)
df_grouped.columns = ["corr", "mae"]
df_grouped
#df_meta = pd.merge(df_grouped, metadata, left_index=True, right_on="pid")
#sns.boxplot(data=df_meta, x="age", y= "corr")
#sns.boxplot(data=df_meta[df_meta["alone"] == "n"], x="separate matress", y= "corr")
#plt.title("Subjects with partner")
#sns.scatterplot(data=df_meta, x="weight", y= "corr")
# --> We see that the performance is different for different subjects in the test set.
# --> We can also see that while the correlation is quite low, the MAE is quite high for each subject
# --> We conclude that the dataset does not have enough variety in HR
#%%
# Look at average HR and HR predictions

fig, axes = plt.subplots(2,len(subjects), figsize=(len(subjects)*5,8), dpi=500, sharex=True)

for i, sub in enumerate(subjects):

    df_sub = df_res[(df_res["sub"] == sub) & (df_res["split"] == split)]
    
    df_sub["Y"].hist(ax=axes[0, i], label="Y", bins=100)
    axes[0, i].set_title(f"Subject {sub} HR distribution")
    df_sub["P"].hist(ax=axes[1, i], label="P", bins=100)
    axes[1, i].set_title(f"Subject {sub} HR prediction distribution")
    fig.tight_layout()

# %%
# Look at HR and HR predictions over time
fig, axes = plt.subplots(len(subjects), 1, figsize=(8,len(subjects)*4), sharex=False, sharey=True)
for i, sub in enumerate(subjects):
    
    df_sub = df_res[(df_res["sub"] == sub )& (df_res["split"] == split)]
    
    df_sub["Y"].rolling(10).median().plot(ax=axes[i], label="Ground Truth", marker="", alpha=1.0, linewidth=2, color='green')
    df_sub["P"].rolling(10).median().plot(ax=axes[i], label="Supervised Regression", marker="", alpha=1.0, color="#A1A9AD")
    #df_bioglass[df_res["sub"] == sub]["hr_pred"].rolling(10).median().plot(ax=axes[i%2,i//2], label="Bioglass", marker="", alpha=0.5, color="#5BC5DB")
    #df_ssa[df_res["sub"] == sub]["hr_pred"].rolling(10).median().plot(ax=axes[i%2,i//2], label="SSA", marker="", alpha=0.5, color="#A12864")
    axes[i].set_title(f"Subject {sub}")
    axes[i].set_xlabel("Time [s]")
    axes[i].set_ylabel("HR [bpm]")

axes[0].legend()

    #print(df_sub[["Y", "P"]].corr().iloc[0,1])

fig.tight_layout()

#%%
# Look at best and worst predictions
for i, sub in enumerate(subjects):
    df_sub = df_res[df_res["sub"] == sub]
    df_sub["diff"] = np.abs(df_sub["Y"] - df_sub["P"])
    df_sub = df_sub.sort_values("diff", ascending=True)
    fig, axes = plt.subplots(2,3, figsize=(15,5))
    for i_best in range(3):
        axes[0,i_best].plot(df_sub["X"].iloc[i_best], alpha=0.5)
        axes[0,i_best].set_title(f"HR: {df_sub['Y'].iloc[i_best]:.1f}, Pred: {df_sub['P'].iloc[i_best]:.1f}")

    for i_worst in range(3):
        axes[1,i_worst].plot(df_sub["X"].iloc[-i_worst -1], alpha=0.5)
        axes[1,i_worst].set_title(f"HR: {df_sub['Y'].iloc[-i_worst -1]:.1f}, Pred: {df_sub['P'].iloc[-i_worst -1]:.1f}")
    fig.suptitle(f"Subject {sub} best and worst predictions")
    fig.tight_layout()


#%%
def plot_true_pred(hr_true, hr_pred, signal_std=[], signal_threshold=0.05, ax=None, title='', hr_lims = [40,120], **kwargs):

    """
    Plot the true vs. predicted heart rate values with optional data filtering and statistics.

    Parameters:
        hr_pred (array-like): Predicted heart rate values.
        hr_true (array-like): True heart rate values.
        signal_std (array-like): Signal standard deviation values for data filtering.
        signal_threshold (float, optional): Threshold for signal filtering. Default is 0.05.
        ax (matplotlib Axes, optional): The Axes to draw the plot on. If not provided, a new figure is created.
        title (str, optional): Title for the plot.

    This function creates a scatter plot of true vs. predicted heart rate values, with points
    categorized based on signal standard deviation. It calculates and displays the Mean Absolute
    Error (MAE) and the correlation coefficient for the entire dataset and the low-signal subset.

    Returns:
        None

    Example:
    >>> plot_true_pred(hr_pred, hr_true, signal_std, signal_threshold=0.1, title='Heart Rate Prediction')
    """

    if ax == None:
        fig, ax = plt.subplots()

    split_by_std = len(signal_std) != 0

    if split_by_std:
        thr_h = signal_std > signal_std.max()*signal_threshold
        num_low = (thr_h).sum()/signal_std.count() * 100
    else:
        thr_h = np.array([False] * hr_pred, dtype='bool')
        num_low = len(hr_pred)

    
    hr_true_h = hr_true[thr_h]
    hr_pred_h = hr_pred[thr_h]
    hr_true_l = hr_true[~thr_h]
    hr_pred_l = hr_pred[~thr_h]


    h_args = {'x': hr_true_l, 'y': hr_pred_l, 'alpha': 0.2, 'label': f'std > {signal_threshold} std_max ({np.round(num_low, 2)}%))'}
    l_args = {'x': hr_true_h, 'y': hr_pred_h, 'alpha': 0.2, 'label': f'std < {signal_threshold} std_max ({np.round(100-num_low, 2)}%))'}
    # This allows us to pass some extra plot arguments into the function. Passed arguments overwrite the default ones
    h_args.update(kwargs)
    l_args.update(kwargs)

    ax.scatter(**h_args)
    ax.scatter(**l_args)

    ax.plot(hr_lims, hr_lims, color='k', linestyle='-', linewidth=2)
    ax.set_xlabel('True HR (bpm)')
    ax.set_ylabel('Predicted HR (bpm)')
    ax.set_title(title)
    #ax.set_ylim([25, 110])
    #ax.set_xlim([35, 85])
    if split_by_std:
        ax.legend(loc='upper right')

    
    mae = np.round(np.abs(hr_true - hr_pred).mean(), 2)
    mae_l = np.round(np.abs(hr_true_l - hr_pred_l).mean(), 2)
    correlation_coefficient = np.round(np.corrcoef(hr_true, hr_pred)[0, 1],3)
    correlation_coefficient_l = np.round(np.corrcoef(hr_true_l, hr_pred_l)[0, 1],3)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    if split_by_std:
        textstr = f'MAE = {mae} \ncorr = {correlation_coefficient} \nMAE_low = {mae_l} \ncorr_low = {correlation_coefficient_l}'
    else:
        textstr = f'MAE = {mae} \ncorr = {correlation_coefficient}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)

    return ax

ax = plot_true_pred(Y*1000, P*1000, hr_lims=[0,250])
ax.set_xlabel("True HRV [ms]")
ax.set_ylabel("Predicted HRV [ms]")
# %%

data_dir = config.data_dir_Apple_processed_all
X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = classical_utils.load_dataset(data_dir, 0)
# %%


X, Y, D, P, M = [], [], [], [], []
# metrices are metric_avg, metric_max, metric_angle, metric_hr
for i, (x, y, d, metric) in enumerate(zip(X_test, Y_test, pid_test, metrics_test)):
    print(i)
    #if not (metric[0] < 0.01 and metric[1] < 0.5 and metric[2] < 0.1 and metric[3] < 5):
    #   continue
    x = torch.Tensor(x).unsqueeze(0).to(DEVICE).float()
    # z normalize x
    x = (x - x.mean()) / x.std()
    y = torch.Tensor(np.array(y).reshape(1,-1)).to(DEVICE)
    #d = torch.Tensor(d).to(DEVICE)
    p,_ = model_test(x)
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    #d = d.detach().cpu().numpy()
    p = p.detach().cpu().numpy()
    X.append(x)
    Y.append(y)
    D.append(d.reshape(1,-1))
    P.append(p.reshape(1,-1))
    M.append(metric.reshape(1,-1))

    
X = np.concatenate(X)
Y = np.concatenate(Y).squeeze()
D = np.concatenate(D)
P = np.concatenate(P).squeeze()
M = np.concatenate(M)

P = P*(args.hr_max - args.hr_min) + args.hr_min
#Y = Y*(args.hr_max - args.hr_min) + args.hr_min
# %%
corr = np.corrcoef(Y, P)[0,1]
mae = np.mean(np.abs(Y-P))
#rmse = np.sqrt(np.mean((Y-P)**2))
print(f"Correlation: {corr}")
print(f"MAE: {mae}")
#print(f"RMSE: {rmse}")


# %%
# train a random forest to predict prediction error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

error = np.abs(Y_test-P)
m_train, m_test, err_train, err_test = train_test_split(M.reshape(-1,4), error, test_size=0.2, random_state=42)
# %%
m_train[np.isnan(m_train)] = 0
estimator = RandomForestRegressor(n_estimators=100, random_state=42)
estimator.fit(m_train, err_train)
# %%
err_pred = estimator.predict(m_test)
mae = np.mean(np.abs(err_pred-err_test))
plt.scatter(err_test, err_pred)
# %%
# find parameter importance
importances = estimator.feature_importances_
indices = np.argsort(importances)[::-1]
# metrices are metric_avg, metric_max, metric_angle, metric_hr
# %%
# Print the feature ranking
print("Feature ranking:")
metric_names = ["metric_avg", "metric_max", "metric_angle", "metric_hr"]
for f in range(m_train.shape[1]):
    print(f"{f+1}. feature {metric_names[indices[f]]} ({importances[indices[f]]})")

# %%
fig, axes = plt.subplots(2,2, figsize=(10,10), sharex=True, sharey=True)
for i in range(4):
    M = M.reshape(-1,4)
    sort_ind = np.argsort(M[:,i])
    maes = []
    rs = list(np.linspace(0,1,11))
    for r in rs:
        print(r)
        mae = np.mean(np.abs(Y_test[sort_ind[:int(r*len(Y))]]-P[sort_ind[:int(r*len(Y))]]))
        maes.append(mae)
    axes[i//2, i%2].plot(rs, maes)
    axes[i//2, i%2].set_title(f"{metric_names[i]}")
    axes[i//2, i%2].set_xlabel("ratio")
    axes[i//2, i%2].set_ylabel("mae")
# %%

# %%

#plot some traces
for i in [1,50,400,800,730, 920, 320, 220]:
    plt.plot(X_test[i,:,0], label="acc_x")
    plt.plot(X_test[i,:,1], label="acc_y")
    plt.plot(X_test[i,:,2], label="acc_z")
    plt.legend()
    plt.xlabel("Time [ms]")
    plt.ylabel("Acceleration")
    plt.savefig(f"/local/home/lhauptmann/thesis/images/midterm/signal_raw{i}.png")
    plt.clf()
# %%
from scipy.signal import convolve
P_smooth = convolve(P, np.ones(5)/5, mode="same")
timestamps = D.reshape(-1,2)[:,1]
plt.plot(timestamps[100:1000], Y[100:1000], label="True")
plt.plot(timestamps[100:1000], P[100:1000], label="Predicted")
plt.xlabel("Time [s]")
plt.ylabel("HR [bpm]")
plt.legend()


#%%
res = []
i = 6847

for freq in np.arange(0,200,1):
    freq = freq / 60
    x_sample = df_res.loc[i]["X"]
    x_sample = x_sample + 1 * np.sin(2 * np.pi * freq * np.arange(0, len(x_sample))).repeat(3).reshape(-1,3)
    x_sample = torch.Tensor(x_sample).unsqueeze(0).to(DEVICE).float()
    y_sample = df_res.loc[i]["Y"]
    hr_pred = (model_test(x_sample)[0] * (args.hr_max - args.hr_min)) + args.hr_min
    #print(f"True HR: {y_sample}, Predicted HR: {hr_pred.item()}")
    res.append((freq*60, hr_pred.item()))
# %%
res = pd.DataFrame(res, columns=["freq", "hr_pred"])
# %%
from scipy.signal import periodogram
frequencies, per = periodogram(res["hr_pred"], fs=100)
# %%
plt.plot(frequencies, per)
# %%
from captum.attr import IntegratedGradients, InputXGradient, Saliency, DeepLift, GuidedBackprop, GuidedGradCam, Deconvolution, Occlusion, Lime, ShapleyValueSampling, FeatureAblation, FeaturePermutation
import seaborn as sns
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)[0]
    
torch.manual_seed(123)
np.random.seed(123)

inp_X = torch.Tensor(X[7000:7001,...]).to(DEVICE).float()
hr_true = Y[7000]

model_test.eval()
pred,_ = model_test(inp_X)
pred = pred * (args.hr_max - args.hr_min) + args.hr_min


methods = {"IntegratedGradients": IntegratedGradients, 
           "InputXGradient": InputXGradient,
            "Saliency": Saliency,
            #"DeepLift": DeepLift,
            "GuidedBackprop": GuidedBackprop,
            #"GuidedGradCam": GuidedGradCam,
            "Deconvolution": Deconvolution,
            #"Occlusion": Occlusion,
            "Lime": Lime,
            "ShapleyValueSampling": ShapleyValueSampling,
            "FeatureAblation": FeatureAblation,
            "FeaturePermutation": FeaturePermutation


           }
#%%
for method_name, method in methods.items():
    model_test.train()
    model_attr = ModelWrapper(model_test)

    ig = method(model_attr)
    #ig = InputXGradient(model_attr)
    attributions = ig.attribute(inp_X, target=0)


    print(f"True HR: {hr_true:.2f}, Predicted HR: {pred.item():.2f}")



    signal = inp_X[0,:,2].detach().cpu().numpy()
    signal_attr = attributions[0,:,2].detach().cpu().numpy()
    plt.figure()
    plt.plot(range(len(signal)), signal, color='blue', alpha=0.2)  # Plot the line in a neutral color with low alpha

    # Overlaying a scatter plot with color-encoded attribution
    
    plt.scatter(range(len(signal)), signal, c=signal_attr, cmap='viridis', s=10)  # Scatter plot with color-encoded attribution
    plt.title(f"{method_name}: True HR: {hr_true:.2f}, Predicted HR: {pred.item():.2f}")
    plt.colorbar(label='Attribution')
# %%
from scipy.signal import butter, sosfiltfilt
res = []
for i in np.arange(0,len(X),10):
    print(i)
    res_i = []
    input_X = X[i,...]
    hr_true = Y[i]
    delta = 2/60

    hr_start = 5
    hr_end = 600

    hr_pred_orig, _ = model_test(torch.Tensor(input_X.copy()).unsqueeze(0).to(DEVICE).float())
    hr_pred_orig = hr_pred_orig.item() * (args.hr_max - args.hr_min) + args.hr_min

    for c_hr in np.arange(hr_start,hr_end, 2):

        c_freq = c_hr / 60
        # apply bandstop filter for different frequencies
        bandpass_filter = butter(6, [c_freq - delta, c_freq + delta], btype="bandstop", fs=100, output="sos")
        input_X_bandpass = sosfiltfilt(bandpass_filter, input_X, axis=0)
        input_X_bandpass = (input_X_bandpass - input_X_bandpass.mean(axis=0)) / input_X_bandpass.std(axis=0)
        #hr_target = 20
        #synth_signal = np.sin(2 * np.pi * hr_target/60 * np.arange(0, 10, 1/100)) + np.random.normal(0, 0.1, 1000)
        #synth_signal = np.stack([synth_signal, synth_signal, synth_signal], axis=1)
        out, feat = model_test(torch.Tensor(input_X_bandpass.copy()).unsqueeze(0).to(DEVICE).float())
        hr_pred = out.item() * (args.hr_max - args.hr_min) + args.hr_min
        res_i.append((c_hr, hr_pred, hr_pred_orig, hr_true))
    res.append(res_i)

res = np.array(res)

#%%

df_res = pandas.DataFrame(res.reshape(-1, res.shape[-1]), columns=["hr_target", "hr_pred", "hr_pred_orig", "hr_true"])
df_res["dist_from_true"] = df_res["hr_target"] - df_res["hr_true"]
df_res["deviation"] = np.abs(df_res["hr_pred"] - df_res["hr_pred_orig"])

bins = pd.cut(df_res["dist_from_true"], bins=300)
# take center of bins
bins = bins.apply(lambda x: np.round(x.mid,3))
#plot median of deviation per bin
df_res["bin"] = bins
#df_res = df_res[df_res["bin"] ]

df_res_group = df_res.groupby("bin").agg({"deviation": ["median", lambda x: np.quantile(x, 0.25), lambda x: np.quantile(x, 0.75)]})

plt.plot(df_res_group.index, df_res_group["deviation"]["median"])
plt.fill_between(df_res_group.index, df_res_group["deviation"].iloc[:,1],  df_res_group["deviation"].iloc[:,2], alpha=0.3)
plt.ylim(0,2)

# %%
plt.figure(figsize=(10,10))
deviation = np.abs(res[...,1] - res[...,3])
mean = np.median(deviation, axis=0)
x = np.mean(res[...,0], axis=0)/60 # they should all be the same
std = np.std(deviation, axis=0)
quant_high = np.quantile(deviation, 0.75, axis=0)
quant_low = np.quantile(deviation, 0.25, axis=0)
plt.plot(x, mean, label="Median")
plt.fill_between(x, quant_low, quant_high, alpha=0.3, label='25 and 75 Quantile')
#plt.axhline(0,color='red', alpha=0.2)
plt.xlim(0,10)
plt.locator_params(axis='x', nbins=20)
plt.legend()
plt.xlabel("Frequency [Hz]")
plt.ylabel("Absolute Error [bpm]")

#%%
all_res = {}

#%%

from scipy.signal import butter, sosfiltfilt
res_swap = []
for i in np.arange(0,len(X),1):
    input_X = X[i,...]
    hr_true = Y[i]

    hr_pred_orig, _ = model_test(torch.Tensor(input_X.copy()).unsqueeze(0).to(DEVICE).float())
    hr_pred_orig = hr_pred_orig.item() * (args.hr_max - args.hr_min) + args.hr_min
    res_swap.append([-1, hr_pred_orig, hr_pred_orig, hr_true])

    for axis in [0,1,2]:
        
        s_axis1 = axis
        s_axis2 = (axis+1) % 3
        input_X_axis = input_X.copy()
        input_X_axis[...,s_axis1] = input_X[...,s_axis2]
        input_X_axis[...,s_axis2] = input_X[...,s_axis1]
        out, feat = model_test(torch.Tensor(input_X_axis).unsqueeze(0).to(DEVICE).float())
        hr_pred = out.item() * (args.hr_max - args.hr_min) + args.hr_min
        res_swap.append([axis, hr_pred, hr_pred_orig, hr_true])


res_swap = np.array(res_swap)

df_res_swap = pandas.DataFrame(res_swap, columns=["axis", "hr_pred", "hr_pred_orig", "hr_true"])
df_res_swap["axis"] = df_res_swap["axis"].map({0: "x <-> y", 1: "y <-> z", 2: "z <-> x", -1: "original"})
df_res_swap["deviation"] = np.abs(df_res_swap["hr_pred"] - df_res_swap["hr_pred_orig"])
df_res_swap["error"] = np.abs(df_res_swap["hr_pred"] - df_res_swap["hr_true"])
df_res_swap.groupby("axis").agg({"deviation": ["mean", "std"], "error": ["mean", "std"]})

# %%
res_occ = []
for i in np.arange(0,len(X),1):
    input_X = X[i,...]
    hr_true = Y[i]

    hr_pred_orig, _ = model_test(torch.Tensor(input_X.copy()).unsqueeze(0).to(DEVICE).float())
    hr_pred_orig = hr_pred_orig.item() * (args.hr_max - args.hr_min) + args.hr_min
    res_occ.append([-1, hr_pred_orig, hr_pred_orig, hr_true])

    for axis in [0,1,2]:
        
        input_X_axis = input_X.copy()
        input_X_axis[...,axis] = 0

        out, feat = model_test(torch.Tensor(input_X_axis).unsqueeze(0).to(DEVICE).float())
        hr_pred = out.item() * (args.hr_max - args.hr_min) + args.hr_min
        res_occ.append([axis, hr_pred, hr_pred_orig, hr_true])


res_occ = np.array(res_occ)

df_res_occ = pandas.DataFrame(res_occ, columns=["axis", "hr_pred", "hr_pred_orig", "hr_true"])
df_res_occ["axis"] = df_res_occ["axis"].map({0: "x", 1: "y", 2: "z", -1: "original"})
df_res_occ["deviation"] = np.abs(df_res_occ["hr_pred"] - df_res_occ["hr_pred_orig"])
df_res_occ["error"] = np.abs(df_res_occ["hr_pred"] - df_res_occ["hr_true"])
df_res_occ.groupby("axis").agg({"deviation": ["mean", "std"], "error": ["mean", "std"]})




# %%
all_res["CorNET"] = (df_res_occ.copy(), df_res_swap.copy())

# %%


# %%

occlusion = pd.concat([all_res["AttentionCorNET"][0], all_res["CorNET"][0]], axis=0, keys=["AttentionCorNET", "CorNET"]).reset_index(level=0)
occlusion = occlusion.rename(columns={"level_0": "model"})
occlusion.boxplot(column="error", by=["axis","model",], figsize=(10,10), rot=45)

#%%
swap = pd.concat([all_res["AttentionCorNET"][1], all_res["CorNET"][1]], axis=0, keys=["AttentionCorNET", "CorNET"]).reset_index(level=0)
swap = swap.rename(columns={"level_0": "model"})
swap.boxplot(column="error", by=["axis","model",], figsize=(10,10), rot=45)
# %%
sns.boxplot(data=swap, x="axis", y="error", hue="model", showfliers=False, showmeans=True)
# %%
sns.boxplot(data=occlusion, x="axis", y="deviation", hue="model",  showfliers=False, showmeans=True)
# %%

swap.to_pickle("/local/home/lhauptmann/thesis/CL-HAR/results/swap.pkl")
occlusion.to_pickle("/local/home/lhauptmann/thesis/CL-HAR/results/occlusion.pkl")
# %%


