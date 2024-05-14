#%%
import os
import json
import pandas
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import subprocess
import re
import pickle

from models.backbones import *
from models.loss import *
from trainer import *
from argparse import Namespace
from main import get_parser


image_save_dir = config.plot_dir
#%%

def get_gpu_info():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.free', '--format=csv,noheader,nounits'])
        gpu_info = output.decode('utf-8')
        return gpu_info.split('\n')[:-1]  # Remove the last empty element
    except subprocess.CalledProcessError:
        return None

def get_free_gpu():
    gpu_info = get_gpu_info()
    if gpu_info:
        gpu_data = [info.split(', ') for info in gpu_info]
        gpu_data = sorted(gpu_data, key=lambda x: int(x[2]), reverse=True)  # Sort by free memory
        return gpu_data[0][0]  # Return index of the GPU with the most free memory
    else:
        return None
    
def predict(model_test, data_loader, args, DEVICE):


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


def load_model(json_file, mode="finetuning", extra_args={}, split_partition="test", data_loader=None):
    
    with open(json_file) as json_file:
        json_args = json.load(json_file)

    parser = get_parser()
    args = parser.parse_args([])
    args_dict = vars(args)
    args_dict.update(json_args)
    args_dict.update(extra_args)
    args = Namespace(**args_dict)
    args.cuda = int(get_free_gpu())


    if not hasattr(args, "model_name"):
        split = 0
        args.model_name = args.backbone + '_'+args.dataset + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) + '_split' + str(split)

    if mode == "finetuning":
        model_weights_path = args.lincl_model_file
    elif mode == "pretraining":
        
        model_weights_path = args.pretrain_model_file
    else:
        raise ValueError(f"Mode {mode} not supported")
    if "rnn_type" not in json_args.keys():
        json_args["rnn_type"] = "lstm"



    # Testing
    #######################
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE, 'dataset:', args.dataset)

    # Load data
    if data_loader != None \
        and data_loader.dataset.dataset == args.dataset \
        and data_loader.dataset.partition == split_partition \
        and (data_loader.dataset.split == args.split \
             or split_partition == "test"):
        #skip dataset loading if passed to function
        print("Using passed dataloader")
    else:
        train_loader, val_loader, test_loader = setup_dataloaders(args, mode=mode)
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

    model_weights_dict = torch.load(model_weights_path)

    if mode == "finetuning":
        classifier = setup_linclf(args, None,  model_test.out_dim)
        model_test.set_classification_head(classifier)
        model_weights = load_best_lincls(args)


    elif mode == "pretraining":
        model_weights = model_weights_dict["model_state_dict"]

    model_test.load_state_dict(model_weights, strict=False)

    return model_test, args, DEVICE, data_loader


def run_frequency_occlusion(model_test, data_loader, args, DEVICE, fs=100, delta=2, take_every_n=10, hr_start=3, hr_end=480, hr_step = 2):

    from scipy.signal import butter, sosfiltfilt
    res = []
    delta = delta / 60
    for i, (input_X, hr_true, _) in enumerate(data_loader):

        print(f"Processing {i}/{len(data_loader)}")
        input_X, hr_true = input_X[::take_every_n,...], hr_true[::take_every_n,...]
        input_X, hr_true = input_X.reshape(-1, input_X.shape[-2], input_X.shape[-1]), hr_true.reshape(-1)
        res_i = []
        if len(input_X) == 0:
            continue
        hr_true = hr_true.numpy() * (args.hr_max - args.hr_min) + args.hr_min
        hr_pred_orig, _ = model_test(input_X.to(DEVICE).float())
        hr_pred_orig = hr_pred_orig* (args.hr_max - args.hr_min) + args.hr_min
        hr_pred_orig = hr_pred_orig.cpu().detach().numpy().squeeze(axis=1)
        for c_hr in np.arange(hr_start,hr_end, hr_step):

            c_freq = c_hr / 60
            # apply bandstop filter for different frequencies
            bandpass_filter = butter(4, [c_freq - delta/2, c_freq + delta/2], btype="bandstop", fs=fs, output="sos")
            input_X_bandpass = sosfiltfilt(bandpass_filter, input_X.numpy(), axis=1)
            input_X_bandpass = (input_X_bandpass - input_X_bandpass.mean(axis=1, keepdims=True)) / input_X_bandpass.std(axis=1, keepdims=True)
            out, feat = model_test(torch.Tensor(input_X_bandpass).to(DEVICE).float())
            hr_pred = out * (args.hr_max - args.hr_min) + args.hr_min
            hr_pred = hr_pred.cpu().detach().numpy().squeeze(axis=1)
            res_i.append(np.array((np.repeat(c_freq, len(hr_true)), hr_pred, hr_pred_orig, hr_true)))
        res.append(np.concatenate(res_i, axis=1))
    res = np.concatenate(res, axis=1).T
    res = pd.DataFrame(res, columns=["freq", "hr_pred", "hr_pred_orig", "hr_true"])
    return res



def plot_frequency_occlusion(df_res, mode='deviation', ax=None, quantiles=.25, measure="mean", legend="", find_peaks=False):
    assert mode in ['deviation', 'dist_from_true']
    if ax is None:
        fig, ax = plt.subplots()
    mode_names = {"deviation": "Deviation from predicted HR", "dist_from_true": "Deviation from True HR"}
    dataset_names = {"appleall": "Apple", "max_v2": "Max"}
    #df_res = pandas.DataFrame(res.reshape(-1, res.shape[-1]), columns=["hr_target", "hr_pred", "hr_pred_orig", "hr_true"])
    df_res["dist_from_true"] = (df_res["hr_pred"] - df_res["hr_true"]).abs()
    df_res["deviation"] = (df_res["hr_pred"] - df_res["hr_pred_orig"]).abs()
    
    legend = f"{mode_names[mode]}" if legend == "" else legend
    if measure == "mean":
        df_res_grouped = df_res.groupby("freq")[mode].agg([(measure,"mean"),("std", "std")])
    else:
        df_res_grouped = df_res.groupby("freq")[mode].agg([(measure,"median"),("quant25",lambda x : np.quantile(x,quantiles)),("quant75",lambda x: np.quantile(x,1-quantiles))])
    if measure == "mean":
        ax.plot(df_res_grouped[measure], label=legend)
        ax.fill_between(df_res_grouped.index, df_res_grouped[measure] - df_res_grouped["std"]*0.1, df_res_grouped[measure] + df_res_grouped["std"]*0.1, alpha=0.3)
    else:
        ax.plot(df_res_grouped[measure], label=legend)
        ax.fill_between(df_res_grouped.index, df_res_grouped["quant25"], df_res_grouped["quant75"], alpha=0.3)
    
    if find_peaks:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(df_res_grouped[measure], distance=5, prominence=0.01)
        ax.plot(df_res_grouped.index[peaks], df_res_grouped[measure].iloc[peaks], "x")
        print(f"Peaks: {df_res_grouped.index[peaks]}")
    
    ax.locator_params(axis='x', nbins=20)
    ax.legend()
    ax.set_xlabel("Frequency [Hz]")
    if mode == "dist_from_true":
        ax.set_ylabel("Absolute Error [bpm]")
    else:
        ax.set_ylabel("Absolute Deviation [bpm]")

def swap_axes(model_test, data_loader, args, DEVICE):
    from scipy.signal import butter, sosfiltfilt
    res_swap = []
    for input_X, hr_true, pid in data_loader:

        hr_pred_orig, _ = model_test(input_X.to(DEVICE).float())
        hr_pred_orig = hr_pred_orig.cpu().detach().numpy().squeeze()
        hr_pred_orig = hr_pred_orig * (args.hr_max - args.hr_min) + args.hr_min
        hr_true = hr_true.numpy() * (args.hr_max - args.hr_min) + args.hr_min
        res_swap.append(np.array([np.repeat(-1,len(hr_pred_orig)), hr_pred_orig, hr_pred_orig, hr_true]))

        for axis in [0,1,2]:
            
            s_axis1 = axis
            s_axis2 = (axis+1) % 3
            input_X_axis = torch.clone(input_X)
            input_X_axis[...,s_axis1] = input_X[...,s_axis2]
            input_X_axis[...,s_axis2] = input_X[...,s_axis1]
            out, _ = model_test(input_X_axis.to(DEVICE).float())
            out = out.cpu().detach().numpy().squeeze()
            del input_X_axis
            hr_pred = out * (args.hr_max - args.hr_min) + args.hr_min
            res_swap.append(np.array([np.repeat(axis,len(hr_pred)), hr_pred, hr_pred_orig, hr_true]))
    return res_swap

def analyze_swap_axes(res_swap):
    res_swap = np.concatenate(res_swap, axis=1).T
    df_res_swap = pandas.DataFrame(res_swap, columns=["axis", "hr_pred", "hr_pred_orig", "hr_true"])
    df_res_swap["axis"] = df_res_swap["axis"].map({0: "x <-> y", 1: "y <-> z", 2: "z <-> x", -1: "original"})
    df_res_swap["deviation"] = np.abs(df_res_swap["hr_pred"] - df_res_swap["hr_pred_orig"])
    df_res_swap["error"] = np.abs(df_res_swap["hr_pred"] - df_res_swap["hr_true"])
    df_res_swap = df_res_swap.groupby("axis").agg({"deviation": ["mean", "std"], "error": ["mean", "std"]})
    return df_res_swap

# %%
# frequency occlusion
split = 0
json_files_dict = {
    "appleall": f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_CorNET_dataset_appleall_split{split}_eps60_bs512_config.json",
    "max_v2": f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_CorNET_dataset_max_v2_split{split}_eps60_bs512_config.json",

    #"appleall": f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_AttentionCorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_AttentionCorNET_dataset_appleall_split{split}_eps60_bs512_config.json",
    #"max_v2": f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_AttentionCorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_AttentionCorNET_dataset_max_v2_split{split}_eps60_bs512_config.json",
}

res_dict = {}
for dataset_name, json_file in json_files_dict.items():
    model_test, args, DEVICE, data_loader = load_model(json_file, mode="finetuning", split_partition = "test")
    model_test = model_test.to(DEVICE)
    res = run_frequency_occlusion(model_test, data_loader, args, DEVICE, fs=100, delta=16, take_every_n=1, hr_start=9, hr_end=600, hr_step = 4)
    res_dict[dataset_name] = res


#%%
with open("expl_freq_occlusion.pickle", "wb") as f:
    pickle.dump(res_dict, f)


#%%

with open("expl_freq_occlusion.pickle", "rb") as f:
    res_dict = pickle.load(f)


#%%
# plotting 
plot_frequency_occlusion(res_dict["appleall"], mode="deviation", quantiles=.40, measure="median", legend="Apple Watch", find_peaks=False)
plt.savefig(os.path.join(image_save_dir, "frequency_occlusion_att_apple.pdf"))
plot_frequency_occlusion(res_dict["max_v2"], mode="deviation", quantiles=.40, measure="median", legend="In-House", find_peaks=False)
plt.savefig(os.path.join(image_save_dir, "frequency_occlusion_att_max.pdf"))
#%%
res_dict_swap = {}
split = 0
json_files_dict = {
    "appleall": f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_CorNET_dataset_appleall_split{split}_eps60_bs512_config.json",
    "max_v2": f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_CorNET_dataset_max_v2_split{split}_eps60_bs512_config.json",
    "appleall_att": f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_AttentionCorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_AttentionCorNET_dataset_appleall_split{split}_eps60_bs512_config.json",
    "max_v2_att": f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_AttentionCorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_AttentionCorNET_dataset_max_v2_split{split}_eps60_bs512_config.json",
}

for dataset_name, json_file in json_files_dict.items():
    res_dict_swap[dataset_name] = []
    data_loader = None
    for split in range(5):
        print(f"Processing {dataset_name} split {split}")
        json_file = re.sub(r"split(\d+)", f"split{split}", json_file)
        model_test, args, DEVICE, data_loader = load_model(json_file, mode="finetuning", split_partition = "test", data_loader = data_loader)
        model_test = model_test.to(DEVICE)
        res_occ = swap_axes(model_test, data_loader, args, DEVICE)
        del model_test
        res_dict_swap[dataset_name].extend(res_occ)
    df_res_occ = analyze_swap_axes(res_dict_swap[dataset_name])
    res_dict_swap[dataset_name] = df_res_occ

#%%
with open("expl_swap_axes.pickle", "wb") as f:
    pickle.dump(res_dict_swap, f)


# %%
def occlude_axes(model_test, data_loader, args, DEVICE):
    res_occ = []
    for input_X, hr_true, pid in data_loader:

        hr_pred_orig, _ = model_test(input_X.to(DEVICE).float())
        hr_pred_orig = hr_pred_orig * (args.hr_max - args.hr_min) + args.hr_min
        hr_pred_orig = hr_pred_orig.cpu().detach().numpy().squeeze()
        hr_true = hr_true.numpy() * (args.hr_max - args.hr_min) + args.hr_min
        res_occ.append(np.array([np.repeat(-1,len(hr_pred_orig)), hr_pred_orig, hr_pred_orig, hr_true]))

        for axis in [0,1,2]:
            
            input_X_axis = torch.clone(input_X)
            input_X_axis[...,axis] = 0

            out, feat = model_test(input_X_axis.to(DEVICE).float())
            hr_pred = out * (args.hr_max - args.hr_min) + args.hr_min
            hr_pred = hr_pred.cpu().detach().numpy().squeeze()
            res_occ.append(np.array([np.repeat(axis,len(hr_pred)), hr_pred, hr_pred_orig, hr_true]))

    
    return res_occ


def analyse_occlusion(res_occ):
    res_occ = np.concatenate(res_occ, axis=1).T
    df_res_occ = pandas.DataFrame(res_occ, columns=["axis", "hr_pred", "hr_pred_orig", "hr_true"])
    df_res_occ["axis"] = df_res_occ["axis"].map({0: "x", 1: "y", 2: "z", -1: "original"})
    df_res_occ["deviation"] = np.abs(df_res_occ["hr_pred"] - df_res_occ["hr_pred_orig"])
    df_res_occ["error"] = np.abs(df_res_occ["hr_pred"] - df_res_occ["hr_true"])
    df_res_occ = df_res_occ.groupby("axis").agg({"deviation": ["mean", "std"], "error": ["mean", "std"]})    

    return df_res_occ
#%%
res_dict_occ = {}
split = 0
json_files_dict = {
    "appleall": f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_CorNET_dataset_appleall_split{split}_eps60_bs512_config.json",
    "max_v2": f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_CorNET_dataset_max_v2_split{split}_eps60_bs512_config.json",
    "appleall_att": f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_AttentionCorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_AttentionCorNET_dataset_appleall_split{split}_eps60_bs512_config.json",
    "max_v2_att": f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_AttentionCorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_AttentionCorNET_dataset_max_v2_split{split}_eps60_bs512_config.json",
}

for dataset_name, json_file in json_files_dict.items():
    res_dict_occ[dataset_name] = []
    data_loader = None
    for split in range(5):
        json_file = re.sub(r"split(\d+)", f"split{split}", json_file)
        model_test, args, DEVICE, data_loader = load_model(json_file, mode="finetuning", split_partition = "test", data_loader = data_loader)
        model_test = model_test.to(DEVICE)
        res_occ = occlude_axes(model_test, data_loader, args, DEVICE)
        res_dict_occ[dataset_name].extend(res_occ)
    df_res_occ = analyse_occlusion(res_dict_occ[dataset_name])
    res_dict_occ[dataset_name] = df_res_occ



#%%
with open("expl_occ_axes.pickle", "wb") as f:
    pickle.dump(res_dict_occ, f)

# %%

def df_to_latex_custom(df, n_digits = 2):
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
    
    for var in df.columns.levels[0]:  # For each variable like deviation, error
        # Concatenate mean and std into a single string in the format: 'mean (std)'
        formatted_df[var] = df[(var, 'mean')].apply(lambda x: f"{x:.{n_digits}f}") + " ± " + \
                            df[(var, 'std')].apply(lambda y: f"{y:.{n_digits-1}f}")
    # Use to_latex to convert the formatted DataFrame to a LaTeX table
    latex_str = formatted_df.to_latex(index=True, header=True)
    print(latex_str)
# %%
suf = ""
dataset1 = f"appleall{suf}"
dataset2 = f"max_v2{suf}"

metric = "error"

df_concat_swap = pd.concat({"Apple Watch":res_dict_swap[dataset1], "In-House":res_dict_swap[dataset2]}, axis=1)
df_concat_swap.columns = pd.MultiIndex.from_tuples([(f"{level[0]} {level[1]}", level[2]) for level in df_concat_swap.columns])
#df_to_latex_custom(df_concat_swap)
df_concat_swap.iloc[1:].mean()[[(f"Apple Watch {metric}", "mean"), (f"In-House {metric}", "mean")]].mean()
#%%

suf = ""
dataset1 = f"appleall{suf}"
dataset2 = f"max_v2{suf}"

metric = "deviation"

df_concat_occ = pd.concat({"Apple Watch":res_dict_occ[dataset1], "In-House":res_dict_occ[dataset2]}, axis=1)
df_concat_occ.columns = pd.MultiIndex.from_tuples([(f"{level[0]} {level[1]}", level[2]) for level in df_concat_occ.columns])
#df_to_latex_custom(df_concat_swap)
df_concat_occ.iloc[1:].mean()[[(f"Apple Watch {metric}", "mean"), (f"In-House {metric}", "mean")]].mean()
# %%
df_concat_occ = pd.concat({"Apple Watch":res_dict_occ["appleall_att"], "In-House":res_dict_occ["max_v2_att"]}, axis=1)
df_concat_occ.columns = pd.MultiIndex.from_tuples([(f"{level[0]} {level[1]}", level[2]) for level in df_concat_occ.columns])
df_to_latex_custom(df_concat_occ)
# %%
