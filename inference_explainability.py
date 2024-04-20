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
import subprocess

from models.backbones import *
from models.loss import *
from trainer import *
from argparse import Namespace
import classical_utils
from main import get_parser

from captum.attr import IntegratedGradients

split_res = {}

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

def run_frequency_occlusion(model_test, X, Y, args, DEVICE, fs=100, delta=2, take_every_n=10, hr_start=3, hr_end=480):

    from scipy.signal import butter, sosfiltfilt
    res = []
    delta = delta / 60
    for i in np.arange(0,len(X),take_every_n):
        print(i)
        res_i = []
        input_X = X[i,...]
        hr_true = Y[i]

        hr_pred_orig, _ = model_test(torch.Tensor(input_X.copy()).unsqueeze(0).to(DEVICE).float())
        hr_pred_orig = hr_pred_orig.item() * (args.hr_max - args.hr_min) + args.hr_min
        
        for c_hr in np.arange(hr_start,hr_end, 2):

            c_freq = c_hr / 60
            # apply bandstop filter for different frequencies
            bandpass_filter = butter(4, [c_freq - delta, c_freq + delta], btype="bandstop", fs=fs, output="sos")
            input_X_bandpass = sosfiltfilt(bandpass_filter, input_X, axis=0)
            input_X_bandpass = (input_X_bandpass - input_X_bandpass.mean(axis=0)) / input_X_bandpass.std(axis=0)
            out, feat = model_test(torch.Tensor(input_X_bandpass.copy()).unsqueeze(0).to(DEVICE).float())
            hr_pred = out.item() * (args.hr_max - args.hr_min) + args.hr_min
            res_i.append((c_hr, hr_pred, hr_pred_orig, hr_true))
        res.append(res_i)
    res = np.array(res)
    res = pd.DataFrame(res.reshape(-1, res.shape[-1]), columns=["hr_target", "hr_pred", "hr_pred_orig", "hr_true"])
    return res

#%%

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


# %%
def load_model(json_file, mode="finetuning", extra_args={}, split_partition="test"):
    
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

    model_weights_dict = torch.load(model_weights_path, map_location = DEVICE)

    if mode == "finetuning":
        classifier = setup_linclf(args, DEVICE, model_test.out_dim)
        model_test.set_classification_head(classifier)
        model_weights = load_best_lincls(args, device=DEVICE)


    elif mode == "pretraining":
        model_weights = model_weights_dict["model_state_dict"]

    model_test.load_state_dict(model_weights, strict=False)
    model_test = model_test.to(DEVICE)

    return model_test, args, DEVICE, data_loader
#%%

split = 1
mode = "finetuning"
split_partition = "test"
extra_args = {}
#json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_reconstruction_backbone_CNN_AE_pretrain_max_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_MSE_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_lr_1.0E-05_lr_lstm_1.0E-04_hrmin_30_hrmax_120_CNN_AE_dataset_max_v2_split{split}_eps60_bs512_config.json"
json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_CorNET_dataset_appleall_split{split}_eps60_bs512_config.json"
#json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_nnclr_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1perm_jit_aug2bioglass_dim-pdim128-128_EMA0.996_criterion_NTXent_lambda1_1.0_lambda2_1.0_tempunit_tsfm_pretrain_subsample_0.100/lincls__lr_0.000_lr_lstm_0.000CorNET_dataset_apple100_split{split}_eps60_bs128_config.json"


model_test, args, DEVICE, data_loader = load_model(json_file, mode="finetuning", split_partition = "test")

X, Y, D, P = predict(model_test, data_loader, args, DEVICE)
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

# %%

data_dir = config.data_dir_Apple_processed_all
X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = classical_utils.load_dataset(data_dir, 0, load_train=False, args=vars(args))
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
# frequency occlusion
json_files_dict = {
    "appleall": f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_CorNET_dataset_appleall_split{split}_eps60_bs512_config.json",
    "max_v2": f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_hrmin_30_hrmax_120_CorNET_dataset_max_v2_split{split}_eps60_bs512_config.json",
}

res_dict = {}
for dataset_name, json_file in json_files_dict.items():
    model_test, args, DEVICE, data_loader = load_model(json_file, mode="finetuning", split_partition = "test", extra_args={"batch_size": 1})

    res = run_frequency_occlusion(model_test, X, Y, args, DEVICE, fs=100, delta=2/60, take_every_n=1, hr_start=0.05, hr_end=480)
    res_dict[dataset_name] = res
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


