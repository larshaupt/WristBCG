
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
import argparse

from models.backbones import *
from models.loss import *
from trainer import *
from argparse import Namespace
import classical_utils
from main import get_parser
#%%
Y_all, M_all, P_all = [], [], []

for split in range(5):

    #json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_stepsize_4/lincls_hrmin_20_hrmax_120_CorNET_dataset_apple100_split0_eps60_bs128_config.json"
    #json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm/lincls_CorNET_dataset_apple100_split0_eps60_bs128_config.json"
    json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm/lincls_hrmin_20_hrmax_120_CorNET_dataset_apple100_split{split}_eps60_bs512_config.json"
    with open(json_file) as json_file:
        json_args = json.load(json_file)



    parser = get_parser()
    args = parser.parse_args([])
    args_dict = vars(args)
    args_dict.update(json_args)
    args = Namespace(**args_dict)


    split = args.split

    mode = "finetune"


    if not hasattr(args, "model_name"):
        args.model_name = args.backbone + '_'+args.dataset + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) + '_split' + str(split)
    else:
        args.model_name = re.sub(r'split\d+', f"split{split}", args.model_name)

    if mode == "finetune":
        model_weights_path = args.lincl_model_file
    elif mode == "pretrain":
        model_weights_path = args.pretrain_model_file
    else:
        raise ValueError(f"Mode {mode} not supported")

    #args.batch_size = 1
    #args.cuda = 3
    #args.dataset = "IEEE"
    args.step_size = 8
    args.window_size = 10
    #args.sampling_rate = 100
    #args.subsample_ranked_train = 1.0
    args.dataset = "appleall"
    #args.hr_min = 30

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
        model_test = CorNET(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=args.num_kernels, kernel_size=args.kernel_size, LSTM_units=args.lstm_units, backbone=True, input_size=args.input_length)
    elif args.backbone == "TCN":
        model_test = TemporalConvNet(num_channels=[32, 64, 128], n_classes=args.n_class,  num_inputs=args.n_feature, input_length = args.input_length, kernel_size=16, dropout=0.2, backbone=True)
    else:
        NotImplementedError

    # Testing
    model_weights_dict = torch.load(model_weights_path)

    if mode == "finetune":
        classifier = setup_linclf(args, DEVICE, model_test.out_dim)
        model_test.set_classification_head(classifier)
        model_weights = load_best_lincls(args)


    elif mode == "pretrain":
        model_weights = model_weights_dict["model_state_dict"]

    model_test.load_state_dict(model_weights, strict=False)
    model_test = model_test.to(DEVICE)




    data_dir = config.data_dir_Apple_processed_all
    X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = classical_utils.load_dataset(data_dir, args.split)

    X, Y, D, P, M = [], [], [], [], []
    # metrices are metric_avg, metric_max, metric_angle, metric_hr
    for i, (x, y, d, metric) in enumerate(zip(X_val, Y_val, pid_val, metrics_val)):

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

    corr = np.corrcoef(Y, P)[0,1]
    mae = np.mean(np.abs(Y-P))
    #rmse = np.sqrt(np.mean((Y-P)**2))
    print(f"Correlation: {corr}")
    print(f"MAE: {mae}")
    #print(f"RMSE: {rmse}")

    Y_all.append(Y)
    M_all.append(M)
    P_all.append(P)

Y_all = np.concatenate(Y_all)
M_all = np.concatenate(M_all)
P_all = np.concatenate(P_all)



#%%
def mae_threshold(avg, max, angle, hr, details=False):
    M_mask = (M_all[:,0] < avg) & (M_all[:,1] < max) & (M_all[:,2] < angle) & (M_all[:,3] < hr)
    ratio = M_mask.sum() / len(M_mask)

    if details:
        corr = np.corrcoef(Y_mask, P_mask)[0,1]
        mae = np.mean(np.abs(Y_mask-P_mask))
        return corr, mae, ratio
    

    
    if ratio < 0.3:
        return -4, ratio
    
    Y_mask = Y_all[M_mask]
    P_mask = P_all[M_mask]


    if len(Y_mask) == 0:
        return -4, ratio

    mae = np.mean(np.abs(Y_mask-P_mask))
    return -mae, ratio

#%%
from bayes_opt import BayesianOptimization
pbounds = {'avg': (0, 5), 'max': (0, 1), "angle": (0, 10), "hr": (0, 10)}

optimizer = BayesianOptimization(
    f=mae_threshold,
    pbounds=pbounds,
    random_state=1,

)

optimizer.maximize(
init_points=2,
n_iter=300,
)
# %%
print(optimizer.max)
max_params = optimizer.max["params"]
max_params["hr"] = 10
# %%
M_mask = (M_all[:,0] < max_params["avg"]) & (M_all[:,1] < max_params["max"]) & (M_all[:,2] < max_params["angle"]) & (M_all[:,3] < max_params["hr"])
ratio = M_mask.sum() / len(M_mask)

Y_mask = Y_all[M_mask]
P_mask = P_all[M_mask]


corr_fio = np.corrcoef(Y_mask, P_mask)[0,1]
mae_fil = np.mean(np.abs(Y_mask-P_mask))
print(f"Ratio: {ratio}")
print(f"Correlation: {corr_fio}")
print(f"MAE: {mae_fil}")
# %%


avg_x = np.linspace(0.1, 5, 100)
avg_y = [mae_threshold(avg, max_params["max"], max_params["angle"], max_params["hr"]) for avg in avg_x]

avg_y = np.array(avg_y)

plt.plot(avg_x, -avg_y[:,0])
plt.axline((max_params["avg"], 3), (max_params["avg"], 4), color="red")
plt.ylim(3,4)
plt.xlabel("avg value")
plt.ylabel("MAE")

twinax = plt.gca().twinx()
twinax.set_ylabel("Ratio")
twinax.plot(avg_x, avg_y[:,1], color="green", label="Ratio")
twinax.legend(loc="upper right")
# %%

max_x = np.linspace(0, 0.3, 100)
max_y = [mae_threshold(max_params["avg"], max, max_params["angle"], max_params["hr"]) for max in max_x]
max_y = np.array(max_y)
plt.plot(max_x, -max_y[:,0], label="MAE")
plt.axline((max_params["max"], 3), (max_params["max"], 4), color="red")
plt.xlabel("max value")
plt.ylabel("MAE")
plt.ylim(3,4)
plt.title("Variation of MAE and amount of data with Max Value")
plt.legend(loc="lower right")
twinax = plt.gca().twinx()
twinax.set_ylabel("Ratio")
twinax.plot(max_x, max_y[:,1], color="green", label="Ratio")
twinax.legend(loc="upper right")
# %%

angle_x = np.linspace(0, 10, 100)
angle_y = [mae_threshold(max_params["avg"], max_params["max"], angle, max_params["hr"]) for angle in angle_x]
angle_y = np.array(angle_y)

plt.xlabel("angle thredhold")
plt.ylabel("MAE")
plt.title("Variation of MAE and amount of data with Angle Value")
plt.plot(angle_x, -angle_y[:,0], label="MAE")
plt.axline((max_params["angle"], 3), (max_params["angle"], 4), color="red")
plt.ylim(3,4)
plt.legend(loc="lower right")
twinax = plt.gca().twinx()
twinax.set_ylabel("Ratio")
twinax.plot(angle_x, angle_y[:,1], color="green", label="Ratio")
twinax.legend(loc="upper right")
# %%

hr_x = np.linspace(0, 15, 100)
hr_y = [mae_threshold(max_params["avg"], max_params["max"], max_params["angle"], hr) for hr in hr_x]

hr_y = np.array(hr_y)

plt.xlabel("hr")
plt.ylabel("MAE")

plt.plot(hr_x, -hr_y[:,0])
plt.axline((max_params["hr"], 3), (max_params["hr"], 4), color="red")
plt.ylim(3,4)

twinax = plt.gca().twinx()
twinax.set_ylabel("Ratio")
twinax.plot(hr_x, hr_y[:,1], color="green")
# %%
