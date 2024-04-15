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
from utils import tsne

from models.backbones import *
from models.loss import *
from trainer import *
from argparse import Namespace
import classical_utils
from main import get_parser

from captum.attr import IntegratedGradients


#%%
X, Y, D, P, feats, splits = [], [], [], [], [], []

for split in range(5):

    #json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_stepsize_4/lincls_hrmin_20_hrmax_120_CorNET_dataset_apple100_split0_eps60_bs128_config.json"
    #json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm/lincls_CorNET_dataset_apple100_split{split}_eps60_bs128_config.json"
    json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_nnclr_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1t_warp_aug2bioglass_dim-pdim128-128_EMA0.996_criterion_NTXent_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_lr_1.0E-05_lr_lstm_1.0E-04_hrmin_30_hrmax_120_CorNET_dataset_appleall_split{split}_eps60_bs512_config.json"
    #json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm/lincls_hrmin_30_hrmax_120_CorNET_dataset_max_split{split}_eps60_bs512_config.json"
    # json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm/lincls_hrmin_30_hrmax_120_CorNET_dataset_max_split4_eps60_bs512_config.json"
    #json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_nnclr_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1perm_jit_aug2bioglass_dim-pdim128-128_EMA0.996_criterion_NTXent_lambda1_1.0_lambda2_1.0_tempunit_tsfm_pretrain_subsample_0.100/lincls__lr_0.000_lr_lstm_0.000CorNET_dataset_apple100_split{split}_eps60_bs128_config.json"
    with open(json_file) as json_file:
        json_args = json.load(json_file)



    parser = get_parser()
    args = parser.parse_args([])
    args_dict = vars(args)
    args_dict.update(json_args)
    args = Namespace(**args_dict)



    mode = "finetune"


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

    # Load data
    train_loader, val_loader, test_loader = setup_dataloaders(args)


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
        model_test = CorNET(n_channels=args.n_feature, 
                                  n_classes=args.n_class, 
                                  conv_kernels=args.num_kernels, 
                                  kernel_size=args.kernel_size, 
                                  LSTM_units=args.lstm_units, 
                                  backbone=True, 
                                  input_size=args.input_length, 
                                  rnn_type=args.rnn_type,
                                  dropout_rate=args.dropout_rate)
    elif args.backbone == "TCN":
        model_test = TemporalConvNet(num_channels=[32, 64, 128], n_classes=args.n_class,  num_inputs=args.n_feature, input_length = args.input_length, kernel_size=16, dropout=0.2, backbone=True)
    else:
        NotImplementedError

    # Testing
    model_weights_dict = torch.load(model_weights_path, map_location = DEVICE)

    if mode == "finetune":
        classifier = setup_linclf(args, DEVICE, model_test.out_dim)
        model_test.set_classification_head(classifier)
        model_weights = load_best_lincls(args, device=DEVICE)


    elif mode == "pretrain":
        model_weights = model_weights_dict["model_state_dict"]

    model_test.load_state_dict(model_weights, strict=False)
    model_test = model_test.to(DEVICE)



    
    model_test.eval()
    for i, (x, y, d) in enumerate(test_loader):

        x = x.to(DEVICE).float()
        y = y.to(DEVICE)
        d = d.to(DEVICE)
        p, feat = model_test(x)
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        d = d.detach().cpu().numpy()
        p = p.detach().cpu().numpy()
        feat = feat.detach().cpu().numpy()
        X.append(x)
        Y.append(y)
        D.append(d)
        P.append(p)
        feats.append(feat)
        splits.append([split]*len(y))
        
X = np.concatenate(X)
Y = np.concatenate(Y)
D = np.concatenate(D)
P = np.concatenate(P).squeeze()
splits = np.concatenate(splits)
feats = np.concatenate(feats)


P = P*(args.hr_max - args.hr_min) + args.hr_min
Y = Y*(args.hr_max - args.hr_min) + args.hr_min


#%%

corr = np.corrcoef(Y, P)[0,1]
mae = np.mean(np.abs(Y-P))
rmse = np.sqrt(np.mean((Y-P)**2))
print(f"Correlation: {corr}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

#%%
df_bioglass = pd.read_pickle(os.path.join(config.classical_results_dir, "predictions_Bioglass_test_Apple100_split0.pkl"))
df_ssa = pd.read_pickle(os.path.join(config.classical_results_dir, "predictions_SSA_test_Apple100_split0.pkl"))


#%%
df_res = pandas.DataFrame({"Y": Y, 
                           "P": P, 
                           "sub": [el[0] for el in D],
                            "time": [el[1] for el in D], 
                           "X": [el for el in X],
                           "split" : splits})

df_res["ae"] = np.abs(df_res["Y"] - df_res["P"])
df_res.set_index("time", inplace=True)
#%%

#df_res["sub"] = df_res["sub"].map(subjects_mapping)

subjects = df_res["sub"].unique()

df_grouped = pd.concat([df_res.groupby("sub").apply(lambda x: x['Y'].corr(x['P'])), df_res.groupby("sub").apply(lambda x: np.mean(np.abs(x['Y']-x['P'])))], axis=1)
df_grouped.columns = ["corr", "mae"]
df_grouped


#%%

from sklearn.manifold import TSNE

# y_ground_truth = y_ground_truth.cpu().detach().numpy()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(P)


# %%

#color = np.max(np.abs(X),axis=1).mean(axis=1)
#color = np.apply_along_axis(lambda x: mean_frequency(x), 1, X).mean(axis=1)
color = Y
plt.scatter(x=tsne_results[:,0], y=tsne_results[:,1], c=color, cmap='viridis', alpha=0.2)

# Add a colorbar to show the mapping between colors and values
plt.colorbar(label='TRUE HR [bpm]')
plt.title("TSNE of HR predictions with true HR as color")
# %%
from scipy.signal import periodogram
def mean_frequency(signal, fs=100):
    f, Pxx = periodogram(signal, fs=fs)
    return np.sum(f*Pxx)/np.sum(Pxx)

mean_frequency(X[20,:,0])
# %%
