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

from models.backbones import *
from models.loss import *
from trainer import *
from argparse import Namespace
import classical_utils
from main import get_parser


#%%

#json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_stepsize_4/lincls_hrmin_20_hrmax_120_CorNET_dataset_apple100_split0_eps60_bs128_config.json"
#json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm/lincls_CorNET_dataset_apple100_split0_eps60_bs128_config.json"
json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm/lincls_hrmin_20_hrmax_120_CorNET_dataset_apple100_split0_eps60_bs512_config.json"
with open(json_file) as json_file:
    json_args = json.load(json_file)



parser = get_parser()
args = parser.parse_args([])
args_dict = vars(args)
args_dict.update(json_args)
args = Namespace(**args_dict)

split = 0

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
# %%
# Testing
#######################
DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE, 'dataset:', args.dataset)
#%%
# Load data
train_loader, val_loader, test_loader = setup_dataloaders(args)

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

model_test.load_state_dict(model_weights)
model_test = model_test.to(DEVICE)

#%%

X, Y, D, P = [], [], [], []

for i, (x, y, d) in enumerate(test_loader):

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
                           "X": [el for el in X]})
#%%
subjects_mapping = {53.0: "5",
                    549511850: "6_v2",
                    509511850: "2_v2"}

#df_res["sub"] = df_res["sub"].map(subjects_mapping)

subjects = df_res["sub"].unique()

df_grouped = pd.concat([df_res.groupby("sub").apply(lambda x: x['Y'].corr(x['P'])), df_res.groupby("sub").apply(lambda x: np.mean(np.abs(x['Y']-x['P'])))], axis=1)
df_grouped.columns = ["corr", "mae"]
df_grouped

# --> We see that the performance is different for different subjects in the test set.
# --> We can also see that while the correlation is quite low, the MAE is quite high for each subject
# --> We conclude that the dataset does not have enough variety in HR
#%%
# Look at average HR and HR predictions

fig, axes = plt.subplots(2,len(subjects), figsize=(len(subjects)*5,8), dpi=500, sharex=True)

for i, sub in enumerate(subjects):

    df_sub = df_res[df_res["sub"] == sub]
    
    df_sub["Y"].hist(ax=axes[0, i], label="Y", bins=100)
    axes[0, i].set_title(f"Subject {sub} HR distribution")
    df_sub["P"].hist(ax=axes[1, i], label="P", bins=100)
    axes[1, i].set_title(f"Subject {sub} HR prediction distribution")
    fig.tight_layout()

# %%
# Look at HR and HR predictions over time
fig, axes = plt.subplots(2,2, figsize=(20,10), sharey=True)
for i, sub in enumerate(subjects):

    if i > 3:
        break
    df_sub = df_res[df_res["sub"] == sub]
    
    df_sub["Y"].rolling(10).median().plot(ax=axes[i%2,i//2], label="Ground Truth", marker="", alpha=1.0, linewidth=2, color='green')
    df_sub["P"].rolling(10).median().plot(ax=axes[i%2,i//2], label="Supervised Regression", marker="", alpha=1.0, color="#A1A9AD")
    #df_bioglass[df_res["sub"] == sub]["hr_pred"].rolling(10).median().plot(ax=axes[i%2,i//2], label="Bioglass", marker="", alpha=0.5, color="#5BC5DB")
    #df_ssa[df_res["sub"] == sub]["hr_pred"].rolling(10).median().plot(ax=axes[i%2,i//2], label="SSA", marker="", alpha=0.5, color="#A12864")
    axes[i%2,i//2].set_title(f"Subject {sub}")
    axes[i%2,i//2].set_xlabel("Time [s]")
    axes[i%2,i//2].set_ylabel("HR [bpm]")

axes[0,0].legend()

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
        axes[0,i_best].plot(df_sub["X"].iloc[i_best])
        axes[0,i_best].set_title(f"HR: {df_sub['Y'].iloc[i_best]:.1f}, Pred: {df_sub['P'].iloc[i_best]:.1f}")

    for i_worst in range(3):
        axes[1,i_worst].plot(df_sub["X"].iloc[-i_worst -1])
        axes[1,i_worst].set_title(f"HR: {df_sub['Y'].iloc[-i_worst -1]:.1f}, Pred: {df_sub['P'].iloc[-i_worst -1]:.1f}")
    fig.suptitle(f"Subject {sub} best and worst predictions")
    fig.tight_layout()


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
