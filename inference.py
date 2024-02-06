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


#%%

json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_simsiam_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1perm_jit_aug2bioglass_dim-pdim128-128_EMA0.0_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_pretrain_subsample_0.100/config.json"
#json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_reconstruction_backbone_AE_pretrain_max_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_MSE_lambda1_1.0_lambda2_1.0_tempunit_tsfm/config.json"
with open(json_file) as json_file:
    json_args = json.load(json_file)

args = Namespace(**json_args)
split = 0

mode = "finetune"


if not hasattr(args, "model_name"):
    args.model_name = args.backbone + '_'+args.dataset + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) + '_split' + str(split)
else:
    args.model_name = re.sub(r'split\d+', f"split{split}", args.model_name)

if mode == "pretrain":
    model_weights_path = os.path.join(args.model_dir_name, 'pretrain_' + args.model_name  + "_bestmodel" + '.pt')
elif mode == "finetune":
    model_weights_path = os.path.join(args.model_dir_name, 'lincls_' + args.model_name + "_split" + str(args.split) + "_bestmodel" + '.pt')
elif mode == "supervised":
    model_weights_path = os.path.join(args.model_dir_name, args.model_name + '_model.pt')
args.batch_size = 1
args.cuda = 3
args.dataset = "apple100"
# %%
# Testing
#######################
DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE, 'dataset:', args.dataset)
#%%
# Load data
train_loader, val_loader, test_loader = setup_dataloaders(args)
#%%

class AE(nn.Module):
    def __init__(self, n_channels, input_size, n_classes, outdim=128, backbone=True):
        super(AE, self).__init__()

        self.backbone = backbone
        self.input_size = input_size

        self.e1 = nn.Linear(n_channels, 8)
        self.e2 = nn.Linear(8 * input_size, 2 * input_size)
        self.e3 = nn.Linear(2 * input_size, outdim)

        self.d1 = nn.Linear(outdim, 2 * input_size)
        self.d2 = nn.Linear(2 * input_size, 8 * input_size)
        self.d3 = nn.Linear(8, 1)

        self.out_dim = outdim

        if backbone == False:
            self.classifier = nn.Linear(outdim, n_classes)

    def forward(self, x):
        x_e1 = self.e1(x)
        x_e1 = x_e1.reshape(x_e1.shape[0], -1)
        x_e2 = self.e2(x_e1)
        x_encoded = self.e3(x_e2)

        x_d1 = self.d1(x_encoded)
        x_d2 = self.d2(x_d1)
        x_d2 = x_d2.reshape(x_d2.shape[0], self.input_size, 8)
        x_decoded = self.d3(x_d2)

        if self.backbone:
            return x_decoded, x_encoded
        else:
            out = self.classifier(x_encoded)
            return out, x_decoded
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
    model_weights = model_weights_dict["trained_backbone"]
    classifier = setup_linclf(args, DEVICE, model_test.out_dim)
    model_test.set_classification_head(classifier)

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

fig, axes = plt.subplots(2,3, figsize=(15,8), dpi=500, sharex=True)

for i, sub in enumerate(subjects):

    df_sub = df_res[df_res["sub"] == sub]
    
    df_sub["Y"].hist(ax=axes[0, i], label="Y", bins=100)
    axes[0, i].set_title(f"Subject {sub} HR distribution")
    df_sub["P"].hist(ax=axes[1, i], label="P", bins=100)
    axes[1, i].set_title(f"Subject {sub} HR prediction distribution")
    fig.tight_layout()

# %%
# Look at HR and HR predictions over time
fig, axes = plt.subplots(len(subjects),1, figsize=(8,15))
for i, sub in enumerate(subjects):
    df_sub = df_res[df_res["sub"] == sub]
    #df_sub = df_sub.iloc[500:-500]
    
    df_sub["Y"].rolling(10).median().plot(ax=axes[i], label="Y")
    df_sub["P"].rolling(10).median().plot(ax=axes[i], label="P")
    axes[i].set_title(f"Subject {sub} HR over time")
    axes[i].legend()
    #print(df_sub[["Y", "P"]].corr().iloc[0,1])

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
i = -8
df_sub = df_res[df_res["sub"] == subjects[0]]
df_sub["diff"] = np.abs(df_sub["Y"] - df_sub["P"])
df_sub = df_sub.sort_values("diff", ascending=True)
plt.plot(df_sub["X"].iloc[i])
plt.xlabel("Time")
plt.title(f"HR: {df_sub['Y'].iloc[i]:.1f}, Pred: {df_sub['P'].iloc[i]:.1f}")
# %%


import pickle
def load_subjects(dataset_dir, subject_paths):
    X, Y, pid, metrics = [], [], [], []
    for sub in subject_paths:
        with open(os.path.join(dataset_dir, sub), "rb") as f:
            data = pickle.load(f)
            if len(data) == 4:
                X_sub, Y_sub, pid_sub, metrics_sub = data
            else:
                X_sub, Y_sub, pid_sub = data
                metrics_sub = np.array([0] * len(Y_sub))

        X.append(X_sub)
        Y.append(Y_sub)
        pid.append(pid_sub)
        metrics.append(metrics_sub)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    pid = np.concatenate(pid, axis=0)
    metrics = np.concatenate(metrics, axis=0)

    return X, Y, pid, metrics

def load_dataset(dataset_dir, split, load_train=False):
    # load split
    split_file = os.path.join(dataset_dir, f'splits.json')

    if not os.path.exists(split_file):
        raise ValueError(f"Split file {split_file} does not exist")
    
    with open(split_file) as f:
        splits = json.load(f)

    split = splits[str(split)]


    # load data
    if load_train:
        X_train, Y_train, pid_train, metrics_train = load_subjects(dataset_dir, split["train"])
    X_val, Y_val, pid_val, metrics_val = load_subjects(dataset_dir, split["val"])
    X_test, Y_test, pid_test, metrics_test = load_subjects(dataset_dir, split["test"])

    if load_train:
        return X_train, Y_train, pid_train, metrics_train, X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test
    else:
        return X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test
    
#%%
data_dir = config.data_dir_Apple_processed_100hz_wmetrics
X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = load_dataset(data_dir, 0)
# %%



X, Y, D, P, M = [], [], [], [], []
# metrices are metric_avg, metric_max, metric_angle, metric_hr
for i, (x, y, d, metric) in enumerate(zip(X_test, Y_test, pid_test, metrics_test)):
    print(i)
    if not (metric[0] < 0.08 and metric[1] < 0.5 and metric[2] < 5 and metric[3] < 40):
       pass
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
# %%
D = D.reshape(-1,2)

selected_indices.shape, D.shape

# find indices of selected_indices in D
subjects = np.unique(D[:,0])
sub_dict = {}
for sub in subjects:
    D_sub = D[D[:,0] == sub]
    sel_sub = selected_indices[selected_indices[:,0] == sub]
    sub_in = np.where(np.isin(D_sub[:,1], sel_sub[:,1]))[0]
    ratio_sub = len(sub_in) / len(D_sub)
    sub_dict[sub] = sub_in
    print(sub)
    print(D_sub.shape, sel_sub.shape,np.intersect1d(D_sub[:,1], sel_sub[:,1]).shape, np.union1d(D_sub[:,1], sel_sub[:,1]).shape, ratio_sub)


# %%


# %%
import statsmodels.api as sm
M_s = sm.add_constant(M)
model = sm.OLS(Y, M_s).fit()
print(model.summary())
for feature in range(4):  # Exclude the constant term
    plt.figure(figsize=(8, 6))
    plt.scatter(M_s[:,feature], Y, label='Actual data')
    
# %%
