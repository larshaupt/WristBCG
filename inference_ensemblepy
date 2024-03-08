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
preds_dict = {}
DEVICE = None
test_loader = None
for random_seed in [0,1,2,3,4,5,6,7,10]:


    for split in [0,1,2,3,4]:


        rseed = f"_rseed_{random_seed}" if random_seed != 10 else ""

        json_file = f"/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm/lincls_hrmin_30_hrmax_120{rseed}_CorNET_dataset_apple100_split{split}_eps60_bs512_config.json"

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

        args.cuda = 1


        
        if DEVICE is None:
            DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
            print('device:', DEVICE, 'dataset:', args.dataset)
        if test_loader is None:       
            train_loader, val_loader, test_loader = setup_dataloaders(args)


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
        model_weights_dict = torch.load(model_weights_path, map_location = DEVICE)

        if mode == "finetune":
            classifier = setup_linclf(args, DEVICE, model_test.out_dim)
            model_test.set_classification_head(classifier)
            model_weights = load_best_lincls(args, device=DEVICE)


        elif mode == "pretrain":
            model_weights = model_weights_dict["model_state_dict"]

        model_test.load_state_dict(model_weights, strict=False)
        model_test = model_test.to(DEVICE)




        X, Y, D, P = [], [], [], []
        model_test.eval()
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

        preds_dict[(random_seed, split)] = P

#%%
#df_preds = pd.Series(preds_dict)

splits_all = [ [0], [1], [2], [3], [4], [0,1,2,3,4]]
random_seeds = [[0],[1],[2],[3],[4],[5],[6],[7],[10], [0,1,2,3,4,5,6,7,10]]

res = {}

for splits in splits_all:

    preds = np.array([preds_dict[(rseed,split)] for split in splits for rseed in random_seeds[-2]])
    #preds = np.array([preds_dict[(rseed,split)] for split in splits])
    #preds = np.array([preds_dict[(rseed,split)] for rseed in random_seeds])
    P = np.mean(preds, axis=0)
    uncert = np.std(preds, axis=0)


    corr = np.corrcoef(Y, P)[0,1]
    mae = np.mean(np.abs(Y-P))
    rmse = np.sqrt(np.mean((Y-P)**2))
    print(f"Correlation: {corr}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

    uncert_corr = np.corrcoef(np.abs(Y-P), uncert)[0,1]

    if  len(splits) == 1:

        res[splits[0]] = {"corr": corr, "mae": mae, "rmse": rmse, "uncert_corr": uncert_corr}
    else:
        res["all"] = {"corr": corr, "mae": mae, "rmse": rmse, "uncert_corr": uncert_corr}

#%%
    
df_res = pandas.DataFrame(res).T
df_res["corr"].plot(kind="bar")
plt.ylabel("Correlation")
plt.xlabel("Split ID")
plt.title("Correlation per Split")

#%%
df_bioglass = pd.read_pickle(os.path.join(config.classical_results_dir, "predictions_Bioglass_test_Apple100_split0.pkl"))
df_ssa = pd.read_pickle(os.path.join(config.classical_results_dir, "predictions_SSA_test_Apple100_split0.pkl"))

#%%

df_res = pandas.DataFrame({"Y": Y, 
                           "P": P, 
                           "sub": [el[0] for el in D],
                            "time": [el[1] for el in D], 
                           "X": [el for el in X]})

df_res["ae"] = np.abs(df_res["Y"] - df_res["P"])
df_res.set_index("time", inplace=True)
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
fig, axes = plt.subplots(len(subjects), 1, figsize=(8,len(subjects)*4), sharex=False, sharey=True)
for i, sub in enumerate(subjects):

    df_sub = df_res[df_res["sub"] == sub]
    
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

from scipy.signal import butter, sosfiltfilt
res = []
for i in np.arange(0,len(X),1):
    print(i)
    input_X = X[i,...]
    hr_true = Y[i]

    hr_pred_orig, _ = model_test(torch.Tensor(input_X.copy()).unsqueeze(0).to(DEVICE).float())
    hr_pred_orig = hr_pred_orig.item() * (args.hr_max - args.hr_min) + args.hr_min
    res.append([-1, hr_pred_orig, hr_pred_orig, hr_true])

    for axis in [0,1,2]:
        
        s_axis1 = axis
        s_axis2 = (axis+1) % 3
        input_X_axis = input_X.copy()
        input_X_axis[...,s_axis1] = input_X[...,s_axis2]
        input_X_axis[...,s_axis2] = input_X[...,s_axis1]
        out, feat = model_test(torch.Tensor(input_X_axis).unsqueeze(0).to(DEVICE).float())
        hr_pred = out.item() * (args.hr_max - args.hr_min) + args.hr_min
        res.append([axis, hr_pred, hr_pred_orig, hr_true])


res = np.array(res)


# %%

df_res = pandas.DataFrame(res, columns=["axis", "hr_pred", "hr_pred_orig", "hr_true"])
df_res["axis"] = df_res["axis"].map({0: "x <-> y", 1: "y <-> z", 2: "z <-> x", -1: "original"})
df_res["deviation"] = np.abs(df_res["hr_pred"] - df_res["hr_pred_orig"])
df_res["error"] = np.abs(df_res["hr_pred"] - df_res["hr_true"])
df_res.groupby("axis").agg({"deviation": ["median", "std"], "error": ["mean", "std"]})

#sns.boxplot(data=df_res, x="axis", y="error")

# %%


def plot_excluion(HR_true, HR_pred, uncertainty, ax=None, metric="MAE", name=""):

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,5))

    df_res = pandas.DataFrame({"hr_true": HR_true, 
                            "hr_pred": HR_pred, 
                            "uncertainty": uncertainty})

    df_res.sort_values("uncertainty", inplace = True, ascending=True)

    p_keep = np.linspace(0,100,100)
    total_num = len(df_res)
    if metric=="MAE":
        df_res["mae"] = np.abs(df_res["hr_true"] - df_res["hr_pred"])
        mae = [df_res["mae"].iloc[:int(p/100*total_num)].mean() for p in p_keep]
        ax.plot(p_keep, mae, label=name)
        ax.set_ylabel("MAE")
    
    elif metric=="Corr":
        corr = [df_res[["hr_true", "hr_pred"]].iloc[:int(p/100*total_num)].corr().iloc[0,1] for p in p_keep]
        ax.plot(p_keep, corr, label=name)
        ax.set_ylabel("Correlation")


    ax.set_xlabel("Percentage of data evaluated")
    ax.set_title("Exclusion of uncertain predictions")
    ax.legend()
# %%
