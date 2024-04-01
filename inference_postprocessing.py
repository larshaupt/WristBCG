#%%
import os
import json
import pandas
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from scipy.stats import invgauss, norm

from models.backbones import *
from models.loss import *
from trainer import *
from main import get_parser
from argparse import Namespace

def ece_loss(y_true, y_pred, bins=10):

    """
    Calculate the Expected Calibration Error (ECE) between predicted probabilities and true labels.

    ECE measures the discrepancy between predicted probabilities (y_pred) and empirical accuracy (y_true).

    Parameters:
    y_true (array-like): True labels. Each row corresponds to a sample, and each column corresponds to a class.
    y_pred (array-like): Predicted probabilities. Each row corresponds to a sample, and each column corresponds to a class.
    bins (int, optional): Number of equally spaced bins for dividing the range of predicted probabilities. Default is 10.

    Returns:
    float: The Expected Calibration Error (ECE) value.

    Notes:
    - The best possible value of ECE is 0, indicating perfect calibration.
    - The worst possible value of ECE is 1, indicating complete miscalibration.
    """
        
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(y_pred, axis=1)
    accuracies = y_true[np.arange(len(y_true)), np.argmax(y_pred, axis=1)]
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return ece

#%%
#json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm/lincls_CorNET_dataset_apple100_split0_eps60_bs128_config.json"
#json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm/lincls_bnn_pretrained_CorNET_dataset_apple100_split0_eps60_bs128_config.json"
#json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_simsiam_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1perm_jit_aug2bioglass_dim-pdim128-128_EMA0.0_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_pretrain_subsample_0.100/config.json"
#json_file = '/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm/lincls_gaussian_classification_disc_hr_64_hrmin_30.0_CorNET_dataset_apple100_split0_eps60_bs128_config.json'
json_file = '/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs512_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm_gru/lincls_NLE_hrmin_30_hrmax_120_CorNET_dataset_apple100_split1_eps60_bs512_config.json'
#json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_reconstruction_backbone_AE_pretrain_max_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_MSE_lambda1_1.0_lambda2_1.0_tempunit_tsfm/config.json"
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

# %%
# Testing
#######################
DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE, 'dataset:', args.dataset)
#%%
# Load data
train_loader, val_loader, test_loader = setup_dataloaders(args, mode="postprocessing")

#%%
# Initialize and load test model
model_test, _ = setup_model_optm(args, DEVICE)

#%%
# Testing

if mode == "finetune":
    
    classifier = setup_linclf(args, DEVICE, model_test.out_dim)
    model_test.set_classification_head(classifier)
    model_weights = load_best_lincls(args)

else:
    NotImplementedError("Only finetuning supported for now")

model_test.load_state_dict(model_weights, strict=False)
model_test = model_test.to(DEVICE)
model_test = add_probability_wrapper(model_test, args, DEVICE)
model_test.return_probs = True
model_test.uncertainty_model = "std"


#%%

X, Y_prob, D, P, P_prob, P_uncert = [], [], [], [], [], []
#%%

#tau = 1.352632
#tau = 1
#model_test.classifier.set_temperature(tau)
model_test.eval()
data_loader = test_loader

with torch.no_grad():
    for i, (x, y, d) in enumerate(data_loader):
        print(f"Batch {i}/{len(data_loader)}")
        x = x.to(DEVICE).float()
        y = y.to(DEVICE)
        d = d.to(DEVICE)
        exp, p_uncert, p = model_test(x)
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        d = d.detach().cpu().numpy()
        p = p.detach().cpu().numpy()
        p_uncert = p_uncert.detach().cpu().numpy()
        X.append(x)
        Y_prob.append(y)
        D.append(d)
        P_prob.append(p)
        P_uncert.append(p_uncert * (args.hr_max - args.hr_min))
        P.append(convert_to_hr(exp, args))


Y = [convert_to_hr(el, args) for el in Y_prob]


Y_concat = np.concatenate(Y)
P_concat = np.concatenate(P).squeeze()
D_concat = np.concatenate(D)
X_concat = np.concatenate(X)
P_prob_concat = np.concatenate(P_prob)
P_uncert_concat = np.concatenate(P_uncert)

ece = ece_loss(np.concatenate(Y_prob), np.concatenate(P_prob))
mae = np.mean(np.abs(Y_concat-P_concat))
corr = np.corrcoef(Y_concat, P_concat)[0,1]
#res.append({"tau": tau, "ece": ece, "mae": mae, "corr": corr})
#res_df = pandas.DataFrame(res, columns=["tau", "ece", "mae", "corr"])


#%%
ece = ece_loss(np.concatenate(Y_prob), np.concatenate(P_prob))
corr = np.corrcoef(Y_concat, P_concat)[0,1]
mae = np.mean(np.abs(Y_concat-P_concat))
rmse = np.sqrt(np.mean((Y_concat-P_concat)**2))

print(f"Correlation: {corr}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"ECE: {ece}")



#%%

df_res = pandas.DataFrame({"Y": Y_concat, 
                           "P": P_concat,
                           "uncert": P_uncert_concat.reshape(-1),
                           "sub": [el[0] for el in D_concat],
                            "time": [el[1] for el in D_concat], 
                           "X": [el for el in X_concat]})

df_res["abs_error"] = np.abs(df_res["Y"] - df_res["P"])
subjects = df_res["sub"].unique()

corr_func = lambda x: x['Y'].corr(x['P'])
mae_func = lambda x: np.mean(np.abs(x['Y']-x['P']))

df_grouped = pd.concat([df_res.groupby("sub").apply(corr_func), df_res.groupby("sub").apply(mae_func)], axis=1)
df_grouped.columns = ["corr", "mae"]
df_grouped.loc["Total"] = [corr_func(df_res), mae_func(df_res)]
df_grouped


#%%
# Look at average HR and HR predictions

fig, axes = plt.subplots(2,len(subjects), figsize=(3*len(subjects),8), dpi=500, sharex=True)

for i, sub in enumerate(subjects):

    df_sub = df_res[df_res["sub"] == sub]
    
    df_sub["Y"].hist(ax=axes[0, i], label="Y", bins=100)
    axes[0, i].set_title(f"Subject {sub} HR distribution")
    df_sub["P"].hist(ax=axes[1, i], label="P", bins=100)
    axes[1, i].set_title(f"Subject {sub} HR prediction distribution")
    fig.tight_layout()

# %%
# Look at HR and HR predictions over time
fig, axes = plt.subplots(len(subjects),1, figsize=(8,3*len(subjects)))
for i, sub in enumerate(subjects):
    df_sub = df_res[df_res["sub"] == sub]
    #df_sub = df_sub.iloc[500:-500]
    
    df_sub["Y"].rolling(10).median().plot(ax=axes[i], label="Y")
    df_sub["P"].rolling(10).median().plot(ax=axes[i], label="P")
    axes[i].set_title(f"Subject {sub} HR over time")
    axes[i].legend()

    ax2 = axes[i].twinx()
    df_sub["uncert"].rolling(10).median().plot(ax=ax2, color="red", label="uncert", alpha=0.5)
    
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
import trainer
import importlib
import models.prior_layer
importlib.reload(models.prior_layer)
importlib.reload(trainer)

postprocessing_model = models.prior_layer.PriorLayer(
            args.n_prob_class, 
            min_hr=args.hr_min, 
            max_hr=args.hr_max, 
            is_online= False,
            return_probs= True,
            uncert = "std")
ys = [target for _, target, _ in train_loader]
# Sometimes we get a Value Error here. This is because the probabilities for some classes are too small. Taking the log will then result in infinite values
postprocessing_model.fit_layer(ys, distr="laplace", learn_state_prior=True)
postprocessing_model = postprocessing_model.to(DEVICE)


#%%

uncerts, probs, preds = np.zeros_like(Y_concat), np.zeros_like(P_prob_concat), np.zeros_like(P_concat)

for sub in subjects:
    sub_index = D_concat[:,0] == sub
    hr_pred, uncert, hr_pred_prob = postprocessing_model(torch.Tensor(np.concatenate(P_prob)[sub_index]).to(DEVICE), 
                method="raw")
    preds[sub_index] = hr_pred.cpu().numpy()

    if uncert is not None:
        uncerts[sub_index] = uncert.cpu().numpy()
        probs[sub_index] = hr_pred_prob.cpu().numpy()
    

ece = ece_loss(np.concatenate(Y_prob), probs)
corr = np.corrcoef(Y_concat, preds)[0,1]
mae = np.mean(np.abs(Y_concat-preds))
rmse = np.sqrt(np.mean((Y_concat-preds)**2))

print(f"Correlation: {corr}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"ECE: {ece}")


# %%
def confidence_plot(uncert, pred_expectation, Y, ax=None, name="", distribution="gaussian", num_points = 10):
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,5))


    def gaussian_confidence_interval(conf, uncert):
        return norm.ppf(0.5 + conf/2, loc=0, scale=uncert)

    def laplace_confidence_interval(conf, uncert):
        return - uncert/np.sqrt(2) * np.log(1-conf)
    if distribution == "gaussian":
        conf_function = gaussian_confidence_interval
    elif distribution == "laplace":
        conf_function = laplace_confidence_interval
    else:
        raise ValueError(f"Distribution {distribution} not supported")

    conf_scores = []
    for conf in np.linspace(0, 1, num_points + 1):
        conf_interval = conf_function(conf, uncert).reshape(-1)
        inside_confidence_interval = (Y > pred_expectation - conf_interval) & (Y < pred_expectation + conf_interval)
        conf_score = inside_confidence_interval.sum() / len(inside_confidence_interval)
        conf_scores.append((conf, conf_score))
    df_conf_scores = pandas.DataFrame(conf_scores, columns=["Confidence", "Score"])
    ece = (df_conf_scores["Confidence"] - df_conf_scores["Score"]).abs().mean()
    
    df_conf_scores["Confidence"] = df_conf_scores["Confidence"].round(1)
    df_conf_scores.plot.bar(x="Confidence", y="Score", ax=ax, label=name)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Percentage of data inside confidence interval")
    ax.set_title("Confidence Calibration")
    ax.set_ylim(0,1)
    ax.plot(np.arange(0,1.1,1/num_points), color="red")
    return ece

confidence_plot(uncerts, preds, Y_concat, distribution="gaussian", name="Raw Prediction", num_points=10)
#%%

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



plot_excluion(Y_concat, preds, uncerts, metric="MAE", name="Raw Prediction")

# %%
def plot_uncert_vs_mae(HR_true, HR_pred, uncertainty, ax=None,  name=""):

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,5))

    df_res = pandas.DataFrame({"hr_true": HR_true, 
                            "hr_pred": HR_pred, 
                            "uncertainty": uncertainty})



    df_res["bin"] = pd.cut(df_res["uncertainty"], bins=10)
    df_res["bin"] = df_res["bin"].apply(lambda x : np.round((x.left + x.right)/2, 4))
    df_res["mse"] = (df_res["hr_true"] - df_res["hr_pred"])**2
    groups = df_res.groupby("bin")["mse"].mean()
    groups.plot.bar(ax=ax, alpha=0.5, label=name)

    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("MSE (Mean)")
    ax.set_title("Uncertainty vs MAE")
    ax.legend()

plot_uncert_vs_mae(Y_concat, preds ,uncerts, name="Raw Prediction")

# %%


def plot_hit_rate(HR_true, HR_pred_probs, ax=None,  name="", hrmin=50, hrmax = 110):

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,5))

    if isinstance(HR_true, torch.Tensor) or isinstance(HR_true, np.ndarray):
        HR_pred_probs = [el for el in HR_pred_probs]

    df_res = pandas.DataFrame({"hr_true": HR_true, 
                            "probs": HR_pred_probs})


    df_res["probs"] = df_res["probs"].apply(lambda x : x.reshape(-1, 8).sum(axis=1))
    df_res["bin_l"] = [np.concatenate([[-np.inf], np.linspace(0,1,7)])]*len(df_res)
    df_res["bin_h"] = [np.concatenate([np.linspace(0,1,7), [np.inf],])]*len(df_res)

    df_res = df_res.explode(["bin_l", "bin_h", "probs"])

    df_res["hit"] = df_res.apply(lambda row: (row["hr_true"] -hrmin)/(hrmax-hrmin) > row["bin_l"] and (row["hr_true"] -hrmin)/(hrmax-hrmin) < row["bin_h"], axis=1)
    df_res["bin"] = pd.cut(df_res["probs"], bins=np.arange(0,1.1,0.1))

    groups = df_res.groupby("bin")["hit"].mean()
    groups.plot.bar()
    plt.plot(np.arange(0.05,1.0,0.1), color="red")
    plt.ylim(0,1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Confidence Calibration (8 bins)")

plot_hit_rate(Y_concat, probs, name="Raw Prediction", hrmin = args.hr_min, hrmax = args.hr_max)



#%%

def plot_hr(HR_true, HR_pred, uncert = None, HR_pred_raw = None, ax = None, title = ""):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,5))
    ax.plot(HR_true, label="True HR")
    ax.plot(HR_pred, label="Pred HR")
    if uncert is not None:
        ax.fill_between(np.arange(len(HR_true)), HR_pred-uncert, HR_pred+uncert, alpha=0.3, label="Uncertainty")
    if HR_pred_raw is not None:
        ax.plot(HR_pred_raw, label="Raw Pred HR", alpha=0.5)
    ax.set_title(title)
    ax.legend()

#plot_hr(Y_concat, preds, uncerts, HR_pred_raw=P_concat, title="HR Prediction")
plot_hr(Y_concat, preds, uncerts, title="HR Prediction")
# %%
