#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
import wandb
import scipy.signal

from classical_utils import *

#%%

results_dir = config.classical_results_dir

dataset = "Apple100"
framework = "SSA"
wandb_mode = "disabled"

if dataset == "max":
    dataset_dir = config.data_dir_Max_processed
elif dataset == "Apple100":
    dataset_dir = config.data_dir_Apple_processed_100hz
else:
    NotImplementedError


#%%

""" 
for split in range(5):
    print(f"Processing split {split}")
    X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = load_dataset(dataset_dir, split)
    predictions_path_val = os.path.join(results_dir, f"predictions_{framework}_val_{dataset}_split{split}.pkl")
    predictions_path_test = os.path.join(results_dir, f"predictions_{framework}_test_{dataset}_split{split}.pkl")
    config_dict = {"peak_distance": 0.5, 
                "peak_prominence": 0.3, 
                "framework": framework,
                "dataset": dataset,
                "split": split,
                "predictions_path_val": predictions_path_val,
                "predictions_path_test": predictions_path_test,}
    print(f"Config: {config_dict}")
    if config_dict["framework"] == 'Bioglass':
        compute_hr = compute_hr_bioglass
    elif config_dict["framework"] == 'SSA':
        compute_hr = compute_hr_ssa

    # Initialize WandB with your project name and optional configuration
    wandb.init(project="hr_results", config=config_dict, group=config_dict["framework"], mode=wandb_mode)
    hr_rr_val = []
    print(f"Processing val set")
    for i, X in enumerate(X_val):
        hr_rr = compute_hr(X, fs=100)
        hr_rr_val.append(hr_rr)
    results_df_val = pd.DataFrame({"y_true": Y_val, "hr_pred": hr_rr_val, "pid": [el for el in pid_val], "metrics": [el for el in metrics_val]})
    corr_val = results_df_val[["y_true", "hr_pred"]].corr().iloc[0,1]
    results_df_val["diff_abs"] = (results_df_val["y_true"] - results_df_val["hr_pred"]).abs()
    mae_val = results_df_val.dropna()["diff_abs"].mean()

    wandb.log({"Val_Corr": corr_val, "Val_MAE": mae_val})
    results_df_val.to_pickle(predictions_path_val)
    print(f"Saves results to {predictions_path_val}")


    hr_rr_test = []
    print(f"Processing test set")
    for i, X in enumerate(X_test):
        hr_rr = compute_hr(X, fs=100)

        hr_rr_test.append(hr_rr)    
    results_df_test = pd.DataFrame({"y_true": Y_test, "hr_pred": hr_rr_test, "pid": [el for el in pid_test], "metrics": [el for el in metrics_test]})
    corr_test = results_df_test[["y_true", "hr_pred"]].corr().iloc[0,1]
    results_df_test["diff_abs"] = (results_df_test["y_true"] - results_df_test["hr_pred"]).abs()
    mae_test = results_df_test.dropna()["diff_abs"].mean()

    wandb.log({"Test_Corr": corr_test, "Test_MAE": mae_test})
    print(f"Saves results to {predictions_path_test}")
    results_df_test.to_pickle(predictions_path_test)
    # Finish the run
    wandb.finish() """

# %%
split = 0
X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = load_dataset(dataset_dir, split)


import TROIKA_bcg
from sklearn.metrics import pairwise_distances

def wcorr(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    weighted correlation of ts1 and ts2.
    w is precomputed for reuse.
    """
    L = 500
    N = 1000
    K = N - L + 1
    w = np.concatenate((np.arange(1, L+1), np.full((K-L,), L), np.arange(L-1, 0, -1)))
    w_covar = (w * ts1 * ts2).sum()
    ts1_w_norm = np.sqrt((w * ts1 * ts1).sum())
    ts2_w_norm = np.sqrt((w * ts2 * ts2).sum())
    
    return w_covar / (ts1_w_norm * ts2_w_norm)


def select_components(acc_groups, threshold=0.1):
    selected_indices = []
    for i in range(acc_groups.shape[0]):
        _, periodogram = scipy.signal.periodogram(acc_groups[i,:], nfft=4096 * 2 - 1)
        frequencies = np.linspace(0,100, 4096)
        max_amplitude = np.max(np.abs(periodogram))
        hr_frequenies = (frequencies > 0.5) & (frequencies < 4)

        if np.any(periodogram[hr_frequenies] > max_amplitude*threshold):
            selected_indices = np.append(selected_indices, i)

    selected_indices = np.array(selected_indices, dtype=int)
    acc_reconstructed = acc_groups[selected_indices,:].sum(axis=0)
    return acc_reconstructed

results = []
for i in range(len(X_test)):
    print(f"Processing {i}")
    acc = X_test[i,:,:]
    hr_true = Y_test[i]



    acc_x = data_utils.butterworth_bandpass(acc[:,0], low=0.5, high=4, fs=100)
    acc_y = data_utils.butterworth_bandpass(acc[:,1], low=0.5, high=4, fs=100)
    acc_z = data_utils.butterworth_bandpass(acc[:,2], low=0.5, high=4, fs=100)
    acc_groups_x, wcorr_x = TROIKA_bcg.ssa(acc_x, 500, perform_grouping=True, ret_Wcorr=True)
    acc_groups_y, wcorr_y = TROIKA_bcg.ssa(acc_y, 500, perform_grouping=True, ret_Wcorr=True)
    acc_groups_z, wcorr_z = TROIKA_bcg.ssa(acc_z, 500, perform_grouping=True, ret_Wcorr=True)


    def select_components(acc_groups, threshold=0.1):
        selected_indices = []
        for i in range(acc_groups.shape[0]):
            frequencies, periodogram = scipy.signal.periodogram(acc_groups[i,:], nfft=4096 * 2 - 1, fs=100)
            max_amplitude = np.max(np.abs(periodogram))
            hr_frequenies = (frequencies > 0.5) & (frequencies < 3)

            if np.any(periodogram[hr_frequenies] > max_amplitude*threshold):
                selected_indices = np.append(selected_indices, i)
        #print(selected_indices)
        selected_indices = np.array(selected_indices, dtype=int)
        acc_reconstructed = acc_groups[selected_indices,:].sum(axis=0)
        return acc_reconstructed


    acc_reconstructed_x = select_components(acc_groups_x)
    acc_reconstructed_y = select_components(acc_groups_y)
    acc_reconstructed_z = select_components(acc_groups_z)

    acc_reconstructed = np.sqrt(acc_reconstructed_x**2 + acc_reconstructed_y**2 + acc_reconstructed_z**2)

    #plt.plot(acc_reconstructed)

    # differentiating for more robustness
    acc_reconstructed = np.diff(acc_reconstructed)


    frequencies, periodogram = scipy.signal.periodogram(acc_reconstructed, nfft=4096 * 2 - 1, fs=100)
    #plt.plot(frequencies, periodogram)
    #plt.xlim(0, 3)

    # plot y log
    #plt.yscale("log")
    hr_frequenies_ind = (frequencies > 0.5) & (frequencies < 2)
    hr_periodogram, hr_frequenies = periodogram[hr_frequenies_ind], frequencies[hr_frequenies_ind]
    hr_peak_ind = scipy.signal.find_peaks(hr_periodogram)[0]
    highest_hr_peak_ind = np.argsort(hr_periodogram[hr_peak_ind])[::-1][:10]
    # take highest peaks in the range 0.5-2 Hz

    highest_hr_peaks = hr_frequenies[hr_peak_ind[highest_hr_peak_ind]] * 60
    highest_hr_peak = highest_hr_peaks[0]
    closest_hr_peak = highest_hr_peaks[np.argmin(np.abs(highest_hr_peaks - hr_true))]
    print(highest_hr_peak,closest_hr_peak, hr_true)

    results.append((highest_hr_peak, closest_hr_peak, hr_true))

results_df = pd.DataFrame(results, columns=["highest_hr_peak", "closest_hr_peak", "hr_true"])
results_df.to_csv("ssa_results.csv")


# %%
"""

ssr = TROIKA_bcg.SSR(1000, 4096, 100, lambda_factor=0.05)
acc_spectrum = ssr.transform(ssa_acc)
import importlib
importlib.reload(TROIKA_bcg)
# %%



 # %%
ssr = TROIKA_bcg.SSR(1000, 4096, 100, lambda_factor=0.0005)
acc_spectrum = ssr.transform(acc_reconstructed)
# %%


acc_reconstructed
# %%

f_s = 100
M = 1000 #num samples in window
N = 4096 #num frequencies
lambda_factor = 1


# %%
n_start = (0.1*N//100) - 1
n_end = (4.4*N//100) + 1
ns = np.arange(n_start, n_end)
m, n = np.meshgrid(np.arange(M), np.arange(n_start, n_end), indexing='ij')
Phi = np.exp(1j * 2 * np.pi / N * m * n)
# %%

# %%
from scipy.optimize import minimize
y = acc_reconstructed
phi_pinv = np.linalg.pinv(Phi)
lambda_factor = 500
x0 = phi_pinv @ y
print(f"Sparsity: {np.linalg.norm(x0, ord=1)}, estimation quality: {np.linalg.norm(Phi @ x0 - y, ord=2) * lambda_factor}")
def print_current_target(xk):
    print(f"Sparsity: {np.linalg.norm(xk, ord=1)}, estimation quality: {np.linalg.norm(Phi @ xk - y, ord=2) * lambda_factor}")

constraints = {
    'type': 'eq',
    'fun': lambda x: np.linalg.norm(Phi @ x - y, ord=2) * lambda_factor
}
optimize_result = minimize(lambda x: np.linalg.norm(x, ord=1),
                            x0, method='SLSQP',
                            options={'maxiter': 5},
                            constraints=constraints,
                            callback=print_current_target)
s_k = optimize_result.x ** 2
# %%

import cr.sparse.pursuit.mp as mp
sol = mp.solve(A, acc_reconstructed)
x = sol.x

#%%
import cr.sparse.cvx.spgl1 as crspgl1
sigma=0.
options = crspgl1.SPGL1Options(max_iters=300)
tracker = crs.ProgressTracker(every=10)
x_init = scipy.signal.periodogram(acc_reconstructed, nfft=4096 * 2 - 1)
sol = crspgl1.solve_bpic_from_jit(A, acc_reconstructed, sigma,
    x_init, options=options, tracker=tracker)
# %%
import cr.sparse.lop as lop
A = lop.matrix(Phi)
# %%
 """