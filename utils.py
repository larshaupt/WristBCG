import os
import logging
import sys
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from torch.optim.lr_scheduler import _LRScheduler
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
import re
import numpy as np
import pickle

def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def tsne(latent, y_ground_truth, save_dir):
    """
        Plot t-SNE embeddings of the features
    """
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(latent)
    plt.figure(figsize=(16,10))
    plt.scatter(x=tsne_results[:,0], y=tsne_results[:,1], c=y_ground_truth, cmap='viridis', alpha=0.2)
    plt.colorbar(label='True HR [bpm]')
    plt.savefig(save_dir)
    plt.clf()
    with open(save_dir.replace('.png', '.pickle'), 'wb') as f:
        pickle.dump((tsne_results, y_ground_truth), f)



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

def main():
    free_gpu = get_free_gpu()
    if free_gpu is not None:
        print(f"The best available GPU is GPU {free_gpu}")
        # Use free_gpu for further processing or training
    else:
        print("No available GPUs found.")

def corr_function(a,b):
    corr = pd.DataFrame({'a':a, 'b':b}).corr().iloc[0,1]
    if np.isnan(corr):
        return 0
    else:
        return corr

def plot_true_pred(hr_true, hr_pred, x_lim=[20, 120], y_lim=[20, 120]):
    figure = plt.figure(figsize=(8, 8))
    hr_true, hr_pred = np.array(hr_true), np.array(hr_pred)
    mae = np.round(np.abs(hr_true - hr_pred).mean(), 2)
    correlation_coefficient = corr_function(hr_true, hr_pred)

    plt.scatter(x = hr_true, y = hr_pred, alpha=0.2, label=f"MAE: {mae:.2f}, Corr: {correlation_coefficient:.2f}")

    plt.plot(x_lim, y_lim, color='k', linestyle='-', linewidth=2)
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.xlabel('True HR (bpm)')
    plt.ylabel('Predicted HR (bpm)')
    plt.legend()
    return figure