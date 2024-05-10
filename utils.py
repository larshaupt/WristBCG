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

def plot_true_pred(hr_true, hr_pred, signal_std=[], signal_threshold=0.05, ax=None, title='', hr_lims = [40,120], **kwargs):

    """
    Plot the true vs. predicted heart rate values with optional data filtering and statistics.

    Parameters:
        hr_pred (array-like): Predicted heart rate values.
        hr_true (array-like): True heart rate values.
        signal_std (array-like): Signal standard deviation values for data filtering.
        signal_threshold (float, optional): Threshold for signal filtering. Default is 0.05.
        ax (matplotlib Axes, optional): The Axes to draw the plot on. If not provided, a new figure is created.
        title (str, optional): Title for the plot.

    This function creates a scatter plot of true vs. predicted heart rate values, with points
    categorized based on signal standard deviation. It calculates and displays the Mean Absolute
    Error (MAE) and the correlation coefficient for the entire dataset and the low-signal subset.

    Returns:
        None

    Example:
    >>> plot_true_pred(hr_pred, hr_true, signal_std, signal_threshold=0.1, title='Heart Rate Prediction')
    """

    if ax == None:
        fig, ax = plt.subplots()

    split_by_std = len(signal_std) != 0

    if split_by_std:
        thr_h = signal_std > signal_std.max()*signal_threshold
        num_low = (thr_h).sum()/signal_std.count() * 100
    else:
        thr_h = np.array([False] * hr_pred, dtype='bool')
        num_low = len(hr_pred)

    
    hr_true_h = hr_true[thr_h]
    hr_pred_h = hr_pred[thr_h]
    hr_true_l = hr_true[~thr_h]
    hr_pred_l = hr_pred[~thr_h]


    h_args = {'x': hr_true_l, 'y': hr_pred_l, 'alpha': 0.2, 'label': f'std > {signal_threshold} std_max ({np.round(num_low, 2)}%))'}
    l_args = {'x': hr_true_h, 'y': hr_pred_h, 'alpha': 0.2, 'label': f'std < {signal_threshold} std_max ({np.round(100-num_low, 2)}%))'}
    # This allows us to pass some extra plot arguments into the function. Passed arguments overwrite the default ones
    h_args.update(kwargs)
    l_args.update(kwargs)

    ax.scatter(**h_args)
    ax.scatter(**l_args)

    ax.plot(hr_lims, [40, 120], color='k', linestyle='-', linewidth=2)
    ax.set_xlabel('True HR (bpm)')
    ax.set_ylabel('Predicted HR (bpm)')
    ax.set_title(title)
    #ax.set_ylim([25, 110])
    #ax.set_xlim([35, 85])
    if split_by_std:
        ax.legend(loc='upper right')

    
    mae = np.round(np.abs(hr_true - hr_pred).mean(), 2)
    mae_l = np.round(np.abs(hr_true_l - hr_pred_l).mean(), 2)
    correlation_coefficient = np.round(np.corrcoef(hr_true, hr_pred)[0, 1],3)
    correlation_coefficient_l = np.round(np.corrcoef(hr_true_l, hr_pred_l)[0, 1],3)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    if split_by_std:
        textstr = f'MAE = {mae} \ncorr = {correlation_coefficient} \nMAE_low = {mae_l} \ncorr_low = {correlation_coefficient_l}'
    else:
        textstr = f'MAE = {mae} \ncorr = {correlation_coefficient}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)
