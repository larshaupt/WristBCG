'''
Data Pre-processing on HHAR dataset.

'''
import sys
import numpy as np
import pandas as pd
import sys
import os
from scipy.interpolate import interp1d
import torch
import pickle as cp
from torch.utils.data import Dataset, DataLoader
import config
import re
import json

NUM_FEATURES = 1



class data_loader_hr(Dataset):
    def __init__(self, samples, labels, domains, hr_norm=True, hr_min=20, hr_max=120):

        self.samples = samples
        self.hr_norm = hr_norm
        self.hr_max = hr_max
        self.hr_min = hr_min
        if self.hr_norm:
            self.labels = (labels - self.hr_min) / (self.hr_max - self.hr_min)
        else:
            self.labels = labels
        self.domains = domains
        



    def __getitem__(self, index):
        #print('index: ', index)
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain
    def __len__(self):
        return len(self.samples)


def downsampling(data_t, data_x, data_y, freq):
    """Recordings are downsamplied to 50Hz
    
    :param data_t: numpy integer array
        time array
    :param data_x: numpy integer array
        sensor recordings
    :param data_y: numpy integer array
        labels
    :return: numpy integer array
        Downsampled input
    """
    idx = np.arange(0, data_t.shape[0], int(freq/50))

    return data_t[idx], data_x[idx], data_y[idx]

def preprocess():
    print('preprocessing the data...')
    print('preprocessing done!')

def load_data(data_path, split=0, from_split_json=False):


    def load_pickle(path):
        with open(path, 'rb') as f:
            x,y,pid = cp.load(f)
        return x,y,pid
    
    def load_split(files, path):
        x, y, pid = [], [], []
        for file in files:
            x_, y_, pid_ = load_pickle(os.path.join(path, file))
            x.append(x_)
            y.append(y_)
            pid.append(pid_)
        x = np.concatenate(x)
        y = np.concatenate(y)
        pid = np.concatenate(pid)
        return x,y,pid

    if from_split_json:
        split_path = os.path.join(data_path, "splits.json")
        with open(split_path, 'r') as f:
            splits = json.load(f)
        split_files = splits[str(split)]

        train = split_files["train"]
        test = split_files["test"]
        val = split_files["val"]

        x_train, y_train, d_train = load_split(train, data_path)
        x_val, y_val, d_val = load_split(val, data_path)
        x_test, y_test, d_test = load_split(test, data_path)
    else:

        split_path = os.path.join(data_path, 'split_'+str(split))
        with open(os.path.join(split_path, 'train.pickle'), 'rb') as f:
            train = cp.load(f)
        with open(os.path.join(split_path, 'test.pickle'), 'rb') as f:
            test = cp.load(f)
        with open(os.path.join(split_path, 'val.pickle'), 'rb') as f:
            val = cp.load(f)

        (x_train, y_train, d_train) = train
        (x_val, y_val, d_val) = val
        (x_test, y_test, d_test) = test

    def transform_domain_to_float(domain):
        domain = np.array([float(re.sub("[^0-9.]", "", el)) if isinstance(el, str) else el for el in domain])
        return domain
    
    #d_train = np.array([transform_domain_to_float(el) for el in d_train])
    #d_val = np.array([transform_domain_to_float(el) for el in d_val])
    #d_test = np.array([transform_domain_to_float(el) for el in d_test])
    d_train = d_train.astype(float)
    d_val = d_val.astype(float)
    d_test = d_test.astype(float)

    return x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test 

def load_data_no_labels(data_path, split:int=0, from_split_json=False):
    # Load data without labels
    # Used for capture24 dataset
    # Substitutes labels with zeros


    def load_pickle(path):
        with open(path, 'rb') as f:
            x,pid = cp.load(f)
        return x,pid
    
    def load_split(files, path):
        x,  pid = [], []
        for file in files:
            x_, pid_ = load_pickle(os.path.join(path, file))
            if len(x_) == 0:
                continue
            x.append(x_)
            pid.append(pid_)
        x = np.concatenate(x)
        pid = np.concatenate(pid)
        return x,pid

    if from_split_json:
        split_path = os.path.join(data_path, "splits.json")
        with open(split_path, 'r') as f:
            splits = json.load(f)
        split = splits[str(split)]

        train = split["train"]
        test = split["test"]
        val = split["val"]

        x_train, d_train = load_split(train, data_path)
        x_val, d_val = load_split(val, data_path)
        x_test, d_test = load_split(test, data_path)
                

    else:
        split_path = os.path.join(data_path, 'split_'+str(split))
        with open(os.path.join(split_path, 'train.pickle'), 'rb') as f:
            train = cp.load(f)
        with open(os.path.join(split_path, 'test.pickle'), 'rb') as f:
            test = cp.load(f)
        with open(os.path.join(split_path, 'val.pickle'), 'rb') as f:
            val = cp.load(f)

        (x_train, d_train) = train
        (x_val, d_val) = val
        (x_test, d_test) = test

    y_train = np.ones((x_train.shape[0],1))
    y_val = np.ones((x_val.shape[0],1))
    y_test = np.ones((x_test.shape[0],1))

    def transform_domain_to_float(domain):
        domain = np.array([float(re.sub("[^0-9.]", "", el)) if isinstance(el, str) else el for el in domain])
        return domain
    
    d_train = np.array([transform_domain_to_float(el) for el in d_train])
    d_val = np.array([transform_domain_to_float(el) for el in d_val])
    d_test = np.array([transform_domain_to_float(el) for el in d_test])

    return x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test 

def prep_hr(args, dataset=None, split=None):
    if dataset is None:
        dataset = args.dataset
    if split is None:
        split = args.split

    if dataset == 'max':
        data_path = config.data_dir_Max_processed
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split)
    elif dataset == 'apple':
        data_path = config.data_dir_Apple_processed
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split)
    elif dataset == 'm2sleep':
        data_path = config.data_dir_M2Sleep_processed
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=split,from_split_json=True)
    elif dataset == 'capture24':
        data_path = config.data_dir_Capture24_processed
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data_no_labels(data_path=data_path, split=split, from_split_json=True)
    
    assert x_train.shape[0] == y_train.shape[0] == d_train.shape[0]


    train_set = data_loader_hr(x_train, y_train, d_train, hr_norm=True, hr_min=args.hr_min, hr_max=args.hr_max)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers, pin_memory=True, shuffle=True)

    test_set = data_loader_hr(x_test, y_test, d_test, hr_norm=True, hr_min=args.hr_min, hr_max=args.hr_max)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    val_set = data_loader_hr(x_val, y_val, d_val, hr_norm=True, hr_min=args.hr_min, hr_max=args.hr_max)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")

    return train_loader, val_loader, test_loader

def test_prep_hr():
    # Define mock arguments
    class Args:
        data_path = config.data_dir_Max_processed
        split = 0
        batch_size = 32  # Adjust the batch size as needed

    # Call the prep_hr function with mock arguments
    train_loader, val_loader, test_loader = prep_hr(Args)

    # Print information about the loaded data
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")

    # Print the first batch in each loader for inspection
    print("\nFirst batch in train_loader:")
    for i, (samples, labels, domains) in enumerate(train_loader):
        print(f"Sample shape: {samples.shape}, Label shape: {labels.shape}, Domain shape: {domains.shape}")
        break  # Print only the first batch

    print("\nFirst batch in val_loader:")
    for i, (samples, labels, domains) in enumerate(val_loader):
        print(f"Sample shape: {samples.shape}, Label shape: {labels.shape}, Domain shape: {domains.shape}")
        break  # Print only the first batch

    print("\nFirst batch in test_loader:")
    for i, (samples, labels, domains) in enumerate(test_loader):
        print(f"Sample shape: {samples.shape}, Label shape: {labels.shape}, Domain shape: {domains.shape}")
        break  # Print only the first batch

    print("Testing prep_hr complete.")

# Call the test function
#test_prep_hr()