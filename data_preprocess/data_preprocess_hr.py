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

NUM_FEATURES = 1


class data_loader_hr(Dataset):
    def __init__(self, samples, labels, domains):
        self.samples = samples
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

def load_data(data_path, split=0):

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


    return x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test 


def prep_hr(args):
    data_path = config.data_dir_Max_processed
    x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = load_data(data_path=data_path, split=args.split)
    assert x_train.shape[0] == y_train.shape[0] == d_train.shape[0]


    train_set = data_loader_hr(x_train, y_train, d_train)
    train_sampler = torch.utils.data.sampler.SequentialSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=False, sampler=train_sampler, num_workers=0)

    test_set = data_loader_hr(x_test, y_test, d_test)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    print('train_loader batch: ', len(train_loader), 'test_loader batch: ', len(test_loader))

    val_set = data_loader_hr(x_val, y_val, d_val)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    print('val_loader batch: ', len(val_loader))

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