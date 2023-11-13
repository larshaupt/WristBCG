# encoding=utf-8
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models.backbones import *
from models.loss import *
from trainer import *
import torch
import torch.nn as nn
import argparse
from datetime import datetime
import pickle
import numpy as np
import os
import logging
import sys
from data_preprocess.data_preprocess_utils import normalize
from scipy import signal
from copy import deepcopy
import wandb
from utils import tsne, mds, _logger
import config
import pdb
from tqdm import tqdm
import pandas as pd

wandb.login()


# Parse command line arguments
##################
parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID，0/1')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers for data loading')
# hyperparameter
parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=60, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_cls', type=float, default=1e-3, help='learning rate for linear classifier')

# dataset
parser.add_argument('--dataset', type=str, default='hr_max', choices=['hr_apple','hr_max'], help='name of dataset')
parser.add_argument('--n_feature', type=int, default=77, help='name of feature dimension')
#parser.add_argument('--len_sw', type=int, default=30, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=1, help='number of class')
parser.add_argument('--split', type=int, default=0, help='split number')
#parser.add_argument('--split_ratio', type=float, default=0.2, help='split ratio of test/val: train(0.64), val(0.16), test(0.2)')

# backbone model
parser.add_argument('--backbone', type=str, default='DCL', choices=['FCN', 'DCL', 'LSTM', 'AE', 'CNN_AE', 'Transformer'], help='name of framework')

# log
parser.add_argument('--logdir', type=str, default='log/', help='log directory')

# AE & CNN_AE
parser.add_argument('--lambda1', type=float, default=1.0, help='weight for reconstruction loss when backbone in [AE, CNN_AE]')





# Create directory for saving and plots
##################
global plot_dir_name
plot_dir_name = config.plot_dir

os.makedirs(plot_dir_name, exist_ok=True)

# Training function
#######################

def train(args, train_loader, val_loader, model, DEVICE, optimizer, criterion):

    min_val_loss = 1e8
    num_epochs = args.n_epoch
   

    for epoch in range(num_epochs): # loop over epochs
            
        logger.debug(f'\nEpoch : {epoch}')

        train_loss = 0
        n_batches = 0
        total = 0
        mae = 0
        mae_train = 0
        
        model.train()
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit='batch') as tepoch:
            for idx, (sample, target, domain) in enumerate(tepoch): # loop over training batches
                #pdb.set_trace()
                n_batches += 1
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).float().reshape(-1, 1)
                if args.backbone[-2:] == 'AE':
                    out, x_decoded = model(sample)
                else:
                    out, _ = model(sample)
                loss = criterion(out, target)
                if args.backbone[-2:] == 'AE':
                    # print(loss.item(), nn.MSELoss()(sample, x_decoded).item())
                    loss += nn.MSELoss()(sample, x_decoded) * args.lambda1
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                predicted = out.data
                total += target.size(0)
                mae += torch.abs(predicted - target).sum()

                tepoch.set_postfix(loss=loss.item())
        mae_train = mae / n_batches
        wandb.log({"Train Loss": train_loss / n_batches, "Train MAE": mae_train}, step=epoch)
        logger.debug(f'Train Loss     : {train_loss / n_batches:.4f}\t | \tTrain MAE     : {mae_train:2.4f}\n')
    
        if val_loader is None:
            best_model = deepcopy(model.state_dict())
            model_dir = save_dir + args.model_name + '.pt'
            print('Saving models at {} epoch to {}'.format(epoch, model_dir))
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_dir)
        else:
            with torch.no_grad():
                model.eval()
                val_loss = 0
                n_batches = 0
                total = 0
                mae_val = 0
                mae = 0
                with tqdm(val_loader, desc=f"Validation", unit='batch') as tepoch:
                    for idx, (sample, target, domain) in enumerate(tepoch):
                        n_batches += 1
                        sample, target = sample.to(DEVICE).float(), target.to(DEVICE).float().reshape(-1, 1)
                        if args.backbone[-2:] == 'AE':
                            out, x_decoded = model(sample)
                        else:
                            out, _ = model(sample)
                        loss = criterion(out, target)
                        if args.backbone[-2:] == 'AE':
                            loss += nn.MSELoss()(sample, x_decoded) * args.lambda1
                        val_loss += loss.item()
                        predicted = out.data
                        total += target.size(0)
                        mae += torch.abs(predicted - target).sum()
                        tepoch.set_postfix(val_loss=loss.item())
                        # Logs some signals for visualization to WandB
                        if idx == 0:
                            _sample = sample[0,...].cpu().numpy()
                            logging_table = pd.DataFrame({
                                                "acc_x": sample[_sample[:,0]], 
                                                "acc_y": sample[_sample[:,1]], 
                                                "acc_z": sample[_sample[:,2]], 
                                                "hr_true": target[0,...].cpu().numpy(), 
                                                "hr_pred": predicted[0,...].cpu().numpy()
                                                })
                            wandb.log({"val_signal": wandb.Table(dataframe = pd.DataFrame(logging_table))})

                mae_val = mae / n_batches
                wandb.log({"Val Loss": val_loss / n_batches, "Val MAE": mae_val}, step=epoch)
                logger.debug(f'Val Loss     : {val_loss / n_batches:.4f}\t | \tVal MAE     : {mae_val:2.4f}\n')

                if val_loss <= min_val_loss:
                    min_val_loss = val_loss
                    best_model = deepcopy(model.state_dict())
                    print('update')
                    model_dir = save_dir + args.model_name + '.pt'
                    print('Saving models at {} epoch to {}'.format(epoch, model_dir))
                    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_dir)

    return best_model

# Testing function
#######################
def test(test_loader, model, DEVICE, criterion, plt=False):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        n_batches = 0
        total = 0
        correct = 0
        feats = None
        prds = None
        trgs = None
        confusion_matrix = torch.zeros(args.n_class, args.n_class)
        for idx, (sample, target, domain) in enumerate(test_loader):
            n_batches += 1
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).float().reshape(-1, 1)
            out, features = model(sample)
            loss = criterion(out, target)
            total_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
            if prds is None:
                prds = predicted
                trgs = target
                feats = features[:, :]
            else:
                prds = torch.cat((prds, predicted))
                trgs = torch.cat((trgs, target))
                feats = torch.cat((feats, features), 0)

        acc_test = float(correct) * 100.0 / total
    wandb.log({"Test Loss": total_loss / n_batches, "Test Acc": acc_test})

    logger.debug(f'Test Loss     : {total_loss / n_batches:.4f}\t | \tTest Accuracy     : {acc_test:2.4f}\n')
    for t, p in zip(trgs.view(-1), prds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    logger.debug(confusion_matrix)
    logger.debug(confusion_matrix.diag() / confusion_matrix.sum(1))
    wandb.log({"conf_mat": confusion_matrix})

    if plt == True:
        tsne(feats, trgs, domain=None, save_dir=plot_dir_name + args.model_name + '_tsne.png')
        mds(feats, trgs, domain=None, save_dir=plot_dir_name + args.model_name + 'mds.png')
        sns_plot = sns.heatmap(confusion_matrix, cmap='Blues', annot=True)
        sns_plot.get_figure().savefig(plot_dir_name + args.model_name + '_confmatrix.png')
    return total_loss

# Main function
#######################
if __name__ == '__main__':
    torch.manual_seed(10)
    np.random.seed(10)
    args = parser.parse_args()

    # Initialize W&B
    run = wandb.init(
    # Set the project where this run will be logged
    project="hr_ssl",
    # Track hyperparameters and run metadata
    config=vars(args))

    # Set device
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE, 'dataset:', args.dataset)

    # Load data
    train_loaders, val_loader, test_loader = setup_dataloaders(args)

    # Initialize model
    if args.backbone == 'FCN':
        model = FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False)
    elif args.backbone == 'DCL':
        model = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=False)
    elif args.backbone == 'LSTM':
        model = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=False)
    elif args.backbone == 'AE':
        model = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=False)
    elif args.backbone == 'CNN_AE':
        model = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=False)
    elif args.backbone == 'Transformer':
        model = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
    else:
        NotImplementedError

    model = model.to(DEVICE)

    # Set model name
    args.model_name = args.backbone + '_'+args.dataset + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) + '_sw' + str(args.len_sw)

    # Create directory for results
    save_dir = 'results/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize logger
    if os.path.isdir(args.logdir) == False:
        os.makedirs(args.logdir)
    log_file_name = os.path.join(args.logdir, args.model_name + f".log")
    logger = _logger(log_file_name)
    logger.debug(args)

    # Loss function for regression
    criterion = nn.MSELoss()

    # Initialize optimizer
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, args.lr)

    # Training
    #######################
    training_start = datetime.now()
    train_loss_list = []
    test_loss_list = []

    best_model = train(args, train_loaders, val_loader, model, DEVICE, optimizer, criterion)

    if args.backbone == 'FCN':
        model_test = FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False)
    elif args.backbone == 'DCL':
        model_test = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=False)
    elif args.backbone == 'LSTM':
        model_test = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=False)
    elif args.backbone == 'AE':
        model_test = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=False)
    elif args.backbone == 'CNN_AE':
        model_test = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=False)
    elif args.backbone == 'Transformer':
        model_test = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
    else:
        NotImplementedError

    # Testing
    model_test.load_state_dict(best_model)
    model_test = model_test.to(DEVICE)
    test_loss = test(test_loader, model_test, DEVICE, criterion, plt=False)
    test_loss_list.append(test_loss)

    training_end = datetime.now()
    training_time = training_end - training_start
    logger.debug(f"Training time is : {training_time}")
