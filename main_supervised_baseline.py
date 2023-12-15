# encoding=utf-8
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import seaborn as sns
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
from scipy import signal
from copy import deepcopy
import wandb
from utils import tsne, mds, _logger, get_free_gpu
from config import results_dir, plot_dir, data_dir_Max_processed, data_dir_Apple_processed
from tqdm import tqdm
import pandas as pd
import json
from torchmetrics.regression import LogCoshError

wandb.login()


# Parse command line arguments
##################
parser = argparse.ArgumentParser(description='argument setting of network')

# arguments for data preprocessing
parser.add_argument("--json_load", type=str, default="", help="json file for loading data")
parser.add_argument("--test_only", action="store_true", help="only test the model")
parser.add_argument("--no_saving", action="store_true", help="do not save the model")

# arguments for training process
parser.add_argument('--cuda', default=-1, type=int, help='cuda device ID，0/1')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers for data loading')

# hyperparameter
parser.add_argument('--batch_size', type=int, default=128, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=60, help='number of training epochs')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-7, help='weight decay')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'RMSprop'], help='optimizer')
parser.add_argument('--loss', type=str, default='MAE', choices=['MSE', 'MAE', 'Huber', 'LogCosh'], help='loss function')
parser.add_argument('--huber_delta', type=float, default=0.1, help='delta for Huber loss')
parser.add_argument('--num_ensemble', type=int, default=1, help='number of ensemble models')

# dataset
parser.add_argument('--dataset', type=str.lower, default='max', choices=['apple','max', 'm2sleep', "m2sleep100", 'capture24', 'apple100'], help='name of dataset')
parser.add_argument('--normalize', type=bool, default=True, help='if or not to normalize data')
parser.add_argument('--n_feature', type=int, default=3, help='name of feature dimension')
parser.add_argument('--n_class', type=int, default=1, help='number of class')
parser.add_argument('--split', type=int, default=0, help='split number')
parser.add_argument('--hr_min', type=float, default=20, help='minimum heart rate for training')
parser.add_argument('--hr_max', type=float, default=120, help='maximum heart rate for training')
parser.add_argument('--sampling_rate', type=int, default=0, help='sampling rate of the data. Warning: this will take longer time to load data')

# backbone model
parser.add_argument('--backbone', type=str, default='CorNET', choices=['FCN', 'DCL', 'LSTM', 'AE', 'CNN_AE', 'Transformer', 'CorNET', 'TCN'], help='name of framework')
parser.add_argument('--num_kernels', type=int, default=16, help='number of kernels in CNN')
parser.add_argument('--kernel_size', type=int, default=16, help='kernel size in CNN')
parser.add_argument('--lstm_units', type=int, default=192, help='number of units in LSTM')

# log
parser.add_argument('--logdir', type=str, default='log/', help='log directory')
parser.add_argument('--wandb_mode', type=str, default='online', choices=['offline', 'online', 'disabled', 'dryrun', 'run'],  help='wandb mode')

# AE & CNN_AE
parser.add_argument('--lambda1', type=float, default=1.0, help='weight for reconstruction loss when backbone in [AE, CNN_AE]')


corr = lambda a, b: pd.DataFrame({'a':a, 'b':b}).corr().iloc[0,1]

# Create directory for saving and plots
##################
global plot_dir_name
plot_dir_name = plot_dir

os.makedirs(plot_dir_name, exist_ok=True)

# Training function
#######################

def train(args, train_loader, val_loader, model, DEVICE, optimizer, criterion, wandb_run):

    min_val_loss = 1e8
    num_epochs = args.n_epoch
   

    for epoch in range(num_epochs): # loop over epochs
            

        train_loss = 0
        n_batches = 0
        total = 0
        mae = 0
        hr_true, hr_pred = [], []

        model.train()
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit='batch') as tepoch:
            for idx, (sample, target, domain) in enumerate(tepoch): # loop over training batches
                
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
                mae += torch.abs(predicted - target).mean() * (args.hr_max - args.hr_min)
                hr_true.extend(target.cpu().numpy().squeeze() * (args.hr_max - args.hr_min) + args.hr_min)
                hr_pred.extend(predicted.cpu().numpy().squeeze() * (args.hr_max - args.hr_min) + args.hr_min)

                tepoch.set_postfix(loss=loss.item())
        mae_train = mae / n_batches
        corr_train = corr(hr_true, hr_pred)
        loss_train = train_loss / n_batches
        wandb_run.log({"Train_Loss": loss_train, "Train_MAE": mae_train, 'Train_Corr': corr_train}, step=epoch)
        logger.debug(f'Train Loss     : {loss_train:.4f}\t | \tTrain MAE     : {mae_train:2.4f}\t | \tTrain Corr     : {corr_train:2.4f}\n')
    
    
        # Validation
        model_dir = os.path.join(args.model_dir_name, args.model_name + '_model.pt')

        if val_loader is None:
            best_model = deepcopy(model.state_dict())
            print('Saving models at {} epoch to {}'.format(epoch, model_dir))
            if args.no_saving == False:
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_dir)
        else:
            with torch.no_grad():
                model.eval()
                val_loss = 0
                n_batches = 0
                total = 0
                mae_val = 0
                mae = 0
                hr_true, hr_pred, pids = [], [], []
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
                        mae += torch.abs(predicted - target).mean() * (args.hr_max - args.hr_min)
                        tepoch.set_postfix(val_loss=loss.item())
                        # Logs some signals for visualization to WandB
                        hr_true.extend(target.cpu().numpy().squeeze() * (args.hr_max - args.hr_min) + args.hr_min)
                        hr_pred.extend(predicted.cpu().numpy().squeeze() * (args.hr_max - args.hr_min) + args.hr_min)
                        pids.extend(domain.cpu().numpy().squeeze())

                logging_table = pd.DataFrame({ 
                                    "hr_true": hr_true, 
                                    "hr_pred": hr_pred,
                                    "pid": pids
                                    })
                
                #wandb_run.log({"hr_true_vs_pred_val": wandb.Table(dataframe = pd.DataFrame(logging_table))}, step=epoch)
                figure = plot_true_pred(hr_true, hr_pred)
                wandb_run.log({"true_pred_val": figure}, step=epoch)

                mae_val = mae / n_batches
                corr_val = corr(hr_true, hr_pred)
                wandb_run.log({"Val_Loss": val_loss / n_batches, "Val_MAE": mae_val, 'Val_Corr': corr_val}, step=epoch)
                logger.debug(f'Val Loss     : {val_loss / n_batches:.4f}\t | \tVal MAE     : {mae_val:2.4f}\t | \tVal Corr     : {corr_val:.4f}\n \n')

                if val_loss <= min_val_loss:
                    min_val_loss = val_loss
                    best_model = deepcopy(model.state_dict())
                    print('Saving models and results at {} epoch to {}'.format(epoch, model_dir))
                    if args.no_saving == False:  
                        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_dir)
                    logging_table.to_csv(os.path.join(args.model_dir_name, args.model_name + '_predictions_val.csv'), index=False)

    return best_model

# Testing function
#######################
def test(test_loader, model, DEVICE, criterion, plt=False):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        n_batches = 0
        mae = 0
        feats = None
        hr_true, hr_pred, feats, pids = [], [], [], []

        with tqdm(test_loader, desc=f"Test", unit='batch') as tepoch:
            for idx, (sample, target, domain) in enumerate(tepoch):
                n_batches += 1
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).float().reshape(-1, 1)
                out, features = model(sample)
                loss = criterion(out, target)
                total_loss += loss.item()
                predicted = out.data
                mae += torch.abs(predicted - target).mean() * (args.hr_max - args.hr_min)
                tepoch.set_postfix(test_loss=loss.item())
                hr_true.extend(target.cpu().numpy().squeeze() * (args.hr_max - args.hr_min) + args.hr_min)
                hr_pred.extend(predicted.cpu().numpy().squeeze() * (args.hr_max - args.hr_min) + args.hr_min)
                feats.append(features.cpu().numpy().squeeze())
                pids.extend(domain.cpu().numpy().squeeze())

            
    logging_table = pd.DataFrame({ 
                        "hr_true": hr_true, 
                        "hr_pred": hr_pred,
                        "pid": pids
                        })
    #wandb_run.log({"hr_true_vs_pred_test": wandb.Table(dataframe = pd.DataFrame(logging_table))})

    mae_test = mae / n_batches
    corr_test = corr(hr_true, hr_pred)
    loss_test = total_loss / n_batches

    figure = plot_true_pred(hr_true, hr_pred)
    wandb.log({"true_pred_test": figure})
    wandb.log({"Test_Loss": loss_test, "Test_MAE": mae_test, "Test_Corr": corr_test})


    logger.debug(f'Test Loss     : {loss_test:.4f}\t | \tTest MAE     : {mae_test:2.4f} \t | \tTest Corr     : {corr_test:.4f}\n')

    if plt == True:
        tsne(feats, hr_true, domain=None, save_dir=plot_dir_name + args.model_name + '_tsne.png')
        mds(feats, hr_true, domain=None, save_dir=plot_dir_name + args.model_name + 'mds.png')
    return loss_test

def plot_true_pred(hr_true, hr_pred, x_lim=[20, 120], y_lim=[20, 120]):
    figure = plt.figure(figsize=(8, 8))
    hr_true, hr_pred = np.array(hr_true), np.array(hr_pred)
    mae = np.round(np.abs(hr_true - hr_pred).mean(), 2)
    correlation_coefficient = corr(hr_true, hr_pred)

    plt.scatter(x = hr_true, y = hr_pred, alpha=0.2, label=f"MAE: {mae:.2f}, Corr: {correlation_coefficient:.2f}")

    plt.plot(x_lim, y_lim, color='k', linestyle='-', linewidth=2)
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.xlabel('True HR (bpm)')
    plt.ylabel('Predicted HR (bpm)')
    plt.legend()
    
    return figure
    



# Main function
#######################
if __name__ == '__main__':
    torch.manual_seed(10)
    np.random.seed(10)
    
    args = parser.parse_args()

    # Load args from json if json is passed
    if args.json_load.endswith(".json") and os.path.isfile(args.json_load):
        with open(args.json_load, "r") as f:
            args.__dict__.update(json.load(f))
        print(f"Loaded args from {args.json_load}")
        args.from_json = True
        args.wandb_mode = "disabled"
    else:
        args.from_json = False


    # Set device
    if args.cuda == -1:
        args.cuda = int(get_free_gpu())
        print(f"Automatically selected GPU {args.cuda}")

    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE, 'dataset:', args.dataset)

    # Load data
    train_loaders, val_loader, test_loader = setup_dataloaders(args)

    # Set model name
    # Creates directory for saving results from models from the same split
    model_name = args.backbone + '_'+args.dataset + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) 
    # Create directory for results
    model_time = int(datetime.now().timestamp())
    #args.model_dir_name = os.path.join(results_dir, f"{model_time}_{model_name}")
    args.model_dir_name = os.path.join(results_dir, f"{model_name}")

    # Initialize logger
    if os.path.isdir(args.logdir) == False:
        os.makedirs(args.logdir)

    
    if args.no_saving == False:
        os.makedirs(args.model_dir_name, exist_ok=True)

    
    with open(os.path.join(args.model_dir_name, "config.json"), "w") as outfile:
        print(f"Saving config file to {args.model_dir_name}")
        json.dump(vars(args), outfile)


    group_id = wandb.util.generate_id()

    for i_ensemble in range(args.num_ensemble):

        # Initialize W&B

        config_dict = vars(args)
        config_dict.update({"i_ensemble": i_ensemble})

        wandb_run = wandb.init(
        resume=None,
        group=group_id,
        job_type = "train",
        # Set the project where this run will be logged
        project="hr_ssl",
        reinit = True,
        # Track hyperparameters and run metadata
        config=config_dict,
        mode=args.wandb_mode,
        )


        # Initialize model
        if args.backbone == 'FCN':
            model = FCN(n_channels=args.n_feature, n_classes=args.n_class, input_size=args.input_length, backbone=False)
        elif args.backbone == 'DCL':
            model = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, input_size=args.input_length, conv_kernels=64, kernel_size=5, LSTM_units=args.lstm_units, backbone=False)
        elif args.backbone == 'LSTM':
            model = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=args.lstm_units, backbone=False)
        elif args.backbone == 'AE':
            model = AE(n_channels=args.n_feature, input_size=args.input_length, n_classes=args.n_class, outdim=128, backbone=False)
        elif args.backbone == 'CNN_AE':
            model = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, input_size=args.input_length, backbone=False)
        elif args.backbone == 'Transformer':
            model = Transformer(n_channels=args.n_feature, input_size=args.input_length, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
        elif args.backbone == "CorNET":
            model = CorNET(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=args.num_kernels, kernel_size=args.kernel_size, LSTM_units=args.lstm_units, backbone=False, input_size=args.input_length)
        elif args.backbone == "TCN":
            model = TemporalConvNet(num_channels=[32, 64, 128], n_classes=args.n_class,  num_inputs=args.n_feature, input_length = args.input_length, kernel_size=16, dropout=0.2, backbone=False)
        else:
            NotImplementedError

        total_params = sum(
            param.numel() for param in model.parameters()
        )
        wandb.watch(model, log='all')

        wandb_run.log({"Total_Params": total_params}, step=0)

        model = model.to(DEVICE)
        
        # saves same models with different splits in the same directory
        args.model_name = model_name + '_split' + str(args.split) 
        if args.num_ensemble != 1:
            args.model_name += '_ensemble' + str(args.num_ensemble)

        
        log_file_name = os.path.join(args.logdir, args.model_dir_name + f".log")
        logger = _logger(log_file_name)
        logger.debug(args)



        # Loss function for regression
        if args.loss == 'MSE':
            criterion = nn.MSELoss()
        elif args.loss == 'MAE':
            criterion = nn.L1Loss()
        elif args.loss == 'Huber':
            criterion = nn.HuberLoss(delta=args.huber_delta)
        elif args.loss == 'LogCosh':
            criterion = LogCoshError()
            criterion = criterion.to(DEVICE)


        # Initialize optimizer
        parameters = model.parameters()
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(parameters, args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(parameters, args.lr, momentum=0.9, weight_decay=args.weight_decay)
        elif args.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(parameters, args.lr, momentum=0.9, weight_decay=args.weight_decay)
        else:
            NotImplementedError
    
        # Training
        #######################
        training_start = datetime.now()
        test_loss_list = []

        best_model = train(args, train_loaders, val_loader, model, DEVICE, optimizer, criterion, wandb_run=wandb_run)

        if args.num_ensemble != 1:
            wandb_run.finish()

    # Testing
    #######################
    # Initialize and load test model
    if args.backbone == 'FCN':
        model_test = FCN(n_channels=args.n_feature, n_classes=args.n_class, input_size=args.input_length, backbone=False)
    elif args.backbone == 'DCL':
        model_test = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=args.lstm_units, input_size=args.input_length, backbone=False)
    elif args.backbone == 'LSTM':
        model_test = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=args.lstm_units, backbone=False)
    elif args.backbone == 'AE':
        model_test = AE(n_channels=args.n_feature, input_size=args.input_length, n_classes=args.n_class, outdim=128, backbone=False)
    elif args.backbone == 'CNN_AE':
        model_test = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=False)
    elif args.backbone == 'Transformer':
        model_test = Transformer(n_channels=args.n_feature, input_size=args.input_length, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
    elif args.backbone == "CorNET":
        model_test = CorNET(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=args.num_kernels, kernel_size=args.kernel_size, LSTM_units=args.lstm_units, backbone=False, input_size=args.input_length)
    elif args.backbone == "TCN":
        model = TemporalConvNet(num_channels=[32, 64, 128], n_classes=args.n_class,  num_inputs=args.n_feature, input_length = args.input_length, kernel_size=2, dropout=0.2, backbone=False)
    else:
        NotImplementedError

    if args.num_ensemble != 1:
        wandb_run = wandb.init(
            resume="allow",
            group=group_id,
            job_type = "test",
            # Set the project where this run will be logged
            project="hr_ssl",
            reinit = False,
            # Track hyperparameters and run metadata
            config=config_dict,
            mode=args.wandb_mode,
            )

    # Testing
    if len(test_loader) != 0:
        model_test.load_state_dict(best_model)
        model_test = model_test.to(DEVICE)
        test_loss = test(test_loader, model_test, DEVICE, criterion, plt=False)
        test_loss_list.append(test_loss)
    else:
        print('No test data. Skip testing...')

    training_end = datetime.now()
    training_time = training_end - training_start
    logger.debug(f"Training time is : {training_time}")
    
    wandb_run.finish()
