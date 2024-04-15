import torch
import torch.nn as nn
import numpy as np
import os
import pickle as cp
from augmentations import gen_aug
from utils import tsne, mds, _logger
import time
from models.frameworks import *

from models.backbones import *
from models.loss import *
from models.postprocessing import Postprocessing, BeliefPPG, KalmanSmoothing
from data_preprocess import data_preprocess_hr
from torchmetrics.regression import LogCoshError
from bayesian_torch.models.dnn_to_bnn import get_kl_loss

from sklearn.metrics import f1_score
import seaborn as sns
import wandb
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import config
import pdb
import matplotlib.pyplot as plt

# create directory for saving models and plots
global model_dir_name
model_dir_name = config.results_dir

global plot_dir_name
plot_dir_name = 'plot'
if not os.path.exists(plot_dir_name):
    os.makedirs(plot_dir_name)

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
    

def setup_dataloaders(args, mode="finetuning"):


    if mode == "pretraining":
        dataset = args.pretrain_dataset
        split = 0 # pretrains network always with split 0
        subsample_rate = args.pretrain_subsample

        if args.framework == "reconstruction":
            reconstruction = True
        else:
            reconstruction = False
        sample_sequences = False
        discrete_hr = False
        sigma = args.label_sigma

    elif mode =="finetuning": # normal dataset, not pretraining
        dataset = args.dataset
        split = args.split
        subsample_rate = args.subsample
        reconstruction = False
        sample_sequences = False
        sigma = args.label_sigma
        discrete_hr = args.discretize_hr

    elif mode == "postprocessing":
        dataset = args.dataset
        split = args.split
        subsample_rate = args.subsample
        reconstruction = False
        sample_sequences = True
        sigma = args.label_sigma
        discrete_hr = True
    elif mode == "postprocessing_no_discrete":
        dataset = args.dataset
        split = args.split
        subsample_rate = args.subsample
        reconstruction = False
        sample_sequences = True
        sigma = args.label_sigma
        discrete_hr = False
    else:
        raise ValueError(f"Mode {mode} not recognized")


    train_loader, val_loader, test_loader = data_preprocess_hr.prep_hr(args, dataset=dataset, split=split, subsample_rate=subsample_rate, reconstruction=reconstruction, sample_sequences=sample_sequences, discrete_hr=discrete_hr, sigma=sigma)
    

    return train_loader, val_loader, test_loader


def setup_linclf(args, DEVICE, bb_dim):
    '''
    @param bb_dim: output dimension of the backbone network
    @return: a linear classifier
    '''

    if args.backbone in ['CNN_AE'] and args.framework == 'reconstruction':
        classifier = LSTM_Classifier(bb_dim=bb_dim, n_classes=args.n_class, rnn_type=args.rnn_type, num_layers=args.num_layers_classifier)
    else:
        if args.model_uncertainty == "NLE":
            classifier = Classifier_with_uncertainty(bb_dim=bb_dim, n_classes=args.n_class, num_layers=args.num_layers_classifier)
        else:
            classifier = Classifier(bb_dim=bb_dim, n_classes=args.n_class, num_layers=args.num_layers_classifier)

    classifier = classifier.to(DEVICE)
    return classifier


def setup_model_optm(args, DEVICE):

    # set up backbone network
    if "bnn" in args.model_uncertainty:

        if args.backbone == "CorNET":

            bayesian_layers = "first_last" if "firstlast" in args.model_uncertainty else "all"

            # loading pretrained model
            if "pretrained" in args.model_uncertainty:

                pretrained_model_dir = args.lincl_model_file.replace("_bnn", "").replace("_pretrained", ""). replace("_firstlast", "")
                if os.path.isfile(pretrained_model_dir):
                    #breakpoint()
                    state_dict = torch.load(pretrained_model_dir)["trained_backbone"]
                    print("Loading backbone from {}".format(pretrained_model_dir))
                else:
                    raise ValueError("No pretrained model found at {}".format(pretrained_model_dir))
                backbone = BayesianCorNET(n_channels=args.n_feature, 
                                          n_classes=args.n_class, 
                                          conv_kernels=args.num_kernels, 
                                          kernel_size=args.kernel_size, 
                                          LSTM_units=args.lstm_units, 
                                          backbone=True, 
                                          input_size=args.input_length, 
                                          state_dict=state_dict, 
                                          bayesian_layers=bayesian_layers, 
                                          dropout=args.dropout_rate,
                                          rnn_type=args.rnn_type)
            
            else: # no pretrained model
                backbone = BayesianCorNET(n_channels=args.n_feature, 
                                          n_classes=args.n_class, 
                                          conv_kernels=args.num_kernels, 
                                          kernel_size=args.kernel_size, 
                                          LSTM_units=args.lstm_units, 
                                          backbone=True, 
                                          input_size=args.input_length, 
                                          bayesian_layers=bayesian_layers, 
                                          dropout=args.dropout_rate,
                                          rnn_type=args.rnn_type)
        else: # BNN, different architecure
            raise NotImplementedError
    else: # no BNN
        if args.backbone == 'FCN':
            backbone = FCN(n_channels=args.n_feature, 
                           n_classes=args.n_class, 
                           conv_kernels=args.num_kernels, 
                           kernel_size=args.kernel_size, 
                           input_size=args.input_length, 
                           backbone=True)
        elif args.backbone == 'DCL':
            backbone = DeepConvLSTM(n_channels=args.n_feature, 
                                    n_classes=args.n_class, 
                                    conv_kernels=args.num_kernels, 
                                    kernel_size=args.kernel_size, 
                                    input_size=args.input_length, 
                                    LSTM_units=args.lstm_units, 
                                    rnn_type = args.rnn_type, 
                                    backbone=True)
        elif args.backbone == 'LSTM':
            backbone = LSTM(n_channels=args.n_feature, 
                            n_classes=args.n_class, 
                            LSTM_units=args.lstm_units, 
                            rnn_type = args.rnn_type, 
                            backbone=True)
        elif args.backbone == 'AE':
            backbone = AE(n_channels=args.n_feature, 
                          input_size=args.input_length, 
                          n_classes=args.n_class, 
                          embdedded_size=128, 
                          backbone=True, 
                          n_channels_out=args.n_channels_out)
        elif args.backbone == 'CNN_AE':
            backbone = CNN_AE(n_channels=args.n_feature, 
                              n_classes=args.n_class, 
                              input_size=args.input_length, 
                              backbone=True, 
                              n_channels_out=args.n_channels_out, 
                              dropout=args.dropout_rate, 
                              num_layers=3,
                              pool_kernel_size=2,
                              kernel_size=args.kernel_size,
                              conv_kernels=args.num_kernels)
        elif args.backbone == 'Transformer':
            backbone = Transformer(n_channels=args.n_feature, 
                                   input_size=args.input_length, 
                                   n_classes=args.n_class, 
                                   dim=128, 
                                   depth=4, 
                                   heads=4, 
                                   mlp_dim=64, 
                                   dropout=0.1, 
                                   backbone=True)
        elif args.backbone == "CorNET":
            if args.add_frequency:
                backbone = CorNETFrequency(n_channels=args.n_feature, 
                                           n_classes=args.n_class, 
                                           conv_kernels=args.num_kernels, 
                                           kernel_size=args.kernel_size, 
                                           LSTM_units=args.lstm_units,
                                           backbone=True, 
                                           input_size=args.input_length, 
                                           num_extra_features=args.n_feature)
            else:
                backbone = CorNET(n_channels=args.n_feature, 
                                  n_classes=args.n_class, 
                                  conv_kernels=args.num_kernels, 
                                  kernel_size=args.kernel_size, 
                                  LSTM_units=args.lstm_units, 
                                  backbone=True, 
                                  input_size=args.input_length, 
                                  rnn_type=args.rnn_type,
                                  dropout_rate=args.dropout_rate)
        elif args.backbone == "AttentionCorNET":
            backbone = AttentionCorNET(n_channels=args.n_feature, 
                                       n_classes=args.n_class, 
                                       conv_kernels=args.num_kernels, 
                                       kernel_size=args.kernel_size, 
                                       LSTM_units=args.lstm_units, 
                                       backbone=True, 
                                       input_size=args.input_length, 
                                       rnn_type=args.rnn_type)
        elif args.backbone == "TCN":
            backbone = TemporalConvNet(num_channels=[32, 64, 128], 
                                       n_classes=args.n_class,  
                                       num_inputs=args.n_feature,
                                       input_length = args.input_length, 
                                       kernel_size=16, 
                                       dropout=0.2, 
                                       backbone=True)
        elif args.backbone == "HRCTPNet":
            backbone = HRCTPNet(num_channels=args.n_feature, 
                                num_classes=args.n_class,
                                backbone=True,)
        else:
            raise NotImplementedError



    # take the overall overall best parameters from "What Makes Good Contrastive Learning on Small-Scale Wearable-based Tasks?"
    # paper for this framework
    if args.framework == "byol":
        args.lr_pretrain = 0.01
        args.pretrain_batch_size = 64
        args.weight_decay_pretrain = 1.5e-6
        args.pretrain_n_epoch = 60
    elif args.framework == "simsiam":
        args.lr_pretrain = 3e-4
        args.pretrain_batch_size = 256
        args.weight_decay_pretrain = 1e-4
        args.pretrain_n_epoch = 60
    elif args.framework == "simclr":
        args.pretrain_lr = 2.5e-3
        args.pretrain_batch_size = 256
        args.weight_decay_pretrain = 1e-6
        args.pretrain_n_epoch = 120
    elif args.framework == "nnclr":
        args.lr_pretrain = 2e-3
        args.pretrain_batch_size = 256
        args.pretrain_weight_decay = 1e-6
        args.pretrain_n_epoch = 120
    elif args.framework == "tstcc":
        args.lr_pretrain = 3e-4
        args.pretrain_batch_size = 128
        args.weight_decay_pretrain = 3e-4
        args.pretrain_n_epoch = 40

    

    # set up model and optimizers
    if args.framework in ['byol', 'simsiam']:



        model = BYOL(DEVICE, backbone, window_size=args.input_length, n_channels=args.n_feature, projection_size=args.p,
                     projection_hidden_size=args.phid, moving_average=args.EMA)
        optimizer1 = torch.optim.Adam(model.online_encoder.parameters(),
                                      args.lr_pretrain,
                                      weight_decay=args.weight_decay_pretrain)
        optimizer2 = torch.optim.Adam(model.online_predictor.parameters(),
                                      args.lr_pretrain * args.lr_mul,
                                      weight_decay=args.weight_decay_pretrain)
        optimizers = [optimizer1, optimizer2]
    elif args.framework == 'simclr':
        model = SimCLR(backbone=backbone, dim=args.p)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_pretrain)
        optimizers = [optimizer]
    elif args.framework == 'nnclr':
        model = NNCLR(backbone=backbone, dim=args.p, pred_dim=args.phid)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay_pretrain)
        optimizers = [optimizer]
    elif args.framework == 'tstcc':
        model = TSTCC(backbone=backbone, DEVICE=DEVICE, temp_unit=args.temp_unit, tc_hidden=100)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_pretrain, betas=(0.9, 0.99), weight_decay=args.weight_decay_pretrain)
        optimizers = [optimizer]
    elif args.framework == 'supervised':
        model = backbone
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay_pretrain)
        optimizers = [optimizer]
    elif args.framework == 'reconstruction':
        model = backbone
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay_pretrain)
        args
        optimizers = [optimizer]
    else:
        NotImplementedError

    model = model.to(DEVICE)


    return model, optimizers


def delete_files(args):
    for epoch in range(args.n_epoch):
        model_dir = os.path.join(model_dir_name, 'pretrain_' + args.model_name + str(epoch) + '.pt')
        if os.path.isfile(model_dir):
            os.remove(model_dir)

        cls_dir = os.path.join(model_dir_name, 'lincls_' + args.model_name + str(epoch) + '.pt')
        if os.path.isfile(cls_dir):
            os.remove(cls_dir)




def setup_args(args):

        
    if args.lr_finetune_lstm == -1:
        args.lr_finetune_lstm = args.lr_finetune_backbone

    if args.model_uncertainty == "gaussian_classification":
        args.discretize_hr = True
        args.n_class = args.n_prob_class
        args.loss = "CrossEntropy"
    elif args.model_uncertainty == "NLE":
        args.loss = "NLE"
        args.discretize_hr = False
    else:
        args.discretize_hr = False

    

    if args.n_class == 1:
        assert args.loss in ['MSE', 'MAE', 'Huber', 'LogCosh', "NLE"]
    else:
        assert args.loss == 'CrossEntropy'

    if args.dataset in ['max', 'apple', 'apple100', 'capture24', 'capture24all', 'm2sleep', 'm2sleep100', "parkinson100", "IEEE", "appleall", "max_v2", "max_hrv"]:
        args.n_feature = 3
    
    if args.backbone == "Transformer":
        # limit batch size to 128 for transformer, otherwise it will run out of memory
        args.pretrain_batch_size = min(args.pretrain_batch_size, 128)
        args.batch_size = min(args.batch_size, 128)

    # set up default hyper-parameters
    if args.framework == 'byol':
        args.weight_decay_pretrain = 1.5e-6
        args.criterion = 'cos_sim'
        args.n_channels_out = args.n_class
    if args.framework == 'simsiam':
        args.weight_decay_pretrain = 1e-4
        args.EMA = 0.0
        args.lr_mul = 1.0
        args.criterion = 'cos_sim'
        args.n_channels_out = args.n_class
    if args.framework in ['simclr', 'nnclr']:
        args.criterion = 'NTXent'
        args.weight_decay_pretrain = 1e-6
        args.n_channels_out = args.n_class
    if args.framework == 'tstcc':
        args.criterion = 'NTXent'
        args.backbone = 'FCN'
        args.weight_decay_pretrain = 3e-4
        args.n_channels_out = args.n_class
    if args.framework == 'supervised':
        args.lr_finetune_backbone = args.lr
        args.lr_finetune_lstm = args.lr
        args.n_channels_out = args.n_class
    if args.framework == 'reconstruction':
        args.criterion = 'MSE'
        args.n_channels_out = 1
        assert args.backbone in ['AE', 'CNN_AE', "CorNET", "LSTM", "Transformer", "Attention_CNN_AE", "HRCTPNet"]

    model_name_opt = ""
    model_name_opt += f"_pretrain_subsample_{args.pretrain_subsample:.3f}"  if args.pretrain_subsample != 1 else ""
    model_name_opt += f"_windowsize_{args.window_size}" if args.window_size != 10 else ""
    model_name_opt += f"_stepsize_{args.step_size}" if args.step_size != 8 else ""
    model_name_opt += f"_szfactor_test_{args.take_every_nth_test}" if args.take_every_nth_test != 1 else ""
    model_name_opt += f"_szfactor_train_{args.take_every_nth_train}" if args.take_every_nth_train != 1 else ""
    model_name_opt += f"_wfrequency" if args.add_frequency else ""
    model_name_opt += f"_{args.rnn_type}" if args.rnn_type != "lstm" else ""
    args.model_name = 'try_scheduler_' + args.framework + '_backbone_' + args.backbone +'_pretrain_' + args.pretrain_dataset + '_eps' + str(args.n_epoch) + '_lr' + str(args.lr_pretrain) + '_bs' + str(args.pretrain_batch_size) \
                      + '_aug1' + args.aug1 + '_aug2' + args.aug2 + '_dim-pdim' + str(args.p) + '-' + str(args.phid) \
                      + '_EMA' + str(args.EMA) + '_criterion_' + args.criterion + '_lambda1_' + str(args.lambda1) + '_lambda2_' + str(args.lambda2) + '_tempunit_' + args.temp_unit  +  model_name_opt


    # set model saving paths
    args.model_dir_name = os.path.join(config.results_dir, args.model_name)
    os.makedirs(args.model_dir_name, exist_ok=True)

    args.pretrain_model_file = os.path.join(args.model_dir_name, 'pretrain_' + args.model_name  + "_bestmodel" + '.pt')

    lincl_model_name_opt = ""
    lincl_model_name_opt += f"_{args.model_uncertainty}" if args.model_uncertainty not in ["none", "mcdropout"] else ""
    lincl_model_name_opt += f"_trainranked_{args.subsample_ranked_train}" if args.subsample_ranked_train not in [None, 0.0] else ""
    lincl_model_name_opt += f"_subsample_{args.subsample:.3f}" if args.subsample != 1 else ""
    lincl_model_name_opt += f"_timesplit" if args.split_by == "time" else ""
    lincl_model_name_opt += f"_disc_hr_{args.n_class}" if args.discretize_hr else ""
    lincl_model_name_opt += f"_hrsmoothing_{args.hr_smoothing}" if args.hr_smoothing > 1 else ""
    lincl_model_name_opt += f"_lr_{args.lr_finetune_backbone:.1E}" if args.lr_finetune_backbone != args.lr else ""
    lincl_model_name_opt += f"_lr_lstm_{args.lr_finetune_lstm:.1E}" if args.lr_finetune_lstm != args.lr_finetune_backbone else ""
    lincl_model_name_opt += f"_hrmin_{args.hr_min}" if args.hr_min != 50 else ""
    lincl_model_name_opt += f"_hrmax_{args.hr_max}" if args.hr_max != 110 else ""
    lincl_model_name_opt += f"_rseed_{args.random_seed}" if args.random_seed != 10 else ""
    lincl_model_name_opt += f"_nlayers_{args.num_layers_classifier}" if args.num_layers_classifier != 1 else ""
    lincl_model_name = 'lincls'+ lincl_model_name_opt + "_" + args.backbone + '_dataset_' + args.dataset + '_split' + str(args.split) + '_eps' + str(args.n_epoch) + '_bs' + str(args.batch_size) + "_bestmodel" + '.pt'

    args.lincl_model_file = os.path.join(args.model_dir_name, lincl_model_name)

    return args
    

def setup(args, DEVICE):
    model, optimizers = setup_model_optm(args, DEVICE)

    # loss fn
    if args.criterion == 'cos_sim':
        criterion = nn.CosineSimilarity(dim=1)
    elif args.criterion == 'NTXent':
        if args.framework == 'tstcc':
            criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.2)
        else:
            criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.1)
    elif args.criterion == 'MSE':
        criterion = nn.MSELoss()
    elif args.criterion == 'MAE':
        criterion = nn.L1Loss()


    # log
    if os.path.isdir(args.logdir) == False:
        os.makedirs(args.logdir)
    log_file_name = os.path.join(args.logdir, args.model_name + f".log")
    logger = _logger(log_file_name)
    logger.debug(args)


    # Initialize W&B
    wandb_run = wandb.init(
    # Set the project where this run will be logged
    project= args.wandb_project,
    group = args.wandb_group if args.wandb_group != '' else None,
    tags=[args.wandb_tag] if args.wandb_tag != '' else None,
    # Track hyperparameters and run metadata
    config=vars(args),
    mode = args.wandb_mode)

    

    schedulers = []
    for optimizer in optimizers:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_n_epoch, eta_min=0)
        schedulers.append(scheduler)

    global nn_replacer
    nn_replacer = None
    if args.framework == 'nnclr':
        nn_replacer = NNMemoryBankModule(size=args.mmb_size)

    global recon
    recon = None
    if args.backbone in ['AE', 'CNN_AE']:
        recon = nn.MSELoss()

    total_params = sum(
        param.numel() for param in model.parameters()
    )

    wandb.log({"Total_Params": total_params}, step=0)


    return model, optimizers, schedulers, criterion, logger

def setup_classifier(args, DEVICE, backbone):


    bb_dim = backbone.out_dim
    classifier = setup_linclf(args, DEVICE, bb_dim)

    # splits the different layers of the model so we can assign different learning rates to them
    lstm_gru_parameters = []
    conv_layers = []
    for name, param in backbone.named_parameters():
        if 'lstm' in name.lower() or 'gru' in name.lower():
            lstm_gru_parameters.append(param)
        else:
            conv_layers.append(param)

    params = [
        {"params": conv_layers, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": lstm_gru_parameters, "lr": args.lr, "weight_decay": args.weight_decay}
    ]

    if args.optimizer == 'Adam':
        optimizer_cls = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        NotImplementedError

    if args.loss == 'MSE':
        criterion_cls = nn.MSELoss()
    elif args.loss == 'MAE':
        criterion_cls = nn.L1Loss()
    elif args.loss == 'Huber':
        criterion_cls = nn.HuberLoss(delta=args.huber_delta)
    elif args.loss == 'LogCosh':
        criterion_cls = LogCoshError()
    elif args.loss == 'CrossEntropy':
        criterion_cls = nn.BCELoss(reduction='mean')
    elif args.loss == 'NLE':
        criterion_cls = NLELoss(ratio = 0)
    else:
        NotImplementedError

    return classifier, criterion_cls, optimizer_cls





def calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=None, nn_replacer=None):

    target = target.to(DEVICE).float()

    if args.framework == 'reconstruction':
        sample = sample.to(DEVICE).float()
        sample_decoded, _ = model(sample)
        loss = criterion(sample_decoded, target) * args.lambda1
    else:
        aug_sample1 = gen_aug(sample, args.aug1)
        aug_sample2 = gen_aug(sample, args.aug2)
        aug_sample1, aug_sample2 = aug_sample1.to(DEVICE).float(), aug_sample2.to(DEVICE).float()
        if args.framework in ['byol', 'simsiam']:
            assert args.criterion == 'cos_sim'
        if args.framework in ['tstcc', 'simclr', 'nnclr']:
            assert args.criterion == 'NTXent'

        if args.framework in ['byol', 'simsiam', 'nnclr']:
            if args.backbone in ['AE', 'CNN_AE']:
                x1_encoded, x2_encoded, p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
                recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
            else:
                p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            if args.framework == 'nnclr':
                z1 = nn_replacer(z1, update=False)
                z2 = nn_replacer(z2, update=True)
            if args.criterion == 'cos_sim':
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            elif args.criterion == 'NTXent':
                loss = (criterion(p1, z2) + criterion(p2, z1)) * 0.5
            if args.backbone in ['AE', 'CNN_AE']:
                loss = loss * args.lambda1 + recon_loss * args.lambda2
        if args.framework == 'simclr':
            if args.backbone in ['AE', 'CNN_AE']:
                x1_encoded, x2_encoded, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
                recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
            else:
                z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            loss = criterion(z1, z2)
            if args.backbone in ['AE', 'CNN_AE']:
                loss = loss * args.lambda1 + recon_loss * args.lambda2
        if args.framework == 'tstcc':
            nce1, nce2, p1, p2 = model(x1=aug_sample1, x2=aug_sample2)
            tmp_loss = nce1 + nce2
            ctx_loss = criterion(p1, p2)
            loss = tmp_loss * args.lambda1 + ctx_loss * args.lambda2
        if args.framework == 'supervised':
            NotImplementedError
        

    return loss


def train(train_loader, val_loader, model, logger, DEVICE, optimizers, schedulers, criterion, args):
    # training and validation
    best_model = None
    min_val_loss = 1e8
    num_epochs = args.n_epoch

    for epoch in range(args.pretrain_n_epoch):
        total_loss = 0
        n_batches = 0
        model.train()

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit='batch') as tepoch:
            for idx, (sample, target, domain) in enumerate(tepoch):

                n_batches += 1
                for optimizer in optimizers:
                    optimizer.zero_grad()
                if sample.size(0) != args.batch_size:
                    continue
                loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
                total_loss += loss.item()
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                if args.framework in ['byol', 'simsiam']:
                    model.update_moving_average()

                tepoch.set_postfix(pretrain_loss=loss.item())
        #wandb.log({'lr': optimizers[0].param_groups[0]['lr']}, step=epoch)
        
        for scheduler in schedulers:
            scheduler.step()

        # save model
        #model_dir = os.path.join(args.model_dir_name, 'pretrain_' + args.model_name + str(epoch) + '.pt')
        #print('Saving model at {} epoch to {}'.format(epoch, model_dir))
        #torch.save({'model_state_dict': model.state_dict()}, model_dir)

        logger.debug(f'Train Loss     : {total_loss / n_batches:.4f}')
        wandb.log({'pretrain_training_loss': total_loss / n_batches}, step=epoch)

        if val_loader is None:
            with torch.no_grad():
                best_model = copy.deepcopy(model.state_dict())
        else:
            with torch.no_grad():
                model.eval()
                total_loss = 0
                n_batches = 0
                with tqdm(val_loader, desc=f"Validation", unit='batch') as tepoch:
                    for idx, (sample, target, domain) in enumerate(tepoch):
                        if sample.size(0) != args.batch_size:
                            continue
                        n_batches += 1
                        loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
                        total_loss += loss.item()
                if total_loss <= min_val_loss:
                    min_val_loss = total_loss
                    best_model = copy.deepcopy(model.state_dict())
                    model_dir = os.path.join(args.model_dir_name, 'pretrain_' + args.model_name  + "_bestmodel" + '.pt')
                    print('update: Saving model at {} epoch to {}'.format(epoch, model_dir))
                    torch.save({'model_state_dict': model.state_dict()}, model_dir)
                    
                logger.debug(f'Val Loss     : {total_loss / n_batches:.4f}')
                wandb.log({"pretrain_validation_loss": total_loss / n_batches}, step=epoch)
                
    return best_model


def load_best_model(args, epoch=None):
    if epoch is None:
        model_dir = os.path.join(args.model_dir_name, 'pretrain_' + args.model_name + "_bestmodel" + '.pt')
    else:
        model_dir = os.path.join(args.model_dir_name, 'pretrain_' + args.model_name + str(epoch) + '.pt')

    if not os.path.exists(model_dir):
        print("No model found at {}".format(model_dir))
        return None
    
    print('Loading model from {}'.format(model_dir))
    best_model = torch.load(model_dir)["model_state_dict"]

    # since the previously saved models included the classifier, we need to delete it
    # only a fix to make it compatible with older versions
    if "online_encoder.net.classifier.weight" in best_model.keys():
        del best_model["online_encoder.net.classifier.weight"]
        del best_model["online_encoder.net.classifier.bias"]
    if "target_encoder.net.classifier.weight" in best_model.keys():
        del best_model["target_encoder.net.classifier.weight"]
        del best_model["target_encoder.net.classifier.bias"]
    
    return best_model

def load_best_lincls(args, device=None):

    if device is None:
        device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')

    model_dir = args.lincl_model_file
    if not os.path.exists(model_dir):
        print("No model found at {}".format(model_dir))
        return None
    best_model = torch.load(model_dir, map_location=device)['trained_backbone']

    return best_model



def test(test_loader, model, logger, DEVICE, criterion, args):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        n_batches = 0
        with tqdm(test_loader, desc=f"Test", unit='batch') as tepoch:
            for idx, (sample, target, domain) in enumerate(tepoch):
                if sample.size(0) != args.batch_size:
                    continue
                n_batches += 1
                loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
                total_loss += loss.item()
        logger.debug(f'Test Loss     : {total_loss / n_batches:.4f}')
        wandb.log({"pretrain_test_loss": total_loss / n_batches})


def lock_backbone(model, args):

    for name, param in model.named_parameters():
        param.requires_grad = False



def extract_backbone(model, args):

    if args.framework in ['simsiam', 'byol']:
        trained_backbone = model.online_encoder.net
    elif args.framework in ['simclr', 'nnclr', 'tstcc']:
        trained_backbone = model.encoder
    elif args.framework == 'supervised':
        trained_backbone = model
    elif args.framework == 'reconstruction':
        trained_backbone = model
    else:
        NotImplementedError
    
    if args.backbone in ["AE", "CNN_AE"]:
        trained_backbone = trained_backbone.encoder

    return trained_backbone

def calculate_lincls_output(sample, target, trained_backbone, criterion, args):

    output, feat = trained_backbone(sample)
    if len(feat.shape) == 3:
        feat = feat.reshape(feat.shape[0], -1)
    loss = criterion(output, target)

    
    if "bnn" in args.model_uncertainty:
        kl_loss = get_kl_loss(trained_backbone)/sample.shape[0]
        loss += kl_loss
        predicted = output.data
    elif args.model_uncertainty == "NLE":
        #predicted = output[..., 0].data
        predicted = output[0].data
    else:
        predicted = output.data
    

    return loss, predicted, feat


def add_probability_wrapper(model, args, DEVICE):

    if args.model_uncertainty == "mcdropout":
        model = MC_Dropout_Wrapper(model, n_classes = args.n_prob_class, n_samples = 100)
        
    elif args.model_uncertainty in ["bnn", "bnn_pretrained", "bnn_pretrained_firstlast"]:
        model = BNN_Wrapper(model, n_classes=args.n_prob_class, n_samples=100)

    elif args.model_uncertainty == "NLE":
        model = NLE_Wrapper(model, n_classes=args.n_prob_class)

    elif args.model_uncertainty == "gaussian_classification":    
        model = Uncertainty_Wrapper(model, n_classes=args.n_prob_class, return_probs=True)

    else: # simple regression
        model = Uncertainty_Regression_Wrapper(model, n_classes=args.n_prob_class, return_probs=True)

    model = model.to(DEVICE)

    return model


def convert_to_hr(values, args):
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()


    if values.ndim > 1 and values.shape[1] > 1: # if we have more than one class

        # construct the bins
        bins = np.linspace(0, 1, args.n_prob_class-1)
        # lookup the values in the bins
        lookup_values = np.concatenate([[bins[0]] , (bins[1:] + bins[:-1])/2, [bins[-1]]])
        # convert to HR computing expectation
        values = np.dot(values, lookup_values)
    
    values = values * (args.hr_max - args.hr_min) + args.hr_min
    return values


def train_lincls(train_loader, val_loader, trained_backbone, logger , DEVICE, optimizer, criterion, args):
    best_model = None
    min_val_corr = -1

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=0)

    for epoch in range(args.n_epoch):
        trained_backbone.train() # TODO: remove this
        total_loss = 0
        n_batches = 0
        hr_true, hr_pred = [], []
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.n_epoch} - Training Lincls", unit='batch') as tepoch:
            for idx, (sample, target, domain) in enumerate(tepoch):
                n_batches += 1
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).float().reshape(target.shape[0], -1)
                loss, predicted, _, = calculate_lincls_output(sample, target, trained_backbone, criterion, args)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predicted = convert_to_hr(predicted, args)
                target = convert_to_hr(target, args)

                hr_true.extend(target.squeeze())
                hr_pred.extend(predicted.squeeze())

                tepoch.set_postfix(loss=loss.item())
        # save model
        hr_true, hr_pred = np.array(hr_true), np.array(hr_pred)
        mae_train = np.abs(hr_true - hr_pred).mean()
        train_loss = total_loss / n_batches
        corr_train = corr_function(hr_true, hr_pred)
        #model_dir = os.path.join(args.model_dir_name, 'lincls_' + args.model_name + '_' + args.dataset + "_split" + str(args.split) + "_" + str(epoch) + '.pt')
        #print('Saving model at {} epoch to {}'.format(epoch, model_dir))
        #torch.save({'trained_backbone': trained_backbone.state_dict(), 'classifier': classifier.state_dict()}, model_dir)


        logger.debug(f'Train Loss     : {train_loss:.4f}\t | \tTrain MAE     : {mae_train:2.4f}\t | \tTrain Corr     : {corr_train:2.4f}\n')
        wandb.log({'Train_Loss': train_loss, 'Train_MAE': mae_train, 'Train_Corr': corr_train}, step=epoch)


        if args.scheduler:
            scheduler.step()

        if val_loader is None:
            with torch.no_grad():
                best_model = copy.deepcopy(trained_backbone.state_dict())
                torch.save({'trained_backbone': trained_backbone.state_dict(), 'classifier': trained_backbone.classifier.state_dict()}, args.lincl_model_file)
        else:
            with torch.no_grad():
                trained_backbone.eval() # TODO: remove this
                val_loss = 0
                n_batches = 0
                hr_true, hr_pred, pids = [], [], []
                with tqdm(val_loader, desc=f"Validation Lincls", unit='batch') as tepoch:
                    for idx, (sample, target, domain) in enumerate(tepoch):
                        n_batches += 1
                        sample, target = sample.to(DEVICE).float(), target.to(DEVICE).float().reshape(target.shape[0], -1)
                        loss, predicted, _ = calculate_lincls_output(sample, target, trained_backbone, criterion, args)
                        total_loss += loss.item()

                        predicted = convert_to_hr(predicted, args)
                        target = convert_to_hr(target, args)

                        hr_true.extend(target.squeeze())
                        hr_pred.extend(predicted.squeeze())
                        pids.extend(domain.cpu().numpy().squeeze())
                hr_true, hr_pred = np.array(hr_true), np.array(hr_pred)
                mae_val = np.abs(hr_true - hr_pred).mean()
                corr_val = corr_function(hr_true, hr_pred)
                val_loss = total_loss / n_batches

                logging_table = pd.DataFrame({ 
                    "hr_true": hr_true, 
                    "hr_pred": hr_pred,
                    "pid": pids
                    })
                
                #wandb.log({"hr_true_vs_pred_val": wandb.Table(dataframe = pd.DataFrame(logging_table))}, step=epoch)
                #figure = plot_true_pred(hr_true, hr_pred, x_lim=[args.hr_min, args.hr_max], y_lim=[args.hr_min, args.hr_max])
                #wandb.log({"true_pred_val": figure}, step=epoch)
                logger.debug(f'Val Loss     : {val_loss:.4f}, Val MAE     : {mae_val:.4f}, Val Corr     : {corr_val:.4f}\n')
                wandb.log({'Val_Loss': val_loss, 'Val_MAE': mae_val, 'Val_Corr': corr_val}, step=epoch)
                if corr_val >= min_val_corr:
                    min_val_corr = corr_val
                    best_model = copy.deepcopy(trained_backbone.state_dict())
                    torch.save({'trained_backbone': trained_backbone.state_dict(), 'classifier': trained_backbone.classifier.state_dict()}, args.lincl_model_file)
                    print('Saving models and results at {} epoch to {}'.format(epoch, args.lincl_model_file))
                    logging_table.to_pickle(os.path.join(args.model_dir_name, args.lincl_model_file.replace(".pt", f"_val.pickle")))

    return best_model


def test_lincls(test_loader, trained_backbone, logger, DEVICE, criterion, args, plt=False):

    total_loss = 0
    feats = None
    hr_true, hr_pred, pids = [], [], []
    total = 0
    with torch.no_grad():
        trained_backbone.eval()
        with tqdm(test_loader, desc=f"Test Lincls", unit='batch') as tepoch:
            for idx, (sample, target, domain) in enumerate(tepoch):
                total += 1
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).float().reshape(target.shape[0], -1)
                loss, predicted, feat = calculate_lincls_output(sample, target, trained_backbone, criterion, args)
                total_loss += loss.item()
                if feats is None:
                    feats = feat
                else:
                    feats = torch.cat((feats, feat), 0)

                predicted = convert_to_hr(predicted, args)
                target = convert_to_hr(target, args)

                hr_true.extend(target.squeeze())
                hr_pred.extend(predicted.squeeze())
                pids.extend(domain.cpu().numpy().squeeze())
        
        logging_table = pd.DataFrame({ 
                    "hr_true": hr_true, 
                    "hr_pred": hr_pred,
                    "pid": pids
                    }) 
        
        # Logging results per participant id
        results_per_pid = []
        pids = np.array(pids)
        for pid in np.unique(pids[:,0]):
            hr_true_pid = np.array(hr_true)[np.where(pids[:,0] == pid)]
            hr_pred_pid = np.array(hr_pred)[np.where(pids[:,1] == pid)]
            mae_pid = np.abs(hr_true_pid - hr_pred_pid).mean()
            corr_pid = corr_function(hr_true_pid, hr_pred_pid)
            results_per_pid.append((pid, mae_pid, corr_pid))
        wandb.log({"results_per_pid": wandb.Table(data=results_per_pid, columns=["pid", "MAE", "Corr"])})
        
        hr_true, hr_pred = np.array(hr_true), np.array(hr_pred)
        
        figure = plot_true_pred(hr_true, hr_pred, x_lim=[args.hr_min, args.hr_max], y_lim=[args.hr_min, args.hr_max])
        wandb.log({"true_pred_test": figure})
        
        mae_test = np.abs(hr_true - hr_pred).mean()
        corr_val = corr_function(hr_true, hr_pred)
        wandb.log({'Test_Loss': total_loss, 'Test_MAE': mae_test, "Test_Corr": corr_val})
        logger.debug(f'Test Loss     : {total_loss:.4f}, Test MAE     : {mae_test:.4f}, Test Corr     : {corr_val:.4f}\n')
        print('Saving results to {}'.format(args.model_dir_name))
        logging_table.to_pickle(os.path.join(args.model_dir_name, args.lincl_model_file.replace(".pt", f"_test.pickle")))


    if plt:
        tsne(feats, hr_true, save_dir=plot_dir_name + '/' + args.model_name + '_tsne.png')
        mds(feats, hr_true, save_dir=plot_dir_name + '/' + args.model_name + '_mds.png')
        print('plots saved to ', plot_dir_name)

def test_postprocessing(test_loader, model, postprocessing, logger, DEVICE, criterion_cls, args, plt=False, prefix="Test"):
    hr_true, hr_pred, hr_pred_post, pids, uncertainties, probs, probs_post, target_probs = [], [], [], [], [], [], [], []
    total_loss = 0
    with torch.no_grad():
        model.eval()
        with tqdm(test_loader, desc=f"{prefix} Postprocessing", unit='batch') as tepoch:
            for idx, (sample, target, domain) in enumerate(tepoch):
                sample, target = sample.to(DEVICE).float(), target.float().reshape(target.shape[0], -1)
                pred, x, prob = model(sample)
                pred, prob = pred.cpu(), prob.cpu()
                # if deterministic model, uncertainty will be 0 for all values, prob_post will be one-hot encoding of pred_post
                pred_post, uncertainty, prob_post = postprocessing(prob, pred)

                loss = criterion_cls(prob, target)
                total_loss += loss.item()

                predicted_hr_post = convert_to_hr(pred_post, args)
                predicted_hr = convert_to_hr(pred, args)
                target_hr = convert_to_hr(target, args)


                hr_true.extend(target_hr.reshape(-1))
                hr_pred.extend(predicted_hr.reshape(-1))
                hr_pred_post.extend(predicted_hr_post.reshape(-1))
                pids.extend(domain.numpy().reshape(-1, 2))
                uncertainties.extend(uncertainty.numpy().reshape(-1))
                target_probs.extend(target.numpy().reshape(-1, args.n_prob_class))

                probs_post.extend(prob_post.numpy().reshape(-1, args.n_prob_class))
                probs.extend(prob.numpy().reshape(-1, args.n_prob_class))

        
            logging_table = { 
                        "hr_true": hr_true, 
                        "hr_pred": hr_pred,
                        "hr_pred_post": hr_pred_post,
                        "pid": pids,
                        "uncertainty": uncertainties,
                        }
            if args.save_probabilities:
                logging_table["probs_post"] = probs_post
                logging_table["probs"] = probs

            hr_true, hr_pred, hr_pred_post, probs, probs_post, target_probs = np.array(hr_true), np.array(hr_pred), np.array(hr_pred_post), np.array(probs), np.array(probs_post), np.array(target_probs)
            total_loss = total_loss / len(test_loader)
            figure = plot_true_pred(hr_true, hr_pred, x_lim=[args.hr_min, args.hr_max], y_lim=[args.hr_min, args.hr_max])
            wandb.log({f"true_pred_{prefix}_post": figure})
            
            mae = np.abs(hr_true - hr_pred).mean()
            mae_post = np.abs(hr_true - hr_pred_post).mean()
            corr = corr_function(hr_true, hr_pred)
            corr_post = corr_function(hr_true, hr_pred_post)
            ece = ece_loss(probs, target_probs)
            nll = nll_loss(probs, target_probs)
            ece_post = ece_loss(probs_post, target_probs)
            nll_post = nll_loss(probs_post, target_probs)
            wandb.log({f'{prefix}_Loss': total_loss, f'{prefix}_MAE': mae, f"{prefix}_Corr": corr, f"Post_{prefix}_MAE": mae_post, f"Post_{prefix}_Corr_Post": corr_post, f"{prefix}_ECE": ece, f"{prefix}_NLL": nll, f"Post_{prefix}_ECE": ece_post, f"Post_{prefix}_NLL": nll_post})
            logger.debug(f'{prefix} Loss     : {total_loss:.4f}, {prefix} MAE     : {mae:.4f}, {prefix} Corr     : {corr:.4f}, Post {prefix} MAE   : {mae_post:.4f}, Post {prefix} Corr   : {corr_post:.4f}   ECE: {ece:.4f}, NLL: {nll:.4f}, Post ECE: {ece_post:.4f}, Post NLL: {nll_post:.4f}\n')
            print('Saving results to {}'.format(args.model_dir_name))
            pd.DataFrame(logging_table).to_pickle(os.path.join(args.model_dir_name, args.lincl_model_file.replace(".pt", f"{prefix}_{args.model_uncertainty}_{args.postprocessing}.pickle")))

def setup_postprocessing_model(args):

    if args.postprocessing == "beliefppg" or args.postprocessing == "sumprod":
        postprocessing_model = BeliefPPG(
                    args.n_prob_class, 
                    return_probs= True,
                    uncert = "std",
                    method="sumprod")
        
    elif args.postprocessing == "viterbi":
        postprocessing_model = BeliefPPG(
            args.n_prob_class, 
            return_probs= True,
            uncert = "std",
            method="viterbi")

    elif args.postprocessing == "raw":
        postprocessing_model = Postprocessing(
            args.n_prob_class, 
            return_probs= True,
            uncert = "std")
        

    elif args.postprocessing == "kalmansmoothing":
        postprocessing_model = KalmanSmoothing(
            args.n_prob_class, 
            return_probs= True,
            uncert = "std",
            step_size=args.step_size,)


    else:
        raise NotImplementedError
    return postprocessing_model

def train_postprocessing(train_loader, postprocessing_model, DEVICE, args):

    def cut_prob(probs):
        exp = np.round(np.dot(probs, np.arange(args.n_prob_class))).astype(int)
        probs = np.eye(args.n_prob_class)[exp]
        return probs


    ys = [target for _, target, _ in train_loader]
    # Sometimes we get a Value Error here. This is because the probabilities for some classes are too small. Taking the log will then result in infinite values
    postprocessing_model.fit_layer(ys, distr=args.transition_distribution)
    return postprocessing_model


def predict_median(train_loader, val_loader, test_loader, args, mode="global"):

    assert mode in ["global", "subject_wise"]

    # predicts the meidan of the training HR values
    wandb_run = wandb.init(
    # Set the project where this run will be logged
    project= args.wandb_project,
    group = args.wandb_group if args.wandb_group != '' else None,
    tags=[args.wandb_tag] if args.wandb_tag != '' else None,
    # Track hyperparameters and run metadata
    config=vars(args),
    mode = args.wandb_mode)

    if mode=="global":
        for _, target, pid in train_loader:
            target = convert_to_hr(target, args)
            hr_true_train.extend(target.squeeze())
        hr_true_train = np.array(hr_true_train)
        hr_median = np.median(hr_true_train)
        hr_pred_train = np.repeat(hr_median, len(hr_true_train))
        mae_train = np.abs(hr_true_train - hr_pred_train).mean()
        corr_train = corr_function(hr_true_train, hr_pred_train)
        wandb.log({'Train_MAE': mae_train, 'Train_Corr': corr_train})

        hr_true_val = []
        for _,target, _ in val_loader:

            target = convert_to_hr(target, args)
            hr_true_val.extend(target.squeeze())

        hr_true_val = np.array(hr_true_val)
        hr_pred_val = np.repeat(hr_median, len(hr_true_val))
        mae_train = np.abs(hr_true_val - hr_pred_val).mean()
        corr_train = corr_function(hr_true_val, hr_pred_val)

        wandb.log({'Val_MAE': mae_train, 'Val_Corr': corr_train})

        hr_true_test = []

        for _, target, _ in test_loader:
            target = convert_to_hr(target, args)
            hr_true_test.extend(target.squeeze())
        
        hr_true_test = np.array(hr_true_test)
        hr_pred_test = np.repeat(hr_median, len(hr_true_test))
        mae_train = np.abs(hr_true_test - hr_pred_test).mean()
        corr_train = corr_function(hr_true_test, hr_pred_test)
        
        wandb.log({'Test_MAE': mae_train, 'Test_Corr': corr_train})

    elif mode=="subject_wise":
        for name, datalaoder in {"Train": train_loader, "Val": val_loader, "Test": test_loader}.items():
            
            targets, pids = [], []
            for _, target, pid in datalaoder:
                targets.extend(target.squeeze())
                pids.extend(pid[:,0])
            targets = np.array(targets)
            pids = np.array(pids)
            pid_unique = np.unique(pids)


            hr_true, hr_pred = [], []
            for pid in pid_unique:
                target_pid = targets[pids == pid]
                if len(target_pid) == 0:
                    continue
                hr_median = np.median(hr_true)
                pred_pid = np.repeat(hr_median, len(target_pid))
                hr_true.extend(target_pid)
                hr_pred.extend(pred_pid)
            hr_true = np.array(hr_true)
            hr_pred = np.array(hr_pred)
            mae = np.abs(hr_true - hr_pred).mean()
            corr_val = corr_function(hr_true, hr_pred)
            wandb.log({f'{name}_MAE': mae, f'{name}_Corr': corr_val})

    wandb.finish()

    

