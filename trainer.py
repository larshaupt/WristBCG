import torch
import torch.nn as nn
import numpy as np
import os
import wandb
from tqdm import tqdm
import pandas as pd
import config
import re
from torchmetrics.regression import LogCoshError
from bayesian_torch.models.dnn_to_bnn import get_kl_loss

from augmentations import gen_aug
from utils import tsne, corr_function, plot_true_pred
from data_preprocess import data_preprocess_hr

from models.frameworks import *
from models.backbones import *
from models.loss import *
from models.postprocessing import Postprocessing, BeliefPPG, KalmanSmoothing


def setup_dataloaders(params, mode="finetuning"):
    """
    Sets up data loaders for different modes of operation: pretraining, finetuning, postprocessing,
    and postprocessing without discretization. 

    Parameters:
    params (Namespace): An argparse namespace containing the dataset parameters and configurations.
    mode (str, optional): The mode of operation. Options are "pretraining", "finetuning", 
                          "postprocessing", and "postprocessing_no_discrete". Defaults to "finetuning".

    Returns:
    tuple: A tuple containing the train_loader, val_loader, and test_loader.
    
    Raises:
    ValueError: If the mode is not recognized.
    """

    # Configuration settings based on the selected mode
    if mode == "pretraining":
        # Use pretraining dataset and specific pretraining settings
        dataset = params.pretrain_dataset
        split = 0  # Always use split 0 for pretraining
        reconstruction = params.framework == "reconstruction"
        sample_sequences = False
        sigma = params.label_sigma
        discrete_hr = False

    elif mode == "finetuning":
        # Use standard dataset and finetuning settings
        dataset = params.dataset
        split = params.split
        reconstruction = False
        sample_sequences = False
        sigma = params.label_sigma
        discrete_hr = params.discretize_hr

    elif mode == "postprocessing":
        # Use standard dataset with postprocessing settings
        dataset = params.dataset
        split = params.split
        reconstruction = False
        sample_sequences = True
        sigma = params.label_sigma
        discrete_hr = True

    elif mode == "postprocessing_no_discrete":
        # Use standard dataset with postprocessing settings, without discretization
        dataset = params.dataset
        split = params.split
        reconstruction = False
        sample_sequences = True
        sigma = params.label_sigma
        discrete_hr = False

    else:
        # Raise an error if the mode is not recognized
        raise ValueError(f"Mode {mode} not recognized")

    # Preprocess data and prepare data loaders
    print(f"Loading dataset {dataset} with split {split} for mode {mode}")
    train_loader, val_loader, test_loader = data_preprocess_hr.prep_hr(
        params, 
        dataset=dataset, 
        split=split, 
        reconstruction=reconstruction, 
        sample_sequences=sample_sequences, 
        discrete_hr=discrete_hr, 
        sigma=sigma
    )
    
    return train_loader, val_loader, test_loader

def setup_linclf(params, DEVICE, bb_dim):
    """
    Sets up a linear classifier based on the provided arguments and device configuration.

    Parameters:
    params (Namespace): An argparse namespace containing model parameters and configurations.
    DEVICE (torch.device): The device on which the model will be run (e.g., CPU or GPU).
    bb_dim (int): The output dimension of the backbone network.

    Returns:
    torch.nn.Module: A linear classifier configured based on the provided arguments.
    """
    
    # Determine the classifier type based on the backbone and framework
    if params.backbone in ['CNN_AE'] and params.framework == 'reconstruction':
        # In this case the backbone is only the convolutional layers
        # Use LSTM classifier for reconstruction with CNN_AE backbone
        classifier = LSTM_Classifier(
            bb_dim=bb_dim, 
            n_classes=params.n_class, 
            rnn_type=params.rnn_type, 
            num_layers=params.num_layers_classifier
        )
    else: # backbone is convolutional and RNN layers
        if params.model_uncertainty == "NLE":
            # Use classifier with uncertainty estimation, i.e. a second head for uncertainty
            classifier = Classifier_with_uncertainty(
                bb_dim=bb_dim, 
                n_classes=params.n_class, 
                num_layers=params.num_layers_classifier
            )
        else:
            # Use standard classifier
            classifier = Classifier(
                bb_dim=bb_dim, 
                n_classes=params.n_class, 
                num_layers=params.num_layers_classifier
            )

    # Move the classifier to the specified device
    classifier = classifier.to(DEVICE)
    return classifier


def setup_model_optm(params, DEVICE):

    """
    Sets up the model and optimizer(s) based on the provided arguments and device configuration.

    Parameters:
    params (Namespace): An argparse namespace containing model parameters and configurations.
    DEVICE (torch.device): The device on which the model will be run (e.g., CPU or GPU).

    Returns:
    tuple: A tuple containing the model and a list of optimizers.
    """

    # set up backbone network


    if "bnn" in params.model_uncertainty: # Bayesian Neural Network

        if params.backbone == "CorNET":

            bayesian_layers = "first_last" if "firstlast" in params.model_uncertainty else "all"

            # loading pretrained model
            if "pretrained" in params.model_uncertainty:

                pretrained_model_dir = params.lincl_model_file.replace("_bnn", "").replace("_pretrained", ""). replace("_firstlast", "")
                if os.path.isfile(pretrained_model_dir):
                    state_dict = torch.load(pretrained_model_dir)["trained_backbone"]
                    print("Loading backbone from {}".format(pretrained_model_dir))
                else:
                    raise ValueError("No pretrained model found at {}".format(pretrained_model_dir))
                backbone = BayesianCorNET(n_channels=params.n_feature, 
                                          n_classes=params.n_class, 
                                          conv_kernels=params.num_kernels, 
                                          kernel_size=params.kernel_size, 
                                          LSTM_units=params.lstm_units, 
                                          backbone=True, 
                                          input_size=params.input_length, 
                                          state_dict=state_dict, 
                                          bayesian_layers=bayesian_layers, 
                                          dropout=params.dropout_rate,
                                          rnn_type=params.rnn_type)
            
            else: # no pretrained model
                backbone = BayesianCorNET(n_channels=params.n_feature, 
                                          n_classes=params.n_class, 
                                          conv_kernels=params.num_kernels, 
                                          kernel_size=params.kernel_size, 
                                          LSTM_units=params.lstm_units, 
                                          backbone=True, 
                                          input_size=params.input_length, 
                                          bayesian_layers=bayesian_layers, 
                                          dropout=params.dropout_rate,
                                          rnn_type=params.rnn_type)
        else: # BNN, different architecure
            raise NotImplementedError
    else: # no BNN
        if params.backbone == 'FCN':
            backbone = FCN(n_channels=params.n_feature, 
                           n_classes=params.n_class, 
                           conv_kernels=params.num_kernels, 
                           kernel_size=params.kernel_size, 
                           input_size=params.input_length, 
                           backbone=True)
        elif params.backbone == 'DCL':
            backbone = DeepConvLSTM(n_channels=params.n_feature, 
                                    n_classes=params.n_class, 
                                    conv_kernels=params.num_kernels, 
                                    kernel_size=params.kernel_size, 
                                    input_size=params.input_length, 
                                    LSTM_units=params.lstm_units, 
                                    rnn_type = params.rnn_type, 
                                    backbone=True)
        elif params.backbone == 'LSTM':
            backbone = LSTM(n_channels=params.n_feature, 
                            n_classes=params.n_class, 
                            LSTM_units=params.lstm_units, 
                            rnn_type = params.rnn_type, 
                            backbone=True)
        elif params.backbone == 'AE':
            backbone = AE(n_channels=params.n_feature, 
                          input_size=params.input_length, 
                          n_classes=params.n_class, 
                          embdedded_size=128, 
                          backbone=True, 
                          n_channels_out=params.n_channels_out)
        elif params.backbone == 'CNN_AE':
            backbone = CNN_AE(n_channels=params.n_feature, 
                              n_classes=params.n_class, 
                              input_size=params.input_length, 
                              backbone=True, 
                              n_channels_out=params.n_channels_out, 
                              dropout=params.dropout_rate, 
                              num_layers=3,
                              pool_kernel_size=2,
                              kernel_size=params.kernel_size,
                              conv_kernels=params.num_kernels)
        elif params.backbone == 'Transformer':
            backbone = Transformer(n_channels=params.n_feature, 
                                   input_size=params.input_length, 
                                   n_classes=params.n_class, 
                                   dim=128, 
                                   depth=4, 
                                   heads=4, 
                                   mlp_dim=64, 
                                   dropout=0.1, 
                                   backbone=True)
            
        elif params.backbone == "CorNET": # CorNET, 
            backbone = CorNET(n_channels=params.n_feature, 
                                n_classes=params.n_class, 
                                conv_kernels=params.num_kernels, 
                                kernel_size=params.kernel_size, 
                                LSTM_units=params.lstm_units, 
                                backbone=True, 
                                input_size=params.input_length, 
                                rnn_type=params.rnn_type,
                                dropout_rate=params.dropout_rate)
            
        elif params.backbone == "AttentionCorNET":
            backbone = AttentionCorNET(n_channels=params.n_feature, 
                                       n_classes=params.n_class, 
                                       conv_kernels=params.num_kernels, 
                                       kernel_size=params.kernel_size, 
                                       LSTM_units=params.lstm_units, 
                                       backbone=True, 
                                       input_size=params.input_length, 
                                       rnn_type=params.rnn_type)
            
        elif params.backbone == "FrequencyCorNET":
            backbone = CorNETFrequency(n_channels=params.n_feature, 
                                           n_classes=params.n_class, 
                                           conv_kernels=params.num_kernels, 
                                           kernel_size=params.kernel_size, 
                                           LSTM_units=params.lstm_units,
                                           backbone=True, 
                                           input_size=params.input_length, 
                                           num_extra_features=params.n_feature)
            
        elif params.backbone == "TCN":
            backbone = TemporalConvNet(num_channels=[32, 64, 128], 
                                       n_classes=params.n_class,  
                                       num_inputs=params.n_feature,
                                       input_length = params.input_length, 
                                       kernel_size=16, 
                                       dropout=0.2, 
                                       backbone=True)
        elif params.backbone == "HRCTPNet":
            backbone = HRCTPNet(num_channels=params.n_feature, 
                                num_classes=params.n_class,
                                backbone=True,)
        elif params.backbone == "ResNET":
            backbone = Resnet(n_channels=params.n_feature, 
                              n_classes=params.n_class, 
                              backbone=True)
        else:
            raise NotImplementedError



    # take the overall overall best parameters from "What Makes Good Contrastive Learning on Small-Scale Wearable-based Tasks?"
    # paper for this framework
    if params.framework == "byol":
        params.lr_pretrain = 0.01
        params.pretrain_batch_size = 64
        params.weight_decay_pretrain = 1.5e-6
        params.pretrain_n_epoch = 60
    elif params.framework == "simsiam":
        params.lr_pretrain = 3e-4
        params.pretrain_batch_size = 256
        params.weight_decay_pretrain = 1e-4
        params.pretrain_n_epoch = 60
    elif params.framework == "simclr":
        params.pretrain_lr = 2.5e-3
        params.pretrain_batch_size = 256
        params.weight_decay_pretrain = 1e-6
        params.pretrain_n_epoch = 120
    elif params.framework == "nnclr":
        params.lr_pretrain = 2e-3
        params.pretrain_batch_size = 256
        params.pretrain_weight_decay = 1e-6
        params.pretrain_n_epoch = 120
    elif params.framework == "tstcc":
        params.lr_pretrain = 3e-4
        params.pretrain_batch_size = 128
        params.weight_decay_pretrain = 3e-4
        params.pretrain_n_epoch = 40

    

    # set up model and optimizers
    if params.framework in ['byol', 'simsiam']:

        model = BYOL(DEVICE, backbone, window_size=params.input_length, n_channels=params.n_feature, projection_size=params.p,
                     projection_hidden_size=params.phid, moving_average=params.EMA)
        optimizer1 = torch.optim.Adam(model.online_encoder.parameters(),
                                      params.lr_pretrain,
                                      weight_decay=params.weight_decay_pretrain)
        optimizer2 = torch.optim.Adam(model.online_predictor.parameters(),
                                      params.lr_pretrain * params.lr_mul,
                                      weight_decay=params.weight_decay_pretrain)
        optimizers = [optimizer1, optimizer2]
    elif params.framework == 'simclr':
        model = SimCLR(backbone=backbone, dim=params.p)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr_pretrain)
        optimizers = [optimizer]
    elif params.framework == 'nnclr':
        model = NNCLR(backbone=backbone, dim=params.p, pred_dim=params.phid)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr_pretrain, weight_decay=params.weight_decay_pretrain)
        optimizers = [optimizer]
    elif params.framework == 'tstcc':
        model = TSTCC(backbone=backbone, DEVICE=DEVICE, temp_unit=params.temp_unit, tc_hidden=100)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr_pretrain, betas=(0.9, 0.99), weight_decay=params.weight_decay_pretrain)
        optimizers = [optimizer]
    elif params.framework == 'supervised':
        model = backbone
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr_pretrain, weight_decay=params.weight_decay_pretrain)
        optimizers = [optimizer]
    elif params.framework == 'reconstruction':
        model = backbone
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr_pretrain, weight_decay=params.weight_decay_pretrain)
        params
        optimizers = [optimizer]
    elif params.framework == 'oxford':
        model = backbone
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr_pretrain, weight_decay=params.weight_decay_pretrain)
        optimizers = [optimizer]
    else:
        raise NotImplementedError

    model = model.to(DEVICE)

    return model, optimizers


def setup_args(params):
    # default is regression setting --> n_class = 1
    params.n_class = params.n_prob_class if params.model_uncertainty == "gaussian_classification" else 1

    if params.lr_finetune_lstm == -1:
        params.lr_finetune_lstm = params.lr_finetune_backbone

    if params.pretrain_dataset in ["appleall", "max_v2"]:
        params.dataset = params.pretrain_dataset

    if params.model_uncertainty == "gaussian_classification":
        params.discretize_hr = True
        params.loss = "CrossEntropy" # this achieves better calibration, still need to evaluate why
    elif params.model_uncertainty == "NLE":
        params.loss = "NLE"
        params.discretize_hr = False
    else:
        params.discretize_hr = False


    if params.n_class == 1:
        assert params.loss in ['MSE', 'MAE', 'Huber', 'LogCosh', "NLE"]
    else:
        assert params.loss in ['CrossEntropy']

    if params.dataset in ['max', 'apple', 'apple100', 'capture24', 'capture24all', 'm2sleep', 'm2sleep100', "parkinson100", "IEEE", "appleall", "max_v2", "max_hrv"]:
        params.n_feature = 3
    
    if params.backbone == "Transformer":
        # limit batch size to 128 for transformer, otherwise it will run out of memory
        params.pretrain_batch_size = min(params.pretrain_batch_size, 128)
        params.batch_size = min(params.batch_size, 128)
    elif params.backbone == "HRCTPNet":
        params.scheduler_finetune = "WarmupRoot"

    params.add_frequency = True if params.backbone == "FrequencyCorNET" else False

    # set up default hyper-parameters
    if params.framework == 'byol':
        params.weight_decay_pretrain = 1.5e-6
        params.criterion = 'cos_sim'
        params.n_channels_out = params.n_class
    if params.framework == 'simsiam':
        params.weight_decay_pretrain = 1e-4
        params.EMA = 0.0
        params.lr_mul = 1.0
        params.criterion = 'cos_sim'
        params.n_channels_out = params.n_class
    if params.framework in ['simclr', 'nnclr']:
        params.criterion = 'NTXent'
        params.weight_decay_pretrain = 1e-6
        params.n_channels_out = params.n_class
    if params.framework == 'tstcc':
        params.criterion = 'NTXent'
        params.backbone = 'FCN'
        params.weight_decay_pretrain = 3e-4
        params.n_channels_out = params.n_class
    if params.framework == 'supervised':
        params.lr_finetune_backbone = params.lr
        params.lr_finetune_lstm = params.lr
        params.n_channels_out = params.n_class
    if params.framework == 'reconstruction':
        params.criterion = 'MSE'
        params.n_channels_out = 1
        assert params.backbone in ['AE', 'CNN_AE', "CorNET", "LSTM", "Transformer", "Attention_CNN_AE", "HRCTPNet"]
    if params.framework == 'oxford':
        params.backbone == "ResNET"

    # for backwards compatibility, all these argument are optionally added to the path and only if they are different from the default
    model_name_opt = ""
    model_name_opt += f"_pretrain_subsample_{params.pretrain_subsample:.3f}"  if params.pretrain_subsample != 1.0 else ""
    model_name_opt += f"_windowsize_{params.window_size}" if params.window_size != 10 else ""
    model_name_opt += f"_stepsize_{params.step_size}" if params.step_size != 8 else ""
    model_name_opt += f"_szfactor_test_{params.take_every_nth_test}" if params.take_every_nth_test != 1 else ""
    model_name_opt += f"_szfactor_train_{params.take_every_nth_train}" if params.take_every_nth_train != 1 else ""
    model_name_opt += f"_{params.rnn_type}" if params.rnn_type != "lstm" else ""
    params.model_name = 'try_scheduler_' + params.framework + '_backbone_' + params.backbone +'_pretrain_' + params.pretrain_dataset + '_eps' + str(params.pretrain_n_epoch) + '_lr' + str(params.lr_pretrain) + '_bs' + str(params.pretrain_batch_size) \
                      + '_aug1' + params.aug1 + '_aug2' + params.aug2 + '_dim-pdim' + str(params.p) + '-' + str(params.phid) \
                      + '_EMA' + str(params.EMA) + '_criterion_' + params.criterion + '_lambda1_' + str(params.lambda1) + '_lambda2_' + str(params.lambda2) + '_tempunit_' + params.temp_unit  +  model_name_opt


    # set model saving paths
    params.model_dir_name = os.path.join(config.results_dir, params.model_name)
    os.makedirs(params.model_dir_name, exist_ok=True)

    params.pretrain_model_file = os.path.join(params.model_dir_name, 'pretrain_' + params.model_name  + "_bestmodel" + '.pt')

    # for backwards compatibility, all these argument are optionally added to the path and only if they are different from the default
    lincl_model_name_opt = ""
    lincl_model_name_opt += f"_{params.model_uncertainty}" if params.model_uncertainty not in ["none", "mcdropout", "ensemble"] else ""
    lincl_model_name_opt += f"_take_nth_train_{params.take_every_nth_train}" if params.take_every_nth_train != 1 else ""
    lincl_model_name_opt += f"_take_nth_test_{params.take_every_nth_test}" if params.take_every_nth_test != 1 else ""
    lincl_model_name_opt += f"_timesplit" if params.split_by == "time" else ""
    lincl_model_name_opt += f"_disc_hr_{params.n_class}" if params.discretize_hr else ""
    lincl_model_name_opt += f"_lr_{params.lr_finetune_backbone:.1E}" if params.lr_finetune_backbone != params.lr else ""
    lincl_model_name_opt += f"_lr_lstm_{params.lr_finetune_lstm:.1E}" if params.lr_finetune_lstm != params.lr_finetune_backbone else ""
    lincl_model_name_opt += f"_hrmin_{params.hr_min}" if params.hr_min != 50 else ""
    lincl_model_name_opt += f"_hrmax_{params.hr_max}" if params.hr_max != 110 else ""
    lincl_model_name_opt += f"_rseed_{params.random_seed}" if params.random_seed != 10 else ""
    lincl_model_name_opt += f"_nlayers_{params.num_layers_classifier}" if params.num_layers_classifier != 1 else ""
    lincl_model_name_opt += f"samplingrate_{params.sampling_rate}" if not (params.sampling_rate == 0 or params.sampling_rate == 100) else ""
    lincl_model_name = 'lincls'+ lincl_model_name_opt + "_" + params.backbone + '_dataset_' + params.dataset + '_split' + str(params.split) + '_eps' + str(params.n_epoch) + '_bs' + str(params.batch_size) + "_bestmodel" + '.pt'

    # directories for saving the model, the predictions, and the tsne plot
    params.lincl_model_file = os.path.join(params.model_dir_name, lincl_model_name)
    params.predictions_dir_test = params.lincl_model_file.replace(".pt", "_test.pickle")
    params.predictions_dir_val = params.lincl_model_file.replace(".pt", "_val.pickle")
    params.predictions_dir_post_test = params.lincl_model_file.replace(".pt", f"_test_{params.model_uncertainty}_{params.postprocessing}.pickle")
    params.predictions_dir_post_val = params.lincl_model_file.replace(".pt", f"_val_{params.model_uncertainty}_{params.postprocessing}.pickle")
    params.tsne_dir = params.lincl_model_file.replace("bestmodel.pt", "tsne.png")
    return params
    

def setup(params, DEVICE):
    model, optimizers = setup_model_optm(params, DEVICE)

    # loss fn
    if params.criterion == 'cos_sim':
        criterion = nn.CosineSimilarity(dim=1)
    elif params.criterion == 'NTXent':
        if params.framework == 'tstcc':
            criterion = NTXentLoss(DEVICE, params.batch_size, temperature=0.2)
        else:
            criterion = NTXentLoss(DEVICE, params.batch_size, temperature=0.1)
    elif params.criterion == 'MSE':
        criterion = nn.MSELoss()
    elif params.criterion == 'MAE':
        criterion = nn.L1Loss()


    # Initialize W&B
    wandb_run = wandb.init(
    # Set the project where this run will be logged
    project= params.wandb_project,
    group = params.wandb_group if params.wandb_group != '' else None,
    tags=[params.wandb_tag] if params.wandb_tag != '' else None,
    # Track hyperparameters and run metadata
    config=vars(params),
    mode = params.wandb_mode)

    

    schedulers = []
    for optimizer in optimizers:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.pretrain_n_epoch, eta_min=0)
        schedulers.append(scheduler)

    global nn_replacer
    nn_replacer = None
    if params.framework == 'nnclr':
        nn_replacer = NNMemoryBankModule(size=params.mmb_size)

    global recon
    recon = None
    if params.backbone in ['AE', 'CNN_AE']:
        recon = nn.MSELoss()

    total_params = sum(
        param.numel() for param in model.parameters()
    )

    wandb.log({"Total_Params": total_params}, step=0)


    return model, optimizers, schedulers, criterion

def setup_classifier(params, DEVICE, backbone):


    bb_dim = backbone.out_dim
    classifier = setup_linclf(params, DEVICE, bb_dim)

    # splits the different layers of the model so we can assign different learning rates to them
    lstm_gru_parameters = []
    conv_layers = []
    for name, param in backbone.named_parameters():
        if 'lstm' in name.lower() or 'gru' in name.lower():
            lstm_gru_parameters.append(param)
        else:
            conv_layers.append(param)

    optimizer_params = [
        {"params": conv_layers, "lr": params.lr, "weight_decay": params.weight_decay},
        {"params": lstm_gru_parameters, "lr": params.lr, "weight_decay": params.weight_decay}
    ]

    if params.optimizer == 'Adam':
        optimizer_cls = torch.optim.Adam(optimizer_params, lr=params.lr, weight_decay=params.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {params.optimizer} not implemented")
    
    if params.scheduler_finetune == "WarmupRoot":
        scheduler_cls = torch.optim.lr_scheduler.LambdaLR(optimizer_cls, lr_lambda=lambda epoch: min(1/np.sqrt(epoch), epoch*np.power(4,-1.5)))
    else: # constant original learning rate
        scheduler_cls = torch.optim.lr_scheduler.LambdaLR(optimizer_cls, lr_lambda=lambda epoch: 1)


    if params.loss == 'MSE':
        criterion_cls = nn.MSELoss()
    elif params.loss == 'MAE':
        criterion_cls = nn.L1Loss()
    elif params.loss == 'Huber':
        criterion_cls = nn.HuberLoss(delta=0.1)
    elif params.loss == 'LogCosh':
        criterion_cls = LogCoshError()
    elif params.loss == 'CrossEntropy':
        criterion_cls = nn.CrossEntropyLoss(reduction='mean')
    elif params.loss == 'BinaryCrossEntropy':
        criterion_cls = nn.BCELoss(reduction='mean')
    elif params.loss == 'NLE':
        criterion_cls = NLELoss(ratio = 0)
    else:
        NotImplementedError

    return classifier, criterion_cls, optimizer_cls, scheduler_cls


def calculate_model_loss(params, sample, target, model, criterion, DEVICE, recon=None, nn_replacer=None):

    target = target.to(DEVICE).float()

    if params.framework == 'reconstruction':
        sample = sample.to(DEVICE).float()
        sample_decoded, _ = model(sample)
        loss = criterion(sample_decoded, target) * params.lambda1
    else:
        aug_sample1 = gen_aug(sample, params.aug1)
        aug_sample2 = gen_aug(sample, params.aug2)
        aug_sample1, aug_sample2 = aug_sample1.to(DEVICE).float(), aug_sample2.to(DEVICE).float()
        if params.framework in ['byol', 'simsiam']:
            assert params.criterion == 'cos_sim'
        if params.framework in ['tstcc', 'simclr', 'nnclr']:
            assert params.criterion == 'NTXent'

        if params.framework in ['byol', 'simsiam', 'nnclr']:
            if params.backbone in ['AE', 'CNN_AE']:
                x1_encoded, x2_encoded, p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
                recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
            else:
                p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            if params.framework == 'nnclr':
                z1 = nn_replacer(z1, update=False)
                z2 = nn_replacer(z2, update=True)
            if params.criterion == 'cos_sim':
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            elif params.criterion == 'NTXent':
                loss = (criterion(p1, z2) + criterion(p2, z1)) * 0.5
            if params.backbone in ['AE', 'CNN_AE']:
                loss = loss * params.lambda1 + recon_loss * params.lambda2
        if params.framework == 'simclr':
            if params.backbone in ['AE', 'CNN_AE']:
                x1_encoded, x2_encoded, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
                recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
            else:
                z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            loss = criterion(z1, z2)
            if params.backbone in ['AE', 'CNN_AE']:
                loss = loss * params.lambda1 + recon_loss * params.lambda2
        if params.framework == 'tstcc':
            nce1, nce2, p1, p2 = model(x1=aug_sample1, x2=aug_sample2)
            tmp_loss = nce1 + nce2
            ctx_loss = criterion(p1, p2)
            loss = tmp_loss * params.lambda1 + ctx_loss * params.lambda2
        else: #params.framework == 'supervised'
            raise NotImplementedError
        

    return loss


def train(train_loader, val_loader, model, DEVICE, optimizers, schedulers, criterion, params):
    # training and validation
    best_model = None
    min_val_loss = 1e8
    num_epochs = params.n_epoch

    for epoch in range(params.pretrain_n_epoch):
        total_loss = 0
        n_batches = 0
        model.train()

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit='batch') as tepoch:
            for idx, (sample, target, domain) in enumerate(tepoch):

                n_batches += 1
                for optimizer in optimizers:
                    optimizer.zero_grad()
                if sample.size(0) != params.batch_size:
                    continue
                loss = calculate_model_loss(params, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
                total_loss += loss.item()
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                if params.framework in ['byol', 'simsiam']:
                    model.update_moving_average()

                tepoch.set_postfix(pretrain_loss=loss.item())

        for scheduler in schedulers:
            scheduler.step()

        wandb.log({'pretrain_training_loss': total_loss / n_batches, 'pretrain_lr': optimizers[0].param_groups[0]['lr']}, step=epoch)

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
                        if sample.size(0) != params.batch_size:
                            continue
                        n_batches += 1
                        loss = calculate_model_loss(params, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
                        total_loss += loss.item()
                if total_loss <= min_val_loss:
                    min_val_loss = total_loss
                    best_model = copy.deepcopy(model.state_dict())
                    if params.save_model:
                        print('update: Saving model at {} epoch to {}'.format(epoch, params.pretrain_model_file))
                        torch.save({'model_state_dict': model.state_dict()}, params.pretrain_model_file)
                    
                wandb.log({"pretrain_validation_loss": total_loss / n_batches}, step=epoch)
                
    return best_model


def load_best_model(params):

    model_dir = os.path.join(params.model_dir_name, 'pretrain_' + params.model_name + "_bestmodel" + '.pt')

    if not os.path.exists(model_dir):
        raise FileNotFoundError("No model found at {}".format(model_dir))
    
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

def load_best_lincls(params, device=None):

    if device is None:
        device = torch.device('cuda:' + str(params.cuda) if torch.cuda.is_available() else 'cpu')

    model_dir = params.lincl_model_file
    if not os.path.exists(model_dir):
        raise FileNotFoundError("No model found at {}".format(model_dir))
    best_model = torch.load(model_dir, map_location=device)['trained_backbone']

    return best_model



def test(test_loader, model, DEVICE, criterion, params):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        n_batches = 0
        with tqdm(test_loader, desc=f"Test", unit='batch') as tepoch:
            for idx, (sample, target, domain) in enumerate(tepoch):
                if sample.size(0) != params.batch_size:
                    continue
                n_batches += 1
                loss = calculate_model_loss(params, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
                total_loss += loss.item()
        wandb.log({"pretrain_test_loss": total_loss / n_batches})

def extract_backbone(model, params):
    # Extracts the encoder part of the model

    if params.framework in ['simsiam', 'byol']:
        trained_backbone = model.online_encoder.net
    elif params.framework in ['simclr', 'nnclr', 'tstcc']:
        trained_backbone = model.encoder
    elif params.framework == 'supervised':
        trained_backbone = model
    elif params.framework == 'reconstruction':
        trained_backbone = model
    elif params.framework == 'oxford':
        trained_backbone = model
    else:
        raise NotImplementedError(f"Framework {params.framework} not implemented")
    
    if params.backbone in ["AE", "CNN_AE"]:
        trained_backbone = trained_backbone.encoder

    return trained_backbone

def calculate_lincls_output(sample, target, trained_backbone, criterion, params):
    # calculate the output of the linear classifier and computes the loss

    output, feat = trained_backbone(sample)
    if len(feat.shape) == 3:
        feat = feat.reshape(feat.shape[0], -1)
    loss = criterion(output, target)

    feat = feat.detach().cpu().numpy()
    if "bnn" in params.model_uncertainty:
        kl_loss = get_kl_loss(trained_backbone)/sample.shape[0]
        loss += kl_loss
        predicted = output.data
    elif params.model_uncertainty == "NLE":
        #predicted = output[..., 0].data
        predicted = output[0].data
    else:
        predicted = output.data
    

    return loss, predicted, feat


def add_probability_wrapper(model, params, DEVICE):
    # Adds the probability wrapper to the model
    # The probability wrapper is customized for different uncertainty models 
    # and outputs
    # - expected value of prediction 
    # - uncertainty of prediction (determined by uncertainty_model parameter, currently only supports std)
    # - discrete probability distribution of prediction (determined by n_classes parameter, only if return_probs is True)

    if params.model_uncertainty == "mcdropout":
        model = MC_Dropout_Wrapper(model, n_classes = params.n_prob_class, n_samples = 100, return_probs=True)
        
    elif params.model_uncertainty in ["bnn", "bnn_pretrained", "bnn_pretrained_firstlast"]:
        model = BNN_Wrapper(model, n_classes=params.n_prob_class, n_samples=100, return_probs=True)

    elif params.model_uncertainty == "NLE":
        model = NLE_Wrapper(model, n_classes=params.n_prob_class, return_probs=True)

    elif params.model_uncertainty == "gaussian_classification":    
        model = Uncertainty_Wrapper(model, n_classes=params.n_prob_class, return_probs=True)
    
    elif params.model_uncertainty == "ensemble":
        # load all models in the ensemble, and create an ensemble wrapper
        # only integer random seeds between 0 and 100 supported
        model_path = params.lincl_model_file
        model_paths_seed = [model_path]
        for seed in range(100):
            if "rseed" in model_path:
                model_path_seed = re.sub(r'_rseed_\d+', f'_rseed_{seed}', model_path)
            else:
                #model_path_seed = re.sub(r'hrmax_\d_',  r'\1' + f'_rseed_{seed}.pt', model_path)
                model_path_seed = re.sub(r'(hrmax_(\d+))', r'\1_rseed' + f"_{seed}", model_path)
            if os.path.exists(model_path_seed):
                model_paths_seed.append(model_path_seed)
        print(f"Found {len(model_paths_seed)} models for ensemble")
    
        model = Ensemble_Wrapper(model, model_paths_seed, n_classes=params.n_prob_class, return_probs=True)

    else: # simple regression, outputs a delta distribution
        model = Uncertainty_Regression_Wrapper(model, n_classes=params.n_prob_class, return_probs=True)

    model = model.to(DEVICE)

    return model


def convert_to_hr(values, params):
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()


    if values.ndim > 1 and values.shape[1] > 1: # if we have more than one class

        # construct the bins
        bins = np.linspace(0, 1, params.n_prob_class-1)
        # lookup the values in the bins
        lookup_values = np.concatenate([[bins[0]] , (bins[1:] + bins[:-1])/2, [bins[-1]]])
        # convert to HR computing expectation
        values = np.dot(values, lookup_values)
    
    values = values * (params.hr_max - params.hr_min) + params.hr_min
    return values


def train_lincls(train_loader, val_loader, trained_backbone , DEVICE, optimizer, criterion, scheduler, params):
    best_model = None
    min_val_corr = -1
    best_model = copy.deepcopy(trained_backbone.state_dict())

    for epoch in range(params.n_epoch):
        trained_backbone.train() # TODO: remove this
        train_loss = 0
        hr_true, hr_pred = [], []
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{params.n_epoch} - Training Lincls", unit='batch') as tepoch:
            for idx, (sample, target, domain) in enumerate(tepoch):
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).float().reshape(target.shape[0], -1)
                loss, predicted, _, = calculate_lincls_output(sample, target, trained_backbone, criterion, params)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predicted = convert_to_hr(predicted, params)
                target = convert_to_hr(target, params)

                hr_true.extend(target.squeeze())
                hr_pred.extend(predicted.squeeze())

                tepoch.set_postfix(loss=loss.item())
        # save model
        hr_true, hr_pred = np.array(hr_true), np.array(hr_pred)
        mae_train = np.abs(hr_true - hr_pred).mean()
        if len(train_loader) != 0:
            train_loss = train_loss / len(train_loader)
        corr_train = corr_function(hr_true, hr_pred)
        wandb.log({'Train_Loss': train_loss, 'Train_MAE': mae_train, 'Train_Corr': corr_train}, step=epoch)


        if params.scheduler_finetune:
            scheduler.step()

        if val_loader is None:
            with torch.no_grad():
                best_model = copy.deepcopy(trained_backbone.state_dict())
                if params.save_model:
                    print('update: Saving model at {} epoch to {}'.format(epoch, params.lincl_model_file))
                    torch.save({'trained_backbone': trained_backbone.state_dict(), 'classifier': trained_backbone.classifier.state_dict()}, params.lincl_model_file)
        else:
            with torch.no_grad():
                val_loss = 0
                n_batches = 0
                hr_true, hr_pred, pids = [], [], []
                with tqdm(val_loader, desc=f"Validation Lincls", unit='batch') as tepoch:
                    for idx, (sample, target, domain) in enumerate(tepoch):
                        n_batches += 1
                        sample, target = sample.to(DEVICE).float(), target.to(DEVICE).float().reshape(target.shape[0], -1)
                        loss, predicted, _ = calculate_lincls_output(sample, target, trained_backbone, criterion, params)
                        val_loss += loss.item()

                        predicted = convert_to_hr(predicted, params)
                        target = convert_to_hr(target, params)

                        hr_true.extend(target.squeeze())
                        hr_pred.extend(predicted.squeeze())
                        pids.extend(domain.cpu().numpy().squeeze())
                hr_true, hr_pred = np.array(hr_true), np.array(hr_pred)
                mae_val = np.abs(hr_true - hr_pred).mean()
                corr_val = corr_function(hr_true, hr_pred)
                val_loss = val_loss / n_batches

                logging_table = pd.DataFrame({ 
                    "hr_true": hr_true, 
                    "hr_pred": hr_pred,
                    "pid": pids
                    })

                wandb.log({'Val_Loss': val_loss, 'Val_MAE': mae_val, 'Val_Corr': corr_val}, step=epoch)
                if corr_val >= min_val_corr:
                    min_val_corr = corr_val
                    best_model = copy.deepcopy(trained_backbone.state_dict())
                    if params.save_model:
                        torch.save({'trained_backbone': trained_backbone.state_dict(), 'classifier': trained_backbone.classifier.state_dict()}, params.lincl_model_file)
                        print('Saving models and results at {} epoch to {}'.format(epoch, params.predictions_dir_val))
                        logging_table.to_pickle(params.predictions_dir_val)

    return best_model


def test_lincls(test_loader, trained_backbone, DEVICE, criterion, params):

    total_loss = 0
    feats = None
    hr_true, hr_pred, pids, feats = [], [], [], []
    total = 0
    with torch.no_grad():
        trained_backbone.eval()
        with tqdm(test_loader, desc=f"Test Lincls", unit='batch') as tepoch:
            for idx, (sample, target, domain) in enumerate(tepoch):
                total += 1
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).float().reshape(target.shape[0], -1)
                loss, predicted, feat = calculate_lincls_output(sample, target, trained_backbone, criterion, params)
                total_loss += loss.item()

                predicted = convert_to_hr(predicted, params)
                target = convert_to_hr(target, params)

                feats.append(feat)
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
        feats = np.concatenate(feats, axis=0)

        for pid in np.unique(pids[:,0]):
            hr_true_pid = np.array(hr_true)[np.where(pids[:,0] == pid)]
            hr_pred_pid = np.array(hr_pred)[np.where(pids[:,0] == pid)]
            mae_pid = np.abs(hr_true_pid - hr_pred_pid).mean()
            corr_pid = corr_function(hr_true_pid, hr_pred_pid)
            results_per_pid.append((pid, mae_pid, corr_pid))
            wandb.log({f"Test_MAE_{pid}": mae_pid, f"Test_Corr_{pid}": corr_pid})
        
        hr_true, hr_pred = np.array(hr_true), np.array(hr_pred)
        
        figure = plot_true_pred(hr_true, hr_pred, x_lim=[params.hr_min, params.hr_max], y_lim=[params.hr_min, params.hr_max])
        wandb.log({"true_pred_test": figure})
        
        mae_test = np.abs(hr_true - hr_pred).mean()
        corr_val = corr_function(hr_true, hr_pred)
        wandb.log({'Test_Loss': total_loss, 'Test_MAE': mae_test, "Test_Corr": corr_val})
        if params.save_model:
            print('Saving results to {}'.format(params.predictions_dir_test))
            logging_table.to_pickle(params.predictions_dir_test)


    if params.plot_tsne:
        tsne(feats, hr_true, save_dir=params.tsne_dir)
        print('tSNE plot saved to ', params.tsne_dir)

def test_postprocessing(test_loader, model, postprocessing, DEVICE, criterion_cls, params, prefix="Test"):
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

                predicted_hr_post = convert_to_hr(pred_post, params)
                predicted_hr = convert_to_hr(pred, params)
                target_hr = convert_to_hr(target, params)


                hr_true.extend(target_hr.reshape(-1))
                hr_pred.extend(predicted_hr.reshape(-1))
                hr_pred_post.extend(predicted_hr_post.reshape(-1))
                pids.extend(domain.numpy().reshape(-1, 2))
                uncertainties.extend(uncertainty.numpy().reshape(-1))
                target_probs.extend(target.numpy().reshape(-1, params.n_prob_class))

                probs_post.extend(prob_post.numpy().reshape(-1, params.n_prob_class))
                probs.extend(prob.numpy().reshape(-1, params.n_prob_class))

        
            logging_table = { 
                        "hr_true": hr_true, 
                        "hr_pred": hr_pred,
                        "hr_pred_post": hr_pred_post,
                        "pid": pids,
                        "uncertainty": uncertainties,
                        }
            if params.save_probabilities:
                logging_table["probs_post"] = probs_post
                logging_table["probs"] = probs

            hr_true, hr_pred, hr_pred_post, probs, probs_post, target_probs = np.array(hr_true), np.array(hr_pred), np.array(hr_pred_post), np.array(probs), np.array(probs_post), np.array(target_probs)
            total_loss = total_loss / len(test_loader)
            figure = plot_true_pred(hr_true, hr_pred, x_lim=[params.hr_min, params.hr_max], y_lim=[params.hr_min, params.hr_max])
            wandb.log({f"true_pred_{prefix}_post": figure})
            
            mae = np.abs(hr_true - hr_pred).mean()
            mae_post = np.abs(hr_true - hr_pred_post).mean()
            corr = corr_function(hr_true, hr_pred)
            corr_post = corr_function(hr_true, hr_pred_post)
            ece = ece_loss(probs, target_probs)
            ece_post = ece_loss(probs_post, target_probs)
            wandb.log({f'{prefix}_Loss': total_loss, f'{prefix}_MAE': mae, f"{prefix}_Corr": corr, f"Post_{prefix}_MAE": mae_post, f"Post_{prefix}_Corr_Post": corr_post, f"{prefix}_ECE": ece, f"Post_{prefix}_ECE": ece_post})
            
            if params.save_model:
                if prefix.lower() =="test":
                    print('Saving results to {}'.format(params.predictions_dir_post_test))
                    pd.DataFrame(logging_table).to_pickle(params.predictions_dir_post_test)
                else: #val
                    print('Saving results to {}'.format(params.predictions_dir_post_val))
                    pd.DataFrame(logging_table).to_pickle(params.predictions_dir_post_val)

def setup_postprocessing_model(params):
    # BeliefPPG contains Viterbi (offline) and Sum-Product (online) as methods, by default we choose sum-product
    if params.postprocessing == "beliefppg" or params.postprocessing == "sumprod":
        postprocessing_model = BeliefPPG(
                    params.n_prob_class, 
                    return_probs= True,
                    uncert = "std",
                    method="sumprod")
        
    elif params.postprocessing == "viterbi":
        postprocessing_model = BeliefPPG(
            params.n_prob_class, 
            return_probs= True,
            uncert = "std",
            method="viterbi")

    # No postprocessing, good for comparison of uncertainty models
    elif params.postprocessing == "raw":
        postprocessing_model = Postprocessing(
            params.n_prob_class, 
            return_probs= True,
            uncert = "std")
        
    # Non-probabilistic postprocessing
    elif params.postprocessing == "kalmansmoothing":
        postprocessing_model = KalmanSmoothing(
            params.n_prob_class, 
            return_probs= True,
            uncert = "std",
            step_size=params.step_size,)


    else:
        raise NotImplementedError
    return postprocessing_model

def train_postprocessing(train_loader, postprocessing_model, params):

    def cut_prob(probs):
        exp = np.round(np.dot(probs, np.arange(params.n_prob_class))).astype(int)
        probs = np.eye(params.n_prob_class)[exp]
        return probs


    ys = [target for _, target, _ in train_loader]
    # Sometimes we get a Value Error here. This is because the probabilities for some classes are too small. Taking the log will then result in infinite values.
    # We can fix this by cutting the probabilities to the nearest class
    if len(ys) > 0 and False:
        postprocessing_model.fit_layer(ys, distr=params.transition_distribution)
    else:
        print("No data found for training postprocessing model. Loading pre-trained model")
        transition_prior_path = os.path.join(config.results_dir, "hr_state_transition_prior.pt")
        postprocessing_model.set_transition_prior(torch.load(transition_prior_path))

    return postprocessing_model


def predict_median(train_loader, val_loader, test_loader, params, mode="global"):

    assert mode in ["global", "subject_wise"]

    # predicts the meidan of the training HR values
    wandb_run = wandb.init(
    # Set the project where this run will be logged
    project= params.wandb_project,
    group = params.wandb_group if params.wandb_group != '' else None,
    tags=[params.wandb_tag] if params.wandb_tag != '' else None,
    # Track hyperparameters and run metadata
    config=vars(params),
    mode = params.wandb_mode)

    if mode=="global":
        for _, target, pid in train_loader:
            target = convert_to_hr(target, params)
            hr_true_train.extend(target.squeeze())
        hr_true_train = np.array(hr_true_train)
        hr_median = np.median(hr_true_train)
        hr_pred_train = np.repeat(hr_median, len(hr_true_train))
        mae_train = np.abs(hr_true_train - hr_pred_train).mean()
        corr_train = corr_function(hr_true_train, hr_pred_train)
        wandb.log({'Train_MAE': mae_train, 'Train_Corr': corr_train})

        hr_true_val = []
        for _,target, _ in val_loader:

            target = convert_to_hr(target, params)
            hr_true_val.extend(target.squeeze())

        hr_true_val = np.array(hr_true_val)
        hr_pred_val = np.repeat(hr_median, len(hr_true_val))
        mae_train = np.abs(hr_true_val - hr_pred_val).mean()
        corr_train = corr_function(hr_true_val, hr_pred_val)

        wandb.log({'Val_MAE': mae_train, 'Val_Corr': corr_train})

        hr_true_test = []

        for _, target, _ in test_loader:
            target = convert_to_hr(target, params)
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

    

