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
from data_preprocess import data_preprocess_hhar, data_preprocess_hr

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

corr = lambda a, b: pd.DataFrame({'a':a, 'b':b}).corr().iloc[0,1]

def plot_true_pred(hr_true, hr_pred, x_lim=[20, 120], y_lim=[20, 120]):
    figure = plt.figure(figsize=(8, 8))
    hr_true, hr_pred = np.array(hr_true), np.array(hr_pred)
    mae = np.round(np.abs(hr_true - hr_pred).mean(), 2)
    correlation_coefficient = corr(hr_true, hr_pred)

    plt.scatter(x = hr_true, y = hr_pred, alpha=0.2, label=f"MAE: {mae}, Corr: {correlation_coefficient}")

    plt.plot(x_lim, y_lim, color='k', linestyle='-', linewidth=2)
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.xlabel('True HR (bpm)')
    plt.ylabel('Predicted HR (bpm)')
    plt.legend()
    return figure
    

def setup_dataloaders(args, pretrain=False):

    if pretrain:
        dataset = args.pretrain_dataset
        split = 0 # pretrains network always with split 0
    else: # normal dataset, not pretraining
        dataset = args.dataset
        split = args.split

    if dataset == 'hhar':
        args.n_feature = 6
        args.len_sw = 100
        args.n_class = 6
        source_domain = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        # source_domain.remove(args.target_domain)
        train_loaders, val_loader, test_loader = data_preprocess_hhar.prep_hhar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(args.len_sw * 0.5),
                                                                                device=args.device,
                                                                                train_user=source_domain,
                                                                                test_user=args.target_domain)
    elif dataset == 'max' or dataset == 'apple' or dataset == 'capture24' or dataset == 'm2sleep':
        args.n_feature = 3
        args.n_class = 1
        
        if dataset == 'max':
            args.input_length = 1000
            args.len_sw = 1000
        elif dataset == "apple":
            args.input_length = 500
            args.len_sw = 500
        elif dataset == "m2sleep":
            args.input_length = 320
            args.len_sw = 320
        else: # capture24
            args.input_length = 1000
            args.len_sw = 1000

        train_loaders, val_loader, test_loader = data_preprocess_hr.prep_hr(args, dataset=dataset, split=split)
    
   
    else:
        NotImplementedError(dataset)

    return train_loaders, val_loader, test_loader


def setup_linclf(args, DEVICE, bb_dim):
    '''
    @param bb_dim: output dimension of the backbone network
    @return: a linear classifier
    '''
    classifier = Classifier(bb_dim=bb_dim, n_classes=args.n_class)
    classifier.classifier.weight.data.normal_(mean=0.0, std=0.01)
    classifier.classifier.bias.data.zero_()
    classifier = classifier.to(DEVICE)
    return classifier


def setup_model_optm(args, DEVICE, classifier=True):

    # set up backbone network
    if args.backbone == 'FCN':
        backbone = FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=True)
    elif args.backbone == 'DCL':
        backbone = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=True)
    elif args.backbone == 'LSTM':
        backbone = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=True)
    elif args.backbone == 'AE':
        backbone = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=True)
    elif args.backbone == 'CNN_AE':
        backbone = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=True)
    elif args.backbone == 'Transformer':
        backbone = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True)
    elif args.backbone == 'CorNET':
        backbone = CorNET(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=32, kernel_size=40, LSTM_units=128, backbone=True)
    else:
        NotImplementedError

    # set up model and optimizers
    if args.framework in ['byol', 'simsiam']:
        model = BYOL(DEVICE, backbone, window_size=args.len_sw, n_channels=args.n_feature, projection_size=args.p,
                     projection_hidden_size=args.phid, moving_average=args.EMA)
        optimizer1 = torch.optim.Adam(model.online_encoder.parameters(),
                                      args.lr,
                                      weight_decay=args.weight_decay)
        optimizer2 = torch.optim.Adam(model.online_predictor.parameters(),
                                      args.lr * args.lr_mul,
                                      weight_decay=args.weight_decay)
        optimizers = [optimizer1, optimizer2]
    elif args.framework == 'simclr':
        model = SimCLR(backbone=backbone, dim=args.p)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizers = [optimizer]
    elif args.framework == 'nnclr':
        model = NNCLR(backbone=backbone, dim=args.p, pred_dim=args.phid)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizers = [optimizer]
    elif args.framework == 'tstcc':
        model = TSTCC(backbone=backbone, DEVICE=DEVICE, temp_unit=args.temp_unit, tc_hidden=100)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        optimizers = [optimizer]

    else:
        NotImplementedError

    model = model.to(DEVICE)

    # set up linear classfier
    if classifier:
        bb_dim = backbone.out_dim
        classifier = setup_linclf(args, DEVICE, bb_dim)
        return model, classifier, optimizers

    else:
        return model, optimizers


def delete_files(args):
    for epoch in range(args.n_epoch):
        model_dir = os.path.join(model_dir_name, 'pretrain_' + args.model_name + str(epoch) + '.pt')
        if os.path.isfile(model_dir):
            os.remove(model_dir)

        cls_dir = os.path.join(model_dir_name, 'lincls_' + args.model_name + str(epoch) + '.pt')
        if os.path.isfile(cls_dir):
            os.remove(cls_dir)


def setup(args, DEVICE):
    # set up default hyper-parameters
    if args.framework == 'byol':
        args.weight_decay = 1.5e-6
    if args.framework == 'simsiam':
        args.weight_decay = 1e-4
        args.EMA = 0.0
        args.lr_mul = 1.0
    if args.framework in ['simclr', 'nnclr']:
        args.criterion = 'NTXent'
        args.weight_decay = 1e-6
    if args.framework == 'tstcc':
        args.criterion = 'NTXent'
        args.backbone = 'FCN'
        args.weight_decay = 3e-4

    model, classifier, optimizers = setup_model_optm(args, DEVICE, classifier=True)

    # loss fn
    if args.criterion == 'cos_sim':
        criterion = nn.CosineSimilarity(dim=1)
    elif args.criterion == 'NTXent':
        if args.framework == 'tstcc':
            criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.2)
        else:
            criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.1)

    args.model_name = 'try_scheduler_' + args.framework + '_backbone_' + args.backbone +'_pretrain_' + args.pretrain_dataset + '_eps' + str(args.n_epoch) + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) \
                      + '_aug1' + args.aug1 + '_aug2' + args.aug2 + '_dim-pdim' + str(args.p) + '-' + str(args.phid) \
                      + '_EMA' + str(args.EMA) + '_criterion_' + args.criterion + '_lambda1_' + str(args.lambda1) + '_lambda2_' + str(args.lambda2) + '_tempunit_' + args.temp_unit

    # log
    if os.path.isdir(args.logdir) == False:
        os.makedirs(args.logdir)
    log_file_name = os.path.join(args.logdir, args.model_name + f".log")
    logger = _logger(log_file_name)
    logger.debug(args)


    # Initialize W&B
    run = wandb.init(
    # Set the project where this run will be logged
    project="hr_ssl",
    # Track hyperparameters and run metadata
    config=vars(args),
    mode = args.wandb_mode)


    criterion_cls = nn.MSELoss()
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr_cls)

    schedulers = []
    for optimizer in optimizers:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=0)
        schedulers.append(scheduler)

    global nn_replacer
    nn_replacer = None
    if args.framework == 'nnclr':
        nn_replacer = NNMemoryBankModule(size=args.mmb_size)

    global recon
    recon = None
    if args.backbone in ['AE', 'CNN_AE']:
        recon = nn.MSELoss()

    return model, optimizers, schedulers, criterion, logger, classifier, criterion_cls, optimizer_cls


def calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=None, nn_replacer=None):
    aug_sample1 = gen_aug(sample, args.aug1)
    aug_sample2 = gen_aug(sample, args.aug2)
    aug_sample1, aug_sample2, target = aug_sample1.to(DEVICE).float(), aug_sample2.to(DEVICE).float(), target.to(
        DEVICE).long()
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
    return loss


def train(train_loader, val_loader, model, logger, DEVICE, optimizers, schedulers, criterion, args):
    best_model = None
    min_val_loss = 1e8
    num_epochs = args.n_epoch

    for epoch in range(args.n_epoch):
        logger.debug(f'\nEpoch : {epoch}')
        total_loss = 0
        n_batches = 0
        model.train()

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit='batch') as tepoch:
            for idx, (sample, target, domain) in enumerate(tepoch):
            
                for optimizer in optimizers:
                    optimizer.zero_grad()
                if sample.size(0) != args.batch_size:
                    continue
                n_batches += 1
                loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
                total_loss += loss.item()
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                if args.framework in ['byol', 'simsiam']:
                    model.update_moving_average()
        wandb.log({'lr': optimizers[0].param_groups[0]['lr']}, step=epoch)
        
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

    if os.path.exists(model_dir) == False:
        print("No model found at {}".format(model_dir))
        return None
    
    print('Loading model from {}'.format(model_dir))
    best_model = torch.load(model_dir)["model_state_dict"]
    return best_model

def test(test_loader, best_model, logger, DEVICE, criterion, args):
    model, _ = setup_model_optm(args, DEVICE, classifier=False)
    model.load_state_dict(best_model)
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
    return model


def lock_backbone(model, args):
    for name, param in model.named_parameters():
        param.requires_grad = False

    if args.framework in ['simsiam', 'byol']:
        trained_backbone = model.online_encoder.net
    elif args.framework in ['simclr', 'nnclr', 'tstcc']:
        trained_backbone = model.encoder
    else:
        NotImplementedError

    return trained_backbone


def calculate_lincls_output(sample, target, trained_backbone, classifier, criterion):
    _, feat = trained_backbone(sample)
    if len(feat.shape) == 3:
        feat = feat.reshape(feat.shape[0], -1)
    output = classifier(feat)
    loss = criterion(output, target)
    predicted = output.data # regression
    #_, predicted = torch.max(output.data, 1)
    return loss, predicted, feat


def train_lincls(train_loader, val_loader, trained_backbone, classifier, logger , DEVICE, optimizer, criterion, args):
    best_lincls = None
    min_val_loss = 1e8

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=0)

    for epoch in range(args.n_epoch):
        classifier.train()
        logger.debug(f'\nEpoch : {epoch}')
        total_loss = 0
        mae = 0
        total = 0
        hr_true, hr_pred = [], []
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.n_epoch} - Training Lincls", unit='batch') as tepoch:
            for idx, (sample, target, domain) in enumerate(tepoch):
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).float().reshape(-1, 1)
                loss, predicted, _ = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion)
                total_loss += loss.item()
                mae += torch.abs(predicted - target).mean() * (args.hr_max - args.hr_min)
                total += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                hr_true.extend(target.cpu().numpy().squeeze() * (args.hr_max - args.hr_min) + args.hr_min)
                hr_pred.extend(predicted.cpu().numpy().squeeze() * (args.hr_max - args.hr_min) + args.hr_min)


        # save model
        mae_train = mae / total
        corr_train = np.round(corr(hr_true, hr_pred),3)
        model_dir = os.path.join(args.model_dir_name, 'lincls_' + args.model_name + "_split" + str(args.split) + "_" + str(epoch) + '.pt')
        #print('Saving model at {} epoch to {}'.format(epoch, model_dir))
        #torch.save({'trained_backbone': trained_backbone.state_dict(), 'classifier': classifier.state_dict()}, model_dir)


        logger.debug(f'epoch train loss     : {total_loss:.4f}\t | \tepoch train MAE    : {mae_train:2.4f}\n')
        wandb.log({'Train_Loss': total_loss, 'Train_MAE': mae_train, 'Train_Corr': corr_train}, step=epoch)


        if args.scheduler:
            scheduler.step()

        if val_loader is None:
            with torch.no_grad():
                best_lincls = copy.deepcopy(classifier.state_dict())
        else:
            with torch.no_grad():
                classifier.eval()
                val_loss = 0
                total = 0
                mae = 0
                hr_true, hr_pred, pids = [], [], []
                with tqdm(val_loader, desc=f"Validation Lincls", unit='batch') as tepoch:
                    for idx, (sample, target, domain) in enumerate(tepoch):
                        total += 1
                        sample, target = sample.to(DEVICE).float(), target.to(DEVICE).float()
                        loss, predicted, _ = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion)
                        total_loss += loss.item()
                        mae = torch.abs(predicted - target).mean() * (args.hr_max - args.hr_min)
                        hr_true.extend(target.cpu().numpy().squeeze() * (args.hr_max - args.hr_min) + args.hr_min)
                        hr_pred.extend(predicted.cpu().numpy().squeeze() * (args.hr_max - args.hr_min) + args.hr_min)
                        pids.extend(domain.cpu().numpy().squeeze())
                mae_val = mae / total
                corr_val = np.round(corr(hr_true, hr_pred),3)

                logging_table = pd.DataFrame({ 
                    "hr_true": hr_true, 
                    "hr_pred": hr_pred,
                    "pid": pids
                    })
                
                wandb.log({"hr_true_vs_pred_val": wandb.Table(dataframe = pd.DataFrame(logging_table))}, step=epoch)
                figure = plot_true_pred(hr_true, hr_pred)
                wandb.log({"true_pred_val": figure}, step=epoch)
                logger.debug(f'epoch val loss     : {val_loss:.4f}, val mae     : {mae_val:.4f}, val corr     : {corr_val:.4f}\n')
                wandb.log({'Val_Loss': val_loss, 'Val_MAE': mae_val, 'Val_Corr': corr_val}, step=epoch)

                if val_loss <= min_val_loss:
                    min_val_loss = val_loss
                    best_lincls = copy.deepcopy(classifier.state_dict())
                    model_dir = os.path.join(args.model_dir_name, 'lincls_' + args.model_name + "_split" + str(args.split) + "_bestmodel" + '.pt')
                    torch.save({'trained_backbone': trained_backbone.state_dict(), 'classifier': classifier.state_dict()}, model_dir)
                    print('Saving models and results at {} epoch to {}'.format(epoch, model_dir))
                    logging_table.to_csv(os.path.join(args.model_dir_name, 'predictions_val.csv'), index=False)

    return best_lincls


def test_lincls(test_loader, trained_backbone, best_lincls, logger, DEVICE, criterion, args, plt=False):
    classifier = setup_linclf(args, DEVICE, trained_backbone.out_dim)
    classifier.load_state_dict(best_lincls)
    total_loss = 0
    mae = 0
    feats = None
    hr_true, hr_pred, pids = [], [], []
    total = 0
    with torch.no_grad():
        classifier.eval()
        with tqdm(test_loader, desc=f"Test Lincls", unit='batch') as tepoch:
            for idx, (sample, target, domain) in enumerate(tepoch):
                total += 1
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                loss, predicted, feat = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion)
                total_loss += loss.item()
                if feats is None:
                    feats = feat
                else:
                    feats = torch.cat((feats, feat), 0)
                hr_true.extend(target.cpu().numpy().squeeze() * (args.hr_max - args.hr_min) + args.hr_min)
                hr_pred.extend(predicted.cpu().numpy().squeeze() * (args.hr_max - args.hr_min) + args.hr_min)
                pids.extend(domain.cpu().numpy().squeeze())
                mae += torch.abs(predicted - target).mean() * (args.hr_max - args.hr_min)
        
        logging_table = pd.DataFrame({ 
                    "hr_true": hr_true, 
                    "hr_pred": hr_pred,
                    "pid": pids
                    }) 
        wandb.log({"hr_true_vs_pred_val": wandb.Table(dataframe = pd.DataFrame(logging_table))})
        figure = plot_true_pred(hr_true, hr_pred)
        wandb.log({"true_pred_test": figure})
        mae_test = mae / total
        corr_val = np.round(corr(hr_true, hr_pred),3)
        wandb.log({'Test_Loss': total_loss, 'Test_MAE': mae_test, "Test_Corr": corr_val})
        logger.debug(f'epoch test loss     : {total_loss:.4f}, test mae     : {mae_test:.4f}')
        print('Saving results to {}'.format(args.model_dir_name))
        logging_table.to_csv(os.path.join(args.model_dir_name, 'predictions_test.csv'), index=False)



    if plt == True:
        tsne(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_tsne.png')
        mds(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_mds.png')
        sns_plot.get_figure().savefig(plot_dir_name + '/' + args.model_name + '_confmatrix.png')
        print('plots saved to ', plot_dir_name)
