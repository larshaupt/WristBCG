import argparse
from trainer import *
import wandb
import json
from config import results_dir
from utils import get_free_gpu



wandb.login()

# Parse command line arguments
##################
parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=-1, type=int, help='cuda device ID，0/1')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers for data loading')
parser.add_argument('--random_seed', default=10, type=int, help='random seed')
parser.add_argument('--pretrain', default=0, type=int, help='if or not to pretrain')
parser.add_argument('--finetune', type=int, default=1, help='if or not to finetune')


# hyperparameter
parser.add_argument('--pretrain_batch_size', type=int, default=128, help='batch size of pretraining')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=60, help='number of training epochs')
parser.add_argument('--lr_pretrain', type=float, default=1e-4, help='learning rate for pretrain')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-7, help='weight decay')
parser.add_argument('--weight_decay_pretrain', type=float, default=1e-7, help='weight decay for pretrain')
parser.add_argument('--scheduler', type=bool, default=False, help='if or not to use a scheduler')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam'], help='optimizer')
parser.add_argument('--loss', type=str, default='Huber', choices=['MSE', 'MAE', 'Huber', 'LogCosh', 'CrossEntropy'], help='loss function')
parser.add_argument('--huber_delta', type=float, default=0.1, help='delta for Huber loss')

# dataset
parser.add_argument('--pretrain_dataset', type=str, default='capture24', choices=['ucihar', 'max', 'apple', 'capture24', 'apple100'], help='name of dataset')
parser.add_argument('--pretrain_subsample', type=float, default=1.0, help='subsample rate for pretraining')
parser.add_argument('--normalize', type=bool, default=True, help='if or not to normalize data')
parser.add_argument('--dataset', type=str, default='apple100', choices=['apple','max', 'm2sleep', "m2sleep100", 'capture24', 'apple100'], help='name of dataset for finetuning')
parser.add_argument('--model_uncertainty', type=str, default="none", choices=["none", "gaussian_classification", "mcdropout", "bnn"], help='which method to use to output a probability distribution')
parser.add_argument('--label_sigma', type=float, default=3.0, help='sigma for gaussian classification')
parser.add_argument('--subsample', type=float, default=1.0, help='subsample rate')
parser.add_argument('--n_feature', type=int, default=3, help='name of feature dimension')
parser.add_argument('--n_class', type=int, default=1, help='number of class')
parser.add_argument('--n_prob_class', type=int, default=64, help='number of class for probability distribution')
parser.add_argument('--split', type=int, default=0, help='split number')
parser.add_argument('--hr_min', type=float, default=20, help='minimum heart rate for training, not needed for pretraining')
parser.add_argument('--hr_max', type=float, default=120, help='maximum heart rate for training, not needed for pretraining')
parser.add_argument('--sampling_rate', type=int, default=0, help='sampling rate of the data. Warning: this will take longer time to load data')
parser.add_argument('--lr_finetune_backbone', type=float, default=1e-5, help='learning rate for finetuning the backbone network')
parser.add_argument('--lr_finetune_lstm', type=float, default=1e-4, help='learning rate for finetuning the lstm layer')

# augmentation
parser.add_argument('--aug1', type=str, default='jit_scal',
                    choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 't_flip', 't_warp', 'resample', 'rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f', 'bioglass'],
                    help='the type of augmentation transformation')
parser.add_argument('--aug2', type=str, default='resample',
                    choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 't_flip', 't_warp', 'resample', 'rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f', 'bioglass'],
                    help='the type of augmentation transformation')

# framework
parser.add_argument('--framework', type=str, default='byol', choices=['byol', 'simsiam', 'simclr', 'nnclr', 'tstcc', 'supervised', 'reconstruction'], help='name of framework')
parser.add_argument('--backbone', type=str, default='CorNET', choices=['FCN', 'DCL', 'LSTM', 'AE', 'CNN_AE', 'Transformer', 'CorNET'], help='name of backbone network')
parser.add_argument('--num_kernels', type=int, default=16, help='number of kernels in CNN')
parser.add_argument('--kernel_size', type=int, default=16, help='kernel size in CNN')
parser.add_argument('--lstm_units', type=int, default=192, help='number of units in LSTM')
parser.add_argument('--criterion', type=str, default='cos_sim', choices=['cos_sim', 'NTXent', 'MSE', 'MAE'],
                    help='type of loss function for contrastive learning')
parser.add_argument('--p', type=int, default=128,
                    help='byol: projector size, simsiam: projector output size, simclr: projector output size')
parser.add_argument('--phid', type=int, default=128,
                    help='byol: projector hidden size, simsiam: predictor hidden size, simclr: na')

# postprocessing
parser.add_argument('--postprocessing', type=str, default='none', choices=['none', 'beliefppg'], help='postprocessing method')
parser.add_argument('--transition_distribution', type=str, default='gauss', choices=['gauss', 'laplace'], help='transition distribution for belief ppg')


# log
parser.add_argument('--logdir', type=str, default='log/', help='log directory')
parser.add_argument('--wandb_mode', type=str, default='online', choices=['offline', 'online', 'disabled', 'dryrun', 'run'],  help='wandb mode')
parser.add_argument('--wandb_group', type=str, default='', help='wandb group')
parser.add_argument('--wandb_project', type=str, default='hr_ssl', help='wandb project')
parser.add_argument('--wandb_tag', type=str, default='', help='wandb name')

# byol
parser.add_argument('--lr_mul', type=float, default=10.0,
                    help='lr multiplier for the second optimizer when training byol')
parser.add_argument('--EMA', type=float, default=0.996, help='exponential moving average parameter')

# nnclr
parser.add_argument('--mmb_size', type=int, default=1024, help='maximum size of NNCLR support set')

# TS-TCC
parser.add_argument('--lambda1', type=float, default=1.0, help='weight for temporal contrastive loss')
parser.add_argument('--lambda2', type=float, default=1.0, help='weight for contextual contrastive loss, also used as the weight for reconstruction loss when AE or CAE being backbone network')
parser.add_argument('--temp_unit', type=str, default='tsfm', choices=['tsfm', 'lstm', 'blstm', 'gru', 'bgru'], help='temporal unit in the TS-TCC')

# plot
parser.add_argument('--plt', type=bool, default=False, help='if or not to plot results')

#saving arguments
parser.add_argument('--save_probabilities', type=bool, default=True, help='if or not to save probabilities')


if __name__ == '__main__':

    ############################################################################################################
    ############################ CONFIGURATION #################################################################
    ############################################################################################################
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Set device, automatically select a free GPU if available and cuda == -1
    if args.cuda == -1:
        args.cuda = int(get_free_gpu())
        print(f"Automatically selected GPU {args.cuda}")

    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    # setup model, optimizer, scheduler, criterion, logger
    model, optimizers, schedulers, criterion, logger = setup(args, DEVICE)
    
    # set model saving paths and saves config file
    args.model_dir_name = os.path.join(results_dir, args.model_name)
    os.makedirs(args.model_dir_name, exist_ok=True)
    with open(os.path.join(args.model_dir_name, "config.json"), "w") as outfile:
        print(f"Saving config file to {args.model_dir_name}")
        json.dump(vars(args), outfile)

    ############################################################################################################
    ############################ PRETRAINING ###################################################################
    ############################################################################################################

    if args.pretrain: # pretraining
        # setup dataloader for pretraining
        train_loaders, val_loader, test_loader = setup_dataloaders(args, pretrain=True)
        print('device:', DEVICE, 'dataset:', args.pretrain_dataset)
        pretrain_model_weights = train(train_loaders, val_loader, model, logger, DEVICE, optimizers, schedulers, criterion, args)
        model.load_state_dict(pretrain_model_weights)

        if len(test_loader) != 0:
            test(test_loader, model, logger, DEVICE, criterion, args)

    else: # no pretraining, load previously trained model
        if args.framework == 'supervised':
            # take random initialization
            pass
        else:
            # load best pretrain model
            pretrain_model_weights = load_best_model(args)
            model.load_state_dict(pretrain_model_weights)

    pretrain_model = model


    ############################################################################################################
    ############################ FINETUNING ####################################################################
    ############################################################################################################

    trained_backbone = extract_backbone(pretrain_model, args)
    classifier, criterion_cls, optimizer_cls  = setup_classifier(args, DEVICE, trained_backbone)

    if args.finetune:

        # setup dataloader for finetuning
        train_loaders, val_loader, test_loader = setup_dataloaders(args, pretrain=False, sample_sequences=False, discrete_hr=args.discretize_hr)
        print('device:', DEVICE, 'dataset:', args.dataset)
        

        trained_backbone.set_classification_head(classifier)
        optimizer_cls.param_groups[0]['lr'] = args.lr_finetune_backbone
        optimizer_cls.param_groups[1]['lr'] = args.lr_finetune_lstm
        optimizer_cls.add_param_group({'params': trained_backbone.classifier.parameters(), 'lr': args.lr})
        
        trained_backbone_weights = train_lincls(train_loaders, val_loader, trained_backbone, logger, DEVICE, optimizer_cls, criterion_cls, args)
        trained_backbone.load_state_dict(trained_backbone_weights)

        if len(test_loader) != 0:
            test_lincls(test_loader, trained_backbone, logger, DEVICE, criterion_cls, args, plt=args.plt)

    else:
        classifier = setup_linclf(args, DEVICE, trained_backbone.out_dim)
        trained_backbone.set_classification_head(classifier)
        trained_backbone_weights = load_best_lincls(args)
        trained_backbone.load_state_dict(trained_backbone_weights)

    ############################################################################################################
    ############################ POSTPROCESSING ################################################################
    ############################################################################################################

    if args.postprocessing != 'none':

        train_loader, val_loader, test_loader = setup_dataloaders(args, pretrain=False, sample_sequences=True, discrete_hr=True)
        trained_backbone = add_probability_wrapper(trained_backbone, args, DEVICE)
        postprocessing_model = setup_postprocessing_model(args)

        postprocessing_model = train_postprocessing(train_loader, postprocessing_model, DEVICE, args)

        test_postprocessing(val_loader, trained_backbone, postprocessing_model,  logger, DEVICE, criterion_cls, args, plt=args.plt, prefix='Val')
        test_postprocessing(test_loader, trained_backbone, postprocessing_model, logger, DEVICE, criterion_cls, args, plt=args.plt, prefix='Test')
        



    # remove saved intermediate models
    delete_files(args)
    wandb.finish()