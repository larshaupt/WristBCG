import argparse
from trainer import *
import wandb
import json
from config import results_dir

wandb.login()

# Parse command line arguments
##################
parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID，0/1')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers for data loading')
parser.add_argument('--pretrain', action='store_true', help='if or not to pretrain the backbone network')

# hyperparameter
parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=60, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_cls', type=float, default=1e-3, help='learning rate for linear classifier')
parser.add_argument('--scheduler', type=bool, default=True, help='if or not to use a scheduler')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')

# dataset
parser.add_argument('--pretrain_dataset', type=str, default='capture24', choices=['ucihar', 'max', 'apple', 'capture24'], help='name of dataset')
parser.add_argument('--dataset', type=str, default='max', choices=['ucihar', 'max', 'apple', 'capture24'], help='name of dataset for finetuning')
parser.add_argument('--n_feature', type=int, default=3, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=1000, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=1, help='number of class')
parser.add_argument('--split', type=int, default=0, help='split number')
parser.add_argument('--hr_min', type=float, default=20, help='minimum heart rate for training, not needed for pretraining')
parser.add_argument('--hr_max', type=float, default=120, help='maximum heart rate for training, not needed for pretraining')

# augmentation
parser.add_argument('--aug1', type=str, default='jit_scal',
                    choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 't_flip', 't_warp', 'resample', 'rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f'],
                    help='the type of augmentation transformation')
parser.add_argument('--aug2', type=str, default='resample',
                    choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 't_flip', 't_warp', 'resample', 'rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f'],
                    help='the type of augmentation transformation')

# framework
parser.add_argument('--framework', type=str, default='byol', choices=['byol', 'simsiam', 'simclr', 'nnclr', 'tstcc'], help='name of framework')
parser.add_argument('--backbone', type=str, default='DCL', choices=['FCN', 'DCL', 'LSTM', 'AE', 'CNN_AE', 'Transformer', 'CorNET'], help='name of backbone network')
parser.add_argument('--criterion', type=str, default='cos_sim', choices=['cos_sim', 'NTXent'],
                    help='type of loss function for contrastive learning')
parser.add_argument('--p', type=int, default=128,
                    help='byol: projector size, simsiam: projector output size, simclr: projector output size')
parser.add_argument('--phid', type=int, default=128,
                    help='byol: projector hidden size, simsiam: predictor hidden size, simclr: na')

# log
parser.add_argument('--logdir', type=str, default='log/', help='log directory')
parser.add_argument('--wandb_mode', type=str, default='online', choices=['offline', 'online', 'disabled', 'dryrun', 'run'],  help='wandb mode')

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


if __name__ == '__main__':
    torch.manual_seed(10)
    np.random.seed(10)
    args = parser.parse_args()
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')

    model, optimizers, schedulers, criterion, logger, classifier, criterion_cls, optimizer_cls = setup(args, DEVICE)

    print('device:', DEVICE, 'dataset:', args.pretrain_dataset)
    train_loaders, val_loader, test_loader = setup_dataloaders(args, pretrain=True)

    args.model_dir_name = os.path.join(results_dir, args.model_name)
    os.makedirs(args.model_dir_name, exist_ok=True)
    with open(os.path.join(args.model_dir_name, "config.json"), "w") as outfile:
        print(f"Saving config file to {args.model_dir_name}")
        json.dump(vars(args), outfile)

    if not args.pretrain: # no pretraining, load previously trained model
        best_pretrain_model = load_best_model(args)


    if args.pretrain or best_pretrain_model == None: # pretraining
        best_pretrain_model = train(train_loaders, val_loader, model, logger, DEVICE, optimizers, schedulers, criterion, args)

    best_pretrain_model = test(test_loader, best_pretrain_model, logger, DEVICE, criterion, args)
    
    ############################################################################################################
    print('device:', DEVICE, 'dataset:', args.dataset)
    train_loaders, val_loader, test_loader = setup_dataloaders(args, pretrain=False)

    trained_backbone = lock_backbone(best_pretrain_model, args)

    best_lincls = train_lincls(train_loaders, val_loader, trained_backbone, classifier, logger, DEVICE, optimizer_cls, criterion_cls, args)
    if len(test_loader) != 0:
        test_lincls(test_loader, trained_backbone, best_lincls, logger, DEVICE, criterion_cls, args, plt=args.plt)

    # remove saved intermediate models
    delete_files(args)