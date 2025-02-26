import argparse
from trainer import *
import wandb
import json
from config import results_dir
from utils import get_free_gpu



def get_parser():

    """
    Creates and configures the argument parser for the training script.

    This function sets up command line arguments for various configurations, including 
    hyperparameters, dataset specifications, training options, model choices, and logging preferences.
    It returns the configured parser, which can then be used to parse the command line arguments.

    Arguments:
    - cuda (int): CUDA device ID for training. Default is -1 (automatically selects a free GPU).
    - num_workers (int): Number of workers for data loading. Default is 0.
    - random_seed (int): Random seed for reproducibility. Default is 10.
    - pretrain (int): Flag to indicate whether to pretrain the model. Default is 0 (no pretraining).
    - finetune (int): Flag to indicate whether to finetune the pretrained model. Default is 1 (finetune).
    
    Hyperparameters:
    - pretrain_batch_size (int): Batch size for pretraining. Default is 512.
    - batch_size (int): Batch size for training. Default is 512.
    - n_epoch (int): Number of epochs for training. Default is 60.
    - lr_pretrain (float): Learning rate for pretraining. Default is 1e-4.
    - lr (float): Learning rate for training. Default is 5e-4.
    - weight_decay (float): Weight decay for regularization. Default is 1e-7.
    - scheduler (bool): Flag to indicate whether to use a scheduler. Default is False.
    - optimizer (str): Optimizer for finetuning. Default is 'Adam'.
    - loss (str): Loss function to use. Default is 'MAE'.
    
    Dataset:
    - dataset (str): Name of the dataset for finetuning. Default is 'appleall'.
    - pretrain_dataset (str): Name of the dataset for pretraining. Default is 'capture24'.
    - normalize (int): Flag to indicate whether to normalize the data. Default is 1 (normalize).
    
    Model configuration:
    - framework (str): Framework to use for training. Default is 'supervised'.
    - backbone (str): Backbone network architecture. Default is 'CorNET'.
    
    Logging:
    - wandb_mode (str): Weights and Biases logging mode. Default is 'online'.
    - wandb_group (str): Wandb group name.
    - wandb_project (str): Wandb project name. Default is 'hr_ssl'.
    - wandb_tag (str): Wandb run tag.
    
    Returns:
    - argparse.ArgumentParser: The argument parser configured with all the specified arguments.
    """
    
    # Parse command line arguments
    ##################
    parser = argparse.ArgumentParser(description='argument setting of network')
    parser.add_argument('--cuda', default=-1, type=int, help='cuda device ID')
    parser.add_argument('--num_workers', default=0, type=int, help='number of workers for data loading')
    parser.add_argument('--random_seed', default=10, type=int, help='random seed for training')
    parser.add_argument('--pretrain', type=int, default=0, help='if or not to pretrain')
    parser.add_argument('--finetune', type=int, default=1, help='if or not to finetune')


    # hyperparameter
    parser.add_argument('--pretrain_batch_size', type=int, default=512, help='batch size of pretraining')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size of training')
    parser.add_argument('--n_epoch', type=int, default=60, help='number of training epochs')
    parser.add_argument('--pretrain_n_epoch', type=int, default=60, help='number of training epochs for pretraining')
    parser.add_argument('--lr_pretrain', type=float, default=1e-4, help='learning rate for pretrain')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='weight decay')
    parser.add_argument('--weight_decay_pretrain', type=float, default=1e-7, help='weight decay for pretrain')
    parser.add_argument('--scheduler', type=bool, default=False, help='if or not to use a scheduler')
    parser.add_argument('--scheduler_finetune', type=bool, default=False, help='if or not to use a scheduler for finetuning/supervised learning')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam'], help='optimizer for finetuning')
    parser.add_argument('--loss', type=str, default='MAE', choices=['MSE', 'MAE', 'Huber', 'LogCosh', 'CrossEntropy', 'NLE'], help='loss function')

    # dataset
    parser.add_argument('--dataset', type=str, default='appleall', choices=['apple','max', 'm2sleep', "m2sleep100", 'capture24', 'apple100', 'parkinson100', 'IEEE', "appleall", "max_v2", "max_hrv"], help='name of dataset for finetuning')
    parser.add_argument('--pretrain_dataset', type=str, default='capture24', choices=['max', 'apple', 'capture24', 'capture24all', 'apple100', 'parkinson100', 'max_v2', 'appleall'], help='name of dataset')
    parser.add_argument('--pretrain_subsample', type=float, default=1.0, help='subsampling rate for pretraining')
    parser.add_argument('--normalize', type=int, default=1, help='if or not to z-normalize data')
    parser.add_argument('--split', type=int, default=0, help='split number, needs to have split file')
    parser.add_argument('--split_by', type=str, default='subject', choices=['subject', 'time'], help='split by subject or time, needs to have split file')

    # dataset characteristics
    parser.add_argument('--hr_min', type=float, default=30, help='minimum heart rate for training, not needed for pretraining')
    parser.add_argument('--hr_max', type=float, default=120, help='maximum heart rate for training, not needed for pretraining')
    parser.add_argument('--take_every_nth_test', type=int, default=1, help='take every nth test sample, similar to increasing step size')
    parser.add_argument('--take_every_nth_train', type=int, default=1, help='take every nth train sample, similar to increasing step size')
    parser.add_argument('--bandpass_freq_min', type=float, default=0.1, help='minimum frequency for bandpass filter')
    parser.add_argument('--bandpass_freq_max', type=float, default=18, help='maximum frequency for bandpass filter')

    # ablations
    parser.add_argument('--sampling_rate', type=int, default=0, help='sampling rate of the data. Warning: this will take longer time to load data')
    parser.add_argument('--window_size', type=int, default=10, help='window size for the dataset in seconds')
    parser.add_argument('--step_size', type=int, default=8, help='step size for the dataset in seconds')

    # data filtering by thresholds, default 0 means no filtering
    parser.add_argument('--data_thr_avg', type=float, default=0, help='threshold for input signal average')
    parser.add_argument('--data_thr_max', type=float, default=0, help='threshold for input signal max')
    parser.add_argument('--data_thr_angle', type=float, default=0, help='threshold for input signal angle')
    parser.add_argument('--data_thr_hr', type=float, default=0, help='threshold for input signal heart rate quality')
    
    # framework
    parser.add_argument('--framework', type=str, default='supervised', choices=['byol', 'simsiam', 'simclr', 'nnclr', 'tstcc', 'supervised', 'reconstruction', 'median', 'subject_median', 'oxford'], help='name of framework')
    parser.add_argument('--backbone', type=str, default='CorNET', choices=['FCN', 'DCL', 'LSTM', 'AE', 'CNN_AE', 'Attention_CNN_AE', 'Transformer', 'CorNET', 'AttentionCorNET', "FrequencyCorNET", 'HRCTPNet', "ResNET"], help='name of backbone network')
    parser.add_argument('--num_kernels', type=int, default=32, help='number of kernels in CNN')
    parser.add_argument('--kernel_size', type=int, default=16, help='kernel size in CNN')
    parser.add_argument('--lstm_units', type=int, default=128, help='number of units in LSTM')
    parser.add_argument('--rnn_type', type=str, default="gru", choices=["lstm", "lstm_bi", "gru", "gru_bi"], help='direction of LSTM')
    parser.add_argument('--num_layers_classifier', type=int, default=1, help='number of layers in the classifier')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='dropout rate')

    # uncertainty
    parser.add_argument('--model_uncertainty', type=str, default="none", choices=["none", "gaussian_classification", "mcdropout", "bnn", "bnn_pretrained", "bnn_pretrained_firstlast", "NLE", "ensemble"], help='which method to use to output a probability distribution')
    parser.add_argument('--label_sigma', type=float, default=3.0, help='sigma for gaussian classification')

    # postprocessing
    # sumprod and beliefppg are the same, just for compatibility
    parser.add_argument('--postprocessing', type=str, default='none', choices=['none', 'beliefppg', 'kalmansmoothing', 'raw', 'sumprod', "viterbi"], help='postprocessing method')
    parser.add_argument('--transition_distribution', type=str, default='laplace', choices=['gauss', 'laplace'], help='transition distribution for belief ppg')
    parser.add_argument('--n_prob_class', type=int, default=64, help='number of class for probability distribution')

    # log
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['offline', 'online', 'disabled', 'dryrun', 'run'],  help='wandb mode')
    parser.add_argument('--wandb_group', type=str, default='', help='wandb group')
    parser.add_argument('--wandb_project', type=str, default='hr_ssl', help='wandb project')
    parser.add_argument('--wandb_tag', type=str, default='', help='wandb name')

    # ssl  - finetune
    parser.add_argument('--lr_finetune_backbone', type=float, default=1e-5, help='learning rate for finetuning the backbone network')
    parser.add_argument('--lr_finetune_lstm', type=float, default=1e-4, help='learning rate for finetuning the lstm layer')

    # ssl - general
    parser.add_argument('--p', type=int, default=128,
                        help='byol: projector size, simsiam: projector output size, simclr: projector output size')
    parser.add_argument('--phid', type=int, default=128,
                        help='byol: projector hidden size, simsiam: predictor hidden size, simclr: na')
    parser.add_argument('--criterion', type=str, default='cos_sim', choices=['cos_sim', 'NTXent', 'MSE', 'MAE'],
                        help='type of loss function for contrastive learning')
    # ssl - augmentation
    parser.add_argument('--aug1', type=str, default='t_warp',
                        choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 't_flip', 't_warp', 'resample', 'rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f', 'bioglass'],
                        help='the type of augmentation transformation')
    parser.add_argument('--aug2', type=str, default='bioglass',
                        choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 't_flip', 't_warp', 'resample', 'rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f', 'bioglass'],
                        help='the type of augmentation transformation')


    # ssl - byol
    parser.add_argument('--lr_mul', type=float, default=10.0,
                        help='lr multiplier for the second optimizer when training byol')
    parser.add_argument('--EMA', type=float, default=0.996, help='exponential moving average parameter')

    # ssl - nnclr
    parser.add_argument('--mmb_size', type=int, default=1024, help='maximum size of NNCLR support set')

    # ssl - TS-TCC
    parser.add_argument('--lambda1', type=float, default=1.0, help='weight for temporal contrastive loss')
    parser.add_argument('--lambda2', type=float, default=1.0, help='weight for contextual contrastive loss, also used as the weight for reconstruction loss when AE or CAE being backbone network')
    parser.add_argument('--temp_unit', type=str, default='tsfm', choices=['tsfm', 'lstm', 'blstm', 'gru', 'bgru'], help='temporal unit in the TS-TCC')

    # plot
    parser.add_argument('--plot_tsne', type=bool, default=False, help='if or not to plot tsne plot')

    #saving arguments
    parser.add_argument('--save_probabilities', type=bool, default=True, help='if or not to save probabilities')
    parser.add_argument('--save_model', type=int, default=1, help='if or not to save model, results, and config')

    
    return parser

if __name__ == '__main__':

    wandb.login()
    ############################################################################################################
    ############################ CONFIGURATION #################################################################
    ############################################################################################################
    
    parser = get_parser()
    params = parser.parse_args()

    torch.manual_seed(params.random_seed)
    np.random.seed(params.random_seed)

    # Set device, automatically select a free GPU if available and cuda == -1
    if params.cuda == -1:
        params.cuda = int(get_free_gpu())
        print(f"Automatically selected GPU {params.cuda}")

    
    DEVICE = torch.device('cuda:' + str(params.cuda) if torch.cuda.is_available() else 'cpu')
    # setup model, optimizer, scheduler, criterion
    params = setup_args(params)
    initial_mode = 'pretraining' if params.pretrain else 'finetuning' #if params.finetune else 'postprocessing'
    train_loader, val_loader, test_loader = setup_dataloaders(params, mode=initial_mode)

    if params.framework == "median":
        predict_median(train_loader, val_loader, test_loader, params, mode="global")
        exit()
    if params.framework == "subject_median":
        predict_median(train_loader, val_loader, test_loader, params, mode="subject_wise")
        exit()
        

    model, optimizers, schedulers, criterion = setup(params, DEVICE)

    # save config file
    if params.save_model:
        config_file = os.path.join(params.lincl_model_file.replace("bestmodel.pt", "config.json"))
        with open(config_file, "w") as outfile:
            print(f"Saving config file to {config_file}")
            json.dump(vars(params), outfile)

    ############################################################################################################
    ############################ PRETRAINING ###################################################################
    ############################################################################################################

    if params.pretrain: # pretraining
        # setup dataloader for pretraining
        #train_loader, val_loader, test_loader = setup_dataloaders(params, mode="pretraining")
        print('device:', DEVICE, 'dataset:', params.pretrain_dataset)
        pretrain_model_weights = train(train_loader, val_loader, model, DEVICE, optimizers, schedulers, criterion, params)
        model.load_state_dict(pretrain_model_weights)

        if len(test_loader) != 0:
            test(test_loader, model, DEVICE, criterion, params)

    else: # no pretraining, load previously trained model
        if params.framework == 'supervised':
            # take random initialization
            pass
        else:
            # load best pretrain model
            if params.backbone == "ResNET":
                # only for oxwearables model
                model.load_weights(config.ResNET_oxwearables_weights_path)
            else:
                pretrain_model_weights = load_best_model(params)
                model.load_state_dict(pretrain_model_weights)

    pretrain_model = model


    ############################################################################################################
    ############################ FINETUNING ####################################################################
    ############################################################################################################

    trained_backbone = extract_backbone(pretrain_model, params)
    del pretrain_model
    classifier, criterion_cls, optimizer_cls, scheduler_cls = setup_classifier(params, DEVICE, trained_backbone)

    if initial_mode != 'finetuning':
        train_loader, val_loader, test_loader = setup_dataloaders(params, mode="finetuning")

    if params.finetune:

        # setup dataloader for finetuning

        print('device:', DEVICE, 'dataset:', params.dataset)
        
        trained_backbone.set_classification_head(classifier)
        optimizer_cls.param_groups[0]['lr'] = params.lr_finetune_backbone
        optimizer_cls.param_groups[1]['lr'] = params.lr_finetune_lstm
        optimizer_cls.add_param_group({'params': trained_backbone.classifier.parameters(), 'lr': params.lr})
        
        trained_backbone_weights = train_lincls(train_loader, val_loader, trained_backbone, DEVICE, optimizer_cls, criterion_cls, scheduler_cls, params)
        trained_backbone.load_state_dict(trained_backbone_weights)
           

    elif params.postprocessing != 'none':
        classifier = setup_linclf(params, DEVICE, trained_backbone.out_dim)
        trained_backbone.set_classification_head(classifier)
        trained_backbone_weights = load_best_lincls(params)
        trained_backbone.load_state_dict(trained_backbone_weights)

    # We'll run the finetune test regardless of whether we finetuned or not
    if len(test_loader) != 0:
        test_lincls(test_loader, trained_backbone, DEVICE, criterion_cls, params)

    ############################################################################################################
    ##########################W## POSTPROCESSING ################################################################
    ############################################################################################################

    if params.postprocessing != 'none':
        criterion_post = nn.BCELoss(reduction='mean')
        if initial_mode != 'postprocessing':
            train_loader, val_loader, test_loader = setup_dataloaders(params, mode="postprocessing")
            print('device:', DEVICE, 'dataset:', params.dataset)

        trained_backbone = add_probability_wrapper(trained_backbone, params, DEVICE)
        postprocessing_model = setup_postprocessing_model(params)
        
        postprocessing_model = train_postprocessing(train_loader, postprocessing_model, params)

        # Validation
        test_postprocessing(val_loader, trained_backbone, postprocessing_model, DEVICE, criterion_post, params, prefix='Val')
        # Test
        test_postprocessing(test_loader, trained_backbone, postprocessing_model, DEVICE, criterion_post, params, prefix='Test')
        
    wandb.finish()