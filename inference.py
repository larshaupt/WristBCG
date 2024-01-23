#%%
import os
import json
import pandas
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
import re

from models.backbones import *
from models.loss import *
from trainer import *
from argparse import Namespace


#%%

json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_supervised_backbone_CorNET_pretrain_capture24_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_cos_sim_lambda1_1.0_lambda2_1.0_tempunit_tsfm/config.json"
#json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_reconstruction_backbone_AE_pretrain_max_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_MSE_lambda1_1.0_lambda2_1.0_tempunit_tsfm/config.json"
with open(json_file) as json_file:
    json_args = json.load(json_file)

args = Namespace(**json_args)
split = 0

mode = "finetune"


if not hasattr(args, "model_name"):
    args.model_name = args.backbone + '_'+args.dataset + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) + '_split' + str(split)
else:
    args.model_name = re.sub(r'split\d+', f"split{split}", args.model_name)

if mode == "pretrain":
    model_weights_path = os.path.join(args.model_dir_name, 'pretrain_' + args.model_name  + "_bestmodel" + '.pt')
elif mode == "finetune":
    model_weights_path = os.path.join(args.model_dir_name, 'lincls_' + args.model_name + "_split" + str(args.split) + "_bestmodel" + '.pt')
elif mode == "supervised":
    model_weights_path = os.path.join(args.model_dir_name, args.model_name + '_model.pt')
args.batch_size = 1
args.cuda = 3

# %%
# Testing
#######################
DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE, 'dataset:', args.dataset)
#%%
# Load data
train_loader, val_loader, test_loader = setup_dataloaders(args)
#%%

class AE(nn.Module):
    def __init__(self, n_channels, input_size, n_classes, outdim=128, backbone=True):
        super(AE, self).__init__()

        self.backbone = backbone
        self.input_size = input_size

        self.e1 = nn.Linear(n_channels, 8)
        self.e2 = nn.Linear(8 * input_size, 2 * input_size)
        self.e3 = nn.Linear(2 * input_size, outdim)

        self.d1 = nn.Linear(outdim, 2 * input_size)
        self.d2 = nn.Linear(2 * input_size, 8 * input_size)
        self.d3 = nn.Linear(8, 1)

        self.out_dim = outdim

        if backbone == False:
            self.classifier = nn.Linear(outdim, n_classes)

    def forward(self, x):
        x_e1 = self.e1(x)
        x_e1 = x_e1.reshape(x_e1.shape[0], -1)
        x_e2 = self.e2(x_e1)
        x_encoded = self.e3(x_e2)

        x_d1 = self.d1(x_encoded)
        x_d2 = self.d2(x_d1)
        x_d2 = x_d2.reshape(x_d2.shape[0], self.input_size, 8)
        x_decoded = self.d3(x_d2)

        if self.backbone:
            return x_decoded, x_encoded
        else:
            out = self.classifier(x_encoded)
            return out, x_decoded

# Initialize and load test model
if args.backbone == 'FCN':
    model_test = FCN(n_channels=args.n_feature, n_classes=args.n_class, input_size=args.input_length, backbone=True)
elif args.backbone == 'DCL':
    model_test = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, input_size=args.input_length, conv_kernels=64, kernel_size=5, LSTM_units=args.lstm_units, backbone=True)
elif args.backbone == 'LSTM':
    model_test = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=args.lstm_units, backbone=True)
elif args.backbone == 'AE':
    model_test = AE(n_channels=args.n_feature, input_size=args.input_length, n_classes=args.n_class, outdim=128, backbone=True)
elif args.backbone == 'CNN_AE':
    model_test = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, embdedded_size=128, input_size=args.input_length, backbone=True, n_channels_out=args.n_channels_out)
elif args.backbone == 'Transformer':
    model_test = Transformer(n_channels=args.n_feature, input_size=args.input_length, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True)
elif args.backbone == "CorNET":
    model_test = CorNET(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=args.num_kernels, kernel_size=args.kernel_size, LSTM_units=args.lstm_units, backbone=True, input_size=args.input_length)
elif args.backbone == "TCN":
    model_test = TemporalConvNet(num_channels=[32, 64, 128], n_classes=args.n_class,  num_inputs=args.n_feature, input_length = args.input_length, kernel_size=16, dropout=0.2, backbone=True)
else:
    NotImplementedError

# Testing
model_weights_dict = torch.load(model_weights_path)

if mode == "finetune":
    model_weights = model_weights_dict["trained_backbone"]
    classifier = setup_linclf(args, DEVICE, model_test.out_dim)
    model_test.set_classification_head(classifier)

elif mode == "pretrain":
    model_weights = model_weights_dict["model_state_dict"]

model_test.load_state_dict(model_weights)
model_test = model_test.to(DEVICE)



# %%

X, Y, D, P = [], [], [], []

for i, (x, y, d) in enumerate(val_loader):
    x = x.to(DEVICE).float()
    y = y.to(DEVICE)
    d = d.to(DEVICE)
    p,_ = model_test(x)
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    d = d.detach().cpu().numpy()
    p = p.detach().cpu().numpy()
    X.append(x)
    Y.append(y)
    D.append(d)
    P.append(p)

    
X = np.concatenate(X)
Y = np.concatenate(Y)
D = np.concatenate(D)
P = np.concatenate(P).squeeze()

P = P*(args.hr_max - args.hr_min) + args.hr_min
Y = Y*(args.hr_max - args.hr_min) + args.hr_min

# %%
plt.scatter(Y,P, alpha=0.1)
plt.plot([args.hr_min, args.hr_max],[args.hr_min, args.hr_max], color="red")
# %%
mag = np.sqrt(X[:,:,0]**2 + X[:,:,1]**2 + X[:,:,2]**2)
std = np.std(mag, axis=1)
diff = np.abs(Y-P)
# %%
plt.plot(diff, std, 'o', alpha=0.1)
# %%
plt.plot(Y, diff, 'o', alpha=0.1)
# %%


X, Y, D, P = [], [], [], []

for i, (x, y, d) in enumerate(val_loader):
    x = x.to(DEVICE).float()
    y = y.to(DEVICE)
    d = d.to(DEVICE)
    p,_ = model_test(x)
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    d = d.detach().cpu().numpy()
    p = p.detach().cpu().numpy()
    X.append(x)
    Y.append(y)
    D.append(d)
    P.append(p)

    
X = np.concatenate(X)
Y = np.concatenate(Y)
D = np.concatenate(D)
P = np.concatenate(P)
# %%
i = 100
plt.plot(P[i,:,0])
plt.plot(X[i,:,0])
# %%


# try monte carlo dropout
preds_list = []
for i, (x,y,d) in enumerate(val_loader):

    if i < 300:
        continue
    print(i)
    x = x.to(DEVICE).float()
    y = y.to(DEVICE)
    d = d.to(DEVICE)
    n = 1000
    x = x.repeat(1000,1,1)
    y = y.repeat(1000,1)
    d = d.repeat(1000,1)

    model_test.train()
    out, p = model_test(x)

    preds = out.detach().cpu().numpy()
    preds = preds * (args.hr_max - args.hr_min) + args.hr_min
    preds_list.append(preds)

    if i > 400:
        break

# %%



for i, preds in enumerate(preds_list):
    plt.hist(preds[:,0], bins=100)
    plt.xlim(50,120)
    plt.savefig(os.path.join("/local/home/lhauptmann/thesis/analysis/AppleDataset/MCDropout", f"hist_{i}.png"))
    plt.clf()
# %%
for i, (x,y,d) in enumerate(val_loader):
    plt.plot(x[0,:,0])
    plt.plot(x[0,:,1])
    plt.plot(x[0,:,2])
    plt.savefig(os.path.join("/local/home/lhauptmann/thesis/analysis/AppleDataset/data_loader", f"signal_{i}.png"))
    plt.clf()
# %%

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
}
    
dnn_to_bnn(model_test, const_bnn_prior_parameters)
# %%
preds = []
for _ in range(100):
    out, x = model_test(x)
    preds.append(out)
# %%
from bayesian_torch.layers.variational_layers.conv_variational import Conv1dReparameterization
from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization
from bayesian_torch.layers.variational_layers.rnn_variational import LSTMReparameterization



class BayesianCorNET(nn.Module):
    # from Biswas et. al: CorNET: Deep Learning Framework for PPG-Based Heart Rate Estimation and Biometric Identification in Ambulant Environment
    def __init__(self, n_channels, n_classes, conv_kernels=32, kernel_size=40, LSTM_units=128, input_size:int=500, backbone=True):
        super(BayesianCorNET, self).__init__()
        # vector size after a convolutional layer is given by:
        # (input_size - kernel_size + 2 * padding) / stride + 1
        
        self.activation = nn.ELU()
        self.backbone = backbone
        self.n_classes = n_classes
        self.dropout = nn.Dropout(0.1)
        self.conv1 = nn.Sequential(Conv1dReparameterization(n_channels, conv_kernels, kernel_size=kernel_size, stride=1, bias=False, padding=0),
                                         nn.BatchNorm1d(conv_kernels),
                                         self.activation
                                         )
        self.conv1[0].dnn_to_bnn_flag = True
        self.maxpool1 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0, return_indices=False)
        out_len = (input_size - kernel_size + 2 * 0) // 1 + 1
        out_len = (out_len - 4 + 2 * 0) // 4 + 1
        self.conv2 = nn.Sequential(Conv1dReparameterization(conv_kernels, conv_kernels, kernel_size=kernel_size, stride=1, bias=False, padding=0),
                                         nn.BatchNorm1d(conv_kernels),
                                         self.activation
                                         )
        self.conv2[0].dnn_to_bnn_flag = True
        self.maxpool2 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0, return_indices=False)
                                         
        out_len = (out_len - kernel_size + 2 * 0) // 1 + 1
        self.out_len = (out_len - 4 + 2 * 0) // 4 + 1
        # should be 50 with default parameters

        self.lstm1 = LSTMReparameterization(in_features=conv_kernels, out_features=LSTM_units)
        self.lstm2 = LSTMReparameterization(in_features=LSTM_units, out_features=LSTM_units)
        self.lstm1.dnn_to_bnn_flag = True
        self.lstm2.dnn_to_bnn_flag = True
        self.out_dim = LSTM_units

        if backbone == False:
            self.classifier = LinearReparameterization(LSTM_units, n_classes) 
            self.classifier.dnn_to_bnn_flag = True
    def forward(self, x):
        #self.lstm.flatten_parameters()
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = x.permute(2, 0, 1)
        # shape is (L, N, H_in)
        # L - seq_len, N - batch_size, H_in - input_size
        # L = 19
        # N = 64
        # H_in = 32
        x = x.reshape(x.shape[0], x.shape[1], -1)


        x, h = self.lstm1(x)
        h = (h[0].squeeze(1), h[1].squeeze(1))
        x, h = self.lstm2(x, h)
        x = x[-1, :, :]

        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x

    def set_classification_head(self, classifier):
        self.classifier = classifier
        self.backbone = False
# %%

model = BayesianCorNET(n_channels=3, n_classes=1, conv_kernels=32, kernel_size=40, LSTM_units=128, input_size=1000, backbone=False)
model = model.to(DEVICE)
# %%
model(x)
# %%
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

kl_loss = get_kl_loss(model)
# %%
