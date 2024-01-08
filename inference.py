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


json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/try_scheduler_reconstruction_backbone_AE_pretrain_max_eps60_lr0.0001_bs128_aug1jit_scal_aug2resample_dim-pdim128-128_EMA0.996_criterion_MSE_lambda1_1.0_lambda2_1.0_tempunit_tsfm/config.json"
with open(json_file) as json_file:
    json_args = json.load(json_file)

args = Namespace(**json_args)
split = 0

mode = "pretrain"




if not hasattr(args, "model_name"):
    args.model_name = args.backbone + '_'+args.dataset + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) + '_split' + str(split)
else:
    args.model_name = re.sub(r'split\d+', f"split{split}", args.model_name)

if mode == "pretrain":
    model_weights_path = os.path.join(args.model_dir_name, 'pretrain_' + args.model_name  + "_bestmodel" + '.pt')
elif mode == "finetune":
    NotImplementedError
elif mode == "supervised":
    model_weights_path = os.path.join(args.model_dir_name, args.model_name + '_model.pt')
#args.batch_size = 1

# %%
# Testing
#######################
DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE, 'dataset:', args.dataset)

# Load data
train_loader, val_loader, test_loader = setup_dataloaders(args)


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
model_weights = torch.load(model_weights_path)["model_state_dict"]
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
