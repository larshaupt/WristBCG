#%%
import os
import json
import pandas
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime

from models.backbones import *
from models.loss import *
from trainer import *
from argparse import Namespace


#%%


json_file = "/local/home/lhauptmann/thesis/CL-HAR/results/CorNET_max_lr0.0001_bs64/config.json"
with open(json_file) as json_file:
    json_args = json.load(json_file)

args = Namespace(**json_args)
model_weights_path = os.path.join(args.model_dir_name, args.model_name + '_model.pt')
#args.batch_size = 1

# %%
# Testing
#######################
DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE, 'dataset:', args.dataset)

# Load data
train_loader, val_loader, test_loader = setup_dataloaders(args)

# Initialize and load test model
if args.backbone == 'FCN':
    model_test = FCN(n_channels=args.n_feature, n_classes=args.n_class, input_size=args.input_length, backbone=False)
elif args.backbone == 'DCL':
    model_test = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, input_size=args.input_length, backbone=False)
elif args.backbone == 'LSTM':
    model_test = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=False)
elif args.backbone == 'AE':
    model_test = AE(n_channels=args.n_feature, input_size=args.input_len, n_classes=args.n_class, outdim=128, backbone=False)
elif args.backbone == 'CNN_AE':
    model_test = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=False)
elif args.backbone == 'Transformer':
    model_test = Transformer(n_channels=args.n_feature, input_size=args.input_len, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
elif args.backbone == "CorNET":
    model_test = CorNET(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=32, kernel_size=40, LSTM_units=128, backbone=False, input_size=args.input_length)
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

    
# %%
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
