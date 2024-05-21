import torch
import torch.nn as nn


class LSTM_Classifier(nn.Module):
    # A classifier, consisting of an LSTM layer and a fully connected layer
    def __init__(self, bb_dim, n_classes, hidden_size=128, rnn_type="lstm", num_layers=1, activation="relu"):
        super(LSTM_Classifier, self).__init__()
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.input_length = bb_dim[1]
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        if activation == "relu":
            self.activation_func = nn.ReLU()
        elif activation == "elu":
            self.activation_func = nn.ELU()
        else:
            raise NotImplementedError

        if self.rnn_type == "lstm":
            self.lstm = nn.LSTM(input_size=self.input_length, hidden_size=self.hidden_size, num_layers=2)
            out_size = self.hidden_size
        elif self.rnn_type == "lstm_bi":
            self.lstm = nn.LSTM(input_size=self.input_length, hidden_size=self.hidden_size, num_layers=2, bidirectional=True)
            out_size = self.hidden_size * 2
        elif self.rnn_type == "gru":
            self.lstm = nn.GRU(input_size=self.input_length, hidden_size=self.hidden_size, num_layers=2)
            out_size = self.hidden_size
        elif self.rnn_type == "gru_bi":
            self.lstm = nn.GRU(input_size=self.input_length, hidden_size=self.hidden_size, num_layers=2, bidirectional=True)
            out_size = self.hidden_size * 2
        

        if self.num_layers == 1:
            self.classifier = nn.Linear(out_size, self.n_classes)
        elif self.num_layers == 2:
            self.classifier = nn.Sequential()
            self.classifier.add_module("fc1", nn.Linear(out_size, 256))
            self.classifier.add_module("act1", self.activation_func)
            self.classifier.add_module("fc2", nn.Linear(256, n_classes))
        elif self.num_layers == 3:
            self.classifier = nn.Sequential()
            self.classifier.add_module("fc1", nn.Linear(out_size, 256))
            self.classifier.add_module("act1", self.activation_func)
            self.classifier.add_module("fc2", nn.Linear(256, 128))
            self.classifier.add_module("act2", self.activation_func)
            self.classifier.add_module("fc3", nn.Linear(128, n_classes))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x, (h, c) = self.lstm(x)
        x = x[-1, :, :]
        out = self.classifier(x)
        return out

class Classifier(nn.Module):
    # A simple classifier with a single head for classification or regression
    def __init__(self, bb_dim, n_classes, tau=1.0, num_layers=1, activation="relu"):
        super(Classifier, self).__init__()
        self.n_classes = n_classes
        self.num_layers = num_layers
        self.tau = nn.Parameter(torch.tensor(tau), requires_grad=False)
        if activation == "relu":
            self.activation_func = nn.ReLU()
        elif activation == "elu":
            self.activation_func = nn.ELU()
        else:
            raise NotImplementedError
        
        if self.num_layers == 1:
            self.classifier = nn.Linear(bb_dim, n_classes)
        elif self.num_layers == 2:
            self.classifier = nn.Sequential()
            self.classifier.add_module("fc1", nn.Linear(bb_dim, 256))
            self.classifier.add_module("act1", self.activation_func)
            self.classifier.add_module("fc2", nn.Linear(256, n_classes))
        elif self.num_layers == 3:
            self.classifier = nn.Sequential()
            self.classifier.add_module("fc1", nn.Linear(bb_dim, 256))
            self.classifier.add_module("act1", self.activation_func)
            self.classifier.add_module("fc2", nn.Linear(256, 128))
            self.classifier.add_module("act2", self.activation_func)
            self.classifier.add_module("fc3", nn.Linear(128, n_classes))
        else:
            raise NotImplementedError


    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        out = self.classifier(x)

        if self.n_classes > 1:
            out = out / self.tau

        return out
    
    def set_temperature(self, tau):
        self.tau.data = torch.tensor(tau)


class Classifier_with_uncertainty(nn.Module):
    # Implements Classifier, but with a second single-output head for uncertainty estimation
    def __init__(self, bb_dim, n_classes, num_layers=1, activation="relu"):
        super(Classifier_with_uncertainty, self).__init__()
        self.n_classes = n_classes
        self.num_layers = num_layers
        if activation == "relu":
            self.activation_func = nn.ReLU()
        elif activation == "elu":
            self.activation_func = nn.ELU()
        else:
            raise NotImplementedError
        if self.num_layers == 1:
            self.classifier = nn.Linear(bb_dim, n_classes)
            self.uncertainty = nn.Linear(bb_dim, 1)
        elif self.num_layers == 2:
            self.classifier = nn.Sequential()
            self.classifier.add_module("fc1", nn.Linear(bb_dim, 256))
            self.classifier.add_module("act1", self.activation_func)
            self.classifier.add_module("fc2", nn.Linear(256, n_classes))
            self.uncertainty = nn.Sequential()
            self.uncertainty.add_module("fc1", nn.Linear(bb_dim, 256))
            self.uncertainty.add_module("act1", self.activation_func)
            self.uncertainty.add_module("fc2", nn.Linear(256, 1))
        elif self.num_layers == 3:
            self.classifier = nn.Sequential()
            self.classifier.add_module("fc1", nn.Linear(bb_dim, 256))
            self.classifier.add_module("act1", self.activation_func)
            self.classifier.add_module("fc2", nn.Linear(256, 128))
            self.classifier.add_module("act2", self.activation_func)
            self.classifier.add_module("fc3", nn.Linear(128, n_classes))
            self.uncertainty = nn.Sequential()
            self.uncertainty.add_module("fc1", nn.Linear(bb_dim, 256))
            self.uncertainty.add_module("act1", self.activation_func)
            self.uncertainty.add_module("fc2", nn.Linear(256, 128))
            self.uncertainty.add_module("act2", self.activation_func)
            self.uncertainty.add_module("fc3", nn.Linear(128, 1))
        else:
            raise NotImplementedError

        if self.n_classes > 1:
            # add softmax layer
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        out = self.classifier(x)

        if self.n_classes > 1:
            out = self.softmax(out)
            
        uncertainty = self.uncertainty(x)

        return out, uncertainty