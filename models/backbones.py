
import torch
from torch import nn
from torch.nn.utils import weight_norm


from .attention import *
from .MMB import *
from .HRCTPNet import HRCTPNet
from .accNet import Resnet
from .uncertainty import Uncertainty_Wrapper, Uncertainty_Regression_Wrapper, Ensemble_Wrapper, NLE_Wrapper, MC_Dropout_Wrapper, BNN_Wrapper
from .classifiers import Classifier, Classifier_with_uncertainty, LSTM_Classifier

class FCN(nn.Module):
    def __init__(self, n_channels, n_classes, conv_kernels=64, kernel_size=8, out_channels=128, input_size:int=500, backbone=True):
        super(FCN, self).__init__()

        self.backbone = backbone

        # vector size after a convolutional layer is given by:
        # (input_size - kernel_size + 2 * padding) / stride + 1

        # vector size after pooling layer is given by:
        # (input_size - kernel_size + 2 * padding) / stride + 1
        
        self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, conv_kernels, kernel_size=kernel_size, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(conv_kernels),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                                         nn.Dropout(0.35))
        out_len = (input_size - kernel_size + 2 * 4) // 1 + 1
        out_len = (out_len - 2 + 2 * 1) // 2 + 1
        self.conv_block2 = nn.Sequential(nn.Conv1d(conv_kernels, conv_kernels, kernel_size=kernel_size, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(conv_kernels),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        out_len = (out_len - kernel_size + 2 * 4) // 1 + 1
        out_len = (out_len - 2 + 2 * 1) // 2 + 1
        self.conv_block3 = nn.Sequential(nn.Conv1d(conv_kernels, out_channels, kernel_size=kernel_size, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        out_len = (out_len - kernel_size + 2 * 4) // 1 + 1
        out_len = (out_len - 2 + 2 * 1) // 2 + 1

        self.out_len = int(out_len)
        

        self.out_channels = out_channels
        self.out_dim = self.out_len * self.out_channels

        if backbone == False:
            self.classifier = nn.Linear(self.out_len * out_channels, n_classes)

    def forward(self, x_in):
        x_in = x_in.permute(0, 2, 1)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        if self.backbone:
            return None, x
        else:
            x_flat = x.reshape(x.shape[0], -1)
            logits = self.classifier(x_flat)
            return logits, x
        
    def set_classification_head(self, classifier):
        self.classifier = classifier
        self.backbone = False


class CorNET(nn.Module):
    # from Biswas et. al: CorNET: Deep Learning Framework for PPG-Based Heart Rate Estimation and Biometric Identification in Ambulant Environment
    def __init__(self, 
                 n_channels, 
                 n_classes, 
                 conv_kernels=32, 
                 kernel_size=40, 
                 LSTM_units=128, 
                 input_size:int=500, 
                 backbone=True, 
                 rnn_type="lstm",
                 dropout_rate=0.1):
        super(CorNET, self).__init__()
        # vector size after a convolutional layer is given by:
        # (input_size - kernel_size + 2 * padding) / stride + 1
        
        self.activation = nn.ELU()
        self.backbone = backbone
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout_rate)
        self.conv1 = nn.Sequential(nn.Conv1d(n_channels, conv_kernels, kernel_size=kernel_size, stride=1, bias=False, padding=0),
                                         nn.BatchNorm1d(conv_kernels),
                                         self.activation
                                         )
        self.maxpool1 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0, return_indices=False)
        out_len = (input_size - kernel_size + 2 * 0) // 1 + 1
        out_len = (out_len - 4 + 2 * 0) // 4 + 1
        self.conv2 = nn.Sequential(nn.Conv1d(conv_kernels, conv_kernels, kernel_size=kernel_size, stride=1, bias=False, padding=0),
                                         nn.BatchNorm1d(conv_kernels),
                                         self.activation
                                         )
        self.maxpool2 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0, return_indices=False)
                                         
        out_len = (out_len - kernel_size + 2 * 0) // 1 + 1
        self.out_len = (out_len - 4 + 2 * 0) // 4 + 1
        # should be 50 with default parameters
        
        if rnn_type == "lstm":
            self.lstm = nn.LSTM(input_size=conv_kernels, hidden_size=LSTM_units, num_layers=2, bidirectional=False)
            self.out_dim = LSTM_units
        elif rnn_type == "lstm_bi":
            self.lstm = nn.LSTM(input_size=conv_kernels, hidden_size=LSTM_units, num_layers=2, bidirectional=True)
            self.out_dim = LSTM_units * 2
        elif rnn_type == "gru":
            self.lstm = nn.GRU(input_size=conv_kernels, hidden_size=LSTM_units, num_layers=2, bidirectional=False)
            self.out_dim = LSTM_units
        elif rnn_type == "gru_bi":
            self.lstm = nn.GRU(input_size=conv_kernels, hidden_size=LSTM_units, num_layers=2, bidirectional=True)
            self.out_dim = LSTM_units * 2
        else:
            raise NotImplementedError
        

        if backbone == False:
            self.classifier = nn.Linear(LSTM_units, n_classes) 

    def forward(self, x):
        self.lstm.flatten_parameters()
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


        x, h = self.lstm(x)
        x = x[-1, :, :]

        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x

    def set_classification_head(self, classifier):
        self.classifier = classifier
        self.backbone = False




class DeepConvLSTM(nn.Module):
    def __init__(self, n_channels, n_classes, conv_kernels=64, kernel_size=5, LSTM_units=128, input_size:int=500, backbone=True):
        super(DeepConvLSTM, self).__init__()
        self.backbone = backbone

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2)

        self.out_dim = LSTM_units

        if backbone == False:
            self.classifier = nn.Linear(LSTM_units, n_classes)

        self.activation = nn.ReLU()

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)
        
        x, h = self.lstm(x)
        x = x[-1, :, :]

        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x
        
    def set_classification_head(self, classifier):
        self.classifier = classifier
        self.backbone = False



class ChannelAttention(nn.Module):
    def __init__(self, feature_channels, n_channels, sequence_length):
        super(ChannelAttention, self).__init__()
        self.feature_channels = feature_channels
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.features_per_channel = feature_channels // n_channels
        assert self.feature_channels % self.n_channels == 0, "in_channels must be divisible by n_channels"
        self.attn = nn.Linear(sequence_length * self.features_per_channel, 1)

    def forward(self, encoder_outputs):
        # Reshape to split features into equal chunks for each channel
        encoder_outputs = encoder_outputs.view(encoder_outputs.size(0), self.n_channels, self.features_per_channel, -1)
        # Flatten to fit linear layer input size
        encoder_outputs = encoder_outputs.view(encoder_outputs.size(0), self.n_channels, -1)
        # Compute attention scores along the channel dimension
        energy = torch.tanh(self.attn(encoder_outputs))
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(energy, dim=1)
        # Weighted sum along the channel dimension
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)
        context_vector = context_vector.view(context_vector.size(0),self.features_per_channel, -1)
        return context_vector



class AttentionCorNET(nn.Module):
    # from Biswas et. al: CorNET: Deep Learning Framework for PPG-Based Heart Rate Estimation and Biometric Identification in Ambulant Environment
    # adapted version with Channel Attention
    def __init__(self, n_channels, n_classes, conv_kernels=32, kernel_size=40, LSTM_units=128, input_size:int=500, backbone=True, rnn_type="lstm", dropout=0.1):
        super(AttentionCorNET, self).__init__()
        # vector size after a convolutional layer is given by:
        # (input_size - kernel_size + 2 * padding) / stride + 1
        
        conv_kernels = conv_kernels * n_channels
        self.activation = nn.ELU()
        self.backbone = backbone
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Sequential(nn.Conv1d(n_channels, conv_kernels, kernel_size=kernel_size, stride=1, bias=False, padding=0, groups=n_channels),
                                         nn.BatchNorm1d(conv_kernels),
                                         self.activation
                                         )
        self.maxpool1 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0, return_indices=False)
        out_len = (input_size - kernel_size + 2 * 0) // 1 + 1
        out_len = (out_len - 4 + 2 * 0) // 4 + 1
        self.conv2 = nn.Sequential(nn.Conv1d(conv_kernels, conv_kernels, kernel_size=kernel_size, stride=1, bias=False, padding=0, groups=n_channels),
                                         nn.BatchNorm1d(conv_kernels),
                                         self.activation
                                         )
        self.maxpool2 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0, return_indices=False)
                                         
        out_len = (out_len - kernel_size + 2 * 0) // 1 + 1
        self.out_len = (out_len - 4 + 2 * 0) // 4 + 1

        self.attention = ChannelAttention(conv_kernels, n_channels, self.out_len)

        out_feat = conv_kernels // n_channels

        # should be 50 with default parameters
        
        if rnn_type == "lstm":
            self.lstm = nn.LSTM(input_size=out_feat, hidden_size=LSTM_units, num_layers=2, bidirectional=False)
            self.out_dim = LSTM_units
        elif rnn_type == "lstm_bi":
            self.lstm = nn.LSTM(input_size=out_feat, hidden_size=LSTM_units, num_layers=2, bidirectional=True)
            self.out_dim = LSTM_units * 2
        elif rnn_type == "gru":
            self.lstm = nn.GRU(input_size=out_feat, hidden_size=LSTM_units, num_layers=2, bidirectional=False)
            self.out_dim = LSTM_units
        elif rnn_type == "gru_bi":
            self.lstm = nn.GRU(input_size=out_feat, hidden_size=LSTM_units, num_layers=2, bidirectional=True)
            self.out_dim = LSTM_units * 2
        else:
            raise NotImplementedError
        

        if backbone == False:
            self.classifier = nn.Linear(LSTM_units, n_classes) 

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = self.attention(x)
        x = x.permute(2, 0, 1)
        # shape is (L, N, H_in)
        # L - seq_len, N - batch_size, H_in - input_size
        # L = 19
        # N = 64
        # H_in = 32
        x = x.reshape(x.shape[0], x.shape[1], -1)


        x, h = self.lstm(x)
        x = x[-1, :, :]

        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x

    def set_classification_head(self, classifier):
        self.classifier = classifier
        self.backbone = False



class DeepConvLSTM(nn.Module):
    def __init__(self, n_channels, n_classes, conv_kernels=64, kernel_size=5, LSTM_units=128, rnn_type = "gru", input_size:int=500, backbone=True):
        super(DeepConvLSTM, self).__init__()
        self.backbone = backbone
        self.rnn_type = rnn_type

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = nn.Dropout(0.5)
        if rnn_type == "lstm":
            self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2)
            self.out_dim = LSTM_units
        elif rnn_type == "lstm_bi":
            self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2, bidirectional=True)
            self.out_dim = LSTM_units * 2
        elif rnn_type == "gru": 
            self.lstm = nn.GRU(n_channels * conv_kernels, LSTM_units, num_layers=2)
            self.out_dim = LSTM_units
        elif rnn_type == "gru_bi":
            self.lstm = nn.GRU(n_channels * conv_kernels, LSTM_units, num_layers=2, bidirectional=True)
            self.out_dim = LSTM_units * 2
        else:
            raise NotImplementedError

        if backbone == False:
            self.classifier = nn.Linear(LSTM_units, n_classes)

        self.activation = nn.ReLU()

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)
        
        x, h = self.lstm(x)
        x = x[-1, :, :]

        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x
        
    def set_classification_head(self, classifier):
        self.classifier = classifier
        self.backbone = False

class LSTM(nn.Module):
    def __init__(self, n_channels, n_classes, LSTM_units=128, rnn_type="lstm", backbone=True):
        super(LSTM, self).__init__()
        self.rnn_type = rnn_type
        self.backbone = backbone
        if rnn_type == "lstm":
            self.lstm = nn.LSTM(n_channels, LSTM_units, num_layers=2)
            self.out_dim = LSTM_units
        elif rnn_type == "lstm_bi":
            self.lstm = nn.LSTM(n_channels, LSTM_units, num_layers=2, bidirectional=True)
            self.out_dim = LSTM_units * 2
        elif rnn_type == "gru":
            self.lstm = nn.GRU(n_channels, LSTM_units, num_layers=2)
            self.out_dim = LSTM_units
        elif rnn_type == "gru_bi":
            self.lstm = nn.GRU(n_channels, LSTM_units, num_layers=2, bidirectional=True)
            self.out_dim = LSTM_units * 2
        else:
            raise NotImplementedError
        
        if backbone == False:
            self.classifier = nn.Linear(LSTM_units, n_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.permute(1, 0, 2)
        x, (h, c) = self.lstm(x)
        x = x[-1, :, :]

        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x

    def set_classification_head(self, classifier):
        self.classifier = classifier
        self.backbone = False

class AE_encoder(nn.Module):
    def __init__(self, n_channels, input_size, n_classes=1, out_dim=128, backbone=True):
        super(AE_encoder, self).__init__()

        self.backbone = backbone
        self.input_size = input_size
        self.n_channels = n_channels
        self.out_dim = out_dim

        self.e1 = nn.Linear(self.n_channels, 8)
        self.e2 = nn.Linear(8 * input_size, 2 * input_size)
        self.e3 = nn.Linear(2 * input_size, self.out_dim)

        self.out_length = self.out_dim

        if not self.backbone:
            self.classifier = nn.Linear(self.out_dim, n_classes)


    def forward(self, x):
        x_e1 = self.e1(x)
        x_e1 = x_e1.reshape(x_e1.shape[0], -1)
        x_e2 = self.e2(x_e1)
        x_encoded = self.e3(x_e2)

        if self.backbone:
            return None, x_encoded
        else:
            out = self.classifier(x_encoded)
            return out, x_encoded
        
    def set_classification_head(self, classifier):
        self.classifier = classifier
        self.backbone = False

class AE_decoder(nn.Module):
    def __init__(self, n_channels, input_size, out_dim=128, n_channels_out=None):
        super(AE_decoder, self).__init__()
        self.out_dim = out_dim
        self.input_size = input_size
        if n_channels_out == None:
            self.n_channels_out = n_channels
        else:
            self.n_channels_out = n_channels_out

        self.d1 = nn.Linear(out_dim, 2 * self.input_size)
        self.d2 = nn.Linear(2 * self.input_size, 8 * self.input_size)
        self.d3 = nn.Linear(8, self.n_channels_out)

    def forward(self, x):
        x_d1 = self.d1(x)
        x_d2 = self.d2(x_d1)
        x_d2 = x_d2.reshape(x_d2.shape[0], self.input_size, 8)
        x_decoded = self.d3(x_d2)

        return x_decoded



class AE(nn.Module):
    def __init__(self, n_channels, input_size, n_classes, embdedded_size=128, n_channels_out=None, backbone=True):
        super(AE, self).__init__()

        self.backbone = backbone
        self.input_size = input_size
        self.n_channels = n_channels
        self.embdedded_size = embdedded_size
        self.out_dim = embdedded_size
        if n_channels_out == None:
            self.n_channels_out = n_channels
        else:
            self.n_channels_out = n_channels_out

        self.encoder = AE_encoder(n_channels, input_size, n_classes, self.embdedded_size, backbone)
        self.decoder = AE_decoder(n_channels, input_size, self.embdedded_size, n_channels_out)

    def forward(self, x):
        
        out, x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)

        if self.backbone:
            return x_decoded, x_encoded
        else:
            out = self.classifier(x_encoded)
            return out, x_decoded
        
    def set_classification_head(self, classifier):
        self.encoder.set_classification_head(classifier)
        self.backbone = False

class CNN_AE_encoder(nn.Module):
    def __init__(self, n_channels, 
                 conv_kernels=64, 
                 kernel_size=8,
                 pool_kernel_size=4, 
                 input_size: int = 1000, 
                 n_classes=1, 
                 out_dim=128, 
                 backbone=True, 
                 activation="elu",
                 num_layers=2,
                 dropout=0.1):
        
        super(CNN_AE_encoder, self).__init__()
        self.input_size = input_size
        self.n_channels = n_channels
        self.backbone = backbone
        self.out_dim = out_dim
        self.num_layers = num_layers
        assert num_layers in [2, 3], "Only 2 or 3 layers are supported"
        if activation == "relu":
            activation_func = nn.ReLU()
        elif activation == "elu":
            activation_func = nn.ELU()
        else:
            raise NotImplementedError


        self.e_conv1 = nn.Sequential(
            nn.Conv1d(n_channels, conv_kernels, kernel_size=kernel_size, stride=1, bias=False, padding=0),
            nn.BatchNorm1d(conv_kernels),
            activation_func
        )
        self.pool1 = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size, padding=0, return_indices=True)
        self.dropout = nn.Dropout(dropout)
        out_size = input_size
        out_size = (out_size - kernel_size + 2 * 0) // 1 + 1
        out_size = (out_size - pool_kernel_size + 2 * 0) // pool_kernel_size + 1

        self.e_conv2 = nn.Sequential(
            nn.Conv1d(conv_kernels, conv_kernels, kernel_size=kernel_size, stride=1, bias=False, padding=0),
            nn.BatchNorm1d(conv_kernels),
            activation_func
        )
        self.pool2 = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size, padding=0, return_indices=True)
        out_size = (out_size - kernel_size + 2 * 0) // 1 + 1
        out_size = (out_size - pool_kernel_size + 2 * 0) // pool_kernel_size + 1

        if num_layers == 3:
            self.e_conv3 = nn.Sequential(
                nn.Conv1d(conv_kernels, self.out_dim, kernel_size=kernel_size, stride=1, bias=False, padding=0),
                nn.BatchNorm1d(self.out_dim),
                activation_func
            )
            self.pool3 = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size, padding=0, return_indices=True)
            out_size = (out_size - kernel_size + 2 * 0) // 1 + 1
            out_size = (out_size - pool_kernel_size + 2 * 0) // pool_kernel_size + 1
        self.output_length = out_size
        if not self.backbone:
            self.classifier = nn.Linear(self.out_dim*out_size, n_classes)

        self.out_dim = (self.out_dim, self.output_length)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.e_conv1(x)
        pool1_size = x.shape
        x, indice1 = self.pool1(x)
        x = self.dropout(x)
        x = self.e_conv2(x)
        pool2_size = x.shape
        x, indice2 = self.pool2(x)
        x = self.dropout(x)
        if self.num_layers == 3:
            x = self.e_conv3(x)
            pool3_size = x.shape
            x, indice3 = self.pool3(x)
        else:
            indice3 = None
            pool3_size = None


        if self.backbone:
            return None, x, (indice1, indice2, indice3), (pool1_size, pool2_size, pool3_size)
        else:
            out = self.classifier(x)
            return out, x #, (indice1, indice2, indice3), (pool1_size, pool2_size, pool3_size)
        
    def set_classification_head(self, classifier):
        self.classifier = classifier
        self.backbone = False


class CNN_AE_decoder(nn.Module):
    def __init__(self, out_channels,
                  n_channels_out, 
                  conv_kernels=64, 
                  kernel_size=8, 
                  input_size: int = 1000, 
                  activation="elu",
                  num_layers=2,
                  pool_kernel_size = 4,
                  dropout=0.1):
        super(CNN_AE_decoder, self).__init__()
        self.input_size = input_size
        self.n_channels_out = n_channels_out
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        out_size = input_size
        if activation == "relu":
            activation_func = nn.ReLU()
        elif activation == "elu":
            activation_func = nn.ELU()
        else:
            raise NotImplementedError
        
        if num_layers == 3:
            self.unpool1 = nn.MaxUnpool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size, padding=0)
            self.d_conv1 = nn.Sequential(
                nn.ConvTranspose1d(out_channels, conv_kernels, kernel_size=kernel_size, stride=1, bias=False, padding=0),
                nn.BatchNorm1d(conv_kernels),
                activation_func
            )
            # out_size = (out_size -)
            out_size = (out_size - 1) * pool_kernel_size - 2 * 0 + pool_kernel_size
            out_size = (out_size - 1) * 1 - 2 * 0 + kernel_size
            self.lin1 = nn.Identity()

        self.unpool2 = nn.MaxUnpool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size, padding=0)
        self.d_conv2 = nn.Sequential(
            nn.ConvTranspose1d(conv_kernels, conv_kernels, kernel_size=kernel_size, stride=1, bias=False, padding=0),
            nn.BatchNorm1d(conv_kernels),
            activation_func
        )
        out_size = (out_size - 1) * pool_kernel_size - 2 * 0 + pool_kernel_size
        out_size = (out_size - 1) * 1 - 2 * 0 + kernel_size

        
        self.unpool3 = nn.MaxUnpool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size, padding=0)
        self.d_conv3 = nn.Sequential(
            nn.ConvTranspose1d(conv_kernels, self.n_channels_out, kernel_size=kernel_size, stride=1, bias=False, padding=0),
            nn.BatchNorm1d(self.n_channels_out),
            activation_func
        )
        out_size = (out_size - 1) * pool_kernel_size - 2 * 0 + pool_kernel_size
        out_size = (out_size - 1) * 1 - 2 * 0 + kernel_size
        self.lin2 = nn.Identity()

    def forward(self, x, indices, pool_sizes):

        indice1, indice2, indice3 = indices
        pool1_size, pool2_size, pool3_size = pool_sizes

        if self.num_layers == 3:
            x = self.d_conv1(self.unpool1(x, indice3, output_size=pool3_size))
            x = self.lin1(x)
            x = self.dropout(x)
        x = self.d_conv2(self.unpool2(x, indice2, output_size=pool2_size))
        x = self.dropout(x)
        x = self.d_conv3(self.unpool3(x, indice1, output_size=pool1_size))
        x = self.dropout(x)
        x_decoded = self.lin2(x)
        x_decoded = x_decoded.permute(0, 2, 1)
        return x_decoded


class CNN_AE(nn.Module):
    def __init__(self, n_channels, 
                 n_classes, 
                 conv_kernels=64, 
                 kernel_size=8, 
                 pool_kernel_size = 4,
                 embdedded_size=64, 
                 n_channels_out=None, 
                 input_size: int = 1000, 
                 backbone=True, 
                 activation="elu",
                 num_layers=2,
                 dropout=0.1):
        super(CNN_AE, self).__init__()
        self.input_size = input_size
        self.backbone = backbone
        self.n_channels = n_channels
        self.embdedded_size = conv_kernels
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool_kernel_size=pool_kernel_size
        if n_channels_out is None:
            self.n_channels_out = n_channels
        else:
            self.n_channels_out = n_channels_out
        

        self.encoder = CNN_AE_encoder(self.n_channels, 
                                      conv_kernels=conv_kernels, 
                                      kernel_size=kernel_size, 
                                      input_size = input_size, 
                                      n_classes=n_classes, 
                                      out_dim=self.embdedded_size, 
                                      backbone=backbone, 
                                      activation=activation,
                                      num_layers=self.num_layers,
                                      dropout=dropout,
                                      pool_kernel_size=pool_kernel_size)
                                      
        self.decoder = CNN_AE_decoder(self.embdedded_size, 
                                      n_channels_out, 
                                      conv_kernels=conv_kernels, 
                                      kernel_size=kernel_size, 
                                      input_size = input_size, 
                                      activation=activation,
                                      num_layers=self.num_layers,
                                      dropout=dropout,
                                      pool_kernel_size=pool_kernel_size)

        self.out_dim = self.encoder.out_dim
        self.output_length = self.encoder.output_length


    def forward(self, x):
        out, x_encoded, indices, pool_sizes = self.encoder(x)
        x_decoded = self.decoder(x_encoded, indices, pool_sizes)
        
        if self.backbone:
            return x_decoded, x_encoded
        else:
            return out, x_decoded

    def set_classification_head(self, classifier):
        self.encoder.set_classification_head(classifier)
        self.backbone = False



class Transformer(nn.Module):
    def __init__(self, n_channels, input_size, n_classes, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True):
        super(Transformer, self).__init__()

        self.backbone = backbone
        self.out_dim = dim
        self.transformer = Seq_Transformer(n_channel=n_channels, input_size=input_size, n_classes=n_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
        if backbone == False:
            self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.transformer(x)
        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x
        
    def set_classification_head(self, classifier):
        self.classifier = classifier
        self.backbone = False




class Projector(nn.Module):
    def __init__(self, model, bb_dim, prev_dim, dim):
        super(Projector, self).__init__()
        if model == 'SimCLR':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim))
        elif model == 'byol':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim, bias=False),
                                           nn.BatchNorm1d(dim, affine=False))
        elif model == 'NNCLR':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim, bias=False),
                                           nn.BatchNorm1d(dim))
        elif model == 'TS-TCC':
            self.projector = nn.Sequential(nn.Linear(dim, bb_dim // 2),
                                           nn.BatchNorm1d(bb_dim // 2),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(bb_dim // 2, bb_dim // 4))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.projector(x)
        return x


class Predictor(nn.Module):
    def __init__(self, model, dim, pred_dim):
        super(Predictor, self).__init__()
        if model == 'SimCLR':
            pass
        elif model == 'byol':
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim),
                                           nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pred_dim, dim))
        elif model == 'NNCLR':
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim),
                                           nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pred_dim, dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.predictor(x)
        return x

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


from functools import wraps


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projector and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, DEVICE, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer
        self.DEVICE = DEVICE

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        children = [*self.net.children()]
        print('children[self.layer]:', children[self.layer])
        return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output.reshape(output.shape[0], -1)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = Projector(model='byol', bb_dim=dim, prev_dim=self.projection_hidden_size, dim=self.projection_size)
        return projector.to(hidden)

    def get_representation(self, x):

        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x):
        if self.net.__class__.__name__ in ['AE', 'CNN_AE']:
            x_decoded, representation = self.get_representation(x)
        else:
            _, representation = self.get_representation(x)

        if len(representation.shape) == 3:
            representation = representation.reshape(representation.shape[0], -1)

        projector = self._get_projector(representation)
        projection = projector(representation)
        if self.net.__class__.__name__ in ['AE', 'CNN_AE']:
            return projection, x_decoded, representation
        else:
            return projection, representation


class NNMemoryBankModule(MemoryBankModule):
    """Nearest Neighbour Memory Bank implementation
    This class implements a nearest neighbour memory bank as described in the
    NNCLR paper[0]. During the forward pass we return the nearest neighbour
    from the memory bank.
    [0] NNCLR, 2021, https://arxiv.org/abs/2104.14548
    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0,
            memory bank is not used.
    """

    def __init__(self, size: int = 2 ** 16):
        super(NNMemoryBankModule, self).__init__(size)

    def forward(self,
                output: torch.Tensor,
                update: bool = False):
        """Returns nearest neighbour of output tensor from memory bank
        Args:
            output: The torch tensor for which you want the nearest neighbour
            update: If `True` updated the memory bank by adding output to it
        """

        output, bank = \
            super(NNMemoryBankModule, self).forward(output, update=update)
        bank = bank.to(output.device).t()

        output_normed = torch.nn.functional.normalize(output, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)

        similarity_matrix = \
            torch.einsum("nd,md->nm", output_normed, bank_normed)
        index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
        nearest_neighbours = \
            torch.index_select(bank, dim=0, index=index_nearest_neighbours)

        return nearest_neighbours



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.kernel_size = kernel_size
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    
    def get_output_size(self, input_size):

        # for a convolutional layer:
        # output_size = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        output_size = (input_size + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        output_size = (output_size + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        # actually the chomp layers cut off the extra values, so that the size is the same as the input size
        output_size = input_size
        return output_size


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, input_length, n_classes, num_channels, kernel_size=2, dropout=0.2, backbone=None):
        super(TemporalConvNet, self).__init__()

        self.backbone = backbone

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.out_len = self.get_output_size(input_length)

        if backbone == False:
            self.classifier = nn.Linear(self.out_len* num_channels[-1], n_classes) 

    def forward(self, x):
        x = x.swapaxes(1, 2)
        x = self.network(x)
        if self.backbone:
            return None, x
        else:
            x_flat = x.reshape(x.shape[0], -1)
            logits = self.classifier(x_flat)
            return logits, x
        
    def get_output_size(self, input_size):
        for layer in self.network:
            input_size = layer.get_output_size(input_size)
        return input_size


            
from bayesian_torch.layers.variational_layers.conv_variational import Conv1dReparameterization
from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization
from bayesian_torch.layers.variational_layers.rnn_variational import LSTMReparameterization
from bayesian_torch.models.dnn_to_bnn import bnn_conv_layer, bnn_linear_layer


class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, *args):
        return args

class BayesianCorNET(nn.Module):
    # from Biswas et. al: CorNET: Deep Learning Framework for PPG-Based Heart Rate Estimation and Biometric Identification in Ambulant Environment
    def __init__(self, n_channels, n_classes, conv_kernels=32, kernel_size=40, LSTM_units=128, input_size:int=500, backbone=True, bayesian_layers="all", state_dict=None, dropout=0.1, rnn_type="lstm"):
        super(BayesianCorNET, self).__init__()
        # vector size after a convolutional layer is given by:
        # (input_size - kernel_size + 2 * padding) / stride + 1
        self.bayesian_layers = bayesian_layers
        self.activation = nn.ELU()
        self.backbone = backbone
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.rnn_type = rnn_type

        

        self.conv1 = nn.Sequential(nn.Conv1d(n_channels, conv_kernels, kernel_size=kernel_size, stride=1, bias=False, padding=0),
                                         nn.BatchNorm1d(conv_kernels),
                                         self.activation
                                         )
        self.maxpool1 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0, return_indices=False)
        out_len = (input_size - kernel_size + 2 * 0) // 1 + 1
        out_len = (out_len - 4 + 2 * 0) // 4 + 1
        self.conv2 = nn.Sequential(nn.Conv1d(conv_kernels, conv_kernels, kernel_size=kernel_size, stride=1, bias=False, padding=0),
                                         nn.BatchNorm1d(conv_kernels),
                                         self.activation
                                         )
        self.maxpool2 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0, return_indices=False)
                                         
        out_len = (out_len - kernel_size + 2 * 0) // 1 + 1
        self.out_len = (out_len - 4 + 2 * 0) // 4 + 1


        if self.bayesian_layers in ["all"]:
            self.lstm = LSTMReparameterization(in_features=conv_kernels, out_features=LSTM_units)
            self.lstm2 = LSTMReparameterization(in_features=LSTM_units, out_features=LSTM_units)
            self.lstm.dnn_to_bnn_flag = True
            self.lstm2.dnn_to_bnn_flag = True
        else:
            if self.rnn_type == "lstm":
                self.lstm = nn.LSTM(input_size=conv_kernels, hidden_size=LSTM_units, num_layers=2, batch_first=True)
            elif self.rnn_type == "gru":
                self.lstm = nn.GRU(input_size=conv_kernels, hidden_size=LSTM_units, num_layers=2, batch_first=True)
            elif self.rn_type == "lstm_bi":
                self.lstm = nn.LSTM(input_size=conv_kernels, hidden_size=LSTM_units, num_layers=2, batch_first=True, bidirectional=True)
            elif self.rnn_type == "gru_bi":
                self.lstm = nn.GRU(input_size=conv_kernels, hidden_size=LSTM_units, num_layers=2, batch_first=True, bidirectional=True)
            else:
                raise NotImplementedError
            
            # the first layer already has two layers, no need for lstm2


        self.out_dim = LSTM_units

        if backbone == False:
            self.classifier = Classifier(LSTM_units, n_classes)

        if state_dict is not None:
            self.load_state_dict(state_dict, strict=False)

        self.const_bnn_prior_parameters = {
                "prior_mu": 0.0,
                "prior_sigma": 1.0,
                "posterior_mu_init": 0.0,
                "posterior_rho_init": -3.0,
                "type": "Reparameterization",  # Flipout or Reparameterization
                "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
                "moped_delta": 0.5,
        }

        if self.bayesian_layers in ["all", "first", "first_last"]:
            self.conv1[0] = bnn_conv_layer(self.const_bnn_prior_parameters, self.conv1[0])


        if self.bayesian_layers in ["all"]:
            self.conv2[0] = bnn_conv_layer(self.const_bnn_prior_parameters, self.conv2[0])

        if backbone == False and self.bayesian_layers in ["all", "last", "first_last"]:
            self.classifier.classifier = bnn_linear_layer(self.const_bnn_prior_parameters, self.classifier.classifier)
            


    def forward(self, x):
        #self.lstm.flatten_parameters()
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)

        # X input should be (batch_size, seq_len, input_size)
        #breakpoint()
        x, h = self.lstm(x)
        if self.bayesian_layers in ["all"]:
            h = (h[0][:,-1,:], h[1][:,-1,:])
            x, h = self.lstm2(x, h)
        x = x[:, -1, :]

        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x

    def set_classification_head(self, classifier):
        self.classifier = classifier
        self.backbone = False

        if self.bayesian_layers in ["all", "last", "first_last"]:
            # transforms the classification layer into a bayesian layer
            try:
                self.classifier.classifier = bnn_linear_layer(self.const_bnn_prior_parameters, self.classifier.classifier).to(next(self.parameters()).device)
            except:
                print("Could not transform the classification layer into a bayesian layer")




class CorNETFrequency(nn.Module):
    def __init__(self, n_channels, n_classes, num_extra_features, conv_kernels=32, kernel_size=40, LSTM_units=128, input_size=1000, backbone=False):
        super(CorNETFrequency, self).__init__()
        
        # Instantiate the CorNET model
        self.cornet = CorNET(n_channels, n_classes, conv_kernels, kernel_size, LSTM_units, input_size, backbone=False)
        self.num_extra_features = num_extra_features
        self.input_size = input_size
        self.backbone = backbone
        self.n_classes = n_classes
        # Fully connected network for frequency features
        self.extra_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_extra_features*self.input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Output layer for the concatenated features
        self.out_dim = self.cornet.out_dim + 64

        if not self.backbone:
            self.classifier = Classifier(self.out_dim, n_classes)
        
    def forward(self, x):
        # Split the input into original features and frequency features
        original_features = x[:, :, :-self.num_extra_features]
        extra_features = x[:, :, :-self.num_extra_features].reshape(x.shape[0], -1)
        
        # Pass the original features through the CorNET model
        _, cornet_output = self.cornet(original_features)
        
        # Pass the extra features through the fully connected network
        extra_output = self.extra_fc(extra_features)
        
        # Concatenate the outputs
        x = torch.cat((cornet_output, extra_output), dim=1)
        
        # Pass through the final output layer
        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x
    
    def set_classification_head(self, classifier):
        self.classifier = classifier
        self.backbone = False

# %%
def test_model_frequency():
    # Define model parameters
    n_channels = 3
    n_classes = 1
    num_extra_features = 3
    input_size = 1000


    # Create dummy input data
    x = torch.randn(32, input_size, n_channels + num_extra_features)

    # Instantiate the model
    model = CorNETFrequency(n_channels, n_classes, num_extra_features, backbone=False)

    # Forward pass
    out, x = model(x)

    # Print output shape
    print("Output shape:", out.shape)


# %%


