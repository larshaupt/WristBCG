#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Disout(nn.Module):
    def __init__(self, p=0.2):
        super(Disout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        mask = torch.empty_like(x).bernoulli_(self.p)
        return x * mask / (1 - self.p)

class Downsample1D(nn.Module):
    def __init__(self, scale_factor):
        super(Downsample1D, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        # Apply interpolate function for downsampling
        input_length = x.size(2)
        output_length = input_length // self.scale_factor
        x = F.interpolate(x, size=output_length, mode='linear')
        return x
    

class CNNBackboneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, do_downsample=False, groups=1):
        super(CNNBackboneBlock, self).__init__()
        self.groups = groups
        self.conv = nn.Conv1d(in_channels*groups, out_channels*groups, kernel_size, padding=kernel_size//2, groups=groups)
        self.bn = nn.BatchNorm1d(out_channels*groups)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(pool_size)
        self.disout = Disout()
        self.do_downsample = do_downsample
        if self.do_downsample:
            self.downsample = Downsample1D(pool_size)


    def forward(self, x):
        if isinstance(x, tuple):
            # got x and downsampled input as input
            x, x_downsampled = x
            batch_size, num_channels, length = x.shape
            x = x.reshape(batch_size, num_channels//self.groups, self.groups, length ) + x_downsampled.reshape(batch_size, 1, self.groups, length)
            x = x.reshape(batch_size, num_channels, length)
        else:
            x_downsampled = x
        if self.do_downsample:
            x_downsampled = self.downsample(x_downsampled)
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.disout(x)
        if self.do_downsample:
            return (x, x_downsampled)
        return x
class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_size, num_heads=6, hidden_size=24, dropout=0.2, groups:int=1, num_layers=6):
        super(TransformerEncoderBlock, self).__init__()
        self.transfomer_layer = nn.TransformerEncoderLayer(d_model=input_size//groups, nhead=num_heads, dim_feedforward=hidden_size, dropout=0.2)
        self.self_attention = nn.TransformerEncoder(self.transfomer_layer, num_layers=num_layers)
        self.groups = groups

    def forward(self, x):
        x = torch.squeeze(x)
        x = x.reshape(x.shape[0], x.shape[1]//self.groups, self.groups, x.shape[2])
        x = x.reshape(x.shape[0], x.shape[1], x.shape[3]*self.groups)
        x = torch.permute(x, (2,0,1))
        # Self-attention layer
        x = self.self_attention(x)
        # Residual connection and layer normalization
        x = torch.mean(x, dim=0)
        return x

class HRCTPNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=1, backbone:bool=True):
        super(HRCTPNet, self).__init__()
        self.cnn_blocks = nn.Sequential(
            CNNBackboneBlock(1, 24, 30, 5, do_downsample = True, groups=num_channels),
            CNNBackboneBlock(24, 24, 22, 5, do_downsample = True, groups=num_channels),
            CNNBackboneBlock(24, 24, 10, 2, do_downsample = True, groups=num_channels),
            CNNBackboneBlock(24, 24, 6, 2, do_downsample = False, groups=num_channels)
        )
        self.transformer_block = TransformerEncoderBlock(24*num_channels, 6,hidden_size=24, groups=1, num_layers = 6)
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.backbone = backbone
    
        """         self.dense = nn.Sequential(
                    nn.Linear(36, 24),
                    nn.ReLU(),
                )
        """
        self.out_dim = 24*num_channels

        if not self.backbone:
            self.classifier = nn.Linear(self.out_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn_blocks(x)

        x = self.transformer_block(x)  # Global average pooling
        
        if not self.backbone:
            out = self.classifier(x)
            return out, x

        return x
    
            
    def set_classification_head(self, classifier):
        self.classifier = classifier
        self.backbone = False
#%%
def test_model(model, input_size=(3, 1000)):
    """
    Function to test the HRCTPNet model with dummy input data.

    Args:
    - model (nn.Module): The HRCTPNet model to be tested.
    - input_size (tuple): The size of the input tensor (num_channels, sequence_length).

    Returns:
    - predictions (torch.Tensor): The model's predictions.
    """
    # Generate random input data
    batch_size = 64
    num_channels, sequence_length = input_size
    input_data = torch.randn(batch_size, sequence_length, num_channels)
    print(input_data.shape)
    # Ensure the model is in evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        predictions = model(input_data)

    return predictions

# Example usage:
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
""" model = HRCTPNet(backbone=True)
predictions = test_model(model)
print("Predictions:", predictions)

summary(model) """

# %%
def test_transformer_encoder_block():
    """
    Function to test the Transformer Encoder Block.

    Returns:
    - output (torch.Tensor): The output tensor.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize the Transformer Encoder Block
    input_size = 12
    transformer_block = TransformerEncoderBlock(input_size)

    # Generate random input tensor
    batch_size = 64
    sequence_length = 5
    input_tensor = torch.randn(batch_size, input_size, 1, sequence_length)

    # Forward pass
    output = transformer_block(input_tensor)

    return output

""" # Example usage:
output = test_transformer_encoder_block()
print("Output shape:", output.shape) """
# %%
