#%%

import torch
import torch.nn as nn
import sys
import numpy as np
sys.path.append("/local/home/lhauptmann/thesis/CL-HAR")

from models.backbones import *

#%%

def test_CNN_AE():
    # Create an instance of CNN_AE
    n_channels = 3
    n_classes = 1
    n_channels_output = 1
    embedded_size = 128
    input_size = 1000

    model = CNN_AE(n_channels, n_classes, embedded_size, n_channels_output,  input_size)
    
    # Create random input data
    batch_size = 5
    sequence_length = 1000
    #input_data = torch.randn(batch_size, sequence_length, n_channels)
    input_data = np.repeat(np.repeat(np.sin(np.linspace(0, np.pi*2*10, sequence_length)).reshape(1,-1), batch_size, axis=0).reshape(batch_size, -1, 1), n_channels, axis=2)
    input_data = input_data + np.random.normal(0, 0.1, input_data.shape)
    input_data = torch.from_numpy(input_data).float()
    
    # Forward pass
    output, encoded = model(input_data)
    
    # Display the shapes of output and encoded tensors
    print("Output Shape:", output.shape)
    print("Encoded Shape:", encoded.shape)

    # Check if shapes match expected shapes
    expected_output_shape = torch.Size([batch_size, sequence_length, n_channels])
    expected_encoded_shape = torch.Size([batch_size, sequence_length, n_channels_output])
    
    assert output.shape == expected_output_shape, f"Expected: {expected_output_shape}, Got: {output.shape}"
    assert encoded.shape == expected_encoded_shape, f"Expected: {expected_encoded_shape}, Got: {encoded.shape}"

def test_AE():
    # Create an instance of AE
    n_channels = 3
    n_classes = 1
    embdedded_size = 128
    input_size = 1000

    model = AE(n_channels, input_size, n_classes, embdedded_size)
    
    # Create random input data
    batch_size = 5
    sequence_length = 1000
    input_data = torch.randn(batch_size,sequence_length,  n_channels )
    
    # Forward pass
    output, encoded = model(input_data)
    
    # Display the shapes of output and encoded tensors
    print("Output Shape:", output.shape)
    print("Encoded Shape:", encoded.shape)

    # Check if shapes match expected shapes
    expected_output_shape = torch.Size([batch_size, sequence_length, n_channels])
    expected_encoded_shape = torch.Size([batch_size, outdim])
    
    assert output.shape == expected_output_shape, f"Expected: {expected_output_shape}, Got: {output.shape}"
    assert encoded.shape == expected_encoded_shape, f"Expected: {expected_encoded_shape}, Got: {encoded.shape}"

#%%
test_CNN_AE()
# %%
