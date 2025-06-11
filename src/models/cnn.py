import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class CNN(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_dim: int,
        channels: List[int] = [32, 64, 64],
        kernel_sizes: List[int] = [8, 4, 3],
        strides: List[int] = [4, 2, 1],
        padding: List[int] = [0, 0, 0],
        hidden_dims: List[int] = [512],
        activation: str = 'relu',
        dropout: float = 0.0,
        batch_norm: bool = True,
        pool_type: Optional[str] = None
    ):
        super().__init__()
        
        assert len(channels) == len(kernel_sizes) == len(strides) == len(padding)
        
        self.activation_fn = self._get_activation(activation)
        
        # Build convolutional layers
        conv_layers = []
        in_channels = input_shape[0] if len(input_shape) == 3 else 1
        
        for i, (out_channels, kernel_size, stride, pad) in enumerate(
            zip(channels, kernel_sizes, strides, padding)
        ):
            conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)
            )
            
            if batch_norm:
                conv_layers.append(nn.BatchNorm2d(out_channels))
                
            conv_layers.append(self.activation_fn)
            
            if pool_type == 'max' and i < len(channels) - 1:
                conv_layers.append(nn.MaxPool2d(2))
            elif pool_type == 'avg' and i < len(channels) - 1:
                conv_layers.append(nn.AvgPool2d(2))
                
            in_channels = out_channels
            
        self.conv_net = nn.Sequential(*conv_layers)
        
        # Calculate flatten dimension
        self.flatten_dim = self._calculate_flatten_dim(input_shape)
        
        # Build fully connected layers
        fc_layers = []
        dims = [self.flatten_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            fc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if i < len(dims) - 2:  # Don't add activation/dropout after output layer
                fc_layers.append(self.activation_fn)
                
                if dropout > 0:
                    fc_layers.append(nn.Dropout(dropout))
                    
        self.fc_net = nn.Sequential(*fc_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        conv_features = self.conv_net(x)
        conv_features = conv_features.view(conv_features.size(0), -1)
        output = self.fc_net(conv_features)
        
        return output
    
    def _calculate_flatten_dim(self, input_shape: Tuple[int, ...]) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            if len(dummy_input.shape) == 3:
                dummy_input = dummy_input.unsqueeze(1)
            conv_output = self.conv_net(dummy_input)
            return int(np.prod(conv_output.shape[1:]))
    
    def _get_activation(self, activation: str):
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'mish': nn.Mish()
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
            
        return activations[activation]