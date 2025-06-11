import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = 'relu',
        dropout: float = 0.0,
        layer_norm: bool = False,
        spectral_norm: bool = False,
        output_activation: Optional[str] = None
    ):
        super().__init__()
        
        self.activation_fn = self._get_activation(activation)
        self.output_activation_fn = self._get_activation(output_activation) if output_activation else None
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            
            if spectral_norm and i < len(dims) - 2:  # Don't apply to output layer
                layer = nn.utils.spectral_norm(layer)
                
            layers.append(layer)
            
            if i < len(dims) - 2:  # Don't add activation/norm/dropout after output layer
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                    
                layers.append(self.activation_fn)
                
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        output = self.model(x)
        
        if self.output_activation_fn:
            output = self.output_activation_fn(output)
            
        return output
    
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