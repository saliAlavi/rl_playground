import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple
from .mlp import MLP
from .cnn import CNN
from .backbones import PretrainedBackbone


def create_model(
    config: Dict[str, Any],
    input_shape: Tuple,
    output_dim: int,
    device: str = "cuda"
) -> nn.Module:
    model_type = config.get('type', 'mlp')
    
    if model_type == 'mlp':
        model = MLP(
            input_dim=input_shape[0] if len(input_shape) == 1 else np.prod(input_shape),
            output_dim=output_dim,
            hidden_dims=config.get('hidden_dims', [128, 128]),
            activation=config.get('activation', 'relu'),
            dropout=config.get('dropout', 0.0),
            layer_norm=config.get('layer_norm', False),
            spectral_norm=config.get('spectral_norm', False)
        )
    elif model_type == 'cnn':
        model = CNN(
            input_shape=input_shape,
            output_dim=output_dim,
            channels=config.get('channels', [32, 64, 64]),
            kernel_sizes=config.get('kernel_sizes', [8, 4, 3]),
            strides=config.get('strides', [4, 2, 1]),
            padding=config.get('padding', [0, 0, 0]),
            hidden_dims=config.get('hidden_dims', [512]),
            activation=config.get('activation', 'relu'),
            dropout=config.get('dropout', 0.0),
            batch_norm=config.get('batch_norm', True)
        )
    elif model_type == 'pretrained':
        model = PretrainedBackbone(
            backbone_name=config.get('backbone', 'resnet18'),
            output_dim=output_dim,
            pretrained=config.get('pretrained', True),
            freeze_backbone=config.get('freeze_backbone', False),
            num_trainable_layers=config.get('num_trainable_layers', 2),
            hidden_dims=config.get('hidden_dims', [512, 256]),
            activation=config.get('activation', 'relu'),
            dropout=config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)