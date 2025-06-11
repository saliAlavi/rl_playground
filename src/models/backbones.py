import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Optional


class PretrainedBackbone(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        output_dim: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_trainable_layers: int = 2,
        hidden_dims: List[int] = [512, 256],
        activation: str = 'relu',
        dropout: float = 0.1,
        input_size: Optional[int] = None
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.backbone, self.feature_dim = self._create_backbone(backbone_name, pretrained)
        
        if freeze_backbone:
            self._freeze_backbone(num_trainable_layers)
            
        # Build head network
        self.activation_fn = self._get_activation(activation)
        
        head_layers = []
        dims = [self.feature_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            head_layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if i < len(dims) - 2:  # Don't add activation/dropout after output layer
                head_layers.append(self.activation_fn)
                
                if dropout > 0:
                    head_layers.append(nn.Dropout(dropout))
                    
        self.head = nn.Sequential(*head_layers)
        
    def _create_backbone(self, backbone_name: str, pretrained: bool):
        if backbone_name.startswith('resnet'):
            if backbone_name == 'resnet18':
                backbone = models.resnet18(pretrained=pretrained)
            elif backbone_name == 'resnet34':
                backbone = models.resnet34(pretrained=pretrained)
            elif backbone_name == 'resnet50':
                backbone = models.resnet50(pretrained=pretrained)
            elif backbone_name == 'resnet101':
                backbone = models.resnet101(pretrained=pretrained)
            else:
                raise ValueError(f"Unknown ResNet variant: {backbone_name}")
                
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            
        elif backbone_name.startswith('efficientnet'):
            if backbone_name == 'efficientnet_b0':
                backbone = models.efficientnet_b0(pretrained=pretrained)
            elif backbone_name == 'efficientnet_b1':
                backbone = models.efficientnet_b1(pretrained=pretrained)
            elif backbone_name == 'efficientnet_b2':
                backbone = models.efficientnet_b2(pretrained=pretrained)
            else:
                raise ValueError(f"Unknown EfficientNet variant: {backbone_name}")
                
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            
        elif backbone_name.startswith('vit'):
            if backbone_name == 'vit_b_16':
                backbone = models.vit_b_16(pretrained=pretrained)
            elif backbone_name == 'vit_b_32':
                backbone = models.vit_b_32(pretrained=pretrained)
            elif backbone_name == 'vit_l_16':
                backbone = models.vit_l_16(pretrained=pretrained)
            else:
                raise ValueError(f"Unknown ViT variant: {backbone_name}")
                
            feature_dim = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
            
        elif backbone_name == 'mobilenet_v2':
            backbone = models.mobilenet_v2(pretrained=pretrained)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            
        elif backbone_name == 'mobilenet_v3_small':
            backbone = models.mobilenet_v3_small(pretrained=pretrained)
            feature_dim = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
            
        elif backbone_name == 'mobilenet_v3_large':
            backbone = models.mobilenet_v3_large(pretrained=pretrained)
            feature_dim = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
            
        elif backbone_name == 'densenet121':
            backbone = models.densenet121(pretrained=pretrained)
            feature_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
            
        elif backbone_name == 'densenet169':
            backbone = models.densenet169(pretrained=pretrained)
            feature_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
            
        return backbone, feature_dim
    
    def _freeze_backbone(self, num_trainable_layers: int):
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze last few layers
        if hasattr(self.backbone, 'layer4'):  # ResNet
            layers = [self.backbone.layer4, self.backbone.layer3, self.backbone.layer2, self.backbone.layer1]
            for i in range(min(num_trainable_layers, len(layers))):
                for param in layers[i].parameters():
                    param.requires_grad = True
                    
        elif hasattr(self.backbone, 'features'):  # EfficientNet, DenseNet
            # Unfreeze last few blocks
            children = list(self.backbone.features.children())
            for i in range(len(children) - num_trainable_layers, len(children)):
                for param in children[i].parameters():
                    param.requires_grad = True
                    
        elif hasattr(self.backbone, 'encoder_layer'):  # ViT
            # Unfreeze last few transformer blocks
            num_blocks = len(self.backbone.encoder.layers)
            for i in range(num_blocks - num_trainable_layers, num_blocks):
                for param in self.backbone.encoder.layers[i].parameters():
                    param.requires_grad = True
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.head(features)
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


def get_backbone(name: str, pretrained: bool = True):
    """Helper function to get just the backbone without head"""
    if name.startswith('resnet'):
        if name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
        elif name == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
        elif name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone: {name}")
            
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        
    elif name.startswith('efficientnet'):
        if name == 'efficientnet_b0':
            backbone = models.efficientnet_b0(pretrained=pretrained)
        elif name == 'efficientnet_b1':
            backbone = models.efficientnet_b1(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone: {name}")
            
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        
    else:
        raise ValueError(f"Unknown backbone: {name}")
        
    return backbone, feature_dim