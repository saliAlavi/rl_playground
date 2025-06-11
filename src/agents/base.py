import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class BaseAgent(ABC):
    def __init__(
        self,
        observation_space,
        action_space,
        config: Dict[str, Any],
        device: str = "cuda",
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.n_actions = action_space.n
        self.observation_shape = observation_space.shape
        
        self.training_step = 0
        self.episodes_done = 0
        
        self._build_networks()
        self._setup_optimizers()
        
    @abstractmethod
    def _build_networks(self):
        pass
    
    @abstractmethod
    def _setup_optimizers(self):
        pass
    
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def train_step(self, *args, **kwargs) -> Dict[str, float]:
        pass
    
    def eval(self):
        for net in self.get_networks().values():
            net.eval()
    
    def train(self):
        for net in self.get_networks().values():
            net.train()
    
    def get_networks(self) -> Dict[str, nn.Module]:
        return {}
    
    def save_checkpoint(self, path: str):
        checkpoint = {
            'training_step': self.training_step,
            'episodes_done': self.episodes_done,
            'config': self.config,
        }
        
        for name, net in self.get_networks().items():
            checkpoint[f'{name}_state_dict'] = net.state_dict()
        
        for name, opt in self.get_optimizers().items():
            checkpoint[f'{name}_optimizer_state_dict'] = opt.state_dict()
        
        if hasattr(self, 'schedulers'):
            for name, scheduler in self.schedulers.items():
                checkpoint[f'{name}_scheduler_state_dict'] = scheduler.state_dict()
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.training_step = checkpoint['training_step']
        self.episodes_done = checkpoint['episodes_done']
        
        for name, net in self.get_networks().items():
            net.load_state_dict(checkpoint[f'{name}_state_dict'])
        
        for name, opt in self.get_optimizers().items():
            opt.load_state_dict(checkpoint[f'{name}_optimizer_state_dict'])
        
        if hasattr(self, 'schedulers'):
            for name, scheduler in self.schedulers.items():
                if f'{name}_scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint[f'{name}_scheduler_state_dict'])
    
    def get_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {}
    
    def to(self, device):
        self.device = torch.device(device)
        for net in self.get_networks().values():
            net.to(self.device)
        return self