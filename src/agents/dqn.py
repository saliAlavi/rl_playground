import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .base import BaseAgent
from ..utils.buffer import ReplayBuffer, PrioritizedReplayBuffer
from ..models import create_model


class DQN(BaseAgent):
    def __init__(
        self,
        observation_space,
        action_space,
        config: Dict[str, Any],
        model_config: Dict[str, Any],
        device: str = "cuda",
    ):
        self.model_config = model_config
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.batch_size = config.get('batch_size', 64)
        self.target_update_interval = config.get('target_update_interval', 10)
        self.buffer_size = config.get('buffer_size', 100000)
        self.min_buffer_size = config.get('min_buffer_size', 1000)
        
        super().__init__(observation_space, action_space, config, device)
        
        # Initialize replay buffer
        use_prioritized = config.get('use_prioritized_replay', False)
        if use_prioritized:
            self.buffer = PrioritizedReplayBuffer(
                self.buffer_size,
                observation_space.shape,
                alpha=config.get('alpha', 0.6),
                beta=config.get('beta', 0.4),
                device=device
            )
        else:
            self.buffer = ReplayBuffer(
                self.buffer_size,
                observation_space.shape,
                device=device
            )
        
    def _build_networks(self):
        # Create Q-network and target network
        self.q_network = create_model(
            self.model_config,
            input_shape=self.observation_shape,
            output_dim=self.n_actions,
            device=self.device
        )
        
        self.target_network = create_model(
            self.model_config,
            input_shape=self.observation_shape,
            output_dim=self.n_actions,
            device=self.device
        )
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
    def _setup_optimizers(self):
        optimizer_name = self.config.get('optimizer', 'adam')
        optimizer_params = self.config.get('optimizer_params', {})
        
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.q_network.parameters(),
                lr=self.config['learning_rate'],
                **optimizer_params
            )
        elif optimizer_name == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(
                self.q_network.parameters(),
                lr=self.config['learning_rate'],
                **optimizer_params
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return self.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
            
        return action
    
    def update(self, state, action, reward, next_state, done):
        # Store transition in replay buffer
        self.buffer.push(state, action, reward, next_state, done)
        
        # Update epsilon
        if self.training_step > 0:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Train if buffer has enough samples
        if len(self.buffer) >= self.min_buffer_size:
            metrics = self.train_step()
            
            # Update target network
            if self.training_step % self.target_update_interval == 0:
                self._update_target_network()
                
            return metrics
        
        return {}
    
    def train_step(self) -> Dict[str, float]:
        self.training_step += 1
        
        # Sample batch from replay buffer
        batch = self.buffer.sample(self.batch_size)
        
        # Compute current Q values
        current_q_values = self.q_network(batch['observations'])
        current_q_values = current_q_values.gather(1, batch['actions'].unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(batch['next_observations'])
            next_q_values, _ = next_q_values.max(dim=1)
            target_q_values = batch['rewards'] + self.gamma * next_q_values * (1 - batch['dones'])
        
        # Compute loss
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            td_errors = (current_q_values - target_q_values).abs()
            loss = (batch['weights'] * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
            
            # Update priorities
            self.buffer.update_priorities(batch['indices'], td_errors.detach().cpu().numpy())
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.get('max_grad_norm'):
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config['max_grad_norm'])
            
        self.optimizer.step()
        
        metrics = {
            'loss': loss.item(),
            'q_value_mean': current_q_values.mean().item(),
            'epsilon': self.epsilon,
            'buffer_size': len(self.buffer)
        }
        
        return metrics
    
    def _update_target_network(self):
        # Soft update
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def get_networks(self) -> Dict[str, nn.Module]:
        return {
            'q_network': self.q_network,
            'target_network': self.target_network
        }
    
    def get_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {
            'optimizer': self.optimizer
        }