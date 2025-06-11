import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .base import BaseAgent
from ..models import create_model


class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def add(self, obs, action, reward, value, log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def get(self, device: str = "cuda"):
        observations = torch.FloatTensor(np.array(self.observations)).to(device)
        actions = torch.LongTensor(self.actions).to(device)
        rewards = torch.FloatTensor(self.rewards).to(device)
        values = torch.FloatTensor(self.values).to(device)
        log_probs = torch.FloatTensor(self.log_probs).to(device)
        dones = torch.FloatTensor(self.dones).to(device)
        
        return observations, actions, rewards, values, log_probs, dones
    
    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
    def __len__(self):
        return len(self.observations)


class PPO(BaseAgent):
    def __init__(
        self,
        observation_space,
        action_space,
        config: Dict[str, Any],
        model_config: Dict[str, Any],
        device: str = "cuda",
    ):
        self.model_config = model_config
        self.n_steps = config.get('n_steps', 2048)
        self.n_epochs = config.get('n_epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.clip_range_vf = config.get('clip_range_vf', None)
        self.normalize_advantage = config.get('normalize_advantage', True)
        self.ent_coef = config.get('ent_coef', 0.0)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.shared_backbone = config.get('shared_backbone', False)
        
        super().__init__(observation_space, action_space, config, device)
        
        self.rollout_buffer = RolloutBuffer()
        
    def _build_networks(self):
        if self.shared_backbone:
            # Shared network with separate heads
            self.feature_extractor = create_model(
                self.model_config,
                input_shape=self.observation_shape,
                output_dim=128,  # Feature dimension
                device=self.device
            )
            
            self.policy_head = nn.Linear(128, self.n_actions).to(self.device)
            self.value_head = nn.Linear(128, 1).to(self.device)
        else:
            # Separate networks for policy and value
            policy_config = self.model_config.copy()
            policy_config['hidden_dims'] = self.config.get('policy_hidden_dims', [64, 64])
            
            self.policy_net = create_model(
                policy_config,
                input_shape=self.observation_shape,
                output_dim=self.n_actions,
                device=self.device
            )
            
            value_config = self.model_config.copy()
            value_config['hidden_dims'] = self.config.get('value_hidden_dims', [64, 64])
            
            self.value_net = create_model(
                value_config,
                input_shape=self.observation_shape,
                output_dim=1,
                device=self.device
            )
            
    def _setup_optimizers(self):
        optimizer_name = self.config.get('optimizer', 'adam')
        optimizer_params = self.config.get('optimizer_params', {})
        
        if self.shared_backbone:
            params = list(self.feature_extractor.parameters()) + \
                     list(self.policy_head.parameters()) + \
                     list(self.value_head.parameters())
        else:
            params = list(self.policy_net.parameters()) + list(self.value_net.parameters())
        
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=self.config['learning_rate'], **optimizer_params)
        elif optimizer_name == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=self.config['learning_rate'], **optimizer_params)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
    def get_action_and_value(self, obs: torch.Tensor):
        if self.shared_backbone:
            features = self.feature_extractor(obs)
            logits = self.policy_head(features)
            value = self.value_head(features)
        else:
            logits = self.policy_net(obs)
            value = self.value_net(obs)
            
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        return dist, value.squeeze(-1)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, value = self.get_action_and_value(state_tensor)
            
            if training:
                action = dist.sample()
            else:
                action = dist.probs.argmax(dim=-1)
                
            log_prob = dist.log_prob(action)
            
        if training:
            self.rollout_buffer.add(
                state,
                action.item(),
                0,  # Reward will be set later
                value.item(),
                log_prob.item(),
                False  # Done will be set later
            )
            
        return action.item()
    
    def update(self, state, action, reward, next_state, done):
        # Update the last transition with reward and done
        if len(self.rollout_buffer) > 0:
            self.rollout_buffer.rewards[-1] = reward
            self.rollout_buffer.dones[-1] = done
            
        # Train if buffer is full
        if len(self.rollout_buffer) >= self.n_steps:
            return self.train_step()
            
        return {}
    
    def train_step(self) -> Dict[str, float]:
        # Get data from rollout buffer
        observations, actions, rewards, values, old_log_probs, dones = self.rollout_buffer.get(self.device)
        
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values
        
        # Normalize advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # Flatten the batch
        b_observations = observations.reshape(-1, *self.observation_shape)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_old_log_probs = old_log_probs.reshape(-1)
        
        # Training metrics
        pg_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        
        # Update epochs
        for epoch in range(self.n_epochs):
            # Random sampling
            indices = torch.randperm(len(b_observations))
            
            for start in range(0, len(b_observations), self.batch_size):
                end = start + self.batch_size
                mb_indices = indices[start:end]
                
                mb_obs = b_observations[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]
                mb_old_log_probs = b_old_log_probs[mb_indices]
                
                # Get current policy distribution and values
                dist, values = self.get_action_and_value(mb_obs)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy()
                
                # Policy loss (PPO clip)
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.clip_range_vf is not None:
                    values_clipped = mb_returns + torch.clamp(
                        values - mb_returns, -self.clip_range_vf, self.clip_range_vf
                    )
                    value_loss = F.mse_loss(values, mb_returns)
                    value_loss_clipped = F.mse_loss(values_clipped, mb_returns)
                    value_loss = torch.max(value_loss, value_loss_clipped)
                else:
                    value_loss = F.mse_loss(values, mb_returns)
                    
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.max_grad_norm:
                    if self.shared_backbone:
                        params = list(self.feature_extractor.parameters()) + \
                                list(self.policy_head.parameters()) + \
                                list(self.value_head.parameters())
                    else:
                        params = list(self.policy_net.parameters()) + list(self.value_net.parameters())
                    torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                    
                self.optimizer.step()
                
                # Track metrics
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                with torch.no_grad():
                    clip_fraction = ((ratio - 1).abs() > self.clip_range).float().mean()
                    clip_fractions.append(clip_fraction.item())
                    
        # Clear rollout buffer
        self.rollout_buffer.clear()
        self.training_step += 1
        
        return {
            'policy_loss': np.mean(pg_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'clip_fraction': np.mean(clip_fractions),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            advantages[t] = last_advantage
            
        return advantages
    
    def get_networks(self) -> Dict[str, nn.Module]:
        if self.shared_backbone:
            return {
                'feature_extractor': self.feature_extractor,
                'policy_head': self.policy_head,
                'value_head': self.value_head
            }
        else:
            return {
                'policy_net': self.policy_net,
                'value_net': self.value_net
            }
    
    def get_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {'optimizer': self.optimizer}