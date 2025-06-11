import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from .base import BaseAgent
from ..models import create_model


class A2CRolloutBuffer:
    def __init__(self, n_steps: int, n_envs: int):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.pos = 0
        self.full = False
        
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def add(self, obs, action, reward, value, log_prob, done):
        if len(self.observations) == self.n_steps:
            self.observations[self.pos] = obs
            self.actions[self.pos] = action
            self.rewards[self.pos] = reward
            self.values[self.pos] = value
            self.log_probs[self.pos] = log_prob
            self.dones[self.pos] = done
        else:
            self.observations.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.dones.append(done)
            
        self.pos = (self.pos + 1) % self.n_steps
        self.full = self.full or self.pos == 0
        
    def get(self, last_values: torch.Tensor, last_dones: torch.Tensor, device: str = "cuda"):
        observations = torch.FloatTensor(np.array(self.observations)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).to(device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(device)
        values = torch.FloatTensor(np.array(self.values)).to(device)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        dones = torch.FloatTensor(np.array(self.dones)).to(device)
        
        return observations, actions, rewards, values, log_probs, dones, last_values, last_dones
    
    def reset(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.pos = 0
        self.full = False


class A2C(BaseAgent):
    def __init__(
        self,
        observation_space,
        action_space,
        config: Dict[str, Any],
        model_config: Dict[str, Any],
        device: str = "cuda",
        n_envs: int = 1
    ):
        self.model_config = model_config
        self.n_steps = config.get('n_steps', 5)
        self.n_envs = n_envs
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 1.0)
        self.ent_coef = config.get('ent_coef', 0.01)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.normalize_advantage = config.get('normalize_advantage', False)
        self.shared_backbone = config.get('shared_backbone', True)
        
        super().__init__(observation_space, action_space, config, device)
        
        self.rollout_buffer = A2CRolloutBuffer(self.n_steps, self.n_envs)
        self.step_count = 0
        
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
        optimizer_name = self.config.get('optimizer', 'rmsprop')
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
            
        return logits, value.squeeze(-1)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, value = self.get_action_and_value(state_tensor)
            
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            if training:
                action = dist.sample()
            else:
                action = probs.argmax(dim=-1)
                
            log_prob = dist.log_prob(action)
            
        if training and hasattr(self, '_current_rollout_data'):
            self._current_rollout_data = {
                'observation': state,
                'action': action.item(),
                'value': value.item(),
                'log_prob': log_prob.item()
            }
            
        return action.item()
    
    def select_actions(self, states: np.ndarray, training: bool = True) -> np.ndarray:
        """For vectorized environments"""
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            logits, values = self.get_action_and_value(states_tensor)
            
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            if training:
                actions = dist.sample()
            else:
                actions = probs.argmax(dim=-1)
                
            log_probs = dist.log_prob(actions)
            
        if training:
            self._current_rollout_data = {
                'observations': states,
                'actions': actions.cpu().numpy(),
                'values': values.cpu().numpy(),
                'log_probs': log_probs.cpu().numpy()
            }
            
        return actions.cpu().numpy()
    
    def update(self, state, action, reward, next_state, done):
        if hasattr(self, '_current_rollout_data'):
            self.rollout_buffer.add(
                self._current_rollout_data['observation'],
                self._current_rollout_data['action'],
                reward,
                self._current_rollout_data['value'],
                self._current_rollout_data['log_prob'],
                done
            )
            delattr(self, '_current_rollout_data')
            
        self.step_count += 1
        
        # Train if buffer is full
        if self.step_count % self.n_steps == 0:
            # Get last values for bootstrap
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                _, last_values = self.get_action_and_value(next_state_tensor)
                last_dones = torch.FloatTensor([done]).to(self.device)
                
            return self.train_step(last_values, last_dones)
            
        return {}
    
    def update_vectorized(self, observations, actions, rewards, next_observations, dones):
        """For vectorized environments"""
        if hasattr(self, '_current_rollout_data'):
            for i in range(self.n_envs):
                self.rollout_buffer.add(
                    self._current_rollout_data['observations'][i],
                    self._current_rollout_data['actions'][i],
                    rewards[i],
                    self._current_rollout_data['values'][i],
                    self._current_rollout_data['log_probs'][i],
                    dones[i]
                )
            delattr(self, '_current_rollout_data')
            
        self.step_count += self.n_envs
        
        # Train if buffer is full
        if self.step_count >= self.n_steps * self.n_envs:
            # Get last values for bootstrap
            with torch.no_grad():
                next_states_tensor = torch.FloatTensor(next_observations).to(self.device)
                _, last_values = self.get_action_and_value(next_states_tensor)
                last_dones = torch.FloatTensor(dones).to(self.device)
                
            metrics = self.train_step(last_values, last_dones)
            self.step_count = 0
            return metrics
            
        return {}
    
    def train_step(self, last_values: torch.Tensor, last_dones: torch.Tensor) -> Dict[str, float]:
        self.training_step += 1
        
        # Get data from rollout buffer
        observations, actions, rewards, values, old_log_probs, dones, last_values, last_dones = \
            self.rollout_buffer.get(last_values, last_dones, self.device)
            
        # Compute returns and advantages
        returns = self._compute_returns(rewards, dones, last_values, last_dones)
        
        if self.gae_lambda < 1.0:
            advantages = self._compute_gae(rewards, values, dones, last_values, last_dones)
        else:
            advantages = returns - values
            
        # Normalize advantages
        if self.normalize_advantage and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # Flatten batch for single update
        b_observations = observations.reshape(-1, *self.observation_shape)
        b_actions = actions.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        
        # Get current policy distribution and values
        logits, values = self.get_action_and_value(b_observations)
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(b_actions)
        entropy = dist.entropy()
        
        # Policy loss
        policy_loss = -(log_probs * b_advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, b_returns)
        
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
        
        # Clear rollout buffer
        self.rollout_buffer.reset()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': loss.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_return': returns.mean().item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def _compute_returns(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        last_values: torch.Tensor,
        last_dones: torch.Tensor
    ) -> torch.Tensor:
        returns = torch.zeros_like(rewards)
        
        # Bootstrap from last value
        if self.n_envs == 1:
            next_return = last_values * (1 - last_dones)
        else:
            next_return = last_values * (1 - last_dones)
            
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma * next_return * (1 - dones[t])
            next_return = returns[t]
            
        return returns
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_values: torch.Tensor,
        last_dones: torch.Tensor
    ) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_values
                next_done = last_dones
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
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