import numpy as np
import torch
from typing import Tuple, Optional, Dict
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity: int, observation_shape: Tuple, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.observation_shape = observation_shape
        
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
        
    def push(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ):
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        batch = {
            'observations': torch.FloatTensor(self.observations[indices]).to(self.device),
            'actions': torch.LongTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_observations': torch.FloatTensor(self.next_observations[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device),
        }
        
        return batch
    
    def __len__(self):
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        device: str = "cpu"
    ):
        super().__init__(capacity, observation_shape, device)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
    def push(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ):
        super().push(observation, action, reward, next_observation, done)
        self.priorities[self.position - 1] = self.max_priority
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
            
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        batch = super().sample(batch_size)
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        batch['weights'] = torch.FloatTensor(weights).to(self.device)
        batch['indices'] = indices
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        priorities = priorities + self.epsilon
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())