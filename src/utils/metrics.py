import numpy as np
from typing import List, Dict, Optional, Deque
from collections import deque
import time


class MetricTracker:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: Dict[str, Deque[float]] = {}
        self.episode_rewards: Deque[float] = deque(maxlen=window_size)
        self.episode_lengths: Deque[int] = deque(maxlen=window_size)
        self.episode_times: Deque[float] = deque(maxlen=window_size)
        
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.episode_start_time = time.time()
        
        self.total_steps = 0
        self.total_episodes = 0
        
    def add_step_reward(self, reward: float):
        self.current_episode_reward += reward
        self.current_episode_length += 1
        self.total_steps += 1
        
    def end_episode(self):
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)
        self.episode_times.append(time.time() - self.episode_start_time)
        
        self.total_episodes += 1
        
        # Reset for next episode
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.episode_start_time = time.time()
        
    def add_metric(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.window_size)
        self.metrics[name].append(value)
        
    def get_episode_stats(self) -> Dict[str, float]:
        if len(self.episode_rewards) == 0:
            return {}
            
        return {
            'reward_mean': np.mean(self.episode_rewards),
            'reward_std': np.std(self.episode_rewards),
            'reward_min': np.min(self.episode_rewards),
            'reward_max': np.max(self.episode_rewards),
            'length_mean': np.mean(self.episode_lengths),
            'length_std': np.std(self.episode_lengths),
            'time_mean': np.mean(self.episode_times),
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
        }
        
    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return {}
            
        values = self.metrics[metric_name]
        return {
            f'{metric_name}_mean': np.mean(values),
            f'{metric_name}_std': np.std(values),
            f'{metric_name}_min': np.min(values),
            f'{metric_name}_max': np.max(values),
        }
        
    def get_all_stats(self) -> Dict[str, float]:
        stats = self.get_episode_stats()
        
        for metric_name in self.metrics:
            stats.update(self.get_metric_stats(metric_name))
            
        return stats
    
    def get_fps(self) -> float:
        if len(self.episode_times) == 0 or len(self.episode_lengths) == 0:
            return 0.0
            
        total_time = sum(self.episode_times)
        total_steps = sum(self.episode_lengths)
        
        if total_time == 0:
            return 0.0
            
        return total_steps / total_time