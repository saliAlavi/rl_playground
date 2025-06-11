import gymnasium as gym
import numpy as np
from typing import List, Optional, Tuple, Union
from collections import deque
import torch


class ObservationNormalizer(gym.ObservationWrapper):
    def __init__(self, env, clip_obs: float = 10.0):
        super().__init__(env)
        self.clip_obs = clip_obs
        self.obs_rms = RunningMeanStd(shape=env.observation_space.shape)
        
    def observation(self, obs):
        self.obs_rms.update(obs)
        normalized_obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(normalized_obs, -self.clip_obs, self.clip_obs)


class RewardNormalizer(gym.RewardWrapper):
    def __init__(self, env, clip_reward: float = 10.0, gamma: float = 0.99):
        super().__init__(env)
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.ret_rms = RunningMeanStd(shape=())
        self.returns = 0
        
    def reward(self, reward):
        self.returns = self.returns * self.gamma + reward
        self.ret_rms.update(self.returns)
        normalized_reward = reward / np.sqrt(self.ret_rms.var + 1e-8)
        return np.clip(normalized_reward, -self.clip_reward, self.clip_reward)
    
    def reset(self, **kwargs):
        self.returns = 0
        return super().reset(**kwargs)


class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames: int):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        
        old_shape = env.observation_space.shape
        new_shape = (n_frames * old_shape[0],) + old_shape[1:]
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=new_shape,
            dtype=env.observation_space.dtype
        )
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)


class VecEnv:
    def __init__(self, env_fns: List):
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
    def reset(self):
        obs_list = []
        info_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        return np.stack(obs_list), info_list
    
    def step(self, actions):
        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                final_obs = obs
                obs, reset_info = env.reset()
                info['terminal_observation'] = final_obs
                info.update(reset_info)
            
            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(info)
            
        return (
            np.stack(obs_list),
            np.array(reward_list),
            np.array(terminated_list),
            np.array(truncated_list),
            info_list
        )
    
    def close(self):
        for env in self.envs:
            env.close()
    
    def render(self):
        return self.envs[0].render()


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        
    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > len(self.mean.shape) else 1
        self.update_from_moments(batch_mean, batch_var, batch_count)
        
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


def make_env(env_config: dict) -> gym.Env:
    env = gym.make(
        env_config['env_id'],
        render_mode=env_config.get('render_mode', None)
    )
    
    if env_config.get('max_episode_steps'):
        env = gym.wrappers.TimeLimit(env, max_episode_steps=env_config['max_episode_steps'])
    
    if env_config.get('normalize_obs', False):
        env = ObservationNormalizer(env, clip_obs=env_config.get('clip_obs', 10.0))
    
    if env_config.get('normalize_reward', False):
        env = RewardNormalizer(env, clip_reward=env_config.get('clip_reward', 10.0))
    
    if env_config.get('frame_stack', 1) > 1:
        env = FrameStack(env, n_frames=env_config['frame_stack'])
    
    return env