import numpy as np
import torch
import cv2
from typing import Tuple, Optional, Union


class ImageProcessor:
    def __init__(
        self,
        resize: Optional[Tuple[int, int]] = None,
        grayscale: bool = False,
        normalize: bool = True,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None
    ):
        self.resize = resize
        self.grayscale = grayscale
        self.normalize = normalize
        self.mean = mean or (0.485, 0.456, 0.406)  # ImageNet defaults
        self.std = std or (0.229, 0.224, 0.225)
        
    def process(self, image: np.ndarray) -> np.ndarray:
        # Handle different input formats
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            
        # Resize if needed
        if self.resize is not None:
            image = cv2.resize(image, self.resize)
            
        # Convert to grayscale if needed
        if self.grayscale and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=-1)
            
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            
        # Apply normalization
        if self.normalize:
            if self.grayscale:
                image = (image - 0.5) / 0.5
            else:
                # Channel-wise normalization
                for i in range(3):
                    image[..., i] = (image[..., i] - self.mean[i]) / self.std[i]
                    
        # Convert to CHW format
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
            
        return image
    
    def process_batch(self, images: Union[list, np.ndarray]) -> np.ndarray:
        if isinstance(images, list):
            processed = [self.process(img) for img in images]
            return np.stack(processed)
        else:
            # Assume numpy array with batch dimension
            processed = []
            for i in range(len(images)):
                processed.append(self.process(images[i]))
            return np.stack(processed)


class StateProcessor:
    def __init__(
        self,
        normalize: bool = True,
        clip_range: Optional[Tuple[float, float]] = None,
        dtype: torch.dtype = torch.float32
    ):
        self.normalize = normalize
        self.clip_range = clip_range
        self.dtype = dtype
        
        # Running statistics for normalization
        self.running_mean = None
        self.running_std = None
        self.count = 0
        
    def update_statistics(self, states: np.ndarray):
        """Update running mean and std"""
        batch_mean = np.mean(states, axis=0)
        batch_std = np.std(states, axis=0)
        
        if self.running_mean is None:
            self.running_mean = batch_mean
            self.running_std = batch_std
            self.count = len(states)
        else:
            # Welford's online algorithm for updating mean and std
            total_count = self.count + len(states)
            delta = batch_mean - self.running_mean
            
            self.running_mean = self.running_mean + delta * len(states) / total_count
            
            # Update variance
            m_a = self.running_std ** 2 * self.count
            m_b = batch_std ** 2 * len(states)
            M2 = m_a + m_b + delta ** 2 * self.count * len(states) / total_count
            self.running_std = np.sqrt(M2 / total_count)
            
            self.count = total_count
            
    def process(self, state: np.ndarray, update_stats: bool = True) -> np.ndarray:
        if update_stats and self.normalize:
            self.update_statistics(state.reshape(1, -1))
            
        # Normalize if needed
        if self.normalize and self.running_mean is not None:
            state = (state - self.running_mean) / (self.running_std + 1e-8)
            
        # Clip if needed
        if self.clip_range is not None:
            state = np.clip(state, self.clip_range[0], self.clip_range[1])
            
        return state.astype(np.float32)
    
    def process_batch(self, states: np.ndarray, update_stats: bool = True) -> np.ndarray:
        if update_stats and self.normalize:
            self.update_statistics(states)
            
        # Normalize if needed
        if self.normalize and self.running_mean is not None:
            states = (states - self.running_mean) / (self.running_std + 1e-8)
            
        # Clip if needed
        if self.clip_range is not None:
            states = np.clip(states, self.clip_range[0], self.clip_range[1])
            
        return states.astype(np.float32)
    
    def save_statistics(self, path: str):
        np.savez(
            path,
            mean=self.running_mean,
            std=self.running_std,
            count=self.count
        )
        
    def load_statistics(self, path: str):
        data = np.load(path)
        self.running_mean = data['mean']
        self.running_std = data['std']
        self.count = data['count']