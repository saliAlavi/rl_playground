import numpy as np
from typing import Optional, Dict, Any
import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(optimizer, scheduler_name: str, scheduler_params: Dict[str, Any], max_steps: Optional[int] = None):
    if scheduler_name is None or scheduler_name == 'none':
        return None
        
    elif scheduler_name == 'linear':
        def linear_schedule(current_step: int):
            if max_steps is None:
                return 1.0
            start_factor = scheduler_params.get('start_factor', 1.0)
            end_factor = scheduler_params.get('end_factor', 0.1)
            progress = min(current_step / max_steps, 1.0)
            return start_factor + (end_factor - start_factor) * progress
            
        return lr_scheduler.LambdaLR(optimizer, linear_schedule)
        
    elif scheduler_name == 'cosine':
        T_max = scheduler_params.get('T_max', max_steps or 1000)
        eta_min = scheduler_params.get('eta_min', 0)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        
    elif scheduler_name == 'step':
        step_size = scheduler_params.get('step_size', 100)
        gamma = scheduler_params.get('gamma', 0.9)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
    elif scheduler_name == 'multistep':
        milestones = scheduler_params.get('milestones', [100, 200, 300])
        gamma = scheduler_params.get('gamma', 0.1)
        return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        
    elif scheduler_name == 'exponential':
        gamma = scheduler_params.get('gamma', 0.99)
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        
    elif scheduler_name == 'plateau':
        mode = scheduler_params.get('mode', 'min')
        factor = scheduler_params.get('factor', 0.5)
        patience = scheduler_params.get('patience', 10)
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
        
    elif scheduler_name == 'warmup_cosine':
        warmup_steps = scheduler_params.get('warmup_steps', 100)
        T_max = scheduler_params.get('T_max', max_steps or 1000)
        eta_min = scheduler_params.get('eta_min', 0)
        
        def warmup_cosine_schedule(current_step: int):
            if current_step < warmup_steps:
                return current_step / warmup_steps
            else:
                progress = (current_step - warmup_steps) / (T_max - warmup_steps)
                return eta_min + (1 - eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
                
        return lr_scheduler.LambdaLR(optimizer, warmup_cosine_schedule)
        
    elif scheduler_name == 'polynomial':
        power = scheduler_params.get('power', 0.9)
        total_steps = scheduler_params.get('total_steps', max_steps or 1000)
        
        def polynomial_schedule(current_step: int):
            return (1 - current_step / total_steps) ** power
            
        return lr_scheduler.LambdaLR(optimizer, polynomial_schedule)
        
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


class LinearSchedule:
    """Simple linear schedule for epsilon greedy"""
    def __init__(self, start_value: float, end_value: float, num_steps: int):
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps
        
    def get_value(self, step: int) -> float:
        if step >= self.num_steps:
            return self.end_value
        progress = step / self.num_steps
        return self.start_value + (self.end_value - self.start_value) * progress
        
        
class ExponentialSchedule:
    """Exponential decay schedule"""
    def __init__(self, start_value: float, end_value: float, decay_rate: float):
        self.start_value = start_value
        self.end_value = end_value
        self.decay_rate = decay_rate
        
    def get_value(self, step: int) -> float:
        value = self.start_value * (self.decay_rate ** step)
        return max(value, self.end_value)