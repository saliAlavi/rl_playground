import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
from pathlib import Path
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns


class Evaluator:
    def __init__(
        self,
        env_fn,
        device: str = "cuda",
        render: bool = False,
        record_video: bool = False,
        video_dir: Optional[str] = None
    ):
        self.env_fn = env_fn
        self.device = device
        self.render = render
        self.record_video = record_video
        self.video_dir = Path(video_dir) if video_dir else None
        
        if self.video_dir:
            self.video_dir.mkdir(parents=True, exist_ok=True)
            
    def evaluate_agent(
        self,
        agent,
        n_episodes: int = 10,
        deterministic: bool = True,
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate agent for multiple episodes"""
        env = self.env_fn()
        
        if seed is not None:
            env.reset(seed=seed)
            
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            frames = [] if self.record_video and episode == 0 else None
            
            while not done:
                # Select action
                action = agent.select_action(obs, training=not deterministic)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                # Render or record
                if self.render:
                    env.render()
                    
                if frames is not None:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                        
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Save video for first episode
            if frames and len(frames) > 0:
                video_path = self.video_dir / f"eval_episode_{agent.episodes_done}.mp4"
                imageio.mimsave(str(video_path), frames, fps=30)
                
        env.close()
        
        # Compute statistics
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
        }
        
    def evaluate_checkpoints(
        self,
        agent_class,
        checkpoint_dir: str,
        n_episodes: int = 10,
        pattern: str = "*.pth"
    ) -> Dict[str, List[Dict[str, float]]]:
        """Evaluate multiple checkpoints"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoints = sorted(checkpoint_dir.glob(pattern))
        
        results = {}
        
        for checkpoint_path in checkpoints:
            # Load agent
            agent = agent_class.load_from_checkpoint(str(checkpoint_path))
            
            # Evaluate
            metrics = self.evaluate_agent(agent, n_episodes)
            
            # Store results
            checkpoint_name = checkpoint_path.stem
            results[checkpoint_name] = metrics
            
        return results
    
    def create_evaluation_plots(
        self,
        results: Dict[str, Dict[str, float]],
        save_dir: Optional[str] = None
    ):
        """Create plots from evaluation results"""
        # Extract data
        checkpoints = list(results.keys())
        mean_rewards = [results[cp]['mean_reward'] for cp in checkpoints]
        std_rewards = [results[cp]['std_reward'] for cp in checkpoints]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot rewards
        x = range(len(checkpoints))
        ax1.plot(x, mean_rewards, 'b-', label='Mean Reward')
        ax1.fill_between(
            x,
            np.array(mean_rewards) - np.array(std_rewards),
            np.array(mean_rewards) + np.array(std_rewards),
            alpha=0.3
        )
        ax1.set_xlabel('Checkpoint')
        ax1.set_ylabel('Reward')
        ax1.set_title('Evaluation Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot lengths
        mean_lengths = [results[cp]['mean_length'] for cp in checkpoints]
        ax2.plot(x, mean_lengths, 'g-', label='Mean Episode Length')
        ax2.set_xlabel('Checkpoint')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Lengths')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'evaluation_results.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig


def visualize_q_values(
    agent,
    env,
    n_steps: int = 100,
    save_path: Optional[str] = None
):
    """Visualize Q-values during an episode"""
    obs, info = env.reset()
    
    q_values_history = []
    actions_taken = []
    rewards_received = []
    
    for step in range(n_steps):
        # Get Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            q_values = agent.q_network(state_tensor).cpu().numpy()[0]
            
        q_values_history.append(q_values)
        
        # Select action
        action = agent.select_action(obs, training=False)
        actions_taken.append(action)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_received.append(reward)
        
        if terminated or truncated:
            break
            
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot Q-values
    q_values_array = np.array(q_values_history)
    for action_idx in range(q_values_array.shape[1]):
        ax1.plot(q_values_array[:, action_idx], label=f'Action {action_idx}')
        
    # Mark taken actions
    for step, action in enumerate(actions_taken):
        ax1.scatter(step, q_values_array[step, action], c='red', s=20, zorder=5)
        
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Q-value')
    ax1.set_title('Q-values During Episode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot rewards
    ax2.plot(rewards_received, 'g-')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    ax2.set_title('Rewards Received')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def create_training_summary(
    metrics_history: Dict[str, List[Tuple[int, float]]],
    save_dir: Optional[str] = None
):
    """Create comprehensive training summary plots"""
    # Set style
    sns.set_style("whitegrid")
    
    # Create subplots
    n_metrics = len(metrics_history)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    # Plot each metric
    for idx, (metric_name, values) in enumerate(metrics_history.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        if len(values) > 0:
            steps, vals = zip(*values)
            
            # Apply smoothing for noisy metrics
            if len(vals) > 100:
                window = min(100, len(vals) // 10)
                smoothed = np.convolve(vals, np.ones(window)/window, mode='valid')
                smooth_steps = steps[window//2:-window//2+1] if len(smoothed) < len(steps) else steps
                
                ax.plot(steps, vals, alpha=0.3, label='Raw')
                ax.plot(smooth_steps, smoothed, label='Smoothed')
            else:
                ax.plot(steps, vals)
                
            ax.set_xlabel('Step')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} over Training')
            if len(vals) > 100:
                ax.legend()
                
    # Remove empty subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])
        
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'training_summary.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig