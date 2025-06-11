import argparse
import torch
import numpy as np
import time
from pathlib import Path
import gymnasium as gym

from evaluate import load_agent_from_checkpoint
from src.environments import make_env


def run_demo(agent, env, episodes: int = 5, delay: float = 0.01):
    """Run interactive demo of trained agent"""
    
    print("\n=== Starting Demo ===")
    print(f"Running {episodes} episodes...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        for episode in range(episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            print(f"\n--- Episode {episode + 1} ---")
            
            while not done:
                # Render
                env.render()
                
                # Select action (deterministic)
                action = agent.select_action(obs, training=False)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                # Add delay for visualization
                time.sleep(delay)
                
            print(f"Episode Reward: {episode_reward:.2f}")
            print(f"Episode Length: {episode_length}")
            
            # Wait a bit between episodes
            if episode < episodes - 1:
                print("\nPress Enter for next episode or Ctrl+C to quit...")
                input()
                
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        
    finally:
        env.close()
        print("\nDemo finished!")


def main():
    parser = argparse.ArgumentParser(description='Demo of trained RL agent')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=5, help='Number of demo episodes')
    parser.add_argument('--delay', type=float, default=0.01, help='Delay between steps (seconds)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    # Load agent
    print(f"Loading checkpoint from {args.checkpoint}...")
    agent, config = load_agent_from_checkpoint(args.checkpoint, args.device)
    
    # Create environment with rendering
    env_config = config['environment'].copy()
    env_config['render_mode'] = 'human'
    env = make_env(env_config)
    
    # Run demo
    run_demo(agent, env, episodes=args.episodes, delay=args.delay)


if __name__ == "__main__":
    main()