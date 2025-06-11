import argparse
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import json

from src.agents import DQN, PPO, A2C
from src.environments import make_env
from src.utils.evaluation import Evaluator, visualize_q_values


def load_agent_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load agent from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Determine agent class
    algorithm_name = config['algorithm']['name']
    agent_classes = {
        'dqn': DQN,
        'ppo': PPO,
        'a2c': A2C
    }
    
    if algorithm_name not in agent_classes:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
    # Create environment to get spaces
    env = make_env(config['environment'])
    
    # Create agent
    agent_class = agent_classes[algorithm_name]
    agent = agent_class(
        observation_space=env.observation_space,
        action_space=env.action_space,
        config=config['algorithm'],
        model_config=config['model'],
        device=device
    )
    
    # Load checkpoint
    agent.load_checkpoint(checkpoint_path)
    env.close()
    
    return agent, config


def main():
    parser = argparse.ArgumentParser(description='Evaluate RL agent')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--record', action='store_true', help='Record video')
    parser.add_argument('--video-dir', type=str, default='experiments/videos', help='Directory to save videos')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--visualize-q', action='store_true', help='Visualize Q-values (DQN only)')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load agent
    print(f"Loading checkpoint from {args.checkpoint}...")
    agent, config = load_agent_from_checkpoint(args.checkpoint, args.device)
    
    # Create evaluator
    evaluator = Evaluator(
        env_fn=lambda: make_env(config['environment']),
        device=args.device,
        render=args.render,
        record_video=args.record,
        video_dir=args.video_dir
    )
    
    # Evaluate agent
    print(f"\nEvaluating for {args.episodes} episodes...")
    metrics = evaluator.evaluate_agent(
        agent,
        n_episodes=args.episodes,
        deterministic=True,
        seed=args.seed
    )
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Min Reward: {metrics['min_reward']:.2f}")
    print(f"Max Reward: {metrics['max_reward']:.2f}")
    print(f"Mean Episode Length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    
    # Save results
    results_path = Path(args.checkpoint).parent / f"eval_results_{Path(args.checkpoint).stem}.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nResults saved to {results_path}")
    
    # Visualize Q-values if requested (DQN only)
    if args.visualize_q and config['algorithm']['name'] == 'dqn':
        print("\nVisualizing Q-values...")
        env = make_env(config['environment'])
        fig = visualize_q_values(
            agent, env,
            n_steps=200,
            save_path=Path(args.checkpoint).parent / "q_values_visualization.png"
        )
        env.close()
        print("Q-values visualization saved!")


if __name__ == "__main__":
    main()