import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any

from src.agents import DQN, PPO, A2C
from src.environments import make_env, VecEnv
from src.utils import Logger, MetricTracker, get_scheduler
from src.utils.evaluation import Evaluator, create_training_summary


def set_seeds(seed: int):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_agent_class(algorithm_name: str):
    """Get agent class from algorithm name"""
    agents = {
        'dqn': DQN,
        'ppo': PPO,
        'a2c': A2C
    }
    
    if algorithm_name not in agents:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {list(agents.keys())}")
        
    return agents[algorithm_name]


def train_episode(agent, env, metric_tracker):
    """Train agent for one episode"""
    obs, info = env.reset()
    done = False
    
    while not done:
        # Select action
        action = agent.select_action(obs, training=True)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update agent
        train_metrics = agent.update(obs, action, reward, next_obs, done)
        
        # Track metrics
        metric_tracker.add_step_reward(reward)
        for key, value in train_metrics.items():
            metric_tracker.add_metric(key, value)
            
        obs = next_obs
        
    metric_tracker.end_episode()
    return train_metrics


def train_vectorized(agent, vec_env, metric_tracker, n_steps: int):
    """Train agent with vectorized environments (for PPO/A2C)"""
    obs, infos = vec_env.reset()
    
    for step in range(n_steps):
        # Select actions for all environments
        actions = agent.select_actions(obs, training=True)
        
        # Step all environments
        next_obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
        
        # Update agent
        train_metrics = agent.update_vectorized(obs, actions, rewards, next_obs, terminateds)
        
        # Track metrics
        for i, (reward, terminated, truncated) in enumerate(zip(rewards, terminateds, truncateds)):
            metric_tracker.add_step_reward(reward)
            if terminated or truncated:
                metric_tracker.end_episode()
                
        for key, value in train_metrics.items():
            metric_tracker.add_metric(key, value)
            
        obs = next_obs
        
    return train_metrics


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    # Set seeds
    set_seeds(cfg.experiment.seed)
    
    # Create directories
    checkpoint_dir = Path(cfg.experiment.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = Logger(
        log_dir=cfg.experiment.log_dir,
        experiment_name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        use_wandb=cfg.wandb.enabled,
        wandb_config=OmegaConf.to_container(cfg.wandb, resolve=True)
    )
    
    # Create environment
    if cfg.algorithm.name in ['ppo', 'a2c'] and cfg.environment.n_envs > 1:
        # Use vectorized environments for on-policy algorithms
        env_fns = [lambda: make_env(OmegaConf.to_container(cfg.environment, resolve=True)) 
                   for _ in range(cfg.environment.n_envs)]
        env = VecEnv(env_fns)
        single_env = env_fns[0]()
        observation_space = single_env.observation_space
        action_space = single_env.action_space
        single_env.close()
    else:
        env = make_env(OmegaConf.to_container(cfg.environment, resolve=True))
        observation_space = env.observation_space
        action_space = env.action_space
        
    # Create agent
    agent_class = get_agent_class(cfg.algorithm.name)
    
    if cfg.algorithm.name in ['ppo', 'a2c'] and cfg.environment.n_envs > 1:
        agent = agent_class(
            observation_space=observation_space,
            action_space=action_space,
            config=OmegaConf.to_container(cfg.algorithm, resolve=True),
            model_config=OmegaConf.to_container(cfg.model, resolve=True),
            device=cfg.experiment.device,
            n_envs=cfg.environment.n_envs
        )
    else:
        agent = agent_class(
            observation_space=observation_space,
            action_space=action_space,
            config=OmegaConf.to_container(cfg.algorithm, resolve=True),
            model_config=OmegaConf.to_container(cfg.model, resolve=True),
            device=cfg.experiment.device
        )
    
    # Setup learning rate scheduler
    scheduler = get_scheduler(
        agent.optimizer,
        cfg.algorithm.scheduler,
        OmegaConf.to_container(cfg.algorithm.scheduler_params, resolve=True),
        max_steps=cfg.experiment.max_episodes * 200  # Approximate
    )
    
    # Create evaluator
    evaluator = Evaluator(
        env_fn=lambda: make_env(OmegaConf.to_container(cfg.environment, resolve=True)),
        device=cfg.experiment.device,
        render=False,
        record_video=True,
        video_dir=cfg.experiment.result_dir
    )
    
    # Create metric tracker
    metric_tracker = MetricTracker(window_size=100)
    
    # Training loop
    episode = 0
    best_reward = -float('inf')
    
    while episode < cfg.experiment.max_episodes:
        # Train
        if cfg.algorithm.name in ['ppo', 'a2c'] and cfg.environment.n_envs > 1:
            train_metrics = train_vectorized(
                agent, env, metric_tracker, 
                n_steps=cfg.algorithm.n_steps
            )
            episode += cfg.environment.n_envs  # Multiple episodes completed
        else:
            train_metrics = train_episode(agent, env, metric_tracker)
            episode += 1
            
        agent.episodes_done = episode
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            train_metrics['learning_rate'] = agent.optimizer.param_groups[0]['lr']
            
        # Log metrics
        if episode % cfg.experiment.log_interval == 0:
            episode_stats = metric_tracker.get_episode_stats()
            all_stats = metric_tracker.get_all_stats()
            
            logger.log_episode(episode_stats, episode)
            logger.log_training(train_metrics, agent.training_step)
            
            print(f"Episode {episode} | "
                  f"Reward: {episode_stats.get('reward_mean', 0):.2f} ± {episode_stats.get('reward_std', 0):.2f} | "
                  f"Length: {episode_stats.get('length_mean', 0):.1f} | "
                  f"FPS: {metric_tracker.get_fps():.1f}")
                  
        # Evaluate
        if episode % cfg.experiment.eval_episodes == 0:
            eval_metrics = evaluator.evaluate_agent(
                agent,
                n_episodes=cfg.experiment.eval_episodes,
                deterministic=True
            )
            logger.log_evaluation(eval_metrics, agent.training_step)
            
            print(f"Evaluation | Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            
            # Save best model
            if eval_metrics['mean_reward'] > best_reward:
                best_reward = eval_metrics['mean_reward']
                best_path = checkpoint_dir / 'best_model.pth'
                agent.save_checkpoint(str(best_path))
                logger.log_model(str(best_path), agent.training_step)
                
        # Save checkpoint
        if episode % cfg.experiment.save_interval == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_ep{episode}.pth'
            agent.save_checkpoint(str(checkpoint_path))
            logger.log_model(str(checkpoint_path), agent.training_step)
            
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_eval_metrics = evaluator.evaluate_agent(
        agent,
        n_episodes=20,
        deterministic=True
    )
    logger.log_evaluation(final_eval_metrics, agent.training_step)
    print(f"Final Mean Reward: {final_eval_metrics['mean_reward']:.2f} ± {final_eval_metrics['std_reward']:.2f}")
    
    # Save final model
    final_path = checkpoint_dir / 'final_model.pth'
    agent.save_checkpoint(str(final_path))
    
    # Create plots
    logger.create_plots()
    training_fig = create_training_summary(
        logger.metrics_history,
        save_dir=cfg.experiment.plot_dir
    )
    logger.log_figure(training_fig, "training_summary", agent.training_step)
    
    # Cleanup
    if hasattr(env, 'close'):
        env.close()
    logger.finish()
    
    print(f"\nTraining completed! Results saved to {cfg.experiment.log_dir}")


if __name__ == "__main__":
    main()