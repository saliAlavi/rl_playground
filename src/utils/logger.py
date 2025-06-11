import os
import json
import wandb
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path


class Logger:
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Dict[str, Any],
        use_wandb: bool = True,
        wandb_config: Optional[Dict[str, Any]] = None
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.config = config
        self.use_wandb = use_wandb
        
        # Initialize logging
        self.metrics_history = {}
        self.step = 0
        
        # Save config
        with open(self.log_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
            
        # Initialize wandb
        if self.use_wandb and wandb_config:
            wandb.init(
                project=wandb_config.get('project', 'rl-experiments'),
                entity=wandb_config.get('entity'),
                name=experiment_name,
                config=config,
                group=wandb_config.get('group'),
                tags=wandb_config.get('tags', []),
                dir=str(self.log_dir)
            )
            
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if step is None:
            step = self.step
        else:
            self.step = step
            
        # Store metrics locally
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append((step, value))
            
        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics, step=step)
            
        # Save to file periodically
        if step % 100 == 0:
            self._save_metrics()
            
    def log_episode(self, episode_metrics: Dict[str, float], episode: int):
        prefixed_metrics = {f"episode/{k}": v for k, v in episode_metrics.items()}
        prefixed_metrics['episode/number'] = episode
        self.log_metrics(prefixed_metrics, step=self.step)
        
    def log_training(self, training_metrics: Dict[str, float], step: Optional[int] = None):
        prefixed_metrics = {f"train/{k}": v for k, v in training_metrics.items()}
        self.log_metrics(prefixed_metrics, step=step)
        
    def log_evaluation(self, eval_metrics: Dict[str, float], step: Optional[int] = None):
        prefixed_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.log_metrics(prefixed_metrics, step=step)
        
    def log_model(self, model_path: str, step: Optional[int] = None):
        if self.use_wandb:
            wandb.save(model_path)
            artifact = wandb.Artifact(
                f"{self.experiment_name}_model",
                type="model",
                metadata={"step": step or self.step}
            )
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            
    def log_video(self, video_path: str, caption: str = "", step: Optional[int] = None):
        if self.use_wandb:
            wandb.log({
                "video": wandb.Video(video_path, caption=caption, fps=30)
            }, step=step or self.step)
            
    def log_figure(self, figure: plt.Figure, name: str, step: Optional[int] = None):
        # Save locally
        fig_path = self.log_dir / f"{name}_{step or self.step}.png"
        figure.savefig(fig_path, dpi=150, bbox_inches='tight')
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({name: wandb.Image(figure)}, step=step or self.step)
            
        plt.close(figure)
        
    def _save_metrics(self):
        metrics_path = self.log_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=4)
            
    def finish(self):
        self._save_metrics()
        if self.use_wandb:
            wandb.finish()
            
    def create_plots(self):
        """Create and save plots for all logged metrics"""
        for metric_name, values in self.metrics_history.items():
            if len(values) > 0:
                steps, vals = zip(*values)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(steps, vals)
                ax.set_xlabel('Step')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name} over training')
                ax.grid(True, alpha=0.3)
                
                # Save plot
                plot_dir = self.log_dir / "plots"
                plot_dir.mkdir(exist_ok=True)
                fig.savefig(plot_dir / f"{metric_name.replace('/', '_')}.png", dpi=150, bbox_inches='tight')
                plt.close(fig)