# RL Playground - Professional Reinforcement Learning Framework

A high-quality, enterprise-grade reinforcement learning framework for solving the CartPole task and other OpenAI Gym environments. This project implements multiple state-of-the-art RL algorithms with comprehensive features including configurable backbones, wandb logging, and professional code structure.

## Features

### Algorithms Implemented
- **DQN (Deep Q-Network)** with prioritized experience replay
- **PPO (Proximal Policy Optimization)** with GAE and parallel environments
- **A2C (Advantage Actor-Critic)** with vectorized environments

### Model Architectures
- **MLP**: Fully connected networks with configurable layers
- **CNN**: Convolutional networks for image-based observations
- **Pretrained Backbones**: Support for ResNet, EfficientNet, ViT, MobileNet, DenseNet from torchvision

### Professional Features
- 📊 **Wandb Integration**: Real-time experiment tracking and visualization
- 🔧 **Hydra Configuration**: Flexible experiment configuration system
- 💾 **Checkpoint System**: Save/resume training with full state preservation
- 📈 **Comprehensive Metrics**: Episode rewards, training losses, learning curves
- 🎥 **Video Recording**: Record evaluation episodes for visualization
- 🧪 **Evaluation Suite**: Systematic agent evaluation with statistics
- 🎨 **Plotting Utilities**: Automatic generation of training summaries
- ⚡ **Vectorized Environments**: Efficient parallel training for on-policy methods
- 🔄 **Learning Rate Schedulers**: Multiple scheduling strategies (linear, cosine, step, etc.)

## Project Structure

```
rl_playground/
├── configs/                    # Hydra configuration files
│   ├── algorithm/             # Algorithm configs (dqn.yaml, ppo.yaml, a2c.yaml)
│   ├── environment/           # Environment configs (cartpole.yaml)
│   ├── model/                 # Model configs (mlp.yaml, cnn.yaml, resnet.yaml, vit.yaml)
│   └── config.yaml           # Main configuration
├── src/
│   ├── agents/               # RL algorithms
│   │   ├── base.py          # Base agent class
│   │   ├── dqn.py           # DQN implementation
│   │   ├── ppo.py           # PPO implementation
│   │   └── a2c.py           # A2C implementation
│   ├── environments/         # Environment wrappers
│   ├── models/              # Neural network architectures
│   │   ├── mlp.py           # Multi-layer perceptron
│   │   ├── cnn.py           # Convolutional networks
│   │   ├── backbones.py     # Pretrained model support
│   │   └── factory.py       # Model factory
│   ├── utils/               # Utilities
│   │   ├── buffer.py        # Experience replay buffers
│   │   ├── logger.py        # Wandb and local logging
│   │   ├── metrics.py       # Metric tracking
│   │   ├── schedulers.py    # Learning rate schedulers
│   │   └── evaluation.py    # Evaluation and visualization
│   └── data/                # Data processing utilities
├── experiments/             # Generated experiment outputs
│   ├── checkpoints/        # Model checkpoints
│   ├── logs/              # Training logs
│   ├── plots/             # Generated plots
│   └── results/           # Evaluation results
├── train.py               # Main training script
├── evaluate.py            # Evaluation script
├── demo.py               # Interactive demo script
└── requirements.txt      # Dependencies
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd rl_playground

# Install dependencies
pip install -r requirements.txt

# Set up wandb (optional)
wandb login
```

### Basic Training

Train a DQN agent on CartPole:
```bash
python train.py
```

Train with different algorithms:
```bash
# PPO
python train.py algorithm=ppo

# A2C
python train.py algorithm=a2c

# DQN with different backbone
python train.py algorithm=dqn model=cnn
```

### Advanced Configuration

Train with custom hyperparameters:
```bash
python train.py \
    algorithm=ppo \
    algorithm.learning_rate=0.001 \
    algorithm.n_epochs=20 \
    experiment.max_episodes=2000 \
    model=resnet \
    model.pretrained=true
```

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py experiments/checkpoints/best_model.pth --episodes 10 --render
```

### Interactive Demo

Run an interactive demo:
```bash
python demo.py experiments/checkpoints/best_model.pth --episodes 5
```

## Configuration System

The project uses Hydra for configuration management. All hyperparameters can be easily modified through YAML files or command line overrides.

### Algorithm Configuration

Choose from different algorithms:
- `algorithm=dqn`: Deep Q-Network
- `algorithm=ppo`: Proximal Policy Optimization  
- `algorithm=a2c`: Advantage Actor-Critic

### Model Configuration

Select different model architectures:
- `model=mlp`: Multi-layer perceptron (default for CartPole)
- `model=cnn`: Convolutional neural network
- `model=resnet`: ResNet backbone from torchvision
- `model=vit`: Vision Transformer backbone

### Environment Configuration

Currently supports CartPole-v1 with options for:
- Observation normalization
- Reward normalization  
- Frame stacking
- Vectorized environments

## Wandb Integration

The framework automatically logs to Weights & Biases:
- Training metrics (loss, rewards, episode length)
- Hyperparameters
- Model checkpoints
- Training videos
- Evaluation results

Configure in `configs/config.yaml`:
```yaml
wandb:
  enabled: true
  project: rl-cartpole
  entity: your-wandb-entity
```

## Model Architectures

### Pretrained Vision Models

The framework supports various pretrained models from torchvision:

```yaml
# ResNet
model=resnet
model.backbone=resnet18  # resnet18, resnet34, resnet50, resnet101

# EfficientNet  
model=efficientnet
model.backbone=efficientnet_b0  # efficientnet_b0, efficientnet_b1, efficientnet_b2

# Vision Transformer
model=vit
model.backbone=vit_b_16  # vit_b_16, vit_b_32, vit_l_16

# MobileNet
model=mobilenet
model.backbone=mobilenet_v2  # mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
```

### Custom Network Architecture

Easily customize network architectures:

```yaml
model:
  type: mlp
  hidden_dims: [256, 256, 128]
  activation: relu
  dropout: 0.1
  layer_norm: true
```

## Learning Rate Schedulers

Multiple scheduler options available:

```yaml
algorithm:
  scheduler: cosine  # linear, cosine, step, multistep, exponential, plateau
  scheduler_params:
    T_max: 1000
    eta_min: 0.0001
```

## Extending the Framework

### Adding New Algorithms

1. Create new algorithm file in `src/agents/`
2. Inherit from `BaseAgent`
3. Add configuration file in `configs/algorithm/`
4. Register in `train.py`

### Adding New Environments

1. Create environment wrapper in `src/environments/`
2. Add configuration in `configs/environment/`
3. Update environment factory

### Adding New Models

1. Implement model in `src/models/`
2. Add to model factory
3. Create configuration file

## Results and Logging

All results are automatically saved to the `experiments/` directory:
- **Checkpoints**: Model weights and training state
- **Logs**: Training metrics and episode statistics
- **Plots**: Training curves and evaluation results
- **Videos**: Recorded evaluation episodes

## Performance Tips

1. **Use GPU**: Set `experiment.device=cuda` for faster training
2. **Vectorized Environments**: Use `environment.n_envs > 1` for PPO/A2C
3. **Wandb Offline**: Set `WANDB_MODE=offline` for faster training
4. **Batch Size**: Tune `algorithm.batch_size` based on GPU memory

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use smaller models
2. **Slow Training**: Enable GPU acceleration or use vectorized environments
3. **Wandb Login**: Run `wandb login` before training

### Debug Mode

Enable debug logging:
```bash
python train.py experiment.log_interval=1 hydra.verbose=true
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes linting (black, flake8)
5. Submit pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{rl_playground,
  title={RL Playground: Professional Reinforcement Learning Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/rl_playground}
}
```