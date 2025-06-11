#!/usr/bin/env python3
"""
Quick start script to test the RL playground setup.
This script runs a simple training session to verify everything works.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and print output"""
    print(f"\n🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
        
    if check and result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        sys.exit(1)
    
    return result


def main():
    print("🎯 RL Playground Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("train.py").exists():
        print("❌ Please run this script from the rl_playground root directory")
        sys.exit(1)
    
    print("\n1. Checking Python environment...")
    run_command([sys.executable, "--version"])
    
    print("\n2. Testing import of key modules...")
    test_imports = [
        "import torch; print(f'PyTorch: {torch.__version__}')",
        "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')",
        "import numpy; print(f'NumPy: {numpy.__version__}')",
        "import hydra; print(f'Hydra: {hydra.__version__}')",
    ]
    
    for import_cmd in test_imports:
        try:
            run_command([sys.executable, "-c", import_cmd])
        except Exception as e:
            print(f"❌ Import failed: {e}")
            print("Please install requirements: pip install -r requirements.txt")
            sys.exit(1)
    
    print("\n3. Running quick training test (DQN, 50 episodes)...")
    
    # Run a short training session
    train_cmd = [
        sys.executable, "train.py",
        "experiment.max_episodes=50",
        "experiment.log_interval=10",
        "experiment.save_interval=25",
        "experiment.eval_episodes=5",
        "wandb.enabled=false",  # Disable wandb for quick test
        "algorithm=dqn",
        "algorithm.learning_rate=0.001",
        "algorithm.epsilon_decay=0.99"
    ]
    
    try:
        result = run_command(train_cmd, check=False)
        if result.returncode == 0:
            print("✅ Training completed successfully!")
        else:
            print("⚠️  Training had issues but script completed")
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted by user")
    
    print("\n4. Checking generated files...")
    
    # Check if experiment files were created
    experiment_dirs = [
        "experiments/checkpoints",
        "experiments/logs", 
        "experiments/plots"
    ]
    
    for exp_dir in experiment_dirs:
        if Path(exp_dir).exists():
            files = list(Path(exp_dir).rglob("*"))
            print(f"✅ {exp_dir}: {len(files)} files")
        else:
            print(f"❌ {exp_dir}: directory not found")
    
    print("\n5. Testing evaluation (if checkpoint exists)...")
    
    # Find latest checkpoint
    checkpoint_dir = Path("experiments/checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.rglob("*.pth"))
        if checkpoints:
            latest_checkpoint = sorted(checkpoints)[-1]
            print(f"Found checkpoint: {latest_checkpoint}")
            
            eval_cmd = [
                sys.executable, "evaluate.py",
                str(latest_checkpoint),
                "--episodes", "3",
                "--device", "cpu"  # Use CPU for compatibility
            ]
            
            try:
                run_command(eval_cmd, check=False)
                print("✅ Evaluation completed!")
            except Exception as e:
                print(f"⚠️  Evaluation failed: {e}")
        else:
            print("❌ No checkpoints found")
    
    print("\n" + "=" * 50)
    print("🎉 Quick start completed!")
    print("\nNext steps:")
    print("1. Try different algorithms: python train.py algorithm=ppo")
    print("2. Experiment with models: python train.py model=cnn")
    print("3. Enable wandb logging: python train.py wandb.enabled=true")
    print("4. Run longer training: python train.py experiment.max_episodes=1000")
    print("\nFor more options, check the README.md file!")


if __name__ == "__main__":
    main()