import os
import torch
import numpy as np
import logging
from datetime import datetime
from environment import SnakeGame, PuckWorldGame, PongEnv
from maml import MAMLAgent
from utils import evaluate_and_record, plot_training_results

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Configure logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = "models"
VIDEO_DIR = "videos"
CHECKPOINT_DIR = "checkpoints"
PLOT_DIR = "plots"
LOG_DIR = "logs"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f'training_{timestamp}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hyperparameter configurations
CONFIGS = [
    {
        "num_iterations": 100,
        "k_shots": 5,
        "meta_lr": 0.01,
        "inner_lr": 0.2,
        "hidden_dims": [256, 128],
        "conv_filters": [32, 64],
        "num_episodes_eval": 3,
        "log_interval": 5,
        "batch_size": 2,
        "max_steps_per_episode": 500,
        "entropy_bonus": 0.05,
        "discount_factor": 0.99,
        "epsilon_start": 0.5,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.98,
        "frame_stack": 6,
        "grad_clip": 1.0,
        "dropout_rate": 0.3
    },
    {
        "num_iterations": 200,
        "k_shots": 3,
        "meta_lr": 0.005,
        "inner_lr": 0.1,
        "hidden_dims": [512, 256],
        "conv_filters": [64, 128],
        "num_episodes_eval": 3,
        "log_interval": 5,
        "batch_size": 2,
        "max_steps_per_episode": 500,
        "entropy_bonus": 0.01,
        "discount_factor": 0.99,
        "epsilon_start": 0.5,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.95,
        "frame_stack": 6,
        "grad_clip": 1.0,
        "dropout_rate": 0.3
    },
    {
        "num_iterations": 100,
        "k_shots": 10,
        "meta_lr": 0.02,
        "inner_lr": 0.3,
        "hidden_dims": [128, 64],
        "conv_filters": [16, 32],
        "num_episodes_eval": 3,
        "log_interval": 5,
        "batch_size": 2,
        "max_steps_per_episode": 500,
        "entropy_bonus": 0.1,
        "discount_factor": 0.99,
        "epsilon_start": 0.5,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.99,
        "frame_stack": 6,
        "grad_clip": 1.0,
        "dropout_rate": 0.3
    }
]

def main(config):
    # Initialize environments
    envs = {
        "snake": SnakeGame(),
        "puckworld": PuckWorldGame(),
        "pong": PongEnv(frame_stack=config["frame_stack"])
    }

    # Reset CUDA state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.init()
        logger.info("CUDA state reset")

    # Initialize agent
    agent = MAMLAgent(envs, config, SAVE_DIR, CHECKPOINT_DIR)
    logger.info("Starting meta-learning training...")
    start_iteration, losses = agent.load_checkpoint()
    if start_iteration > 0:
        logger.info(f"Resuming training from iteration {start_iteration}")
    else:
        losses = []

    # Train
    new_losses = agent.train(num_iterations=config["num_iterations"] - start_iteration, k_shots=config["k_shots"])
    losses.extend(new_losses)

    # Visualize results
    plot_training_results(losses, agent.game_rewards, envs, PLOT_DIR)

    # Evaluate and record videos
    for env_name, env in envs.items():
        avg_reward = evaluate_and_record(agent, env, env_name, num_episodes=config["num_episodes_eval"], video_dir=VIDEO_DIR)
        logger.info(f"Average reward after adaptation on {env_name}: {avg_reward:.4f}")

if __name__ == "__main__":
    # Run with the first configuration (default)
    main(CONFIGS[0])
    # To experiment with other configurations, you can loop through CONFIGS or select a specific one:
    # for config in CONFIGS:
    #     logger.info(f"Running with config: {config}")
    #     main(config)