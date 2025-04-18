Meta-Learning for Snake, PuckWorld, and Pong

This repository implements a Model-Agnostic Meta-Learning (MAML) algorithm to train a reinforcement learning agent across three environments: Snake, PuckWorld, and Pong. The agent learns a shared policy initialization that can quickly adapt to each game, inspired by generalized meta-learning frameworks (e.g., arXiv:2209.14110). The code is modular, parameterized for hyperparameter tuning, and supports batch execution for multiple configurations.

Features





Environments:





Snake: A grid-based game where the agent collects food while avoiding collisions.



PuckWorld: A continuous 2D environment where the agent moves to a target.



Pong: Atari Pong (ALE/Pong-v5) where the agent plays against an opponent.



MAML Algorithm:





Trains a policy network with task-specific adaptation using inner-loop SGD and meta-optimization via Adam.



Supports distinct architectures: MLP for Snake/PuckWorld, CNN for Pong.



Outputs:





Training logs with detailed metrics (e.g., meta-loss, rewards, steps, entropy, memory usage).



Plots: Meta-loss (training_loss_<timestamp>.png), per-game rewards (game_rewards_<timestamp>.png).



Saved models: .pth files for each game and iteration.



Videos: .mp4 recordings of evaluation episodes.



Metrics: CSV file (training_metrics.csv) with iteration-wise statistics.



Hyperparameter Tuning:





Configurable parameters (e.g., learning rates, network sizes) defined in main.py.



Batch execution for multiple configurations via run_batch.sh.



Execution:





Modular Python scripts for local or cluster runs (e.g., VS Code, SLURM).



Comprehensive logging to files and console for monitoring.

Prerequisites





Python: 3.8+ (tested with 3.8 to match dependency compatibility).



System:





Linux/macOS/Windows with ffmpeg installed for video saving.



Optional: GPU for faster training (CUDA-compatible, 8GB+ VRAM recommended).



Dependencies: Listed in requirements.txt.



Hardware: At least 8GB RAM, 4 CPU cores; GPU recommended for Pong.

Setup





Clone Repository:

git clone https://github.com/lakujayanth/ITCS_6050_Group4.git
cd ITCS_6050_Group4/Meta_Learning



Create Virtual Environment (recommended):

python3 -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate  # Windows



Install Dependencies:

pip install -r requirements.txt





Includes numpy==1.24.4, gym==0.26.2, ale-py==0.8.1, torch==2.0.1, matplotlib==3.7.1, opencv-python-headless==4.7.0.72, imageio-ffmpeg==0.4.8, psutil==5.9.5.



Installs Atari ROMs via gym[atari,accept-rom-license].



Install FFmpeg:





Ubuntu/Debian:

sudo apt-get install -y ffmpeg



macOS:

brew install ffmpeg



Windows: Download from ffmpeg.org, add to PATH.



Verify Setup:

python -c "import gym; env = gym.make('ALE/Pong-v5'); print('Pong loaded successfully')"





Should output without errors if ROMs are installed.

Project Structure

Meta_Learning/
├── environment.py        # Snake, PuckWorld, Pong environment classes
├── model.py              # PolicyNetwork (MLP for Snake/PuckWorld, CNN for Pong)
├── maml.py               # MAMLAgent with training/evaluation logic
├── utils.py              # Utilities for evaluation, video recording, plotting
├── main.py               # Main script with hyperparameter configs and training
├── run_batch.sh          # Bash script for batch execution
├── requirements.txt      # Python dependencies
├── logs/                 # Training logs (training_<timestamp>.log)
├── models/               # Saved models (.pth) and metrics (training_metrics.csv)
├── videos/               # Evaluation videos (.mp4)
├── checkpoints/          # Training checkpoints (.pth)
├── plots/                # Loss/reward plots (.png)
├── experiments/          # Batch run outputs (config_<index>_<timestamp>/)

Usage

Run a Single Experiment

Run the default configuration (index 0 in main.py):

python main.py





Outputs save to logs/, models/, videos/, checkpoints/, and plots/.



Logs print to console and save to logs/training_<timestamp>.log.



Metrics (meta-loss, rewards, etc.) save to models/training_metrics.csv.

Run Batch Experiments

Execute all configurations defined in main.py (CONFIGS):

chmod +x run_batch.sh
./run_batch.sh





Runs configurations 0, 1, 2 sequentially.



Outputs save to experiments/config_0_<timestamp>/, experiments/config_1_<timestamp>/, experiments/config_2_<timestamp>/, each containing models/, videos/, checkpoints/, plots/, and logs/.



To modify configurations, edit CONFIG_INDICES in run_batch.sh.

Hyperparameter Tuning

The main.py file defines three configurations in the CONFIGS list. Key parameters include:





CONFIGS[0] (Default):





num_iterations: 100



k_shots: 5



meta_lr: 0.01



inner_lr: 0.2



hidden_dims: [256, 128]



conv_filters: [32, 64]



entropy_bonus: 0.05



epsilon_decay: 0.98



num_episodes_eval: 3



log_interval: 5



CONFIGS[1]:





num_iterations: 200 (longer training)



k_shots: 3 (fewer shots)



meta_lr: 0.005 (more stable)



inner_lr: 0.1



hidden_dims: [512, 256] (larger network)



conv_filters: [64, 128] (more features)



entropy_bonus: 0.01 (less exploration)



epsilon_decay: 0.95 (slower decay)



CONFIGS[2]:





num_iterations: 100



k_shots: 10 (more shots)



meta_lr: 0.02 (faster updates)



inner_lr: 0.3



hidden_dims: [128, 64] (smaller network)



conv_filters: [16, 32] (fewer features)



entropy_bonus: 0.1 (more exploration)



epsilon_decay: 0.99 (very slow decay)

To run a specific configuration, modify main.py to call main(CONFIGS[<index>]). To add new configurations, extend the CONFIGS list in main.py. Example:

CONFIGS.append({
    "num_iterations": 150,
    "k_shots": 7,
    "meta_lr": 0.002,
    "inner_lr": 0.15,
    "hidden_dims": [256, 256],
    "conv_filters": [32, 64],
    "entropy_bonus": 0.05,
    "epsilon_decay": 0.97,
    "num_episodes_eval": 3,
    "log_interval": 5,
    "batch_size": 2,
    "max_steps_per_episode": 500,
    "discount_factor": 0.99,
    "epsilon_start": 0.5,
    "epsilon_end": 0.01,
    "frame_stack": 6,
    "grad_clip": 1.0,
    "dropout_rate": 0.3
})

Update run_batch.sh to include the new index (e.g., CONFIG_INDICES=(0 1 2 3)).

Cluster Submission (SLURM Example)

For clusters, create submit_job.sh:

#!/bin/bash
#SBATCH --job-name=meta_learning
#SBATCH --output=experiments/config_%a_%j/log.txt
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

source venv/bin/activate
python main.py
deactivate

Submit:

sbatch submit_job.sh





Adjust resources (--mem, --gpus, --time) based on your cluster. Note: Modify main.py to select CONFIGS[$SLURM_ARRAY_TASK_ID] if running specific configs.

Outputs





Logs: logs/training_<timestamp>.log (or experiments/config_X_<timestamp>/logs/ for batch runs)





Contains iteration details, losses, rewards, entropy, memory usage, and more.



Example:

2025-04-17 12:00:00 [INFO] Starting Iteration 0/100
2025-04-17 12:00:01 [DEBUG] Game: snake, Inner Step: 0, Episode Reward: 0.5000, Steps: 10, Entropy: 1.0986
2025-04-17 12:00:02 [INFO] Iteration 0/100, Meta Loss: 123.4567, Avg Steps: 15.67, Grad Norm: 2.3456



Plots:





plots/training_loss_<timestamp>.png: Meta-loss over iterations.



plots/game_rewards_<timestamp>.png: Per-game rewards (Snake, PuckWorld, Pong).



Models:





models/<env_name>_model_iter<iteration>_<timestamp>.pth (e.g., snake_model_iter20_20250417_120000.pth).



Videos:





videos/<env_name>_eval_<timestamp>.mp4 (e.g., pong_eval_20250417_120000.mp4 for the first evaluation episode).



Checkpoints:





checkpoints/checkpoint.pth: Saves model states, optimizer, losses, and rewards for resuming training.



Metrics:





models/training_metrics.csv: Columns include iteration, meta_loss, snake_reward, puckworld_reward, pong_reward, epsilon, avg_steps, grad_norm, entropy, memory_mb.

Learning Validation

The agent’s performance is validated by:





Reward Trends (game_rewards_<timestamp>.png):





Snake: Expected to increase from ~0 to 1–10 (food collection).



PuckWorld: Should improve from ~-1 to ~0 or positive (target tracking).



Pong: Should move from ~-21 (losing) to ~-15 or better (ball returns).



Evaluation: Post-training rewards confirm adaptation:





Expected: Snake > 0, PuckWorld > -1, Pong > -21.



If stagnant, try CONFIGS[1] (lower meta_lr=0.005, more num_iterations=200) or CONFIGS[2] (more k_shots=10).



Logs/Metrics: Check training_metrics.csv for trends:





meta_loss: Should stabilize or decrease.



<env_name>_reward: Should trend upward.



entropy: Should remain non-zero to ensure exploration.



memory_mb: Monitor for memory leaks (should stay <16GB on GPU).

Previous runs showed:





Meta-Loss: High variability (~-400 to +400), suggesting meta_lr=0.005 or more k_shots may help.



Snake: Peaks at ~10, but inconsistent; more iterations (200) recommended.



PuckWorld: Peaks at ~10, stabilizes ~0; larger networks (hidden_dims=[512, 256]) may improve.



Pong: Stays ~-1; needs more conv_filters (e.g., [64, 128]) and slower epsilon_decay (e.g., 0.95).

Troubleshooting





Pong ROM Error:





Verify ale-py==0.8.1 and ROMs (~/.local/lib/python3.8/site-packages/AutoROM/roms/pong.bin).



Reinstall: pip install gym[atari,accept-rom-license].



Video Saving:





Ensure ffmpeg is in PATH (which ffmpeg).



Check imageio-ffmpeg==0.4.8 installation.



Learning Issues:





Flat rewards: Lower meta_lr (e.g., 0.005), increase num_iterations (e.g., 200).



Poor Pong performance: Increase conv_filters (e.g., [64, 128]), k_shots (e.g., 10).



High loss variability: Try CONFIGS[1] for stability.



Memory Errors:





Reduce k_shots (e.g., 3) or use CONFIGS[2] (smaller network).



Ensure GPU memory is cleared (torch.cuda.empty_cache() is included).



CUDA Errors:





Verify PyTorch CUDA support: python -c "import torch; print(torch.cuda.is_available())".



Update NVIDIA drivers/CUDA toolkit.

Development





VS Code:





Install Python extension.



Select venv interpreter (Meta_Learning/venv/bin/python).



Run/debug main.py with default config.



Add Environments:





Extend environment.py with new classes (inherit gym.Env or similar).



Update main.py to include in envs dictionary.



New Metrics:





Modify maml.py to log additional metrics in training_metrics.csv (e.g., reward variance).



Update utils.py for new plots.



Tuning:





Expand CONFIGS in main.py with new hyperparameters.



Compare game_rewards_<timestamp>.png and training_metrics.csv across runs.



Example: Test meta_lr=0.002, hidden_dims=[256, 256] for balanced capacity.

License

This project uses Atari ROMs via gym[accept-rom-license], intended for research purposes per ALE licensing. Ensure compliance with licensing terms.

Acknowledgments





Based on MAML concepts from arXiv:2209.14110.



Built with gym, torch, and ale-py for reinforcement learning.



Developed as part of ITCS_6050_Group4 coursework.



Last updated: April 17, 2025
