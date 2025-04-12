# Meta-Learning for Snake, PuckWorld, and Pong

This repository implements a Model-Agnostic Meta-Learning (MAML) algorithm to train a reinforcement learning agent across three environments: Snake, PuckWorld, and Pong. The agent learns a shared policy initialization that can quickly adapt to each game, inspired by generalized meta-learning frameworks (e.g., [arXiv:2209.14110](https://arxiv.org/pdf/2209.14110)). The code is modular, parameterized for hyperparameter tuning, and supports batch execution for multiple configurations.

## Features
- **Environments**:
  - **Snake**: A grid-based game where the agent collects food while avoiding collisions.
  - **PuckWorld**: A continuous 2D environment where the agent moves to a target.
  - **Pong**: Atari Pong (ALE/Pong-v5) where the agent plays against an opponent.
- **MAML Algorithm**:
  - Trains a policy network with task-specific adaptation using inner-loop SGD and meta-optimization via Adam.
  - Supports distinct architectures: MLP for Snake/PuckWorld, CNN for Pong.
- **Outputs**:
  - Training/evaluation logs with per-game rewards and steps.
  - Plots: Meta-loss (`training_loss.png`), per-game rewards (`game_rewards.png`).
  - Saved models: `.pth` files for each game and iteration.
  - Videos: `.mp4` recordings of evaluation episodes.
- **Hyperparameter Tuning**:
  - Configurable parameters (e.g., learning rates, network sizes) via `config.py`.
  - Batch execution for multiple configurations.
- **Execution**:
  - Modular Python scripts for local or cluster runs (e.g., VS Code, SLURM).
  - Logging to files for easy monitoring.

## Prerequisites
- **Python**: 3.11 (matching Google Colab’s environment).
- **System**:
  - Linux/macOS/Windows with `ffmpeg` installed for video saving.
  - Optional: GPU for faster training (CUDA-compatible).
- **Dependencies**: Listed in `requirements.txt`.
- **Hardware**: At least 8GB RAM, 4 CPU cores; GPU recommended for Pong.

## Setup

1. **Clone Repository**:
   ```bash
   git clone <repository-url>
   cd meta_learning
   ```

2. **Create Virtual Environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   .\venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   - Includes `gym==0.25.2`, `ale-py>=0.10.1`, `torch`, `numpy`, `matplotlib`, `opencv-python-headless`, `imageio-ffmpeg`.
   - Installs Atari ROMs via `gym[atari,accept-rom-license]` for Pong.

4. **Install FFmpeg**:
   - **Ubuntu/Debian**:
     ```bash
     sudo apt-get install -y ffmpeg
     ```
   - **macOS**:
     ```bash
     brew install ffmpeg
     ```
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html), add to PATH.

5. **Verify Setup**:
   ```bash
   python -c "import gym; env = gym.make('ALE/Pong-v5'); print('Pong loaded successfully')"
   ```
   - Should output without errors if ROMs are installed.

## Project Structure

```
meta_learning/
├── environments.py       # Snake, PuckWorld, Pong environment classes
├── model.py             # PolicyNetwork (MLP for Snake/PuckWorld, CNN for Pong)
├── maml.py              # MAMLAgent with training/evaluation logic
├── config.py            # Hyperparameter configurations for tuning
├── main.py              # Main script to run training/evaluation
├── run_batch.sh         # Bash script for batch execution
├── requirements.txt     # Python dependencies
├── outputs/
│   ├── config_0/        # Outputs for config index 0
│   │   ├── models/      # Saved models (.pth)
│   │   ├── videos/      # Evaluation videos (.mp4)
│   │   ├── plots/       # Loss/reward plots (.png)
│   │   ├── log.txt      # Training/evaluation logs
│   ├── config_1/
│   ├── config_2/
```

## Usage

### Run a Single Experiment
Run the default configuration (index 0):
```bash
python main.py --config_idx 0
```
- Outputs save to `outputs/config_0/`.
- Logs print to console and save to `outputs/config_0/log.txt`.

### Run Batch Experiments
Execute all configurations defined in `config.py`:
```bash
chmod +x run_batch.sh
./run_batch.sh
```
- Runs configs 0, 1, 2 sequentially.
- Outputs save to `outputs/config_0/`, `outputs/config_1/`, `outputs/config_2/`.

### Hyperparameter Tuning
Edit `config.py` to add/modify `TUNING_CONFIGS`. Example:
```python
TUNING_CONFIGS = [
    {
        "num_iterations": 50,
        "k_shots": 5,
        "meta_lr": 0.001,
        "inner_lr": 0.01,
        "hidden_dims": [128, 64],
        "conv_filters": [16, 32],
        "num_episodes_eval": 3,
        "log_interval": 5,
    },
    # Add new config
    {
        "num_iterations": 100,
        "k_shots": 10,
        "meta_lr": 0.0002,
        "inner_lr": 0.02,
        "hidden_dims": [256, 128],
        "conv_filters": [32, 64],
        "num_episodes_eval": 5,
        "log_interval": 2,
    },
]
```
- Update `run_batch.sh` to loop over new config indices (e.g., `for config_idx in 0 1`).

### Cluster Submission (SLURM Example)
For clusters, create `submit_job.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=meta_learning
#SBATCH --output=outputs/config_%a/log.txt
#SBATCH --array=0-2
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

source venv/bin/activate
python main.py --config_idx $SLURM_ARRAY_TASK_ID
deactivate
```
Submit:
```bash
sbatch submit_job.sh
```
- Adjust resources (`--mem`, `--gpus`) based on your cluster.

## Outputs

- **Logs**: `outputs/config_X/log.txt`
  - Example:
    ```
    Starting meta-learning training...
    Game: snake, Inner Step: 0, Episode Reward: 0.0000, Steps: 8
    Game: puckworld, Inner Step: 0, Episode Reward: -0.6000, Steps: 20
    Game: pong, Inner Step: 0, Episode Reward: -21.0000, Steps: 150
    Iteration 0, Game: snake, Meta Episode Reward: 1.0000, Steps: 12
    ...
    Average reward after adaptation on snake: 2.0000
    Average reward after adaptation on puckworld: 0.1000
    Average reward after adaptation on pong: -15.0000
    ```
- **Plots**:
  - `outputs/config_X/plots/training_loss.png`: Meta-loss over iterations.
  - `outputs/config_X/plots/game_rewards.png`: Per-game rewards (Snake: ~1–3, PuckWorld: ~-0.5 to 0.1, Pong: ~-21 to -15).
- **Models**:
  - `outputs/config_X/models/snake_model_iter20_YYYYMMDD_HHMMSS.pth` (and for other games/iterations).
- **Videos**:
  - `outputs/config_X/videos/pong_eval_YYYYMMDD_HHMMSS.mp4` (first evaluation episode per game).

## Learning Validation

The agent’s performance is validated by:
- **Reward Trends** (`game_rewards.png`):
  - **Snake**: Increases from ~0 to 1–3 (food collection).
  - **PuckWorld**: Improves from ~-1 to ~0 or positive (target tracking).
  - **Pong**: Moves from ~-21 (losing) to ~-15 or better (ball returns).
- **Evaluation**: Post-training rewards confirm adaptation:
  - Expected: Snake > 0, PuckWorld > -1, Pong > -21.
  - If stagnant, tune `meta_lr` (e.g., 0.0005), `k_shots` (e.g., 10), or network sizes.
- **Logs**: Per-game rewards/steps printed every `log_interval` iterations.

## Troubleshooting

- **Pong ROM Error**:
  - Verify `ale-py>=0.10.1` and ROMs (`~/.local/lib/python3.11/site-packages/AutoROM/roms/pong.bin`).
  - Reinstall: `pip install gym[atari,accept-rom-license]`.
- **Video Saving**:
  - Ensure `ffmpeg` is in PATH.
  - Check `imageio-ffmpeg` installation.
- **Learning Issues**:
  - Flat rewards: Lower `meta_lr` (e.g., 0.0001), increase `num_iterations` (e.g., 100).
  - Poor Pong performance: Adjust `conv_filters` (e.g., `[32, 64]`), increase `k_shots`.
- **Memory Errors**:
  - Reduce `num_iterations` (e.g., 20) or `k_shots` (e.g., 3).
  - Use GPU if available.

## Development

- **VS Code**:
  - Install Python extension.
  - Select `venv` interpreter (`meta_learning/venv/bin/python`).
  - Run/debug `main.py` with `--config_idx`.
- **Add Environments**:
  - Extend `environments.py` with new classes.
  - Update `main.py` to include in `envs`.
- **New Metrics**:
  - Modify `maml.py` (e.g., add variance in `evaluate_and_record`).
- **Tuning**:
  - Expand `TUNING_CONFIGS` in `config.py`.
  - Compare `game_rewards.png` across `outputs/config_X/`.

## License

This project uses Atari ROMs via `gym[accept-rom-license]`, intended for research purposes per [ALE licensing](https://github.com/mgbellemare/Arcade-Learning-Environment#rom-management). Ensure compliance with licensing terms.

## Acknowledgments

- Based on MAML concepts from [arXiv:2209.14110](https://arxiv.org/pdf/2209.14110).
- Built with `gym`, `torch`, and `ale-py` for reinforcement learning.
