
# Generalized Transfer Learning in RL

This project investigates transfer learning for the following games:

- 🕹️ Pong (Atari)
- 🐍 Snake (custom)
- 🟡 PuckWorld (custom with distance-to-goal tracking)

---

## Setup

```bash
pip install requirements.txt
AutoROM --accept-license
```

---

## Training Scripts

```bash
python train_pong.py         # Train PPO agent on Pong
python train_snake.py        # Train PPO agent on Snake
python train_puckworld.py    # Train PPO agent on PuckWorld (tracks distance)
```
---

## Directory Overview

- `ppo_framework.py` – General PPO + CNN setup
- `*_env.py` – Environment wrappers
- `train_*.py` – Training scripts
- `videos/` – Output video folder (future)

---
