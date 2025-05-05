import math
import torch
from pathlib import Path

# ──────────────── Hyperparameters & Paths ────────────────
EPISODES         = 20_000
INNER_STEPS      = 5
MAML_INNER_LR    = 1e-2
MAML_META_LR     = 1e-3
MAML_NUM_TASKS   = 5
EVAL_ADAPT_STEPS = 200
EWC_LAMBDA       = 500
LEARNING_RATE    = 1e-3
LOG_INTERVAL     = 2_000
N_SEEDS          = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Path("outputs_v7").mkdir(exist_ok=True)

# ──────────── Task Definitions ────────────
TASK_PARAMS = [
    (1.0, 0.0), (2.0, math.pi/4), (0.5, math.pi/2),
    (1.5, math.pi/3), (0.8, math.pi/6),
    (3.5, 2.4), (0.2, 0.1), (4.8, 0.0)
]
TRANSFER_PAIRS = [
    ((1.0, 0.0), (2.0, math.pi/4)),
    ((2.0, math.pi/4), (0.5, math.pi/2)),
    ((0.5, math.pi/2), (3.5, 2.4)),
    ((1.5, math.pi/3), (0.2, 0.1)),
]