import math
import csv
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from config import DEVICE

# ────────── Model Definition ──────────
class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)

# ────────── Data Generation ──────────
def generate_sine(amplitude, phase, n=100):
    x = np.random.uniform(-5, 5, (n, 1)).astype(np.float32)
    y = (amplitude * np.sin(x + phase)).astype(np.float32)
    return torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).to(DEVICE)

# ────────── Utilities ──────────
def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += (p.grad.detach() ** 2).sum().item()
    return math.sqrt(total)


def write_csv(path: Path, rows, header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)