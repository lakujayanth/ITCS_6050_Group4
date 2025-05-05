import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from experiments import transfer_experiment, maml_experiment, ewc_experiment
from config import EPISODES, N_SEEDS, EVAL_ADAPT_STEPS, TRANSFER_PAIRS, TASK_PARAMS
from core import DEVICE, grad_norm

# ────────── Plot Helpers ──────────
def plot_with_band(xs, mean, std, label):
    plt.fill_between(xs, mean - std, mean + std, alpha=0.25)
    plt.plot(xs, mean, label=label)

# ────────── Transfer Comparison ──────────
def plot_transfer_comparison():
    transfer_curves = {"scratch": [], "freeze": [], "finetune": []}
    for seed in range(N_SEEDS):
        for pair in TRANSFER_PAIRS:
            res = transfer_experiment(pair[0], pair[1], seed)
            for mode in transfer_curves:
                transfer_curves[mode].append(res[mode][0])

    x = np.arange(EPISODES)
    plt.figure(figsize=(8,5))
    for mode, curves in transfer_curves.items():
        arr = np.stack(curves)
        plot_with_band(x, arr.mean(0), arr.std(0), mode.capitalize())
    plt.title("Transfer Strategies Comparison")
    plt.xlabel("Episode")
    plt.ylabel("MSE ↓")
    plt.grid(True)
    plt.legend()
    Path("outputs_v7/transfer").mkdir(exist_ok=True)
    plt.savefig("outputs_v7/transfer/comparison.png")
    plt.close()

# ────────── Meta-learning Adaptation ──────────
def plot_meta_adaptation():
    meta_curves = []
    for seed in range(N_SEEDS):
        meta_curves.append(maml_experiment(seed))
    arr = np.stack(meta_curves)

    xs = np.arange(EVAL_ADAPT_STEPS)
    plt.figure(figsize=(8,5))
    plot_with_band(xs, arr.mean(0), arr.std(0), "MAML")
    plt.title("Meta-learning Adaptation")
    plt.xlabel("Adapt Step")
    plt.ylabel("MSE ↓")
    plt.grid(True)
    plt.legend()
    Path("outputs_v7/maml").mkdir(exist_ok=True)
    plt.savefig("outputs_v7/maml/meta_comparison.png")
    plt.close()

# ────────── EWC Forgetting Matrix ──────────
def plot_ewc_forgetting():
    fmats = []
    for seed in range(N_SEEDS):
        fmats.append(ewc_experiment(seed))
    mean_fmat = np.stack(fmats).mean(0)

    plt.figure(figsize=(6,5))
    plt.imshow(mean_fmat, cmap="viridis", origin="lower")
    plt.colorbar(label="MSE")
    ticks = np.arange(len(TASK_PARAMS))
    plt.xticks(ticks, [f"T{i+1}" for i in ticks])
    plt.yticks(ticks, [f"S{i+1}" for i in ticks])
    plt.title("EWC Forgetting Matrix (mean over seeds)")
    Path("outputs_v7/ewc").mkdir(exist_ok=True)
    plt.savefig("outputs_v7/ewc/forgetting.png")
    plt.close()