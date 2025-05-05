import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from config import (
    EPISODES, INNER_STEPS, MAML_INNER_LR, MAML_META_LR,
    MAML_NUM_TASKS, EVAL_ADAPT_STEPS, LEARNING_RATE,
    LOG_INTERVAL, DEVICE, TASK_PARAMS, TRANSFER_PAIRS, EWC_LAMBDA
)
from core import MLP, generate_sine, write_csv, grad_norm

mse = nn.MSELoss()

# ────────── Transfer Experiment ──────────
def transfer_experiment(src, tgt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    (amp_s, ph_s), (amp_t, ph_t) = src, tgt
    tag = f"A{amp_s}_P{ph_s}_to_A{amp_t}_P{ph_t}_seed{seed}"
    out_dir = Path("outputs_v7/transfer")
    out_dir.mkdir(exist_ok=True)

    # Train on source
    src_model = MLP().to(DEVICE)
    opt_s = optim.Adam(src_model.parameters(), lr=LEARNING_RATE)
    for ep in range(EPISODES):
        x, y = generate_sine(amp_s, ph_s)
        loss = mse(src_model(x), y)
        opt_s.zero_grad()
        loss.backward()
        opt_s.step()
    torch.save(src_model.state_dict(), out_dir / f"src_{tag}.pth")

    results = {}
    for mode in ("scratch", "freeze", "finetune"):
        # Initialize or load
        model = MLP().to(DEVICE)
        if mode != "scratch":
            model.load_state_dict(src_model.state_dict())
            if mode == "freeze":
                for name, p in model.named_parameters():
                    p.requires_grad = (".2" in name)
        opt = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE
        )

        curve, grad_curve = [], []
        for ep in range(EPISODES):
            x, y = generate_sine(amp_t, ph_t)
            pred = model(x)
            loss = mse(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            curve.append(loss.item())
            if ep % LOG_INTERVAL == 0 or ep == EPISODES - 1:
                grad_curve.append((ep, grad_norm(model)))

        results[mode] = (np.array(curve), grad_curve)
        write_csv(out_dir / f"{mode}_{tag}.csv", enumerate(curve), header=["episode","mse"])
        write_csv(out_dir / f"{mode}_grad_{tag}.csv", grad_curve, header=["episode","grad_norm"])

    return results

# ────────── MAML Experiment ──────────
def maml_experiment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tag = f"seed{seed}"
    out_dir = Path("outputs_v7/maml")
    out_dir.mkdir(exist_ok=True)

    meta_model = MLP().to(DEVICE)
    opt_meta = optim.Adam(meta_model.parameters(), lr=MAML_META_LR)

    for it in range(EPISODES):
        opt_meta.zero_grad()
        for _ in range(MAML_NUM_TASKS):
            amp = random.uniform(0.1, 5.0)
            ph  = random.uniform(0, math.pi)
            # Inner loop copy
            tmp = MLP().to(DEVICE)
            tmp.load_state_dict(meta_model.state_dict())
            inner_opt = optim.SGD(tmp.parameters(), lr=MAML_INNER_LR)

            for _ in range(INNER_STEPS):
                x_i, y_i = generate_sine(amp, ph)
                loss_i = mse(tmp(x_i), y_i)
                inner_opt.zero_grad()
                loss_i.backward()
                inner_opt.step()

            # Outer loop
            x_q, y_q = generate_sine(amp, ph)
            mse(tmp(x_q), y_q).backward()

        opt_meta.step()
        if it % LOG_INTERVAL == 0:
            print(f"[MAML] iter={it}")

    # Adaptation evaluation
    adapt = MLP().to(DEVICE)
    adapt.load_state_dict(meta_model.state_dict())
    opt_adapt = optim.SGD(adapt.parameters(), lr=MAML_INNER_LR)

    adapt_curve = []
    for step in range(EVAL_ADAPT_STEPS):
        x, y = generate_sine(1.5, 0.5)
        loss = mse(adapt(x), y)
        adapt_curve.append(loss.item())
        opt_adapt.zero_grad()
        loss.backward()
        opt_adapt.step()

    write_csv(out_dir / f"adapt_curve_{tag}.csv", enumerate(adapt_curve), header=["step","mse"])
    return np.array(adapt_curve)

# ────────── EWC Experiment ──────────

def compute_fisher(model, data_loader):
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    for x_batch, y_batch in data_loader:
        model.zero_grad()
        loss = mse(model(x_batch), y_batch)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
    for n in fisher:
        fisher[n] /= len(data_loader)
    return fisher

class EWC:
    def __init__(self, ref_model, data_loader):
        self.params = {n: p.clone().detach() for n, p in ref_model.named_parameters()}
        self.fisher = compute_fisher(ref_model, data_loader)

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            loss += (self.fisher[n] * (p - self.params[n])**2).sum()
        return (EWC_LAMBDA / 2) * loss


def ewc_experiment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tag = f"seed{seed}"
    out_dir = Path("outputs_v7/ewc")
    out_dir.mkdir(exist_ok=True)

    model = MLP().to(DEVICE)
    prev_loader, ewc_obj = None, None
    forgetting = np.zeros((len(TASK_PARAMS), len(TASK_PARAMS)))

    for i, (amp, ph) in enumerate(TASK_PARAMS):
        opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        for ep in range(EPISODES):
            x, y = generate_sine(amp, ph)
            loss = mse(model(x), y)
            if ewc_obj:
                loss += ewc_obj.penalty(model)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Build loader of current task
        xs, ys = generate_sine(amp, ph, n=100)
        loader = [(xs[k:k+10], ys[k:k+10]) for k in range(0, 100, 10)]
        ewc_obj = EWC(model, loader)

        # Evaluate on seen tasks
        for j, (amp_j, ph_j) in enumerate(TASK_PARAMS[:i+1]):
            x_t, y_t = generate_sine(amp_j, ph_j, n=100)
            forgetting[i, j] = mse(model(x_t), y_t).item()

    write_csv(out_dir / f"forgetting_{tag}.csv", forgetting.tolist())
    return forgetting