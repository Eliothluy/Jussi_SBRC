#!/usr/bin/env python3
"""Plot training loss/reward curves for the 3 offline DRL models.

Generates two figures:
  - fig16: Loss curves (convergence comparison)
  - fig17: Reward proxy (predicted score over epochs)

Usage:
    cd /home/elioth/Documentos/artigo_jussi/ns-o-ran-gym
    .venv-dashboard/bin/python examples/plot_drl_rewards.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path("/home/elioth/Documentos/artigo_jussi")
MODELS_DIR = REPO_ROOT / "models"
FIG_DIR = REPO_ROOT / "figuras_overleaf"

MODELS = [
    ("DDQN", "ddqn_offline.pt", "#1f77b4"),
    ("SAC", "sac_offline.pt", "#d62728"),
    ("PPO", "ppo_offline.pt", "#9467bd"),
]

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.fontsize": 10,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})


def load_loss_histories() -> dict[str, list[float]]:
    histories = {}
    for name, fname, _ in MODELS:
        path = MODELS_DIR / fname
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hist = ckpt.get("loss_history", [])
        histories[name] = hist
        print(f"  {name}: {len(hist)} epochs, final loss={hist[-1]:.6f}" if hist else f"  {name}: no history")
    return histories


def plot_loss_curves(histories: dict[str, list[float]]) -> None:
    """Fig 16: Loss convergence curves (log scale for cross-model comparison)."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    for name, _, color in MODELS:
        hist = histories.get(name, [])
        if not hist:
            continue
        epochs = np.arange(1, len(hist) + 1)
        ax.plot(epochs, hist, linewidth=2, color=color, label=name, alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (MSE)")
    ax.set_title("Offline DRL training: loss convergence")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    fig.savefig(FIG_DIR / "fig16_drl_loss_curves.png")
    plt.close(fig)
    print("  + fig16_drl_loss_curves.png")


def plot_reward_curves(histories: dict[str, list[float]]) -> None:
    """Fig 17: Normalized reward proxy (1 - normalized loss) over epochs.

    Since these are offline supervised models (not Bellman RL), the "reward"
    is the inverse of the loss: lower loss = better prediction of optimal weights
    = higher implicit reward. We plot 1 - normalized(loss) so higher = better,
    matching the intuition of a reward curve.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    for name, _, color in MODELS:
        hist = histories.get(name, [])
        if not hist:
            continue
        hist_arr = np.array(hist)
        # Normalize loss to [0,1] then invert: reward = 1 - norm(loss)
        loss_min, loss_max = hist_arr.min(), hist_arr.max()
        if loss_max > loss_min:
            norm_loss = (hist_arr - loss_min) / (loss_max - loss_min)
        else:
            norm_loss = np.zeros_like(hist_arr)
        reward_proxy = 1.0 - norm_loss
        epochs = np.arange(1, len(hist) + 1)
        ax.plot(epochs, reward_proxy, linewidth=2, color=color, label=name, alpha=0.85)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized reward proxy (1 - normalized loss)")
    ax.set_title("Offline DRL training: reward proxy convergence")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    fig.savefig(FIG_DIR / "fig17_drl_reward_curves.png")
    plt.close(fig)
    print("  + fig17_drl_reward_curves.png")


def main() -> None:
    print("Loading loss histories...")
    histories = load_loss_histories()
    print("\nGenerating figures...")
    plot_loss_curves(histories)
    plot_reward_curves(histories)
    print("Done.")


if __name__ == "__main__":
    main()
