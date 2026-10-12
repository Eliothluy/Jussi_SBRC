#!/usr/bin/env python3
"""Generate article-quality comparison plots for GRU vs LSTM forecasters.

Reads forecaster_comparison.json produced by compare_forecasters.py
and outputs .pdf + .png figures to the specified directory.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib is required for plotting. Install it with `pip install matplotlib`.")


def set_article_style():
    """Configure matplotlib for IEEE-style article figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig_both(fig, output_dir, name):
    """Save figure as both PDF and PNG."""
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)


def plot_latency_bar(data, output_dir):
    """Fig 1: Inference latency comparison (bar plot)."""
    gru = data["gru"]["latency_ms"]
    lstm = data["lstm"]["latency_ms"]

    labels = ["Mean", "P50", "P95", "P99"]
    gru_vals = [gru["mean_ms"], gru["p50_ms"], gru["p95_ms"], gru["p99_ms"]]
    lstm_vals = [lstm["mean_ms"], lstm["p50_ms"], lstm["p95_ms"], lstm["p99_ms"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars1 = ax.bar(x - width / 2, gru_vals, width, label="GRU", color="#4472C4", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, lstm_vals, width, label="LSTM", color="#ED7D31", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    ax.set_ylim(0, max(max(gru_vals), max(lstm_vals)) * 1.2)

    # Annotate bars
    for bar in bars1:
        ax.annotate(f"{bar.get_height():.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    savefig_both(fig, output_dir, "fig1_latency_comparison")


def plot_loss_comparison(data, output_dir):
    """Fig 2: Loss comparison (grouped bar)."""
    categories = ["Total Loss", "Risk Loss (BCE)", "KPI Loss (MSE)"]
    gru_vals = [data["gru"]["total_loss"], data["gru"]["risk_loss"], data["gru"]["kpi_loss"]]
    lstm_vals = [data["lstm"]["total_loss"], data["lstm"]["risk_loss"], data["lstm"]["kpi_loss"]]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars1 = ax.bar(x - width / 2, gru_vals, width, label="GRU", color="#4472C4", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, lstm_vals, width, label="LSTM", color="#ED7D31", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Loss")
    ax.set_title("Validation Loss Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(frameon=False)
    ax.set_ylim(0, max(max(gru_vals), max(lstm_vals)) * 1.2)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.4f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.4f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    savefig_both(fig, output_dir, "fig2_loss_comparison")


def plot_pr_auc_radar(data, output_dir):
    """Fig 3: PR-AUC radar chart per risk type and slice."""
    pr_names = list(data["gru"]["pr_aucs"].keys())
    gru_vals = [data["gru"]["pr_aucs"][k] for k in pr_names]
    lstm_vals = [data["lstm"]["pr_aucs"][k] for k in pr_names]

    # Convert to polar
    angles = np.linspace(0, 2 * np.pi, len(pr_names), endpoint=False).tolist()
    gru_vals += gru_vals[:1]
    lstm_vals += lstm_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, gru_vals, "o-", linewidth=1.5, label="GRU", color="#4472C4")
    ax.fill(angles, gru_vals, alpha=0.15, color="#4472C4")
    ax.plot(angles, lstm_vals, "s-", linewidth=1.5, label="LSTM", color="#ED7D31")
    ax.fill(angles, lstm_vals, alpha=0.15, color="#ED7D31")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([n.replace("_", "\n") for n in pr_names], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title("PR-AUC per Risk Type and Slice", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False)

    savefig_both(fig, output_dir, "fig3_pr_auc_radar")


def plot_training_curves(gru_dir, lstm_dir, output_dir):
    """Fig 4: Training/validation loss curves from forecaster_metrics.json."""
    def load_history(path):
        if not path.exists():
            return []
        with open(path) as f:
            data = json.load(f)
        return data.get("history", [])

    gru_hist = load_history(Path(gru_dir) / "forecaster_metrics.json")
    lstm_hist = load_history(Path(lstm_dir) / "forecaster_metrics.json")

    if not gru_hist or not lstm_hist:
        print("[plot] Warning: training histories not found, skipping fig4")
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))
    epochs = [r["epoch"] for r in gru_hist]
    ax.plot(epochs, [r["train_loss"] for r in gru_hist], "-", label="GRU Train", color="#4472C4")
    ax.plot(epochs, [r["val_loss"] for r in gru_hist if r["val_loss"] is not None], "--", label="GRU Val", color="#4472C4", alpha=0.7)
    ax.plot(epochs, [r["train_loss"] for r in lstm_hist], "-", label="LSTM Train", color="#ED7D31")
    ax.plot(epochs, [r["val_loss"] for r in lstm_hist if r["val_loss"] is not None], "--", label="LSTM Val", color="#ED7D31", alpha=0.7)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Curves")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(1, len(gru_hist))

    savefig_both(fig, output_dir, "fig4_training_curves")


def main():
    parser = argparse.ArgumentParser(description="Plot forecaster comparison figures")
    parser.add_argument("--comparison_json", required=True)
    parser.add_argument("--gru_dir", default="")
    parser.add_argument("--lstm_dir", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    set_article_style()

    with open(args.comparison_json) as f:
        data = json.load(f)

    plot_latency_bar(data, args.output)
    plot_loss_comparison(data, args.output)
    plot_pr_auc_radar(data, args.output)

    if args.gru_dir and args.lstm_dir:
        plot_training_curves(args.gru_dir, args.lstm_dir, args.output)

    print(f"[plot] Figures saved to {args.output}")


if __name__ == "__main__":
    main()
