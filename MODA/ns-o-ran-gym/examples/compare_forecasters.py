#!/usr/bin/env python3
"""Compare two RSLAQ forecasters (e.g. GRU vs LSTM) offline on validation data.

Outputs metrics and latency comparisons to CSV/JSON for article tables.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from environments.rslaq_predictive import (
    FEATURE_DIM,
    FORECAST_DIM,
    RISK_DIM,
    load_forecaster,
    build_forecast_sequences,
    forecast_from_history,
)


def compute_pr_auc(y_true, y_score):
    """Compute Average Precision (PR-AUC) manually for binary labels."""
    # Sort by score descending
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    recall = tp / (tp[-1] + 1e-12)
    precision = tp / (tp + fp + 1e-12)

    # Average precision = sum(precision * delta_recall)
    recall_diff = np.diff(recall, prepend=0.0)
    ap = np.sum(precision * recall_diff)
    return float(ap)


def evaluate_forecaster(model, x_val, y_val, device, batch_size=256):
    """Evaluate a forecaster on validation data and return metrics."""
    model.eval()
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    all_pred = []
    with torch.no_grad():
        for i in range(0, x_val.shape[0], batch_size):
            xb = torch.from_numpy(x_val[i : i + batch_size]).to(device=device, dtype=torch.float32)
            pred = model(xb)
            all_pred.append(pred.cpu().numpy())
    pred = np.concatenate(all_pred, axis=0)

    risk_loss = float(bce_loss(
        torch.from_numpy(pred[:, :RISK_DIM]),
        torch.from_numpy(y_val[:, :RISK_DIM])
    ))
    kpi_loss = float(mse_loss(
        torch.from_numpy(pred[:, RISK_DIM:]),
        torch.from_numpy(y_val[:, RISK_DIM:])
    ))
    total_loss = risk_loss + kpi_loss

    # PR-AUC per risk dimension (outage 0-2, soft 3-5)
    pr_aucs = {}
    for idx, name in enumerate([
        "outage_embb", "outage_urllc", "outage_mtc",
        "soft_embb", "soft_urllc", "soft_mtc",
    ]):
        pr_aucs[name] = compute_pr_auc(y_val[:, idx], pred[:, idx])

    # MAE per dimension
    mae_per_dim = np.mean(np.abs(pred - y_val), axis=0).tolist()

    # Latency benchmark (single sample, repeated)
    sample_x = torch.from_numpy(x_val[:1]).to(device=device, dtype=torch.float32)
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(sample_x)
    # Timing
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(sample_x)
        if device == torch.device("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)  # ms

    latency = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
    }

    return {
        "total_loss": total_loss,
        "risk_loss": risk_loss,
        "kpi_loss": kpi_loss,
        "pr_aucs": pr_aucs,
        "mae_per_dim": mae_per_dim,
        "latency_ms": latency,
        "val_samples": int(x_val.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare RSLAQ forecasters offline")
    parser.add_argument("--gru_checkpoint", required=True)
    parser.add_argument("--lstm_checkpoint", required=True)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--source_format", default="baseline", choices=["auto", "step_metrics", "baseline"])
    parser.add_argument("--sequence_len", type=int, default=8)
    parser.add_argument("--forecast_horizon", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--limit_files", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output, exist_ok=True)

    # Build sequences once
    x, y, meta = build_forecast_sequences(
        args.source_root,
        sequence_len=args.sequence_len,
        horizon=args.forecast_horizon,
        limit_files=args.limit_files or None,
        source_format=args.source_format,
    )
    if x.shape[0] == 0:
        raise RuntimeError(f"No forecast sequences found under {args.source_root}")

    # Simple hold-out by sim_id (reuse same split logic as training)
    from collections import defaultdict
    import random
    by_sim = defaultdict(list)
    for idx, item in enumerate(meta):
        by_sim[item["sim_id"]].append(idx)
    sim_ids = sorted(by_sim)
    rng = random.Random(args.seed)
    rng.shuffle(sim_ids)
    val_count = max(1, int(round(len(sim_ids) * 0.2)))
    val_sims = set(sim_ids[:val_count])
    val_idx = []
    for sim_id, indices in by_sim.items():
        if sim_id in val_sims:
            val_idx.extend(indices)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    x_val = x[val_idx]
    y_val = y[val_idx]

    # Evaluate GRU
    print("[compare] Evaluating GRU...")
    gru_model, _, gru_cell = load_forecaster(args.gru_checkpoint, device=device)
    gru_metrics = evaluate_forecaster(gru_model, x_val, y_val, device, args.batch_size)
    gru_metrics["cell_type"] = gru_cell

    # Evaluate LSTM
    print("[compare] Evaluating LSTM...")
    lstm_model, _, lstm_cell = load_forecaster(args.lstm_checkpoint, device=device)
    lstm_metrics = evaluate_forecaster(lstm_model, x_val, y_val, device, args.batch_size)
    lstm_metrics["cell_type"] = lstm_cell

    # Save JSON
    result = {
        "gru": gru_metrics,
        "lstm": lstm_metrics,
        "config": {
            "source_root": os.path.abspath(args.source_root),
            "source_format": args.source_format,
            "sequence_len": args.sequence_len,
            "forecast_horizon": args.forecast_horizon,
            "val_samples": int(x_val.shape[0]),
        },
    }
    json_path = os.path.join(args.output, "forecaster_comparison.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save CSV summary
    csv_path = os.path.join(args.output, "forecaster_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "gru", "lstm"])
        writer.writerow(["total_loss", f"{gru_metrics['total_loss']:.6f}", f"{lstm_metrics['total_loss']:.6f}"])
        writer.writerow(["risk_loss", f"{gru_metrics['risk_loss']:.6f}", f"{lstm_metrics['risk_loss']:.6f}"])
        writer.writerow(["kpi_loss", f"{gru_metrics['kpi_loss']:.6f}", f"{lstm_metrics['kpi_loss']:.6f}"])
        for name in gru_metrics["pr_aucs"]:
            writer.writerow([f"prauc_{name}", f"{gru_metrics['pr_aucs'][name]:.4f}", f"{lstm_metrics['pr_aucs'][name]:.4f}"])
        writer.writerow(["latency_mean_ms", f"{gru_metrics['latency_ms']['mean_ms']:.4f}", f"{lstm_metrics['latency_ms']['mean_ms']:.4f}"])
        writer.writerow(["latency_p95_ms", f"{gru_metrics['latency_ms']['p95_ms']:.4f}", f"{lstm_metrics['latency_ms']['p95_ms']:.4f}"])
        writer.writerow(["val_samples", gru_metrics["val_samples"], lstm_metrics["val_samples"]])

    print(f"[compare] Results saved to {args.output}")
    print(f"[compare] GRU total_loss={gru_metrics['total_loss']:.5f}  LSTM total_loss={lstm_metrics['total_loss']:.5f}")
    print(f"[compare] GRU latency p95={gru_metrics['latency_ms']['p95_ms']:.3f}ms  LSTM latency p95={lstm_metrics['latency_ms']['p95_ms']:.3f}ms")


if __name__ == "__main__":
    main()
