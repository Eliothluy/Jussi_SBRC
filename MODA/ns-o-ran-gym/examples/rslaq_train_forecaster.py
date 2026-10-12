#!/usr/bin/env python3
"""Train the RSLAQ temporal KPI forecaster from existing step or baseline logs."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from environments.rslaq_predictive import (
    FEATURE_DIM,
    FORECAST_DIM,
    RISK_DIM,
    TemporalKpiForecaster,
    TemporalKpiForecasterLSTM,
    create_forecaster,
    build_forecast_sequences,
)
from nsoran.compute_accounting import ComputeAccounting


DEFAULT_SOURCE_ROOT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "ns-3-dev",
    "results_rslaq_network_only",
)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results_controlled",
    "predictive_forecaster",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_sim_id(meta, val_fraction: float, seed: int):
    by_sim = defaultdict(list)
    for idx, item in enumerate(meta):
        by_sim[item["sim_id"]].append(idx)
    sim_ids = sorted(by_sim)
    rng = random.Random(seed)
    rng.shuffle(sim_ids)
    val_count = max(1, int(round(len(sim_ids) * val_fraction))) if sim_ids else 0
    val_sims = set(sim_ids[:val_count])
    train_idx, val_idx = [], []
    for sim_id, indices in by_sim.items():
        if sim_id in val_sims:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)
    if not train_idx and val_idx:
        train_idx, val_idx = val_idx, []
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def compute_loss(pred, target, bce_loss, mse_loss, risk_weight: float):
    risk_loss = bce_loss(pred[:, :RISK_DIM], target[:, :RISK_DIM])
    kpi_loss = mse_loss(pred[:, RISK_DIM:], target[:, RISK_DIM:])
    return risk_weight * risk_loss + kpi_loss, risk_loss, kpi_loss


def main():
    parser = argparse.ArgumentParser(description="Train RSLAQ temporal KPI forecaster")
    parser.add_argument("--source_root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--sequence_len", type=int, default=8)
    parser.add_argument("--forecast_horizon", type=int, default=5)
    parser.add_argument("--source_format", default="auto",
                        choices=["auto", "step_metrics", "baseline"],
                        help="Input format: DRL step_metrics, ns-3 baseline, or auto-detect both")
    parser.add_argument("--cell_type", type=str, default="gru", choices=["gru", "lstm"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--risk_loss_weight", type=float, default=2.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit_files", type=int, default=0)
    parser.add_argument("--compute_cost_per_hour_usd", type=float, default=0.0)
    parser.add_argument("--compute_avg_power_watts", type=float, default=0.0)
    parser.add_argument("--compute_electricity_cost_usd_per_kwh", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y, meta = build_forecast_sequences(
        args.source_root,
        sequence_len=args.sequence_len,
        horizon=args.forecast_horizon,
        limit_files=args.limit_files or None,
        source_format=args.source_format,
    )
    if x.shape[0] == 0:
        raise RuntimeError(f"No forecast sequences found under {args.source_root}")

    train_idx, val_idx = split_by_sim_id(meta, args.val_fraction, args.seed)
    compute_accounting = ComputeAccounting(
        interaction_budget=int(train_idx.size) * args.epochs,
        period_ms=0.0,
        cost_per_hour_usd=args.compute_cost_per_hour_usd,
        avg_power_watts=args.compute_avg_power_watts,
        electricity_cost_usd_per_kwh=args.compute_electricity_cost_usd_per_kwh,
    )
    train_ds = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = None
    if val_idx.size:
        val_ds = TensorDataset(torch.from_numpy(x[val_idx]), torch.from_numpy(y[val_idx]))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = create_forecaster(
        cell_type=args.cell_type,
        input_dim=FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        output_dim=FORECAST_DIM,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    best_val = float("inf")
    history = []
    best_path = os.path.join(args.output, "forecaster_best.pt")
    final_path = os.path.join(args.output, "forecaster_final.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            pred = model(xb)
            loss, risk_loss, kpi_loss = compute_loss(
                pred, yb, bce_loss, mse_loss, args.risk_loss_weight
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append((loss.item(), risk_loss.item(), kpi_loss.item()))

        val_summary = None
        monitor = float(np.mean([item[0] for item in train_losses]))
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device=device, dtype=torch.float32)
                    yb = yb.to(device=device, dtype=torch.float32)
                    pred = model(xb)
                    loss, risk_loss, kpi_loss = compute_loss(
                        pred, yb, bce_loss, mse_loss, args.risk_loss_weight
                    )
                    val_losses.append((loss.item(), risk_loss.item(), kpi_loss.item()))
            val_summary = np.mean(val_losses, axis=0).tolist()
            monitor = val_summary[0]

        train_summary = np.mean(train_losses, axis=0).tolist()
        row = {
            "epoch": epoch,
            "train_loss": train_summary[0],
            "train_risk_loss": train_summary[1],
            "train_kpi_loss": train_summary[2],
            "val_loss": val_summary[0] if val_summary else None,
            "val_risk_loss": val_summary[1] if val_summary else None,
            "val_kpi_loss": val_summary[2] if val_summary else None,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train={row['train_loss']:.5f} "
            f"val={row['val_loss']:.5f}" if row["val_loss"] is not None
            else f"Epoch {epoch}/{args.epochs} train={row['train_loss']:.5f}"
        )

        if monitor < best_val:
            best_val = monitor
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cell_type": args.cell_type,
                    "input_dim": FEATURE_DIM,
                    "output_dim": FORECAST_DIM,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "sequence_len": args.sequence_len,
                    "forecast_horizon": args.forecast_horizon,
                },
                best_path,
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "cell_type": args.cell_type,
            "input_dim": FEATURE_DIM,
            "output_dim": FORECAST_DIM,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "sequence_len": args.sequence_len,
            "forecast_horizon": args.forecast_horizon,
        },
        final_path,
    )

    metrics = {
        "source_root": os.path.abspath(args.source_root),
        "source_format": args.source_format,
        "num_sequences": int(x.shape[0]),
        "train_sequences": int(train_idx.size),
        "val_sequences": int(val_idx.size),
        "best_loss": best_val,
        "history": history,
        "best_checkpoint": best_path,
        "final_checkpoint": final_path,
        "compute": compute_accounting.finish(),
    }
    with open(os.path.join(args.output, "forecaster_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Forecaster saved to {best_path}")


if __name__ == "__main__":
    main()
