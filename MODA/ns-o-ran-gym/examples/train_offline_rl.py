#!/usr/bin/env python3
"""Offline training of the 3 slice-weight policies (xApp line).

Trains 3 neural policies from the consolidated Parquet dataset:
  1. ddqn: Q-regression over discretized action space (66 simplex bins)
  2. sac:  behavioral cloning of top-20% candidates (actor MLP with softmax)
  3. ppo:  reward-weighted regression (RWR)

The data does NOT form an MDP (no next_state), so these are contextual-bandit /
supervised policies, not TD-learning — see nsoran/offline_models.py.

Usage:
    cd ns-o-ran-gym
    python3 examples/train_offline_rl.py
    python3 examples/train_offline_rl.py --epochs 20 --models sac
    python3 examples/train_offline_rl.py --seeds 42 43 44 45 46   # multi-seed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsoran.offline_models import (
    ACTION_COLS,
    DEVICE,
    STATE_COLS,
    ActorNet,
    PolicyNet,
    QNet,
    build_simplex_actions,
    compute_norm_params,
    nearest_action_index,
    save_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (REPO_ROOT / "resultados_cenarios_finalizados_20260715"
                   / "offline_dataset_v2.parquet")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_and_split(parquet_path: Path, train_split: float, seed: int):
    """Load Parquet, normalize, split train/test by (scenario, seed) groups."""
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from {parquet_path.name}")

    norm_params = compute_norm_params(df)
    state_min = pd.Series(norm_params["min"])[STATE_COLS]
    state_range = pd.Series(norm_params["range"])[STATE_COLS]
    state_norm = (df[STATE_COLS] - state_min) / state_range

    actions = df[ACTION_COLS].values.astype(np.float32)
    scores = df["score"].values.astype(np.float32)

    # Split by (scenario, seed) — ensures no leakage
    rng = np.random.RandomState(seed)
    groups = df.groupby(["scenario", "seed"]).size().reset_index()[["scenario", "seed"]].values
    rng.shuffle(groups)
    n_train = int(len(groups) * train_split)
    train_groups = set(tuple(g) for g in groups[:n_train])

    group_keys = list(zip(df["scenario"], df["seed"]))
    train_mask = np.array([g in train_groups for g in group_keys])

    train = {"X": state_norm.values[train_mask].astype(np.float32),
             "A": actions[train_mask], "R": scores[train_mask],
             "df": df[train_mask].reset_index(drop=True)}
    test = {"X": state_norm.values[~train_mask].astype(np.float32),
            "A": actions[~train_mask], "R": scores[~train_mask],
            "df": df[~train_mask].reset_index(drop=True)}

    print(f"Train: {len(train['X'])} rows | Test: {len(test['X'])} rows")
    print(f"Train score: {train['R'].mean():.2f} +/- {train['R'].std():.2f}")
    print(f"Test  score: {test['R'].mean():.2f} +/- {test['R'].std():.2f}")
    return train, test, df, norm_params


# ---------------------------------------------------------------------------
# Model 1: ddqn (Q-regression)
# ---------------------------------------------------------------------------
def train_ddqn_offline(train, test, action_table, args, ckpt_kwargs) -> dict:
    print("\n=== ddqn (contextual-bandit Q-regression) ===")
    n_actions = len(action_table)
    state_dim = train["X"].shape[1]

    train_aidx = np.array([nearest_action_index(a, action_table) for a in train["A"]])

    model = QNet(state_dim, n_actions).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    X_t = torch.tensor(train["X"], device=DEVICE)
    aidx_t = torch.tensor(train_aidx, device=DEVICE, dtype=torch.long)
    R_t = torch.tensor(train["R"], device=DEVICE)

    loss_history = []
    for epoch in range(args.epochs):
        model.train()
        q_pred = model(X_t).gather(1, aidx_t.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q_pred, R_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}: loss={loss.item():.2f}")

    model_path = args.models_dir / "ddqn_offline.pt"
    save_checkpoint(model_path, model, model_type="ddqn", state_dim=state_dim,
                    loss_history=loss_history, action_table=action_table,
                    **ckpt_kwargs)
    print(f"  Saved: {model_path}")

    model.eval()
    with torch.no_grad():
        best_aidx = model(torch.tensor(test["X"], device=DEVICE)).argmax(dim=1).cpu().numpy()
    return {"model_path": str(model_path), "pred_weights_test": action_table[best_aidx],
            "test_df": test["df"]}


# ---------------------------------------------------------------------------
# Model 2: sac (behavioral cloning of top-K)
# ---------------------------------------------------------------------------
def train_sac_offline(train, test, args, ckpt_kwargs) -> dict:
    print("\n=== sac (behavioral cloning top-20%) ===")
    state_dim = train["X"].shape[1]

    df_train = train["df"]
    top_k_list = []
    for sc in df_train["scenario"].unique():
        sub = df_train[df_train["scenario"] == sc]
        threshold = sub["score"].quantile(0.80)
        top_k_list.append(sub[sub["score"] >= threshold])
    top_df = pd.concat(top_k_list)
    print(f"  Top-K candidates: {len(top_df)} (from {len(df_train)} train rows)")

    X_t = torch.tensor(train["X"][top_df.index.values], device=DEVICE)
    A_t = torch.tensor(train["A"][top_df.index.values], device=DEVICE)

    model = ActorNet(state_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_history = []
    for epoch in range(args.epochs):
        model.train()
        loss = F.mse_loss(model(X_t), A_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}: loss={loss.item():.6f}")

    model_path = args.models_dir / "sac_offline.pt"
    save_checkpoint(model_path, model, model_type="sac", state_dim=state_dim,
                    loss_history=loss_history, **ckpt_kwargs)
    print(f"  Saved: {model_path}")

    model.eval()
    with torch.no_grad():
        pred_weights = model(torch.tensor(test["X"], device=DEVICE)).cpu().numpy()
    return {"model_path": str(model_path), "pred_weights_test": pred_weights,
            "test_df": test["df"]}


# ---------------------------------------------------------------------------
# Model 3: ppo (reward-weighted regression)
# ---------------------------------------------------------------------------
def train_ppo_offline(train, test, args, ckpt_kwargs) -> dict:
    print("\n=== ppo (reward-weighted regression) ===")
    state_dim = train["X"].shape[1]

    X_t = torch.tensor(train["X"], device=DEVICE)
    A_t = torch.tensor(train["A"], device=DEVICE)
    R_t = torch.tensor(train["R"], device=DEVICE)

    R_min, R_max = R_t.min(), R_t.max()
    R_norm = (R_t - R_min) / (R_max - R_min + 1e-8)
    tau = 0.1
    weights = torch.exp(R_norm / tau)
    weights = weights / weights.sum()

    model = PolicyNet(state_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_history = []
    for epoch in range(args.epochs):
        model.train()
        mean, _ = model(X_t)
        pred_weights = F.softmax(mean, dim=-1)
        mse = ((pred_weights - A_t) ** 2).sum(dim=1)
        loss = (weights * mse).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}: weighted_loss={loss.item():.6f}")

    model_path = args.models_dir / "ppo_offline.pt"
    save_checkpoint(model_path, model, model_type="ppo", state_dim=state_dim,
                    loss_history=loss_history, **ckpt_kwargs)
    print(f"  Saved: {model_path}")

    model.eval()
    with torch.no_grad():
        pred_weights = model.predict(torch.tensor(test["X"], device=DEVICE)).cpu().numpy()
    return {"model_path": str(model_path), "pred_weights_test": pred_weights,
            "test_df": test["df"]}


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_predictions(results: dict, df_full: pd.DataFrame) -> dict:
    """Per test scenario: mean predicted weights vs best score in the data."""
    summary = {}
    test_df = results["test_df"]
    pred_w = results["pred_weights_test"]
    for sc in test_df["scenario"].unique():
        mask = test_df["scenario"].values == sc
        if mask.sum() == 0:
            continue
        summary[sc] = {
            "avg_pred_weights": pred_w[mask].mean(axis=0).tolist(),
            "best_meta_score": float(df_full[df_full["scenario"] == sc]["score"].max()),
        }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--models", nargs="+", choices=["ddqn", "sac", "ppo"],
                        default=["ddqn", "sac", "ppo"])
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=42)
    seed_group.add_argument("--seeds", nargs="+", type=int, default=None,
                            help="Multi-seed training: one run per seed in "
                                 "<models-dir>/seed<k>/, plus metrics_multiseed.json")
    return parser.parse_args()


def run_one_seed(args, seed: int) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    train, test, df_full, norm_params = load_and_split(
        args.dataset, args.train_split, seed)
    ckpt_kwargs = {
        "norm_params": norm_params,
        "train_seed": seed,
        "dataset_path": str(args.dataset),
        "dataset_version": args.dataset.stem.replace("offline_dataset", "").strip("_") or "v1",
    }

    trainers = {
        "ddqn": lambda: train_ddqn_offline(train, test, build_simplex_actions(0.1),
                                           args, ckpt_kwargs),
        "sac": lambda: train_sac_offline(train, test, args, ckpt_kwargs),
        "ppo": lambda: train_ppo_offline(train, test, args, ckpt_kwargs),
    }

    all_metrics = {}
    for name in args.models:
        results = trainers[name]()
        all_metrics[name] = evaluate_predictions(results, df_full)

    print("\n" + "=" * 70)
    print(f"EVALUATION (seed {seed}): predicted weights per scenario (test set)")
    print("=" * 70)
    for name, summ in all_metrics.items():
        print(f"\n--- {name} ---")
        for sc in sorted(summ.keys()):
            w = summ[sc]["avg_pred_weights"]
            print(f"  {sc:<18}: pred weights=[{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}] "
                  f"| best score in data={summ[sc]['best_meta_score']:.2f}")

    metrics_path = args.models_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nMetrics saved: {metrics_path}")
    return all_metrics


def main() -> None:
    args = parse_args()

    if not args.seeds:
        run_one_seed(args, args.seed)
        return

    # Multi-seed: one full run per seed, then aggregate mean +/- std.
    base_dir = args.models_dir
    per_seed = {}
    for seed in args.seeds:
        print(f"\n{'#' * 70}\n# SEED {seed}\n{'#' * 70}")
        args.models_dir = base_dir / f"seed{seed}"
        per_seed[seed] = run_one_seed(args, seed)
    args.models_dir = base_dir

    aggregate: dict = {}
    for name in args.models:
        aggregate[name] = {}
        scenarios = sorted({sc for m in per_seed.values() for sc in m.get(name, {})})
        for sc in scenarios:
            weights = np.array([m[name][sc]["avg_pred_weights"]
                                for m in per_seed.values() if sc in m.get(name, {})])
            aggregate[name][sc] = {
                "mean_pred_weights": weights.mean(axis=0).tolist(),
                "std_pred_weights": weights.std(axis=0).tolist(),
                "n_seeds": int(len(weights)),
                "per_seed": {str(s): m[name][sc]["avg_pred_weights"]
                             for s, m in per_seed.items() if sc in m.get(name, {})},
            }
    out = base_dir / "metrics_multiseed.json"
    out.write_text(json.dumps({"seeds": args.seeds, "models": aggregate}, indent=2))
    print(f"\nMulti-seed aggregate saved: {out}")


if __name__ == "__main__":
    main()
