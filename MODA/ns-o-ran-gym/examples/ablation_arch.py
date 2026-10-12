#!/usr/bin/env python3
"""Architecture/optimizer sensitivity ablation for the MODA policies (CPU-only).

Extends the R8 study (ablation_offline_rl.py, which covers delta/tau/q) with
the axes the reviewer explicitly asked about: model architecture choices and
loss formulation. Sweeps, per model:

  - hidden width h in {64, 128, 256}
  - MLP depth d in {1, 2, 3} hidden layers
  - learning rate lr in {1e-4, 1e-3, 1e-2}

for the three recipes (MODA-Q Q-regression, MODA-BC behavioral cloning,
MODA-RWR reward-weighted regression), evaluated with the SAME proxy as the
existing ablations (nearest dataset candidate in weight space, mean over the
four training scenarios). Also reports the L1 divergence of each config's
scenario predictions vs the production baseline (h=128, d=2, lr=1e-3, seed 42).

No ns-3 runs; production checkpoints in models/ are not touched.

Usage:
    cd ns-o-ran-gym
    python3 examples/ablation_arch.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from nsoran.offline_models import ACTION_COLS, DEVICE, build_simplex_actions, nearest_action_index  # noqa: E402
from ablation_offline_rl import OBS_COLS, ONE_HOT, SCENARIOS, TRAIN_SPLIT, prepare, nearest_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (REPO_ROOT / "resultados_cenarios_finalizados_20260715"
                   / "offline_dataset_v2.parquet")
OUT_DIR = REPO_ROOT / "models" / "ablation"
EPOCHS = 300
SEED = 42
BASELINE = {"hidden": 128, "depth": 2, "lr": 1e-3}


# ---------------------------------------------------------------------------
# Parameterizable architectures (local — production checkpoints untouched)
# ---------------------------------------------------------------------------
def _mlp(state_dim: int, hidden: int, depth: int, out: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = state_dim
    for _ in range(depth):
        layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
        in_dim = hidden
    layers.append(nn.Linear(in_dim, out))
    return nn.Sequential(*layers)


class QNetH(nn.Module):
    def __init__(self, state_dim, n_actions, hidden, depth):
        super().__init__()
        self.net = _mlp(state_dim, hidden, depth, n_actions)

    def forward(self, x):
        return self.net(x)


class ActorNetH(nn.Module):
    def __init__(self, state_dim, hidden, depth):
        super().__init__()
        self.net = _mlp(state_dim, hidden, depth, 3)

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


class PolicyNetH(nn.Module):
    def __init__(self, state_dim, hidden, depth):
        super().__init__()
        self.trunk = _mlp(state_dim, hidden, depth, hidden)
        self.mean_head = nn.Linear(hidden, 3)

    def forward(self, x):
        return self.mean_head(self.trunk(x))

    def predict(self, x):
        return F.softmax(self.forward(x), dim=-1)


# ---------------------------------------------------------------------------
def train(model, loss_fn, X, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(EPOCHS):
        loss = loss_fn(model, X)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def train_variant(recipe: str, tr, df_train_scen, hidden, depth, lr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    action_table = build_simplex_actions(0.1)
    sd = tr["X"].shape[1]
    X = torch.tensor(tr["X"], device=DEVICE)
    A = torch.tensor(tr["A"], device=DEVICE)
    R = torch.tensor(tr["R"], device=DEVICE)

    if recipe == "q":
        aidx = torch.tensor(
            np.array([nearest_action_index(a, action_table) for a in tr["A"]]),
            device=DEVICE, dtype=torch.long)
        m = QNetH(sd, len(action_table), hidden, depth).to(DEVICE)
        train(m, lambda net, _: F.mse_loss(
            net(X).gather(1, aidx.unsqueeze(1)).squeeze(1), R), X, lr)
        return lambda s: action_table[int(m(torch.tensor(s[None], device=DEVICE)).argmax())]

    if recipe == "bc":
        idx = np.concatenate([
            np.where((df_train_scen == sc)
                     & (tr["R"] >= np.quantile(tr["R"][df_train_scen == sc], 0.80)))[0]
            for sc in np.unique(df_train_scen)])
        Xt, At = X[torch.tensor(idx)], A[torch.tensor(idx)]
        m = ActorNetH(sd, hidden, depth).to(DEVICE)
        train(m, lambda net, _: F.mse_loss(net(Xt), At), X, lr)
        return lambda s: m(torch.tensor(s[None], device=DEVICE)).detach().cpu().numpy()[0]

    # rwr
    Rn = (R - R.min()) / (R.max() - R.min() + 1e-8)
    w = torch.exp(Rn / 0.1); w = w / w.sum()
    m = PolicyNetH(sd, hidden, depth).to(DEVICE)
    train(m, lambda net, _: (w * ((F.softmax(net(X), dim=-1) - A) ** 2).sum(dim=1)).mean(),
          X, lr)
    return lambda s: m.predict(torch.tensor(s[None], device=DEVICE)).detach().cpu().numpy()[0]


def predict_scenario_weights(predict, scen_state):
    out = {}
    for sc, s in scen_state.items():
        w = np.asarray(predict(s), dtype=float)
        w = np.clip(w, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.full(3, 1 / 3)
        out[sc] = w
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    cols = OBS_COLS + ONE_HOT  # FULL state, as in production
    tr, scen_state = prepare(df, cols, SEED)
    # rebuild per-row scenario labels for the training split (mirrors prepare's mask)
    rng = np.random.RandomState(SEED)
    groups = df.groupby(["scenario", "seed"]).size().reset_index()[["scenario", "seed"]].values
    rng.shuffle(groups)
    train_groups = set(tuple(g) for g in groups[:int(len(groups) * TRAIN_SPLIT)])
    mask = np.array([g in train_groups for g in zip(df["scenario"], df["seed"])])
    df_train_scen = df["scenario"].values[mask]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    baseline_preds = {}

    grid = [(h, d, lr) for h in (64, 128, 256) for d in (1, 2, 3)
            for lr in (1e-4, 1e-3, 1e-2)]
    for recipe in ("q", "bc", "rwr"):
        for h, d, lr in grid:
            predict = train_variant(recipe, tr, df_train_scen, h, d, lr, SEED)
            preds = predict_scenario_weights(predict, scen_state)
            proxy = float(np.mean([nearest_score(df, sc, w) for sc, w in preds.items()]))
            if (h, d, lr) == (BASELINE["hidden"], BASELINE["depth"], BASELINE["lr"]):
                baseline_preds[recipe] = preds
            results.append({"recipe": recipe, "hidden": h, "depth": d, "lr": lr,
                            "proxy_mean_score": round(proxy, 3),
                            "preds": {sc: w.tolist() for sc, w in preds.items()}})
            print(f"{recipe:<4} h={h:<4} d={d} lr={lr:<7} proxy={proxy:7.2f}")

    # divergence vs baseline
    for r in results:
        base = baseline_preds[r["recipe"]]
        l1 = sum(np.abs(r["preds"][sc] - base[sc]).sum() for sc in base) / len(base)
        r["l1_divergence_vs_baseline"] = round(float(l1), 4)

    out_csv = OUT_DIR / "ablation_arch_results.csv"
    fieldnames = ["recipe", "hidden", "depth", "lr", "proxy_mean_score",
                  "l1_divergence_vs_baseline"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\nSaved {len(results)} rows -> {out_csv}")

    # quick summary: per recipe, spread of proxy across the grid
    print("\nGrid spread (proxy_mean_score):")
    for recipe in ("q", "bc", "rwr"):
        vals = [r["proxy_mean_score"] for r in results if r["recipe"] == recipe]
        l1s = [r["l1_divergence_vs_baseline"] for r in results if r["recipe"] == recipe]
        print(f"  {recipe:<4} min={min(vals):7.2f} max={max(vals):7.2f} "
              f"spread={max(vals) - min(vals):6.2f} | L1 div max={max(l1s):.3f}")


if __name__ == "__main__":
    main()
