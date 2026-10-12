#!/usr/bin/env python3
"""Ablation study for the MODA offline policies (peer-review R5a + R8).

Answers two reviewer requirements without touching the frozen production
checkpoints in models/:

  R5a  One-hot scenario indicators: do the policies still work if the four
       one-hot scenario flags are removed from the state, leaving only the
       observable features (per-slice UE counts, offered load, packet size)?
       In a real network there is no oracle telling the policy which of the
       four scenarios it is in; a deployable policy must map observable state
       to weights. We compare FULL (with one-hot) vs OBS (observable only).

  R8   Hyperparameter sensitivity of the three recipes:
         - MODA-Q  : discretization step delta -> action-table size |A|
         - MODA-RWR: temperature tau of the reward weighting
         - MODA-BC : elite quantile q for behavioral cloning

Evaluation is by PROXY (nearest dataset candidate in weight space, per
scenario) — fast, CPU-only, no ns-3. Proxy scores are on the dataset's
composite-score scale and are used only for RELATIVE comparison across
ablation settings; absolute closed-loop validation of the best/worst points
is deferred to ns-3 (post-pilot). All scores here share the same proxy, so
the comparison is fair.

Usage:
    cd ns-o-ran-gym
    python3 examples/ablation_offline_rl.py
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsoran.offline_models import (  # noqa: E402
    ACTION_COLS,
    DEVICE,
    ActorNet,
    PolicyNet,
    QNet,
    build_simplex_actions,
    nearest_action_index,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (REPO_ROOT / "resultados_cenarios_finalizados_20260715"
                   / "offline_dataset.parquet")  # v1: the set the frozen MODA models use
OUT_DIR = REPO_ROOT / "models" / "ablation"

# Feature sets for R5a
ONE_HOT = ["is_low_traffic", "is_normal", "is_congestion", "is_stressed"]
OBS_COLS = ["num_ues_total", "num_ues_embb", "num_ues_urllc", "num_ues_mtc",
            "offered_load_embb", "offered_load_urllc", "offered_load_mtc",
            "pkt_size_embb", "pkt_size_urllc", "pkt_size_mtc"]
STATE_SETS = {"FULL": OBS_COLS + ONE_HOT, "OBS": OBS_COLS}

EPOCHS = 300
LR = 1e-3
TRAIN_SPLIT = 0.8
SCENARIOS = ["low_traffic", "normal", "congestion", "stressed"]


# ---------------------------------------------------------------------------
def prepare(df: pd.DataFrame, state_cols: list[str], seed: int):
    """Normalize the chosen feature set and split by (scenario, seed) groups."""
    smin = df[state_cols].min()
    srange = (df[state_cols].max() - smin).replace(0, 1)
    Xn = ((df[state_cols] - smin) / srange).values.astype(np.float32)
    A = df[ACTION_COLS].values.astype(np.float32)
    R = df["score"].values.astype(np.float32)

    rng = np.random.RandomState(seed)
    groups = df.groupby(["scenario", "seed"]).size().reset_index()[["scenario", "seed"]].values
    rng.shuffle(groups)
    train_groups = set(tuple(g) for g in groups[:int(len(groups) * TRAIN_SPLIT)])
    mask = np.array([g in train_groups for g in zip(df["scenario"], df["seed"])])

    # per-scenario mean normalized state (for proxy prediction)
    scen_state = {}
    for sc in SCENARIOS:
        sub = df[df["scenario"] == sc]
        if len(sub):
            raw = sub[state_cols].mean()
            scen_state[sc] = ((raw - smin) / srange).values.astype(np.float32)
    return {"X": Xn[mask], "A": A[mask], "R": R[mask]}, scen_state


def train_q(tr, action_table, seed):
    torch.manual_seed(seed)
    aidx = torch.tensor(np.array([nearest_action_index(a, action_table) for a in tr["A"]]),
                        device=DEVICE, dtype=torch.long)
    m = QNet(tr["X"].shape[1], len(action_table)).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    X, R = torch.tensor(tr["X"], device=DEVICE), torch.tensor(tr["R"], device=DEVICE)
    for _ in range(EPOCHS):
        loss = F.mse_loss(m(X).gather(1, aidx.unsqueeze(1)).squeeze(1), R)
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda s: action_table[int(m(torch.tensor(s[None], device=DEVICE)).argmax())]


def train_bc(tr, df_train_scen, quantile, seed):
    torch.manual_seed(seed)
    # elite selection per scenario is applied on the *training rows*; here tr is
    # already the training split, so we filter by score quantile per scenario.
    idx = np.concatenate([
        np.where((df_train_scen == sc) & (tr["R"] >= np.quantile(tr["R"][df_train_scen == sc], quantile)))[0]
        for sc in np.unique(df_train_scen)
    ])
    m = ActorNet(tr["X"].shape[1]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    X, A = torch.tensor(tr["X"][idx], device=DEVICE), torch.tensor(tr["A"][idx], device=DEVICE)
    for _ in range(EPOCHS):
        loss = F.mse_loss(m(X), A)
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda s: m(torch.tensor(s[None], device=DEVICE)).detach().cpu().numpy()[0]


def train_rwr(tr, tau, seed):
    torch.manual_seed(seed)
    X = torch.tensor(tr["X"], device=DEVICE)
    A = torch.tensor(tr["A"], device=DEVICE)
    R = torch.tensor(tr["R"], device=DEVICE)
    Rn = (R - R.min()) / (R.max() - R.min() + 1e-8)
    w = torch.exp(Rn / tau); w = w / w.sum()
    m = PolicyNet(tr["X"].shape[1]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    for _ in range(EPOCHS):
        mean, _ = m(X)
        mse = ((F.softmax(mean, dim=-1) - A) ** 2).sum(dim=1)
        loss = (w * mse).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda s: m.predict(torch.tensor(s[None], device=DEVICE)).detach().cpu().numpy()[0]


def nearest_score(df: pd.DataFrame, sc: str, weights: np.ndarray) -> float:
    sub = df[df["scenario"] == sc]
    d = ((sub[ACTION_COLS].values - weights) ** 2).sum(axis=1)
    return float(sub.iloc[int(d.argmin())]["score"])


def proxy_mean_score(predict, scen_state, df) -> float:
    """Mean over scenarios of the proxy score of the predicted weights."""
    vals = []
    for sc, s in scen_state.items():
        w = np.asarray(predict(s), dtype=float)
        w = np.clip(w, 0, None); w = w / w.sum() if w.sum() > 0 else np.full(3, 1/3)
        vals.append(nearest_score(df, sc, w))
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.dataset)
    print(f"Loaded {len(df)} rows from {args.dataset.name}\n")
    rows = []

    # --- R5a: one-hot vs observable-only, all 3 recipes at default hypers ---
    print("=== R5a: FULL (one-hot) vs OBS (observable-only) ===")
    for fs_name, cols in STATE_SETS.items():
        tr, scen_state = prepare(df, cols, args.seed)
        # df_train_scen: scenario label aligned to tr rows (for BC elite filter)
        rng = np.random.RandomState(args.seed)
        groups = df.groupby(["scenario", "seed"]).size().reset_index()[["scenario", "seed"]].values
        rng.shuffle(groups)
        tg = set(tuple(g) for g in groups[:int(len(groups) * TRAIN_SPLIT)])
        scen_lbl = df["scenario"].values[np.array([g in tg for g in zip(df["scenario"], df["seed"])])]

        preds = {
            "MODA-Q": train_q(tr, build_simplex_actions(0.1), args.seed),
            "MODA-BC": train_bc(tr, scen_lbl, 0.80, args.seed),
            "MODA-RWR": train_rwr(tr, 0.1, args.seed),
        }
        for name, pred in preds.items():
            sc = proxy_mean_score(pred, scen_state, df)
            rows.append({"study": "R5a", "knob": "feature_set", "value": fs_name,
                         "model": name, "proxy_mean_score": sc, "state_dim": len(cols)})
            print(f"  {fs_name:<5} {name:<9} state_dim={len(cols):<2} proxy_mean={sc:.2f}")
    print()

    # --- R8: hyperparameter sweeps (FULL feature set) ---
    tr, scen_state = prepare(df, STATE_SETS["FULL"], args.seed)
    rng = np.random.RandomState(args.seed)
    groups = df.groupby(["scenario", "seed"]).size().reset_index()[["scenario", "seed"]].values
    rng.shuffle(groups)
    tg = set(tuple(g) for g in groups[:int(len(groups) * TRAIN_SPLIT)])
    scen_lbl = df["scenario"].values[np.array([g in tg for g in zip(df["scenario"], df["seed"])])]

    print("=== R8a: MODA-Q — discretization step delta ===")
    for delta in [0.05, 0.10, 0.20]:
        at = build_simplex_actions(delta)
        pred = train_q(tr, at, args.seed)
        sc = proxy_mean_score(pred, scen_state, df)
        rows.append({"study": "R8", "knob": "delta", "value": delta, "model": "MODA-Q",
                     "proxy_mean_score": sc, "n_actions": len(at)})
        print(f"  delta={delta:<5} |A|={len(at):<4} proxy_mean={sc:.2f}")

    print("\n=== R8b: MODA-RWR — temperature tau ===")
    for tau in [0.05, 0.10, 0.50, 1.00]:
        pred = train_rwr(tr, tau, args.seed)
        sc = proxy_mean_score(pred, scen_state, df)
        rows.append({"study": "R8", "knob": "tau", "value": tau, "model": "MODA-RWR",
                     "proxy_mean_score": sc})
        print(f"  tau={tau:<5} proxy_mean={sc:.2f}")

    print("\n=== R8c: MODA-BC — elite quantile q ===")
    for q in [0.70, 0.80, 0.90]:
        pred = train_bc(tr, scen_lbl, q, args.seed)
        sc = proxy_mean_score(pred, scen_state, df)
        rows.append({"study": "R8", "knob": "elite_quantile", "value": q, "model": "MODA-BC",
                     "proxy_mean_score": sc})
        print(f"  q={q:<5} proxy_mean={sc:.2f}")

    out = OUT_DIR / "ablation_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
