#!/usr/bin/env python3
"""Out-of-distribution (OOD) evaluation of the MODA policies on the held-out
scenario `insufficient_resources` (295 Mbps / 70 UEs — never in training).

Responds to the peer-review request for robustness under traffic peaks outside
the training distribution. Protocol:

1. OBS (observable-only) variants: retrain the 3 policies WITHOUT the scenario
   one-hot features (state_dim=10). R5a (ablation_R5a_R8.md) showed BC/RWR
   produce identical predictions without one-hot, so this is the honest
   zero-shot protocol: the agent must map unseen observables -> weights with
   no scenario oracle. MODA-Q collapses to [0.5,0.1,0.4] regardless.
2. FULL-state variants: predict with the existing checkpoints (state_dim=14)
   feeding all one-hot flags = 0 (a vector never seen in training). When the
   predicted weights match an OBS prediction (<=1e-9), the ns-3 result is
   shared; otherwise the prediction is reported but NOT evaluated (approved
   simulation budget: 9 runs = 3 OBS models x 3 seeds).
3. Each (model-variant, seed) runs standalone ns-3 (`slice_custom`, SIM_TIME=5,
   same protocol as the MODA closed-loop line) and is scored with BOTH the
   v1 composite score (MODA line metric) and the v2 Path C score (labeled
   secondary; comparable to the v2 references on disk).

Outputs under models/ood_insufficient/:
  method=moda-{q,bc,rwr}-{obs,full}/scenario=insufficient_resources/seed=N/
      candidate.json (weights, score_v1, score_v2, failed, reason)
  ood_results.csv  (one row per model-variant x seed)

Usage:
    cd ns-o-ran-gym
    python3 examples/eval_moda_ood.py --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from nsoran.offline_models import (  # noqa: E402
    ACTION_COLS,
    DEVICE,
    STATE_COLS,
    ActorNet,
    PolicyNet,
    QNet,
    build_simplex_actions,
    compute_norm_params,
    nearest_action_index,
)
from nsoran.scoring import score_summary_rows, score_summary_rows_v2  # noqa: E402
from run_rslaq_metaheuristics import build_sim_command  # noqa: E402

REPO_ROOT = Path(HERE).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
DEFAULT_DATASET = (REPO_ROOT / "resultados_cenarios_finalizados_20260715"
                   / "offline_dataset_v2.parquet")
NS3_DIR_REAL = REPO_ROOT / "ns-3-dev"

# Observable-only state columns (drop the 4 one-hot scenario flags)
OBS_COLS = STATE_COLS[:10]

# insufficient_resources profile (rslaq-sim.cc:251) — 20/10/40 UEs,
# 220/5/70 Mbps offered, pkt 1500/100/500 B
HELDOUT_OBS = {
    "num_ues_total": 70,
    "num_ues_embb": 20,
    "num_ues_urllc": 10,
    "num_ues_mtc": 40,
    "offered_load_embb": 220.0,
    "offered_load_urllc": 5.0,
    "offered_load_mtc": 70.0,
    "pkt_size_embb": 1500,
    "pkt_size_urllc": 100,
    "pkt_size_mtc": 500,
}
HELDOUT_SCENARIO = "insufficient_resources"

SIM_TIME = 5.0
APP_START = 0.5
DRAIN = 0.2
PERIOD_MS = 10
EPOCHS = 300
LR = 1e-3
TRAIN_SPLIT = 0.8
TRAIN_SEED = 42  # same as the committed checkpoints


def log(msg: str) -> None:
    print(f"[ood] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data prep (mirrors train_offline_rl.load_and_split, parameterized by cols)
# ---------------------------------------------------------------------------
def load_train_data(parquet: Path, cols: list[str]):
    df = pd.read_parquet(parquet)
    norm_params = compute_norm_params(df)  # over full STATE_COLS; we slice after
    mins = np.array([norm_params["min"][c] for c in cols], dtype=np.float32)
    ranges = np.array([norm_params["range"][c] for c in cols], dtype=np.float32)
    X = ((df[cols] - np.array([norm_params["min"][c] for c in cols]))
         / np.array([norm_params["range"][c] for c in cols])).values.astype(np.float32)
    actions = df[ACTION_COLS].values.astype(np.float32)
    scores = df["score"].values.astype(np.float32)

    rng = np.random.RandomState(TRAIN_SEED)
    groups = df.groupby(["scenario", "seed"]).size().reset_index()[["scenario", "seed"]].values
    rng.shuffle(groups)
    n_train = int(len(groups) * TRAIN_SPLIT)
    train_groups = set(tuple(g) for g in groups[:n_train])
    mask = np.array([g in train_groups for g in zip(df["scenario"], df["seed"])])
    return {"X": X[mask], "A": actions[mask], "R": scores[mask],
            "df_train": df[mask].reset_index(drop=True)}, norm_params


# ---------------------------------------------------------------------------
# OBS retraining (same losses as train_offline_rl, state_dim=10)
# ---------------------------------------------------------------------------
def train_obs_models(train: dict):
    action_table = build_simplex_actions(0.1)
    state_dim = train["X"].shape[1]
    X_t = torch.tensor(train["X"], device=DEVICE)
    A_t = torch.tensor(train["A"], device=DEVICE)
    R_t = torch.tensor(train["R"], device=DEVICE)
    models = {}

    # ddqn: Q-regression on the taken action
    aidx = np.array([nearest_action_index(a, action_table) for a in train["A"]])
    q = QNet(state_dim, len(action_table)).to(DEVICE)
    opt = torch.optim.Adam(q.parameters(), lr=LR)
    for _ in range(EPOCHS):
        loss = F.mse_loss(q(X_t).gather(1, torch.tensor(aidx).unsqueeze(1)).squeeze(1), R_t)
        opt.zero_grad(); loss.backward(); opt.step()
    q.eval()
    models["q"] = (q, "ddqn", action_table)

    # sac: BC of top-20% per scenario
    df = train["df_train"]
    tops = []
    for sc in df["scenario"].unique():
        sub = df[df["scenario"] == sc]
        tops.append(sub[sub["score"] >= sub["score"].quantile(0.80)])
    top_idx = pd.concat(tops).index.values
    Xt, At = X_t[torch.tensor(top_idx)], A_t[torch.tensor(top_idx)]
    bc = ActorNet(state_dim).to(DEVICE)
    opt = torch.optim.Adam(bc.parameters(), lr=LR)
    for _ in range(EPOCHS):
        loss = F.mse_loss(bc(Xt), At)
        opt.zero_grad(); loss.backward(); opt.step()
    bc.eval()
    models["bc"] = (bc, "sac", None)

    # ppo: reward-weighted regression (tau=0.1)
    R_norm = (R_t - R_t.min()) / (R_t.max() - R_t.min() + 1e-8)
    w = torch.exp(R_norm / 0.1); w = w / w.sum()
    rwr = PolicyNet(state_dim).to(DEVICE)
    opt = torch.optim.Adam(rwr.parameters(), lr=LR)
    for _ in range(EPOCHS):
        mean, _ = rwr(X_t)
        pred = F.softmax(mean, dim=-1)
        loss = (w * ((pred - A_t) ** 2).sum(dim=1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    rwr.eval()
    models["rwr"] = (rwr, "ppo", None)
    return models


def predict(model_pack, state_norm: np.ndarray) -> np.ndarray:
    model, mtype, action_table = model_pack
    x = torch.tensor(state_norm.reshape(1, -1), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        if mtype == "ddqn":
            idx = model(x).argmax(dim=1).item()
            return np.asarray(action_table)[idx]
        if mtype == "sac":
            return model(x).cpu().numpy()[0]
        return model.predict(x).cpu().numpy()[0]


# ---------------------------------------------------------------------------
# ns-3 execution + scoring
# ---------------------------------------------------------------------------
def run_dir_for(out_root: Path, method: str, seed: int) -> Path:
    return out_root / f"method=moda-{method}" / f"scenario={HELDOUT_SCENARIO}" / f"seed={seed}"


def read_summary(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def execute(out_root: Path, method: str, seed: int, weights) -> dict:
    run_dir = run_dir_for(out_root, method, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_sim_command(
        scenario=HELDOUT_SCENARIO, output_root=run_dir, weights=weights,
        intra_algo="PF", sim_time=SIM_TIME, app_start=APP_START, drain_time=DRAIN,
        period_ms=PERIOD_MS, seed=seed, run=1,
    )
    log(f"run moda-{method} seed={seed} w={np.round(weights, 4).tolist()}")
    with (run_dir / "ns3.log").open("w") as lf:
        proc = subprocess.run(["./ns3", "run", cmd], cwd=str(NS3_DIR_REAL),
                              stdout=lf, stderr=subprocess.STDOUT, text=True, check=False)
    summaries = list(run_dir.rglob("summary.csv"))
    if proc.returncode != 0 or not summaries:
        return {"weights": weights.tolist(), "score_v1": None, "score_v2": None,
                "failed": True, "reason": f"ns-3 exit {proc.returncode}"}
    rows = read_summary(summaries[0])
    return {"weights": weights.tolist(), "score_v1": float(score_summary_rows(rows)),
            "score_v2": float(score_summary_rows_v2(rows)),
            "failed": False, "reason": ""}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--jobs", type=int, default=6,
                    help="Parallel ns-3 runs for the OBS evaluation")
    ap.add_argument("--out-root", type=Path, default=MODELS_DIR / "ood_insufficient")
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    # 1. Train OBS variants (CPU, seconds) — seeds fixed for reproducibility,
    #    mirroring run_one_seed in train_offline_rl.py
    log(f"training OBS variants (state_dim={len(OBS_COLS)}) on "
        f"{DEFAULT_DATASET.name} ...")
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    train, _ = load_train_data(DEFAULT_DATASET, OBS_COLS)
    obs_models = train_obs_models(train)

    # held-out state, normalized with dataset params (sliced to OBS cols)
    df = pd.read_parquet(DEFAULT_DATASET)
    norm = compute_norm_params(df)
    obs_raw = np.array([[HELDOUT_OBS[c] for c in OBS_COLS]], dtype=np.float32)
    obs_norm = (obs_raw - np.array([norm["min"][c] for c in OBS_COLS])) \
        / np.array([norm["range"][c] for c in OBS_COLS])

    # 2. Predictions: OBS variants + FULL checkpoints (one-hot = 0)
    preds = {}
    for key in ("q", "bc", "rwr"):
        preds[f"{key}-obs"] = predict(obs_models[key], obs_norm[0])
        log(f"pred moda-{key}-obs = {np.round(preds[f'{key}-obs'], 4).tolist()}")
    from nsoran.offline_models import load_model
    full_raw = list(obs_raw[0]) + [0.0, 0.0, 0.0, 0.0]  # all one-hots zero
    full_norm = (np.array(full_raw) - np.array([norm["min"][c] for c in STATE_COLS])) \
        / np.array([norm["range"][c] for c in STATE_COLS])
    for key, fname in (("q", "ddqn_offline.pt"), ("bc", "sac_offline.pt"),
                       ("ppo", "ppo_offline.pt")):
        mtype = {"q": "ddqn", "bc": "sac", "ppo": "ppo"}[key]
        model, ckpt = load_model(MODELS_DIR / fname, mtype)
        table = ckpt.get("action_table")
        preds[f"{key}-full"] = predict((model, mtype, table), full_norm)
        log(f"pred moda-{key}-full (one-hot=0) = {np.round(preds[f'{key}-full'], 4).tolist()}")

    # 3. ns-3 runs: OBS approved (3 models x seeds); FULL shares results when
    #    weights coincide (<=1e-9), else reported unevaluated (budget).
    # OBS runs go through a thread pool (ns-3 subprocesses are IO/CPU bound;
    # --jobs mirrors eval_offline_rl_closedloop behavior).
    from concurrent.futures import ThreadPoolExecutor

    obs_jobs = [(variant, seed) for variant in ("q-obs", "bc-obs", "rwr-obs")
                for seed in args.seeds]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(execute, args.out_root, variant, seed,
                               np.array(preds[variant])): (variant, seed)
                   for variant, seed in obs_jobs}
        executed: dict[tuple[int, tuple], dict] = {}
        for fut in futures:
            pass  # results consumed below after completion
        for fut, (variant, seed) in futures.items():
            executed[(seed, tuple(np.round(preds[variant], 9)))] = fut.result()

    results = []
    for variant in ("q-obs", "bc-obs", "rwr-obs", "q-full", "bc-full", "rwr-full"):
        approved = variant.endswith("-obs")
        for seed in args.seeds:
            w = preds[variant]
            wk = tuple(np.round(w, 9))
            if approved:
                res = executed[(seed, wk)]
            else:
                match = executed.get((seed, wk))
                if match is not None:
                    res = dict(match)
                    res["shared_with_obs"] = True
                else:
                    res = {"weights": list(w), "score_v1": None, "score_v2": None,
                           "failed": None, "reason": "not evaluated (approved budget: OBS runs only)"}
            res.update({"method": f"moda-{variant}", "scenario": HELDOUT_SCENARIO,
                        "seed": seed})
            results.append(res)
            sidecar_dir = run_dir_for(args.out_root, variant, seed)
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            with (sidecar_dir / "candidate.json").open("w") as f:
                json.dump(res, f, indent=2)
            log(f"moda-{variant} seed={seed}: v1={res['score_v1']} v2={res['score_v2']}")

    out_csv = args.out_root / "ood_results.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "scenario", "seed", "weights",
                                          "score_v1", "score_v2", "failed",
                                          "reason", "shared_with_obs"])
        w.writeheader()
        for r in results:
            w.writerow({**r, "weights": str(r["weights"]),
                        "shared_with_obs": r.get("shared_with_obs", "")})
    log(f"done -> {out_csv}")


if __name__ == "__main__":
    main()
