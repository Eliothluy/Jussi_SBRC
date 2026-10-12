#!/usr/bin/env python3
"""Closed-loop evaluation of the offline slice-weight policies against ns-3.

Unlike eval_offline_rl.py (which scores predicted weights by the nearest
candidate already present in the offline dataset — a proxy that is circular
and discontinuous), this script closes the loop: each predicted weight vector
is executed as a real ns-3 run (--baselineMode=slice_custom, same command and
parameters as the metaheuristic campaign) and scored with the campaign
objective (nsoran.scoring.score_summary_rows). Scores are therefore directly
comparable to the metaheuristic search scores in the dataset.

Evaluated methods per scenario x seed (seeds 1,2,3 as in the campaign):
  - DDQN/SAC/PPO offline policies (weights predicted from the scenario state)
  - Fixed baselines: RSLAQ [0.3333,0.40,0.2667] and Equal [1/3,1/3,1/3]
  - Best metaheuristic candidate (score read from best_candidate_*.json —
    already a real ns-3 score; not re-run)

Runs are checkpointed with a candidate.json sidecar (same convention as the
campaign): a completed run with matching weights is never re-executed.

Usage:
    cd ns-o-ran-gym
    python3 examples/eval_offline_rl_closedloop.py            # full matrix
    python3 examples/eval_offline_rl_closedloop.py --smoke    # 1 run only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parents[1]
sys.path.insert(0, str(EXAMPLES_DIR.parent / "src"))
sys.path.insert(0, str(EXAMPLES_DIR))

from nsoran.scoring import (  # noqa: E402
    format_weights,
    read_summary,
    score_summary_rows,
)
from nsoran.offline_models import (  # noqa: E402
    STATE_COLS,
    SCENARIOS,
    load_model,
    norm_params_from_checkpoint,
    normalize_state,
    predict_weights,
)
from run_rslaq_metaheuristics import build_sim_command, summary_path  # noqa: E402

DEFAULT_DATA_ROOT = REPO_ROOT / "resultados_cenarios_finalizados_20260715"
DEFAULT_DATASET = DEFAULT_DATA_ROOT / "offline_dataset.parquet"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_NS3_DIR = REPO_ROOT / "ns-3-dev"
DEFAULT_OUT_ROOT = DEFAULT_MODELS_DIR / "closedloop_eval"

# Campaign parameters (must match run_all_scenarios.sh Phase 2 defaults)
SIM_TIME = 5.0
APP_START = 0.4
DRAIN_TIME = 0.2
PERIOD_MS = 10
INTRA_ALGO = "PF"
SEEDS = (1, 2, 3)

FIXED_BASELINES = {
    "RSLAQ fixed [0.33,0.40,0.27]": [0.3333, 0.4000, 0.2667],
    "Equal [0.33,0.33,0.33]": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
}

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def build_scenario_state(df: pd.DataFrame, scenario: str, norm_params: dict) -> np.ndarray:
    sub = df[df["scenario"] == scenario]
    if sub.empty:
        raise ValueError(f"No rows for scenario {scenario} in dataset")
    raw = sub[STATE_COLS].mean().values.astype(np.float32)
    return normalize_state(raw.reshape(1, -1), norm_params)


def predict_all(models_dir: Path, dataset_path: Path, df: pd.DataFrame) -> dict:
    """Return {method: {scenario: weights}} for the three offline policies."""
    preds: dict[str, dict[str, list[float]]] = {}
    for mtype, fname, label in (
        ("ddqn", "ddqn_offline.pt", "DDQN-offline"),
        ("sac", "sac_offline.pt", "SAC-offline"),
        ("ppo", "ppo_offline.pt", "PPO-offline"),
    ):
        path = models_dir / fname
        if not path.exists():
            log(f"[warn] checkpoint missing, skipping: {path}")
            continue
        model, ckpt = load_model(path, mtype)
        norm_params = norm_params_from_checkpoint(ckpt, dataset_path)
        atab = ckpt.get("action_table")
        preds[label] = {}
        for sc in SCENARIOS:
            state = build_scenario_state(df, sc, norm_params)
            w = predict_weights(model, mtype, state, atab)[0]
            w = np.clip(np.asarray(w, dtype=float), 0.0, 1.0)
            w = (w / w.sum()).tolist()
            preds[label][sc] = w
    return preds


def run_dir_for(out_root: Path, method: str, scenario: str, seed: int) -> Path:
    slug = (
        method.lower()
        .replace(" ", "_")
        .replace("[", "")
        .replace("]", "")
        .replace(",", "-")
        .replace(".", "")
    )
    return out_root / f"method={slug}" / f"scenario={scenario}" / f"seed={seed}"


def load_sidecar(run_dir: Path, weights: list[float]) -> float | None:
    sidecar = run_dir / "candidate.json"
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    cached_w = data.get("weights")
    if cached_w is None or data.get("failed"):
        return None
    if max(abs(a - b) for a, b in zip(cached_w, weights)) > 1e-9:
        return None
    return float(data["score"])


def save_sidecar(run_dir: Path, weights: list[float], score: float, *, failed: bool, reason: str = "") -> None:
    payload = {"weights": weights, "score": score, "failed": failed}
    if reason:
        payload["reason"] = reason
    (run_dir / "candidate.json").write_text(json.dumps(payload, indent=2))


def execute_run(
    ns3_dir: Path,
    out_root: Path,
    method: str,
    scenario: str,
    seed: int,
    weights: list[float],
) -> dict:
    run_dir = run_dir_for(out_root, method, scenario, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    cached = load_sidecar(run_dir, weights)
    if cached is not None:
        log(f"[cache] {method} | {scenario} | seed={seed} -> {cached:.2f}")
        return {"scenario": scenario, "method": method, "seed": seed,
                "w_embb": weights[0], "w_urllc": weights[1], "w_mtc": weights[2],
                "score": cached, "source": "cache"}

    command = build_sim_command(
        scenario=scenario, output_root=run_dir, weights=weights,
        intra_algo=INTRA_ALGO, sim_time=SIM_TIME, app_start=APP_START,
        drain_time=DRAIN_TIME, period_ms=PERIOD_MS, seed=seed, run=1,
    )
    log(f"[run  ] {method} | {scenario} | seed={seed} | w={format_weights(weights)}")
    log_file = run_dir / "ns3.log"
    with log_file.open("w") as lf:
        completed = subprocess.run(
            ["./ns3", "run", command], cwd=ns3_dir,
            stdout=lf, stderr=subprocess.STDOUT, text=True, check=False,
        )

    if completed.returncode != 0:
        save_sidecar(run_dir, weights, float("nan"), failed=True,
                     reason=f"ns-3 exit code {completed.returncode}")
        log(f"[FAIL ] {method} | {scenario} | seed={seed}: ns-3 exit {completed.returncode}")
        return {"scenario": scenario, "method": method, "seed": seed,
                "w_embb": weights[0], "w_urllc": weights[1], "w_mtc": weights[2],
                "score": float("nan"), "source": "ns3_crash"}

    summary = summary_path(run_dir, scenario, seed, 1)
    if not summary.exists():
        save_sidecar(run_dir, weights, float("nan"), failed=True, reason="summary.csv missing")
        log(f"[FAIL ] {method} | {scenario} | seed={seed}: summary.csv missing")
        return {"scenario": scenario, "method": method, "seed": seed,
                "w_embb": weights[0], "w_urllc": weights[1], "w_mtc": weights[2],
                "score": float("nan"), "source": "no_summary"}

    score = score_summary_rows(read_summary(summary))
    save_sidecar(run_dir, weights, score, failed=False)
    log(f"[done ] {method} | {scenario} | seed={seed} -> {score:.2f}")
    return {"scenario": scenario, "method": method, "seed": seed,
            "w_embb": weights[0], "w_urllc": weights[1], "w_mtc": weights[2],
            "score": score, "source": "ns3"}


def parse_candidate_weights(data: dict) -> list[float]:
    """best_candidate JSONs store weights as {"eMBB":..,"URLLC":..,"MTC":..}."""
    w = data["weights"]
    if isinstance(w, dict):
        return [float(w["eMBB"]), float(w["URLLC"]), float(w["MTC"])]
    return [float(x) for x in w]


def best_meta_rows(data_root: Path) -> list[dict]:
    """Read real per-seed best-candidate scores from the campaign snapshot.

    NOTE: these scores were measured in the campaign environment (old
    machine/toolchain) and are NOT directly comparable to runs executed
    today — identical weights were observed to differ by tens of points
    across environments. Kept for reference under method
    "Best meta (campaign env)"; the fair comparison is the re-run rows.
    """
    rows = []
    for sc in SCENARIOS:
        for seed in SEEDS:
            path = (data_root / "metaheuristics" / f"scenario={sc}" / f"seed={seed}"
                    / "metaheuristic_search" / f"best_candidate_{sc}_seed{seed}.json")
            if not path.exists():
                log(f"[warn] best candidate missing: {path}")
                continue
            data = json.loads(path.read_text())
            w = parse_candidate_weights(data)
            rows.append({"scenario": sc, "method": "Best meta (campaign env)", "seed": seed,
                         "w_embb": w[0], "w_urllc": w[1], "w_mtc": w[2],
                         "score": float(data["score"]), "source": "best_candidate_json"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ns3-dir", type=Path, default=DEFAULT_NS3_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--smoke", action="store_true",
                        help="execute only the first pending ns-3 run and exit")
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    preds = predict_all(args.models_dir, args.dataset, df)

    # Assemble the run matrix: {(method, scenario, seed): weights}
    matrix: list[tuple[str, str, int, list[float]]] = []
    for method, per_sc in preds.items():
        for sc, w in per_sc.items():
            for seed in args.seeds:
                matrix.append((method, sc, seed, w))
    for method, w in FIXED_BASELINES.items():
        for sc in SCENARIOS:
            for seed in args.seeds:
                matrix.append((method, sc, seed, list(w)))

    # Best metaheuristic candidate per (scenario, seed), re-executed in the
    # CURRENT environment so the comparison with the policies is fair
    # (campaign-era scores are not reproducible across machines/toolchains).
    for sc in SCENARIOS:
        for seed in args.seeds:
            path = (args.data_root / "metaheuristics" / f"scenario={sc}" / f"seed={seed}"
                    / "metaheuristic_search" / f"best_candidate_{sc}_seed{seed}.json")
            if not path.exists():
                log(f"[warn] best candidate missing: {path}")
                continue
            data = json.loads(path.read_text())
            matrix.append(("Best meta (rerun)", sc, seed, parse_candidate_weights(data)))

    print("\nPredicted weights per scenario:")
    for method, per_sc in preds.items():
        for sc, w in per_sc.items():
            print(f"  {method:<14} {sc:<16} {format_weights(w)}")
    print(f"\nTotal ns-3 runs in matrix: {len(matrix)} (cached runs are skipped)")

    if args.smoke:
        for method, sc, seed, w in matrix:
            run_dir = run_dir_for(args.out_root, method, sc, seed)
            if load_sidecar(run_dir, w) is None:
                res = execute_run(args.ns3_dir, args.out_root, method, sc, seed, w)
                print(f"\nSmoke result: {res}")
                return
        print("Nothing pending — all runs cached.")
        return

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(execute_run, args.ns3_dir, args.out_root, m, sc, seed, w)
                   for m, sc, seed, w in matrix]
        for fut in as_completed(futures):
            results.append(fut.result())

    results.extend(best_meta_rows(args.data_root))

    res_df = pd.DataFrame(results).sort_values(["scenario", "method", "seed"])
    out_csv = args.models_dir / "evaluation_closedloop.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved per-seed results: {out_csv}")

    # Aggregate (mean +/- std over the 3 seeds; N=3 by design, no significance claims)
    agg = (res_df.groupby(["scenario", "method"])["score"]
           .agg(["mean", "std", "min", "max", "count"]).reset_index())
    out_agg = args.models_dir / "evaluation_closedloop_aggregate.csv"
    agg.to_csv(out_agg, index=False)
    print(f"Saved aggregate: {out_agg}\n")

    for sc in SCENARIOS:
        sub = agg[agg["scenario"] == sc].sort_values("mean", ascending=False)
        if sub.empty:
            continue
        print(f"=== {sc} ===")
        for _, r in sub.iterrows():
            std = 0.0 if pd.isna(r["std"]) else r["std"]
            print(f"  {r['method']:<30} {r['mean']:>8.2f} +/- {std:<6.2f}"
                  f"  (min {r['min']:.2f}, max {r['max']:.2f}, n={int(r['count'])})")
        print()


if __name__ == "__main__":
    main()
