#!/usr/bin/env python3
"""Evaluate the trained offline policies against baselines and online RSLAQ.

Two evaluation modes:
  - proxy (default): score = nearest candidate in the dataset (weight-space
    Euclidean). Fast, no simulation; scores inherit the dataset's sim_time
    (5 s search runs). Development use.
  - ns3: re-run each policy's predicted weights in ns-3 as slice_custom (same
    path as run_meta_evaluation.py) and score the produced summary.csv with
    nsoran.scoring. Fair, publication-grade comparison. Default sim_time is
    20 s — NEVER mix proxy (5 s) and ns3 (20 s) scores in one unlabeled table;
    the output CSV carries an eval_mode column for this reason.

Online RSLAQ (trained on another machine) enters via --online-results, pointing
at the CSV produced by examples/ingest_online_rslaq.py (or the raw results dir
containing evaluation_log.csv). Only its WEIGHTS are comparable: its episodic
total_reward is on the RSLAQ paper-reward scale, not the scoring scale.

Usage:
    cd ns-o-ran-gym
    python3 examples/eval_offline_rl.py
    python3 examples/eval_offline_rl.py --mode ns3 --models sac --scenarios normal --ns3-seeds 1
    python3 examples/eval_offline_rl.py --online-results results/online_rslaq_summary.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nsoran.offline_models import (
    STATE_COLS,
    load_model,
    norm_params_from_checkpoint,
    predict_weights,
)
from nsoran.scoring import format_weights, read_summary, score_summary_rows
from run_meta_evaluation import build_eval_command

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (REPO_ROOT / "resultados_cenarios_finalizados_20260715"
                   / "offline_dataset_v2.parquet")

SCENARIOS = ["low_traffic", "normal", "congestion", "stressed"]
SCENARIO_LABELS = {"low_traffic": "Low Traffic", "normal": "Normal",
                   "congestion": "Congestion", "stressed": "Stressed"}

BASELINES = {
    "P_STA default [0.33,0.40,0.27]": np.array([0.3333, 0.4000, 0.2667]),
    "Equal [0.33,0.33,0.33]": np.array([0.3333, 0.3333, 0.3333]),
}

METHOD_COLORS = {
    "Best in dataset": "#2ca02c",
    "P_STA default [0.33,0.40,0.27]": "#7f7f7f",
    "Equal [0.33,0.33,0.33]": "#bcbd22",
    "ddqn-offline": "#1f77b4",
    "sac-offline": "#d62728",
    "ppo-offline": "#9467bd",
    "RSLAQ-online": "#ff7f0e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--models", nargs="+", choices=["ddqn", "sac", "ppo"],
                        default=["ddqn", "sac", "ppo"])
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "figuras_overleaf")
    parser.add_argument("--output", type=Path, default=None,
                        help="Comparison CSV (default: <models-dir>/evaluation_comparison.csv)")
    parser.add_argument("--mode", choices=["proxy", "ns3"], default="proxy")
    parser.add_argument("--online-results", type=Path, default=None,
                        help="online_rslaq_summary.csv from ingest_online_rslaq.py "
                             "(or the raw online results dir)")
    # ns3 mode options
    parser.add_argument("--ns3-dir", type=Path, default=REPO_ROOT / "ns-3-dev")
    parser.add_argument("--ns3-output-root", type=Path,
                        default=REPO_ROOT / "ns-3-dev" / "results_offline_drl_eval")
    parser.add_argument("--ns3-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--ns3-include-baselines", action="store_true",
                        help="Also re-run the fixed baselines in ns-3")
    parser.add_argument("--sim-time", type=float, default=20.0)
    parser.add_argument("--app-start", type=float, default=0.4)
    parser.add_argument("--drain-time", type=float, default=0.2)
    parser.add_argument("--period-ms", type=int, default=10)
    parser.add_argument("--build-ns3", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Weight sources
# ---------------------------------------------------------------------------
def build_state_for_scenario(scenario: str, df: pd.DataFrame, norm_params: dict):
    """Averaged, normalized state vector for a scenario."""
    sub = df[df["scenario"] == scenario]
    if sub.empty:
        return None
    raw = sub[STATE_COLS].mean().values.astype(np.float32)
    mins = np.array([norm_params["min"][c] for c in STATE_COLS])
    ranges = np.array([norm_params["range"][c] for c in STATE_COLS])
    return ((raw - mins) / ranges).reshape(1, -1)


def load_trained_models(args) -> dict:
    """{method_name: (model, model_type, action_table, norm_params)}."""
    models = {}
    for mtype in args.models:
        path = args.models_dir / f"{mtype}_offline.pt"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {mtype}")
            continue
        model, ckpt = load_model(path, mtype)
        norm = norm_params_from_checkpoint(ckpt, args.dataset)
        models[f"{mtype}-offline"] = (model, mtype, ckpt.get("action_table"), norm)
        print(f"Loaded {mtype} from {path}"
              + (" (legacy checkpoint, dataset-derived norm)" if not ckpt.get("norm_params") else ""))
    return models


def load_online_weights(path: Path) -> pd.DataFrame | None:
    """Per-(scenario, seed) weights of the online-trained RSLAQ agent."""
    if path is None:
        return None
    if path.is_dir():
        from ingest_online_rslaq import ingest_results_dir
        online = ingest_results_dir(path)
    else:
        online = pd.read_csv(path)
    required = {"scenario", "seed", "w_embb", "w_urllc", "w_mtc"}
    missing = required - set(online.columns)
    if missing:
        raise ValueError(f"Online results missing columns: {sorted(missing)}")
    return online


def collect_method_weights(args, df: pd.DataFrame, models: dict,
                           online: pd.DataFrame | None) -> list[dict]:
    """One entry per (method, scenario): the weight vector to evaluate."""
    entries = []
    for sc in args.scenarios:
        sub = df[df["scenario"] == sc]
        if sub.empty:
            continue
        best_row = sub.nlargest(1, "score").iloc[0]
        entries.append({"scenario": sc, "method": "Best in dataset",
                        "weights": best_row[["w_embb", "w_urllc", "w_mtc"]].values.astype(float),
                        "source_of_weights": f"dataset:{best_row['mode']}",
                        "dataset_score": float(best_row["score"])})
        for bname, bw in BASELINES.items():
            entries.append({"scenario": sc, "method": bname, "weights": bw,
                            "source_of_weights": "fixed"})
        for mname, (model, mtype, atab, norm) in models.items():
            state = build_state_for_scenario(sc, df, norm)
            if state is None:
                continue
            weights = predict_weights(model, mtype, state, atab)[0]
            entries.append({"scenario": sc, "method": mname,
                            "weights": np.asarray(weights, dtype=float),
                            "source_of_weights": "policy"})
        if online is not None:
            osub = online[online["scenario"] == sc]
            if not osub.empty:
                weights = osub[["w_embb", "w_urllc", "w_mtc"]].mean().values
                entries.append({"scenario": sc, "method": "RSLAQ-online",
                                "weights": np.asarray(weights, dtype=float),
                                "source_of_weights": "online_training"})
    return entries


# ---------------------------------------------------------------------------
# Proxy evaluation
# ---------------------------------------------------------------------------
def find_nearest_score(weights: np.ndarray, df: pd.DataFrame, scenario: str) -> float:
    sub = df[df["scenario"] == scenario]
    if sub.empty:
        return float("nan")
    dist = ((sub[["w_embb", "w_urllc", "w_mtc"]].values - weights) ** 2).sum(axis=1)
    return float(sub.iloc[int(dist.argmin())]["score"])


def evaluate_proxy(entries: list[dict], df: pd.DataFrame) -> list[dict]:
    results = []
    for entry in entries:
        score = (entry["dataset_score"] if "dataset_score" in entry
                 else find_nearest_score(entry["weights"], df, entry["scenario"]))
        results.append(_result_row(entry, score, eval_mode="proxy", seed=None))
    return results


# ---------------------------------------------------------------------------
# ns-3 re-evaluation
# ---------------------------------------------------------------------------
def evaluate_ns3(entries: list[dict], args) -> list[dict]:
    if args.build_ns3:
        subprocess.run(["./ns3", "build", "rslaq-sim"], cwd=args.ns3_dir, check=True)

    slug = {"Best in dataset": "best_dataset",
            "RSLAQ-online": "rslaq_online",
            "P_STA default [0.33,0.40,0.27]": "psta_default",
            "Equal [0.33,0.33,0.33]": "equal"}

    results = []
    for entry in entries:
        method = entry["method"]
        if method == "Best in dataset":
            continue  # its 20 s re-run is run_meta_evaluation.py's job
        if entry["source_of_weights"] == "fixed" and not args.ns3_include_baselines:
            continue
        mode_slug = slug.get(method, method.replace("-offline", ""))
        weights_str = format_weights(entry["weights"])
        for seed in args.ns3_seeds:
            eval_dir = (args.ns3_output_root / f"scenario={entry['scenario']}"
                        / f"mode=drl_{mode_slug}" / f"seed={seed}_run=1")
            summary_path = (eval_dir / "results_rslaq_network_only"
                            / f"scenario={entry['scenario']}" / "mode=slice_custom"
                            / f"seed={seed}_run=1" / "summary.csv")
            if not summary_path.exists():
                eval_dir.mkdir(parents=True, exist_ok=True)
                cmd = build_eval_command(
                    scenario=entry["scenario"], output_root=eval_dir,
                    weights_str=weights_str, intra_algo="PF",
                    sim_time=args.sim_time, app_start=args.app_start,
                    drain_time=args.drain_time, period_ms=args.period_ms,
                    seed=seed, run=1)
                print(f"[ns3] {method} scenario={entry['scenario']} seed={seed} "
                      f"weights={weights_str}")
                log_file = eval_dir / "ns3.log"
                with log_file.open("w") as log:
                    proc = subprocess.run(f"./ns3 run \"{cmd}\"", shell=True,
                                          cwd=args.ns3_dir, stdout=log,
                                          stderr=subprocess.STDOUT)
                if proc.returncode != 0 or not summary_path.exists():
                    print(f"  WARNING: run failed (see {log_file}); skipping")
                    continue
            else:
                print(f"[ns3] {method} scenario={entry['scenario']} seed={seed}: "
                      "cached summary.csv, skipping run")
            score = score_summary_rows(read_summary(summary_path))
            print(f"  score={score:.4f}")
            results.append(_result_row(entry, score, eval_mode="ns3", seed=seed))
    return results


def _result_row(entry: dict, score: float, eval_mode: str, seed: int | None) -> dict:
    weights = entry["weights"]
    return {"scenario": entry["scenario"], "method": entry["method"],
            "w_embb": float(weights[0]), "w_urllc": float(weights[1]),
            "w_mtc": float(weights[2]), "score": score,
            "eval_mode": eval_mode, "seed": seed,
            "source_of_weights": entry["source_of_weights"]}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_table(res_df: pd.DataFrame) -> None:
    print("\n" + "=" * 96)
    print(f"COMPARISON ({res_df['eval_mode'].iloc[0]} scores)")
    print("=" * 96)
    print(f"{'Scenario':<16} {'Method':<30} {'w_eMBB':>7} {'w_URLLC':>8} "
          f"{'w_MTC':>7} {'Score':>10} {'Seed':>5}")
    print("-" * 96)
    for _, r in res_df.iterrows():
        seed = "" if pd.isna(r["seed"]) else int(r["seed"])
        print(f"{r['scenario']:<16} {r['method']:<30} {r['w_embb']:>7.3f} "
              f"{r['w_urllc']:>8.3f} {r['w_mtc']:>7.3f} {r['score']:>10.2f} {seed:>5}")


def plot_comparison(res_df: pd.DataFrame, fig_path: Path, eval_mode: str) -> None:
    # Average over seeds for ns3 mode; proxy rows are already one per scenario.
    agg = (res_df.groupby(["scenario", "method"], sort=False)["score"]
           .agg(["mean", "std"]).reset_index())
    scenarios = [sc for sc in SCENARIOS if sc in set(agg["scenario"])]
    methods = list(dict.fromkeys(agg["method"]))
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for i, method in enumerate(methods):
        vals, errs = [], []
        for sc in scenarios:
            row = agg[(agg["scenario"] == sc) & (agg["method"] == method)]
            vals.append(row["mean"].values[0] if len(row) else 0)
            errs.append(row["std"].values[0] if len(row) and not pd.isna(row["std"].values[0]) else 0)
        ax.bar(x + i * width, vals, width, yerr=errs if any(errs) else None,
               capsize=3, label=method, color=METHOD_COLORS.get(method, "gray"),
               alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x + width * (len(methods) / 2 - 0.5))
    ax.set_xticklabels([SCENARIO_LABELS.get(sc, sc) for sc in scenarios])
    ylabel = ("Proxy score (nearest dataset candidate)" if eval_mode == "proxy"
              else "ns-3 re-evaluation score (nsoran.scoring)")
    ax.set_ylabel(ylabel)
    ax.set_title("Offline policies vs. baselines"
                 + (" vs. online RSLAQ" if "RSLAQ-online" in methods else ""))
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.dataset)
    models = load_trained_models(args)
    online = load_online_weights(args.online_results)

    entries = collect_method_weights(args, df, models, online)
    if args.mode == "proxy":
        results = evaluate_proxy(entries, df)
    else:
        results = evaluate_ns3(entries, args)

    if not results:
        print("No results produced.", file=sys.stderr)
        sys.exit(1)

    res_df = pd.DataFrame(results)
    print_table(res_df)

    output = args.output or args.models_dir / "evaluation_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output, index=False)
    print(f"\nSaved: {output}")

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(res_df, args.fig_dir / "fig15_offline_drl_comparison.png", args.mode)


if __name__ == "__main__":
    main()
