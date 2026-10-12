#!/usr/bin/env python3
"""Ingest online RSLAQ DDQN results (trained on another machine).

Reads the output directory of examples/rslaq_train_ddqn.py — it must contain
evaluation_log.csv (preferred) or ddqn_training_log.csv, with columns:
  episode, scenario, seed, reward_mode, total_reward, outage_count, soft_count,
  steps, action_embb, action_urllc, action_mtc
and optionally ddqn_summary.json.

Per (scenario, seed) it extracts the agent's action weights from the final (or
best-reward) episode plus reward statistics, writing online_rslaq_summary.csv
for consumption by eval_offline_rl.py --online-results.

HONESTY RULE: total_reward is on the RSLAQ paper-reward scale and is NOT
comparable to nsoran.scoring scores. Only the extracted WEIGHTS are comparable,
by scoring them through the same path as every other method (proxy or ns-3
re-run). total_reward is carried in its own clearly-named column and must never
be plotted on a scoring axis.

Usage:
    cd ns-o-ran-gym
    python3 examples/ingest_online_rslaq.py --results-dir /path/to/online/results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsoran.scoring import normalize_weights

REPO_ROOT = Path(__file__).resolve().parents[2]

LOG_CANDIDATES = ["evaluation_log.csv", "ddqn_training_log.csv"]
WEIGHT_COLS = ["action_embb", "action_urllc", "action_mtc"]
REQUIRED_COLS = {"episode", "scenario", "seed", "total_reward", *WEIGHT_COLS}


def _find_log(results_dir: Path) -> Path:
    for name in LOG_CANDIDATES:
        matches = sorted(results_dir.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"None of {LOG_CANDIDATES} found under {results_dir}. "
        "Expected the output directory of rslaq_train_ddqn.py.")


def ingest_results_dir(results_dir: Path, select: str = "final") -> pd.DataFrame:
    """Extract per-(scenario, seed) final policy weights and reward stats."""
    log_path = _find_log(results_dir)
    log = pd.read_csv(log_path)
    missing = REQUIRED_COLS - set(log.columns)
    if missing:
        raise ValueError(f"{log_path} is missing columns: {sorted(missing)}")
    print(f"Ingesting {log_path} ({len(log)} episodes, select={select})")

    rows = []
    for (scenario, seed), group in log.groupby(["scenario", "seed"]):
        group = group.sort_values("episode")
        picked = (group.iloc[-1] if select == "final"
                  else group.loc[group["total_reward"].idxmax()])
        weights = normalize_weights([float(picked[c]) for c in WEIGHT_COLS])
        rows.append({
            "scenario": scenario,
            "seed": int(seed),
            "w_embb": weights[0],
            "w_urllc": weights[1],
            "w_mtc": weights[2],
            "selected_episode": int(picked["episode"]),
            "selection": select,
            # RSLAQ paper-reward scale — NOT comparable to nsoran.scoring.
            "online_total_reward_selected": float(picked["total_reward"]),
            "online_total_reward_mean": float(group["total_reward"].mean()),
            "online_total_reward_std": float(group["total_reward"].std()),
            "outage_count_selected": (int(picked["outage_count"])
                                      if "outage_count" in group.columns else None),
            "episodes": int(len(group)),
            "source_log": log_path.name,
        })
    if not rows:
        raise ValueError(f"{log_path} has no (scenario, seed) groups")

    summary_json = next(iter(sorted(results_dir.rglob("ddqn_summary.json"))), None)
    if summary_json is not None:
        try:
            json.loads(summary_json.read_text(encoding="utf-8"))
            print(f"Found {summary_json} (kept alongside, not merged)")
        except (OSError, ValueError):
            pass

    return pd.DataFrame(rows).sort_values(["scenario", "seed"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Output dir of rslaq_train_ddqn.py from the online machine")
    parser.add_argument("--select", choices=["final", "best"], default="final",
                        help="Which episode's weights represent the trained policy")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: <models>/online_rslaq_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = ingest_results_dir(args.results_dir, args.select)

    output = args.output or REPO_ROOT / "models" / "online_rslaq_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\nSaved: {output}")
    print(df[["scenario", "seed", "w_embb", "w_urllc", "w_mtc",
              "selected_episode", "online_total_reward_selected"]].to_string(index=False))
    print("\nNext: python3 examples/eval_offline_rl.py --online-results", output)


if __name__ == "__main__":
    main()
