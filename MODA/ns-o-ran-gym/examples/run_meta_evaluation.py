#!/usr/bin/env python3
"""Post-search evaluation: re-run meta-heuristic best candidates as slice_custom.

Reads best_candidate_<scenario>_seed<seed>.json files produced by
run_rslaq_metaheuristics.py and re-executes ns-3 with the optimized weights
on the same scenario/seed grid used by the baselines, so that meta-heuristics
can be compared side-by-side with RR, BCQI, PF, RSLAQ, AQPS, and heuristics.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsoran.scoring import read_summary, score_summary_rows


def load_best_candidates(search_root: Path) -> list[dict]:
    candidates: list[dict] = []
    for json_path in sorted(search_root.rglob("best_candidate_*_seed*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        payload["_source_file"] = str(json_path)
        candidates.append(payload)
    return candidates


def build_eval_command(
    *,
    scenario: str,
    output_root: Path,
    weights_str: str,
    intra_algo: str,
    sim_time: float,
    app_start: float,
    drain_time: float,
    period_ms: int,
    seed: int,
    run: int,
    tx_power: float = 43.0,
    tdd_pattern: str = "D|D|8D|4GB|4U|U|U",
    rlc_mode: str = "um",
) -> str:
    return (
        "scratch/rslaq/rslaq-sim "
        f"--scenario={scenario} "
        "--baselineMode=slice_custom "
        f"--weights={weights_str} "
        f"--intraAlgo={intra_algo} "
        f"--simTime={sim_time:g} "
        f"--appStart={app_start:g} "
        f"--drainTimeSec={drain_time:g} "
        f"--periodMs={period_ms} "
        f"--seed={seed} "
        f"--run={run} "
        f"--txPower={tx_power:g} "
        f"--tddPattern={tdd_pattern} "
        f"--rlcMode={rlc_mode} "
        f"--outputDir={output_root}"
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Re-run meta-heuristic best candidates for fair comparison")
    parser.add_argument("--search_root", type=Path, required=True, help="Root containing scenario=.../seed=.../metaheuristic_search/")
    parser.add_argument("--output_root", type=Path, default=repo_root / "ns-3-dev" / "results_meta_evaluation")
    parser.add_argument("--ns3_dir", type=Path, default=repo_root / "ns-3-dev")
    parser.add_argument("--intra_algo", choices=["RR", "PF", "BCQI"], default="PF")
    parser.add_argument("--sim_time", type=float, default=20.0)
    parser.add_argument("--app_start", type=float, default=0.4)
    parser.add_argument("--drain_time", type=float, default=0.2)
    parser.add_argument("--period_ms", type=int, default=10)
    parser.add_argument("--tx_power", type=float, default=43.0)
    parser.add_argument("--tdd_pattern", default="D|D|8D|4GB|4U|U|U")
    parser.add_argument("--rlc_mode", choices=["um", "am"], default="um")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--build_ns3", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.build_ns3:
        subprocess.run(["./ns3", "build", "rslaq-sim"], cwd=args.ns3_dir, check=True)

    candidates = load_best_candidates(args.search_root)
    if not candidates:
        print(f"No best_candidate_*_seed*.json found under {args.search_root}")
        return

    print(f"[EVAL] Found {len(candidates)} best candidate(s)")

    for candidate in candidates:
        scenario = candidate["scenario"]
        seed = candidate["seed"]
        method = candidate["method"]
        weights_str = candidate["ns3_args"]["weights"]

        eval_dir = args.output_root / f"scenario={scenario}" / f"mode=meta_{method}" / f"seed={seed}_run={args.run}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        log_file = eval_dir / "ns3.log"

        cmd = build_eval_command(
            scenario=scenario,
            output_root=eval_dir,
            weights_str=weights_str,
            intra_algo=args.intra_algo,
            sim_time=args.sim_time,
            app_start=args.app_start,
            drain_time=args.drain_time,
            period_ms=args.period_ms,
            seed=seed,
            run=args.run,
            tx_power=args.tx_power,
            tdd_pattern=args.tdd_pattern,
            rlc_mode=args.rlc_mode,
        )

        print(f"[EVAL] method={method} scenario={scenario} seed={seed} weights={weights_str}")
        subprocess.run(f"./ns3 run \"{cmd}\"", shell=True, cwd=args.ns3_dir, check=True, stdout=log_file.open("w"), stderr=subprocess.STDOUT)

        summary_path = eval_dir / "results_rslaq_network_only" / f"scenario={scenario}" / "mode=slice_custom" / f"seed={seed}_run={args.run}" / "summary.csv"
        if summary_path.exists():
            rows = read_summary(summary_path)
            score = score_summary_rows(rows)
            print(f"  score={score:.4f}")
        else:
            print(f"  WARNING: summary.csv not found at {summary_path}")

    print(f"[EVAL] Output: {args.output_root}")


if __name__ == "__main__":
    main()
