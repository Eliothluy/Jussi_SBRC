#!/usr/bin/env python3
"""Re-run DDQN-extracted weights in standalone ns-3 and score with Path C.

After extract_ddqn_weights.py produces ddqn_extracted_weights.csv, this script
re-executes each (scenario, seed) as a standalone ns-3 run with
--baselineMode=slice_custom --weights=<w_embb,w_urllc,w_mtc>, then scores the
resulting summary.csv with score_summary_rows_v2 (Path C). The output
(ddqn_scored_v2.csv) is directly comparable to the meta-heuristic best
candidates scored under the same protocol.

Usage:
    python3 examples/rescore_ddqn_standalone.py \
        --weights-csv ddqn_extracted_weights.csv \
        --ns3-dir ../ns-3-dev \
        --output-root results_controlled/heuristics_metaheuristics/20260811_rslaq_ddqn_paper/rslaq_ddqn_paper/ddqn_rescored \
        --output-csv ddqn_scored_v2.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
GYM_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(GYM_DIR, "src"))

from nsoran.scoring import score_summary_rows_v2  # noqa: E402


def read_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_summary(result_dir: Path) -> Path | None:
    """Locate summary.csv produced by rslaq-sim under an output dir."""
    matches = list(result_dir.rglob("summary.csv"))
    return matches[0] if matches else None


def build_command(ns3_dir: Path, scenario: str, weights: list[float],
                  seed: int, sim_time: float, output_dir: Path) -> list[str]:
    # Renormalize to absorb rounding drift from the extracted weights (the
    # scheduler asserts sum=1.0 within 1e-6; rounded 6-decimal weights can
    # drift by exactly 1e-6 and trip the assert).
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    w = f"{weights[0]:.6f},{weights[1]:.6f},{weights[2]:.6f}"
    sim_cmd = (
        f"scratch/rslaq/rslaq-sim "
        f"--scenario={scenario} "
        f"--baselineMode=slice_custom "
        f"--weights={w} "
        f"--intraAlgo=PF "
        f"--simTime={sim_time:g} "
        f"--appStart=0.5 "
        f"--drainTimeSec=0.2 "
        f"--periodMs=10 "
        f"--seed={seed} "
        f"--run=1 "
        f"--txPower=43 "
        f"--tddPattern=D|D|8D|4GB|4U|U|U "
        f"--rlcMode=um "
        f"--outputDir={output_dir}"
    )
    return ["./ns3", "run", sim_cmd]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights-csv", required=True)
    ap.add_argument("--ns3-dir", required=True)
    ap.add_argument("--output-root", required=True,
                    help="Base dir for standalone re-runs")
    ap.add_argument("--output-csv", required=True,
                    help="Output CSV with Path-C scores")
    ap.add_argument("--sim-time", type=float, default=5.0)
    ap.add_argument("--seeds", default="",
                    help="Comma-separated seed filter (e.g. '2,4,8'); empty = all")
    ap.add_argument("--scenarios", default="",
                    help="Comma-separated scenario filter (e.g. 'stressed'); empty = all")
    args = ap.parse_args()

    ns3_dir = Path(args.ns3_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with open(args.weights_csv, newline="") as f:
        weight_rows = list(csv.DictReader(f))

    # Apply optional --seeds / --scenarios filters
    if args.seeds:
        wanted_seeds = {int(s) for s in args.seeds.split(",") if s.strip()}
        weight_rows = [r for r in weight_rows if int(r["seed"]) in wanted_seeds]
        print(f"[FILTER] --seeds={sorted(wanted_seeds)} -> {len(weight_rows)} rows")
    if args.scenarios:
        wanted_scn = {s.strip() for s in args.scenarios.split(",") if s.strip()}
        weight_rows = [r for r in weight_rows if r["scenario"] in wanted_scn]
        print(f"[FILTER] --scenarios={sorted(wanted_scn)} -> {len(weight_rows)} rows")

    scored = []
    for row in weight_rows:
        scenario = row["scenario"]
        seed = int(row["seed"])
        weights = [float(row["w_embb"]), float(row["w_urllc"]), float(row["w_mtc"])]

        run_dir = output_root / f"scenario={scenario}" / "mode=rslaq_ddqn" / f"seed={seed}_run=1"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = run_dir.parent.parent / f"ns3_seed{seed}.log"

        cmd = build_command(ns3_dir, scenario, weights, seed, args.sim_time, run_dir)
        print(f"[RESCORE] scenario={scenario} seed={seed} "
              f"w=[{weights[0]:.4f},{weights[1]:.4f},{weights[2]:.4f}]")
        try:
            with open(log_file, "w") as lf:
                proc = subprocess.run(
                    cmd, cwd=str(ns3_dir), stdout=lf, stderr=subprocess.STDOUT,
                    timeout=3600,
                )
            if proc.returncode != 0:
                print(f"  [WARN] ns-3 exit={proc.returncode}, see {log_file}",
                      file=sys.stderr)
        except subprocess.TimeoutExpired:
            print(f"  [WARN] ns-3 timeout for scenario={scenario} seed={seed}",
                  file=sys.stderr)

        summary = find_summary(run_dir)
        if summary is None:
            print(f"  [WARN] no summary.csv under {run_dir}", file=sys.stderr)
            scored.append({**row, "score_v2": "", "status": "no_summary"})
            continue

        rows = read_summary(summary)
        score = score_summary_rows_v2(rows) if rows else None
        # Extract per-slice KPIs for the comparison
        kpis = {r.get("slice", ""): r for r in rows}
        urllc = kpis.get("URLLC", {})
        rec = {
            "scenario": scenario,
            "seed": seed,
            "method": "rslaq_ddqn",
            "score_v2": round(score, 4) if score is not None else "",
            "w_embb": weights[0],
            "w_urllc": weights[1],
            "w_mtc": weights[2],
            "urllc_p99_ms": urllc.get("delay_ms_p99", ""),
            "urllc_pdr_pct": urllc.get("pdr_pct", ""),
            "min_sla_pct": min(
                (float(r.get("sla_satisfaction_pct", 100)) for r in rows),
                default=0,
            ),
            "mean_sla_pct": (
                sum(float(r.get("sla_satisfaction_pct", 0)) for r in rows) / len(rows)
                if rows else 0
            ),
            "status": "ok" if score is not None else "score_failed",
        }
        scored.append(rec)
        print(f"  -> score_v2={score:.4f}" if score is not None else "  -> score FAILED")

    fieldnames = [
        "scenario", "seed", "method", "score_v2",
        "w_embb", "w_urllc", "w_mtc",
        "urllc_p99_ms", "urllc_pdr_pct",
        "min_sla_pct", "mean_sla_pct", "status",
    ]
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in scored:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    ok = sum(1 for r in scored if r.get("status") == "ok")
    print(f"\n[OK] {ok}/{len(scored)} scored -> {out_path}")
    return 0 if ok == len(scored) else 1


if __name__ == "__main__":
    sys.exit(main())
