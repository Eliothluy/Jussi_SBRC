#!/usr/bin/env python3
"""Radio-resource efficiency comparison across closed-loop evaluation runs.

Reads every summary.csv produced by eval_offline_rl_closedloop.py (fresh runs,
all in the CURRENT environment, hence mutually comparable) and compares, per
method x scenario (mean over seeds), how efficiently the resulting slice
allocations use the radio resources:

  - RBG share actually allocated per slice (allocated_rbg_total, normalized)
  - Real resource share used (rsh_real_pct_mean)
  - Unused granted budget (unused_budget_pct_mean)
  - Offered-load satisfaction per slice (served/offered traffic)
  - Goodput per allocated RBG (rx_bytes_total / allocated_rbg_total)
  - QoS outcome: PDR, mean delay, SLA satisfaction per slice

Interpretation caveats (AGENTS.md 14.4): budget_utilization_pct=100 means the
slice consumed its grant, NOT that the cell is full; FlowMonitor throughput
includes IP/UDP headers; congestion should be judged by PDR and mean delay.

Usage:
    cd ns-o-ran-gym
    python3 examples/analyze_closedloop_radio_efficiency.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_ROOT = REPO_ROOT / "models" / "closedloop_eval"
DEFAULT_OUT_DIR = REPO_ROOT / "models"

SCENARIO_ORDER = ["low_traffic", "normal", "congestion", "stressed"]
SLICE_ORDER = ["eMBB", "URLLC", "MTC"]
METHOD_LABELS = {
    "ddqn-offline": "DDQN-offline",
    "sac-offline": "SAC-offline",
    "ppo-offline": "PPO-offline",
    "rslaq_fixed_033-040-027": "RSLAQ-fixed",
    "equal_033-033-033": "Equal",
    "best_meta_(rerun)": "Best meta (rerun)",
}

KPI_COLS = [
    "throughput_mbps_mean", "delay_ms_mean", "pdr_pct",
    "sla_satisfaction_pct", "offered_load_satisfaction_pct",
    "rsh_real_pct_mean", "budget_utilization_pct_mean",
    "unused_budget_pct_mean", "allocated_rbg_total", "rx_bytes_total",
]


def method_label(slug: str) -> str:
    return METHOD_LABELS.get(slug, slug)


def collect(eval_root: Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"method=([^/]+)/scenario=([^/]+)/seed=(\d+)/")
    for summary in sorted(eval_root.glob(
            "method=*/scenario=*/seed=*/results_rslaq_network_only/"
            "scenario=*/mode=*/seed=*_run=*/summary.csv")):
        m = pattern.search(str(summary))
        if not m:
            continue
        method_slug, scenario, seed = m.group(1), m.group(2), int(m.group(3))
        df = pd.read_csv(summary)
        for _, r in df.iterrows():
            row = {"method": method_label(method_slug), "scenario": scenario,
                   "seed": seed, "slice": r["slice"]}
            for c in KPI_COLS:
                row[c] = float(r[c])
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit(f"No summary.csv found under {eval_root}")
    return out


def derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-run derived efficiency metrics."""
    df = df.copy()
    # RBG share of the slice within its own run (cell-level normalization)
    total_rbg = df.groupby(["method", "scenario", "seed"])["allocated_rbg_total"].transform("sum")
    df["rbg_share_pct"] = 100.0 * df["allocated_rbg_total"] / total_rbg
    # Goodput per allocated RBG (bytes per RBG): payload actually delivered per
    # unit of granted spectrum — the core "efficiency of use" metric.
    df["goodput_b_per_rbg"] = df["rx_bytes_total"] / df["allocated_rbg_total"].clip(lower=1)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    df = derive_metrics(collect(args.eval_root))
    n_runs = df.groupby(["method", "scenario"])["seed"].nunique().min()
    print(f"Runs collected: {len(df) // 3} (min seeds per cell: {n_runs})\n")

    # ----- per-slice aggregate (mean over seeds) -----
    per_slice = (df.groupby(["scenario", "method", "slice"])
                 [["rbg_share_pct", "rsh_real_pct_mean", "unused_budget_pct_mean",
                   "offered_load_satisfaction_pct", "goodput_b_per_rbg",
                   "pdr_pct", "delay_ms_mean", "sla_satisfaction_pct"]]
                 .mean().reset_index())
    out_slice = args.out_dir / "closedloop_radio_efficiency_per_slice.csv"
    per_slice.to_csv(out_slice, index=False)

    # ----- cell-level aggregate: one efficiency line per method x scenario -----
    cell = (df.groupby(["scenario", "method", "seed"])
            .apply(lambda g: pd.Series({
                "served_mbps_total": g["throughput_mbps_mean"].sum(),
                "goodput_b_per_rbg_cell": g["rx_bytes_total"].sum() / max(g["allocated_rbg_total"].sum(), 1),
                "unused_budget_pct_mean": g["unused_budget_pct_mean"].mean(),
                "offered_satisfaction_min": g["offered_load_satisfaction_pct"].min(),
                "offered_satisfaction_mean": g["offered_load_satisfaction_pct"].mean(),
                "pdr_min": g["pdr_pct"].min(),
                "sla_min": g["sla_satisfaction_pct"].min(),
            }), include_groups=False)
            .reset_index())
    cell_agg = (cell.groupby(["scenario", "method"])
                [["served_mbps_total", "goodput_b_per_rbg_cell", "unused_budget_pct_mean",
                  "offered_satisfaction_min", "offered_satisfaction_mean", "pdr_min", "sla_min"]]
                .agg(["mean", "std"]))
    out_cell = args.out_dir / "closedloop_radio_efficiency_cell.csv"
    cell_agg.to_csv(out_cell)

    print(f"Saved: {out_slice}")
    print(f"Saved: {out_cell}\n")

    # ----- printed report -----
    for sc in SCENARIO_ORDER:
        sub = cell_agg.loc[sc] if sc in cell_agg.index.get_level_values(0) else None
        if sub is None:
            continue
        print("=" * 100)
        print(f"CENARIO: {sc}  (media +/- desvio sobre seeds)")
        print("=" * 100)
        hdr = (f"{'metodo':<20} {'servido Mb/s':>13} {'goodput/RBG B':>14} "
               f"{'satisf. min %':>14} {'PDR min %':>10} {'SLA min %':>10}")
        print(hdr)
        order = sub[("served_mbps_total", "mean")].sort_values(ascending=False).index
        for method in order:
            r = sub.loc[method]
            print(f"{method:<20} "
                  f"{r[('served_mbps_total','mean')]:>7.1f}±{r[('served_mbps_total','std')]:<5.1f} "
                  f"{r[('goodput_b_per_rbg_cell','mean')]:>8.1f}±{r[('goodput_b_per_rbg_cell','std')]:<5.1f} "
                  f"{r[('offered_satisfaction_min','mean')]:>8.1f}±{r[('offered_satisfaction_min','std')]:<5.1f} "
                  f"{r[('pdr_min','mean')]:>5.1f}±{r[('pdr_min','std')]:<4.1f} "
                  f"{r[('sla_min','mean')]:>5.1f}±{r[('sla_min','std')]:<4.1f}")
        print()

        # per-slice detail: where does each method put the spectrum, and does it pay off?
        det = per_slice[per_slice["scenario"] == sc]
        print(f"{'metodo':<20} {'slice':<6} {'RBG %':>6} {'uso real %':>10} "
              f"{'satisf %':>8} {'PDR %':>6} {'delay ms':>9} {'SLA %':>6}")
        for method in order:
            for sl in SLICE_ORDER:
                r = det[(det["method"] == method) & (det["slice"] == sl)]
                if r.empty:
                    continue
                r = r.iloc[0]
                print(f"{method:<20} {sl:<6} {r['rbg_share_pct']:>6.1f} {r['rsh_real_pct_mean']:>10.1f} "
                      f"{r['offered_load_satisfaction_pct']:>8.1f} {r['pdr_pct']:>6.1f} "
                      f"{r['delay_ms_mean']:>9.1f} {r['sla_satisfaction_pct']:>6.1f}")
            print()
        print()


if __name__ == "__main__":
    main()
