#!/usr/bin/env python3
"""Unified comparison: RSLAQ DDQN (paper-faithful) vs meta-heuristics vs baselines.

Produces:
  - comparison_ddqn_vs_meta_v2.csv: per-(scenario, seed, method) Path-C score
    for {ga, pso, sa, hybrid, rslaq_ddqn} + best baseline per scenario.
  - wilcoxon_ddqn_vs_meta.csv: paired Wilcoxon p-values (DDQN vs each meta +
    DDQN vs best baseline), valid for n>=5.
  - fig_ddqn_vs_meta.pdf: boxplot of scores by method per scenario.

Inputs:
  --meta-csv    paper_v2_campaign/v2_best_candidates.csv (160 rows)
  --ddqn-csv    ddqn_scored_v2.csv (from rescore_ddqn_standalone.py)
  --baseline-csv v2_opt_vs_baseline_paired.csv (optional, for context)

Usage:
    python3 examples/analyze_ddqn_vs_meta.py \
        --meta-csv paper_v2_campaign/v2_best_candidates.csv \
        --ddqn-csv results_controlled/.../ddqn_scored_v2.csv \
        --output-dir paper_v2_campaign/
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from statistics import mean, median, stdev

import numpy as np

META_METHODS = ["ga", "pso", "sa", "hybrid"]


def load_csv(path):
    if not os.path.exists(path):
        print(f"[WARN] {path} not found", file=sys.stderr)
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def wilcoxon_paired(x, y):
    """Two-sided Wilcoxon signed-rank test on paired samples.

    Returns (statistic, p_value). Uses scipy if available; falls back to
    a normal approximation with continuity correction.
    """
    diffs = [a - b for a, b in zip(x, y) if a != b]
    n = len(diffs)
    if n < 5:
        return None, None  # too few non-tied pairs
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        return float(stat), float(p)
    except ImportError:
        # Normal approximation fallback
        abs_diffs = [abs(d) for d in diffs]
        ranks = np.argsort(np.argsort(abs_diffs)) + 1
        w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
        w_minus = sum(r for r, d in zip(ranks, diffs) if d < 0)
        w = min(w_plus, w_minus)
        mu = n * (n + 1) / 4
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        if sigma == 0:
            return float(w), 1.0
        z = (w - mu) / sigma
        # two-sided p from normal
        from math import erf
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))
        return float(w), float(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-csv", required=True)
    ap.add_argument("--ddqn-csv", required=True)
    ap.add_argument("--baseline-csv", default="",
                    help="Optional v2_opt_vs_baseline_paired.csv for context")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    meta_rows = load_csv(args.meta_csv)
    ddqn_rows = load_csv(args.ddqn_csv)
    baseline_rows = load_csv(args.baseline_csv) if args.baseline_csv else []

    # Normalize: meta CSV has scenario/seed/method/score
    # ddqn CSV has scenario/seed/method/score_v2
    # Align to common schema.
    scenarios_meta = sorted(set(r["scenario"] for r in meta_rows))
    scenarios_ddqn = sorted(set(r["scenario"] for r in ddqn_rows))
    scenarios = sorted(set(scenarios_meta) & set(scenarios_ddqn))
    print(f"Scenarios in meta: {scenarios_meta}")
    print(f"Scenarios in ddqn: {scenarios_ddqn}")
    print(f"Common scenarios:  {scenarios}")

    seeds_meta = sorted(set(int(r["seed"]) for r in meta_rows))
    seeds_ddqn = sorted(set(int(r["seed"]) for r in ddqn_rows))
    print(f"Seeds in meta: {seeds_meta}")
    print(f"Seeds in ddqn: {seeds_ddqn}")

    # Build unified table
    unified = []  # rows: scenario, seed, method, score
    for r in meta_rows:
        if r["scenario"] not in scenarios:
            continue
        unified.append({
            "scenario": r["scenario"],
            "seed": int(r["seed"]),
            "method": r["method"],
            "score": float(r["score"]),
            "source": "meta",
        })
    for r in ddqn_rows:
        if r.get("status") != "ok" or r["scenario"] not in scenarios:
            continue
        unified.append({
            "scenario": r["scenario"],
            "seed": int(r["seed"]),
            "method": "rslaq_ddqn",
            "score": float(r["score_v2"]),
            "source": "ddqn",
        })

    # Write unified CSV
    unified_path = os.path.join(args.output_dir, "comparison_ddqn_vs_meta_v2.csv")
    with open(unified_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "seed", "method", "score", "source"])
        w.writeheader()
        w.writerows(unified)
    print(f"\n[OK] unified -> {unified_path} ({len(unified)} rows)")

    # Summary table: mean/std/best per method per scenario
    print("\n" + "=" * 80)
    print("SUMMARY: score Path C by method per scenario")
    print("=" * 80)
    print(f"{'scenario':<22} {'method':<12} {'n':>3} {'mean':>8} {'std':>7} "
          f"{'median':>8} {'best':>8}")
    all_methods = META_METHODS + ["rslaq_ddqn"]
    summary_path = os.path.join(args.output_dir, "summary_ddqn_vs_meta_v2.csv")
    with open(summary_path, "w", newline="") as f:
        sw = csv.writer(f)
        sw.writerow(["scenario", "method", "n", "mean", "std", "median", "best"])
        for sc in scenarios:
            for m in all_methods:
                scores = [row["score"] for row in unified
                          if row["scenario"] == sc and row["method"] == m]
                if not scores:
                    continue
                mu = mean(scores)
                sd = stdev(scores) if len(scores) > 1 else 0.0
                md = median(scores)
                best = max(scores)
                print(f"{sc:<22} {m:<12} {len(scores):>3} {mu:>8.2f} "
                      f"{sd:>7.2f} {md:>8.2f} {best:>8.2f}")
                sw.writerow([sc, m, len(scores), f"{mu:.4f}", f"{sd:.4f}",
                             f"{md:.4f}", f"{best:.4f}"])
    print(f"\n[OK] summary -> {summary_path}")

    # Paired Wilcoxon: DDQN vs each meta method (per scenario)
    print("\n" + "=" * 80)
    print("WILCOXON paired (two-sided): rslaq_ddqn vs each meta-heuristic")
    print("=" * 80)
    print(f"{'scenario':<22} {'comparison':<24} {'n':>3} {'W':>8} {'p-value':>10}")
    wilcox_path = os.path.join(args.output_dir, "wilcoxon_ddqn_vs_meta.csv")
    with open(wilcox_path, "w", newline="") as f:
        ww = csv.writer(f)
        ww.writerow(["scenario", "comparison", "n", "W_statistic", "p_value",
                     "ddqn_mean", "meta_mean", "delta_mean"])
        for sc in scenarios:
            ddqn_scores = {row["seed"]: row["score"] for row in unified
                           if row["scenario"] == sc and row["method"] == "rslaq_ddqn"}
            if len(ddqn_scores) < 5:
                print(f"{sc:<22} (DDQN n={len(ddqn_scores)} < 5, skip)")
                continue
            for m in META_METHODS:
                meta_scores = {row["seed"]: row["score"] for row in unified
                               if row["scenario"] == sc and row["method"] == m}
                common = sorted(set(ddqn_scores) & set(meta_scores))
                if len(common) < 5:
                    print(f"{sc:<22} ddqn vs {m:<16} n={len(common)} < 5, skip")
                    continue
                x = [ddqn_scores[s] for s in common]
                y = [meta_scores[s] for s in common]
                W, p = wilcoxon_paired(x, y)
                dm = mean(x)
                mm = mean(y)
                delta = dm - mm
                pstr = f"{p:.4f}" if p is not None else "n/a"
                wstr = f"{W:.1f}" if W is not None else "n/a"
                print(f"{sc:<22} ddqn vs {m:<16} {len(common):>3} "
                      f"{wstr:>8} {pstr:>10}  Δ={delta:+.2f}")
                ww.writerow([sc, f"ddqn_vs_{m}", len(common),
                             wstr, pstr, f"{dm:.4f}", f"{mm:.4f}", f"{delta:.4f}"])
    print(f"\n[OK] wilcoxon -> {wilcox_path}")

    # Optional baseline context
    if baseline_rows:
        print("\n" + "=" * 80)
        print("BASELINE context: best baseline per scenario (paired vs optimized)")
        print("=" * 80)
        for sc in scenarios:
            sub = [r for r in baseline_rows if r["scenario"] == sc]
            if not sub:
                continue
            # The "optimized" column is the best meta; for context we show it
            sub.sort(key=lambda r: -int(r.get("opt_wins", 0)))
            print(f"\n--- {sc} (top 3 baselines by opt_wins) ---")
            for r in sub[:3]:
                print(f"  {r['baseline']:<24} opt_wins={r['opt_wins']}/10 "
                      f"ties={r['ties']} Δ_med={float(r['median_delta']):.2f}")

    # Boxplot figure (optional, needs matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 5),
                                 squeeze=False)
        for idx, sc in enumerate(scenarios):
            ax = axes[0][idx]
            data = []
            labels = []
            for m in all_methods:
                scores = [row["score"] for row in unified
                          if row["scenario"] == sc and row["method"] == m]
                if scores:
                    data.append(scores)
                    labels.append(m)
            ax.boxplot(data, labels=labels, showmeans=True)
            ax.set_title(sc)
            ax.set_ylabel("Score Path C")
            ax.tick_params(axis="x", rotation=30)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5,
                       label="viability threshold")
        fig.suptitle("RSLAQ DDQN (paper-faithful) vs meta-heuristics — Path C score",
                     fontsize=12)
        fig.tight_layout()
        fig_path = os.path.join(args.output_dir, "fig_ddqn_vs_meta.pdf")
        fig.savefig(fig_path)
        fig.savefig(fig_path.replace(".pdf", ".png"), dpi=150)
        print(f"\n[OK] figure -> {fig_path}")
    except ImportError:
        print("\n[INFO] matplotlib not available, skipping figure")

    return 0


if __name__ == "__main__":
    sys.exit(main())
