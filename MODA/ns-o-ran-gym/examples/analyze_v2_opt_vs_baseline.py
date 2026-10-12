#!/usr/bin/env python3
"""Optimization vs non-optimization: full per-slice allocation comparison (v2).

The paper's central claim under congestion is NOT that a slice's SLA is met
(physically impossible when the cell is overloaded) but that metaheuristic
optimization ALLOCATES RESOURCES WELL — protecting the latency-sensitive slice
(URLLC) and avoiding starvation of MTC — better than non-optimized baselines.
A scalar score hides this; this script reports the WHOLE allocation per slice.

Compares, per (scenario), scored uniformly with Path C (score_summary_rows_v2):
  - OPTIMIZED: the best metaheuristic candidate per (scenario, seed) — the max
    over GA/PSO/SA/hybrid of each seed's search.
  - BASELINES (no offline optimization): pure schedulers (RR/PF/BCQI), fixed
    weights (psta_equal, slice_weighted_*), adaptive heuristics (AQPS, greedy).

For each method it reports, per slice (mean over seeds): RBG share, offered-load
satisfaction, PDR, delay p99, deadline-violation rate; plus cell-level Path C
score, feasibility rate (URLLC p99<=10ms with a live URLLC), and worst-slice
composite SLA. The point of comparison is the ALLOCATION PROFILE, not one number.

Usage:
    cd ns-o-ran-gym
    python3 examples/analyze_v2_opt_vs_baseline.py \
        --meta-root  results_controlled/heuristics_metaheuristics_v2/v2_pilot_congestion/metaheuristics \
        --baseline-root results_controlled/heuristics_metaheuristics_v2/v2_baselines_congestion/heuristics_ns3
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nsoran.scoring import read_summary, score_summary_rows_v2  # noqa: E402

SLICES = ["eMBB", "URLLC", "MTC"]
BASELINE_LABELS = {
    "pure_rr": "Pure RR", "pure_pf": "Pure PF", "pure_bcqi": "Pure BCQI",
    "psta_equal": "P-STA equal", "slice_weighted_pf": "Weighted PF",
    "slice_weighted_rr": "Weighted RR", "slice_weighted_bcqi": "Weighted BCQI",
    "slice_aqps": "AQPS", "slice_demand_greedy": "Demand-greedy",
    "slice_sla_greedy": "SLA-greedy", "slice_least_waste": "Least-waste",
    "slice_qos_mixed": "QoS-mixed", "slice_random_vine": "Random-vine",
    "slice_meta_risk_elastic": "Meta-risk-elastic",
}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def per_slice_profile(summary_path: Path) -> dict | None:
    """Return {slice: {rbg_pct, satisf, pdr, p99, dv}} + cell score/feasible."""
    rows = {r["slice"]: r for r in read_summary(summary_path)}
    if not all(s in rows for s in SLICES):
        return None
    rbg_tot = sum(_f(rows[s].get("allocated_rbg_total")) for s in SLICES) or 1.0
    prof = {}
    for s in SLICES:
        r = rows[s]
        prof[s] = {
            "rbg_pct": 100.0 * _f(r.get("allocated_rbg_total")) / rbg_tot,
            "satisf": _f(r.get("offered_load_satisfaction_pct")),
            "pdr": _f(r.get("pdr_pct")),
            "p99": _f(r.get("delay_ms_p99")),
            "dv": _f(r.get("deadline_violation_pct")),
        }
    score = score_summary_rows_v2(read_summary(summary_path))
    return {"slices": prof, "score": score, "feasible": score >= 0.0}


def collect_optimized(meta_root: Path) -> dict:
    """Best metaheuristic candidate per (scenario, seed), re-scored with Path C fix."""
    best = {}  # (scenario, seed) -> (score, summary_path)
    for f in glob.glob(str(meta_root / "**/candidate.json"), recursive=True):
        d = json.loads(Path(f).read_text())
        if d.get("failed"):
            continue
        summ = glob.glob(os.path.join(os.path.dirname(f), "**/summary.csv"), recursive=True)
        if not summ:
            continue
        score = score_summary_rows_v2(read_summary(Path(summ[0])))  # corrected score
        key = (d["scenario"], d["seed"])
        if key not in best or score > best[key][0]:
            best[key] = (score, Path(summ[0]))
    profiles = defaultdict(dict)  # scenario -> seed -> profile
    for (sc, seed), (score, path) in best.items():
        p = per_slice_profile(path)
        if p:
            profiles[sc][int(seed)] = p
    return profiles


def collect_baselines(baseline_root: Path) -> dict:
    """Per (scenario, mode) profiles keyed by seed (seeds pair against optimization)."""
    profiles = defaultdict(lambda: defaultdict(dict))  # scenario -> mode -> seed -> profile
    for summ in glob.glob(str(baseline_root / "**/summary.csv"), recursive=True):
        parts = Path(summ).parts
        sc = next((p.split("=")[1] for p in parts if p.startswith("scenario=")), None)
        mode = next((p.split("=")[1] for p in parts if p.startswith("mode=")), None)
        seed = next((int(p.split("=")[1].split("_")[0]) for p in parts
                     if p.startswith("seed=")), None)
        if not sc or not mode or seed is None:
            continue
        p = per_slice_profile(Path(summ))
        if p:
            profiles[sc][mode][seed] = p
    return profiles


def _mean(profs, slice_, key):
    vals = [p["slices"][slice_][key] for p in profs if not _is_nan(p["slices"][slice_][key])]
    return st.mean(vals) if vals else float("nan")


def _is_nan(x):
    return x != x


def paired_comparison(opt_by_seed: dict, base_by_seed: dict) -> list:
    """Per-seed paired comparison of optimization against one baseline mode.

    Comparing the median score *among feasible runs* is a selection artifact:
    each method is feasible on a different subset of seeds, and a baseline that
    only survives the easy seeds gets an inflated conditional median. Pairing on
    the seed removes that — both methods face the same channel realization, so
    the difference is attributable to the allocation policy alone.

    Returns (wins, ties, losses, median_delta) over the seeds both methods ran.
    The win/tie/loss counts are a sign test: 10/10 is p~0.002 two-sided, 9/10
    p~0.021. With n=10 that is the strongest claim this campaign supports —
    report it instead of unpaired medians.
    """
    common = sorted(set(opt_by_seed) & set(base_by_seed))
    if len(common) < 3:
        return None
    deltas = [opt_by_seed[s] - base_by_seed[s] for s in common]
    wins = sum(1 for d in deltas if d > 1e-6)
    losses = sum(1 for d in deltas if d < -1e-6)
    return wins, len(deltas) - wins - losses, losses, st.median(deltas), len(common)


def print_paired_block(scenario: str, opt_by_seed: dict, base_modes: dict) -> None:
    feasible = sum(1 for v in opt_by_seed.values() if v >= 0.0)
    print(f"\n  PAREADO POR SEED — otimização viável em {feasible}/{len(opt_by_seed)} seeds")
    print(f"    {'baseline':<24}{'viáv.':>7}{'V-E-D':>10}{'Δ mediano':>13}")
    rows = []
    for mode, by_seed in base_modes.items():
        res = paired_comparison(opt_by_seed, by_seed)
        if res is None:
            continue
        w, t, l, med, n = res
        feas_b = sum(1 for v in by_seed.values() if v >= 0.0)
        rows.append((med, mode, feas_b, len(by_seed), w, t, l))
    for med, mode, fb, nb, w, t, l in sorted(rows, key=lambda r: -r[0]):
        label = BASELINE_LABELS.get(mode, mode)
        print(f"    {label:<24}{fb:>3}/{nb:<3}{w}-{t}-{l:<6}{med:>+13.1f}")


def print_method_block(name: str, profs: list) -> None:
    if not profs:
        return
    feas = sum(p["feasible"] for p in profs)
    scores = [p["score"] for p in profs if p["feasible"]]
    med = f"{st.median(scores):.1f}" if scores else "—"
    print(f"\n  {name}  | viáveis {feas}/{len(profs)} | score viável mediano {med}")
    print(f"    {'slice':<7}{'RBG%':>7}{'satisf%':>9}{'PDR%':>7}{'p99 ms':>9}{'dviol%':>8}")
    for s in SLICES:
        print(f"    {s:<7}{_mean(profs,s,'rbg_pct'):>7.1f}{_mean(profs,s,'satisf'):>9.1f}"
              f"{_mean(profs,s,'pdr'):>7.1f}{_mean(profs,s,'p99'):>9.1f}{_mean(profs,s,'dv'):>8.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-root", type=Path, required=True)
    ap.add_argument("--baseline-root", type=Path, default=None,
                    help="…/heuristics_ns3 (results_rslaq_network_only) of the v2 baseline run")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    opt = collect_optimized(args.meta_root)
    base = collect_baselines(args.baseline_root) if args.baseline_root else {}

    scenarios = sorted(set(opt) | set(base))
    for sc in scenarios:
        print("=" * 74)
        print(f"CENÁRIO: {sc}  — otimização vs baselines (perfil por-slice, média sobre seeds)")
        print("=" * 74)
        opt_profs = list(opt.get(sc, {}).values())
        print_method_block("OTIMIZAÇÃO (best metaheur.)", opt_profs)
        if sc in base:
            for mode in sorted(base[sc]):
                print_method_block(f"baseline: {BASELINE_LABELS.get(mode, mode)}",
                                   list(base[sc][mode].values()))
            # The headline test: same seed, same channel realization, both methods.
            print_paired_block(
                sc,
                {s: p["score"] for s, p in opt.get(sc, {}).items()},
                {m: {s: p["score"] for s, p in by_seed.items()}
                 for m, by_seed in base[sc].items()},
            )
        elif not base:
            print("\n  (baselines ainda não disponíveis — rode a Fase 1 v2 e re-execute)")
        print()

    # Persist tidy CSV for the paper figures
    out_dir = args.out_dir or args.meta_root.parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    def dump(method_kind, method_name, sc, profs):
        for s in SLICES:
            rows.append({"scenario": sc, "kind": method_kind, "method": method_name, "slice": s,
                         "rbg_pct": _mean(profs, s, "rbg_pct"), "satisf": _mean(profs, s, "satisf"),
                         "pdr": _mean(profs, s, "pdr"), "p99_ms": _mean(profs, s, "p99"),
                         "deadline_viol_pct": _mean(profs, s, "dv"),
                         "feasible_rate": sum(p["feasible"] for p in profs) / max(len(profs), 1)})
    paired_rows = []
    for sc in scenarios:
        if opt.get(sc):
            dump("optimized", "best_metaheuristic", sc, list(opt[sc].values()))
        for mode in base.get(sc, {}):
            dump("baseline", mode, sc, list(base[sc][mode].values()))
            res = paired_comparison(
                {s: p["score"] for s, p in opt.get(sc, {}).items()},
                {s: p["score"] for s, p in base[sc][mode].items()},
            )
            if res:
                w, t, l, med, n = res
                paired_rows.append({"scenario": sc, "baseline": mode, "n_seeds": n,
                                    "opt_wins": w, "ties": t, "opt_losses": l,
                                    "median_delta": med})
    if paired_rows:
        import csv as _csv
        out = out_dir / "v2_opt_vs_baseline_paired.csv"
        with out.open("w", newline="") as fh:
            w_ = _csv.DictWriter(fh, fieldnames=list(paired_rows[0].keys()))
            w_.writeheader(); w_.writerows(paired_rows)
        print(f"Saved: {out}")
    if rows:
        import csv as _csv
        out = out_dir / "v2_opt_vs_baseline_per_slice.csv"
        with out.open("w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
