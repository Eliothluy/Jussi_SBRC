#!/usr/bin/env python3
"""Camera-ready (WPMC 2026) consolidation from sidecar checkpoints.

Source of truth: candidate.json sidecars written by
run_rslaq_metaheuristics.py (with the camera-ready instrumentation:
elapsed_s / cached / score_version). Covers the stopped campaign:
  - low_traffic + normal: COMPLETE (meta GA/PSO/SA/hybrid + grid, seeds 4-8)
  - congestion: GA complete on seeds 4-8, PSO partial, grid ~49%, SA/hybrid
    not started (simulations stopped for the submission deadline)
  - baselines RR/PF/BCQI x 3 scenarios x seeds 4-8 (summary.csv)

Produces (in results_controlled/v1_camera_ready/consolidated/):
  - best_by_seed.csv        : best J(w) per (scenario, seed, method)
  - proximity_to_grid.csv   : Q1 — meta best vs grid reference per seed
  - search_cost.csv         : Q3 — n_evals, total time, mean time, time-to-best
  - evals_to_reference.csv  : Q2 — evaluations until 95%/99% of grid reference
  - baseline_scores.csv     : baselines scored with the same v1 objective
  - summary_tables.tex      : ready-to-paste LaTeX tables

Rules honored: J(w) = score_summary_rows (v1, frozen); wall-clock only from
cached=False sidecars; grid treated as a quality/cost REFERENCE (no
convergence metrics); descriptive statistics (n=5, no inferential tests).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsoran.scoring import score_summary_rows  # noqa: E402  (v1, frozen)

REPO = Path(__file__).resolve().parents[2]
CR = REPO / "ns-o-ran-gym" / "results_controlled" / "v1_camera_ready"
OUT = CR / "consolidated"
OUT.mkdir(parents=True, exist_ok=True)

META_ROOT = CR / "metaheuristics"
GRID_ROOT = CR / "metaheuristics_grid"
BASE_ROOT = CR / "baselines_ns3" / "results_rslaq_network_only"

SCENARIOS = ["low_traffic", "normal", "congestion"]
SEEDS = [4, 5, 6, 7, 8]
META_METHODS = ["ga", "pso", "sa", "hybrid"]
GRID_BUDGET = 231


def load_sidecars(root: Path):
    out = []
    for p in root.rglob("candidate.json"):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if d.get("score_version") != "v1":
            continue
        out.append(d)
    return out


def w_str(w):
    if isinstance(w, dict):
        return f"[{w['eMBB']:.4f},{w['URLLC']:.4f},{w['MTC']:.4f}]"
    return "[" + ",".join(f"{float(x):.4f}" for x in w) + "]"


def main() -> None:
    meta = load_sidecars(META_ROOT)
    grid = load_sidecars(GRID_ROOT)
    print(f"sidecars: meta={len(meta)} grid={len(grid)}")

    # ------------------------------------------------------------------
    # best per (scenario, seed, method) + search cost per (scenario, method)
    # ------------------------------------------------------------------
    best_rows, cost_rows, eval_hist = [], [], {}
    for sc in SCENARIOS:
        for seed in SEEDS:
            for group, root, methods in (("meta", META_ROOT, META_METHODS),
                                         ("grid", GRID_ROOT, ["grid"])):
                for m in methods:
                    evs = [d for d in (meta if group == "meta" else grid)
                           if d["scenario"] == sc and d["seed"] == seed
                           and d["method"] == m]
                    if not evs:
                        continue
                    valid = [d for d in evs if not d.get("failed")]
                    n_total, n_failed = len(evs), len(evs) - len(valid)
                    # best in EVALUATION ORDER (search trajectory)
                    valid.sort(key=lambda d: d["evaluation_id"])
                    best = max(valid, key=lambda d: d["score"]) if valid else None
                    if best:
                        best_rows.append({
                            "scenario": sc, "seed": seed, "method": m,
                            "score": round(best["score"], 4),
                            "weights": w_str(best["weights"]),
                            "eval_of_best": best["evaluation_id"],
                            "n_evals_total": n_total, "n_failed": n_failed,
                            "complete": (m == "grid" and n_total == GRID_BUDGET)
                                        or (m != "grid" and n_total >= 72 * (2 if False else 1)),
                        })
                    ex = [d for d in evs if not d.get("cached")]
                    tot = sum(d.get("elapsed_s", 0.0) for d in ex)
                    # time-to-best: cumulative wall-clock up to best eval
                    t_best = None
                    if best:
                        acc = 0.0
                        for d in sorted(evs, key=lambda d: d["evaluation_id"]):
                            if not d.get("cached"):
                                acc += d.get("elapsed_s", 0.0)
                            if d["evaluation_id"] == best["evaluation_id"]:
                                t_best = acc
                                break
                    cost_rows.append({
                        "scenario": sc, "method": m, "seed": seed,
                        "n_evals": n_total, "n_failed": n_failed,
                        "total_h": round(tot / 3600, 3),
                        "mean_s": round(tot / max(len(ex), 1), 1),
                        "time_to_best_h": round(t_best / 3600, 3) if t_best else "",
                    })
                    eval_hist[(sc, seed, m)] = valid

    with open(OUT / "best_by_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
        w.writeheader(); w.writerows(best_rows)

    with open(OUT / "search_cost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cost_rows[0].keys()))
        w.writeheader(); w.writerows(cost_rows)

    # ------------------------------------------------------------------
    # Q1: proximity to grid reference (grid COMPLETE scenarios only)
    # ------------------------------------------------------------------
    prox_rows = []
    for sc in SCENARIOS:
        grid_by_seed = {s: [d for d in grid if d["scenario"] == sc
                            and d["seed"] == s and not d.get("failed")]
                        for s in SEEDS}
        grid_complete = all(len(v) == GRID_BUDGET for v in grid_by_seed.values())
        if not grid_complete:
            print(f"[Q1] {sc}: grid incomplete "
                  f"({[len(grid_by_seed[s]) for s in SEEDS]}) — skipped")
            continue
        for seed in SEEDS:
            gbest = max(grid_by_seed[seed], key=lambda d: d["score"])
            row = {"scenario": sc, "seed": seed,
                   "grid_best": round(gbest["score"], 4),
                   "grid_weights": w_str(gbest["weights"])}
            for m in META_METHODS:
                mb = [r for r in best_rows if r["scenario"] == sc
                      and r["seed"] == seed and r["method"] == m]
                if mb:
                    r = mb[0]
                    row[f"{m}_best"] = r["score"]
                    row[f"{m}_pct_of_grid"] = round(
                        100 * r["score"] / gbest["score"], 2) if gbest["score"] else ""
            prox_rows.append(row)
    if prox_rows:
        fields = list(prox_rows[0].keys())
        with open(OUT / "proximity_to_grid.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(prox_rows)

    # ------------------------------------------------------------------
    # Q2: evaluations until 95%/99% of the grid reference (complete scen.)
    # ------------------------------------------------------------------
    q2_rows = []
    for sc in SCENARIOS:
        if not any(r["scenario"] == sc for r in prox_rows):
            continue
        for seed in SEEDS:
            gref = next(r["grid_best"] for r in prox_rows
                        if r["scenario"] == sc and r["seed"] == seed)
            for m in META_METHODS:
                hist = eval_hist.get((sc, seed, m), [])
                if not hist:
                    continue
                hist = sorted(hist, key=lambda d: d["evaluation_id"])
                cum, hit95, hit99 = -1e9, None, None
                for d in hist:
                    cum = max(cum, d["score"])
                    if hit95 is None and cum >= 0.95 * gref:
                        hit95 = d["evaluation_id"]
                    if hit99 is None and cum >= 0.99 * gref:
                        hit99 = d["evaluation_id"]
                q2_rows.append({"scenario": sc, "seed": seed, "method": m,
                                "grid_ref": round(gref, 3),
                                "evals_to_95pct": hit95 or "",
                                "evals_to_99pct": hit99 or ""})
    if q2_rows:
        with open(OUT / "evals_to_reference.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(q2_rows[0].keys()))
            w.writeheader(); w.writerows(q2_rows)

    # ------------------------------------------------------------------
    # Baselines (same v1 objective) — only completed summary.csv
    # ------------------------------------------------------------------
    base_rows = []
    for sc in SCENARIOS:
        for mode in ("pure_rr", "pure_pf", "pure_bcqi"):
            for seed in SEEDS:
                sd = BASE_ROOT / f"scenario={sc}" / f"mode={mode}" / f"seed={seed}_run=1"
                sm = sd / "summary.csv"
                if not sm.exists():
                    continue
                rows = list(csv.DictReader(open(sm)))
                base_rows.append({"scenario": sc, "baseline": mode, "seed": seed,
                                  "score": round(score_summary_rows(rows), 4)})
    if base_rows:
        with open(OUT / "baseline_scores.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["scenario", "baseline", "seed", "score"])
            w.writeheader(); w.writerows(base_rows)

    # ------------------------------------------------------------------
    # Descriptive aggregates + LaTeX tables
    # ------------------------------------------------------------------
    def agg(rows, key_m, key_v):
        out = {}
        for sc in SCENARIOS:
            for m in sorted({r[key_m] for r in rows if r["scenario"] == sc}):
                vals = [r[key_v] for r in rows
                        if r["scenario"] == sc and r[key_m] == m]
                if len(vals) >= 2:
                    out[(sc, m)] = (mean(vals), stdev(vals), len(vals))
        return out

    best_meta_agg = agg(best_rows, "method", "score")
    tex = []
    tex.append("% == T1: best J(w) mean±std on INDEPENDENT seeds 4-8 ==\n")
    tex.append("% Complete scenarios: low_traffic, normal; congestion: GA only\n")
    tex.append("\\begin{table}[t]\\centering\\caption{Best objective value $J(w)$ "
               "per method on five independent seeds (4--8); "
               "mean $\\pm$ std. Grid = uniform grid-search reference "
               "($\\Delta w=0.05$, 231 points).}\\label{tab:independent}\\small\n")
    tex.append("\\begin{tabular}{l" + "c" * 6 + "}\n\\toprule\n")
    tex.append("Scenario & GA & PSO & SA & Hybrid & Grid ref. & Grid evals \\\\\n\\midrule\n")
    for sc in SCENARIOS:
        cells = []
        for m in META_METHODS:
            if (sc, m) in best_meta_agg:
                mu, sd_, n = best_meta_agg[(sc, m)]
                cells.append(f"{mu:.2f}$\\pm${sd_:.2f}")
            else:
                cells.append("--")
        gb = [r for r in prox_rows if r["scenario"] == sc]
        if gb:
            gmean = mean(r["grid_best"] for r in gb)
            cells.append(f"{gmean:.2f}$\\pm${stdev([r['grid_best'] for r in gb]):.2f}")
            cells.append("231")
        else:
            cells += ["--", "--"]
        tex.append(f"{sc.replace('_', ' ')} & " + " & ".join(cells) + " \\\\\n")
    tex.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    tex.append("% == T2: proximity to grid reference (complete scenarios) ==\n")
    if prox_rows:
        tex.append("\\begin{table}[t]\\centering\\caption{Proximity of the "
                   "meta-heuristic best solution to the grid-search reference "
                   "(\\% of $J_{\\mathrm{grid}}$), per seed.}\\label{tab:prox}\\small\n")
        tex.append("\\begin{tabular}{llccccc}\n\\toprule\n"
                   "Scenario & Seed & GA & PSO & SA & Hybrid \\\\\n\\midrule\n")
        for r in prox_rows:
            cells = [f"{r.get(f'{m}_pct_of_grid','--'):.2f}"
                     if r.get(f"{m}_pct_of_grid") not in (None, "") else "--"
                     for m in META_METHODS]
            tex.append(f"{r['scenario'].replace('_',' ')} & {r['seed']} & "
                       + " & ".join(cells) + " \\\\\n")
        tex.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    tex.append("% == T3: search cost (wall-clock, cached=False; constant 10-proc policy) ==\n")
    cost_agg = {}
    for sc in SCENARIOS:
        for m in META_METHODS + ["grid"]:
            rows = [r for r in cost_rows if r["scenario"] == sc and r["method"] == m]
            if not rows:
                continue
            th = [r["total_h"] for r in rows]
            me = [r["mean_s"] for r in rows]
            ne = [r["n_evals"] for r in rows]
            cost_agg[(sc, m)] = (sum(th), sum(ne), mean(me))
    tex.append("\\begin{table}[t]\\centering\\caption{Search cost per scenario "
               "(5 seeds; wall-clock under a constant 10-process policy; "
               "grid = full sweep).}\\label{tab:cost}\\small\n")
    tex.append("\\begin{tabular}{lrrrr}\n\\toprule\n"
               "Scenario & Method & Evals & Total (h) & Mean (s) \\\\\n\\midrule\n")
    for sc in SCENARIOS:
        for m in META_METHODS + ["grid"]:
            if (sc, m) in cost_agg:
                th, ne, me = cost_agg[(sc, m)]
                tex.append(f"{sc.replace('_',' ')} & {m} & {ne} & {th:.1f} & {me:.0f} \\\\\n")
        if any((sc, m) in cost_agg for m in META_METHODS + ["grid"]):
            tex.append("\\midrule\n" if sc != "congestion" else "")
    tex.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    with open(OUT / "summary_tables.tex", "w") as f:
        f.writelines(tex)

    print(f"\noutputs -> {OUT}")
    print(f"  best_by_seed.csv      {len(best_rows)} rows")
    print(f"  search_cost.csv       {len(cost_rows)} rows")
    print(f"  proximity_to_grid.csv {len(prox_rows)} rows (complete scen. only)")
    print(f"  evals_to_reference.csv {len(q2_rows)} rows")
    print(f"  baseline_scores.csv   {len(base_rows)} rows")
    print(f"  summary_tables.tex    3 tables")


if __name__ == "__main__":
    main()
