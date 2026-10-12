#!/usr/bin/env python3
"""Statistical analysis of the v2 campaign (audit reformulation).

With n>=5 seeds the audit's blocking objection to v1 (n=3 -> no valid test) is
resolved: this script runs the paired non-parametric tests the audit asked for.

Per scenario it computes, over the best-score-per-seed of each method:
  - mean +/- std, median, min/max, and a bootstrap 95% CI of the mean
  - Friedman test across the 4 methods (omnibus: are they different at all?)
  - pairwise Wilcoxon signed-rank tests (which pairs differ?)
  - matched-pairs effect size (rank-biserial) for each pair

It also reports the physical KPIs of each method's best candidate (URLLC p99,
deadline-violation rate, worst-slice SLA) averaged over seeds, so conclusions
rest on radio metrics, not only the composite score.

Reads the metaheuristic search output written by run_rslaq_metaheuristics.py.

Usage:
    cd ns-o-ran-gym
    python3 examples/analyze_v2_stats.py \
        --meta-root results_controlled/heuristics_metaheuristics_v2/v2_pilot_congestion/metaheuristics
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
except ImportError:  # pragma: no cover
    sps = None

REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS = ["ga", "pso", "sa", "hybrid"]
SLICES = ("eMBB", "URLLC", "MTC")


def collect_best(meta_root: Path) -> pd.DataFrame:
    """Best (max-score) candidate per (scenario, seed, method) from sidecars."""
    rows = []
    for sidecar in meta_root.rglob("evals/*/candidate.json"):
        try:
            d = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
        if d.get("failed"):
            continue
        rows.append({
            "scenario": d.get("scenario"),
            "seed": int(d.get("seed", -1)),
            "method": d.get("method"),
            "score": float(d.get("score", float("nan"))),
            "score_version": d.get("score_version", "v1"),
            "sidecar": str(sidecar),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    idx = df.groupby(["scenario", "seed", "method"])["score"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def best_candidate_summary(sidecar_path: Path) -> dict | None:
    """Load the summary.csv KPIs for the best candidate pointed to by a sidecar."""
    eval_dir = sidecar_path.parent
    matches = list(eval_dir.rglob("summary.csv"))
    if not matches:
        return None
    df = pd.read_csv(matches[0])
    by = {r["slice"]: r for _, r in df.iterrows()}
    if not all(s in by for s in SLICES):
        return None
    return {
        "urllc_p99": float(by["URLLC"].get("delay_ms_p99", float("nan"))),
        "urllc_deadline_viol": float(by["URLLC"].get("deadline_violation_pct", float("nan"))),
        "min_sla": float(min(by[s].get("sla_satisfaction_pct", 0.0) for s in SLICES)),
        "mtc_pdr": float(by["MTC"].get("pdr_pct", float("nan"))),
    }


def bootstrap_ci(x: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 12345) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial effect size for Wilcoxon signed-rank."""
    d = a - b
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = sps.rankdata(np.abs(d))
    r_pos = ranks[d > 0].sum()
    r_neg = ranks[d < 0].sum()
    total = r_pos + r_neg
    return float((r_pos - r_neg) / total) if total > 0 else 0.0


def analyze_scenario(sc: str, best: pd.DataFrame, kpi: pd.DataFrame) -> None:
    sub = best[best["scenario"] == sc]
    # pivot: rows=seed, cols=method, values=score (aligned/paired by seed)
    piv = sub.pivot_table(index="seed", columns="method", values="score")
    piv = piv.dropna(axis=0, how="any")  # keep only fully-paired seeds
    methods = [m for m in METHODS if m in piv.columns]
    n = len(piv)

    print("=" * 78)
    print(f"CENARIO: {sc}   (n={n} seeds pareadas)")
    print("=" * 78)
    if n == 0:
        print("  sem seeds completas ainda\n")
        return

    print(f"{'metodo':<8} {'media':>8} {'std':>7} {'mediana':>8} {'IC95% media':>20}")
    for m in methods:
        x = piv[m].values
        lo, hi = bootstrap_ci(x)
        print(f"{m:<8} {x.mean():>8.2f} {x.std(ddof=1) if n>1 else 0:>7.2f} "
              f"{np.median(x):>8.2f}   [{lo:>7.2f}, {hi:>7.2f}]")

    if sps is None:
        print("\n  scipy ausente -> testes estatisticos pulados\n")
        return

    if len(methods) >= 3 and n >= 2:
        try:
            stat, p = sps.friedmanchisquare(*[piv[m].values for m in methods])
            verdict = "DIFERENCA detectada" if p < 0.05 else "sem diferenca significativa"
            print(f"\nFriedman (omnibus): chi2={stat:.3f} p={p:.4f} -> {verdict}")
        except ValueError as exc:
            print(f"\nFriedman indisponivel: {exc}")

    if n >= 5:  # Wilcoxon pareado exige n>=5 amostras nao-nulas
        print("\nWilcoxon pareado + tamanho de efeito (rank-biserial):")
        for a, b in combinations(methods, 2):
            xa, xb = piv[a].values, piv[b].values
            try:
                w, p = sps.wilcoxon(xa, xb)
                es = rank_biserial(xa, xb)
                sig = "*" if p < 0.05 else " "
                print(f"  {a:>6} vs {b:<6}: W={w:>6.1f} p={p:.4f}{sig}  effect={es:+.2f}  "
                      f"(mean {xa.mean():.2f} vs {xb.mean():.2f})")
            except ValueError as exc:
                print(f"  {a:>6} vs {b:<6}: indisponivel ({exc})")
    else:
        print(f"\n  n={n} < 5: Wilcoxon pareado ainda sem potencia (aguarde mais seeds)")

    # Physical KPIs of the best candidate per method (mean over seeds)
    ksub = kpi[kpi["scenario"] == sc]
    if not ksub.empty:
        print("\nKPIs fisicos do melhor candidato (media sobre seeds):")
        print(f"{'metodo':<8} {'URLLC p99 ms':>13} {'deadline viol %':>16} "
              f"{'MTC PDR %':>10} {'minSLA %':>9}")
        for m in methods:
            g = ksub[ksub["method"] == m]
            if g.empty:
                continue
            print(f"{m:<8} {g['urllc_p99'].mean():>13.1f} "
                  f"{g['urllc_deadline_viol'].mean():>16.2f} "
                  f"{g['mtc_pdr'].mean():>10.1f} {g['min_sla'].mean():>9.1f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-root", type=Path, required=True,
                    help="…/metaheuristics dir of the v2 campaign")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write CSVs (default: alongside meta-root)")
    args = ap.parse_args()

    best = collect_best(args.meta_root)
    if best.empty:
        raise SystemExit(f"No candidate.json sidecars under {args.meta_root}")

    versions = set(best["score_version"].unique())
    print(f"score_version(s) presentes: {versions}")
    if len(versions) > 1:
        print("  AVISO: cache com versoes de score mistas — nao compare diretamente!")

    # physical KPIs for each best candidate
    krows = []
    for _, r in best.iterrows():
        s = best_candidate_summary(Path(r["sidecar"]))
        if s:
            s.update({"scenario": r["scenario"], "seed": r["seed"], "method": r["method"]})
            krows.append(s)
    kpi = pd.DataFrame(krows)

    out_dir = args.out_dir or args.meta_root.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    best.to_csv(out_dir / "v2_best_per_seed.csv", index=False)
    if not kpi.empty:
        kpi.to_csv(out_dir / "v2_best_kpis.csv", index=False)
    print(f"Saved: {out_dir / 'v2_best_per_seed.csv'}\n")

    for sc in sorted(best["scenario"].unique()):
        analyze_scenario(sc, best, kpi)


if __name__ == "__main__":
    main()
