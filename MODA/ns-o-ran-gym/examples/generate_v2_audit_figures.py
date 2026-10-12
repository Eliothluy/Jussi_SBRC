#!/usr/bin/env python3
"""IEEE figures for the v2 campaign, answering the audit's 'graficos necessarios'.

Reads the metaheuristic search output (best candidate per scenario/seed/method
plus its summary.csv) and produces publication figures + tidy CSVs under
paper_v2_campaign/. Designed to run on partial data (pilot = congestion only);
it plots whatever scenarios/seeds are present.

Figures (audit §C, feasible from the v2 summary columns):
  fig_v2_1  composite score v2 per method (bar +/- std over seeds)
  fig_v2_2  URLLC latency p99 / p99.9 per method (grouped bar +/- std)
  fig_v2_3  URLLC deadline-violation rate per method
  fig_v2_4  worst-slice SLA satisfaction per method
  fig_v2_5  physical Pareto: total throughput vs URLLC p99 (per method)
  fig_v2_6  per-seed score dispersion (strip) — shows seed variability directly
  fig_v2_7  weight distribution per slice across seeds (violin) — weight stability
  fig_v2_8  URLLC delay CCDF across seeds (from per-seed percentile anchors)

Usage:
    cd ns-o-ran-gym
    python3 examples/generate_v2_audit_figures.py \
        --meta-root results_controlled/heuristics_metaheuristics_v2/v2_pilot_congestion/metaheuristics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS = ["ga", "pso", "sa", "hybrid"]
METHOD_LABEL = {"ga": "GA", "pso": "PSO", "sa": "SA", "hybrid": "Hybrid"}
SLICES = ("eMBB", "URLLC", "MTC")
COLOR = {"ga": "#0072B2", "pso": "#009E73", "sa": "#D55E00", "hybrid": "#CC79A7"}
SLICE_COLOR = {"eMBB": "#0072B2", "URLLC": "#E69F00", "MTC": "#009E73"}


def set_ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
        "axes.axisbelow": True, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def collect(meta_root: Path) -> pd.DataFrame:
    """One row per (scenario, seed, method) best candidate, with score+KPIs+weights."""
    rows = []
    for sidecar in meta_root.rglob("evals/*/candidate.json"):
        try:
            d = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
        if d.get("failed"):
            continue
        rows.append({"scenario": d.get("scenario"), "seed": int(d.get("seed", -1)),
                     "method": d.get("method"), "score": float(d.get("score", float("nan"))),
                     "weights": d.get("weights"), "sidecar": sidecar})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    best = df.loc[df.groupby(["scenario", "seed", "method"])["score"].idxmax()].reset_index(drop=True)

    # attach KPIs + weights from the best candidate's summary.csv
    recs = []
    for _, r in best.iterrows():
        rec = {"scenario": r["scenario"], "seed": r["seed"], "method": r["method"],
               "score": r["score"]}
        w = r["weights"] or [np.nan] * 3
        rec.update({"w_embb": w[0], "w_urllc": w[1], "w_mtc": w[2]})
        matches = list(Path(r["sidecar"]).parent.rglob("summary.csv"))
        if matches:
            sm = pd.read_csv(matches[0])
            by = {x["slice"]: x for _, x in sm.iterrows()}
            if all(s in by for s in SLICES):
                rec["thr_total"] = sum(float(by[s].get("throughput_mbps_mean", 0)) for s in SLICES)
                rec["urllc_p99"] = float(by["URLLC"].get("delay_ms_p99", np.nan))
                rec["urllc_p999"] = float(by["URLLC"].get("delay_ms_p999", np.nan))
                rec["urllc_p95"] = float(by["URLLC"].get("delay_ms_p95", np.nan))
                rec["urllc_mean"] = float(by["URLLC"].get("delay_ms_mean", np.nan))
                rec["urllc_deadline_viol"] = float(by["URLLC"].get("deadline_violation_pct", np.nan))
                rec["min_sla"] = min(float(by[s].get("sla_satisfaction_pct", 0)) for s in SLICES)
        recs.append(rec)
    return pd.DataFrame(recs)


def _bar_mean_std(ax, df, sc, value, ylabel):
    sub = df[df["scenario"] == sc]
    methods = [m for m in METHODS if m in sub["method"].unique()]
    means = [sub[sub.method == m][value].mean() for m in methods]
    stds = [sub[sub.method == m][value].std(ddof=1) for m in methods]
    x = np.arange(len(methods))
    ax.bar(x, means, 0.6, yerr=stds, color=[COLOR[m] for m in methods],
           edgecolor="black", linewidth=0.4, error_kw={"elinewidth": 0.6, "capsize": 2})
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods])
    ax.set_ylabel(ylabel)


def fig_scores(df, sc, out):
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    _bar_mean_std(ax, df, sc, "score", "v2 composite score")
    ax.set_title(sc)
    fig.savefig(out / f"fig_v2_1_score_{sc}.pdf")
    fig.savefig(out / f"fig_v2_1_score_{sc}.png")
    plt.close(fig)


def fig_urllc_percentiles(df, sc, out):
    set_ieee_style()
    sub = df[df["scenario"] == sc]
    methods = [m for m in METHODS if m in sub["method"].unique()]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(methods)); w = 0.4
    for i, (col, lab) in enumerate([("urllc_p99", "p99"), ("urllc_p999", "p99.9")]):
        means = [sub[sub.method == m][col].mean() for m in methods]
        stds = [sub[sub.method == m][col].std(ddof=1) for m in methods]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, label=lab,
               edgecolor="black", linewidth=0.4, error_kw={"elinewidth": 0.6, "capsize": 2})
    ax.axhline(10.0, color="red", ls="--", lw=0.8, label="10 ms alvo")
    ax.set_xticks(x); ax.set_xticklabels([METHOD_LABEL[m] for m in methods])
    ax.set_ylabel("URLLC delay (ms)"); ax.set_title(sc); ax.legend(frameon=False)
    fig.savefig(out / f"fig_v2_2_urllc_percentiles_{sc}.pdf")
    fig.savefig(out / f"fig_v2_2_urllc_percentiles_{sc}.png")
    plt.close(fig)


def fig_deadline(df, sc, out):
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    _bar_mean_std(ax, df, sc, "urllc_deadline_viol", "URLLC deadline violation (%)")
    ax.set_title(sc)
    fig.savefig(out / f"fig_v2_3_deadline_{sc}.pdf")
    fig.savefig(out / f"fig_v2_3_deadline_{sc}.png")
    plt.close(fig)


def fig_minsla(df, sc, out):
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    _bar_mean_std(ax, df, sc, "min_sla", "Worst-slice SLA satisf. (%)")
    ax.set_title(sc); ax.set_ylim(0, 108)
    fig.savefig(out / f"fig_v2_4_minsla_{sc}.pdf")
    fig.savefig(out / f"fig_v2_4_minsla_{sc}.png")
    plt.close(fig)


def fig_pareto(df, sc, out):
    set_ieee_style()
    sub = df[df["scenario"] == sc]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for m in [mm for mm in METHODS if mm in sub["method"].unique()]:
        g = sub[sub.method == m]
        ax.scatter(g["urllc_p99"].mean(), g["thr_total"].mean(), s=45, color=COLOR[m],
                   edgecolor="black", linewidth=0.5, label=METHOD_LABEL[m], zorder=3)
    ax.axvline(10.0, color="red", ls="--", lw=0.8)
    ax.set_xlabel("URLLC p99 delay (ms)"); ax.set_ylabel("Total throughput (Mb/s)")
    ax.set_title(sc); ax.legend(frameon=False)
    fig.savefig(out / f"fig_v2_5_pareto_{sc}.pdf")
    fig.savefig(out / f"fig_v2_5_pareto_{sc}.png")
    plt.close(fig)


def fig_dispersion(df, sc, out):
    set_ieee_style()
    sub = df[df["scenario"] == sc]
    methods = [m for m in METHODS if m in sub["method"].unique()]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for i, m in enumerate(methods):
        y = sub[sub.method == m]["score"].values
        x = np.full_like(y, i, dtype=float) + np.linspace(-0.12, 0.12, len(y))
        ax.scatter(x, y, s=18, color=COLOR[m], edgecolor="black", linewidth=0.3, zorder=3)
        ax.hlines(y.mean(), i - 0.25, i + 0.25, color="black", lw=1.2, zorder=4)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels([METHOD_LABEL[m] for m in methods])
    ax.set_ylabel("v2 score (per seed)"); ax.set_title(f"{sc} — seed dispersion")
    fig.savefig(out / f"fig_v2_6_dispersion_{sc}.pdf")
    fig.savefig(out / f"fig_v2_6_dispersion_{sc}.png")
    plt.close(fig)


def fig_weight_violin(df, sc, out):
    set_ieee_style()
    sub = df[df["scenario"] == sc]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    data = [sub[c].dropna().values for c in ("w_embb", "w_urllc", "w_mtc")]
    data = [d for d in data if len(d) > 0]
    if not data:
        plt.close(fig); return
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(list(SLICE_COLOR.values())[i]); pc.set_alpha(0.6)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(list(SLICES))
    ax.set_ylabel("Best-candidate weight"); ax.set_title(f"{sc} — weight stability")
    fig.savefig(out / f"fig_v2_7_weights_{sc}.pdf")
    fig.savefig(out / f"fig_v2_7_weights_{sc}.png")
    plt.close(fig)


def fig_ccdf(df, sc, out):
    """CCDF of URLLC delay across seeds from per-seed percentile anchors.

    Honest approximation: each seed contributes its (mean,p95,p99,p999) anchor
    points; we plot the empirical complementary CDF of the pooled p99/p999 to
    show the tail spread across seeds. Labeled as percentile-anchor based."""
    set_ieee_style()
    sub = df[df["scenario"] == sc]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for m in [mm for mm in METHODS if mm in sub["method"].unique()]:
        g = sub[sub.method == m]
        p99s = np.sort(g["urllc_p99"].dropna().values)
        if len(p99s) == 0:
            continue
        ccdf = 1.0 - np.arange(1, len(p99s) + 1) / len(p99s)
        ax.step(p99s, ccdf, where="post", color=COLOR[m], label=METHOD_LABEL[m], lw=1.2)
    ax.axvline(10.0, color="red", ls="--", lw=0.8)
    ax.set_xlabel("URLLC p99 delay across seeds (ms)")
    ax.set_ylabel("CCDF (fraction of seeds)")
    ax.set_title(f"{sc} — tail spread"); ax.legend(frameon=False)
    fig.savefig(out / f"fig_v2_8_ccdf_{sc}.pdf")
    fig.savefig(out / f"fig_v2_8_ccdf_{sc}.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "paper_v2_campaign" / "figures")
    args = ap.parse_args()

    df = collect(args.meta_root)
    if df.empty:
        raise SystemExit(f"No data under {args.meta_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir.parent / "v2_best_candidates.csv", index=False)

    scenarios = sorted(df["scenario"].unique())
    print(f"Cenarios com dados: {scenarios}")
    for sc in scenarios:
        for fn in (fig_scores, fig_urllc_percentiles, fig_deadline, fig_minsla,
                   fig_pareto, fig_dispersion, fig_weight_violin, fig_ccdf):
            try:
                fn(df, sc, args.out_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] {fn.__name__}({sc}) falhou: {exc}")
    print(f"Figuras em: {args.out_dir}")


if __name__ == "__main__":
    main()
