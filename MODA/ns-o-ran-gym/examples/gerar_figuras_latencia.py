#!/usr/bin/env python3
"""Gera gráficos de latência URLLC das meta-heurísticas para o trabalho.

Foco: mostrar que a otimização encontra configurações com latência URLLC
excelente (delay_mean e p99 bem abaixo do SLA de 10ms).

Filtra apenas evals onde URLLC recebeu recursos (w_urllc > 0.05) e de fato
transmitiu pacotes (PDR > 0), para evitar o artefato de delay=0 quando o
slice não tem espectro.

Como rodar:
    cd /home/elioth/Documentos/artigo_jussi/ns-o-ran-gym
    .venv-dashboard/bin/python examples/gerar_figuras_latencia.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/elioth/Documentos/artigo_jussi")
RESULTS_ROOT = REPO_ROOT / "ns-o-ran-gym/results_controlled/heuristics_metaheuristics/20260626_122326"
META_ROOT = RESULTS_ROOT / "metaheuristics"
BASELINE_ROOT = RESULTS_ROOT / "heuristics_ns3/results_rslaq_network_only"
FIG_DIR = REPO_ROOT / "figuras_overleaf"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["low_traffic", "normal", "congestion"]
SCENARIO_LABELS = {"low_traffic": "Low Traffic", "normal": "Normal", "congestion": "Congestion"}
SEEDS = [1, 2, 3]
METHODS = ["ga", "pso", "sa", "hybrid"]
METHOD_LABELS = {"ga": "GA", "pso": "PSO", "sa": "SA", "hybrid": "Hybrid"}
METHOD_COLORS = {"ga": "#1f77b4", "pso": "#d62728", "sa": "#2ca02c", "hybrid": "#9467bd"}

PURE_MODES = ["pure_rr", "pure_pf", "pure_bcqi"]
PURE_LABELS = {"pure_rr": "RR", "pure_pf": "PF", "pure_bcqi": "BCQI"}
PURE_COLORS = {"pure_rr": "#7f7f7f", "pure_pf": "#bcbd22", "pure_bcqi": "#17becf"}

SLA_LIMIT_MS = 10.0  # URLLC latency budget

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.fontsize": 10,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

sys.path.insert(0, str(REPO_ROOT / "ns-o-ran-gym/src"))
from nsoran.scoring import read_summary  # noqa: E402


def load_meta_urllc() -> pd.DataFrame:
    """Carrega latência URLLC de todos evals onde URLLC esteve ativo."""
    rows = []
    for sc in SCENARIOS:
        for sd in SEEDS:
            base = META_ROOT / f"scenario={sc}/seed={sd}/metaheuristic_search/evals"
            if not base.exists():
                continue
            for sidecar in base.glob("*_eval_*/candidate.json"):
                try:
                    p = json.loads(sidecar.read_text())
                    if p.get("failed"):
                        continue
                    w = p["weights"]
                    if w[1] < 0.05:  # URLLC sem recurso
                        continue
                    summary = sidecar.parent / f"results_rslaq_network_only/scenario={sc}/mode=slice_custom/seed={sd}_run=1/summary.csv"
                    if not summary.exists():
                        continue
                    data = read_summary(summary)
                    u = [r for r in data if r.get("slice") == "URLLC"][0]
                    pdr = float(u.get("pdr_pct", 0))
                    if pdr <= 0:
                        continue
                    rows.append({
                        "method": p["method"],
                        "scenario": sc,
                        "seed": sd,
                        "score": p["score"],
                        "delay_mean": float(u.get("delay_ms_mean", 0)),
                        "delay_p95": float(u.get("delay_ms_p95", 0)),
                        "delay_p99": float(u.get("delay_ms_p99", 0)),
                        "jitter": float(u.get("jitter_ms_mean", 0)),
                        "pdr": pdr,
                        "w_urllc": w[1],
                    })
                except (KeyError, ValueError, IndexError):
                    continue
    return pd.DataFrame(rows)


def load_baseline_urllc() -> pd.DataFrame:
    """Carrega latência URLLC dos schedulers puros."""
    rows = []
    for sc in SCENARIOS:
        for mode in PURE_MODES:
            for sd in SEEDS:
                summary = BASELINE_ROOT / f"scenario={sc}/mode={mode}/seed={sd}_run=1/summary.csv"
                if not summary.exists():
                    continue
                try:
                    data = read_summary(summary)
                    u = [r for r in data if r.get("slice") == "URLLC"][0]
                    rows.append({
                        "method": mode,
                        "scenario": sc,
                        "seed": sd,
                        "delay_mean": float(u.get("delay_ms_mean", 0)),
                        "delay_p95": float(u.get("delay_ms_p95", 0)),
                        "delay_p99": float(u.get("delay_ms_p99", 0)),
                        "jitter": float(u.get("jitter_ms_mean", 0)),
                        "pdr": float(u.get("pdr_pct", 0)),
                    })
                except Exception:
                    continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_boxplot_latencia_meta(df: pd.DataFrame) -> None:
    """Fig 8: Boxplot da latência URLLC (delay_mean) por meta-heurística.

    Mostra que as 4 meta-heurísticas encontram configurações com latência
    URLLC bem abaixo do SLA de 10ms na maioria dos casos.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, sc in zip(axes, SCENARIOS):
        sub = df[df.scenario == sc]
        data = []
        labels = []
        colors = []
        for m in METHODS:
            d = sub[sub.method == m]["delay_mean"].values
            if len(d) > 0:
                data.append(d)
                labels.append(METHOD_LABELS[m])
                colors.append(METHOD_COLORS[m])
        if not data:
            ax.set_title(f"{SCENARIO_LABELS[sc]} (no data)")
            continue
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                        showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white",
                                                    "markeredgecolor": "black", "markersize": 5})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(y=SLA_LIMIT_MS, color="red", linestyle="--", linewidth=1.5,
                   label=f"URLLC SLA ({SLA_LIMIT_MS:.0f} ms)")
        ax.set_title(SCENARIO_LABELS[sc])
        if ax == axes[0]:
            ax.set_ylabel("URLLC mean delay (ms)")
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("URLLC latency distribution per meta-heuristic (URLLC-active evaluations only)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(FIG_DIR / "fig08_urllc_latency_boxplot.png")
    plt.close(fig)
    print("  + fig08_urllc_latency_boxplot.png")


def fig_cdf_latencia(df: pd.DataFrame) -> None:
    """Fig 9: CDF (ECDF) da latência URLLC por método, cenário congestion.

    Mostra a fração de avaliações que cumprem delay <= 10ms.
    """
    sc = "congestion"
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for m in METHODS:
        sub = df[(df.scenario == sc) & (df.method == m)]
        if sub.empty:
            continue
        delays = np.sort(sub["delay_mean"].values)
        cdf = np.arange(1, len(delays) + 1) / len(delays)
        ax.plot(delays, cdf, linewidth=2, color=METHOD_COLORS[m],
                label=f"{METHOD_LABELS[m]} ({len(delays)} evals)")
    ax.axvline(x=SLA_LIMIT_MS, color="red", linestyle="--", linewidth=1.5,
               label=f"URLLC SLA ({SLA_LIMIT_MS:.0f} ms)")
    ax.set_xlabel("URLLC mean delay (ms)")
    ax.set_ylabel("Cumulative fraction of evaluations")
    ax.set_title(f"ECDF of URLLC latency — {SCENARIO_LABELS[sc]} (URLLC-active evals)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(left=0)
    fig.savefig(FIG_DIR / "fig09_urllc_latency_cdf.png")
    plt.close(fig)
    print("  + fig09_urllc_latency_cdf.png")


def fig_p99_meta_vs_baseline(meta: pd.DataFrame, base: pd.DataFrame) -> None:
    """Fig 10: URLLC p99 delay — meta-heurísticas (melhor eval) vs schedulers puros.

    Barras agrupadas por cenário. Mostra que as meta-heurísticas, no melhor caso,
    mantêm p99 competitivo ou melhor que RR/PF, embora BCQI seja difícil de bater.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)
    for ax, sc in zip(axes, SCENARIOS):
        names = []
        p99_vals = []
        colors_bar = []
        # Baselines (média 3 seeds)
        for mode in PURE_MODES:
            sub = base[(base.scenario == sc) & (base.method == mode)]
            if sub.empty:
                continue
            names.append(PURE_LABELS[mode])
            p99_vals.append(sub["delay_p99"].mean())
            colors_bar.append(PURE_COLORS[mode])
        # Meta-heurísticas: melhor (menor) p99 por método (média 3 seeds do min)
        for m in METHODS:
            sub = meta[(meta.scenario == sc) & (meta.method == m)]
            if sub.empty:
                continue
            # melhor p99 por seed, depois média
            best_per_seed = sub.groupby("seed")["delay_p99"].min()
            names.append(METHOD_LABELS[m])
            p99_vals.append(best_per_seed.mean())
            colors_bar.append(METHOD_COLORS[m])

        x = np.arange(len(names))
        ax.bar(x, p99_vals, color=colors_bar, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(y=SLA_LIMIT_MS, color="red", linestyle="--", linewidth=1.5,
                   label=f"SLA ({SLA_LIMIT_MS:.0f} ms)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_title(SCENARIO_LABELS[sc])
        if ax == axes[0]:
            ax.set_ylabel("URLLC P99 delay (ms)")
        ax.legend(loc="upper left", fontsize=8)
        # separador puros/meta
        n_pure = sum(1 for mode in PURE_MODES if not base[(base.scenario == sc) & (base.method == mode)].empty)
        ax.axvline(x=n_pure - 0.5, color="gray", linestyle=":", alpha=0.4)
    fig.suptitle("URLLC P99 delay: pure schedulers vs. meta-heuristics (best case per method)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(FIG_DIR / "fig10_urllc_p99_comparison.png")
    plt.close(fig)
    print("  + fig10_urllc_p99_comparison.png")


def fig_sla_compliance(df: pd.DataFrame) -> None:
    """Fig 11: % de avaliações que cumprem SLA URLLC (delay_mean <= 10ms) por método.

    Gráfico de barras mostrando a taxa de compliance por método e cenário.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    width = 0.2
    x = np.arange(len(SCENARIOS))
    for i, m in enumerate(METHODS):
        rates = []
        for sc in SCENARIOS:
            sub = df[(df.scenario == sc) & (df.method == m)]
            if sub.empty:
                rates.append(0)
            else:
                compliant = (sub["delay_mean"] <= SLA_LIMIT_MS).sum()
                rates.append(100 * compliant / len(sub))
        bars = ax.bar(x + i * width, rates, width, label=METHOD_LABELS[m],
                      color=METHOD_COLORS[m], alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, rate in zip(bars, rates):
            ax.annotate(f"{rate:.0f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(y=100, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([SCENARIO_LABELS[sc] for sc in SCENARIOS])
    ax.set_ylabel("% evaluations with URLLC delay <= 10 ms")
    ax.set_title("URLLC SLA compliance rate per meta-heuristic (URLLC-active evaluations)")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")
    fig.savefig(FIG_DIR / "fig11_urllc_sla_compliance.png")
    plt.close(fig)
    print("  + fig11_urllc_sla_compliance.png")


# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading URLLC latency data...")
    meta_df = load_meta_urllc()
    base_df = load_baseline_urllc()
    print(f"  Meta-heuristics (URLLC-active): {len(meta_df)} evals")
    print(f"  Baselines: {len(base_df)} points")
    print()

    # Resumo rápido
    for sc in SCENARIOS:
        sub = meta_df[meta_df.scenario == sc]
        if sub.empty:
            continue
        compliant = (sub["delay_mean"] <= SLA_LIMIT_MS).sum()
        print(f"  {sc}: {len(sub)} evals, {compliant} ({100*compliant/len(sub):.0f}%) com delay<=10ms, "
              f"melhor={sub['delay_mean'].min():.2f}ms, mediana={sub['delay_mean'].median():.2f}ms")
    print()

    print("Generating figures in", FIG_DIR)
    fig_boxplot_latencia_meta(meta_df)
    fig_cdf_latencia(meta_df)
    fig_p99_meta_vs_baseline(meta_df, base_df)
    fig_sla_compliance(meta_df)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
