#!/usr/bin/env python3
"""Gera gráficos que diferenciam as meta-heurísticas de forma honesta.

Como os scores finais são similares (plateau na paisagem), a diferença real
entre GA/PSO/SA/Híbrida está em:
1. Velocidade de convergência (evals até atingir X% do ótimo)
2. Eficiência de busca (AUC normalizada da curva de convergência)
3. Estabilidade (consistência entre seeds)

Como rodar:
    cd /home/elioth/Documentos/artigo_jussi/ns-o-ran-gym
    .venv-dashboard/bin/python examples/gerar_figuras_eficiencia.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path("/home/elioth/Documentos/artigo_jussi")
META_ROOT = REPO_ROOT / "ns-o-ran-gym/results_controlled/heuristics_metaheuristics/20260626_122326/metaheuristics"
FIG_DIR = REPO_ROOT / "figuras_overleaf"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["low_traffic", "normal", "congestion"]
SCENARIO_LABELS = {"low_traffic": "Low Traffic", "normal": "Normal", "congestion": "Congestion"}
SEEDS = [1, 2, 3]
METHODS = ["ga", "pso", "sa", "hybrid"]
METHOD_LABELS = {"ga": "GA", "pso": "PSO", "sa": "SA", "hybrid": "Hybrid"}
METHOD_COLORS = {"ga": "#1f77b4", "pso": "#d62728", "sa": "#2ca02c", "hybrid": "#9467bd"}

plt.rcParams.update({
    "figure.figsize": (8, 5),
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


def load_convergence_data() -> dict:
    """Para cada (cenário, método, seed), retorna scores cummax e métricas."""
    data = {}
    for sc in SCENARIOS:
        data[sc] = {}
        for method in METHODS:
            curves = []
            for sd in SEEDS:
                base = META_ROOT / f"scenario={sc}/seed={sd}/metaheuristic_search/evals"
                if not base.exists():
                    continue
                evals = []
                for sidecar in sorted(base.glob(f"{method}_eval_*/candidate.json"),
                                       key=lambda p: int(p.parent.name.split("_")[-1])):
                    try:
                        p = json.loads(sidecar.read_text())
                        if p.get("failed"):
                            evals.append(-1e6)
                        else:
                            evals.append(p["score"])
                    except (KeyError, ValueError):
                        evals.append(-1e6)
                if not evals:
                    continue
                scores = np.array(evals)
                cummax = np.maximum.accumulate(scores)
                curves.append(cummax)
            data[sc][method] = curves
    return data


def fig_convergencia_normalizada(data: dict) -> None:
    """Fig 12: Convergência normalizada (score/best) por método.

    Mostra quão rápido cada método chega a 90%, 95% e 99% do seu melhor resultado.
    Eixo X: avaliações. Eixo Y: score normalizado [0,1].
    Curva média de 3 seeds com banda de ±1 std.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, sc in zip(axes, SCENARIOS):
        for method in METHODS:
            curves = data[sc].get(method, [])
            if not curves:
                continue
            # Normalizar cada curva pelo seu próprio máximo
            norm_curves = []
            for c in curves:
                best = c[-1]
                if best > 0:
                    norm_curves.append(c / best)
            if not norm_curves:
                continue
            min_len = min(len(c) for c in norm_curves)
            trimmed = np.array([c[:min_len] for c in norm_curves])
            mean = trimmed.mean(axis=0)
            std = trimmed.std(axis=0)
            x = np.arange(1, min_len + 1)
            ax.plot(x, mean, linewidth=2, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=METHOD_COLORS[method])
        ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_title(SCENARIO_LABELS[sc])
        ax.set_xlabel("Evaluation number")
        if ax == axes[0]:
            ax.set_ylabel("Normalized score (cumulative best / final best)")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
    fig.suptitle("Convergence efficiency: normalized cumulative best (mean +/- std, 3 seeds)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(FIG_DIR / "fig12_convergence_normalized.png")
    plt.close(fig)
    print("  + fig12_convergence_normalized.png")


def fig_evals_to_target(data: dict) -> None:
    """Fig 13: Avaliações necessárias para atingir 95% do ótimo por método.

    Barras agrupadas por cenário. Mostra que alguns métodos convergem mais rápido.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    width = 0.2
    x = np.arange(len(SCENARIOS))

    for i, method in enumerate(METHODS):
        means = []
        stds = []
        for sc in SCENARIOS:
            curves = data[sc].get(method, [])
            vals = []
            for c in curves:
                best = c[-1]
                if best <= 0:
                    continue
                target = 0.95 * best
                idx = np.argmax(c >= target)
                vals.append(idx + 1 if c[idx] >= target else len(c))
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if len(vals) > 1 else 0)
        bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
                      label=METHOD_LABELS[method], color=METHOD_COLORS[method], alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        # Annotate values
        for bar, val in zip(bars, means):
            ax.annotate(f"{val:.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([SCENARIO_LABELS[sc] for sc in SCENARIOS])
    ax.set_ylabel("Evaluations to reach 95% of best score")
    ax.set_title("Search efficiency: evaluations needed to reach 95% of the optimum (mean +/- std, 3 seeds)")
    ax.legend(loc="upper left")
    fig.savefig(FIG_DIR / "fig13_evals_to_95pct.png")
    plt.close(fig)
    print("  + fig13_evals_to_95pct.png")


def fig_auc_efficiency(data: dict) -> None:
    """Fig 14: Eficiência de busca (AUC normalizada) por método.

    AUC alto = boa qualidade média ao longo de toda a busca (não só no final).
    Mede quão rápido o método encontra boas soluções em média.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    width = 0.2
    x = np.arange(len(SCENARIOS))

    for i, method in enumerate(METHODS):
        means = []
        stds = []
        for sc in SCENARIOS:
            curves = data[sc].get(method, [])
            vals = []
            for c in curves:
                best = c[-1]
                if best <= 0:
                    continue
                auc = np.trapezoid(c) / (best * len(c)) if hasattr(np, "trapezoid") else np.sum(c) / (best * len(c))
                vals.append(auc)
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if len(vals) > 1 else 0)
        bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
                      label=METHOD_LABELS[method], color=METHOD_COLORS[method], alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, means):
            ax.annotate(f"{val:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([SCENARIO_LABELS[sc] for sc in SCENARIOS])
    ax.set_ylabel("Normalized AUC of convergence curve")
    ax.set_title("Search efficiency: area under the convergence curve (higher = faster good solutions)")
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc="lower right")
    fig.savefig(FIG_DIR / "fig14_auc_efficiency.png")
    plt.close(fig)
    print("  + fig14_auc_efficiency.png")


def main() -> None:
    print("Loading convergence data...")
    data = load_convergence_data()
    print("Generating figures...")
    fig_convergencia_normalizada(data)
    fig_evals_to_target(data)
    fig_auc_efficiency(data)
    print("Done.")


if __name__ == "__main__":
    main()
