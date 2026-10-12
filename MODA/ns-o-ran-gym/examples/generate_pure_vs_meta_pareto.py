#!/usr/bin/env python3
"""Generate defense figures comparing pure schedulers with meta-heuristics.

Only RR, PF, BCQI, GA, PSO, SA, and Hybrid are shown.  The figures make the
central trade-off explicit: a pure scheduler can maximize its native utility
(e.g., channel efficiency/throughput), while optimized inter-slice weights can
improve worst-slice SLA protection and balance across heterogeneous slices.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from generate_completed_scenario_figures import (
    COLORS,
    DEFAULT_RESULTS,
    LABELS,
    SCENARIOS,
    SCENARIO_LABELS,
    configure_style,
    load_baselines,
    load_final_meta,
    load_meta_evaluations,
    validate_completed_meta,
)


ARTICLE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ARTICLE_ROOT / "paper_figures_completed_scenarios_en"
PURE_METHODS = ("pure_rr", "pure_pf", "pure_bcqi")
META_METHODS = ("ga", "pso", "sa", "hybrid")
METHOD_ORDER = PURE_METHODS + META_METHODS
MARKERS = {
    "pure_rr": "o",
    "pure_pf": "s",
    "pure_bcqi": "^",
    "ga": "o",
    "pso": "s",
    "sa": "^",
    "hybrid": "D",
}


def jain_sla_fairness(row: pd.Series) -> float:
    """Jain index (%) over the three per-slice SLA satisfaction values."""
    values = row[["sla_embb_pct", "sla_urllc_pct", "sla_mtc_pct"]].to_numpy(
        dtype=float
    )
    denominator = len(values) * float(np.square(values).sum())
    if denominator <= 0.0:
        return 0.0
    return 100.0 * float(values.sum() ** 2) / denominator


def load_selected_results(results_root: Path) -> pd.DataFrame:
    meta_root = results_root / "metaheuristics"
    baseline_root = results_root / "heuristics_ns3/results_rslaq_network_only"
    evaluations = load_meta_evaluations(meta_root)
    validate_completed_meta(evaluations)
    final_meta = load_final_meta(evaluations)
    baselines = load_baselines(baseline_root)
    baselines = baselines[baselines["method"].isin(PURE_METHODS)].copy()
    data = pd.concat([baselines, final_meta], ignore_index=True)
    data = data[data["method"].isin(METHOD_ORDER)].copy()
    data["sla_jain_pct"] = data.apply(jain_sla_fairness, axis=1)
    data["method_order"] = pd.Categorical(
        data["method"], categories=METHOD_ORDER, ordered=True
    )
    return data.sort_values(["scenario", "method_order", "seed"])


def pareto_front(means: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Return the mean solutions not dominated when both KPIs are maximized."""
    points = means[[x_col, y_col]].to_numpy(dtype=float)
    efficient = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        dominates = np.all(points >= point, axis=1) & np.any(points > point, axis=1)
        efficient[index] = not dominates.any()
    return means[efficient].sort_values(x_col)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def legend_handles() -> list[Line2D]:
    handles: list[Line2D] = []
    for method in METHOD_ORDER:
        edge = "#222222" if method == "pure_pf" else "white"
        handles.append(
            Line2D(
                [0],
                [0],
                marker=MARKERS[method],
                color="none",
                markerfacecolor=COLORS[method],
                markeredgecolor=edge,
                markeredgewidth=0.6,
                markersize=6.2,
                label=LABELS[method],
            )
        )
    return handles


def draw_method_points(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    annotate: bool = False,
) -> pd.DataFrame:
    means = (
        data.groupby("method", observed=True, as_index=False)[[x_col, y_col]]
        .mean()
        .copy()
    )
    for method in METHOD_ORDER:
        runs = data[data["method"] == method]
        ax.scatter(
            runs[x_col],
            runs[y_col],
            s=16,
            marker=MARKERS[method],
            color=COLORS[method],
            alpha=0.25,
            edgecolor="none",
            zorder=2,
        )
        mean = means[means["method"] == method]
        ax.scatter(
            mean[x_col],
            mean[y_col],
            s=57,
            marker=MARKERS[method],
            color=COLORS[method],
            edgecolor="#222222" if method == "pure_pf" else "white",
            linewidth=0.75,
            zorder=4,
        )
    front = pareto_front(means, x_col, y_col)
    ax.plot(
        front[x_col],
        front[y_col],
        color="#202020",
        lw=1.0,
        ls=(0, (2.5, 2.0)),
        zorder=1,
    )
    if annotate:
        annotate_means(ax, means, x_col, y_col)
    return means


def annotate_means(
    ax: plt.Axes, means: pd.DataFrame, x_col: str, y_col: str
) -> None:
    if y_col == "min_sla_pct":
        offsets = {
            "pure_rr": (-18, -11),
            "pure_pf": (5, 4),
            "pure_bcqi": (5, 4),
            "ga": (5, 4),
            "pso": (-18, 4),
            "sa": (5, 4),
            "hybrid": (5, -11),
        }
    else:
        offsets = {
            "pure_rr": (-18, -11),
            "pure_pf": (5, -11),
            "pure_bcqi": (5, 4),
            "ga": (5, 4),
            "pso": (-18, 4),
            "sa": (5, 4),
            "hybrid": (5, -11),
        }
    for row in means.itertuples(index=False):
        dx, dy = offsets[row.method]
        ax.annotate(
            LABELS[row.method],
            (getattr(row, x_col), getattr(row, y_col)),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.1,
            color="#222222",
        )


def plot_pareto_across_scenarios(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.85), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        subset = data[data["scenario"] == scenario]
        draw_method_points(
            ax,
            subset,
            "total_throughput_mbps",
            "min_sla_pct",
            annotate=False,
        )
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_xlabel("Aggregate throughput (Mbps)")
        ax.grid(axis="both", visible=True)
    axes[0].set_ylabel("Worst-slice SLA satisfaction (%)")
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=7,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    fig.suptitle(
        "The throughput–SLA trade-off emerges when resources become scarce",
        y=1.13,
    )
    fig.text(
        0.5,
        -0.04,
        "Large markers: seed means; translucent markers: individual seeds; dotted line: mean Pareto front.",
        ha="center",
        fontsize=7.3,
        color="#444444",
    )
    fig.subplots_adjust(wspace=0.18, top=0.80, bottom=0.22)
    save_figure(fig, output_dir, "fig01_pareto_throughput_sla_three_scenarios")


def plot_congestion_pareto(data: pd.DataFrame, output_dir: Path) -> None:
    congestion = data[data["scenario"] == "congestion"]
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.35))
    draw_method_points(
        axes[0],
        congestion,
        "total_throughput_mbps",
        "min_sla_pct",
        annotate=True,
    )
    draw_method_points(
        axes[1],
        congestion,
        "total_throughput_mbps",
        "sla_jain_pct",
        annotate=True,
    )
    axes[0].set_xlabel("Aggregate throughput (Mbps; higher is better)")
    axes[0].set_ylabel("Worst-slice SLA satisfaction (%)")
    axes[0].set_title("SLA protection")
    axes[1].set_xlabel("Aggregate throughput (Mbps; higher is better)")
    axes[1].set_ylabel("SLA fairness across slices (Jain index, %)")
    axes[1].set_title("Inter-slice SLA balance")
    for ax in axes:
        ax.grid(axis="both", visible=True)
    fig.suptitle(
        "Congestion: throughput maximization and balanced SLA fulfillment are competing goals",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.035,
        "Fairness is Jain's index over [eMBB, URLLC, MTC] SLA satisfaction. GA/PSO/SA/Hybrid optimize weights; PF remains intra-slice.",
        ha="center",
        fontsize=7.2,
        color="#444444",
    )
    fig.subplots_adjust(wspace=0.31, bottom=0.21)
    save_figure(fig, output_dir, "fig02_pareto_congestion_sla_fairness")


def plot_throughput_and_slice_sla(data: pd.DataFrame, output_dir: Path) -> None:
    congestion = data[data["scenario"] == "congestion"].copy()
    means = congestion.groupby("method", observed=True, as_index=False).agg(
        throughput=("total_throughput_mbps", "mean"),
        sla_embb=("sla_embb_pct", "mean"),
        sla_urllc=("sla_urllc_pct", "mean"),
        sla_mtc=("sla_mtc_pct", "mean"),
    )
    means["method"] = pd.Categorical(
        means["method"], categories=METHOD_ORDER, ordered=True
    )
    means = means.sort_values("method")
    y = np.arange(len(METHOD_ORDER))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.35, 3.75),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
        sharey=True,
    )
    colors = [COLORS[method] for method in METHOD_ORDER]
    axes[0].barh(y, means["throughput"], color=colors, alpha=0.78, height=0.58)
    for row_index, method in enumerate(METHOD_ORDER):
        runs = congestion[congestion["method"] == method]
        axes[0].scatter(
            runs["total_throughput_mbps"],
            np.full(len(runs), row_index),
            s=16,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        axes[0].text(
            means.iloc[row_index]["throughput"] + 1.5,
            row_index,
            f"{means.iloc[row_index]['throughput']:.1f}",
            va="center",
            fontsize=7.2,
        )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABELS[m] for m in METHOD_ORDER])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Aggregate throughput (Mbps)")
    axes[0].set_title("Native throughput outcome")
    axes[0].grid(axis="x", visible=True)
    axes[0].grid(axis="y", visible=False)

    matrix = means[["sla_embb", "sla_urllc", "sla_mtc"]].to_numpy(dtype=float)
    image = axes[1].imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axes[1].text(
                column,
                row,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=7.7,
                color="white" if value >= 62 else "#222222",
            )
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(("eMBB", "URLLC", "MTC"))
    axes[1].set_title("Per-slice SLA satisfaction (%)")
    axes[1].grid(False)
    axes[1].axhline(2.5, color="white", lw=2.0)
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.055, pad=0.04)
    colorbar.set_label("SLA satisfaction (%)")
    axes[0].axhline(2.5, color="#333333", lw=0.9, ls="--")
    fig.suptitle(
        "The throughput winner can leave a slice behind; optimized weights protect SLA balance",
        y=0.99,
    )
    fig.text(
        0.5,
        0.025,
        "Dashed separator: pure schedulers (top) versus meta-heuristic weight optimization (bottom). Values are seed means.",
        ha="center",
        fontsize=7.2,
        color="#444444",
    )
    fig.subplots_adjust(left=0.11, right=0.94, top=0.87, bottom=0.16, wspace=0.20)
    save_figure(fig, output_dir, "fig03_congestion_throughput_vs_slice_sla")


def validate_outputs(data: pd.DataFrame, output_dir: Path) -> None:
    expected_rows = len(SCENARIOS) * len(METHOD_ORDER) * len((1, 2, 3))
    if len(data) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} selected runs, found {len(data)}")
    expected_files = {
        f"{stem}.{extension}"
        for stem in (
            "fig01_pareto_throughput_sla_three_scenarios",
            "fig02_pareto_congestion_sla_fairness",
            "fig03_congestion_throughput_vs_slice_sla",
        )
        for extension in ("png", "pdf", "svg")
    }
    actual_files = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(
            f"Output directory must contain only the 9 figure files; found {sorted(actual_files)}"
        )


def print_defense_summary(data: pd.DataFrame) -> None:
    congestion = data[data["scenario"] == "congestion"]
    means = congestion.groupby("method", observed=True).agg(
        throughput=("total_throughput_mbps", "mean"),
        worst_sla=("min_sla_pct", "mean"),
        sla_fairness=("sla_jain_pct", "mean"),
    )
    throughput_winner = means["throughput"].idxmax()
    sla_winner = means["worst_sla"].idxmax()
    print(
        f"Congestion throughput leader: {LABELS[throughput_winner]} "
        f"({means.loc[throughput_winner, 'throughput']:.2f} Mbps)"
    )
    print(
        f"Congestion best worst-slice SLA: {LABELS[sla_winner]} "
        f"({means.loc[sla_winner, 'worst_sla']:.2f}%)"
    )
    print(f"Figures: {DEFAULT_OUTPUT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    data = load_selected_results(args.results.resolve())
    output_dir = args.output.resolve()
    plot_pareto_across_scenarios(data, output_dir)
    plot_congestion_pareto(data, output_dir)
    plot_throughput_and_slice_sla(data, output_dir)
    validate_outputs(data, output_dir)
    print_defense_summary(data)


if __name__ == "__main__":
    main()
