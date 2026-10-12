#!/usr/bin/env python3
"""Generate IEEE-style pure-scheduler versus meta-heuristic figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from generate_pure_vs_meta_pareto import (
    DEFAULT_OUTPUT,
    DEFAULT_RESULTS,
    METHOD_ORDER,
    SCENARIOS,
    SCENARIO_LABELS,
    load_selected_results,
    pareto_front,
)
from generate_completed_scenario_figures import LABELS


# IEEE double-column width: 7.16 in.  Styles redundantly encode each method
# using luminance/color and a unique marker so figures survive grayscale print.
IEEE_WIDTH = 7.16
PURE_METHODS = ("pure_rr", "pure_pf", "pure_bcqi")
META_METHODS = ("ga", "pso", "sa", "hybrid")
METHOD_STYLES = {
    "pure_rr": {"color": "#111111", "marker": "o", "filled": False},
    "pure_pf": {"color": "#555555", "marker": "s", "filled": False},
    "pure_bcqi": {"color": "#999999", "marker": "^", "filled": False},
    "ga": {"color": "#0072B2", "marker": "D", "filled": True},
    "pso": {"color": "#D55E00", "marker": "v", "filled": True},
    "sa": {"color": "#009E73", "marker": "P", "filled": True},
    "hybrid": {"color": "#7A5195", "marker": "X", "filled": True},
}
SLICE_STYLES = {
    "eMBB": {"color": "#111111", "marker": "o"},
    "URLLC": {"color": "#666666", "marker": "s"},
    "MTC": {"color": "#AAAAAA", "marker": "^"},
}


def configure_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.75,
            "axes.edgecolor": "black",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.color": "#D0D0D0",
            "grid.linewidth": 0.45,
            "grid.linestyle": ":",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stixsans",
        }
    )


def save_ieee(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_dir / f"{stem}.eps", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def method_handle(method: str) -> Line2D:
    style = METHOD_STYLES[method]
    face = style["color"] if style["filled"] else "white"
    return Line2D(
        [0],
        [0],
        linestyle="none",
        marker=style["marker"],
        markersize=5.3,
        markerfacecolor=face,
        markeredgecolor=style["color"],
        markeredgewidth=0.9,
        label=LABELS[method],
    )


def draw_pareto_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    y_col: str,
) -> pd.DataFrame:
    means = (
        data.groupby("method", observed=True, as_index=False)[
            ["total_throughput_mbps", y_col]
        ]
        .mean()
        .copy()
    )
    for method in METHOD_ORDER:
        style = METHOD_STYLES[method]
        runs = data[data["method"] == method]
        # Open, small symbols: individual seeds.
        ax.scatter(
            runs["total_throughput_mbps"],
            runs[y_col],
            s=19,
            marker=style["marker"],
            facecolors="white",
            edgecolors=style["color"],
            linewidths=0.65,
            zorder=2,
        )
        mean = means[means["method"] == method]
        face = style["color"] if style["filled"] else "white"
        ax.scatter(
            mean["total_throughput_mbps"],
            mean[y_col],
            s=43,
            marker=style["marker"],
            facecolors=face,
            edgecolors=style["color"],
            linewidths=1.0,
            zorder=4,
        )
    front = pareto_front(means, "total_throughput_mbps", y_col)
    ax.plot(
        front["total_throughput_mbps"],
        front[y_col],
        color="black",
        linewidth=0.9,
        linestyle=(0, (4, 2)),
        zorder=1,
    )
    ax.grid(True, axis="both")
    return means


def plot_three_scenarios(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(IEEE_WIDTH, 2.35),
        sharey=True,
    )
    panel_labels = ("(a)", "(b)", "(c)")
    for ax, scenario, panel in zip(axes, SCENARIOS, panel_labels):
        subset = data[data["scenario"] == scenario]
        draw_pareto_panel(ax, subset, "min_sla_pct")
        ax.set_title(f"{panel} {SCENARIO_LABELS[scenario]}", pad=3.0)
        ax.set_xlabel("Throughput (Mbit/s)", labelpad=2.0)
    axes[0].set_ylabel("Minimum SLA satisfaction (%)", labelpad=2.0)
    axes[0].set_ylim(0, 105)
    fig.legend(
        handles=[method_handle(method) for method in METHOD_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=7,
        columnspacing=0.85,
        handletextpad=0.3,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.19, top=0.80, wspace=0.18)
    save_ieee(fig, output_dir, "fig01_pareto_throughput_sla_three_scenarios")


def plot_congestion_pareto(data: pd.DataFrame, output_dir: Path) -> None:
    congestion = data[data["scenario"] == "congestion"]
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_WIDTH, 2.75))
    draw_pareto_panel(axes[0], congestion, "min_sla_pct")
    draw_pareto_panel(axes[1], congestion, "sla_jain_pct")
    axes[0].set_title("(a) Worst-slice protection", pad=3.0)
    axes[1].set_title("(b) Inter-slice SLA balance", pad=3.0)
    axes[0].set_xlabel("Throughput (Mbit/s)", labelpad=2.0)
    axes[1].set_xlabel("Throughput (Mbit/s)", labelpad=2.0)
    axes[0].set_ylabel("Minimum SLA satisfaction (%)", labelpad=2.0)
    axes[1].set_ylabel("SLA Jain index (%)", labelpad=2.0)
    axes[0].set_ylim(0, 105)
    axes[1].set_ylim(60, 102)
    fig.legend(
        handles=[method_handle(method) for method in METHOD_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=7,
        columnspacing=0.9,
        handletextpad=0.3,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.17, top=0.83, wspace=0.29)
    save_ieee(fig, output_dir, "fig02_pareto_congestion_sla_fairness")


def plot_kpi_contrast(data: pd.DataFrame, output_dir: Path) -> None:
    congestion = data[data["scenario"] == "congestion"].copy()
    y_base = np.arange(len(METHOD_ORDER), dtype=float)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(IEEE_WIDTH, 3.05),
        sharey=True,
        gridspec_kw={"width_ratios": [0.92, 1.30]},
    )

    # (a) Aggregate throughput: individual seeds plus method mean.
    for row, method in enumerate(METHOD_ORDER):
        style = METHOD_STYLES[method]
        values = congestion[congestion["method"] == method][
            "total_throughput_mbps"
        ].to_numpy(dtype=float)
        axes[0].scatter(
            values,
            np.full(len(values), row),
            s=18,
            marker=style["marker"],
            facecolors="white",
            edgecolors=style["color"],
            linewidths=0.65,
            zorder=2,
        )
        face = style["color"] if style["filled"] else "white"
        axes[0].scatter(
            values.mean(),
            row,
            s=43,
            marker=style["marker"],
            facecolors=face,
            edgecolors=style["color"],
            linewidths=1.0,
            zorder=4,
        )
    axes[0].set_title("(a) Aggregate throughput", pad=3.0)
    axes[0].set_xlabel("Throughput (Mbit/s)", labelpad=2.0)
    axes[0].set_xlim(0, 170)
    axes[0].set_yticks(y_base)
    axes[0].set_yticklabels([LABELS[method] for method in METHOD_ORDER])
    axes[0].invert_yaxis()
    axes[0].grid(True, axis="x")
    axes[0].grid(False, axis="y")

    # (b) Per-slice SLA: unique grayscale marker per slice, with raw seeds
    # open and means filled. Small vertical offsets prevent occlusion.
    slice_columns = {
        "eMBB": "sla_embb_pct",
        "URLLC": "sla_urllc_pct",
        "MTC": "sla_mtc_pct",
    }
    slice_offsets = {"eMBB": -0.18, "URLLC": 0.0, "MTC": 0.18}
    for row, method in enumerate(METHOD_ORDER):
        method_runs = congestion[congestion["method"] == method]
        for slice_name, column in slice_columns.items():
            style = SLICE_STYLES[slice_name]
            values = method_runs[column].to_numpy(dtype=float)
            y = row + slice_offsets[slice_name]
            axes[1].scatter(
                values,
                np.full(len(values), y),
                s=15,
                marker=style["marker"],
                facecolors="white",
                edgecolors=style["color"],
                linewidths=0.6,
                zorder=2,
            )
            axes[1].scatter(
                values.mean(),
                y,
                s=36,
                marker=style["marker"],
                facecolors=style["color"],
                edgecolors="black",
                linewidths=0.55,
                zorder=4,
            )
    axes[1].set_title("(b) Per-slice SLA satisfaction", pad=3.0)
    axes[1].set_xlabel("SLA satisfaction (%)", labelpad=2.0)
    axes[1].set_xlim(0, 103)
    axes[1].grid(True, axis="x")
    axes[1].grid(False, axis="y")
    axes[1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=SLICE_STYLES[name]["marker"],
                markersize=5.1,
                markerfacecolor=SLICE_STYLES[name]["color"],
                markeredgecolor="black",
                markeredgewidth=0.55,
                label=name,
            )
            for name in ("eMBB", "URLLC", "MTC")
        ],
        loc="lower left",
        ncol=3,
        columnspacing=0.8,
        handletextpad=0.3,
    )

    for ax in axes:
        ax.axhline(2.5, color="black", linewidth=0.75, linestyle=(0, (3, 2)))
        ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.15, top=0.94, wspace=0.20)
    save_ieee(fig, output_dir, "fig03_congestion_throughput_vs_slice_sla")


def clean_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()


def validate_output(output_dir: Path) -> None:
    stems = (
        "fig01_pareto_throughput_sla_three_scenarios",
        "fig02_pareto_congestion_sla_fairness",
        "fig03_congestion_throughput_vs_slice_sla",
    )
    expected = {
        f"{stem}.{extension}"
        for stem in stems
        for extension in ("pdf", "eps", "png")
    }
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual != expected:
        raise RuntimeError(f"Unexpected output set: {sorted(actual)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_ieee_style()
    data = load_selected_results(args.results.resolve())
    output_dir = args.output.resolve()
    clean_output(output_dir)
    plot_three_scenarios(data, output_dir)
    plot_congestion_pareto(data, output_dir)
    plot_kpi_contrast(data, output_dir)
    validate_output(output_dir)
    print(f"IEEE figures written to: {output_dir}")


if __name__ == "__main__":
    main()
