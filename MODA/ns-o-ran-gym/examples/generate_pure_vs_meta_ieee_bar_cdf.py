#!/usr/bin/env python3
"""Generate IEEE-style bar charts and ECDFs after averaging random seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from generate_completed_scenario_figures import LABELS
from generate_pure_vs_meta_pareto import (
    DEFAULT_OUTPUT,
    DEFAULT_RESULTS,
    METHOD_ORDER,
    SCENARIOS,
    SCENARIO_LABELS,
    load_selected_results,
)
from nsoran.scoring import read_summary


IEEE_WIDTH = 7.16
SLICE_NAMES = ("eMBB", "URLLC", "MTC")
METHOD_STYLES = {
    "pure_rr": {
        "color": "#FFFFFF",
        "edge": "#111111",
        "hatch": "////",
        "marker": "o",
        "linestyle": "-",
    },
    "pure_pf": {
        "color": "#D9D9D9",
        "edge": "#444444",
        "hatch": "\\\\\\\\",
        "marker": "s",
        "linestyle": "--",
    },
    "pure_bcqi": {
        "color": "#8C8C8C",
        "edge": "#222222",
        "hatch": "xxxx",
        "marker": "^",
        "linestyle": ":",
    },
    "ga": {
        "color": "#0072B2",
        "edge": "#00517D",
        "hatch": "",
        "marker": "D",
        "linestyle": "-",
    },
    "pso": {
        "color": "#D55E00",
        "edge": "#963F00",
        "hatch": "..",
        "marker": "v",
        "linestyle": "--",
    },
    "sa": {
        "color": "#009E73",
        "edge": "#006A4E",
        "hatch": "++",
        "marker": "P",
        "linestyle": "-.",
    },
    "hybrid": {
        "color": "#7A5195",
        "edge": "#513563",
        "hatch": "oo",
        "marker": "X",
        "linestyle": (0, (5, 1, 1, 1)),
    },
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
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 8.4,
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
        }
    )


def safe_float(value: object) -> float:
    if value in (None, "", "NA"):
        return 0.0
    return float(value)


def load_slice_rows(run_data: pd.DataFrame) -> pd.DataFrame:
    """Load per-slice KPIs, then average the three random seeds."""
    records: list[dict[str, object]] = []
    for run in run_data.itertuples(index=False):
        summary_path = Path(run.result_dir) / "summary.csv"
        rows = read_summary(summary_path)
        by_slice = {row["slice"]: row for row in rows}
        for slice_name in SLICE_NAMES:
            row = by_slice[slice_name]
            records.append(
                {
                    "scenario": run.scenario,
                    "method": run.method,
                    "slice": slice_name,
                    "sla_pct": safe_float(row.get("sla_satisfaction_pct")),
                    "offered_pct": safe_float(
                        row.get("offered_load_satisfaction_pct")
                    ),
                    "throughput_mbps": safe_float(
                        row.get("throughput_mbps_mean")
                    ),
                }
            )
    raw = pd.DataFrame.from_records(records)
    # The seed dimension disappears here. All subsequent plots consume only
    # these method/scenario/slice means.
    return (
        raw.groupby(["scenario", "method", "slice"], as_index=False)
        .agg(
            sla_pct=("sla_pct", "mean"),
            offered_pct=("offered_pct", "mean"),
            throughput_mbps=("throughput_mbps", "mean"),
        )
        .copy()
    )


def aggregate_methods(slice_means: pd.DataFrame) -> pd.DataFrame:
    """Compute scenario-level metrics only after seed averaging."""
    records: list[dict[str, object]] = []
    for (scenario, method), group in slice_means.groupby(
        ["scenario", "method"], observed=True
    ):
        records.append(
            {
                "scenario": scenario,
                "method": method,
                "total_throughput_mbps": group["throughput_mbps"].sum(),
                "minimum_sla_pct": group["sla_pct"].min(),
            }
        )
    return pd.DataFrame.from_records(records)


def bar_legend() -> list[Patch]:
    return [
        Patch(
            facecolor=METHOD_STYLES[method]["color"],
            edgecolor=METHOD_STYLES[method]["edge"],
            hatch=METHOD_STYLES[method]["hatch"],
            linewidth=0.75,
            label=LABELS[method],
        )
        for method in METHOD_ORDER
    ]


def draw_bar_panels(
    aggregated: pd.DataFrame,
    metric: str,
    ylabel: str,
    ylim: tuple[float, float],
    output_dir: Path,
    stem: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(IEEE_WIDTH, 2.45), sharey=True)
    x = np.arange(len(METHOD_ORDER))
    panels = ("(a)", "(b)", "(c)")
    for ax, scenario, panel in zip(axes, SCENARIOS, panels):
        subset = aggregated[aggregated["scenario"] == scenario].set_index(
            "method"
        )
        values = np.array([subset.loc[method, metric] for method in METHOD_ORDER])
        for index, method in enumerate(METHOD_ORDER):
            style = METHOD_STYLES[method]
            ax.bar(
                index,
                values[index],
                width=0.72,
                color=style["color"],
                edgecolor=style["edge"],
                hatch=style["hatch"],
                linewidth=0.75,
                zorder=3,
            )
        ax.axvline(2.5, color="black", linewidth=0.65, linestyle=(0, (3, 2)))
        ax.set_title(f"{panel} {SCENARIO_LABELS[scenario]}", pad=3.0)
        ax.set_xticks(x)
        ax.set_xticklabels(
            ("RR", "PF", "BCQI", "GA", "PSO", "SA", "HYB"),
            rotation=0,
            ha="center",
        )
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    axes[0].set_ylabel(ylabel, labelpad=2.0)
    fig.legend(
        handles=bar_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=7,
        columnspacing=0.8,
        handlelength=1.4,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.80, wspace=0.17)
    save_ieee(fig, output_dir, stem)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(np.asarray(values, dtype=float))
    probabilities = np.arange(1, len(sorted_values) + 1, dtype=float) / len(
        sorted_values
    )
    x = np.concatenate(([sorted_values[0]], sorted_values))
    y = np.concatenate(([0.0], probabilities))
    return x, y


def line_legend() -> list[Line2D]:
    handles: list[Line2D] = []
    for method in METHOD_ORDER:
        style = METHOD_STYLES[method]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["edge"],
                linestyle=style["linestyle"],
                linewidth=1.15,
                marker=style["marker"],
                markersize=4.3,
                markerfacecolor=style["color"],
                markeredgecolor=style["edge"],
                markeredgewidth=0.6,
                label=LABELS[method],
            )
        )
    return handles


def plot_cdfs(slice_means: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_WIDTH, 2.70), sharey=True)
    metrics = (
        ("sla_pct", "(a) SLA satisfaction", "SLA satisfaction (%)"),
        (
            "offered_pct",
            "(b) Offered-load satisfaction",
            "Offered-load satisfaction (%)",
        ),
    )
    for ax, (metric, title, xlabel) in zip(axes, metrics):
        for method in METHOD_ORDER:
            style = METHOD_STYLES[method]
            values = slice_means[slice_means["method"] == method][metric].to_numpy(
                dtype=float
            )
            x, y = ecdf(values)
            ax.step(
                x,
                y,
                where="post",
                color=style["edge"],
                linestyle=style["linestyle"],
                linewidth=1.15,
                marker=style["marker"],
                markersize=3.7,
                markerfacecolor=style["color"],
                markeredgecolor=style["edge"],
                markeredgewidth=0.55,
            )
        ax.set_title(title, pad=3.0)
        ax.set_xlabel(xlabel, labelpad=2.0)
        ax.set_xlim(0, 101)
        ax.set_ylim(0, 1.02)
        ax.grid(True, axis="both")
    axes[0].set_ylabel("Empirical CDF", labelpad=2.0)
    fig.legend(
        handles=line_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=7,
        columnspacing=0.8,
        handlelength=1.7,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.17, top=0.82, wspace=0.22)
    save_ieee(fig, output_dir, "fig03_cdf_mean_qos")


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


def clean_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()


def validate_output(output_dir: Path) -> None:
    stems = (
        "fig01_bar_mean_throughput",
        "fig02_bar_mean_minimum_sla",
        "fig03_cdf_mean_qos",
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
    run_data = load_selected_results(args.results.resolve())
    slice_means = load_slice_rows(run_data)
    aggregated = aggregate_methods(slice_means)
    output_dir = args.output.resolve()
    clean_output(output_dir)
    draw_bar_panels(
        aggregated,
        "total_throughput_mbps",
        "Mean aggregate throughput (Mbit/s)",
        (0, 145),
        output_dir,
        "fig01_bar_mean_throughput",
    )
    draw_bar_panels(
        aggregated,
        "minimum_sla_pct",
        "Minimum SLA satisfaction (%)",
        (0, 105),
        output_dir,
        "fig02_bar_mean_minimum_sla",
    )
    plot_cdfs(slice_means, output_dir)
    validate_output(output_dir)
    print(f"IEEE bar/CDF figures written to: {output_dir}")


if __name__ == "__main__":
    main()
