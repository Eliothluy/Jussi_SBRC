#!/usr/bin/env python3
"""Generate IEEE figures for RBG allocation and effective resource use."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from generate_completed_scenario_figures import LABELS
from generate_pure_vs_meta_ieee_bar_cdf import (
    DEFAULT_OUTPUT,
    DEFAULT_RESULTS,
    SCENARIOS,
    SCENARIO_LABELS,
    configure_ieee_style,
    save_ieee,
)
from generate_pure_vs_meta_pareto import load_selected_results
from nsoran.scoring import read_summary


IEEE_WIDTH = 7.16
META_METHODS = ("ga", "pso", "sa", "hybrid")
SLICE_ORDER = ("eMBB", "URLLC", "MTC")
SLICE_ID_TO_NAME = {0: "eMBB", 1: "URLLC", 2: "MTC"}
SLICE_STYLES = {
    "eMBB": {"color": "#4C78A8", "edge": "#2D4A67", "hatch": ""},
    "URLLC": {"color": "#F58518", "edge": "#9B4E08", "hatch": "////"},
    "MTC": {"color": "#54A24B", "edge": "#32612D", "hatch": "xx"},
}
METHOD_STYLES = {
    "ga": {"color": "#0072B2", "edge": "#00517D", "hatch": ""},
    "pso": {"color": "#D55E00", "edge": "#963F00", "hatch": ".."},
    "sa": {"color": "#009E73", "edge": "#006A4E", "hatch": "++"},
    "hybrid": {"color": "#7A5195", "edge": "#513563", "hatch": "oo"},
}


def load_resource_metrics(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-run metrics and average the three random seeds afterward."""
    runs = load_selected_results(results_root)
    runs = runs[runs["method"].isin(META_METHODS)].copy()
    allocation_records: list[dict[str, object]] = []
    efficiency_records: list[dict[str, object]] = []

    for run in runs.itertuples(index=False):
        result_dir = Path(run.result_dir)
        allocation = pd.read_csv(result_dir / "slice_alloc.csv")
        summary = read_summary(result_dir / "summary.csv")

        # Mean allocated RBGs per scheduling record for each slice. Since each
        # timestamp contains one row per slice, the stacked sum is the mean
        # total RBG allocation per scheduling opportunity.
        allocation["slice_name"] = allocation["slice"].map(SLICE_ID_TO_NAME)
        per_slice = allocation.groupby("slice_name", as_index=False)[
            "allocated_rbg"
        ].mean()
        for row in per_slice.itertuples(index=False):
            allocation_records.append(
                {
                    "scenario": run.scenario,
                    "method": run.method,
                    "slice": row.slice_name,
                    "mean_allocated_rbg": float(row.allocated_rbg),
                }
            )

        with_budget = allocation[allocation["budget_rbg"] > 0]
        total_budget = float(with_budget["budget_rbg"].sum())
        assigned_budget = float(with_budget["allocated_rbg"].sum())
        utilization_pct = (
            100.0 * assigned_budget / total_budget if total_budget > 0.0 else 0.0
        )
        total_allocated_rbg = sum(
            float(row["allocated_rbg_total"]) for row in summary
        )
        received_bytes = sum(float(row["rx_bytes_total"]) for row in summary)
        delivered_bytes_per_rbg = (
            received_bytes / total_allocated_rbg
            if total_allocated_rbg > 0.0
            else 0.0
        )
        efficiency_records.append(
            {
                "scenario": run.scenario,
                "method": run.method,
                "budget_utilization_pct": utilization_pct,
                "delivered_bytes_per_rbg": delivered_bytes_per_rbg,
            }
        )

    allocation_means = (
        pd.DataFrame.from_records(allocation_records)
        .groupby(["scenario", "method", "slice"], as_index=False)
        .agg(mean_allocated_rbg=("mean_allocated_rbg", "mean"))
    )
    efficiency_means = (
        pd.DataFrame.from_records(efficiency_records)
        .groupby(["scenario", "method"], as_index=False)
        .agg(
            budget_utilization_pct=("budget_utilization_pct", "mean"),
            delivered_bytes_per_rbg=("delivered_bytes_per_rbg", "mean"),
        )
    )
    return allocation_means, efficiency_means


def slice_legend() -> list[Patch]:
    return [
        Patch(
            facecolor=SLICE_STYLES[slice_name]["color"],
            edgecolor=SLICE_STYLES[slice_name]["edge"],
            hatch=SLICE_STYLES[slice_name]["hatch"],
            linewidth=0.75,
            label=slice_name,
        )
        for slice_name in SLICE_ORDER
    ]


def method_legend() -> list[Patch]:
    return [
        Patch(
            facecolor=METHOD_STYLES[method]["color"],
            edgecolor=METHOD_STYLES[method]["edge"],
            hatch=METHOD_STYLES[method]["hatch"],
            linewidth=0.75,
            label=LABELS[method],
        )
        for method in META_METHODS
    ]


def plot_rbg_allocation(allocation: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(IEEE_WIDTH, 2.45), sharey=True)
    x = np.arange(len(META_METHODS))
    panels = ("(a)", "(b)", "(c)")

    for ax, scenario, panel in zip(axes, SCENARIOS, panels):
        subset = allocation[allocation["scenario"] == scenario]
        bottom = np.zeros(len(META_METHODS), dtype=float)
        for slice_name in SLICE_ORDER:
            slice_data = subset[subset["slice"] == slice_name].set_index("method")
            values = np.array(
                [slice_data.loc[method, "mean_allocated_rbg"] for method in META_METHODS]
            )
            style = SLICE_STYLES[slice_name]
            ax.bar(
                x,
                values,
                width=0.68,
                bottom=bottom,
                color=style["color"],
                edgecolor=style["edge"],
                hatch=style["hatch"],
                linewidth=0.75,
                zorder=3,
            )
            bottom += values
        ax.set_title(f"{panel} {SCENARIO_LABELS[scenario]}", pad=3.0)
        ax.set_xticks(x)
        ax.set_xticklabels(("GA", "PSO", "SA", "HYB"))
        ax.set_ylim(0, 285)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    axes[0].set_ylabel("Allocated RBGs / scheduling event", labelpad=2.0)
    fig.legend(
        handles=slice_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        columnspacing=1.2,
        handlelength=1.5,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.17, top=0.80, wspace=0.17)
    save_ieee(fig, output_dir, "fig04_bar_mean_rbg_allocation")


def plot_resource_efficiency(efficiency: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_WIDTH, 2.65))
    scenario_x = np.arange(len(SCENARIOS), dtype=float)
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    metrics = (
        (
            "budget_utilization_pct",
            "(a) Allocated-budget utilization",
            "Budget utilization (%)",
            (0, 105),
        ),
        (
            "delivered_bytes_per_rbg",
            "(b) Delivered-payload efficiency",
            "Delivered payload (byte/RBG)",
            (0, 55),
        ),
    )

    for ax, (metric, title, ylabel, ylim) in zip(axes, metrics):
        for method_index, method in enumerate(META_METHODS):
            method_data = efficiency[efficiency["method"] == method].set_index(
                "scenario"
            )
            values = np.array(
                [method_data.loc[scenario, metric] for scenario in SCENARIOS]
            )
            style = METHOD_STYLES[method]
            ax.bar(
                scenario_x + offsets[method_index],
                values,
                width=width,
                color=style["color"],
                edgecolor=style["edge"],
                hatch=style["hatch"],
                linewidth=0.75,
                zorder=3,
            )
        ax.set_title(title, pad=3.0)
        ax.set_ylabel(ylabel, labelpad=2.0)
        ax.set_xticks(scenario_x)
        ax.set_xticklabels(("Low traffic", "Normal", "Congestion"))
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    fig.legend(
        handles=method_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        columnspacing=1.0,
        handlelength=1.5,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.17, top=0.81, wspace=0.24)
    save_ieee(fig, output_dir, "fig05_bar_mean_rbg_efficiency")


def validate(allocation: pd.DataFrame, efficiency: pd.DataFrame) -> None:
    expected_alloc = len(SCENARIOS) * len(META_METHODS) * len(SLICE_ORDER)
    expected_eff = len(SCENARIOS) * len(META_METHODS)
    if len(allocation) != expected_alloc or len(efficiency) != expected_eff:
        raise RuntimeError("Incomplete resource-efficiency aggregation")
    if not np.allclose(efficiency["budget_utilization_pct"], 100.0, atol=1e-9):
        raise RuntimeError("Unexpected non-100% budget utilization in selected candidates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_ieee_style()
    allocation, efficiency = load_resource_metrics(args.results.resolve())
    validate(allocation, efficiency)
    output_dir = args.output.resolve()
    plot_rbg_allocation(allocation, output_dir)
    plot_resource_efficiency(efficiency, output_dir)
    print(f"Resource-allocation figures written to: {output_dir}")


if __name__ == "__main__":
    main()
