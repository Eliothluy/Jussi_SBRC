#!/usr/bin/env python3
"""Generate publication-ready figures for the three completed scenarios.

The script compares pure schedulers, selected slice-aware baselines, and the
four meta-heuristics without treating dependent search evaluations as
independent experimental replicates.  The unit of replication is the ns-3
seed (n=3, except congestion/AQPS where seed 1 failed).

Outputs are written as 450-DPI PNG and vector PDF/SVG files.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ARTICLE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ARTICLE_ROOT / "ns-o-ran-gym"
DEFAULT_RESULTS = (
    PROJECT_ROOT
    / "results_controlled/heuristics_metaheuristics/20260626_122326"
)
DEFAULT_OUTPUT = ARTICLE_ROOT / "paper_figures_completed_scenarios_en"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from nsoran.scoring import read_summary, score_summary_rows  # noqa: E402


SCENARIOS = ("low_traffic", "normal", "congestion")
SCENARIO_LABELS = {
    "low_traffic": "Low traffic",
    "normal": "Normal load",
    "congestion": "Congestion",
}
SEEDS = (1, 2, 3)

BASELINE_MODES = (
    "pure_rr",
    "pure_pf",
    "pure_bcqi",
    "slice_aqps",
    "slice_meta_risk_elastic",
)
META_METHODS = ("ga", "pso", "sa", "hybrid")
METHOD_ORDER = BASELINE_MODES + META_METHODS
COMPARISON_METHODS = ("slice_aqps", "slice_meta_risk_elastic") + META_METHODS

LABELS = {
    "pure_rr": "RR",
    "pure_pf": "PF",
    "pure_bcqi": "BCQI",
    "slice_aqps": "AQPS",
    "slice_meta_risk_elastic": "Meta-Risk",
    "ga": "GA",
    "pso": "PSO",
    "sa": "SA",
    "hybrid": "Hybrid",
}
FAMILIES = {
    "pure_rr": "Pure scheduler",
    "pure_pf": "Pure scheduler",
    "pure_bcqi": "Pure scheduler",
    "slice_aqps": "Slice-aware baseline",
    "slice_meta_risk_elastic": "Slice-aware baseline",
    "ga": "Meta-heuristic",
    "pso": "Meta-heuristic",
    "sa": "Meta-heuristic",
    "hybrid": "Meta-heuristic",
}

# Okabe-Ito-derived, color-blind-safe palette plus neutral baselines.
COLORS = {
    "pure_rr": "#9A9A9A",
    "pure_pf": "#4D4D4D",
    "pure_bcqi": "#B8B8B8",
    "slice_aqps": "#E69F00",
    "slice_meta_risk_elastic": "#009E73",
    "ga": "#56B4E9",
    "pso": "#D55E00",
    "sa": "#7A9E2F",
    "hybrid": "#7B61A8",
}
MARKERS = {
    "pure_rr": "o",
    "pure_pf": "s",
    "pure_bcqi": "^",
    "slice_aqps": "P",
    "slice_meta_risk_elastic": "X",
    "ga": "o",
    "pso": "s",
    "sa": "^",
    "hybrid": "D",
}
EXPECTED_EVALUATIONS = {"ga": 72, "pso": 72, "sa": 72, "hybrid": 84}
SLICE_NAMES = ("eMBB", "URLLC", "MTC")
SLICE_COLORS = {"eMBB": "#0072B2", "URLLC": "#D55E00", "MTC": "#009E73"}


def configure_style() -> None:
    """Use a restrained two-column-paper style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.0,
            "figure.titlesize": 12.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def safe_float(value: object, default: float = math.nan) -> float:
    try:
        if value in (None, "", "NA"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_summary(path: Path) -> dict[str, float]:
    """Return the composite objective and interpretable network KPIs."""
    rows = read_summary(path)
    by_slice = {row["slice"]: row for row in rows}
    if not all(name in by_slice for name in SLICE_NAMES):
        raise ValueError(f"Incomplete slice summary: {path}")

    throughput = {
        name: safe_float(by_slice[name].get("throughput_mbps_mean"), 0.0)
        for name in SLICE_NAMES
    }
    sla = {
        name: safe_float(by_slice[name].get("sla_satisfaction_pct"), 0.0)
        for name in SLICE_NAMES
    }
    offered = {
        name: safe_float(by_slice[name].get("offered_load_satisfaction_pct"), 0.0)
        for name in SLICE_NAMES
    }
    pdr = {
        name: safe_float(by_slice[name].get("pdr_pct"), 0.0)
        for name in SLICE_NAMES
    }
    return {
        "objective": float(score_summary_rows(rows)),
        "total_throughput_mbps": sum(throughput.values()),
        "min_sla_pct": min(sla.values()),
        "mean_sla_pct": float(np.mean(list(sla.values()))),
        "mean_offered_load_satisfaction_pct": float(np.mean(list(offered.values()))),
        "mean_pdr_pct": float(np.mean(list(pdr.values()))),
        "urllc_delay_ms": safe_float(by_slice["URLLC"].get("delay_ms_mean")),
        "throughput_embb_mbps": throughput["eMBB"],
        "throughput_urllc_mbps": throughput["URLLC"],
        "throughput_mtc_mbps": throughput["MTC"],
        "sla_embb_pct": sla["eMBB"],
        "sla_urllc_pct": sla["URLLC"],
        "sla_mtc_pct": sla["MTC"],
    }


def load_meta_evaluations(meta_root: Path) -> pd.DataFrame:
    paths = sorted(meta_root.rglob("metaheuristic_results_*_seed*.csv"))
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame[frame["scenario"].isin(SCENARIOS)].copy()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No completed meta-heuristic CSVs under {meta_root}")
    data = pd.concat(frames, ignore_index=True)
    data["method"] = data["method"].str.lower()
    data["evaluation_id"] = data["evaluation_id"].astype(int)
    data["seed"] = data["seed"].astype(int)
    return data.sort_values(["scenario", "seed", "method", "evaluation_id"])


def validate_completed_meta(meta: pd.DataFrame) -> None:
    problems: list[str] = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            for method in META_METHODS:
                count = len(
                    meta[
                        (meta["scenario"] == scenario)
                        & (meta["seed"] == seed)
                        & (meta["method"] == method)
                    ]
                )
                expected = EXPECTED_EVALUATIONS[method]
                if count != expected:
                    problems.append(
                        f"{scenario}/seed={seed}/{method}: {count} evaluations (expected {expected})"
                    )
    if problems:
        raise RuntimeError("Incomplete selected scenario results:\n  " + "\n  ".join(problems))


def load_final_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """Load the network summary for each method's best candidate per seed."""
    best_idx = meta.groupby(["scenario", "seed", "method"])["score"].idxmax()
    best = meta.loc[best_idx].copy()
    records: list[dict[str, object]] = []
    for row in best.itertuples(index=False):
        summary_path = Path(row.result_dir) / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing best-candidate summary: {summary_path}")
        kpis = parse_summary(summary_path)
        score_delta = abs(float(row.score) - kpis["objective"])
        if score_delta > 1e-4:
            raise ValueError(
                f"Stored/rescored objective mismatch ({score_delta:.6g}) for {summary_path}"
            )
        records.append(
            {
                "scenario": row.scenario,
                "seed": int(row.seed),
                "method": row.method,
                "label": LABELS[row.method],
                "family": FAMILIES[row.method],
                "evaluation_id": int(row.evaluation_id),
                "weight_embb": float(row.weight_embb),
                "weight_urllc": float(row.weight_urllc),
                "weight_mtc": float(row.weight_mtc),
                "result_dir": str(row.result_dir),
                **kpis,
            }
        )
    return pd.DataFrame.from_records(records)


def load_baselines(baseline_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for method in BASELINE_MODES:
            for seed in SEEDS:
                summary_path = (
                    baseline_root
                    / f"scenario={scenario}"
                    / f"mode={method}"
                    / f"seed={seed}_run=1"
                    / "summary.csv"
                )
                if not summary_path.exists():
                    continue
                records.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "method": method,
                        "label": LABELS[method],
                        "family": FAMILIES[method],
                        "evaluation_id": math.nan,
                        "weight_embb": math.nan,
                        "weight_urllc": math.nan,
                        "weight_mtc": math.nan,
                        "result_dir": str(summary_path.parent),
                        **parse_summary(summary_path),
                    }
                )
    return pd.DataFrame.from_records(records)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def method_legend(methods: tuple[str, ...] = METHOD_ORDER) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=MARKERS[method],
            color="none",
            markerfacecolor=COLORS[method],
            markeredgecolor="white" if method != "pure_pf" else "#222222",
            markeredgewidth=0.7,
            markersize=6.5,
            label=LABELS[method],
        )
        for method in methods
    ]


def plot_objective_comparison(all_results: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 3.35), sharey=True)
    seed_offsets = {1: -0.13, 2: 0.0, 3: 0.13}
    for ax, scenario in zip(axes, SCENARIOS):
        subset = all_results[all_results["scenario"] == scenario]
        for x, method in enumerate(METHOD_ORDER):
            values = subset[subset["method"] == method].sort_values("seed")
            for row in values.itertuples(index=False):
                ax.scatter(
                    x + seed_offsets[int(row.seed)],
                    row.objective,
                    s=18,
                    marker=MARKERS[method],
                    color=COLORS[method],
                    edgecolor="white" if method != "pure_pf" else "#222222",
                    linewidth=0.45,
                    zorder=3,
                )
            if not values.empty:
                mean = values["objective"].mean()
                ax.plot([x - 0.24, x + 0.24], [mean, mean], color="#111111", lw=1.25, zorder=4)
        ax.axvline(2.5, color="#B5B5B5", lw=0.8)
        ax.axvline(4.5, color="#B5B5B5", lw=0.8)
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([LABELS[m] for m in METHOD_ORDER], rotation=62, ha="right")
        ax.set_xlim(-0.55, len(METHOD_ORDER) - 0.45)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Composite objective $F(\\mathbf{w})$ (higher is better)")
    fig.suptitle("The composite objective measures multi-SLA balance—not universal scheduler quality", y=1.01)
    fig.text(
        0.5,
        -0.03,
        "Dots: individual ns-3 seeds; black segment: mean. Congestion AQPS has n=2 (seed 1 failed).",
        ha="center",
        fontsize=7.4,
        color="#444444",
    )
    fig.subplots_adjust(wspace=0.10, bottom=0.30)
    save_figure(fig, output_dir, "fig01_objective_vs_baselines")


def best_so_far_curves(meta: pd.DataFrame, scenario: str, method: str) -> np.ndarray:
    curves: list[np.ndarray] = []
    for seed in SEEDS:
        values = (
            meta[
                (meta["scenario"] == scenario)
                & (meta["seed"] == seed)
                & (meta["method"] == method)
            ]
            .sort_values("evaluation_id")["score"]
            .to_numpy(dtype=float)
        )
        curves.append(np.maximum.accumulate(values))
    min_len = min(map(len, curves))
    return np.vstack([curve[:min_len] for curve in curves])


def plot_convergence(meta: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.75), sharex=True)
    for ax, scenario in zip(axes, SCENARIOS):
        for method in META_METHODS:
            curves = best_so_far_curves(meta, scenario, method)
            x = np.arange(1, curves.shape[1] + 1)
            mean = curves.mean(axis=0)
            ax.fill_between(
                x,
                curves.min(axis=0),
                curves.max(axis=0),
                color=COLORS[method],
                alpha=0.10,
                linewidth=0,
            )
            ax.plot(x, mean, color=COLORS[method], lw=1.65, label=LABELS[method])
        ax.axvline(72, color="#555555", ls=(0, (3, 2)), lw=0.8)
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_xlim(1, 84)
        ax.set_xlabel("Objective evaluations")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Best-so-far $F(\\mathbf{w})$")
    axes[-1].text(
        72.8,
        0.03,
        "72-eval budget",
        rotation=90,
        transform=axes[-1].get_xaxis_transform(),
        va="bottom",
        fontsize=6.8,
        color="#555555",
    )
    fig.legend(
        handles=method_legend(META_METHODS),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
    )
    fig.suptitle("Search efficiency and final solution quality", y=1.14)
    fig.text(
        0.5,
        -0.04,
        "Lines: mean across 3 seeds; bands: observed seed range. Hybrid uses 84 evaluations; GA/PSO/SA use 72.",
        ha="center",
        fontsize=7.4,
        color="#444444",
    )
    fig.subplots_adjust(wspace=0.24, top=0.80, bottom=0.23)
    save_figure(fig, output_dir, "fig02_best_so_far_convergence")


def compute_gain_over_best_pure(all_results: pd.DataFrame) -> pd.DataFrame:
    pure = all_results[all_results["method"].isin(BASELINE_MODES[:3])]
    reference = (
        pure.groupby(["scenario", "seed"], as_index=False)["objective"]
        .max()
        .rename(columns={"objective": "best_pure_objective"})
    )
    compared = all_results[all_results["method"].isin(COMPARISON_METHODS)].merge(
        reference, on=["scenario", "seed"], how="inner"
    )
    compared["gain"] = compared["objective"] - compared["best_pure_objective"]
    return compared


def plot_gain_over_pure(all_results: pd.DataFrame, output_dir: Path) -> None:
    gains = compute_gain_over_best_pure(all_results)
    groups = (
        ("Slice-aware baselines", ("slice_aqps", "slice_meta_risk_elastic")),
        ("Meta-heuristic weight optimization", META_METHODS),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.05), sharey=True)
    x = np.arange(len(SCENARIOS))
    seed_offsets = {1: -0.055, 2: 0.0, 3: 0.055}
    for ax, (title, methods) in zip(axes, groups):
        for method in methods:
            means: list[float] = []
            for scenario_index, scenario in enumerate(SCENARIOS):
                values = gains[
                    (gains["scenario"] == scenario) & (gains["method"] == method)
                ].sort_values("seed")
                means.append(values["gain"].mean())
                for row in values.itertuples(index=False):
                    ax.scatter(
                        scenario_index + seed_offsets[int(row.seed)],
                        row.gain,
                        s=21,
                        color=COLORS[method],
                        marker=MARKERS[method],
                        edgecolor="white",
                        linewidth=0.45,
                        alpha=0.72,
                        zorder=3,
                    )
            ax.plot(
                x,
                means,
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=5.5,
                lw=1.6,
                label=LABELS[method],
                zorder=4,
            )
        ax.axhline(0, color="#222222", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="best")
    axes[0].set_ylabel("Objective gain over best pure scheduler (points)")
    fig.suptitle("Gains are specific to the composite multi-SLA objective", y=1.02)
    fig.text(
        0.5,
        -0.04,
        "Reference is max(RR, PF, BCQI) per scenario/seed. Pure schedulers were not tuned to maximize this objective.",
        ha="center",
        fontsize=7.4,
        color="#444444",
    )
    fig.subplots_adjust(wspace=0.12, bottom=0.22)
    save_figure(fig, output_dir, "fig03_gain_over_best_pure_scheduler")


def pareto_mask(points: np.ndarray, maximize: tuple[bool, bool]) -> np.ndarray:
    """Return non-dominated points for two objectives."""
    transformed = points.copy()
    for column, should_maximize in enumerate(maximize):
        if not should_maximize:
            transformed[:, column] *= -1
    efficient = np.ones(len(transformed), dtype=bool)
    for i, point in enumerate(transformed):
        dominates_i = np.all(transformed >= point, axis=1) & np.any(
            transformed > point, axis=1
        )
        efficient[i] = not dominates_i.any()
    return efficient


def annotate_points(ax: plt.Axes, means: pd.DataFrame, x_col: str, y_col: str) -> None:
    # Panel-specific offsets keep labels readable around the dense optimized
    # solutions. They are expressed in display points, not data units.
    if x_col == "total_throughput_mbps":
        offsets = {
            "pure_rr": (-16, -10),
            "pure_pf": (4, 3),
            "pure_bcqi": (4, 3),
            "slice_aqps": (-17, -11),
            "slice_meta_risk_elastic": (-41, 3),
            "ga": (5, 4),
            "pso": (-18, 4),
            "sa": (5, 4),
            "hybrid": (5, -11),
        }
    else:
        offsets = {
            "pure_rr": (5, 3),
            "pure_pf": (5, -11),
            "pure_bcqi": (5, 3),
            "slice_aqps": (-34, -11),
            "slice_meta_risk_elastic": (5, 3),
            "ga": (5, 4),
            "pso": (5, -11),
            "sa": (-16, 4),
            "hybrid": (-31, -11),
        }
    for row in means.itertuples(index=False):
        dx, dy = offsets[row.method]
        ax.annotate(
            LABELS[row.method],
            (getattr(row, x_col), getattr(row, y_col)),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.9,
            color="#222222",
        )


def plot_congestion_tradeoffs(all_results: pd.DataFrame, output_dir: Path) -> None:
    data = all_results[all_results["scenario"] == "congestion"].copy()
    means = data.groupby("method", as_index=False).agg(
        total_throughput_mbps=("total_throughput_mbps", "mean"),
        min_sla_pct=("min_sla_pct", "mean"),
        urllc_delay_ms=("urllc_delay_ms", "mean"),
        objective=("objective", "mean"),
    )
    means = means[means["method"].isin(METHOD_ORDER)]

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.20))
    panels = (
        (
            "total_throughput_mbps",
            "min_sla_pct",
            "Aggregate throughput (Mbps)",
            "Worst-slice SLA satisfaction (%)",
            (True, True),
            "Throughput–isolation trade-off",
        ),
        (
            "urllc_delay_ms",
            "objective",
            "URLLC mean delay (ms; lower is better)",
            "Composite objective $F(\\mathbf{w})$",
            (False, True),
            "Latency–objective trade-off",
        ),
    )
    for ax, (x_col, y_col, x_label, y_label, maximize, title) in zip(axes, panels):
        for method in METHOD_ORDER:
            values = data[data["method"] == method]
            ax.scatter(
                values[x_col],
                values[y_col],
                s=16,
                color=COLORS[method],
                marker=MARKERS[method],
                alpha=0.28,
                edgecolor="none",
                zorder=2,
            )
            mean_row = means[means["method"] == method]
            if mean_row.empty:
                continue
            ax.scatter(
                mean_row[x_col],
                mean_row[y_col],
                s=53,
                color=COLORS[method],
                marker=MARKERS[method],
                edgecolor="white" if method != "pure_pf" else "#222222",
                linewidth=0.7,
                zorder=4,
            )
        points = means[[x_col, y_col]].to_numpy(dtype=float)
        front = means[pareto_mask(points, maximize)].sort_values(x_col)
        ax.plot(
            front[x_col],
            front[y_col],
            color="#222222",
            lw=0.9,
            ls=(0, (2, 2)),
            zorder=1,
        )
        annotate_points(ax, means, x_col, y_col)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(axis="both", visible=True)
    axes[1].axvline(10.0, color="#B2182B", lw=1.0, ls="--", zorder=1)
    axes[1].text(
        10.1,
        0.02,
        "10 ms URLLC target",
        rotation=90,
        transform=axes[1].get_xaxis_transform(),
        va="bottom",
        color="#8A1421",
        fontsize=7.0,
    )
    fig.suptitle("Congestion: optimization navigates competing network objectives", y=1.02)
    fig.text(
        0.5,
        -0.04,
        "Large markers: seed means; translucent markers: individual seeds; dotted line: non-dominated mean solutions.",
        ha="center",
        fontsize=7.4,
        color="#444444",
    )
    fig.subplots_adjust(wspace=0.33, bottom=0.22)
    save_figure(fig, output_dir, "fig04_congestion_tradeoffs")


def plot_slice_sla_heatmap(all_results: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 4.25), sharey=True)
    image = None
    for ax, scenario in zip(axes, SCENARIOS):
        subset = all_results[all_results["scenario"] == scenario]
        matrix = np.full((len(METHOD_ORDER), len(SLICE_NAMES)), np.nan)
        for row_index, method in enumerate(METHOD_ORDER):
            method_data = subset[subset["method"] == method]
            for col_index, column in enumerate(
                ("sla_embb_pct", "sla_urllc_pct", "sla_mtc_pct")
            ):
                matrix[row_index, col_index] = method_data[column].mean()
        image = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                if np.isnan(value):
                    label = "NA"
                    color = "#444444"
                else:
                    label = f"{value:.0f}"
                    color = "white" if value >= 62 else "#222222"
                ax.text(
                    col_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=color,
                )
        ax.set_xticks(range(len(SLICE_NAMES)))
        ax.set_xticklabels(SLICE_NAMES)
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.grid(False)
    axes[0].set_yticks(range(len(METHOD_ORDER)))
    axes[0].set_yticklabels([LABELS[m] for m in METHOD_ORDER])
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, fraction=0.024, pad=0.025)
        cbar.set_label("SLA satisfaction (%)")
    fig.suptitle("Slice-level SLA satisfaction exposes the effect of congestion", y=0.99)
    fig.text(
        0.5,
        0.02,
        "Cell values are seed means. Congestion AQPS is based on n=2; all other cells use n=3.",
        ha="center",
        fontsize=7.4,
        color="#444444",
    )
    fig.subplots_adjust(left=0.14, right=0.91, top=0.88, bottom=0.12, wspace=0.08)
    save_figure(fig, output_dir, "fig05_slice_sla_heatmap")


def simplex_xy(embb: np.ndarray, urllc: np.ndarray, mtc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = embb + urllc + mtc
    embb = embb / total
    urllc = urllc / total
    mtc = mtc / total
    x = urllc + 0.5 * mtc
    y = (np.sqrt(3.0) / 2.0) * mtc
    return x, y


def draw_simplex(ax: plt.Axes) -> None:
    height = np.sqrt(3.0) / 2.0
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, height], [0.0, 0.0]])
    ax.plot(triangle[:, 0], triangle[:, 1], color="#333333", lw=0.9)
    for fraction in (0.25, 0.50, 0.75):
        # Constant MTC.
        y = height * fraction
        ax.plot([0.5 * fraction, 1 - 0.5 * fraction], [y, y], color="#DDDDDD", lw=0.55)
        # Constant eMBB and constant URLLC.
        ax.plot(
            [1 - fraction, 0.5 * (1 - fraction)],
            [0, height * (1 - fraction)],
            color="#DDDDDD",
            lw=0.55,
        )
        ax.plot(
            [fraction, 0.5 + 0.5 * fraction],
            [0, height * (1 - fraction)],
            color="#DDDDDD",
            lw=0.55,
        )
    ax.text(-0.035, -0.035, "eMBB", ha="right", va="top", fontsize=8.0)
    ax.text(1.035, -0.035, "URLLC", ha="left", va="top", fontsize=8.0)
    ax.text(0.5, height + 0.035, "MTC", ha="center", va="bottom", fontsize=8.0)
    ax.set_xlim(-0.09, 1.09)
    ax.set_ylim(-0.08, height + 0.10)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_weight_simplex(final_meta: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.70))
    seed_markers = {1: "o", 2: "s", 3: "^"}
    for ax, scenario in zip(axes, SCENARIOS):
        draw_simplex(ax)
        subset = final_meta[final_meta["scenario"] == scenario]
        for method in META_METHODS:
            values = subset[subset["method"] == method]
            for row in values.itertuples(index=False):
                x, y = simplex_xy(
                    np.array([row.weight_embb]),
                    np.array([row.weight_urllc]),
                    np.array([row.weight_mtc]),
                )
                ax.scatter(
                    x,
                    y,
                    s=25,
                    marker=seed_markers[int(row.seed)],
                    color=COLORS[method],
                    edgecolor="white",
                    linewidth=0.55,
                    alpha=0.82,
                    zorder=3,
                )
            mean_weights = values[["weight_embb", "weight_urllc", "weight_mtc"]].mean()
            x_mean, y_mean = simplex_xy(
                np.array([mean_weights["weight_embb"]]),
                np.array([mean_weights["weight_urllc"]]),
                np.array([mean_weights["weight_mtc"]]),
            )
            ax.scatter(
                x_mean,
                y_mean,
                s=68,
                marker="*",
                color=COLORS[method],
                edgecolor="#222222",
                linewidth=0.45,
                zorder=4,
            )
        ax.set_title(SCENARIO_LABELS[scenario], pad=0)
    method_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[m], markersize=5.8, label=LABELS[m])
        for m in META_METHODS
    ]
    seed_handles = [
        Line2D([0], [0], marker=seed_markers[s], color="none", markerfacecolor="#777777", markersize=5.2, label=f"Seed {s}")
        for s in SEEDS
    ]
    mean_handle = Line2D([0], [0], marker="*", color="none", markerfacecolor="#777777", markeredgecolor="#222222", markersize=8, label="Method mean")
    fig.legend(
        handles=method_handles + seed_handles + [mean_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=8,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.suptitle("Best allocation weights occupy different regions of the simplex", y=1.02)
    fig.subplots_adjust(wspace=0.16, bottom=0.23, top=0.86)
    save_figure(fig, output_dir, "fig06_best_weight_simplex")



def plot_congestion_kpi_profile(all_results: pd.DataFrame, output_dir: Path) -> None:
    """Contrast native scheduler outcomes with the paper's multi-SLA target."""
    data = all_results[all_results["scenario"] == "congestion"].copy()
    metrics = (
        ("total_throughput_mbps", "Aggregate throughput\n(Mbps; higher is better)"),
        ("min_sla_pct", "Worst-slice SLA\n(%; higher is better)"),
        ("urllc_delay_ms", "URLLC mean delay\n(ms; lower is better)"),
        ("objective", "Composite objective $F(\\mathbf{w})$\n(higher is better)"),
    )
    family_bands = (
        (-0.5, 2.5, "#F1F1F1", "Pure schedulers"),
        (2.5, 4.5, "#FFF3D6", "Slice-aware baselines"),
        (4.5, 8.5, "#EAF3FA", "Optimized weights"),
    )
    seed_offsets = {1: -0.12, 2: 0.0, 3: 0.12}
    y_positions = np.arange(len(METHOD_ORDER))
    fig, axes = plt.subplots(1, 4, figsize=(7.35, 4.25), sharey=True)
    for ax, (metric, xlabel) in zip(axes, metrics):
        for low, high, color, _ in family_bands:
            ax.axhspan(low, high, color=color, zorder=0)
        for y, method in enumerate(METHOD_ORDER):
            values = data[data["method"] == method].sort_values("seed")
            for row in values.itertuples(index=False):
                ax.scatter(
                    getattr(row, metric),
                    y + seed_offsets[int(row.seed)],
                    s=18,
                    marker="o",
                    color=COLORS[method],
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.68,
                    zorder=3,
                )
            if not values.empty:
                ax.scatter(
                    values[metric].mean(),
                    y,
                    s=39,
                    marker="D",
                    color=COLORS[method],
                    edgecolor="#222222",
                    linewidth=0.55,
                    zorder=4,
                )
        ax.axhline(2.5, color="white", lw=1.6, zorder=1)
        ax.axhline(4.5, color="white", lw=1.6, zorder=1)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([LABELS[m] for m in METHOD_ORDER])
    axes[0].invert_yaxis()
    axes[2].axvline(10.0, color="#B2182B", lw=1.0, ls="--", zorder=2)
    axes[2].text(
        10.25,
        0.98,
        "10 ms target",
        rotation=90,
        transform=axes[2].get_xaxis_transform(),
        va="top",
        color="#8A1421",
        fontsize=6.8,
    )
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markersize=4.8, label="Individual seed"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#777777", markeredgecolor="#222222", markersize=5.3, label="Seed mean"),
    ] + [Patch(facecolor=color, edgecolor="none", label=label) for _, _, color, label in family_bands]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    fig.suptitle("Congestion: different policies optimize different notions of utility", y=0.99)
    fig.text(
        0.5,
        0.045,
        "BCQI emphasizes channel efficiency/throughput; meta-heuristics tune inter-slice weights for multi-SLA balance (intra-slice PF).",
        ha="center",
        fontsize=7.2,
        color="#444444",
    )
    fig.subplots_adjust(left=0.12, right=0.99, top=0.89, bottom=0.22, wspace=0.28)
    save_figure(fig, output_dir, "fig07_congestion_kpi_profile")
def write_summary_files(
    all_results: pd.DataFrame,
    meta: pd.DataFrame,
    final_meta: pd.DataFrame,
    baseline: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "scenario",
        "seed",
        "method",
        "label",
        "family",
        "objective",
        "total_throughput_mbps",
        "min_sla_pct",
        "mean_sla_pct",
        "urllc_delay_ms",
        "throughput_embb_mbps",
        "throughput_urllc_mbps",
        "throughput_mtc_mbps",
        "sla_embb_pct",
        "sla_urllc_pct",
        "sla_mtc_pct",
        "weight_embb",
        "weight_urllc",
        "weight_mtc",
        "evaluation_id",
        "result_dir",
    ]
    all_results[columns].sort_values(["scenario", "method", "seed"]).to_csv(
        output_dir / "figure_source_data.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )

    counts = (
        meta.groupby(["scenario", "seed", "method"]).size().rename("evaluations").reset_index()
    )
    counts.to_csv(output_dir / "metaheuristic_evaluation_counts.csv", index=False)

    means = (
        all_results.groupby(["scenario", "method", "label", "family"], as_index=False)
        .agg(
            n=("seed", "count"),
            objective_mean=("objective", "mean"),
            objective_sd=("objective", "std"),
            total_throughput_mean=("total_throughput_mbps", "mean"),
            min_sla_mean=("min_sla_pct", "mean"),
            urllc_delay_mean=("urllc_delay_ms", "mean"),
        )
        .sort_values(["scenario", "objective_mean"], ascending=[True, False])
    )
    means.to_csv(output_dir / "descriptive_summary.csv", index=False)

    gains = compute_gain_over_best_pure(all_results)
    gain_means = (
        gains.groupby(["scenario", "method"])["gain"].agg(["count", "mean", "std"]).reset_index()
    )
    gain_means.to_csv(output_dir / "gain_over_best_pure_summary.csv", index=False)

    best_lines: list[str] = []
    for scenario in SCENARIOS:
        meta_means = means[
            (means["scenario"] == scenario) & (means["method"].isin(META_METHODS))
        ].sort_values("objective_mean", ascending=False)
        winner = meta_means.iloc[0]
        best_lines.append(
            f"- **{SCENARIO_LABELS[scenario]}:** highest meta-heuristic mean = "
            f"{winner['label']} ({winner['objective_mean']:.2f} ± "
            f"{winner['objective_sd']:.2f}, n={int(winner['n'])})."
        )

    readme = f"""# Publication figures: completed scenarios

These figures use the completed `low_traffic`, `normal`, and `congestion`
results from run tag `20260626_122326`. All plot text and legends are in
English. PNG files are rendered at 450 DPI; PDF and SVG files are vector
versions suitable for typesetting.

## Figure set

1. `fig01_objective_vs_baselines`: direct comparison of pure schedulers,
   slice-aware baselines, and the best solution found by each meta-heuristic.
2. `fig02_best_so_far_convergence`: mean best-so-far curves with the observed
   seed range; the 72-evaluation reference makes the hybrid's extra local
   refinement budget explicit.
3. `fig03_gain_over_best_pure_scheduler`: paired gain over the strongest pure
   scheduler for each scenario and seed.
4. `fig04_congestion_tradeoffs`: domain-level trade-offs under the only load
   regime that materially separates the policies.
5. `fig05_slice_sla_heatmap`: per-slice SLA satisfaction, showing where
   congestion is absorbed.
6. `fig06_best_weight_simplex`: best allocation weights for all methods and
   seeds, without hiding seed-to-seed variability behind a stacked mean.
7. `fig07_congestion_kpi_profile`: raw KPI profile showing that pure schedulers
   can lead in their native priority while optimized weights lead in SLA balance.

## Interpretation framework

RR, PF, and BCQI are operating schedulers, not optimizers of the paper's
composite objective. RR emphasizes equal scheduling opportunity, BCQI favors
users with stronger channels and therefore channel efficiency/throughput, and
PF balances instantaneous rate with historical service. In contrast, GA, PSO,
SA, and Hybrid search for inter-slice weights that maximize the multi-SLA
objective while PF remains the intra-slice scheduler. A higher composite score
therefore means better alignment with the stated multi-SLA objective; it does
not mean that the method dominates every pure scheduler on every raw KPI.

## Descriptive result (not a significance claim)

{chr(10).join(best_lines)}

## Required caveats

- The experimental unit is the ns-3 seed. With only three seeds, the figures
  support descriptive comparisons, not formal statistical significance.
- Congestion/AQPS uses two successful seeds because seed 1 failed; the missing
  run is shown as reduced sample size rather than imputed.
- Low-traffic and normal-load policies may visually overlap because the system
  is not resource-limited there. This is a meaningful negative result.
- The composite objective includes SLA satisfaction, offered-load satisfaction,
  PDR, utilization, and explicit penalties. It must not be labeled simply as
  throughput, fairness, or latency.

## Reproduction

```bash
cd {PROJECT_ROOT}
python3 examples/generate_completed_scenario_figures.py
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results.resolve()
    output_dir = args.output.resolve()
    meta_root = results_root / "metaheuristics"
    baseline_root = results_root / "heuristics_ns3/results_rslaq_network_only"

    configure_style()
    meta = load_meta_evaluations(meta_root)
    validate_completed_meta(meta)
    final_meta = load_final_meta(meta)
    baseline = load_baselines(baseline_root)
    all_results = pd.concat([baseline, final_meta], ignore_index=True)

    plot_objective_comparison(all_results, output_dir)
    plot_convergence(meta, output_dir)
    plot_gain_over_pure(all_results, output_dir)
    plot_congestion_tradeoffs(all_results, output_dir)
    plot_slice_sla_heatmap(all_results, output_dir)
    plot_weight_simplex(final_meta, output_dir)
    plot_congestion_kpi_profile(all_results, output_dir)
    write_summary_files(all_results, meta, final_meta, baseline, output_dir)

    print(f"Validated meta-heuristic evaluations: {len(meta):,}")
    print(f"Final meta-heuristic solutions: {len(final_meta):,}")
    print(f"Baseline runs: {len(baseline):,}")
    print(f"Figures and source tables: {output_dir}")


if __name__ == "__main__":
    main()
