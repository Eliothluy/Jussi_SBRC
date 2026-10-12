#!/usr/bin/env python3
"""Generate figures for the M.Sc. thesis (low_traffic, normal, congestion).

All figures are in English and saved as PNG (300 DPI) ready for Overleaf in
/home/elioth/Documentos/artigo_jussi/figuras_overleaf/.

How to run:
    cd /home/elioth/Documentos/artigo_jussi/ns-o-ran-gym
    .venv-dashboard/bin/python examples/gerar_figuras.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display (background)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
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

# Pure baseline schedulers (for the core comparison)
PURE_MODES = ["pure_rr", "pure_pf", "pure_bcqi"]
PURE_LABELS = {"pure_rr": "RR", "pure_pf": "PF", "pure_bcqi": "BCQI"}
PURE_COLORS = {"pure_rr": "#7f7f7f", "pure_pf": "#bcbd22", "pure_bcqi": "#17becf"}

# Style (clean, white background, readable)
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
from nsoran.scoring import read_summary, score_summary_rows  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_meta_evals() -> pd.DataFrame:
    """Load all candidate.json sidecars (excluding tolerated crashes)."""
    rows = []
    for sidecar in META_ROOT.rglob("candidate.json"):
        try:
            p = json.loads(sidecar.read_text())
            if p.get("failed"):
                continue
            if p.get("scenario") not in SCENARIOS:
                continue
            rows.append({
                "method": p["method"],
                "evaluation_id": int(p["evaluation_id"]),
                "scenario": p["scenario"],
                "seed": int(p["seed"]),
                "score": float(p["score"]),
                "w_embb": float(p["weights"][0]),
                "w_urllc": float(p["weights"][1]),
                "w_mtc": float(p["weights"][2]),
            })
        except (KeyError, ValueError):
            continue
    return pd.DataFrame(rows)


def load_throughput_data(modes: list[str]) -> pd.DataFrame:
    """Extract throughput per slice + total throughput + score for given modes."""
    rows = []
    for sc in SCENARIOS:
        for mode in modes:
            for seed in SEEDS:
                summary = BASELINE_ROOT / f"scenario={sc}/mode={mode}/seed={seed}_run=1/summary.csv"
                if not summary.exists():
                    continue
                try:
                    data = read_summary(summary)
                    by = {r.get("slice", "?"): r for r in data}
                    thr = {s: float(by[s].get("throughput_mbps_mean", 0)) for s in ["eMBB", "URLLC", "MTC"]}
                    sla = {s: float(by[s].get("sla_satisfaction_pct", 0)) / 100.0 for s in ["eMBB", "URLLC", "MTC"]}
                    total_thr = sum(thr.values())
                    score = score_summary_rows(data)
                    rows.append({
                        "scenario": sc, "mode": mode, "seed": seed,
                        "total_thr": total_thr, "thr_embb": thr["eMBB"],
                        "thr_urllc": thr["URLLC"], "thr_mtc": thr["MTC"],
                        "sla_embb": sla["eMBB"], "sla_urllc": sla["URLLC"], "sla_mtc": sla["MTC"],
                        "min_sla": min(sla.values()), "score": score,
                    })
                except Exception:
                    continue
    return pd.DataFrame(rows)


def load_best_candidates() -> pd.DataFrame:
    """Load official best_candidate_*.json files."""
    rows = []
    for sidecar in META_ROOT.rglob("best_candidate_*_seed*.json"):
        try:
            p = json.loads(sidecar.read_text())
            if p.get("scenario") not in SCENARIOS:
                continue
            w = p.get("weights", {})
            rows.append({
                "scenario": p["scenario"],
                "seed": int(p["seed"]),
                "method": p["method"],
                "score": float(p["score"]),
                "w_embb": float(w.get("eMBB", 0)),
                "w_urllc": float(w.get("URLLC", 0)),
                "w_mtc": float(w.get("MTC", 0)),
            })
        except (KeyError, ValueError):
            continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_convergencia(df: pd.DataFrame) -> None:
    """Fig 1: Convergence curves (cumulative best) per method, 3 scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, sc in zip(axes, SCENARIOS):
        for method in METHODS:
            sub = df[(df.scenario == sc) & (df.method == method)].copy()
            if sub.empty:
                continue
            curves = []
            for seed in SEEDS:
                s = sub[sub.seed == seed].sort_values("evaluation_id")
                if s.empty:
                    continue
                curves.append(s["score"].cummax().values)
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            mean_curve = np.mean([c[:min_len] for c in curves], axis=0)
            x = np.arange(1, min_len + 1)
            ax.plot(x, mean_curve, label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method], linewidth=2)
        ax.set_title(SCENARIO_LABELS[sc])
        ax.set_xlabel("Evaluation number")
        if ax == axes[0]:
            ax.set_ylabel("Cumulative best score $F(\\mathbf{w})$")
        ax.legend(loc="lower right")
    fig.suptitle("Convergence of meta-heuristics (mean of 3 seeds, cumulative best)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(FIG_DIR / "fig01_convergencia.png")
    plt.close(fig)
    print("  + fig01_convergencia.png")


def fig_distribuicao_scores(df: pd.DataFrame) -> None:
    """Fig 2: Boxplot of score distribution per method and scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, sc in zip(axes, SCENARIOS):
        sub = df[df.scenario == sc]
        data = []
        labels = []
        colors = []
        for m in METHODS:
            d = sub[sub.method == m]["score"].values
            if len(d) > 0:
                data.append(d)
                labels.append(METHOD_LABELS[m])
                colors.append(METHOD_COLORS[m])
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                        showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white",
                                                   "markeredgecolor": "black", "markersize": 5})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(SCENARIO_LABELS[sc])
        ax.set_ylabel("Score $F(\\mathbf{w})$" if ax == axes[0] else "")
    fig.suptitle("Score distribution per meta-heuristic and scenario (3 seeds, diamond = mean)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(FIG_DIR / "fig02_distribuicao_scores.png")
    plt.close(fig)
    print("  + fig02_distribuicao_scores.png")


def fig_comparacao_baselines(meta: pd.DataFrame, tf: pd.DataFrame) -> None:
    """Fig 3: Score comparison — pure schedulers vs. meta-heuristics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, sc in zip(axes, SCENARIOS):
        scores = {}
        for mode in PURE_MODES:
            sub = tf[(tf.scenario == sc) & (tf.mode == mode)]
            if not sub.empty:
                scores[PURE_LABELS[mode]] = sub["score"].mean()
        for m in METHODS:
            sub = meta[(meta.scenario == sc) & (meta.method == m)]
            if sub.empty:
                continue
            maxs = sub.groupby("seed")["score"].max()
            scores[METHOD_LABELS[m]] = maxs.mean()

        names = list(scores.keys())
        values = list(scores.values())
        n_pure = len([m for m in PURE_MODES if PURE_LABELS[m] in scores])
        colors_bar = [PURE_COLORS[m] for m in PURE_MODES if PURE_LABELS[m] in scores] + \
                     [METHOD_COLORS[m] for m in METHODS if METHOD_LABELS[m] in scores]
        x = np.arange(len(names))
        ax.bar(x, values, color=colors_bar, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=0, fontsize=9)
        ax.set_title(SCENARIO_LABELS[sc])
        if ax == axes[0]:
            ax.set_ylabel("Mean score $F(\\mathbf{w})$")
        ax.axvline(x=n_pure - 0.5, color="gray", linestyle="--", alpha=0.5)
    fig.suptitle("Score comparison: pure schedulers vs. meta-heuristics (mean of 3 seeds)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(FIG_DIR / "fig03_comparacao_baselines.png")
    plt.close(fig)
    print("  + fig03_comparacao_baselines.png")


def _nondominated_mask(xs: np.ndarray, ys: np.ndarray) -> list[int]:
    """Return indices of points non-dominated when MAXIMIZING both x and y."""
    n = len(xs)
    out = []
    for i in range(n):
        dom = False
        for j in range(n):
            if i == j:
                continue
            if xs[j] >= xs[i] and ys[j] >= ys[i] and (xs[j] > xs[i] or ys[j] > ys[i]):
                dom = True
                break
        if not dom:
            out.append(i)
    return out


# Distinct filled markers so the 4 meta-heuristics remain visually separable
# even when their throughput values cluster tightly.
META_MARKERS = {"ga": "*", "pso": "D", "sa": "^", "hybrid": "P"}
META_MARKER_SIZES = {"ga": 330, "pso": 230, "sa": 260, "hybrid": 280}
# Purely visual horizontal jitter applied around a meta-heuristic's REAL
# throughput so overlapping markers separate. It is scaled to 4% of the
# per-panel throughput range so it stays honest: large when methods differ a
# lot (congestion), negligible when they collapse (low load). NEVER used for
# the Pareto computation, which runs on real (unjittered) values.
META_VISUAL_JITTER_FRACTION = 0.04


def _jain_index(vals: list[float]) -> float:
    """Jain fairness index for >=1 non-negative values. Returns 0 for all-zero."""
    n = len(vals)
    s = sum(vals)
    sq = sum(v * v for v in vals)
    if s <= 0.0 or sq <= 0.0:
        return 0.0
    return (s * s) / (n * sq)


def _load_pareto_points() -> dict[str, list[dict]]:
    """Build the 7 (label, throughput, Jain) points per scenario.

    For each scenario returns a list of dicts (one per method) with keys:
        label, throughput, jain, kind ("pure" | "meta"), key (mode or method)

    Pure schedulers (RR, PF, BCQI): throughput_total and Jain computed from the
    pure_* summary.csv, averaged over the 3 seeds.

    Meta-heuristics (ga, pso, sa, hybrid): for each seed, pick the eval with the
    highest score (failed:true excluded), read its slice_custom summary.csv and
    compute throughput_total + Jain, then average those per-seed values over the
    3 seeds. This uses REAL throughput from the best eval, not a proxy.
    """
    out: dict[str, list[dict]] = {sc: [] for sc in SCENARIOS}
    slices = ["eMBB", "URLLC", "MTC"]

    for sc in SCENARIOS:
        # --- Pure schedulers ---
        for mode in PURE_MODES:
            totals = []
            jains = []
            for sd in SEEDS:
                summary = BASELINE_ROOT / f"scenario={sc}/mode={mode}/seed={sd}_run=1/summary.csv"
                if not summary.exists():
                    continue
                data = read_summary(summary)
                by = {r.get("slice", "?"): r for r in data}
                thr = [float(by[s].get("throughput_mbps_mean", 0.0)) for s in slices]
                totals.append(sum(thr))
                jains.append(_jain_index(thr))
            if not totals:
                continue
            out[sc].append({
                "label": PURE_LABELS[mode],
                "throughput": float(np.mean(totals)),
                "jain": float(np.mean(jains)),
                "kind": "pure",
                "key": mode,
            })

        # --- Meta-heuristics: best eval per seed, then average ---
        for method in METHODS:
            totals = []
            jains = []
            for sd in SEEDS:
                base = META_ROOT / f"scenario={sc}/seed={sd}/metaheuristic_search/evals"
                if not base.exists():
                    continue
                best_score = -1e18
                best_dir: Path | None = None
                for sidecar in base.glob(f"{method}_eval_*/candidate.json"):
                    try:
                        p = json.loads(sidecar.read_text())
                    except (ValueError, OSError):
                        continue
                    if p.get("failed"):
                        continue
                    try:
                        sc_eff = float(p["score"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if sc_eff > best_score:
                        best_score = sc_eff
                        best_dir = sidecar.parent
                if best_dir is None:
                    continue
                summary = (best_dir
                           / f"results_rslaq_network_only/scenario={sc}/mode=slice_custom/seed={sd}_run=1/summary.csv")
                if not summary.exists():
                    continue
                data = read_summary(summary)
                by = {r.get("slice", "?"): r for r in data}
                thr = [float(by[s].get("throughput_mbps_mean", 0.0)) for s in slices]
                totals.append(sum(thr))
                jains.append(_jain_index(thr))
            if not totals:
                continue
            out[sc].append({
                "label": METHOD_LABELS[method],
                "throughput": float(np.mean(totals)),
                "jain": float(np.mean(jains)),
                "kind": "meta",
                "key": method,
            })

    return out


def fig_tradeoff_pareto(*_args, **_kwargs) -> None:
    """Fig 4: Throughput vs. fairness (Jain index) Pareto scatter.

    Each of the 3 subplots (Low Traffic / Normal / Congestion) shows 7 points:
      - 3 pure schedulers (RR, PF, BCQI) as filled circles
      - 4 meta-heuristics (GA, PSO, SA, Hybrid) with distinct markers/colors

    X = aggregated throughput across the 3 slices (Mbps); Y = Jain fairness
    index in [0,1]. Both axes are maximized, so the non-dominated set is the
    upper-right frontier. Data points use REAL throughput + Jain computed from
    each method's actual summary.csv (best eval per seed for meta-heuristics).
    A small, purely visual horizontal jitter is applied to the meta markers so
    near-coincident ones stay legible; the Pareto frontier is computed BEFORE
    any jitter.

    The function ignores the legacy ``meta``/``tf`` DataFrame arguments and
    rebuilds the points from disk via :func:`_load_pareto_points`, because the
    Jain index is not present in those tables.
    """
    points_by_sc = _load_pareto_points()

    # Print numeric validation (method, throughput, jain) per scenario.
    print("  Pareto points (label, throughput Mbps, Jain):")
    pareto_summary: dict[str, list[str]] = {}
    for sc in SCENARIOS:
        pts = points_by_sc[sc]
        for pt in pts:
            print(f"    {sc:12s} {pt['label']:7s} thr={pt['throughput']:8.2f}  jain={pt['jain']:.4f}")
        xs_real = np.array([p["throughput"] for p in pts], dtype=float)
        ys_real = np.array([p["jain"] for p in pts], dtype=float)
        nd_idx = _nondominated_mask(xs_real, ys_real)
        pareto_summary[sc] = [pts[i]["label"] for i in
                              sorted(nd_idx, key=lambda k: xs_real[k])]
    for sc in SCENARIOS:
        print(f"    Pareto frontier [{sc}]: {pareto_summary[sc]}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax, sc in zip(axes, SCENARIOS):
        pts = points_by_sc[sc]
        labels: list[str] = [p["label"] for p in pts]
        xs_real = np.array([p["throughput"] for p in pts], dtype=float)
        ys_real = np.array([p["jain"] for p in pts], dtype=float)
        kinds = [p["kind"] for p in pts]
        keys = [p["key"] for p in pts]

        # --- Visual jitter ONLY for plotting position (never for Pareto) ---
        # Scale jitter to the throughput range of THIS panel so it stays honest:
        # large when methods differ (congestion), negligible when they collapse
        # (low load, where all methods return ~identical throughput).
        xs_plot = xs_real.copy()
        meta_positions = [i for i, k in enumerate(kinds) if k == "meta"]
        if len(meta_positions) > 1:
            thr_range = float(xs_real.max() - xs_real.min()) if len(xs_real) > 1 else 1.0
            thr_range = max(thr_range, 1e-6)
            jitter_step = META_VISUAL_JITTER_FRACTION * thr_range
            offsets = np.linspace(-jitter_step, jitter_step, len(meta_positions))
            for k_idx, i in enumerate(meta_positions):
                xs_plot[i] = xs_real[i] + float(offsets[k_idx])

        # --- Pareto frontier computed on REAL (unjittered) values ---
        nd_idx = _nondominated_mask(xs_real, ys_real)
        nd_set = set(nd_idx)
        if len(nd_idx) >= 2:
            nd_sorted = sorted(nd_idx, key=lambda i: xs_real[i])
            ax.plot(xs_real[nd_sorted], ys_real[nd_sorted], "--",
                    color="#d62728", alpha=0.85, linewidth=2.0, zorder=3,
                    label="Pareto frontier" if ax == axes[0] else None)

        # --- Markers ---
        for i, p in enumerate(pts):
            if p["kind"] == "pure":
                marker, color, size = "o", PURE_COLORS[p["key"]], 220
            else:
                marker = META_MARKERS[p["key"]]
                color = METHOD_COLORS[p["key"]]
                size = META_MARKER_SIZES[p["key"]]
            if i in nd_set:
                ax.scatter(xs_plot[i], ys_real[i], marker="o", s=size * 2.6,
                           facecolor="none", edgecolor="#d62728",
                           linewidth=1.8, zorder=3, alpha=0.7)
            ax.scatter(xs_plot[i], ys_real[i], marker=marker, s=size,
                       color=color, edgecolor="black", linewidth=1.1,
                       zorder=4, alpha=0.95)

        # --- Axes headroom so labels fit ---
        x_min, x_max = (xs_real.min(), xs_real.max()) if len(xs_real) else (0.0, 1.0)
        y_min, y_max = (ys_real.min(), ys_real.max()) if len(ys_real) else (0.0, 1.0)
        x_pad = max((x_max - x_min) * 0.18, 2.0)
        y_pad = max((y_max - y_min) * 0.20, 0.03)
        ax.set_xlim(x_min - x_pad * 0.35, x_max + x_pad)
        ax.set_ylim(max(0.0, y_min - y_pad * 0.30), min(1.05, y_max + y_pad))

        # --- Region annotations (discrete) ---
        ax.annotate("Higher fairness", xy=(0.02, 0.97), xycoords="axes fraction",
                    ha="left", va="top", fontsize=8.5, style="italic",
                    color="#555555",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#cccccc", alpha=0.8))
        ax.annotate("Higher throughput \u2192", xy=(0.98, 0.03),
                    xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=8.5, style="italic", color="#555555",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#cccccc", alpha=0.8))

        # --- Labels (use offset-points so fallback never escapes the axes) ---
        texts = []
        for i in range(len(labels)):
            side = -1 if (i % 2 == 0) else 1
            lift = 6 + (i % 3) * 4
            t = ax.annotate(
                labels[i], (xs_plot[i], ys_real[i]),
                textcoords="offset points",
                xytext=(side * 12, lift),
                fontsize=8.5, fontweight="bold",
                ha="right" if side < 0 else "left", va="center",
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#999999", alpha=0.85),
            )
            texts.append(t)
        try:
            from adjustText import adjust_text  # type: ignore
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6))
        except ImportError:
            # Offsets already applied above; nothing more to do.
            pass

        ax.set_title(f"Throughput vs. fairness (Jain index): {SCENARIO_LABELS[sc]}")
        ax.set_xlabel("Aggregated throughput (Mbps)")
        if ax == axes[0]:
            ax.set_ylabel("Jain fairness index")

        # Flag degenerate panels where all methods collapse to near-identical
        # operating points (no real trade-off exists in that load regime).
        thr_span = float(xs_real.max() - xs_real.min()) if len(xs_real) > 1 else 0.0
        jain_span = float(ys_real.max() - ys_real.min()) if len(ys_real) > 1 else 0.0
        if thr_span < 1.0 and jain_span < 0.02:
            ax.text(0.5, 0.06, "Low-load regime: schedulers converge\n"
                    "(no throughput/fairness trade-off)",
                    transform=ax.transAxes, ha="center", fontsize=8,
                    style="italic", color="#666666",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#f9f9f9",
                              ec="#cccccc", alpha=0.9))

    # Shared legend (manually built) using pure colors and meta colors/markers
    from matplotlib.lines import Line2D
    legend_handles = []
    for mode in PURE_MODES:
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor=PURE_COLORS[mode],
                                     markeredgecolor="black", markersize=10,
                                     label=PURE_LABELS[mode]))
    for m in METHODS:
        legend_handles.append(Line2D([0], [0], marker=META_MARKERS[m], color="w",
                                     markerfacecolor=METHOD_COLORS[m],
                                     markeredgecolor="black", markersize=11,
                                     label=METHOD_LABELS[m]))
    legend_handles.append(Line2D([0], [0], linestyle="--", color="#d62728",
                                 alpha=0.85, label="Pareto frontier"))
    fig.legend(handles=legend_handles, loc="lower center", ncol=8,
               bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=10)

    fig.suptitle("Throughput vs. fairness (Jain index): pure schedulers vs. all meta-heuristics",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.98))
    fig.savefig(FIG_DIR / "fig04_tradeoff_pareto.png")
    plt.close(fig)
    print("  + fig04_tradeoff_pareto.png")


def fig_variabilidade_seeds(meta: pd.DataFrame) -> None:
    """Fig 5: Variability of best score across seeds (error bars)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    width = 0.2
    x = np.arange(len(SCENARIOS))
    for i, m in enumerate(METHODS):
        means = []
        stds = []
        for sc in SCENARIOS:
            sub = meta[(meta.scenario == sc) & (meta.method == m)]
            maxs = sub.groupby("seed")["score"].max()
            means.append(maxs.mean())
            stds.append(maxs.std() if len(maxs) > 1 else 0)
        ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
               label=METHOD_LABELS[m], color=METHOD_COLORS[m], alpha=0.85,
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([SCENARIO_LABELS[sc] for sc in SCENARIOS])
    ax.set_ylabel("Best score $F(\\mathbf{w})$ per seed")
    ax.set_title("Variability of best score across seeds (mean +/- std, n=3)")
    ax.legend(loc="upper right")
    fig.savefig(FIG_DIR / "fig05_variabilidade_seeds.png")
    plt.close(fig)
    print("  + fig05_variabilidade_seeds.png")


def fig_pesos_otimos(best: pd.DataFrame) -> None:
    """Fig 6: Optimal weights [eMBB, URLLC, MTC] per scenario (mean 3 seeds)."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    width = 0.5
    x = np.arange(len(SCENARIOS))
    slices = [("w_embb", "eMBB", "#1f77b4"),
              ("w_urllc", "URLLC", "#d62728"),
              ("w_mtc", "MTC", "#2ca02c")]
    bottom = np.zeros(len(SCENARIOS))
    for col, label, color in slices:
        vals = [best[best.scenario == sc][col].mean() for sc in SCENARIOS]
        ax.bar(x, vals, width, bottom=bottom, label=label, color=color, alpha=0.85,
               edgecolor="white", linewidth=0.8)
        for xi, v, b in zip(x, vals, bottom):
            if v > 0.05:
                ax.text(xi, b + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[sc] for sc in SCENARIOS])
    ax.set_ylabel("Allocation weight")
    ax.set_ylim(0, 1.0)
    ax.set_title("Optimal weights found by the meta-heuristics (mean of 3 seeds)")
    ax.legend(loc="upper right", ncol=3)
    fig.savefig(FIG_DIR / "fig06_pesos_otimos.png")
    plt.close(fig)
    print("  + fig06_pesos_otimos.png")


def fig_throughput_por_slice(tf: pd.DataFrame) -> None:
    """Fig 7: Throughput per slice (eMBB/URLLC/MTC) for pure schedulers."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, sc in zip(axes, SCENARIOS):
        sub = tf[(tf.scenario == sc) & (tf["mode"].isin(PURE_MODES))]
        if sub.empty:
            ax.set_title(f"{SCENARIO_LABELS[sc]} (no data)")
            continue
        agg = sub.groupby("mode").agg({"thr_embb": "mean", "thr_urllc": "mean", "thr_mtc": "mean"}).reset_index()
        modes_order = [m for m in PURE_MODES if m in set(agg["mode"])]
        agg = agg.set_index("mode").loc[modes_order].reset_index()
        x = np.arange(len(agg))
        w = 0.25
        ax.bar(x - w, agg["thr_embb"], w, label="eMBB", color="#1f77b4", edgecolor="black", linewidth=0.4)
        ax.bar(x, agg["thr_urllc"], w, label="URLLC", color="#d62728", edgecolor="black", linewidth=0.4)
        ax.bar(x + w, agg["thr_mtc"], w, label="MTC", color="#2ca02c", edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([PURE_LABELS[m] for m in agg["mode"]])
        ax.set_title(SCENARIO_LABELS[sc])
        if ax == axes[0]:
            ax.legend(loc="upper right", fontsize=9)
        ax.set_ylabel("Throughput (Mbps)")
    fig.suptitle("Throughput per slice and scenario - pure schedulers (mean of 3 seeds)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(FIG_DIR / "fig07_throughput_por_slice.png")
    plt.close(fig)
    print("  + fig07_throughput_por_slice.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading data...")
    meta_df = load_meta_evals()
    best_df = load_best_candidates()
    tf_df = load_throughput_data(PURE_MODES + ["slice_weighted_pf"])

    print(f"  Meta-heuristics: {len(meta_df)} valid evals (crashes discarded)")
    print(f"  Best candidates: {len(best_df)}")
    print(f"  Throughput data: {len(tf_df)} points")
    print()
    print("Generating figures in", FIG_DIR)

    fig_convergencia(meta_df)
    fig_distribuicao_scores(meta_df)
    fig_comparacao_baselines(meta_df, tf_df)
    fig_tradeoff_pareto(meta_df, tf_df)
    fig_variabilidade_seeds(meta_df)
    fig_pesos_otimos(best_df)
    fig_throughput_por_slice(tf_df)

    print()
    print("Done. Figures saved in:", FIG_DIR)
    for f in sorted(FIG_DIR.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<40} {size_kb:>6.0f} KB")


if __name__ == "__main__":
    main()
