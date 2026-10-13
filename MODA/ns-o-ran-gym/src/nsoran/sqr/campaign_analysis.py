"""Campaign analysis for the SQR Fase 4 — mirror of the paper's Figs. 2-4.

Aggregates the ``step_metrics.csv`` of every run of a campaign directory
(layout produced by ``examples/run_sqr_campaign.sh``:
``<campaign>/<method>/seed_<seed>/step_metrics.csv``) and produces, PER
METHOD, with the frozen registers of ``MODA/docs/sqr/assumptions_register.md``:

- **Violation rates with explicit denominators (R19/R26)**:
  * UE-time (primary, R19): ``sum(violation_rate * n_ues) / sum(n_ues)``
    over ELIGIBLE (slice, window) rows — a window with no UE in the slice
    is excluded from the denominator; pooled over the 3 slices and logged
    per slice.
  * Slice-time (harmonization level, R19/KBRL): fraction of eligible
    (slice, window) pairs with ``y_t (p5) < Lambda_slice``.
- **Cumulative violations over time (Fig. 3 mirror)**: running sum of
  UE-time violations (count) vs simulated time.
- **Per-window violation rate vs the delta = 1% target (Fig. 2 mirror)**:
  pooled UE-time rate per 100 ms window, aggregated across seeds.
- **RB usage (Fig. 4 mirror, R26)**: distribution/mean of PRBs
  requested / executed / scheduled / idle per decision (mean across the
  slices of each decision; per-slice means logged in the tables).
  ``prb_scheduled`` comes from the ns-3 telemetry (contract §3.5) and may
  be missing in contract mocks — handled as NaN, never as 0.
- **Mode diversity**: KBRL ``scan`` vs ``cold`` (kbrl_controller.py) and
  SQR confident (``confident``) vs uncertain (``softmax``/``knn``/
  ``knn_nearest``/``fallback``) plus the static ``equal`` mode.
- **95% CI by bootstrap on the SEED unit** (GUIA §11 protocol): every
  scalar metric is first computed PER SEED and only then resampled
  (percentile bootstrap over seeds); the point estimate is the seed mean.
  With a single seed the CI degenerates to the point value.

Outputs: PNG figures (matplotlib, Agg) in ``<campaign>/figures/`` and CSV
tables in ``<campaign>/tables/``, all with an optional suffix (the pilot
uses ``_pilot``).

Usage:
    python src/nsoran/sqr/campaign_analysis.py \
        --campaign-dir results/sqr_campaign [--suffix _pilot] [--methods ...]
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

if __package__ in (None, ""):  # executed as a script: bootstrap the package path
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless: campaign boxes have no display
import matplotlib.pyplot as plt
from matplotlib import colormaps

#: Method -> plot color (campaign methods of the ask; extras get tab10).
METHOD_COLORS = {
    "sqr-g0": "#1f77b4",
    "sqr-g1": "#ff7f0e",
    "kbrl": "#2ca02c",
    "equal": "#7f7f7f",
}

#: Decision modes per controller family (``mode`` column of step_metrics.csv).
SQR_MODES = ("confident", "softmax", "knn", "knn_nearest", "fallback")
KBRL_MODES = ("scan", "cold")
STATIC_MODES = ("equal",)
#: SQR confident vs uncertain grouping (Fase 4 metric "modos confiante/incerto").
CONFIDENT_MODES = frozenset({"confident"})
UNCERTAIN_MODES = frozenset({"softmax", "knn", "knn_nearest", "fallback"})

#: RB categories of the Fig. 4 mirror: (step_metrics column, label).
RB_CATEGORIES = (
    ("prb_requested", "requested"),
    ("prb_executed", "executed"),
    ("prb_scheduled", "scheduled"),
    ("prb_idle_est", "idle (est.)"),
)

DEFAULT_SLAS = (0.5, 1.0, 1.5)  # Mbps per UE [ARTIGO, Tabela I]
DEFAULT_BOOTSTRAP_SEED = 20261012

STEP_CSV = "step_metrics.csv"
CHECKPOINT_FILE = "sqr_checkpoint.pkl"
DONE_FILE = "run_done.json"


# ---------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------


@dataclass
class RunInfo:
    """One campaign run (method, seed, artifact paths, completeness)."""

    method: str
    seed: int
    run_dir: str
    complete: bool = False
    step_csv: str = ""
    skipped_reason: str = ""


def discover_runs(campaign_dir: str, methods: list[str] | None = None) -> list[RunInfo]:
    """List runs under ``<campaign>/<method>/seed_<seed>/`` with completeness.

    A run is USABLE when ``step_metrics.csv`` exists; it is marked COMPLETE
    when the campaign driver's ``run_done.json`` is present (falls back to
    checkpoint presence).  Incomplete runs are reported and excluded from
    the aggregation (never silently mixed into the means).
    """
    runs: list[RunInfo] = []
    if not os.path.isdir(campaign_dir):
        return runs
    for method in sorted(os.listdir(campaign_dir)):
        method_dir = os.path.join(campaign_dir, method)
        if not os.path.isdir(method_dir) or method in ("figures", "tables"):
            continue
        if methods and method not in methods:
            continue
        for entry in sorted(os.listdir(method_dir)):
            if not entry.startswith("seed_"):
                continue
            run_dir = os.path.join(method_dir, entry)
            step_csv = os.path.join(run_dir, STEP_CSV)
            if not os.path.isfile(step_csv):
                continue
            try:
                seed = int(entry[len("seed_"):])
            except ValueError:
                continue
            done = os.path.isfile(os.path.join(run_dir, DONE_FILE))
            ckpt = os.path.isfile(os.path.join(run_dir, CHECKPOINT_FILE))
            runs.append(
                RunInfo(
                    method=method,
                    seed=seed,
                    run_dir=run_dir,
                    complete=done or ckpt,
                    step_csv=step_csv,
                    skipped_reason="" if (done or ckpt) else "no run_done.json/checkpoint",
                )
            )
    return runs


def load_run_df(step_csv: str) -> pd.DataFrame:
    """One step_metrics.csv -> typed frame (numeric coercion, NaN-safe)."""
    df = pd.read_csv(step_csv)
    numeric_cols = [
        "episode", "step", "timestamp_ms", "seed", "gamma", "slice_id",
        "raw_action", "action", "prb_requested", "prb_executed",
        "prb_scheduled", "prb_idle_est", "sigma", "margin_m", "margin_applied",
        "tau", "y_t_p5_mbps", "lowest_thr_mbps", "n_ues", "violation_rate",
        "n_snr_imputed", "kpm_missing", "learned", "virtual_updates",
        "decision_time_ms", "reward", "warmup",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Per-seed metrics (the bootstrap unit — GUIA §11)
# ---------------------------------------------------------------------------


def eligible_mask(df: pd.DataFrame) -> pd.Series:
    """R19 eligibility: (slice, window) rows with UEs and a defined rate."""
    return (df["n_ues"] > 0) & df["violation_rate"].notna()


def slice_time_violations(df: pd.DataFrame, slas: dict[int, float]) -> pd.Series:
    """y_t (p5) < Lambda per row (NaN where the row is not eligible)."""
    lam = df["slice_id"].map(lambda s: slas.get(int(s)) if pd.notna(s) else np.nan)
    y = df["y_t_p5_mbps"]
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    ok = eligible_mask(df) & y.notna() & lam.notna()
    out[ok] = (y[ok] < lam[ok]).astype("float64")
    return out


def seed_series(df: pd.DataFrame, period_ms: int) -> dict[str, pd.Series]:
    """Per-step series of one run, indexed by decision step (1..N).

    - ``window_rate_pct``: pooled UE-time violation rate of the window (%),
      NaN when no slice had UEs (period excluded from the denominator).
    - ``cum_violations``: running UE-time violation COUNT.
    - ``prb_*``: per-decision mean ACROSS SLICES of each PRB column (R26
      Fig. 4 aggregation); ``prb_scheduled`` is NaN when the telemetry
      column is empty (mock runs).
    """
    elig = eligible_mask(df)
    viol = df["violation_rate"] * df["n_ues"]

    ue_num = viol.where(elig, 0.0).groupby(df["step"]).sum()
    ue_den = df["n_ues"].where(elig, 0.0).groupby(df["step"]).sum()
    window_rate = 100.0 * ue_num / ue_den.where(ue_den > 0)
    cum = ue_num.cumsum()

    series: dict[str, pd.Series] = {
        "window_rate_pct": window_rate,
        "cum_violations": cum,
    }
    for col, _label in RB_CATEGORIES:
        series[col] = df[col].groupby(df["step"]).mean()  # NaN-aware mean
    del period_ms  # time axis is derived by the callers from the step index
    return series


def seed_scalars(
    df: pd.DataFrame, slas: dict[int, float], period_ms: int
) -> dict[str, float]:
    """Scalar metrics of ONE run (one bootstrap unit)."""
    elig = eligible_mask(df)
    n_rows = int(len(df))
    n_steps = int(df["step"].nunique())
    slice_ids = sorted(df["slice_id"].dropna().unique().astype(int))

    ue_num = float((df["violation_rate"] * df["n_ues"])[elig].sum())
    ue_den = float(df["n_ues"][elig].sum())
    slice_viol = slice_time_violations(df, slas)
    slice_elig = int(slice_viol.notna().sum())
    slice_num = float(np.nansum(slice_viol.values))

    out: dict[str, float] = {
        "n_steps": float(n_steps),
        "n_slice_decisions": float(n_rows),
        "ue_time_violation_pct": 100.0 * ue_num / ue_den if ue_den > 0 else np.nan,
        "ue_time_violation_count": ue_num,
        "slice_time_violation_pct": 100.0 * slice_num / slice_elig if slice_elig > 0 else np.nan,
        "ue_time_denominator": ue_den,
        "slice_time_denominator": float(slice_elig),
    }

    # Per-slice breakdown (R26: logged separately).
    for sid in slice_ids:
        sub = df[df["slice_id"] == sid]
        e = eligible_mask(sub)
        den = float(sub["n_ues"][e].sum())
        sv = slice_time_violations(sub, slas)
        se = int(sv.notna().sum())
        out[f"ue_time_pct_slice{sid}"] = (
            100.0 * float((sub["violation_rate"] * sub["n_ues"])[e].sum()) / den
            if den > 0 else np.nan
        )
        out[f"slice_time_pct_slice{sid}"] = (
            100.0 * float(np.nansum(sv.values)) / se if se > 0 else np.nan
        )
        out[f"ue_denominator_slice{sid}"] = den
        out[f"slice_denominator_slice{sid}"] = float(se)

    # RB usage (Fig. 4): mean over decisions of the per-decision mean across
    # slices; per-slice means and cell totals logged too (R26).
    for col, _label in RB_CATEGORIES:
        key = col.replace("prb_", "")
        per_step = df[col].groupby(df["step"]).mean()
        out[f"rb_{key}_mean"] = (
            float(per_step.mean()) if per_step.notna().any() else np.nan
        )
        for sid in slice_ids:
            v = df.loc[df["slice_id"] == sid, col]
            out[f"rb_{key}_mean_slice{sid}"] = (
                float(v.mean()) if v.notna().any() else np.nan
            )
        cell = df.groupby("step")[col].sum(min_count=1)
        out[f"rb_cell_total_{key}_mean"] = (
            float(cell.mean()) if cell.notna().any() else np.nan
        )

    # Decision wall-time (xApp accounting; one sample per decision).
    per_step_dt = df.groupby("step")["decision_time_ms"].first()
    dt = per_step_dt.dropna()
    out["decision_ms_p50"] = float(dt.quantile(0.50)) if len(dt) else np.nan
    out["decision_ms_p95"] = float(dt.quantile(0.95)) if len(dt) else np.nan

    # Countable KPM-missing fallback (contract §5): cumulative counter.
    km = df.groupby("step")["kpm_missing"].max()
    out["kpm_missing_pct"] = (
        100.0 * float(km.iloc[-1]) / n_steps if n_steps and len(km) else np.nan
    )

    # Mode diversity (rows = slice-decisions).  Raw per-mode shares use the
    # ``mode_<mode>_pct`` namespace; the confident-vs-uncertain GROUP shares
    # use ``modes_*_group_pct`` so the raw "confident" column is preserved.
    modes = df["mode"].astype("string")
    n_mode_rows = int(modes.notna().sum())
    for mode in sorted(set(modes.dropna().unique())):
        out[f"mode_{mode}_pct"] = 100.0 * float((modes == mode).sum()) / max(n_mode_rows, 1)
    conf = int(modes.isin(CONFIDENT_MODES).sum())
    unc = int(modes.isin(UNCERTAIN_MODES).sum())
    denom = conf + unc
    out["modes_confident_group_pct"] = 100.0 * conf / denom if denom else np.nan
    out["modes_uncertain_group_pct"] = 100.0 * unc / denom if denom else np.nan
    del period_ms  # kept in the signature for symmetry with seed_series
    return out


# ---------------------------------------------------------------------------
# Bootstrap (seed unit) and aggregation
# ---------------------------------------------------------------------------


def _nanmean(values: np.ndarray) -> float:
    """nanmean of a possibly empty/all-NaN vector -> NaN instead of a
    RuntimeWarning (all-NaN is a LEGAL answer here: e.g. the confident-mode
    share of a KBRL run, whose modes are scan/cold only)."""
    vals = np.asarray(values, dtype="float64")
    return float(np.nanmean(vals)) if vals.size and np.isfinite(vals).any() else np.nan


def bootstrap_ci(
    values: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Percentile bootstrap CI of the MEAN over seeds (GUIA §11).

    Resamples the per-seed values with replacement; NaN seeds are dropped
    first.  With a single seed the CI degenerates to the point value.
    """
    vals = np.asarray(values, dtype="float64")
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return (np.nan, np.nan)
    if vals.size == 1:
        return (float(vals[0]), float(vals[0]))
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def aggregate_seeds(
    per_seed: pd.DataFrame, n_boot: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Per-seed scalar frame -> summary frame (mean, ci_lo, ci_hi per metric)."""
    numeric = per_seed.select_dtypes(include=[np.number])
    rows = {}
    for metric in numeric.columns:
        vals = numeric[metric].to_numpy(dtype="float64")
        lo, hi = bootstrap_ci(vals, n_boot, rng)
        rows[metric] = {
            "mean": _nanmean(vals),
            "ci_lo": lo,
            "ci_hi": hi,
            "n_seeds": int(vals.size),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def align_series(frames: dict[int, dict[str, pd.Series]], key: str) -> pd.DataFrame:
    """Per-seed series -> steps (rows) x seeds (cols) frame, NaN-padded."""
    return pd.DataFrame({seed: s[key] for seed, s in frames.items()})


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _method_color(method: str) -> str:
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    cmap = colormaps["tab10"]
    return cmap(hash(method) % 10)


def fig2_window_violation(
    series_by_method: dict[str, pd.DataFrame],
    delta_pct: float,
    rolling: int,
    period_ms: int,
    path: str,
) -> None:
    """Fig. 2 mirror: per-window violation rate (%) vs time, target delta."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, frame in sorted(series_by_method.items()):
        t = frame.index.to_numpy(dtype="float64") * (period_ms / 1000.0)
        mean = frame.mean(axis=1, skipna=True)
        smooth = mean.rolling(rolling, min_periods=1, center=True).mean()
        ax.plot(
            t, smooth.to_numpy(),
            label=f"{method} ({rolling}-window mean)",
            color=_method_color(method),
        )
    ax.axhline(delta_pct, color="k", linestyle="--", linewidth=1,
               label=fr"target $\delta$ = {delta_pct:g}%")
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("UE-time SLA violation rate per window (%)")
    ax.set_title("Per-window SLA violation rate (R19/R26; Fig. 2 mirror)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig3_cumulative_violations(
    series_by_method: dict[str, pd.DataFrame], period_ms: int, path: str
) -> None:
    """Fig. 3 mirror: accumulated UE-time violations vs time (mean ± std)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, frame in sorted(series_by_method.items()):
        t = frame.index.to_numpy(dtype="float64") * (period_ms / 1000.0)
        mean = frame.mean(axis=1, skipna=True)
        std = frame.std(axis=1, skipna=True)
        color = _method_color(method)
        ax.plot(t, mean.to_numpy(), label=method, color=color)
        if frame.shape[1] > 1:
            ax.fill_between(
                t, (mean - std).to_numpy(), (mean + std).to_numpy(),
                color=color, alpha=0.18, linewidth=0,
            )
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("cumulative UE-time SLA violations (count)")
    ax.set_title(
        "Accumulated SLA violations (R19 count; Fig. 3 mirror; "
        "band = ±1 std across seeds)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig4_prb_distribution(
    series_by_method: dict[str, dict[str, pd.DataFrame]], path: str
) -> None:
    """Fig. 4 mirror: distribution of PRBs per decision (mean across slices).

    One group per RB category (requested/executed/scheduled/idle), one box
    per method inside the group; every observation is one 100 ms decision
    of one seed (pooled over seeds).
    """
    methods = sorted(series_by_method)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.8 / max(len(methods), 1)
    for gi, (_key, label) in enumerate(RB_CATEGORIES):
        for mi, method in enumerate(methods):
            vals = series_by_method[method][_key].to_numpy(dtype="float64").ravel()
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            pos = gi + (mi - (len(methods) - 1) / 2.0) * width
            ax.boxplot(
                vals, positions=[pos], widths=width * 0.9,
                showfliers=False, patch_artist=True,
                boxprops=dict(facecolor=_method_color(method), alpha=0.6),
                medianprops=dict(color="k", linewidth=1),
            )
    ax.set_xticks(range(len(RB_CATEGORIES)))
    ax.set_xticklabels([lbl for _k, lbl in RB_CATEGORIES])
    ax.set_ylabel("PRBs per decision (mean across slices)")
    ax.set_title("RB usage per decision (R26/R27; Fig. 4 mirror; pooled seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=_method_color(m), alpha=0.6)
        for m in methods
    ]
    ax.legend(handles, methods, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_summary_bars(summary: pd.DataFrame, path: str) -> None:
    """Overall violation rates (%) with seed-bootstrap 95% CI error bars."""
    metrics = (
        ("ue_time_violation_pct", "UE-time (pooled, R19/R26)"),
        ("slice_time_violation_pct", "slice-time (p5 < Lambda)"),
    )
    methods = list(dict.fromkeys(summary.index.get_level_values("method")))
    x = np.arange(len(metrics))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for mi, method in enumerate(methods):
        for gi, (metric, _label) in enumerate(metrics):
            if (method, metric) not in summary.index:
                continue
            row = summary.loc[(method, metric)]
            mean, lo, hi = row["mean"], row["ci_lo"], row["ci_hi"]
            if not np.isfinite(mean):
                continue
            err_lo = max(0.0, mean - lo) if np.isfinite(lo) else 0.0
            err_hi = max(0.0, hi - mean) if np.isfinite(hi) else 0.0
            ax.bar(
                gi + (mi - (len(methods) - 1) / 2.0) * width, mean,
                width=width * 0.9, color=_method_color(method), alpha=0.85,
                yerr=[[err_lo], [err_hi]],
                error_kw=dict(elinewidth=1, capsize=3),
            )
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _m, lbl in metrics])
    ax.set_ylabel("SLA violation rate (%)")
    ax.set_title("Violation rates by method (mean over seeds; 95% bootstrap CI on seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=_method_color(m), alpha=0.85)
        for m in methods
    ]
    ax.legend(handles, methods, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_modes(modes_summary: pd.DataFrame, path: str) -> None:
    """Mode diversity: % of slice-decisions per decision mode (KBRL scan/cold,
    SQR confident/uncertain modes, static equal) — stacked bars per method."""
    methods = list(dict.fromkeys(modes_summary.index.get_level_values("method")))
    modes = list(dict.fromkeys(modes_summary.index.get_level_values("mode")))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(methods))
    left = np.zeros(len(methods))
    palette = colormaps["tab20"]
    for mi_mode, mode in enumerate(modes):
        vals = np.asarray(
            [
                float(modes_summary.loc[(m, mode), "mean_pct"])
                if (m, mode) in modes_summary.index else 0.0
                for m in methods
            ],
            dtype="float64",
        )
        ax.barh(y, vals, left=left, height=0.6,
                color=palette(mi_mode % 20), label=mode)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlabel("% of slice-decisions")
    ax.set_xlim(0, 100)
    ax.set_title("Decision-mode diversity (KBRL scan/cold; SQR confident vs uncertain)")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    default_campaign = os.path.abspath(
        os.path.join(here, "..", "..", "..", "results", "sqr_campaign")
    )
    parser = argparse.ArgumentParser(
        description="Aggregate SQR campaign runs into Figs. 2-4 mirrors + tables"
    )
    parser.add_argument("--campaign-dir", type=str, default=default_campaign,
                        help="campaign root (default: results/sqr_campaign)")
    parser.add_argument("--methods", type=str, default="",
                        help="comma list restricting methods (default: all found)")
    parser.add_argument("--suffix", type=str, default="",
                        help="suffix for every output file (e.g. _pilot)")
    parser.add_argument("--out-figures", type=str, default="",
                        help="figures dir (default: <campaign>/figures)")
    parser.add_argument("--out-tables", type=str, default="",
                        help="tables dir (default: <campaign>/tables)")
    parser.add_argument("--bootstraps", type=int, default=10000,
                        help="bootstrap resamples (GUIA §11)")
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED,
                        help="RNG seed of the bootstrap (reproducibility)")
    parser.add_argument("--exclude-warmup", action="store_true",
                        help="drop rows with warmup == 1 (R14; the 600 s "
                             "campaign runs mark the first 100 s)")
    parser.add_argument("--slas", type=str,
                        default=",".join(str(v) for v in DEFAULT_SLAS),
                        help="per-slice SLA in Mbps [ARTIGO, Tabela I]")
    parser.add_argument("--period-ms", type=int, default=100,
                        help="decision window (contract §2)")
    parser.add_argument("--rolling", type=int, default=50,
                        help="rolling window (decisions) of the Fig. 2 curve")
    parser.add_argument("--delta", type=float, default=0.01,
                        help="violation target delta of Fig. 2 [ARTIGO]")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    campaign_dir = os.path.abspath(args.campaign_dir)
    fig_dir = os.path.abspath(args.out_figures or os.path.join(campaign_dir, "figures"))
    tab_dir = os.path.abspath(args.out_tables or os.path.join(campaign_dir, "tables"))
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    sfx = args.suffix

    slas = {
        i: float(v) for i, v in enumerate(float(x) for x in args.slas.split(","))
    }
    methods_filter = (
        [m.strip() for m in args.methods.split(",") if m.strip()]
        if args.methods else None
    )

    runs = discover_runs(campaign_dir, methods_filter)
    if not runs:
        print(f"[analysis] nenhum run encontrado em {campaign_dir}")
        return 1

    rng = np.random.default_rng(args.bootstrap_seed)

    # ---- load + per-seed metrics -----------------------------------------
    per_seed_rows: list[dict] = []
    per_seed_series: dict[str, dict[int, dict[str, pd.Series]]] = {}
    skipped: list[str] = []
    for run in runs:
        if not run.complete:
            skipped.append(f"{run.method}/seed_{run.seed} ({run.skipped_reason})")
            continue
        df = load_run_df(run.step_csv)
        if args.exclude_warmup and "warmup" in df.columns:
            df = df[df["warmup"] == 0]
        if df.empty:
            skipped.append(f"{run.method}/seed_{run.seed} (empty after warmup filter)")
            continue
        scalars = seed_scalars(df, slas, args.period_ms)
        scalars.update({"method": run.method, "seed": run.seed})
        per_seed_rows.append(scalars)
        per_seed_series.setdefault(run.method, {})[run.seed] = seed_series(
            df, args.period_ms
        )
    if skipped:
        print("[analysis] runs ignorados (incompletos):")
        for s in skipped:
            print(f"  - {s}")
    if not per_seed_rows:
        print("[analysis] nenhum run utilizável — nada a agregar")
        return 1

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(
        os.path.join(tab_dir, f"per_seed_metrics{sfx}.csv"), index=False
    )

    # ---- per-method aggregation (bootstrap on the seed unit) -------------
    summary_rows = []
    modes_rows = []
    series_means: dict[str, dict[str, pd.DataFrame]] = {}
    for method in sorted(per_seed_series):
        sub = per_seed_df[per_seed_df["method"] == method].drop(
            columns=["method", "seed"], errors="ignore"
        )
        agg = aggregate_seeds(sub, args.bootstraps, rng)
        for metric, row in agg.iterrows():
            summary_rows.append({"method": method, "metric": metric, **row.to_dict()})

        # mode breakdown from the per-seed mode_*_pct columns (seed unit);
        # the grouped confident/uncertain shares are named modes_*_group_pct
        # and are therefore excluded here automatically.
        mode_cols = [
            c for c in sub.columns
            if c.startswith("mode_") and c.endswith("_pct")
        ]
        n_rows_per_seed = sub["n_slice_decisions"].to_numpy(dtype="float64")
        for col in mode_cols:
            mode = col[len("mode_"):-len("_pct")]
            vals = sub[col].to_numpy(dtype="float64")
            lo, hi = bootstrap_ci(vals, args.bootstraps, rng)
            counts = vals * n_rows_per_seed / 100.0
            modes_rows.append({
                "method": method,
                "mode": mode,
                "mean_pct": _nanmean(vals),
                "ci_lo_pct": lo,
                "ci_hi_pct": hi,
                "mean_count": _nanmean(counts),
            })

        frames = per_seed_series[method]
        series_means[method] = {
            "window_rate_pct": align_series(frames, "window_rate_pct"),
            "cum_violations": align_series(frames, "cum_violations"),
            **{col: align_series(frames, col) for col, _l in RB_CATEGORIES},
        }

    summary = pd.DataFrame(summary_rows).set_index(["method", "metric"]).sort_index()
    summary.to_csv(os.path.join(tab_dir, f"summary_by_method{sfx}.csv"))

    modes_summary = (
        pd.DataFrame(modes_rows).set_index(["method", "mode"]).sort_index()
        if modes_rows else pd.DataFrame()
    )
    if not modes_summary.empty:
        modes_summary.to_csv(os.path.join(tab_dir, f"modes_by_method{sfx}.csv"))

    # per-slice violation table (R26: curves per slice logged separately)
    slice_rows = []
    for method in sorted(per_seed_df["method"].unique()):
        sub = per_seed_df[per_seed_df["method"] == method]
        for sid in range(len(slas)):
            ue_col = f"ue_time_pct_slice{sid}"
            st_col = f"slice_time_pct_slice{sid}"
            if ue_col not in sub.columns:
                continue
            ue_vals = sub[ue_col].to_numpy(dtype="float64")
            st_vals = sub[st_col].to_numpy(dtype="float64")
            ue_lo, ue_hi = bootstrap_ci(ue_vals, args.bootstraps, rng)
            st_lo, st_hi = bootstrap_ci(st_vals, args.bootstraps, rng)
            slice_rows.append({
                "method": method,
                "slice_id": sid,
                "sla_mbps": slas[sid],
                "ue_time_violation_pct_mean": _nanmean(ue_vals),
                "ue_time_violation_pct_ci_lo": ue_lo,
                "ue_time_violation_pct_ci_hi": ue_hi,
                "slice_time_violation_pct_mean": _nanmean(st_vals),
                "slice_time_violation_pct_ci_lo": st_lo,
                "slice_time_violation_pct_ci_hi": st_hi,
                "ue_denominator_mean": _nanmean(
                    sub[f"ue_denominator_slice{sid}"].to_numpy(dtype="float64")
                ),
                "slice_denominator_mean": _nanmean(
                    sub[f"slice_denominator_slice{sid}"].to_numpy(dtype="float64")
                ),
                "n_seeds": int(sub[ue_col].notna().sum()),
            })
    pd.DataFrame(slice_rows).to_csv(
        os.path.join(tab_dir, f"violation_by_slice{sfx}.csv"), index=False
    )

    # series exports (means across seeds) for reproducibility
    for method in sorted(series_means):
        steps = series_means[method]["cum_violations"].index
        out = pd.DataFrame({
            "time_s": steps.to_numpy(dtype="float64") * (args.period_ms / 1000.0),
            "window_rate_pct_mean": series_means[method]["window_rate_pct"].mean(axis=1),
            "cum_violations_mean": series_means[method]["cum_violations"].mean(axis=1),
        })
        out.to_csv(
            os.path.join(tab_dir, f"series_{method}{sfx}.csv"), index=False
        )

    # ---- figures ----------------------------------------------------------
    fig2_window_violation(
        {m: s["window_rate_pct"] for m, s in series_means.items()},
        delta_pct=100.0 * args.delta, rolling=args.rolling,
        period_ms=args.period_ms,
        path=os.path.join(fig_dir, f"fig2_window_violation_rate{sfx}.png"),
    )
    fig3_cumulative_violations(
        {m: s["cum_violations"] for m, s in series_means.items()},
        period_ms=args.period_ms,
        path=os.path.join(fig_dir, f"fig3_cumulative_violation{sfx}.png"),
    )
    fig4_prb_distribution(
        series_means,
        path=os.path.join(fig_dir, f"fig4_prb_distribution{sfx}.png"),
    )
    fig_summary_bars(
        summary,
        path=os.path.join(fig_dir, f"fig_summary_violation_bars{sfx}.png"),
    )
    if not modes_summary.empty:
        fig_modes(
            modes_summary,
            path=os.path.join(fig_dir, f"fig_modes{sfx}.png"),
        )

    # ---- human summary ----------------------------------------------------
    print("\n=== Resumo por método (média sobre seeds; IC 95% bootstrap na seed) ===")
    key_metrics = (
        "ue_time_violation_pct", "slice_time_violation_pct",
        "rb_requested_mean", "rb_executed_mean", "rb_scheduled_mean",
        "rb_idle_est_mean", "decision_ms_p95", "kpm_missing_pct",
        "modes_confident_group_pct",
    )
    for method in sorted(per_seed_df["method"].unique()):
        n_seeds = int((per_seed_df["method"] == method).sum())
        print(f"\n[{method}] seeds={n_seeds}")
        for metric in key_metrics:
            if (method, metric) not in summary.index:
                continue
            row = summary.loc[(method, metric)]
            print(
                f"  {metric:28s} mean={row['mean']:10.4f} "
                f"CI95=[{row['ci_lo']:10.4f}, {row['ci_hi']:10.4f}] "
                f"(n={int(row['n_seeds'])})"
            )
    print(f"\n[analysis] figuras: {fig_dir}")
    print(f"[analysis] tabelas: {tab_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
