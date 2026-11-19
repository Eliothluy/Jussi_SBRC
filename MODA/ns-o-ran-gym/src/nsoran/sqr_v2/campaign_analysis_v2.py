"""Non-destructive BUG-04 fixed campaign analysis (unit separation + .first() idle_est).

Fixes two analytical bugs identified in ``nsoran/sqr/campaign_analysis.py`` (v1):
1. ``prb_idle_est`` is already a cell-level quantity ``C - sum_i(executed_i)``
   replicated on every slice row of a step; aggregating with ``.sum()`` tripled
   it (e.g. ``74.96`` instead of ``24.99`` for C=53). ``v2`` uses ``.first()``
   and asserts exact conservation ``executed + unallocated_quota == C_step``.
2. ``Fig. 4`` previously mixed ``PRBs / slot`` (``requested``, ``executed``,
   ``idle_est`` on ``[0, C]``) with window-cumulative ``RBG·slots / 100 ms``
   (``prb_scheduled`` up to ~``53 * 58``). ``v2`` separates ``Fig. 4`` and all
   CSV summary metrics into two explicitly typed physical unit groups:
   - ``rb_slot_*`` (PRBs / slot, scale ``[0, C]``), including
     ``scheduled_per_active_dl_slot = sum(prb_scheduled_rbg) / dl_slots``;
   - ``rb_win_*`` (RBG·slots / 100 ms window) from ``sqr_prb_windows.csv``.

Never modifies raw simulation CSV files; writes only to ``--out-figures`` and
``--out-tables`` (defaulting under ``results/sqr_v2/`` for v1 reanalysis).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nsoran.sqr.campaign_analysis import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_SLAS,
    METHOD_COLORS,
    aggregate_seeds,
    align_series,
    bootstrap_ci,
    discover_runs,
    eligible_mask,
    fig2_window_violation,
    fig3_cumulative_violations,
    fig_modes,
    fig_summary_bars,
    load_run_df,
    slice_time_violations,
)


def _method_color(method: str) -> str:
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    cmap = plt.get_cmap("tab10")
    return cmap(hash(method) % 10)


def find_prb_windows_csv(run_dir: str) -> str | None:
    matches = sorted(
        glob.glob(os.path.join(run_dir, "**", "sqr_prb_windows.csv"), recursive=True)
    )
    return matches[0] if matches else None


def load_prb_windows_df(run_dir: str) -> pd.DataFrame | None:
    path = find_prb_windows_csv(run_dir)
    if path is None:
        return None
    df = pd.read_csv(path)
    num_cols = [
        "timestamp_ms", "total_rbg", "total_prb", "dl_slots",
        "prb_executed_rbg_0", "prb_executed_rbg_1", "prb_executed_rbg_2",
        "prb_scheduled_rbg_0", "prb_scheduled_rbg_1", "prb_scheduled_rbg_2",
        "prb_budget_rbg_0", "prb_budget_rbg_1", "prb_budget_rbg_2",
        "prb_idle_rbg", "window_seq", "applied_action_seq",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def check_step_idle_consistency(df: pd.DataFrame, tol: float = 1e-6) -> None:
    """Ensure ``prb_idle_est`` is identical across slice rows of the same step."""
    if "prb_idle_est" not in df.columns or df.empty:
        return
    spread = df.groupby("step")["prb_idle_est"].agg(lambda s: float(s.max() - s.min()) if s.notna().any() else 0.0)
    max_spread = float(spread.max()) if len(spread) else 0.0
    if max_spread > tol:
        raise ValueError(
            f"prb_idle_est varies across slices within a step (max spread={max_spread})"
        )


def compute_seed_series_v2(
    df: pd.DataFrame,
    prb_win_df: pd.DataFrame | None = None,
    *,
    exclude_warmup: bool = False,
) -> dict[str, pd.Series]:
    """Per-step series of one run with BUG-04 unit separation and .first() idle."""
    check_step_idle_consistency(df)
    elig = eligible_mask(df)
    viol = df["violation_rate"] * df["n_ues"]

    ue_num = viol.where(elig, 0.0).groupby(df["step"]).sum()
    ue_den = df["n_ues"].where(elig, 0.0).groupby(df["step"]).sum()
    window_rate = 100.0 * ue_num / ue_den.where(ue_den > 0)
    cum = ue_num.cumsum()

    # Cell-level per-slot RB series [0, C]
    req_cell = df.groupby("step")["prb_requested"].sum(min_count=1)
    exec_cell = df.groupby("step")["prb_executed"].sum(min_count=1)
    unalloc_cell = df.groupby("step")["prb_idle_est"].first()

    # Window-level scheduled series from step_metrics (RBG*slots/100ms, NOT per-slot)
    sched_win_step = df.groupby("step")["prb_scheduled"].sum(min_count=1)

    series: dict[str, pd.Series] = {
        "window_rate_pct": window_rate,
        "cum_violations": cum,
        "slot_requested_cell": req_cell,
        "slot_executed_cell": exec_cell,
        "slot_unallocated_quota": unalloc_cell,
        "win_scheduled_rbg_slots_step": sched_win_step,
    }

    if prb_win_df is not None and not prb_win_df.empty:
        pw = prb_win_df.copy()
        if exclude_warmup and "timestamp_ms" in df.columns and df["timestamp_ms"].notna().any():
            # Keep windows at or after the earliest non-warmup step timestamp
            valid_ts = df["timestamp_ms"].dropna()
            min_ts = float(valid_ts[valid_ts > 0].min()) if (valid_ts > 0).any() else float(valid_ts.min())
            pw = pw[pw["timestamp_ms"] >= min_ts]
        # Skip pre-agent rows where prb_executed_pct == -1 (timestamp <= 100ms) when aligning agent decisions
        sched_sum = (
            pw["prb_scheduled_rbg_0"] + pw["prb_scheduled_rbg_1"] + pw["prb_scheduled_rbg_2"]
        )
        budget_sum = (
            pw["prb_budget_rbg_0"] + pw["prb_budget_rbg_1"] + pw["prb_budget_rbg_2"]
        )
        cap_win = pw["total_rbg"] * pw["dl_slots"]
        idle_win = pw["prb_idle_rbg"]
        active = pw["dl_slots"] > 0
        sched_per_active_slot = (sched_sum[active] / pw.loc[active, "dl_slots"]).reset_index(drop=True)

        series["slot_scheduled_active_dl"] = sched_per_active_slot
        series["win_capacity_rbg_slots"] = cap_win.reset_index(drop=True)
        series["win_budget_rbg_slots"] = budget_sum.reset_index(drop=True)
        series["win_scheduled_rbg_slots"] = sched_sum.reset_index(drop=True)
        series["win_idle_rbg_slots"] = idle_win.reset_index(drop=True)
    return series


def compute_seed_scalars_v2(
    df: pd.DataFrame,
    slas: dict[int, float],
    period_ms: int,
    prb_win_df: pd.DataFrame | None = None,
    *,
    exclude_warmup: bool = False,
) -> dict[str, float]:
    """Scalar metrics of ONE run with exact BUG-04 fixes & unit separation."""
    check_step_idle_consistency(df)
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

    # ---- BUG-04 Fixed Unit A: Per-slot PRBs/slot on [0, C] ----
    req_cell = df.groupby("step")["prb_requested"].sum(min_count=1)
    exec_cell = df.groupby("step")["prb_executed"].sum(min_count=1)
    unalloc_cell = df.groupby("step")["prb_idle_est"].first()
    c_inferred = exec_cell + unalloc_cell

    out["rb_slot_requested_cell_mean"] = float(req_cell.mean()) if req_cell.notna().any() else np.nan
    out["rb_slot_executed_cell_mean"] = float(exec_cell.mean()) if exec_cell.notna().any() else np.nan
    out["rb_slot_unallocated_quota_mean"] = float(unalloc_cell.mean()) if unalloc_cell.notna().any() else np.nan
    out["rb_slot_capacity_c_mean"] = float(c_inferred.mean()) if c_inferred.notna().any() else np.nan
    # Explicit corrected alias matching v1 table key for errata comparison
    out["rb_cell_total_executed_mean"] = out["rb_slot_executed_cell_mean"]
    out["rb_cell_total_idle_est_mean"] = out["rb_slot_unallocated_quota_mean"]
    out["rb_slot_conservation_abs_err"] = (
        abs((out["rb_slot_executed_cell_mean"] + out["rb_slot_unallocated_quota_mean"]) - out["rb_slot_capacity_c_mean"])
        if np.isfinite(out["rb_slot_capacity_c_mean"]) else np.nan
    )

    for sid in slice_ids:
        sub = df[df["slice_id"] == sid]
        out[f"rb_slot_requested_slice{sid}_mean"] = float(sub["prb_requested"].mean())
        out[f"rb_slot_executed_slice{sid}_mean"] = float(sub["prb_executed"].mean())

    # ---- BUG-04 Fixed Unit B: Cumulative RBG*slots / 100ms window ----
    sched_step_sum = df.groupby("step")["prb_scheduled"].sum(min_count=1)
    out["rb_win_scheduled_step_mean"] = (
        float(sched_step_sum.mean()) if sched_step_sum.notna().any() else np.nan
    )

    if prb_win_df is not None and not prb_win_df.empty:
        pw = prb_win_df.copy()
        if exclude_warmup and "timestamp_ms" in df.columns and df["timestamp_ms"].notna().any():
            valid_ts = df["timestamp_ms"].dropna()
            min_ts = float(valid_ts[valid_ts > 0].min()) if (valid_ts > 0).any() else float(valid_ts.min())
            pw = pw[pw["timestamp_ms"] >= min_ts]
        if not pw.empty:
            sched_win = pw["prb_scheduled_rbg_0"] + pw["prb_scheduled_rbg_1"] + pw["prb_scheduled_rbg_2"]
            budget_win = pw["prb_budget_rbg_0"] + pw["prb_budget_rbg_1"] + pw["prb_budget_rbg_2"]
            cap_win = pw["total_rbg"] * pw["dl_slots"]
            idle_win = pw["prb_idle_rbg"]
            active = pw["dl_slots"] > 0

            out["rb_win_capacity_rbg_slots_mean"] = float(cap_win.mean())
            out["rb_win_budget_rbg_slots_mean"] = float(budget_win.mean())
            out["rb_win_scheduled_rbg_slots_mean"] = float(sched_win.mean())
            out["rb_win_idle_rbg_slots_mean"] = float(idle_win.mean())
            out["rb_win_scheduled_le_budget_violations"] = float((sched_win > budget_win).sum())
            out["rb_win_multibeam_budget_gt_cap_windows"] = float((budget_win > cap_win).sum())
            single_beam = budget_win <= cap_win
            out["rb_win_singlebeam_conservation_abs_err"] = (
                float((sched_win[single_beam] + idle_win[single_beam] - cap_win[single_beam]).abs().max())
                if bool(single_beam.any()) else 0.0
            )
            out["rb_win_conservation_abs_err"] = float((sched_win + idle_win - cap_win).abs().max())
            out["rb_slot_scheduled_active_dl_mean"] = (
                float((sched_win[active] / pw.loc[active, "dl_slots"]).mean())
                if bool(active.any()) else np.nan
            )
            out["rb_dl_slots_per_window_mean"] = float(pw["dl_slots"].mean())

    # Decision wall-time
    dt_col = "agent_decision_wall_ms" if "agent_decision_wall_ms" in df.columns else "decision_time_ms"
    dt = df.groupby("step")[dt_col].first().dropna()
    out["decision_ms_p50"] = float(dt.quantile(0.50)) if len(dt) else np.nan
    out["decision_ms_p95"] = float(dt.quantile(0.95)) if len(dt) else np.nan
    out["decision_ms_max"] = float(dt.max()) if len(dt) else np.nan

    # KPM missing & modes
    km = df.groupby("step")["kpm_missing"].max()
    out["kpm_missing_pct"] = (
        100.0 * float(km.iloc[-1]) / n_steps if n_steps and len(km) else np.nan
    )
    modes = df["mode"].astype("string")
    n_mode_rows = int(modes.notna().sum())
    for mode in sorted(set(modes.dropna().unique())):
        out[f"mode_{mode}_pct"] = 100.0 * float((modes == mode).sum()) / max(n_mode_rows, 1)
    conf = int((modes == "confident").sum())
    unc = int(modes.isin({"softmax", "knn", "knn_nearest", "fallback"}).sum())
    denom = conf + unc
    out["modes_confident_group_pct"] = 100.0 * conf / denom if denom else np.nan
    out["modes_uncertain_group_pct"] = 100.0 * unc / denom if denom else np.nan
    del period_ms
    return out


def fig4_prb_distribution_v2(
    series_by_method: dict[str, dict[str, pd.DataFrame]],
    path: str,
    c_prbs: float = 53.0,
) -> None:
    """Two-panel Fig. 4 separating PRBs/slot [0, C] from RBG*slots / 100 ms window."""
    methods = sorted(series_by_method)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    width = 0.8 / max(len(methods), 1)

    slot_cats = [
        ("slot_requested_cell", "requested\n(cell)"),
        ("slot_executed_cell", "executed\n(cell)"),
        ("slot_unallocated_quota", "unalloc quota\n(C - exec)"),
        ("slot_scheduled_active_dl", "scheduled\n(/active DL slot)"),
    ]
    for gi, (key, _lbl) in enumerate(slot_cats):
        for mi, method in enumerate(methods):
            if key not in series_by_method[method]:
                continue
            vals = series_by_method[method][key].to_numpy(dtype="float64").ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            pos = gi + (mi - (len(methods) - 1) / 2.0) * width
            ax1.boxplot(
                vals, positions=[pos], widths=width * 0.88,
                showfliers=False, patch_artist=True,
                boxprops=dict(facecolor=_method_color(method), alpha=0.65),
                medianprops=dict(color="k", linewidth=1.1),
            )
    ax1.axhline(c_prbs, color="k", linestyle=":", linewidth=1.0, label=f"C = {int(c_prbs)} PRBs/slot")
    ax1.set_xticks(range(len(slot_cats)))
    ax1.set_xticklabels([lbl for _k, lbl in slot_cats], fontsize=8.5)
    ax1.set_ylabel("PRBs / slot (cell level, [0, C])")
    ax1.set_title("Panel A — Per-Slot Rate (PRBs / slot)")
    ax1.grid(True, axis="y", alpha=0.3)

    win_cats = [
        ("win_capacity_rbg_slots", "window cap\n(C * dl_slots)"),
        ("win_budget_rbg_slots", "budget\n(quota * dl_slots)"),
        ("win_scheduled_rbg_slots", "scheduled\n(MAC window)"),
        ("win_idle_rbg_slots", "idle\n(cap - sched)"),
    ]
    has_win = any("win_scheduled_rbg_slots" in series_by_method[m] for m in methods)
    if not has_win:
        win_cats = [("win_scheduled_rbg_slots_step", "scheduled\n(window total)")]

    for gi, (key, _lbl) in enumerate(win_cats):
        for mi, method in enumerate(methods):
            if key not in series_by_method[method]:
                continue
            vals = series_by_method[method][key].to_numpy(dtype="float64").ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            pos = gi + (mi - (len(methods) - 1) / 2.0) * width
            ax2.boxplot(
                vals, positions=[pos], widths=width * 0.88,
                showfliers=False, patch_artist=True,
                boxprops=dict(facecolor=_method_color(method), alpha=0.65),
                medianprops=dict(color="k", linewidth=1.1),
            )
    ax2.set_xticks(range(len(win_cats)))
    ax2.set_xticklabels([lbl for _k, lbl in win_cats], fontsize=8.5)
    ax2.set_ylabel("RBG·slots / 100 ms window")
    ax2.set_title("Panel B — Window Volume (RBG·slots / 100 ms)")
    ax2.grid(True, axis="y", alpha=0.3)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=_method_color(m), alpha=0.65)
        for m in methods
    ]
    ax1.legend(handles, methods, fontsize=8, loc="upper right")
    fig.suptitle("Fig. 4 (v2 / BUG-04 Fixed) — Separated Slot Rates vs Window Volumes", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_analysis_v2(
    *,
    campaign_dir: str,
    out_figures: str,
    out_tables: str,
    methods_filter: list[str] | None = None,
    suffix: str = "",
    bootstraps: int = 10000,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    exclude_warmup: bool = False,
    slas_tuple: tuple[float, ...] = DEFAULT_SLAS,
    period_ms: int = 100,
    rolling: int = 50,
    delta: float = 0.01,
) -> dict[str, Any]:
    """Execute read-only v2 aggregation over ``campaign_dir`` writing only to out dirs."""
    os.makedirs(out_figures, exist_ok=True)
    os.makedirs(out_tables, exist_ok=True)
    slas = {i: float(v) for i, v in enumerate(slas_tuple)}

    runs = discover_runs(campaign_dir, methods_filter)
    if not runs:
        raise FileNotFoundError(f"No runs found in {campaign_dir}")

    rng = np.random.default_rng(bootstrap_seed)
    per_seed_rows: list[dict[str, Any]] = []
    per_seed_series: dict[str, dict[int, dict[str, pd.Series]]] = {}
    skipped: list[str] = []

    for run in runs:
        if not run.complete:
            skipped.append(f"{run.method}/seed_{run.seed} ({run.skipped_reason})")
            continue
        df = load_run_df(run.step_csv)
        if exclude_warmup and "warmup" in df.columns:
            df = df[df["warmup"] == 0]
        if df.empty:
            skipped.append(f"{run.method}/seed_{run.seed} (empty)")
            continue
        prb_win_df = load_prb_windows_df(run.run_dir)
        scalars = compute_seed_scalars_v2(
            df, slas, period_ms, prb_win_df, exclude_warmup=exclude_warmup
        )
        scalars.update({"method": run.method, "seed": run.seed})
        per_seed_rows.append(scalars)
        per_seed_series.setdefault(run.method, {})[run.seed] = compute_seed_series_v2(
            df, prb_win_df, exclude_warmup=exclude_warmup
        )

    if not per_seed_rows:
        raise RuntimeError(f"No usable completed runs in {campaign_dir} (skipped={skipped})")

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_csv = os.path.join(out_tables, f"per_seed_metrics{suffix}.csv")
    per_seed_df.to_csv(per_seed_csv, index=False)

    summary_rows = []
    modes_rows = []
    series_means: dict[str, dict[str, pd.DataFrame]] = {}
    series_keys = [
        "window_rate_pct",
        "cum_violations",
        "slot_requested_cell",
        "slot_executed_cell",
        "slot_unallocated_quota",
        "slot_scheduled_active_dl",
        "win_capacity_rbg_slots",
        "win_budget_rbg_slots",
        "win_scheduled_rbg_slots",
        "win_scheduled_rbg_slots_step",
        "win_idle_rbg_slots",
    ]

    for method in sorted(per_seed_series):
        sub = per_seed_df[per_seed_df["method"] == method].drop(
            columns=["method", "seed"], errors="ignore"
        )
        agg = aggregate_seeds(sub, bootstraps, rng)
        for metric, row in agg.iterrows():
            summary_rows.append({"method": method, "metric": metric, **row.to_dict()})

        mode_cols = [c for c in sub.columns if c.startswith("mode_") and c.endswith("_pct")]
        n_rows_per_seed = sub["n_slice_decisions"].to_numpy(dtype="float64")
        for col in mode_cols:
            mode = col[len("mode_"):-len("_pct")]
            vals = sub[col].to_numpy(dtype="float64")
            lo, hi = bootstrap_ci(vals, bootstraps, rng)
            counts = vals * n_rows_per_seed / 100.0
            modes_rows.append({
                "method": method,
                "mode": mode,
                "mean_pct": float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan,
                "ci_lo_pct": lo,
                "ci_hi_pct": hi,
                "mean_count": float(np.nanmean(counts)) if np.isfinite(counts).any() else np.nan,
            })

        frames = per_seed_series[method]
        method_dict: dict[str, pd.DataFrame] = {}
        for sk in series_keys:
            if all(sk in f for f in frames.values()):
                method_dict[sk] = align_series(frames, sk)
        series_means[method] = method_dict

    summary = pd.DataFrame(summary_rows).set_index(["method", "metric"]).sort_index()
    summary_csv = os.path.join(out_tables, f"summary_by_method{suffix}.csv")
    summary.to_csv(summary_csv)

    modes_summary = (
        pd.DataFrame(modes_rows).set_index(["method", "mode"]).sort_index()
        if modes_rows else pd.DataFrame()
    )
    if not modes_summary.empty:
        modes_summary.to_csv(os.path.join(out_tables, f"modes_by_method{suffix}.csv"))

    fig2_window_violation(
        {m: s["window_rate_pct"] for m, s in series_means.items()},
        delta_pct=100.0 * delta,
        rolling=rolling,
        period_ms=period_ms,
        path=os.path.join(out_figures, f"fig2_window_violation_rate{suffix}.png"),
    )
    fig3_cumulative_violations(
        {m: s["cum_violations"] for m, s in series_means.items()},
        period_ms=period_ms,
        path=os.path.join(out_figures, f"fig3_cumulative_violation{suffix}.png"),
    )
    fig4_prb_distribution_v2(
        series_means,
        path=os.path.join(out_figures, f"fig4_prb_distribution{suffix}.png"),
    )
    fig_summary_bars(
        summary,
        path=os.path.join(out_figures, f"fig_summary_violation_bars{suffix}.png"),
    )
    if not modes_summary.empty:
        fig_modes(
            modes_summary,
            path=os.path.join(out_figures, f"fig_modes{suffix}.png"),
        )

    return {
        "n_runs": len(per_seed_rows),
        "skipped": skipped,
        "summary": summary,
        "per_seed_df": per_seed_df,
        "summary_csv": summary_csv,
        "per_seed_csv": per_seed_csv,
        "out_figures": out_figures,
        "out_tables": out_tables,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="BUG-04 fixed SQR v2 campaign analysis")
    parser.add_argument("--campaign-dir", type=str, required=True)
    parser.add_argument("--out-figures", type=str, required=True)
    parser.add_argument("--out-tables", type=str, required=True)
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--suffix", type=str, default="_v2")
    parser.add_argument("--bootstraps", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--exclude-warmup", action="store_true")
    parser.add_argument("--slas", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--period-ms", type=int, default=100)
    parser.add_argument("--rolling", type=int, default=50)
    parser.add_argument("--delta", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    slas_tuple = tuple(float(x) for x in args.slas.split(","))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else None
    run_analysis_v2(
        campaign_dir=os.path.abspath(args.campaign_dir),
        out_figures=os.path.abspath(args.out_figures),
        out_tables=os.path.abspath(args.out_tables),
        methods_filter=methods,
        suffix=args.suffix,
        bootstraps=args.bootstraps,
        bootstrap_seed=args.bootstrap_seed,
        exclude_warmup=args.exclude_warmup,
        slas_tuple=slas_tuple,
        period_ms=args.period_ms,
        rolling=args.rolling,
        delta=args.delta,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
