"""Read-only BUG-04 errata reanalysis of the preserved v1 2 km campaign."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsoran.sqr_v2.campaign_analysis_v2 import run_analysis_v2

GYM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_V1_CAMPAIGN = os.path.join(GYM_ROOT, "results", "sqr_campaign")
DEFAULT_OUT_DIR = os.path.join(GYM_ROOT, "results", "sqr_v2", "reanalysis_v1_bug04")


def build_v1_errata_comparison(
    v1_summary_csv: str, v2_summary_df: pd.DataFrame, out_csv: str
) -> pd.DataFrame:
    """Compare original v1 BUG-04 table values vs read-only corrected v2 values."""
    v1 = pd.read_csv(v1_summary_csv).set_index(["method", "metric"])
    rows = []
    for method in sorted(set(v2_summary_df.index.get_level_values("method"))):
        def _v1(m: str) -> float:
            return float(v1.loc[(method, m), "mean"]) if (method, m) in v1.index else float("nan")

        def _v2(m: str) -> float:
            return float(v2_summary_df.loc[(method, m), "mean"]) if (method, m) in v2_summary_df.index else float("nan")

        rows.append({
            "method": method,
            "v1_buggy_idle_est_3x_sum": _v1("rb_cell_total_idle_est_mean"),
            "v2_fixed_idle_est_first_slot": _v2("rb_slot_unallocated_quota_mean"),
            "v2_slot_executed_cell": _v2("rb_slot_executed_cell_mean"),
            "v2_slot_capacity_c_inferred": _v2("rb_slot_capacity_c_mean"),
            "v2_slot_conservation_err": _v2("rb_slot_conservation_abs_err"),
            "v1_mixed_cell_total_scheduled": _v1("rb_cell_total_scheduled_mean"),
            "v2_slot_scheduled_active_dl": _v2("rb_slot_scheduled_active_dl_mean"),
            "v2_win_scheduled_rbg_slots": _v2("rb_win_scheduled_rbg_slots_mean"),
            "v2_win_idle_rbg_slots": _v2("rb_win_idle_rbg_slots_mean"),
            "v2_win_capacity_rbg_slots": _v2("rb_win_capacity_rbg_slots_mean"),
            "v2_win_conservation_err": _v2("rb_win_conservation_abs_err"),
            "ue_time_violation_pct": _v2("ue_time_violation_pct"),
            "slice_time_violation_pct": _v2("slice_time_violation_pct"),
        })
    comp = pd.DataFrame(rows)
    comp.to_csv(out_csv, index=False)
    return comp


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only BUG-04 reanalysis of v1 sqr_campaign")
    parser.add_argument("--campaign-dir", type=str, default=DEFAULT_V1_CAMPAIGN)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--exclude-warmup", action="store_true", default=False)
    args = parser.parse_args(argv)

    fig_dir = os.path.join(args.out_dir, "figures")
    tab_dir = os.path.join(args.out_dir, "tables")
    res = run_analysis_v2(
        campaign_dir=os.path.abspath(args.campaign_dir),
        out_figures=fig_dir,
        out_tables=tab_dir,
        suffix="_bug04_fixed",
        exclude_warmup=bool(args.exclude_warmup),
    )

    v1_summary = os.path.join(os.path.abspath(args.campaign_dir), "tables", "summary_by_method.csv")
    if os.path.isfile(v1_summary):
        errata_csv = os.path.join(tab_dir, "errata_bug04_v1_vs_v2.csv")
        comp = build_v1_errata_comparison(v1_summary, res["summary"], errata_csv)
        print("\n=== Errata BUG-04 (v1 somente-leitura) ===")
        print(comp.to_string(index=False))
    print(f"\n[reanalyze_v1] OK: n_runs={res['n_runs']}, out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
