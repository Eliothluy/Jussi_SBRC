"""Tests for the Fase 4 campaign tooling (ask items 1-4, no ns-3).

Covers:
- ``nsoran.sqr.static_controllers.EqualSplitController`` — the "equal"
  campaign method: split sums to exactly C, constant decisions, dynamic-C
  sync, checkpoint round-trip, integration with the shared run loop of
  ``examples/sqr_run.py`` (mode column, learned=0);
- ``nsoran.sqr.campaign_analysis`` — violation rates with EXPLICIT
  denominators (R19/R26) on a hand-checked synthetic run, per-step series,
  seed-unit bootstrap, run discovery, and the end-to-end main() on a
  synthetic campaign directory (figures + tables produced).
"""

import csv
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_TEST_DIR, "..", "src")
sys.path.insert(0, _SRC)

_SPEC = importlib.util.spec_from_file_location(
    "sqr_run_example",
    os.path.join(_TEST_DIR, "..", "examples", "sqr_run.py"),
)
sqr_run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sqr_run)

from nsoran.sqr.campaign_analysis import (
    align_series,
    bootstrap_ci,
    discover_runs,
    load_run_df,
    main as analysis_main,
    seed_scalars,
    seed_series,
    slice_time_violations,
)
from nsoran.sqr.checkpoint import build_checkpoint, load_checkpoint, restore_controllers
from nsoran.sqr.static_controllers import MODE_EQUAL, EqualSplitController

C = 52
NUM_SLICES = 3
SLAS = {0: 0.5, 1: 1.0, 2: 1.5}


# ---------------------------------------------------------------------------
# EqualSplitController (campaign method "equal", ask item 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c_prbs", [52, 53, 30, 9])
def test_equal_split_sums_exactly_to_c(c_prbs):
    split = EqualSplitController(0, c_prbs).equal_split()
    assert int(split.sum()) == c_prbs
    assert split.shape == (NUM_SLICES,)
    assert np.all(split >= 0)
    # largest-remainder style: balanced within 1 PRB
    assert int(split.max()) - int(split.min()) <= 1


def test_equal_controller_act_is_constant_and_contract_shaped():
    ctl = EqualSplitController(1, C)
    z = np.full(6, 0.5)
    d1, d2 = ctl.act(z), ctl.act(np.zeros(6))
    expected = int(ctl.equal_split()[1])
    assert d1.action == expected and d2.action == expected
    assert d1.mode == MODE_EQUAL == "equal"
    assert d1.raw_action == d1.action
    assert np.isnan(d1.sigma) and np.isnan(d1.tau)
    assert d1.margin_applied == 0


def test_equal_controller_dynamic_c_and_checkpoint_roundtrip():
    ctl = EqualSplitController(2, 52)
    ctl.set_n_prbs(53)  # register R01 as amended: telemetry C
    assert int(ctl.equal_split().sum()) == 53
    payload = build_checkpoint({0: EqualSplitController(0, 52), 2: ctl})
    fresh = {0: EqualSplitController(0, 52), 2: EqualSplitController(2, 52)}
    restore_controllers(payload, fresh)
    assert fresh[2].n_prbs == 53
    with pytest.raises(ValueError):
        fresh[2].load_state_dict({"kind": "other"})


class _FakeEnv:
    """Minimal mock of SqrEnv for run_episode (no projection cuts)."""

    def __init__(self, n_steps=3):
        self.n_steps = n_steps
        self.step_count = 0
        self.received = []

    def reset(self):
        self.step_count = 0
        return np.full((NUM_SLICES, 6), 0.5, dtype=np.float32), {}

    def step(self, proposals):
        proposals = [int(v) for v in proposals]
        self.received.append(proposals)
        self.step_count += 1
        info = {
            "total_prbs": C,           # unchanged -> no dynamic-C sync
            "kpm_missing": False,
            "kpm_missing_steps": 0,
            "kpm_timestamp": 100 * self.step_count,
            "projection": {
                "prb_requested": list(proposals),
                "prb_executed": list(proposals),  # feasible: Pi is identity
                "cut": [False] * NUM_SLICES,
                "over_capacity": False,
                "virtual_updates": [],
            },
        }
        for sid in range(NUM_SLICES):
            info[f"slice_{sid}"] = {
                "n_ues": 2, "p5_thr_mbps": 0.4, "lowest_thr_mbps": 0.3,
                "mean_thr_mbps": 0.6, "queue_bytes_sum": 0.0,
                "violation_rate": 0.5, "n_snr_imputed": 0, "snr_imputations": {},
            }
        done = self.step_count >= self.n_steps
        return np.full((NUM_SLICES, 6), 0.5, dtype=np.float32), -0.5, False, done, info


def test_equal_method_through_shared_run_loop(tmp_path):
    env = _FakeEnv(n_steps=3)
    controllers = {sid: EqualSplitController(sid, C) for sid in range(NUM_SLICES)}
    log_path = tmp_path / "step_metrics.csv"
    logger = sqr_run.StepMetricsLogger(str(log_path))
    summary = sqr_run.run_episode(
        env, controllers, gamma=1.0, freeze=False, seed=1,
        step_logger=logger,
        checkpoint_path=str(tmp_path / "ckpt.pkl"), checkpoint_interval=0,
    )
    split = EqualSplitController(0, C).equal_split().tolist()
    assert env.received == [split] * 3          # constant equal proposals
    df = pd.read_csv(log_path)
    assert set(df["mode"]) == {"equal"}
    assert (df["learned"] == 0).all()           # static: honest learned column
    assert (df["prb_requested"] == df["slice_id"].map(lambda s: split[int(s)])).all()
    assert (df["prb_executed"] == df["prb_requested"]).all()
    assert (df["prb_idle_est"] == 0).all()      # sum(split) == C -> nothing idle
    assert summary["learned_windows"] == {0: 0, 1: 0, 2: 0}


def test_sqr_run_accepts_equal_choice():
    args = sqr_run.parse_args(["--controller", "equal", "--simTime", "60"])
    controllers = sqr_run.build_controllers(args)
    assert all(isinstance(c, EqualSplitController) for c in controllers.values())


# ---------------------------------------------------------------------------
# campaign_analysis: hand-checked synthetic run (R19/R26 denominators)
# ---------------------------------------------------------------------------


def _synthetic_df() -> pd.DataFrame:
    rows = [
        # step, slice, n_ues, violation_rate, y_t (p5), mode, prbs, dt, kpm, warmup
        (1, 0, 2, 0.5, 0.4, "confident", 10, 20, 30, 1.1, 0, 0),
        (1, 1, 1, 0.0, 1.2, "softmax", 20, 20, 25, 1.1, 0, 0),
        # slice 2 empty at step 1: excluded from every denominator (R19)
        (1, 2, 0, np.nan, np.nan, "confident", 22, 20, 0, 1.1, 0, 0),
        (2, 0, 2, 1.0, 0.1, "confident", 12, 14, 12, 0.9, 0, 0),
        (2, 1, 3, 1.0 / 3.0, 0.9, "knn", 18, 14, 16, 0.9, 0, 0),
        (2, 2, 1, 0.0, 2.0, "confident", 23, 14, 25, 0.9, 1, 0),
    ]
    cols = ["step", "slice_id", "n_ues", "violation_rate", "y_t_p5_mbps",
            "mode", "prb_requested", "prb_executed", "prb_scheduled",
            "decision_time_ms", "kpm_missing", "warmup"]
    df = pd.DataFrame(rows, columns=cols)
    df["prb_idle_est"] = 10  # constant cell-level value
    return df


def test_seed_scalars_explicit_denominators():
    df = load_run_df_from_frame(_synthetic_df())
    m = seed_scalars(df, SLAS, period_ms=100)
    # UE-time: count = 0.5*2 + 0 + 1.0*2 + (1/3)*3 + 0 = 4; den = 2+1+2+3+1 = 9
    assert m["ue_time_violation_count"] == pytest.approx(4.0)
    assert m["ue_time_denominator"] == pytest.approx(9.0)
    assert m["ue_time_violation_pct"] == pytest.approx(100.0 * 4.0 / 9.0)
    # slice-time: 3 of the 5 eligible pairs have y_t < Lambda
    assert m["slice_time_denominator"] == pytest.approx(5.0)
    assert m["slice_time_violation_pct"] == pytest.approx(60.0)
    # per-slice breakdown (R26 logged separately)
    assert m["ue_time_pct_slice0"] == pytest.approx(75.0)   # (0.5*2 + 1.0*2) / 4
    assert m["slice_time_pct_slice0"] == pytest.approx(100.0)
    assert m["ue_denominator_slice1"] == pytest.approx(4.0)
    # RB means: per-decision mean across slices = (10+20+22)/3, (12+18+23)/3
    assert m["rb_requested_mean"] == pytest.approx((52 / 3 + 53 / 3) / 2)
    assert m["rb_scheduled_mean"] == pytest.approx((55 / 3 + 53 / 3) / 2)
    assert m["rb_cell_total_requested_mean"] == pytest.approx((52 + 53) / 2)
    # decision time: per-step first values {1: 1.1, 2: 0.9} (pandas interpolation)
    assert m["decision_ms_p50"] == pytest.approx(1.0)
    assert m["decision_ms_p95"] == pytest.approx(1.09)
    # modes: 4 confident + 1 softmax + 1 knn of 6 slice-decisions
    assert m["mode_confident_pct"] == pytest.approx(100.0 * 4 / 6)
    assert m["modes_confident_group_pct"] == pytest.approx(100.0 * 4 / 6)
    assert m["modes_uncertain_group_pct"] == pytest.approx(100.0 * 2 / 6)
    assert m["mode_softmax_pct"] == pytest.approx(100.0 / 6)
    # kpm_missing is a cumulative counter: 1 of 2 windows
    assert m["kpm_missing_pct"] == pytest.approx(50.0)


def test_seed_series_window_rate_and_cumsum():
    df = load_run_df_from_frame(_synthetic_df())
    s = seed_series(df, period_ms=100)
    assert s["window_rate_pct"].loc[1] == pytest.approx(100.0 / 3.0)  # 1 viol of 3 UEs
    assert s["window_rate_pct"].loc[2] == pytest.approx(50.0)         # 3 of 6
    assert s["cum_violations"].tolist() == pytest.approx([1.0, 4.0])
    # scheduled NaN handling: mean across slices of the window, NaN-aware
    assert s["prb_scheduled"].loc[1] == pytest.approx((30 + 25 + 0) / 3)
    # align_series pads different step ranges
    aligned = align_series({1: s, 2: s}, "cum_violations")
    assert aligned.shape == (2, 2)
    assert aligned.mean(axis=1).tolist() == pytest.approx([1.0, 4.0])


def test_slice_time_violation_empty_excluded():
    df = load_run_df_from_frame(_synthetic_df())
    sv = slice_time_violations(df, SLAS)
    assert int(sv.notna().sum()) == 5          # empty (slice2, step1) excluded
    assert float(np.nansum(sv.values)) == 3.0


def load_run_df_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Route the synthetic frame through the real loader (csv round-trip)."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="") as f:
        df.to_csv(f, index=False)
        path = f.name
    try:
        return load_run_df(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Bootstrap on the seed unit + end-to-end main()
# ---------------------------------------------------------------------------


def test_bootstrap_ci_constant_and_binary():
    rng = np.random.default_rng(0)
    lo, hi = bootstrap_ci(np.array([7.0, 7.0, 7.0, 7.0]), 500, rng)
    assert lo == pytest.approx(7.0) and hi == pytest.approx(7.0)
    lo, hi = bootstrap_ci(np.array([0.0, 1.0] * 5), 2000, rng)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    lo, hi = bootstrap_ci(np.array([np.nan, 2.0, 4.0]), 500, rng)
    assert lo <= 3.0 <= hi
    assert bootstrap_ci(np.array([np.nan]), 10, rng) == (np.nan, np.nan)


def _write_run(tmp_path, method, seed, n_steps=6):
    """Deterministic synthetic run: 'sqr-g1' violates on odd steps, 'kbrl'
    only on multiples of 4; slice 2 empty on even steps (R19 exclusion)."""
    run_dir = tmp_path / method / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(1, n_steps + 1):
        for sid in range(NUM_SLICES):
            n_ues = 0 if (sid == 2 and step % 2 == 0) else 2
            if method == "sqr-g1":
                viol = step % 2 == 1
            elif method == "kbrl":
                viol = step % 4 == 0
            else:
                viol = False
            rate = 1.0 if (viol and n_ues) else 0.0
            y = SLAS[sid] * (0.5 if (viol and n_ues) else 1.5)
            rows.append([
                1,                # episode
                step,             # step
                step * 100,       # timestamp_ms
                seed,             # seed
                1,                # gamma
                sid,              # slice_id
                "confident",      # mode
                10 + sid,         # raw_action
                10 + sid,         # action
                10 + sid,         # prb_requested
                10 + sid,         # prb_executed
                10 + sid,         # prb_scheduled (telemetry)
                3,                # prb_idle_est
                0.5,              # sigma
                10.0,             # margin_m
                0,                # margin_applied
                0.05,             # tau
                y,                # y_t_p5_mbps
                y,                # lowest_thr_mbps
                n_ues,            # n_ues
                rate,             # violation_rate
                0,                # n_snr_imputed
                0,                # kpm_missing
                1,                # learned
                0,                # virtual_updates
                1.0,              # decision_time_ms
                float(-rate),     # reward
                0,                # warmup
                False,            # terminated
                step == n_steps,  # truncated
            ])
    header = list(sqr_run.STEP_METRICS_COLUMNS)
    with open(run_dir / "step_metrics.csv", "w", newline="") as f:
        csv.writer(f).writerow(header)
        csv.writer(f).writerows(rows)
    (run_dir / "run_done.json").write_text("{}")


def test_discover_runs_skips_incomplete(tmp_path):
    _write_run(tmp_path, "sqr-g1", 1)
    _write_run(tmp_path, "sqr-g1", 2)
    (tmp_path / "sqr-g1" / "seed_3").mkdir()
    runs = discover_runs(str(tmp_path))
    assert len(runs) == 2
    assert all(r.complete for r in runs)
    (tmp_path / "sqr-g1" / "seed_3" / "step_metrics.csv").write_text("x\n")
    runs = discover_runs(str(tmp_path))
    incomplete = [r for r in runs if not r.complete]
    assert len(incomplete) == 1 and incomplete[0].seed == 3


def test_analysis_main_end_to_end(tmp_path):
    for method in ("sqr-g1", "kbrl"):
        for seed in (1, 2):
            _write_run(tmp_path, method, seed)
    rc = analysis_main([
        "--campaign-dir", str(tmp_path), "--suffix", "_t",
        "--bootstraps", "200",
        "--out-figures", str(tmp_path / "figures"),
        "--out-tables", str(tmp_path / "tables"),
    ])
    assert rc == 0
    for name in (
        "fig2_window_violation_rate_t.png", "fig3_cumulative_violation_t.png",
        "fig4_prb_distribution_t.png", "fig_summary_violation_bars_t.png",
        "fig_modes_t.png",
    ):
        assert (tmp_path / "figures" / name).stat().st_size > 0
    for name in (
        "summary_by_method_t.csv", "violation_by_slice_t.csv",
        "modes_by_method_t.csv", "per_seed_metrics_t.csv",
        "series_sqr-g1_t.csv", "series_kbrl_t.csv",
    ):
        assert (tmp_path / "tables" / name).stat().st_size > 0
    summary = pd.read_csv(tmp_path / "tables" / "summary_by_method_t.csv")
    ue = summary[summary["metric"] == "ue_time_violation_pct"]
    assert set(ue["method"]) == {"sqr-g1", "kbrl"}
    # deterministic synthetic: sqr-g1 violates on the 3 odd steps (18/30
    # UE-windows), kbrl only on step 4 (4/30) — aggregation must reproduce it
    kbrl_mean = float(ue[ue["method"] == "kbrl"]["mean"].iloc[0])
    sqrg1_mean = float(ue[ue["method"] == "sqr-g1"]["mean"].iloc[0])
    assert sqrg1_mean == pytest.approx(100.0 * 18 / 30)
    assert kbrl_mean == pytest.approx(100.0 * 4 / 30)
    assert sqrg1_mean > kbrl_mean
