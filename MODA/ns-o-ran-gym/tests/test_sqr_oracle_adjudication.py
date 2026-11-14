"""Tests of the Etapa-2.4-II-c adjudication (Despacho II-c).

Synthetic coverage of the pure machinery (prefix determinism checks, frozen
SLA judgments, group adjudicators, confusion matrix, confidence gate, G4
verdict) + one campaign-gated integration test running the FULL §4+§5 build
over the real 28 replays (skipped when the replay outputs are absent).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsoran.sqr.oracle import adjudication as adj
from nsoran.sqr.oracle.adjudication import (
    RESULTS_COLUMNS,
    _kpm_prefix_identical,
    _monotonic,
    _prb_prefix_identical,
    _row_at,
    _sla_oks,
    adjudicate_g1,
    adjudicate_g2,
    adjudicate_g3,
    adjudicate_g4,
    confidence_gate,
    confusion_counts,
    g4_verdict,
)
from nsoran.sqr.oracle.replay_sources import replay_paths, rlf_count_from_logs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROUND2 = os.path.join(ROOT, "results", "sqr_characterization_round2")
ORACLE = os.path.join(ROUND2, "oracle")
REPLAYS = os.path.join(ORACLE, "executable_validation", "replays")
HAS_CAMPAIGN = os.path.isdir(REPLAYS) and len(os.listdir(REPLAYS)) > 0 \
    and os.path.isdir(os.path.join(ROUND2, "hetero_sweep"))


# ---------------------------------------------------------------------------
# §4 machinery — monotonic stamps and prefix determinism
# ---------------------------------------------------------------------------

def test_monotonic_allows_repeated_stamps_per_ue():
    """Per stamp the telemetry repeats one row per UE/slice: non-decreasing
    file order is the check; any negative step is a disorder."""
    stamps = np.array([100, 100, 100, 200, 200, 90])
    assert not _monotonic(stamps)
    assert _monotonic(np.array([100, 100, 100, 200, 200]))


def test_kpm_prefix_identical_basic_and_broken():
    a = pd.DataFrame({"timestamp": [100, 100, 200], "v": [1.0, 2.0, 3.0]})
    b = a.copy()
    ok, msg = _kpm_prefix_identical(a, b, stamp_max=200)
    assert ok and "2" in msg  # wait: 3 rows <= 200
    b2 = b.copy()
    b2.loc[2, "v"] = 99.0
    ok, msg = _kpm_prefix_identical(a, b2, stamp_max=200)
    assert not ok and "column v" in msg


def test_prb_prefix_ignores_ipc_bookkeeping_but_requires_quota_match():
    # original network-only: sentinels -1/0; quota 18 via budget = 18*slots
    orig = pd.DataFrame({
        "timestamp_ms": [200, 300, 400],
        "dl_slots": [52.0, 0.0, 52.0],
        "prb_scheduled_rbg_0": [100.0, 0.0, 90.0],
        "prb_budget_rbg_0": [18 * 52.0, 0.0, 18 * 52.0],
        "prb_executed_rbg_0": [0, 0, 0],       # network-only sentinel
        "prb_requested_pct_0": [-1.0, -1.0, -1.0],
    })
    for s in (1, 2):  # the checker walks all three slices
        orig[f"prb_scheduled_rbg_{s}"] = 1.0
        orig[f"prb_budget_rbg_{s}"] = 17 * orig["dl_slots"]
        orig[f"prb_executed_rbg_{s}"] = 0
        orig[f"prb_requested_pct_{s}"] = -1.0
    rep = orig.copy()
    rep["prb_executed_rbg_0"] = [18, 18, 18]   # IPC quota visible in replay
    for s in (1, 2):
        rep[f"prb_executed_rbg_{s}"] = 17
    rep["prb_requested_pct_0"] = 33.96
    ok, msg = _prb_prefix_identical(rep, orig, stamp_max=400)
    assert ok, msg
    # dl_slots == 0 rows: quota not recoverable from budget -> skipped there
    rep_bad = rep.copy()
    rep_bad.loc[2, "prb_executed_rbg_0"] = 19   # real divergence with slots
    ok, msg = _prb_prefix_identical(rep_bad, orig, stamp_max=400)
    assert not ok and "quota in force diverges" in msg


# ---------------------------------------------------------------------------
# §5 machinery — frozen SLA judgments (vacuous = OK)
# ---------------------------------------------------------------------------

def test_sla_oks_frozen_rules_and_vacuous():
    ok = _sla_oks({"n_embb": 2, "embb_throughput": 3.5, "urllc_n_pkts": 10,
                   "urllc_latency_p95": 10.0, "mmtc_demand": False,
                   "mmtc_pdr": np.nan})
    assert ok == {"observed_embb_ok": True, "observed_urllc_ok": True,
                  "observed_mmtc_ok": True}  # thresholds are >= / <=
    bad = _sla_oks({"n_embb": 1, "embb_throughput": 3.49, "urllc_n_pkts": 1,
                    "urllc_latency_p95": 10.01, "mmtc_demand": True,
                    "mmtc_pdr": 0.94})
    assert not any(bad.values())


def test_results_columns_exact_protocol_section5_order():
    assert RESULTS_COLUMNS[0:4] == ("group", "replay_id", "lambda_total", "seed")
    assert RESULTS_COLUMNS[-1] == "replay_valid"
    assert len(RESULTS_COLUMNS) == 28
    for col in ("oracle_predicted_feasible", "oracle_predicted_infeasible",
                "observed_joint_ok", "snr_p05", "rlf_count"):
        assert col in RESULTS_COLUMNS


# ---------------------------------------------------------------------------
# §6–§11 adjudicators on synthetic frames
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_results():
    rows = []
    # G1: 2 replays all-OK (false-infeasible), 1 violates
    for i, ok in enumerate([True, True, False]):
        rows.append({"group": "G1", "replay_id": f"G1_{i}", "lambda_total": 2.0,
                     "seed": 1, "window_id": 100 + i,
                     "time_start_ms": (100 + i) * 100,
                     "time_end_ms": (101 + i) * 100,
                     "oracle_predicted_feasible": False,
                     "oracle_predicted_infeasible": True,
                     "oracle_num_feasible_actions": 0,
                     "observed_joint_ok": ok, "replay_valid": True})
    # G2: 2 oracle-feasible, 1 all-OK (TP), 1 violates (FP)
    for i, ok in enumerate([True, False]):
        rows.append({"group": "G2", "replay_id": f"G2_{i}", "lambda_total": 5.0,
                     "seed": 1, "window_id": 200 + i,
                     "time_start_ms": (200 + i) * 100,
                     "time_end_ms": (201 + i) * 100,
                     "oracle_predicted_feasible": True,
                     "oracle_predicted_infeasible": False,
                     "oracle_num_feasible_actions": 56,
                     "observed_joint_ok": ok, "replay_valid": True})
    # G3: 1 oracle-infeasible violating (TN)
    rows.append({"group": "G3", "replay_id": "G3_0", "lambda_total": 7.0,
                 "seed": 1, "window_id": 300, "time_start_ms": 30000,
                 "time_end_ms": 30100, "oracle_predicted_feasible": False,
                 "oracle_predicted_infeasible": True,
                 "oracle_num_feasible_actions": 0,
                 "observed_joint_ok": False, "replay_valid": True})
    # G4 excluded from the confusion matrix
    rows.append({"group": "G4", "replay_id": "G4_0", "lambda_total": 5.0,
                 "seed": 1, "window_id": 400, "time_start_ms": 40000,
                 "time_end_ms": 40100, "oracle_predicted_feasible": True,
                 "oracle_predicted_infeasible": False,
                 "oracle_num_feasible_actions": 27720,
                 "observed_joint_ok": True, "replay_valid": True})
    df = pd.DataFrame(rows)
    df["embb_throughput"] = 4.0
    df["urllc_deadline_miss"] = 5.0
    df["urllc_latency_p95"] = 9.0
    df["mmtc_pdr"] = 1.0
    df["util_active_dl"] = 50.0
    df["override_embb"] = [12, 12, 12, 48, 48, 48, 39]
    df["override_urllc"] = [13, 13, 13, 5, 5, 5, 13]
    df["override_mmtc"] = [28, 28, 28, 0, 0, 0, 1]
    return df


def test_confusion_counts_infeasible_positive_class(synth_results):
    cm = confusion_counts(synth_results)
    assert cm["n_replays"] == 6
    assert cm["TP_feas"] == 1 and cm["FP_feas"] == 1
    assert cm["FN_feas_false_infeasible"] == 2 and cm["TN_infeas"] == 2
    assert cm["accuracy"] == pytest.approx(3 / 6)
    assert cm["precision_inf"] == pytest.approx(2 / 4)
    assert cm["recall_inf"] == pytest.approx(2 / 3)
    assert cm["false_infeasible_rate"] == pytest.approx(2 / 4)


def test_adjudicate_g1_counts_p25_auxiliary_criterion(synth_results):
    g1 = adjudicate_g1(synth_results)
    assert g1["n_executed"] == 3
    assert g1["false_infeasible_count"] == 2
    assert g1["false_infeasible_rate"] == pytest.approx(2 / 3)
    assert "p25" in g1["criterion"]


def test_row_at_mask_lookup_time_and_time_ms_columns():
    ex = pd.DataFrame({"arm": ["equal", "equal"], "lam": [5.0, 5.0],
                       "seed": [1, 3], "time_ms": [11100, 11100],
                       "a_u": [17, 17]})
    r = _row_at(ex, "equal", 5.0, 3, 11100)
    assert r is not None and int(r["a_u"]) == 17
    assert _row_at(ex, "prop", 5.0, 3, 11100) is None
    ow = pd.DataFrame({"arm": ["equal"], "lambda_total": [7.0], "seed": [1],
                       "time": [30100], "min_total_deficit": [0.022]})
    r2 = _row_at(ow, "equal", 7.0, 1, 30100)
    assert r2 is not None and float(r2["min_total_deficit"]) == pytest.approx(0.022)


def test_adjudicate_g3_grades_literal_vs_strict(synth_results):
    ow = pd.DataFrame({
        "arm": ["equal", "prop"], "lambda_total": [7.0, 7.0], "seed": [1, 1],
        "time": [30100, 30100], "min_total_deficit": [0.022, 0.022],
        "observed_embb_violation": [1.0, 0.0], "observed_urllc_violation": [0.0, 0.0],
        "observed_mmtc_violation": [np.nan, np.nan]})
    ex = pd.DataFrame({"arm": ["equal"], "lam": [7.0], "seed": [1],
                       "time_ms": [30100], "thr_embb_med": [1.28]})
    g3 = adjudicate_g3(synth_results[synth_results["group"] == "G3"], ow, ex)
    # replay violated at a*; prop all-OK => executable counterexample
    assert bool(g3.loc[0, "corrobora_a_star"])
    assert bool(g3.loc[0, "contraprova_executavel"])
    assert not bool(g3.loc[0, "corrobora_estrita"])
    # observed URLLC deficit from the synthetic p95 = 9.0 -> zero deficit
    assert float(g3.loc[0, "D_min_observado"]) == pytest.approx(0.0)


def test_adjudicate_g4_curve_includes_base_reference(synth_results):
    ex = pd.DataFrame({"arm": ["equal"], "lam": [5.0], "seed": [1],
                       "time_ms": [40100], "a_u": [17], "a_e": [18],
                       "latency_urllc_miss_pct": [5.5],
                       "latency_urllc_p95_ms": [10.03]})
    curve = adjudicate_g4(synth_results[synth_results["group"] == "G4"], ex)
    assert set(curve["fonte"]) == {"replay_override", "base_equal_original"}


def test_g4_verdict_resource_dependent_vs_scheduler():
    flat = pd.DataFrame([
        {"seed": 1, "win": 1, "fonte": "replay_override", "a_u": a,
         "deadline_miss_pct": 5.5, "latency_p95_ms": 10.03}
        for a in (13, 25, 35)])
    v = g4_verdict(flat)
    assert v["interp_B_scheduler_tdd"] == 1
    drop = pd.DataFrame([
        {"seed": 1, "win": 1, "fonte": "replay_override", "a_u": a,
         "deadline_miss_pct": m, "latency_p95_ms": 12.0 - 0.5 * i}
        for i, (a, m) in enumerate([(13, 12.0), (25, 6.0), (35, 2.0)])])
    v2 = g4_verdict(drop)
    assert v2["interp_A_resource_dependent"] == 1


def test_confidence_gate_stop_and_pass_paths(synth_results):
    g1 = adjudicate_g1(synth_results)
    g2 = pd.DataFrame({"sla_todos_ok_no_override": [True] * 7 + [False]})
    g3 = pd.DataFrame({"corrobora_a_star": [True] * 6,
                       "corrobora_estrita": [True] * 6,
                       "contraprova_executavel": [False] * 6})
    cm = confusion_counts(synth_results)
    gate = confidence_gate(synth_results, g1, g2, g3, cm)
    assert gate["verdict"] == "STOP"  # FIR 2/3 > 0.25 on this synthetic G1
    g1_ok = {"false_infeasible_rate": 0.0, "false_infeasible_count": 0,
             "n_executed": 8, "criterion": "p25 auxiliary witness port (R48)"}
    gate2 = confidence_gate(synth_results, g1_ok, g2, g3, cm)
    assert gate2["verdict"] == "PASS"
    g3_bad = pd.DataFrame({"corrobora_a_star": [True] * 6,
                           "corrobora_estrita": [True, True] + [False] * 4,
                           "contraprova_executavel": [False, False] + [True] * 4})
    gate3 = confidence_gate(synth_results, g1_ok, g2, g3_bad, cm)
    assert gate3["verdict"] == "STOP"  # strict ICR 2/6 < 5/6


# ---------------------------------------------------------------------------
# replay_sources
# ---------------------------------------------------------------------------

def test_rlf_count_and_replay_paths(tmp_path):
    (tmp_path / "stdout").write_text("nothing here\nRLF seen\n")
    (tmp_path / "stderr").write_text("RadioLinkFailure traceline\n")
    assert rlf_count_from_logs(str(tmp_path)) == 2
    with pytest.raises(FileNotFoundError):
        replay_paths(str(tmp_path))


# ---------------------------------------------------------------------------
# Full §4+§5 build over the REAL 28 replays (campaign-gated)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CAMPAIGN, reason="replays 2.4-II-b ausentes neste checkout")
def test_full_adjudication_build_real_replays():
    res, validations = adj.build_replay_results(ORACLE, ROUND2)
    assert len(res) == 28
    assert int(res["replay_valid"].sum()) == 28
    assert list(res.columns)[:len(RESULTS_COLUMNS)] == list(RESULTS_COLUMNS)
    # every replay executed its plan override at the target window
    assert res["executed_action"].str.startswith("(").all()
    cm = confusion_counts(res)
    assert cm["n_replays"] == 22
    assert cm["TP_feas"] == 5 and cm["FP_feas"] == 3
    assert cm["FN_feas_false_infeasible"] == 7 and cm["TN_infeas"] == 7
    g1 = adjudicate_g1(res)
    assert g1["n_executed"] == 8 and g1["false_infeasible_count"] == 7
