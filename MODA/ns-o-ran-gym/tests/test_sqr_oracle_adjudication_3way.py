"""Tests of the Etapa-2.4-II-d parte 2 adjudication (3-way oracle gate).

Synthetic coverage of the pure machinery (group adjudicators D1–D6 against
the frozen §20 expectations, the §21 confusion + conservative views and
metrics, the §22 gate C1–C7, the post-PASS freezing §24–§27 and the SQR-JC
decision §28–§31) + campaign-gated integration tests over the real 49
replays when the replay outputs are present.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsoran.sqr.oracle import adjudication_3way as a3

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROUND2 = os.path.join(ROOT, "results", "sqr_characterization_round2")
ORACLE = os.path.join(ROUND2, "oracle", "oracle_recalibrated")
HAS_PLAN = os.path.isfile(os.path.join(ORACLE, "replay_v2",
                                       "plan_d1d5_metadata.csv"))
HAS_G2 = os.path.isfile(os.path.join(ROUND2, "oracle", "adjudication",
                                     "g2_deltas.csv"))


# ---------------------------------------------------------------------------
# Synthetic frame builders
# ---------------------------------------------------------------------------

def _witRow(rid, group, pred, ok, thr=6.0, p95=5.0, band_e=("FEASIBLE",
                                                            np.nan, np.nan),
            band_u=("FEASIBLE", np.nan, np.nan), in_support=True):
    return {
        "group": group, "replay_id": rid, "role": "witness",
        "in_support_plan": in_support, "replay_valid": True,
        "oracle_state_class": pred, "observed_joint_ok": ok,
        "embb_throughput": thr, "urllc_latency_p95": p95,
        "band_embb_class": band_e[0], "band_embb_p25": band_e[1],
        "band_embb_p75": band_e[2],
        "band_urllc_class": band_u[0], "band_urllc_p25": band_u[1],
        "band_urllc_p75": band_u[2],
        "lambda_total": 2.0, "seed": 1, "window_id": 100,
    }


def _res(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §20 — group adjudicators
# ---------------------------------------------------------------------------

def test_d1_verdict_mapping():
    rows = [
        _witRow("a", "D1", "FEASIBLE", True),
        _witRow("b", "D1", "MARGINAL", True),
        _witRow("c", "D1", "MARGINAL", False),
        _witRow("d", "D1", "INFEASIBLE", True),    # failure of the model
        _witRow("e", "D1", "FEASIBLE", False),     # false feasible
    ]
    d = a3.adjudicate_d1(_res(rows))
    v = dict(zip(d["replay_id"], d["verdict"]))
    assert v["a"].startswith("CONFIRMED_FEASIBLE")
    assert v["b"].startswith("ABSORVED_MARGINAL")
    assert v["c"].startswith("ABSORVED_MARGINAL")
    assert v["d"].startswith("FALSE_INFEASIBLE_NOVO")
    assert v["e"].startswith("FALSE_FEASIBLE")


def test_d2_t1423_infeasible_semantics():
    ok = _witRow("t1423", "D2", "INFEASIBLE", False)
    refuted = _witRow("t1423", "D2", "INFEASIBLE", True)
    assert "CORRECT_INFEASIBLE" in a3.adjudicate_d2(_res([ok]))["verdict"][0]
    assert "FALSE_INFEASIBLE_NOVO" in \
        a3.adjudicate_d2(_res([refuted]))["verdict"][0]


def test_d3_replay_decides():
    viol = _witRow("x", "D3", "INFEASIBLE", False)
    ok = _witRow("x", "D3", "INFEASIBLE", True)
    d1 = a3.adjudicate_d3(_res([viol]))
    d2 = a3.adjudicate_d3(_res([ok]))
    assert bool(d1["corroborates_new_label"][0])
    assert "CORROBORATED" in d1["verdict"][0]
    assert not bool(d2["corroborates_new_label"][0])
    assert "REFUTED" in d2["verdict"][0] and "II-d-v2" in d2["verdict"][0]


def test_d4_coherence():
    good = _witRow("g", "D4", "INFEASIBLE", False)
    bad = _witRow("b", "D4", "INFEASIBLE", True)
    edge = _witRow("e", "D4", "MARGINAL", False)
    d = a3.adjudicate_d4(_res([good, bad, edge]))
    v = dict(zip(d["replay_id"], d["verdict"]))
    assert v["g"].startswith("COHERENT")
    assert v["b"].startswith("INCOHERENT")
    assert v["e"].startswith("COHERENT")


def test_d6_floor_measured_and_decision():
    d6 = pd.DataFrame([
        {"group": "D6", "replay_id": f"am{k}", "role": "calibration",
         "replay_valid": True, "lambda_total": 5.0, "seed": 1,
         "window_id": 202, "override_mmtc": k, "observed_mmtc_ok":
             k >= 2, "mmtc_pdr": 1.0 if k >= 2 else 0.0,
         "mmtc_pdr_100ms": 1.0 if k >= 2 else 0.0,
         "mmtc_pdr_100ms_original": 1.0, "mmtc_pdr_100ms_delta": 0.0}
        for k in range(6)
    ])
    curve, summary = a3.adjudicate_d6(d6, None)
    assert summary["a_m_min_measured"] == 2
    assert summary["a_m_min_decision"] == 3          # orchestrator's call
    assert "sobrante" in summary["decision_rule"]
    assert len(curve) == 6 and set(curve["a_m"]) == set(range(6))


# ---------------------------------------------------------------------------
# §21 — confusion + conservative views
# ---------------------------------------------------------------------------

def test_confusion_matrix_counts_and_metrics():
    rows = [
        # 2 predicted FEASIBLE: 1 all-OK, 1 violated
        _witRow("f1", "D1", "FEASIBLE", True),
        _witRow("f2", "D1", "FEASIBLE", False, thr=2.0),
        # 4 predicted MARGINAL: 3 all-OK, 1 violated
        _witRow("m1", "D5", "MARGINAL", True),
        _witRow("m2", "D5", "MARGINAL", True),
        _witRow("m3", "D5", "MARGINAL", True),
        _witRow("m4", "D5", "MARGINAL", False),
        # 4 predicted INFEASIBLE: 3 violated, 1 all-OK
        _witRow("i1", "D3", "INFEASIBLE", False),
        _witRow("i2", "D4", "INFEASIBLE", False),
        _witRow("i3", "D4", "INFEASIBLE", False),
        _witRow("i4", "D2", "INFEASIBLE", True),
    ]
    conf = a3.confusion_matrix_3way(_res(rows))
    m = conf["matrix"]
    assert m == {"FEASIBLE_all_ok": 1, "FEASIBLE_violation": 1,
                 "MARGINAL_all_ok": 3, "MARGINAL_violation": 1,
                 "INFEASIBLE_all_ok": 1, "INFEASIBLE_violation": 3}
    assert conf["n_witness"] == 10
    assert conf["n_pred_feasible"] == 2 and conf["n_pred_marginal"] == 4 \
        and conf["n_pred_infeasible"] == 4
    # FIR = I∧OK / I = 1/4; FFR = F∧V / F = 1/2
    assert conf["false_infeasible_rate"] == pytest.approx(0.25)
    assert conf["false_feasible_rate"] == pytest.approx(0.5)
    assert conf["marginal_rate"] == pytest.approx(0.4)
    # precision/recall feasible: TP=1, FP=1, FN=M∧OK + I∧OK = 4
    assert conf["precision_feasible"] == pytest.approx(0.5)
    assert conf["recall_feasible"] == pytest.approx(1 / 5)
    # precision/recall infeasible: TP=3, FP=1, FN=F∧V + M∧V = 2
    assert conf["precision_infeasible"] == pytest.approx(0.75)
    assert conf["recall_infeasible"] == pytest.approx(3 / 5)
    # conservative views are plain binary projections
    assert conf["view_feasible"]["TP (pred F, all-OK)"] == 1
    assert conf["view_feasible"]["FN (pred ¬F, all-OK)"] == 4
    assert conf["view_infeasible"]["TP (pred I, violated)"] == 3
    assert conf["view_infeasible"]["FP (pred I, all-OK)"] == 1


def test_confusion_excludes_controls_and_calibration():
    rows = [
        _witRow("w", "D1", "FEASIBLE", True),
        {"group": "G4", "replay_id": "c", "role": "control",
         "in_support_plan": False, "replay_valid": True,
         "oracle_state_class": "FEASIBLE", "observed_joint_ok": True,
         "embb_throughput": 6.0, "urllc_latency_p95": 5.0,
         "band_embb_class": "", "band_embb_p25": np.nan,
         "band_embb_p75": np.nan, "band_urllc_class": "",
         "band_urllc_p25": np.nan, "band_urllc_p75": np.nan,
         "lambda_total": 5.0, "seed": 1, "window_id": 187},
        {"group": "D6", "replay_id": "d", "role": "calibration",
         "in_support_plan": False, "replay_valid": True,
         "oracle_state_class": "FEASIBLE", "observed_joint_ok": True,
         "embb_throughput": 6.0, "urllc_latency_p95": 5.0,
         "band_embb_class": "", "band_embb_p25": np.nan,
         "band_embb_p75": np.nan, "band_urllc_class": "",
         "band_urllc_p25": np.nan, "band_urllc_p75": np.nan,
         "lambda_total": 5.0, "seed": 1, "window_id": 202},
    ]
    conf = a3.confusion_matrix_3way(_res(rows))
    assert conf["n_witness"] == 1


# ---------------------------------------------------------------------------
# §22 — gate C1–C7
# ---------------------------------------------------------------------------

def _cls_frame(state_by_key):
    rows = []
    for (lam, seed, win), sc in state_by_key.items():
        rows.append({"arm": "equal", "lam": lam, "seed": seed, "win": win,
                     "time_ms": (win + 1) * 100, "state_class": sc})
    return pd.DataFrame(rows)


def test_gate_pass_path():
    # 4 F all OK, 1 I violated; D3 violated; C3 states not INFEASIBLE
    rows = [
        _witRow("f1", "D1", "FEASIBLE", True),
        _witRow("f2", "D1", "FEASIBLE", True),
        _witRow("f3", "D1", "MARGINAL", True),
        _witRow("i1", "D3", "INFEASIBLE", False),
        _witRow("i2", "D4", "INFEASIBLE", False),
        _witRow("i3", "D4", "INFEASIBLE", False),
        _witRow("i4", "D2", "INFEASIBLE", False),
    ]
    res = _res(rows)
    cls = _cls_frame({(6.0, 3, 298): "FEASIBLE", (4.0, 3, 1812): "MARGINAL",
                      (5.0, 1, 1955): "FEASIBLE", (4.0, 1, 2210): "MARGINAL",
                      (5.0, 3, 1677): "FEASIBLE"})
    d6s = {"a_m_min_measured": 3, "a_m_min_decision": 3,
           "decision_rule": "x", "levels": []}
    gate = a3.gate_ii_d(res, cls, d6s, a3.confusion_matrix_3way(res))
    ids = {c["id"]: c for c in gate["checks"]}
    assert gate["verdict"] == "PASS"
    assert ids["C1_FIR"]["passed"] and ids["C2_FFR"]["passed"]
    assert ids["C3_avoidable_not_infeasible"]["passed"]
    assert ids["C4_corroborated_coherent"]["passed"]
    assert ids["C5_counterproofs_via_D3"]["passed"]
    assert ids["C6_witnesses_in_support"]["passed"]
    assert ids["C7_near_threshold_absorbed"]["passed"]


def test_gate_c1_fir_threshold():
    rows = [
        _witRow("i1", "D3", "INFEASIBLE", False),
        _witRow("i2", "D4", "INFEASIBLE", True),   # false infeasible
        _witRow("i3", "D4", "INFEASIBLE", True),   # false infeasible
    ]   # FIR = 2/3 > 0.25
    res = _res(rows)
    cls = _cls_frame({(6.0, 3, 298): "FEASIBLE", (4.0, 3, 1812): "FEASIBLE",
                      (5.0, 1, 1955): "FEASIBLE", (4.0, 1, 2210): "FEASIBLE",
                      (5.0, 3, 1677): "FEASIBLE"})
    d6s = {"a_m_min_measured": 3, "a_m_min_decision": 3, "decision_rule": "",
           "levels": []}
    gate = a3.gate_ii_d(res, cls, d6s, a3.confusion_matrix_3way(res))
    ids = {c["id"]: c for c in gate["checks"]}
    assert gate["verdict"] == "STOP"
    assert not ids["C1_FIR"]["passed"]
    # C4 needs both D4 coherent — here both incoherent
    assert not ids["C4_corroborated_coherent"]["passed"]


def test_gate_c3_confirmed_avoidable_becoming_infeasible_fails():
    rows = [_witRow("f1", "D1", "FEASIBLE", True)]
    cls = _cls_frame({(6.0, 3, 298): "INFEASIBLE",      # refuted by II-c
                      (4.0, 3, 1812): "FEASIBLE", (5.0, 1, 1955): "FEASIBLE",
                      (4.0, 1, 2210): "FEASIBLE", (5.0, 3, 1677): "FEASIBLE"})
    d6s = {"a_m_min_measured": 3, "a_m_min_decision": 3, "decision_rule": "",
           "levels": []}
    gate = a3.gate_ii_d(_res(rows), cls, d6s,
                        a3.confusion_matrix_3way(_res(rows)))
    c3 = [c for c in gate["checks"] if c["id"].startswith("C3")][0]
    assert not c3["passed"] and gate["verdict"] == "STOP"


def test_gate_c5_d3_all_ok_fails():
    rows = [_witRow("d3", "D3", "INFEASIBLE", True)]   # the II-d-v2 case
    cls = _cls_frame({(6.0, 3, 298): "FEASIBLE", (4.0, 3, 1812): "FEASIBLE",
                      (5.0, 1, 1955): "FEASIBLE", (4.0, 1, 2210): "FEASIBLE",
                      (5.0, 3, 1677): "FEASIBLE"})
    d6s = {"a_m_min_measured": 3, "a_m_min_decision": 3, "decision_rule": "",
           "levels": []}
    gate = a3.gate_ii_d(_res(rows), cls, d6s,
                        a3.confusion_matrix_3way(_res(rows)))
    c5 = [c for c in gate["checks"] if c["id"].startswith("C5")][0]
    assert not c5["passed"]


def test_gate_c6_witness_out_of_support_fails():
    rows = [_witRow("w", "D1", "FEASIBLE", True, in_support=False)]
    cls = _cls_frame({(6.0, 3, 298): "FEASIBLE", (4.0, 3, 1812): "FEASIBLE",
                      (5.0, 1, 1955): "FEASIBLE", (4.0, 1, 2210): "FEASIBLE",
                      (5.0, 3, 1677): "FEASIBLE"})
    d6s = {"a_m_min_measured": 3, "a_m_min_decision": 3, "decision_rule": "",
           "levels": []}
    gate = a3.gate_ii_d(_res(rows), cls, d6s,
                        a3.confusion_matrix_3way(_res(rows)))
    c6 = [c for c in gate["checks"] if c["id"].startswith("C6")][0]
    assert not c6["passed"]


def test_gate_c7_far_hard_error_fails_and_marginal_boundary():
    # FEASIBLE violated FAR from the SLA (thr 1.0) -> far hard error
    rows = [
        _witRow("ff", "D1", "FEASIBLE", False, thr=1.0),
        # MARGINAL with band covering the realization -> boundary ok
        _witRow("mm", "D5", "MARGINAL", False, thr=2.8,
                band_e=("MARGINAL", 2.5, 4.5)),
    ]
    res = _res(rows)
    cls = _cls_frame({(6.0, 3, 298): "FEASIBLE", (4.0, 3, 1812): "FEASIBLE",
                      (5.0, 1, 1955): "FEASIBLE", (4.0, 1, 2210): "FEASIBLE",
                      (5.0, 3, 1677): "FEASIBLE"})
    d6s = {"a_m_min_measured": 3, "a_m_min_decision": 3, "decision_rule": "",
           "levels": []}
    gate = a3.gate_ii_d(res, cls, d6s, a3.confusion_matrix_3way(res))
    c7 = [c for c in gate["checks"] if c["id"].startswith("C7")][0]
    assert not c7["passed"]            # far hard error (thr 1.0)
    # now fix the hard error to near-threshold and break ONLY the M band
    rows[0] = _witRow("ff", "D1", "FEASIBLE", False, thr=3.1)
    rows[1] = _witRow("mm", "D5", "MARGINAL", False, thr=8.0,
                      band_e=("MARGINAL", 2.5, 4.5))
    res = _res(rows)
    gate = a3.gate_ii_d(res, cls, d6s, a3.confusion_matrix_3way(res))
    c7 = [c for c in gate["checks"] if c["id"].startswith("C7")][0]
    assert not c7["passed"]            # M realization outside band & near
    # and the fully honest variant passes C7
    rows[1] = _witRow("mm", "D5", "MARGINAL", False, thr=2.8,
                      band_e=("MARGINAL", 2.5, 4.5))
    res = _res(rows)
    gate = a3.gate_ii_d(res, cls, d6s, a3.confusion_matrix_3way(res))
    c7 = [c for c in gate["checks"] if c["id"].startswith("C7")][0]
    assert c7["passed"]


# ---------------------------------------------------------------------------
# §24–§27 — freezing + L_N/L_O + SQR-JC decision
# ---------------------------------------------------------------------------

def test_freeze_by_lambda_and_regimes():
    cls = pd.DataFrame({
        "lam": [2.0] * 10 + [7.0] * 10,
        "state_class": ["FEASIBLE"] * 9 + ["INFEASIBLE"]
        + ["INFEASIBLE"] * 9 + ["MARGINAL"],
        "D_min_uncertain": [False] * 20,
        "num_feasible_actions": [10] * 20,
        "num_marginal_actions": [5] * 20,
    })
    bl = a3.freeze_by_lambda(cls)
    assert np.allclose(bl["P_F"] + bl["P_M"] + bl["P_I"], 1.0)
    assert np.allclose(bl["P_inf"], bl["P_I"])
    lam2 = bl[bl["lambda_total"] == 2.0].iloc[0]
    lam7 = bl[bl["lambda_total"] == 7.0].iloc[0]
    assert lam2["P_I"] == pytest.approx(0.1)
    assert lam2["regime"] == "OVERLOAD_TRANSITORIO"   # 0 < P_I < 1 (§24)
    assert lam7["P_I"] == pytest.approx(0.9)
    assert lam7["regime"] == "OVERLOAD_TRANSITORIO" or \
        lam7["regime"] == "OVERLOAD_SEVERO"
    # a load with P_I ~ 0 and high P_F reads NORMAL (§24)
    cls2 = pd.DataFrame({
        "lam": [1.0] * 10,
        "state_class": ["FEASIBLE"] * 9 + ["MARGINAL"],
        "D_min_uncertain": [False] * 10,
        "num_feasible_actions": [10] * 10,
        "num_marginal_actions": [5] * 10,
    })
    bl2 = a3.freeze_by_lambda(cls2)
    assert bl2.iloc[0]["regime"] == "NORMAL"


def test_select_ln_lo_prefers_transitory_not_collapse():
    bl = pd.DataFrame([
        {"lambda_total": 2.0, "P_F": 0.90, "P_M": 0.09, "P_I": 0.01,
         "regime": "NORMAL"},
        {"lambda_total": 4.0, "P_F": 0.60, "P_M": 0.22, "P_I": 0.18,
         "regime": "OVERLOAD_TRANSITORIO"},
        {"lambda_total": 5.0, "P_F": 0.40, "P_M": 0.19, "P_I": 0.41,
         "regime": "OVERLOAD_TRANSITORIO"},
        {"lambda_total": 7.0, "P_F": 0.14, "P_M": 0.14, "P_I": 0.72,
         "regime": "OVERLOAD_SEVERO"},
    ])
    sel = a3.select_ln_lo(bl)
    assert sel["L_N"]["lambda_total"] == 2.0
    assert sel["L_O"]["lambda_total"] == 4.0     # transitory, NOT λ7


def test_sqr_jc_decision_components():
    bl = pd.DataFrame([
        {"lambda_total": 2.0, "P_F": 0.90, "P_M": 0.09, "P_I": 0.01,
         "regime": "NORMAL"},
        {"lambda_total": 5.0, "P_F": 0.27, "P_M": 0.19, "P_I": 0.54,
         "regime": "OVERLOAD_TRANSITORIO"},
    ])
    avr = pd.DataFrame([
        {"lambda_total": 2.0, "scope": "exec_in_support", "AVR": 0.15,
         "AVR_FM": 0.2, "n": 100},
        {"lambda_total": 5.0, "scope": "exec_in_support", "AVR": 0.30,
         "AVR_FM": 0.4, "n": 100},
    ])
    sel = a3.select_ln_lo(bl)
    jc = a3.sqr_jc_decision(bl, avr, sel, sel)
    assert jc["joint_coordination_first"] is True
    assert any("Joint_Coordination" in c for c in jc["components"])
    assert any("Feasibility_Awareness" in c for c in jc["components"])
    assert any("Graceful_Degradation" in c for c in jc["components"])
    assert any("MARGINAL != INFEASIBLE" in inv for inv in jc["invariants"])
    assert any("a_u >= 13" in inv for inv in jc["invariants"])


# ---------------------------------------------------------------------------
# Frozen constants + campaign-gated integration
# ---------------------------------------------------------------------------

def test_confirmed_avoidable_frozen_list():
    # R50: the 5 II-c all-OK G2 windows (g2_deltas sla_todos_ok=True)
    assert sorted(a3.CONFIRMED_AVOIDABLE_II_C) == sorted([
        (6.0, 3, 298), (4.0, 3, 1812), (5.0, 1, 1955), (4.0, 1, 2210),
        (5.0, 3, 1677)])


@pytest.mark.skipif(not HAS_G2, reason="II-c artifacts absent")
def test_confirmed_avoidable_matches_iic_artifact():
    g2 = pd.read_csv(os.path.join(ROUND2, "oracle", "adjudication",
                                  "g2_deltas.csv"))
    ok = g2[g2["sla_todos_ok_no_override"] == True]  # noqa: E712
    art = {(float(r.lam), int(r.seed), int(r.win)) for r in ok.itertuples()}
    assert art == set(a3.CONFIRMED_AVOIDABLE_II_C)


@pytest.mark.skipif(not HAS_PLAN, reason="II-d plan absent")
def test_plan_load_and_support():
    plan = a3.load_plan_d1d5(ORACLE)
    assert len(plan) == 31
    wit = plan[plan["_role"] == "witness"]
    assert len(wit) == 25
    assert bool(wit["_in_support"].all())          # §22-C6 precondition
    plan6, meta6 = a3.load_plan_d6(ORACLE)
    assert len(plan6) == 18 and len(meta6) == 3
    assert set(plan6["action_m"]) == set(range(6))


@pytest.mark.skipif(not HAS_PLAN, reason="II-d plan absent")
def test_replay_dir_uses_int_lam_tag():
    plan = a3.load_plan_d1d5(ORACLE)
    d = a3.replay_dir_3way(ORACLE, plan.iloc[0])
    assert os.path.isdir(d), d                     # .done dirs use lam2 etc.
