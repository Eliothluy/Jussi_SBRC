"""§76 regression tests — qualitative patterns of Round 2.

Protocol §76: IF the unit and semantics remain compatible (G1 audit says
yes: R27 absolute RBG quota, 1 RBG = 1 PRB, ALLOCATED_USAGE), the Round-2
findings enter as QUALITATIVE regression tests — no identical numeric
reproduction is demanded after the physical-scenario change:

    - cases recovered by (48, 5, 0);
    - cases NOT recovered by (37, 13, 3);
    - URLLC-sensitive windows;
    - mMTC zero-allocation failure;
    - robust high-load failures.

Two evidence layers:

1. FROZEN EVIDENCE (real Round-2 CSVs): when the frozen adjudication
   artifacts exist, verify the qualitative patterns the protocol names are
   PRESENT in them (data availability guard, honest skip otherwise):
   ``oracle/adjudication/g2_deltas.csv`` — the G2 group of (48,5,0)
   recoveries with ``sla_todos_ok_no_override``; and
   ``oracle/oracle_recalibrated/adjudication/oracle_3way_replay_results.csv``
   — the (37,13,3) probes and pisos-family bases.
2. SYNTHETIC EMBODIMENT: the SQR-JC machinery (classify → floors → select
   → decide) reproduces the qualitative classes over HANDCRAFTED kernel
   memories that embody the Round-2 findings (anchors in the same 26-dim
   feature space).  The online-learning mechanics themselves are covered
   by test_sqr_jc_kernel; here the fixture memory realizes the qualitative
   response surfaces directly so the patterns are tested deterministically.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

DIM = 26  # 23 joint-state + 3 action dims

#: Frozen Round-2 evidence (§76 patterns).
EVIDENCE_G2_CSV = os.path.join(
    os.path.dirname(__file__), "..", "results", "sqr_characterization_round2",
    "oracle", "adjudication", "g2_deltas.csv",
)
EVIDENCE_3WAY_CSV = os.path.join(
    os.path.dirname(__file__), "..", "results", "sqr_characterization_round2",
    "oracle", "oracle_recalibrated", "adjudication",
    "oracle_3way_replay_results.csv",
)


def _make_scaler():
    from nsoran.sqr_jc.feature_scaler import FeatureScaler

    rng = np.random.default_rng(23)
    data = np.column_stack([
        rng.uniform(0, 8, 120), rng.uniform(0, 4, 120), rng.uniform(0, 6, 120),
        rng.uniform(0, 2e6, 120), rng.uniform(0, 1e4, 120), rng.uniform(0, 1e5, 120),
        rng.uniform(0, 10, 120), rng.uniform(0, 20, 120), rng.uniform(0.5, 1, 120),
        rng.uniform(0, 30, 120), rng.uniform(0, 1, 120), rng.uniform(0, 1, 120),
        rng.uniform(0, 30, 120), rng.uniform(0, 30, 120), rng.uniform(0, 30, 120),
        rng.uniform(0, 53, 120), rng.uniform(0, 53, 120), rng.uniform(0, 53, 120),
        rng.integers(0, 2, 120), rng.integers(0, 2, 120), rng.integers(0, 2, 120),
        rng.uniform(0, 1, 120), rng.uniform(0, 20, 120),
    ]).astype(float)
    s = FeatureScaler(c=53)
    s.fit(data)
    return s


def _state(**kw):
    from nsoran.sqr_jc.joint_state import JointState

    base = dict(n_e=1, n_u=1, n_m=2, queue_e=2e5, queue_u=100.0, queue_m=1e3,
                kpi_e_raw=3.0, kpi_u_raw=8.0, kpi_m_raw=0.99,
                offered_e=5.0, offered_u=0.26, offered_m=0.004,
                snr_e=20.0, snr_u=20.0, snr_m=20.0,
                prev_a_e=18, prev_a_u=13, prev_a_m=3,
                viol_e=1, viol_u=0, viol_m=0,
                urllc_miss_raw=0.02, urllc_p99_raw=11.0, window_index=1500)
    base.update(kw)
    return JointState(**base)


def _fixture_controller(state_probe, sigma_support=0.55):
    """Controller whose slice memories EMBODY the Round-2 qualitative
    physics around ``state_probe`` (anchors in the 26-dim x space):

    eMBB  : margin ≥ 0 iff a_e ≥ ~30 (low/mid load)  — the (48,5,0)
            recovery region; a_e ≲ 12 violates.
    URLLC : margin < 0 when a_u ≤ 2 (zero/tiny allocation), ≥ 0 at a_u ≥ 5
            — the URLLC-sensitive windows.
    mMTC  : margin < 0 at a_m = 0 (zero-allocation failure), ≥ 0 at a_m ≥ 3.
    HIGH-load state (n_e ≥ 4): NO allocation satisfies eMBB — robust
            failure anchors at the high-load region.
    """
    from nsoran.sqr_jc.joint_controller import JointController
    from nsoran.sqr_jc.kernel_models import MultiQuantileKernelModel
    from nsoran.sqr_jc.feature_scaler import FeatureScaler

    scaler = _make_scaler()
    z = state_probe.vector(scaler)  # normalized 23-dim state

    def x_of(a):
        return np.concatenate([z, np.asarray(a, dtype=np.float64) / 53.0])

    def _model(anchors):
        """anchors: list of (action, alpha_safe, alpha_q25, alpha_q75)."""
        m = MultiQuantileKernelModel(dim=DIM, length_scale=0.05)
        pts = np.stack([x_of(a) for a, *_ in anchors])
        m.set_memory(
            pts,
            np.array([a[1] for a in anchors]),
            np.array([a[2] for a in anchors]),
            np.array([a[3] for a in anchors]),
        )
        return m

    high = state_probe.n_e >= 4
    # eMBB: recovery region at a_e = 48; violation anchors at a_e ≈ 10 and
    # (robust high-load) at a_e = 37 with the pisos split.
    embb_anchors = [
        ((48, 5, 0), +0.30, +0.25, +0.35),
        ((10, 5, 0), -0.25, -0.20, -0.15),
    ]
    if high:
        embb_anchors.append(((37, 13, 3), -0.30, -0.25, -0.20))
    urllc_anchors = [
        ((48, 5, 0), +0.15, +0.10, +0.20),
        ((10, 0, 3), -0.40, -0.35, -0.30),
    ]
    mmtc_anchors = [
        ((40, 5, 0), -0.25, -0.20, -0.15),
        ((40, 5, 6), +0.05, +0.04, +0.06),
    ]
    models = (_model(embb_anchors), _model(urllc_anchors), _model(mmtc_anchors))
    return JointController(models, scaler, sigma_support=sigma_support, c=53,
                           bootstrap_seconds=0.0), scaler


def _verdict_of(jc, scaler, st, action):
    from nsoran.sqr_jc.action_search import JointActionOracleEvaluator
    from nsoran.sqr_jc.feasibility import classify_joint

    ev = JointActionOracleEvaluator(jc.models, sigma_support=jc.sigma_support, c=53)
    res = ev.evaluate(st.vector(scaler))
    idx = int(np.where((res.actions == np.asarray(action)).all(axis=1))[0][0])
    return classify_joint(res.actions[idx], res, idx)


# ---------------------------------------------------------------------------
# FROZEN EVIDENCE layer (skipped honestly when the file is absent)
# ---------------------------------------------------------------------------

def test_round2_evidence_patterns_present():
    """The frozen Round-2 evidence must contain the §76 qualitative
    patterns: (48,5,0) recoveries (G2) and (37,13,3)/pisos-family probes
    with the 3-way state classes."""
    if not os.path.exists(EVIDENCE_G2_CSV):
        pytest.skip("Round-2 G2 evidence not present: " + EVIDENCE_G2_CSV)
    import csv

    with open(EVIDENCE_G2_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "empty G2 evidence file"
    rec = [r for r in rows if r["acao_alternativa"].strip() == "(48,5,0)"
           and r["sla_todos_ok_no_override"].strip().lower() == "true"]
    assert rec, "no (48,5,0) recovery row in the frozen G2 evidence"

    if not os.path.exists(EVIDENCE_3WAY_CSV):
        pytest.skip("Round-2 3-way evidence not present: " + EVIDENCE_3WAY_CSV)
    with open(EVIDENCE_3WAY_CSV, newline="") as fh:
        rows3 = list(csv.DictReader(fh))
    assert rows3
    for col in ("base_action_embb", "override_embb", "oracle_state_class",
                "observed_joint_ok"):
        assert col in rows3[0], f"3-way evidence lacks {col}"
    overrides = {(r["override_embb"], r["override_urllc"], r["override_mmtc"])
                 for r in rows3}
    assert ("37", "13", "3") in overrides
    # pisos-family probes (a_u=13, a_m=3) exist among the overrides and the
    # bases are the historical equal-split allocation
    assert any(tuple(map(int, o))[1:] == (13, 3) for o in overrides)
    bases = {(r["base_action_embb"], r["base_action_urllc"], r["base_action_mmtc"])
             for r in rows3}
    assert ("18", "17", "18") in bases


# ---------------------------------------------------------------------------
# SYNTHETIC EMBODIMENT layer (machinery regression on embodied patterns)
# ---------------------------------------------------------------------------

def test_recovered_by_48_5_0_pattern():
    """§76 'casos recuperados por (48,5,0)': at low/mid load the eMBB slice
    is NOT confidently infeasible at (48,5,0) — the recovery region."""
    st = _state(n_e=1, n_m=0)
    jc, scaler = _fixture_controller(st)
    verdict = _verdict_of(jc, scaler, st, (48, 5, 0))
    embb = verdict.slices[0]
    assert embb.cls in ("FEASIBLE", "MARGINAL")
    assert embb.q25_used >= -0.05
    assert embb.q75_used > 0.0


def test_not_recovered_by_37_13_3_pattern():
    """§76 'casos NÃO recuperados por (37,13,3)': at high load the pisos
    split cannot satisfy eMBB — (37,13,3) is not a recovery."""
    st = _state(n_e=4, queue_e=1.9e6)
    jc, scaler = _fixture_controller(st)
    verdict = _verdict_of(jc, scaler, st, (37, 13, 3))
    assert verdict.slices[0].cls == "INFEASIBLE"
    assert verdict.cls == "JOINT_INFEASIBLE"


def test_urllc_sensitive_windows():
    """§76 'URLLC-sensitive windows': a_u = 0 is URLLC-INFEASIBLE; a_u = 5
    is not."""
    st = _state(n_e=1, n_m=0)
    jc, scaler = _fixture_controller(st)
    bad = _verdict_of(jc, scaler, st, (10, 0, 3))
    assert bad.slices[1].cls == "INFEASIBLE"
    good = _verdict_of(jc, scaler, st, (48, 5, 0))
    assert good.slices[1].cls != "INFEASIBLE"


def test_mmtc_zero_allocation_failure():
    """§76 'mMTC zero-allocation failure': a_m = 0 is mMTC-INFEASIBLE; the
    served anchor (a_m = 6) is not."""
    st = _state(n_e=1, n_m=2)
    jc, scaler = _fixture_controller(st)
    verdict = _verdict_of(jc, scaler, st, (40, 5, 0))
    assert verdict.slices[2].cls == "INFEASIBLE"
    ok = _verdict_of(jc, scaler, st, (40, 5, 6))
    assert ok.slices[2].cls != "INFEASIBLE"


def test_robust_high_load_failures():
    """§76 'robust high-load failures': at high load the controller never
    returns a fake FEASIBLE claim; the decision degrades gracefully with
    floors respected and Σ ≤ C."""
    st = _state(n_e=4, queue_e=1.9e6, kpi_e_raw=1.5)
    jc, scaler = _fixture_controller(st)
    d = jc.decide(st, now_s=200.0)
    assert sum(d.action) <= 53
    assert all(a >= f for a, f in zip(d.action, d.floors.floors))
    # the chosen action cannot claim a satisfying eMBB margin at a tiny a_e
    if d.state_class == "FEASIBLE":
        assert d.selected_verdict.slices[0].q25_used >= 0.0
        assert d.action[0] >= 30  # evidenced region, not a fake claim
    else:
        assert d.state_class in ("MARGINAL", "INFEASIBLE", "MARGINAL_UNRESOLVED")
