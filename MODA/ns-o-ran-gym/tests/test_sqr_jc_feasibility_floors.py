"""Feasibility + floors tests (§39–§49).

Covers: test_joint_feasible, test_joint_marginal, test_joint_infeasible,
test_unknown_not_infeasible, test_marginal_not_infeasible,
test_no_demand_floor_zero, test_conservative_floor_fallback.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from nsoran.sqr_jc.action_search import EvaluationResult
from nsoran.sqr_jc.conditional_floors import (
    DIAGNOSTIC_ANCHOR_MMTC,
    DIAGNOSTIC_ANCHOR_URLLC,
    FALLBACKS,
    FALLBACK_EMBB,
    FALLBACK_MMTC,
    FALLBACK_URLLC,
    compute_floors,
    fallback_from_cal,
)
from nsoran.sqr_jc.feasibility import (
    SLICE_INFEASIBLE,
    SLICE_MARGINAL,
    STATE_FEASIBLE,
    STATE_INFEASIBLE,
    STATE_MARGINAL,
    STATE_MARGINAL_UNRESOLVED,
    classify_slice,
    classify_state,
)


def _result(n: int = 6, exhaustive: bool = True) -> EvaluationResult:
    """Hand-built evaluation: per-action (q25, q75, sigma, supported) per slice.

    Actions (a_e, a_u, a_m) chosen around the historical pattern:
      0: (48, 5, 0)  1: (37, 13, 3)  2: (18, 13, 3)  3: (0, 13, 3)
      4: (0, 0, 0)   5: (53, 0, 0)
    Slice matrix rows: [q25_e, q25_u, q25_m] / [q75_*].
    """
    actions = np.array([
        [48, 5, 0], [37, 13, 3], [18, 13, 3], [0, 13, 3], [0, 0, 0], [53, 0, 0],
    ], dtype=np.int64)
    q25 = np.array([
        [0.20, 0.10, 0.05],   # all positive -> JOINT_FEASIBLE
        [-0.30, 0.05, 0.10],  # eMBB infeasible -> JOINT_INFEASIBLE
        [-0.10, -0.02, 0.05], # q25<0 but q75>=0 (e,u) -> JOINT_MARGINAL
        [-0.40, -0.30, 0.05], # INFEASIBLE
        [ 0.10, 0.05, 0.05],  # FEASIBLE
        [-0.20, 0.05, 0.05],  # MARGINAL
    ])
    q75 = np.array([
        [0.40, 0.30, 0.20],
        [-0.10, 0.25, 0.15],
        [ 0.05, 0.03, 0.15],
        [-0.20, -0.10, 0.15],
        [ 0.30, 0.20, 0.20],
        [ 0.10, 0.25, 0.15],
    ])
    sigma = np.full((n, 3), 0.2)
    supported = np.ones((n, 3), dtype=bool)
    return EvaluationResult(
        actions=actions, safe=q25 + 0.05, q25=q25, q75=q75, sigma=sigma,
        supported=supported, eval_time_ms=0.0, exhaustive=exhaustive,
    )


def _slice_verdict_of(action_row, result):
    from nsoran.sqr_jc.feasibility import classify_joint

    index = int(np.where((result.actions == np.asarray(action_row)).all(axis=1))[0][0])
    return classify_joint(result.actions[index], result, index)


# ---------------------------------------------------------------------------
# Slice classes (§46) and §49 absolutes
# ---------------------------------------------------------------------------

def test_slice_three_way_and_support_rule():
    assert classify_slice(0, 0.1, 0.3, True).cls == "FEASIBLE"
    assert classify_slice(0, -0.5, -0.2, True).cls == SLICE_INFEASIBLE
    assert classify_slice(0, -0.1, 0.2, True).cls == SLICE_MARGINAL
    # out of support -> MARGINAL, NEVER INFEASIBLE (§49)
    assert classify_slice(0, -5.0, -4.0, supported=False).cls == SLICE_MARGINAL
    assert classify_slice(0, float("nan"), float("nan"), supported=False).cls == SLICE_MARGINAL


def test_marginal_not_infeasible():
    """§49 absolute rule at the state level."""
    res = _result()
    # action (18,13,3) is MARGINAL (e,u); never reported INFEASIBLE
    verdict = _slice_verdict_of([18, 13, 3], res)
    assert verdict.cls == "JOINT_MARGINAL"
    st = classify_state(res, total_space_size=6)
    assert st.state_class in (STATE_FEASIBLE, STATE_MARGINAL)
    assert st.state_class != STATE_INFEASIBLE


def test_unknown_not_infeasible():
    """Non-exhaustive search without feasible actions => MARGINAL_UNRESOLVED
    (§39), never INFEASIBLE; analysis family MARGINAL."""
    res = _result(exhaustive=False)
    st = classify_state(res, total_space_size=27720)
    assert st.n_actions_evaluated < st.n_actions_total_space
    if st.n_joint_feasible == 0:
        assert st.state_class == STATE_MARGINAL_UNRESOLVED
        assert st.state_class != STATE_INFEASIBLE
    # even an all-INFEASIBLE evaluated subset is not state-INFEASIBLE
    res_bad = EvaluationResult(
        actions=np.array([[0, 13, 3]]),
        safe=np.array([[-1.0, -1.0, -1.0]]),
        q25=np.array([[-0.5, -0.4, -0.3]]),
        q75=np.array([[-0.2, -0.1, -0.1]]),
        sigma=np.array([[0.1, 0.1, 0.1]]),
        supported=np.array([[True, True, True]]),
        eval_time_ms=0.0, exhaustive=False,
    )
    st_bad = classify_state(res_bad, total_space_size=27720)
    assert st_bad.state_class == STATE_MARGINAL_UNRESOLVED


# ---------------------------------------------------------------------------
# Joint classes (§47) and state classes (§48)
# ---------------------------------------------------------------------------

def test_joint_feasible():
    res = _result()
    verdict = _slice_verdict_of([48, 5, 0], res)
    assert verdict.cls == "JOINT_FEASIBLE"
    assert verdict.n_infeasible == 0
    st = classify_state(res, total_space_size=6)
    assert st.n_joint_feasible >= 1
    assert st.state_class == STATE_FEASIBLE


def test_joint_marginal():
    res = _result()
    verdict = _slice_verdict_of([18, 13, 3], res)
    assert verdict.cls == "JOINT_MARGINAL"
    assert verdict.n_infeasible == 0
    assert 0 < verdict.n_marginal


def test_joint_infeasible():
    res = _result()
    verdict = _slice_verdict_of([37, 13, 3], res)
    assert verdict.cls == "JOINT_INFEASIBLE"
    assert verdict.n_infeasible == 1
    verdict2 = _slice_verdict_of([0, 13, 3], res)
    assert verdict2.cls == "JOINT_INFEASIBLE"


def test_state_infeasible_requires_exhaustive_coverage():
    """§48: INFEASIBLE only when ALL actions of the validated space are
    JOINT_INFEASIBLE and the search covers the whole space."""
    res = _result()
    all_bad = EvaluationResult(
        actions=res.actions.copy(),
        safe=np.full((6, 3), -1.0),
        q25=np.full((6, 3), -0.5),
        q75=np.full((6, 3), -0.2),
        sigma=np.full((6, 3), 0.1),
        supported=np.ones((6, 3), dtype=bool),
        eval_time_ms=0.0, exhaustive=True,
    )
    st = classify_state(all_bad, total_space_size=6)  # 6 == total space
    assert st.state_class == STATE_INFEASIBLE
    # same evidence but a PARTIAL search => NOT infeasible (§39/§48)
    partial = EvaluationResult(
        actions=all_bad.actions[:3].copy(),
        safe=all_bad.safe[:3].copy(), q25=all_bad.q25[:3].copy(),
        q75=all_bad.q75[:3].copy(), sigma=all_bad.sigma[:3].copy(),
        supported=all_bad.supported[:3].copy(),
        eval_time_ms=0.0, exhaustive=False,
    )
    st2 = classify_state(partial, total_space_size=6)
    assert st2.state_class == STATE_MARGINAL_UNRESOLVED


# ---------------------------------------------------------------------------
# Floors (§40–§44)
# ---------------------------------------------------------------------------

def _floors_fixture():
    """Exhaustive evaluation where safe evidence exists: eMBB safe ≥ 0 at
    a_e ∈ {18, 48, 37, 53} rows; URLLC safe ≥ 0 only at a_u = 13 rows; mMTC
    safe ≥ 0 everywhere supported except the last row."""
    actions = np.array([
        [48, 5, 0],     # 0
        [37, 13, 3],    # 1
        [18, 13, 3],    # 2
        [0, 13, 3],     # 3
        [0, 0, 0],      # 4
        [53, 0, 0],     # 5
    ], dtype=np.int64)
    safe = np.array([
        [-0.2, 0.5, 0.4],
        [ 0.1, 0.3, 0.3],
        [ 0.2, 0.2, 0.5],
        [-0.5, 0.1, 0.4],
        [-0.3, -0.1, 0.3],
        [ 0.3, -0.1, 0.2],
    ])
    res = EvaluationResult(
        actions=actions, safe=safe, q25=safe - 0.05, q75=safe + 0.05,
        sigma=np.full((6, 3), 0.2), supported=np.ones((6, 3), dtype=bool),
        eval_time_ms=0.0, exhaustive=True,
    )
    return res


def test_floor_evidence_smallest_supported_safe():
    """§41: floors are the componentwise minima of a_i over the SUPPORTED
    JOINTLY SAFE set (every slice safe on the SAME action) — jointly
    admissible by construction (Σ floors ≤ Σ of any member ≤ C)."""
    res = _floors_fixture()
    fr = compute_floors(res, n_active=(2, 1, 2), queues=(1e5, 100.0, 500.0), c=53)
    # jointly-safe rows: (37,13,3) and (18,13,3) → floors = componentwise min
    assert fr.floors == (18, 13, 3)
    assert fr.source == ("evidence", "evidence", "evidence")
    assert fr.witness_action == (37, 13, 3)  # first jointly-safe witness
    assert sum(fr.floors) <= 53


def test_floor_joint_admissibility_guarantee():
    """§41 step 3: per-slice minima taken from DIFFERENT witnesses can sum
    above C — the jointly-safe rule prevents that by construction."""
    actions = np.array([[48, 5, 0], [5, 13, 3], [0, 0, 40]], dtype=np.int64)
    # per-slice safe rows live on DIFFERENT actions (jointly Σ=48+13+3... the
    # componentwise minima (5,5,0) come from rows 0 and 1 — admissible), but
    # a naive per-slice scan would yield (48→? no) — the guarantee tested is
    # Σ floors ≤ C for ANY safe pattern.
    safe = np.array([
        [0.3, -0.5, -0.5],
        [-0.5, 0.3, 0.3],
        [-0.5, -0.5, 0.3],
    ])
    sup = np.ones((3, 3), dtype=bool)
    # no jointly-safe row → all fallbacks → Σ = 16 ≤ C
    res = EvaluationResult(actions=actions, safe=safe, q25=safe - 0.05,
                           q75=safe + 0.05, sigma=np.full((3, 3), 0.2),
                           supported=sup, eval_time_ms=0.0, exhaustive=True)
    fr = compute_floors(res, n_active=(1, 1, 1), queues=(1.0, 1.0, 1.0), c=53)
    assert sum(fr.floors) <= 53
    # a jointly-safe pattern: floors = componentwise minima of the safe set
    safe2 = np.array([
        [0.3, 0.3, 0.3],   # (48,5,0) jointly safe
        [0.2, 0.2, 0.2],   # (5,13,3) jointly safe
    ])
    res2 = EvaluationResult(actions=actions[:2], safe=safe2, q25=safe2 - 0.05,
                            q75=safe2 + 0.05, sigma=np.full((2, 3), 0.2),
                            supported=sup[:2], eval_time_ms=0.0, exhaustive=True)
    fr2 = compute_floors(res2, n_active=(1, 1, 1), queues=(1.0, 1.0, 1.0), c=53)
    assert fr2.floors == (5, 5, 0)
    assert sum(fr2.floors) <= 53


def test_floor_fallback_when_no_evidence():
    res = _floors_fixture()
    # all-unsafe evaluation -> conservative §43 fallbacks (audited values)
    bad = EvaluationResult(
        actions=res.actions.copy(), safe=np.full((6, 3), -1.0),
        q25=np.full((6, 3), -1.1), q75=np.full((6, 3), -0.9),
        sigma=res.sigma.copy(), supported=res.supported.copy(),
        eval_time_ms=0.0, exhaustive=True,
    )
    fr = compute_floors(bad, n_active=(2, 1, 2), queues=(1e5, 100.0, 500.0), c=53)
    assert fr.floors == (FALLBACK_EMBB, FALLBACK_URLLC, FALLBACK_MMTC)
    assert fr.source == ("fallback", "fallback", "fallback")
    assert (FALLBACK_URLLC, FALLBACK_MMTC) == (13, 3)
    assert sum(fr.floors) <= 53


def test_no_demand_floor_zero():
    res = _floors_fixture()
    # §42: N_active == 0 and queue == 0 => floor 0 (evidence ignored)
    fr = compute_floors(res, n_active=(0, 0, 0), queues=(0.0, 0.0, 0.0), c=53)
    assert fr.floors == (0, 0, 0)
    assert fr.source == ("no_demand", "no_demand", "no_demand")
    # only one slice empty -> only that slice is 0
    fr2 = compute_floors(res, n_active=(0, 1, 2), queues=(0.0, 100.0, 500.0), c=53)
    assert fr2.floors[0] == 0 and fr2.source[0] == "no_demand"
    # queue non-zero with no UE -> NOT no-demand (backlog exists)
    fr3 = compute_floors(res, n_active=(0, 1, 2), queues=(5.0, 100.0, 500.0), c=53)
    assert fr3.source[0] != "no_demand"


def test_floor_requires_exhaustive_evaluation():
    res = _floors_fixture()
    partial = EvaluationResult(
        actions=res.actions[:2].copy(), safe=res.safe[:2].copy(),
        q25=res.q25[:2].copy(), q75=res.q75[:2].copy(),
        sigma=res.sigma[:2].copy(), supported=res.supported[:2].copy(),
        eval_time_ms=0.0, exhaustive=False,
    )
    with pytest.raises(ValueError):
        compute_floors(partial, n_active=(1, 1, 1), queues=(1.0, 1.0, 1.0), c=53)


def test_fallback_from_cal_procedure():
    """§43 recalibration: smallest allocation with SLA ok in ≥95% of states."""
    states = [{"a_i": a} for a in (0, 1, 2, 3, 4, 5)] * 20
    ok = [a >= 3 for s in states for a in [s["a_i"]]]
    assert fallback_from_cal(states, ok, max_allocation=10) == 3


def test_anchors_are_diagnostic_only():
    """§44: 5/1 are NOT applied as rules anywhere; they exist as constants."""
    assert FALLBACKS == (0, 13, 3)          # the rule
    assert (DIAGNOSTIC_ANCHOR_URLLC, DIAGNOSTIC_ANCHOR_MMTC) == (5, 1)


def test_conservative_floor_fallback():
    """§75 exact name: no supported safe evidence -> frozen §43 fallbacks."""
    res = _floors_fixture()
    bad = EvaluationResult(
        actions=res.actions.copy(), safe=np.full((6, 3), -1.0),
        q25=np.full((6, 3), -1.1), q75=np.full((6, 3), -0.9),
        sigma=res.sigma.copy(), supported=res.supported.copy(),
        eval_time_ms=0.0, exhaustive=True,
    )
    fr = compute_floors(bad, n_active=(2, 1, 2), queues=(1e5, 100.0, 500.0), c=53)
    assert fr.floors == (FALLBACK_EMBB, FALLBACK_URLLC, FALLBACK_MMTC)
    assert fr.source == ("fallback", "fallback", "fallback")
