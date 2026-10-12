"""Tests for nsoran.sqr.coordinator (integer proportional projection).

Checklist (PLANO_IMPLEMENTACAO_SQR_NORI.md §3-Fase 2 / GUIA §3.6):
integer viability (sum <= C, 0 <= a_i <= proposal, whole PRBs), never above
the proposal, determinism, virtual negative updates registered for cut
slices, edge cases (C=0, zero proposals) and input validation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from nsoran.sqr.coordinator import VirtualUpdate, project_prbs


# ── Feasible proposals pass through unchanged ───────────────────────

def test_within_capacity_passthrough():
    res = project_prbs([10, 20, 10], total_prbs=52)
    assert np.array_equal(res.allocated, np.array([10, 20, 10]))
    assert not res.cut.any()
    assert res.virtual_updates == ()
    assert res.total_allocated == 40
    assert not res.over_capacity


def test_exact_capacity_passthrough():
    res = project_prbs([20, 20, 12], total_prbs=52)
    assert np.array_equal(res.allocated, np.array([20, 20, 12]))
    assert res.total_allocated == 52
    assert not res.over_capacity


def test_all_zero_proposals():
    res = project_prbs([0, 0, 0], total_prbs=52)
    assert np.array_equal(res.allocated, np.zeros(3, dtype=int))
    assert res.virtual_updates == ()


# ── Over capacity: integer viability ────────────────────────────────

def test_over_capacity_sum_equals_c_and_caps_at_proposal():
    res = project_prbs([40, 40, 40], total_prbs=52)
    assert res.allocated.dtype.kind == "i"
    assert int(res.allocated.sum()) == 52
    assert np.all(res.allocated >= 0)
    assert np.all(res.allocated <= 40)  # never above the proposal
    # Equal proposals -> near-equal shares (largest remainder).
    assert int(res.allocated.max() - res.allocated.min()) <= 1


def test_never_above_proposal_with_zero_slices():
    res = project_prbs([0, 60, 0], total_prbs=52)
    assert np.array_equal(res.allocated, np.array([0, 52, 0]))


def test_single_slice_takes_exactly_c():
    res = project_prbs([100], total_prbs=52)
    assert np.array_equal(res.allocated, np.array([52]))


def test_proportionality_preserves_ordering():
    """Larger proposals keep larger (or equal) shares after projection."""
    proposals = [70, 10, 40]
    res = project_prbs(proposals, total_prbs=52)
    a = res.allocated
    assert a[0] >= a[2] >= a[1]


def test_tie_break_lowest_index_deterministic():
    """[33,33,33], C=52: floors are 17 each, one unit left -> slice 0."""
    res = project_prbs([33, 33, 33], total_prbs=52)
    assert np.array_equal(res.allocated, np.array([18, 17, 17]))


def test_determinism_repeated_calls():
    p = [41, 7, 63, 25]
    r1 = project_prbs(p, total_prbs=52)
    r2 = project_prbs(p, total_prbs=52)
    assert np.array_equal(r1.allocated, r2.allocated)
    assert r1.virtual_updates == r2.virtual_updates


def test_randomized_viability_invariants():
    """Property test over seeded random cases: integers, sum <= C,
    0 <= a_i <= p_i, sum == min(C, sum(p)) when C > 0, deterministic."""
    rng = np.random.default_rng(2026)
    for _ in range(1000):
        n = int(rng.integers(1, 7))
        p = rng.integers(0, 81, size=n)
        c = int(rng.integers(0, 107))
        r1 = project_prbs(p, c)
        r2 = project_prbs(p, c)
        assert r1.allocated.dtype.kind == "i"
        assert int(r1.allocated.sum()) <= c
        assert np.all(r1.allocated >= 0)
        assert np.all(r1.allocated <= p)
        assert np.array_equal(r1.allocated, r2.allocated)
        if c > 0:
            assert int(r1.allocated.sum()) == min(c, int(p.sum()))
        expected_cut = r1.allocated < p
        assert np.array_equal(r1.cut, expected_cut)


def test_zero_capacity_yields_zeros():
    res = project_prbs([30, 40, 50], total_prbs=0)
    assert np.array_equal(res.allocated, np.zeros(3, dtype=int))
    assert res.over_capacity


# ── Virtual negative updates for cut slices ─────────────────────────

def test_virtual_updates_registered_for_every_cut_slice():
    res = project_prbs([40, 40, 40], total_prbs=52)
    assert res.cut.tolist() == [True, True, True]
    assert len(res.virtual_updates) == 3
    total_deficit = sum(u.deficit for u in res.virtual_updates)
    assert total_deficit == 120 - 52
    for i, u in enumerate(res.virtual_updates):
        assert isinstance(u, VirtualUpdate)
        assert u.slice_id == i
        assert u.proposed == 40
        assert u.allocated == res.allocated[i]
        assert u.deficit == 40 - int(res.allocated[i])
        assert u.deficit > 0


def test_no_virtual_update_for_uncut_slices():
    res = project_prbs([0, 60, 0], total_prbs=52)
    assert len(res.virtual_updates) == 1
    u = res.virtual_updates[0]
    assert u.slice_id == 1
    assert (u.proposed, u.allocated, u.deficit) == (60, 52, 8)


def test_no_virtual_updates_when_feasible():
    res = project_prbs([5, 5, 5], total_prbs=52)
    assert res.virtual_updates == ()
    assert not res.cut.any()


def test_updates_match_cut_mask_and_deficits():
    rng = np.random.default_rng(5)
    for _ in range(200):
        n = int(rng.integers(1, 6))
        p = rng.integers(0, 81, size=n)
        c = int(rng.integers(0, 107))
        res = project_prbs(p, c)
        cut_ids = sorted(u.slice_id for u in res.virtual_updates)
        assert cut_ids == [i for i in range(n) if res.cut[i]]
        for u in res.virtual_updates:
            assert u.deficit == u.proposed - u.allocated
            assert u.allocated == int(res.allocated[u.slice_id])


# ── Input validation ────────────────────────────────────────────────

def test_fractional_proposals_rejected():
    for bad in ([10.5, 20], [10, np.pi]):
        try:
            project_prbs(bad, total_prbs=52)
        except ValueError:
            continue
        raise AssertionError(f"fractional proposals {bad} must raise")


def test_negative_proposals_rejected():
    try:
        project_prbs([-1, 20], total_prbs=52)
    except ValueError:
        pass
    else:
        raise AssertionError("negative proposals must raise")


def test_empty_and_2d_proposals_rejected():
    for bad in ([], np.zeros((2, 2), dtype=int)):
        try:
            project_prbs(bad, total_prbs=52)
        except ValueError:
            continue
        raise AssertionError(f"proposals {bad} must raise")


def test_negative_capacity_rejected():
    try:
        project_prbs([1, 2], total_prbs=-5)
    except ValueError:
        pass
    else:
        raise AssertionError("negative capacity must raise")


def test_non_finite_proposals_rejected():
    try:
        project_prbs([np.nan, 20], total_prbs=52)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN proposals must raise")


# ── Run manually ────────────────────────────────────────────────────

if __name__ == "__main__":
    test_within_capacity_passthrough()
    test_exact_capacity_passthrough()
    test_all_zero_proposals()
    test_over_capacity_sum_equals_c_and_caps_at_proposal()
    test_never_above_proposal_with_zero_slices()
    test_single_slice_takes_exactly_c()
    test_proportionality_preserves_ordering()
    test_tie_break_lowest_index_deterministic()
    test_determinism_repeated_calls()
    test_randomized_viability_invariants()
    test_zero_capacity_yields_zeros()
    test_virtual_updates_registered_for_every_cut_slice()
    test_no_virtual_update_for_uncut_slices()
    test_no_virtual_updates_when_feasible()
    test_updates_match_cut_mask_and_deficits()
    test_fractional_proposals_rejected()
    test_negative_proposals_rejected()
    test_empty_and_2d_proposals_rejected()
    test_negative_capacity_rejected()
    test_non_finite_proposals_rejected()
    print("All coordinator tests passed.")
