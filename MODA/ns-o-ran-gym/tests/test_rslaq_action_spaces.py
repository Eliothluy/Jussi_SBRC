"""Tests for rslaq_action_spaces module."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from environments.rslaq_action_spaces import (
    build_discrete_action_table,
    continuous_action_to_prb,
    discrete_action_to_prb,
    DEFAULT_WEIGHTS,
)


def test_discrete_table_size():
    table = build_discrete_action_table(step=0.1, include_scheduler=False)
    assert len(table) == 66, f"Expected 66 actions, got {len(table)}"


def test_discrete_table_sum():
    table = build_discrete_action_table(step=0.1, include_scheduler=False)
    for entry in table:
        p0, p1, p2 = entry
        assert abs(p0 + p1 + p2 - 1.0) < 1e-6, f"Action {entry} does not sum to 1.0"


def test_continuous_sums_to_100():
    for _ in range(10):
        raw = np.random.uniform(-1, 1, size=3)
        prb = continuous_action_to_prb(raw)
        assert abs(prb.sum() - 100.0) < 0.1, f"Sum {prb.sum()} != 100.0"
        assert prb.shape == (3,)


def test_discrete_prb_sums_to_100():
    table = build_discrete_action_table(step=0.1, include_scheduler=False)
    for idx in range(len(table)):
        prb, sch = discrete_action_to_prb(idx, table)
        assert abs(prb.sum() - 100.0) < 0.1, f"Action {idx} sum {prb.sum()} != 100.0"
        assert sch == -1


def test_discrete_scheduler_table():
    table = build_discrete_action_table(step=0.1, include_scheduler=True)
    assert len(table) == 198, f"Expected 198 actions, got {len(table)}"
    for entry in table:
        assert len(entry) == 4
        p0, p1, p2, sch = entry
        assert sch in {0, 1, 2}
        assert abs(p0 + p1 + p2 - 1.0) < 1e-6


def test_continuous_apply_p_sta_false():
    """Test that apply_p_sta=False returns raw softmax without static weights."""
    raw = np.array([0.0, 0.0, 0.0])  # Uniform input
    prb = continuous_action_to_prb(raw, apply_p_sta=False)
    # With apply_p_sta=False, should be uniform distribution (no static weights bias)
    assert abs(prb[0] - prb[1]) < 1e-6
    assert abs(prb[1] - prb[2]) < 1e-6
    assert abs(prb.sum() - 100.0) < 0.1


def test_continuous_apply_p_sta_true():
    """Test that apply_p_sta=True applies P_STA decomposition."""
    raw = np.array([0.0, 0.0, 0.0])  # Uniform input
    prb = continuous_action_to_prb(raw, apply_p_sta=True)
    # With apply_p_sta=True, should be biased toward static weights
    # Static weights are [0.3333, 0.4000, 0.2667]
    # With uniform softmax [0.333, 0.333, 0.333], P_STA gives:
    # 0.5*weights + 0.5*softmax = [0.333, 0.367, 0.300]
    assert prb[0] > 33.0  # eMBB
    assert prb[1] > 36.0  # URLLC (highest weight, but blended with uniform)
    assert prb[2] > 29.0  # MTC (lowest weight)
    assert abs(prb.sum() - 100.0) < 0.1


def test_discrete_apply_p_sta_false():
    """Test that apply_p_sta=False returns raw action table values."""
    table = build_discrete_action_table(step=0.5, include_scheduler=False)
    # Action at index 0 is (0.0, 0.0, 1.0) for MTC
    prb, sch = discrete_action_to_prb(0, table, apply_p_sta=False)
    assert sch == -1
    # With apply_p_sta=False, should match the raw action table (MTC gets all)
    assert prb[0] < 1.0  # eMBB gets nothing
    assert prb[1] < 1.0  # URLLC gets nothing
    assert prb[2] > 98.0  # MTC gets almost all
    assert abs(prb.sum() - 100.0) < 0.1


def test_discrete_apply_p_sta_true():
    """Test that apply_p_sta=True applies P_STA decomposition."""
    table = build_discrete_action_table(step=0.5, include_scheduler=False)
    # Action at index 0 is (0.0, 0.0, 1.0) for MTC
    prb, sch = discrete_action_to_prb(0, table, apply_p_sta=True)
    assert sch == -1
    # With apply_p_sta=True, even MTC-heavy action gets static weight bias
    assert prb[0] > 15.0  # eMBB gets some due to static weight
    assert prb[1] > 18.0  # URLLC gets some due to static weight
    assert prb[2] > 50.0  # MTC still gets most due to raw action
    assert abs(prb.sum() - 100.0) < 0.1


if __name__ == "__main__":
    test_discrete_table_size()
    test_discrete_table_sum()
    test_continuous_sums_to_100()
    test_discrete_prb_sums_to_100()
    test_discrete_scheduler_table()
    test_continuous_apply_p_sta_false()
    test_continuous_apply_p_sta_true()
    test_discrete_apply_p_sta_false()
    test_discrete_apply_p_sta_true()
    print("All action_spaces tests passed.")
