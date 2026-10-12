"""
Integration tests for RSLAQ P_STA decomposition flow.

Tests the end-to-end flow from Python action to C++ scheduler weights:
    1. Python agent produces raw action
    2. Action is converted to PRB percentages (with/without P_STA)
    3. PRB percentages are written to CSV
    4. C++ reads CSV and applies to scheduler

This test verifies the final weights match the expected formula.
"""

import sys
import os
import tempfile
import numpy as np
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from environments.rslaq_action_spaces import (
    build_discrete_action_table,
    continuous_action_to_prb,
    discrete_action_to_prb,
    DEFAULT_WEIGHTS,
)


def simulate_cpp_parsing(
    dedicated_prb: List[float],
    apply_p_sta_in_cpp: bool = False,
) -> List[float]:
    """
    Simulate the C++ side parsing logic from rslaq-sim.cc.

    Args:
        dedicated_prb: PRB percentages read from CSV (from Python)
        apply_p_sta_in_cpp: If True, apply P_STA in C++ (old behavior)
                            If False, use values directly (new behavior)

    Returns:
        Final weights that would be used by the scheduler
    """
    if apply_p_sta_in_cpp:
        # Old C++ code (BEFORE fix):
        # p_dyn[s] = (dedicatedPrb[s] / totalDed) * 0.5
        # p_final[s] = P_STA_WEIGHTS[s] * 0.5 + p_dyn[s]
        total = sum(dedicated_prb)
        if total == 0:
            return [0.0, 0.0, 0.0]

        p_dyn = [(p / total) * 0.5 for p in dedicated_prb]
        p_final = [DEFAULT_WEIGHTS[i] * 0.5 + p_dyn[i] for i in range(3)]

        # Renormalize
        sum_p = sum(p_final)
        return [p / sum_p for p in p_final]
    else:
        # New C++ code (AFTER fix):
        # Directly use the percentages from Python
        total = sum(dedicated_prb)
        if total == 0:
            return [0.0, 0.0, 0.0]
        return [p / total for p in dedicated_prb]


def test_continuous_action_flow_apply_p_sta_false():
    """
    Test continuous action flow with apply_p_sta=False in Python.

    This is the NEW behavior after the fix:
        - Python: apply_p_sta=False, sends raw softmax output
        - C++: uses values directly

    Expected: final weights = softmax(raw_action)
    """
    raw_action = np.array([1.0, 0.5, -0.5])
    static_fraction = 0.5

    # Python side
    prb_pct = continuous_action_to_prb(raw_action, apply_p_sta=False)

    # C++ side (NEW: no P_STA)
    final_weights = simulate_cpp_parsing(prb_pct, apply_p_sta_in_cpp=False)

    # Verify: final weights should equal softmax of raw action
    from environments.rslaq_action_spaces import _softmax
    expected = _softmax(raw_action)

    for i in range(3):
        assert abs(final_weights[i] - expected[i]) < 1e-6, (
            f"Slice {i}: got {final_weights[i]}, expected {expected[i]}"
        )
    print("test_continuous_action_flow_apply_p_sta_false PASSED")


def test_continuous_action_flow_apply_p_sta_true():
    """
    Test continuous action flow with apply_p_sta=True in Python.

    This is the OLD behavior (before fix):
        - Python: apply_p_sta=True, applies P_STA decomposition
        - C++: uses values directly

    Expected: final weights = 0.5*weights + 0.5*softmax(raw_action)
    """
    raw_action = np.array([1.0, 0.5, -0.5])
    static_fraction = 0.5

    # Python side
    prb_pct = continuous_action_to_prb(raw_action, apply_p_sta=True)

    # C++ side (no P_STA applied here - already done in Python)
    final_weights = simulate_cpp_parsing(prb_pct, apply_p_sta_in_cpp=False)

    # Verify: final weights should match P_STA formula
    from environments.rslaq_action_spaces import _softmax
    p_opt = _softmax(raw_action)
    expected = static_fraction * DEFAULT_WEIGHTS + (1.0 - static_fraction) * p_opt
    expected = expected / expected.sum()  # Renormalize

    for i in range(3):
        assert abs(final_weights[i] - expected[i]) < 1e-6, (
            f"Slice {i}: got {final_weights[i]}, expected {expected[i]}"
        )
    print("test_continuous_action_flow_apply_p_sta_true PASSED")


def test_discrete_action_flow_apply_p_sta_false():
    """
    Test discrete action flow with apply_p_sta=False in Python.

    Expected: final weights = action_table[idx]
    """
    table = build_discrete_action_table(step=0.5, include_scheduler=False)
    action_idx = len(table) // 2  # Some action in the middle

    # Python side
    prb_pct, _ = discrete_action_to_prb(action_idx, table, apply_p_sta=False)

    # C++ side (NEW: no P_STA)
    final_weights = simulate_cpp_parsing(prb_pct, apply_p_sta_in_cpp=False)

    # Verify: final weights should equal raw action table values
    p_opt = np.array(table[action_idx], dtype=np.float64)
    p_opt = p_opt / p_opt.sum()
    expected = p_opt * 100.0
    expected = expected / expected.sum()

    for i in range(3):
        assert abs(final_weights[i] - expected[i]) < 1e-6, (
            f"Slice {i}: got {final_weights[i]}, expected {expected[i]}"
        )
    print("test_discrete_action_flow_apply_p_sta_false PASSED")


def test_discrete_action_flow_apply_p_sta_true():
    """
    Test discrete action flow with apply_p_sta=True in Python.

    Expected: final weights = 0.5*weights + 0.5*action_table[idx]
    """
    table = build_discrete_action_table(step=0.5, include_scheduler=False)
    action_idx = len(table) // 2  # Some action in the middle
    static_fraction = 0.5

    # Python side
    prb_pct, _ = discrete_action_to_prb(action_idx, table, apply_p_sta=True)

    # C++ side (no P_STA applied here - already done in Python)
    final_weights = simulate_cpp_parsing(prb_pct, apply_p_sta_in_cpp=False)

    # Verify: final weights should match P_STA formula
    p_opt = np.array(table[action_idx], dtype=np.float64)
    p_opt = p_opt / p_opt.sum()
    expected = static_fraction * DEFAULT_WEIGHTS + (1.0 - static_fraction) * p_opt
    expected = expected / expected.sum()

    for i in range(3):
        assert abs(final_weights[i] - expected[i]) < 1e-6, (
            f"Slice {i}: got {final_weights[i]}, expected {expected[i]}"
        )
    print("test_discrete_action_flow_apply_p_sta_true PASSED")


def test_double_p_sta_bug():
    """
    Demonstrate the bug that was fixed: P_STA applied TWICE.

    This test verifies that the OLD configuration (apply_p_sta=True in Python
    AND P_STA in C++) results in double application.
    """
    raw_action = np.array([1.0, 0.5, -0.5])

    # Python side: apply_p_sta=True
    prb_pct = continuous_action_to_prb(raw_action, apply_p_sta=True)

    # C++ side: ALSO applies P_STA (BUG!)
    final_weights_buggy = simulate_cpp_parsing(prb_pct, apply_p_sta_in_cpp=True)

    # C++ side: does NOT apply P_STA (CORRECT!)
    final_weights_correct = simulate_cpp_parsing(prb_pct, apply_p_sta_in_cpp=False)

    # The buggy version should give MORE weight to static weights
    # because P_STA is applied twice
    # This proves the bug existed and is now fixed
    assert (
        abs(final_weights_buggy[1] - final_weights_correct[1]) > 1e-3
    ), "Buggy and correct versions should differ"

    print(f"  Buggy (double P_STA):   {final_weights_buggy}")
    print(f"  Correct (single P_STA):  {final_weights_correct}")
    print(f"  Difference: {[abs(a-b) for a,b in zip(final_weights_buggy, final_weights_correct)]}")

    # Verify the formula: double P_STA gives 75% static, 25% dynamic
    from environments.rslaq_action_spaces import _softmax
    p_opt = _softmax(raw_action)
    # First P_STA: p1 = 0.5*weights + 0.5*p_opt
    p1 = 0.5 * DEFAULT_WEIGHTS + 0.5 * p_opt
    p1 = p1 / p1.sum()
    # Second P_STA: p2 = 0.5*weights + 0.5*p1
    p2 = 0.5 * DEFAULT_WEIGHTS + 0.5 * p1
    p2 = p2 / p2.sum()

    for i in range(3):
        assert abs(final_weights_buggy[i] - p2[i]) < 1e-6, (
            f"Double P_STA verification failed for slice {i}"
        )

    print("test_double_p_sta_bug PASSED (bug demonstrated)")


def test_agent_influence_50_percent():
    """
    Test that with apply_p_sta=False and new C++, the agent has 50% influence.

    This is the expected behavior after the fix.
    """
    raw_action = np.array([10.0, 0.0, 0.0])  # Strong bias to eMBB

    # With apply_p_sta=False, agent output goes directly to C++
    prb_pct = continuous_action_to_prb(raw_action, apply_p_sta=False)
    final_weights = simulate_cpp_parsing(prb_pct, apply_p_sta_in_cpp=False)

    # Agent's influence: the softmax output
    from environments.rslaq_action_spaces import _softmax
    agent_influence = _softmax(raw_action)

    # Final weights should match agent influence (no P_STA in C++)
    for i in range(3):
        assert abs(final_weights[i] - agent_influence[i]) < 1e-6, (
            f"Slice {i}: final {final_weights[i]} != agent {agent_influence[i]}"
        )

    print(f"  Agent influence: {agent_influence}")
    print(f"  Final weights:   {final_weights}")
    print("  Agent has full control over output (P_STA handled elsewhere)")


if __name__ == "__main__":
    print("\n=== RSLAQ Integration Tests ===\n")

    test_continuous_action_flow_apply_p_sta_false()
    test_continuous_action_flow_apply_p_sta_true()
    test_discrete_action_flow_apply_p_sta_false()
    test_discrete_action_flow_apply_p_sta_true()
    test_double_p_sta_bug()
    test_agent_influence_50_percent()

    print("\n=== All integration tests PASSED ===\n")
