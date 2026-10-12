"""Tests for nsoran.sqr.sqr_controller (hierarchical safety-first policy).

Checklist (PLANO_IMPLEMENTACAO_SQR_NORI.md §3-Fase 2 / GUIA §3.6):
gamma=0 vs gamma=1 differ ONLY in the cost term; empty A_safe -> uncertain
mode (softmax / k-NN / fallback); determinism per seed; safety margin
a + round(sigma*M); tau/M adaptation; edges (empty memory, no neighbors).
P_penalty stays active for both gamma variants (AGENTS.md §5 rule 7).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from nsoran.sqr.kernel import MaternKernel
from nsoran.sqr.quantile_regressor import CRITICAL, SUBPREDICTION
from nsoran.sqr.sqr_controller import (
    MODE_CONFIDENT,
    MODE_FALLBACK,
    MODE_KNN,
    MODE_SOFTMAX,
    SqrSliceController,
)

C = 52          # [ASSUMIDO] default cell capacity (plan Fase 0)
LAMBDA = 0.5    # SLA level for A_safe / critical detection


def _kernel(length_scale=0.05):
    """Tight length-scale so action candidates decorrelate quickly."""
    return MaternKernel(dim=3, length_scale=length_scale, nu=1.5)


def _controller(gamma=0.0, seed=None, kernel=None, **kwargs):
    kwargs.setdefault("fallback_action", 0)
    return SqrSliceController(
        state_dim=2,
        n_prbs=C,
        gamma=gamma,
        sla_threshold=LAMBDA,
        kernel=kernel if kernel is not None else _kernel(),
        seed=seed,
        **kwargs,
    )


def _inject_confident_memory(controller, f_low=0.9, f_high=1.0):
    """Two support vectors at z*=[0.5, 0.5]: a=10 -> f_low, a=40 -> f_high.

    With length_scale=0.05 in the action dimension, candidates more than a
    few PRBs away decorrelate (k ~ 0), so f(a=10) ~= f_low, f(a=40) ~= f_high.
    """
    z = np.array([0.5, 0.5])
    X = np.array([
        [0.5, 0.5, 10.0 / C],
        [0.5, 0.5, 40.0 / C],
    ])
    controller.regressor.set_memory(X, np.array([f_low, f_high]))
    return z


# ── gamma alternates ONLY the cost term ─────────────────────────────

def test_scores_identical_f_minus_gamma_a_over_c():
    """scores must equal f_tau - gamma * a/C with the SAME f for both."""
    kern = _kernel()
    z = np.array([0.5, 0.5])
    for gamma in (0.0, 1.0):
        ctl = _controller(gamma=gamma, kernel=kern)
        _inject_confident_memory(ctl)
        info = ctl.score_candidates(z)
        a = np.arange(C + 1, dtype=np.float64)
        expected = info["f"] - gamma * a / C
        assert np.allclose(info["scores"], expected, atol=1e-15)


def test_a_safe_mask_identical_across_gamma():
    """The safe set depends only on f_tau and Lambda — never on gamma."""
    kern = _kernel()
    z = np.array([0.5, 0.5])
    masks = []
    for gamma in (0.0, 1.0):
        ctl = _controller(gamma=gamma, kernel=kern)
        _inject_confident_memory(ctl)
        masks.append(ctl.score_candidates(z)["a_safe"])
    assert np.array_equal(masks[0], masks[1])
    assert masks[0].sum() >= 2  # both support actions are safe


def test_gamma_one_shifts_choice_to_cheaper_action():
    """f(a=10)=0.9 < f(a=40)=1.0 but f(a=10) - a/C > f(a=40) - a/C:
    gamma=0 picks 40 (max quantile), gamma=1 picks 10 (cheaper safe action)."""
    kern = _kernel()
    z = np.array([0.5, 0.5])
    dec0 = _controller(gamma=0.0, kernel=kern)
    _inject_confident_memory(dec0, 0.9, 1.0)
    d0 = dec0.act(z)
    assert d0.mode == MODE_CONFIDENT
    assert d0.raw_action == 40

    ctl1 = _controller(gamma=1.0, kernel=kern)
    _inject_confident_memory(ctl1, 0.9, 1.0)
    d1 = ctl1.act(z)
    assert d1.mode == MODE_CONFIDENT
    assert d1.raw_action == 10


def test_gamma_zero_and_one_agree_when_quantile_prefers_cheap():
    """f(a=10)=1.0 > f(a=40)=0.9: both variants pick 10."""
    kern = _kernel()
    z = np.array([0.5, 0.5])
    for gamma in (0.0, 1.0):
        ctl = _controller(gamma=gamma, kernel=kern)
        _inject_confident_memory(ctl, 1.0, 0.9)
        d = ctl.act(z)
        assert d.raw_action == 10, f"gamma={gamma}"


def test_p_penalty_identical_across_gamma_variants():
    """A predicted-safe action that violated the SLA yields the same
    amplified critical update under gamma=0 and gamma=1."""
    records = []
    for gamma in (0.0, 1.0):
        ctl = _controller(gamma=gamma)
        z = np.array([0.5, 0.5])
        # Make f(z, a=20) >= Lambda, then observe a violation.
        X = np.array([[0.5, 0.5, 20.0 / C]])
        ctl.regressor.set_memory(X, np.array([0.8]))
        rec = ctl.observe(z, action=20, y=0.1)
        records.append(rec)
    assert all(r.kind == CRITICAL for r in records)
    assert records[0].alpha == records[1].alpha
    assert abs(records[0].alpha - 0.1 * (0.05 - 1.0) * 2.0) < 1e-15


# ── Empty A_safe -> uncertain modes ─────────────────────────────────

def test_uncertain_no_neighbors_falls_back():
    """Empty regressor (f=0 < Lambda), no successes, epsilon=0 -> fallback."""
    ctl = _controller(gamma=0.0, epsilon=0.0, fallback_action=17)
    d = ctl.act(np.array([0.1, 0.2]))
    assert d.mode == MODE_FALLBACK
    assert d.raw_action == 17
    assert d.a_safe_size == 0


def test_uncertain_with_neighbors_uses_knn():
    """epsilon=0 + successes -> safety-first k-NN over the success memory."""
    ctl = _controller(gamma=0.0, epsilon=0.0)
    z = np.array([0.05, 0.0])
    ctl.observe(np.array([0.0, 0.0]), action=10, y=0.6)   # success (y >= Lambda)
    ctl.observe(np.array([0.1, 0.0]), action=30, y=1.0)   # success
    d = ctl.act(z)
    assert d.mode == MODE_KNN
    assert d.raw_action == 30  # equidistant -> safety-first picks max y


def test_knn_restricted_to_k_nearest_successes():
    """The best overall success must be ignored when outside the K nearest."""
    ctl = SqrSliceController(
        state_dim=2, n_prbs=C, gamma=0.0, sla_threshold=LAMBDA,
        kernel=_kernel(), epsilon=0.0, knn_k=2, fallback_action=0, seed=1,
    )
    ctl.observe(np.array([0.00, 0.0]), action=10, y=1.0)  # best y, far away
    ctl.observe(np.array([0.90, 0.0]), action=20, y=0.6)  # within K
    ctl.observe(np.array([0.95, 0.0]), action=30, y=0.5)  # nearest
    d = ctl.act(np.array([0.93, 0.0]))
    assert d.mode == MODE_KNN
    assert d.raw_action == 20  # max y among the K=2 nearest (0.90, 0.95)


def test_uncertain_epsilon_one_samples_softmax():
    """epsilon=1 + no successes -> softmax sampling over f_tau."""
    ctl = _controller(gamma=0.0, epsilon=1.0, seed=3)
    d = ctl.act(np.array([0.3, 0.3]))
    assert d.mode == MODE_SOFTMAX
    assert 0 <= d.action <= C


def test_softmax_prefers_high_f_when_temperature_low():
    """With T -> 0 the softmax argmaxes f; the SV at a=40 must dominate."""
    ctl = _controller(gamma=0.0, epsilon=1.0, temperature=1e-6, seed=5)
    z = _inject_confident_memory(ctl, 0.9, 1.0)
    # Force the uncertain path: lower Lambda would make A_safe non-empty,
    # so instead raise Lambda above every f.
    ctl.sla_threshold = 2.0
    d = ctl.act(z)
    assert d.mode == MODE_SOFTMAX
    assert d.raw_action == 40


# ── Safety margin a + round(sigma * M) ──────────────────────────────

def test_margin_adds_round_sigma_m_with_sigma_one():
    """Empty memory -> sigma=1 -> margin adds exactly M to the raw action."""
    ctl = _controller(gamma=0.0, epsilon=0.0, fallback_action=10, margin=4)
    d = ctl.act(np.array([0.0, 0.0]))
    assert d.raw_action == 10
    assert d.sigma == 1.0
    assert d.action == 14
    assert d.margin_applied == 4


def test_margin_clipped_at_capacity():
    ctl = _controller(gamma=0.0, epsilon=0.0, fallback_action=50, margin=4)
    d = ctl.act(np.array([0.0, 0.0]))
    assert d.action == C  # 50 + 4 clipped to 52


def test_margin_zero_by_default_keeps_raw_action():
    ctl = _controller(gamma=0.0, epsilon=0.0, fallback_action=23)
    d = ctl.act(np.array([0.0, 0.0]))
    assert d.margin_m == 0.0
    assert d.action == d.raw_action == 23


def test_margin_zero_at_support_vector():
    """sigma=0 exactly at a support vector -> no margin shift."""
    ctl = _controller(gamma=0.0, epsilon=0.0, margin=5)
    z = np.array([0.5, 0.5])
    _inject_confident_memory(ctl, 0.9, 1.0)
    d = ctl.act(z)
    assert d.sigma == 0.0
    assert d.action == d.raw_action


# ── tau / M adaptation heuristic ────────────────────────────────────

def test_violations_drive_tau_down_and_margin_up():
    ctl = _controller(
        gamma=0.0,
        violation_window=10,
        target_violation_rate=0.05,
    )
    z = np.array([0.2, 0.2])
    for _ in range(10):
        ctl.observe(z, action=20, y=0.1)  # SLA violated every step
    assert ctl.tau == 0.01                 # 0.05 - 4*0.01 floored at tau_min
    assert ctl.margin == 10.0              # +1 per violation, capped at 10


def test_successes_recover_tau_and_margin():
    ctl = _controller(
        gamma=0.0,
        violation_window=10,
        target_violation_rate=0.05,
    )
    z = np.array([0.2, 0.2])
    for _ in range(10):
        ctl.observe(z, action=20, y=0.1)   # all violations
    assert ctl.tau == 0.01 and ctl.margin == 10.0
    for _ in range(20):                    # window refills with successes
        ctl.observe(z, action=20, y=1.0)
    assert ctl.tau == 0.05                 # back to tau_max
    assert ctl.margin == 0.0               # back to margin lower bound


def test_tau_stays_inside_published_range():
    """tau must remain inside [0.01, 0.05] through any adaptation sequence
    [ARTIGO range]; the controller constructor rejects out-of-range tau."""
    ctl = _controller(gamma=0.0, violation_window=5, target_violation_rate=0.9)
    z = np.array([0.1, 0.1])
    for _ in range(60):
        ctl.observe(z, action=5, y=0.0)
        assert 0.01 - 1e-12 <= ctl.tau <= 0.05 + 1e-12
        assert 0.0 <= ctl.margin <= 10.0
    assert ctl.tau == 0.01  # pinned at the floor


def test_out_of_range_tau_rejected():
    for bad in (0.001, 0.06):
        try:
            _controller(tau=bad)
        except ValueError:
            continue
        raise AssertionError(f"tau={bad} must be rejected")


# ── Determinism per seed ────────────────────────────────────────────

def test_same_seed_same_decision_sequence():
    """Same seed + same observation stream -> identical (mode, action) log."""
    z_stream = np.random.default_rng(99).normal(size=(40, 2)) * 0.4 + 0.5

    def run(seed):
        ctl = _controller(gamma=1.0, seed=seed, epsilon=0.5)
        # Seed a few successes so both k-NN and softmax are reachable.
        ctl.observe(np.array([0.5, 0.5]), action=12, y=0.9)
        ctl.observe(np.array([0.7, 0.3]), action=30, y=0.8)
        return [(ctl.act(z).mode, ctl.act(z).action) for z in z_stream]

    # Two independent controllers with the same seed: identical sequences.
    # (Each z is consumed twice below, so build the log in one pass instead.)
    def run_once(seed):
        ctl = _controller(gamma=1.0, seed=seed, epsilon=0.5)
        ctl.observe(np.array([0.5, 0.5]), action=12, y=0.9)
        ctl.observe(np.array([0.7, 0.3]), action=30, y=0.8)
        log = []
        for z in z_stream:
            d = ctl.act(z)
            log.append((d.mode, d.action))
        return log

    assert run_once(42) == run_once(42)


def test_different_seeds_can_diverge():
    """With epsilon=0.5 over 60 uncertain decisions, different seeds must
    produce at least one different (mode, action) pair."""
    z_stream = np.random.default_rng(7).normal(size=(60, 2)) * 0.4 + 0.5

    def run_once(seed):
        ctl = _controller(gamma=1.0, seed=seed, epsilon=0.5)
        ctl.observe(np.array([0.5, 0.5]), action=12, y=0.9)
        log = []
        for z in z_stream:
            d = ctl.act(z)
            log.append((d.mode, d.action))
        return log

    assert run_once(7) != run_once(42)


def test_actions_always_within_bounds():
    rng = np.random.default_rng(21)
    ctl = _controller(gamma=1.0, seed=8, epsilon=1.0, margin=7, fallback_action=C)
    for _ in range(50):
        d = ctl.act(rng.uniform(0, 1, size=2))
        assert 0 <= d.action <= C
        assert 0 <= d.raw_action <= C


# ── Observe bookkeeping ─────────────────────────────────────────────

def test_observe_records_successes_and_violations():
    ctl = _controller(gamma=0.0, epsilon=0.0)
    ctl.observe(np.array([0.1, 0.1]), action=15, y=0.9)   # success
    ctl.observe(np.array([0.2, 0.2]), action=10, y=0.2)   # violation
    assert ctl.n_successes == 1
    assert ctl.violation_rate() == 0.5
    zs, acts, ys = ctl.success_memory()
    assert acts[0] == 15 and ys[0] == 0.9


def test_observe_rejects_out_of_range_action():
    ctl = _controller()
    for bad in (-1, C + 1):
        try:
            ctl.observe(np.array([0.1, 0.1]), action=bad, y=1.0)
        except ValueError:
            continue
        raise AssertionError(f"action={bad} must be rejected")


def test_subprediction_update_from_observed_improvement():
    """An outcome above the prediction is a positive (subprediction) update."""
    ctl = _controller(gamma=0.0)
    rec = ctl.observe(np.array([0.3, 0.3]), action=25, y=1.0)  # f was 0
    assert rec.kind == SUBPREDICTION
    assert rec.alpha > 0.0


# ── Run manually ────────────────────────────────────────────────────

if __name__ == "__main__":
    test_scores_identical_f_minus_gamma_a_over_c()
    test_a_safe_mask_identical_across_gamma()
    test_gamma_one_shifts_choice_to_cheaper_action()
    test_gamma_zero_and_one_agree_when_quantile_prefers_cheap()
    test_p_penalty_identical_across_gamma_variants()
    test_uncertain_no_neighbors_falls_back()
    test_uncertain_with_neighbors_uses_knn()
    test_knn_restricted_to_k_nearest_successes()
    test_uncertain_epsilon_one_samples_softmax()
    test_softmax_prefers_high_f_when_temperature_low()
    test_margin_adds_round_sigma_m_with_sigma_one()
    test_margin_clipped_at_capacity()
    test_margin_zero_by_default_keeps_raw_action()
    test_margin_zero_at_support_vector()
    test_violations_drive_tau_down_and_margin_up()
    test_successes_recover_tau_and_margin()
    test_tau_stays_inside_published_range()
    test_out_of_range_tau_rejected()
    test_same_seed_same_decision_sequence()
    test_different_seeds_can_diverge()
    test_actions_always_within_bounds()
    test_observe_records_successes_and_violations()
    test_observe_rejects_out_of_range_action()
    test_subprediction_update_from_observed_improvement()
    print("All controller tests passed.")
