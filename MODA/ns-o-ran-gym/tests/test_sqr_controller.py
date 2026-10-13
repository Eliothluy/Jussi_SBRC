"""Tests for nsoran.sqr.sqr_controller (hierarchical safety-first policy).

Checklist (PLANO_IMPLEMENTACAO_SQR_NORI.md §3-Fase 2 / GUIA §3.6):
gamma=0 vs gamma=1 differ ONLY in the cost term; empty A_safe -> uncertain
mode (softmax / R21 k-NN with argmax f_tau + edges / fallback = argmax
f_tau); determinism per seed; safety margin a + round(sigma*M) with the
frozen R11 defaults (M0=10, bounds [2,50]); block-window tau/M adaptation
(W=100, delta_tau=0.005, doubling/halving, recovery only after 2 clean
windows); R12 virtual support with exact negative weight; checkpoint
round-trip including the RNG state.  P_penalty stays active for both gamma
variants (AGENTS.md §5 rule 7).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from nsoran.sqr.kernel import MaternKernel
from nsoran.sqr.quantile_regressor import CRITICAL, SUBPREDICTION
from nsoran.sqr.sqr_controller import (
    MARGIN_BOUNDS,
    MARGIN_INIT,
    MODE_CONFIDENT,
    MODE_FALLBACK,
    MODE_KNN,
    MODE_KNN_NEAREST,
    MODE_SOFTMAX,
    RECOVERY_CLEAN_WINDOWS,
    TARGET_VIOLATION_RATE,
    TAU_INIT,
    TAU_STEP,
    VIOLATION_WINDOW,
    SqrSliceController,
)

C = 52          # [ASSUMIDO] default cell capacity (plan Fase 0)
LAMBDA = 0.5    # SLA level for A_safe / critical detection


def _kernel(length_scale=0.05):
    """Tight length-scale so action candidates decorrelate quickly."""
    return MaternKernel(dim=3, length_scale=length_scale, nu=1.5)


def _controller(gamma=0.0, seed=None, kernel=None, **kwargs):
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


# ── Frozen R11 constants ────────────────────────────────────────────

def test_frozen_adaptation_defaults():
    """Register R11 (congelado): W=100, delta_tau=0.005, tau0=0.05, M0=10,
    M bounds [2, 50], delta=1%, recovery after 2 consecutive clean windows."""
    assert VIOLATION_WINDOW == 100
    assert TAU_STEP == 0.005
    assert TAU_INIT == 0.05
    assert MARGIN_INIT == 10.0
    assert MARGIN_BOUNDS == (2.0, 50.0)
    assert TARGET_VIOLATION_RATE == 0.01
    assert RECOVERY_CLEAN_WINDOWS == 2

    ctl = _controller()
    assert ctl.violation_window == 100
    assert ctl.tau_step == 0.005
    assert ctl.tau == 0.05
    assert ctl.margin == 10.0
    assert ctl.margin_bounds == (2.0, 50.0)
    assert ctl.target_violation_rate == 0.01
    assert ctl.recovery_windows == 2
    assert ctl.margin_bounds[0] <= ctl.margin <= ctl.margin_bounds[1]


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


# ── Empty A_safe -> uncertain modes (register R21) ──────────────────

def test_uncertain_no_neighbors_uses_argmax_ftau():
    """Edge 2 (frozen): with NO memorized experience the action is the
    candidate with the highest f_tau — with an empty regressor all f are 0,
    so the tie resolves to the smallest action (0), deterministically."""
    ctl = _controller(gamma=0.0, epsilon=0.0)
    d = ctl.act(np.array([0.1, 0.2]))
    assert d.mode == MODE_FALLBACK
    assert d.raw_action == 0
    assert d.a_safe_size == 0


def test_uncertain_no_neighbors_but_model_prefers_best_ftau():
    """Edge 2 is NOT a fixed fallback constant: with a populated regressor
    but zero experience, the highest-f_tau candidate wins."""
    ctl = _controller(gamma=0.0, epsilon=0.0)
    _inject_confident_memory(ctl, 0.1, 2.5)   # f(a=40) = 2.5
    ctl.sla_threshold = 5.0                   # nothing is safe
    d = ctl.act(np.array([0.5, 0.5]))
    assert d.mode == MODE_FALLBACK
    assert d.raw_action == 40
    assert ctl.n_experiences == 0


def test_knn_picks_argmax_ftau_not_max_y():
    """R21 branch 1: among the K nearest SUCCESSES, choose argmax f_tau
    (z, a_j) — NOT the highest observed y.  Here y(a=10)=0.9 > y(a=40)=0.8
    but f(a=10)=0.6 < f(a=40)=0.9, so the chosen action is 40."""
    ctl = _controller(gamma=0.0, epsilon=0.0)
    z = np.array([0.5, 0.5])
    ctl.observe(z, action=10, y=0.9)   # success
    ctl.observe(z, action=40, y=0.8)   # success
    # Overwrite the model so f(z, a) is exactly known at the two actions.
    X = np.array([[0.5, 0.5, 10.0 / C], [0.5, 0.5, 40.0 / C]])
    ctl.regressor.set_memory(X, np.array([0.6, 0.9]))
    ctl.sla_threshold = 2.0            # force the uncertain branch
    d = ctl.act(z)
    assert d.mode == MODE_KNN
    assert d.raw_action == 40


def test_knn_restricted_to_k_nearest_successes():
    """The best-f_tau action must be ignored when its experience lies
    outside the K nearest successes."""
    ctl = SqrSliceController(
        state_dim=2, n_prbs=C, gamma=0.0, sla_threshold=LAMBDA,
        kernel=_kernel(), epsilon=0.0, knn_k=2, seed=1,
    )
    ctl.observe(np.array([0.00, 0.0]), action=10, y=1.0)  # far away (3rd)
    ctl.observe(np.array([0.90, 0.0]), action=20, y=0.6)
    ctl.observe(np.array([0.95, 0.0]), action=30, y=0.5)
    # f at the query: action 10 would clearly win if the far experience
    # leaked into the neighbor set (f=2.0 > 0.6 > 0.5).
    q = np.array([0.93, 0.0])
    X = np.array([
        [0.93, 0.0, 10.0 / C],
        [0.93, 0.0, 20.0 / C],
        [0.93, 0.0, 30.0 / C],
    ])
    ctl.regressor.set_memory(X, np.array([2.0, 0.6, 0.5]))
    ctl.sla_threshold = 3.0    # above every f (cross-terms push f(a=10) > 2)
    d = ctl.act(q)
    assert d.mode == MODE_KNN
    assert d.raw_action == 20  # argmax f among the K=2 nearest successes {20, 30}


def test_knn_edge_no_success_uses_nearest_experience():
    """Edge 1 (frozen): experiences memorized but NONE succeeded -> reuse
    the action of the single nearest experience (success or failure)."""
    ctl = _controller(gamma=0.0, epsilon=0.0)
    ctl.observe(np.array([0.0, 0.0]), action=10, y=0.1)   # violation
    ctl.observe(np.array([0.3, 0.0]), action=30, y=0.2)   # violation
    ctl.sla_threshold = 2.0
    d = ctl.act(np.array([0.05, 0.0]))
    assert d.mode == MODE_KNN_NEAREST
    assert d.raw_action == 10   # nearest (dist 0.05), NOT the argmax-f action
    assert ctl.n_successes == 0
    assert ctl.n_experiences == 2


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
    # Force the uncertain path: raise Lambda above every f.
    ctl.sla_threshold = 2.0
    d = ctl.act(z)
    assert d.mode == MODE_SOFTMAX
    assert d.raw_action == 40


# ── Safety margin a + round(sigma * M) ──────────────────────────────

def test_margin_adds_round_sigma_m_with_sigma_one():
    """Empty memory -> sigma=1 -> margin adds exactly M to the raw action."""
    ctl = _controller(gamma=0.0, epsilon=0.0, margin=4)
    d = ctl.act(np.array([0.0, 0.0]))
    assert d.raw_action == 0    # edge 2 with empty memory: argmax f -> 0
    assert d.sigma == 1.0
    assert d.action == 4
    assert d.margin_applied == 4


def test_margin_clipped_at_capacity():
    """raw=50 comes from edge 1 (nearest memorized experience, a violation);
    the regressor memory is replaced afterwards so sigma(raw) ~ 1 and the
    margin pushes past C, where it must clip."""
    ctl = _controller(gamma=0.0, epsilon=0.0, margin=4)
    z = np.array([0.5, 0.5])
    ctl.observe(z, action=50, y=0.1)   # violation memorized (edge-1 fodder)
    # Replace the regressor memory: f ~ 0 everywhere (nothing safe) and
    # sigma(z, a=50) ~ 1.
    ctl.regressor.set_memory(np.array([[9.0, 9.0, 0.0]]), np.array([1.0]))
    d = ctl.act(z)
    assert d.mode == MODE_KNN_NEAREST
    assert d.raw_action == 50
    assert d.sigma > 0.9
    assert d.action == C  # 50 + 4 clipped to 52


def test_margin_default_is_frozen_m0():
    """The frozen initial margin is M0 = 10 PRBs (register R11), so a cold
    uncertain decision adds +10 before any adaptation."""
    ctl = _controller(gamma=0.0, epsilon=0.0)
    d = ctl.act(np.array([0.0, 0.0]))
    assert d.margin_m == 10.0
    assert d.raw_action == 0
    assert d.action == 10


def test_margin_zero_at_support_vector():
    """sigma=0 exactly at a support vector -> no margin shift."""
    ctl = _controller(gamma=0.0, epsilon=0.0, margin=5)
    z = np.array([0.5, 0.5])
    _inject_confident_memory(ctl, 0.9, 1.0)
    d = ctl.act(z)
    assert d.sigma == 0.0
    assert d.action == d.raw_action


# ── tau / M adaptation (register R11: block windows) ────────────────

def test_no_adaptation_before_first_full_window():
    """The heuristic evaluates once per BLOCK window of W periods — a
    partial window never adapts."""
    ctl = _controller(gamma=0.0, violation_window=10)
    z = np.array([0.2, 0.2])
    for _ in range(9):                     # one short of the window
        ctl.observe(z, action=20, y=0.1)   # violations
    assert ctl.tau == 0.05
    assert ctl.margin == 10.0


def test_violations_drive_tau_down_and_margin_doubling():
    """v_W > delta: tau -= 0.005 (floor 0.01), M <- min(50, 2M).
    W=10, all windows fully violated:
      win1: tau 0.045, M 20; win2: tau 0.040, M 40; win3: tau 0.035, M 50."""
    ctl = _controller(gamma=0.0, violation_window=10)
    z = np.array([0.2, 0.2])
    for _ in range(30):
        ctl.observe(z, action=20, y=0.1)
    assert ctl.tau == pytest.approx(0.05 - 3 * 0.005)
    assert ctl.margin == 50.0


def test_recovery_requires_two_consecutive_clean_windows():
    """Recovery (tau += 0.005 capped at 0.05; M <- max(2, M/2)) fires ONLY
    after v_W = 0 in two consecutive windows."""
    ctl = _controller(gamma=0.0, violation_window=10)
    z = np.array([0.2, 0.2])
    for _ in range(10):
        ctl.observe(z, action=20, y=0.1)       # window 1: v_W = 1 > delta
    assert ctl.tau == pytest.approx(0.045) and ctl.margin == 20.0
    for _ in range(10):
        ctl.observe(z, action=20, y=1.0)       # window 2: v_W = 0 (streak 1)
    assert ctl.tau == pytest.approx(0.045) and ctl.margin == 20.0   # NO early recovery
    for _ in range(10):
        ctl.observe(z, action=20, y=1.0)       # window 3: v_W = 0 (streak 2)
    assert ctl.tau == pytest.approx(0.05)      # 0.045 + 0.005
    assert ctl.margin == 10.0                  # max(2, 20/2)
    for _ in range(10):
        ctl.observe(z, action=20, y=1.0)       # window 4: clean again
    assert ctl.tau == pytest.approx(0.05) and ctl.margin == 10.0  # streak reset


def test_dirty_window_breaks_the_clean_streak():
    ctl = _controller(gamma=0.0, violation_window=10)
    z = np.array([0.2, 0.2])
    for _ in range(10):
        ctl.observe(z, action=20, y=0.1)       # window 1: adapt down
    for _ in range(10):
        ctl.observe(z, action=20, y=1.0)       # window 2: clean, streak 1
    for _ in range(9):
        ctl.observe(z, action=20, y=1.0)
    ctl.observe(z, action=20, y=0.1)           # window 3: 1 violation -> dirty
    assert ctl.tau == pytest.approx(0.04)      # adapted DOWN again (0.045-0.005)
    assert ctl.margin == 40.0                  # 2 * 20


def test_tau_stays_inside_published_range_and_margin_within_bounds():
    """tau must remain inside [0.01, 0.05] and M inside [2, 50] through any
    adaptation sequence; tau pins at the published floor."""
    ctl = _controller(gamma=0.0, violation_window=5, target_violation_rate=0.9)
    z = np.array([0.1, 0.1])
    for _ in range(60):     # 12 windows, all violated
        ctl.observe(z, action=5, y=0.0)
        assert 0.01 - 1e-12 <= ctl.tau <= 0.05 + 1e-12
        assert ctl.margin_bounds[0] <= ctl.margin <= ctl.margin_bounds[1]
    assert ctl.tau == pytest.approx(0.01)  # pinned at the floor
    assert ctl.margin == 50.0


def test_recovery_margin_floors_at_two():
    """M <- max(2, M/2): the halving stops at the frozen floor 2."""
    ctl = _controller(gamma=0.0, violation_window=5)
    z = np.array([0.3, 0.3])
    for _ in range(5):
        ctl.observe(z, action=5, y=0.1)    # win1: M 10 -> 20
    assert ctl.margin == 20.0
    for _ in range(10):
        ctl.observe(z, action=5, y=1.0)    # win2+win3 clean -> one recovery
    assert ctl.margin == 10.0              # max(2, 20/2)
    for _ in range(10):
        ctl.observe(z, action=5, y=1.0)    # win4+win5 -> another recovery
    assert ctl.margin == 5.0               # max(2, 10/2)


def test_out_of_range_tau_rejected():
    for bad in (0.001, 0.06):
        try:
            _controller(tau=bad)
        except ValueError:
            continue
        raise AssertionError(f"tau={bad} must be rejected")


def test_out_of_range_margin_rejected():
    for bad in (1.0, 51.0):    # outside the frozen [2, 50]
        try:
            _controller(margin=bad)
        except ValueError:
            continue
        raise AssertionError(f"margin={bad} must be rejected")


# ── R12: virtual support at the pre-projection action ───────────────

def test_insert_virtual_support_exact_negative_weight():
    ctl = _controller(gamma=0.0, epsilon=0.0)
    z = np.array([0.4, 0.6])
    rec = ctl.insert_virtual_support(z, action=30, alpha=0.05)
    assert rec.alpha == -0.05                # the sign is applied inside
    x = np.concatenate([z, [30.0 / C]])
    idx = rec.inserted_index
    assert reg_alpha_at(ctl, idx) == -0.05
    # The inserted feature vector is exactly [z, a/C].
    np.testing.assert_array_equal(ctl.regressor.support_points[idx], x)
    # predict at the same point equals the coefficient (k(x,x)=1).
    assert abs(ctl.regressor.predict(x[None, :])[0] - (-0.05)) < 1e-15


def test_insert_virtual_support_rejects_out_of_range_action():
    ctl = _controller()
    for bad in (-1, C + 1):
        try:
            ctl.insert_virtual_support(np.array([0.1, 0.1]), bad, 0.05)
        except ValueError:
            continue
        raise AssertionError(f"action={bad} must be rejected")


def reg_alpha_at(controller, index):
    return float(controller.regressor.alphas[index])


# ── Determinism per seed ────────────────────────────────────────────

def test_same_seed_same_decision_sequence():
    """Same seed + same observation stream -> identical (mode, action) log."""
    z_stream = np.random.default_rng(99).normal(size=(40, 2)) * 0.4 + 0.5

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
    ctl = _controller(gamma=1.0, seed=8, epsilon=1.0, margin=7)
    for _ in range(50):
        d = ctl.act(rng.uniform(0, 1, size=2))
        assert 0 <= d.action <= C
        assert 0 <= d.raw_action <= C


# ── Observe bookkeeping (R21: failures are memorized too) ───────────

def test_observe_records_successes_failures_and_violations():
    ctl = _controller(gamma=0.0, epsilon=0.0)
    ctl.observe(np.array([0.1, 0.1]), action=15, y=0.9)   # success
    ctl.observe(np.array([0.2, 0.2]), action=10, y=0.2)   # violation
    assert ctl.n_successes == 1
    assert ctl.n_experiences == 2                          # BOTH memorized (R21)
    assert ctl.violation_rate() == 0.5
    zs, acts, ys, ok = ctl.experience_memory()
    assert acts[0] == 15 and ys[0] == 0.9 and bool(ok[0])
    assert acts[1] == 10 and ys[1] == 0.2 and not bool(ok[1])


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


# ── Checkpoint round-trip (support, alphas, tau, M, window, RNG) ────

def test_checkpoint_roundtrip_reproduces_decisions():
    """A controller restored from state_dict must produce EXACTLY the same
    decision sequence as the original on a shared future stream — the RNG
    state (epsilon coin + softmax) is part of the checkpoint."""
    z_stream = np.random.default_rng(31).normal(size=(40, 2)) * 0.4 + 0.5
    y_stream = np.random.default_rng(32).uniform(0.0, 1.2, size=40)

    original = _controller(gamma=1.0, seed=17, epsilon=0.4)
    for i in range(20):
        original.observe(z_stream[i], action=int(5 + i % 30), y=float(y_stream[i]))
    # Advance the decision RNG before the snapshot.
    for i in range(20, 26):
        original.act(z_stream[i])

    state = original.state_dict()
    required = {
        "regressor", "margin_m", "violations", "window_count",
        "clean_window_streak", "experience_z", "experience_a",
        "experience_y", "experience_ok", "rng_state",
    }
    assert required.issubset(state.keys())
    assert state["margin_m"] == original.margin
    assert state["regressor"]["tau"] == original.tau

    clone = _controller(gamma=1.0, seed=999, epsilon=0.4)  # DIFFERENT seed
    clone.load_state_dict(state)

    rest = []
    for i in range(26, 40):
        do = original.act(z_stream[i])
        dc = clone.act(z_stream[i])
        rest.append(
            (do.mode, do.action, dc.mode, dc.action)
        )
        assert (do.mode, do.action) == (dc.mode, dc.action)
    assert len(rest) == 14
    # Learning stays in lockstep after the restore.
    assert original.tau == clone.tau
    assert original.margin == clone.margin
    assert original.n_experiences == clone.n_experiences


def test_checkpoint_rejects_topology_mismatch():
    state = _controller(gamma=0.0).state_dict()
    clone = SqrSliceController(
        state_dim=3, n_prbs=C, sla_threshold=LAMBDA,
        kernel=MaternKernel(dim=4, length_scale=0.05, nu=1.5),
    )
    try:
        clone.load_state_dict(state)
    except ValueError as e:
        assert "state_dim" in str(e)
    else:
        raise AssertionError("state_dim mismatch must raise")


def test_checkpoint_rejects_prb_mismatch():
    state = _controller(gamma=0.0).state_dict()
    clone = SqrSliceController(
        state_dim=2, n_prbs=C + 1, sla_threshold=LAMBDA,
        kernel=MaternKernel(dim=3, length_scale=0.05, nu=1.5),
    )
    try:
        clone.load_state_dict(state)
    except ValueError as e:
        assert "n_prbs" in str(e)
    else:
        raise AssertionError("n_prbs mismatch must raise")


# ── Run manually ────────────────────────────────────────────────────

if __name__ == "__main__":
    test_frozen_adaptation_defaults()
    test_scores_identical_f_minus_gamma_a_over_c()
    test_a_safe_mask_identical_across_gamma()
    test_gamma_one_shifts_choice_to_cheaper_action()
    test_gamma_zero_and_one_agree_when_quantile_prefers_cheap()
    test_p_penalty_identical_across_gamma_variants()
    test_uncertain_no_neighbors_uses_argmax_ftau()
    test_uncertain_no_neighbors_but_model_prefers_best_ftau()
    test_knn_picks_argmax_ftau_not_max_y()
    test_knn_restricted_to_k_nearest_successes()
    test_knn_edge_no_success_uses_nearest_experience()
    test_uncertain_epsilon_one_samples_softmax()
    test_softmax_prefers_high_f_when_temperature_low()
    test_margin_adds_round_sigma_m_with_sigma_one()
    test_margin_clipped_at_capacity()
    test_margin_default_is_frozen_m0()
    test_margin_zero_at_support_vector()
    test_no_adaptation_before_first_full_window()
    test_violations_drive_tau_down_and_margin_doubling()
    test_recovery_requires_two_consecutive_clean_windows()
    test_dirty_window_breaks_the_clean_streak()
    test_tau_stays_inside_published_range_and_margin_within_bounds()
    test_recovery_margin_floors_at_two()
    test_out_of_range_tau_rejected()
    test_out_of_range_margin_rejected()
    test_insert_virtual_support_exact_negative_weight()
    test_insert_virtual_support_rejects_out_of_range_action()
    test_same_seed_same_decision_sequence()
    test_different_seeds_can_diverge()
    test_actions_always_within_bounds()
    test_observe_records_successes_failures_and_violations()
    test_observe_rejects_out_of_range_action()
    test_subprediction_update_from_observed_improvement()
    test_checkpoint_roundtrip_reproduces_decisions()
    test_checkpoint_rejects_topology_mismatch()
    test_checkpoint_rejects_prb_mismatch()
    print("All controller tests passed.")
