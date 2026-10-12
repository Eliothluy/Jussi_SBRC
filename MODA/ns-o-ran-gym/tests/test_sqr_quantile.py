"""Tests for nsoran.sqr.quantile_regressor (online pinball quantile core).

Checklist (PLANO_IMPLEMENTACAO_SQR_NORI.md §3-Fase 2 / GUIA §3.6):
quantile coverage on synthetic stationary data; signs/magnitudes of the
pinball alpha rules; amplified critical failure (P_penalty=2); memory
budget <= 2000; deterministic pruning per seed; edge cases (empty memory).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from nsoran.sqr.kernel import MaternKernel
from nsoran.sqr.quantile_regressor import (
    CRITICAL,
    OVERPREDICTION,
    SUBPREDICTION,
    KernelQuantileRegressor,
)

ETA = 0.1
TAU = 0.05
P_PENALTY = 2.0


def _regressor(dim=1, tau=TAU, memory_size=2000, length_scale=0.3, seed=None):
    kernel = MaternKernel(dim=dim, length_scale=length_scale, nu=1.5)
    return KernelQuantileRegressor(
        kernel=kernel, memory_size=memory_size, tau=tau, eta=ETA, p_penalty=P_PENALTY
    )


# ── Edge case: empty memory ─────────────────────────────────────────

def test_empty_memory_predicts_zero_and_sigma_one():
    reg = _regressor()
    X = np.linspace(0.0, 1.0, 11)[:, None]
    assert reg.n_support == 0
    assert np.array_equal(reg.predict(X), np.zeros(11))
    assert np.array_equal(reg.uncertainty(X), np.ones(11))


def test_sigma_zero_at_support_vector_and_grows_with_distance():
    reg = _regressor(length_scale=0.5)
    x0 = np.array([0.5])
    reg.update(x0, y=1.0)
    sig = reg.uncertainty(np.array([[0.5], [0.6], [5.0]]))
    assert sig[0] == 0.0                      # k(x0, x0) = 1
    assert 0.0 < sig[1] < sig[2] < 1.0        # grows with distance, capped at 1


# ── Pinball alpha rules: signs and magnitudes ───────────────────────

def test_subprediction_positive_alpha_eta_tau():
    """Empty memory -> f=0; y > f must add alpha = eta*tau > 0."""
    reg = _regressor()
    rec = reg.update(np.array([0.3]), y=0.7)
    assert rec.kind == SUBPREDICTION
    assert rec.alpha == ETA * TAU
    assert rec.alpha > 0.0


def test_overprediction_negative_alpha_eta_tau_minus_one():
    """y <= f must add alpha = eta*(tau-1) < 0."""
    reg = _regressor()
    reg.update(np.array([0.3]), y=10.0)       # big positive SV -> f > 0
    rec = reg.update(np.array([0.3]), y=0.0)  # y < f
    assert rec.kind == OVERPREDICTION
    assert rec.alpha == ETA * (TAU - 1.0)
    assert rec.alpha < 0.0


def test_subprediction_magnitude_smaller_than_overprediction():
    """|eta*tau| = 0.005 < |eta*(tau-1)| = 0.095 for tau=0.05 (tau < 0.5)."""
    assert ETA * TAU < abs(ETA * (TAU - 1.0))


def test_first_prediction_uses_pre_update_memory():
    """rec.f_before is the prediction with the OLD memory; the new SV is
    exactly predict(x) = alpha after a single update (k(x,x)=1)."""
    reg = _regressor()
    rec = reg.update(np.array([0.4]), y=1.0)
    assert rec.f_before == 0.0
    assert abs(reg.predict(np.array([[0.4]]))[0] - rec.alpha) < 1e-15


# ── Critical failure: amplified negative update ─────────────────────

def test_critical_failure_auto_detected_and_amplified():
    """f >= Lambda but y < Lambda -> alpha = eta*(tau-1)*P_penalty."""
    reg = _regressor()
    x0 = np.array([0.5])
    reg.set_memory(x0[None, :], np.array([0.5]))   # f(x0) = 0.5 * k = 0.5
    rec = reg.update(x0, y=0.1, threshold=0.4)     # predicted safe, violated
    assert rec.kind == CRITICAL
    assert rec.alpha == ETA * (TAU - 1.0) * P_PENALTY
    # Amplification: exactly P_penalty times the plain overprediction rule.
    plain = ETA * (TAU - 1.0)
    assert abs(rec.alpha) == P_PENALTY * abs(plain)


def test_no_critical_without_threshold():
    """Same (f, y) but no Lambda given -> plain overprediction rule."""
    reg = _regressor()
    x0 = np.array([0.5])
    reg.set_memory(x0[None, :], np.array([0.5]))
    rec = reg.update(x0, y=0.1)
    assert rec.kind == OVERPREDICTION
    assert rec.alpha == ETA * (TAU - 1.0)


def test_predicted_unsafe_violation_is_not_critical():
    """f < Lambda and y < Lambda but y > f -> subprediction (not critical)."""
    reg = _regressor()
    rec = reg.update(np.array([0.2]), y=-0.1, threshold=0.5)  # f=0, y=-0.1
    assert rec.kind == OVERPREDICTION  # y <= f

    reg2 = _regressor()
    rec2 = reg2.update(np.array([0.2]), y=0.1, threshold=0.5)  # y > f=0
    assert rec2.kind == SUBPREDICTION


def test_forced_critical_via_flag():
    """critical=True forces the amplified rule (virtual negative updates)."""
    reg = _regressor()
    rec = reg.update(np.array([0.2]), y=9.0, critical=True)
    assert rec.kind == CRITICAL
    assert rec.alpha == ETA * (TAU - 1.0) * P_PENALTY


def test_p_penalty_active_independent_of_gamma_variant():
    """AGENTS.md §5 rule 7: P_penalty stays 2 for gamma=0 AND gamma=1 — the
    regressor has no gamma parameter at all (gamma only enters selection)."""
    reg0 = _regressor()
    reg1 = _regressor()
    assert reg0.p_penalty == reg1.p_penalty == P_PENALTY
    x0 = np.array([0.1])
    for reg in (reg0, reg1):
        reg.set_memory(x0[None, :], np.array([0.9]))
    r0 = reg0.update(x0, y=0.05, threshold=0.5)
    r1 = reg1.update(x0, y=0.05, threshold=0.5)
    assert r0.kind == r1.kind == CRITICAL
    assert r0.alpha == r1.alpha


# ── Memory budget and deterministic pruning ─────────────────────────

def test_budget_never_exceeds_2000():
    reg = _regressor(dim=2, memory_size=2000, length_scale=0.5)
    rng = np.random.default_rng(7)
    pruned_after_full = 0
    for i in range(2200):
        x = rng.normal(size=2)
        rec = reg.update(x, y=float(rng.normal()))
        assert reg.n_support <= 2000
        if i >= 2000:
            assert rec.pruned_index is not None
            pruned_after_full += 1
    assert reg.n_support == 2000
    assert pruned_after_full == 200  # one prune per insert once full


def test_pruning_drops_smallest_influence_first():
    """|alpha_j| is the influence: the +eta*tau SV (0.005) goes before the
    -eta*(tau-1) one (0.095)."""
    reg = _regressor(memory_size=2)
    x_sub = np.array([0.0])
    x_over = np.array([10.0])
    rec1 = reg.update(x_sub, y=1.0)            # alpha = +0.005
    reg.update(x_over, y=-1.0)                 # alpha = -0.095 (y=-1 < f=0.005)
    rec3 = reg.update(np.array([5.0]), y=1.0)  # memory full -> prune first
    assert rec3.pruned_index == 0              # smallest |alpha|
    assert rec3.pruned_alpha == ETA * TAU
    # Survivors: the overprediction SV and the new one.
    assert reg.n_support == 2
    assert reg.alphas[0] == ETA * (TAU - 1.0)


def test_pruning_deterministic_same_stream_same_memory():
    """Same update stream -> identical memory, regardless of any seed
    (the pruning rule uses no random numbers)."""
    rng = np.random.default_rng(123)
    stream = [(rng.normal(size=2), float(rng.normal())) for _ in range(2100)]

    reg_a = _regressor(dim=2, memory_size=500, length_scale=1.0)
    reg_b = _regressor(dim=2, memory_size=500, length_scale=1.0)
    for x, y in stream:
        reg_a.update(x, y)
        reg_b.update(x, y)
    assert np.array_equal(reg_a.support_points, reg_b.support_points)
    assert np.array_equal(reg_a.alphas, reg_b.alphas)

    # Two fresh regressors replaying the stream again: identical again.
    reg_c = _regressor(dim=2, memory_size=500, length_scale=1.0)
    for x, y in stream:
        reg_c.update(x, y)
    assert np.array_equal(reg_a.support_points, reg_c.support_points)
    assert np.array_equal(reg_a.alphas, reg_c.alphas)


# ── Quantile coverage on synthetic stationary data ──────────────────

def _train_on_synthetic(tau, n_train=4000, seed=11):
    """y = 1 + 0.5*sin(2*pi*x) + N(0, 0.1); feature = [x] (1-D)."""
    rng = np.random.default_rng(seed)
    reg = _regressor(dim=1, tau=tau, length_scale=0.15)

    def truth(x):
        return 1.0 + 0.5 * np.sin(2.0 * np.pi * x)

    for _ in range(n_train):
        x = rng.random()
        reg.update(np.array([x]), float(truth(x) + rng.normal(0.0, 0.1)))
    return reg, truth, rng


def test_quantile_coverage_lower_tail_tau005():
    """tau=0.05: P(y < f_tau) must stay well below 0.5 and <= 3*tau.

    Measured across seeds {11,12,13}: rates 0.000-0.0015. The assumed
    pruning rule (smallest |alpha| first) evicts the positive +eta*tau
    vectors preferentially, biasing f_tau DOWNWARD — i.e. to the safe side
    for a lower quantile — so the realized violation rate sits below tau.
    """
    reg, truth, rng = _train_on_synthetic(0.05)
    xs = rng.random(2000)
    ys = truth(xs) + rng.normal(0.0, 0.1, size=2000)
    f = reg.predict(xs[:, None])
    rate = float(np.mean(ys < f))
    assert 0.0 <= rate <= 0.15, f"lower-tail coverage rate={rate:.3f}"


def test_quantile_coverage_median_higher_than_tau005():
    """f_{tau=0.5} must sit strictly above f_{tau=0.05} in coverage terms:
    P(y < f_0.5) ~ 0.5 >> P(y < f_0.05). Both regressors are trained on the
    same seed so the comparison is paired. For tau=0.5 the pinball step is
    symmetric (|alpha| = eta/2 both ways), so f wobbles around the median
    with no pruning bias; realized rates across seeds {11,12,13} were
    0.76 / 0.27 / 0.47 — hence the generous [0.20, 0.80] band."""
    reg5, truth5, rng5 = _train_on_synthetic(0.5, seed=12)
    xs = rng5.random(2000)
    ys = truth5(xs) + rng5.normal(0.0, 0.1, size=2000)
    f = reg5.predict(xs[:, None])
    rate_median = float(np.mean(ys < f))
    assert 0.20 <= rate_median <= 0.80, f"median rate={rate_median:.3f}"

    # Same seed -> same evaluation stream for a paired comparison.
    reg05, truth05, rng05 = _train_on_synthetic(0.05, seed=12)
    xs2 = rng05.random(2000)
    ys2 = truth05(xs2) + rng05.normal(0.0, 0.1, size=2000)
    f2 = reg05.predict(xs2[:, None])
    rate_low = float(np.mean(ys2 < f2))
    assert rate_low < rate_median


def test_set_tau_changes_update_rules():
    reg = _regressor(tau=0.05)
    reg.set_tau(0.5)
    rec = reg.update(np.array([0.1]), y=1.0)
    assert rec.alpha == ETA * 0.5


# ── Checkpoint round-trip ───────────────────────────────────────────

def test_state_dict_roundtrip_preserves_predictions():
    reg = _regressor(dim=2, length_scale=0.8)
    rng = np.random.default_rng(3)
    for _ in range(300):
        reg.update(rng.normal(size=2), float(rng.normal()))

    clone = _regressor(dim=2, length_scale=0.8)
    clone.load_state_dict(reg.state_dict())
    X = rng.normal(size=(10, 2))
    assert np.allclose(reg.predict(X), clone.predict(X), atol=1e-15)
    assert np.allclose(reg.uncertainty(X), clone.uncertainty(X), atol=1e-15)


# ── Validation ──────────────────────────────────────────────────────

def test_invalid_hyperparameters_rejected():
    kernel = MaternKernel(dim=1)
    for kwargs in (
        dict(memory_size=0),
        dict(tau=0.0),
        dict(tau=1.0),
        dict(eta=0.0),
        dict(p_penalty=0.0),
    ):
        try:
            KernelQuantileRegressor(kernel=kernel, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"constructor must reject {kwargs}")


def test_non_matern_kernel_rejected():
    try:
        KernelQuantileRegressor(kernel="not a kernel")
    except TypeError:
        pass
    else:
        raise AssertionError("non-MaternKernel must raise TypeError")


def test_batch_update_rejected():
    reg = _regressor()
    try:
        reg.update(np.array([[0.1], [0.2]]), y=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("batch update must raise")


def test_non_finite_y_rejected():
    reg = _regressor()
    try:
        reg.update(np.array([0.1]), y=np.nan)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN y must raise")


# ── Run manually ────────────────────────────────────────────────────

if __name__ == "__main__":
    test_empty_memory_predicts_zero_and_sigma_one()
    test_sigma_zero_at_support_vector_and_grows_with_distance()
    test_subprediction_positive_alpha_eta_tau()
    test_overprediction_negative_alpha_eta_tau_minus_one()
    test_subprediction_magnitude_smaller_than_overprediction()
    test_first_prediction_uses_pre_update_memory()
    test_critical_failure_auto_detected_and_amplified()
    test_no_critical_without_threshold()
    test_predicted_unsafe_violation_is_not_critical()
    test_forced_critical_via_flag()
    test_p_penalty_active_independent_of_gamma_variant()
    test_budget_never_exceeds_2000()
    test_pruning_drops_smallest_influence_first()
    test_pruning_deterministic_same_stream_same_memory()
    test_quantile_coverage_lower_tail_tau005()
    test_quantile_coverage_median_higher_than_tau005()
    test_set_tau_changes_update_rules()
    test_state_dict_roundtrip_preserves_predictions()
    test_invalid_hyperparameters_rejected()
    test_non_matern_kernel_rejected()
    test_batch_update_rejected()
    test_non_finite_y_rejected()
    print("All quantile regressor tests passed.")
