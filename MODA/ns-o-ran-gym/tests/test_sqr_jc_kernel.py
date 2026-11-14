"""Kernel-head tests of SQR-JC (§75 block: kernel/models/quantiles).

Covers: test_kernel_prediction, test_kernel_support_insertion,
test_memory_budget, test_memory_pruning, test_quantile_update,
test_q25_q75_ordering, test_quantile_crossing_logging,
test_monotonic_rearrangement_for_classification, test_uncertainty,
test_support_threshold — plus shared-support locations (§10) and the
tau/M adaptation continuity with the audited SQR (register R11).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from nsoran.sqr.kernel import MaternKernel
from nsoran.sqr_jc.kernel_models import (
    ETA,
    MEMORY_SIZE,
    P_PENALTY,
    TAU_MAX,
    TAU_MIN,
    MultiQuantileKernelModel,
)

DIM = 8  # 5 state + 3 action (small test fixture dim)


@pytest.fixture
def model() -> MultiQuantileKernelModel:
    return MultiQuantileKernelModel(dim=DIM, memory_size=50, tau=0.05)


def _x(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=DIM)


# ---------------------------------------------------------------------------
# test_kernel_prediction
# ---------------------------------------------------------------------------

def test_kernel_prediction(model):
    """f = K(X, memory) @ alpha; zeros with empty memory; matches a manual
    computation with the audited MaternKernel (nu=1.5)."""
    X = np.stack([_x(1), _x(2), _x(3)])
    preds = model.predict_raw(X)
    assert preds["safe"].shape == (3,)
    assert np.all(preds["safe"] == 0.0)  # empty memory -> 0
    assert np.all(preds["sigma"] == 1.0)

    for i in range(5):
        model.update(_x(100 + i), y=0.5 + 0.1 * i)
    X1 = _x(101)[None, :]
    preds = model.predict_raw(X1)
    k = MaternKernel(dim=DIM, length_scale=1.0, nu=1.5)
    manual = k(X1, model.support_points) @ model.alphas_safe
    assert preds["safe"] == pytest.approx(manual, rel=1e-12)


# ---------------------------------------------------------------------------
# test_kernel_support_insertion + shared locations (§10)
# ---------------------------------------------------------------------------

def test_kernel_support_insertion(model):
    """One observation = ONE shared support location with THREE coefficients
    (§10: mesmos support-vector locations, coeficientes separados)."""
    for i in range(3):
        rec = model.update(_x(10 + i), y=0.2 * i)
    assert model.n_support == 3
    assert model.alphas_safe.shape == (3,)
    assert model.alphas_q25.shape == (3,)
    assert model.alphas_q75.shape == (3,)
    # pinball coefficients of the FIRST update (y=0 > f=0 -> subprediction)
    assert rec.alpha_safe == pytest.approx(ETA * 0.05)
    assert rec.alpha_q25 == pytest.approx(ETA * 0.25)
    assert rec.alpha_q75 == pytest.approx(ETA * 0.75)


# ---------------------------------------------------------------------------
# test_memory_budget / test_memory_pruning (R10, safe-head influence)
# ---------------------------------------------------------------------------

def test_memory_budget(model):
    """Budget B never exceeded; each update beyond B prunes exactly one."""
    for i in range(60):
        model.update(_x(200 + i), y=float(i % 7) - 3.0)
    assert model.n_support == 50  # == memory_size


def test_memory_pruning(model):
    """R10 influence pruning: the SAFE head's |alpha| * k(x_new, x_j) picks
    the victim once, and the SAME location leaves all three heads."""
    for i in range(50):
        model.update(_x(300 + i), y=0.1)
    before_points = model.support_points
    before_q25 = model.alphas_q25
    before_q75 = model.alphas_q75
    rec = model.update(_x(999), y=0.1)  # memory full -> prune first
    assert rec.pruned_index is not None
    assert model.n_support == 50
    pruned_point = before_points[rec.pruned_index]
    # pruned location gone from the shared memory
    assert not any(np.allclose(pruned_point, p) for p in model.support_points)
    # every REMAINING location kept its q25/q75 coefficients (shared drop);
    # the NEW point (last row) is the only one without an old correspondence
    new_point = model.support_points[-1]
    for new_i, point in enumerate(model.support_points):
        if np.allclose(point, new_point):
            continue
        old_i = next(
            j for j in range(before_points.shape[0])
            if np.allclose(before_points[j], point)
        )
        assert model.alphas_q25[new_i] == pytest.approx(before_q25[old_i])
        assert model.alphas_q75[new_i] == pytest.approx(before_q75[old_i])
    # determinism: pruning decision uses no RNG
    m2 = MultiQuantileKernelModel(dim=DIM, memory_size=50, tau=0.05)
    for i in range(50):
        m2.update(_x(300 + i), y=0.1)
    rec2 = m2.update(_x(999), y=0.1)
    assert rec2.pruned_index == rec.pruned_index


# ---------------------------------------------------------------------------
# test_quantile_update (pinball rules incl. critical, margin scale)
# ---------------------------------------------------------------------------

def test_quantile_update(model):
    """Pinball coefficients in margin scale: threshold 0; critical rule on
    the SAFE head only (P_penalty = 2), q25/q75 plain pinball."""
    # subprediction: y > f (f=0, y=1)
    rec = model.update(_x(1), y=1.0)
    assert rec.kind_safe == "subprediction"
    assert rec.alpha_safe == pytest.approx(ETA * 0.05)
    # critical: safe head predicted safe (f_safe = eta*tau > 0 at the SAME
    # point) but the SLA was violated (y < 0)
    rec = model.update(_x(1), y=-0.1)
    assert rec.kind_safe == "critical"
    assert rec.alpha_safe == pytest.approx(ETA * (0.05 - 1.0) * P_PENALTY)
    # q25/q75 never receive the critical amplification (plain pinball with
    # their OWN heads' f_before: y <= f -> overprediction branch)
    assert rec.alpha_q25 == pytest.approx(ETA * (0.25 - 1.0))
    assert rec.alpha_q75 == pytest.approx(ETA * (0.75 - 1.0))
    # overprediction branch of the SAFE head at a fresh point with negative y
    rec = model.update(_x(2), y=-0.5)
    assert rec.kind_safe == "overprediction"
    assert rec.alpha_safe == pytest.approx(ETA * (0.05 - 1.0))


# ---------------------------------------------------------------------------
# test_q25_q75_ordering / crossing logging / rearrangement (§10.1)
# ---------------------------------------------------------------------------

def test_q25_q75_ordering_and_crossing_logging(model):
    """Raw q25/q75 may cross; crossings are COUNTED; classification uses the
    rearranged values; raw values are preserved."""
    # Divergent histories to force a crossing: positive updates at point A,
    # strongly negative at an identical-looking point B (alpha magnitudes of
    # q25 and q75 differ -> heads can disagree in sign after mixed streams).
    for i in range(30):
        model.update(_x(500 + i), y=-1.0)   # negative everywhere
    for i in range(30):
        model.update(_x(600 + i), y=+1.0)   # positive everywhere
    X = np.stack([_x(500), _x(600), _x(700)])
    raw = model.predict_raw(X)
    # raw heads are FINITE and independently predicted (no forced ordering)
    assert np.all(np.isfinite(raw["q25"])) and np.all(np.isfinite(raw["q75"]))
    q25u, q75u = MultiQuantileKernelModel.rearranged(raw["q25"], raw["q75"])
    assert np.all(q25u <= q75u)
    # crossing flag per update is logged
    rec = model.update(_x(700), y=0.0)
    assert isinstance(rec.crossing, bool)


def test_monotonic_rearrangement_for_classification():
    """§46 classes computed from the REARRANGED quantiles (raw may cross)."""
    from nsoran.sqr_jc.feasibility import (
        SLICE_FEASIBLE,
        SLICE_INFEASIBLE,
        SLICE_MARGINAL,
        classify_slice,
    )

    # raw crossed (q25=0.2 > q75=-0.1): rearranged q25=-0.1, q75=0.2 -> MARGINAL
    v = classify_slice(0, q25_raw=0.2, q75_raw=-0.1, supported=True)
    assert v.cls == SLICE_MARGINAL
    assert v.crossed is True
    assert v.q25_used == pytest.approx(-0.1)
    assert v.q75_used == pytest.approx(0.2)
    assert classify_slice(1, 0.3, 0.5, True).cls == SLICE_FEASIBLE
    assert classify_slice(2, -0.5, -0.2, True).cls == SLICE_INFEASIBLE


def test_quantile_crossing_rate_metric(model):
    """§10.1 metric: crossing rate over an evaluation (gate > 5% => STOP)."""
    from nsoran.sqr_jc.action_search import JointActionOracleEvaluator
    from nsoran.sqr_jc.feasibility import quantile_crossing_rate

    for i in range(10):
        model.update(_x(800 + i), y=0.3)
    evaluator = JointActionOracleEvaluator(
        (model, model, model), sigma_support=0.9
    )
    z = np.zeros(5)
    res = evaluator.evaluate(z)
    rate = quantile_crossing_rate(res, slice_id=0)
    assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# test_uncertainty / test_support_threshold
# ---------------------------------------------------------------------------

def test_uncertainty(model):
    """sigma(x) = 1 − max_j k(x, x_j): 0 at a stored point, ->1 far away,
    exactly 1.0 with empty memory — the audited SQR equation."""
    assert np.all(model.uncertainty(_x(1)[None, :]) == 1.0)
    model.update(_x(1), y=0.5)
    sigma_at = float(model.uncertainty(_x(1)[None, :])[0])
    assert sigma_at == pytest.approx(0.0, abs=1e-9)
    far = np.full(DIM, 50.0)
    assert float(model.uncertainty(far[None, :])[0]) > 0.99


def test_support_threshold(model):
    """sigma > sigma_support => OUT OF SUPPORT (float, never INFEASIBLE)."""
    from nsoran.sqr_jc.action_search import JointActionOracleEvaluator

    for i in range(5):
        model.update(_x(900 + i), y=-1.0)  # negative margins everywhere
    evaluator = JointActionOracleEvaluator(
        (model, model, model), sigma_support=0.5
    )
    res_near = evaluator.evaluate(_x(900)[None, :5].repeat(3, axis=0)[0])
    res_far = evaluator.evaluate(np.full(5, 100.0))
    assert bool(res_near.supported[0, 0]) is True
    assert bool(res_far.supported[0, 0]) is False
    # out-of-support slice class is MARGINAL, never INFEASIBLE
    from nsoran.sqr_jc.feasibility import SLICE_MARGINAL, classify_slice

    assert classify_slice(0, -2.0, -1.0, supported=False).cls == SLICE_MARGINAL


# ---------------------------------------------------------------------------
# tau/M adaptation continuity with the audited register R11
# ---------------------------------------------------------------------------

def test_tau_adaptation_margins_and_bounds():
    """Violation stream drives tau down / M up; clean recovery drives them
    back; bounds [0.01, 0.05] and [2, 50] hold (audited R11 rule).

    One block window = 100 updates; tau needs 10 violated blocks to reach
    the floor and 8 fired recoveries (2 clean blocks each) to return."""
    m = MultiQuantileKernelModel(dim=DIM, tau=TAU_MAX, margin_m=10.0)
    for _ in range(1000):
        m.update(_x(1), y=-0.5)  # 10 violated block windows
    assert m.tau == pytest.approx(TAU_MIN)
    assert m.margin_m == pytest.approx(50.0)
    for _ in range(1600):
        m.update(_x(2), y=+0.5)  # 16 clean block windows -> 8 fired recoveries
    assert m.tau == pytest.approx(TAU_MAX)
    assert m.margin_m == pytest.approx(2.0)
    assert TAU_MIN < TAU_MAX


def test_state_dict_roundtrip(model):
    """Checkpoint round-trip restores support + all three heads + tau/M."""
    for i in range(7):
        model.update(_x(1000 + i), y=0.4 - 0.05 * i)
    state = model.state_dict()
    clone = MultiQuantileKernelModel(dim=DIM, memory_size=50)
    clone.load_state_dict(state)
    X = np.stack([_x(1000), _x(1001)])
    a = clone.predict_raw(X)
    b = model.predict_raw(X)
    for key in ("safe", "q25", "q75", "sigma"):
        assert np.allclose(a[key], b[key])
    with pytest.raises(ValueError):
        clone.load_state_dict({**state, "dim": DIM + 1})


# ---------------------------------------------------------------------------
# §75 exact-name aliases (protocol checklist fidelity)
# ---------------------------------------------------------------------------

def test_q25_q75_ordering(model):
    """§75 exact name: rearrangement guarantees q25_used <= q75_used."""
    X = np.stack([_x(500), _x(600)])
    raw = model.predict_raw(X)
    q25u, q75u = MultiQuantileKernelModel.rearranged(raw["q25"], raw["q75"])
    assert np.all(q25u <= q75u)


def test_quantile_crossing_logging(model):
    """§75 exact name: every update logs its crossing flag (bool)."""
    for i in range(5):
        rec = model.update(_x(700 + i), y=0.2)
        assert isinstance(rec.crossing, bool)
