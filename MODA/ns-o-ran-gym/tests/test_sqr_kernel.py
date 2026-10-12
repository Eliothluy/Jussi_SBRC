"""Tests for nsoran.sqr.kernel (normalized Matérn kernel).

Checklist (PLANO_IMPLEMENTACAO_SQR_NORI.md §3-Fase 2 / GUIA §3.6):
symmetry, k(x,x)=1, similarity decreasing with distance, per-dimension
length-scales, nu variants, vectorization vs loop, validation errors.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from nsoran.sqr.kernel import MaternKernel

RNG = np.random.default_rng(2026)


# ── Normalization and symmetry ──────────────────────────────────────

def test_kxx_equals_one_exact():
    """k(x, x) == 1.0 exactly for every supported nu."""
    X = RNG.normal(size=(20, 4))
    for nu in (0.5, 1.5, 2.5, np.inf):
        K = MaternKernel(dim=4, length_scale=1.0, nu=nu)(X)
        assert np.all(np.diag(K) == 1.0), f"nu={nu}"


def test_symmetry_of_gram_matrix():
    """K(X, X) is exactly symmetric."""
    X = RNG.normal(size=(15, 3))
    K = MaternKernel(dim=3, length_scale=[0.5, 1.0, 2.0], nu=1.5)(X)
    assert K.shape == (15, 15)
    assert np.array_equal(K, K.T)


def test_cross_matrix_symmetry():
    """k(X, Y) == k(Y, X).T."""
    X = RNG.normal(size=(6, 2))
    Y = RNG.normal(size=(9, 2))
    kern = MaternKernel(dim=2, length_scale=0.7, nu=1.5)
    assert np.allclose(kern(X, Y), kern(Y, X).T, atol=1e-12)


def test_values_in_unit_interval():
    """Normalized Matérn values stay in (0, 1] for finite distances."""
    X = RNG.normal(size=(30, 3)) * 5
    K = MaternKernel(dim=3, length_scale=1.0, nu=1.5)(X)
    assert np.all(K > 0.0)
    assert np.all(K <= 1.0 + 1e-12)


# ── Monotonicity with distance ──────────────────────────────────────

def test_similarity_decreases_with_distance():
    """k(0, d) is strictly decreasing for d >= 0 (all nu)."""
    xs = np.linspace(0.0, 4.0, 41)[:, None]
    zeros = np.zeros((1, 1))
    for nu in (0.5, 1.5, 2.5, np.inf):
        k = MaternKernel(dim=1, length_scale=1.0, nu=nu)
        vals = k(xs, zeros)[:, 0]
        assert np.all(np.diff(vals) < 0.0), f"nu={nu}"


def test_closed_form_nu_half():
    """nu=0.5 is the exponential kernel exp(-d)."""
    kern = MaternKernel(dim=1, length_scale=2.0, nu=0.5)
    d = 3.0
    expected = math.exp(-d / 2.0)
    got = kern(np.array([[0.0]]), np.array([[d]]))[0, 0]
    assert abs(got - expected) < 1e-12


def test_closed_form_nu_one_half_at_unit_distance():
    """nu=1.5: k(1) = (1 + sqrt(3)) * exp(-sqrt(3)) for l=1."""
    kern = MaternKernel(dim=1, length_scale=1.0, nu=1.5)
    expected = (1.0 + math.sqrt(3.0)) * math.exp(-math.sqrt(3.0))
    got = kern(np.array([[0.0]]), np.array([[1.0]]))[0, 0]
    assert abs(got - expected) < 1e-12


# ── Per-dimension length-scales ─────────────────────────────────────

def test_per_dimension_length_scale_anisotropy():
    """A large l_d makes dim d more similar than a dim with small l_d."""
    kern = MaternKernel(dim=2, length_scale=[10.0, 0.1], nu=1.5)
    x = np.array([[0.0, 0.0]])
    y_far_in_dim0 = np.array([[1.0, 0.0]])   # distance 1 with l=10
    y_far_in_dim1 = np.array([[0.0, 1.0]])   # distance 1 with l=0.1
    k0 = kern(x, y_far_in_dim0)[0, 0]
    k1 = kern(x, y_far_in_dim1)[0, 0]
    assert k0 > k1
    assert k1 < 0.01  # effectively dissimilar along the tight dimension


def test_larger_length_scale_increases_similarity():
    X = np.array([[0.0]])
    Y = np.array([[2.0]])
    sims = [MaternKernel(dim=1, length_scale=l, nu=1.5)(X, Y)[0, 0]
            for l in (0.5, 1.0, 2.0, 4.0)]
    assert np.all(np.diff(sims) > 0.0)


# ── Vectorization and shapes ────────────────────────────────────────

def test_matrix_matches_pairwise_loop():
    X = RNG.normal(size=(7, 3))
    Y = RNG.normal(size=(5, 3))
    kern = MaternKernel(dim=3, length_scale=[0.3, 1.0, 2.5], nu=1.5)
    K = kern(X, Y)
    assert K.shape == (7, 5)
    for i in range(7):
        for j in range(5):
            d = np.sqrt(np.sum(((X[i] - Y[j]) / kern.length_scale) ** 2))
            expected = (1.0 + math.sqrt(3.0) * d) * math.exp(-math.sqrt(3.0) * d)
            assert abs(K[i, j] - expected) < 1e-12


def test_single_point_input_shapes():
    kern = MaternKernel(dim=2, length_scale=1.0, nu=1.5)
    K = kern(np.array([0.0, 0.0]))  # single 1-D point
    assert K.shape == (1, 1)
    assert K[0, 0] == 1.0
    K2 = kern(np.array([0.0, 0.0]), np.array([[1.0, 0.0], [0.0, 2.0]]))
    assert K2.shape == (1, 2)


def test_diag_matches_full_matrix():
    X = RNG.normal(size=(8, 2))
    Y = RNG.normal(size=(8, 2))
    kern = MaternKernel(dim=2, length_scale=0.9, nu=2.5)
    full = kern(X, Y)
    assert np.allclose(kern.diag(X, Y), np.diag(full), atol=1e-15)
    assert np.all(kern.diag(X, X) == 1.0)


# ── Validation ──────────────────────────────────────────────────────

def test_invalid_dim_rejected():
    try:
        MaternKernel(dim=0)
    except ValueError:
        pass
    else:
        raise AssertionError("dim=0 must raise")


def test_invalid_length_scale_rejected():
    for bad in (0.0, -1.0, np.nan, np.inf):
        try:
            MaternKernel(dim=2, length_scale=bad)
        except ValueError:
            continue
        raise AssertionError(f"length_scale={bad} must raise")


def test_wrong_length_scale_shape_rejected():
    try:
        MaternKernel(dim=3, length_scale=np.ones(2))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong length_scale shape must raise")


def test_invalid_nu_rejected():
    try:
        MaternKernel(dim=2, nu=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("nu=1.0 must raise (only 0.5/1.5/2.5/inf supported)")


def test_non_finite_points_rejected():
    kern = MaternKernel(dim=2)
    try:
        kern(np.array([[np.nan, 0.0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("NaN coordinates must raise")


def test_wrong_point_dimension_rejected():
    kern = MaternKernel(dim=2)
    try:
        kern(np.array([[1.0, 2.0, 3.0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong feature dimension must raise")


# ── Run manually ────────────────────────────────────────────────────

if __name__ == "__main__":
    test_kxx_equals_one_exact()
    test_symmetry_of_gram_matrix()
    test_cross_matrix_symmetry()
    test_values_in_unit_interval()
    test_similarity_decreases_with_distance()
    test_closed_form_nu_half()
    test_closed_form_nu_one_half_at_unit_distance()
    test_per_dimension_length_scale_anisotropy()
    test_larger_length_scale_increases_similarity()
    test_matrix_matches_pairwise_loop()
    test_single_point_input_shapes()
    test_diag_matches_full_matrix()
    test_invalid_dim_rejected()
    test_invalid_length_scale_rejected()
    test_wrong_length_scale_shape_rejected()
    test_invalid_nu_rejected()
    test_non_finite_points_rejected()
    test_wrong_point_dimension_rejected()
    print("All kernel tests passed.")
