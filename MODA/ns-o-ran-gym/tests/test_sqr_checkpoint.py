"""Tests for the SQR checkpoint file round-trip (plan §3-Fase 2, item 9).

The checkpoint must carry {support matrix, alphas, hyperparameters, tau, M,
violation window, success/non-success memory, RNG state} and a restored
controller must reproduce the decision stream EXACTLY (bit-for-bit modes
and actions), including through the pickle file helpers.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from nsoran.sqr.checkpoint import (
    CHECKPOINT_FORMAT,
    build_checkpoint,
    load_checkpoint,
    restore_controllers,
    save_checkpoint,
)
from nsoran.sqr.kernel import MaternKernel
from nsoran.sqr.sqr_controller import SqrSliceController

C = 52


def _controller(seed, gamma=1.0):
    return SqrSliceController(
        state_dim=6, n_prbs=C, gamma=gamma, sla_threshold=1.0,
        kernel=MaternKernel(dim=7, length_scale=0.3, nu=1.5),
        epsilon=0.4, seed=seed,
    )


def _stream(rng_z, n=30):
    zs = rng_z.normal(size=(n, 6)) * 0.4 + 0.5
    ys = rng_z.uniform(0.0, 1.5, size=n)
    acts = rng_z.integers(0, C + 1, size=n)
    return zs, acts, ys


def test_file_roundtrip_reproduces_decisions(tmp_path):
    rng_z = np.random.default_rng(77)
    zs, acts, ys = _stream(rng_z)

    original = _controller(seed=5)
    for i in range(15):
        original.observe(zs[i], int(acts[i]), float(ys[i]))
    for i in range(15, 20):          # advance the decision RNG pre-snapshot
        original.act(zs[i])

    path = str(tmp_path / "ckpt.pkl")
    meta = {"gamma": 1.0, "seed": 5, "note": "roundtrip"}
    save_checkpoint(path, build_checkpoint({0: original}, meta=meta))
    assert os.path.exists(path)

    payload = load_checkpoint(path)
    clone = _controller(seed=999)    # DIFFERENT seed: state must overwrite it
    returned_meta = restore_controllers(payload, {0: clone})
    assert returned_meta == meta

    for i in range(20, 30):
        do = original.act(zs[i])
        dc = clone.act(zs[i])
        assert (do.mode, do.action) == (dc.mode, dc.action)
        assert do.raw_action == dc.raw_action
        assert do.tau == dc.tau
        assert do.margin_m == dc.margin_m
    # Learning stays in lockstep through the file round-trip.
    assert original.tau == clone.tau
    assert original.margin == clone.margin
    assert original.n_experiences == clone.n_experiences
    np.testing.assert_array_equal(
        original.regressor.alphas, clone.regressor.alphas
    )


def test_multi_slice_checkpoint(tmp_path):
    rng_z = np.random.default_rng(11)
    ctls = {sid: _controller(seed=100 + sid) for sid in range(3)}
    zs, acts, ys = _stream(rng_z, n=12)
    for i in range(12):
        for sid in range(3):
            ctls[sid].observe(zs[i], int(acts[i]), float(ys[i]))

    path = str(tmp_path / "multi.pkl")
    save_checkpoint(path, build_checkpoint(ctls, meta={"slices": 3}))
    restored = {sid: _controller(seed=1) for sid in range(3)}
    restore_controllers(load_checkpoint(path), restored)

    for sid in range(3):
        assert restored[sid].tau == ctls[sid].tau
        assert restored[sid].margin == ctls[sid].margin
        assert restored[sid].n_experiences == ctls[sid].n_experiences
        np.testing.assert_array_equal(
            restored[sid].regressor.support_points,
            ctls[sid].regressor.support_points,
        )


def test_checkpoint_carries_violation_window_and_streak():
    ctl = _controller(seed=2)
    ctl.regressor_state = None  # no-op attribute guard
    z = np.full(6, 0.5)
    for _ in range(3):
        ctl.observe(z, action=10, y=0.1)   # violations (W=100: no adapt yet)
    state = ctl.state_dict()
    assert len(state["violations"]) == 3
    assert all(state["violations"])
    assert state["window_count"] == 3
    assert state["margin_m"] == 10.0
    assert state["experience_ok"].tolist() == [False] * 3
    assert state["experience_z"].shape == (3, 6)


def test_restore_into_missing_slice_raises(tmp_path):
    ctl = _controller(seed=3)
    path = str(tmp_path / "one.pkl")
    save_checkpoint(path, build_checkpoint({0: ctl}))
    payload = load_checkpoint(path)
    with pytest.raises(ValueError, match="no controller"):
        restore_controllers(payload, {1: _controller(seed=4)})


def test_restore_into_extra_controller_raises(tmp_path):
    """Topology must match in BOTH directions: a pre-built controller with
    no slice in the checkpoint is an error (a partial restore would leave
    that slice at its fresh-init state silently)."""
    ctl = _controller(seed=3)
    path = str(tmp_path / "one.pkl")
    save_checkpoint(path, build_checkpoint({0: ctl}))
    payload = load_checkpoint(path)
    with pytest.raises(ValueError, match="both directions"):
        restore_controllers(
            payload, {0: _controller(seed=4), 1: _controller(seed=5)}
        )


def test_load_state_dict_validates_format_version():
    ctl = _controller(seed=8)
    state = ctl.state_dict()
    assert state["format_version"] == 2  # current payload version

    clone = _controller(seed=9)
    clone.load_state_dict(state)         # current version loads fine

    stale = dict(state)
    stale["format_version"] = 1
    with pytest.raises(ValueError, match="format_version"):
        _controller(seed=10).load_state_dict(stale)

    future = dict(state)
    future["format_version"] = 99
    with pytest.raises(ValueError, match="format_version"):
        _controller(seed=11).load_state_dict(future)

    missing = {k: v for k, v in state.items() if k != "format_version"}
    with pytest.raises(ValueError, match="format_version"):
        _controller(seed=12).load_state_dict(missing)


def test_load_rejects_non_sqr_and_wrong_version(tmp_path):
    bad = tmp_path / "bad.pkl"
    save_checkpoint(bad, {"format": "other", "version": 1})
    with pytest.raises(ValueError, match="not an SQR checkpoint"):
        load_checkpoint(str(bad))

    old = tmp_path / "old.pkl"
    payload = build_checkpoint({0: _controller(seed=6)})
    payload["version"] = 1
    save_checkpoint(old, payload)
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(str(old))


def test_checkpoint_format_constant():
    assert CHECKPOINT_FORMAT == "sqr-checkpoint"
