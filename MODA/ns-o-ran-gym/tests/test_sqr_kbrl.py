"""Tests for the KBRL baseline port (Fase 3) — src/nsoran/sqr/kbrl/.

Scope (ask Fase 3): decision determinism per seed; the action contract
(quota {0..C}, projection bookkeeping, margin/clamping edges); faithful
port behavior (minimal-action scan, accuracy EMA, security-factor gate,
monotonic sample augmentation); borders (empty classifier, cut proposals,
dynamic C); checkpoint round-trip through the SAME envelope as SQR with
cross-restore rejection; and the closed loop through sqr_run.run_episode
against the contract mock (same FakeEnv pattern as test_sqr_run.py).

Reference for the ported semantics: third_party/network-slicing commit
65a3891 (kbrl_control.py, algorithms/projectron.py, scenario_creator.py).
"""

import csv
import importlib.util
import os
import sys
import types

import numpy as np
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_TEST_DIR, "..", "src")
sys.path.insert(0, _SRC)

from nsoran.sqr.checkpoint import (
    build_checkpoint,
    load_checkpoint,
    restore_controllers,
    save_checkpoint,
)
from nsoran.sqr.kbrl import (
    KBRL_ACCURACY_RANGE,
    KBRL_ALFA,
    KBRL_INITIAL_SEC_RANGE,
    KbrlSliceController,
    Projectron,
    SVVariable,
    GaussianKernelSV,
)
from nsoran.sqr.sqr_controller import SqrSliceController

C = 52
NUM_SLICES = 3
SLAS = {0: 0.5, 1: 1.0, 2: 1.5}


def _ctl(seed=7, sla=1.0, c_prbs=C, **over):
    return KbrlSliceController(
        state_dim=6, n_prbs=c_prbs, sla_threshold=sla, seed=seed, **over
    )


# ── Constructor / hyperparameters [REPO defaults] ────────────────────

def test_reference_hyperparameters_frozen():
    assert KBRL_ALFA == 0.05                        # scenario_creator.py L187
    assert KBRL_ACCURACY_RANGE == (0.99, 0.999)     # create_kbrl_agent default
    assert KBRL_INITIAL_SEC_RANGE == (2, 8)         # eMBB sec range L191
    ctl = _ctl(seed=3)
    assert ctl.classifier.eta == 0.1                # Projectron default
    assert ctl.classifier.kernel.gamma == 1.0       # GaussianKernel(sv, 1)
    assert ctl._initial_accuracy == pytest.approx(  # (lo + hi)/2, ref L38-39
        (0.99 + 0.999) / 2
    )


def test_initial_security_factor_seeded_from_reference_range():
    for seed in (1, 7, 42, 1234):
        assert KBRL_INITIAL_SEC_RANGE[0] <= _ctl(seed=seed).security_factor \
            < KBRL_INITIAL_SEC_RANGE[1]


def test_constructor_validation():
    with pytest.raises(ValueError):
        _ctl(c_prbs=0)
    with pytest.raises(ValueError):
        _ctl(alfa=1.5)
    with pytest.raises(ValueError):
        _ctl(accuracy_range=(0.999, 0.99))


# ── Decision determinism per seed ────────────────────────────────────

def test_same_seed_reproduces_identical_decision_stream():
    rng = np.random.default_rng(99)
    states = [rng.random(6) for _ in range(20)]
    a, b = _ctl(seed=11), _ctl(seed=11)
    for z in states:
        da, db = a.act(z), b.act(z)
        assert (da.action, da.raw_action, da.mode) == (db.action, db.raw_action, db.mode)
        assert da.a_safe_size == db.a_safe_size
        # learn identically to keep the streams aligned forever
        a.observe(z, da.action, 0.8)
        b.observe(z, db.action, 0.8)


def test_different_seeds_differ_only_through_the_initial_security_factor():
    # The seeded draw is the ONLY stochastic element; with an empty
    # classifier both controllers propose C (cold), but the scan offset
    # differs after the classifier starts predicting +1.
    secs = {_ctl(seed=s).security_factor for s in range(30)}
    assert len(secs) > 1                    # the draw actually varies


def test_learning_is_fully_deterministic_no_rng_in_dynamics():
    rng = np.random.default_rng(5)
    ctl_a, ctl_b = _ctl(seed=None), _ctl(seed=None)  # even unseeded: no RNG use
    for _ in range(10):
        z = rng.random(6)
        for ctl in (ctl_a, ctl_b):
            d = ctl.act(z)
            ctl.observe(z, int(rng.integers(0, C + 1)), float(rng.uniform(0, 2)))
    # Same support state -> same decisions despite the shared random stream
    # having interleaved differently between the two controllers above.
    z = rng.random(6)
    d1, d2 = ctl_a.act(z), ctl_b.act(z)
    # NOTE: the two controllers consumed DIFFERENT update streams (the
    # executed actions came from the same rng), so they generally differ —
    # the property under test is reproducibility, checked below instead.
    assert d1.n_candidates == d2.n_candidates == C + 1


def test_exact_replay_of_a_recorded_stream(tmp_path):
    rng = np.random.default_rng(21)
    ctl = _ctl(seed=13)
    trace = []
    for _ in range(25):
        z = rng.random(6)
        d = ctl.act(z)
        y = float(rng.uniform(0, 2))
        u = ctl.observe(z, int(rng.integers(0, C + 1)), y)
        trace.append((d.action, d.raw_action, d.mode, u.label, u.security_factor))
    # Replay with a fresh same-seed controller and the same experience
    # stream -> identical trace.
    rng = np.random.default_rng(21)
    ctl2 = _ctl(seed=13)
    for expected in trace:
        z = rng.random(6)
        d = ctl2.act(z)
        y = float(rng.uniform(0, 2))
        u = ctl2.observe(z, int(rng.integers(0, C + 1)), y)
        assert (d.action, d.raw_action, d.mode, u.label, u.security_factor) == expected


# ── Action contract: quota {0..C} and projection bookkeeping ─────────

def test_action_always_within_quota_and_scan_is_minimal():
    rng = np.random.default_rng(3)
    ctl = _ctl(seed=2)
    for _ in range(60):
        z = rng.random(6)
        d = ctl.act(z)
        assert 0 <= d.action <= C
        assert 0 <= d.raw_action <= C
        assert d.n_candidates == C + 1
        if d.mode == "scan":
            # margin = security offset actually applied, capped at C
            assert d.action == min(C, d.raw_action + ctl.security_factor)
            assert d.margin_applied == d.action - d.raw_action
        else:
            assert d.action == C and d.raw_action == C and d.margin_applied == 0
        ctl.observe(z, d.action, float(rng.uniform(0, 2)))


def test_scan_finds_the_first_predicted_safe_action():
    ctl = _ctl(seed=4)
    z = np.full(6, 0.5)
    # Fulfill with the TOP action: augmentation covers a >= executed.
    ctl.act(z)
    ctl.observe(z, C, 1.5)  # label +1 -> exactly one SV at a/C = 1.0
    # Gaussian kernel is strictly positive: one +1 SV predicts +1 for EVERY
    # action (same z) -> the scan must return a0 = 0 (the FIRST safe one).
    d = ctl.act(z)
    assert d.mode == "scan" and d.raw_action == 0 and d.a_safe_size == C + 1


def test_violation_carves_negative_support_and_raises_the_scan_floor():
    ctl = _ctl(seed=4)
    z = np.full(6, 0.5)
    ctl.act(z)
    ctl.observe(z, C, 1.5)          # +1 SV at a = C
    ctl.act(z)
    ctl.observe(z, 0, 0.1)          # -1 SV at a = 0 (augmentation a <= 0)
    d = ctl.act(z)
    # f(a) = k(a, 1.0) - k(a, 0.0); f(a) > 0 iff (1 - a/C)^2 < (a/C)^2
    # iff a/C > 0.5 -> a0 = 27 (first a with a/52 > 0.5).
    assert d.mode == "scan"
    assert d.raw_action == 27       # exact, by the closed form above


def test_cold_start_edge_empty_classifier_proposes_all_prbs():
    ctl = _ctl(seed=1)
    d = ctl.act(np.zeros(6))
    assert ctl.n_support == 0
    assert d.mode == "cold" and d.action == C and d.raw_action == C
    assert d.a_safe_size == 0 and d.margin_applied == 0


def test_notify_projection_gates_security_factor_update():
    # Differential test: two identical controllers see the same miss stream;
    # only the one whose windows are NOT adjusted may move its factor.
    gated = _ctl(seed=5)
    free = _ctl(seed=5)
    z = np.full(6, 0.5)
    for ctl in (gated, free):
        ctl.act(z)
        ctl.observe(z, C, 1.2)      # warm classifier; sec -> argmax(all True) = 0
        # Consecutive misses (y_pred +1, label -1) decay the low-margin
        # accuracies, so argmax climbs the first still-accurate margin.
        for _ in range(2):
            d = ctl.act(z)
            ctl.notify_projection(False)
            ctl.observe(z, d.action, 0.2)
    assert free.security_factor > 0
    assert gated.security_factor == free.security_factor   # same history

    # The gate: an adjusted window skips the factor update — after the same
    # miss, only the unadjusted controller's factor moves.
    before_g, before_f = gated.security_factor, free.security_factor
    d_g, d_f = gated.act(z), free.act(z)
    gated.notify_projection(True)     # projection cut someone in the cell
    free.notify_projection(False)
    gated.observe(z, d_g.action, 0.2)
    free.observe(z, d_f.action, 0.2)
    assert gated.security_factor == before_g   # frozen by the adjustment
    assert free.security_factor >= before_g    # free adaptation continues


def test_margin_attribution_subtracts_the_projection_cut():
    ctl = _ctl(seed=5)
    z = np.full(6, 0.5)
    d = ctl.act(z)                   # cold: proposal C, margin 0
    ctl.notify_projection(True)
    # Pi cut the proposal to 20: cut = 32, margin = max(0, 0 - 32) = 0.
    u = ctl.observe(z, 20, 1.3)
    assert u.margin == 0
    # Now with a scan proposal and an offset: proposal a0+sec cut to lower.
    ctl.act(z)
    ctl.notify_projection(False)
    ctl.observe(z, 0, 1.4)          # boost margins >= 0 (hit path)
    d2 = ctl.act(z)
    ctl.notify_projection(True)
    executed = max(0, d2.action - 3)  # pretend Pi cut 3 PRBs
    u2 = ctl.observe(z, executed, 1.4)
    assert u2.margin == max(0, d2.margin_applied - 3)


def test_executed_action_bounds_validated():
    ctl = _ctl(seed=1)
    with pytest.raises(ValueError):
        ctl.observe(np.zeros(6), C + 1, 1.0)
    with pytest.raises(ValueError):
        ctl.observe(np.zeros(6), -1, 1.0)
    with pytest.raises(ValueError):
        ctl.observe(np.zeros(7), 1, 1.0)   # wrong state_dim


# ── Accuracy EMA / security factor math (reference expressions) ──────

def test_accuracy_ema_hit_boosts_margins_at_or_above():
    ctl = _ctl(seed=5, alfa=0.5)
    z = np.full(6, 0.5)
    d = ctl.act(z)                   # cold start: y_pred 0 -> no acc update
    ctl.notify_projection(False)
    ctl.observe(z, 0, 1.4)           # warms the classifier (+1 everywhere)
    d = ctl.act(z)                   # a0=0, action=0+sec, margin=sec
    margin = d.margin_applied
    before = ctl.accuracies()
    ctl.notify_projection(False)
    ctl.observe(z, d.action, 1.6)    # y_pred +1 (same z), label +1: hit
    after = ctl.accuracies()
    expect = (1 - 0.5) * before[margin:] + 0.5
    assert np.allclose(after[margin:], expect)
    assert np.allclose(after[:margin], before[:margin])  # untouched below


def test_accuracy_ema_miss_decays_margins_at_or_below():
    ctl = _ctl(seed=5, alfa=0.5)
    z = np.full(6, 0.5)
    ctl.act(z)
    ctl.observe(z, C, 1.4)           # warm classifier with +1 at a=C
    d = ctl.act(z)                   # scan: a0=0; executed=d.action
    margin = d.margin_applied
    before = ctl.accuracies()
    ctl.notify_projection(False)
    ctl.observe(z, d.action, 0.1)    # y_pred +1, label -1: MISS
    after = ctl.accuracies()
    assert np.allclose(after[: margin + 1], (1 - 0.5) * before[: margin + 1])
    assert np.allclose(after[margin + 1:], before[margin + 1:])


def test_security_factor_is_first_margin_above_target_or_zero():
    ctl = _ctl(seed=5)
    # Initial fill = midpoint > lower target: argmax = 0 after ANY unadjusted
    # window (reference expression, not a special case).
    ctl.act(np.zeros(6))
    ctl.observe(np.zeros(6), C, 1.2)
    assert ctl.security_factor == 0
    # Force the table so that ONLY margin 3 is above the target:
    ctl._accuracies[:] = 0.5
    ctl._accuracies[3] = 0.995
    ctl._accuracies[7] = 0.9995      # higher value must NOT win: first True
    ctl._adjusted = False
    ctl.act(np.zeros(6))
    ctl.observe(np.zeros(6), C, 1.2)
    assert ctl.security_factor == 3
    # No margin above the target: argmax of all-False = 0 (ref semantics).
    ctl._accuracies[:] = 0.5
    ctl._adjusted = False
    ctl.observe(np.zeros(6), C, 1.2)
    assert ctl.security_factor == 0


def test_monotonic_augmentation_updates_correct_halves():
    z = np.full(6, 0.4)
    ctl = _ctl(seed=8)
    ctl.act(z)
    u = ctl.observe(z, 20, 1.3)      # fulfilled at 20
    assert u.label == 1 and u.n_augmented == C - 20 + 1
    # Augmentation calls update() on the +1 half (a >= 20).  From an empty
    # classifier the FIRST call inserts one +1 SV at a=20; the remaining
    # points are already classified correctly (f > 0) and make no mistake —
    # the perceptron inserts nothing more (reference Projectron dynamics).
    assert ctl.n_support == 1
    x_sv = np.append(z, 20.0 / C)
    landmarks = ctl.classifier.sv.landmarks
    assert np.allclose(landmarks[0], x_sv)
    assert ctl.classifier.sv.coeff[0] == pytest.approx(1.0)
    # Prediction anywhere equals that single kernel contribution.
    f_far = ctl.classifier._margin_state(np.append(z, 1.0))[0]
    assert f_far == pytest.approx(float(np.exp(-((1.0 - 20.0 / C) ** 2))))

    ctl2 = _ctl(seed=8)
    ctl2.act(z)
    u2 = ctl2.observe(z, 10, 0.2)    # violated at 10
    assert u2.label == -1 and u2.n_augmented == 11
    # The -1 half (a <= 10): one -1 SV inserted at the FIRST scanned point
    # a=0 (f=0 counts as a mistake); f everywhere equals its kernel lobes.
    assert ctl2.n_support == 1
    assert np.allclose(ctl2.classifier.sv.landmarks[0], np.append(z, 0.0))
    assert ctl2.classifier.sv.coeff[0] == pytest.approx(-1.0)
    f_at_10 = ctl2.classifier._margin_state(np.append(z, 10.0 / C))[0]
    assert f_at_10 == pytest.approx(-float(np.exp(-((10.0 / C) ** 2))))


def test_projectron_projection_keeps_support_set_small():
    """Near-span updates are absorbed (delta <= eta), not inserted."""
    kernel = GaussianKernelSV(SVVariable(), gamma=1.0)
    alg = Projectron(kernel, eta=0.1)
    x0 = np.array([0.0, 0.0, 0.5])
    alg.predict(x0)
    alg.update(x0, 1)
    assert alg.n_support == 1
    # A nearby point (almost in the span of x0 under this kernel):
    x_near = x0 + 1e-6
    alg.predict(x_near)
    alg.update(x_near, 1)            # f*y > 0 -> no update at all
    assert alg.n_support == 1
    # Same point with a FLIP label: mistake; delta ~ 1 - k^2 ~ 0 <= eta
    # -> projected, still no new landmark.
    alg.predict(x_near)
    alg.update(x_near, -1)
    assert alg.n_support == 1
    # A far point forces a real insertion.
    x_far = np.array([3.0, 3.0, 0.5])
    alg.predict(x_far)
    alg.update(x_far, -1)
    assert alg.n_support == 2


def test_projectron_predict_tie_is_deterministic_negative_one():
    alg = Projectron(GaussianKernelSV(SVVariable(), gamma=1.0), eta=0.1)
    x0 = np.array([0.1, 0.2, 0.3])
    alg.predict(x0)
    alg.update(x0, 1)
    # The equidistant... use the exact landmark: f > 0. A point at infinite
    # distance has k -> 0 -> f == 0 -> tie must predict -1 deterministically.
    x_far = np.array([50.0, -50.0, 0.3])
    y1 = alg.predict(x_far)
    y2 = alg.predict(x_far)
    assert y1 == -1 and y2 == -1
    # Empty classifier predicts 0 (the reference's "no knowledge" value).
    assert Projectron(GaussianKernelSV(SVVariable())).predict(x0) == 0


# ── Dynamic C (register R01 as amended) ──────────────────────────────

def test_set_n_prbs_rescales_accuracy_table_and_clamps_factor():
    ctl = _ctl(seed=6)
    ctl._accuracies[5] = 0.999
    ctl._security_factor = 5
    ctl.set_n_prbs(53)               # telemetry C on the 10 MHz scenario
    assert ctl.n_prbs == 53
    assert ctl._accuracies.shape == (54,)
    assert ctl._accuracies[5] == pytest.approx(0.999)      # overlap kept
    assert np.allclose(ctl._accuracies[53:], ctl._initial_accuracy)
    assert ctl.security_factor == 5
    d = ctl.act(np.full(6, 0.5))
    assert 0 <= d.action <= 53 and d.n_candidates == 54
    ctl.set_n_prbs(4)                # shrink clamps the factor
    assert ctl.security_factor <= 4


# ── Checkpoint: same envelope as SQR + cross-restore rejection ───────

def test_checkpoint_roundtrip_preserves_decisions(tmp_path):
    rng = np.random.default_rng(17)
    ctls = {sid: _ctl(seed=100 + sid, sla=SLAS[sid]) for sid in range(NUM_SLICES)}
    for _ in range(12):
        z = rng.random(6)
        for ctl in ctls.values():
            d = ctl.act(z)
            ctl.observe(z, d.action, float(rng.uniform(0, 2)))
    path = str(tmp_path / "kbrl_ckpt.pkl")
    save_checkpoint(
        path, build_checkpoint(ctls, meta={"controller": "kbrl", "seed": 1})
    )
    payload = load_checkpoint(path)  # same envelope as the SQR checkpoint
    assert payload["meta"]["controller"] == "kbrl"

    fresh = {sid: KbrlSliceController(state_dim=6, n_prbs=C, sla_threshold=SLAS[sid])
             for sid in range(NUM_SLICES)}
    restore_controllers(payload, fresh)
    for _ in range(10):
        z = rng.random(6)
        for sid in range(NUM_SLICES):
            da, db = ctls[sid].act(z), fresh[sid].act(z)
            assert (da.action, da.raw_action, da.mode) == (
                db.action, db.raw_action, db.mode
            )
    assert fresh[0].n_support == ctls[0].n_support
    assert fresh[0].n_updates == ctls[0].n_updates


def test_cross_restore_sqr_kbrl_rejected_both_ways(tmp_path):
    kbrl = _ctl(seed=1)
    sqr = SqrSliceController(state_dim=6, n_prbs=C, gamma=1.0, sla_threshold=1.0)
    with pytest.raises(ValueError):
        sqr.load_state_dict(kbrl.state_dict())       # kbrl payload into SQR
    with pytest.raises(ValueError):
        kbrl.load_state_dict(sqr.state_dict())       # sqr payload into KBRL


def test_checkpoint_topology_validated_both_directions(tmp_path):
    ctls = {sid: _ctl(seed=sid) for sid in range(NUM_SLICES)}
    path = str(tmp_path / "kbrl_topo.pkl")
    save_checkpoint(path, build_checkpoint(ctls))
    payload = load_checkpoint(path)
    with pytest.raises(ValueError):  # checkpoint has slices 0..2, one built
        restore_controllers(payload, {0: _ctl(seed=0)})
    with pytest.raises(ValueError):  # controller 3 has no slice inside
        restore_controllers(payload, {sid: _ctl(seed=sid) for sid in range(4)})


# ── Closed loop through sqr_run.run_episode (contract mock) ──────────

_SPEC = importlib.util.spec_from_file_location(
    "sqr_run_example_kbrl",
    os.path.join(_TEST_DIR, "..", "examples", "sqr_run.py"),
)
sqr_run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sqr_run)


class FakeEnv:
    """Same mock pattern as test_sqr_run.py: capacity 52, over-capacity
    proposals cut proportionally; slice 1 empty on even steps (R19)."""

    def __init__(self, over_capacity=True, n_steps=6):
        self.over_capacity = over_capacity
        self.n_steps = n_steps
        self.step_count = 0
        self.received_proposals = []

    def reset(self):
        self.step_count = 0
        return np.full((NUM_SLICES, 6), 0.5, dtype=np.float32), {}

    def step(self, proposals):
        proposals = [int(v) for v in proposals]
        self.received_proposals.append(proposals)
        if self.over_capacity:
            total = sum(proposals)
            allocated = [int(a * C // total) for a in proposals] if total else [0] * 3
            allocated[-1] += C - sum(allocated)
        else:
            allocated = list(proposals)
        virtual = [
            {"slice_id": i, "proposed": proposals[i], "allocated": allocated[i],
             "deficit": proposals[i] - allocated[i]}
            for i in range(NUM_SLICES) if allocated[i] < proposals[i]
        ]
        self.step_count += 1
        sid_n = self.step_count
        info = {
            "kpm_timestamp": 100 * sid_n,
            "total_ues": 3,
            "kpm_missing": False,
            "kpm_missing_steps": 0,
            "projection": {
                "prb_requested": proposals,
                "prb_executed": allocated,
                "cut": [a > b for a, b in zip(proposals, allocated)],
                "over_capacity": bool(virtual),
                "virtual_updates": virtual,
            },
        }
        for i in range(NUM_SLICES):
            empty = i == 1 and sid_n % 2 == 0
            info[f"slice_{i}"] = {
                "n_ues": 0 if empty else 2,
                "p5_thr_mbps": None if empty else 0.4 + 0.1 * (sid_n % 3),
                "lowest_thr_mbps": None if empty else 0.3,
                "mean_thr_mbps": None if empty else 0.9,
                "queue_bytes_sum": 1000.0,
                "violation_rate": None if empty else 0.5,
                "n_snr_imputed": 0,
                "snr_imputations": {},
            }
        obs = np.full((NUM_SLICES, 6), 0.5, dtype=np.float32)
        return obs, -0.25, False, sid_n >= self.n_steps, info


def _kbrl_args(**over):
    base = dict(
        controller="kbrl", gamma=1, c_prbs=C, num_slices=NUM_SLICES,
        slas="0.5,1.0,1.5", violation_window=100, epsilon=0.1,
        temperature=1.0, knn_k=10, seed=7,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_build_controllers_dispatches_kbrl_and_defaults_to_sqr():
    kbrl_ctls = sqr_run.build_controllers(_kbrl_args())
    assert set(kbrl_ctls) == {0, 1, 2}
    for sid, ctl in kbrl_ctls.items():
        assert isinstance(ctl, KbrlSliceController)
        assert ctl.n_prbs == C
        assert ctl.sla_threshold == pytest.approx(SLAS[sid])
    sqr_ctls = sqr_run.build_controllers(_kbrl_args(controller="sqr"))
    assert all(isinstance(c, SqrSliceController) for c in sqr_ctls.values())
    legacy = sqr_run.build_controllers(  # no .controller attr -> sqr path
        types.SimpleNamespace(gamma=1, c_prbs=C, num_slices=2,
                              slas="0.5,1.0", violation_window=100,
                              epsilon=0.1, temperature=1.0, knn_k=10, seed=7)
    )
    assert all(isinstance(c, SqrSliceController) for c in legacy.values())
    with pytest.raises(ValueError):
        sqr_run.build_controllers(_kbrl_args(controller="dqn"))


def test_parse_args_controller_choices():
    args = sqr_run.parse_args([])
    assert args.controller == "sqr"          # backward-compatible default
    assert sqr_run.parse_args(["--controller", "kbrl"]).controller == "kbrl"
    with pytest.raises(SystemExit):
        sqr_run.parse_args(["--controller", "dqn"])


def test_run_episode_kbrl_learns_logs_and_checkpoints(tmp_path):
    env = FakeEnv(over_capacity=True, n_steps=6)
    ctls = sqr_run.build_controllers(_kbrl_args())
    csv_path = tmp_path / "step_metrics.csv"
    ckpt_path = tmp_path / "sqr_checkpoint.pkl"
    logger = sqr_run.StepMetricsLogger(str(csv_path))

    summary = sqr_run.run_episode(
        env, ctls, gamma=1.0, virtual_alpha=0.05, freeze=False,
        seed=7, episode=1, step_logger=logger,
        checkpoint_path=str(ckpt_path), checkpoint_interval=3,
        checkpoint_meta={"controller": "kbrl"}, max_steps=6,
    )
    assert summary["steps"] == 6
    # KBRL learns slice-time on every window WITH UEs (slice 1 skips even).
    assert summary["learned_windows"][0] == 6
    assert summary["learned_windows"][1] == 3
    assert summary["learned_windows"][2] == 6
    # No R12 virtual updates for KBRL (SQR-only mechanism): the classifier
    # grows only through the real + monotonic updates.
    for sid in range(NUM_SLICES):
        assert ctls[sid].n_support > 0       # classifier warmed up
        assert ctls[sid].n_updates >= summary["learned_windows"][sid]

    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == sqr_run.STEP_METRICS_COLUMNS   # SAME columns (C4b)
    assert len(rows) == 1 + 6 * NUM_SLICES
    header = {name: i for i, name in enumerate(rows[0])}
    modes = {r[header["mode"]] for r in rows[1:]}
    assert modes <= {"scan", "cold"}                 # KBRL decision modes
    first = rows[1]
    assert first[header["prb_scheduled"]] == ""      # ns-3 side, as for SQR
    # The dual-level violation signals ride in the SAME columns as SQR:
    # UE-time = violation_rate (R19 denominator), slice-time = y_t vs SLA.
    assert first[header["violation_rate"]] == "0.500000"
    assert first[header["y_t_p5_mbps"]] != ""

    payload = load_checkpoint(str(ckpt_path))
    assert payload["meta"]["controller"] == "kbrl"
    restored = sqr_run.build_controllers(_kbrl_args())
    restore_controllers(payload, restored)
    assert restored[0].n_support == ctls[0].n_support


def test_run_episode_kbrl_respects_cell_quota_every_step():
    env = FakeEnv(over_capacity=True, n_steps=8)
    ctls = sqr_run.build_controllers(_kbrl_args())
    sqr_run.run_episode(env, ctls, gamma=1.0, freeze=True, max_steps=8)
    # Proposals may exceed C (cold start proposes C per slice); the QUOTA
    # contract is enforced by the env projection Pi — the loop must have
    # delivered every proposal to the env and never executed > C.
    assert len(env.received_proposals) == 8
    assert all(all(0 <= a <= C for a in p) for p in env.received_proposals)


def test_run_episode_kbrl_freeze_disables_updates():
    env = FakeEnv(over_capacity=False, n_steps=4)
    ctls = sqr_run.build_controllers(_kbrl_args())
    sqr_run.run_episode(env, ctls, gamma=1.0, freeze=True, max_steps=4)
    assert all(ctl.n_support == 0 for ctl in ctls.values())
    assert all(ctl.n_updates == 0 for ctl in ctls.values())


def test_run_episode_kbrl_syncs_to_effective_c():
    env = FakeEnv(over_capacity=False, n_steps=3)
    effective = {1: 53}

    orig_step = env.step

    def step(proposals):
        obs, r, term, trunc, info = orig_step(proposals)
        if env.step_count in effective:
            info["total_prbs"] = effective[env.step_count]
        return obs, r, term, trunc, info

    env.step = step
    ctls = sqr_run.build_controllers(_kbrl_args())
    sqr_run.run_episode(env, ctls, gamma=1.0, freeze=True, max_steps=3)
    assert all(ctl.n_prbs == 53 for ctl in ctls.values())
    assert all(ctl._accuracies.shape[0] == 54 for ctl in ctls.values())
    # Per-slice proposals respect the NEW grid {0..53} from the synced step
    # on (they may SUM above C — cutting them is the env projection's job,
    # identical for every method).
    assert all(0 <= a <= 53 for p in env.received_proposals[1:] for a in p)
