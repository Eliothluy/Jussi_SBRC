"""Tests for the "prop" demand-proportional sanity controller (Rodada 2,
Etapa 2.2) — no ns-3.

Covers ``nsoran.sqr.static_controllers.PropDemandController`` (register
R39): hand-checked demand-proportional split with demand only in eMBB,
service floors guaranteed (and decay-to-floor without demand), Σa <= C in
every configuration, demand capping at the service target, determinism,
dynamic-C sync + checkpoint round-trip, and the integration with the
shared run loop of ``examples/sqr_run.py`` (mode column, learned = 0).
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_TEST_DIR, "..", "src")
sys.path.insert(0, _SRC)

_SPEC = importlib.util.spec_from_file_location(
    "sqr_run_example",
    os.path.join(_TEST_DIR, "..", "examples", "sqr_run.py"),
)
sqr_run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sqr_run)

from nsoran.sqr.checkpoint import build_checkpoint, restore_controllers
from nsoran.sqr.sqr_kpis import UeKpmRow, SqrKpmWindow
from nsoran.sqr.static_controllers import (
    MODE_PROP,
    PROP_CAPACITY_PER_RBG_MBPS,
    PROP_EMBB_DEMAND_MARGIN,
    PROP_EMBB_SLA_MBPS,
    PROP_FLOOR_FRACS,
    PROP_SLA_TARGETS,
    PropDemandController,
)

NUM_SLICES = 3


# ---------------------------------------------------------------------------
# Helpers (parsed-window stubs — the controller reads SqrKpmWindow.rows only)
# ---------------------------------------------------------------------------


def _row(
    imsi: int, slice_id: int, thr_mbps: float, queue_bytes: float = 0.0
) -> UeKpmRow:
    return UeKpmRow(
        timestamp=100,
        ue_imsi=imsi,
        slice_id=slice_id,
        d_tx_bytes=0.0,
        d_rx_bytes=0.0,
        plr=0.0,
        resource_share_pct=0.0,
        d_lost_packets=0.0,
        throughput_mbps=thr_mbps,
        buffer_bytes=0.0,
        window_throughput_mbps=thr_mbps,
        snr_db=15.0,
        snr_was_imputed=False,
        snr_imputation=None,
        queue_bytes=queue_bytes,
    )


def _window(rows: list[UeKpmRow]) -> SqrKpmWindow:
    return SqrKpmWindow(
        timestamp=100,
        slices={},
        total_ues=len(rows),
        cell_snr_median=None,
        rows=tuple(rows),
    )


def _floor_split(c: int) -> np.ndarray:
    """Expected no-demand split: rounded floors (largest remainder)."""
    ctl = PropDemandController(0, c, NUM_SLICES)
    return ctl.prop_split(None)


# ---------------------------------------------------------------------------
# Split: demand only in eMBB (hand-checked), floors, Σa <= C
# ---------------------------------------------------------------------------


def test_prop_demand_only_embb_hand_checked():
    """C=53, 1 eMBB UE saturated (capped at the 5.25 = 1.5×3.5 demand
    target): all the 65% remainder goes to eMBB (excess = 5.25 − 2.65·0.55
    > 0; URLLC/mMTC have no UE) -> quotas (37.1, 13.25, 2.65), largest-
    remainder rounding -> (37, 13, 3)."""
    c = 53
    ctl = PropDemandController(0, c, NUM_SLICES)
    window = _window([_row(1, 0, 100.0)])  # greedy UE: offered capped at 5.25
    split = ctl.prop_split(window)
    assert split.tolist() == [37, 13, 3]
    assert int(split.sum()) == c
    # the per-slice act() views agree with the shared split ...
    ctls = [
        PropDemandController(sid, c, NUM_SLICES, window_provider=lambda w=window: w)
        for sid in range(NUM_SLICES)
    ]
    assert [ctl.act(np.zeros(6)).action for ctl in ctls] == split.tolist()
    # ... and a controller without a provider stays at its floor
    assert (
        PropDemandController(0, c, NUM_SLICES).act(np.zeros(6)).action
        == _floor_split(c)[0]
    )


def test_floors_guaranteed_without_demand():
    """No window (or window without UEs): every slice sits at its floor;
    the integer floor never undershoots floor(frac_i * C)."""
    for c in (53, 52, 30, 9):
        split = _floor_split(c)
        assert int(split.sum()) <= c
        for sid in range(NUM_SLICES):
            floor_int = int(PROP_FLOOR_FRACS[sid] * c)
            assert split[sid] >= floor_int, (c, sid, split)
        # window whose rows exist but belong to no slice with demand: same
        ctl = PropDemandController(0, c, NUM_SLICES)
        empty = _window([])
        assert ctl.prop_split(empty).tolist() == split.tolist()


def test_sum_le_c_and_floor_property_over_patterns():
    """Property: for every demand pattern and C, Σ split <= C and every
    slice keeps at least floor(frac_i·C) PRBs."""
    patterns = [
        [],
        [_row(1, 0, 100.0)],
        [_row(1, 0, 100.0), _row(2, 0, 100.0)],
        [_row(3, 1, 0.256)],
        [_row(4, 2, 0.0016)],
        [_row(1, 0, 100.0), _row(3, 1, 0.256), _row(4, 2, 0.0016)],
        [_row(*p) for p in ((5, 1, 0.256), (6, 1, 0.256), (7, 1, 0.256))],
        [_row(8, 0, 0.0)],  # a UE with zero throughput carries no demand
    ]
    for c in (53, 52, 20, 7):
        for rows in patterns:
            ctl = PropDemandController(0, c, NUM_SLICES)
            split = ctl.prop_split(_window(rows))
            assert int(split.sum()) <= c, (c, rows, split)
            assert np.all(split >= 0) and split.dtype == np.int64
            for sid in range(NUM_SLICES):
                assert split[sid] >= int(PROP_FLOOR_FRACS[sid] * c)


def test_excess_demand_is_proportional_across_slices():
    """Two slices with excess demand share the 65% pool proportionally;
    the slice without excess stays at its floor (freed capacity flows to
    the demanding slices only)."""
    c = 53
    ctl = PropDemandController(0, c, NUM_SLICES)
    # 2 eMBB UEs capped at 3.5 each (d0 = 7) and 3 URLLC UEs at 0.256
    # (d1 = 0.768 < floor·cap -> NO excess: URLLC stays at its floor).
    window = _window(
        [_row(1, 0, 100.0), _row(2, 0, 100.0), _row(3, 1, 0.256)]
    )
    split = ctl.prop_split(window)
    assert int(split.sum()) == c
    assert split[0] > split[1] > split[2]   # demand flows to eMBB only
    # slices without excess keep their FLOAT floor quota; only the integer
    # leftover unit migrates between slices with the (equal) fractional part
    assert ctl.quota_vector(window)[1] == pytest.approx(ctl.quota_vector(None)[1])
    assert ctl.quota_vector(window)[2] == pytest.approx(ctl.quota_vector(None)[2])
    assert abs(split[1] - _floor_split(c)[1]) <= 1   # URLLC at its floor ±1u
    assert abs(split[2] - _floor_split(c)[2]) <= 1   # mMTC at its floor ±1u
    # with enough URLLC UEs its excess is positive: it takes share from the
    # pool and eMBB's share drops relative to the URLLC-less case
    many_urllc = _window([_row(10 + k, 1, 0.256) for k in range(40)])
    split2 = ctl.prop_split(many_urllc)
    assert split2[1] > split[1]             # URLLC grows above its floor
    assert split2[0] < split[0]             # ... at eMBB's expense
    assert int(split2.sum()) == c


def test_demand_capped_at_service_target():
    """demand_i = Σ min(offered, target): eMBB capped at 5.25 = 1.5×3.5
    (R36×margin, v2 calibration), URLLC and mMTC weights far above their
    offered rates (never bind, R39).  A BACKLOGGED UE (queueBytes > 0)
    offers at least its service target (greedy R04: full buffer) — its
    demand saturates at the target even when the floor serves less; an
    empty-queue UE offers what it was served."""
    ctl = PropDemandController(0, 53, NUM_SLICES)
    window = _window(
        [
            _row(1, 0, 100.0), _row(2, 0, 2.0),   # 5.25 + 2.0 = 7.25
            _row(3, 1, 0.256),                     # 0.256 (< 1.0)
            _row(4, 2, 0.0016),                    # 0.0016 (< 0.05)
        ]
    )
    d = ctl.demand_vector(window)
    assert d.tolist() == pytest.approx([7.25, 0.256, 0.0016])
    # backlogged eMBB UE served only 1.0 by a tiny floor: still offers 5.25
    backlogged = _window([_row(5, 0, 1.0, queue_bytes=500_000.0)])
    assert ctl.demand_vector(backlogged).tolist() == pytest.approx(
        [5.25, 0.0, 0.0]
    )
    # the real URLLC profile leaves small queue residue (observed ~69 B in
    # the seed-41 run): demand = target 1.0, still below its floor coverage
    urllc_residue = _window([_row(6, 1, 0.48, queue_bytes=69.0)])
    assert ctl.demand_vector(urllc_residue).tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert PROP_SLA_TARGETS == {0: 5.25, 1: 1.0, 2: 0.05}
    assert PROP_EMBB_SLA_MBPS == 3.5 and PROP_EMBB_DEMAND_MARGIN == 1.5
    assert PROP_CAPACITY_PER_RBG_MBPS == 0.55


def test_empty_window_decays_to_floor():
    """A slice with UEs in one window and none in the next decays back to
    its floor, freeing the remainder to the others (ask: "a cota dela decai
    ao piso")."""
    c = 53
    busy = _window([_row(1, 0, 100.0)])
    empty = _window([])
    busy_expected = [
        PropDemandController(
            sid, c, NUM_SLICES, window_provider=lambda w=busy: w
        ).act(np.zeros(6)).action
        for sid in range(NUM_SLICES)
    ]
    assert busy_expected != _floor_split(c).tolist()  # demand moved the split

    feed = {"w": busy, "calls": 0}

    def provider():
        # all 3 controllers of a step must see the SAME window (the run loop
        # calls act() once per slice against one closed window)
        w = busy if feed["calls"] // NUM_SLICES == 0 else empty
        feed["calls"] += 1
        return w

    ctls = {
        sid: PropDemandController(
            sid, c, NUM_SLICES, window_provider=provider
        )
        for sid in range(NUM_SLICES)
    }
    first = [ctls[sid].act(np.zeros(6)).action for sid in range(NUM_SLICES)]
    second = [ctls[sid].act(np.zeros(6)).action for sid in range(NUM_SLICES)]
    assert first == busy_expected                  # busy window: demand split
    assert second == _floor_split(c).tolist()      # decayed to floors
    assert first != second


# ---------------------------------------------------------------------------
# Determinism, decision shape, validation
# ---------------------------------------------------------------------------


def test_prop_determinism_same_inputs_same_outputs():
    window = _window([_row(1, 0, 100.0), _row(2, 1, 0.256)])
    a = PropDemandController(0, 53, NUM_SLICES, window_provider=lambda: window)
    b = PropDemandController(0, 53, NUM_SLICES, window_provider=lambda: window)
    d1, d2, d3 = a.act(np.full(6, 0.5)), a.act(np.zeros(6)), b.act(np.ones(6))
    assert (d1.action, d1.raw_action, d1.mode) == (d2.action, d2.raw_action, d2.mode)
    # NaN fields (sigma/tau/margin_m) never compare equal — compare the rest
    assert (d1.action, d1.raw_action, d1.mode, d1.margin_applied) == (
        d3.action, d3.raw_action, d3.mode, d3.margin_applied
    )
    assert d1.mode == MODE_PROP == "prop"
    assert np.isnan(d1.sigma) and np.isnan(d1.tau) and np.isnan(d1.margin_m)
    assert d1.margin_applied == 0


def test_no_window_provider_defaults_to_floors():
    ctl = PropDemandController(1, 53, NUM_SLICES)
    assert ctl.demand_vector(None).tolist() == [0.0, 0.0, 0.0]
    d = ctl.act(np.zeros(6))
    assert d.action == int(_floor_split(53)[1])
    assert d.mode == MODE_PROP


def test_constructor_validation():
    with pytest.raises(ValueError):
        PropDemandController(3, 53, NUM_SLICES)  # slice_id out of range
    with pytest.raises(ValueError):
        PropDemandController(0, 0)               # C must be positive
    with pytest.raises(ValueError):
        PropDemandController(0, 53, NUM_SLICES, floor_fracs={0: 0.05})  # gap
    with pytest.raises(ValueError):
        PropDemandController(                   # floors above the cell
            0, 53, NUM_SLICES, floor_fracs={0: 0.5, 1: 0.4, 2: 0.2}
        )
    with pytest.raises(ValueError):
        PropDemandController(0, 53, NUM_SLICES, capacity_per_rbg=0.0)
    with pytest.raises(ValueError):
        PropDemandController(0, 53, NUM_SLICES, capacity_per_rbg=float("nan"))
    with pytest.raises(ValueError):
        PropDemandController(0, 53).set_capacity_per_rbg(-1.0)
    with pytest.raises(ValueError):
        PropDemandController(0, 53).set_n_prbs(0)


def test_dynamic_c_sync_and_checkpoint_roundtrip():
    ctl = PropDemandController(0, 52)
    before = ctl.prop_split(None).copy()
    ctl.set_n_prbs(53)
    after = ctl.prop_split(None)
    assert int(after.sum()) <= 53 and not np.array_equal(before, after)
    payload = build_checkpoint({0: PropDemandController(0, 52), 1: ctl})
    fresh = {0: PropDemandController(0, 52), 1: PropDemandController(1, 52)}
    restore_controllers(payload, fresh)
    assert fresh[1].n_prbs == 53
    assert fresh[1].sla_targets == {0: 5.25, 1: 1.0, 2: 0.05}
    assert fresh[1].floor_fracs[1] == pytest.approx(0.25)
    assert fresh[1].capacity_per_rbg == pytest.approx(0.55)
    with pytest.raises(ValueError):
        fresh[1].load_state_dict({"kind": "equal_split"})
    with pytest.raises(ValueError):
        fresh[1].load_state_dict({"kind": "prop_demand", "state_version": 99})


# ---------------------------------------------------------------------------
# Driver wiring (build_controllers) + shared run loop
# ---------------------------------------------------------------------------


def test_sqr_run_accepts_prop_choice():
    args = sqr_run.parse_args(["--controller", "prop", "--simTime", "60"])
    controllers = sqr_run.build_controllers(args)
    assert all(isinstance(c, PropDemandController) for c in controllers.values())
    assert all(c._window_provider is None for c in controllers.values())
    # pass-through flags parse (absent by default -> not sent to the C++)
    args2 = sqr_run.parse_args(
        ["--controller", "prop", "--sliceLambdas", "1,1,1",
         "--sliceApps", "greedy,urllc,mmtc", "--cellRadius", "300"]
    )
    assert args2.slice_lambdas == "1,1,1"
    assert args2.slice_apps == "greedy,urllc,mmtc"
    assert args2.cell_radius == 300.0


def test_build_controllers_wires_window_provider_from_env_builder():
    class _Builder:
        def __init__(self):
            self.last_window = None

    class _Env:
        def __init__(self):
            self.builder = _Builder()

    env = _Env()
    args = sqr_run.parse_args(["--controller", "prop"])
    controllers = sqr_run.build_controllers(args, env)
    floor = _floor_split(52)
    assert [c.act(np.zeros(6)).action for c in controllers.values()] == floor.tolist()
    env.builder.last_window = _window([_row(1, 0, 100.0)])
    actions = [c.act(np.zeros(6)).action for c in controllers.values()]
    assert actions != floor.tolist()               # demand moved the split
    assert int(sum(actions)) <= 52
    assert actions[0] > floor[0]                   # eMBB takes the remainder


class _FakeEnv:
    """Minimal mock of SqrEnv for run_episode (no projection cuts)."""

    def __init__(self, n_steps=2):
        self.n_steps = n_steps
        self.step_count = 0
        self.received = []

    def reset(self):
        self.step_count = 0
        return np.full((NUM_SLICES, 6), 0.5, dtype=np.float32), {}

    def step(self, proposals):
        proposals = [int(v) for v in proposals]
        self.received.append(proposals)
        self.step_count += 1
        info = {
            "total_prbs": 53,          # unchanged -> no dynamic-C sync
            "kpm_missing": False,
            "kpm_missing_steps": 0,
            "kpm_timestamp": 100 * self.step_count,
            "projection": {
                "prb_requested": list(proposals),
                "prb_executed": list(proposals),
                "cut": [False] * NUM_SLICES,
                "over_capacity": False,
                "virtual_updates": [],
            },
        }
        for sid in range(NUM_SLICES):
            info[f"slice_{sid}"] = {
                "n_ues": 1, "p5_thr_mbps": 0.4, "lowest_thr_mbps": 0.3,
                "mean_thr_mbps": 0.6, "queue_bytes_sum": 0.0,
                "violation_rate": 0.5, "n_snr_imputed": 0, "snr_imputations": {},
            }
        done = self.step_count >= self.n_steps
        return (
            np.full((NUM_SLICES, 6), 0.5, dtype=np.float32),
            -0.5, False, done, info,
        )


class _WindowFeed:
    """One distinct window per step (3 controllers call the provider/step)."""

    def __init__(self, windows):
        self.windows = list(windows)
        self.calls = 0

    def __call__(self):
        w = self.windows[min(self.calls // NUM_SLICES, len(self.windows) - 1)]
        self.calls += 1
        return w


def test_prop_method_through_shared_run_loop(tmp_path):
    env = _FakeEnv(n_steps=2)
    windows = [
        _window([_row(1, 0, 100.0)]),                       # step 1: eMBB demand
        _window([]),                                        # step 2: no UEs
    ]
    feed = _WindowFeed(windows)  # ONE shared feed: 3 act() calls per step
    controllers = {
        sid: PropDemandController(
            sid, 53, NUM_SLICES, window_provider=feed
        )
        for sid in range(NUM_SLICES)
    }
    log_path = tmp_path / "step_metrics.csv"
    logger = sqr_run.StepMetricsLogger(str(log_path))
    summary = sqr_run.run_episode(
        env, controllers, gamma=1.0, freeze=False, seed=41,
        step_logger=logger,
        checkpoint_path=str(tmp_path / "ckpt.pkl"), checkpoint_interval=0,
    )
    assert len(env.received) == 2
    busy, idle = (int(sum(r)) for r in env.received)
    assert busy <= 53 and idle <= 53
    assert env.received[0] != env.received[1]   # actions VARY per window
    assert env.received[1] == _floor_split(53).tolist()  # decayed to floors
    assert env.received[0][0] > env.received[1][0]       # eMBB had the demand
    df = pd.read_csv(log_path)
    assert set(df["mode"]) == {"prop"}
    assert (df["learned"] == 0).all()           # static: honest learned column
    assert (df["prb_executed"] == df["prb_requested"]).all()
    assert summary["learned_windows"] == {0: 0, 1: 0, 2: 0}
