"""Tests for the Etapa-2.4 replay instrument (Despacho II-a).

Covers (ask items 1-3):
- ``nsoran.sqr.replay_controller``: the hand-checked single-window override
  (action at T exact, base before/after), the equal base mirror of the
  scenario's initial partition (18, 17, 18) @ C=53, the prop base
  reproducing the ``PropDemandController`` sequence, determinism (same
  windows ⇒ identical decision sequences), input validation, dynamic-C sync
  and the checkpoint round-trip;
- ``examples/sqr_run.py --controller replay``: the shared run loop logs the
  override row (mode ``replay_override``, exact actions) with base rows
  around it (FakeEnv pattern of test_sqr_run.py, no ns-3);
- ``examples/sqr_oracle_select_replays.py``: on a self-consistent synthetic
  mini-oracle the plan has the 4 groups with the required counts, valid
  actions (Σ ≤ C), the G4 override grid, the G2 witness action and
  bit-a-bit determinism.
"""

import csv
import importlib.util
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_TEST_DIR, "..", "src")
sys.path.insert(0, _SRC)

_SPEC = importlib.util.spec_from_file_location(
    "sqr_run_example_replay",
    os.path.join(_TEST_DIR, "..", "examples", "sqr_run.py"),
)
sqr_run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sqr_run)

_SEL_SPEC = importlib.util.spec_from_file_location(
    "sqr_oracle_select_replays_example",
    os.path.join(_TEST_DIR, "..", "examples", "sqr_oracle_select_replays.py"),
)
sel = importlib.util.module_from_spec(_SEL_SPEC)
_SEL_SPEC.loader.exec_module(sel)

from nsoran.sqr.checkpoint import build_checkpoint, restore_controllers
from nsoran.sqr.replay_controller import (
    MODE_REPLAY_EQUAL,
    MODE_REPLAY_OVERRIDE,
    MODE_REPLAY_PROP,
    ReplayController,
    initial_equal_split,
    parse_replay_action,
)
from nsoran.sqr.sqr_kpis import SqrKpmWindow, UeKpmRow
from nsoran.sqr.static_controllers import PropDemandController

NUM_SLICES = 3
C53 = 53


# ---------------------------------------------------------------------------
# Base equal: mirror of the sqr-sim initial partition
# ---------------------------------------------------------------------------


def test_initial_equal_split_hand_checked():
    """C=53 → (18, 17, 18): the measured network-only equal-arm quota
    (Despacho I §1).  Pure 1/3 weights would give (18, 18, 17) — the mirror
    uses the scenario's exact initial weights (0.333333, 0.333333, 0.333334)
    with the C++ floor + largest-remainder rule."""
    s = initial_equal_split(53, 3)
    assert s.tolist() == [18, 17, 18]
    assert int(s.sum()) == 53
    # C=52: floors (17,17,17), extra 1, largest frac = slice 2.
    assert initial_equal_split(52, 3).tolist() == [17, 17, 18]
    # Deterministic and Σ == C on a grid of capacities.
    for c in (10, 24, 53, 104):
        s2 = initial_equal_split(c, 3)
        assert int(s2.sum()) == c
        assert s2.tolist() == initial_equal_split(c, 3).tolist()
    with pytest.raises(ValueError):
        initial_equal_split(0, 3)
    with pytest.raises(ValueError):
        initial_equal_split(53, 4)  # mirror is defined for the 3-slice layout


# ---------------------------------------------------------------------------
# Action parsing / validation
# ---------------------------------------------------------------------------


def test_parse_replay_action_valid_and_errors():
    assert parse_replay_action("40,12,1", 3, 53) == (40, 12, 1)
    assert parse_replay_action(" 0 , 53 , 0 ", 3, 53) == (0, 53, 0)
    assert parse_replay_action("17,35,1", 3, 53) == (17, 35, 1)
    with pytest.raises(ValueError):  # Σ > C
        parse_replay_action("40,12,2", 3, 53)
    with pytest.raises(ValueError):  # wrong arity
        parse_replay_action("40,12", 3, 53)
    with pytest.raises(ValueError):  # negative
        parse_replay_action("-1,27,27", 3, 53)
    with pytest.raises(ValueError):  # non-integer
        parse_replay_action("40.5,12,0.5", 3, 53)


def _make_replay_triple(base="equal", at_win=3, action=(40, 12, 1), c=C53,
                        provider=None):
    return {
        sid: ReplayController(
            slice_id=sid, n_prbs=c, num_slices=NUM_SLICES, base=base,
            replay_at_win=at_win, replay_action=action,
            window_provider=provider,
        )
        for sid in range(NUM_SLICES)
    }


# ---------------------------------------------------------------------------
# Hand-checked override semantics (before / at / after T)
# ---------------------------------------------------------------------------


def test_replay_override_hand_checked_equal_base():
    """Windows < T and > T run the base (18, 17, 18); window T runs the
    override exactly; the mode column marks the override window."""
    ctls = _make_replay_triple(base="equal", at_win=3, action=(40, 12, 1))
    z = np.zeros(6)
    seq = [
        tuple(int(ctls[sid].act(z).action) for sid in range(NUM_SLICES))
        for _ in range(5)
    ]
    assert seq == [
        (18, 17, 18),  # win 1: base
        (18, 17, 18),  # win 2: base
        (40, 12, 1),   # win 3 == T: override
        (18, 17, 18),  # win 4: back to base
        (18, 17, 18),  # win 5: base
    ]
    # Modes: replay_equal everywhere except the override window.
    ctls2 = _make_replay_triple(base="equal", at_win=2, action=(53, 0, 0))
    modes = [ctls2[0].act(z).mode for _ in range(3)]
    assert modes == [MODE_REPLAY_EQUAL, MODE_REPLAY_OVERRIDE, MODE_REPLAY_EQUAL]


def test_replay_at_win_must_be_reachable_and_valid():
    with pytest.raises(ValueError):
        _make_replay_triple(at_win=0)      # window 0 precedes any decision
    with pytest.raises(ValueError):
        _make_replay_triple(at_win=-5)
    with pytest.raises(ValueError):        # Σ override > C
        _make_replay_triple(action=(40, 12, 2))
    with pytest.raises(ValueError):        # negative component
        _make_replay_triple(action=(-1, 27, 27))
    with pytest.raises(ValueError):        # wrong number of components
        _make_replay_triple(action=(40, 13))
    with pytest.raises(ValueError):        # unknown base
        _make_replay_triple(base="sqr")


# ---------------------------------------------------------------------------
# prop base reproduces the PropDemandController sequence
# ---------------------------------------------------------------------------


def _row(imsi, slice_id, thr_mbps, queue_bytes=0.0) -> UeKpmRow:
    return UeKpmRow(
        timestamp=100, ue_imsi=imsi, slice_id=slice_id, d_tx_bytes=0.0,
        d_rx_bytes=0.0, plr=0.0, resource_share_pct=0.0, d_lost_packets=0.0,
        throughput_mbps=thr_mbps, buffer_bytes=0.0,
        window_throughput_mbps=thr_mbps, snr_db=15.0, snr_was_imputed=False,
        snr_imputation=None, queue_bytes=queue_bytes,
    )


def _window(rows) -> SqrKpmWindow:
    return SqrKpmWindow(timestamp=100, slices={}, total_ues=len(rows),
                        cell_snr_median=None, rows=tuple(rows))


def test_replay_base_prop_reproduces_prop_sequence():
    """The prop base delegates to the UNCHANGED PropDemandController: over a
    synthetic window stream the replay sequence equals the original
    controller's sequence everywhere except the single override window."""
    busy = _window([_row(1, 0, 100.0, queue_bytes=1.0), _row(2, 1, 0.256)])
    idle = _window([])
    stream = [busy, busy, idle, busy, idle, busy]

    def provider_seq(window_stream):
        # Models the real wiring (``builder.last_window``): every call
        # WITHIN a window returns the same window — the run loop calls the
        # provider once per slice per window (NUM_SLICES calls / window).
        state = {"calls": 0}

        def provider():
            k = state["calls"] // NUM_SLICES
            state["calls"] += 1
            return window_stream[min(k, len(window_stream) - 1)]
        return provider

    ctls = _make_replay_triple(base="prop", at_win=4, action=(7, 40, 6),
                               provider=provider_seq(stream))
    # Faithful wiring: ONE shared provider for the whole triple (the real
    # run wires every controller to the same ``builder.last_window``).
    orig_provider = provider_seq(list(stream))
    orig = {
        sid: PropDemandController(slice_id=sid, n_prbs=C53,
                                  num_slices=NUM_SLICES,
                                  window_provider=orig_provider)
        for sid in range(NUM_SLICES)
    }
    z = np.zeros(6)
    replay_seq, orig_seq = [], []
    for _ in range(len(stream)):
        replay_seq.append(
            tuple(int(ctls[sid].act(z).action) for sid in range(NUM_SLICES)))
        orig_seq.append(
            tuple(int(orig[sid].act(z).action) for sid in range(NUM_SLICES)))
    assert replay_seq[3] == (7, 40, 6)                    # the override
    for k in (0, 1, 2, 4, 5):
        assert replay_seq[k] == orig_seq[k]               # base reproduced
    assert any(a != b for a, b in zip(replay_seq[3], orig_seq[3]))
    # Mode marks: prop windows vs override.
    ctls_modes = _make_replay_triple(base="prop", at_win=1,
                                     action=(0, 0, 0),
                                     provider=provider_seq(stream))
    assert ctls_modes[0].act(z).mode == MODE_REPLAY_OVERRIDE
    assert ctls_modes[0].act(z).mode == MODE_REPLAY_PROP


def test_replay_determinism_same_windows_same_sequence():
    """Two executions over the same (deterministic) window stream produce
    IDENTICAL decision sequences (modes + actions) — the controller-level
    determinism of the replay (the ns-3 side determinism is the pilot's
    job: same seed ⇒ same KPM stream)."""
    busy = _window([_row(1, 0, 100.0, queue_bytes=1.0), _row(2, 1, 0.256)])
    stream = [busy, busy, None, busy, None]

    def run_triple():
        state = {"calls": 0}

        def provider():
            k = state["calls"] // NUM_SLICES
            state["calls"] += 1
            return stream[min(k, len(stream) - 1)]

        ctls = _make_replay_triple(base="prop", at_win=3, action=(10, 40, 3),
                                   provider=provider)
        z = np.zeros(6)
        out = []
        for _ in range(len(stream)):
            ds = [ctls[sid].act(z) for sid in range(NUM_SLICES)]
            out.append(tuple((d.mode, int(d.action)) for d in ds))
        return out

    assert run_triple() == run_triple()


# ---------------------------------------------------------------------------
# Dynamic C sync + checkpoint round-trip
# ---------------------------------------------------------------------------


def test_replay_set_n_prbs_follows_dynamic_c():
    ctls = _make_replay_triple(base="equal", at_win=99, action=(0, 0, 0))
    for ctl in ctls.values():
        ctl.set_n_prbs(52)
    z = np.zeros(6)
    assert tuple(int(ctls[s].act(z).action) for s in range(3)) == (17, 17, 18)
    # The prop delegate follows the sync too.
    busy = _window([_row(1, 0, 100.0, queue_bytes=1.0)])
    propt = _make_replay_triple(base="prop", at_win=99, action=(0, 0, 0),
                                provider=lambda: busy)
    for ctl in propt.values():
        ctl.set_n_prbs(52)
    ref = {
        s: PropDemandController(slice_id=s, n_prbs=52, num_slices=NUM_SLICES,
                                window_provider=lambda: busy)
        for s in range(NUM_SLICES)
    }
    assert tuple(int(propt[s].act(z).action) for s in range(3)) == \
        tuple(int(ref[s].prop_split(busy)[s]) for s in range(3))


def test_replay_checkpoint_roundtrip():
    from nsoran.sqr.replay_controller import REPLAY_STATE_VERSION

    ctls = _make_replay_triple(base="prop", at_win=7, action=(9, 40, 4))
    z = np.zeros(6)
    for ctl in ctls.values():
        ctl.act(z)
    path = os.path.join(_TEST_DIR, "_tmp_replay_checkpoint.pkl")
    try:
        from nsoran.sqr.checkpoint import save_checkpoint

        save_checkpoint(path, build_checkpoint(
            ctls, meta={"controller": "replay"}))
        fresh = _make_replay_triple(base="equal", at_win=1, action=(0, 0, 0))
        restore_controllers(load_checkpoint_file(path), fresh)
        for sid, ctl in fresh.items():
            assert ctl.base == "prop"
            assert ctl.replay_at_win == 7
            assert ctl.replay_action == (9, 40, 4)
            assert ctl._windows_seen == 1
            assert ctl.state_dict()["state_version"] == REPLAY_STATE_VERSION
    finally:
        if os.path.exists(path):
            os.remove(path)


def load_checkpoint_file(path):
    from nsoran.sqr.checkpoint import load_checkpoint

    return load_checkpoint(path)


# ---------------------------------------------------------------------------
# Shared run loop: the override row lands in step_metrics.csv
# ---------------------------------------------------------------------------


class FakeEnv:
    """Mock of SqrEnv (same pattern as test_sqr_run.py): capacity 53, no
    cuts (Σ proposals ≤ C), 6 windows."""

    def __init__(self, n_steps=6):
        self.n_steps = n_steps
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.full((NUM_SLICES, 6), 0.5, dtype=np.float32), {}

    def step(self, proposals):
        self.step_count += 1
        sid = self.step_count
        info = {
            "kpm_timestamp": 100 * sid,
            "total_prbs": C53,
            "total_prbs_source": "telemetry",
            "kpm_missing": False,
            "kpm_missing_steps": 0,
            "projection": {
                "prb_requested": list(int(v) for v in proposals),
                "prb_executed": list(int(v) for v in proposals),
                "cut": [False] * NUM_SLICES,
                "over_capacity": False,
                "virtual_updates": [],
            },
        }
        for i in range(NUM_SLICES):
            info[f"slice_{i}"] = {
                "n_ues": 2, "p5_thr_mbps": 1.0, "lowest_thr_mbps": 0.9,
                "mean_thr_mbps": 1.1, "queue_bytes_sum": 0.0,
                "violation_rate": 0.0, "n_snr_imputed": 0,
            }
        obs = np.full((NUM_SLICES, 6), 0.5, dtype=np.float32)
        return obs, -0.25, False, sid >= self.n_steps, info


def test_run_loop_logs_override_row(tmp_path):
    """run_episode + --controller replay: step T rows carry mode
    replay_override and the EXACT override actions; the steps before/after
    carry the base (18, 17, 18); prb_executed == action (no cut, Σ ≤ C)."""
    ctls = _make_replay_triple(base="equal", at_win=4, action=(40, 12, 1))
    env = FakeEnv(n_steps=6)
    log = os.path.join(str(tmp_path), "step_metrics.csv")
    sqr_run.run_episode(
        env, ctls, gamma=1.0, freeze=True, warmup_steps=0, seed=1,
        step_logger=sqr_run.StepMetricsLogger(log), max_steps=6,
    )
    with open(log) as f:
        rows = list(csv.DictReader(f))
    by_step = {}
    for r in rows:
        by_step.setdefault(int(r["step"]), {})[int(r["slice_id"])] = r
    assert sorted(by_step) == [1, 2, 3, 4, 5, 6]
    # base / base / base / OVERRIDE / base / base
    for step, expected in [(1, (18, 17, 18)), (2, (18, 17, 18)),
                           (3, (18, 17, 18)), (4, (40, 12, 1)),
                           (5, (18, 17, 18)), (6, (18, 17, 18))]:
        for sid in range(NUM_SLICES):
            r = by_step[step][sid]
            assert int(r["action"]) == expected[sid], (step, sid, r)
            assert int(r["prb_requested"]) == expected[sid]
            assert int(r["prb_executed"]) == expected[sid]
            assert r["mode"] == (
                MODE_REPLAY_OVERRIDE if step == 4 else MODE_REPLAY_EQUAL
            )


def test_build_controllers_replay_registration(tmp_path):
    """--controller replay is registered: the driver builds one validated
    ReplayController per slice; bad plans raise."""
    args = types.SimpleNamespace(
        controller="replay", replay_base="equal", replay_at_win=500,
        replay_action="40,12,1", c_prbs=53, num_slices=3, slas="3.5,3.5,3.5",
        seed=1, gamma=1,
    )
    ctls = sqr_run.build_controllers(args, env=None)
    assert set(ctls) == {0, 1, 2}
    assert all(isinstance(c, ReplayController) for c in ctls.values())
    assert ctls[1].replay_action == (40, 12, 1)
    assert ctls[1].replay_at_win == 500
    with pytest.raises(ValueError):  # Σ > C caught at build time
        sqr_run.build_controllers(
            types.SimpleNamespace(**{**vars(args), "replay_action": "54,0,0"}),
            env=None,
        )
    with pytest.raises(ValueError):  # at_win < 1 caught at build time
        sqr_run.build_controllers(
            types.SimpleNamespace(**{**vars(args), "replay_at_win": 0}),
            env=None,
        )


# ---------------------------------------------------------------------------
# Selector: synthetic self-consistent mini-oracle
# ---------------------------------------------------------------------------

LAM_A, LAM_B, LAM_C = 2.0, 5.0, 7.0


def _synthetic_oracle(n_per_run=24):
    """A self-consistent mini version of the Despacho-I artifacts: per-window
    state/KPIs with noise (deterministic seed), the classification columns
    computed with the SAME oracle definitions, and a critical-windows CSV
    (the refit cross-check must PASS on it — fixtures with inconsistent
    columns fail loudly here)."""
    rng = np.random.default_rng(20261022)
    rows = []
    for lam, n_e_grid in ((LAM_A, (2, 3, 2, 3)), (LAM_B, (2, 2, 4, 4)),
                          (LAM_C, (4, 5, 5, 4))):
        for seed in (1, 2):
            for k in range(n_per_run):
                n_e = int(n_e_grid[k % len(n_e_grid)])
                # URLLC population per load (the model looks up (n_u, n_e)):
                # λ2 -> n_u=1 (median tail > 10 ms, p25 < 10: the G1
                # false-infeasible pool); λ5/λ7 -> n_u=2 (median tail <= 10:
                # gate OPEN -> G2 avoidable / G3 capacity-infeasible pools).
                n_u = 1 if lam == LAM_A else 2
                n_m = int(rng.integers(0, 2))
                # Executed quota; λ7 VARIES a_e so its CQI-bin cells have
                # support in the HIGH-share bins too (otherwise the R42 flat
                # extrapolation borrows other loads' high yields and the
                # capacity infeasibility disappears from the grid).
                a_e = 18 if lam != LAM_C else (18, 30, 44, 53)[k % 4]
                a_u, a_m = 17, 18
                # eMBB: thr monotone in the share s with small jitter; the
                # yield coefficient per load engineers the regimes:
                #   λ2: always feasible (thr well above 3.5 at s = a_e/N_e);
                #   λ5: feasible at n_e=2, eMBB-violating at n_e=4;
                #   λ7: capacity-infeasible (yield so low that even a_e=C
                #       stays below 3.5 Mbps/UE) with the URLLC gate OPEN.
                coef = {LAM_A: 0.62, LAM_B: 0.50, LAM_C: 0.055}[lam]
                s = a_e / max(n_e, 1)
                thr = float(np.clip(coef * s + rng.normal(0, 0.05), 0.0, 12.0))
                # λ7 gets its own CQI bin (cb=0 vs cb=2 for λ2/λ5) so the
                # (cqi, share) lookup cells keep the load profiles separate.
                cqi = 14.0 if lam != LAM_C else 10.0
                # URLLC p95 strata: n_u=1 -> 10.6 ± 1.3 (median > 10, p25
                # < 10); n_u=2 -> 8.9 ± 1.0 (median <= 10, some windows > 10).
                p95_med = 10.6 if n_u == 1 else 8.9
                p95 = float(np.clip(p95_med + rng.normal(0, 1.3 if n_u == 1
                                                        else 1.0), 4.0, 25.0))
                miss = 100.0 * float(p95 > 10.0)
                pdr = 1.0
                rows.append({
                    "arm": "equal", "lam": lam, "seed": seed, "win": 100 + k,
                    "time_ms": (100 + k + 1) * 100,
                    "n_embb": n_e, "n_urllc": n_u, "n_mmtc": n_m,
                    "n_embb_backlog": n_e,
                    "thr_embb_med": thr,
                    "latency_urllc_p95_ms": p95,
                    "latency_urllc_miss_pct": miss,
                    "urllc_n_pkts": 40.0,
                    "pdr_mmtc": pdr,
                    "cqi_embb": cqi, "cqi_eff": cqi,
                    "a_e": a_e, "a_u": a_u, "a_m": a_m,
                    "e_min": -1, "urllc_gate": False,  # recomputed below
                    "num_feasible_actions": 0,
                    "avoidable_any": np.nan,
                    "min_total_deficit": np.nan,
                    "observed_embb_violation": float(thr < 3.5),
                    "observed_urllc_violation": float(p95 > 10.0),
                    "observed_mmtc_violation": np.nan,
                })
    ex = pd.DataFrame(rows)

    # Classify every synthetic window with the REAL oracle machinery and
    # fill the derived columns the selector reads.
    from nsoran.sqr.oracle.models import fit_models
    from nsoran.sqr.oracle.simplex import window_feasibility

    fit_frame = ex.copy()
    models = fit_models(fit_frame)
    jf, nfa, emin, gate, amin, av, dmin = [], [], [], [], [], [], []
    for r in ex.itertuples():
        res = window_feasibility(
            models["embb"], models["urllc"], models["mmtc"],
            n_embb=int(r.n_embb), n_urllc=int(r.n_urllc),
            n_mmtc=int(r.n_mmtc), cqi=float(r.cqi_eff))
        jf.append(res.joint_feasible)
        nfa.append(res.num_feasible_actions)
        emin.append(res.e_min)
        gate.append(res.urllc_gate)
        amin.append(res.min_total_deficit)
        v_any = (r.observed_embb_violation == 1.0
                 or r.observed_urllc_violation == 1.0)
        av.append((1.0 if (v_any and res.joint_feasible) else 0.0))
        dmin.append(res.min_total_deficit)
    ex["joint_feasible"] = jf
    ex["num_feasible_actions"] = nfa
    ex["e_min"] = emin
    ex["urllc_gate"] = gate
    ex["avoidable_any"] = av
    ex["min_total_deficit"] = dmin
    # §19 stand-in: same columns/names as oracle_windows.csv (the selector
    # renames lambda_total -> lam and time -> time_ms on load).  NO per-window
    # KPI columns here (thr/p95 live ONLY in the extra table, as in the real
    # artifacts) — otherwise the 1:1 merge would suffix-collide.
    w19 = ex[[
        "arm", "lam", "seed", "time_ms", "n_embb", "n_urllc", "n_mmtc",
        "observed_embb_violation", "observed_urllc_violation",
        "observed_mmtc_violation", "joint_feasible", "num_feasible_actions",
        "avoidable_any", "min_total_deficit",
    ]].rename(columns={"lam": "lambda_total", "time_ms": "time"}).copy()

    # Critical CSV: a witness/a* sample for the refit cross-check.
    crit_rows = []
    for r in ex.itertuples():
        res = window_feasibility(
            models["embb"], models["urllc"], models["mmtc"],
            n_embb=int(r.n_embb), n_urllc=int(r.n_urllc),
            n_mmtc=int(r.n_mmtc), cqi=float(r.cqi_eff))
        w = res.witness or (-1, -1, -1)
        crit_rows.append({
            "priority_class": "D_oracle_inviavel",
            "lam": r.lam, "seed": r.seed, "win": r.win, "arm": r.arm,
            "tag": f"A_lam{int(r.lam)}_s{r.seed}",
            "time_ms": r.time_ms,
            "n_embb": r.n_embb, "n_urllc": r.n_urllc, "n_mmtc": r.n_mmtc,
            "thr_embb_med": r.thr_embb_med,
            "latency_urllc_p95_ms": r.latency_urllc_p95_ms,
            "a_e": r.a_e, "a_u": r.a_u, "a_m": r.a_m,
            "joint_feasible": res.joint_feasible,
            "num_feasible_actions": res.num_feasible_actions,
            "e_min": res.e_min,
            "witness_a_e": w[0], "witness_a_u": w[1], "witness_a_m": w[2],
            "a_star_e": res.best_deg_action[0],
            "a_star_u": res.best_deg_action[1],
            "a_star_m": res.best_deg_action[2],
            "observed_any_violation": float(
                r.observed_embb_violation == 1.0
                or r.observed_urllc_violation == 1.0),
        })
        if len(crit_rows) >= 12:
            break
    critical = pd.DataFrame(crit_rows)
    return w19, ex, critical


def _write_mini_oracle(tmp_path, w19, ex, critical):
    oracle_dir = tmp_path / "oracle"
    (oracle_dir / "tables").mkdir(parents=True)
    w19.to_csv(oracle_dir / "oracle_windows.csv", index=False)
    # Mirror the REAL extra table: only the extra columns (never the §19
    # derived ones — otherwise the 1:1 merge would suffix-collide).
    extra_cols = [
        "arm", "lam", "seed", "win", "time_ms", "n_embb_backlog",
        "thr_embb_med", "latency_urllc_p95_ms",
        "latency_urllc_miss_pct", "urllc_n_pkts", "pdr_mmtc",
        "cqi_embb", "cqi_eff", "a_e", "a_u", "a_m", "e_min", "urllc_gate",
    ]
    ex[extra_cols].to_csv(
        oracle_dir / "tables" / "oracle_windows_extra.csv", index=False
    )
    critical.to_csv(oracle_dir / "tables" / "oracle_critical_windows.csv",
                    index=False)
    return str(oracle_dir)


def test_selector_plan_groups_counts_and_actions(tmp_path, monkeypatch):
    """On the synthetic mini-oracle (small group sizes via monkeypatch): the
    plan has all 4 groups, valid Σ ≤ C actions, the G4 grid, the G2 witness
    action and G1 executed-OK + p25-witness windows."""
    monkeypatch.setattr(sel, "GROUP_SIZES",
                        {"G1": 2, "G2": 2, "G3": 2, "G4_WINDOWS": 1})
    monkeypatch.setattr(sel, "CAPS_PER_RUN", {"G1": 1, "G2": 1, "G3": 1,
                                              "G4": 1})
    w19, ex, critical = _synthetic_oracle()
    oracle_dir = _write_mini_oracle(tmp_path, w19, ex, critical)

    plan, stats = sel.build_plan(oracle_dir)
    assert list(plan.columns) == list(sel.PLAN_COLUMNS)
    assert stats["by_group"] == {"G1": 2, "G2": 2, "G3": 2, "G4": 3}
    assert stats["windows_by_group"]["G4"] == 1
    sums = plan[["action_e", "action_u", "action_m"]].sum(axis=1)
    assert bool((sums <= sel.C_PRB).all())
    assert bool((plan[["action_e", "action_u", "action_m"]] >= 0)
                .all().all())

    merged = sel.load_merged_windows(oracle_dir)
    g1 = plan[plan.grupo == "G1"]
    assert (g1.lam == 2).all()
    models = sel.fit_models(merged)
    for r in g1.itertuples():
        row = merged[(merged.lam == 2.0) & (merged.seed == r.seed)
                     & (merged.win == r.win_t)].iloc[0]
        assert not bool(row["joint_feasible"])          # median-infeasible
        v = row[["observed_embb_violation", "observed_urllc_violation",
                 "observed_mmtc_violation"]]
        assert v.fillna(0).eq(0).all()                  # executed all-OK
        # The action IS the p25-port witness of this state (R44 recipe).
        res = sel.window_feasibility(
            models["embb"], models["urllc"], models["mmtc"],
            n_embb=int(row["n_embb"]), n_urllc=int(row["n_urllc"]),
            n_mmtc=int(row["n_mmtc"]), cqi=float(row["cqi_eff"]),
            embb_q=0.25, urllc_q=0.25)
        assert res.e_min >= 0                            # p25-witnessable
        wit = sel._witness_from_e_min(res.e_min)
        assert (r.action_e, r.action_u, r.action_m) == wit
    g2 = plan[plan.grupo == "G2"]
    assert g2.lam.between(4, 6).all()
    for r in g2.itertuples():
        row = merged[(merged.lam == r.lam) & (merged.seed == r.seed)
                     & (merged.win == r.win_t)].iloc[0]
        assert bool(row["joint_feasible"]) and row["avoidable_any"] == 1.0
        wit = sel._witness_from_e_min(int(row["e_min"]))
        assert (r.action_e, r.action_u, r.action_m) == wit
    g3 = plan[plan.grupo == "G3"]
    assert g3.lam.isin([6, 7]).all()
    for r in g3.itertuples():
        row = merged[(merged.lam == r.lam) & (merged.seed == r.seed)
                     & (merged.win == r.win_t)].iloc[0]
        assert not bool(row["joint_feasible"])
        assert bool(row["urllc_gate"])                  # capacity-driven
    g4 = plan[plan.grupo == "G4"]
    assert len(g4) == 3 and g4.win_t.nunique() == 1
    assert sorted(g4.action_u) == [13, 25, 35]
    for r in g4.itertuples():
        assert r.action_e == sel.C_PRB - r.action_u - 1
        assert r.action_m == 1


def test_selector_plan_is_deterministic(tmp_path, monkeypatch):
    """Two build_plan runs on the same inputs emit BIT-A-BIT identical
    plans (same group counts, same rows, same order)."""
    monkeypatch.setattr(sel, "GROUP_SIZES",
                        {"G1": 2, "G2": 2, "G3": 2, "G4_WINDOWS": 1})
    monkeypatch.setattr(sel, "CAPS_PER_RUN", {"G1": 1, "G2": 1, "G3": 1,
                                              "G4": 1})
    w19, ex, critical = _synthetic_oracle()
    oracle_dir = _write_mini_oracle(tmp_path, w19, ex, critical)
    p1, s1 = sel.build_plan(oracle_dir)
    p2, s2 = sel.build_plan(oracle_dir)
    pd.testing.assert_frame_equal(p1, p2)
    assert s1 == s2


def test_selector_witness_and_spread_helpers():
    """Unit checks of the R44 witness recipe and the deterministic spread
    picker (per-(lam, seed) cap)."""
    assert sel._witness_from_e_min(0) == (0, 13, 40)
    assert sel._witness_from_e_min(12) == (12, 13, 28)
    assert sel._witness_from_e_min(48) == (48, 5, 0)
    assert sel._witness_from_e_min(41) == (41, 12, 0)  # a_u clamped to C−e_min
    with pytest.raises(ValueError):
        sel._witness_from_e_min(-1)
    cand = pd.DataFrame({
        "lam": [2.0] * 4 + [4.0] * 4,
        "seed": [1, 1, 2, 2, 1, 1, 2, 2],
        "win": [10, 11, 12, 13, 14, 15, 16, 17],
        "score": [1.0, 0.5, 0.1, 0.2, 9.0, 0.3, 0.4, 8.0],
    })
    # Sort by score: 12(0.1, λ2s2), 13(0.2, λ2s2), 15(0.3, λ4s1), 16(0.4,
    # λ4s2), 11(0.5, λ2s1), 17(8.0), 14(9.0).  Cap 1 per (lam, seed):
    # 12 → (2,2); 13 skipped (same combo); 15 → (4,1); 16 → (4,2).
    picked = sel._pick_spread(cand, 3, 1, ["score"])
    assert list(picked["win"]) == [12, 15, 16]
    assert sel._pick_spread(cand, 3, 1, ["score"]).index.equals(picked.index)
    # With cap 2 the third pick fills (2,2) again (win 13).
    assert list(sel._pick_spread(cand, 3, 2, ["score"])["win"]) == [12, 13, 15]
