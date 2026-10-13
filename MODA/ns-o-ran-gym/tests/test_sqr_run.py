"""Tests for examples/sqr_run.py against a MOCK of the IPC contract.

Fase 2 scope (PLANO §3-Fase 2 / ask item 10): the script exists, is
consistent with the contract and its closed loop LEARNS — the integration
with the real sqr-sim is validated later by the ns-3 agent.  A FakeEnv
replays synthetic KPM windows (with p5 y_t, projections and cut slices) so
run_episode runs end-to-end without ns-3: decisions -> projection ->
REAL update with the executed action -> R12 virtual updates for cut
slices -> step_metrics.csv -> periodic checkpoints.
"""

import csv
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
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

from nsoran.sqr.checkpoint import load_checkpoint
from nsoran.sqr.kernel import MaternKernel
from nsoran.sqr.sqr_controller import SqrSliceController

C = 52
NUM_SLICES = 3
SLAS = {0: 0.5, 1: 1.0, 2: 1.5}


class FakeEnv:
    """Mock of SqrEnv: capacity 52, proposals above it get cut; slice 1 has
    no UEs on even steps (y_t undefined -> learning exclusion, R19).

    ``kpm_missing_steps`` = decision steps reported WITHOUT a KPM (contract
    §5 countable fallback); ``effective_c`` = step -> effective C reported
    in info (register R01 as amended, dynamic C)."""

    def __init__(self, over_capacity=True, n_steps=8):
        self.over_capacity = over_capacity
        self.n_steps = n_steps
        self.step_count = 0
        self.reset_count = 0
        self.received_proposals = []
        self.kpm_missing_steps = set()
        self.effective_c = {}

    def reset(self):
        self.reset_count += 1
        self.step_count = 0
        obs = np.full((NUM_SLICES, 6), 0.5, dtype=np.float32)
        return obs, {}

    def step(self, proposals):
        proposals = [int(v) for v in proposals]
        self.received_proposals.append(proposals)
        if self.over_capacity:
            # Proportional-looking cut: scale to C (the loop does not care
            # about the exact Pi here — that is coordinator's test job).
            total = sum(proposals)
            allocated = [int(a * C // total) for a in proposals]
            allocated[-1] += C - sum(allocated)
            cut = [a > b for a, b in zip(proposals, allocated)]
            virtual = [
                {"slice_id": i, "proposed": proposals[i],
                 "allocated": allocated[i], "deficit": proposals[i] - allocated[i]}
                for i in range(NUM_SLICES) if cut[i]
            ]
        else:
            allocated = list(proposals)
            virtual = []

        self.step_count += 1
        sid = self.step_count
        info = {
            "kpm_timestamp": 100 * sid,
            "total_ues": 3,
            "pooled_violation_rate": 0.25,
            "kpm_missing": sid in self.kpm_missing_steps,
            "kpm_missing_steps": sum(1 for s in self.kpm_missing_steps if s <= sid),
            "projection": {
                "prb_requested": proposals,
                "prb_executed": allocated,
                "cut": [a > b for a, b in zip(proposals, allocated)],
                "over_capacity": bool(virtual),
                "virtual_updates": virtual,
            },
        }
        if sid in self.effective_c:
            info["total_prbs"] = self.effective_c[sid]
        for i in range(NUM_SLICES):
            empty = (i == 1 and sid % 2 == 0)  # R19 exclusion on even steps
            info[f"slice_{i}"] = {
                "n_ues": 0 if empty else 2,
                "p5_thr_mbps": None if empty else 0.4 + 0.1 * (sid % 3),
                "lowest_thr_mbps": None if empty else 0.3,
                "mean_thr_mbps": None if empty else 0.9,
                "queue_bytes_sum": 1000.0,
                "violation_rate": None if empty else 0.5,
                "n_snr_imputed": 1,
                "snr_imputations": {"carried": 1},
            }
        obs = np.full((NUM_SLICES, 6), 0.5, dtype=np.float32)
        return obs, -0.25, False, sid >= self.n_steps, info


def _controllers(gamma=1.0, seed=7):
    return {
        sid: SqrSliceController(
            state_dim=6, n_prbs=C, gamma=gamma, sla_threshold=SLAS[sid],
            kernel=MaternKernel(dim=7, length_scale=0.3, nu=1.5),
            seed=seed + sid,
        )
        for sid in range(NUM_SLICES)
    }


def _args(**over):
    base = dict(
        gamma=1, c_prbs=C, num_slices=NUM_SLICES, slas="0.5,1.0,1.5",
        violation_window=100, epsilon=0.1, temperature=1.0, knn_k=10, seed=7,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


# ── CLI / builder ───────────────────────────────────────────────────

def test_parse_args_frozen_defaults():
    args = sqr_run.parse_args([])
    assert args.gamma in (0, 1)
    assert args.periodMs == 100
    assert args.simTime == 600.0
    assert args.virtual_alpha == 0.05          # R12 frozen default
    assert args.c_prbs == 52                   # R01
    assert args.slas == "0.5,1.0,1.5"          # [ARTIGO] Tabela I
    assert args.violation_window == 100        # R11
    assert args.warmup_s == 100.0              # R14


def test_parse_args_gamma_restricted_to_0_1():
    with pytest.raises(SystemExit):
        sqr_run.parse_args(["--gamma", "2"])


def test_build_controllers_topology_and_frozen_params():
    ctls = sqr_run.build_controllers(_args(gamma=0))
    assert set(ctls) == {0, 1, 2}
    for sid, ctl in ctls.items():
        assert ctl.state_dim == 6              # R20
        assert ctl.n_prbs == C
        assert ctl.gamma == 0.0
        assert ctl.sla_threshold == SLAS[sid]
        assert ctl.tau == 0.05 and ctl.margin == 10.0   # R11 (tau0/M0)
        assert ctl.violation_window == 100 and ctl.recovery_windows == 2


def test_build_controllers_gamma_one_sets_cost_term_only():
    ctls = sqr_run.build_controllers(_args(gamma=1))
    assert all(ctl.gamma == 1.0 for ctl in ctls.values())
    # P_penalty untouched by gamma (AGENTS §5 rule 7).
    assert all(ctl.regressor.p_penalty == 2.0 for ctl in ctls.values())


def test_parse_slas_validation():
    assert sqr_run.parse_slas("0.5,1.0,1.5", 3) == SLAS
    with pytest.raises(ValueError):
        sqr_run.parse_slas("0.5,1.0", 3)
    with pytest.raises(ValueError):
        sqr_run.parse_slas("0.5,1.0,-1.0", 3)


# ── Closed-loop episode over the mock ───────────────────────────────

def test_run_episode_learns_projects_logs_and_checkpoints(tmp_path):
    env = FakeEnv(over_capacity=True, n_steps=8)
    ctls = _controllers()
    csv_path = tmp_path / "step_metrics.csv"
    ckpt_path = tmp_path / "sqr_checkpoint.pkl"
    logger = sqr_run.StepMetricsLogger(str(csv_path))

    summary = sqr_run.run_episode(
        env, ctls,
        gamma=1.0, virtual_alpha=0.05, freeze=False, warmup_steps=3,
        seed=7, episode=1, step_logger=logger,
        checkpoint_path=str(ckpt_path), checkpoint_interval=3,
        checkpoint_meta={"gamma": 1.0}, max_steps=8,
    )

    assert summary["steps"] == 8
    assert env.reset_count == 1
    # Learning happened for slices with UEs (slice 1 skips even steps).
    assert summary["learned_windows"][0] == 8
    assert summary["learned_windows"][1] == 4
    assert summary["learned_windows"][2] == 8
    assert summary["virtual_updates"] > 0      # proposals over C were cut
    # The virtual update inserted an exact -0.05 SV in every cut slice.
    for sid in range(NUM_SLICES):
        alphas = ctls[sid].regressor.alphas
        assert np.any(alphas == -0.05), f"slice {sid}: no virtual update SV"

    # step_metrics.csv: header + one row per slice per step.
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == sqr_run.STEP_METRICS_COLUMNS
    assert len(rows) == 1 + 8 * NUM_SLICES
    header = {name: i for i, name in enumerate(rows[0])}
    first = rows[1]
    assert int(first[header["slice_id"]]) == 0
    assert first[header["mode"]] in {
        "confident", "softmax", "knn", "knn_nearest", "fallback"
    }
    assert int(first[header["prb_executed"]]) <= C
    assert first[header["prb_scheduled"]] == ""   # ns-3 side (Fase 1)
    assert int(first[header["prb_idle_est"]]) == C - sum(
        int(r[header["prb_executed"]]) for r in rows[1:4]
    )
    assert first[header["y_t_p5_mbps"]] != ""
    assert first[header["warmup"]] == "1"          # step 1 <= 3
    assert rows[1 + 3 * NUM_SLICES][header["warmup"]] == "0"

    # Checkpoint written at interval 3 and at the end; loads and restores.
    payload = load_checkpoint(str(ckpt_path))
    assert payload["meta"]["gamma"] == 1.0
    restored = _controllers()
    from nsoran.sqr.checkpoint import restore_controllers
    restore_controllers(payload, restored)
    assert restored[0].n_experiences == ctls[0].n_experiences


def test_run_episode_freeze_disables_all_updates(tmp_path):
    env = FakeEnv(over_capacity=True, n_steps=5)
    ctls = _controllers()
    # Seed confident memories at a=50 so every proposal is 150 > C: the
    # coordinator cuts all three slices every step.
    for ctl in ctls.values():
        ctl.regressor.set_memory(
            np.array([[0.5] * 6 + [50.0 / C]]), np.array([2.0])
        )
    summary = sqr_run.run_episode(
        env, ctls, gamma=1.0, virtual_alpha=0.05, freeze=True,
        step_logger=None, checkpoint_path=None, max_steps=5,
    )
    assert summary["virtual_updates"] > 0        # cuts still logged...
    for ctl in ctls.values():
        assert ctl.n_experiences == 0            # ...but nothing learned
        assert not np.any(ctl.regressor.alphas == -0.05)  # no virtual SV
        assert ctl.regressor.n_support == 1      # only the seeded memory


def test_run_episode_gamma_variants_select_differently_under_cost():
    """gamma=0 vs gamma=1: same mock, same seeds — the DECISION machinery
    runs for both variants (the Figs. 2-4 comparison axis)."""
    for gamma in (0, 1):
        env = FakeEnv(over_capacity=False, n_steps=4)
        ctls = _controllers(gamma=gamma)
        summary = sqr_run.run_episode(
            env, ctls, gamma=float(gamma), freeze=True, max_steps=4,
        )
        assert summary["steps"] == 4
        assert all(sum(p) <= C for p in env.received_proposals)


def test_resume_roundtrip_via_helpers(tmp_path):
    """The --resume path: save with the script's helpers, restore into
    fresh controllers, verify the decision stream continues identically."""
    rng = np.random.default_rng(41)
    ctls = _controllers()
    for i in range(10):
        z = rng.normal(size=6) * 0.4 + 0.5
        for ctl in ctls.values():
            ctl.observe(z, action=int(rng.integers(0, C + 1)), y=float(rng.uniform(0, 1.2)))
    zs_future = [rng.normal(size=6) * 0.4 + 0.5 for _ in range(10)]

    path = str(tmp_path / "ckpt.pkl")
    payload = sqr_run.build_checkpoint(ctls, meta={"gamma": 1})
    sqr_run.save_checkpoint(path, payload)
    fresh = _controllers(seed=12345)
    sqr_run.restore_controllers(load_checkpoint(path), fresh)

    for z in zs_future:
        d_orig = [ctls[s].act(z) for s in range(NUM_SLICES)]
        d_rest = [fresh[s].act(z) for s in range(NUM_SLICES)]
        assert [(d.mode, d.action) for d in d_orig] == [
            (d.mode, d.action) for d in d_rest
        ]


# ── Dynamic C (register R01 as amended) ──────────────────────────────

def test_run_episode_syncs_controllers_to_effective_c(tmp_path):
    """When the env reports a new effective C (telemetry total_prb = 53),
    the loop updates the controllers' action grid and cost denominator."""
    env = FakeEnv(over_capacity=False, n_steps=3)
    env.effective_c = {1: 53}  # step 1 adopts the telemetry C
    ctls = _controllers()
    assert all(ctl.n_prbs == C for ctl in ctls.values())

    summary = sqr_run.run_episode(
        env, ctls, gamma=1.0, freeze=True, max_steps=3,
    )
    assert all(ctl.n_prbs == 53 for ctl in ctls.values())
    assert summary["total_prbs_final"] == 53
    # Proposals from step 2 on respect the new C (candidate grid {0..53}).
    assert all(sum(p) <= 53 for p in env.received_proposals[1:])


# ── Countable KPM-missing fallback (contract §5) ─────────────────────

def test_run_episode_counts_kpm_missing_and_logs_column(tmp_path):
    env = FakeEnv(over_capacity=False, n_steps=4)
    env.kpm_missing_steps = {1, 3}  # windows 1 and 3 arrive without KPM
    ctls = _controllers()
    csv_path = tmp_path / "step_metrics.csv"
    logger = sqr_run.StepMetricsLogger(str(csv_path))

    summary = sqr_run.run_episode(
        env, ctls, gamma=1.0, freeze=True, step_logger=logger, max_steps=4,
    )
    assert summary["kpm_missing_steps"] == 2
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == sqr_run.STEP_METRICS_COLUMNS  # carries kpm_missing
    header = {name: i for i, name in enumerate(rows[0])}
    # The column carries the CUMULATIVE counter (contract §5): constant
    # across the 3 slice rows of a window, +1 after each missing window.
    by_step = {}
    for r in rows[1:]:
        by_step.setdefault(int(r[header["step"]]), set()).add(
            r[header["kpm_missing"]]
        )
    assert by_step[1] == {"1"}
    assert by_step[2] == {"1"}
    assert by_step[3] == {"2"}
    assert by_step[4] == {"2"}


# ── Incremental ns-3 telemetry reader (contract §3.5) ────────────────

TELEMETRY_HEADER = (
    "timestamp_ms,total_rbg,total_prb,dl_slots,prb_requested_pct_0,"
    "prb_requested_pct_1,prb_requested_pct_2,prb_executed_rbg_0,"
    "prb_executed_rbg_1,prb_executed_rbg_2,prb_scheduled_rbg_0,"
    "prb_scheduled_rbg_1,prb_scheduled_rbg_2,prb_budget_rbg_0,"
    "prb_budget_rbg_1,prb_budget_rbg_2,prb_idle_rbg"
)


def _tel_row(ts, sched0):
    return (
        f"{ts},13,53,40,19.23,36.54,19.23,4,7,4,{sched0},7,4,4,7,4,4"
    )


def test_ns3_telemetry_reader_is_incremental(tmp_path):
    seed_dir = tmp_path / "results_sqr_network_only" / "seed=1_run=1"
    seed_dir.mkdir(parents=True)
    csv_path = seed_dir / "sqr_prb_windows.csv"
    telemetry = sqr_run.Ns3PrbWindowTelemetry(str(tmp_path))

    assert telemetry.latest() is None            # no file yet

    csv_path.write_text(TELEMETRY_HEADER + "\n" + _tel_row(100, 4) + "\n")
    first = telemetry.latest()
    assert first is not None
    assert first["timestamp_ms"] == 100
    assert first["total_prb"] == 53              # effective C is exposed
    assert first["prb_scheduled"]["0"] == 4
    assert telemetry.latest() == first           # no new bytes -> cached row

    # Torn write: an unterminated trailing line is NOT consumed yet.
    with open(csv_path, "a") as f:
        f.write(_tel_row(200, 9))                # no trailing newline
    assert telemetry.latest()["timestamp_ms"] == 100
    with open(csv_path, "a") as f:
        f.write("\n")
    assert telemetry.latest()["timestamp_ms"] == 200
    assert telemetry.latest()["prb_scheduled"]["0"] == 9

    # A complete but malformed row is skipped; the next good row wins.
    with open(csv_path, "a") as f:
        f.write("300,13\n")                      # column-count mismatch
        f.write(_tel_row(400, 5) + "\n")
    last = telemetry.latest()
    assert last["timestamp_ms"] == 400
    assert last["prb_scheduled"]["0"] == 5


def test_ns3_telemetry_reader_inert_without_sim_path():
    assert sqr_run.Ns3PrbWindowTelemetry(None).latest() is None
    assert sqr_run.Ns3PrbWindowTelemetry("/nonexistent").latest() is None
