"""Tests for nsoran.sqr.sqr_env against a MOCK of the IPC contract (no ns-3).

The ns-3 lifecycle (semaphores, subprocess) is NOT exercised here — the
integration with the real sqr-sim is validated in Fase 1/4 (PLANO §3).
What IS tested: the frozen-contract pieces that are pure computation —
spaces/defaults, action→projection→CSV tuples (Σa ≤ C, dedicatedPRB =
100·a/C, min=max, PF, no renormalization), KPM→observation with the R09
scales, empty-slice encoding, log-only reward, and the info payload.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from nsoran.ns_env import NsOranEnv
from nsoran.sqr.sqr_env import (
    DEFAULT_SLAS,
    SQR_ALGORITHM_PF,
    SQR_CONTROL_FILENAME,
    SQR_KPM_FILENAME,
    QUEUE_CAP_BYTES,
    SqrEnv,
    SqrObservationBuilder,
    read_effective_total_prbs,
)
from nsoran.sqr.sqr_kpis import SQR_KPM_COLUMNS, SNR_SENTINEL

SLAS = {0: 0.5, 1: 1.0, 2: 1.5}


def _make_env(monkeypatch, tmp_path, total_prbs=52, num_slices=3, **kw):
    """SqrEnv without the ns-3 lifecycle: setup_sim patched out (the
    sqr-sim executable only exists after Fase 1)."""
    monkeypatch.setattr(NsOranEnv, "setup_sim", lambda self: None)
    config = {
        "simTime": [600.0],
        "appStart": [0.5],
        "periodMs": [100],
        "seed": [1],
    }
    return SqrEnv(
        ns3_path="/nonexistent",
        scenario_configuration=config,
        output_folder=str(tmp_path),
        total_prbs=total_prbs,
        num_slices=num_slices,
        sla_thresholds=SLAS,
        **kw,
    )


def _write_kpm(path, rows):
    lines = [",".join(SQR_KPM_COLUMNS)]
    for row in rows:
        lines.append(",".join(str(v) for v in row))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _kpm_row(ts=100, imsi=1, sid=0, wthr=1.0, snr=20.0, queue=1000):
    return [ts, imsi, sid, 15000, 12500, 0.0, 33.3, 0, wthr, 2000, wthr, snr, queue]


# Real sqr_prb_windows.csv header (sqr-sim.cc, contract §3.5) and one row
# shaped like the C++ output (13 RBGs x 4 PRBs + 1 -> total_prb = 53).
PRB_HEADER = (
    "timestamp_ms,total_rbg,total_prb,dl_slots,prb_requested_pct_0,"
    "prb_requested_pct_1,prb_requested_pct_2,prb_executed_rbg_0,"
    "prb_executed_rbg_1,prb_executed_rbg_2,prb_scheduled_rbg_0,"
    "prb_scheduled_rbg_1,prb_scheduled_rbg_2,prb_budget_rbg_0,"
    "prb_budget_rbg_1,prb_budget_rbg_2,prb_idle_rbg"
)


def _write_prb_telemetry(path, rows):
    with open(path, "w") as f:
        f.write(PRB_HEADER + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def _make_telemetry_dir(sim_path):
    subdir = os.path.join(
        sim_path, "results_sqr_network_only", "seed=1_run=1"
    )
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, "sqr_prb_windows.csv")


# ── Spaces and frozen contract defaults ─────────────────────────────

def test_spaces_and_contract_defaults(monkeypatch, tmp_path):
    env = _make_env(monkeypatch, tmp_path)
    assert env.observation_space.shape == (3, 6)
    assert env.observation_space.dtype == np.float32
    assert list(env.observation_space.low) == [0.0] * 6 or np.all(
        env.observation_space.low == 0.0
    )
    assert np.all(env.observation_space.high == 1.0)
    # action: integer PRBs per slice in {0..C}
    assert list(env.action_space.nvec) == [53, 53, 53]
    assert env.control_file == SQR_CONTROL_FILENAME == "rslaq_actions_for_ns3.csv"
    assert SQR_KPM_FILENAME == "sqr-kpms.txt"
    assert SQR_ALGORITHM_PF == 1  # PF fixed (register R17)
    # 100 ms period (contract §2); max_steps from 600 s − 0.5 s / 0.1 s
    # (scenario_configuration is flattened to scalars by NsOranEnv).
    assert env.scenario_configuration["periodMs"] == 100
    assert env.max_steps == int((600.0 - 0.5) * 1000 / 100)
    assert env.sla_thresholds == DEFAULT_SLAS
    assert env.builder.parser.last_timestamp == 0


# ── Action -> projection -> CSV tuples (contract §3) ────────────────

def test_compute_action_feasible_passthrough(monkeypatch, tmp_path):
    env = _make_env(monkeypatch, tmp_path)
    actions = env._compute_action(np.array([10, 20, 10]))
    assert env.last_projection.total_allocated == 40
    assert not env.last_projection.cut.any()
    for sid, (sid_out, dedicated, mn, mx, algo) in enumerate(actions):
        assert sid_out == sid
        assert dedicated == pytest.approx(100.0 * [10, 20, 10][sid] / 52.0)
        assert mn == dedicated and mx == dedicated  # exact quote, §3.4
        assert algo == SQR_ALGORITHM_PF


def test_compute_action_projects_over_capacity_without_renormalizing(monkeypatch, tmp_path):
    """[40, 40, 40] on C=52 must be projected to integers summing 52 with
    dedicatedPRB = 100·a_executed/C — the percentages are NEVER re-expanded
    to a full partition (register R27: the C++ side must see Σ% = 100·52/52
    only because Σa = C here; try Σa < C next test)."""
    env = _make_env(monkeypatch, tmp_path)
    actions = env._compute_action(np.array([40, 40, 40]))
    allocated = env.last_projection.allocated
    assert int(allocated.sum()) == 52
    assert env.last_projection.total_proposed == 120
    assert len(env.last_projection.virtual_updates) == 3
    total_pct = sum(a[1] for a in actions)
    assert total_pct == pytest.approx(100.0)
    for sid, (_, dedicated, mn, mx, _) in enumerate(actions):
        assert dedicated == pytest.approx(100.0 * allocated[sid] / 52.0)
        assert mn == dedicated and mx == dedicated


def test_compute_action_under_capacity_stays_under(monkeypatch, tmp_path):
    """Σa < C: the written percentages must sum to 100·Σa/C < 100 — the
    env must NOT renormalize (idle RBGs stay idle, R27/Fig. 4)."""
    env = _make_env(monkeypatch, tmp_path)
    actions = env._compute_action(np.array([10, 10, 10]))
    total_pct = sum(a[1] for a in actions)
    assert total_pct == pytest.approx(100.0 * 30 / 52.0)
    assert total_pct < 100.0
    info = env._build_info(np.array([10, 10, 10]))
    assert info["projection"]["prb_executed"] == [10, 10, 10]
    assert info["projection"]["prb_requested"] == [10, 10, 10]


def test_compute_action_rejects_fractional_and_wrong_shape(monkeypatch, tmp_path):
    env = _make_env(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        env._compute_action(np.array([10.5, 20, 10]))
    with pytest.raises(ValueError):
        env._compute_action(np.array([10, 20]))
    with pytest.raises(ValueError):
        env._compute_action(np.array([[10], [20], [10]]))


# ── KPM -> observation (R09/R20) ────────────────────────────────────

def test_fill_datalake_builds_observation(monkeypatch, tmp_path):
    env = _make_env(monkeypatch, tmp_path)
    env.sim_path = str(tmp_path)
    _write_kpm(os.path.join(env.sim_path, SQR_KPM_FILENAME), [
        _kpm_row(ts=100, imsi=1, sid=0, wthr=0.75, snr=15.0, queue=100_000),
        _kpm_row(ts=100, imsi=2, sid=0, wthr=1.5, snr=30.0, queue=300_000),
        _kpm_row(ts=100, imsi=3, sid=1, wthr=0.5, snr=SNR_SENTINEL, queue=2_000_000),
    ])
    env._fill_datalake_usecase()
    obs = env._get_obs()
    assert obs.shape == (3, 6)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    # slice 0: lowest thr 0.75/1.5 = 0.5; n_ues 2/10; queue 400k/2MB;
    # snr mean (15+30)/2 = 22.5 -> 0.75
    assert obs[0, 0] == pytest.approx(0.5, abs=1e-6)
    assert obs[0, 1] == pytest.approx(0.2, abs=1e-6)
    assert obs[0, 2] == pytest.approx(400_000 / QUEUE_CAP_BYTES, abs=1e-6)
    assert obs[0, 3] == pytest.approx(0.75, abs=1e-6)
    # slice 1: sentinel imputed to 15 dB -> snr_norm 0.5; queue 2 MB -> 1.0
    assert obs[1, 3] == pytest.approx(0.5, abs=1e-6)
    assert obs[1, 2] == pytest.approx(1.0, abs=1e-6)
    # globals: no action yet -> prb_outros 0; total UEs 3/20
    assert np.all(obs[:, 4] == 0.0)
    assert obs[0, 5] == pytest.approx(3.0 / 20.0, abs=1e-6)
    assert env.builder.parser.last_timestamp == 100


def test_obs_prb_outros_uses_last_allocation(monkeypatch, tmp_path):
    env = _make_env(monkeypatch, tmp_path)
    env.sim_path = str(tmp_path)
    env._compute_action(np.array([10, 20, 10]))
    _write_kpm(os.path.join(env.sim_path, SQR_KPM_FILENAME), [
        _kpm_row(ts=100, imsi=1, sid=0, wthr=1.0),
    ])
    env._fill_datalake_usecase()
    obs = env._get_obs()
    # slice 0 sees (20 + 10)/52; slice 1 sees (10 + 10)/52
    assert obs[0, 4] == pytest.approx(30.0 / 52.0, abs=1e-6)
    assert obs[1, 4] == pytest.approx(20.0 / 52.0, abs=1e-6)
    assert obs[2, 4] == pytest.approx(30.0 / 52.0, abs=1e-6)


def test_obs_empty_slice_encodes_zeros(monkeypatch, tmp_path):
    env = _make_env(monkeypatch, tmp_path)
    env.sim_path = str(tmp_path)
    _write_kpm(os.path.join(env.sim_path, SQR_KPM_FILENAME), [
        _kpm_row(ts=100, imsi=1, sid=0, wthr=1.0),
    ])
    env._fill_datalake_usecase()
    obs = env._get_obs()
    # empty slice 2: local features 0 (nothing on the air), globals live.
    assert np.all(obs[2, :4] == 0.0)
    assert obs[2, 5] > 0.0
    # y_t of the empty slice is NOT an observation; it rides in info as None.
    env.num_steps = 1
    info = env._build_info(np.array([5, 5, 5]))
    assert info["slice_2"]["p5_thr_mbps"] is None
    assert info["slice_2"]["violation_rate"] is None
    assert info["slice_0"]["p5_thr_mbps"] == pytest.approx(1.0)


def test_reward_is_negative_pooled_violation(monkeypatch, tmp_path):
    env = _make_env(monkeypatch, tmp_path)
    env.sim_path = str(tmp_path)
    _write_kpm(os.path.join(env.sim_path, SQR_KPM_FILENAME), [
        _kpm_row(ts=100, imsi=1, sid=0, wthr=0.2),   # violating UE
        _kpm_row(ts=100, imsi=2, sid=0, wthr=1.0),   # ok UE
    ])
    env._fill_datalake_usecase()
    # pooled per-UE rate (R19): 1 violating UE / 2 eligible UEs
    assert env._compute_reward() == pytest.approx(-0.5)


def test_reward_zero_when_no_window_yet(monkeypatch, tmp_path):
    env = _make_env(monkeypatch, tmp_path)
    env.sim_path = str(tmp_path)  # no KPM file written
    assert env._compute_reward() == 0.0


def test_builder_standalone_matches_env_observation(tmp_path):
    """The pure builder (used by tests/mocks without ns-3) must agree with
    the env's pipeline."""
    builder = SqrObservationBuilder(total_prbs=52, sla_thresholds=SLAS)
    kpm = str(tmp_path / SQR_KPM_FILENAME)
    _write_kpm(kpm, [_kpm_row(ts=100, imsi=1, sid=0, wthr=1.2, snr=24.0)])
    w = builder.update_from_kpm(kpm)
    obs = builder.build_obs(w, allocation=np.array([10, 20, 10]))
    assert builder.pooled_violation_rate(w) == pytest.approx(0.0)
    assert obs[0, 0] == pytest.approx(1.2 / 1.5, abs=1e-6)
    z = builder.slice_states(w)[0]
    assert z.shape == (6,)


# ── Dynamic C from the ns-3 telemetry (register R01 as amended) ──────

def test_read_effective_total_prbs(tmp_path):
    sim_path = str(tmp_path)
    assert read_effective_total_prbs(sim_path) is None   # no telemetry yet
    csv_path = _make_telemetry_dir(sim_path)
    assert read_effective_total_prbs(sim_path) is None   # file not written
    _write_prb_telemetry(csv_path, [[100, 13, 53, 40] + [19.23] * 3 + [4] * 9 + [4]])
    assert read_effective_total_prbs(sim_path) == 53
    assert read_effective_total_prbs(None) is None


def test_dynamic_c_adopted_from_telemetry(monkeypatch, tmp_path):
    """The action space, projection and dedicatedPRB conversion follow the
    effective C read from sqr_prb_windows.csv (53, not the fixed 52)."""
    env = _make_env(monkeypatch, tmp_path)
    env.sim_path = str(tmp_path)
    csv_path = _make_telemetry_dir(env.sim_path)
    _write_prb_telemetry(csv_path, [
        [100, 13, 53, 40, 19.23, 19.23, 19.23] + [4] * 9 + [4],
    ])
    _write_kpm(os.path.join(env.sim_path, SQR_KPM_FILENAME), [_kpm_row(ts=100)])
    with pytest.warns(RuntimeWarning, match="C efetivo"):
        env._fill_datalake_usecase()
    assert env.total_prbs == 53
    assert env.total_prbs_source == "telemetry"
    assert list(env.action_space.nvec) == [54, 54, 54]
    assert env.builder.total_prbs == 53
    # Projection and conversion now run on the effective C (coordinator's
    # project_prbs is parameterized; the env passes the dynamic C).
    actions = env._compute_action(np.array([40, 40, 40]))
    allocated = env.last_projection.allocated
    assert int(allocated.sum()) == 53
    assert env.last_projection.total_prbs == 53
    for sid, (_, dedicated, mn, mx, _) in enumerate(actions):
        assert dedicated == pytest.approx(100.0 * allocated[sid] / 53.0)
        assert mn == dedicated and mx == dedicated
    # Idempotent: re-reading the same telemetry changes nothing.
    assert env.refresh_effective_c() is None
    assert env.total_prbs == 53
    # info carries the effective C so the run loop can sync controllers.
    env.num_steps = 1
    info = env._build_info(np.array([10, 20, 10]))
    assert info["total_prbs"] == 53
    assert info["total_prbs_source"] == "telemetry"
    # The obs feature prb_outros_norm uses the effective C too. Last
    # allocation was Pi([40,40,40], C=53) = [18,18,17] -> slice 0 sees
    # (53 - 18)/53.
    obs = env._get_obs()
    assert obs[0, 4] == pytest.approx(35.0 / 53.0, abs=1e-6)


def test_dynamic_c_ignores_malformed_or_invalid_telemetry(monkeypatch, tmp_path):
    """Torn write / missing column / non-positive C: fail-static, the env
    keeps its current C (never a crash inside the control loop)."""
    env = _make_env(monkeypatch, tmp_path)
    env.sim_path = str(tmp_path)
    csv_path = _make_telemetry_dir(env.sim_path)
    _write_kpm(os.path.join(env.sim_path, SQR_KPM_FILENAME), [_kpm_row(ts=100)])

    _write_prb_telemetry(csv_path, [[100, 13]])  # torn write (column count)
    env._fill_datalake_usecase()
    assert env.total_prbs == 52
    assert env.total_prbs_source == "config"

    with open(csv_path, "a") as f:  # complete row, but total_prb <= 0
        f.write(
            "200,13,0,40,19.23,19.23,19.23,4,4,4,4,4,4,4,4,4,4\n"
        )
    env._fill_datalake_usecase()
    assert env.total_prbs == 52

    with open(csv_path, "a") as f:  # garbage in the column
        f.write(
            "300,13,abc,40,19.23,19.23,19.23,4,4,4,4,4,4,4,4,4,4\n"
        )
    env._fill_datalake_usecase()
    assert env.total_prbs == 52


# ── Countable KPM-missing fallback (contract §5) ─────────────────────

def test_kpm_missing_is_countable_and_warns(monkeypatch, tmp_path):
    env = _make_env(monkeypatch, tmp_path)
    env.sim_path = str(tmp_path)
    env.num_steps = 3

    # Missing KPM: counts + warns + stale observation; NOT a silent return.
    with pytest.warns(RuntimeWarning, match="KPM ausente"):
        env._fill_datalake_usecase()
    assert env.kpm_missing_steps == 1
    assert env._last_kpm_missing is True
    info = env._build_info(np.array([1, 1, 1]))
    assert info["kpm_missing"] is True
    assert info["kpm_missing_steps"] == 1

    # A valid window clears the flag WITHOUT resetting the counter.
    _write_kpm(os.path.join(env.sim_path, SQR_KPM_FILENAME), [
        _kpm_row(ts=200, imsi=1, sid=0, wthr=1.0),
    ])
    env._fill_datalake_usecase()
    assert env.kpm_missing_steps == 1
    assert env._last_kpm_missing is False
    info = env._build_info(np.array([1, 1, 1]))
    assert info["kpm_missing"] is False
    assert env._get_obs()[0, 0] == pytest.approx(1.0 / 1.5, abs=1e-6)

    # A second missing window counts again (counter is cumulative).
    os.remove(os.path.join(env.sim_path, SQR_KPM_FILENAME))
    with pytest.warns(RuntimeWarning, match="KPM ausente"):
        env._fill_datalake_usecase()
    assert env.kpm_missing_steps == 2


def test_kpm_counter_resets_per_episode(monkeypatch, tmp_path):
    env = _make_env(monkeypatch, tmp_path)
    env.sim_path = str(tmp_path)
    with pytest.warns(RuntimeWarning, match="KPM ausente"):
        env._fill_datalake_usecase()
    assert env.kpm_missing_steps == 1
    # NsOranEnv.reset needs a live sim: neutralize ONLY the lifecycle bits,
    # keeping SqrEnv's own counter-reset logic under test.
    monkeypatch.setattr(NsOranEnv, "close", lambda self: None)
    monkeypatch.setattr(SqrEnv, "start_sim", lambda self: None)
    monkeypatch.setattr(
        SqrEnv, "_wait_data_availability", lambda self, timeout=10: None
    )
    monkeypatch.setattr(NsOranEnv, "_fill_datalake", lambda self: None)
    env.reset()
    assert env.kpm_missing_steps == 0
    assert env._last_kpm_missing is False
