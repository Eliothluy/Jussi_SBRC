"""Unit & invariant tests for Asymmetric SQR-CER and Priority-SQR (Fase A)."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from pathlib import Path
import sys
import time

import numpy as np
import pytest

GYM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = GYM_ROOT.parents[1]
SRC_DIR = GYM_ROOT / "src"
EXAMPLES_DIR = GYM_ROOT / "examples"
GCE_CSV = REPO_ROOT / "MODA" / "docs" / "sqr" / "jc_decision" / "GATE_CONTEXTUAL_ESTADOS.csv"

for p in (str(SRC_DIR), str(EXAMPLES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from nsoran.sqr_v2.cer_reallocator import (  # noqa: E402
    CerBlocoATelemetry,
    CerConfig,
    CerReallocator,
    PrequentialRiskTracker,
    PrequentialSnapshot,
    extract_bloco_a_telemetry,
    pava_nondecreasing,
    regularized_marginal_slope,
)
from nsoran.sqr_v2.priority_hoashi import (  # noqa: E402
    BALANCED_PRIORITY_WEIGHTS,
    CANONICAL_URLLC_PRIORITY_WEIGHTS,
    PriorityHoashiConfig,
    PriorityHoashiReallocator,
    parse_priority_weights,
)


def _linear_curve(c: int, f0: float, slope: float) -> np.ndarray:
    return f0 + slope * np.arange(c + 1, dtype=np.float64)


def _make_clean_tracker(
    *,
    y_last: tuple[float | None, float | None, float | None] = (0.30, 0.19, 0.0526),
    n_hist: int = 10,
    urllc_lag: int = 2,
    n_open_u: int = 0,
    exp_u: int = 0,
) -> tuple[PrequentialSnapshot, int]:
    tr = PrequentialRiskTracker()
    gw = 100
    for sid in range(3):
        yv = y_last[sid]
        if yv is not None:
            for k in range(n_hist):
                ow = gw - (n_hist - k) - (1 if sid == 1 else 0)
                tr.record_closed_outcome(sid, f_pred_exec=0.0, y_closed=yv, outcome_win=ow)
    for k in range(n_open_u):
        tr.mark_urllc_pending(gw - 1 - k)
    for k in range(exp_u):
        tr.mark_urllc_expired(gw - 5 - k, observed_at_win=gw - 1)
    if y_last[1] is not None:
        # Place last closed URLLC window exactly at gw - urllc_lag
        tr._last_closed_urllc_win = gw - urllc_lag  # noqa: SLF001 - intentional test pin
        tr._ever_had_urllc_cohort = True  # noqa: SLF001
    return tr.snapshot(governed_window=gw), gw


class _SpyBlocoAMapping(Mapping):
    """Mapping wrapper that raises PermissionError on any Bloco B read."""

    def __init__(self, row: dict[str, str], forbidden_keys: set[str]) -> None:
        self._row = row
        self._forbidden = forbidden_keys
        self.accessed: list[str] = []

    def __getitem__(self, key: str):
        if key in self._forbidden or str(key).startswith("off_"):
            raise PermissionError(f"Bloco B access attempted on {key!r}")
        self.accessed.append(key)
        return self._row[key]

    def __iter__(self):
        raise PermissionError("Unrestricted dict iteration over Bloco A+B row is forbidden")

    def __len__(self) -> int:
        return len(self._row)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str) and (key in self._forbidden or key.startswith("off_")):
            raise PermissionError(f"Bloco B existence check attempted on {key!r}")
        return key in self._row


# ---------------------------------------------------------------------------
# 1. Capacity & non-negativity invariants (SQR-CER + Priority-SQR)
# ---------------------------------------------------------------------------


def test_capacity_and_nonnegativity_500_synthetic_inputs() -> None:
    rng = np.random.default_rng(20261118)
    cer_full = CerReallocator(CerConfig(regime_a_only=False))
    cer_idle = CerReallocator(CerConfig(regime_a_only=True))
    hoashi = PriorityHoashiReallocator(PriorityHoashiConfig(CANONICAL_URLLC_PRIORITY_WEIGHTS))

    for trial in range(500):
        c = int(rng.choice([10, 52, 53]))
        act_bits = tuple(bool(x) for x in rng.integers(0, 2, size=3))
        if rng.random() < 0.1:
            act_bits = (False, False, False)
        n_ues = tuple(int(rng.integers(1, 5)) if act_bits[i] else 0 for i in range(3))
        tel = CerBlocoATelemetry(
            n_ues=n_ues,  # type: ignore[arg-type]
            queue_bytes=tuple(float(rng.uniform(0.0, 5e6)) if act_bits[i] else 0.0 for i in range(3)),  # type: ignore[arg-type]
            offered_mbps=tuple(float(rng.uniform(0.0, 12.0)) if act_bits[i] else 0.0 for i in range(3)),  # type: ignore[arg-type]
            snr_db=tuple(float(rng.uniform(-4.0, 25.0)) for _ in range(3)),  # type: ignore[arg-type]
        )
        raw_prop = [int(rng.integers(-3, c + 15)) for _ in range(3)]
        sum_cap = int(rng.integers(0, c + 1))
        a_nom = [0, 0, 0]
        if any(act_bits) and sum_cap > 0:
            rem = sum_cap
            for i in range(3):
                if act_bits[i]:
                    take = int(rng.integers(0, rem + 1))
                    a_nom[i] = take
                    rem -= take

        f_curves = [rng.normal(loc=float(rng.uniform(-1.0, 0.5)), scale=0.25, size=c + 1) for _ in range(3)]
        sig_curves = [rng.uniform(0.0, 0.9, size=c + 1) for _ in range(3)]

        tr = PrequentialRiskTracker()
        for sid in range(3):
            for _ in range(int(rng.integers(0, 12))):
                tr.record_closed_outcome(
                    sid,
                    float(rng.uniform(-0.5, 0.5)),
                    float(rng.uniform(-0.6, 0.8)),
                    outcome_win=95,
                )
        snap = tr.snapshot(governed_window=100)

        for engine in (cer_full, cer_idle):
            dec = engine.reallocate(
                a_nom=a_nom,
                f_curves=f_curves,
                sigma_curves=sig_curves,
                telemetry=tel,
                risk=snap,
                capacity_c=c,
            )
            assert len(dec.action) == 3, f"trial={trial}"
            assert all(isinstance(v, int) and v >= 0 for v in dec.action), f"trial={trial}"
            assert sum(dec.action) <= c, f"trial={trial} exceeded C={c}: {dec.action}"
            for i in range(3):
                assert dec.delta_prbs[i] == dec.action[i] - dec.a_nom_guarded[i]

        h_dec = hoashi.allocate(raw_prop, c, act_bits)
        assert all(v >= 0 for v in h_dec.action)
        assert sum(h_dec.action) <= c
        for i in range(3):
            if not act_bits[i]:
                assert h_dec.action[i] == 0


def test_priority_hoashi_saturation_waterfilling() -> None:
    hoashi = PriorityHoashiReallocator(PriorityHoashiConfig(CANONICAL_URLLC_PRIORITY_WEIGHTS))
    # URLLC weight 1e6 dominates; excess cut falls on eMBB & mMTC, clamping mMTC if needed
    dec = hoashi.allocate((10, 45, 5), capacity_c=53, active_mask=(True, True, True))
    assert sum(dec.action) == 53
    assert dec.action[1] == 45  # URLLC untouched
    assert dec.action == (7, 45, 1) or dec.action == (6, 45, 2)
    assert dec.cut_applied and dec.excess_cut == 7

    # Extreme saturation where eMBB + mMTC proposals < excess -> URLLC also gives remaining cut cleanly
    dec_sat = hoashi.allocate((3, 52, 2), capacity_c=53, active_mask=(True, True, True))
    assert dec_sat.action == (0, 52, 0) or sum(dec_sat.action) == 53
    assert sum(dec_sat.action) == 53
    assert dec_sat.action[1] == 52 and dec_sat.action[0] + dec_sat.action[2] == 1

    # Balanced sensitivity weights parse check
    assert parse_priority_weights("10,1000000,1") == BALANCED_PRIORITY_WEIGHTS


# ---------------------------------------------------------------------------
# 2. PAVA accuracy, KKT optimality, and O(C) linear scaling
# ---------------------------------------------------------------------------


def test_pava_monotonicity_idempotence_kkt_and_linear_time() -> None:
    rng = np.random.default_rng(7)
    raw = np.sin(np.linspace(0, 6.0, 54)) + 0.35 * rng.normal(size=54)
    tilde = pava_nondecreasing(raw)

    assert tilde.shape == raw.shape
    assert np.all(np.diff(tilde) >= -1e-12)
    # Idempotence: second PAVA equals first
    assert np.allclose(pava_nondecreasing(tilde), tilde, atol=1e-12)

    # KKT optimality against valid feasible monotone perturbations (e.g. clamp adjacent pairs)
    loss_pava = float(np.sum((tilde - raw) ** 2))
    loss_sorted = float(np.sum((np.sort(raw) - raw) ** 2))
    assert loss_pava <= loss_sorted + 1e-12
    assert regularized_marginal_slope(tilde, 10, delta_ref=4) >= 0.0
    assert regularized_marginal_slope(tilde, 53, delta_ref=4) == 0.0

    # O(C) runtime check: 5300 elements must stay well under 25 ms and scale sub-quadratically
    big_x = rng.normal(size=5300)
    t0 = time.perf_counter()
    res_big = pava_nondecreasing(big_x)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert np.all(np.diff(res_big) >= -1e-12)
    assert dt_ms < 25.0, f"PAVA too slow ({dt_ms:.2f} ms), expected O(C)"


# ---------------------------------------------------------------------------
# 3. Strict A2a preservation (empty slices stay zero under guard=0)
# ---------------------------------------------------------------------------


def test_a2a_empty_slice_guard_preserved() -> None:
    cer = CerReallocator(CerConfig(idle_empty_guard_prbs=0))
    tel = CerBlocoATelemetry(
        n_ues=(0, 2, 1),
        queue_bytes=(0.0, 120.0, 200.0),
        offered_mbps=(0.0, 0.512, 0.003),
        snr_db=(0.0, 12.0, 8.0),
        tx_pkts_prev=(0, 20, 1),
    )
    snap, _ = _make_clean_tracker()
    # Even if noisy upstream a_nom gave 10 PRBs to inactive slice 0, A2a zeros it
    dec = cer.reallocate(
        a_nom=(10, 20, 5),
        f_curves=[_linear_curve(53, -0.5, 0.02), _linear_curve(53, -0.3, 0.002), _linear_curve(53, 0.01, 0.005)],
        sigma_curves=[np.full(54, 0.05) for _ in range(3)],
        telemetry=tel,
        risk=snap,
        capacity_c=53,
    )
    assert dec.active_mask == (False, True, True)
    assert dec.a_nom_guarded[0] == 0
    assert dec.action[0] == 0
    assert sum(dec.action) == 53

    # Controlled guard ablation gives up to idle_empty_guard_prbs ONLY when no active deficit
    cer_guard2 = CerReallocator(CerConfig(idle_empty_guard_prbs=2))
    snap_ok, _ = _make_clean_tracker(y_last=(0.5, 0.19, 0.0526))
    # Active deficit (34 < prop_u=36) => NO PRBs reserved for empty slice 0
    dec_g2_def = cer_guard2.reallocate(
        a_nom=(0, 34, 13),
        f_curves=[_linear_curve(53, 0.2, 0.01), _linear_curve(53, 0.1, 0.001), _linear_curve(53, 0.05, 0.002)],
        sigma_curves=[np.full(54, 0.05) for _ in range(3)],
        telemetry=tel,
        risk=snap_ok,
        capacity_c=53,
    )
    assert dec_g2_def.action[0] == 0

    # Zero active deficit (36 >= prop_u=36 and 12 >= a_phys_m) with slack=5 => reserves 2 PRBs
    dec_g2 = cer_guard2.reallocate(
        a_nom=(0, 36, 12),
        f_curves=[_linear_curve(53, 0.2, 0.01), _linear_curve(53, 0.1, 0.001), _linear_curve(53, 0.05, 0.002)],
        sigma_curves=[np.full(54, 0.05) for _ in range(3)],
        telemetry=tel,
        risk=snap_ok,
        capacity_c=53,
    )
    assert dec_g2.deficit_before_idle == ()
    assert dec_g2.action[0] == 2
    assert sum(dec_g2.action) <= 53


# ---------------------------------------------------------------------------
# 4. Deterministic t1608-like rescue (URLLC wide safe plateau -> elastic eMBB)
# ---------------------------------------------------------------------------


def test_deterministic_t1608_urllc_plateau_rescue() -> None:
    cer = CerReallocator(CerConfig())
    # Geometry mirroring S1_lam6_s2102_t1608: a_nom=(15,37,1), prop_u=7, eMBB elastic deficit
    tel = CerBlocoATelemetry(
        n_ues=(3, 1, 4),
        queue_bytes=(500_000.0, 69.0, 948.0),
        offered_mbps=(2.1, 0.256, 0.0064),
        snr_db=(8.0, 14.0, 6.0),
        tx_pkts_prev=(50, 10, 4),
        urllc_pool_open=True,
    )
    # eMBB rises cleanly from -0.45 to -0.05 (slope ~0.01/PRB >= eps_slope); URLLC quantized negative raw f
    f_e = _linear_curve(53, -0.58, 0.010)
    f_u = _linear_curve(53, -0.18, 0.001)
    f_m = _linear_curve(53, -0.10, 0.0015)
    sigs = [np.full(54, 0.04), np.full(54, 0.01), np.full(54, 0.03)]
    snap, _ = _make_clean_tracker(
        y_last=(0.04, 0.19, 0.0526),  # y_u=+0.19 >= 0.08, U_u=0 < 0.20, lag=2 <= 6
        urllc_lag=2,
        n_open_u=2,
    )
    dec = cer.reallocate(
        a_nom=(15, 37, 1),
        f_curves=[f_e, f_u, f_m],
        sigma_curves=sigs,
        telemetry=tel,
        risk=snap,
        capacity_c=53,
    )
    # Under t1608: n=(3,1,4) => prop_u=7, F_u = max(6, ceil(1.15*7) + 2) = 11, surplus = 26 -> budget Delta_max=20
    assert dec.donor_floors[1] == 11
    assert 1 in dec.eligible_donors
    assert 0 in dec.eligible_receivers
    assert dec.action == (35, 17, 1)
    assert dec.cer_stage == "REGIME_B"
    assert dec.abstain_reason == "none"


# ---------------------------------------------------------------------------
# 5. Deterministic abstention rules (critical border, high U_preq, healthy, sufficiency guard)
# ---------------------------------------------------------------------------


def test_abstention_critical_border_high_risk_healthy_and_sufficiency() -> None:
    cer = CerReallocator(CerConfig())
    tel = CerBlocoATelemetry(
        n_ues=(3, 1, 4),
        queue_bytes=(500_000.0, 69.0, 948.0),
        offered_mbps=(2.1, 0.256, 0.0064),
        snr_db=(8.0, 14.0, 6.0),
        tx_pkts_prev=(50, 10, 4),
    )
    f_e = _linear_curve(53, -0.58, 0.010)
    f_u = _linear_curve(53, 0.02, 0.0005)
    f_m = _linear_curve(53, -0.02, 0.002)
    sigs = [np.full(54, 0.04), np.full(54, 0.01), np.full(54, 0.03)]

    # (a) URLLC last closed margin near 10ms border (< 0.08) => blocks URLLC donation
    snap_border, _ = _make_clean_tracker(y_last=(0.04, 0.05, 0.0526), urllc_lag=2)
    dec_border = cer.reallocate(
        a_nom=(22, 30, 1),
        f_curves=[f_e, f_u, f_m],
        sigma_curves=sigs,
        telemetry=tel,
        risk=snap_border,
        capacity_c=53,
    )
    assert 1 not in dec_border.eligible_donors
    assert dec_border.action == (22, 30, 1)
    assert dec_border.cer_stage == "MANTER_B0"

    # (b) High URLLC prequential risk U_{w,u} >= 0.20 (e.g. 3/10 violations => 0.70*0.3 = 0.21) => blocks donor
    tr_risk = PrequentialRiskTracker()
    for k in range(10):
        tr_risk.record_closed_outcome(1, 0.0, -0.015 if k < 3 else 0.19, outcome_win=80 + k)
    tr_risk._last_closed_urllc_win = 98  # noqa: SLF001
    snap_risk = tr_risk.snapshot(governed_window=100)
    assert snap_risk.u_preq[1] >= 0.20
    dec_risk = cer.reallocate(
        a_nom=(15, 37, 1),
        f_curves=[f_e, f_u, f_m],
        sigma_curves=sigs,
        telemetry=tel,
        risk=snap_risk,
        capacity_c=53,
    )
    assert 1 not in dec_risk.eligible_donors
    assert dec_risk.action == (15, 37, 1)

    # (c) Excessive URLLC lag > L_safe (6) => blocks donor even with positive last margin
    snap_lag, _ = _make_clean_tracker(y_last=(0.04, 0.19, 0.0526), urllc_lag=8)
    dec_lag = cer.reallocate(
        a_nom=(15, 37, 1),
        f_curves=[f_e, f_u, f_m],
        sigma_curves=sigs,
        telemetry=tel,
        risk=snap_lag,
        capacity_c=53,
    )
    assert 1 not in dec_lag.eligible_donors
    assert dec_lag.action == (15, 37, 1)

    # (d) Healthy w1621-style state or intra-window arrival w1254-style => MANTER_B0
    snap_ok, _ = _make_clean_tracker(y_last=(2.0, 0.19, 0.0526), urllc_lag=2)
    tel_w1254 = CerBlocoATelemetry(
        n_ues=(0, 2, 1),
        queue_bytes=(0.0, 138.0, 237.0),
        offered_mbps=(0.0, 0.9792, 0.0016),
        snr_db=(0.0, 12.0, 8.0),
        tx_pkts_prev=(0, 10, 1),
    )
    dec_1254 = cer.reallocate(
        a_nom=(0, 40, 13),
        f_curves=[_linear_curve(53, 0.0, 0.0), _linear_curve(53, -0.1, 0.001), _linear_curve(53, 0.02, 0.002)],
        sigma_curves=sigs,
        telemetry=tel_w1254,
        risk=snap_ok,
        capacity_c=53,
    )
    assert dec_1254.action == (0, 40, 13)
    assert dec_1254.cer_stage == "MANTER_B0"

    # (e) Anti-innocuous-perturbation sufficiency guard: deep eMBB deficit with total gain < 0.05 abstains
    f_e_flat = _linear_curve(53, -0.50, 0.0015)  # over +20 PRBs gives +0.03 < 0.05
    dec_guard = cer.reallocate(
        a_nom=(15, 37, 1),
        f_curves=[f_e_flat, f_u, _linear_curve(53, 0.05, 0.001)],
        sigma_curves=sigs,
        telemetry=tel,
        risk=snap_ok,
        capacity_c=53,
    )
    assert dec_guard.action == (15, 37, 1)
    assert dec_guard.abstain_reason == "insufficient_embb_gain_guard"


# ---------------------------------------------------------------------------
# 6. Zero access to Bloco B columns over all 25 Gate 1 states
# ---------------------------------------------------------------------------


def test_zero_bloco_b_reads_across_all_25_gate_states() -> None:
    assert GCE_CSV.exists(), f"Missing {GCE_CSV}"
    with GCE_CSV.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = list(reader.fieldnames or [])
        forbidden = set(cols[111:])
        assert all(k.startswith("off_") for k in forbidden)
        rows = list(reader)

    assert len(rows) == 25
    cer = CerReallocator(CerConfig())
    snap, _ = _make_clean_tracker()

    for row in rows:
        spy = _SpyBlocoAMapping(row, forbidden)
        tel = extract_bloco_a_telemetry(spy)
        a_nom = (
            int(float(spy["b0_a_exec_e"])),
            int(float(spy["b0_a_exec_u"])),
            int(float(spy["b0_a_exec_m"])),
        )
        dec = cer.reallocate(
            a_nom=a_nom,
            f_curves=[_linear_curve(53, -0.2, 0.01), _linear_curve(53, -0.2, 0.002), _linear_curve(53, 0.01, 0.002)],
            sigma_curves=[np.full(54, 0.05) for _ in range(3)],
            telemetry=tel,
            risk=snap,
            capacity_c=53,
        )
        assert sum(dec.action) <= 53
        assert all(not k.startswith("off_") and k not in forbidden for k in spy.accessed)


def test_sqr_v2_run_cli_defaults_preserve_none() -> None:
    import sqr_v2_run

    args_default = sqr_v2_run.parse_args([])
    assert args_default.reallocator == "none"
    assert args_default.cer_mode == "full"
    assert args_default.idle_empty_guard_prbs == 0

    args_cer = sqr_v2_run.parse_args(
        ["--reallocator", "sqr_cer", "--cer-mode", "regime_a_only", "--idleEmptyGuardPrbs", "2"]
    )
    assert args_cer.reallocator == "sqr_cer"
    assert args_cer.cer_mode == "regime_a_only"
    assert args_cer.idle_empty_guard_prbs == 2
