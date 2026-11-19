"""Asymmetric SQR-CER reallocator (ESPECIFICACAO_SQR_CER_V2 v1.0).

Implements Stages 0-3 of Safe Quantile Regression with Constrained Elasticity
Reallocation (``SQR-CER``) under ``sqr_v2``:
  - Stage 0: O(C) Pool-Adjacent-Violators (PAVA) isotonic projection and
    regularized marginal slope computation.
  - Stage 1: Causal unilateral prequential risk index U_{w,i} (strictly
    observed_at <= decision_at) and class-specific demand/physical envelopes.
  - Stage 2: Conservative absorption of unallocated capacity S_w (Regime A).
  - Stage 3: Envelope-confined inter-slice micro-transfers (Regime B) with
    explicit abstention and anti-innocuous-perturbation sufficiency guard.

Design invariants:
  - Strictly causal (Bloco A only): never accesses Bloco B (``off_*``) fields.
  - Never loads ``sqr_checkpoint.pkl`` files or touches ``sqr`` / ``sqr_jc``.
  - Deterministic, typed, pure Python/NumPy (< 1 ms p95 per decision).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

NUM_SLICES = 3
SLICE_NAMES: tuple[str, str, str] = ("eMBB", "URLLC", "mMTC")

# Canonical PropDemand / causal active demand weights & physical constants
PROP_DEMAND_WEIGHT_FLOOR_QUEUE = 0.5
PROP_DEMAND_WEIGHT_FLOOR_PENDING = 0.25
URLLC_NOMINAL_MBPS_PER_UE = 0.256
MMTC_NOMINAL_MBPS_PER_UE = 0.0016
WIN_DURATION_S = 0.1
DL_SLOTS_PER_WIN = 50
# ~11 data symbols/slot * 12 subcarriers = 132 REs per PRB per DL slot
RES_PER_PRB_SLOT = 132.0
SNR_SENTINEL_CUTOFF = -100.0


@dataclass(frozen=True)
class CerConfig:
    """Hyperparameters of Asymmetric SQR-CER (ESPECIFICACAO_SQR_CER_V2 §3)."""

    # Stage 0 & Stage 1 Prequential Risk
    delta_ref: int = 4
    w_preq: int = 20
    beta_v: float = 0.70
    beta_o: float = 0.30
    beta_c: float = 0.30
    l_max: int = 10

    # Deficit detection thresholds (§3.2 item 4)
    theta_u_def_e: float = 0.25
    theta_u_def_u: float = 0.30

    # Stage 2 Regime A (idle capacity absorption)
    a_cap_u_idle: int = 18
    idle_empty_guard_prbs: int = 0

    # Stage 3 Donor certificates (§3.4 item 1)
    a_min_u: int = 6
    kappa_u: float = 1.15
    b_open: int = 1
    mu_margin_u: float = 0.08
    l_safe: int = 6
    theta_u_don_u: float = 0.20

    a_min_m: int = 2
    b_m: int = 1
    mu_margin_m: float = 0.01
    theta_u_don_m: float = 0.15

    mu_don_e: float = 0.10
    mu_margin_e: float = 0.10
    sigma_max_don_e: float = 0.20
    theta_u_don_e: float = 0.15

    # Stage 3 Receiver certificates & micro-transfer loop (§3.4 items 2-3)
    eps_slope: float = 0.001
    sigma_max_rec: float = 0.55
    mu_target_e: float = 0.05
    mu_target_m: float = 0.02
    delta_0: int = 2
    delta_max: int = 20
    deep_deficit_f_e: float = -0.15
    eta_min_gain_e: float = 0.05

    # Operational mode toggle ("full" vs "regime_a_only") & frame geometry
    regime_a_only: bool = False
    dl_slots_per_win: int = DL_SLOTS_PER_WIN

    def __post_init__(self) -> None:
        if self.delta_ref <= 0 or self.w_preq <= 0:
            raise ValueError("delta_ref and w_preq must be positive")
        if self.delta_0 <= 0 or self.delta_max < 0:
            raise ValueError("delta_0 must be > 0 and delta_max >= 0")
        if self.idle_empty_guard_prbs < 0:
            raise ValueError("idle_empty_guard_prbs must be >= 0")
        if self.dl_slots_per_win <= 0:
            raise ValueError("dl_slots_per_win must be positive")


@dataclass(frozen=True)
class CerBlocoATelemetry:
    """Strictly causal pre-decision Bloco A telemetry at window boundary w."""

    n_ues: tuple[int, int, int]
    queue_bytes: tuple[float, float, float]
    offered_mbps: tuple[float, float, float]
    snr_db: tuple[float, float, float]
    tx_pkts_prev: tuple[int, int, int] = (0, 0, 0)
    embb_pending_pipeline: bool = False
    urllc_pool_open: bool = False
    mmtc_rolling_active: bool = False
    active_mask_override: tuple[bool, bool, bool] | None = None


@dataclass(frozen=True)
class PrequentialSnapshot:
    """Causal prequential risk snapshot computed strictly before decision w."""

    u_preq: tuple[float, float, float]
    r_rel: tuple[float, float, float]
    v_rate: tuple[float, float, float]
    overpred_pen: tuple[float, float, float]
    c_urllc: float
    last_closed_margins: tuple[float | None, float | None, float | None]
    n_open_urllc: int
    n_exp_urllc: int
    urllc_lag_windows: int
    hist_counts: tuple[int, int, int]


@dataclass(frozen=True)
class CerDecision:
    """Full auditable record of one SQR-CER Stage 0-3 decision."""

    action: tuple[int, int, int]
    a_nom_guarded: tuple[int, int, int]
    a_stage1_idle: tuple[int, int, int]
    active_mask: tuple[bool, bool, bool]
    cer_stage: str
    slack_prbs: int
    a_prop: tuple[int, int, int]
    a_phys_m: int
    donor_floors: tuple[int | None, int | None, int | None]
    donor_surplus: tuple[int, int, int]
    eligible_donors: tuple[int, ...]
    eligible_receivers: tuple[int, ...]
    deficit_before_idle: tuple[int, ...]
    deficit_after_idle: tuple[int, ...]
    u_preq: tuple[float, float, float]
    r_rel: tuple[float, float, float]
    delta_prbs: tuple[int, int, int]
    abstain_reason: str
    f_pava: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    slopes_at_nom: tuple[float, float, float]


def pava_nondecreasing(f: Sequence[float] | np.ndarray) -> np.ndarray:
    """Exact O(C) isotonic regression (Pool-Adjacent-Violators) without sorting."""
    vals = np.asarray(f, dtype=np.float64)
    n = vals.size
    if n == 0:
        return vals.copy()
    if n == 1:
        return np.array([float(vals[0])], dtype=np.float64)

    # Monotonic stack of blocks [sum_value, length]
    blocks: list[list[Any]] = []
    for raw_v in vals:
        s = float(raw_v)
        c = 1
        while blocks:
            prev_s, prev_c = blocks[-1]
            if (prev_s / prev_c) > (s / c):
                blocks.pop()
                s += prev_s
                c += prev_c
            else:
                break
        blocks.append([s, c])

    out = np.empty(n, dtype=np.float64)
    pos = 0
    for s, c in blocks:
        out[pos : pos + c] = s / float(c)
        pos += c
    return out


def regularized_marginal_slope(
    f_tilde: Sequence[float] | np.ndarray,
    a: int,
    delta_ref: int = 4,
) -> float:
    """Regularized forward marginal slope s_{w,i}(a) over delta_ref PRBs (Eq. 52)."""
    n = len(f_tilde)
    if n <= 1:
        return 0.0
    c = n - 1
    a_lo = max(0, min(c, int(a)))
    a_hi = min(c, a_lo + max(1, int(delta_ref)))
    step = a_hi - a_lo
    if step <= 0:
        return 0.0
    diff = float(f_tilde[a_hi]) - float(f_tilde[a_lo])
    return float(max(0.0, diff) / float(step))


def _assert_not_bloco_b_key(key: str) -> None:
    if str(key).startswith("off_"):
        raise PermissionError(f"Causal violation: Bloco B key {key!r} is forbidden in SQR-CER")


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or val == "":
        return float(default)
    f = float(val)
    return float(default) if math.isnan(f) else f


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None or val == "":
        return int(default)
    return int(round(float(val)))


def _lookup_mapping_whitelisted(mapping: Mapping[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    """Lookup strictly from pre-approved Bloco A keys without iterating Bloco B keys."""
    for key in keys:
        _assert_not_bloco_b_key(key)
        if key in mapping:
            return mapping[key]
    return default


def extract_bloco_a_telemetry(
    source: CerBlocoATelemetry | Mapping[str, Any] | Any,
) -> CerBlocoATelemetry:
    """Extract strictly pre-decision Bloco A telemetry from JointState, KPM window, or Bloco A mapping."""
    if isinstance(source, CerBlocoATelemetry):
        return source

    if isinstance(source, Mapping):
        n_e = _safe_int(_lookup_mapping_whitelisted(source, ("n_e",), 0))
        n_u = _safe_int(_lookup_mapping_whitelisted(source, ("n_u",), 0))
        n_m = _safe_int(_lookup_mapping_whitelisted(source, ("n_m",), 0))

        q_e = _safe_float(_lookup_mapping_whitelisted(source, ("queue_e",), 0.0))
        q_u = _safe_float(_lookup_mapping_whitelisted(source, ("queue_u",), 0.0))
        q_m = _safe_float(_lookup_mapping_whitelisted(source, ("queue_m",), 0.0))

        off_e = _safe_float(_lookup_mapping_whitelisted(source, ("offered_e", "offered_e_mbps"), 0.0))
        off_u = _safe_float(_lookup_mapping_whitelisted(source, ("offered_u", "offered_u_mbps"), 0.0))
        off_m = _safe_float(_lookup_mapping_whitelisted(source, ("offered_m", "offered_m_mbps"), 0.0))

        snr_e = _safe_float(_lookup_mapping_whitelisted(source, ("snr_e", "snr_e_db"), 10.0), 10.0)
        snr_u = _safe_float(_lookup_mapping_whitelisted(source, ("snr_u", "snr_u_db"), 10.0), 10.0)
        snr_m = _safe_float(_lookup_mapping_whitelisted(source, ("snr_m", "snr_m_db"), 10.0), 10.0)

        extras = _lookup_mapping_whitelisted(source, ("extras",), None)
        if isinstance(extras, Mapping) and "kpm_tx_packets" in extras and extras["kpm_tx_packets"] is not None:
            tx_tup = tuple(int(v) for v in extras["kpm_tx_packets"][:3])
        else:
            tx_tup = (
                _safe_int(_lookup_mapping_whitelisted(source, ("tx_pkts_e_wprev",), 0)),
                _safe_int(_lookup_mapping_whitelisted(source, ("tx_pkts_u_wprev",), 0)),
                _safe_int(_lookup_mapping_whitelisted(source, ("tx_pkts_m_wprev",), 0)),
            )

        embb_pend = bool(extras.get("embb_pending_pipeline", False)) if isinstance(extras, Mapping) else False
        urllc_open = (
            bool(extras.get("urllc_pool_open", False))
            if isinstance(extras, Mapping) and "urllc_pool_open" in extras
            else bool(_lookup_mapping_whitelisted(source, ("urllc_pool_open_at_dec",), False))
        )
        prev_m = _lookup_mapping_whitelisted(source, ("prev_kpi_m_pdr", "kpi_m_raw"), None)
        mmtc_roll_flag = (
            bool(extras.get("mmtc_rolling_active", False))
            if isinstance(extras, Mapping) and "mmtc_rolling_active" in extras
            else bool(_lookup_mapping_whitelisted(source, ("mmtc_rolling_active_at_dec",), False))
        )
        mmtc_active = mmtc_roll_flag or (prev_m is not None and not (isinstance(prev_m, float) and math.isnan(prev_m)))

        return CerBlocoATelemetry(
            n_ues=(n_e, n_u, n_m),
            queue_bytes=(q_e, q_u, q_m),
            offered_mbps=(off_e, off_u, off_m),
            snr_db=(snr_e, snr_u, snr_m),
            tx_pkts_prev=(tx_tup[0], tx_tup[1], tx_tup[2]),
            embb_pending_pipeline=embb_pend,
            urllc_pool_open=urllc_open,
            mmtc_rolling_active=mmtc_active,
        )

    # JointState or SqrKpmWindow-like object
    if hasattr(source, "n_e") and hasattr(source, "queue_e"):
        extras = getattr(source, "extras", None) or {}
        tx_raw = extras.get("kpm_tx_packets") or (0, 0, 0)
        kpi_m = getattr(source, "kpi_m_raw", None)
        return CerBlocoATelemetry(
            n_ues=(int(source.n_e), int(source.n_u), int(source.n_m)),
            queue_bytes=(float(source.queue_e), float(source.queue_u), float(source.queue_m)),
            offered_mbps=(float(source.offered_e), float(source.offered_u), float(source.offered_m)),
            snr_db=(float(source.snr_e), float(source.snr_u), float(source.snr_m)),
            tx_pkts_prev=(int(tx_raw[0]), int(tx_raw[1]), int(tx_raw[2])),
            embb_pending_pipeline=bool(extras.get("embb_pending_pipeline", False)),
            urllc_pool_open=bool(extras.get("urllc_pool_open", False)),
            mmtc_rolling_active=(kpi_m is not None) or bool(extras.get("mmtc_rolling_active", False)),
        )

    if hasattr(source, "slices") and hasattr(source, "rows"):
        # SqrKpmWindow from online SqrV2Env.builder.last_window
        n_list, q_list, off_list, snr_list, tx_list = [], [], [], [], []
        for sid in range(NUM_SLICES):
            sk = source.slices.get(sid)
            s_rows = [r for r in source.rows if int(getattr(r, "slice_id", -1)) == sid]
            if sk is not None:
                n_list.append(int(sk.n_ues))
                q_list.append(float(sk.queue_bytes_sum))
                snrs = getattr(sk, "snr_values", ())
                snr_list.append(float(np.mean(snrs)) if snrs else 10.0)
            else:
                n_list.append(len(s_rows))
                q_list.append(float(sum(float(getattr(r, "queue_bytes", 0.0)) for r in s_rows)))
                snr_list.append(float(np.mean([float(getattr(r, "snr_db", 10.0)) for r in s_rows])) if s_rows else 10.0)
            off_list.append(float(sum(max(0.0, float(getattr(r, "window_throughput_mbps", 0.0))) for r in s_rows)))
            tx_list.append(int(sum(int(getattr(r, "d_tx_packets", 0)) for r in s_rows)))
        return CerBlocoATelemetry(
            n_ues=(n_list[0], n_list[1], n_list[2]),
            queue_bytes=(q_list[0], q_list[1], q_list[2]),
            offered_mbps=(off_list[0], off_list[1], off_list[2]),
            snr_db=(snr_list[0], snr_list[1], snr_list[2]),
            tx_pkts_prev=(tx_list[0], tx_list[1], tx_list[2]),
        )

    raise TypeError(f"Unsupported Bloco A telemetry source type: {type(source)!r}")


def causal_active_mask(telemetry: CerBlocoATelemetry) -> tuple[bool, bool, bool]:
    """Causal A2a active-slice indicator d_{w,i} (matches sqr_het.is_slice_provably_empty)."""
    if telemetry.active_mask_override is not None:
        m = telemetry.active_mask_override
        return (bool(m[0]), bool(m[1]), bool(m[2]))
    pendings = (
        bool(telemetry.embb_pending_pipeline),
        bool(telemetry.urllc_pool_open),
        bool(telemetry.mmtc_rolling_active),
    )
    out = []
    for i in range(NUM_SLICES):
        provably_empty = (
            telemetry.n_ues[i] == 0
            and telemetry.queue_bytes[i] <= 0.0
            and telemetry.offered_mbps[i] <= 0.0
            and telemetry.tx_pkts_prev[i] <= 0
            and not pendings[i]
        )
        out.append(not provably_empty)
    return (out[0], out[1], out[2])


def prop_active_demand_weights(
    telemetry: CerBlocoATelemetry,
    active_mask: tuple[bool, bool, bool],
) -> tuple[float, float, float]:
    """Normalized demand weights rho_{w,i} aligned with ``prop_active_split``."""
    w = []
    for i in range(NUM_SLICES):
        if not active_mask[i]:
            w.append(0.0)
        elif telemetry.n_ues[i] > 0:
            w.append(float(telemetry.n_ues[i]))
        elif telemetry.queue_bytes[i] > 0.0:
            w.append(PROP_DEMAND_WEIGHT_FLOOR_QUEUE)
        else:
            w.append(PROP_DEMAND_WEIGHT_FLOOR_PENDING)
    return (w[0], w[1], w[2])


def compute_prop_active_envelope(
    telemetry: CerBlocoATelemetry,
    capacity_c: int,
    active_mask: tuple[bool, bool, bool],
) -> tuple[int, int, int]:
    """Ceiling proportional active quotas a_{w,i}^{prop} (Eq. 80)."""
    c = max(0, int(capacity_c))
    if c == 0 or not any(active_mask):
        return (0, 0, 0)
    weights = prop_active_demand_weights(telemetry, active_mask)
    denom = math.fsum(weights)
    if denom <= 0.0:
        return (0, 0, 0)
    out = []
    for i in range(NUM_SLICES):
        if active_mask[i] and weights[i] > 0.0:
            out.append(min(c, int(math.ceil(float(c) * weights[i] / denom))))
        else:
            out.append(0)
    return (out[0], out[1], out[2])


def bits_per_prb_slot_from_snr(snr_db: float) -> float:
    """Conservative bits/PRB/slot spectral efficiency from causal SNR report (Eq. 82)."""
    if not math.isfinite(snr_db) or snr_db <= SNR_SENTINEL_CUTOFF:
        eff = 0.15
    else:
        clamped_snr = min(30.0, max(-5.0, float(snr_db)))
        eff = min(5.5, max(0.15, math.log2(1.0 + 10.0 ** (clamped_snr / 10.0))))
    return eff * RES_PER_PRB_SLOT


def compute_mmtc_physical_floor(
    telemetry: CerBlocoATelemetry,
    capacity_c: int,
    active_mask: tuple[bool, bool, bool],
    *,
    a_min_m: int = 2,
    b_m: int = 1,
    dl_slots_per_win: int = DL_SLOTS_PER_WIN,
) -> int:
    """Conservative causal RLC queue + offered rate + SNR floor a_{w,m}^{phys} (Eq. 82)."""
    if not active_mask[2] or capacity_c <= 0:
        return 0
    q_bits = 8.0 * max(0.0, float(telemetry.queue_bytes[2]))
    r_mbps = max(
        float(telemetry.offered_mbps[2]),
        MMTC_NOMINAL_MBPS_PER_UE * max(0, int(telemetry.n_ues[2])),
    )
    req_bits = q_bits + 1.0e6 * r_mbps * WIN_DURATION_S
    cap_per_prb_win = float(dl_slots_per_win) * bits_per_prb_slot_from_snr(telemetry.snr_db[2])
    prbs_needed = int(math.ceil(req_bits / max(1.0, cap_per_prb_win)))
    return min(int(capacity_c), max(int(a_min_m), prbs_needed + int(b_m)))


class PrequentialRiskTracker:
    """Strictly causal unilateral prequential risk tracker (ESPECIFICACAO §3.2)."""

    def __init__(
        self,
        *,
        w_preq: int = 20,
        beta_v: float = 0.70,
        beta_o: float = 0.30,
        beta_c: float = 0.30,
        l_max: int = 10,
    ) -> None:
        self.w_preq = int(w_preq)
        self.beta_v = float(beta_v)
        self.beta_o = float(beta_o)
        self.beta_c = float(beta_c)
        self.l_max = int(l_max)

        self._closed_hist: dict[int, deque[tuple[float, float]]] = {
            sid: deque(maxlen=self.w_preq) for sid in range(NUM_SLICES)
        }
        self._last_closed_margin: list[float | None] = [None, None, None]
        self._open_urllc_cohorts: set[int] = set()
        self._exp_obs_windows: deque[int] = deque(maxlen=max(64, 4 * self.w_preq))
        self._last_closed_urllc_win: int | None = None
        self._ever_had_urllc_cohort: bool = False

    def record_closed_outcome(
        self,
        slice_id: int,
        f_pred_exec: float,
        y_closed: float,
        *,
        outcome_win: int | None = None,
    ) -> None:
        """Record one closed SLA outcome associated with executed MAC quota a_exec."""
        sid = int(slice_id)
        if sid not in (0, 1, 2):
            raise ValueError(f"invalid slice_id={sid}")
        yv = float(y_closed)
        fv = float(f_pred_exec)
        if not math.isfinite(yv) or not math.isfinite(fv):
            return
        self._closed_hist[sid].append((fv, yv))
        self._last_closed_margin[sid] = yv
        if sid == 1 and outcome_win is not None:
            ow = int(outcome_win)
            self._open_urllc_cohorts.discard(ow)
            self._ever_had_urllc_cohort = True
            if self._last_closed_urllc_win is None or ow > self._last_closed_urllc_win:
                self._last_closed_urllc_win = ow

    def mark_urllc_pending(self, cohort_win: int) -> None:
        """Register an active URLLC arrival cohort still waiting delayed closure."""
        self._open_urllc_cohorts.add(int(cohort_win))
        self._ever_had_urllc_cohort = True

    def mark_urllc_expired(self, cohort_win: int, *, observed_at_win: int) -> None:
        """Register Amendment-4 MAX_AGE expiration (``expired_vacuous_R19``)."""
        self._open_urllc_cohorts.discard(int(cohort_win))
        self._exp_obs_windows.append(int(observed_at_win))
        self._ever_had_urllc_cohort = True

    def snapshot(
        self,
        *,
        governed_window: int | None = None,
        n_open_urllc_override: int | None = None,
        lag_override: int | None = None,
    ) -> PrequentialSnapshot:
        """Compute causal risk indices U_{w,i} and reliability R_{w,i} at boundary w."""
        v_list, o_list, u_list, r_list, cnt_list = [], [], [], [], []
        n_open = (
            int(n_open_urllc_override)
            if n_open_urllc_override is not None
            else len(self._open_urllc_cohorts)
        )

        if governed_window is not None:
            gw = int(governed_window)
            state_w = gw - 1
            n_exp = sum(1 for wo in self._exp_obs_windows if wo > state_w - self.w_preq)
        else:
            gw = None
            n_exp = len(self._exp_obs_windows)

        if lag_override is not None:
            lag_u = max(1, int(lag_override))
        elif gw is not None and self._last_closed_urllc_win is not None:
            lag_u = max(1, gw - self._last_closed_urllc_win)
        elif gw is not None and self._open_urllc_cohorts:
            lag_u = max(1, gw - min(self._open_urllc_cohorts))
        else:
            lag_u = 1

        c_u = min(
            1.0,
            float(n_exp) / float(self.w_preq)
            + (0.5 if (self._ever_had_urllc_cohort and lag_u > self.l_max) else 0.0),
        )

        for sid in range(NUM_SLICES):
            hist = self._closed_hist[sid]
            cnt = len(hist)
            cnt_list.append(cnt)
            denom = float(max(1, cnt))
            v_hat = math.fsum(1.0 for _, y in hist if y < 0.0) / denom
            o_hat = math.fsum(max(0.0, f - y) for f, y in hist if y < 0.0) / denom
            c_i = c_u if sid == 1 else 0.0
            u_val = min(1.0, max(0.0, self.beta_v * v_hat + self.beta_o * o_hat + self.beta_c * c_i))
            v_list.append(v_hat)
            o_list.append(o_hat)
            u_list.append(u_val)
            r_list.append(1.0 - u_val)

        return PrequentialSnapshot(
            u_preq=(u_list[0], u_list[1], u_list[2]),
            r_rel=(r_list[0], r_list[1], r_list[2]),
            v_rate=(v_list[0], v_list[1], v_list[2]),
            overpred_pen=(o_list[0], o_list[1], o_list[2]),
            c_urllc=c_u,
            last_closed_margins=(
                self._last_closed_margin[0],
                self._last_closed_margin[1],
                self._last_closed_margin[2],
            ),
            n_open_urllc=n_open,
            n_exp_urllc=n_exp,
            urllc_lag_windows=lag_u,
            hist_counts=(cnt_list[0], cnt_list[1], cnt_list[2]),
        )


class CerReallocator:
    """Asymmetric SQR-CER Stages 0-3 joint budget reallocator."""

    def __init__(self, config: CerConfig | None = None) -> None:
        self.config = config or CerConfig()

    def _deficit_slices(
        self,
        alloc: Sequence[int],
        active_mask: tuple[bool, bool, bool],
        f_pava: tuple[np.ndarray, np.ndarray, np.ndarray],
        risk: PrequentialSnapshot,
        a_prop: tuple[int, int, int],
        a_phys_m: int,
        capacity_c: int,
    ) -> list[int]:
        cfg = self.config
        last_y = risk.last_closed_margins
        u_risk = risk.u_preq
        deficit: list[int] = []
        for sid in range(NUM_SLICES):
            if not active_mask[sid]:
                continue
            a_i = max(0, min(capacity_c, int(alloc[sid])))
            y_last = last_y[sid]
            y_neg = (y_last is not None) and (y_last < 0.0)
            if sid == 0:
                if f_pava[0][a_i] < 0.0 or y_neg or u_risk[0] >= cfg.theta_u_def_e:
                    deficit.append(0)
            elif sid == 1:
                # Never use f_tilde_u < 0 alone for URLLC (chronic false negative bias)
                if y_neg or u_risk[1] >= cfg.theta_u_def_u or a_i < min(capacity_c, a_prop[1]):
                    deficit.append(1)
            else:
                if f_pava[2][a_i] < 0.0 or y_neg or a_i < a_phys_m:
                    deficit.append(2)
        return deficit

    def _donor_floor_and_eligibility(
        self,
        sid: int,
        alloc_stage1: int,
        active_mask: tuple[bool, bool, bool],
        deficit_stage1: Sequence[int],
        f_pava: np.ndarray,
        sigma: np.ndarray,
        risk: PrequentialSnapshot,
        a_prop: tuple[int, int, int],
        a_phys_m: int,
        capacity_c: int,
    ) -> tuple[bool, int | None]:
        cfg = self.config
        if not active_mask[sid] or sid in deficit_stage1:
            return False, None

        y_last = risk.last_closed_margins[sid]
        u_i = risk.u_preq[sid]

        if sid == 1:
            if u_i >= cfg.theta_u_don_u or risk.urllc_lag_windows > cfg.l_safe:
                return False, None
            if y_last is not None and y_last < cfg.mu_margin_u:
                return False, None
            f_u = max(
                int(cfg.a_min_u),
                int(math.ceil(cfg.kappa_u * float(a_prop[1]))) + int(cfg.b_open) * int(risk.n_open_urllc),
            )
            f_u = min(capacity_c, f_u)
            return (alloc_stage1 - f_u) >= cfg.delta_0, f_u

        if sid == 2:
            if u_i >= cfg.theta_u_don_m:
                return False, None
            if y_last is not None and y_last < cfg.mu_margin_m:
                return False, None
            f_m = min(capacity_c, int(a_phys_m))
            return (alloc_stage1 - f_m) >= cfg.delta_0, f_m

        # sid == 0 (eMBB conjunctive lock against 55.3% overprediction FDR)
        if u_i >= cfg.theta_u_don_e:
            return False, None
        if y_last is None or y_last < cfg.mu_margin_e:
            return False, None
        idx_minus = max(0, int(alloc_stage1) - cfg.delta_0)
        if float(sigma[idx_minus]) > cfg.sigma_max_don_e:
            return False, None
        safe_as = [
            a
            for a in range(capacity_c + 1)
            if float(f_pava[a]) >= cfg.mu_don_e and float(sigma[a]) <= cfg.sigma_max_don_e
        ]
        if not safe_as:
            return False, capacity_c
        f_e = max(int(a_prop[0]), int(min(safe_as)))
        f_e = min(capacity_c, f_e)
        return (alloc_stage1 - f_e) >= cfg.delta_0, f_e

    def reallocate(
        self,
        *,
        a_nom: Sequence[int],
        f_curves: Sequence[Sequence[float] | np.ndarray],
        sigma_curves: Sequence[Sequence[float] | np.ndarray],
        telemetry: CerBlocoATelemetry | Mapping[str, Any] | Any,
        risk: PrequentialSnapshot,
        capacity_c: int,
    ) -> CerDecision:
        """Execute deterministic Stages 0-3 of Asymmetric SQR-CER."""
        cfg = self.config
        c = max(0, int(capacity_c))
        tel = extract_bloco_a_telemetry(telemetry)
        act_mask = causal_active_mask(tel)
        weights_norm = prop_active_demand_weights(tel, act_mask)

        # Stage 0: A2a empty guard + PAVA projection
        guarded = [
            max(0, min(c, int(a_nom[i]))) if act_mask[i] else 0
            for i in range(NUM_SLICES)
        ]
        # Defensive cap if upstream proposal exceeded c before Stage 0
        if sum(guarded) > c:
            excess = sum(guarded) - c
            for sid in sorted(range(NUM_SLICES), key=lambda i: (-guarded[i], i)):
                if excess <= 0:
                    break
                cut = min(excess, guarded[sid])
                guarded[sid] -= cut
                excess -= cut
        a_nom_guarded = (guarded[0], guarded[1], guarded[2])

        f_pava_arr = tuple(pava_nondecreasing(f_curves[i]) for i in range(NUM_SLICES))
        sig_arr = tuple(np.asarray(sigma_curves[i], dtype=np.float64) for i in range(NUM_SLICES))
        for i in range(NUM_SLICES):
            if f_pava_arr[i].size != c + 1 or sig_arr[i].size != c + 1:
                raise ValueError(
                    f"Curve length mismatch on slice {i}: expected {c + 1}, "
                    f"got f={f_pava_arr[i].size}, sigma={sig_arr[i].size}"
                )

        slopes_nom = (
            regularized_marginal_slope(f_pava_arr[0], a_nom_guarded[0], cfg.delta_ref),
            regularized_marginal_slope(f_pava_arr[1], a_nom_guarded[1], cfg.delta_ref),
            regularized_marginal_slope(f_pava_arr[2], a_nom_guarded[2], cfg.delta_ref),
        )

        # Stage 1: Causal envelopes and initial deficit set
        a_prop = compute_prop_active_envelope(tel, c, act_mask)
        a_phys_m = compute_mmtc_physical_floor(
            tel,
            c,
            act_mask,
            a_min_m=cfg.a_min_m,
            b_m=cfg.b_m,
            dl_slots_per_win=cfg.dl_slots_per_win,
        )
        def_before = self._deficit_slices(
            a_nom_guarded, act_mask, f_pava_arr, risk, a_prop, a_phys_m, c
        )

        # Stage 2: Regime A — conservative idle absorption
        alloc = list(a_nom_guarded)
        slack = max(0, c - sum(alloc))
        if slack > 0 and any(act_mask):
            # Step 2a: under-envelope floor coverage for URLLC / mMTC
            under_floors = {
                1: min(int(cfg.a_cap_u_idle), int(a_prop[1])),
                2: int(a_phys_m),
            }
            while slack > 0:
                needs = [
                    sid
                    for sid in (1, 2)
                    if act_mask[sid] and alloc[sid] < under_floors[sid]
                ]
                if not needs:
                    break
                chosen = min(needs, key=lambda sid: (alloc[sid], 0 if sid == 1 else 1))
                alloc[chosen] += 1
                slack -= 1

            # Optional idleEmptyGuardPrbs ablation before Step 2b (default 0 => strict A2a)
            if (
                cfg.idle_empty_guard_prbs > 0
                and len(def_before) == 0
                and any(not act_mask[sid] for sid in range(NUM_SLICES))
                and slack > 0
            ):
                empty_sids = [sid for sid in range(NUM_SLICES) if not act_mask[sid]]
                budget_guard = min(slack, int(cfg.idle_empty_guard_prbs))
                idx_g = 0
                while budget_guard > 0 and empty_sids:
                    alloc[empty_sids[idx_g % len(empty_sids)]] += 1
                    slack -= 1
                    budget_guard -= 1
                    idx_g += 1

            # Step 2b: iterative weighted marginal slope greedy assignment
            while slack > 0:
                def_now = self._deficit_slices(
                    alloc, act_mask, f_pava_arr, risk, a_prop, a_phys_m, c
                )
                cands = def_now if def_now else [sid for sid in range(NUM_SLICES) if act_mask[sid]]
                cands = [sid for sid in cands if alloc[sid] < c]
                if not cands:
                    break

                def _idle_key(sid: int) -> tuple[float, int, float, int]:
                    slope = regularized_marginal_slope(f_pava_arr[sid], alloc[sid], cfg.delta_ref)
                    score = (1.0 + risk.u_preq[sid]) * slope
                    env_gap = max(0, a_prop[sid] - alloc[sid])
                    return (round(score, 12), env_gap, weights_norm[sid], -sid)

                best_sid = max(cands, key=_idle_key)
                alloc[best_sid] += 1
                slack -= 1

        a_stage1_idle = (int(alloc[0]), int(alloc[1]), int(alloc[2]))
        def_after = self._deficit_slices(
            a_stage1_idle, act_mask, f_pava_arr, risk, a_prop, a_phys_m, c
        )

        # Evaluate donor floors & certificates (computed for Stage 3 auditability even if abstaining)
        donor_floors: list[int | None] = [None, None, None]
        donor_surplus = [0, 0, 0]
        elig_donors: list[int] = []
        for sid in range(NUM_SLICES):
            ok_d, f_d = self._donor_floor_and_eligibility(
                sid,
                a_stage1_idle[sid],
                act_mask,
                def_after,
                f_pava_arr[sid],
                sig_arr[sid],
                risk,
                a_prop,
                a_phys_m,
                c,
            )
            donor_floors[sid] = f_d
            if f_d is not None:
                surp = max(0, a_stage1_idle[sid] - f_d)
                donor_surplus[sid] = surp
            if ok_d:
                elig_donors.append(sid)

        # Receiver certificates (§3.4 item 2)
        elig_receivers: list[int] = []
        rec_by_slope: set[int] = set()
        rec_floor_targets: dict[int, int] = {}
        d0 = int(cfg.delta_0)
        for sid in def_after:
            a_i = a_stage1_idle[sid]
            if sid in (0, 2):
                a_next = min(c, a_i + d0)
                s_rec = regularized_marginal_slope(f_pava_arr[sid], a_i, cfg.delta_ref)
                if (risk.r_rel[sid] * s_rec) >= cfg.eps_slope and float(sig_arr[sid][a_next]) <= cfg.sigma_max_rec:
                    elig_receivers.append(sid)
                    rec_by_slope.add(sid)
            if sid in (1, 2):
                floor_ref = a_phys_m if sid == 2 else a_prop[1]
                floor_target = min(int(a_prop[sid]), int(floor_ref))
                if a_i < floor_target:
                    if sid not in elig_receivers:
                        elig_receivers.append(sid)
                    rec_floor_targets[sid] = floor_target

        def _finish(
            final_vec: tuple[int, int, int],
            stage_label: str,
            abstain_reason: str,
        ) -> CerDecision:
            f_tup = (
                tuple(float(x) for x in f_pava_arr[0]),
                tuple(float(x) for x in f_pava_arr[1]),
                tuple(float(x) for x in f_pava_arr[2]),
            )
            return CerDecision(
                action=final_vec,
                a_nom_guarded=a_nom_guarded,
                a_stage1_idle=a_stage1_idle,
                active_mask=act_mask,
                cer_stage=stage_label,
                slack_prbs=int(c - sum(a_nom_guarded)),
                a_prop=a_prop,
                a_phys_m=int(a_phys_m),
                donor_floors=(donor_floors[0], donor_floors[1], donor_floors[2]),
                donor_surplus=(donor_surplus[0], donor_surplus[1], donor_surplus[2]),
                eligible_donors=tuple(elig_donors),
                eligible_receivers=tuple(elig_receivers),
                deficit_before_idle=tuple(def_before),
                deficit_after_idle=tuple(def_after),
                u_preq=risk.u_preq,
                r_rel=risk.r_rel,
                delta_prbs=(
                    final_vec[0] - a_nom_guarded[0],
                    final_vec[1] - a_nom_guarded[1],
                    final_vec[2] - a_nom_guarded[2],
                ),
                abstain_reason=abstain_reason,
                f_pava=f_tup,
                slopes_at_nom=slopes_nom,
            )

        idle_used = any(a_stage1_idle[i] != a_nom_guarded[i] for i in range(NUM_SLICES))
        if cfg.regime_a_only:
            return _finish(
                a_stage1_idle,
                "REGIME_A" if idle_used else "MANTER_B0",
                "regime_a_only_mode" if not idle_used else "none",
            )

        if not def_after:
            return _finish(
                a_stage1_idle,
                "REGIME_A" if idle_used else "MANTER_B0",
                "no_remaining_deficit" if not idle_used else "none",
            )

        if sum(a_stage1_idle) < c:
            return _finish(
                a_stage1_idle,
                "REGIME_A" if idle_used else "MANTER_B0",
                "under_capacity_after_idle",
            )

        total_surplus = sum(donor_surplus[d] for d in elig_donors if donor_surplus[d] >= d0)
        if total_surplus < d0:
            reason = "no_eligible_donor" if not elig_donors else "insufficient_donor_surplus"
            return _finish(
                a_stage1_idle,
                "REGIME_A_ABSTAIN" if idle_used else "MANTER_B0",
                reason,
            )

        if not elig_receivers:
            return _finish(
                a_stage1_idle,
                "REGIME_A_ABSTAIN" if idle_used else "MANTER_B0",
                "no_eligible_receiver",
            )

        # Stage 3 item 3: Iterative constrained micro-transfers
        cur = list(a_stage1_idle)
        moved = 0
        budget = int(cfg.delta_max)

        while moved + d0 <= budget:
            cand_d = [
                d
                for d in elig_donors
                if donor_floors[d] is not None and (cur[d] - int(donor_floors[d])) >= d0
            ]
            if not cand_d:
                break

            cand_r = []
            for r in elig_receivers:
                if cur[r] + d0 > c:
                    continue
                if r == 0:
                    if f_pava_arr[0][cur[0]] < cfg.mu_target_e:
                        cand_r.append(r)
                elif r == 1:
                    if cur[1] < a_prop[1]:
                        cand_r.append(r)
                else:
                    need_floor = r in rec_floor_targets and cur[2] < rec_floor_targets[2]
                    need_quant = r in rec_by_slope and f_pava_arr[2][cur[2]] < cfg.mu_target_m
                    if need_floor or need_quant:
                        cand_r.append(r)

            if not cand_r:
                break

            d_star = max(
                cand_d,
                key=lambda d: (
                    cur[d] - int(donor_floors[d]),  # type: ignore[arg-type]
                    -regularized_marginal_slope(f_pava_arr[d], max(0, cur[d] - d0), cfg.delta_ref),
                    -d,
                ),
            )
            r_star = max(
                cand_r,
                key=lambda r: (
                    risk.r_rel[r] * regularized_marginal_slope(f_pava_arr[r], cur[r], cfg.delta_ref),
                    1 if (r in rec_floor_targets and cur[r] < rec_floor_targets[r]) else 0,
                    max(0, a_prop[r] - cur[r]),
                    -r,
                ),
            )
            if d_star == r_star:
                break
            cur[d_star] -= d0
            cur[r_star] += d0
            moved += d0

        if moved == 0:
            return _finish(
                a_stage1_idle,
                "REGIME_A_ABSTAIN" if idle_used else "MANTER_B0",
                "receiver_already_at_target",
            )

        # Final sufficiency guard for eMBB (anti-innocuous-perturbation, §3.4 item 3 line 149)
        if cur[0] > a_stage1_idle[0]:
            f_start_e = float(f_pava_arr[0][a_stage1_idle[0]])
            f_end_e = float(f_pava_arr[0][cur[0]])
            if f_start_e < cfg.deep_deficit_f_e and (f_end_e - f_start_e) < cfg.eta_min_gain_e:
                return _finish(
                    a_stage1_idle,
                    "REGIME_A_ABSTAIN" if idle_used else "MANTER_B0",
                    "insufficient_embb_gain_guard",
                )

        stage_lbl = "REGIME_A+B" if idle_used else "REGIME_B"
        return _finish((int(cur[0]), int(cur[1]), int(cur[2])), stage_lbl, "none")
