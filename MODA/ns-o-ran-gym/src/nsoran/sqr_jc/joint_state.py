"""Joint state z_joint — PRE-decision information only (§11/§12).

The joint feature vector consumed by every SQR-JC head is

    x_t = [ z_joint_t , a_e/C , a_u/C , a_m/C ]                (§5/§54)

z_joint_t (23 dims, all pre-decision, per §11):

    idx feature                     scale (§13)
     0  n_active_eMBB               CAL p1/p99 -> [0,1]   (count)
     1  n_active_URLLC              CAL p1/p99
     2  n_active_mMTC               CAL p1/p99
     3  queue_eMBB bytes            CAL p1/p99            (queue)
     4  queue_URLLC bytes           CAL p1/p99
     5  queue_mMTC bytes            CAL p1/p99
     6  prev_KPI_eMBB (thr med)     /SLA  -> clip[0,1]    (KPI)
     7  prev_KPI_URLLC (p95 ms)     /deadline -> clip[0,1]
     8  prev_KPI_mMTC (PDR)         natural [0,1]
     9  offered_eMBB Mbps           CAL p1/p99            (no natural cap)
    10  offered_URLLC Mbps          CAL p1/p99
    11  offered_mMTC Mbps           CAL p1/p99
    12  snr_eMBB dB                 /30 clip[0,1]         (audited R09 scale)
    13  snr_URLLC dB                /30 clip[0,1]
    14  snr_mMTC dB                 /30 clip[0,1]
    15  prev_a_e / C                /C                    (allocation)
    16  prev_a_u / C                /C
    17  prev_a_m / C                /C
    18  recent_violation_eMBB       {0,1}
    19  recent_violation_URLLC      {0,1}
    20  recent_violation_mMTC       {0,1}
    21  URLLC recent miss rate      natural [0,1] (fraction, not %)
    22  URLLC p99 latency           /deadline -> clip[0,1]

§11 lists "pending/retransmission state" for mMTC: the frozen KPM contract
(cols 1–17) exposes NO RLC retransmission/pending counter — that feature
does not exist in the audited environment and is NOT invented here
(registered as absent in the preflight report; the no-demand rule §42
therefore tests N_active == 0 and queue == 0, documented).

TEMPORAL LEAKAGE (§12 — test_no_future_information): every field of
``JointState`` is observable BEFORE the decision of window t: counts/queues/
SNR/offered come from the KPM window that JUST CLOSED; prev_KPI_* are the
outcomes of the window before that; prev_a_* and recent violations are the
controller's own history.  The builder REFUSES any post-window quantity by
construction (no field accepts the outcome of the window the action is
about to govern) — the update path (online_update) only ever receives the
outcome AFTER ``act`` has run, paired with the PREVIOUS decision's
JointState via the frozen alignment rule of §23:

    action k (decided at window k, stamp k*100 ms)
        governs window k+1 -> outcome stamp (k+1)*100 ms
        (adjudication.py:343 ``stamp = (win_t + 1) * WINDOW_MS``;
         sqr-sim.cc:1545-1553: the action received at border t governs
         (t, t+period] and drains at border t+period)
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np

from nsoran.sqr.sqr_kpis import SqrKpmWindow

#: Number of pre-decision features of z_joint (documented layout above).
JOINT_STATE_DIM = 23

#: Audited R09 SNR scale (sqr_env.py:80).
SNR_SCALE_DB = 30.0

#: Feature indices (documentation + test anchors).
FEATURE_INDEX: dict[str, int] = {
    "n_e": 0, "n_u": 1, "n_m": 2,
    "queue_e": 3, "queue_u": 4, "queue_m": 5,
    "kpi_e": 6, "kpi_u": 7, "kpi_m": 8,
    "offered_e": 9, "offered_u": 10, "offered_m": 11,
    "snr_e": 12, "snr_u": 13, "snr_m": 14,
    "prev_a_e": 15, "prev_a_u": 16, "prev_a_m": 17,
    "viol_e": 18, "viol_u": 19, "viol_m": 20,
    "urllc_miss": 21, "urllc_p99": 22,
}

#: Raw (unnormalized) quantities carried alongside for logs/scalers.
_RAW_FIELDS = (
    "n_e", "n_u", "n_m",
    "queue_e", "queue_u", "queue_m",
    "kpi_e_raw", "kpi_u_raw", "kpi_m_raw",
    "offered_e", "offered_u", "offered_m",
    "snr_e", "snr_u", "snr_m",
    "prev_a_e", "prev_a_u", "prev_a_m",
    "viol_e", "viol_u", "viol_m",
    "urllc_miss_raw", "urllc_p99_raw",
)


@dataclass
class JointState:
    """Raw pre-decision state; ``.vector(scaler)`` renders the feature.

    Every field is pre-decision information (§11/§12).  There is no field
    that could carry the outcome of the window being decided — the class
    has no slot for it, so "future information" cannot even be attached.
    """

    n_e: int = 0
    n_u: int = 0
    n_m: int = 0
    queue_e: float = 0.0
    queue_u: float = 0.0
    queue_m: float = 0.0
    kpi_e_raw: Optional[float] = None    # median windowThroughput, Mbps (None = no demand)
    kpi_u_raw: Optional[float] = None    # window p95 latency, ms (None = no packets)
    kpi_m_raw: Optional[float] = None    # 1 s rolling PDR fraction (None = no tx)
    offered_e: float = 0.0               # Mbps, served-rate sum over UEs
    offered_u: float = 0.0
    offered_m: float = 0.0
    snr_e: float = 0.0                   # mean imputed SNR dB of the slice
    snr_u: float = 0.0
    snr_m: float = 0.0
    prev_a_e: int = 0
    prev_a_u: int = 0
    prev_a_m: int = 0
    viol_e: int = 0                      # 0/1 recent violation indicator
    viol_u: int = 0
    viol_m: int = 0
    urllc_miss_raw: float = 0.0          # miss RATE fraction [0,1] (not %)
    urllc_p99_raw: float = 0.0           # ms

    window_index: int = -1               # decision window id (metadata)
    extras: dict = field(default_factory=dict)  # logs only, never a feature

    # ------------------------------------------------------------------
    # Feature rendering
    # ------------------------------------------------------------------

    def raw_vector(self) -> np.ndarray:
        """Unnormalized 23-dim z_joint (documented order; None KPIs -> 0)."""
        return np.array([
            float(self.n_e), float(self.n_u), float(self.n_m),
            float(self.queue_e), float(self.queue_u), float(self.queue_m),
            float(self.kpi_e_raw or 0.0), float(self.kpi_u_raw or 0.0),
            float(self.kpi_m_raw or 0.0),
            float(self.offered_e), float(self.offered_u), float(self.offered_m),
            float(self.snr_e), float(self.snr_u), float(self.snr_m),
            float(self.prev_a_e), float(self.prev_a_u), float(self.prev_a_m),
            float(self.viol_e), float(self.viol_u), float(self.viol_m),
            float(self.urllc_miss_raw), float(self.urllc_p99_raw),
        ], dtype=np.float64)

    def vector(self, scaler) -> np.ndarray:
        """Normalized z_joint via the FROZEN FeatureScaler (§13)."""
        return scaler.transform_joint(self.raw_vector())

    def feature_x(self, action: tuple[int, int, int], c: int, scaler) -> np.ndarray:
        """x_t = [z_joint, a/C] (§5) — the head feature vector (dim 26)."""
        z = self.vector(scaler)
        a = np.asarray(action, dtype=np.float64) / float(c)
        return np.concatenate([z, a])


# ---------------------------------------------------------------------------
# Builders (from the audited parses — no re-derivation, no future info)
# ---------------------------------------------------------------------------


def state_from_kpm_window(
    window: SqrKpmWindow,
    prev_action: tuple[int, int, int] = (0, 0, 0),
    closed_window_kpis: Optional[dict] = None,
    prev_violations: tuple[int, int, int] = (0, 0, 0),
    urllc_delay_stats: Optional[dict] = None,
    window_index: int = -1,
) -> JointState:
    """Build the JointState of the decision taken when ``window`` just closed.

    DECISION-WINDOW BOOKKEEPING (§23, frozen replay rule): a decision is
    taken at the border after the KPM window ``W`` (``window_index=W``)
    closes; it GOVERNS outcome window ``W+1`` whose rows are stamped
    ``(W+2)*100 ms`` in the KPM (``sqr-sim.cc:1545-1553`` — the action
    received at border t governs (t, t+period]).  ``online_update`` pairs
    THIS state with the outcome of :func:`governed_outcome_window`.

    Everything consumed here is pre-decision: counts/queues/SNR/offered and
    the URLLC delay stats come from the window that JUST CLOSED (already
    fully observable); ``closed_window_kpis`` are §11's ``previous_KPI_i``
    (the KPIs of that same just-closed window, as reported by the frozen
    Gate II-d aggregations; ``sqr_run.py`` executes observe with the same
    pairing).  NO quantity of the governed window W+1 is ever consumed.

    ``closed_window_kpis`` keys: ``embb_throughput_mbps`` / ``urllc_p95_ms``
    / ``mmtc_pdr_rolling`` (None = vacuous -> 0 after the physical scale).
    """
    kpis = closed_window_kpis or {}
    stats = urllc_delay_stats or {}

    k_e, k_u, k_m = (window.slices.get(0), window.slices.get(1), window.slices.get(2))

    offered = {0: 0.0, 1: 0.0, 2: 0.0}
    for row in window.rows:
        offered[int(row.slice_id)] += float(row.window_throughput_mbps)

    def _mean_snr(k) -> float:
        if k is None or not k.snr_values:
            return 0.0
        return float(np.mean(k.snr_values))

    state = JointState(
        n_e=k_e.n_ues if k_e else 0,
        n_u=k_u.n_ues if k_u else 0,
        n_m=k_m.n_ues if k_m else 0,
        queue_e=k_e.queue_bytes_sum if k_e else 0.0,
        queue_u=k_u.queue_bytes_sum if k_u else 0.0,
        queue_m=k_m.queue_bytes_sum if k_m else 0.0,
        kpi_e_raw=kpis.get("embb_throughput_mbps"),
        kpi_u_raw=kpis.get("urllc_p95_ms"),
        kpi_m_raw=kpis.get("mmtc_pdr_rolling"),
        offered_e=offered[0],
        offered_u=offered[1],
        offered_m=offered[2],
        snr_e=_mean_snr(k_e),
        snr_u=_mean_snr(k_u),
        snr_m=_mean_snr(k_m),
        prev_a_e=int(prev_action[0]),
        prev_a_u=int(prev_action[1]),
        prev_a_m=int(prev_action[2]),
        viol_e=int(prev_violations[0]),
        viol_u=int(prev_violations[1]),
        viol_m=int(prev_violations[2]),
        urllc_miss_raw=float(stats.get("miss_rate", 0.0)),
        urllc_p99_raw=float(stats.get("p99_ms", 0.0)),
        window_index=int(window_index),
    )
    return state


def governed_outcome_window(state_window_index: int) -> int:
    """Outcome window governed by the decision built on state window ``W``."""
    return int(state_window_index) + 1


def urllc_delay_window_stats(
    delay_rows: list[dict], window_index: int
) -> dict:
    """URLLC per-window delay aggregates by ARRIVAL window (frozen convention).

    ``delay_rows``: dicts with keys ``txSec`` (arrival/send time, s) and
    ``delaySec`` — the frozen ``sqr_packet_delays.csv`` columns (R32),
    slice URLLC rows only (caller filters ``sliceId == 1``; the frozen
    builder dataset.py::_urllc_packet_windows groups by
    ``win = floor(txSec * 10)``).

    Returns {} when the window has no URLLC packets (vacuous state).
    """
    sel = [r for r in delay_rows if int(np.floor(float(r["txSec"]) * 10.0)) == int(window_index)]
    if not sel:
        return {}
    ms = np.array([float(r["delaySec"]) * 1000.0 for r in sel], dtype=np.float64)
    return {
        "n_pkts": int(ms.size),
        "p50_ms": float(np.percentile(ms, 50.0)),
        "p95_ms": float(np.percentile(ms, 95.0)),
        "p99_ms": float(np.percentile(ms, 99.0)),
        "miss_rate": float(np.count_nonzero(ms > 10.0) / ms.size),
    }


# ---------------------------------------------------------------------------
# §23 decision-window alignment (frozen replay rule)
# ---------------------------------------------------------------------------

DECISION_PERIOD_MS = 100


def governing_window(decision_window: int) -> int:
    """The outcome window an action governs: k+1 (§23; frozen replay rule)."""
    return int(decision_window) + 1


def kpm_stamp_ms(outcome_window: int) -> int:
    """KPM stamp of an outcome window ``w``: ``(w + 1) * 100 ms``.

    Frozen convention (adjudication.py:343 ``stamp = (win_t + 1) * WINDOW_MS``;
    dataset.py:177 ``win = timestamp // 100 - 1``): outcome window ``w`` has
    its KPM rows stamped ``stamp(w) = (w + 1) * 100``.  The decision built on
    state window ``W`` governs outcome window ``W + 1``, whose KPI rows are
    therefore stamped ``(W + 2) * 100 ms``.
    """
    return (int(outcome_window) + 1) * DECISION_PERIOD_MS


def window_of_stamp(stamp_ms: int) -> int:
    """Inverse of :func:`kpm_stamp_ms` (dataset.py:177 ``win = ts//100 − 1``)."""
    return int(stamp_ms) // DECISION_PERIOD_MS - 1


def decision_window_of_stamp(stamp_ms: int) -> int:
    """The decision window whose action produced the rows with this stamp."""
    return window_of_stamp(stamp_ms) - 1


#: Fields that may never become features (guards test_no_future_information):
#: any attribute outside ``_RAW_FIELDS``/metadata slots is rejected.
_ALLOWED_SLOTS = frozenset(f.name for f in fields(JointState))
