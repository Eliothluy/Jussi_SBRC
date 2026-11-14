"""Common normalized SLA margin for the heterogeneous slices (protocol §7/§8).

All SQR-JC heads predict the SAME quantity per slice: the normalized SLA
margin m_i, so that

    m_i > 0  -> SLA satisfied
    m_i = 0  -> boundary
    m_i < 0  -> SLA violated

The KPIs, units and thresholds below were EXTRACTED FROM THE FROZEN ROUND-2
CODE (protocol §8 hard gate: "Inventar ou alterar SLA durante a
implementação" is forbidden; §7: "Usar somente uma definição primária,
conforme o código auditado"):

- eMBB  (R36): windowThroughputMbps per UE-window of 100 ms with demand;
  operational SLA 3.5 Mbps/UE (7 Mbps is the DORA source target).  The
  frozen Gate 2.4-II-d window judgment is the MEDIAN over demand UE-windows
  of the governed window: ``embb_throughput >= 3.5``
  (oracle/adjudication.py::_sla_oks; dataset.py::SLA_EMBB_MBPS).
- URLLC (R37): payload 32 B @ 1 ms; experimental deadline 10 ms app-to-app;
  frozen Gate 2.4-II-d window judgment is the per-window p95 of URLLC packet
  delays grouped by ARRIVAL window: ``p95 <= 10 ms``
  (oracle/adjudication.py:407-408; oracle/models.py:222; constants
  oracle/dataset.py:71-73).  The deadline-miss ratio
  (fraction of packets with delaySec > 0.010) rides along as a REPORTED
  metric only — it was never a gate in the audited Round-2 code.
- mMTC  (R38): payload 200 B @ 1000 ms; SLA = delivery PDR >= 95%; the
  frozen window analog is the 1 s ROLLING PDR
  (oracle/adjudication.py:384-390; dataset.py add_rolling_pdr).

**DIVERGÊNCIA REGISTRADA (protocolo §0 — nunca corrigir silenciosamente):**
o despacho PRE-FLIGHT pedia "URLLC miss-rate ≤5% OU p95 ≤10 ms (USE O QUE O
GATE II-d USOU: miss≤5)". O código congelado do Gate II-d NÃO usou miss≤5:
usou **p95 ≤ 10 ms** como julgamento de janela (adjudication.py:404-412,
models.py:222); ``miss_pct`` é métrica de relatório em todo o código da
Rodada 2.  O protocolo §7 manda usar a definição primária "conforme o código
auditado" — portanto a primária aqui é p95 (10 ms), e a variante miss-rate
fica registrada como SECUNDÁRIA (never a gate).  Qualquer troca da primária
exige nova decisão do orquestrador e invalida o congelamento.

Vacuous windows (no demand in the slice) are SATISFIED — the audited Gate
II-d semantics: ``urllc_ok = True if urllc_n_pkts < 1 else ...``
(adjudication.py:406-410), register R19's exclusion of empty slices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Frozen SLA constants (Round 2, registers R36–R38; oracle/dataset.py:71-73)
# ---------------------------------------------------------------------------

#: eMBB operational throughput SLA per UE-window median, Mbps (R36 as amended;
#: source target 7 Mbps = DORA Eq. (5); 3.5 = 53/106 PRB linear scaling).
SLA_EMBB_MBPS = 3.5

#: URLLC experimental deadline, ms (R37); the frozen window judgment is
#: p95 <= this value.
SLA_URLLC_MS = 10.0

#: mMTC delivery PDR SLA (R38): 1 s rolling PDR >= 0.95.
SLA_MMTC_PDR = 0.95

#: URLLC deadline-miss secondary threshold in percent (despacho pre-flight
#: "miss<=5" — SECONDARY only; never a gate in the audited Round-2 code).
SLA_URLLC_MISS_PCT_SECONDARY = 5.0

#: mMTC PDR is already a fraction in [0, 1] — natural normalization.
MMTC_PDR_IS_FRACTION = True

# ---------------------------------------------------------------------------
# Margin formulas (§7) — one primary definition per slice, audited code.
# ---------------------------------------------------------------------------


def embb_margin(embb_throughput_mbps: Optional[float]) -> Optional[float]:
    """m_e = (T_e − SLA_e) / SLA_e; None = vacuous (no demand, satisfied).

    ``embb_throughput_mbps`` is EXACTLY the frozen Gate II-d KPI: the median
    of ``windowThroughputMbps`` over demand eMBB UE-windows of the governed
    window (window stamp (t+1)·100 ms).
    """
    if embb_throughput_mbps is None:
        return None
    return (float(embb_throughput_mbps) - SLA_EMBB_MBPS) / SLA_EMBB_MBPS


def urllc_margin(urllc_p95_ms: Optional[float]) -> Optional[float]:
    """m_u = (D_SLA − D_u) / D_SLA with D_u = window p95 latency in ms.

    Primary definition per the audited Gate II-d (p95 <= 10 ms); None =
    vacuous (no URLLC packets in the window, satisfied).
    """
    if urllc_p95_ms is None:
        return None
    return (SLA_URLLC_MS - float(urllc_p95_ms)) / SLA_URLLC_MS


def urllc_margin_miss_secondary(miss_pct: Optional[float]) -> Optional[float]:
    """SECONDARY margin m_u^miss = (MR_SLA − MR_u) / MR_SLA (miss% scale).

    Registered because the pre-flight dispatch mentioned "miss<=5"; the
    audited Round-2 code never gated on it.  Do NOT use as the primary.
    """
    if miss_pct is None:
        return None
    return (SLA_URLLC_MISS_PCT_SECONDARY - float(miss_pct)) / SLA_URLLC_MISS_PCT_SECONDARY


def mmtc_margin(mmtc_pdr: Optional[float]) -> Optional[float]:
    """m_m = (PDR_m − PDR_SLA) / PDR_SLA; None = vacuous (no tx, satisfied).

    ``mmtc_pdr`` is the 1 s ROLLING PDR of the slice (frozen Gate II-d
    window analog of the R38 session PDR).
    """
    if mmtc_pdr is None:
        return None
    return (float(mmtc_pdr) - SLA_MMTC_PDR) / SLA_MMTC_PDR


# ---------------------------------------------------------------------------
# Window judgment (identical predicates to the audited Gate II-d)
# ---------------------------------------------------------------------------


def embb_sla_ok(embb_throughput_mbps: Optional[float]) -> bool:
    """Vacuous when no demand (Gate II-d: ``n_embb < 1`` -> True)."""
    return embb_throughput_mbps is None or float(embb_throughput_mbps) >= SLA_EMBB_MBPS


def urllc_sla_ok(urllc_p95_ms: Optional[float]) -> bool:
    """Vacuous when no packets (Gate II-d: ``urllc_n_pkts < 1`` -> True)."""
    return urllc_p95_ms is None or float(urllc_p95_ms) <= SLA_URLLC_MS


def mmtc_sla_ok(mmtc_pdr: Optional[float]) -> bool:
    """Vacuous when no tx (Gate II-d: ``mmtc_demand`` False -> True)."""
    return mmtc_pdr is None or float(mmtc_pdr) >= SLA_MMTC_PDR


def severity(margin: Optional[float]) -> float:
    """Violation severity of §63: −m_i when m_i < 0, else 0 (comparable)."""
    if margin is None:
        return 0.0
    return max(0.0, -float(margin))


@dataclass(frozen=True)
class SliceOutcome:
    """Observed per-slice outcome of one governed window (§54 input)."""

    #: frozen Gate II-d window KPIs (None = vacuous / no demand)
    embb_throughput_mbps: Optional[float] = None
    urllc_p95_ms: Optional[float] = None
    urllc_miss_pct: Optional[float] = None   # secondary, log-only
    mmtc_pdr_rolling: Optional[float] = None

    @property
    def margin_embb(self) -> Optional[float]:
        return embb_margin(self.embb_throughput_mbps)

    @property
    def margin_urllc(self) -> Optional[float]:
        return urllc_margin(self.urllc_p95_ms)

    @property
    def margin_mmtc(self) -> Optional[float]:
        return mmtc_margin(self.mmtc_pdr_rolling)

    @property
    def margins(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """(m_e, m_u, m_m) in slice order (0=eMBB, 1=URLLC, 2=mMTC)."""
        return (self.margin_embb, self.margin_urllc, self.margin_mmtc)

    @property
    def sla_ok(self) -> tuple[bool, bool, bool]:
        """(eMBB_ok, URLLC_ok, mMTC_ok) with the frozen Gate II-d predicates."""
        return (
            embb_sla_ok(self.embb_throughput_mbps),
            urllc_sla_ok(self.urllc_p95_ms),
            mmtc_sla_ok(self.mmtc_pdr_rolling),
        )

    @property
    def joint_ok(self) -> bool:
        return all(self.sla_ok)
