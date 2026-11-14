"""Runtime glue for SQR-HET (B0) runs — D1, protocolo v2 §7.1.

Tails the ns-3 per-packet delay stream (``sqr_packet_delays.csv`` — flushed
per packet by ``sqr-sim.cc`` ``pktDelaysFile.flush()``), maintains the frozen
Gate II-d window aggregations needed by ``JointState``/``SliceOutcome``
(eMBB = MEDIAN windowThroughputMbps over demand UE-windows; URLLC =
p95/p99/miss% by ARRIVAL window floor(txSec·10); mMTC = 1 s ROLLING PDR)
and converts the env's parsed KPM windows into the SqrHetController inputs.

Runtime censoring (documented divergence from the offline replay): a
window's URLLC stats use only delay rows flushed by read time — packets
with delay > ~1 window (>100 ms) land later and are subcounted in the
STATE feature and in the update target.  Deadline is 10 ms, so every
packet relevant to the SLA (hits and misses ≤ 100 ms) is present; only
extreme outliers are censored.  Logged per decision (``u_censored``).
"""

from __future__ import annotations

import os
from collections import deque
from typing import Optional

import numpy as np

from nsoran.sqr.sqr_kpis import SqrKpmWindow
from nsoran.sqr_jc.joint_state import JointState, state_from_kpm_window
from nsoran.sqr_jc.sla_margin import SliceOutcome

_URLLC_SLICE = 1
_EMBB_SLICE = 0
_MMTc_SLICE = 2
DEADLINE_MS = 10.0


class DelayStream:
    """Incremental reader of ``sqr_packet_delays.csv`` (flushed per packet).

    Rows are buffered by ARRIVAL window ``win = floor(txSec*10)``; stats for
    a window are computed on demand from the rows seen so far (censoring
    documented in the module docstring).
    """

    def __init__(self, path: str) -> None:
        self.path = path          # absolute path OR glob (see HetRuntime)
        self._fh = None
        self._offset = 0
        #: column-name -> index map from the file's OWN header (the wire
        #: order is ``ueImsi,sliceId,txSec,delaySec`` — parse BY NAME, never
        #: by position; incident D1-v1: positional parsing read ueImsi as
        #: txSec and the URLLC stream came out empty).
        self._cols: dict[str, int] = {}
        #: win -> list of delay ms (URLLC rows only — sliceId == 1)
        self.by_win: dict[int, list[float]] = {}

    def poll(self) -> int:
        """Read newly flushed URLLC rows; return the count added."""
        import glob as _glob
        if self._fh is None:
            hits = ([self.path] if os.path.exists(self.path)
                    else sorted(_glob.glob(self.path, recursive=True)))
            if not hits:
                return 0
            self._fh = open(hits[0], "r", encoding="utf-8", errors="ignore")
            self._fh.seek(self._offset)
        added = 0
        for line in self._fh:
            parts = [p.strip() for p in line.split(",")]
            if not parts or not parts[0]:
                continue
            if not self._cols:
                # first non-empty line = header (whatever its order)
                if "txSec" in parts:
                    self._cols = {name: i for i, name in enumerate(parts)}
                    continue
            try:
                sid = int(float(parts[self._cols["sliceId"]]))
                tx = float(parts[self._cols["txSec"]])
                delay = float(parts[self._cols["delaySec"]])
            except (KeyError, IndexError, ValueError):
                continue
            if sid != _URLLC_SLICE:
                continue
            self.by_win.setdefault(int(np.floor(tx * 10.0)), []).append(
                delay * 1000.0)
            added += 1
        self._offset = self._fh.tell()
        return added

    def stats(self, win: int) -> Optional[dict]:
        """p50/p95/p99/miss/n of one arrival window (None if no packets)."""
        ms = self.by_win.get(int(win))
        if not ms:
            return None
        a = np.asarray(ms, dtype=float)
        return {
            "n_pkts": int(a.size),
            "p50_ms": float(np.percentile(a, 50.0)),
            "p95_ms": float(np.percentile(a, 95.0)),
            "p99_ms": float(np.percentile(a, 99.0)),
            "miss_rate": float(np.count_nonzero(a > DEADLINE_MS) / a.size),
        }


class HetRuntime:
    """Builds JointState/SliceOutcome for one SQR-HET run (D1)."""

    def __init__(self, sim_path: str) -> None:
        # ns-3 writes the delay stream at g_outputDir — the telemetry subdir
        # ``<sim_path>/results_sqr_network_only/seed=*/sqr_packet_delays.csv``
        # (two levels BELOW the sim uuid dir), so resolve by GLOB (incident
        # D1-v2: watching ``<sim_path>/sqr_packet_delays.csv`` never found
        # the file and the URLLC stream stayed empty for a whole campaign).
        self.delays = DelayStream(os.path.join(
            str(sim_path), "**", "sqr_packet_delays.csv"))
        #: (win, tx, rx) of the last 10 mMTC windows (R38 rolling PDR).
        self._mmtc_roll: deque[tuple[int, float, float]] = deque(maxlen=10)
        #: per-window eMBB demand presence (for R19 evidence flags).
        self.last_window_index: Optional[int] = None
        #: win -> {"embb_median"} cached at absorb (late-pending outcomes;
        #: mMTC rolling PDR é acumulado no deque, URLLC vem do stream).
        self._win_cache: dict[int, dict] = {}

    # -- ingestion ----------------------------------------------------------

    @staticmethod
    def window_index(kpm_window: SqrKpmWindow) -> int:
        """win = timestamp//100 − 1 (frozen dataset convention)."""
        return int(kpm_window.timestamp // 100) - 1

    def absorb(self, kpm_window: SqrKpmWindow) -> None:
        """Record one CLOSED window (call once per env step, in order).

        Caches the per-window KPIs needed by :meth:`outcome_for_window` so
        a PENDING decision can be observed LATE (after empty-cell holds)
        with the KPIs of its OWN governed window (Opção B, Emenda 3)."""
        w = self.window_index(kpm_window)
        tx = sum(r.d_tx_packets for r in kpm_window.rows
                 if r.slice_id == _MMTc_SLICE)
        rx = sum(r.d_rx_packets for r in kpm_window.rows
                 if r.slice_id == _MMTc_SLICE)
        # dedupe/reorder guard: keep one entry per window index
        if self._mmtc_roll and self._mmtc_roll[-1][0] == w:
            self._mmtc_roll.pop()
        self._mmtc_roll.append((w, float(tx), float(rx)))
        self.delays.poll()
        # rolling PDR AT W (deque é [W-9..W] neste instante — congelar no
        # cache; perguntar tarde devolve o valor da época de W)
        _txs = sum(t for _, t, _ in self._mmtc_roll)
        _rxs = sum(r for _, _, r in self._mmtc_roll)
        self._win_cache[w] = {
            "embb_median": self.embb_median(kpm_window),
            "mmtc_pdr": (_rxs / _txs) if _txs > 0 else None,
            "mmtc_demand": _txs > 0,
            # delayed-feedback probe (Emenda candidata): URLLC tx of W is
            # CLOSED KPM data — available online at W+1; delivered(W) grows
            # in the delay stream, so delivered==tx is an ONLINE-testable
            # pool-closure condition (no future data, no timeout ad hoc).
            "urllc_tx": sum(r.d_tx_packets for r in kpm_window.rows
                            if r.slice_id == _URLLC_SLICE),
        }
        self.last_window_index = w

    # -- frozen Gate II-d aggregations of ONE closed window ------------------

    def embb_median(self, kpm_window: SqrKpmWindow) -> Optional[float]:
        """Median windowThroughputMbps over demand eMBB UE-windows (R36)."""
        thr = [r.window_throughput_mbps for r in kpm_window.rows
               if r.slice_id == _EMBB_SLICE and r.d_tx_bytes > 0]
        return float(np.median(thr)) if thr else None

    def mmtc_rolling_pdr(self) -> tuple[Optional[float], bool]:
        """(1 s rolling PDR, demand flag) over the last 10 absorbed windows."""
        if not self._mmtc_roll:
            return None, False
        txs = sum(t for _, t, _ in self._mmtc_roll)
        rxs = sum(r for _, _, r in self._mmtc_roll)
        if txs <= 0:
            return None, False
        return rxs / txs, True

    def closed_kpis(self, kpm_window: SqrKpmWindow) -> dict:
        """closed_window_kpis of ONE window for state_from_kpm_window."""
        w = self.window_index(kpm_window)
        u = self.delays.stats(w)
        pdr, _ = self.mmtc_rolling_pdr()
        return {
            "embb_throughput_mbps": self.embb_median(kpm_window),
            "urllc_p95_ms": (u["p95_ms"] if u else None),
            "mmtc_pdr_rolling": pdr,
        }

    # -- state / outcome ------------------------------------------------------

    def joint_state(self, kpm_window: SqrKpmWindow,
                    prev_action: tuple[int, int, int],
                    prev_violations: tuple[int, int, int]) -> JointState:
        w = self.window_index(kpm_window)
        st = state_from_kpm_window(
            kpm_window,
            prev_action=prev_action,
            closed_window_kpis=self.closed_kpis(kpm_window),
            prev_violations=prev_violations,
            urllc_delay_stats=self.delays.stats(w) or {},
            window_index=w,
        )
        cch = self._win_cache.get(w) or {}
        u_tag, _ = self.urllc_pool_status(w)
        st.extras["kpm_tx_packets"] = (
            int(sum(r.d_tx_packets for r in kpm_window.rows if r.slice_id == _EMBB_SLICE)),
            int(cch.get("urllc_tx") or 0),
            int(sum(r.d_tx_packets for r in kpm_window.rows if r.slice_id == _MMTc_SLICE)),
        )
        st.extras["urllc_pool_open"] = (u_tag == "OPEN")
        st.extras["mmtc_rolling_active"] = bool(cch.get("mmtc_demand", False))
        return st

    def outcome_for_window(self, w: int) -> SliceOutcome:
        """SliceOutcome of window ``w`` — PODE ser perguntado tarde.

        Atraso de entrega (bug v4/v5): pacotes de W são entregues/flushados
        até ~700 ms (p90) após o fim de W.  O stream de delays é
        ACUMULATIVO por janela de chegada (arquivo append-only), então
        ``stats(w)`` pergunta TARDE devolve o outcome COMPLETO da janela —
        Opção B (Emenda 3): pendentes observados contra a PRÓPRIA janela
        governada, sem censura quando o gap entre perguntas é longo.
        eMBB/mMTC vêm do cache do absorb (KPM fechado de W).
        """
        self.delays.poll()
        u = self.delays.stats(w)
        c = self._win_cache.get(w) or {}
        pdr, demand = c.get("mmtc_pdr"), c.get("mmtc_demand", False)
        return SliceOutcome(
            embb_throughput_mbps=c.get("embb_median"),
            urllc_p95_ms=(u["p95_ms"] if u else None),
            urllc_miss_pct=(100.0 * u["miss_rate"]) if u else None,
            mmtc_pdr_rolling=(pdr if demand else None),
        )

    def outcome(self, kpm_window: SqrKpmWindow) -> SliceOutcome:
        """SliceOutcome of the window this kpm_window closes (delegate)."""
        return self.outcome_for_window(self.window_index(kpm_window))

    # -- delayed feedback (Emenda candidata — flag-only, canônica intacta) --

    def urllc_pool_status(self, w: int) -> tuple[str, Optional[dict]]:
        """Online pool-closure test for window ``w`` (no future data).

        Returns ``(tag, stats)`` where tag is one of
        ``NO_DEMAND``  — KPM tx of W is 0 (R19: vacuous, never a label);
        ``OPEN``       — delivered < tx: pool not closed yet (keep pending);
        ``CLOSED``     — delivered == tx: every tx of W was delivered and
                         is already flushed ⇒ ``stats`` is the COMPLETE
                         URLLC stats of W (valid margin, R19-consistent).
        Perda real (delivered < tx para sempre) ⇒ never CLOSED ⇒ pending
        expires at MAX age and is discarded as vacuous (R19: sem-RX nunca
        é miss nem sucesso implícito).
        """
        self.delays.poll()
        c = self._win_cache.get(w)
        if c is None:
            return "NO_KPM", None
        tx = int(c.get("urllc_tx") or 0)
        if tx == 0:
            return "NO_DEMAND", None
        n = len(self.delays.by_win.get(int(w), ()))
        if n < tx:
            return "OPEN", None
        return "CLOSED", self.delays.stats(w)

    def urllc_only_outcome(self, w: int) -> Optional[SliceOutcome]:
        """SliceOutcome com APENAS a margem URLLC (pool CLOSED) — os outros
        dois slices ficam None ⇒ R19 os exclui ⇒ observe_outcome atualiza
        SÓ o modelo URLLC, 1× por janela, sem duplicar eMBB/mMTC."""
        tag, stats = self.urllc_pool_status(w)
        if tag != "CLOSED":
            return None
        return SliceOutcome(
            embb_throughput_mbps=None,
            urllc_p95_ms=(stats["p95_ms"] if stats else None),
            urllc_miss_pct=(100.0 * stats["miss_rate"]) if stats else None,
            mmtc_pdr_rolling=None,
        )

    # -- R19 evidence flags (§4.2 log) ---------------------------------------

    def evidence(self, w: int) -> list[str]:
        """R19 evidence flags da JANELA ``w`` (índice; Opção B por época).

        eMBB/mMTC do cache do absorb (KPM fechado de W); URLLC do stream.
        Semântica do URLLC (correção do ofício pré-D3, 13/11 — t1420):
        ``stats(w)=None`` com demanda real no KPM (urllc_tx>0) é
        NOT_YET_OBSERVED (resultado ainda não flushado — censura lag=1),
        NÃO ausência de demanda; ``NO_ACTIVE_DEMAND`` só quando o KPM da
        janela não registrou tráfego URLLC. Rótulo de diagnóstico apenas:
        nenhum julgamento (frozen/STRICT/R19) depende dele."""
        self.delays.poll()
        c = self._win_cache.get(w) or {}
        ev = []
        ev.append("COMPLETE" if c.get("embb_median") is not None
                  else "NO_ACTIVE_DEMAND")
        if self.delays.stats(w):
            ev.append("COMPLETE")
        elif c.get("urllc_tx"):
            ev.append("PENDING_OBSERVATION")
        else:
            ev.append("NO_ACTIVE_DEMAND")
        ev.append("COMPLETE" if c.get("mmtc_demand") else "NO_ACTIVE_DEMAND")
        return ev
