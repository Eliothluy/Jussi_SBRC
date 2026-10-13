"""KPM wrapper for the SQR scenario — parses ``sqr-kpms.txt`` (contract §4).

The ns-3 ``sqr-sim`` scenario (Fase 1, Agente A) writes ONE line per active
UE per 100 ms decision window, with the 13 frozen columns:

    timestamp,ueImsi,sliceId,dTxBytes,dRxBytes,plr,resourceSharePct,
    dLostPackets,throughputMbps,bufferBytes,windowThroughputMbps,snrDb,
    queueBytes

(columns 1-10 = legacy rslaq header; 11-13 = SQR extensions, contract §4).

This wrapper is NEW code in ``src/nsoran/sqr/`` — the core
``environments.rslaq_kpis`` module is untouched (AGENTS.md §5 rule 2).

Frozen behaviors implemented here (registers R09/R18/R19):

- **SNR sentinel −999 never reaches the state.**  Wire sentinel ``snrDb =
  -999`` (or an unparsable/absent field) is imputed deterministically:
  (1) the UE's last valid snrDb (carried across windows); (2) if none, the
  cell median of the PREVIOUS window (valid values only); (3) if none,
  15 dB (scale centre).  Every imputation is counted and classified
  (``carried`` / ``cell_median`` / ``default_15``) — the per-window
  imputation rate is logged, never silently absorbed.
- **y_t = 5th percentile of per-UE windowThroughputMbps** within the slice
  (linear interpolation between order statistics, register R18); a slice
  with no UE in the window has y_t = None and is EXCLUDED from violation
  denominators (register R19).
- **Per-slice derivations** (nothing of this goes into the file):
  lowest-user-throughput, p5, UE count, queue-bytes sum, per-UE violation
  rate against the slice SLA, and the cell median of valid SNRs (fed back
  as the next window's imputation fallback).
"""

from __future__ import annotations

import csv
import os
import warnings
from dataclasses import dataclass, field

import numpy as np

#: Frozen column order of sqr-kpms.txt (contract §4).
SQR_KPM_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "ueImsi",
    "sliceId",
    "dTxBytes",
    "dRxBytes",
    "plr",
    "resourceSharePct",
    "dLostPackets",
    "throughputMbps",
    "bufferBytes",
    "windowThroughputMbps",
    "snrDb",
    "queueBytes",
)

#: Wire sentinel for "no SNR available" (contract §4).  NEVER fed to the
#: state — clip(-999/30, 0, 1) = 0 would be indistinguishable from a real
#: 0 dB SNR (register R09, correção v1.1).
SNR_SENTINEL = -999.0

#: Last-resort SNR imputation: centre of the snr_norm = clip(snr/30, 0, 1)
#: scale (register R09).
DEFAULT_IMPUTED_SNR_DB = 15.0

#: Imputation method labels (also log values).
IMP_CARRIED = "carried"          # UE's own last valid snrDb
IMP_CELL_MEDIAN = "cell_median"  # previous window's cell median (valid only)
IMP_DEFAULT = "default_15"       # 15 dB fallback

_NUM_SLICES = 3
_VALID_SLICE_IDS = frozenset(range(_NUM_SLICES))


@dataclass(frozen=True)
class UeKpmRow:
    """One parsed UE line with the IMPUTED snr (state-safe, never sentinel)."""

    timestamp: int
    ue_imsi: int
    slice_id: int
    d_tx_bytes: float
    d_rx_bytes: float
    plr: float
    resource_share_pct: float
    d_lost_packets: float
    throughput_mbps: float
    buffer_bytes: float
    window_throughput_mbps: float
    snr_db: float              # imputed value; guaranteed != SNR_SENTINEL
    snr_was_imputed: bool
    snr_imputation: str | None  # IMP_* label, None when the wire value was used
    queue_bytes: float


@dataclass(frozen=True)
class SliceKpis:
    """Per-slice derivations of one KPM window (registers R18/R19)."""

    slice_id: int
    n_ues: int
    lowest_thr_mbps: float | None       # min windowThroughputMbps (None if empty)
    p5_thr_mbps: float | None           # y_t: 5th percentile (None if empty)
    mean_thr_mbps: float | None
    queue_bytes_sum: float
    violation_rate: float | None        # per-UE rate, None if no UE (R19)
    n_snr_imputed: int
    snr_imputations: dict[str, int] = field(default_factory=dict)
    snr_values: tuple[float, ...] = ()  # imputed per-UE SNRs, in row order


@dataclass(frozen=True)
class SqrKpmWindow:
    """Parsed view of one 100 ms KPM window."""

    timestamp: int
    slices: dict[int, SliceKpis]
    total_ues: int
    cell_snr_median: float | None  # median of VALID snrDb values, for the
                                   # NEXT window's imputation fallback
    rows: tuple[UeKpmRow, ...]


class SqrKpmParser:
    """Stateful parser: feeds ``sqr-kpms.txt`` windows one by one.

    Holds the cross-window state required by the frozen SNR imputation
    (per-UE last valid snrDb and the previous window's cell median) and the
    ``last_timestamp`` watermark so replayed/partial files only yield NEW
    rows (same contract as ``rslaq_kpis.parse_kpm_file``).
    """

    def __init__(
        self,
        num_slices: int = _NUM_SLICES,
        sla_thresholds: dict[int, float] | None = None,
        last_timestamp: int = 0,
    ) -> None:
        self.num_slices = int(num_slices)
        if self.num_slices <= 0:
            raise ValueError(f"num_slices must be positive, got {num_slices}")
        self.sla_thresholds = dict(sla_thresholds or {})
        self.last_timestamp = int(last_timestamp)
        self._last_valid_snr: dict[int, float] = {}   # per UE, across windows
        self._previous_cell_median: float | None = None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_window(self, kpm_path: str) -> SqrKpmWindow:
        """Parse the current file contents into one :class:`SqrKpmWindow`.

        Rows with ``timestamp < last_timestamp`` are skipped (watermark);
        the watermark advances to the newest timestamp seen.
        """
        raw_rows: list[dict] = []
        latest_ts = self.last_timestamp
        raw_slice_ids: set[int] = set()

        if os.path.exists(kpm_path):
            with open(kpm_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = int(row["timestamp"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if ts < self.last_timestamp:
                        continue
                    try:
                        raw_slice_ids.add(int(row.get("sliceId", -1)))
                    except (TypeError, ValueError):
                        continue
                    raw_rows.append(row)
                    latest_ts = max(latest_ts, ts)

        # Legacy 1-based detection follows rslaq_kpis: only when NO row is
        # 0-based AND id 3 appears (a window with a single slice-1 row must
        # NOT be mistaken for a 1-based window).
        legacy_one_based = (
            0 not in raw_slice_ids and 3 in raw_slice_ids
        )
        if legacy_one_based:
            warnings.warn(
                "Detected legacy 1-based slice ids in the SQR KPM file; "
                "converting to 0-based ids.",
                stacklevel=2,
            )

        rows: list[UeKpmRow] = []
        current_window_valid_snr: list[float] = []
        for row in raw_rows:
            parsed = self._parse_row(row, legacy_one_based)
            if parsed is None:
                continue
            if parsed.snr_was_imputed is False:
                current_window_valid_snr.append(parsed.snr_db)
            rows.append(parsed)

        self.last_timestamp = latest_ts

        by_slice: dict[int, list[UeKpmRow]] = {
            sid: [] for sid in range(self.num_slices)
        }
        for r in rows:
            by_slice.setdefault(r.slice_id, []).append(r)

        slices: dict[int, SliceKpis] = {}
        for sid in range(self.num_slices):
            slices[sid] = self._summarize(sid, by_slice.get(sid, []))

        cell_median = (
            float(np.median(current_window_valid_snr))
            if current_window_valid_snr
            else None
        )
        # The median becomes the fallback of the NEXT window only.
        self._previous_cell_median = cell_median

        return SqrKpmWindow(
            timestamp=latest_ts,
            slices=slices,
            total_ues=len(rows),
            cell_snr_median=cell_median,
            rows=tuple(rows),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_row(self, row: dict, legacy_one_based: bool) -> UeKpmRow | None:
        try:
            ts = int(row["timestamp"])
            ue_imsi = int(row["ueImsi"])
            raw_sid = int(row.get("sliceId", -1))
        except (KeyError, TypeError, ValueError):
            return None
        sid = raw_sid - 1 if legacy_one_based else raw_sid
        if sid not in range(self.num_slices):
            return None

        def _f(name: str, default: float = 0.0) -> float:
            try:
                v = float(row.get(name, default))
            except (TypeError, ValueError):
                return default
            return v if np.isfinite(v) else default

        snr_raw, snr_valid = self._impute_snr(ue_imsi, row.get("snrDb"))
        return UeKpmRow(
            timestamp=ts,
            ue_imsi=ue_imsi,
            slice_id=sid,
            d_tx_bytes=_f("dTxBytes"),
            d_rx_bytes=_f("dRxBytes"),
            plr=_f("plr"),
            resource_share_pct=_f("resourceSharePct"),
            d_lost_packets=_f("dLostPackets"),
            throughput_mbps=_f("throughputMbps"),
            buffer_bytes=_f("bufferBytes"),
            window_throughput_mbps=_f("windowThroughputMbps"),
            snr_db=snr_raw,
            snr_was_imputed=not snr_valid,
            snr_imputation=None if snr_valid else self._last_method,
            queue_bytes=_f("queueBytes"),
        )

    def _impute_snr(self, ue_imsi: int, wire_value) -> tuple[float, bool]:
        """Frozen imputation chain of register R09 / contract §4.

        Returns ``(snr_db, was_valid)``; the returned value is NEVER the
        sentinel.
        """
        try:
            v = float(wire_value)
        except (TypeError, ValueError):
            v = SNR_SENTINEL
        if np.isfinite(v) and v != SNR_SENTINEL:
            self._last_valid_snr[ue_imsi] = v
            self._last_method = None
            return v, True

        # 1) UE's own last valid value (carried across windows).
        carried = self._last_valid_snr.get(ue_imsi)
        if carried is not None:
            self._last_method = IMP_CARRIED
            return float(carried), False
        # 2) Cell median of the PREVIOUS window (valid values only).
        if self._previous_cell_median is not None:
            self._last_method = IMP_CELL_MEDIAN
            return float(self._previous_cell_median), False
        # 3) Scale centre.
        self._last_method = IMP_DEFAULT
        return DEFAULT_IMPUTED_SNR_DB, False

    def _summarize(self, slice_id: int, rows: list[UeKpmRow]) -> SliceKpis:
        n = len(rows)
        imputations: dict[str, int] = {}
        for r in rows:
            if r.snr_was_imputed:
                key = r.snr_imputation or IMP_DEFAULT
                imputations[key] = imputations.get(key, 0) + 1

        if n == 0:
            # Empty slice: y_t undefined, period excluded from the slice's
            # violation denominator (register R19).
            return SliceKpis(
                slice_id=slice_id,
                n_ues=0,
                lowest_thr_mbps=None,
                p5_thr_mbps=None,
                mean_thr_mbps=None,
                queue_bytes_sum=0.0,
                violation_rate=None,
                n_snr_imputed=0,
                snr_imputations={},
                snr_values=(),
            )

        thr = np.asarray([r.window_throughput_mbps for r in rows], dtype=np.float64)
        lam = self.sla_thresholds.get(slice_id)
        if lam is None:
            violation_rate = None  # no SLA configured for this slice
        else:
            violation_rate = float(np.count_nonzero(thr < float(lam)) / n)

        return SliceKpis(
            slice_id=slice_id,
            n_ues=n,
            lowest_thr_mbps=float(np.min(thr)),
            # R18: linear interpolation between order statistics.
            p5_thr_mbps=float(np.percentile(thr, 5.0, method="linear")),
            mean_thr_mbps=float(np.mean(thr)),
            queue_bytes_sum=float(sum(r.queue_bytes for r in rows)),
            violation_rate=violation_rate,
            n_snr_imputed=int(sum(imputations.values())),
            snr_imputations=imputations,
            snr_values=tuple(r.snr_db for r in rows),
        )


def parse_sqr_kpm_file(
    kpm_path: str,
    sla_thresholds: dict[int, float] | None = None,
    last_timestamp: int = 0,
) -> SqrKpmWindow:
    """One-shot stateless parse (fresh imputation state)."""
    return SqrKpmParser(
        sla_thresholds=sla_thresholds, last_timestamp=last_timestamp
    ).parse_window(kpm_path)
