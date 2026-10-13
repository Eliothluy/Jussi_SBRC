"""Tests for nsoran.sqr.sqr_kpis (KPM wrapper for sqr-kpms.txt, contract §4).

Checklist: 13-column contract parse; sentinel -999 NEVER reaches the state
(imputation chain carried -> cell_median -> 15 dB, register R09); per-slice
derivations (lowest, p5 y_t with linear interpolation, UE count, queue sum);
per-UE violation rate with empty-slice exclusion (register R19); watermark
(last_timestamp); legacy 1-based slice ids; malformed rows skipped.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from nsoran.sqr.sqr_kpis import (
    DEFAULT_IMPUTED_SNR_DB,
    IMP_CARRIED,
    IMP_CELL_MEDIAN,
    IMP_DEFAULT,
    SNR_SENTINEL,
    SQR_KPM_COLUMNS,
    SqrKpmParser,
    parse_sqr_kpm_file,
)

SLAS = {0: 0.5, 1: 1.0, 2: 1.5}


def _write_kpm(path, rows, header=SQR_KPM_COLUMNS):
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(v) for v in row))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _row(ts=100, imsi=1, sid=0, wthr=1.0, snr=20.0, queue=1000, **over):
    base = [
        ts, imsi, sid,
        over.get("dTxBytes", 15000), over.get("dRxBytes", 12500),
        over.get("plr", 0.0), over.get("resourceSharePct", 33.3),
        over.get("dLostPackets", 0), over.get("throughputMbps", wthr),
        over.get("bufferBytes", 2000), wthr, snr, queue,
    ]
    return base


# ── Contract parse: 13 columns, per-slice derivations ───────────────

def test_header_columns_match_contract():
    assert SQR_KPM_COLUMNS == (
        "timestamp", "ueImsi", "sliceId", "dTxBytes", "dRxBytes", "plr",
        "resourceSharePct", "dLostPackets", "throughputMbps", "bufferBytes",
        "windowThroughputMbps", "snrDb", "queueBytes",
    )


def test_per_slice_derivations(tmp_path):
    path = _write_kpm(tmp_path / "sqr-kpms.txt", [
        _row(ts=100, imsi=1, sid=0, wthr=0.4, snr=18.0, queue=100_000),
        _row(ts=100, imsi=2, sid=0, wthr=1.2, snr=22.0, queue=300_000),
        _row(ts=100, imsi=3, sid=1, wthr=0.9, snr=25.0, queue=50),
        _row(ts=100, imsi=4, sid=2, wthr=1.6, snr=12.0, queue=7),
    ])
    w = parse_sqr_kpm_file(str(path), sla_thresholds=SLAS)

    assert w.timestamp == 100
    assert w.total_ues == 4
    s0 = w.slices[0]
    assert s0.n_ues == 2
    assert s0.lowest_thr_mbps == pytest.approx(0.4)
    # R18: p5 with linear interpolation of {0.4, 1.2}.
    expected_p5 = float(np.percentile([0.4, 1.2], 5.0, method="linear"))
    assert s0.p5_thr_mbps == pytest.approx(expected_p5)
    assert s0.queue_bytes_sum == pytest.approx(400_000)
    assert s0.violation_rate == pytest.approx(0.5)  # 0.4 < 0.5, 1.2 >= 0.5
    assert s0.n_snr_imputed == 0
    assert w.slices[1].p5_thr_mbps == pytest.approx(0.9)
    assert w.slices[1].violation_rate == pytest.approx(1.0)  # 0.9 < 1.0 SLA
    assert w.slices[2].violation_rate == pytest.approx(0.0)  # 1.6 >= 1.5
    assert w.cell_snr_median == pytest.approx(np.median([18.0, 22.0, 25.0, 12.0]))


def test_empty_slice_y_t_undefined_and_excluded(tmp_path):
    """Register R19: a slice with no UE has p5 = None and is excluded from
    the pooled denominator."""
    path = _write_kpm(tmp_path / "sqr-kpms.txt", [
        _row(ts=100, imsi=1, sid=0, wthr=0.2, snr=10.0),
    ])
    w = parse_sqr_kpm_file(str(path), sla_thresholds=SLAS)
    for sid in (1, 2):
        assert w.slices[sid].n_ues == 0
        assert w.slices[sid].p5_thr_mbps is None
        assert w.slices[sid].violation_rate is None
    assert w.slices[0].violation_rate == 1.0


# ── SNR sentinel: the frozen imputation chain (register R09) ────────

def test_sentinel_never_reaches_state_first_window_default_15(tmp_path):
    """First window, no imputation history: the sentinel row falls through
    the whole chain to the 15 dB default; the valid row passes through."""
    path = _write_kpm(tmp_path / "sqr-kpms.txt", [
        _row(ts=100, imsi=1, sid=0, snr=SNR_SENTINEL),
        _row(ts=100, imsi=2, sid=0, snr=20.0),
    ])
    w = parse_sqr_kpm_file(str(path), sla_thresholds=SLAS)
    s0 = w.slices[0]
    assert all(v != SNR_SENTINEL for v in s0.snr_values)
    assert s0.n_snr_imputed == 1
    assert s0.snr_imputations == {IMP_DEFAULT: 1}
    assert s0.snr_values[0] == DEFAULT_IMPUTED_SNR_DB  # 15 dB fallback
    assert s0.snr_values[1] == 20.0


def test_imputation_chain_carried_then_cell_median(tmp_path):
    """Window 1: UE1 valid 20 dB, UE2 sentinel (no history) -> 15 dB.
    Window 2: UE1 sentinel -> carried 20; UE2 sentinel again -> previous
    window's cell median (20); new UE3 sentinel -> also the median."""
    path = _write_kpm(tmp_path / "sqr-kpms.txt", [
        _row(ts=100, imsi=1, sid=0, snr=20.0),
        _row(ts=100, imsi=2, sid=0, snr=SNR_SENTINEL),
    ])
    parser = SqrKpmParser(sla_thresholds=SLAS)
    w1 = parser.parse_window(str(path))
    assert w1.slices[0].snr_values[0] == 20.0
    assert w1.slices[0].snr_values[1] == DEFAULT_IMPUTED_SNR_DB

    path2 = _write_kpm(tmp_path / "sqr-kpms2.txt", [
        _row(ts=200, imsi=1, sid=0, snr=SNR_SENTINEL),
        _row(ts=200, imsi=2, sid=0, snr=SNR_SENTINEL),
        _row(ts=200, imsi=3, sid=0, snr=SNR_SENTINEL),
    ])
    w2 = parser.parse_window(str(path2))
    s0 = w2.slices[0]
    assert s0.snr_values[0] == 20.0                       # carried (own UE)
    assert s0.snr_imputations[IMP_CARRIED] == 1
    # UE2 and UE3: no own valid value -> cell median of the previous window.
    assert s0.snr_values[1] == pytest.approx(20.0)
    assert s0.snr_values[2] == pytest.approx(20.0)
    assert s0.snr_imputations.get(IMP_CELL_MEDIAN) == 2


def test_cell_median_uses_valid_values_only(tmp_path):
    """The previous window's median must be computed over VALID snrDb
    values only (sentinels excluded)."""
    path = _write_kpm(tmp_path / "sqr-kpms.txt", [
        _row(ts=100, imsi=1, sid=0, snr=10.0),
        _row(ts=100, imsi=2, sid=0, snr=30.0),
        _row(ts=100, imsi=3, sid=1, snr=SNR_SENTINEL),
    ])
    parser = SqrKpmParser(sla_thresholds=SLAS)
    w1 = parser.parse_window(str(path))
    assert w1.cell_snr_median == pytest.approx(20.0)  # median(10, 30)

    path2 = _write_kpm(tmp_path / "sqr-kpms2.txt", [
        _row(ts=200, imsi=9, sid=1, snr=SNR_SENTINEL),
    ])
    w2 = parser.parse_window(str(path2))
    assert w2.slices[1].snr_values[0] == pytest.approx(20.0)


def test_median_fallback_only_from_previous_window_not_current(tmp_path):
    """A valid value later IN THE SAME window must not be used to impute an
    earlier sentinel row (fallback is the PREVIOUS window's median)."""
    path = _write_kpm(tmp_path / "sqr-kpms.txt", [
        _row(ts=100, imsi=1, sid=0, snr=SNR_SENTINEL),  # no history at all
        _row(ts=100, imsi=2, sid=0, snr=25.0),          # valid, same window
    ])
    w = parse_sqr_kpm_file(str(path), sla_thresholds=SLAS)
    assert w.slices[0].snr_values[0] == DEFAULT_IMPUTED_SNR_DB  # 15, not 25


# ── Watermark, legacy ids, malformed rows ───────────────────────────

def test_watermark_skips_old_rows(tmp_path):
    """Rows strictly older than the watermark are skipped (inherited
    rslaq contract: ``ts < last_timestamp``); newer rows parse."""
    path = _write_kpm(tmp_path / "sqr-kpms.txt", [
        _row(ts=100, imsi=1, sid=0, wthr=0.4),
        _row(ts=200, imsi=1, sid=0, wthr=0.8),
    ])
    parser = SqrKpmParser(sla_thresholds=SLAS, last_timestamp=150)
    w = parser.parse_window(str(path))
    assert w.total_ues == 1            # only ts=200
    assert w.timestamp == 200
    assert parser.last_timestamp == 200
    assert w.slices[0].lowest_thr_mbps == pytest.approx(0.8)


def test_legacy_one_based_slice_ids_converted(tmp_path):
    path = _write_kpm(tmp_path / "sqr-kpms.txt", [
        _row(ts=100, imsi=1, sid=1, wthr=0.4),  # legacy slice 1 -> 0
        _row(ts=100, imsi=2, sid=3, wthr=0.8),  # legacy slice 3 -> 2
    ])
    w = parse_sqr_kpm_file(str(path), sla_thresholds=SLAS)
    assert w.slices[0].n_ues == 1
    assert w.slices[2].n_ues == 1
    assert w.slices[1].n_ues == 0


def test_malformed_rows_skipped(tmp_path):
    path = tmp_path / "sqr-kpms.txt"
    lines = [
        ",".join(SQR_KPM_COLUMNS),
        "abc,1,0,1,1,0,1,0,1,1,1.0,20,10",          # bad timestamp
        "100,notanimsi,0,1,1,0,1,0,1,1,1.0,20,10",  # bad imsi
        "100,7,9,1,1,0,1,0,1,1,1.0,20,10",          # invalid slice id
        "100,8,0,1,1,0,1,0,1,1,1.0,nan,10",         # NaN snr -> imputed
        "100,9,0,1,1,0,1,0,1,1,1.5,20,10",          # good row
    ]
    path.write_text("\n".join(lines) + "\n")
    w = parse_sqr_kpm_file(str(path), sla_thresholds=SLAS)
    assert w.total_ues == 2
    assert w.slices[0].n_ues == 2
    assert w.slices[0].snr_values[0] == DEFAULT_IMPUTED_SNR_DB
    assert w.slices[0].snr_values[1] == 20.0


def test_missing_file_returns_empty_window():
    w = parse_sqr_kpm_file("/nonexistent/sqr-kpms.txt", sla_thresholds=SLAS)
    assert w.total_ues == 0
    assert w.timestamp == 0
    assert all(w.slices[sid].p5_thr_mbps is None for sid in range(3))


# ── Run manually ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("use pytest: the tests rely on the tmp_path fixture")
