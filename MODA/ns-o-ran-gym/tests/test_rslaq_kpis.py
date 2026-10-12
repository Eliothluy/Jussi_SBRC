"""Tests for rslaq_kpis module."""

import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from environments.rslaq_kpis import parse_kpm_file, build_observation


def _make_kpm_file(rows):
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write("timestamp,ueImsi,sliceId,dTxBytes,dRxBytes,plr,resourceSharePct,dLostPackets,throughputMbps\n")
        for r in rows:
            f.write(
                f"{r['timestamp']},{r['ueImsi']},{r['sliceId']},"
                f"{r['dTxBytes']},{r['dRxBytes']},{r['plr']},"
                f"{r['resourceSharePct']},{r['dLostPackets']},{r['throughputMbps']}\n"
            )
    return path


def test_parse_and_build_paper():
    rows = [
        {"timestamp": 1, "ueImsi": 1, "sliceId": 0, "dTxBytes": 100000, "dRxBytes": 90000, "plr": 2.0, "resourceSharePct": 33.3, "dLostPackets": 10, "throughputMbps": 50.0},
        {"timestamp": 1, "ueImsi": 2, "sliceId": 1, "dTxBytes": 50000, "dRxBytes": 48000, "plr": 1.0, "resourceSharePct": 40.0, "dLostPackets": 5, "throughputMbps": 20.0},
        {"timestamp": 1, "ueImsi": 3, "sliceId": 2, "dTxBytes": 20000, "dRxBytes": 19000, "plr": 5.0, "resourceSharePct": 26.7, "dLostPackets": 20, "throughputMbps": 10.0},
    ]
    path = _make_kpm_file(rows)
    kpi_dict = parse_kpm_file(path, last_timestamp=0)
    os.unlink(path)

    assert 0 in kpi_dict
    assert 1 in kpi_dict
    assert 2 in kpi_dict
    assert 3 in kpi_dict  # cell total

    obs = build_observation(kpi_dict, mode="paper")
    assert obs.shape == (4, 4)
    assert obs.dtype == np.float32
    assert np.all((obs >= 0) & (obs <= 1))


def test_parse_zero_based_ids():
    # Test that 0-based ids are parsed correctly
    rows = [
        {"timestamp": 1, "ueImsi": 1, "sliceId": 0, "dTxBytes": 100000, "dRxBytes": 90000, "plr": 2.0, "resourceSharePct": 33.3, "dLostPackets": 10, "throughputMbps": 50.0},
    ]
    path = _make_kpm_file(rows)
    kpi_dict = parse_kpm_file(path, last_timestamp=0)
    os.unlink(path)
    assert kpi_dict[0]["ue_count"] == 1


def test_missing_slice():
    rows = [
        {"timestamp": 1, "ueImsi": 1, "sliceId": 0, "dTxBytes": 100000, "dRxBytes": 90000, "plr": 2.0, "resourceSharePct": 33.3, "dLostPackets": 10, "throughputMbps": 50.0},
    ]
    path = _make_kpm_file(rows)
    kpi_dict = parse_kpm_file(path, last_timestamp=0)
    os.unlink(path)
    # Slice 1 and 2 should have zero counts
    assert kpi_dict[1]["ue_count"] == 0
    assert kpi_dict[2]["ue_count"] == 0


def test_debug_mode():
    rows = [
        {"timestamp": 1, "ueImsi": 1, "sliceId": 0, "dTxBytes": 100000, "dRxBytes": 90000, "plr": 2.0, "resourceSharePct": 33.3, "dLostPackets": 10, "throughputMbps": 50.0},
    ]
    path = _make_kpm_file(rows)
    kpi_dict = parse_kpm_file(path, last_timestamp=0)
    os.unlink(path)
    obs = build_observation(kpi_dict, mode="debug")
    assert obs.shape == (3, 5)
    assert np.all((obs >= 0) & (obs <= 1))


if __name__ == "__main__":
    test_parse_and_build_paper()
    test_parse_zero_based_ids()
    test_missing_slice()
    test_debug_mode()
    print("All kpis tests passed.")
