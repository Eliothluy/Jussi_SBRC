import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsoran.scoring import score_summary_rows, SLICE_NAMES


def _row(slice, **kwargs):
    base = {
        "slice": slice,
        "throughput_mbps_mean": "10",
        "sla_satisfaction_pct": "80",
        "offered_load_satisfaction_pct": "70",
        "pdr_pct": "90",
        "delay_ms_mean": "20",
        "buffer_bytes_mean": "100000",
        "budget_utilization_pct_mean": "80",
    }
    base.update(kwargs)
    return base


class ScoringTest(unittest.TestCase):
    def test_score_summary_rows_exists_and_returns_float(self):
        rows = [_row("eMBB"), _row("URLLC"), _row("MTC")]
        score = score_summary_rows(rows)
        self.assertIsInstance(score, float)

    def test_score_summary_rows_returns_negative_for_missing_slices(self):
        rows = [_row("eMBB"), _row("URLLC")]
        self.assertLess(score_summary_rows(rows), -1e8)

    def test_balanced_outscores_starved_mtc(self):
        balanced = [
            _row("eMBB", throughput_mbps_mean="45", sla_satisfaction_pct="90", buffer_bytes_mean="2000000"),
            _row("URLLC", throughput_mbps_mean="4", sla_satisfaction_pct="100", delay_ms_mean="5", buffer_bytes_mean="1000"),
            _row("MTC", throughput_mbps_mean="30", sla_satisfaction_pct="100", delay_ms_mean="50", buffer_bytes_mean="100000"),
        ]
        starved = [
            _row("eMBB", throughput_mbps_mean="110", sla_satisfaction_pct="100", buffer_bytes_mean="1000000"),
            _row("URLLC", throughput_mbps_mean="4", sla_satisfaction_pct="100", delay_ms_mean="5", buffer_bytes_mean="1000"),
            _row("MTC", throughput_mbps_mean="0", sla_satisfaction_pct="0", pdr_pct="0", delay_ms_mean="0", buffer_bytes_mean="0", offered_load_satisfaction_pct="0", budget_utilization_pct_mean="0"),
        ]
        self.assertGreater(score_summary_rows(balanced), score_summary_rows(starved))

    def test_slice_names_constant(self):
        self.assertEqual(SLICE_NAMES, ("eMBB", "URLLC", "MTC"))


if __name__ == "__main__":
    unittest.main()
