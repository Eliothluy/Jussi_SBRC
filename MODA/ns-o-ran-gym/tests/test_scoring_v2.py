import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsoran.scoring import (
    score_summary_rows_v2,
    SLICE_NAMES,
    V2_URLLC_DELAY_MS_MAX,
)


def _row(slice, **kwargs):
    """A feasible, well-served row by default (URLLC p99 under deadline)."""
    base = {
        "slice": slice,
        "throughput_mbps_mean": "10",
        "sla_satisfaction_pct": "90",
        "offered_load_satisfaction_pct": "80",
        "pdr_pct": "95",
        "delay_ms_mean": "5",
        "delay_ms_p99": "5",
        "buffer_bytes_mean": "100000",
        "budget_utilization_pct_mean": "80",
    }
    base.update(kwargs)
    return base


def _feasible_case():
    # URLLC p99 <= 10ms (feasible); all slices decently served.
    return [
        _row("eMBB", sla_satisfaction_pct="80", delay_ms_p99="40"),
        _row("URLLC", sla_satisfaction_pct="85", delay_ms_p99="8"),
        _row("MTC", sla_satisfaction_pct="88", delay_ms_p99="120"),
    ]


class ScoringV2Test(unittest.TestCase):
    def test_returns_float(self):
        self.assertIsInstance(score_summary_rows_v2(_feasible_case()), float)

    def test_missing_slices_returns_large_negative(self):
        self.assertLess(score_summary_rows_v2([_row("eMBB"), _row("URLLC")]), -1e8)

    def test_feasible_is_nonnegative(self):
        self.assertGreaterEqual(score_summary_rows_v2(_feasible_case()), 0.0)

    def test_urllc_latency_is_the_hard_constraint(self):
        # URLLC p99 above the deadline -> infeasible (negative), regardless of
        # how well the slices are otherwise served.
        rows = _feasible_case()
        rows[1]["delay_ms_p99"] = str(V2_URLLC_DELAY_MS_MAX * 5)
        self.assertLess(score_summary_rows_v2(rows), 0.0)

    def test_feasible_beats_infeasible(self):
        feasible = _feasible_case()
        infeasible = _feasible_case()
        infeasible[1]["delay_ms_p99"] = str(V2_URLLC_DELAY_MS_MAX * 3)
        self.assertGreater(
            score_summary_rows_v2(feasible), score_summary_rows_v2(infeasible)
        )

    def test_starvation_is_penalized_via_minmax(self):
        """A feasible allocation that starves one slice (low composite SLA on
        that slice) must score well below a balanced one, because the min-SLA
        term dominates. This is the audit's core requirement expressed under
        Path C: URLLC latency is met, but starving MTC is still punished."""
        balanced = _feasible_case()  # all slices ~0.8-0.88
        starved = [
            _row("eMBB", sla_satisfaction_pct="95", delay_ms_p99="40"),
            _row("URLLC", sla_satisfaction_pct="90", delay_ms_p99="6"),
            _row("MTC", sla_satisfaction_pct="3", delay_ms_p99="120"),  # starved
        ]
        self.assertGreater(
            score_summary_rows_v2(balanced), score_summary_rows_v2(starved)
        )

    def test_acid_case_sa_eval_0021_scores_low(self):
        """sa_eval_0021 (weights [0.816,0.180,0.004]) had URLLC delay ~3.4ms
        (feasible under Path C) but starved MTC (PDR=2.4% -> composite SLA ~0).
        v2 must score it far below a balanced feasible allocation."""
        acid = [
            _row("eMBB", sla_satisfaction_pct="70", delay_ms_p99="1721"),
            _row("URLLC", sla_satisfaction_pct="78", delay_ms_p99="3.4"),
            _row("MTC", sla_satisfaction_pct="3", delay_ms_p99="4479"),  # starved
        ]
        self.assertGreater(
            score_summary_rows_v2(_feasible_case()), score_summary_rows_v2(acid)
        )

    def test_dead_urllc_is_infeasible_not_feasible(self):
        """Loophole closed: a URLLC slice that delivers ZERO packets has no
        delay samples (p99='NA'). It must be treated as an outage (infeasible),
        NOT as '0ms <= 10ms feasible' — otherwise the optimizer can satisfy the
        latency constraint by starving URLLC entirely (v1 starvation paradox)."""
        rows = _feasible_case()
        rows[1]["delay_ms_p99"] = "NA"   # empty histogram
        rows[1]["pdr_pct"] = "0"          # zero packets delivered
        rows[1]["throughput_mbps_mean"] = "0"
        self.assertLess(score_summary_rows_v2(rows), 0.0)
        # And it must lose to a genuinely feasible allocation.
        self.assertGreater(
            score_summary_rows_v2(_feasible_case()), score_summary_rows_v2(rows)
        )

    def test_better_served_scores_higher(self):
        """Monotonicity: raising the worst slice's composite SLA raises the score."""
        worse = _feasible_case()
        worse[2]["sla_satisfaction_pct"] = "40"
        better = _feasible_case()
        better[2]["sla_satisfaction_pct"] = "80"
        self.assertGreater(
            score_summary_rows_v2(better), score_summary_rows_v2(worse)
        )


if __name__ == "__main__":
    unittest.main()
