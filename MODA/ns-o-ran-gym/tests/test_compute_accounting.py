"""Tests for training and xApp compute accounting helpers."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsoran.compute_accounting import (
    ComputeAccounting,
    DecisionTimer,
    estimate_compute_cost,
)


def test_estimate_compute_cost_from_wall_time_and_budget():
    cost = estimate_compute_cost(
        wall_time_s=7200.0,
        interaction_budget=20000,
        cost_per_hour_usd=0.50,
        avg_power_watts=100.0,
        electricity_cost_usd_per_kwh=0.20,
    )

    assert cost["cloud_cost_usd"] == 1.0
    assert cost["cost_per_1000_steps_usd"] == 0.05
    assert cost["energy_kwh"] == 0.2
    assert cost["energy_cost_usd"] == 0.04


def test_decision_timer_reports_mean_p95_and_deadline_usage():
    timer = DecisionTimer(period_ms=10.0)
    timer.record_ms(1.0)
    timer.record_ms(3.0)
    timer.record_ms(2.0)

    summary = timer.summary()

    assert summary["decision_count"] == 3
    assert summary["inference_ms_mean"] == 2.0
    assert summary["inference_ms_p95"] == 3.0
    assert summary["deadline_usage_pct"] == 30.0


def test_compute_accounting_summary_contains_training_and_xapp_sections():
    accounting = ComputeAccounting(
        interaction_budget=100,
        period_ms=10.0,
        cost_per_hour_usd=1.0,
    )
    accounting.decision_timer.record_ms(0.5)

    summary = accounting.finish()

    assert summary["training"]["interaction_budget"] == 100
    assert summary["training"]["wall_time_s"] >= 0.0
    assert summary["training"]["steps_per_second"] >= 0.0
    assert summary["xapp_runtime"]["decision_count"] == 1
    assert "cloud_cost_usd" in summary["cost_estimate"]
