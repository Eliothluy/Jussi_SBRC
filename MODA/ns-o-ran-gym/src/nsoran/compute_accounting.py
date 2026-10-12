"""Lightweight compute accounting for RSLAQ training and xApp inference."""

from __future__ import annotations

import math
import resource
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


def _round(value: float) -> float:
    return round(float(value), 6)


def _resource_cpu_time(usage: resource.struct_rusage) -> float:
    return float(usage.ru_utime + usage.ru_stime)


def _max_rss_mb(usage: resource.struct_rusage) -> float:
    # Linux reports ru_maxrss in KiB; macOS reports bytes. This project runs on Linux.
    return float(usage.ru_maxrss) / 1024.0


def estimate_compute_cost(
    *,
    wall_time_s: float,
    interaction_budget: int,
    cost_per_hour_usd: float = 0.0,
    avg_power_watts: float = 0.0,
    electricity_cost_usd_per_kwh: float = 0.0,
) -> dict[str, float]:
    """Estimate cloud and energy cost from elapsed wall-clock time."""
    wall_hours = max(float(wall_time_s), 0.0) / 3600.0
    cloud_cost = wall_hours * max(float(cost_per_hour_usd), 0.0)
    budget = max(int(interaction_budget), 0)
    energy_kwh = wall_hours * max(float(avg_power_watts), 0.0) / 1000.0
    energy_cost = energy_kwh * max(float(electricity_cost_usd_per_kwh), 0.0)
    return {
        "cost_per_hour_usd": _round(cost_per_hour_usd),
        "cloud_cost_usd": _round(cloud_cost),
        "cost_per_1000_steps_usd": _round((1000.0 * cloud_cost / budget) if budget else 0.0),
        "avg_power_watts": _round(avg_power_watts),
        "electricity_cost_usd_per_kwh": _round(electricity_cost_usd_per_kwh),
        "energy_kwh": _round(energy_kwh),
        "energy_cost_usd": _round(energy_cost),
    }


@dataclass
class DecisionTimer:
    """Collect per-step policy decision latency for xApp runtime accounting."""

    period_ms: float
    samples_ms: list[float] = field(default_factory=list)

    @contextmanager
    def measure(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record_ms((time.perf_counter() - start) * 1000.0)

    def record_ms(self, value: float) -> None:
        self.samples_ms.append(max(float(value), 0.0))

    def summary(self) -> dict[str, float | int]:
        if not self.samples_ms:
            return {
                "decision_count": 0,
                "inference_ms_mean": 0.0,
                "inference_ms_p50": 0.0,
                "inference_ms_p95": 0.0,
                "inference_ms_max": 0.0,
                "period_ms": _round(self.period_ms),
                "deadline_usage_pct": 0.0,
            }

        samples = sorted(self.samples_ms)
        p95_idx = max(math.ceil(0.95 * len(samples)) - 1, 0)
        p50 = statistics.median(samples)
        p95 = samples[p95_idx]
        period = max(float(self.period_ms), 0.0)
        return {
            "decision_count": len(samples),
            "inference_ms_mean": _round(statistics.fmean(samples)),
            "inference_ms_p50": _round(p50),
            "inference_ms_p95": _round(p95),
            "inference_ms_max": _round(samples[-1]),
            "period_ms": _round(period),
            "deadline_usage_pct": _round((100.0 * p95 / period) if period else 0.0),
        }


@dataclass
class ComputeAccounting:
    """Track training cost and decision latency for one training run."""

    interaction_budget: int
    period_ms: float
    cost_per_hour_usd: float = 0.0
    avg_power_watts: float = 0.0
    electricity_cost_usd_per_kwh: float = 0.0
    decision_timer: DecisionTimer = field(init=False)
    _start_wall: float = field(init=False)
    _start_process: float = field(init=False)
    _start_self_usage: resource.struct_rusage = field(init=False)
    _start_child_usage: resource.struct_rusage = field(init=False)

    def __post_init__(self) -> None:
        self.decision_timer = DecisionTimer(period_ms=self.period_ms)
        self._start_wall = time.perf_counter()
        self._start_process = time.process_time()
        self._start_self_usage = resource.getrusage(resource.RUSAGE_SELF)
        self._start_child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)

    def finish(self) -> dict[str, object]:
        wall_time_s = max(time.perf_counter() - self._start_wall, 0.0)
        cpu_time_s = max(time.process_time() - self._start_process, 0.0)
        self_usage = resource.getrusage(resource.RUSAGE_SELF)
        child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu_time_s = max(
            _resource_cpu_time(child_usage) - _resource_cpu_time(self._start_child_usage),
            0.0,
        )
        budget = max(int(self.interaction_budget), 0)
        return {
            "training": {
                "wall_time_s": _round(wall_time_s),
                "cpu_time_s": _round(cpu_time_s),
                "child_cpu_time_s": _round(child_cpu_time_s),
                "peak_rss_mb": _round(_max_rss_mb(self_usage)),
                "child_peak_rss_mb": _round(_max_rss_mb(child_usage)),
                "interaction_budget": budget,
                "steps_per_second": _round((budget / wall_time_s) if wall_time_s else 0.0),
            },
            "xapp_runtime": self.decision_timer.summary(),
            "cost_estimate": estimate_compute_cost(
                wall_time_s=wall_time_s,
                interaction_budget=budget,
                cost_per_hour_usd=self.cost_per_hour_usd,
                avg_power_watts=self.avg_power_watts,
                electricity_cost_usd_per_kwh=self.electricity_cost_usd_per_kwh,
            ),
        }
