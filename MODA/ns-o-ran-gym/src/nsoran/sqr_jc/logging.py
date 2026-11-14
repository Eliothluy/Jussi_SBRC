"""SQR-JC decision logging (§56 exact columns + §57 kernel logs).

``DECISION_COLUMNS`` is the protocol's §56 column list, in order — the CSV
writer validates every row against it (length + name check on open; value
order by construction).  Post-window columns (observed KPI / normalized
margin / SLA status) ride in ``OUTCOME_COLUMNS`` of the outcome CSV, keyed
by the governed window (§56 "Após janela").

``KernelLogRow``/``kernel_log_columns`` implement §57: support vector count,
insertions, evictions, kernel evaluations, prediction time, update time,
out-of-support count, tau_safe(t) and M_i(t) — per slice, per decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

#: §56 per-decision columns — EXACT order, no additions, no removals.
DECISION_COLUMNS: tuple[str, ...] = (
    "time",
    "rng_run",
    "controller",
    "phase",
    "lambda_total",
    "N_e", "N_u", "N_m",
    "queue_e", "queue_u", "queue_m",
    "prev_a_e", "prev_a_u", "prev_a_m",
    "floor_e", "floor_u", "floor_m",
    "state_class",
    "search_mode",
    "num_actions_total",
    "num_actions_evaluated",
    "num_joint_feasible",
    "num_joint_marginal",
    "num_joint_infeasible",
    "selected_a_e", "selected_a_u", "selected_a_m",
    "safe_q_e", "q25_e", "q75_e", "sigma_e",
    "safe_q_u", "q25_u", "q75_u", "sigma_u",
    "safe_q_m", "q25_m", "q75_m", "sigma_m",
    "support_e", "support_u", "support_m",
    "quantile_crossing_e", "quantile_crossing_u", "quantile_crossing_m",
    "decision_time_ms",
)

#: Post-window columns of the outcome CSV (§56 "Após janela").
OUTCOME_COLUMNS: tuple[str, ...] = (
    "outcome_stamp_ms",
    "outcome_window",
    "executed_a_e", "executed_a_u", "executed_a_m",
    "kpi_embb_mbps",
    "kpi_urllc_p95_ms",
    "kpi_urllc_miss_pct",
    "kpi_mmtc_pdr",
    "margin_e", "margin_u", "margin_m",
    "sla_ok_e", "sla_ok_u", "sla_ok_m",
    "joint_sla_ok",
    "severity_max",
)


def format_row(values: Sequence) -> str:
    """One CSV line with the exact §56 column count."""
    if len(values) != len(DECISION_COLUMNS):
        raise ValueError(
            f"decision row has {len(values)} values, "
            f"expected {len(DECISION_COLUMNS)} (§56 exact columns)"
        )
    return ",".join(_fmt(v) for v in values)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    if isinstance(v, bool):
        return str(int(v))
    return str(v)


@dataclass(frozen=True)
class KernelLogRow:
    """§57 kernel log of one decision (per-slice model state + counters)."""

    time_ms: float
    slice_support_count: tuple[int, int, int]
    insertions: int
    evictions: int
    kernel_evaluations: int
    prediction_time_ms: float
    update_time_ms: float
    out_of_support_count: int
    tau_safe: tuple[float, float, float]
    margin_m: tuple[float, float, float]

    def row(self) -> str:
        vals = [
            f"{self.time_ms:.3f}",
            *[str(int(v)) for v in self.slice_support_count],
            str(int(self.insertions)),
            str(int(self.evictions)),
            str(int(self.kernel_evaluations)),
            f"{self.prediction_time_ms:.6f}",
            f"{self.update_time_ms:.6f}",
            str(int(self.out_of_support_count)),
            *[f"{v:.4f}" for v in self.tau_safe],
            *[f"{v:.4f}" for v in self.margin_m],
        ]
        return ",".join(vals)


KERNEL_LOG_COLUMNS: tuple[str, ...] = (
    "time_ms",
    "support_count_e", "support_count_u", "support_count_m",
    "insertions",
    "evictions",
    "kernel_evaluations",
    "prediction_time_ms",
    "update_time_ms",
    "out_of_support_count",
    "tau_safe_e", "tau_safe_u", "tau_safe_m",
    "margin_m_e", "margin_m_u", "margin_m_m",
)


class DecisionLogWriter:
    """Append-only writer for the §56 decision CSV (+ outcome CSV)."""

    def __init__(self, decision_path: str, outcome_path: Optional[str] = None,
                 kernel_log_path: Optional[str] = None) -> None:
        self.decision_path = decision_path
        self.outcome_path = outcome_path
        self.kernel_log_path = kernel_log_path
        self._fh = open(decision_path, "w", encoding="utf-8", newline="")
        self._fh.write(",".join(DECISION_COLUMNS) + "\n")
        self._fh_out = None
        if outcome_path is not None:
            self._fh_out = open(outcome_path, "w", encoding="utf-8", newline="")
            self._fh_out.write(",".join(OUTCOME_COLUMNS) + "\n")
        self._fh_k = None
        if kernel_log_path is not None:
            self._fh_k = open(kernel_log_path, "w", encoding="utf-8", newline="")
            self._fh_k.write(",".join(KERNEL_LOG_COLUMNS) + "\n")
        self.rows_written = 0

    def write_decision(self, values: Sequence) -> None:
        self._fh.write(format_row(values) + "\n")
        self.rows_written += 1

    def write_outcome(self, values: Sequence) -> None:
        if self._fh_out is None:
            raise RuntimeError("no outcome path configured")
        if len(values) != len(OUTCOME_COLUMNS):
            raise ValueError(
                f"outcome row has {len(values)} values, expected "
                f"{len(OUTCOME_COLUMNS)}"
            )
        self._fh_out.write(",".join(_fmt(v) for v in values) + "\n")

    def write_kernel_row(self, row: KernelLogRow) -> None:
        if self._fh_k is None:
            raise RuntimeError("no kernel log path configured")
        self._fh_k.write(row.row() + "\n")

    def close(self) -> None:
        for fh in (self._fh, self._fh_out, self._fh_k):
            if fh is not None:
                fh.flush()
                fh.close()

    def __enter__(self) -> "DecisionLogWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
