"""Online update of the SQR-JC heads from observed outcomes (§54).

After executing a_t = (a_e, a_u, a_m):

    observe KPI_e / KPI_u / KPI_m  (the FROZEN Gate II-d window KPIs of the
        governed window W+1: eMBB median windowThroughput @ stamp (W+2)*100ms;
        URLLC per-window p95 by arrival grouping; mMTC 1 s rolling PDR)
    -> convert to margins m_e / m_u / m_m        (sla_margin, §7)
    -> update safe head (pinball + critical rule at threshold 0), q25, q75
       of each slice at x_t = [z_joint(W), a_e/C, a_u/C, a_m/C]

The safe head's tau/M adaptation runs per observation with the margin-scale
violation indicator m_i < 0 (kernel_models).  A slice with a vacuous margin
in the governed window is EXCLUDED from that slice's update (audited R19);
the others still update.  §55: SQR-HET receives the same outcomes but keeps
its per-slice feature x_i = [z_i, a_i/C].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from nsoran.sqr_jc.feature_scaler import FeatureScaler
from nsoran.sqr_jc.joint_action_space import C_PRB
from nsoran.sqr_jc.joint_state import (
    JointState,
    governed_outcome_window,
    kpm_stamp_ms,
)
from nsoran.sqr_jc.kernel_models import MultiQuantileKernelModel, MultiQuantileUpdate
from nsoran.sqr_jc.sla_margin import SliceOutcome

#: Margin-scale SLA boundary of the critical rule (m > 0 = satisfied).
MARGIN_THRESHOLD = 0.0


@dataclass(frozen=True)
class WindowUpdateRecord:
    """Bookkeeping of one outcome absorbed by the three slice models."""

    state_window_index: int
    outcome_window_index: int
    outcome_kpm_stamp_ms: int
    action: tuple[int, int, int]
    margins: tuple[Optional[float], Optional[float], Optional[float]]
    sla_ok: tuple[bool, bool, bool]
    updates: tuple[Optional[MultiQuantileUpdate], ...]   # None for vacuous slice
    learned_slices: tuple[int, ...]


@dataclass
class OnlineUpdater:
    """Applies §54 updates to the three slice models."""

    models: tuple[MultiQuantileKernelModel, MultiQuantileKernelModel, MultiQuantileKernelModel]
    scaler: FeatureScaler
    c: int = C_PRB
    #: §57 kernel-log counters (cumulative since construction).
    insertions: int = 0
    evictions: int = 0
    kernel_evaluations: int = 0
    out_of_support_events: int = 0
    update_time_ms_total: float = 0.0
    last_record: Optional[WindowUpdateRecord] = None

    def __post_init__(self) -> None:
        if len(self.models) != 3:
            raise ValueError(f"need exactly 3 slice models, got {len(self.models)}")
        if not isinstance(self.scaler, FeatureScaler):
            raise TypeError("scaler must be a FeatureScaler (frozen §13)")
        if int(self.c) <= 0:
            raise ValueError(f"c must be positive, got {self.c}")

    def observe_outcome(
        self,
        state: JointState,
        action: tuple[int, int, int],
        outcome: SliceOutcome,
    ) -> WindowUpdateRecord:
        """Absorb one governed-window outcome (§54; R19 vacuous exclusion)."""
        import time

        w = int(state.window_index)
        out_w = governed_outcome_window(w)
        stamp = kpm_stamp_ms(out_w)

        x = state.feature_x(action, self.c, self.scaler)
        margins = outcome.margins
        sla_ok = outcome.sla_ok

        updates: list[Optional[MultiQuantileUpdate]] = []
        learned: list[int] = []
        t0 = time.perf_counter()
        for si in range(3):
            m = margins[si]
            if m is None:
                updates.append(None)   # vacuous: excluded from the update (R19)
                continue
            rec = self.models[si].update(x, float(m), threshold=MARGIN_THRESHOLD)
            updates.append(rec)
            learned.append(si)
            self.insertions += 1
            if rec.pruned_index is not None:
                self.evictions += 1
            self.kernel_evaluations += 2  # k_new row + crossing re-check row
        self.update_time_ms_total += (time.perf_counter() - t0) * 1000.0

        record = WindowUpdateRecord(
            state_window_index=w,
            outcome_window_index=out_w,
            outcome_kpm_stamp_ms=int(stamp),
            action=tuple(int(v) for v in action),
            margins=tuple(margins),
            sla_ok=tuple(bool(v) for v in sla_ok),
            updates=tuple(updates),
            learned_slices=tuple(learned),
        )
        self.last_record = record
        return record
