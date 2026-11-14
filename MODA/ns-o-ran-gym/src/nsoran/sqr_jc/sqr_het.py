"""SQR-HET baseline (§27) — the fair main baseline for the campaign.

Definition frozen by protocol §27:

> SQR original adaptado apenas para utilizar os KPIs/SLA heterogêneos e a
> margem normalizada comum, mantendo a arquitetura de decisão slice-wise
> seguida de projeção.

What SQR-HET KEEPS (all of it — the audited SQR original machinery,
REUSED BY IMPORT, not modified, not reimplemented):
    - per-slice state/action model  (three ``SqrSliceController``)
    - kernel quantile regression    (``nsoran.sqr.quantile_regressor``)
    - dynamic tau + M               (audited R11 adaptation)
    - uncertainty + safety margin   (a <- a + round(sigma * M) — the
      ORIGINAL post-selection margin, part of SQR's architecture)
    - original fallback/exploration (epsilon-softmax, safety-first kNN)
    - original projection           (``nsoran.sqr.coordinator.project_prbs``,
      largest-remainder cut + R12 virtual negative updates)

What SQR-HET does NOT have (§27 — the isolated delta of SQR-JC):
    - joint action evaluation
    - feasibility-aware selection (no q25/q75 heads, no 3-way classes)
    - conditional floors
    - graceful degradation
    - joint feature (its feature stays x_i = [z_i, a_i/C], §55)

Adaptation: each slice's regressor is trained on the COMMON normalized
margin m_i (sla_margin) instead of a raw KPI, with SLA boundary
Lambda = 0 in margin scale (m > 0 = satisfied) — heterogeneous KPIs, one
comparable target.  Success/violation for the experience memory, kNN,
tau/M heuristic and critical rule follow the margin (m < 0 = violated).

Per-slice local state z_i (6 dims — the SQR original z layout adapted to
heterogeneous features, all normalized by the SAME frozen rules):

    [ n_i, queue_i, prev_KPI_i, snr_i, prev_a_i/C, viol_i ]

(no z_joint — that is precisely the SQR-JC delta).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from nsoran.sqr.coordinator import (
    DEFAULT_VIRTUAL_ALPHA,
    apply_virtual_updates,
    project_prbs,
)
from nsoran.sqr.sqr_controller import SqrDecision, SqrSliceController
from nsoran.sqr_jc.feature_scaler import FeatureScaler
from nsoran.sqr_jc.joint_state import JointState
from nsoran.sqr_jc.sla_margin import SliceOutcome

#: Controller identity logged in step_metrics/decision logs.
CONTROLLER_NAME = "sqr_het"

#: Per-slice local state dim (see module docstring).
HET_STATE_DIM = 6


@dataclass(frozen=True)
class SqrHetDecision:
    """One SQR-HET decision: three slice decisions + the projected action."""

    slice_decisions: tuple[SqrDecision, SqrDecision, SqrDecision]
    proposals: tuple[int, int, int]
    action: tuple[int, int, int]     # after the ORIGINAL projection Π
    cut: tuple[bool, bool, bool]
    unconstrained_raw: tuple[int, int, int] = (0, 0, 0)
    unconstrained_proposals: tuple[int, int, int] = (0, 0, 0)
    guard_suppressed: tuple[bool, bool, bool] = (False, False, False)


class SqrHetController:
    """Slice-wise SQR on the common margin + original projection (§27)."""

    def __init__(
        self,
        controllers: tuple[SqrSliceController, SqrSliceController, SqrSliceController],
        scaler: FeatureScaler,
        c: int,
        virtual_alpha: float = DEFAULT_VIRTUAL_ALPHA,
        empty_slice_guard: bool = False,
    ) -> None:
        if len(controllers) != 3:
            raise ValueError(f"need exactly 3 SqrSliceController, got {len(controllers)}")
        for ctl in controllers:
            if not isinstance(ctl, SqrSliceController):
                raise TypeError(
                    "SQR-HET must reuse the AUDITED SqrSliceController "
                    "(learner NOT replaced — §6/§27)"
                )
            if ctl.state_dim != HET_STATE_DIM:
                raise ValueError(
                    f"slice controller state_dim must be {HET_STATE_DIM}, "
                    f"got {ctl.state_dim}"
                )
            if ctl.sla_threshold != 0.0:
                raise ValueError(
                    "SQR-HET Lambda must be 0 (margin scale: m > 0 satisfied)"
                )
        self.controllers = tuple(controllers)
        self.scaler = scaler
        self.c = int(c)
        self.virtual_alpha = float(virtual_alpha)
        self.empty_slice_guard = bool(empty_slice_guard)
        self.prev_action: tuple[int, int, int] = (0, 0, 0)

    # ------------------------------------------------------------------
    # Per-slice local state z_i (6 dims; NO joint features)
    # ------------------------------------------------------------------

    def slice_state(self, state: JointState, slice_id: int) -> np.ndarray:
        """z_i = [n_i, queue_i, prev_KPI_i, snr_i, prev_a_i/C, viol_i]."""
        letter = "emu"[slice_id]
        n = (state.n_e, state.n_u, state.n_m)[slice_id]
        queue = (state.queue_e, state.queue_u, state.queue_m)[slice_id]
        kpi = (state.kpi_e_raw, state.kpi_u_raw, state.kpi_m_raw)[slice_id]
        snr = (state.snr_e, state.snr_u, state.snr_m)[slice_id]
        viol = (state.viol_e, state.viol_u, state.viol_m)[slice_id]
        prev_a = (state.prev_a_e, state.prev_a_u, state.prev_a_m)[slice_id]
        return np.array([
            float(self.scaler.transform_feature(f"n_{letter}", float(n))),
            float(self.scaler.transform_feature(f"queue_{letter}", float(queue))),
            float(self.scaler.transform_feature(_KPI_FEATURE[slice_id], float(kpi or 0.0))),
            float(self.scaler.transform_feature(f"snr_{letter}", float(snr))),
            float(prev_a) / float(self.c),
            float(viol),
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Decision (slice-wise + original projection — NO joint machinery)
    # ------------------------------------------------------------------

    def is_slice_provably_empty(self, state: JointState, slice_id: int) -> bool:
        """True iff slice has zero active UEs, zero queue, zero offered traffic,
        zero closed-window TX packets, and no pending pipeline work (A2a)."""
        n_i = (state.n_e, state.n_u, state.n_m)[slice_id]
        q_i = (state.queue_e, state.queue_u, state.queue_m)[slice_id]
        off_i = (state.offered_e, state.offered_u, state.offered_m)[slice_id]
        extras = state.extras or {}
        tx_tuple = extras.get("kpm_tx_packets") or (0, 0, 0)
        tx_i = tx_tuple[slice_id] if slice_id < len(tx_tuple) else 0
        if slice_id == 0:
            pending = bool(extras.get("embb_pending_pipeline", False))
        elif slice_id == 1:
            pending = bool(extras.get("urllc_pool_open", False))
        else:
            pending = (state.kpi_m_raw is not None) or bool(
                extras.get("mmtc_rolling_active", False)
            )
        return (
            int(n_i) == 0
            and float(q_i) <= 0.0
            and float(off_i) <= 0.0
            and int(tx_i) <= 0
            and not pending
        )

    def decide(self, state: JointState) -> SqrHetDecision:
        zs = [self.slice_state(state, sid) for sid in range(3)]
        raw_decisions = tuple(self.controllers[sid].act(zs[sid]) for sid in range(3))
        unconstrained_raw = tuple(int(d.raw_action) for d in raw_decisions)
        unconstrained_proposals = tuple(int(d.action) for d in raw_decisions)

        decisions_list = list(raw_decisions)
        guard_suppressed = [False, False, False]
        if self.empty_slice_guard:
            for sid in range(3):
                if self.is_slice_provably_empty(state, sid):
                    decisions_list[sid] = replace(
                        raw_decisions[sid],
                        action=0,
                        raw_action=0,
                        margin_applied=0,
                    )
                    guard_suppressed[sid] = True

        decisions = tuple(decisions_list)
        proposals = tuple(int(d.action) for d in decisions)
        result = project_prbs(np.asarray(proposals, dtype=np.int64), self.c)
        action = tuple(int(v) for v in result.allocated)
        self.prev_action = action
        return SqrHetDecision(
            slice_decisions=decisions,
            proposals=proposals,
            action=action,
            cut=tuple(bool(v) for v in result.cut),
            unconstrained_raw=unconstrained_raw,
            unconstrained_proposals=unconstrained_proposals,
            guard_suppressed=tuple(guard_suppressed),
        )

    def apply_virtual_updates(self, state: JointState, proposals, allocated) -> list:
        """R12 virtual negative update at the PRE-projection proposal."""
        zs = [self.slice_state(state, sid) for sid in range(3)]
        from nsoran.sqr.coordinator import ProjectionResult, VirtualUpdate

        updates = tuple(
            VirtualUpdate(
                slice_id=i,
                proposed=int(p),
                allocated=int(a),
                deficit=int(p) - int(a),
            )
            for i, (p, a) in enumerate(zip(proposals, allocated))
            if int(a) < int(p)
        )
        result = ProjectionResult(
            allocated=np.asarray(allocated, dtype=np.int64),
            proposals=np.asarray(proposals, dtype=np.int64),
            total_prbs=self.c,
            cut=np.asarray(allocated) < np.asarray(proposals),
            virtual_updates=updates,
        )
        return apply_virtual_updates(result, zs, self.controllers, alpha=self.virtual_alpha)

    # ------------------------------------------------------------------
    # Online update on the common margin (§55: same outcomes, per-slice
    # feature x_i = [z_i, a_i/C])
    # ------------------------------------------------------------------

    def observe_outcome(
        self, state: JointState, executed: tuple[int, int, int], outcome: SliceOutcome
    ) -> list:
        zs = [self.slice_state(state, sid) for sid in range(3)]
        margins = outcome.margins
        records = []
        for sid in range(3):
            m = margins[sid]
            if m is None:
                continue  # vacuous: excluded (R19)
            records.append(
                self.controllers[sid].observe(
                    zs[sid], int(executed[sid]), float(m), threshold=0.0
                )
            )
        return records

    # ------------------------------------------------------------------
    # Dynamic C sync (R01) — plain attribute sync like sqr_run.py does
    # ------------------------------------------------------------------

    def set_n_prbs(self, c: int) -> None:
        self.c = int(c)
        for ctl in self.controllers:
            ctl.n_prbs = int(c)


#: Feature names for the per-slice prev_KPI normalization rule.
_KPI_FEATURE = ("kpi_e", "kpi_u", "kpi_m")
