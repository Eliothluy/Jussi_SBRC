"""Priority-SQR baseline (Hoashi, Ono & Nakao, INFOCOM Wkshps 2026, Eqs. 2-5).

Isolated implementation under ``sqr_v2`` used solely as a fair literature
comparator alongside ``B0-v2-ref`` and ``SQR-CER``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


CANONICAL_URLLC_PRIORITY_WEIGHTS: tuple[float, float, float] = (1.0, 1e6, 1.0)
BALANCED_PRIORITY_WEIGHTS: tuple[float, float, float] = (10.0, 1e6, 1.0)


@dataclass(frozen=True)
class PriorityHoashiConfig:
    """Configuration for priority-weighted proportional excess cutting."""

    weights: tuple[float, float, float] = CANONICAL_URLLC_PRIORITY_WEIGHTS

    def __post_init__(self) -> None:
        if len(self.weights) != 3:
            raise ValueError(f"weights must have length 3, got {len(self.weights)}")
        for idx, w in enumerate(self.weights):
            wf = float(w)
            if not math.isfinite(wf) or wf <= 0.0:
                raise ValueError(f"weight[{idx}] must be finite and > 0, got {w!r}")


@dataclass(frozen=True)
class PriorityHoashiDecision:
    """Auditable output of one Priority-SQR allocation step."""

    action: tuple[int, int, int]
    proposals_guarded: tuple[int, int, int]
    excess_cut: int
    weights: tuple[float, float, float]
    cut_applied: bool


def parse_priority_weights(raw: str | Sequence[float]) -> tuple[float, float, float]:
    """Parse ``'w_e,w_u,w_m'`` or 3-element sequence into priority weights."""
    if isinstance(raw, str):
        parts = [float(p.strip()) for p in raw.split(",") if p.strip()]
    else:
        parts = [float(p) for p in raw]
    if len(parts) != 3:
        raise ValueError(f"priority weights must have 3 values, got {raw!r}")
    cfg = PriorityHoashiConfig(weights=(parts[0], parts[1], parts[2]))
    return cfg.weights


class PriorityHoashiReallocator:
    """Deterministic inverse-priority excess cutter (Hoashi et al., Eqs. 2-5)."""

    def __init__(self, config: PriorityHoashiConfig | None = None) -> None:
        self.config = config or PriorityHoashiConfig()

    def allocate(
        self,
        proposals: Sequence[int],
        capacity_c: int,
        active_mask: Sequence[bool] = (True, True, True),
    ) -> PriorityHoashiDecision:
        """Apply A2a empty-slice guard and cut excess inversely to weights."""
        if len(proposals) != 3 or len(active_mask) != 3:
            raise ValueError("proposals and active_mask must have length 3")
        c = max(0, int(capacity_c))
        weights = tuple(float(w) for w in self.config.weights)

        guarded = (
            max(0, int(proposals[0])) if bool(active_mask[0]) else 0,
            max(0, int(proposals[1])) if bool(active_mask[1]) else 0,
            max(0, int(proposals[2])) if bool(active_mask[2]) else 0,
        )
        if c == 0:
            return PriorityHoashiDecision(
                action=(0, 0, 0),
                proposals_guarded=guarded,
                excess_cut=int(sum(guarded)),
                weights=(weights[0], weights[1], weights[2]),
                cut_applied=bool(sum(guarded) > 0),
            )

        total_prop = int(sum(guarded))
        if total_prop <= c:
            return PriorityHoashiDecision(
                action=guarded,
                proposals_guarded=guarded,
                excess_cut=0,
                weights=(weights[0], weights[1], weights[2]),
                cut_applied=False,
            )

        r_cut = float(total_prop - c)
        q = [float(v) for v in guarded]

        # Iterative inverse-priority water-filling so cuts saturating a low-priority
        # slice at 0 redistribute the remaining excess across surviving active slices.
        rem_cut = r_cut
        for _ in range(3):
            if rem_cut <= 1e-12:
                break
            surv = [i for i in range(3) if bool(active_mask[i]) and q[i] > 1e-12]
            if not surv:
                break
            inv_w = {i: 1.0 / weights[i] for i in surv}
            denom = math.fsum(inv_w.values())
            shaved_total = 0.0
            for i in surv:
                p_i = inv_w[i] / denom
                cut_i = min(q[i], p_i * rem_cut)
                q[i] -= cut_i
                shaved_total += cut_i
            rem_cut -= shaved_total

        # Conservative integer rounding guaranteeing sum <= c and non-negativity.
        alloc = [min(guarded[i], max(0, int(math.floor(q[i] + 1e-12)))) for i in range(3)]
        while sum(alloc) > c:
            # Trim 1 PRB from lowest-priority slice with positive allocation
            victim = max(
                (i for i in range(3) if alloc[i] > 0),
                key=lambda idx: (1.0 / weights[idx], idx),
            )
            alloc[victim] -= 1

        rem_slots = min(c, total_prop) - sum(alloc)
        while rem_slots > 0:
            elig = [i for i in range(3) if bool(active_mask[i]) and alloc[i] < guarded[i]]
            if not elig:
                break
            best = max(
                elig,
                key=lambda idx: (q[idx] - alloc[idx], weights[idx], -idx),
            )
            alloc[best] += 1
            rem_slots -= 1

        return PriorityHoashiDecision(
            action=(int(alloc[0]), int(alloc[1]), int(alloc[2])),
            proposals_guarded=guarded,
            excess_cut=int(total_prop - c),
            weights=(weights[0], weights[1], weights[2]),
            cut_applied=True,
        )
