"""Deterministic integer projection of per-slice PRB proposals onto sum(a) <= C.

The SQR agents propose integer actions a_i in {0, ..., C} per slice; the
action contract requires sum(a) <= C [ARTIGO, AGENTS.md §4]. When the total
exceeds C, the coordinator applies a *proportional* projection Pi and
signals a *virtual negative update* for every cut slice [ARTIGO:

"se exceder, projeção proporcional Π + atualização virtual punitiva da
proposta original" — PLANO_IMPLEMENTACAO_SQR_NORI.md §2.1].

Algorithm (deterministic largest-remainder, exact integer arithmetic):
  1. If sum(proposals) <= C, the proposals are feasible: pass through
     unchanged (the cell may under-use resources; ns-3 renormalizes the
     weights anyway).
  2. Otherwise, with s = C / sum(p): base_i = floor(p_i * C / total) using
     integer division (no float error), then distribute the
     R = C - sum(base) remaining units one at a time to the slices with the
     largest fractional remainder (p_i*C mod total), ties broken by the
     LOWEST slice index. No slice ever receives more than its own proposal
     (a fractional part is always < 1 unit short of the proposal).

The rule uses no random numbers, so the output depends only on the inputs.
The virtual-update signal carries (slice, proposed, allocated, deficit);
the training loop applies it to the slice's regressor as a punitive
observation (weight of the virtual update is an ablation knob [ASSUMIDO],
PLANO §3-Fase 2).

Dependencies: NumPy only (no torch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VirtualUpdate:
    """Signal that slice ``slice_id`` had its proposal cut by ``deficit`` PRBs."""

    slice_id: int
    proposed: int
    allocated: int
    deficit: int  # proposed - allocated > 0


@dataclass(frozen=True)
class ProjectionResult:
    """Outcome of :func:`project_prbs`."""

    allocated: np.ndarray      # int array, sum <= C, 0 <= a_i <= p_i
    proposals: np.ndarray      # int array, as given
    total_prbs: int            # C
    cut: np.ndarray            # bool array, True where allocated < proposed
    virtual_updates: tuple[VirtualUpdate, ...]

    @property
    def total_allocated(self) -> int:
        return int(self.allocated.sum())

    @property
    def total_proposed(self) -> int:
        return int(self.proposals.sum())

    @property
    def over_capacity(self) -> bool:
        return self.total_proposed > self.total_prbs


def _validate(proposals: np.ndarray, total_prbs: int) -> tuple[np.ndarray, int]:
    arr = np.asarray(proposals)
    if arr.ndim != 1 or arr.shape[0] == 0:
        raise ValueError(f"proposals must be a non-empty 1-D array, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("proposals must be finite")
    if not np.all(arr == np.floor(arr)):
        raise ValueError(
            "proposals must be integers (SQR actions are whole PRBs); "
            f"got {arr[arr != np.floor(arr)]}"
        )
    p = arr.astype(np.int64)
    if np.any(p < 0):
        raise ValueError(f"proposals must be non-negative, got {p[p < 0]}")
    c = int(total_prbs)
    if c < 0:
        raise ValueError(f"total_prbs must be non-negative, got {total_prbs}")
    return p, c


def project_prbs(proposals, total_prbs: int) -> ProjectionResult:
    """Project per-slice proposals onto the feasible integer simplex sum <= C.

    Parameters
    ----------
    proposals:
        1-D array-like of non-negative integers (one per slice).
    total_prbs:
        C, the total number of PRBs of the cell.

    Returns
    -------
    ProjectionResult
        ``allocated`` (integers, sum <= C, never above the proposal),
        ``cut`` mask and one :class:`VirtualUpdate` per cut slice.
    """
    p, c = _validate(proposals, total_prbs)
    n = p.shape[0]
    total = int(p.sum())

    if total <= c:
        alloc = p.copy()
        cut = np.zeros(n, dtype=bool)
        updates: tuple[VirtualUpdate, ...] = ()
        return ProjectionResult(alloc, p.copy(), c, cut, updates)

    # Over capacity: exact integer largest-remainder apportionment.
    numer = p * c                    # p_i * C, exact in int64 for realistic C
    base = numer // total            # floor(p_i * C / total)
    frac = numer - base * total      # fractional remainder numerator
    alloc = base.copy()
    remaining = c - int(base.sum())  # units left to hand out (> 0)

    # Hand out one unit at a time, largest remainder first (tie -> lowest
    # index). A slice's room (p_i - alloc_i) never runs out before the units
    # do, because sum(p) > C guarantees sum(p - alloc) > remaining.
    order = np.lexsort((np.arange(n), -frac))
    while remaining > 0:
        progressed = False
        for idx in order:
            if remaining == 0:
                break
            i = int(idx)
            if alloc[i] < p[i]:
                alloc[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:  # unreachable by the invariant above; guard anyway
            raise RuntimeError("projection failed to distribute all units")

    cut = alloc < p
    updates = tuple(
        VirtualUpdate(
            slice_id=i,
            proposed=int(p[i]),
            allocated=int(alloc[i]),
            deficit=int(p[i] - alloc[i]),
        )
        for i in range(n)
        if cut[i]
    )
    return ProjectionResult(alloc, p.copy(), c, cut, updates)
