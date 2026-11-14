"""Joint action space with the AUDITED semantics (§21/§22/§35).

G1 conclusion frozen in ``resource_unit_manifest.md`` (read the code:
``sqr-mac-scheduler.cc`` + ``rslaq-mac-scheduler.cc`` + telemetry):

- C = 53 is simultaneously the number of RBGs of the DL grid AND the number
  of PRBs: 5G-LENA's ``NrHelper`` attribute ``NumRbPerRbg`` defaults to 1
  (nr-helper.cc:158-161), so each RBG covers exactly 1 PRB, and the RB
  count is floor(10 MHz × (1 − 0.04 RB overhead) / 180 kHz) = 53
  (nr-phy.cc:584, nr-helper.cc:155).  Frozen telemetry confirms it
  byte-for-byte: ``total_rbg=53, total_prb=53``.
- The action is an ABSOLUTE per-DL-slot RBG quota: the scheduler serves
  each slice UP TO its quota; unused budget is NEVER redistributed (stays
  idle) and the quota is never exceeded (register R27,
  sqr-mac-scheduler.h:19-24, .cc:494-498/716-719).

Therefore ACTION_SEMANTICS = **ALLOCATED_USAGE** (§22): the admissible set is

    a = (a_e, a_u, a_m) ∈ Z³≥0,  a_i ≤ C,  Σ a_i ≤ C      (idle allowed)

with **27,720** integer combinations for C=53 (vs 1,485 for the Σ = C
BUDGET_PARTITION alternative — computed here only as a documented
comparison; the partition reading is REFUTED by R27's idle-budget
semantics, so it is NOT the action space of SQR-JC).

§35 invariants enforced here: no negative action, no action above C, no
violation of an active floor, unit = scheduler unit (RBG=PRB).
"""

from __future__ import annotations

import numpy as np

#: Effective cell capacity (R01 as amended; telemetry total_rbg = 53).
C_PRB = 53

#: ACTION_SEMANTICS frozen by the G1 audit (protocol §22).
ACTION_SEMANTICS = "ALLOCATED_USAGE"

#: Number of ALLOCATED_USAGE combinations for C=53: C(56,3) = 27,720.
NUM_ACTIONS_ALLOCATED_USAGE_C53 = 27720

#: Number of BUDGET_PARTITION combinations for C=53: C(55,2) = 1,485
#: (documented comparison only — NOT the frozen semantics).
NUM_ACTIONS_BUDGET_PARTITION_C53 = 1485


def num_actions_budget_partition(c: int) -> int:
    """|{(a_e,a_u,a_m) ≥ 0 : Σa = c}| = (c+1)(c+2)/2."""
    c = int(c)
    return (c + 1) * (c + 2) // 2


def num_actions_allocated_usage(c: int) -> int:
    """|{(a_e,a_u,a_m) ≥ 0 : Σa ≤ c}| = (c+1)(c+2)(c+3)/6."""
    c = int(c)
    return (c + 1) * (c + 2) * (c + 3) // 6


def enumerate_actions(c: int = C_PRB, step: int = 1) -> np.ndarray:
    """All ALLOCATED_USAGE actions, lexicographic (a_e, a_u, a_m ascending).

    Vectorized in blocks; returns an int64 array (N, 3).  N = 27,720 for
    C=53, step 1 — the deterministic order used by every tie-break.
    """
    if int(step) < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    grid = np.arange(0, c + 1, int(step), dtype=np.int64)
    blocks = []
    for ae in grid:
        for au in grid:
            rem = c - int(ae) - int(au)
            if rem < 0:
                break
            am = grid[grid <= rem]
            blocks.append(np.column_stack([
                np.full(am.shape, int(ae), dtype=np.int64),
                np.full(am.shape, int(au), dtype=np.int64),
                am,
            ]))
    if not blocks:
        return np.zeros((0, 3), dtype=np.int64)
    return np.vstack(blocks)


def filter_feasible_mask(
    actions: np.ndarray, c: int = C_PRB, floors: tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """§35 invariant mask: 0 ≤ a_i ≤ C, Σa ≤ C, a_i ≥ floor_i (active floors).

    Works on an (N, 3) int array; returns a bool mask.
    """
    A = np.asarray(actions, dtype=np.int64)
    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError(f"expected (N, 3) actions, got {A.shape}")
    floors_arr = np.asarray(floors, dtype=np.int64)
    if np.any(floors_arr < 0) or np.any(floors_arr > c):
        raise ValueError(f"floors {floors} outside [0, {c}]")
    ok = np.all(A >= floors_arr[None, :], axis=1)
    ok &= np.all(A <= int(c), axis=1)
    ok &= A.sum(axis=1) <= int(c)
    return ok


def validate_action(action: tuple[int, int, int], c: int = C_PRB,
                    floors: tuple[int, int, int] = (0, 0, 0)) -> tuple[int, int, int]:
    """Validate ONE action against §35; raises on any violation."""
    a = tuple(int(v) for v in action)
    for name, v in zip(("a_e", "a_u", "a_m"), a):
        if v < 0:
            raise ValueError(f"{name} negative: {a}")
        if v > int(c):
            raise ValueError(f"{name} exceeds C={c}: {a}")
    if sum(a) > int(c):
        raise ValueError(f"Σa exceeds C={c}: {a}")
    for i, (v, f) in enumerate(zip(a, floors)):
        if v < int(f):
            raise ValueError(f"action {a} violates active floor {i}={f}")
    return a


def switching_distance(a: tuple[int, int, int], prev: tuple[int, int, int]) -> int:
    """L1 switching distance |a − a(t−1)|₁ (§50 key 3 / §51 key 4)."""
    return int(np.abs(np.asarray(a, dtype=np.int64) - np.asarray(prev, dtype=np.int64)).sum())
