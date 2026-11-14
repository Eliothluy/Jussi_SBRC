"""State-conditioned protection floors (§40–§44).

NOT hardcoded universal floors (§40 forbids permanent URLLC=13/mMTC=3 as
global rules).  The floor of slice i at state z is the SMALLEST resource
level sustained by safe kernelized evidence at that state (§41, operational
form): scanning the exhaustive evaluation, floor_i = min { a_i : ∃
supported action a with that a_i whose SAFE head predicts m_i ≥ 0 }.  The
"does not require clearly infeasible behavior from the others" check of
§41 step 3 is implemented by requiring the witness action itself to be
within the admissible space (Σa ≤ C — enforced by the enumeration): a
witness is a fully admissible joint action, so the floor never demands
more than the cell capacity.

No-demand rule (§42): floor_i = 0 when N_active_i == 0 and queue_i == 0
(and no pending retransmission — NOT OBSERVABLE in the frozen KPM contract,
documented; the condition reduces to the two observable signals).

Fallback (§43): when NO supported safe evidence exists for slice i, the
conservative fallback applies.  The G1 audit confirms same unit and
semantics of Round 2 (R27 absolute RBG quota, 1 RBG = 1 PRB), so the
frozen historical fallbacks are used:

    URLLC fallback = 13      (audited Round-2 protective allocation)
    mMTC  fallback = 3
    eMBB  fallback = 0       (no historical eMBB floor exists in the
                              audited code — 0 = no protection claim;
                              CAL may recalibrate via fallback_from_cal
                              before PILOT, §43 procedure)

Historical URLLC=5 / mMTC=1 remain DIAGNOSTIC ANCHORS only (§44) — never
applied as rules.

floors are recomputed per decision from the state's evaluation; they enter
the search as the §35 lower bound a_i ≥ floor_i(z).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from nsoran.sqr_jc.action_search import EvaluationResult

#: §43 conservative fallbacks (audited same-unit condition met by G1).
FALLBACK_URLLC = 13
FALLBACK_MMTC = 3
FALLBACK_EMBB = 0  # no historical eMBB floor in the audited code (documented)

#: §44 diagnostic anchors — NEVER a rule; calibration comparisons only.
DIAGNOSTIC_ANCHOR_URLLC = 5
DIAGNOSTIC_ANCHOR_MMTC = 1

#: Slice order convention (0=eMBB, 1=URLLC, 2=mMTC — contract v1.4 §4.2/R35).
FALLBACKS: tuple[int, int, int] = (FALLBACK_EMBB, FALLBACK_URLLC, FALLBACK_MMTC)


@dataclass(frozen=True)
class FloorResult:
    """Per-slice floor for one decision, with evidence provenance."""

    floors: tuple[int, int, int]
    source: tuple[str, str, str]   # "no_demand" | "evidence" | "fallback" | "anchor"
    witness_action: Optional[tuple[int, int, int]]  # first evidence witness


def compute_floors(
    result: EvaluationResult,
    n_active: Sequence[int],
    queues: Sequence[float],
    c: int,
    pending_retransmission: Sequence[bool] = (False, False, False),
) -> FloorResult:
    """§41/§42/§43 floors from one exhaustive evaluation of the state.

    Requires an EXHAUSTIVE evaluation (floors defined by §41 scan the full
    supported set); callers pass the oracle evaluator's result.

    JOINT ADMISSIBILITY BY CONSTRUCTION (§41 step 3): the evidence floors
    are the componentwise minima of a_i over the set of SUPPORTED JOINTLY
    SAFE actions (safe head ≥ 0 for EVERY slice on the SAME action).  The
    componentwise minimum of admissible vectors satisfies Σ floors ≤ Σ of
    any member ≤ C, so the floor triple can never make the simplex empty
    (the failure mode this rule exists to prevent: per-slice minima taken
    from DIFFERENT witnesses can sum above C).

    Slices with no jointly-safe supported evidence use the §43 fallback;
    if the mixed triple (evidence + fallbacks) still exceeds C, fallback
    floors are dropped largest-first (a fallback is a preference, not
    evidence — dropping it keeps the triple admissible without touching
    evidenced floors).
    """
    if len(n_active) != 3 or len(queues) != 3:
        raise ValueError("n_active/queues must have 3 entries")
    if not result.exhaustive:
        raise ValueError(
            "compute_floors requires an exhaustive evaluation (§41 scans the "
            "full supported set; non-exhaustive floors would be ungrounded)"
        )

    floors: list[int] = []
    sources: list[str] = []
    witness: Optional[tuple[int, int, int]] = None

    # §10.1 rearrangement is NOT needed on the safe head (single head).
    q_safe = result.safe
    sup = result.supported

    # Set of SUPPORTED JOINTLY SAFE actions (every slice safe + supported).
    joint_safe = np.all((q_safe >= 0.0) & sup, axis=1)
    has_joint_evidence = bool(np.any(joint_safe))
    if has_joint_evidence:
        witness_row = result.actions[int(np.flatnonzero(joint_safe)[0])]
        witness = tuple(int(v) for v in witness_row)

    for si in range(3):
        # §42 no-demand rule (pending/retransmission not observable in the
        # frozen KPM contract — caller passes False; documented absence).
        if int(n_active[si]) == 0 and float(queues[si]) == 0.0 and not pending_retransmission[si]:
            floors.append(0)
            sources.append("no_demand")
            continue

        if has_joint_evidence:
            # §41: smallest a_i over the jointly-safe supported set.
            floors.append(int(np.min(result.actions[joint_safe, si])))
            sources.append("evidence")
        else:
            # §43 fallback (same unit/semantics confirmed by the G1 audit).
            floors.append(int(FALLBACKS[si]))
            sources.append("fallback")

    # Joint-admissibility cap: drop FALLBACK floors largest-first while
    # Σ > C (evidence floors from the jointly-safe set satisfy Σ ≤ C by
    # the componentwise-minimum argument; no-evidence fallbacks yield).
    floors = [min(int(f), int(c)) for f in floors]
    while sum(floors) > int(c):
        candidates = [i for i, s in enumerate(sources) if s == "fallback" and floors[i] > 0]
        if not candidates:
            break  # defensive: cannot happen (see the joint-safe argument)
        i = max(candidates, key=lambda j: floors[j])
        floors[i] = 0
    return FloorResult(
        floors=tuple(floors),
        source=tuple(sources),
        witness_action=witness,
    )


def fallback_from_cal(
    cal_states: Sequence[dict],
    sla_ok_flags: Sequence[Sequence[bool]],
    max_allocation: int,
    target_coverage: float = 0.95,
) -> int:
    """§43 recalibration procedure (used only when historical fallbacks are
    NOT compatible): the smallest allocation that keeps the SLA in ≥95% of
    supported CAL states.  ``cal_states`` entries carry ``a_i`` under key
    "a_i"; ``sla_ok_flags[i]`` is the SLA-ok boolean of sample i.  Frozen
    before PILOT; returns the smallest a meeting the coverage.
    """
    if len(cal_states) != len(sla_ok_flags):
        raise ValueError("state/flag length mismatch")
    for a in range(0, int(max_allocation) + 1):
        ok = [ok_i for s, ok_i in zip(cal_states, sla_ok_flags) if int(s["a_i"]) >= a]
        if not ok:
            continue
        if float(np.mean(ok)) >= target_coverage:
            return a
    return int(max_allocation)
