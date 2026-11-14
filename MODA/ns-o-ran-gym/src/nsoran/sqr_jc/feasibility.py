"""Three-way feasibility classification (§46–§49) + quantile crossing rule.

Slice class of action a (§46), from the AUXILIARY quantiles of the
normalized margin m_i (never from the safe head):

    FEASIBLE    q25_i(m_i | z, a) ≥ 0
    INFEASIBLE  q75_i(m_i | z, a) < 0
    MARGINAL    otherwise

Support rule (§46): if q25/q75 are OUT OF SUPPORT for the slice, the slice
class is MARGINAL — never INFEASIBLE (§49: UNKNOWN != INFEASIBLE,
OUT_OF_SUPPORT != INFEASIBLE).

Quantile crossing (§10.1): classification uses the monotonic rearrangement
(q25_used = min(raw), q75_used = max(raw)); raw values are kept for logs
and every crossing is counted via ``quantile_crossing_rate`` (gate: > 5%
crossing in CAL/PILOT for any slice ⇒ STOP per §10.1).

Joint action class (§47) and state class (§48):

    JOINT_FEASIBLE    all slices FEASIBLE
    JOINT_MARGINAL    none confidently INFEASIBLE and ≥ 1 MARGINAL
    JOINT_INFEASIBLE  ≥ 1 slice INFEASIBLE

    state FEASIBLE    ∃ JOINT_FEASIBLE action in the evaluated set
    state MARGINAL    no JOINT_FEASIBLE but ∃ JOINT_MARGINAL (or the
                      evidence is out of support)
    state INFEASIBLE  ONLY if ALL actions of the validated space are
                      JOINT_INFEASIBLE AND the search has adequate
                      coverage/certification (exhaustive flag)

§39: a non-exhaustive search that finds nothing feasible emits
``MARGINAL_UNRESOLVED`` (analysis family: MARGINAL) + conservative fallback —
never INFEASIBLE.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from nsoran.sqr_jc.action_search import EvaluationResult

#: Slice-level classes (§46).
SLICE_FEASIBLE = "FEASIBLE"
SLICE_MARGINAL = "MARGINAL"
SLICE_INFEASIBLE = "INFEASIBLE"

#: Joint-action classes (§47).
JOINT_FEASIBLE = "JOINT_FEASIBLE"
JOINT_MARGINAL = "JOINT_MARGINAL"
JOINT_INFEASIBLE = "JOINT_INFEASIBLE"

#: State classes (§48 + §39).
STATE_FEASIBLE = "FEASIBLE"
STATE_MARGINAL = "MARGINAL"
STATE_INFEASIBLE = "INFEASIBLE"
STATE_MARGINAL_UNRESOLVED = "MARGINAL_UNRESOLVED"


class SliceClass(Enum):
    FEASIBLE = SLICE_FEASIBLE
    MARGINAL = SLICE_MARGINAL
    INFEASIBLE = SLICE_INFEASIBLE


@dataclass(frozen=True)
class SliceVerdict:
    """One slice's class of one action (with evidence bookkeeping)."""

    slice_id: int
    cls: str
    q25_raw: float
    q75_raw: float
    q25_used: float
    q75_used: float
    supported: bool
    crossed: bool


def classify_slice(
    slice_id: int, q25_raw: float, q75_raw: float, supported: bool
) -> SliceVerdict:
    """§46 rule (with §10.1 rearrangement for classification only)."""
    q25_used = min(q25_raw, q75_raw)
    q75_used = max(q25_raw, q75_raw)
    if not supported:
        cls = SLICE_MARGINAL
    elif q25_used >= 0.0:
        cls = SLICE_FEASIBLE
    elif q75_used < 0.0:
        cls = SLICE_INFEASIBLE
    else:
        cls = SLICE_MARGINAL
    return SliceVerdict(
        slice_id=slice_id,
        cls=cls,
        q25_raw=float(q25_raw),
        q75_raw=float(q75_raw),
        q25_used=float(q25_used),
        q75_used=float(q75_used),
        supported=bool(supported),
        crossed=bool(q25_raw > q75_raw),
    )


@dataclass(frozen=True)
class JointVerdict:
    """Classification of ONE action over the three slices (§47)."""

    action: tuple[int, int, int]
    slices: tuple[SliceVerdict, SliceVerdict, SliceVerdict]
    cls: str

    @property
    def n_infeasible(self) -> int:
        return sum(1 for s in self.slices if s.cls == SLICE_INFEASIBLE)

    @property
    def n_feasible(self) -> int:
        return sum(1 for s in self.slices if s.cls == SLICE_FEASIBLE)

    @property
    def n_marginal(self) -> int:
        return sum(1 for s in self.slices if s.cls == SLICE_MARGINAL)

    @property
    def non_infeasible_count(self) -> int:
        return 3 - self.n_infeasible

    @property
    def worst_q25(self) -> float:
        """§51 key 2: the worst (min) q25_used normalized margin."""
        return float(min(s.q25_used for s in self.slices))


def classify_joint(action: np.ndarray, result: EvaluationResult, index: int) -> JointVerdict:
    """Classify action ``result.actions[index]`` from the evaluation arrays."""
    a = tuple(int(v) for v in result.actions[index])
    slices = tuple(
        classify_slice(
            si,
            float(result.q25[index, si]),
            float(result.q75[index, si]),
            bool(result.supported[index, si]),
        )
        for si in range(3)
    )
    if any(s.cls == SLICE_INFEASIBLE for s in slices):
        cls = JOINT_INFEASIBLE
    elif all(s.cls == SLICE_FEASIBLE for s in slices):
        cls = JOINT_FEASIBLE
    else:
        cls = JOINT_MARGINAL
    return JointVerdict(action=a, slices=slices, cls=cls)


@dataclass(frozen=True)
class StateClassification:
    """§48/§39 state classification over one evaluated action set."""

    state_class: str                      # FEASIBLE | MARGINAL | INFEASIBLE | MARGINAL_UNRESOLVED
    n_joint_feasible: int
    n_joint_marginal: int
    n_joint_infeasible: int
    n_actions_evaluated: int
    n_actions_total_space: int
    exhaustive: bool
    out_of_support_actions: int

    @property
    def search_complete(self) -> bool:
        """The evaluated set is the WHOLE validated action space."""
        return bool(self.exhaustive and self.n_actions_evaluated >= self.n_actions_total_space)


def classify_state(result: EvaluationResult, total_space_size: int) -> StateClassification:
    """Vectorized §48 classification of the evaluated action set.

    INFEASIBLE is emitted ONLY when the evaluation is exhaustive over the
    validated space (§48's coverage requirement + §39's absolute rule).
    A non-exhaustive evaluation with zero joint-feasible actions yields
    ``MARGINAL_UNRESOLVED`` (§39 — belongs to the MARGINAL family).
    """
    if result.n_actions == 0:
        return StateClassification(
            state_class=STATE_MARGINAL_UNRESOLVED if not result.exhaustive else STATE_MARGINAL,
            n_joint_feasible=0,
            n_joint_marginal=0,
            n_joint_infeasible=0,
            n_actions_evaluated=0,
            n_actions_total_space=int(total_space_size),
            exhaustive=bool(result.exhaustive),
            out_of_support_actions=0,
        )

    # §10.1 rearrangement for classification (raw kept in result arrays).
    q25u = np.minimum(result.q25, result.q75)
    q75u = np.maximum(result.q25, result.q75)
    sup = result.supported
    n_sup = sup.sum(axis=1)              # supported slices per action

    feas_slice = (q25u >= 0.0) & sup
    infeas_slice = (q75u < 0.0) & sup
    marg_slice = ~(feas_slice | infeas_slice)  # includes out-of-support

    joint_feasible = np.all(feas_slice, axis=1)
    joint_infeasible = np.any(infeas_slice, axis=1)
    joint_marginal = ~(joint_feasible | joint_infeasible)

    n_feas = int(np.count_nonzero(joint_feasible))
    n_marg = int(np.count_nonzero(joint_marginal))
    n_inf = int(np.count_nonzero(joint_infeasible))
    out_support = int(np.count_nonzero(~np.all(sup, axis=1)))

    complete = bool(result.exhaustive) and result.n_actions >= total_space_size
    if n_feas > 0:
        state_class = STATE_FEASIBLE
    elif complete and n_marg == 0 and n_inf == result.n_actions and out_support == 0:
        # Every action of the validated, supported space is confidently
        # INFEASIBLE, and the search covered the whole space: §48 INFEASIBLE.
        state_class = STATE_INFEASIBLE
    elif result.exhaustive and complete:
        # Exhaustive but the state has marginal/out-of-support evidence.
        state_class = STATE_MARGINAL
    else:
        # Non-exhaustive search: absence of a feasible action is NOT
        # evidence of infeasibility (§39).
        state_class = STATE_MARGINAL_UNRESOLVED
    return StateClassification(
        state_class=state_class,
        n_joint_feasible=n_feas,
        n_joint_marginal=n_marg,
        n_joint_infeasible=n_inf,
        n_actions_evaluated=int(result.n_actions),
        n_actions_total_space=int(total_space_size),
        exhaustive=bool(result.exhaustive),
        out_of_support_actions=out_support,
    )


def quantile_crossing_rate(result: EvaluationResult, slice_id: int) -> float:
    """Crossing rate of slice ``slice_id`` over the evaluated actions (§10.1)."""
    if result.n_actions == 0:
        return 0.0
    crossed = result.q25[:, slice_id] > result.q75[:, slice_id]
    return float(np.count_nonzero(crossed) / result.n_actions)
