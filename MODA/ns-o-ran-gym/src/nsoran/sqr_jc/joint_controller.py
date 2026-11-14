"""SQR-JC joint controller (§29/§35/§39/§45/§50–§52).

Decision pipeline of ONE 100 ms window (everything below consumes ONLY the
pre-decision JointState of §11 — no future information):

1. bootstrap (§29): while ``t < bootstrap_seconds`` (0–100 s) ALL learning
   controllers emit the SAME PROP allocation and observe the same data
   (models may update online); from ``t >= bootstrap_seconds`` the joint
   policy decides.
2. exhaustive evaluation of the admissible space through the three slice
   models (vectorized; §36 evaluator is the runtime search until the §37
   gate replaces it — never silently).
3. floors from the full-space evaluation (§40–§44) — the search then
   restricts to a_i ≥ floor_i (§35).
4. state classification 3-way (§46–§49); a non-exhaustive search with no
   feasible action yields MARGINAL_UNRESOLVED + conservative fallback
   (floors), never INFEASIBLE (§39).
5. lexicographic selection (§50 FEASIBLE / §51 MARGINAL) or minimax
   graceful degradation (§52 INFEASIBLE) — NO continuous weights.
6. NO post-hoc uncertainty margin on the joint action (§45) — safety is
   carried by support filtering, floors, MARGINAL and uncertainty-aware
   tie-breaking.  The audited M state is maintained/logged only.

Tie-breaks (all classes): the FIRST action in the deterministic
enumeration order (a_e, a_u, a_m ascending) — ``np.lexsort`` is stable and
the enumeration order is the final key.  No RNG is used anywhere in the
decision path: identical (state, memory) ⇒ identical decision (replay
identity).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from nsoran.sqr.sqr_kpis import SqrKpmWindow
from nsoran.sqr.static_controllers import MODE_PROP, PropDemandController

from nsoran.sqr_jc.action_search import (
    EvaluationResult,
    JointActionOracleEvaluator,
)
from nsoran.sqr_jc.conditional_floors import FloorResult, compute_floors
from nsoran.sqr_jc.feature_scaler import FeatureScaler
from nsoran.sqr_jc.feasibility import (
    JOINT_FEASIBLE,
    JOINT_INFEASIBLE,
    JOINT_MARGINAL,
    STATE_FEASIBLE,
    STATE_INFEASIBLE,
    STATE_MARGINAL,
    STATE_MARGINAL_UNRESOLVED,
    SliceVerdict,
    JointVerdict,
    classify_joint,
    classify_slice,
    classify_state,
)
from nsoran.sqr_jc.joint_action_space import (
    ACTION_SEMANTICS,
    C_PRB,
    NUM_ACTIONS_ALLOCATED_USAGE_C53,
    switching_distance,
    validate_action,
)
from nsoran.sqr_jc.joint_state import JointState
from nsoran.sqr_jc.kernel_models import MultiQuantileKernelModel
from nsoran.sqr_jc.online_update import OnlineUpdater

#: Shared bootstrap controller of §29 (PROP; the same controller every
#: learning controller runs 0–100 s).
BOOTSTRAP_SECONDS = 100.0

#: Controller identity logged in the §56 ``controller`` column.
CONTROLLER_NAME = "sqr_jc"

#: Search modes logged in §56 ``search_mode``.
SEARCH_EXHAUSTIVE = "exhaustive"
SEARCH_COARSE_TO_FINE = "coarse_to_fine"


@dataclass(frozen=True)
class JointDecision:
    """One SQR-JC decision (policy + evidence for the §56 log row)."""

    action: tuple[int, int, int]
    bootstrap: bool
    state_class: str
    floors: FloorResult
    n_actions_total: int
    n_actions_evaluated: int
    n_joint_feasible: int
    n_joint_marginal: int
    n_joint_infeasible: int
    selected_verdict: object           # feasibility.JointVerdict of the chosen action
    safe_q: tuple[float, float, float]
    q25_raw: tuple[float, float, float]
    q75_raw: tuple[float, float, float]
    q25_used: tuple[float, float, float]
    q75_used: tuple[float, float, float]
    sigma: tuple[float, float, float]
    supported: tuple[bool, bool, bool]
    crossing: tuple[bool, bool, bool]
    search_mode: str
    decision_time_ms: float


class JointController:
    """SQR-JC policy over the three kernelized slice models."""

    def __init__(
        self,
        models: tuple[MultiQuantileKernelModel, MultiQuantileKernelModel, MultiQuantileKernelModel],
        scaler: FeatureScaler,
        sigma_support: float,
        c: int = C_PRB,
        bootstrap_seconds: float = BOOTSTRAP_SECONDS,
        bootstrap_controller: Optional[PropDemandController] = None,
    ) -> None:
        if len(models) != 3:
            raise ValueError(f"need exactly 3 slice models, got {len(models)}")
        if not isinstance(scaler, FeatureScaler):
            raise TypeError("scaler must be a frozen FeatureScaler (§13)")
        s = float(sigma_support)
        if not (0.0 <= s <= 1.0):
            raise ValueError(f"sigma_support must be in [0, 1], got {s}")
        self.models = tuple(models)
        self.scaler = scaler
        self.sigma_support = s
        self.c = int(c)
        self.bootstrap_seconds = float(bootstrap_seconds)
        self.bootstrap_controller = bootstrap_controller or PropDemandController(
            slice_id=0, n_prbs=self.c
        )
        self.evaluator = JointActionOracleEvaluator(
            self.models, sigma_support=s, c=self.c
        )
        self.updater = OnlineUpdater(self.models, scaler=self.scaler, c=self.c)
        self.prev_action: tuple[int, int, int] = (0, 0, 0)
        self.search_mode = SEARCH_EXHAUSTIVE  # runtime search chosen by the §37 gate

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(
        self,
        state: JointState,
        now_s: float,
        kpm_window: Optional[SqrKpmWindow] = None,
    ) -> JointDecision:
        """One joint decision for the window after ``state`` (§29/§50–§52)."""
        if float(now_s) < self.bootstrap_seconds:
            return self._decide_bootstrap(state, now_s, kpm_window)
        return self._decide_joint(state)

    def _decide_bootstrap(
        self,
        state: JointState,
        now_s: float,
        kpm_window: Optional[SqrKpmWindow],
    ) -> JointDecision:
        """§29: same PROP allocation for every learning controller."""
        t0 = time.perf_counter()
        split = self.bootstrap_controller.prop_split(kpm_window)
        action = tuple(int(v) for v in split)
        validate_action(action, self.c)
        action = (action[0], action[1], action[2])
        floors = FloorResult(
            floors=(0, 0, 0),
            source=("bootstrap_prop", "bootstrap_prop", "bootstrap_prop"),
            witness_action=None,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        # Placeholder verdict (no model evidence consumed while bootstrapping).
        verdict = JointVerdict(
            action=action,
            slices=tuple(
                SliceVerdict(si, "BOOTSTRAP_PROP", 0.0, 0.0, 0.0, 0.0, True, False)
                for si in range(3)
            ),
            cls="BOOTSTRAP_PROP",
        )
        return JointDecision(
            action=action,
            bootstrap=True,
            state_class="BOOTSTRAP_PROP",
            floors=floors,
            n_actions_total=NUM_ACTIONS_ALLOCATED_USAGE_C53,
            n_actions_evaluated=0,
            n_joint_feasible=0,
            n_joint_marginal=0,
            n_joint_infeasible=0,
            selected_verdict=verdict,
            safe_q=(0.0, 0.0, 0.0),
            q25_raw=(0.0, 0.0, 0.0),
            q75_raw=(0.0, 0.0, 0.0),
            q25_used=(0.0, 0.0, 0.0),
            q75_used=(0.0, 0.0, 0.0),
            sigma=(1.0, 1.0, 1.0),
            supported=(False, False, False),
            crossing=(False, False, False),
            search_mode=MODE_PROP,
            decision_time_ms=elapsed,
        )

    def _decide_joint(self, state: JointState) -> JointDecision:
        """Full joint pipeline (§35–§52). Deterministic; no RNG."""
        t0 = time.perf_counter()

        # 1. exhaustive evaluation of the WHOLE admissible space (no floors).
        z_scaled = state.vector(self.scaler)
        result = self.evaluator.evaluate(z_scaled)  # exhaustive over 27,720

        # 2. floors from the full-space evidence (§40–§44).
        floors = compute_floors(
            result,
            n_active=(state.n_e, state.n_u, state.n_m),
            queues=(state.queue_e, state.queue_u, state.queue_m),
            c=self.c,
        )

        # 3. state classification over the exhaustive evaluation (§48).
        classification = classify_state(result, NUM_ACTIONS_ALLOCATED_USAGE_C53)

        # 4. restrict to actions respecting the active floors (§35).
        floor_mask = np.all(result.actions >= np.asarray(floors.floors)[None, :], axis=1)
        total_space = NUM_ACTIONS_ALLOCATED_USAGE_C53
        if not np.any(floor_mask):  # defensive (cannot happen: floors ≤ C and
            # §43 fallbacks sum 16 ≤ C with the enumeration containing them)
            floor_mask = np.ones(result.n_actions, dtype=bool)

        # 5. selection per state class (§50/§51/§52/§39).
        if classification.state_class == STATE_FEASIBLE:
            index = self._select_feasible(result, floor_mask)
        elif classification.state_class == STATE_MARGINAL:
            index = self._select_marginal(result, floor_mask)
        elif classification.state_class == STATE_MARGINAL_UNRESOLVED:
            index = self._select_marginal(result, floor_mask)
        elif classification.state_class == STATE_INFEASIBLE:
            index = self._select_infeasible(result, floor_mask)
        else:  # pragma: no cover — classification is closed
            index = self._select_infeasible(result, floor_mask)

        action = tuple(int(v) for v in result.actions[index])
        validate_action(action, self.c, floors.floors)
        verdict = classify_joint(result.actions[index], result, index)

        elapsed = (time.perf_counter() - t0) * 1000.0
        self.prev_action = action

        return JointDecision(
            action=action,
            bootstrap=False,
            state_class=classification.state_class,
            floors=floors,
            n_actions_total=total_space,
            n_actions_evaluated=classification.n_actions_evaluated,
            n_joint_feasible=classification.n_joint_feasible,
            n_joint_marginal=classification.n_joint_marginal,
            n_joint_infeasible=classification.n_joint_infeasible,
            selected_verdict=verdict,
            safe_q=tuple(float(result.safe[index, si]) for si in range(3)),
            q25_raw=tuple(float(result.q25[index, si]) for si in range(3)),
            q75_raw=tuple(float(result.q75[index, si]) for si in range(3)),
            q25_used=tuple(float(min(result.q25[index, si], result.q75[index, si])) for si in range(3)),
            q75_used=tuple(float(max(result.q25[index, si], result.q75[index, si])) for si in range(3)),
            sigma=tuple(float(result.sigma[index, si]) for si in range(3)),
            supported=tuple(bool(result.supported[index, si]) for si in range(3)),
            crossing=tuple(bool(result.q25[index, si] > result.q75[index, si]) for si in range(3)),
            search_mode=self.search_mode,
            decision_time_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Lexicographic / minimax selection (vectorized; stable enumeration ties)
    # ------------------------------------------------------------------

    def _lexsort_keys(self, result: EvaluationResult, mask: np.ndarray, keys: list[np.ndarray]) -> int:
        """np.lexsort: the LAST key in the list is the PRIMARY sort key and
        the first is the least important.  The enumeration index is placed
        FIRST (least priority) so ties always break to the deterministic
        enumeration order (§50/§51/§52 final key)."""
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            raise RuntimeError(
                "selection over an EMPTY floor-respecting set: floors must "
                "come from compute_floors (jointly admissible by "
                "construction, §41/§35)"
            )
        stacked = [idx.astype(np.float64)] + [k[idx] for k in keys]
        order = np.lexsort(stacked)
        return int(idx[order[0]])

    @staticmethod
    def _severity_columns(result: EvaluationResult) -> np.ndarray:
        """Predicted severity per slice: max(0, −safe margin) — margin scale."""
        return np.maximum(0.0, -result.safe)

    def _select_feasible(self, result: EvaluationResult, mask: np.ndarray) -> int:
        """§50 FEASIBLE keys (all minimized lexicographically):
        1. aggregate quantile risk Σ_i max(0, −safe_i)
        2. total uncertainty Σ_i σ_i
        3. switching distance |a − a(t−1)|₁
        4. total resources Σ a_i (ALLOCATED_USAGE allows Σ < C)
        5. deterministic tie-break (enumeration order)."""
        risk = self._severity_columns(result).sum(axis=1)
        uncertainty = result.sigma.sum(axis=1)
        switching = np.abs(
            result.actions - np.asarray(self.prev_action, dtype=np.int64)[None, :]
        ).sum(axis=1).astype(np.float64)
        total = result.actions.sum(axis=1).astype(np.float64)
        return self._lexsort_keys(
            result, mask, [total, switching, uncertainty, risk]
        )

    def _select_marginal(self, result: EvaluationResult, mask: np.ndarray) -> int:
        """§51 MARGINAL keys:
        1. MAXIMIZE non-INFEASIBLE slice count     -> minimize −count
        2. MAXIMIZE worst q25_used normalized margin -> minimize −worst
        3. minimize total uncertainty Σ σ
        4. minimize switching
        5. deterministic tie-break."""
        q25u = np.minimum(result.q25, result.q75)
        q75u = np.maximum(result.q25, result.q75)
        sup = result.supported
        feas = (q25u >= 0.0) & sup
        infeas = (q75u < 0.0) & sup
        non_infeasible = (3 - infeas.sum(axis=1)).astype(np.float64)
        worst_q25 = q25u.min(axis=1)   # §51 key 2: worst slice q25_used
        uncertainty = result.sigma.sum(axis=1)
        switching = np.abs(
            result.actions - np.asarray(self.prev_action, dtype=np.int64)[None, :]
        ).sum(axis=1).astype(np.float64)
        return self._lexsort_keys(
            result, mask, [switching, uncertainty, -worst_q25, -non_infeasible]
        )

    def _select_infeasible(self, result: EvaluationResult, mask: np.ndarray) -> int:
        """§52 INFEASIBLE (graceful degradation, minimax fairness):
        1. minimize maximum predicted normalized violation severity
        2. minimize sum of predicted severities
        3. minimize switching
        4. deterministic tie-break."""
        sev = self._severity_columns(result)
        max_sev = sev.max(axis=1)
        sum_sev = sev.sum(axis=1)
        switching = np.abs(
            result.actions - np.asarray(self.prev_action, dtype=np.int64)[None, :]
        ).sum(axis=1).astype(np.float64)
        return self._lexsort_keys(result, mask, [switching, sum_sev, max_sev])

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def switch_search_mode(self, mode: str) -> None:
        """Runtime search chosen by the §37/§38 gates (explicit, never silent)."""
        if mode not in (SEARCH_EXHAUSTIVE, SEARCH_COARSE_TO_FINE):
            raise ValueError(f"unknown search mode {mode!r}")
        self.search_mode = mode
