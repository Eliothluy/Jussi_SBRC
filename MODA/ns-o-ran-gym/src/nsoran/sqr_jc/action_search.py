"""Joint action search (§36–§38): exhaustive vectorized + coarse-to-fine.

``JointActionOracleEvaluator`` (§36) is the REFERENCE exhaustive evaluator:
for one joint state it evaluates the FULL enumerated action set through the
three slice models (vectorized in blocks) and returns every head's raw
predictions, sigmas and support flags — used offline (calibration, replay,
runtime-search benchmarking) and as the runtime search when §37's gate
passes.

``ExhaustiveSearch`` is the runtime wrapper that applies floors + the
feasibility/selection machinery on top of the oracle evaluator.

``CoarseToFineSearch`` (§38, authorized ONLY if the exhaustive gate fails):
deterministic two-stage grid search on the normalized simplex (coarse step
s_c, refine one coarse cell with step 1).  Parameters may be tuned with CAL
data ONLY; it must pass the §38 quality gate against the oracle evaluator
before MAIN (feasibility-class agreement ≥98%, median normalized objective
regret ≤2%, p95 ≤5%, p95 decision time ≤80 ms).  Until its certificate is
validated, a search that finds no feasible action can only produce
MARGINAL_UNRESOLVED — never INFEASIBLE (§39).

``benchmark_exhaustive`` implements the §37 gate harness: decision wall-time
over a list of joint states (the CAL states at runtime), reporting
median/p95 — the orchestrator runs it against real CAL states
(``03_calibration/run_cal.sh`` produces them; PASS at p95 ≤ 80 ms).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from nsoran.sqr_jc.joint_action_space import C_PRB, enumerate_actions, filter_feasible_mask
from nsoran.sqr_jc.kernel_models import MultiQuantileKernelModel

#: Kernel-evaluation block size (bounds temporary memory: 4096×2000×8B ≈ 65MB).
_BLOCK = 4096

#: sigma_support percentile candidates (§16 calibration grid is external; this
#: module only consumes the frozen threshold value).
SUPPORT_SIGMA_MAX = 1.0


@dataclass(frozen=True)
class EvaluationResult:
    """Full oracle-evaluator output for ONE state over the action set."""

    actions: np.ndarray            # (N, 3) int64, the evaluated set
    safe: np.ndarray               # (N, 3) safe-head margin predictions
    q25: np.ndarray                # (N, 3) raw q25
    q75: np.ndarray                # (N, 3) raw q75
    sigma: np.ndarray              # (N, 3) uncertainties
    supported: np.ndarray          # (N, 3) bool, sigma <= sigma_support
    eval_time_ms: float
    exhaustive: bool = True

    @property
    def n_actions(self) -> int:
        return int(self.actions.shape[0])


class JointActionOracleEvaluator:
    """§36 reference evaluator: full action set through the slice models."""

    def __init__(
        self,
        models: Sequence[MultiQuantileKernelModel],
        sigma_support: float,
        c: int = C_PRB,
    ) -> None:
        if len(models) != 3:
            raise ValueError(f"need exactly 3 slice models, got {len(models)}")
        s = float(sigma_support)
        if not (0.0 <= s <= 1.0):
            raise ValueError(f"sigma_support must be in [0, 1], got {s}")
        self.models = tuple(models)
        self.sigma_support = s
        self.c = int(c)
        self.actions = enumerate_actions(self.c)

    def evaluate(
        self,
        z_scaled: np.ndarray,
        actions: Optional[np.ndarray] = None,
        floors: tuple[int, int, int] = (0, 0, 0),
    ) -> EvaluationResult:
        """Evaluate actions at the SCALED joint state z (23,) — vectorized.

        ``floors`` restricts the evaluated set (§35: an action may not
        violate an active floor).  Head features are x = [z, a/C].
        """
        A = self.actions if actions is None else np.asarray(actions, dtype=np.int64)
        if A.ndim != 2 or A.shape[1] != 3:
            raise ValueError(f"actions must be (N, 3), got {A.shape}")
        keep = filter_feasible_mask(A, self.c, floors)
        A = A[keep]

        z = np.asarray(z_scaled, dtype=np.float64).flatten()
        if z.shape[0] != self.models[0].dim - 3:
            raise ValueError(
                f"z dim {z.shape[0]} != model dim − 3 "
                f"({self.models[0].dim - 3})"
            )
        n = A.shape[0]
        safe = np.empty((n, 3), dtype=np.float64)
        q25 = np.empty((n, 3), dtype=np.float64)
        q75 = np.empty((n, 3), dtype=np.float64)
        sigma = np.empty((n, 3), dtype=np.float64)
        supported = np.empty((n, 3), dtype=bool)

        t0 = time.perf_counter()
        # Action block: x_block = [z, a/C] shared across the three models.
        for start in range(0, max(n, 1), _BLOCK):
            stop = min(start + _BLOCK, n)
            if stop <= start:
                break
            ab = A[start:stop].astype(np.float64) / float(self.c)
            X = np.column_stack([np.tile(z, (stop - start, 1)), ab])
            for si, model in enumerate(self.models):
                preds = model.predict_raw(X)
                safe[start:stop, si] = preds["safe"]
                q25[start:stop, si] = preds["q25"]
                q75[start:stop, si] = preds["q75"]
                sigma[start:stop, si] = preds["sigma"]
                supported[start:stop, si] = preds["sigma"] <= self.sigma_support
        elapsed = (time.perf_counter() - t0) * 1000.0
        return EvaluationResult(
            actions=A,
            safe=safe,
            q25=q25,
            q75=q75,
            sigma=sigma,
            supported=supported,
            eval_time_ms=elapsed,
            exhaustive=True,
        )


# ---------------------------------------------------------------------------
# Coarse-to-fine (§38 — authorized fallback; certificate required)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoarseToFineParams:
    """Deterministic search parameters (tunable with CAL only, then frozen)."""

    coarse_step: int = 4
    refine_radius: int = 4


class CoarseToFineSearch:
    """Two-stage deterministic search over the ALLOCATED_USAGE simplex.

    Stage 1 evaluates the coarse grid (step ``coarse_step``); stage 2
    refines, with unit step, a ±``refine_radius`` unit box around the best
    coarse corner (by the SAME lexicographic key the runtime selector
    uses — passed in as ``key_fn``).  Coverage is a documented certificate,
    NOT exhaustiveness: the caller must keep §39's rule (no INFEASIBLE
    from an incomplete search) and pass §38's agreement gate before MAIN.
    """

    def __init__(
        self,
        evaluator: JointActionOracleEvaluator,
        params: CoarseToFineParams = CoarseToFineParams(),
    ) -> None:
        self.evaluator = evaluator
        self.params = params

    def candidate_set(self, c: int | None = None) -> np.ndarray:
        """Deterministic candidate actions: coarse grid + full unit refinement
        happens per-state; this returns the coarse grid for the certificate."""
        c = self.evaluator.c if c is None else int(c)
        step = max(1, int(self.params.coarse_step))
        return enumerate_actions(c, step=step)

    def search(
        self,
        z_scaled: np.ndarray,
        floors: tuple[int, int, int],
        key_fn,
    ) -> tuple[EvaluationResult, Optional[tuple[int, int, int]]]:
        """Return (evaluated union, argmin by ``key_fn``) or (…, None)."""
        c = self.evaluator.c
        coarse = self.candidate_set(c)
        stage1 = self.evaluator.evaluate(z_scaled, actions=coarse, floors=floors)
        if stage1.n_actions == 0:
            return stage1, None
        order = np.lexsort(key_fn(stage1))
        best = stage1.actions[order[0]]
        lo = np.maximum(best - self.params.refine_radius, 0)
        hi = np.minimum(best + self.params.refine_radius, c)
        fine = enumerate_actions(int(hi.max() - lo.min()) + 1)  # local unit grid
        # clip the unit grid to the refinement box:
        keep = np.all((fine >= lo[None, :]) & (fine <= hi[None, :]), axis=1)
        fine = fine[keep]
        stage2 = self.evaluator.evaluate(z_scaled, actions=fine, floors=floors)
        actions = np.vstack([stage1.actions, stage2.actions])
        exhaustive = False
        result = EvaluationResult(
            actions=actions,
            safe=np.vstack([stage1.safe, stage2.safe]),
            q25=np.vstack([stage1.q25, stage2.q25]),
            q75=np.vstack([stage1.q75, stage2.q75]),
            sigma=np.vstack([stage1.sigma, stage2.sigma]),
            supported=np.vstack([stage1.supported, stage2.supported]),
            eval_time_ms=stage1.eval_time_ms + stage2.eval_time_ms,
            exhaustive=exhaustive,
        )
        order = np.lexsort(key_fn(result))
        return result, tuple(int(v) for v in result.actions[order[0]])


# ---------------------------------------------------------------------------
# §37 runtime gate harness
# ---------------------------------------------------------------------------

def benchmark_exhaustive(
    evaluator: JointActionOracleEvaluator,
    z_states_scaled: Sequence[np.ndarray],
) -> dict:
    """Benchmark the exhaustive evaluator over states (§37: ≥100 CAL states
    covering low/mid/high load; PASS at p95 decision time ≤ 80 ms).

    Returns {n_states, median_ms, p95_ms, p99_ms, max_ms, n_actions,
    passed_p95_80ms}.  The ORCHESTRATOR runs this on real CAL states — the
    preflight only validates the harness on synthetic states.
    """
    times = []
    n_actions = 0
    for z in z_states_scaled:
        res = evaluator.evaluate(np.asarray(z, dtype=np.float64))
        times.append(res.eval_time_ms)
        n_actions = res.n_actions
    arr = np.asarray(times, dtype=np.float64)
    out = {
        "n_states": int(arr.size),
        "median_ms": float(np.median(arr)) if arr.size else float("nan"),
        "p95_ms": float(np.percentile(arr, 95)) if arr.size else float("nan"),
        "p99_ms": float(np.percentile(arr, 99)) if arr.size else float("nan"),
        "max_ms": float(arr.max()) if arr.size else float("nan"),
        "n_actions": int(n_actions),
        "passed_p95_80ms": bool(arr.size and np.percentile(arr, 95) <= 80.0),
    }
    return out
