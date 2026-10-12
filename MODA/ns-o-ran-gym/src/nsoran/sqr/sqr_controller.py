"""Hierarchical safety-first policy of one SQR slice controller.

Per slice i, the SQR policy [ARTIGO, PLANO_IMPLEMENTACAO_SQR_NORI.md §2.1]:

1. Feature per candidate action:  x = [z, a / C]  with integer actions
   a in {0, ..., C} (C = total PRBs of the cell).
2. Safe set:  A_safe = { a : f_tau(z, a) >= Lambda }  (Lambda = SLA level).
3. Confident mode (A_safe non-empty):
       a* = argmax_{a in A_safe} [ f_tau(z, a) - gamma * a / C ]
   gamma ALTERNATES ONLY the resource-cost term (AGENTS.md §5 rule 7):
   gamma = 0 maximizes the quantile alone; gamma = 1 trades quantile for
   PRB economy. The critical-failure penalty P_penalty stays active in
   both variants.
4. Uncertain mode (A_safe empty):
     - with probability epsilon = 0.1 [ARTIGO]: softmax sampling over
       f_tau(z, a) with temperature T ([ASSUMIDO]; T not published);
     - otherwise: safety-first k-NN (K = 10 [ARTIGO]) over past
       *successful* (z, a, y) experiences (y >= Lambda): take the K nearest
       neighbors in z-space and reuse the action of the one with the best
       observed outcome ([ASSUMIDO] tie-break: higher y first, then nearer,
       then older);
     - with no successful experience at all: a fixed fallback action
       ([ASSUMIDO]; callers typically pass C // n_slices so the cold start
       is an equal split before the coordinator rebalances).
5. Safety margin [ARTIGO]:  a <- a + round(sigma(z, a) * M), clipped to
   [0, C].  M starts at margin_bounds[0] (= 0).

Adaptation of (tau, M) ([ASSUMIDO] heuristic — "more violations -> tau down
and M up" is published as behavior, not as equations; PLANO §2.1): over a
sliding window of the last ``violation_window`` observations, if the SLA
violation rate exceeds ``target_violation_rate``, tau decreases by
``tau_step`` (floor tau_bounds[0] = 0.01) and M increases by ``margin_step``
(cap margin_bounds[1]); under hysteresis (rate < target / 2) both recover.
tau stays inside the published range [0.01, 0.05] at all times [ARTIGO].

All stochastic choices (epsilon coin, softmax sample) come from one
``numpy.random.default_rng(seed)``: the same seed reproduces the same
decision sequence given the same observation stream.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from nsoran.sqr.kernel import MaternKernel
from nsoran.sqr.quantile_regressor import (
    KernelQuantileRegressor,
    QuantileUpdate,
    _as_points,
)

#: Published SQR constants [ARTIGO].
MEMORY_SIZE = 2000          # B, support-vector budget
ETA = 0.1                   # pinball step size
P_PENALTY = 2.0             # critical-failure multiplier (both gamma variants)
EPSILON = 0.1               # uncertain-mode exploration probability
KNN_K = 10                  # safety-first k-NN neighbors
TAU_MIN = 0.01              # published tau range ...
TAU_MAX = 0.05              # ... [0.01, 0.05]

#: Assumed defaults [ASSUMIDO] (not published; sweep in sensitivity studies).
DEFAULT_TEMPERATURE = 1.0   # T_softmax
DEFAULT_STATE_DIM = 6       # 4 local + 2 global KPIs (plan §3-Fase 2)
SUCCESS_MEMORY_SIZE = 2000  # ring buffer of successful (z, a, y) experiences

#: Controller decision modes (also log values).
MODE_CONFIDENT = "confident"
MODE_SOFTMAX = "softmax"
MODE_KNN = "knn"
MODE_FALLBACK = "fallback"


@dataclass(frozen=True)
class SqrDecision:
    """One decision of the slice controller (for logs and tests)."""

    action: int          # final action after the safety margin, in [0, C]
    raw_action: int      # action chosen by the policy, before the margin
    mode: str            # confident | softmax | knn | fallback
    sigma: float         # uncertainty at the raw action
    margin_applied: int  # round(sigma * M) actually added
    a_safe_size: int
    n_candidates: int
    tau: float
    margin_m: float


class SqrSliceController:
    """SQR controller for ONE slice (own regressor + own success memory).

    Parameters
    ----------
    state_dim:
        Dimension of the local state z.
    n_prbs:
        C, total PRBs of the cell; candidate actions are {0, ..., C}.
    gamma:
        Resource-cost weight of the confident-mode objective
        f_tau(z,a) - gamma * a/C. Only this term changes with gamma.
    sla_threshold:
        Lambda, the per-decision SLA level used for A_safe, critical-failure
        detection and the success memory (e.g. the per-UE 5th-percentile
        throughput target of the slice).
    seed:
        Seed for the decision RNG (epsilon coin + softmax sampling).
    """

    def __init__(
        self,
        state_dim: int,
        n_prbs: int,
        gamma: float = 0.0,
        sla_threshold: float = 1.0,
        kernel: MaternKernel | None = None,
        memory_size: int = MEMORY_SIZE,
        tau: float = TAU_MAX,
        tau_bounds: tuple[float, float] = (TAU_MIN, TAU_MAX),
        tau_step: float = 0.01,
        eta: float = ETA,
        p_penalty: float = P_PENALTY,
        epsilon: float = EPSILON,
        temperature: float = DEFAULT_TEMPERATURE,
        knn_k: int = KNN_K,
        margin: float = 0.0,
        margin_bounds: tuple[float, float] = (0.0, 10.0),
        margin_step: float = 1.0,
        violation_window: int = 50,
        target_violation_rate: float = 0.01,
        fallback_action: int = 0,
        success_memory_size: int = SUCCESS_MEMORY_SIZE,
        seed: int | None = None,
    ) -> None:
        state_dim = int(state_dim)
        n_prbs = int(n_prbs)
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if n_prbs <= 0:
            raise ValueError(f"n_prbs must be positive, got {n_prbs}")
        if float(gamma) < 0.0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if not (0.0 <= float(epsilon) <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if float(temperature) <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if int(knn_k) <= 0:
            raise ValueError(f"knn_k must be positive, got {knn_k}")
        if not (tau_bounds[0] <= tau <= tau_bounds[1]):
            raise ValueError(
                f"tau={tau} outside tau_bounds {tau_bounds}"
            )
        if not (0 <= int(fallback_action) <= n_prbs):
            raise ValueError(
                f"fallback_action must be in [0, {n_prbs}], got {fallback_action}"
            )

        self.state_dim = state_dim
        self.n_prbs = n_prbs
        self.gamma = float(gamma)
        self.sla_threshold = float(sla_threshold)
        self.epsilon = float(epsilon)
        self.temperature = float(temperature)
        self.knn_k = int(knn_k)
        self.tau_bounds = (float(tau_bounds[0]), float(tau_bounds[1]))
        self.tau_step = float(tau_step)
        self.margin_bounds = (float(margin_bounds[0]), float(margin_bounds[1]))
        self.margin_step = float(margin_step)
        self.violation_window = int(violation_window)
        self.target_violation_rate = float(target_violation_rate)
        self.fallback_action = int(fallback_action)

        if kernel is None:
            kernel = MaternKernel(dim=state_dim + 1, nu=1.5, length_scale=1.0)
        self.kernel = kernel
        self.regressor = KernelQuantileRegressor(
            kernel=kernel,
            memory_size=memory_size,
            tau=tau,
            eta=eta,
            p_penalty=p_penalty,
        )

        # Successful experiences (y >= Lambda): ring buffers, oldest evicted.
        self._suc_z = np.empty((0, state_dim), dtype=np.float64)
        self._suc_a = np.empty((0,), dtype=np.int64)
        self._suc_y = np.empty((0,), dtype=np.float64)
        self._success_memory_size = int(success_memory_size)

        # Sliding window of SLA outcomes for the tau/M heuristic.
        self._violations: deque[bool] = deque(maxlen=self.violation_window)

        self._margin = float(margin_bounds[0]) if margin is None else float(margin)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def tau(self) -> float:
        return self.regressor.tau

    @property
    def margin(self) -> float:
        return self._margin

    @property
    def n_successes(self) -> int:
        return int(self._suc_z.shape[0])

    def success_memory(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Copies of (z, a, y) of the successful experiences."""
        return self._suc_z.copy(), self._suc_a.copy(), self._suc_y.copy()

    def violation_rate(self) -> float:
        """Current SLA violation rate over the sliding window."""
        if not self._violations:
            return 0.0
        return sum(self._violations) / len(self._violations)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def candidate_features(self, z: np.ndarray) -> np.ndarray:
        """Feature matrix [z, a/C] for a = 0..C, shape (C+1, state_dim+1)."""
        zp = _as_points(z, self.state_dim, "z")[0]
        actions = np.arange(self.n_prbs + 1, dtype=np.float64) / float(self.n_prbs)
        return np.column_stack([np.tile(zp, (self.n_prbs + 1, 1)), actions])

    def score_candidates(self, z: np.ndarray) -> dict:
        """f_tau, sigma, A_safe mask and gamma-objective for every a in 0..C.

        The objective is exactly ``f_tau(z, a) - gamma * a / C`` evaluated on
        the SAME f_tau for every gamma — gamma alternates only the cost term
        (AGENTS.md §5 rule 7).
        """
        X = self.candidate_features(z)
        f = self.regressor.predict(X)
        sigma = self.regressor.uncertainty(X)
        a_safe = f >= self.sla_threshold
        actions = np.arange(self.n_prbs + 1, dtype=np.float64)
        scores = f - self.gamma * actions / float(self.n_prbs)
        return {"f": f, "sigma": sigma, "a_safe": a_safe, "scores": scores}

    def act(self, z: np.ndarray) -> SqrDecision:
        """Choose the per-slice action a in {0, ..., C} (hierarchical)."""
        info = self.score_candidates(z)
        f, sigma_all, a_safe = info["f"], info["sigma"], info["a_safe"]
        n_candidates = self.n_prbs + 1

        if bool(np.any(a_safe)):
            # Confident mode: argmax over A_safe of f_tau - gamma * a/C.
            masked = np.where(a_safe, info["scores"], -np.inf)
            raw = int(np.argmax(masked))  # tie -> smallest action
            mode = MODE_CONFIDENT
        else:
            if self._rng.random() < self.epsilon:
                # Softmax exploration over f_tau with temperature T.
                scaled = f / self.temperature
                scaled = scaled - np.max(scaled)
                p = np.exp(scaled)
                p /= p.sum()
                raw = int(self._rng.choice(n_candidates, p=p))
                mode = MODE_SOFTMAX
            else:
                raw, mode = self.k_nn_action(z)

        sigma = float(sigma_all[raw])
        margin_units = int(round(sigma * self._margin))
        action = int(np.clip(raw + margin_units, 0, self.n_prbs))

        return SqrDecision(
            action=action,
            raw_action=raw,
            mode=mode,
            sigma=sigma,
            margin_applied=int(action - raw),
            a_safe_size=int(np.count_nonzero(a_safe)),
            n_candidates=n_candidates,
            tau=self.regressor.tau,
            margin_m=self._margin,
        )

    def k_nn_action(self, z: np.ndarray) -> tuple[int, str]:
        """Safety-first k-NN: K nearest successes, best observed y wins.

        Tie-break ([ASSUMIDO]): higher y, then nearer, then older.
        """
        if self._suc_z.shape[0] == 0:
            return self.fallback_action, MODE_FALLBACK
        zp = _as_points(z, self.state_dim, "z")[0]
        dist = np.sqrt(np.sum((self._suc_z - zp) ** 2, axis=1))
        k = min(self.knn_k, self._suc_z.shape[0])
        nearest = np.argsort(dist, kind="stable")[:k]
        # lexsort: last key is primary -> max y, then min distance, then age.
        order = np.arange(nearest.shape[0])
        best = nearest[int(np.lexsort((order, dist[nearest], -self._suc_y[nearest]))[0])]
        return int(self._suc_a[best]), MODE_KNN

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def observe(
        self,
        z: np.ndarray,
        action: int,
        y: float,
        threshold: float | None = None,
    ) -> QuantileUpdate:
        """Feed one outcome back to the regressor and the tau/M heuristic.

        ``y`` is the KPI the SLA is written on (per-UE 5th-percentile
        throughput in the paper's formulation). ``threshold`` defaults to
        the controller's Lambda. Returns the regressor's update record.
        """
        action = int(action)
        if not (0 <= action <= self.n_prbs):
            raise ValueError(f"action must be in [0, {self.n_prbs}], got {action}")
        lam = self.sla_threshold if threshold is None else float(threshold)
        zp = _as_points(z, self.state_dim, "z")[0]
        record = self.regressor.update(
            np.concatenate([zp, [action / float(self.n_prbs)]]),
            y,
            threshold=lam,
        )

        violated = float(y) < lam
        self._violations.append(bool(violated))
        if not violated:
            self._remember_success(zp, action, float(y))
        self._adapt()
        return record

    def _remember_success(self, zp: np.ndarray, action: int, y: float) -> None:
        if self._suc_z.shape[0] >= self._success_memory_size:
            self._suc_z = self._suc_z[1:]
            self._suc_a = self._suc_a[1:]
            self._suc_y = self._suc_y[1:]
        self._suc_z = np.vstack([self._suc_z, zp[None, :]])
        self._suc_a = np.append(self._suc_a, np.int64(action))
        self._suc_y = np.append(self._suc_y, np.float64(y))

    def _adapt(self) -> None:
        """[ASSUMIDO] heuristic: more violations -> tau down, M up."""
        if not self._violations:
            return
        rate = self.violation_rate()
        tau_lo, tau_hi = self.tau_bounds
        m_lo, m_hi = self.margin_bounds
        if rate > self.target_violation_rate:
            new_tau = max(tau_lo, self.regressor.tau - self.tau_step)
            new_margin = min(m_hi, self._margin + self.margin_step)
        elif rate < self.target_violation_rate / 2.0:
            new_tau = min(tau_hi, self.regressor.tau + self.tau_step)
            new_margin = max(m_lo, self._margin - self.margin_step)
        else:
            return
        self.regressor.set_tau(new_tau)
        self._margin = float(new_margin)
