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
     - otherwise: safety-first k-NN (K = 10 [ARTIGO], register R21) over
       past experiences (z, a, y, success):
         * successes (y >= Lambda) exist: take the K nearest SUCCESSFUL
           neighbors in z-space and reuse the action with the highest
           f_tau(z, a_j) — the model's quantile, NOT the highest observed y;
         * edge 1 (memorized experiences but no successful one): reuse the
           action of the single nearest experience (success or not);
         * edge 2 (no experience at all): the candidate action with the
           highest f_tau(z, a) (with an empty regressor this is a = 0 —
           deterministic; ties break to the smallest action).
     All three branches are distinguishable in the decision log (mode).
5. Safety margin [ARTIGO]:  a <- a + round(sigma(z, a) * M), clipped to
   [0, C].  M starts at M0 = 10 PRBs (register R11).

Adaptation of (tau, M) (register R11, [ASSUMIDO] — "more violations -> tau
down and M up" is published as behavior [ARTIGO §IV-C], not as equations):
the control signal is v_W = (number of periods with y_t < Lambda in the
last W periods) / W, evaluated once per BLOCK window of W = 100 periods
(10 s simulated; y_t is the same p5 statistic as the constraint of Eq. (1)).
With delta = 1% [ARTIGO §III-C]:

    v_W >  delta : tau <- max(0.01, tau - 0.005);  M <- min(50, 2 * M)
    v_W == 0 for 2 consecutive windows:
                   tau <- min(0.05, tau + 0.005);  M <- max(2, M / 2)
    otherwise    : no change

Frozen constants: W = 100, delta_tau = 0.005, tau0 = 0.05, M0 = 10 PRBs,
M floor/cap = 2/50.  tau stays inside the published range [0.01, 0.05] at
all times [ARTIGO].  The per-UE violation rate (register R19) is a
secondary, log-only metric — it never feeds this controller.

All stochastic choices (epsilon coin, softmax sample) come from one
``numpy.random.default_rng(seed)``: the same seed reproduces the same
decision sequence given the same observation stream, and the RNG state is
part of the checkpoint (``state_dict``/``load_state_dict``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from nsoran.sqr.kernel import MaternKernel
from nsoran.sqr.quantile_regressor import (
    KernelQuantileRegressor,
    QuantileUpdate,
    SupportInsertion,
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

#: Frozen tau/M adaptation constants (register R11, [ASSUMIDO]).
VIOLATION_WINDOW = 100        # W periods = 10 s simulated (100 ms periods)
TAU_STEP = 0.005              # delta_tau
TAU_INIT = TAU_MAX            # tau0 = 0.05
MARGIN_INIT = 10.0            # M0 = 10 PRBs
MARGIN_BOUNDS = (2.0, 50.0)   # M <- max(2, M/2) ... min(50, 2M)
TARGET_VIOLATION_RATE = 0.01  # delta = 1% [ARTIGO §III-C]
RECOVERY_CLEAN_WINDOWS = 2    # recovery only after v_W = 0 twice in a row

#: Assumed defaults [ASSUMIDO] (not published; sweep in sensitivity studies).
DEFAULT_TEMPERATURE = 1.0   # T_softmax
DEFAULT_STATE_DIM = 6       # 4 local + 2 global KPIs (plan §3-Fase 2, R20)
SUCCESS_MEMORY_SIZE = 2000  # ring buffer of (z, a, y, success) experiences

#: Controller decision modes (also log values).
MODE_CONFIDENT = "confident"
MODE_SOFTMAX = "softmax"
MODE_KNN = "knn"                    # K nearest successes -> argmax f_tau
MODE_KNN_NEAREST = "knn_nearest"    # edge 1: nearest memorized experience
MODE_FALLBACK = "fallback"          # edge 2: argmax f_tau (no experience)

#: Version of the state_dict payload produced/consumed by this class
#: (validated on load — a stale checkpoint must fail loudly, not restore
#: into a mismatched layout).
STATE_DICT_VERSION = 2


@dataclass(frozen=True)
class SqrDecision:
    """One decision of the slice controller (for logs and tests)."""

    action: int          # final action after the safety margin, in [0, C]
    raw_action: int      # action chosen by the policy, before the margin
    mode: str            # confident | softmax | knn | knn_nearest | fallback
    sigma: float         # uncertainty at the raw action
    margin_applied: int  # round(sigma * M) actually added
    a_safe_size: int
    n_candidates: int
    tau: float
    margin_m: float


class SqrSliceController:
    """SQR controller for ONE slice (own regressor + own experience memory).

    Parameters
    ----------
    state_dim:
        Dimension of the local state z (4 local + 2 global KPIs -> 6).
    n_prbs:
        C, total PRBs of the cell; candidate actions are {0, ..., C}.
    gamma:
        Resource-cost weight of the confident-mode objective
        f_tau(z,a) - gamma * a/C. Only this term changes with gamma.
    sla_threshold:
        Lambda, the per-decision SLA level used for A_safe, critical-failure
        detection, the success flag of experiences and the tau/M window.
    seed:
        Seed for the decision RNG (epsilon coin + softmax sampling).

    Adaptation parameters (register R11, frozen defaults): violation_window
    W = 100, tau_step = 0.005, margin M0 = 10 with bounds [2, 50],
    target_violation_rate delta = 0.01, recovery after 2 consecutive clean
    windows.
    """

    def __init__(
        self,
        state_dim: int,
        n_prbs: int,
        gamma: float = 0.0,
        sla_threshold: float = 1.0,
        kernel: MaternKernel | None = None,
        memory_size: int = MEMORY_SIZE,
        tau: float = TAU_INIT,
        tau_bounds: tuple[float, float] = (TAU_MIN, TAU_MAX),
        tau_step: float = TAU_STEP,
        eta: float = ETA,
        p_penalty: float = P_PENALTY,
        epsilon: float = EPSILON,
        temperature: float = DEFAULT_TEMPERATURE,
        knn_k: int = KNN_K,
        margin: float = MARGIN_INIT,
        margin_bounds: tuple[float, float] = MARGIN_BOUNDS,
        violation_window: int = VIOLATION_WINDOW,
        target_violation_rate: float = TARGET_VIOLATION_RATE,
        recovery_windows: int = RECOVERY_CLEAN_WINDOWS,
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
        if int(violation_window) <= 0:
            raise ValueError(
                f"violation_window must be positive, got {violation_window}"
            )
        if int(recovery_windows) <= 0:
            raise ValueError(
                f"recovery_windows must be positive, got {recovery_windows}"
            )
        if not (tau_bounds[0] <= tau <= tau_bounds[1]):
            raise ValueError(
                f"tau={tau} outside tau_bounds {tau_bounds}"
            )
        if not (margin_bounds[0] <= margin <= margin_bounds[1]):
            raise ValueError(
                f"margin={margin} outside margin_bounds {margin_bounds}"
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
        self.violation_window = int(violation_window)
        self.target_violation_rate = float(target_violation_rate)
        self.recovery_windows = int(recovery_windows)

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

        # Experience memory (register R21): ring buffer of (z, a, y, success),
        # oldest evicted.  Non-successes are memorized too — edge 1 of the
        # uncertain mode ("no successful neighbor -> nearest experience")
        # needs them.
        self._exp_z = np.empty((0, state_dim), dtype=np.float64)
        self._exp_a = np.empty((0,), dtype=np.int64)
        self._exp_y = np.empty((0,), dtype=np.float64)
        self._exp_ok = np.empty((0,), dtype=bool)
        self._success_memory_size = int(success_memory_size)

        # Sliding window of y_t < Lambda indicators for the tau/M heuristic
        # (register R11); evaluated once per block window of W periods.
        self._violations: deque[bool] = deque(maxlen=self.violation_window)
        self._window_count = 0        # periods accumulated in the open block
        self._clean_window_streak = 0  # consecutive windows with v_W = 0

        self._margin = float(margin)
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
        return int(np.count_nonzero(self._exp_ok))

    @property
    def n_experiences(self) -> int:
        return int(self._exp_z.shape[0])

    def experience_memory(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Copies of (z, a, y, success) of the memorized experiences."""
        return (
            self._exp_z.copy(),
            self._exp_a.copy(),
            self._exp_y.copy(),
            self._exp_ok.copy(),
        )

    def violation_rate(self) -> float:
        """Fraction of y_t < Lambda indicators in the sliding window.

        Log-only secondary signal (register R11 keeps the CONTROL on the
        block-window evaluation in :meth:`_adapt`).
        """
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
        """Safety-first k-NN of the uncertain mode (register R21).

        Branches, in order:
        1. Successful experiences exist -> K nearest successes, action with
           the highest f_tau(z, a_j) (ties: nearer, then older).
        2. Edge 1 — experiences exist but none succeeded -> action of the
           single nearest experience (ties: older).
        3. Edge 2 — no experience at all -> candidate action of highest
           f_tau(z, a) (ties: smallest action; a = 0 with empty memory).
        """
        zp = _as_points(z, self.state_dim, "z")[0]

        if self._exp_z.shape[0] == 0:
            # Edge 2: no experience at all -> argmax f_tau over candidates.
            f = self.regressor.predict(self.candidate_features(zp))
            return int(np.argmax(f)), MODE_FALLBACK

        dist = np.sqrt(np.sum((self._exp_z - zp) ** 2, axis=1))
        order = np.argsort(dist, kind="stable")  # tie -> older first

        success_idx = np.flatnonzero(self._exp_ok)
        if success_idx.shape[0] > 0:
            # K nearest SUCCESSES, then argmax f_tau(z, a_j).
            success_order = order[np.isin(order, success_idx)]
            nearest = success_order[: min(self.knn_k, success_order.shape[0])]
            X = np.column_stack([
                np.tile(zp, (nearest.shape[0], 1)),
                self._exp_a[nearest] / float(self.n_prbs),
            ])
            f_neighbors = self.regressor.predict(X)
            best = int(nearest[int(np.argmax(f_neighbors))])
            return int(self._exp_a[best]), MODE_KNN

        # Edge 1: no successful neighbor -> nearest memorized experience.
        nearest = int(order[0])
        return int(self._exp_a[nearest]), MODE_KNN_NEAREST

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
        the controller's Lambda. The REAL update uses the FINAL (executed)
        action (register R22, [ARTIGO] Algorithm 1). Returns the regressor's
        update record.
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
        self._remember_experience(zp, action, float(y), success=not violated)
        self._adapt()
        return record

    def insert_virtual_support(
        self, z: np.ndarray, action: int, alpha: float
    ) -> SupportInsertion:
        """Register R12: negative support vector at the PRE-projection action.

        The coordinator cuts a proposal; this inserts one support vector at
        (z, a_proposed) with alpha = -alpha (a small negative weight, by
        default eta/2 = 0.05) so the model learns that the over-aggressive
        proposal was infeasible.  The sign is applied here; callers pass the
        positive magnitude (ablation knob {0.025, 0.05, 0.1}).
        """
        action = int(action)
        if not (0 <= action <= self.n_prbs):
            raise ValueError(f"action must be in [0, {self.n_prbs}], got {action}")
        zp = _as_points(z, self.state_dim, "z")[0]
        return self.regressor.insert_support_vector(
            np.concatenate([zp, [action / float(self.n_prbs)]]),
            -float(alpha),
        )

    def _remember_experience(
        self, zp: np.ndarray, action: int, y: float, success: bool
    ) -> None:
        if self._exp_z.shape[0] >= self._success_memory_size:
            self._exp_z = self._exp_z[1:]
            self._exp_a = self._exp_a[1:]
            self._exp_y = self._exp_y[1:]
            self._exp_ok = self._exp_ok[1:]
        self._exp_z = np.vstack([self._exp_z, zp[None, :]])
        self._exp_a = np.append(self._exp_a, np.int64(action))
        self._exp_y = np.append(self._exp_y, np.float64(y))
        self._exp_ok = np.append(self._exp_ok, np.bool_(success))

    def _adapt(self) -> None:
        """tau/M adaptation on block windows of W periods (register R11).

        Evaluated once every ``violation_window`` observations, so "2
        consecutive windows" is exact: recovery fires only after two full
        windows with zero y_t < Lambda indicators.
        """
        self._window_count += 1
        if self._window_count < self.violation_window:
            return
        self._window_count = 0

        # The deque holds exactly the last W indicators == the closed block.
        n_violations = int(sum(self._violations))
        v_w = n_violations / self.violation_window
        tau_lo, tau_hi = self.tau_bounds
        m_lo, m_hi = self.margin_bounds

        if v_w > self.target_violation_rate:
            self._clean_window_streak = 0
            self.regressor.set_tau(max(tau_lo, self.regressor.tau - self.tau_step))
            self._margin = float(min(m_hi, 2.0 * self._margin))
        elif n_violations == 0:
            self._clean_window_streak += 1
            if self._clean_window_streak >= self.recovery_windows:
                self._clean_window_streak = 0
                self.regressor.set_tau(min(tau_hi, self.regressor.tau + self.tau_step))
                self._margin = float(max(m_lo, self._margin / 2.0))
        else:
            # 0 < v_W <= delta: hold, and a dirty window breaks the streak.
            self._clean_window_streak = 0

    # ------------------------------------------------------------------
    # Checkpoint (plan §3-Fase 2 item: support matrix, alphas,
    # hyperparameters, tau, M, violation window, experience memory, RNG).
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Full restorable state of the slice controller."""
        return {
            "format_version": STATE_DICT_VERSION,
            "state_dim": self.state_dim,
            "n_prbs": self.n_prbs,
            "gamma": self.gamma,
            "sla_threshold": self.sla_threshold,
            "epsilon": self.epsilon,
            "temperature": self.temperature,
            "knn_k": self.knn_k,
            "tau_bounds": self.tau_bounds,
            "tau_step": self.tau_step,
            "margin_bounds": self.margin_bounds,
            "violation_window": self.violation_window,
            "target_violation_rate": self.target_violation_rate,
            "recovery_windows": self.recovery_windows,
            "margin_m": self._margin,
            "violations": list(self._violations),
            "window_count": self._window_count,
            "clean_window_streak": self._clean_window_streak,
            "experience_z": self._exp_z.copy(),
            "experience_a": self._exp_a.copy(),
            "experience_y": self._exp_y.copy(),
            "experience_ok": self._exp_ok.copy(),
            "success_memory_size": self._success_memory_size,
            "rng_state": self._rng.bit_generator.state,
            "regressor": self.regressor.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a checkpoint saved by :meth:`state_dict`.

        The payload's ``format_version`` must match
        :data:`STATE_DICT_VERSION` (a stale/foreign layout raises instead
        of restoring garbage), and the structural fields (state_dim,
        n_prbs) must match this controller — every mismatch raises an
        explicit ValueError.
        """
        try:
            version = int(state.get("format_version", -1))
        except (TypeError, ValueError):
            version = -1
        if version != STATE_DICT_VERSION:
            raise ValueError(
                f"unsupported controller state format_version "
                f"{state.get('format_version')!r} (expected "
                f"{STATE_DICT_VERSION}); regenerate the checkpoint"
            )
        if int(state["state_dim"]) != self.state_dim:
            raise ValueError(
                f"state_dim mismatch: checkpoint has {state['state_dim']}, "
                f"controller has {self.state_dim}"
            )
        if int(state["n_prbs"]) != self.n_prbs:
            raise ValueError(
                f"n_prbs mismatch: checkpoint has {state['n_prbs']}, "
                f"controller has {self.n_prbs}"
            )
        self.regressor.load_state_dict(state["regressor"])
        self.gamma = float(state["gamma"])
        self.sla_threshold = float(state["sla_threshold"])
        self.epsilon = float(state["epsilon"])
        self.temperature = float(state["temperature"])
        self.knn_k = int(state["knn_k"])
        self.tau_bounds = (float(state["tau_bounds"][0]), float(state["tau_bounds"][1]))
        self.tau_step = float(state["tau_step"])
        self.margin_bounds = (
            float(state["margin_bounds"][0]),
            float(state["margin_bounds"][1]),
        )
        self.violation_window = int(state["violation_window"])
        self.target_violation_rate = float(state["target_violation_rate"])
        self.recovery_windows = int(state["recovery_windows"])
        self._margin = float(state["margin_m"])
        self._violations = deque(
            (bool(v) for v in state["violations"]), maxlen=self.violation_window
        )
        self._window_count = int(state["window_count"])
        self._clean_window_streak = int(state["clean_window_streak"])
        self._exp_z = np.asarray(state["experience_z"], dtype=np.float64).reshape(
            -1, self.state_dim
        )
        self._exp_a = np.asarray(state["experience_a"], dtype=np.int64).flatten()
        self._exp_y = np.asarray(state["experience_y"], dtype=np.float64).flatten()
        self._exp_ok = np.asarray(state["experience_ok"], dtype=bool).flatten()
        if not (
            self._exp_z.shape[0] == self._exp_a.shape[0]
            == self._exp_y.shape[0] == self._exp_ok.shape[0]
        ):
            raise ValueError("experience memory arrays have inconsistent lengths")
        self._success_memory_size = int(state["success_memory_size"])
        rng_state = state["rng_state"]
        self._rng.bit_generator.state = rng_state
