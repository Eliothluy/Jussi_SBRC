"""Kernelized multi-quantile heads on SHARED support locations (§9/§10).

Preserves the audited SQR learner backbone (protocol §3/§6 — the learner is
NOT replaced): same normalized Matérn kernel family (``nsoran.sqr.kernel``,
nu=1.5, anisotropic per-dimension length-scales, k(x,x)=1), same online
pinball update with eta=0.1 and the critical-failure rule (P_penalty=2) for
the safe head, same bounded memory B=2000 with the influence pruning rule
R10 (I_j = |alpha_j| * k(x_new, x_j), ties to the oldest index), same
uncertainty sigma(x) = 1 − max_j k(x, x_j).

What is new is ONLY the multi-head layout (§9.2/§10):

    per slice: one support-location memory + THREE coefficient vectors

        alpha_safe  -> dynamic safety quantile tau_safe(t) in [0.01, 0.05]
                       (the SQR mechanism, §9.1 — kept verbatim)
        alpha_q25   -> auxiliary feasibility quantile tau = 0.25
        alpha_q75   -> auxiliary feasibility quantile tau = 0.75

Sharing ONE set of support locations is the recommended implementation of
§10 ("mesmos support-vector locations, coeficientes separados por quantil").
Because all heads update on the SAME observations and the pruning decision
must keep the locations shared, the pruned index is decided ONCE per update
by the SAFE head's influence (the safe model is the primary safety learner,
§9.1) and applied to all three coefficient vectors simultaneously.  This is
a documented design decision (registered in the preflight report): the
pruning rule remains deterministic and seed-free.

Quantile crossing (§10.1): raw q25/q75 predictions are kept; classification
uses the monotonic rearrangement (q25_used = min(raw), q75_used = max(raw));
every crossing is counted (``crossing_events``, ``crossing_rate``) and
logged per decision — crossing is never hidden by the rearrangement.

tau/M adaptation state (audit register R11) is carried by the safe head's
adaptation controller exactly as in the audited SQR controller: block
window W=100 periods, delta=1%, tau_step=0.005, M0=10 with bounds [2, 50],
recovery after 2 clean windows.  IMPORTANT (§45): in SQR-JC the M margin is
NEVER added post-hoc to the joint action — the M state is maintained and
LOGGED (§57 "M_i(t), se ainda mantido") but does not touch actions; the
safety mechanism is support filtering + floors + MARGINAL + uncertainty-aware
tie-breaking.  This is the intentional policy change of SQR-JC.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from nsoran.sqr.kernel import MaternKernel, _as_points

#: Published SQR constants [ARTIGO] — identical to the audited controller.
MEMORY_SIZE = 2000          # B, support-vector budget
ETA = 0.1                   # pinball step size
P_PENALTY = 2.0             # critical-failure multiplier (safe head only)
TAU_MIN = 0.01
TAU_MAX = 0.05
TAU_INIT = TAU_MAX          # tau0 = 0.05
TAU_STEP = 0.005
TARGET_VIOLATION_RATE = 0.01  # delta = 1%
VIOLATION_WINDOW = 100      # W periods = 10 s simulated
RECOVERY_CLEAN_WINDOWS = 2
MARGIN_INIT = 10.0
MARGIN_BOUNDS = (2.0, 50.0)

#: Auxiliary feasibility quantiles (§9.2).
TAU_Q25 = 0.25
TAU_Q75 = 0.75

#: Frozen kernel hyperparameters of the audited SQR (audit: nu=1.5 [ASSUMIDO],
#: length_scale=1.0 scalar — sqr_controller.py:213-214 default).
KERNEL_NU = 1.5
KERNEL_LENGTH_SCALE = 1.0


@dataclass(frozen=True)
class MultiQuantileUpdate:
    """Record of one online update across the three heads."""

    kind_safe: str            # subprediction | overprediction | critical
    alpha_safe: float
    alpha_q25: float
    alpha_q75: float
    pruned_index: int | None
    crossing: bool            # raw q25 > raw q75 AFTER this update


class MultiQuantileKernelModel:
    """One slice's kernelized heads over a shared bounded support memory.

    Parameters mirror the audited ``KernelQuantileRegressor``; ``tau`` is the
    safe quantile (kept inside [0.01, 0.05] by the adaptation state).
    """

    def __init__(
        self,
        dim: int,
        length_scale: float | np.ndarray = KERNEL_LENGTH_SCALE,
        nu: float = KERNEL_NU,
        memory_size: int = MEMORY_SIZE,
        tau: float = TAU_INIT,
        eta: float = ETA,
        p_penalty: float = P_PENALTY,
        margin_m: float = MARGIN_INIT,
    ) -> None:
        dim = int(dim)
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.kernel = MaternKernel(dim=dim, length_scale=length_scale, nu=nu)
        self.memory_size = int(memory_size)
        self.eta = float(eta)
        self.p_penalty = float(p_penalty)
        self.dim = dim

        self.tau = float(tau)
        if not (0.0 < self.tau < 1.0):
            raise ValueError(f"tau must be in (0, 1), got {tau}")

        # Shared support locations; per-quantile coefficient vectors (§10).
        self._sx = np.empty((0, dim), dtype=np.float64)
        self._alpha_safe = np.empty((0,), dtype=np.float64)
        self._alpha_q25 = np.empty((0,), dtype=np.float64)
        self._alpha_q75 = np.empty((0,), dtype=np.float64)

        # tau/M adaptation state (audit register R11, carried verbatim).
        self.margin_m = float(margin_m)
        if not (
            MARGIN_BOUNDS[0] <= self.margin_m <= MARGIN_BOUNDS[1]
        ):
            raise ValueError(
                f"margin_m={margin_m} outside bounds {MARGIN_BOUNDS}"
            )
        self._violations: deque[bool] = deque(maxlen=VIOLATION_WINDOW)
        self._window_count = 0
        self._clean_window_streak = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def n_support(self) -> int:
        return int(self._sx.shape[0])

    @property
    def support_points(self) -> np.ndarray:
        return self._sx.copy()

    @property
    def alphas_safe(self) -> np.ndarray:
        return self._alpha_safe.copy()

    @property
    def alphas_q25(self) -> np.ndarray:
        return self._alpha_q25.copy()

    @property
    def alphas_q75(self) -> np.ndarray:
        return self._alpha_q75.copy()

    @property
    def margin_bounds(self) -> tuple[float, float]:
        return MARGIN_BOUNDS

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _head_predict(self, X: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        if self._sx.shape[0] == 0:
            return np.zeros(X.shape[0], dtype=np.float64)
        return self.kernel(X, self._sx) @ alpha

    def predict_raw(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Raw head predictions at X, shape (n,): safe/q25/q75 + sigma.

        ``safe`` uses the dynamic tau head; ``q25``/``q75`` are the RAW
        auxiliary predictions (§10.1 keeps raw values); ``sigma`` is the
        audited uncertainty 1 − max_j k(x, x_j).  Computes the kernel
        matrix ONCE and reuses it across the three heads + sigma.
        """
        Xp = _as_points(X, self.dim, "X")
        if self._sx.shape[0] == 0:
            n = Xp.shape[0]
            zeros = np.zeros(n, dtype=np.float64)
            return {
                "safe": zeros.copy(),
                "q25": zeros.copy(),
                "q75": zeros.copy(),
                "sigma": np.ones(n, dtype=np.float64),
            }
        K = self.kernel(Xp, self._sx)          # ONE kernel evaluation
        return {
            "safe": K @ self._alpha_safe,
            "q25": K @ self._alpha_q25,
            "q75": K @ self._alpha_q75,
            "sigma": 1.0 - np.max(K, axis=1),
        }

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        """sigma(x) = 1 − max_j k(x, x_j); exactly 1.0 with empty memory."""
        Xp = _as_points(X, self.dim, "X")
        if self._sx.shape[0] == 0:
            return np.ones(Xp.shape[0], dtype=np.float64)
        return 1.0 - np.max(self.kernel(Xp, self._sx), axis=1)

    @staticmethod
    def rearranged(q25_raw: np.ndarray, q75_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Monotonic rearrangement for CLASSIFICATION only (§10.1 rule 2)."""
        return np.minimum(q25_raw, q75_raw), np.maximum(q25_raw, q75_raw)

    # ------------------------------------------------------------------
    # Online update (pinball; critical rule on the safe head only)
    # ------------------------------------------------------------------

    def update(
        self,
        x: np.ndarray,
        y: float,
        threshold: float = 0.0,
    ) -> MultiQuantileUpdate:
        """One observation (x, y) into all three heads.

        ``y`` is the observed normalized SLA margin m_i of the governed
        window (§54); ``threshold`` is the SLA boundary in margin scale —
        0.0, because m_i > 0 means satisfied.  The critical-failure rule
        (audited semantics: predicted safe but SLA violated) applies to the
        SAFE head only; q25/q75 get the plain pinball coefficients.

        When the memory is full, the R10 influence rule of the SAFE head
        picks the pruned index ONCE and the same location is dropped from
        all three coefficient vectors (shared locations, §10).
        """
        xp = _as_points(x, self.dim, "x")
        if xp.shape[0] != 1:
            raise ValueError("update accepts exactly one point")
        x1 = xp[0]
        y = float(y)
        if not np.isfinite(y):
            raise ValueError(f"y must be finite, got {y}")

        if self._sx.shape[0] > 0:
            k_new = self.kernel(x1[None, :], self._sx)[0]
        else:
            k_new = np.empty(0, dtype=np.float64)
        f_safe_before = float(k_new @ self._alpha_safe) if k_new.size else 0.0
        f_q25_before = float(k_new @ self._alpha_q25) if k_new.size else 0.0
        f_q75_before = float(k_new @ self._alpha_q75) if k_new.size else 0.0

        is_critical = f_safe_before >= float(threshold) and y < float(threshold)
        if is_critical:
            kind_safe = "critical"
            alpha_safe = self.eta * (self.tau - 1.0) * self.p_penalty
        elif y > f_safe_before:
            kind_safe = "subprediction"
            alpha_safe = self.eta * self.tau
        else:
            kind_safe = "overprediction"
            alpha_safe = self.eta * (self.tau - 1.0)

        # Pinball rule per head (protocol §9.2/§54): each auxiliary quantile
        # compares y against ITS OWN prediction, not the safe head's. The
        # previous version compared against f_safe_before (tau~0.01-0.05
        # underpredicts by design) -> chronic positive updates -> q25/q75
        # degenerated to near-maxima (G4 STOP of 03/11; fix §82: bug registrado,
        # CAL reprocessado offline — sem novas simulações, dados CAL idênticos).
        alpha_q25 = self.eta * TAU_Q25 if y > f_q25_before else self.eta * (TAU_Q25 - 1.0)
        alpha_q75 = self.eta * TAU_Q75 if y > f_q75_before else self.eta * (TAU_Q75 - 1.0)

        pruned_index: int | None = None
        if self._sx.shape[0] >= self.memory_size:
            # R10 influence of the SAFE head (primary learner, §9.1); ties
            # break to the lowest insertion index (argmin first-occurrence).
            pruned_index = int(np.argmin(np.abs(self._alpha_safe) * k_new))
            keep = np.ones(self._sx.shape[0], dtype=bool)
            keep[pruned_index] = False
            self._sx = self._sx[keep]
            self._alpha_safe = self._alpha_safe[keep]
            self._alpha_q25 = self._alpha_q25[keep]
            self._alpha_q75 = self._alpha_q75[keep]

        self._sx = np.vstack([self._sx, x1[None, :]])
        self._alpha_safe = np.append(self._alpha_safe, alpha_safe)
        self._alpha_q25 = np.append(self._alpha_q25, alpha_q25)
        self._alpha_q75 = np.append(self._alpha_q75, alpha_q75)

        # Crossing state AFTER the update (measured on the updated heads).
        preds = self.predict_raw(x1[None, :])
        crossing = bool(preds["q25"][0] > preds["q75"][0])

        self._remember_violation(y < float(threshold))
        return MultiQuantileUpdate(
            kind_safe=kind_safe,
            alpha_safe=float(alpha_safe),
            alpha_q25=float(alpha_q25),
            alpha_q75=float(alpha_q75),
            pruned_index=pruned_index,
            crossing=crossing,
        )

    # ------------------------------------------------------------------
    # tau/M adaptation (audit register R11, verbatim block-window rule)
    # ------------------------------------------------------------------

    def _remember_violation(self, violated: bool) -> None:
        """Feed the block-window violation indicator and adapt (tau, M).

        In SQR-JC the violation indicator is ``m_i < 0`` (margin scale);
        the rule is otherwise identical to the audited controller.  M is
        maintained for log continuity but NEVER applied post-hoc (§45).
        """
        self._violations.append(bool(violated))
        self._window_count += 1
        if self._window_count < VIOLATION_WINDOW:
            return
        self._window_count = 0
        n_violations = int(sum(self._violations))
        v_w = n_violations / VIOLATION_WINDOW
        if v_w > TARGET_VIOLATION_RATE:
            self._clean_window_streak = 0
            self.tau = max(TAU_MIN, self.tau - TAU_STEP)
            self.margin_m = float(min(MARGIN_BOUNDS[1], 2.0 * self.margin_m))
        elif n_violations == 0:
            self._clean_window_streak += 1
            if self._clean_window_streak >= RECOVERY_CLEAN_WINDOWS:
                self._clean_window_streak = 0
                self.tau = min(TAU_MAX, self.tau + TAU_STEP)
                self.margin_m = float(max(MARGIN_BOUNDS[0], self.margin_m / 2.0))
        else:
            self._clean_window_streak = 0

    @property
    def violation_rate(self) -> float:
        if not self._violations:
            return 0.0
        return sum(self._violations) / len(self._violations)

    # ------------------------------------------------------------------
    # Checkpoint (deterministic replay / state identity tests)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "dim": self.dim,
            "kernel_nu": self.kernel.nu,
            "kernel_length_scale": self.kernel.length_scale.copy(),
            "memory_size": self.memory_size,
            "tau": self.tau,
            "eta": self.eta,
            "p_penalty": self.p_penalty,
            "margin_m": self.margin_m,
            "support_points": self._sx.copy(),
            "alphas_safe": self._alpha_safe.copy(),
            "alphas_q25": self._alpha_q25.copy(),
            "alphas_q75": self._alpha_q75.copy(),
            "violations": list(self._violations),
            "window_count": self._window_count,
            "clean_window_streak": self._clean_window_streak,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["dim"]) != self.dim:
            raise ValueError(
                f"dim mismatch: checkpoint {state['dim']} vs {self.dim}"
            )
        if float(state["kernel_nu"]) != self.kernel.nu:
            raise ValueError("kernel_nu mismatch")
        saved_ls = np.asarray(state["kernel_length_scale"], dtype=np.float64).flatten()
        if not np.array_equal(saved_ls, self.kernel.length_scale):
            raise ValueError("kernel_length_scale mismatch")
        self.set_memory(
            state["support_points"],
            state["alphas_safe"],
            state["alphas_q25"],
            state["alphas_q75"],
        )
        self.tau = float(state["tau"])
        self.margin_m = float(state["margin_m"])
        self._violations = deque(
            (bool(v) for v in state["violations"]), maxlen=VIOLATION_WINDOW
        )
        self._window_count = int(state["window_count"])
        self._clean_window_streak = int(state["clean_window_streak"])

    def set_memory(
        self,
        X: np.ndarray,
        alphas_safe: np.ndarray,
        alphas_q25: np.ndarray,
        alphas_q75: np.ndarray,
    ) -> None:
        """Replace the shared memory (checkpoint restore / tests)."""
        Xp = _as_points(X, self.dim, "X")
        a_s = np.asarray(alphas_safe, dtype=np.float64).flatten()
        a_lo = np.asarray(alphas_q25, dtype=np.float64).flatten()
        a_hi = np.asarray(alphas_q75, dtype=np.float64).flatten()
        if not (Xp.shape[0] == a_s.shape[0] == a_lo.shape[0] == a_hi.shape[0]):
            raise ValueError("memory arrays have inconsistent lengths")
        if Xp.shape[0] > self.memory_size:
            raise ValueError(
                f"{Xp.shape[0]} points exceed the memory budget {self.memory_size}"
            )
        self._sx = Xp
        self._alpha_safe = a_s
        self._alpha_q25 = a_lo
        self._alpha_q75 = a_hi
