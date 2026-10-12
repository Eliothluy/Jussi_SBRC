"""Online kernelized pinball quantile regressor (the "QR" core of SQR).

Implements the model of the SQR paper [ARTIGO, method summary in
PLANO_IMPLEMENTACAO_SQR_NORI.md §2.1]:

    f_tau(x)      = sum_{j=1..B} alpha_j * k(x, x_j)
    sigma(x)      = 1 - max_j k(x, x_j)          (1.0 with empty memory)

with a bounded memory of B = 2000 support vectors [ARTIGO].

Online update (pinball loss, step size eta = 0.1 [ARTIGO]).  Each observed
pair (x, y) is inserted as ONE new support vector whose coefficient follows
the published rules [ARTIGO §III, eq. in PLANO §2.1]:

    subprediction  (y > f):            alpha = eta * tau            (> 0)
    overprediction  (y <= f):          alpha = eta * (tau - 1)      (< 0)
    critical failure:                  alpha = eta * (tau - 1) * P_penalty
      (x predicted safe, f >= Lambda, but the SLA was violated, y < Lambda)
                                       (< 0, amplified; P_penalty = 2 [ARTIGO])

A critical failure is auto-detected when a ``threshold`` (Lambda) is given;
it can also be forced explicitly (used by the virtual negative updates
signaled by the coordinator when a proposal is cut).

Pruning rule ([ASSUMIDO] — the paper says "adaptive pruning" without giving
the rule; hypothesis from PLANO_IMPLEMENTACAO_SQR_NORI.md §3-Fase 2):
when the memory is full, drop the support vector of smallest *influence*
I_j = |alpha_j| — since k <= 1, |alpha_j| upper-bounds the vector's total
contribution |alpha_j * k(x, x_j)| to any prediction.  Ties are broken by
the lowest insertion index (oldest first).  The rule is deterministic and
uses no random numbers, so two regressors fed the same stream of updates
always hold identical memories regardless of seed.

Measured property of this rule (tests/test_sqr_quantile.py, coverage tests):
because subprediction coefficients (+eta*tau) are smaller than overprediction
ones (eta*(1-tau)), full-memory operation evicts positive vectors faster and
biases f_tau slightly DOWNWARD — the safe side for a lower quantile, which
is the conservative direction for the SQR safety filter.

tau is adaptable at runtime (``set_tau``); the SQR controller keeps it
inside the published range tau in [0.01, 0.05] [ARTIGO].

Dependencies: NumPy only (no torch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nsoran.sqr.kernel import MaternKernel, _as_points

#: Kinds of online updates (also used as log values).
SUBPREDICTION = "subprediction"      # y > f        -> alpha = +eta * tau
OVERPREDICTION = "overprediction"    # y <= f       -> alpha = eta * (tau - 1)
CRITICAL = "critical"                # safe-forecast + SLA violated -> amplified


@dataclass(frozen=True)
class QuantileUpdate:
    """Record of one online update (for logging and unit tests)."""

    kind: str
    f_before: float          # f_tau(x) computed with the pre-update memory
    y: float
    alpha: float             # coefficient assigned to the new support vector
    pruned_index: int | None  # index pruned to respect the budget, else None
    pruned_alpha: float | None


class KernelQuantileRegressor:
    """Bounded-memory online quantile regressor with a Matérn kernel.

    Parameters
    ----------
    kernel:
        A :class:`~nsoran.sqr.kernel.MaternKernel` instance (k(x,x)=1).
    memory_size:
        Support-vector budget B [ARTIGO: B = 2000].
    tau:
        Target quantile in (0, 1). The published operating range is
        [0.01, 0.05] [ARTIGO]; any value in (0, 1) is accepted so tests and
        sensitivity studies can use e.g. tau=0.5 (median).
    eta:
        Pinball step size [ARTIGO: eta = 0.1].
    p_penalty:
        Critical-failure multiplier [ARTIGO: P_penalty = 2]. Active for
        every gamma variant — gamma only changes the action-selection cost,
        never this penalty (AGENTS.md §5 rule 7).
    """

    def __init__(
        self,
        kernel: MaternKernel,
        memory_size: int = 2000,
        tau: float = 0.05,
        eta: float = 0.1,
        p_penalty: float = 2.0,
    ) -> None:
        if not isinstance(kernel, MaternKernel):
            raise TypeError("kernel must be a MaternKernel instance")
        if int(memory_size) <= 0:
            raise ValueError(f"memory_size must be positive, got {memory_size}")
        if not (0.0 < float(tau) < 1.0):
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        if not (0.0 < float(eta) <= 1.0):
            raise ValueError(f"eta must be in (0, 1], got {eta}")
        if float(p_penalty) <= 0.0:
            raise ValueError(f"p_penalty must be positive, got {p_penalty}")

        self.kernel = kernel
        self.memory_size = int(memory_size)
        self.tau = float(tau)
        self.eta = float(eta)
        self.p_penalty = float(p_penalty)

        self._sx = np.empty((0, kernel.dim), dtype=np.float64)
        self._alpha = np.empty((0,), dtype=np.float64)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return self.kernel.dim

    @property
    def n_support(self) -> int:
        return int(self._sx.shape[0])

    @property
    def support_points(self) -> np.ndarray:
        """Copy of the support vectors, shape (n_support, dim)."""
        return self._sx.copy()

    @property
    def alphas(self) -> np.ndarray:
        """Copy of the coefficients, shape (n_support,)."""
        return self._alpha.copy()

    def set_tau(self, tau: float) -> None:
        """Adapt the target quantile (controller keeps it in [0.01, 0.05])."""
        if not (0.0 < float(tau) < 1.0):
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        self.tau = float(tau)

    def set_memory(self, X: np.ndarray, alphas: np.ndarray) -> None:
        """Replace the memory (checkpoint restore / deterministic tests)."""
        Xp = _as_points(X, self.dim, "X")
        a = np.asarray(alphas, dtype=np.float64).flatten()
        if Xp.shape[0] != a.shape[0]:
            raise ValueError(
                f"X has {Xp.shape[0]} points but {a.shape[0]} alphas"
            )
        if Xp.shape[0] > self.memory_size:
            raise ValueError(
                f"{Xp.shape[0]} points exceed the memory budget {self.memory_size}"
            )
        self._sx = Xp
        self._alpha = a

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """f_tau(X) = K(X, memory) @ alpha; zeros with an empty memory."""
        Xp = _as_points(X, self.dim, "X")
        if self._sx.shape[0] == 0:
            return np.zeros(Xp.shape[0], dtype=np.float64)
        return self.kernel(Xp, self._sx) @ self._alpha

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        """sigma(x) = 1 - max_j k(x, x_j); exactly 1.0 with an empty memory.

        sigma is in [0, 1]: 0 at a stored support vector, -> 1 far from all
        stored data.  The SQR controller uses it as round(sigma * M), the
        published safety margin [ARTIGO].
        """
        Xp = _as_points(X, self.dim, "X")
        if self._sx.shape[0] == 0:
            return np.ones(Xp.shape[0], dtype=np.float64)
        return 1.0 - np.max(self.kernel(Xp, self._sx), axis=1)

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(
        self,
        x: np.ndarray,
        y: float,
        threshold: float | None = None,
        critical: bool = False,
    ) -> QuantileUpdate:
        """Insert one observation as a new support vector (pinball rules).

        Parameters
        ----------
        x:
            Single feature point, shape (dim,).
        y:
            Observed KPI value (e.g. the 5th percentile of per-UE throughput).
        threshold:
            SLA threshold Lambda. When given and the model predicted safe
            (f_tau(x) >= Lambda) while y < Lambda, the critical-failure rule
            applies.
        critical:
            Force the critical-failure rule regardless of auto-detection
            (used for the coordinator's virtual negative updates).
        """
        xp = _as_points(x, self.dim, "x")
        if xp.shape[0] != 1:
            raise ValueError("update accepts exactly one point, got a batch")
        xp = xp[0]
        y = float(y)
        if not np.isfinite(y):
            raise ValueError(f"y must be finite, got {y}")

        f_before = float(self.predict(xp[None, :])[0])

        is_critical = bool(critical) or (
            threshold is not None and f_before >= float(threshold) and y < float(threshold)
        )
        if is_critical:
            kind = CRITICAL
            alpha = self.eta * (self.tau - 1.0) * self.p_penalty
        elif y > f_before:
            kind = SUBPREDICTION
            alpha = self.eta * self.tau
        else:
            kind = OVERPREDICTION
            alpha = self.eta * (self.tau - 1.0)

        pruned_index: int | None = None
        pruned_alpha: float | None = None
        if self._sx.shape[0] >= self.memory_size:
            # Prune first (deterministic rule in the module docstring), then
            # insert — the budget is never exceeded.
            pruned_index = int(np.argmin(np.abs(self._alpha)))  # tie -> oldest
            pruned_alpha = float(self._alpha[pruned_index])
            keep = np.ones(self._sx.shape[0], dtype=bool)
            keep[pruned_index] = False
            self._sx = self._sx[keep]
            self._alpha = self._alpha[keep]

        self._sx = np.vstack([self._sx, xp[None, :]])
        self._alpha = np.append(self._alpha, alpha)

        return QuantileUpdate(
            kind=kind,
            f_before=f_before,
            y=y,
            alpha=float(alpha),
            pruned_index=pruned_index,
            pruned_alpha=pruned_alpha,
        )

    # ------------------------------------------------------------------
    # Checkpointing (NumPy only — plan §3-Fase 2: checkpoint = support
    # matrix + alphas + hyperparameters)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "support_points": self._sx.copy(),
            "alphas": self._alpha.copy(),
            "tau": self.tau,
            "eta": self.eta,
            "p_penalty": self.p_penalty,
            "memory_size": self.memory_size,
            "kernel_dim": self.kernel.dim,
            "kernel_nu": self.kernel.nu,
            "kernel_length_scale": self.kernel.length_scale.copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.set_memory(state["support_points"], state["alphas"])
        self.set_tau(state["tau"])
        self.eta = float(state["eta"])
        self.p_penalty = float(state["p_penalty"])
