"""SQR v2 controllers with single-pass kernel evaluation (BUG-03 speedup)."""

from __future__ import annotations

import numpy as np

from nsoran.sqr.kernel import MaternKernel, _as_points
from nsoran.sqr.quantile_regressor import KernelQuantileRegressor
from nsoran.sqr.sqr_controller import (
    MARGIN_BOUNDS,
    MARGIN_INIT,
    TAU_INIT,
    TAU_STEP,
    VIOLATION_WINDOW,
    SqrSliceController,
)


class KernelQuantileRegressorV2(KernelQuantileRegressor):
    """KernelQuantileRegressor with single-pass ``predict_and_uncertainty``."""

    def predict_and_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute ``f_tau(X)`` and ``sigma(X)`` from a single kernel pass.

        Numerically equivalent to ``(self.predict(X), self.uncertainty(X))``
        while cutting the Matérn distance matrix computation in half inside
        ``score_candidates``.
        """
        Xp = _as_points(X, self.dim, "X")
        n = Xp.shape[0]
        if self._sx.shape[0] == 0:
            return np.zeros(n, dtype=np.float64), np.ones(n, dtype=np.float64)
        K = self.kernel(Xp, self._sx)
        f = K @ self._alpha
        sigma = 1.0 - np.max(K, axis=1)
        return f, sigma


class SqrSliceControllerV2(SqrSliceController):
    """Drop-in v2 SqrSliceController computing K once per decision window."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Upgrade regressor in-place to V2 preserving exact state and hyperparams
        reg = KernelQuantileRegressorV2(
            kernel=self.kernel,
            memory_size=self.regressor.memory_size,
            tau=self.regressor.tau,
            eta=self.regressor.eta,
            p_penalty=self.regressor.p_penalty,
        )
        self.regressor = reg

    def score_candidates(self, z: np.ndarray) -> dict:
        """Evaluate f_tau, sigma, A_safe mask and gamma-objective in one kernel pass."""
        X = self.candidate_features(z)
        if hasattr(self.regressor, "predict_and_uncertainty"):
            f, sigma = self.regressor.predict_and_uncertainty(X)
        else:
            f = self.regressor.predict(X)
            sigma = self.regressor.uncertainty(X)
        a_safe = f >= self.sla_threshold
        actions = np.arange(self.n_prbs + 1, dtype=np.float64)
        scores = f - self.gamma * actions / float(self.n_prbs)
        return {"f": f, "sigma": sigma, "a_safe": a_safe, "scores": scores}


def parse_slas_v2(slas_arg: str, num_slices: int) -> dict[int, float]:
    values = [float(v) for v in slas_arg.split(",")]
    if len(values) != num_slices:
        raise ValueError(f"--slas must have {num_slices} values, got {len(values)}")
    if any(v <= 0 for v in values):
        raise ValueError("--slas values must be positive (Mbps per UE)")
    return dict(enumerate(values))


def parse_margin_bounds_v2(bounds_arg) -> tuple[float, float]:
    if isinstance(bounds_arg, (tuple, list)):
        if len(bounds_arg) != 2:
            raise ValueError(f"--margin_bounds must have 2 values, got {bounds_arg!r}")
        lo, hi = float(bounds_arg[0]), float(bounds_arg[1])
    else:
        parts = [p.strip() for p in str(bounds_arg).split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"--margin_bounds must be 'lo,hi', got {bounds_arg!r}")
        lo, hi = float(parts[0]), float(parts[1])
    if lo < 0.0 or hi < lo:
        raise ValueError(f"--margin_bounds requires 0 <= lo <= hi, got ({lo}, {hi})")
    return (lo, hi)


def build_controllers_v2(args, env=None):
    """Build v2 per-slice controllers (supports ``sqr``, ``kbrl``, ``equal``, ``prop``)."""
    slas = parse_slas_v2(args.slas, args.num_slices)
    controller = getattr(args, "controller", "sqr")
    if controller == "kbrl":
        from nsoran.sqr.kbrl import KbrlSliceController

        return {
            sid: KbrlSliceController(
                state_dim=6,
                n_prbs=args.c_prbs,
                sla_threshold=slas[sid],
                seed=None if args.seed is None else int(args.seed) + 1000 * sid,
            )
            for sid in range(args.num_slices)
        }
    if controller == "equal":
        from nsoran.sqr.static_controllers import EqualSplitController

        return {
            sid: EqualSplitController(
                slice_id=sid,
                n_prbs=args.c_prbs,
                num_slices=args.num_slices,
            )
            for sid in range(args.num_slices)
        }
    if controller == "prop":
        from nsoran.sqr.static_controllers import PropDemandController

        builder = getattr(env, "builder", None)
        window_provider = (
            (lambda b: lambda: b.last_window)(builder)
            if builder is not None
            else None
        )
        return {
            sid: PropDemandController(
                slice_id=sid,
                n_prbs=args.c_prbs,
                num_slices=args.num_slices,
                window_provider=window_provider,
            )
            for sid in range(args.num_slices)
        }
    if controller != "sqr":
        raise ValueError(
            f"unknown v2 controller {controller!r} (supported in Etapa 2: sqr, kbrl, equal, prop)"
        )

    raw_margin_init = getattr(args, "margin_init", None)
    margin_init = MARGIN_INIT if raw_margin_init is None else float(raw_margin_init)
    raw_margin_bounds = getattr(args, "margin_bounds", None)
    margin_bounds = (
        MARGIN_BOUNDS
        if raw_margin_bounds is None
        else parse_margin_bounds_v2(raw_margin_bounds)
    )
    return {
        sid: SqrSliceControllerV2(
            state_dim=6,
            n_prbs=args.c_prbs,
            gamma=float(args.gamma),
            sla_threshold=slas[sid],
            kernel=MaternKernel(dim=7, nu=1.5, length_scale=1.0),
            tau=TAU_INIT,
            tau_step=TAU_STEP,
            margin=margin_init,
            margin_bounds=margin_bounds,
            violation_window=args.violation_window,
            epsilon=args.epsilon,
            temperature=args.temperature,
            knn_k=args.knn_k,
            seed=None if args.seed is None else int(args.seed) + 1000 * sid,
        )
        for sid in range(args.num_slices)
    }
