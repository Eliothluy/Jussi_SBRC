"""SQR v2 controllers with single-pass kernel evaluation (BUG-03 speedup)."""

from __future__ import annotations

import math

import numpy as np

_SQRT3 = math.sqrt(3.0)
_SQRT5 = math.sqrt(5.0)

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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._np_meta: tuple[int, int, int] | None = None
        self._np_payload: tuple | None = None

    def _sync_candidate_cache(self, acts: np.ndarray):
        meta = (id(self._sx), id(self._alpha), int(acts.shape[0]))
        if self._np_meta == meta and self._np_payload is not None:
            return self._np_payload

        sd = self.dim - 1
        inv_l = np.asarray(1.0 / self.kernel.length_scale, dtype=np.float64)
        inv_l_z = np.ascontiguousarray(inv_l[:sd])
        sx_z_s = np.ascontiguousarray(self._sx[:, :sd] * inv_l_z)
        da2_pre = np.ascontiguousarray(
            ((acts[:, None] - self._sx[None, :, sd]) * inv_l[sd]) ** 2
        )
        m = int(self._sx.shape[0])
        buf = np.empty((acts.shape[0], m), dtype=np.float64)
        buf2 = np.empty_like(buf)
        dz_buf = np.empty((m, sd), dtype=np.float64)
        self._np_payload = (inv_l_z, sx_z_s, da2_pre, buf, buf2, dz_buf)
        self._np_meta = meta
        return self._np_payload

    def warm_candidate_cache(self, n_prbs: int) -> None:
        """Pre-sync scaled support matrix & action deltas after memory updates."""
        if self._sx.shape[0] > 0:
            acts = np.arange(int(n_prbs) + 1, dtype=np.float64) / float(n_prbs)
            self._sync_candidate_cache(acts)

    def predict_and_uncertainty_candidates(
        self, z: np.ndarray, actions_norm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact fast path for ``X = [z, actions_norm]`` using pre-allocated scratch buffers.

        Exploits identical state prefix ``z`` across candidates (O(M * d_z) state distance
        instead of O(C * M * d_z)) and precomputed action-axis squared deltas, running in
        pure NumPy with zero 3D tensor allocations and exact glibc IEEE-754 parity.
        """
        zp = _as_points(z, self.dim - 1, "z")[0]
        acts = np.asarray(actions_norm, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(acts)):
            raise ValueError("actions_norm must be finite")
        n = acts.shape[0]
        if self._sx.shape[0] == 0:
            return np.zeros(n, dtype=np.float64), np.ones(n, dtype=np.float64)

        inv_l_z, sx_z_s, da2_pre, buf, buf2, dz_buf = self._sync_candidate_cache(acts)
        zp_s = zp * inv_l_z
        np.subtract(zp_s, sx_z_s, out=dz_buf)
        dz2 = np.einsum("ij,ij->i", dz_buf, dz_buf)
        np.add(da2_pre, dz2[None, :], out=buf)

        nu = float(self.kernel.nu)
        if nu == 1.5:
            np.sqrt(buf, out=buf)
            buf *= _SQRT3
            np.negative(buf, out=buf2)
            np.exp(buf2, out=buf2)
            buf += 1.0
            buf *= buf2
        elif nu == 0.5:
            np.sqrt(buf, out=buf)
            np.negative(buf, out=buf)
            np.exp(buf, out=buf)
        elif nu == 2.5:
            np.sqrt(buf, out=buf)
            np.multiply(buf, _SQRT5, out=buf2)
            buf *= buf
            buf *= (5.0 / 3.0)
            buf += 1.0
            buf += buf2
            np.negative(buf2, out=buf2)
            np.exp(buf2, out=buf2)
            buf *= buf2
        else:
            buf *= -0.5
            np.exp(buf, out=buf)

        f = buf @ self._alpha
        sigma = 1.0 - buf.max(axis=1)
        return f, sigma

    def predict_and_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute ``f_tau(X)`` and ``sigma(X)`` from a single kernel pass.

        Numerically equivalent to ``(self.predict(X), self.uncertainty(X))``
        while avoiding 3D ``(n, m, dim)`` intermediate tensor allocations.
        """
        Xp = _as_points(X, self.dim, "X")
        n = Xp.shape[0]
        if self._sx.shape[0] == 0:
            return np.zeros(n, dtype=np.float64), np.ones(n, dtype=np.float64)
        # Fast path when all rows share the same state prefix (candidate scan)
        if n > 1 and self.dim > 1 and np.all(Xp[:, :-1] == Xp[0, :-1]):
            return self.predict_and_uncertainty_candidates(Xp[0, :-1], Xp[:, -1])
        inv_l = 1.0 / self.kernel.length_scale
        try:
            from scipy.spatial.distance import cdist

            d = cdist(Xp * inv_l, self._sx * inv_l, metric="euclidean")
            K = self.kernel._eval_pairwise(d)
        except Exception:
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
        self._sc_cache_key: tuple[int, float, float, float, int] | None = None
        self._sc_cache_z: np.ndarray | None = None
        self._sc_cache_out: dict | None = None

    def _invalidate_sc_cache(self) -> None:
        self._sc_cache_key = None
        self._sc_cache_z = None
        self._sc_cache_out = None

    def observe(self, z: np.ndarray, action: int, y: float, threshold: float | None = None):
        self._invalidate_sc_cache()
        rec = super().observe(z, action, y, threshold=threshold)
        if hasattr(self.regressor, "warm_candidate_cache"):
            self.regressor.warm_candidate_cache(self.n_prbs)
        return rec

    def insert_virtual_support(self, z: np.ndarray, action: int, alpha: float):
        self._invalidate_sc_cache()
        rec = super().insert_virtual_support(z, action, alpha)
        if hasattr(self.regressor, "warm_candidate_cache"):
            self.regressor.warm_candidate_cache(self.n_prbs)
        return rec

    def load_state_dict(self, state: dict) -> None:
        self._invalidate_sc_cache()
        super().load_state_dict(state)
        if hasattr(self.regressor, "warm_candidate_cache"):
            self.regressor.warm_candidate_cache(self.n_prbs)

    def score_candidates(self, z: np.ndarray) -> dict:
        """Evaluate f_tau, sigma, A_safe mask and gamma-objective in one cached pass."""
        zp = _as_points(z, self.state_dim, "z")[0]
        cur_key = (
            int(self.n_prbs),
            float(self.gamma),
            float(self.sla_threshold),
            float(self.regressor.tau),
            int(self.regressor._sx.shape[0]),
        )
        if (
            self._sc_cache_out is not None
            and self._sc_cache_key == cur_key
            and self._sc_cache_z is not None
            and np.array_equal(zp, self._sc_cache_z)
        ):
            return self._sc_cache_out

        actions_norm = np.arange(self.n_prbs + 1, dtype=np.float64) / float(self.n_prbs)
        if hasattr(self.regressor, "predict_and_uncertainty_candidates"):
            f, sigma = self.regressor.predict_and_uncertainty_candidates(zp, actions_norm)
        elif hasattr(self.regressor, "predict_and_uncertainty"):
            f, sigma = self.regressor.predict_and_uncertainty(self.candidate_features(zp))
        else:
            X = self.candidate_features(zp)
            f = self.regressor.predict(X)
            sigma = self.regressor.uncertainty(X)
        a_safe = f >= self.sla_threshold
        scores = f - self.gamma * actions_norm
        out = {"f": f, "sigma": sigma, "a_safe": a_safe, "scores": scores}
        self._sc_cache_key = cur_key
        self._sc_cache_z = zp.copy()
        self._sc_cache_out = out
        return out


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


def parse_action_length_scales_v2(
    scales_arg: str | list[float] | tuple[float, ...] | None,
    num_slices: int = 3,
) -> dict[int, float]:
    """Parse L3 per-slice Matérn length-scale ell_a for the normalized action axis x_6 = a_i / C."""
    if scales_arg is None or (isinstance(scales_arg, str) and not scales_arg.strip()):
        return {sid: 1.0 for sid in range(num_slices)}
    if isinstance(scales_arg, (tuple, list)):
        values = [float(v) for v in scales_arg]
    else:
        values = [float(p.strip()) for p in str(scales_arg).split(",") if p.strip()]
    if len(values) == 1:
        values = values * num_slices
    if len(values) != num_slices:
        raise ValueError(
            f"--action-length-scales must have 1 or {num_slices} values, got {len(values)}"
        )
    if any((not np.isfinite(v)) or v <= 0.0 for v in values):
        raise ValueError(
            f"--action-length-scales values must be finite and > 0, got {values!r}"
        )
    return dict(enumerate(values))


def _kernel_length_scale_for_slice(state_dim: int, ell_a: float) -> float | np.ndarray:
    """Return isotropic scalar 1.0 when ell_a == 1.0, or L3 anisotropic vector otherwise."""
    if abs(float(ell_a) - 1.0) <= 1e-12:
        return 1.0
    vec = np.ones(int(state_dim) + 1, dtype=np.float64)
    vec[-1] = float(ell_a)
    return vec


def build_controllers_v2(args, env=None):
    """Build v2 per-slice controllers (supports ``sqr``, ``kbrl``, ``equal``, ``prop``)."""
    slas = parse_slas_v2(args.slas, args.num_slices)
    het_mode = bool(getattr(args, "het", False))
    effective_slas = {sid: 0.0 for sid in range(args.num_slices)} if het_mode else slas
    action_scales = parse_action_length_scales_v2(
        getattr(args, "action_length_scales", None),
        args.num_slices,
    )
    controller = getattr(args, "controller", "sqr")
    if controller == "kbrl":
        from nsoran.sqr.kbrl import KbrlSliceController

        return {
            sid: KbrlSliceController(
                state_dim=6,
                n_prbs=args.c_prbs,
                sla_threshold=effective_slas[sid],
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
            f"unknown v2 controller {controller!r} (supported in v2: sqr, kbrl, equal, prop)"
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
            sla_threshold=effective_slas[sid],
            kernel=MaternKernel(
                dim=7,
                nu=1.5,
                length_scale=_kernel_length_scale_for_slice(6, action_scales[sid]),
            ),
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
