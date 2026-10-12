"""Normalized Matérn kernel with per-dimension length-scales.

The SQR quantile regressor is kernelized with a Matérn kernel [ARTIGO,
method summary in PLANO_IMPLEMENTACAO_SQR_NORI.md §2.1: "kernel Matérn"].
The paper does not publish nu or the length-scales; this module assumes
nu=1.5 by default ([ASSUMIDO], Fase 0 of the plan) and exposes both as
constructor arguments so sensitivity studies can sweep them.

The kernel is normalized: k(x, x) = 1 exactly for every supported nu,
because every Matérn autocorrelation evaluates to (1 + 0) * exp(0) = 1
(or exp(0) = 1 for the RBF limit) at zero distance.

All operations are vectorized with NumPy broadcasting (no explicit Python
loops over points); k(X, Y) for n x m point pairs costs O(n*m*dim).
"""

from __future__ import annotations

import numpy as np

# Supported smoothness parameters and their closed forms.
_SUPPORTED_NU = (0.5, 1.5, 2.5, np.inf)


def _as_points(X: np.ndarray, dim: int, name: str) -> np.ndarray:
    """Validate and coerce ``X`` to a float64 array of shape (n, dim).

    Accepts a single point of shape (dim,) or a batch of shape (n, dim).
    A 1-D input whose length differs from ``dim`` is rejected (ambiguous).
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != dim:
            raise ValueError(
                f"{name}: expected shape ({dim},) for a single point or "
                f"(n, {dim}) for a batch, got {arr.shape}"
            )
        arr = arr[None, :]
    elif arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(
            f"{name}: expected shape (n, {dim}), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: all coordinates must be finite")
    return arr


class MaternKernel:
    """Matérn kernel k(x, x') with anisotropic (per-dimension) length-scales.

    Parameters
    ----------
    dim:
        Feature dimensionality.
    length_scale:
        Scalar or array of shape (dim,). Distances are scaled per dimension:
        d(x, x') = sqrt(sum_d ((x_d - x'_d) / l_d)^2). Must be finite and > 0.
    nu:
        Smoothness parameter. Supported: 0.5 (exponential), 1.5 (default,
        [ASSUMIDO] per Fase 0), 2.5 and inf (squared-exponential limit).

    Notes
    -----
    Closed forms used (Rasmussen & Williams, 2006, eq. 4.21 / 4.22):
      nu = 1/2: k(d) = exp(-d)
      nu = 3/2: k(d) = (1 + sqrt(3) d) * exp(-sqrt(3) d)
      nu = 5/2: k(d) = (1 + sqrt(5) d + (5/3) d^2) * exp(-sqrt(5) d)
      nu = inf: k(d) = exp(-d^2 / 2)
    All give k(0) = 1, hence k(x, x) = 1.
    """

    def __init__(
        self,
        dim: int,
        length_scale: float | np.ndarray = 1.0,
        nu: float = 1.5,
    ) -> None:
        dim = int(dim)
        if dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")

        scales = np.asarray(length_scale, dtype=np.float64)
        if scales.ndim == 0:
            scales = np.full(dim, float(scales), dtype=np.float64)
        scales = scales.flatten()
        if scales.shape[0] != dim:
            raise ValueError(
                f"length_scale must be a scalar or have shape ({dim},), "
                f"got {np.asarray(length_scale).shape}"
            )
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("length_scale must be finite and strictly positive")

        if nu not in _SUPPORTED_NU:
            raise ValueError(
                f"nu must be one of {_SUPPORTED_NU}, got {nu}"
            )

        self.dim = dim
        self.nu = float(nu)
        self.length_scale = scales

    # ------------------------------------------------------------------
    # Distances
    # ------------------------------------------------------------------

    def scaled_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Pairwise scaled Euclidean distances, shape (len(X), len(Y)).

        Computed by direct differencing in row blocks: unlike the expanded
        Gram trick (||x||^2 + ||y||^2 - 2 x.y), this is numerically exact
        for identical points, which the normalization k(x, x) = 1 requires.
        """
        Xp = _as_points(X, self.dim, "X")
        Yp = _as_points(Y, self.dim, "Y")
        inv_l = 1.0 / self.length_scale
        n, m = Xp.shape[0], Yp.shape[0]
        out = np.empty((n, m), dtype=np.float64)
        # Cap each block at ~8M elements to bound temporary memory.
        block = max(1, int(8_000_000 // max(m * self.dim, 1)))
        for start in range(0, n, block):
            stop = min(start + block, n)
            diff = (Xp[start:stop, None, :] - Yp[None, :, :]) * inv_l
            out[start:stop] = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        return out

    # ------------------------------------------------------------------
    # Kernel evaluation
    # ------------------------------------------------------------------

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        """Kernel matrix k(X, Y) of shape (len(X), len(Y)).

        ``Y=None`` computes k(X, X); the result is symmetric and its
        diagonal is exactly 1.0.
        """
        d = self.scaled_distance(X, X if Y is None else Y)
        if self.nu == 0.5:
            return np.exp(-d)
        if self.nu == 1.5:
            s3d = np.sqrt(3.0) * d
            return (1.0 + s3d) * np.exp(-s3d)
        if self.nu == 2.5:
            s5d = np.sqrt(5.0) * d
            return (1.0 + s5d + (5.0 / 3.0) * d ** 2) * np.exp(-s5d)
        # nu == inf: squared-exponential limit
        return np.exp(-0.5 * d ** 2)

    def diag(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Element-wise kernel k(X[i], Y[i]) without forming the full matrix."""
        Xp = _as_points(X, self.dim, "X")
        Yp = _as_points(Y, self.dim, "Y")
        if Xp.shape[0] != Yp.shape[0]:
            raise ValueError("diag requires X and Y with the same number of points")
        d = np.sqrt(np.sum(((Xp - Yp) / self.length_scale) ** 2, axis=1))
        return self._eval_pairwise(d)

    def _eval_pairwise(self, d: np.ndarray) -> np.ndarray:
        if self.nu == 0.5:
            return np.exp(-d)
        if self.nu == 1.5:
            s3d = np.sqrt(3.0) * d
            return (1.0 + s3d) * np.exp(-s3d)
        if self.nu == 2.5:
            s5d = np.sqrt(5.0) * d
            return (1.0 + s5d + (5.0 / 3.0) * d ** 2) * np.exp(-s5d)
        return np.exp(-0.5 * d ** 2)
