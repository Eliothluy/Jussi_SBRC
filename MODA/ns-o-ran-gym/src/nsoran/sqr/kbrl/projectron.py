"""Faithful NumPy port of the KBRL classifier of the reference repo.

Ported from ``third_party/network-slicing`` (commit 65a3891):
``algorithms/projectron.py`` (``SVvariable`` + ``Projectron``) and the
``GaussianKernel`` of ``algorithms/kernel.py`` — the exact combination used
by ``create_kbrl_agent`` (``scenario_creator.py:216-219``):

    sv = SVvariable()               # unbounded support set ...
    kernel = GaussianKernel(sv, 1)  # k(x,x') = exp(-||x-x'||^2), gamma = 1
    algorithm = Projectron(kernel)  # kernelized perceptron, eta = 0.1

The Projectron (Orabona et al., 2009) keeps the support set SMALL by
projection: an update that is (almost) in the span of the current support
vectors (``delta <= eta``) is absorbed into the coefficients instead of
inserting a new landmark.  This is the KBRL memory mechanism — there is NO
budget B here (that is the SQR regressor's mechanism); the ``SVvariable``
memory grows only when the projection fails, which keeps it near-linear in
practice.  ``n_support`` is the analogue of the SQR "memory size" in logs.

Registered deviations from the reference code (deterministic-behavior and
bookkeeping only; no algorithmic change):

1. **Deterministic tie-break** [DESVIO REGISTRADO]: ``GaussianKernel.predict``
   breaks ``sign(f) == 0`` with ``np.random.choice([-1, 1])`` on the GLOBAL
   numpy RNG (``algorithms/kernel.py:26-28``) — nondeterministic.  Here a
   tie predicts **-1** (conservative: "unknown -> assume SLA not fulfilled"),
   which also matches the empty-memory behavior (``Projectron.predict``
   returns 0 -> treated as not fulfilling).  No RNG is consumed anywhere in
   this module, so predictions are pure functions of the support state.
2. **float64 instead of float32** [DESVIO REGISTRADO]: the reference stores
   landmarks/coefficients/Kinv in float32; float64 is used here for
   cross-platform determinism of the ``delta <= eta`` comparisons.
3. **predict-then-update is self-contained** [DESVIO REGISTRADO]: the
   reference ``update`` consumes ``self.f``/``self.K_f`` cached by the last
   ``predict`` call (``projectron.py:32-52``); the port recomputes ``f`` and
   the kernel row from the ``x`` being updated.  For the documented call
   pattern (``predict(x)`` immediately before ``update(x, y)`` — the ONLY
   pattern in ``kbrl_control.py``) both are numerically identical; the port
   cannot silently consume a stale cache.
4. **state_dict/load_state_dict**: checkpoint support (the reference has
   none) so KBRL controllers round-trip through the SAME checkpoint
   envelope as SQR (``nsoran.sqr.checkpoint``).

Dependencies: NumPy only (no torch, no gym).
"""

from __future__ import annotations

import numpy as np

#: Format of the payload produced/consumed by :meth:`Projectron.state_dict`.
PROJECTRON_STATE_VERSION = 1


class SVVariable:
    """Unbounded support set (port of ``SVvariable``, projectron.py:3-20).

    ``landmarks`` grows by vstack on insert; ``coeff`` grows by append on
    extend.  ``update`` adds to the coefficient vector (the Projectron
    projection branch), exactly like the reference.
    """

    def __init__(self) -> None:
        self.counter = 0
        self.coeff: np.ndarray = np.zeros((0,), dtype=np.float64)
        self.landmarks: np.ndarray = np.zeros((0, 0), dtype=np.float64)

    def extend(self, new_coeff: float) -> None:
        """Append one coefficient (called right before :meth:`insert`)."""
        self.coeff = np.append(self.coeff, np.float64(new_coeff))

    def update(self, new_coeff: np.ndarray) -> None:
        """Absorb a projection: coeff += y * d_star (like ref)."""
        self.coeff = self.coeff + new_coeff

    def insert(self, x: np.ndarray) -> None:
        """Append one landmark."""
        if self.counter == 0:
            self.landmarks = np.asarray(x, dtype=np.float64)[None, :].copy()
        else:
            self.landmarks = np.vstack(
                [self.landmarks, np.asarray(x, dtype=np.float64)]
            )
        self.counter += 1


class GaussianKernelSV:
    """Gaussian kernel over the support set (port of ``GaussianKernel``).

    ``k(x1, x2) = exp(-gamma * ||x1 - x2||^2)`` with the reference default
    ``gamma = 1`` [REPO, scenario_creator.py:217].  k(x, x) = 1 always
    (normalized on the diagonal — the property the SQR Matérn kernel also
    enforces for its sigma definition).
    """

    def __init__(self, sv: SVVariable, gamma: float = 1.0) -> None:
        self.sv = sv
        self.gamma = float(gamma)

    def k_eval(self, x1: np.ndarray, x2: np.ndarray) -> float:
        diff = np.asarray(x1, dtype=np.float64) - np.asarray(x2, dtype=np.float64)
        return float(np.exp(-self.gamma * float(np.sum(diff**2))))

    def k(self, x: np.ndarray) -> np.ndarray:
        """Kernel values between every landmark and x, shape (n_sv,)."""
        if self.sv.counter == 0:
            return np.zeros((0,), dtype=np.float64)
        diff = self.sv.landmarks - np.asarray(x, dtype=np.float64)
        return np.exp(-self.gamma * np.sum(diff**2, axis=1))

    def predict(self, x: np.ndarray) -> tuple[int, float, np.ndarray]:
        """``(y, f, k)``; sign(f)==0 -> **-1** (deviation 1, deterministic)."""
        k_vec = self.k(x)
        f = float(k_vec @ self.sv.coeff) if k_vec.size else 0.0
        y = 1 if f > 0.0 else -1  # tie f == 0 -> -1
        return y, f, k_vec


class Projectron:
    """Kernelized perceptron with projection (port of ``Projectron``).

    Reference contract [REPO, algorithms/projectron.py:23-61]:

    - ``predict(x)`` returns sign(f) of the current support set (0 when
      empty — an empty classifier predicts NO action fulfills the SLA).
    - ``update(x, y)`` acts only when the margin condition ``f*y <= 0`` holds
      (perceptron mistake): with ``Kii = k(x,x) = 1``,
      ``d_star = Kinv @ K_f`` and ``delta = max(Kii - d_star @ K_f, 0)``:
        * ``delta <= eta``: project — ``coeff += y * d_star`` (no new SV);
        * else: insert a new SV with coefficient ``y`` and grow Kinv with
          the rank-1 update ``Kinv += outer(d+, d+)/delta``, d+ = [d*, -1].
    """

    def __init__(self, kernel: GaussianKernelSV, eta: float = 0.1) -> None:
        self.kernel = kernel
        self.sv = kernel.sv
        self.eta = float(eta)
        self.Kinv: np.ndarray = np.zeros((1,), dtype=np.float64)
        self.counter = 0

    # -- prediction ----------------------------------------------------

    def predict(self, x: np.ndarray) -> int:
        """Label of x: sign(f), 0 when the support set is empty (like ref)."""
        if self.counter == 0:
            return 0
        y, _f, _k = self.kernel.predict(np.asarray(x, dtype=np.float64))
        return y

    def _margin_state(self, x: np.ndarray) -> tuple[float, np.ndarray, float]:
        """(f, K_f, Kii) of x under the CURRENT support set (fresh compute)."""
        x = np.asarray(x, dtype=np.float64)
        kii = self.kernel.k_eval(x, x)
        if self.counter == 0:
            return 0.0, np.zeros((0,), dtype=np.float64), kii
        _y, f, k_f = self.kernel.predict(x)
        return float(f), k_f, kii

    # -- learning ------------------------------------------------------

    def update(self, x: np.ndarray, y: int) -> None:
        """One online perceptron step at x with label y in {-1, +1}.

        Recomputes ``f``/``K_f`` at x under the current support set (see
        deviation 3) — numerically identical to the reference's
        predict-immediately-before-update pattern.
        """
        y = int(y)
        if y not in (-1, 1):
            raise ValueError(f"y must be -1 or +1, got {y}")
        x = np.asarray(x, dtype=np.float64)
        f, k_f, kii = self._margin_state(x)
        if f * y > 0:
            return  # correct prediction: no update (perceptron)

        if k_f.size:
            d_star = self.Kinv @ k_f
        else:
            d_star = np.zeros((0,), dtype=np.float64)
        if d_star.ndim == 0:
            d_star = np.array([d_star], dtype=np.float64)
        delta = max(kii - float(d_star @ k_f), 0.0) if k_f.size else kii

        if delta <= self.eta:
            # Projection: absorb into the existing coefficients.
            if d_star.size:
                self.sv.update(y * d_star)
        else:
            # Insert a new support vector with coefficient y.
            self.sv.extend(y)
            self.sv.insert(x)
            self.counter += 1
            if self.counter > 1:
                kinv = np.vstack(
                    [self.Kinv, np.zeros(self.counter - 1, dtype=np.float64)]
                )
                kinv = np.column_stack(
                    [kinv, np.zeros(self.counter, dtype=np.float64)]
                )
                d_star_extend = np.append(d_star, -1.0)
                kinv += np.outer(d_star_extend, d_star_extend) / delta
                self.Kinv = kinv
            else:
                self.Kinv = np.array([1.0 / kii], dtype=np.float64)

    # -- introspection / checkpoint -------------------------------------

    @property
    def n_support(self) -> int:
        """Number of support vectors (the KBRL analogue of memory usage)."""
        return int(self.sv.counter)

    def state_dict(self) -> dict:
        return {
            "format_version": PROJECTRON_STATE_VERSION,
            "eta": self.eta,
            "gamma": self.kernel.gamma,
            "counter": int(self.sv.counter),
            "landmarks": self.sv.landmarks.copy(),
            "coeff": self.sv.coeff.copy(),
            "Kinv": self.Kinv.copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        version = int(state.get("format_version", -1))
        if version != PROJECTRON_STATE_VERSION:
            raise ValueError(
                f"unsupported projectron state format_version {version!r} "
                f"(expected {PROJECTRON_STATE_VERSION})"
            )
        self.eta = float(state["eta"])
        self.kernel.gamma = float(state["gamma"])
        # Projectron.counter mirrors SVVariable.counter (both advance on
        # insert); BOTH are restored, then cross-checked.
        self.counter = int(state["counter"])
        self.sv.counter = self.counter
        self.sv.landmarks = np.asarray(state["landmarks"], dtype=np.float64).copy()
        self.sv.coeff = np.asarray(state["coeff"], dtype=np.float64).copy()
        self.Kinv = np.asarray(state["Kinv"], dtype=np.float64).copy()
        if self.sv.coeff.shape[0] != self.counter:
            raise ValueError(
                f"coeff length {self.sv.coeff.shape[0]} != counter "
                f"{self.counter}"
            )
        expected_kinv = 1 if self.counter <= 1 else self.counter
        if self.Kinv.shape[0] != expected_kinv:
            raise ValueError(
                f"Kinv shape {self.Kinv.shape} inconsistent with counter "
                f"{self.counter} (expected {expected_kinv})"
            )
