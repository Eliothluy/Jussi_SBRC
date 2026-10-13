"""KBRL slice controller — port of the KBRL baseline as an SQR-compatible
per-slice controller (Fase 3, PLANO_IMPLEMENTACAO_SQR_NORI.md).

Reference: Alcaraz et al., "Model-Based Reinforcement Learning with Kernels
for Resource Allocation in RAN Slices", IEEE TWC 2022 — code
``third_party/network-slicing`` (commit 65a3891), files ``kbrl_control.py``
(``KBRL_Control``/``Learner``), ``algorithms/projectron.py`` and
``scenario_creator.py`` (agent hyperparameters).  The port keeps the
reference STRUCTURE (safety/monotonicity per slice) adapted to the frozen
SQR contract (``MODA/docs/sqr/contrato_interface.md``):

- **Same observation**: ``act(z_i)`` consumes the per-slice z (dim 6,
  register R20) exactly like ``SqrSliceController.act``.
- **Same action space**: integer PRBs ``a_i in {0..C}``; the cross-slice
  feasibility (``sum a_i <= C``) is NOT resolved here — the environment's
  deterministic projection Pi (register R24, identical for ALL methods)
  cuts over-capacity proposals exactly like it cuts SQR proposals.  The
  reference's own ``KBRL_Control.adjust_action`` (``kbrl_control.py:75-78``)
  is superseded by Pi for experimental fairness (same projection, same
  arithmetics for SQR gamma=0/gamma=1/KBRL).

Per-decision algorithm (faithful port of ``KBRL_Control.select_action``,
``kbrl_control.py:41-63``):

1. Scan candidate actions a = 0, 1, ..., C.  The FIRST a whose classifier
   prediction is +1 ("SLA fulfilled") is the minimal safe action a0.
2. The proposal is ``a = min(C, a0 + security_factor)``; the offset
   ``security_factor`` is the KBRL safety margin (analogue of the SQR
   ``round(sigma*M)`` margin) and ``margin = a - a0``.
3. If NO candidate is predicted +1 (cold start: empty support set predicts
   0 for everything), the proposal is ``a = C`` with margin 0 — the
   reference behavior when the scan exhausts (``kbrl_control.py:54-62``);
   the projection Pi then cuts it down.

Per-window learning (faithful port of ``KBRL_Control.update_control``,
``kbrl_control.py:80-114``):

1. Label the window: ``y_hat = +1 if y_t >= Lambda else -1`` — the KBRL
   paper imposes the SLA at SLICE level, so the label uses the SAME window
   statistic as the SQR constraint (y_t = p5 per-UE throughput of the
   slice, register R18) against the same SLA Lambda.  This is the declared
   granularity harmonization (PLANO Fase 3): same SLA semantics for every
   method; violations remain logged at BOTH levels (UE-time: register R19
   rate; slice-time: 1[y_t < Lambda] per slice-window) with explicit
   denominators.
2. Predict at x = (z, a_executed/C); hit = (prediction == label).
3. Accuracy EMA over MARGINS (alfa decay): when the classifier predicted
   +1, a miss (label -1) decays ``accuracies[0:margin+1]`` (the same or a
   smaller margin would have failed too); a hit boosts
   ``accuracies[margin:]`` (the same or a larger margin would have
   succeeded too).  Margin here is the post-projection margin
   ``max(0, margin - cut)`` — the reference subtracts the adjustment from
   the margins before attributing accuracy (``kbrl_control.py:69``).
4. The security factor is updated ONLY when the projection did NOT cut
   anything in the window (the reference's global ``adjusted`` flag,
   ``kbrl_control.py:98-99``): ``security_factor = argmax(accuracies >
   accuracy_range[0])`` — the FIRST margin whose accuracy exceeds the
   lower accuracy target (argmax of the boolean = first True; 0 when no
   margin qualifies, like the reference).
5. **Monotonic sample augmentation** (``kbrl_control.py:102-112``): with
   label +1, update the classifier at (z, a/C) for every a >= a_executed
   ("same or more PRBs would fulfill as well"); with label -1, for every
   a <= a_executed.  This encodes the per-slice monotonicity assumption —
   the structural safety of KBRL.

Determinism: the only stochastic element of the reference agent is the
INITIAL security factor (``rng.integers(embb_sec)``, eMBB range [2, 8) —
``scenario_creator.py:191,221``).  Here it is drawn from
``numpy.random.default_rng(seed)`` (same-seed reproducibility, same
convention as ``SqrSliceController``); classifier predictions consume no
RNG at all (f==0 ties predict -1, see ``projectron.py`` deviation 1).

Hyperparameters [REPO — reference defaults, scenario_creator.py]:
``alfa = 0.05`` (L187), ``accuracy_range = (0.99, 0.999)`` (L197 default;
both (0.97, 0.99) and (0.99, 0.999) appear in experiments_kbrl.py L26),
Gaussian ``gamma = 1`` (L217), Projectron ``eta = 0.1`` (L219),
initial security factor uniform in [2, 8) (L191/L221).
``gamma`` (SQR's resource-cost weight) has NO KBRL analogue — KBRL is
resource-economical STRUCTURALLY (minimal-action scan); ``--gamma`` is
ignored by this controller.

Dynamic C (register R01 as amended): the run loop updates ``n_prbs`` when
the ns-3 telemetry reports the effective capacity; the accuracy table and
the security factor are re-scaled (deviation from the reference, which has
a fixed C — [ASSUMIDO] adaptation required by our contract).

Dependencies: NumPy only (no torch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nsoran.sqr.kbrl.projectron import (
    PROJECTRON_STATE_VERSION,
    GaussianKernelSV,
    Projectron,
    SVVariable,
)

#: Hyperparameters frozen from the reference repo [REPO].
KBRL_ALFA = 0.05                     # accuracy EMA rate (scenario_creator L187)
KBRL_ACCURACY_RANGE = (0.99, 0.999)  # create_kbrl_agent default (L197)
KBRL_GAMMA = 1.0                     # Gaussian kernel width (L217)
KBRL_ETA = 0.1                       # Projectron projection threshold (L219)
KBRL_INITIAL_SEC_RANGE = (2, 8)      # eMBB initial security factor (L191)
KBRL_STATE_VERSION = 1

#: Decision modes of this controller (log values, "mode" column).
MODE_SCAN = "scan"  # a candidate action was predicted +1
MODE_COLD = "cold"  # no candidate predicted +1 (incl. empty support set)


@dataclass(frozen=True)
class KbrlDecision:
    """One decision of the KBRL slice controller (for logs and tests).

    Field names mirror :class:`nsoran.sqr.sqr_controller.SqrDecision` so
    the shared run loop and step_metrics.csv keep the SAME columns for both
    controllers; fields without a KBRL analogue carry NaN (documented):
    ``sigma``/``tau`` belong to the SQR quantile machinery.
    """

    action: int          # proposal for this slice, in [0, C]
    raw_action: int      # minimal predicted-fulfilling action a0
    mode: str            # scan | cold
    sigma: float         # NaN (no KBRL analogue; column kept for the contract)
    margin_applied: int  # security offset actually added (min(C, a0+sec) - a0)
    a_safe_size: int     # candidates predicted +1 (KBRL analogue of |A_safe|)
    n_candidates: int    # C + 1
    tau: float           # NaN (no KBRL analogue)
    margin_m: float      # current security factor (the KBRL margin analog)


@dataclass(frozen=True)
class KbrlUpdate:
    """Record of one learning window (for logs and tests)."""

    label: int           # +1 fulfilled / -1 violated (slice-time indicator)
    y_pred: int          # classifier prediction at the executed action
    hit: int             # 1 if y_pred == label
    margin: int          # post-projection margin used for the accuracy EMA
    security_factor: int # security factor AFTER the (possibly skipped) update
    n_support: int       # support-set size after the updates
    n_augmented: int     # monotonic augmentation updates applied this window


class KbrlSliceController:
    """KBRL controller for ONE slice (own Projectron + own accuracy table).

    Parameters
    ----------
    state_dim:
        Dimension of the local state z (6, register R20).
    n_prbs:
        C, total PRBs of the cell; candidate actions are {0, ..., C}.
    sla_threshold:
        Lambda, the SLA level (Mbps) the label is written on — the SAME
        per-slice Lambda used by the SQR controller (harmonization).
    alfa:
        Accuracy EMA decay [REPO, 0.05].
    accuracy_range:
        (lower, upper) accuracy targets [REPO, default (0.99, 0.999)]; the
        security factor is the first margin with accuracy > lower.
    seed:
        Seed for the INITIAL security factor draw only (determinism); the
        decision/learning dynamics themselves are deterministic.
    """

    def __init__(
        self,
        state_dim: int,
        n_prbs: int,
        sla_threshold: float = 1.0,
        alfa: float = KBRL_ALFA,
        accuracy_range: tuple[float, float] = KBRL_ACCURACY_RANGE,
        kernel_gamma: float = KBRL_GAMMA,
        eta: float = KBRL_ETA,
        initial_security_factor: int | None = None,
        seed: int | None = None,
    ) -> None:
        state_dim = int(state_dim)
        n_prbs = int(n_prbs)
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if n_prbs <= 0:
            raise ValueError(f"n_prbs must be positive, got {n_prbs}")
        if not (0.0 <= float(alfa) <= 1.0):
            raise ValueError(f"alfa must be in [0, 1], got {alfa}")
        if not (0.0 <= accuracy_range[0] <= accuracy_range[1] <= 1.0):
            raise ValueError(
                f"accuracy_range must be 0 <= lo <= hi <= 1, got {accuracy_range}"
            )

        self.state_dim = state_dim
        self.n_prbs = n_prbs
        self.sla_threshold = float(sla_threshold)
        self.alfa = float(alfa)
        self.accuracy_range = (float(accuracy_range[0]), float(accuracy_range[1]))

        self.classifier = Projectron(
            GaussianKernelSV(SVVariable(), gamma=kernel_gamma), eta=eta
        )

        # Initial accuracy = midpoint of the accuracy range (the reference
        # initializes the table with (lo + hi)/2, kbrl_control.py:38-39).
        initial_value = 0.5 * (self.accuracy_range[0] + self.accuracy_range[1])
        self._initial_accuracy = float(initial_value)
        self._accuracies = np.full(n_prbs + 1, self._initial_accuracy)

        # Initial security factor: explicit value or seeded draw from the
        # reference eMBB range [2, 8) [REPO, scenario_creator.py:191,221].
        if initial_security_factor is not None:
            sec = int(initial_security_factor)
        elif seed is not None:
            sec = int(np.random.default_rng(seed).integers(*KBRL_INITIAL_SEC_RANGE))
        else:
            # Deterministic fallback: first-quartile draw of the same range.
            sec = KBRL_INITIAL_SEC_RANGE[0] + 1
        if not (0 <= sec <= n_prbs):
            sec = int(np.clip(sec, 0, n_prbs))
        self._security_factor = sec

        # Per-decision bookkeeping consumed by observe() (the reference
        # keeps self.margins/self.adjusted on the agent, kbrl_control.py
        # L37/L66-69 — same state, per slice).
        self._last_a0 = 0
        self._last_margin = 0
        self._last_proposal = 0
        self._adjusted = False
        self._n_updates = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def security_factor(self) -> int:
        """Current security offset (the KBRL margin analog, in PRBs)."""
        return int(self._security_factor)

    @property
    def n_support(self) -> int:
        """Support-vector count of the slice classifier (memory usage)."""
        return self.classifier.n_support

    @property
    def n_updates(self) -> int:
        """Total classifier updates applied (real + monotonic augmentation)."""
        return int(self._n_updates)

    def accuracies(self) -> np.ndarray:
        """Copy of the per-margin accuracy table (length C + 1)."""
        return self._accuracies.copy()

    # ------------------------------------------------------------------
    # Policy (faithful port of KBRL_Control.select_action, per slice)
    # ------------------------------------------------------------------

    def _candidate_features(self, z: np.ndarray) -> np.ndarray:
        """Feature matrix [z, a/C] for a = 0..C, shape (C+1, state_dim+1)."""
        zp = np.asarray(z, dtype=np.float64).reshape(-1)
        if zp.shape[0] != self.state_dim:
            raise ValueError(
                f"z must have {self.state_dim} features, got {zp.shape[0]}"
            )
        actions = np.arange(self.n_prbs + 1, dtype=np.float64) / float(self.n_prbs)
        return np.column_stack([np.tile(zp, (self.n_prbs + 1, 1)), actions])

    def act(self, z: np.ndarray) -> KbrlDecision:
        """Choose the per-slice proposal a in {0, ..., C} (minimal scan)."""
        # The reference select_action assigns ``adjusted`` on EVERY call
        # (kbrl_control.py:67,73 — no adjustment is the default); the run
        # loop re-marks it via notify_projection() before observe().
        self._adjusted = False
        X = self._candidate_features(z)
        # Predictions are pure (no RNG): evaluate the whole grid to count
        # the predicted-safe set for the log.  The SCAN result is identical
        # to the reference's early-break loop (first predicted +1 from 0).
        preds = np.array([self.classifier.predict(X[a]) for a in range(X.shape[0])])
        a_safe = np.flatnonzero(preds == 1)

        if a_safe.size > 0:
            a0 = int(a_safe[0])
            action = min(self.n_prbs, a0 + self._security_factor)
            margin = action - a0
            mode = MODE_SCAN
        else:
            # Scan exhausted (or empty classifier predicting 0/-1 for all):
            # everything to this slice, margin 0; Pi cuts it (ref behavior).
            a0 = self.n_prbs
            action = self.n_prbs
            margin = 0
            mode = MODE_COLD

        # Bookkeeping for observe() (post-projection attribution).
        self._last_a0 = a0
        self._last_margin = margin
        self._last_proposal = action

        return KbrlDecision(
            action=int(action),
            raw_action=int(a0),
            mode=mode,
            sigma=float("nan"),
            margin_applied=int(margin),
            a_safe_size=int(a_safe.size),
            n_candidates=int(self.n_prbs + 1),
            tau=float("nan"),
            margin_m=float(self._security_factor),
        )

    def notify_projection(self, over_capacity: bool) -> None:
        """Record the cross-slice projection outcome of the current window.

        The reference gates the security-factor update on the GLOBAL
        ``adjusted`` flag (``kbrl_control.py:66-68,98``): whenever the sum
        of proposals exceeded C, NO slice's security factor is updated in
        that window (the margins actually executed are unknown to the
        agent's bookkeeping until the per-slice cut is subtracted in
        :meth:`observe`).  Called by the run loop with
        ``info["projection"]["over_capacity"]`` BEFORE the observe calls.
        """
        self._adjusted = bool(over_capacity)

    # ------------------------------------------------------------------
    # Learning (faithful port of KBRL_Control.update_control, per slice)
    # ------------------------------------------------------------------

    def observe(self, z: np.ndarray, action: int, y: float) -> KbrlUpdate:
        """Feed one window outcome back (label from y_t vs Lambda).

        ``action`` is the EXECUTED (post-projection) action — the reference
        also updates with the executed action (``kbrl_control.py:128-132``:
        ``update_control(state, action, ...)`` receives the adjusted
        action).  ``y`` is the window statistic (p5 per-UE throughput,
        Mbps); the slice-time label is ``+1 if y >= Lambda else -1``.

        Windows where the slice has NO UE (y undefined) never reach this
        method — the shared run loop excludes them from learning (R19),
        keeping the denominators explicit.
        """
        executed = int(action)
        if not (0 <= executed <= self.n_prbs):
            raise ValueError(
                f"action must be in [0, {self.n_prbs}], got {executed}"
            )
        zp = np.asarray(z, dtype=np.float64).reshape(-1)
        if zp.shape[0] != self.state_dim:
            raise ValueError(
                f"z must have {self.state_dim} features, got {zp.shape[0]}"
            )
        label = 1 if float(y) >= self.sla_threshold else -1

        # Post-projection margin (ref subtracts the adjustment diff first,
        # kbrl_control.py:69; negative cut means the executed action is
        # BELOW a0 — possible only via Pi, clamp at 0 like the ref).
        cut = max(0, self._last_proposal - executed)
        margin = max(0, self._last_margin - cut)

        x = np.append(zp, executed / float(self.n_prbs))
        y_pred = self.classifier.predict(x)
        hit = 1 if y_pred == label else 0

        if y_pred == 1:
            if hit == 0:
                # Same or less margin would have made the same mistake.
                end = min(margin + 1, self._accuracies.shape[0])
                self._accuracies[:end] *= 1.0 - self.alfa
            else:
                # Same or more margin would have succeeded as well.
                start = min(margin, self._accuracies.shape[0] - 1)
                tail = self._accuracies[start:]
                self._accuracies[start:] = (1.0 - self.alfa) * tail + self.alfa
        if not self._adjusted:
            # argmax of the boolean mask = FIRST margin above the accuracy
            # target; 0 when none qualifies (numpy argmax semantics, same
            # as the reference expression).
            self._security_factor = int(
                np.argmax(self._accuracies > self.accuracy_range[0])
            )

        # Monotonic sample augmentation (kbrl_control.py:102-112).
        n_augmented = 0
        if label == 1:
            grid = range(executed, self.n_prbs + 1)
        else:
            grid = range(0, executed + 1)
        for a in grid:
            xa = np.append(zp, a / float(self.n_prbs))
            self.classifier.predict(xa)
            self.classifier.update(xa, label)
            n_augmented += 1
        self._n_updates += n_augmented

        return KbrlUpdate(
            label=label,
            y_pred=int(y_pred),
            hit=hit,
            margin=int(margin),
            security_factor=int(self._security_factor),
            n_support=self.classifier.n_support,
            n_augmented=int(n_augmented),
        )

    # ------------------------------------------------------------------
    # Dynamic C (register R01 as amended; deviation from the fixed-C ref)
    # ------------------------------------------------------------------

    def set_n_prbs(self, n_prbs: int) -> None:
        """Adopt a new cell capacity C (telemetry-driven, R01).

        The accuracy table is rebuilt at the new size: existing entries are
        preserved up to the overlap and the remainder is filled with the
        initial value; the security factor is clamped into [0, C].
        """
        n_prbs = int(n_prbs)
        if n_prbs <= 0:
            raise ValueError(f"n_prbs must be positive, got {n_prbs}")
        if n_prbs == self.n_prbs:
            return
        new_acc = np.full(n_prbs + 1, self._initial_accuracy)
        keep = min(self._accuracies.shape[0], new_acc.shape[0])
        new_acc[:keep] = self._accuracies[:keep]
        self._accuracies = new_acc
        self.n_prbs = n_prbs
        self._security_factor = int(
            np.clip(self._security_factor, 0, self.n_prbs)
        )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Full restorable state of the KBRL slice controller."""
        return {
            "controller": "kbrl",
            "format_version": KBRL_STATE_VERSION,
            "state_dim": self.state_dim,
            "n_prbs": self.n_prbs,
            "sla_threshold": self.sla_threshold,
            "alfa": self.alfa,
            "accuracy_range": self.accuracy_range,
            "security_factor": int(self._security_factor),
            "accuracies": self._accuracies.copy(),
            "initial_accuracy": self._initial_accuracy,
            "n_updates": int(getattr(self, "_n_updates", 0)),
            "projectron": self.classifier.state_dict(),
            "projectron_version": PROJECTRON_STATE_VERSION,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a checkpoint saved by :meth:`state_dict` (fail loudly)."""
        if state.get("controller") != "kbrl":
            raise ValueError(
                f"not a KBRL controller state (controller="
                f"{state.get('controller')!r}); refusing to restore it into "
                f"a KbrlSliceController"
            )
        version = int(state.get("format_version", -1))
        if version != KBRL_STATE_VERSION:
            raise ValueError(
                f"unsupported KBRL state format_version {version!r} "
                f"(expected {KBRL_STATE_VERSION})"
            )
        if int(state["state_dim"]) != self.state_dim:
            raise ValueError(
                f"state_dim mismatch: checkpoint has {state['state_dim']}, "
                f"controller has {self.state_dim}"
            )
        n_prbs = int(state["n_prbs"])
        self.classifier.load_state_dict(state["projectron"])
        self.n_prbs = n_prbs
        self.sla_threshold = float(state["sla_threshold"])
        self.alfa = float(state["alfa"])
        self.accuracy_range = (
            float(state["accuracy_range"][0]),
            float(state["accuracy_range"][1]),
        )
        acc = np.asarray(state["accuracies"], dtype=np.float64)
        if acc.shape[0] != n_prbs + 1:
            raise ValueError(
                f"accuracies length {acc.shape[0]} != n_prbs + 1 "
                f"({n_prbs + 1})"
            )
        self._accuracies = acc.copy()
        self._initial_accuracy = float(state["initial_accuracy"])
        self._security_factor = int(state["security_factor"])
        self._n_updates = int(state.get("n_updates", 0))
        self._last_a0 = 0
        self._last_margin = 0
        self._last_proposal = 0
        self._adjusted = False
