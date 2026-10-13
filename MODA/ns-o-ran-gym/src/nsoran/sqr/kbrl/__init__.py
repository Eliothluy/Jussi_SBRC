"""KBRL baseline (Alcaraz et al., IEEE TWC 2022) as an SQR-compatible
controller — Fase 3 (PLANO_IMPLEMENTACAO_SQR_NORI.md §3-Fase 3).

Faithful NumPy port of the reference implementation
(``third_party/network-slicing``, commit 65a3891 — ``kbrl_control.py``,
``algorithms/kernel.py``, ``algorithms/projectron.py``), adapted to the
frozen SQR contract: same per-slice observations z (register R20), same
integer-PRB action space {0..C} with the SAME cross-slice projection Pi
(register R24) applied by the environment, same SLA statistic (y_t = p5
per-UE throughput, register R18) and same per-slice Lambda as the SQR
agent.  The reference repo's old dependencies (gym 0.15, stable-baselines,
TensorFlow 1.x) are NOT needed: only NumPy (see
``third_party/README.md``).

Structure kept from the reference (safety/monotonicity per slice):

- one kernelized online binary classifier per slice (Projectron, Gaussian
  kernel gamma=1, eta=0.1) predicting whether the SLA is fulfilled at
  x = [z, a/C];
- minimal-action scan: the proposal is the FIRST a predicted +1 plus the
  ``security_factor`` offset (resource-economical STRUCTURALLY — this is
  why KBRL has no gamma cost term; ``--gamma`` does not apply);
- accuracy EMA over margins (alfa = 0.05) drives the security factor:
  ``argmax(accuracies > accuracy_range[0])`` with the default accuracy
  range (0.99, 0.999);
- monotonic sample augmentation: label +1 updates every a >= executed,
  label -1 every a <= executed.

Registered deviations (determinism/bookkeeping, documented in the module
docstrings): f==0 tie-breaks to -1 instead of the reference's global-RNG
coin; float64 arithmetic; self-contained predict-then-update; checkpoint
support; dynamic C (register R01); the reference ``adjust_action`` is
superseded by the environment's projection Pi for experimental fairness.

Violations are logged at BOTH levels with explicit denominators
(granularity harmonization, PLANO Fase 3): UE-time = register R19 rate
(denominator: eligible UE-window observations); slice-time = 1[y_t <
Lambda] per slice-window (denominator: windows with UEs in the slice) —
both derivable from step_metrics.csv and reported by
``examples/run_sqr_integration_kbrl.sh``.

Dependencies: NumPy only (no torch).
"""

from nsoran.sqr.kbrl.projectron import (
    PROJECTRON_STATE_VERSION,
    GaussianKernelSV,
    Projectron,
    SVVariable,
)
from nsoran.sqr.kbrl.kbrl_controller import (
    KBRL_ACCURACY_RANGE,
    KBRL_ALFA,
    KBRL_ETA,
    KBRL_GAMMA,
    KBRL_INITIAL_SEC_RANGE,
    KBRL_STATE_VERSION,
    KbrlDecision,
    KbrlSliceController,
    KbrlUpdate,
    MODE_COLD,
    MODE_SCAN,
)

__all__ = [
    "PROJECTRON_STATE_VERSION",
    "GaussianKernelSV",
    "Projectron",
    "SVVariable",
    "KBRL_ACCURACY_RANGE",
    "KBRL_ALFA",
    "KBRL_ETA",
    "KBRL_GAMMA",
    "KBRL_INITIAL_SEC_RANGE",
    "KBRL_STATE_VERSION",
    "KbrlDecision",
    "KbrlSliceController",
    "KbrlUpdate",
    "MODE_COLD",
    "MODE_SCAN",
]
