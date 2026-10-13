"""SQR — Safe Quantile Regression agent (model-based RL, no neural network).

Python reimplementation of the SQR method from:
  Aerman Tuerxun and Akihiro Nakao, "Safe RAN Slicing in O-RAN: Minimizing
  SLA Violations via Model-Based Reinforcement Learning", IEEE CCNC 2026,
  DOI 10.1109/CCNC65079.2026.11366589.

Modules
-------
- ``kernel``             : normalized Matérn kernel (k(x,x)=1, per-dim length-scales).
- ``quantile_regressor`` : online kernelized pinball quantile regressor with a
                           bounded (B=2000) support-vector memory.
- ``sqr_controller``     : hierarchical safety-first policy (A_safe / softmax /
                           k-NN) with the round(sigma*M) safety margin and the
                           tau/M adaptation heuristic.
- ``coordinator``        : deterministic integer proportional projection of the
                           per-slice PRB proposals onto sum(a) <= C, with the
                           virtual-update signal (exact weight, register R12)
                           for cut slices.
- ``sqr_kpis``           : KPM wrapper for sqr-kpms.txt (sentinel-aware SNR
                           imputation, register R09; per-slice lowest/p5/UE
                           count derivations).
- ``sqr_env``            : Gymnasium environment over the IPC contract
                           (obs per slice + globals; integer-PRB actions).
- ``checkpoint``         : file round-trip helpers for controller state.

Traceability convention (GUIA_IMPLEMENTACAO_SQR_O_RAN_PYTHON_NS3.md):
values tagged ``[ARTIGO]`` appear in the paper (section cited); values tagged
``[ASSUMIDO]`` are not published and carry a justification here or in
``MODA/docs/sqr/assumptions_register.md``. Published parameters used across
this package: B=2000 support vectors, eta=0.1, P_penalty=2, K=10 (k-NN),
epsilon=0.1, tau in [0.01, 0.05], Matérn kernel. Not published (assumed here):
Matérn nu=1.5, length-scales, T_softmax, pruning rule, tau/M adaptation
equations, k-NN tie-breaks.

Dependencies: NumPy/SciPy only (no torch).
"""

from nsoran.sqr.kernel import MaternKernel
from nsoran.sqr.quantile_regressor import (
    KernelQuantileRegressor,
    QuantileUpdate,
    SupportInsertion,
)
from nsoran.sqr.sqr_controller import (
    SqrDecision,
    SqrSliceController,
)
from nsoran.sqr.coordinator import (
    DEFAULT_VIRTUAL_ALPHA,
    ProjectionResult,
    VirtualUpdate,
    apply_virtual_updates,
    project_prbs,
)

__all__ = [
    "MaternKernel",
    "KernelQuantileRegressor",
    "QuantileUpdate",
    "SupportInsertion",
    "SqrDecision",
    "SqrSliceController",
    "DEFAULT_VIRTUAL_ALPHA",
    "ProjectionResult",
    "VirtualUpdate",
    "apply_virtual_updates",
    "project_prbs",
]
