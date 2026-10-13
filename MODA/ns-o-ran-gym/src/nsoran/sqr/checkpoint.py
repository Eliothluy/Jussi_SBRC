"""File round-trip for SQR controller checkpoints (plan §3-Fase 2).

The checkpoint payload (per slice) is exactly what
:meth:`nsoran.sqr.sqr_controller.SqrSliceController.state_dict` produces:
support matrix, alphas, hyperparameters, tau, M, the violation window,
the experience memory (successes AND non-successes) and the RNG state —
so a resumed run reproduces the same decision sequence.

Pickle is used on purpose: the payload carries NumPy arrays and the NumPy
bit-generator state, and the SQR package has no torch/JSON-schema
dependency.  Checkpoints are only ever written next to the run's own
artifacts by ``examples/sqr_run.py``.
"""

from __future__ import annotations

import os
import pickle

CHECKPOINT_FORMAT = "sqr-checkpoint"
CHECKPOINT_VERSION = 2


def build_checkpoint(
    controllers: dict[int, "SqrSliceController"],  # noqa: F821
    meta: dict | None = None,
) -> dict:
    """Assemble the payload for a dict of per-slice controllers."""
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "meta": dict(meta or {}),
        "slices": {int(sid): ctl.state_dict() for sid, ctl in controllers.items()},
    }


def save_checkpoint(path: str, payload: dict) -> str:
    """Atomically write the checkpoint (tmp file + rename)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)
    return path


def load_checkpoint(path: str) -> dict:
    """Read a checkpoint written by :func:`save_checkpoint`."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path}: not an SQR checkpoint (format={payload.get('format')!r})"
        )
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"{path}: unsupported checkpoint version {payload.get('version')!r} "
            f"(expected {CHECKPOINT_VERSION})"
        )
    return payload


def restore_controllers(
    payload: dict,
    controllers: dict[int, "SqrSliceController"],  # noqa: F821
) -> dict:
    """Load each slice's state into a pre-built controller (same topology).

    The topology is validated in BOTH directions (Fase 0 gate rule: fail
    loudly on bad config):

    - a checkpoint slice with no pre-built controller -> error;
    - a pre-built controller with no slice in the checkpoint -> error
      (restoring a partial checkpoint would silently leave that slice's
      regressor/memory/RNG at their fresh-init state).

    Beyond topology, ``SqrSliceController.load_state_dict`` validates the
    payload's ``format_version`` and the structural fields
    (state_dim/n_prbs) and raises explicitly on mismatch.
    Returns the checkpoint meta for logging.
    """
    slices = payload.get("slices", {})
    missing = sorted(set(slices) - set(controllers))
    if missing:
        raise ValueError(f"checkpoint has slices with no controller: {missing}")
    extra = sorted(set(controllers) - set(slices))
    if extra:
        raise ValueError(
            f"controllers with no slice in the checkpoint (topology must "
            f"match in both directions): {extra}"
        )
    for sid, state in slices.items():
        controllers[sid].load_state_dict(state)
    return dict(payload.get("meta", {}))
