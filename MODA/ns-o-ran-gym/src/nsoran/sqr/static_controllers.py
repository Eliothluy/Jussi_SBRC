"""Static baselines for the SQR campaign (Fase 4, ask item 4).

``EqualSplitController`` is the "equal" campaign method: a TRIVIAL Python
controller that always proposes the fixed equal split of the cell capacity
C across the slices (33.33%/slice, ``sqr_dummy_controller.py`` phase-C
pattern ``33.3333/33.3333/33.3334`` rendered as deterministic INTEGERS on
the SQR action space ``a_i in {0..C}``): ``base = C // L`` to every slice
and the remainder ``C % L`` to the first ``C % L`` slices (largest-remainder
tie-break by slice index, same deterministic style of the coordinator's
projection).  For C = 53, L = 3 -> proposals (18, 18, 17), sum = C.

It is a consumer of the SAME obs/action contract as SQR/KBRL
(:meth:`act` receives z and returns a decision dataclass with the SAME
field names the shared run loop logs in ``step_metrics.csv``), it follows
the dynamic-C sync (``set_n_prbs``, register R01 as amended) and it
round-trips through the checkpoint file (``state_dict``), but it has NO
learning state: :meth:`observe` is a documented no-op and the R12 virtual
update is skipped for it (the run loop checks ``insert_virtual_support``
on every controller before applying any).

Static by construction: identical proposals for the same C, independent of
the observation — the sanity floor the campaign compares the adaptive
methods against (PLANO §3-Fase 3 "estáticos de sanidade").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Decision "mode" logged in step_metrics.csv for the static method
#: (single constant value — no SQR/KBRL decision modes exist here).
MODE_EQUAL = "equal"

#: Checkpoint payload version of the static controller (own lineage).
STATIC_STATE_VERSION = 1


@dataclass(frozen=True)
class StaticDecision:
    """One decision of a static controller.

    Field names mirror :class:`nsoran.sqr.sqr_controller.SqrDecision` and
    :class:`nsoran.sqr.kbrl.kbrl_controller.KbrlDecision` so the shared
    run loop and step_metrics.csv keep the SAME columns for every method;
    fields without a static analogue carry NaN (documented, same
    convention as the KBRL decision).
    """

    action: int          # fixed equal-split proposal, in [0, C]
    raw_action: int      # == action (no quantile machinery behind it)
    mode: str            # "equal" (constant)
    sigma: float         # NaN (no SQR quantile analogue)
    margin_applied: int  # 0 (no margin machinery)
    tau: float           # NaN (no SQR tau analogue)
    margin_m: float      # NaN (no SQR/KBRL margin analogue)


class EqualSplitController:
    """Fixed equal split of C across ``num_slices`` (campaign method "equal").

    Parameters
    ----------
    slice_id:
        Index of this slice (the split is a per-slice controller like the
        SQR/KBRL ones; ``slice_id`` determines who receives the remainder).
    n_prbs:
        C, total PRBs of the cell (initial prior 52; synced to the
        effective telemetry C by :meth:`set_n_prbs` — R01 as amended).
    num_slices:
        L, number of slices sharing the capacity (default 3).

    Class attribute ``LEARNING = False`` tells the shared run loop that
    ``observe`` carries no update — the ``learned`` column of
    step_metrics.csv stays 0 for this method (honest diagnostics: the
    static baseline never learns).
    """

    LEARNING = False

    def __init__(self, slice_id: int, n_prbs: int, num_slices: int = 3):
        if int(slice_id) not in range(int(num_slices)):
            raise ValueError(
                f"slice_id {slice_id} out of range for num_slices={num_slices}"
            )
        if int(n_prbs) <= 0:
            raise ValueError(f"n_prbs must be positive, got {n_prbs}")
        self.slice_id = int(slice_id)
        self.n_prbs = int(n_prbs)
        self.num_slices = int(num_slices)

    # ------------------------------------------------------------------
    # Equal split (deterministic integers, sum == C)
    # ------------------------------------------------------------------

    def equal_split(self) -> np.ndarray:
        """The fixed split (L,) int64: base C//L each, remainder to the
        first C%L slices; sums to exactly C (nothing left idle)."""
        c = int(self.n_prbs)
        base, rem = divmod(c, int(self.num_slices))
        split = np.full(int(self.num_slices), base, dtype=np.int64)
        split[:rem] += 1
        return split

    # ------------------------------------------------------------------
    # Controller interface (same shape the run loop expects)
    # ------------------------------------------------------------------

    def act(self, z: np.ndarray) -> StaticDecision:  # noqa: ARG002 — z unused
        """Fixed proposal, independent of the observation (static method).

        ``z`` is accepted (and ignored) so the shared run loop
        ``controllers[sid].act(states[sid])`` calls this controller exactly
        like the SQR/KBRL ones.
        """
        action = int(self.equal_split()[self.slice_id])
        return StaticDecision(
            action=action,
            raw_action=action,
            mode=MODE_EQUAL,
            sigma=float("nan"),
            margin_applied=0,
            tau=float("nan"),
            margin_m=float("nan"),
        )

    def set_n_prbs(self, n_prbs: int) -> None:
        """Follow the dynamic C (register R01 as amended): the split
        recomputes on the next :meth:`act` with the new capacity."""
        if int(n_prbs) <= 0:
            raise ValueError(f"n_prbs must be positive, got {n_prbs}")
        self.n_prbs = int(n_prbs)

    def observe(self, z: np.ndarray, action: int, y: float) -> None:
        """No learning state (static baseline) — documented no-op.

        The signature matches ``SqrSliceController.observe`` so the shared
        run loop can call it uniformly.
        """
        return None

    # ------------------------------------------------------------------
    # Checkpoint round-trip (build_checkpoint calls state_dict on every
    # controller; the payload is trivial but keeps the file format valid)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "kind": "equal_split",
            "state_version": STATIC_STATE_VERSION,
            "slice_id": self.slice_id,
            "n_prbs": self.n_prbs,
            "num_slices": self.num_slices,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("kind") != "equal_split":
            raise ValueError(
                f"not an equal_split state (kind={state.get('kind')!r})"
            )
        if int(state.get("state_version", -1)) != STATIC_STATE_VERSION:
            raise ValueError(
                f"unsupported static state version {state.get('state_version')!r}"
            )
        self.slice_id = int(state["slice_id"])
        self.n_prbs = int(state["n_prbs"])
        self.num_slices = int(state["num_slices"])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"EqualSplitController(slice_id={self.slice_id}, "
            f"n_prbs={self.n_prbs}, num_slices={self.num_slices})"
        )
