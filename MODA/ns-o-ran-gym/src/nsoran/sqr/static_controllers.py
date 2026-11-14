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

``PropDemandController`` (Rodada 2, Etapa 2.2 sanity method "prop") is the
DEMAND-PROPORTIONAL sanity baseline: at every window it reads the parsed
KPM window (``SqrKpmWindow.rows``, the same ``sqr_kpis`` parse that builds
the observation — per-UE ``windowThroughputMbps``), computes per-slice
demand ``d_i = Σ_ue min(offered, target_i)`` against the frozen
Rodada-2 service targets (R36/R39), and splits C as

    quota_i = floor_i + (C − Σ_j floor_j) · max(0, d_i − floor_i·cap)
                                           / Σ_j max(0, d_j − floor_j·cap)

with service floors (R39 [ASSUMIDO], v3 calibration): URLLC 25% of C (the
quota size is proven NOT to drive its lowload deadline tail — see the
constant's comment; the floor protects the slice under load), eMBB 5% of
C, mMTC 5% of C.  A slice
with no UE has d_i = 0, zero excess and therefore decays to its floor,
freeing the remainder to the slices with demand.  The float quotas are
rounded to deterministic INTEGERS by largest remainder (ties by lowest
slice index, the coordinator's R24 style) with Σa ≤ C; the same
``StaticDecision`` contract of "equal" feeds the shared run loop (mode
column "prop", learned = 0 — no learning state).  Like "equal", it is a
CONSUMER of the obs/action contract, follows the dynamic-C sync (R01) and
    round-trips through the checkpoint file; it is NOT a learning method and
    NOT part of the SQR article — it is the Etapa 2.2 sanity control.

    The per-UE demand term uses the KPM-derived OFFERED measure of
    :meth:`demand_vector` (backlogged UE ⇒ offers its service target; empty
    queue ⇒ offers what it was served) — NOT the served rate alone, which
    would pin the controller on the floors (see the method's docstring).
    """

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from nsoran.sqr.sqr_kpis import SqrKpmWindow

#: Decision "mode" logged in step_metrics.csv for the static method
#: (single constant value — no SQR/KBRL decision modes exist here).
MODE_EQUAL = "equal"

#: Checkpoint payload version of the static controller (own lineage).
STATIC_STATE_VERSION = 1

# ---------------------------------------------------------------------------
# "prop" (Rodada 2, Etapa 2.2): demand-proportional sanity controller.
# Frozen parameters below are REGISTER R39 [ASSUMIDO] — see
# MODA/docs/sqr/assumptions_register.md; none is published by the article.
# ---------------------------------------------------------------------------

#: Decision "mode" logged in step_metrics.csv for "prop".
MODE_PROP = "prop"

#: Checkpoint payload version of the prop controller (own lineage).
PROP_STATE_VERSION = 1

#: Demand-service target per slice (Mbps per UE), cap of the per-UE demand
#: term min(offered, target).  Slice layout is the canonical sqr-sim one
#: (0 = eMBB/greedy, 1 = URLLC, 2 = mMTC — contract v1.4 §4.2 / R35):
#:   0: 5.25 Mbps/UE = 1.5 × the 3.5 Mbps/UE operational eMBB SLA (R36 as
#:      amended) — the 1.5× MARGIN [ASSUMIDO R39, emenda de calibração v2]
#:      absorbs MCS/scheduling fluctuation: capping the demand AT the SLA
#:      delivered a median ≈ the SLA with ~half the windows below it (gate
#:      v1 of the sanity: eMBB violation 8.6% s1 / 0.11% s2 / 37.2% s3 with
#:      N = 2–5 UE clusters); the SLA of judgment remains 3.5 (R36);
#:   1: 1.0 Mbps/UE-equivalent service weight [ASSUMIDO R39] — far above the
#:      URLLC offered rate (32 B @ 1 ms = 0.256 Mbps/UE, R35), so the cap
#:      never binds and demand = the actual offered traffic;
#:   2: 0.05 Mbps/UE-equivalent service weight [ASSUMIDO R39] — likewise far
#:      above the mMTC offered rate (200 B @ 1 s = 0.0016 Mbps/UE, R35).
PROP_EMBB_SLA_MBPS = 3.5            # R36 operational SLA (judgment target)
PROP_EMBB_DEMAND_MARGIN = 1.5       # R39 amendment (calibration v2) [ASSUMIDO]
PROP_SLA_TARGETS: dict[int, float] = {
    0: PROP_EMBB_SLA_MBPS * PROP_EMBB_DEMAND_MARGIN,  # 5.25 Mbps/UE
    1: 1.0,
    2: 0.05,
}

#: Per-slice service floor as a FRACTION of C [ASSUMIDO R39, v3 calibration]:
#: URLLC 25% of C.  The v2-calibration value (30%) was REFUTED by the paired
#: gate experiment (same seeds/channels, only the policy changed): the URLLC
#: deadline tail is NOT quota-bound at low load — giving the slice MORE quota
#: (+1 RBG/slot) made the miss WORSE (15.0% -> 29.4% on seed 2) while its
#: queues stayed empty (median 69 B, ~17% of the budget served) — the tail is
#: scheduler-level per-UE redistribution (PF/RBG placement over the TDD
#: pattern), not starvation.  30% therefore bought nothing for URLLC and cost
#: eMBB 3 PRBs of ceiling (34 vs 37 of 53), worsening the 4-UE cluster
#: violations (38.1% vs 37.2% on seed 3).  25% still covers the URLLC need
#: under load (5x its realized lowload usage); eMBB 5% and mMTC 5% remain
#: anti-starvation floors.  Σ floors = 35% of C; the remaining 65% is the
#: demand-proportional pool.
PROP_FLOOR_FRACS: dict[int, float] = {0: 0.05, 1: 0.25, 2: 0.05}

#: Conservative realized capacity per RBG over a 100 ms window, in Mbps
#: [ASSUMIDO R39]: converts the floor (PRBs) into Mbps in the excess term
#: max(0, d_i − floor_i·cap).  Order of magnitude: cell peak ~24 Mbps @10 MHz
#: (R04 arithmetic) over 13 RBGs ≈ 1.8 Mbps/RBG; 0.55 ≈ 30% of that,
#: discounting partial DL duty (TDD 3DL+1S/7, R03) and mid MCS.  A measured
#: per-RBG capacity (prb windows telemetry) can replace it at runtime via
#: :meth:`PropDemandController.set_capacity_per_rbg` / the provider hook.
PROP_CAPACITY_PER_RBG_MBPS = 0.55


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


def _largest_remainder_round(quotas: np.ndarray, total: int) -> np.ndarray:
    """Round float quotas to int64 summing to ``total`` (<= len(quotas) units
    of rounding), largest fractional remainder first, ties by LOWEST index —
    the coordinator's deterministic R24 style applied to rounding instead of
    cutting (quotas already sum <= C by construction)."""
    base = np.floor(np.asarray(quotas, dtype=np.float64)).astype(np.int64)
    leftover = int(total) - int(base.sum())
    if leftover <= 0:
        return base
    frac = np.asarray(quotas, dtype=np.float64) - base.astype(np.float64)
    order = np.lexsort((np.arange(base.shape[0]), -frac))
    out = base.copy()
    for k in range(leftover):  # leftover <= L by construction; guard modulo
        out[int(order[k % base.shape[0]])] += 1
    return out


class PropDemandController:
    """Demand-proportional sanity baseline (Rodada 2 Etapa 2.2, method "prop").

    At every window the controller reads the KPM window that JUST CLOSED —
    via ``window_provider`` (wired by ``examples/sqr_run.py`` to the env's
    ``SqrObservationBuilder.last_window``, i.e. the SAME ``sqr_kpis`` parse
    that produced the observation z) — and computes, per slice,

        demand_i  = Σ_ue min(offered_ue, target_i)               [Mbps]
        excess_i  = max(0, demand_i − floor_i · capacity_per_rbg) [Mbps]
        quota_i   = floor_i + (C − Σ_j floor_j) · excess_i / Σ_j excess_j

    then rounds the quotas to deterministic integers summing to
    min(C, round(Σquota)) (largest remainder, ties by lowest slice index).

    Parameters mirror the frozen registers: ``sla_targets`` defaults to
    ``PROP_SLA_TARGETS`` (R36×margin / R39), ``floor_fracs`` to
    ``PROP_FLOOR_FRACS`` (R39 [ASSUMIDO], v3 calibration: URLLC 25% /
    eMBB 5% / mMTC 5% of C) and
    ``capacity_per_rbg`` to ``PROP_CAPACITY_PER_RBG_MBPS`` (0.55 Mbps/RBG,
    R39 [ASSUMIDO]).  A measured per-RBG capacity from the prb-windows
    telemetry can be pushed in at runtime via
    :meth:`set_capacity_per_rbg`; when no provider/window exists (mocks,
    pre-first-window) the demand is zero and every slice sits at its floor.

    Same consumer contract as :class:`EqualSplitController`: ``act`` returns
    a :class:`StaticDecision` (mode ``"prop"``), ``observe`` is a no-op
    (``LEARNING = False`` — the run loop logs learned = 0), ``set_n_prbs``
    follows the dynamic C (R01) and the checkpoint round-trips.
    """

    LEARNING = False

    def __init__(
        self,
        slice_id: int,
        n_prbs: int,
        num_slices: int = 3,
        sla_targets: dict[int, float] | None = None,
        floor_fracs: dict[int, float] | None = None,
        capacity_per_rbg: float = PROP_CAPACITY_PER_RBG_MBPS,
        window_provider: Callable[[], SqrKpmWindow | None] | None = None,
    ):
        if int(slice_id) not in range(int(num_slices)):
            raise ValueError(
                f"slice_id {slice_id} out of range for num_slices={num_slices}"
            )
        if int(n_prbs) <= 0:
            raise ValueError(f"n_prbs must be positive, got {n_prbs}")
        floors = dict(
            floor_fracs if floor_fracs is not None else PROP_FLOOR_FRACS
        )
        for sid in range(int(num_slices)):
            f = floors.get(sid)
            if f is None or not np.isfinite(f) or f < 0.0:
                raise ValueError(
                    f"floor_fracs must define a finite fraction >= 0 for "
                    f"every slice; missing/invalid for slice {sid}"
                )
        if sum(floors.get(sid, 0.0) for sid in range(int(num_slices))) > 1.0:
            raise ValueError(
                "sum of floor fractions exceeds 1.0 (floors would exceed C)"
            )
        cap = float(capacity_per_rbg)
        if not np.isfinite(cap) or cap <= 0.0:
            raise ValueError(f"capacity_per_rbg must be positive, got {cap}")
        self.slice_id = int(slice_id)
        self.n_prbs = int(n_prbs)
        self.num_slices = int(num_slices)
        self.sla_targets: dict[int, float] = dict(
            sla_targets if sla_targets is not None else PROP_SLA_TARGETS
        )
        self.floor_fracs: dict[int, float] = floors
        self.capacity_per_rbg = cap
        self._window_provider = window_provider

    # ------------------------------------------------------------------
    # Demand -> quotas -> deterministic integer split
    # ------------------------------------------------------------------

    def demand_vector(self, window: SqrKpmWindow | None) -> np.ndarray:
        """Per-slice demand d_i (Mbps): Σ_ue min(offered, target_i); 0.0
        for a slice with no UE in the window (quota decays to its floor).

        The per-UE "offered" measure is KPM-derived (contract §4): a UE with
        queue backlog (``queueBytes`` > 0, col. 13) is offering at least its
        service target — for the greedy eMBB profile the application is
        full-buffer (R04: 100 Mbps offered, fila sempre não vazia), so its
        demand saturates at the target; a UE with an empty queue is
        bandwidth-satisfied and offers what it was served
        (``windowThroughputMbps``, col. 11).  Using the SERVED rate alone
        would self-train the controller on the floor (a 5%-of-C floor serves
        ~1.4 Mbps < the 1.46 Mbps the floor is credited for — the quota
        could never grow), defeating demand proportionality.
        """
        d = np.zeros(self.num_slices, dtype=np.float64)
        if window is None:
            return d
        for r in window.rows:
            sid = int(getattr(r, "slice_id", -1))
            if sid not in range(self.num_slices):
                continue
            target = float(self.sla_targets.get(sid, 0.0))
            if target <= 0.0:
                continue  # no demand model for this slice
            offered = max(float(getattr(r, "window_throughput_mbps", 0.0)), 0.0)
            if float(getattr(r, "queue_bytes", 0.0)) > 0.0:
                offered = target  # backlogged UE: offers at least the target
            d[sid] += min(offered, target)
        return d

    def quota_vector(self, window: SqrKpmWindow | None) -> np.ndarray:
        """Float quotas (floor + demand-proportional share of the remainder);
        Σ = Σpisos (no demand anywhere) or exactly C (any excess demand)."""
        c = float(self.n_prbs)
        floors = np.array(
            [
                float(self.floor_fracs.get(sid, 0.0)) * c
                for sid in range(self.num_slices)
            ],
            dtype=np.float64,
        )
        demand = self.demand_vector(window)
        capacity = float(self.capacity_per_rbg)
        excess = np.maximum(0.0, demand - floors * capacity)
        total_excess = float(excess.sum())
        quota = floors.copy()
        remainder = c - float(floors.sum())
        if total_excess > 0.0 and remainder > 0.0:
            quota += remainder * excess / total_excess
        return quota

    def prop_split(self, window: SqrKpmWindow | None) -> np.ndarray:
        """Deterministic integer split (L,) int64 with Σ <= C."""
        quota = self.quota_vector(window)
        total = min(int(self.n_prbs), int(round(float(quota.sum()))))
        return _largest_remainder_round(quota, total)

    # ------------------------------------------------------------------
    # Controller interface (same shape the run loop expects)
    # ------------------------------------------------------------------

    def act(self, z: np.ndarray) -> StaticDecision:  # noqa: ARG002 — z unused
        """Demand-proportional proposal for this slice's window.

        ``z`` is accepted (and ignored): the demand comes from the parsed
        KPM window (per-UE offered throughputs), not from the normalized
        observation features (which saturate at their R09 scales).
        """
        window = (
            self._window_provider() if self._window_provider is not None else None
        )
        action = int(self.prop_split(window)[self.slice_id])
        return StaticDecision(
            action=action,
            raw_action=action,
            mode=MODE_PROP,
            sigma=float("nan"),
            margin_applied=0,
            tau=float("nan"),
            margin_m=float("nan"),
        )

    # ------------------------------------------------------------------
    # Runtime wiring (dynamic C, measured capacity, KPM window source)
    # ------------------------------------------------------------------

    def set_n_prbs(self, n_prbs: int) -> None:
        """Follow the dynamic C (register R01 as amended)."""
        if int(n_prbs) <= 0:
            raise ValueError(f"n_prbs must be positive, got {n_prbs}")
        self.n_prbs = int(n_prbs)

    def set_capacity_per_rbg(self, capacity_mbps: float) -> None:
        """Replace the conservative R39 constant by a MEASURED per-RBG
        capacity (e.g. from the prb-windows telemetry)."""
        cap = float(capacity_mbps)
        if not np.isfinite(cap) or cap <= 0.0:
            raise ValueError(f"capacity must be positive, got {capacity_mbps}")
        self.capacity_per_rbg = cap

    def attach_window_provider(
        self, provider: Callable[[], SqrKpmWindow | None] | None
    ) -> None:
        """Wire the KPM-window source (a callable returning the latest
        :class:`SqrKpmWindow` or None).  ``None`` detaches (floor-only)."""
        self._window_provider = provider

    def observe(self, z: np.ndarray, action: int, y: float) -> None:
        """No learning state (static sanity baseline) — documented no-op."""
        return None

    # ------------------------------------------------------------------
    # Checkpoint round-trip (same file format conventions as "equal")
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "kind": "prop_demand",
            "state_version": PROP_STATE_VERSION,
            "slice_id": self.slice_id,
            "n_prbs": self.n_prbs,
            "num_slices": self.num_slices,
            "sla_targets": {str(k): v for k, v in self.sla_targets.items()},
            "floor_fracs": {str(k): v for k, v in self.floor_fracs.items()},
            "capacity_per_rbg": self.capacity_per_rbg,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("kind") != "prop_demand":
            raise ValueError(
                f"not a prop_demand state (kind={state.get('kind')!r})"
            )
        if int(state.get("state_version", -1)) != PROP_STATE_VERSION:
            raise ValueError(
                f"unsupported prop state version {state.get('state_version')!r}"
            )
        self.slice_id = int(state["slice_id"])
        self.n_prbs = int(state["n_prbs"])
        self.num_slices = int(state["num_slices"])
        self.sla_targets = {
            int(k): float(v) for k, v in state.get("sla_targets", {}).items()
        }
        self.floor_fracs = {
            int(k): float(v) for k, v in state.get("floor_fracs", {}).items()
        }
        self.capacity_per_rbg = float(state["capacity_per_rbg"])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PropDemandController(slice_id={self.slice_id}, "
            f"n_prbs={self.n_prbs}, num_slices={self.num_slices}, "
            f"capacity_per_rbg={self.capacity_per_rbg})"
        )
