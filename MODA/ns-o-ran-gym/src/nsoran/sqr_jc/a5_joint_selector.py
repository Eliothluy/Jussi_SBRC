"""a5_joint_selector.py — Modular joint action selection (Phase A5 / SQR-JC).

Implements the three causal, zero-RNG ablation selectors strictly over the
common feasible PRB/RBG simplex A_common(C) = {a in Z^3, a >= 0 : sum(a) <= C}
using frozen B0-rest (A1+A2a+A_scaler) quantile models:

- ``b1_raw``     : Pure feasible bottleneck optimization over all 3 slices
                   using frozen B0-rest margin predictions f_tau(z_i, a_i) and
                   utilities u_i(a_i) = f_tau(z_i, a_i) - gamma * a_i / C.
- ``b1_active``  : Causal active-demand gating (matches B0-rest's
                   ``is_slice_provably_empty``): fixes a_i = 0 for provably
                   inactive slices and optimizes the bottleneck strictly over
                   the causal active/pending slice set I_act(W).
- ``b1_support`` : Demand- and kernel-support-aware joint selection (SQR-JC):
                   restricts to I_act(W) with non-starvation floors a_i >= 1 on
                   live demand slices, enforces Matérn support trust
                   sigma_i(z_i, a_i) <= sigma_max with LCB score
                   g_i^{supp}(a_i) = f_i(a_i) - beta * sigma_i(a_i), and runs a
                   deterministic 4-tier hierarchy down to safe k-NN /
                   prop_active / equal_active fallback when the trusted
                   envelope has no feasible joint solution.

Design invariants:
- Never modifies historical ``joint_selector.py`` (hash 3d719d7bcbbf3fd6).
- Strictly causal: consumes ONLY pre-decision state S_W, scaled states z_i(W),
  and pre-decision frozen model predictions/uncertainties. Zero RNG, zero
  controller mutation, zero future KPI/label access.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from nsoran.sqr.checkpoint import load_checkpoint, restore_controllers
from nsoran.sqr.replay_controller import initial_equal_split
from nsoran.sqr.sqr_controller import (
    MARGIN_BOUNDS,
    MARGIN_INIT,
    TAU_INIT,
    TAU_STEP,
    SqrSliceController,
)
from nsoran.sqr_jc.feature_scaler import FeatureScaler
from nsoran.sqr_jc.freeze_contract import (
    verify_causal_replay_no_future_checkpoint,
    verify_resume_contract,
)
from nsoran.sqr_jc.joint_state import JointState
from nsoran.sqr_jc.sqr_het import SqrHetController

NUM_SLICES = 3
A5_VARIANTS: tuple[str, ...] = ("b1_raw", "b1_active", "b1_support")

# Calibrated to Matérn nu=1.5, length_scale=1.0 on [z_i(6), a_i/C]:
# at identical state z (d_z=0), moving 22..53 PRBs along a/C yields
# sigma in [0.15, 0.517]; sigma > 0.55 occurs only under true state novelty (OOD).
DEFAULT_SIGMA_MAX = 0.15
DEFAULT_SIGMA_SOFT = 0.55
DEFAULT_BETA_UNCERTAINTY = 1.0
DEFAULT_FALLBACK_POLICY = "knn_or_prop_active"
VALID_FALLBACK_POLICIES: tuple[str, ...] = (
    "knn_or_prop_active",
    "prop_active",
    "equal_active",
)

# Cache vectorized integer simplex grids keyed by (capacity_c, active_mask, floors)
_SIMPLEX_CACHE: dict[
    tuple[int, tuple[bool, bool, bool], tuple[int, int, int]],
    tuple[np.ndarray, np.ndarray, np.ndarray],
] = {}


@dataclass(frozen=True)
class A5SelectorConfig:
    """Immutable hyperparameter configuration for an A5 joint selector."""

    variant: str = "b1_support"
    capacity_c: int = 53
    gamma: float = 1.0
    sigma_max: float = DEFAULT_SIGMA_MAX
    sigma_soft: float = DEFAULT_SIGMA_SOFT
    beta_uncertainty: float = DEFAULT_BETA_UNCERTAINTY
    enforce_active_floor_one: bool = True
    fallback_policy: str = DEFAULT_FALLBACK_POLICY

    def __post_init__(self) -> None:
        if self.variant not in A5_VARIANTS:
            raise ValueError(f"variant must be one of {A5_VARIANTS}, got {self.variant!r}")
        if int(self.capacity_c) < 0:
            raise ValueError(f"capacity_c must be >= 0, got {self.capacity_c}")
        if not (0.0 <= float(self.sigma_max) <= float(self.sigma_soft) <= 1.0):
            raise ValueError(
                "must satisfy 0.0 <= sigma_max <= sigma_soft <= 1.0, got "
                f"sigma_max={self.sigma_max}, sigma_soft={self.sigma_soft}"
            )
        if float(self.beta_uncertainty) < 0.0 or not np.isfinite(float(self.beta_uncertainty)):
            raise ValueError(
                f"beta_uncertainty must be finite >= 0, got {self.beta_uncertainty}"
            )
        if self.fallback_policy not in VALID_FALLBACK_POLICIES:
            raise ValueError(
                f"fallback_policy must be one of {VALID_FALLBACK_POLICIES}, got {self.fallback_policy!r}"
            )


@dataclass(frozen=True)
class A5Decision:
    """Complete auditable decision output of one A5 selector call."""

    action: tuple[int, int, int]
    variant: str
    tier: str
    active_mask: tuple[bool, bool, bool]
    live_floor_mask: tuple[bool, bool, bool]
    critical_score: float
    efficient_score: float
    sum_prbs: int
    n_feasible_candidates: int
    max_chosen_sigma: float
    chosen_sigma: tuple[float, float, float]
    chosen_f: tuple[float, float, float]
    chosen_g: tuple[float, float, float]
    fallback_used: bool
    degenerate_avoided: bool
    decision_time_ms: float


def _extract_state_field(state_or_dict: JointState | Mapping[str, Any], key: str, default: Any = 0) -> Any:
    if isinstance(state_or_dict, Mapping):
        return state_or_dict.get(key, default)
    return getattr(state_or_dict, key, default)


def causal_active_slices(
    state_or_dict: JointState | Mapping[str, Any],
) -> tuple[tuple[bool, bool, bool], tuple[bool, bool, bool]]:
    """Return ``(active_mask, live_floor_mask)`` strictly from pre-decision state.

    Matches ``SqrHetController.is_slice_provably_empty`` so B1-active and
    B1-support align 100% with B0-rest's ``--empty_slice_guard`` semantics:
    a slice is provably empty iff ``n == 0 and queue <= 0 and offered <= 0
    and kpm_tx_packets == 0 and not pending``. ``live_floor_mask[i]`` is
    True when slice ``i`` currently has live UEs (``n_i > 0``) or buffered
    backlog (``queue_i > 0``), meaning ``a_i = 0`` would starve active traffic.
    """
    ns = (
        int(_extract_state_field(state_or_dict, "n_e", 0)),
        int(_extract_state_field(state_or_dict, "n_u", 0)),
        int(_extract_state_field(state_or_dict, "n_m", 0)),
    )
    qs = (
        float(_extract_state_field(state_or_dict, "queue_e", 0.0)),
        float(_extract_state_field(state_or_dict, "queue_u", 0.0)),
        float(_extract_state_field(state_or_dict, "queue_m", 0.0)),
    )
    offs = (
        float(_extract_state_field(state_or_dict, "offered_e", 0.0)),
        float(_extract_state_field(state_or_dict, "offered_u", 0.0)),
        float(_extract_state_field(state_or_dict, "offered_m", 0.0)),
    )
    extras = _extract_state_field(state_or_dict, "extras", None) or {}
    tx_tuple = extras.get("kpm_tx_packets") or (0, 0, 0)
    kpi_m = _extract_state_field(state_or_dict, "kpi_m_raw", None)
    pendings = (
        bool(extras.get("embb_pending_pipeline", False)),
        bool(extras.get("urllc_pool_open", False)),
        (kpi_m is not None) or bool(extras.get("mmtc_rolling_active", False)),
    )

    active = []
    live_floor = []
    for sid in range(NUM_SLICES):
        tx_i = int(tx_tuple[sid]) if sid < len(tx_tuple) else 0
        provably_empty = (
            ns[sid] == 0
            and qs[sid] <= 0.0
            and offs[sid] <= 0.0
            and tx_i <= 0
            and not pendings[sid]
        )
        active.append(not provably_empty)
        live_floor.append((ns[sid] > 0) or (qs[sid] > 0.0))

    return (
        (active[0], active[1], active[2]),
        (live_floor[0], live_floor[1], live_floor[2]),
    )


def _simplex_grid(
    capacity_c: int,
    active_mask: tuple[bool, bool, bool],
    floors: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vectorized candidate arrays ``(I0, I1, I2)`` sorted in ascending
    lexicographic order over the admissible integer simplex subset."""
    key = (int(capacity_c), active_mask, floors)
    cached = _SIMPLEX_CACHE.get(key)
    if cached is not None:
        return cached

    c = int(capacity_c)
    lo = [
        (int(floors[sid]) if active_mask[sid] else 0) for sid in range(NUM_SLICES)
    ]
    hi = [(c if active_mask[sid] else 0) for sid in range(NUM_SLICES)]

    if sum(lo) > c:
        empty = np.zeros(0, dtype=np.int64)
        _SIMPLEX_CACHE[key] = (empty, empty, empty)
        return empty, empty, empty

    r0_list: list[np.ndarray] = []
    r1_list: list[np.ndarray] = []
    r2_list: list[np.ndarray] = []

    for a0 in range(lo[0], hi[0] + 1):
        rem_after_0 = c - a0
        max_a1 = min(hi[1], rem_after_0 - lo[2])
        if max_a1 < lo[1]:
            continue
        for a1 in range(lo[1], max_a1 + 1):
            max_a2 = min(hi[2], rem_after_0 - a1)
            if max_a2 < lo[2]:
                continue
            a2_vec = np.arange(lo[2], max_a2 + 1, dtype=np.int64)
            n = a2_vec.size
            r0_list.append(np.full(n, a0, dtype=np.int64))
            r1_list.append(np.full(n, a1, dtype=np.int64))
            r2_list.append(a2_vec)

    if not r0_list:
        empty = np.zeros(0, dtype=np.int64)
        res = (empty, empty, empty)
    else:
        res = (
            np.concatenate(r0_list),
            np.concatenate(r1_list),
            np.concatenate(r2_list),
        )
    _SIMPLEX_CACHE[key] = res
    return res


def _select_lexicographic(
    g_tbl: np.ndarray,
    u_tbl: np.ndarray,
    scope_sids: Sequence[int],
    i0: np.ndarray,
    i1: np.ndarray,
    i2: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[int, float, float, int] | None:
    """Deterministic lexicographic winner over pre-sorted ``(i0, i1, i2)``.

    Order matches PROTOCOL_FROZEN §5.4:
      1. max critical(a) = min_{i in scope_sids} g_i(a_i)
      2. max efficient(a) = sum_{i in scope_sids} u_i(a_i)
      3. min sum(a) = a0 + a1 + a2
      4. lexicographically smallest (a0, a1, a2) [index 0 in ascending grid order]
    """
    if i0.size == 0:
        return None
    if mask is not None:
        idx_pool = np.flatnonzero(mask)
        if idx_pool.size == 0:
            return None
        s0, s1, s2 = i0[idx_pool], i1[idx_pool], i2[idx_pool]
    else:
        idx_pool = None
        s0, s1, s2 = i0, i1, i2

    idx_arr = (s0, s1, s2)
    crit = np.minimum.reduce([g_tbl[sid, idx_arr[sid]] for sid in scope_sids])
    eff = np.add.reduce([u_tbl[sid, idx_arr[sid]] for sid in scope_sids])
    sums = s0 + s1 + s2

    max_crit = float(np.max(crit))
    cand = np.flatnonzero(crit == max_crit)
    max_eff = float(np.max(eff[cand]))
    cand = cand[eff[cand] == max_eff]
    min_sum = int(np.min(sums[cand]))
    cand = cand[sums[cand] == min_sum]
    local_best = int(cand[0])  # sorted (a0, a1, a2) ensures smallest lex tie-break
    orig_idx = int(idx_pool[local_best]) if idx_pool is not None else local_best
    return orig_idx, max_crit, max_eff, min_sum


def _largest_remainder_allocate(weights: Sequence[float], budget: int) -> list[int]:
    """Deterministic floor + largest-remainder integer split summing to <= budget."""
    budget = int(budget)
    if budget <= 0:
        return [0] * len(weights)
    w = [max(0.0, float(v)) for v in weights]
    w_sum = math.fsum(w)
    if w_sum <= 0.0:
        return [0] * len(w)
    target = min(budget, int(math.floor(math.fsum(v * budget / w_sum for v in w) + 1e-12)))
    raw = [v * budget / w_sum for v in w]
    quotas = [int(math.floor(r)) for r in raw]
    fracs = [r - q for r, q in zip(raw, quotas)]
    extra = target - sum(quotas)
    while extra > 0:
        best = max(range(len(fracs)), key=lambda i: (fracs[i], -i))
        quotas[best] += 1
        fracs[best] = -1.0
        extra -= 1
    return quotas


def bounded_project_active(
    proposals: Sequence[int],
    capacity_c: int,
    active_mask: tuple[bool, bool, bool],
    floors: tuple[int, int, int] = (0, 0, 0),
) -> tuple[int, int, int]:
    """Project integer proposals onto A_act(W, C) while preserving live floors."""
    c = int(capacity_c)
    floored = [
        (max(0, int(floors[sid])) if active_mask[sid] else 0)
        for sid in range(NUM_SLICES)
    ]
    if sum(floored) > c:
        # Degenerate tiny-capacity fallback (e.g., C < active_count): prioritize lower sid
        rem = c
        out = [0, 0, 0]
        for sid in range(NUM_SLICES):
            if active_mask[sid] and rem > 0:
                give = min(rem, floored[sid])
                out[sid] = give
                rem -= give
        return (out[0], out[1], out[2])

    req = [
        (max(floored[sid], int(proposals[sid])) if active_mask[sid] else 0)
        for sid in range(NUM_SLICES)
    ]
    if sum(req) <= c:
        return (req[0], req[1], req[2])

    rem_cap = c - sum(floored)
    excess = [req[sid] - floored[sid] for sid in range(NUM_SLICES)]
    cut_excess = _largest_remainder_allocate(excess, rem_cap)
    return (
        floored[0] + cut_excess[0],
        floored[1] + cut_excess[1],
        floored[2] + cut_excess[2],
    )


def equal_active_split(
    capacity_c: int,
    active_mask: tuple[bool, bool, bool],
) -> tuple[int, int, int]:
    """Deterministic uniform split across active slices (mirrors (18,17,18) for 3@53)."""
    c = int(capacity_c)
    n_act = sum(1 for a in active_mask if a)
    if n_act == 0 or c <= 0:
        return (0, 0, 0)
    if n_act == NUM_SLICES and c >= NUM_SLICES:
        eq = initial_equal_split(c, NUM_SLICES)
        return (int(eq[0]), int(eq[1]), int(eq[2]))
    weights = [1.0 if active_mask[sid] else 0.0 for sid in range(NUM_SLICES)]
    alloc = _largest_remainder_allocate(weights, c)
    return (alloc[0], alloc[1], alloc[2])


def prop_active_split(
    state_or_dict: JointState | Mapping[str, Any],
    capacity_c: int,
    active_mask: tuple[bool, bool, bool],
    floors: tuple[int, int, int] = (0, 0, 0),
) -> tuple[int, int, int]:
    """Deterministic active-demand proportional allocation respecting floors."""
    c = int(capacity_c)
    if not any(active_mask) or c <= 0:
        return (0, 0, 0)
    ns = (
        int(_extract_state_field(state_or_dict, "n_e", 0)),
        int(_extract_state_field(state_or_dict, "n_u", 0)),
        int(_extract_state_field(state_or_dict, "n_m", 0)),
    )
    qs = (
        float(_extract_state_field(state_or_dict, "queue_e", 0.0)),
        float(_extract_state_field(state_or_dict, "queue_u", 0.0)),
        float(_extract_state_field(state_or_dict, "queue_m", 0.0)),
    )
    # Use UE count as primary proportional weight across heterogeneous profiles
    # (backstopped by >0 epsilon when active only by queue/pool) to avoid raw byte scaling bias.
    raw_weights = [
        (float(ns[sid]) if ns[sid] > 0 else (0.5 if qs[sid] > 0.0 else 0.25))
        if active_mask[sid]
        else 0.0
        for sid in range(NUM_SLICES)
    ]
    floored = [
        (max(0, int(floors[sid])) if active_mask[sid] else 0)
        for sid in range(NUM_SLICES)
    ]
    if sum(floored) > c:
        return bounded_project_active(floored, c, active_mask, floors)
    rem_cap = c - sum(floored)
    extra = _largest_remainder_allocate(raw_weights, rem_cap)
    return (
        floored[0] + extra[0],
        floored[1] + extra[1],
        floored[2] + extra[2],
    )


class A5JointSelector:
    """Modular causal joint selector for B1-raw, B1-active, and B1-support."""

    def __init__(self, config: A5SelectorConfig | None = None) -> None:
        self.config = config or A5SelectorConfig()

    def _finalize_decision(
        self,
        *,
        action: tuple[int, int, int],
        tier: str,
        active_mask: tuple[bool, bool, bool],
        live_floor_mask: tuple[bool, bool, bool],
        scope_sids: Sequence[int],
        f_tbl: np.ndarray,
        g_tbl: np.ndarray,
        u_tbl: np.ndarray,
        sigma_tbl: np.ndarray,
        n_feasible_candidates: int,
        fallback_used: bool,
        degenerate_avoided: bool,
        t0_ns: int,
    ) -> A5Decision:
        ae, au, am = int(action[0]), int(action[1]), int(action[2])
        c = self.config.capacity_c
        # Safety invariant assertions
        if not (0 <= ae <= c and 0 <= au <= c and 0 <= am <= c and (ae + au + am) <= c):
            raise AssertionError(f"illegal A5 action {action} for capacity C={c}")
        chosen_f = (float(f_tbl[0, ae]), float(f_tbl[1, au]), float(f_tbl[2, am]))
        chosen_g = (float(g_tbl[0, ae]), float(g_tbl[1, au]), float(g_tbl[2, am]))
        chosen_sigma = (
            float(sigma_tbl[0, ae]),
            float(sigma_tbl[1, au]),
            float(sigma_tbl[2, am]),
        )
        crit = min(chosen_g[sid] for sid in scope_sids) if scope_sids else 0.0
        eff = sum(
            float(u_tbl[sid, (ae, au, am)[sid]]) for sid in scope_sids
        ) if scope_sids else 0.0
        max_sig = max(chosen_sigma[sid] for sid in scope_sids) if scope_sids else 0.0
        dt_ms = (time.perf_counter_ns() - t0_ns) / 1e6
        return A5Decision(
            action=(ae, au, am),
            variant=self.config.variant,
            tier=tier,
            active_mask=active_mask,
            live_floor_mask=live_floor_mask,
            critical_score=float(crit),
            efficient_score=float(eff),
            sum_prbs=ae + au + am,
            n_feasible_candidates=int(n_feasible_candidates),
            max_chosen_sigma=float(max_sig),
            chosen_sigma=chosen_sigma,
            chosen_f=chosen_f,
            chosen_g=chosen_g,
            fallback_used=bool(fallback_used),
            degenerate_avoided=bool(degenerate_avoided),
            decision_time_ms=float(dt_ms),
        )

    def select_from_tables(
        self,
        state_or_dict: JointState | Mapping[str, Any],
        f_tbl: np.ndarray,
        u_tbl: np.ndarray,
        sigma_tbl: np.ndarray,
        *,
        knn_actions: Sequence[int] | None = None,
    ) -> A5Decision:
        """Select a legal joint allocation strictly from pre-decision tables.

        Parameters ``f_tbl``, ``u_tbl``, and ``sigma_tbl`` have shape
        ``(3, C + 1)``, holding predictions, B0-gamma utilities, and Matérn
        uncertainties for each slice and RBG count ``0..C``.
        """
        t0_ns = time.perf_counter_ns()
        c = int(self.config.capacity_c)
        f_arr = np.asarray(f_tbl, dtype=np.float64)[:, : c + 1]
        u_arr = np.asarray(u_tbl, dtype=np.float64)[:, : c + 1]
        sig_arr = np.asarray(sigma_tbl, dtype=np.float64)[:, : c + 1]
        if f_arr.shape != (NUM_SLICES, c + 1) or u_arr.shape != (NUM_SLICES, c + 1) or sig_arr.shape != (NUM_SLICES, c + 1):
            raise ValueError(
                f"expected candidate tables of shape (3, {c + 1}), got "
                f"f={f_arr.shape}, u={u_arr.shape}, sigma={sig_arr.shape}"
            )
        if not (np.all(np.isfinite(f_arr)) and np.all(np.isfinite(u_arr)) and np.all(np.isfinite(sig_arr))):
            raise ValueError("candidate tables must be finite")

        active_mask, live_floor_mask = causal_active_slices(state_or_dict)
        active_sids = tuple(sid for sid in range(NUM_SLICES) if active_mask[sid])
        variant = self.config.variant

        # ---- Variant 1: B1-raw (unfiltered A_common over all 3 slices) ----
        if variant == "b1_raw":
            i0, i1, i2 = _simplex_grid(c, (True, True, True), (0, 0, 0))
            winner = _select_lexicographic(f_arr, u_arr, (0, 1, 2), i0, i1, i2)
            assert winner is not None
            idx, _, _, _ = winner
            return self._finalize_decision(
                action=(int(i0[idx]), int(i1[idx]), int(i2[idx])),
                tier="joint_raw",
                active_mask=active_mask,
                live_floor_mask=live_floor_mask,
                scope_sids=(0, 1, 2),
                f_tbl=f_arr,
                g_tbl=f_arr,
                u_tbl=u_arr,
                sigma_tbl=sig_arr,
                n_feasible_candidates=int(i0.size),
                fallback_used=False,
                degenerate_avoided=False,
                t0_ns=t0_ns,
            )

        # ---- Shared zero-demand guard for B1-active & B1-support -----------
        if not active_sids:
            return self._finalize_decision(
                action=(0, 0, 0),
                tier="all_empty",
                active_mask=active_mask,
                live_floor_mask=live_floor_mask,
                scope_sids=(),
                f_tbl=f_arr,
                g_tbl=f_arr,
                u_tbl=u_arr,
                sigma_tbl=sig_arr,
                n_feasible_candidates=1,
                fallback_used=False,
                degenerate_avoided=False,
                t0_ns=t0_ns,
            )

        # ---- Variant 2: B1-active (active subspace ONLY, no sigma filter) --
        if variant == "b1_active":
            i0, i1, i2 = _simplex_grid(c, active_mask, (0, 0, 0))
            winner = _select_lexicographic(f_arr, u_arr, active_sids, i0, i1, i2)
            assert winner is not None
            idx, _, _, _ = winner
            return self._finalize_decision(
                action=(int(i0[idx]), int(i1[idx]), int(i2[idx])),
                tier="joint_active",
                active_mask=active_mask,
                live_floor_mask=live_floor_mask,
                scope_sids=active_sids,
                f_tbl=f_arr,
                g_tbl=f_arr,
                u_tbl=u_arr,
                sigma_tbl=sig_arr,
                n_feasible_candidates=int(i0.size),
                fallback_used=False,
                degenerate_avoided=False,
                t0_ns=t0_ns,
            )

        # ---- Variant 3: B1-support (active subspace + LCB + 4-tier safety) --
        beta = float(self.config.beta_uncertainty)
        g_supp = f_arr - beta * sig_arr
        actions_grid = np.arange(c + 1, dtype=np.float64) / float(max(1, c))
        u_supp = g_supp - float(self.config.gamma) * actions_grid[None, :]

        n_live = sum(1 for lf in live_floor_mask if lf)
        if self.config.enforce_active_floor_one and n_live <= c:
            floors = (
                1 if live_floor_mask[0] else 0,
                1 if live_floor_mask[1] else 0,
                1 if live_floor_mask[2] else 0,
            )
        else:
            floors = (0, 0, 0)

        i0, i1, i2 = _simplex_grid(c, active_mask, floors)
        idx_arr = (i0, i1, i2)
        max_sig_vec = np.maximum.reduce([sig_arr[sid, idx_arr[sid]] for sid in active_sids])

        # Tier 1: joint trusted support envelope (all active slices sigma <= sigma_max)
        mask_t1 = max_sig_vec <= float(self.config.sigma_max)
        n_t1 = int(np.count_nonzero(mask_t1))
        if n_t1 > 0:
            winner = _select_lexicographic(g_supp, u_supp, active_sids, i0, i1, i2, mask=mask_t1)
            assert winner is not None
            idx, _, _, _ = winner
            chosen = (int(i0[idx]), int(i1[idx]), int(i2[idx]))
            # Track if non-starvation floor or sigma filter prevented degenerate (0,0,0)
            degen_avoided = any(chosen[s] > 0 for s in active_sids)
            return self._finalize_decision(
                action=chosen,
                tier="joint_support",
                active_mask=active_mask,
                live_floor_mask=live_floor_mask,
                scope_sids=active_sids,
                f_tbl=f_arr,
                g_tbl=g_supp,
                u_tbl=u_supp,
                sigma_tbl=sig_arr,
                n_feasible_candidates=n_t1,
                fallback_used=False,
                degenerate_avoided=degen_avoided,
                t0_ns=t0_ns,
            )

        # Tier 2: relaxed support envelope (all active slices sigma <= sigma_soft)
        mask_t2 = max_sig_vec <= float(self.config.sigma_soft)
        n_t2 = int(np.count_nonzero(mask_t2))
        if n_t2 > 0:
            winner = _select_lexicographic(g_supp, u_supp, active_sids, i0, i1, i2, mask=mask_t2)
            assert winner is not None
            idx, _, _, _ = winner
            chosen = (int(i0[idx]), int(i1[idx]), int(i2[idx]))
            return self._finalize_decision(
                action=chosen,
                tier="joint_lcb_relaxed",
                active_mask=active_mask,
                live_floor_mask=live_floor_mask,
                scope_sids=active_sids,
                f_tbl=f_arr,
                g_tbl=g_supp,
                u_tbl=u_supp,
                sigma_tbl=sig_arr,
                n_feasible_candidates=n_t2,
                fallback_used=True,
                degenerate_avoided=True,
                t0_ns=t0_ns,
            )

        # Tier 3: OOD fallback — kernel lacks even relaxed support on >=1 active slice.
        # Delegate deterministically (zero RNG) avoiding blind extrapolation -> 0 or 53.
        fb_policy = self.config.fallback_policy
        chosen: tuple[int, int, int]
        tier_name: str
        if fb_policy == "equal_active":
            chosen = equal_active_split(c, active_mask)
            # Ensure live floors even on equal split if feasible
            chosen = bounded_project_active(chosen, c, active_mask, floors)
            tier_name = "fallback_equal_active"
        elif (
            fb_policy == "knn_or_prop_active"
            and knn_actions is not None
            and len(knn_actions) == NUM_SLICES
            and any(int(knn_actions[s]) > 0 for s in active_sids)
            and all(int(knn_actions[s]) > 0 for s in active_sids if live_floor_mask[s])
        ):
            raw_knn = [max(floors[s], int(knn_actions[s])) if active_mask[s] else 0 for s in range(NUM_SLICES)]
            chosen = bounded_project_active(raw_knn, c, active_mask, floors)
            tier_name = "fallback_knn"
        else:
            chosen = prop_active_split(state_or_dict, c, active_mask, floors)
            tier_name = "fallback_prop_active"

        return self._finalize_decision(
            action=chosen,
            tier=tier_name,
            active_mask=active_mask,
            live_floor_mask=live_floor_mask,
            scope_sids=active_sids,
            f_tbl=f_arr,
            g_tbl=g_supp,
            u_tbl=u_supp,
            sigma_tbl=sig_arr,
            n_feasible_candidates=0,
            fallback_used=True,
            degenerate_avoided=True,
            t0_ns=t0_ns,
        )

    def select_from_het(self, het: SqrHetController, state: JointState) -> A5Decision:
        """Read-only decision directly from a (frozen) B0-rest ``SqrHetController``."""
        t0_e2e_ns = time.perf_counter_ns()
        if int(het.c) != int(self.config.capacity_c):
            raise ValueError(
                f"capacity mismatch: selector C={self.config.capacity_c} vs het.c={het.c}"
            )
        c = int(self.config.capacity_c)
        f_tbl = np.zeros((NUM_SLICES, c + 1), dtype=np.float64)
        u_tbl = np.zeros((NUM_SLICES, c + 1), dtype=np.float64)
        sig_tbl = np.zeros((NUM_SLICES, c + 1), dtype=np.float64)
        knn_actions: list[int] = []
        for sid in range(NUM_SLICES):
            z = het.slice_state(state, sid)
            sc = het.controllers[sid].score_candidates(z)  # pure: zero RNG
            knn_a, _ = het.controllers[sid].k_nn_action(z)  # pure: zero RNG
            f_tbl[sid] = sc["f"][: c + 1]
            u_tbl[sid] = sc["scores"][: c + 1]
            sig_tbl[sid] = sc["sigma"][: c + 1]
            knn_actions.append(int(knn_a))
        dec = self.select_from_tables(
            state, f_tbl, u_tbl, sig_tbl, knn_actions=knn_actions
        )
        total_ms = (time.perf_counter_ns() - t0_e2e_ns) / 1e6
        return replace(dec, decision_time_ms=float(total_ms))

    def select_from_trace_record(self, trace_record: Mapping[str, Any]) -> A5Decision:
        """Offline strictly causal selection from ONE ``CausalTraceWriter`` snapshot."""
        if trace_record.get("record_type") != "causal_window":
            raise ValueError(
                f"expected record_type='causal_window', got {trace_record.get('record_type')!r}"
            )
        cap = int(trace_record["capacity_C"])
        if cap != int(self.config.capacity_c):
            raise ValueError(
                f"trace capacity mismatch: snapshot C={cap} vs selector C={self.config.capacity_c}"
            )
        per_slice = sorted(trace_record["per_slice_model"], key=lambda e: int(e["slice_id"]))
        if [int(e["slice_id"]) for e in per_slice] != [0, 1, 2]:
            raise ValueError("trace snapshot must carry slice_ids [0, 1, 2]")
        f_tbl = np.asarray([e["f_candidates"] for e in per_slice], dtype=np.float64)
        u_tbl = np.asarray([e["scores_candidates"] for e in per_slice], dtype=np.float64)
        sig_tbl = np.asarray([e["sigma_candidates"] for e in per_slice], dtype=np.float64)
        knn_actions = (
            [int(e["knn_action_raw"]) for e in per_slice]
            if all("knn_action_raw" in e for e in per_slice)
            else None
        )
        return self.select_from_tables(
            trace_record["state"], f_tbl, u_tbl, sig_tbl, knn_actions=knn_actions
        )


def load_b0_rest_het_from_checkpoint(
    checkpoint_path: str,
    *,
    scaler_manifest_path: str,
    expected_meta: Mapping[str, Any] | None = None,
    target_window_index: int | None = None,
    target_sim_time_ms: int | None = None,
) -> tuple[dict[int, SqrSliceController], SqrHetController, dict[str, Any]]:
    """Strictly reconstruct frozen B0-rest controllers from a verified checkpoint.

    Fixes historical ``joint_selector.load_run_controllers`` by honoring
    ``fix_scaler_index=True``, ``empty_slice_guard=True``, ``mean_field=False``,
    and ``virtual_alpha=0.0`` via ``verify_resume_contract`` and (when
    ``target_window_index`` is supplied) ``verify_causal_replay_no_future_checkpoint``.
    """
    payload = load_checkpoint(checkpoint_path)
    meta = dict(payload.get("meta") or {})
    if target_window_index is not None:
        verify_causal_replay_no_future_checkpoint(
            payload,
            target_window_index=int(target_window_index),
            target_sim_time_ms=target_sim_time_ms,
        )

    slices_dict = payload.get("slices", {})
    first_slice = slices_dict.get(0, slices_dict.get("0", {}))
    c_eff = int(first_slice.get("n_prbs", meta.get("c_prbs", 53)))
    controllers = {
        sid: SqrSliceController(
            state_dim=6,
            n_prbs=c_eff,
            gamma=float(meta.get("gamma", 1.0)),
            sla_threshold=0.0,
            tau=TAU_INIT,
            tau_step=TAU_STEP,
            margin=MARGIN_INIT,
            margin_bounds=tuple(float(x) for x in meta.get("margin_bounds", MARGIN_BOUNDS)),
            violation_window=100,
            epsilon=0.1,
            temperature=1.0,
            knn_k=10,
            seed=int(meta.get("seed", 1)) + 1000 * sid,
        )
        for sid in range(NUM_SLICES)
    }
    req_meta = (
        dict(expected_meta)
        if expected_meta is not None
        else {
            "controller": "sqr_het",
            "gamma": float(meta.get("gamma", 1.0)),
            "virtual_alpha": 0.0,
            "empty_slice_guard": True,
            "mean_field": False,
            "fix_scaler_index": True,
            "margin_init": float(meta.get("margin_init", 10.0)),
            "margin_bounds": list(meta.get("margin_bounds", [2.0, 50.0])),
        }
    )
    verified_meta = verify_resume_contract(payload, req_meta, controllers)
    restore_controllers(payload, controllers)
    scaler = FeatureScaler.load_manifest(scaler_manifest_path)
    het = SqrHetController(
        tuple(controllers[sid] for sid in range(NUM_SLICES)),
        scaler,
        c_eff,
        virtual_alpha=float(verified_meta.get("virtual_alpha", 0.0)),
        empty_slice_guard=bool(verified_meta.get("empty_slice_guard", True)),
        mean_field=bool(verified_meta.get("mean_field", False)),
        fix_scaler_index=bool(verified_meta.get("fix_scaler_index", True)),
    )
    return controllers, het, verified_meta
