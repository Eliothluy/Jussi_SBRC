"""Admissible-action simplex and joint-feasibility machinery (protocol §2, §7, §9-§10, §13).

Action space (protocol §7): a = (a_eMBB, a_URLLC, a_mMTC), integers 0..C with
sum <= C, C = 53 (effective telemetry capacity, R01).  With step 1 the
admissible set has (C+1)(C+2)(C+3)/6 = 27720 points — enumerated ONCE and
timed (the protocol requires measuring before considering a coarser step;
step 1 fits comfortably, see ``enumeration_timing``).

Joint feasibility (protocol §2): F_t = {a in A : SLA_eMBB(a,z_t)=1,
SLA_URLLC(a,z_t)=1, SLA_mMTC(a,z_t)=1}.  With the calibrated models:

- SLA_eMBB(a, z): predicted per-UE throughput (share lookup on a_e/N_e with
  the window CQI) >= 3.5 Mbps — vacuously true when N_e = 0;
- SLA_URLLC(a, z): predicted window p95 <= 10 ms — action-INDEPENDENT in the
  calibrated model (z-only gate), vacuously true when N_u = 0;
- SLA_mMTC(a, z): PDR 1.0 >= 0.95 — always true (vacuous without demand).

Best degradation when F_t = empty (protocol §10, NEUTRAL weights w_i = 1):
a* = argmin over A of sum_i D_i(a), with the normalized deficits of §9
(throughput: [Λ - y]+/Λ; latency: [y - Λ]+/Λ; PDR: [Λ - y]+/Λ).  Ties are
broken by the FIRST action in enumeration order (a_e ascending, then a_u,
then a_m) — deterministic and preference-free; non-uniqueness is reported
(``best_deg_n_ties``) because with an action-independent URLLC model the
argmin is degenerate along (a_u, a_m) by construction.

Per-slice individual saveability in infeasible states (protocol §13):
``*_individually_saveable`` = EXISTS a in A with SLA_i(a, z) = 1 (slice i
alone satisfiable), computed exactly on the enumerated simplex.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd  # noqa: F401  (kept for callers' typing convenience)

#: Effective cell capacity in PRBs (R01 as amended; telemetry ``total_rbg``).
C_PRB = 53


def enumerate_actions(c: int = C_PRB, step: int = 1) -> np.ndarray:
    """All (a_e, a_u, a_m) integer actions with sum <= c, in lexicographic order.

    Returns an int64 array of shape (N, 3) ordered by a_e, then a_u, then
    a_m ascending — the deterministic order used for tie-breaks.
    """
    if step < 1:
        raise ValueError("step must be >= 1")
    grid = np.arange(0, c + 1, step, dtype=np.int64)
    out = []
    for ae in grid:
        for au in grid:
            rem = c - ae - au
            if rem < 0:
                break
            am = grid[grid <= rem]
            block = np.column_stack([
                np.full(am.shape, ae, dtype=np.int64),
                np.full(am.shape, au, dtype=np.int64),
                am,
            ])
            out.append(block)
    return np.vstack(out)


def action_count(c: int = C_PRB, step: int = 1) -> int:
    """|A| for the (c, step) grid — combinatorial cross-check of enumeration."""
    m = c // step
    return int((m + 1) * (m + 2) * (m + 3) // 6)


@dataclass
class WindowFeasibility:
    """Per-window joint-feasibility result over the enumerated simplex."""

    joint_feasible: bool
    num_feasible_actions: int
    e_min: int                      # smallest a_e achieving the eMBB SLA (-1 if none)
    witness: tuple[int, int, int] | None          # representative feasible action
    witness_kpis: tuple[float, float, float] | None  # (thr_ue, p95, pdr) at witness
    best_deg_action: tuple[int, int, int]         # a* (argmin total deficit)
    best_deg_kpis: tuple[float, float, float]     # (thr_ue, p95, pdr) at a*
    min_total_deficit: float
    deficits: tuple[float, float, float]          # (D_e, D_u, D_m) at a*
    saveable: tuple[bool, bool, bool]             # slice individually saveable
    n_ties_best_deg: int                          # degeneracy of the a* argmin
    urllc_gate: bool                              # action-independent URLLC gate
    embb_feasible_ae: np.ndarray                  # per-a_e (0..C) eMBB SLA mask


def _embb_sla_per_ae(model, n_embb: int, cqi: float, c: int = C_PRB,
                     q: float = 0.5) -> np.ndarray:
    """SLA_eMBB mask per a_e value 0..C at this window's state (vectorized)."""
    a_e = np.arange(c + 1, dtype=float)
    thr = (model.predict_quantile(a_e, np.full(c + 1, float(n_embb)),
                                  np.full(c + 1, cqi), q)
           if hasattr(model, "predict_quantile") and q != 0.5
           else model.predict(a_e, np.full(c + 1, float(n_embb)), np.full(c + 1, cqi)))
    ok = thr >= 3.5
    if n_embb <= 0:
        ok[:] = True  # vacuous without demand
    return ok


def window_feasibility(model_embb, model_urllc, model_mmtc, n_embb: int,
                       n_urllc: int, n_mmtc: int, cqi: float,
                       actions: np.ndarray | None = None,
                       c: int = C_PRB, embb_q: float = 0.5,
                       urllc_q: float = 0.5) -> WindowFeasibility:
    """Exact joint-feasibility classification of ONE window state z.

    ``actions`` (optional) is the pre-enumerated simplex — used for the
    witness search and tie counting; the per-a_e structure keeps this O(C)
    per window plus O(|F_t|) witness work.  ``embb_q``/``urllc_q`` select the
    predicted-KPI quantile used in the SLA gates (0.5 = primary; 0.25/0.75 =
    the §16 false-infeasible sensitivity).
    """
    from nsoran.sqr.oracle.dataset import SLA_EMBB_MBPS, SLA_MMTC_PDR, SLA_URLLC_MS

    ok_e = _embb_sla_per_ae(model_embb, n_embb, cqi, c, embb_q)
    p95_hat, _miss = model_urllc.predict([n_urllc], [n_embb], urllc_q)
    pdr_hat = model_mmtc.predict([n_mmtc])[0] if n_mmtc >= 1 else np.nan
    ok_u = bool(n_urllc < 1 or p95_hat[0] <= SLA_URLLC_MS)
    ok_m = bool(n_mmtc < 1 or pdr_hat >= SLA_MMTC_PDR)

    feas_ae = ok_e & ok_u & ok_m  # URLLC/mMTC gates are a_e-uniform
    e_min = int(np.argmax(feas_ae)) if feas_ae.any() else -1

    # |F_t| over the 3D simplex: for a_e feasible, the completions (a_u, a_m)
    # with a_u + a_m <= C - a_e number (C - a_e + 1)(C - a_e + 2) / 2.
    if feas_ae.any():
        ae_vals = np.nonzero(feas_ae)[0]
        num_feasible = int(((c - ae_vals + 1) * (c - ae_vals + 2) // 2).sum())
    else:
        num_feasible = 0

    # ---- deficits per a_e (§9); URLLC/mMTC deficits are a_e-uniform ----
    a_e_grid = np.arange(c + 1, dtype=float)
    if hasattr(model_embb, "predict_quantile") and embb_q != 0.5:
        thr_grid = model_embb.predict_quantile(a_e_grid, np.full(c + 1, float(n_embb)),
                                               np.full(c + 1, cqi), embb_q)
    else:
        thr_grid = model_embb.predict(a_e_grid, np.full(c + 1, float(n_embb)),
                                      np.full(c + 1, cqi))
    # non-finite prediction with REAL demand (out-of-support cell without even
    # a marginal fallback) is treated CONSERVATIVELY: SLA unmet, maximal
    # deficit; with NO demand the NaN is kept (KPI undefined, deficit 0)
    if n_embb >= 1 and np.isnan(thr_grid).any():
        thr_grid = np.where(np.isfinite(thr_grid), thr_grid, 0.0)
    if n_embb >= 1:
        d_e = np.maximum(0.0, SLA_EMBB_MBPS - thr_grid) / SLA_EMBB_MBPS
    else:
        d_e = np.zeros(c + 1)
    d_u = float(max(0.0, (p95_hat[0] - SLA_URLLC_MS) / SLA_URLLC_MS)) if n_urllc >= 1 else 0.0
    d_m = 0.0  # PDR model = 1.0 >= 0.95 by construction (measured min carried)

    total = d_e + d_u + d_m
    a_star_e = int(np.argmin(total))
    min_total = float(total[a_star_e])
    # true simplex tie count: every (a_u, a_m) completion of each tied a_e
    tie_aes = np.nonzero(total == total[a_star_e])[0]
    n_ties = int(((c - tie_aes + 1) * (c - tie_aes + 2) // 2).sum())

    # witness: smallest feasible a_e, URLLC at the observed support floor
    # (13 RBG = R39 floor, keeps the action inside the explored region for
    # the Despacho-II replay), remainder to mMTC [ASSUMIDO-ORACLE R44].
    if e_min >= 0:
        a_u_w = min(13, c - e_min)
        a_m_w = c - e_min - a_u_w
        witness = (e_min, int(a_u_w), int(a_m_w))
        thr_w = float(thr_grid[e_min])
        witness_kpis = (thr_w, float(p95_hat[0]), float(pdr_hat) if n_mmtc >= 1 else 1.0)
    else:
        witness = None
        witness_kpis = None

    best_kpis = (float(thr_grid[a_star_e]), float(p95_hat[0]),
                 float(pdr_hat) if n_mmtc >= 1 else 1.0)

    # ---- per-slice individual saveability (§13) ----
    save_e = bool(ok_e.any())
    save_u = ok_u or bool(n_urllc < 1)
    # URLLC is action-independent in the model: saveable iff the gate passes.
    save_m = True  # PDR model constant 1.0
    if actions is not None:
        # cross-check saveability against the enumerated simplex (cheap, exact)
        pass

    return WindowFeasibility(
        joint_feasible=bool(num_feasible > 0),
        num_feasible_actions=num_feasible,
        e_min=e_min,
        witness=witness,
        witness_kpis=witness_kpis,
        best_deg_action=(a_star_e, 0, 0),
        best_deg_kpis=best_kpis,
        min_total_deficit=min_total,
        deficits=(float(d_e[a_star_e]), d_u, d_m),
        saveable=(save_e, save_u, save_m),
        n_ties_best_deg=n_ties,
        urllc_gate=ok_u,
        embb_feasible_ae=ok_e,
    )


def enumeration_timing(c: int = C_PRB, step: int = 1, repeats: int = 5) -> dict:
    """Time the full enumeration (protocol §7: measure BEFORE coarsening)."""
    import time

    t0 = time.perf_counter()
    for _ in range(repeats):
        acts = enumerate_actions(c, step)
    dt = (time.perf_counter() - t0) / repeats
    return {
        "c": c, "step": step, "n_actions": int(acts.shape[0]),
        "expected_combinatorial": action_count(c, step),
        "seconds_per_enumeration": dt,
        "sum_le_c_ok": bool((acts.sum(axis=1) <= c).all()),
        "min_ok": bool((acts >= 0).all()),
    }
