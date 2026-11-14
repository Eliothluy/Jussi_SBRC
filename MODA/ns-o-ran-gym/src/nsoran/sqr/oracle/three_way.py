"""3-way feasibility decision over the SUPPORTED action space (Etapa 2.4-II-d).

Protocol ``docs/artigo/fase_2_4_II_d_recalibracao_oracle.md`` §4-§7 (band
decision), §14 (supported space), §15 (D_min in support), §17 (full
reprocessing, provisional names), §23 (frozen before the gate).

Band rule (§5, frozen):
- higher-better KPI (eMBB throughput):  FEASIBLE if p25 >= SLA;
  INFEASIBLE if p75 < SLA; else MARGINAL.
- lower-better KPI (URLLC p95):  FEASIBLE if p75 <= SLA; INFEASIBLE if
  p25 > SLA; else MARGINAL.
Per action: JOINT_FEASIBLE (all critical KPIs FEASIBLE) / JOINT_INFEASIBLE
(any INFEASIBLE) / JOINT_MARGINAL (else).  Per state (§7): FEASIBLE iff
|F_t| > 0; else MARGINAL iff |M_t| > 0; else INFEASIBLE — with the
num_feasible/marginal/infeasible_actions counts of §7.

Supported space (§14, frozen — see the model cards for the calibration
evidence):
- ``a_u >= 13``            (URLLC floor, protocol §12 / R50);
- ``a_m >= a_m_min``       (mMTC floor — parameter; placeholder 3 until the
                            D6 microcalibration measures it, §13);
- ``18 <= a_e <= 37``      (eMBB claim support: the ONLY executed eMBB quotas
                            of the frozen 2.3 campaigns are 18 (equal arm)
                            and 37 (prop arm); the recalibrated eMBB model
                            is calibrated ON that manifold and every
                            counterfactual here stays on it — interpolation,
                            never extrapolation.  Actions below 18 / above 37
                            are OUT of model support and carry NO feasibility
                            claim; protocol §37 "suporte insuficiente" is
                            reported, not hidden.)
- ``a_e + a_u + a_m <= 53`` (R01 capacity; idle RBGs allowed, R27).

D_min (§15): ``min_a sum_i w_i D_i(a)`` over the supported space with the
frozen §9 normalized deficits (w=1).  Envelopes: p25 = PESSIMISTIC member
(thr at p25, p95 at p75), p50 = median, p75 = OPTIMISTIC member (thr at p75,
p95 at p25).  ``D_min_uncertain`` = True when the interval touches zero:
optimistic envelope == 0 < pessimistic envelope (the state might be
feasible or not — §15 "cruzar zero" => UNCERTAIN/MARGINAL).

Reprocessing (§17): every window gets state_class, the three action counts,
the deterministic best actions and the D_min envelopes.  Only PROVISIONAL
names are produced (P_F_candidate / P_M_candidate / P_I_candidate) — never
P_inf_final (§17; frozen only after a PASS).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nsoran.sqr.oracle.quantile import (
    SLA_EMBB_MBPS,
    SLA_URLLC_MS,
    BAND_QUANTILES,
    embb_band,
    urllc_band,
)

# ---------------------------------------------------------------------------
# Supported space (frozen; see module docstring)
# ---------------------------------------------------------------------------
C_PRB = 53
URLLC_FLOOR = 13
EMBB_SUPPORT_MIN = 18
EMBB_SUPPORT_MAX = 37

Q_PESS, Q_MED, Q_OPT = BAND_QUANTILES   # 0.25, 0.50, 0.75

#: Band codes shared with quantile.py: 0=FEASIBLE 1=MARGINAL 2=INFEASIBLE.
FEASIBLE, MARGINAL, INFEASIBLE = 0, 1, 2
STATE_NAMES = {0: "FEASIBLE", 1: "MARGINAL", 2: "INFEASIBLE"}


def supported_grids(am_min: int = 3):
    """The supported (a_e, a_u) grids and the completion counts.

    Returns (a_e_grid, a_u_grid, comp) where ``comp[i, j]`` is the number of
    supported a_m completions (a_m from am_min to C - a_e - a_u) of the pair
    (0 when the pair admits none).  |A_support| = comp.sum().
    """
    a_e_grid = np.arange(EMBB_SUPPORT_MIN, EMBB_SUPPORT_MAX + 1)
    a_u_max = C_PRB - EMBB_SUPPORT_MIN - int(am_min)
    a_u_grid = np.arange(URLLC_FLOOR, a_u_max + 1)
    comp = np.clip(C_PRB - a_e_grid[:, None] - a_u_grid[None, :]
                   - int(am_min) + 1, 0, None).astype(np.int64)
    return a_e_grid, a_u_grid, comp


# ---------------------------------------------------------------------------
# Per-window joint classification (vectorized over windows, chunked)
# ---------------------------------------------------------------------------

#: combine-table: joint band from (e_band, u_band): any INFEASIBLE -> I;
#: all FEASIBLE -> F; else M.
_COMBINE = np.array([
    # u=F  u=M  u=I   (rows: e_band)
    [FEASIBLE, MARGINAL, INFEASIBLE],   # e = F
    [MARGINAL, MARGINAL, INFEASIBLE],   # e = M
    [INFEASIBLE, INFEASIBLE, INFEASIBLE],  # e = I
], dtype=np.int8)


def classify_windows_3way(windows: pd.DataFrame, embb_model, urllc_model,
                          am_min: int = 3, chunk: int = 4096) -> pd.DataFrame:
    """§17 reprocessing: one row per window with the 3-way classification.

    Requires the per-window state columns of the frozen dataset (n_embb,
    n_urllc, n_mmtc, cqi_eff, queue_*, n_embb_backlog, offered_load_embb,
    prev_thr_embb_med, urllc_n_pkts, dl_slots, prev_p95, prev_miss_pct) plus
    the identity columns (arm, lam, seed, win, time_ms) and the observed
    violations.  Windows WITHOUT eMBB demand are vacuously FEASIBLE on the
    eMBB axis for every a_e (and likewise URLLC); mMTC is vacuous without
    demand and inside the support the placeholder model is PDR = 1.0 >= 0.95
    (always FEASIBLE — the floor is enforced by the SUPPORT, not by the
    band).
    """
    d = windows.reset_index(drop=True)
    a_e_grid, a_u_grid, comp = supported_grids(am_min)
    ne = d["n_embb"].fillna(0).to_numpy(float)
    nu = d["n_urllc"].fillna(0).to_numpy(float)
    has_e = ne >= 1
    has_u = nu >= 1

    n = len(d)
    e_band = np.full((n, len(a_e_grid)), FEASIBLE, dtype=np.int8)
    u_band = np.full((n, len(a_u_grid)), FEASIBLE, dtype=np.int8)
    # deficit tables per band member (pessimistic/median/optimistic envelopes)
    #   eMBB: D_e = max(0, SLA - thr)/SLA  -> band member (pess, med, opt)
    #   URLLC: D_u = max(0, p95 - SLA)/SLA -> band member (pess, med, opt)
    d_e = np.zeros((n, len(a_e_grid), 3))
    d_u = np.zeros((n, len(a_u_grid), 3))
    # executed-action diagnostics
    thr_hat_exec = np.full(n, np.nan)
    p95_hat_exec = np.full(n, np.nan)

    idx_e = np.nonzero(has_e)[0]
    idx_u = np.nonzero(has_u)[0]
    for lo in range(0, len(idx_e), chunk):
        sel = idx_e[lo:lo + chunk]
        q = embb_model.predict_grid_bands(
            ne[sel], d["cqi_eff"].to_numpy(float)[sel],
            d["queue_embb"].fillna(0).to_numpy(float)[sel],
            d["n_embb_backlog"].fillna(0).to_numpy(float)[sel],
            d["offered_load_embb"].fillna(0).to_numpy(float)[sel],
            d["prev_thr_embb_med"].to_numpy(float)[sel], a_e_grid)
        e_band[sel] = embb_band(q[Q_PESS], q[Q_OPT])
        for m in range(3):
            # band member: pessimistic = p25, median = p50, optimistic = p75
            mem = q[Q_PESS] if m == 0 else (q[Q_MED] if m == 1 else q[Q_OPT])
            d_e[sel, :, m] = np.maximum(0.0, SLA_EMBB_MBPS - mem) / SLA_EMBB_MBPS
        # executed-action diagnostics (nearest grid column to executed a_e)
        ae_exec = np.clip(np.rint(d["a_e"].to_numpy(float)[sel]), 18, 37)
        k = np.searchsorted(a_e_grid, ae_exec)
        thr_hat_exec[sel] = q[Q_MED][np.arange(len(sel)), k]

    for lo in range(0, len(idx_u), chunk):
        sel = idx_u[lo:lo + chunk]
        q = urllc_model.predict_grid_bands(
            nu[sel], d["n_embb"].to_numpy(float)[sel],
            d["queue_urllc"].fillna(0).to_numpy(float)[sel],
            d["queue_embb"].fillna(0).to_numpy(float)[sel],
            d["urllc_n_pkts"].fillna(0).to_numpy(float)[sel],
            d["dl_slots"].to_numpy(float)[sel],
            d["prev_p95"].to_numpy(float)[sel],
            d["prev_miss_pct"].to_numpy(float)[sel], a_u_grid)
        u_band[sel] = urllc_band(q[Q_PESS], q[Q_OPT])
        for m in range(3):
            # LOWER-BETTER KPI: pessimistic = p75 (large latency),
            # median = p50, optimistic = p25 (small latency)
            mem = q[Q_OPT] if m == 0 else (q[Q_MED] if m == 1 else q[Q_PESS])
            d_u[sel, :, m] = np.maximum(0.0, mem - SLA_URLLC_MS) / SLA_URLLC_MS
        au_exec = np.clip(np.rint(d["a_u"].to_numpy(float)[sel]), 13, 50)
        k = np.searchsorted(a_u_grid, au_exec)
        p95_hat_exec[sel] = q[Q_MED][np.arange(len(sel)), k]

    # ---- joint classification over the supported pairs (chunked) ----
    num_f = np.zeros(n, dtype=np.int64)
    num_m = np.zeros(n, dtype=np.int64)
    num_i = np.zeros(n, dtype=np.int64)
    dmin = np.full((n, 3), np.nan)          # pessimistic, median, optimistic
    dmin_arg = np.full((n, 3, 2), -1, dtype=np.int64)  # (i_e, j_u) of the min
    n_ties = np.zeros(n, dtype=np.int64)
    jclass = np.zeros((n, len(a_e_grid), len(a_u_grid)), dtype=np.int8)
    med_total = np.full((n, len(a_e_grid), len(a_u_grid)), np.nan)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        eb = e_band[lo:hi]                   # (m, Ke)
        ub = u_band[lo:hi]                   # (m, Ku)
        m_len = hi - lo
        jc = _COMBINE[eb[:, :, None], np.broadcast_to(ub[:, None, :],
                                                      (m_len, len(a_e_grid),
                                                       len(a_u_grid)))]
        jclass[lo:hi] = jc
        cf = num_f[lo:hi]
        cm = num_m[lo:hi]
        ci = num_i[lo:hi]
        for c, target in ((FEASIBLE, cf), (MARGINAL, cm), (INFEASIBLE, ci)):
            target += (comp[None, :, :] * (jc == c)).sum(axis=(1, 2))
        # D_min envelopes over supported (i, j)
        de = d_e[lo:hi]                      # (m, Ke, 3)
        du = d_u[lo:hi]                      # (m, Ku, 3)
        mask = comp[None, :, :] > 0
        big = np.inf
        for m in range(3):
            tot = de[:, :, m][:, :, None] + du[:, :, m][:, None, :]
            tot = np.where(mask, tot, big)
            flat = tot.reshape(m_len, -1)
            amin = flat.argmin(axis=1)
            mn = flat[np.arange(m_len), amin]
            mn = np.where(np.isfinite(mn), mn, np.nan)
            dmin[lo:hi, m] = mn
            dmin_arg[lo:hi, m, 0] = amin // len(a_u_grid)
            dmin_arg[lo:hi, m, 1] = amin % len(a_u_grid)
            nties_ = (flat == mn[:, None]).sum(axis=1)
            n_ties[lo:hi] = np.where(np.isfinite(mn), nties_, 0)
            if m == 1:
                med_total[lo:hi] = np.where(mask, tot, np.nan)

    state = np.where(num_f > 0, FEASIBLE,
                     np.where(num_m > 0, MARGINAL, INFEASIBLE)).astype(np.int8)

    # ---- deterministic best actions -------------------------------------
    # a_m is always at the floor (the only calibrated region — D6 extends
    # it); the (a_e, a_u) part is the lexicographically smallest argmin of
    # the median-band total deficit over the respective class (np.argmin on
    # the row-major (a_e asc, a_u asc) flat order = lexicographic).
    am_col = int(am_min)
    supported = comp > 0
    # flat indices of the first supported FEASIBLE / MARGINAL pair per window
    feas_mask = (jclass == FEASIBLE) & supported[None]
    marg_mask = (jclass == MARGINAL) & supported[None]
    best_f = np.full((n, 3), -1, dtype=np.int64)
    best_m = np.full((n, 3), -1, dtype=np.int64)
    has_f = feas_mask.reshape(n, -1).any(axis=1)
    has_m = marg_mask.reshape(n, -1).any(axis=1)
    if has_f.any():
        flat_f = feas_mask.reshape(n, -1).astype(np.int8)
        idx_f = np.argmax(flat_f, axis=1)          # first True (lexicographic)
        i_f, j_f = idx_f // len(a_u_grid), idx_f % len(a_u_grid)
        best_f[has_f, 0] = a_e_grid[i_f[has_f]]
        best_f[has_f, 1] = a_u_grid[j_f[has_f]]
        best_f[has_f, 2] = am_col
    if has_m.any():
        mt_nan = np.where(marg_mask, med_total, np.inf)
        idx_m = mt_nan.reshape(n, -1).argmin(axis=1)
        i_m, j_m = idx_m // len(a_u_grid), idx_m % len(a_u_grid)
        best_m[has_m, 0] = a_e_grid[i_m[has_m]]
        best_m[has_m, 1] = a_u_grid[j_m[has_m]]
        best_m[has_m, 2] = am_col
    best_d = np.column_stack([a_e_grid[dmin_arg[:, 1, 0]],
                              a_u_grid[dmin_arg[:, 1, 1]],
                              np.full(n, am_col)]).astype(np.int64)

    def _fmt(arr: np.ndarray, which: np.ndarray) -> list[str]:
        return [("(%d,%d,%d)" % tuple(arr[r])) if which[r] else ""
                for r in range(len(arr))]

    out = pd.DataFrame({
        "arm": d["arm"], "lam": d["lam"], "seed": d["seed"], "win": d["win"],
        "time_ms": d["time_ms"],
        "n_embb": d["n_embb"].fillna(0).astype(int),
        "n_urllc": d["n_urllc"].fillna(0).astype(int),
        "n_mmtc": d["n_mmtc"].fillna(0).astype(int),
        "state_class": [STATE_NAMES[int(s)] for s in state],
        "num_feasible_actions": num_f,
        "num_marginal_actions": num_m,
        "num_infeasible_actions": num_i,
        "n_supported_actions": int(comp.sum()),
        "best_feasible_action": _fmt(best_f, has_f),
        "best_marginal_action": _fmt(best_m, has_m),
        "best_degradation_action": _fmt(best_d, np.ones(n, dtype=bool)),
        "D_min_support_p25_pessimistic": dmin[:, 0],
        "D_min_support_p50": dmin[:, 1],
        "D_min_support_p75_optimistic": dmin[:, 2],
        "D_min_uncertain": (dmin[:, 2] == 0) & (dmin[:, 0] > 0),
        "best_deg_n_ties_pairs": n_ties,
        "thr_hat_exec": thr_hat_exec,
        "p95_hat_exec": p95_hat_exec,
        "a_m_min_used": am_col,
        "observed_embb_violation": d["observed_embb_violation"],
        "observed_urllc_violation": d["observed_urllc_violation"],
        "observed_mmtc_violation": d["observed_mmtc_violation"],
        "observed_any_violation": d["observed_any_violation"],
    })
    if {"a_e", "a_u", "a_m"} <= set(d.columns):
        out["a_e_exec"] = d["a_e"]
        out["a_u_exec"] = d["a_u"]
        out["a_m_exec"] = d["a_m"]
        out["exec_in_support"] = (
            (d["a_e"].between(EMBB_SUPPORT_MIN, EMBB_SUPPORT_MAX))
            & (d["a_u"] >= URLLC_FLOOR) & (d["a_m"] >= am_min)).astype(float)
    return out


# ---------------------------------------------------------------------------
# Candidate aggregates by lambda (§7/§17 — PROVISIONAL names only)
# ---------------------------------------------------------------------------

def aggregate_3way_by_lambda(cls: pd.DataFrame) -> pd.DataFrame:
    """P_F/P_M/P_I_candidate per lambda (+ counts; provisional, §17)."""
    rows = []
    for lam, g in cls.groupby("lam"):
        row = {"lambda_total": float(lam), "n_windows": int(len(g)),
               "P_F_candidate": float((g["state_class"] == "FEASIBLE").mean()),
               "P_M_candidate": float((g["state_class"] == "MARGINAL").mean()),
               "P_I_candidate": float((g["state_class"] == "INFEASIBLE").mean()),
               "marginal_rate": float((g["state_class"] == "MARGINAL").mean()),
               "n_state_feasible": int((g["state_class"] == "FEASIBLE").sum()),
               "n_state_marginal": int((g["state_class"] == "MARGINAL").sum()),
               "n_state_infeasible": int((g["state_class"] == "INFEASIBLE").sum()),
               "median_supported_actions": float(g["n_supported_actions"].median()),
               "median_num_feasible": (float(g.loc[g["num_feasible_actions"] > 0,
                                                   "num_feasible_actions"].median())
                                       if (g["num_feasible_actions"] > 0).any()
                                       else np.nan),
               "D_min_p50_median_infeasible": (
                   float(g.loc[g["state_class"] == "INFEASIBLE",
                               "D_min_support_p50"].median())
                   if (g["state_class"] == "INFEASIBLE").any() else np.nan),
               "frac_D_min_uncertain": float(g["D_min_uncertain"].mean()),
               "am_min_used": int(g["a_m_min_used"].iloc[0]),
               "n_supported": int(g["n_supported_actions"].iloc[0]),
        }
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("lambda_total").reset_index(drop=True)
    s = out["P_F_candidate"] + out["P_M_candidate"] + out["P_I_candidate"]
    assert np.allclose(s, 1.0), "P_F + P_M + P_I must be 1"
    return out


# ---------------------------------------------------------------------------
# Observed-vs-model consistency diagnostics (candidates for the §22 gate)
# ---------------------------------------------------------------------------

def consistency_candidates(cls: pd.DataFrame) -> dict:
    """Provisional FIR/FFR analogs of the reprocessing (NOT the §22 gate).

    - fir_candidate: of the windows whose EXECUTED outcome satisfied all
      three SLAs, the fraction classified state INFEASIBLE.  An all-OK
      window whose EXECUTED action is itself in-support is a genuine
      refutation of INFEASIBLE; one whose executed action is out-of-support
      only flags the model (no supported action was ever tried there).
    - ffr_candidate: windows classified FEASIBLE whose executed outcome
      violated any SLA (a genuine refutation only when the executed action
      is in-support; otherwise the witness was never tried).
    - marginal_rate overall (§21).
    """
    d = cls
    v = d[["observed_embb_violation", "observed_urllc_violation",
           "observed_mmtc_violation"]]
    all_ok = v.fillna(0).eq(0).all(axis=1) & v.notna().any(axis=1)
    infeas = d["state_class"] == "INFEASIBLE"
    feas = d["state_class"] == "FEASIBLE"
    viol = d["observed_any_violation"] == 1
    insup = (d["exec_in_support"] == 1) if "exec_in_support" in d.columns \
        else pd.Series(False, index=d.index)
    out = {
        "n_windows": int(len(d)),
        "n_executed_all_ok": int(all_ok.sum()),
        "fir_candidate": float((all_ok & infeas).sum()
                               / max(all_ok.sum(), 1)),
        "fir_candidate_insupport_exec_only": (
            float((all_ok & infeas & insup).sum()
                  / max((all_ok & insup).sum(), 1))
            if insup.any() else np.nan),
        "n_any_violation": int(viol.sum()),
        "ffr_candidate": float((viol & feas).sum() / max(viol.sum(), 1)),
        "ffr_candidate_insupport_exec_only": (
            float((viol & feas & insup).sum()
                  / max((viol & insup).sum(), 1))
            if (viol & insup).any() else np.nan),
        "marginal_rate": float((d["state_class"] == "MARGINAL").mean()),
        "n_state_feasible": int(feas.sum()),
        "n_state_marginal": int((d["state_class"] == "MARGINAL").sum()),
        "n_state_infeasible": int(infeas.sum()),
    }
    return out
