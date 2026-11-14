"""Quantile (pinball) KPI models of the recalibrated oracle — Etapa 2.4-II-d.

Recalibration after the II-c executable STOP (protocol
``docs/artigo/fase_2_4_II_d_recalibracao_oracle.md`` §8-§12, §16, §23).

Design frozen BEFORE the II-d replay gate (rule §23 — thresholds, quantiles,
floors and the main features are FIXED here and must not change after the
gate results are seen; any change reopens ``II-d-v2``):

- f_eMBB: batch pinball quantile regression (linear-in-features, LP-exact via
  HiGHS) predicting the WINDOW MEDIAN per-UE eMBB throughput (the frozen R41
  judgment quantity) from CONTINUOUS CQI (no coarse CQI bins — the II-c
  failure ``thr_hat 3.42 vs SLA 3.5`` came from the cqi_bin=1 stratum) plus
  the state/share features of protocol §8.  Quantiles p25/p50/p75 give the
  uncertainty band of the 3-way decision (§5).
- f_URLLC: same machinery predicting the WINDOW p95 of URLLC packet delays
  (ms) per window (protocol §10) from state features INCLUDING the burst /
  TDD-context / persistence signals the II-c adjudication showed dominate
  the tail (dl_slots proxy corr -0.67; prev_p95 corr +0.74 on the frozen
  2.3 windows).  ``a_urllc`` enters through the SATURATED basis
  ``[min(a_u,13), max(a_u-13,0)]``: the post-floor coefficient is FREE (no
  simple monotonicity is imposed) but the form encodes the preserved causal
  finding R50 (above the floor the tail responds little to extra quota).
- f_mMTC: placeholder — inside the supported space (``a_m >= a_m_min``) the
  model is the constant PDR = 1.0 (measured at every executed ``a_m``: 3 in
  the prop arm, 18 in the equal arm; R50 corroborates the collapse only at
  ``a_m = 0``, OUTSIDE the support).  The floor itself is measured by the
  D6 microcalibration replays (protocol §13) — see ``replay_plan_v2``.

Feature lists are exactly the frozen ones (see :data:`FROZEN_RECAL`);
``util_active_dl`` is deliberately NOT an eMBB feature (deviation from the
§8 preference list, documented in the model card): utilization is a
CONSEQUENCE of the action (``100 * scheduled / (C * dl_slots)``) and is not
definable for a counterfactual action; the (a_e, share, offered, queue)
block carries the saturation signal instead.

Validation: leave-one-seed-out (protocol §16) with the §9/§11 error gates —
overall MAE/RMSE, NEAR-THRESHOLD MAE (eMBB ``|thr-3.5| <= 1 Mbps``; URLLC
``|p95-10| <= 2 ms``), eMBB CQI-low MAE (``cqi <= 13``), band
classification at the executed action (false-infeasible / false-feasible /
absorbed-by-MARGINAL) and quantile coverage (nominal 50 %).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# FROZEN design (rule §23 — do not edit after seeing the II-d gate results)
# ---------------------------------------------------------------------------
SLA_EMBB_MBPS = 3.5
SLA_URLLC_MS = 10.0
SLA_MMTC_PDR = 0.95          # frozen R36-R38 (never recalibrated here)

#: Band quantiles of the 3-way decision (protocol §5).
BAND_QUANTILES: tuple[float, float, float] = (0.25, 0.50, 0.75)

#: Near-threshold regions of the §9/§11 error gates.
EMMB_NEAR_HALF_WIDTH_MBPS = 1.0     # |thr - 3.5| <= 1
URLLC_NEAR_HALF_WIDTH_MS = 2.0      # |p95 - 10| <= 2
#: CQI-low region (the II-c failure stratum): continuous CQI at or below 13.
CQI_LOW_MAX = 13.0

#: URLLC support floor (protocol §12, R50) — frozen provisionally.
URLLC_FLOOR_RBGS = 13

#: mMTC floor placeholder (protocol §13): measured PDR=1.0 at a_m=3 (prop arm)
#: and 18 (equal arm); D6 microcalibration measures {0..5} and the orchestrator
#: re-runs with the measured value (this is a PARAMETER downstream, not a
#: constant of the band rule).
MMTC_FLOOR_PLACEHOLDER_RBGS = 3

#: eMBB feature basis (frozen; model card documents each column).
#: The ACTION enters ONLY through the per-UE share s = a_e / n_embb, as a
#: LINEAR SPLINE with knots at the II-c stratum edges (0, 2, 4, 6, 8, 12,
#: 18, 27) — the pooled s -> thr curve of the frozen 2.3 windows is strongly
#: nonlinear (medians 0.0 / 1.7 / 2.3 / 3.3 / 6.4 / 6.8 / 14.4 Mbps), and a
#: plain linear-in-s basis degenerates off the trained (a_e in {18, 37})
#: manifold.  Counterfactual predictions are clipped to the measured cell
#: capacity [0, 24] Mbps (R39 peak-cell note).
#:   state (9):
#:   0  const
#:   1  cqi        (continuous wideband CQI of the window)
#:   2  cqi_low    = max(13 - cqi, 0)     (extra sensitivity below 13)
#:   3  n_embb     (demand UEs)
#:   4  n_backlog  (queueBytes > 0)
#:   5  qe_mb      = queue_embb / 1e6
#:   6  offered    (R40 proxy, Mbps)
#:   7  prev_thr   (previous window median throughput)
#:   8  prev_missing (indicator)
#:   action (10):
#:   9   s
#:   10  h2..16 h27 = max(s - knot, 0) for knots (2, 4, 6, 8, 12, 18, 27)
#:   17  cqi_s9     = cqi * min(s, 9)     (continuous-CQI x share)
#:   18  cqi_low_s9 = cqi_low * min(s, 9)
EMMB_SPLINE_KNOTS: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 27.0)
EMMB_THR_CAP_MBPS = 24.0
EMMB_FEATURES: tuple[str, ...] = (
    "const", "cqi", "cqi_low", "n_embb", "n_backlog", "qe_mb", "offered",
    "prev_thr", "prev_missing",
    "s", "h2", "h4", "h6", "h8", "h12", "h18", "h27",
    "cqi_s9", "cqi_low_s9",
)
#: Columns of the eMBB basis that vary with the ACTION (grid factorization).
EMMB_ACTION_COLS: tuple[str, ...] = (
    "s", "h2", "h4", "h6", "h8", "h12", "h18", "h27", "cqi_s9", "cqi_low_s9",
)

#: URLLC feature basis (frozen; protocol §10 list, model card documents).
#:   0  const
#:   1  n_urllc
#:   2  nu_ge2, 3 nu_ge3          (contention tiers)
#:   4  n_embb, 5 ne_ge2          (cross-slice state)
#:   6  qu_100k  = queue_urllc/1e5
#:   7  qe_mb
#:   8  burst     = 1[n_pkts/n_urllc < 100]  (cadence deviation — partial
#:                 window activity; frozen definition)
#:   9  dl_slots  (TDD context proxy — the II-c diagnostic variable)
#:  10  prev_p95, 11 prev_miss = prev_miss_pct/100
#:  12  au_sat = min(a_u, 13), 13 au_over = max(a_u-13, 0)  (saturated form,
#:      post-floor slope FREE — no monotonicity imposed, R50 preserved)
#:  14  prev_missing
URLLC_FEATURES: tuple[str, ...] = (
    "const", "n_urllc", "nu_ge2", "nu_ge3", "n_embb", "ne_ge2",
    "qu_100k", "qe_mb", "burst", "dl_slots", "prev_p95", "prev_miss",
    "au_sat", "au_over", "prev_missing",
)
URLLC_ACTION_COLS: tuple[str, ...] = ("au_sat", "au_over")

FROZEN_RECAL: dict = {
    "protocol": "docs/artigo/fase_2_4_II_d_recalibracao_oracle.md §8-§12, §16, §23",
    "slas": {"embb_mbps": SLA_EMBB_MBPS, "urllc_ms": SLA_URLLC_MS,
             "mmtc_pdr": SLA_MMTC_PDR},
    "band_quantiles": list(BAND_QUANTILES),
    "band_rule": {
        "higher_better": "FEASIBLE if p25 >= SLA; INFEASIBLE if p75 < SLA; else MARGINAL",
        "lower_better": "FEASIBLE if p75 <= SLA; INFEASIBLE if p25 > SLA; else MARGINAL",
    },
    "embb_features": list(EMMB_FEATURES),
    "embb_action_cols": list(EMMB_ACTION_COLS),
    "urllc_features": list(URLLC_FEATURES),
    "urllc_action_cols": list(URLLC_ACTION_COLS),
    "embb_near_threshold": "abs(thr - 3.5) <= 1.0 Mbps",
    "embb_cqi_low": "cqi <= 13.0",
    "urllc_near_threshold": "abs(p95 - 10) <= 2.0 ms",
    "urllc_burst": "urllc_n_pkts / n_urllc < 100 (cadence deviation)",
    "urllc_floor_rbg": URLLC_FLOOR_RBGS,
    "mmtc_floor_placeholder_rbg": MMTC_FLOOR_PLACEHOLDER_RBGS,
    "mmtc_model": "PDR = 1.0 constant inside the support (placeholder until D6)",
    "split": "leave-one-seed-out (3 seeds), final fit on all seeds",
    "deviations_from_protocol_preference_lists": [
        "eMBB: util_active_dl EXCLUDED (consequence of the action; undefined "
        "for counterfactual actions) — §8 preference list",
        "eMBB: zero-quota anchor — a_e=0 with n_embb>=1 predicted as 0 Mbps "
        "(R27 absolute-quota semantics; no training rows at a_e=0)",
        "eMBB: counterfactual grid clipped to the measured cell capacity "
        "[0, 24] Mbps and monotone non-decreasing in a_e (frozen shape "
        "constraints, pre-gate)",
        "URLLC: post-floor a_u effect clipped to +/- 2 ms in the "
        "counterfactual grid (R43 measured bound; fit unclipped)",
    ],
}


# ---------------------------------------------------------------------------
# Batch pinball (quantile) regression — exact LP
# ---------------------------------------------------------------------------

def pinball_lp(X: np.ndarray, y: np.ndarray, q: float) -> np.ndarray:
    """Exact batch quantile regression via the pinball-loss LP (HiGHS IPM).

    min_{beta,u,v}  q*1'u + (1-q)*1'v   s.t.  X beta + u - v = y, u,v >= 0
    (beta free).  Deterministic; returns beta of length p.

    Method note (measured 23/10/2026 on the frozen 2.3 windows): the scipy
    ``method="highs"`` AUTO selection returns a spuriously "optimal"
    beta = 0 on this structure at n ~ 45k (objective ~2x the true optimum);
    ``"highs-ipm"`` and ``"highs-ds"`` both return the verified optimum
    (const-only check: beta = 6.907 = median(y)); IPM is ~40x faster here,
    so IPM is pinned.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    c = np.concatenate([np.zeros(p), np.full(n, q), np.full(n, 1.0 - q)])
    a_eq = sparse.hstack([
        sparse.csr_matrix(-X),
        -sparse.identity(n, format="csr"),
        sparse.identity(n, format="csr"),
    ], format="csr")
    b_eq = -y
    bounds = [(None, None)] * p + [(0.0, None)] * (2 * n)
    res = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs-ipm")
    if not res.success:  # pragma: no cover — convex feasible LP
        raise RuntimeError(f"pinball LP failed (q={q}): {res.message}")
    return np.asarray(res.x[:p], dtype=float)


def rearrange(qs: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Quantile crossing fix: per-point monotone rearrangement (sorted)."""
    stack = np.vstack([qs[q] for q in sorted(qs)])
    stack = np.sort(stack, axis=0)
    return {q: stack[i].copy() for i, q in enumerate(sorted(qs))}


class PinballQuantileModel:
    """Three pinball LP fits (p25/p50/p75) on a frozen standardized basis."""

    def __init__(self, quantiles: tuple[float, ...] = BAND_QUANTILES):
        self.quantiles = tuple(quantiles)
        self.betas: dict[float, np.ndarray] = {}
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PinballQuantileModel":
        X = np.asarray(X, dtype=float)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        # CONSTANT columns (the intercept) are kept AS-IS: centering them
        # would delete the intercept from the design (this exact bug made
        # every prediction collapse to ~0 in the first calibration attempt).
        self._keep = (self.sd != 0).astype(float)
        self.sd = np.where(self.sd == 0, 1.0, self.sd)
        Xs = (X - self.mu * self._keep) / self.sd
        for q in self.quantiles:
            self.betas[q] = pinball_lp(Xs, y, q)
        return self

    def predict_quantiles(self, X: np.ndarray) -> dict[float, np.ndarray]:
        Xs = (np.asarray(X, dtype=float) - self.mu * self._keep) / self.sd
        return rearrange({q: Xs @ self.betas[q] for q in self.quantiles})

    def predict_median(self, X: np.ndarray) -> np.ndarray:
        return self.predict_quantiles(X)[0.5]


# ---------------------------------------------------------------------------
# Frozen bases
# ---------------------------------------------------------------------------

def embb_state_features(df: pd.DataFrame) -> np.ndarray:
    """State columns of the eMBB basis (everything except the action cols)."""
    cqi = df["cqi_eff"].to_numpy(dtype=float)
    prev = df["prev_thr_embb_med"].to_numpy(dtype=float)
    missing = ~np.isfinite(prev)
    n_backlog = df["n_embb_backlog"].fillna(0).to_numpy(dtype=float)
    return np.column_stack([
        np.ones(len(df)),
        cqi,
        np.maximum(CQI_LOW_MAX - cqi, 0.0),
        df["n_embb"].to_numpy(dtype=float),
        n_backlog,
        df["queue_embb"].fillna(0).to_numpy(dtype=float) / 1e6,
        df["offered_load_embb"].fillna(0).to_numpy(dtype=float),
        np.nan_to_num(prev, nan=0.0),
        missing.astype(float),
    ])


def embb_action_features(a_e: np.ndarray, n_embb: np.ndarray,
                         cqi: np.ndarray) -> np.ndarray:
    """Action columns of the eMBB basis for arbitrary (a_e, n_embb, cqi).

    The action acts through the per-UE share s = a_e / max(n, 1) via the
    frozen linear spline + continuous-CQI interactions.
    """
    a_e = np.asarray(a_e, dtype=float)
    n = np.maximum(np.asarray(n_embb, dtype=float), 1.0)
    cqi = np.broadcast_to(np.asarray(cqi, dtype=float), a_e.shape)
    s = a_e / n
    s9 = np.minimum(s, 9.0)
    cols = [s] + [np.maximum(s - k, 0.0) for k in EMMB_SPLINE_KNOTS]
    cols += [cqi * s9, np.maximum(CQI_LOW_MAX - cqi, 0.0) * s9]
    return np.column_stack(cols)


def _place(state: np.ndarray, act: np.ndarray, n_feat: int,
           act_idx: tuple[int, ...]) -> np.ndarray:
    """Assemble the FULL-ordered design: state cols at their canonical
    positions, action cols at ``act_idx`` — the SAME layout the
    counterfactual grid path uses (fit and grid share one ordering)."""
    X = np.zeros((len(state), n_feat))
    state_idx = [i for i in range(n_feat) if i not in act_idx]
    X[:, state_idx] = state
    X[:, act_idx] = act
    return X


def embb_design(df: pd.DataFrame) -> tuple[np.ndarray, tuple[int, ...]]:
    """(X, action_col_positions) of the eMBB basis, canonical feature order."""
    state = embb_state_features(df)
    act = embb_action_features(df["a_e"].to_numpy(dtype=float),
                               df["n_embb"].to_numpy(dtype=float),
                               df["cqi_eff"].to_numpy(dtype=float))
    act_idx = tuple(EMMB_FEATURES.index(c) for c in EMMB_ACTION_COLS)
    return _place(state, act, len(EMMB_FEATURES), act_idx), act_idx


def urllc_state_features(df: pd.DataFrame) -> np.ndarray:
    """State columns of the URLLC basis (everything except au_sat/au_over)."""
    n_u = df["n_urllc"].to_numpy(dtype=float)
    n_e = df["n_embb"].to_numpy(dtype=float)
    ppu = (df["urllc_n_pkts"].to_numpy(dtype=float)
           / np.maximum(n_u, 1.0))
    prev_p95 = df["prev_p95"].to_numpy(dtype=float)
    prev_miss = df["prev_miss_pct"].to_numpy(dtype=float)
    missing = ~np.isfinite(prev_p95)
    return np.column_stack([
        np.ones(len(df)),
        n_u,
        (n_u >= 2).astype(float),
        (n_u >= 3).astype(float),
        n_e,
        (n_e >= 2).astype(float),
        df["queue_urllc"].fillna(0).to_numpy(dtype=float) / 1e5,
        df["queue_embb"].fillna(0).to_numpy(dtype=float) / 1e6,
        (ppu < 100.0).astype(float),          # frozen burst definition
        df["dl_slots"].to_numpy(dtype=float),
        np.nan_to_num(prev_p95, nan=0.0),
        np.nan_to_num(prev_miss, nan=0.0) / 100.0,
        missing.astype(float),
    ])


def urllc_action_features(a_u: np.ndarray) -> np.ndarray:
    """Saturated action basis (R50: no simple monotonicity above the floor)."""
    a_u = np.asarray(a_u, dtype=float)
    return np.column_stack([np.minimum(a_u, float(URLLC_FLOOR_RBGS)),
                            np.maximum(a_u - float(URLLC_FLOOR_RBGS), 0.0)])


def urllc_design(df: pd.DataFrame) -> tuple[np.ndarray, tuple[int, ...]]:
    """(X, action_col_positions) of the URLLC basis, canonical feature order."""
    state = urllc_state_features(df)
    act = urllc_action_features(df["a_u"].to_numpy(dtype=float))
    act_idx = tuple(URLLC_FEATURES.index(c) for c in URLLC_ACTION_COLS)
    return _place(state, act, len(URLLC_FEATURES), act_idx), act_idx


# ---------------------------------------------------------------------------
# Recalibrated oracle models (fit + counterfactual grids)
# ---------------------------------------------------------------------------

class EmmbQuantileRecal:
    """f_eMBB-II-d: quantile model of the window median per-UE throughput.

    ``predict_grid`` evaluates the counterfactual band over the supported
    a_e grid by factorizing state (constant per window) x action columns —
    numerically identical to evaluating the full basis row by row.
    """

    def __init__(self):
        self.model = PinballQuantileModel()
        self.act_idx = tuple(EMMB_FEATURES.index(c) for c in EMMB_ACTION_COLS)
        self.state_idx = tuple(i for i in range(len(EMMB_FEATURES))
                               if i not in self.act_idx)

    def fit(self, windows: pd.DataFrame) -> "EmmbQuantileRecal":
        d = windows[(windows["n_embb"] >= 1) & windows["thr_embb_med"].notna()]
        X, _ = embb_design(d)
        self.model.fit(X, d["thr_embb_med"].to_numpy(dtype=float))
        return self

    def _state_block(self, n_embb, cqi, queue_embb, n_backlog, offered,
                     prev_thr) -> np.ndarray:
        prev = np.asarray(prev_thr, dtype=float)
        missing = ~np.isfinite(prev)
        cqi = np.asarray(cqi, dtype=float)
        return np.column_stack([
            np.ones(len(cqi)), cqi,
            np.maximum(CQI_LOW_MAX - cqi, 0.0),
            np.asarray(n_embb, dtype=float),
            np.asarray(n_backlog, dtype=float),
            np.asarray(queue_embb, dtype=float) / 1e6,   # qe_mb (frozen scale)
            np.asarray(offered, dtype=float),
            np.nan_to_num(prev, nan=0.0), missing.astype(float),
        ])

    def predict_grid_bands(self, n_embb, cqi, queue_embb, n_backlog, offered,
                           prev_thr, a_e_grid: np.ndarray) -> dict[float, np.ndarray]:
        """Band (q -> (N, K)) of predicted median throughput on the a_e grid.

        Counterfactual evaluation factorizes state (constant per window) x
        the frozen action columns — numerically identical to evaluating the
        full basis row by row.  Zero-quota anchor: ``a_e = 0`` with demand
        predicts 0 Mbps (R27 absolute-quota semantics — nothing is served
        outside the quota; no training rows exist there).
        """
        state = self._state_block(n_embb, cqi, queue_embb, n_backlog, offered,
                                  prev_thr)
        mu, sd = self.model.mu, self.model.sd
        keep = self.model._keep
        z = np.zeros((len(state), len(EMMB_FEATURES)))
        z[:, self.state_idx] = state
        zs = (z - mu * keep) / sd
        out = {}
        for q in self.model.quantiles:
            vals = np.empty((len(state), len(a_e_grid)))
            for k, a_e in enumerate(a_e_grid):
                zk = zs.copy()
                # the model lives in STANDARDIZED space: the raw action
                # block must be standardized with the SAME mu/sd as the fit
                act = embb_action_features(
                    np.full(len(state), float(a_e)),
                    np.asarray(n_embb, dtype=float),
                    np.asarray(cqi, dtype=float))
                ai = list(self.act_idx)
                zk[:, self.act_idx] = (
                    (act - self.model.mu[ai] * self.model._keep[ai])
                    / self.model.sd[ai])
                vals[:, k] = zk @ self.model.betas[q]
            out[q] = vals
        out = {q: np.clip(v, 0.0, EMMB_THR_CAP_MBPS) for q, v in out.items()}
        k0 = int(np.where(np.asarray(a_e_grid) == 0)[0][0]) \
            if (np.asarray(a_e_grid) == 0).any() else -1
        if k0 >= 0:  # physical anchor: a_e = 0 -> nothing served (R27)
            for q in out:
                out[q][:, k0] = 0.0
        # frozen shape constraint: more quota never serves less (R27 absolute
        # quota) — enforce monotone non-decreasing along the a_e grid per
        # quantile (isotonic-by-cumulative-maximum; pre-gate decision)
        for q in out:
            out[q] = np.maximum.accumulate(out[q], axis=1)
        return rearrange_grid(out)

    def predict_point(self, df: pd.DataFrame) -> dict[float, np.ndarray]:
        X, _ = embb_design(df)
        return self.model.predict_quantiles(X)


def rearrange_grid(out: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Rearrangement over the LAST axis (quantile) of a grid dict."""
    qs = sorted(out)
    stack = np.stack([out[q] for q in qs], axis=0)
    stack = np.sort(stack, axis=0)
    return {q: stack[i] for i, q in enumerate(qs)}


class UrllcQuantileRecal:
    """f_URLLC-II-d: quantile model of the window p95 delay (ms)."""

    def __init__(self):
        self.model = PinballQuantileModel()
        self.act_idx = tuple(URLLC_FEATURES.index(c) for c in URLLC_ACTION_COLS)
        self.state_idx = tuple(i for i in range(len(URLLC_FEATURES))
                               if i not in self.act_idx)

    def fit(self, windows: pd.DataFrame) -> "UrllcQuantileRecal":
        d = windows[(windows["n_urllc"] >= 1)
                    & windows["latency_urllc_p95_ms"].notna()]
        X, _ = urllc_design(d)
        self.model.fit(X, d["latency_urllc_p95_ms"].to_numpy(dtype=float))
        return self

    def _state_block(self, n_urllc, n_embb, queue_urllc, queue_embb, n_pkts,
                     dl_slots, prev_p95, prev_miss) -> np.ndarray:
        n_u = np.asarray(n_urllc, dtype=float)
        n_e = np.asarray(n_embb, dtype=float)
        ppu = np.asarray(n_pkts, dtype=float) / np.maximum(n_u, 1.0)
        prev_p95 = np.asarray(prev_p95, dtype=float)
        missing = ~np.isfinite(prev_p95)
        return np.column_stack([
            np.ones(len(n_u)), n_u, (n_u >= 2).astype(float),
            (n_u >= 3).astype(float), n_e, (n_e >= 2).astype(float),
            np.asarray(queue_urllc, dtype=float) / 1e5,   # qu_100k (frozen)
            np.asarray(queue_embb, dtype=float) / 1e6,    # qe_mb  (frozen)
            (ppu < 100.0).astype(float),
            np.asarray(dl_slots, dtype=float),
            np.nan_to_num(prev_p95, nan=0.0),
            np.nan_to_num(np.asarray(prev_miss, dtype=float), nan=0.0) / 100.0,
            missing.astype(float),
        ])

    def predict_grid_bands(self, n_urllc, n_embb, queue_urllc, queue_embb,
                           n_pkts, dl_slots, prev_p95, prev_miss,
                           a_u_grid: np.ndarray) -> dict[float, np.ndarray]:
        state = self._state_block(n_urllc, n_embb, queue_urllc, queue_embb,
                                  n_pkts, dl_slots, prev_p95, prev_miss)
        mu, sd = self.model.mu, self.model.sd
        keep = self.model._keep
        zs = np.zeros((len(state), len(URLLC_FEATURES)))
        zs[:, self.state_idx] = state
        zs = (zs - mu * keep) / sd
        # frozen shape constraint (R43/R50): the post-floor a_u effect is
        # identified in training ONLY by the weak 13<->17 contrast; for the
        # counterfactual grid its contribution is CLIPPED to +/- 2 ms (the
        # measured action-class effect bound at fixed strata).  No clipping
        # happens at the executed actions (|a_u - 13| <= 4 in training).
        j_over = URLLC_FEATURES.index("au_over")
        out = {}
        for q in self.model.quantiles:
            gamma = float(self.model.betas[q][j_over]) / float(sd[j_over])
            vals = np.empty((len(state), len(a_u_grid)))
            for k, a_u in enumerate(a_u_grid):
                zk = zs.copy()
                over = max(float(a_u) - float(URLLC_FLOOR_RBGS), 0.0)
                # standardized action block (same mu/sd as the fit)
                act = urllc_action_features(np.full(len(state), float(a_u)))
                ai = list(self.act_idx)
                zk[:, self.act_idx] = (
                    (act - self.model.mu[ai] * self.model._keep[ai])
                    / self.model.sd[ai])
                p = zk @ self.model.betas[q]
                lin = gamma * over
                vals[:, k] = p - lin + float(np.clip(lin, -2.0, 2.0))
            out[q] = vals
        return rearrange_grid(out)

    def predict_point(self, df: pd.DataFrame) -> dict[float, np.ndarray]:
        X, _ = urllc_design(df)
        return self.model.predict_quantiles(X)


# ---------------------------------------------------------------------------
# mMTC placeholder
# ---------------------------------------------------------------------------

class MmtcFloorPlaceholder:
    """f_mMTC-II-d: PDR = 1.0 constant INSIDE the support (a_m >= a_m_min).

    Measured: PDR(1 s rolling) median = 1.0 at every executed a_m (3 in the
    prop arm, 18 in the equal arm; 35,661 demand-adjacent windows); the II-c
    G3 replays measured the collapse (PDR 0.0-0.67) only at a_m = 0, which
    the supported space FORBIDS.  Actions below the floor carry NO model
    claim — they are simply not in the admissible set (protocol §13/§14).
    """

    def __init__(self, floor: int = MMTC_FLOOR_PLACEHOLDER_RBGS,
                 pdr: float = 1.0):
        self.floor = int(floor)
        self.pdr = float(pdr)

    def band(self, n_mmtc: np.ndarray, a_m: np.ndarray) -> np.ndarray:
        """(N, 3) p25/p50/p75 band; NaN where the action is out of support."""
        n = np.asarray(n_mmtc, dtype=float)
        am = np.asarray(a_m, dtype=float)
        ok = (am >= self.floor) & (n >= 1)
        out = np.full((len(n), 3), np.nan)
        out[ok] = self.pdr
        return out


# ---------------------------------------------------------------------------
# §9 deficits + band gates
# ---------------------------------------------------------------------------

def embb_band(thr_p25: np.ndarray, thr_p75: np.ndarray) -> np.ndarray:
    """§5 band codes for the higher-better eMBB KPI: 0=FEASIBLE 1=MARGINAL 2=INFEASIBLE."""
    out = np.ones(thr_p25.shape, dtype=np.int8)          # MARGINAL default
    out[thr_p25 >= SLA_EMBB_MBPS] = 0
    out[thr_p75 < SLA_EMBB_MBPS] = 2
    return out


def urllc_band(p95_p25: np.ndarray, p95_p75: np.ndarray) -> np.ndarray:
    """§5 band codes for the lower-better URLLC KPI."""
    out = np.ones(p95_p25.shape, dtype=np.int8)
    out[p95_p75 <= SLA_URLLC_MS] = 0
    out[p95_p25 > SLA_URLLC_MS] = 2
    return out


BAND_NAMES = {0: "FEASIBLE", 1: "MARGINAL", 2: "INFEASIBLE"}


# ---------------------------------------------------------------------------
# Leave-one-seed-out gates (§9/§11/§16)
# ---------------------------------------------------------------------------

def _band_disagreement(obs_ok: np.ndarray, band: np.ndarray) -> dict:
    """Band outcome vs the observed 2-class truth at the executed action."""
    ok = np.asarray(obs_ok, dtype=bool)
    f = band == 0
    m = band == 1
    i = band == 2
    n = len(ok)
    return {
        "n": int(n),
        "confident_correct": int(((f & ok) | (i & ~ok)).sum()),
        "absorbed_marginal": int(m.sum()),
        "false_infeasible": int((i & ok).sum()),
        "false_feasible": int((f & ~ok).sum()),
        "false_infeasible_rate": (float((i & ok).sum() / max(i.sum(), 1))
                                  if i.sum() else np.nan),
        "false_feasible_rate": (float((f & ~ok).sum() / max(f.sum(), 1))
                                if f.sum() else np.nan),
    }


def loo_gates(windows: pd.DataFrame) -> dict:
    """Leave-one-seed-out §9/§11/§16 gates for both recalibrated models."""
    rows_e, rows_u = [], []
    for held in sorted(windows["seed"].unique()):
        tr = windows[windows["seed"] != held]
        te = windows[windows["seed"] == held].copy()
        me = EmmbQuantileRecal().fit(tr)
        mu = UrllcQuantileRecal().fit(tr)

        de = te[te["n_embb"] >= 1].copy()
        Xe, _ = embb_design(de)
        qe = me.model.predict_quantiles(Xe)
        obs = de["thr_embb_med"].to_numpy(dtype=float)
        band_e = embb_band(qe[0.25], qe[0.75])
        near = np.abs(obs - SLA_EMBB_MBPS) <= EMMB_NEAR_HALF_WIDTH_MBPS
        cqi_low = de["cqi_eff"].to_numpy(dtype=float) <= CQI_LOW_MAX
        cov = float(((obs >= qe[0.25]) & (obs <= qe[0.75])).mean())
        g_e = _band_disagreement(obs >= SLA_EMBB_MBPS, band_e)
        rows_e.append({
            "held_seed": int(held), "n": len(de),
            "mae": float(np.abs(qe[0.5] - obs).mean()),
            "rmse": float(np.sqrt(((qe[0.5] - obs) ** 2).mean())),
            "near_n": int(near.sum()),
            "near_mae": float(np.abs((qe[0.5] - obs)[near]).mean())
            if near.any() else np.nan,
            "cqi_low_n": int(cqi_low.sum()),
            "cqi_low_mae": float(np.abs((qe[0.5] - obs)[cqi_low]).mean())
            if cqi_low.any() else np.nan,
            "coverage_p25_p75": cov,
            "near_band_absorbed_marginal_rate": (
                float((band_e[near] == 1).mean()) if near.any() else np.nan),
            "near_false_infeasible": int(((band_e == 2) & near
                                          & (obs >= SLA_EMBB_MBPS)).sum()),
            "near_false_feasible": int(((band_e == 0) & near
                                        & (obs < SLA_EMBB_MBPS)).sum()),
            **{f"band_{k}": v for k, v in g_e.items()},
        })

        du = te[te["n_urllc"] >= 1].copy()
        Xu, _ = urllc_design(du)
        qu = mu.model.predict_quantiles(Xu)
        obsu = du["latency_urllc_p95_ms"].to_numpy(dtype=float)
        band_u = urllc_band(qu[0.25], qu[0.75])
        nearu = np.abs(obsu - SLA_URLLC_MS) <= URLLC_NEAR_HALF_WIDTH_MS
        covu = float(((obsu >= qu[0.25]) & (obsu <= qu[0.75])).mean())
        g_u = _band_disagreement(obsu <= SLA_URLLC_MS, band_u)
        rows_u.append({
            "held_seed": int(held), "n": len(du),
            "mae": float(np.abs(qu[0.5] - obsu).mean()),
            "rmse": float(np.sqrt(((qu[0.5] - obsu) ** 2).mean())),
            "near_n": int(nearu.sum()),
            "near_mae": float(np.abs((qu[0.5] - obsu)[nearu]).mean())
            if nearu.any() else np.nan,
            "coverage_p25_p75": covu,
            "near_band_absorbed_marginal_rate": (
                float((band_u[nearu] == 1).mean()) if nearu.any() else np.nan),
            "near_false_infeasible": int(((band_u == 2) & nearu
                                          & (obsu <= SLA_URLLC_MS)).sum()),
            "near_false_feasible": int(((band_u == 0) & nearu
                                        & (obsu > SLA_URLLC_MS)).sum()),
            **{f"band_{k}": v for k, v in g_u.items()},
        })
    return {"embb": pd.DataFrame(rows_e), "urllc": pd.DataFrame(rows_u)}
