"""Semi-analytic KPI models of the offline oracle (protocol section 6).

Three models, PREFERRING data-calibrated stratum lookups over heavy
regression (ask item 2):

f_eMBB — ``EmmbShareLookup``: per-UE eMBB throughput as a function of the
per-UE share s = a_e / max(N_e, 1) (RBGs of quota per active eMBB UE) with an
optional wideband-CQI refinement.  Rationale (measured, Etapa 2.3 data): the
scheduler serves the whole backlogged slice quota every window (quota
utilization = 1.0 in the equal arm), so the delivered cell throughput is set
by the quota; per-UE throughput follows the share; the per-RBG YIELD degrades
when the per-UE share collapses (measured: yield 0.41 -> 0.18 Mbps/RBG as
s goes 9 -> 4.5 at equal) — hence a LOOKUP on s, not a constant efficiency.
The prediction is the per-UE throughput; the frozen R36 window judgment
(median UE-window >= 3.5 Mbps) is applied on exactly that quantity.

f_URLLC — ``UrllcStrataLookup``: window p95 (ms) and deadline-miss% of URLLC
packet delays as a lookup over strata (N_urllc, N_embb with demand).  The
Etapa-2.2/2.3 evidence (R39 v1.7-cal2 paired gate; lowload §11.3 diagnosis;
the equal-vs-prop stratification of the 2.3 sweep) shows the deadline tail is
dominated by intra-slice per-UE contention (N_urllc) with a small residual
action-class effect (<= ~2 ms p95 at fixed strata between the two executed
classes), and is NOT quota-bound above the observed floors — so the primary
model is ACTION-INDEPENDENT within the explored support
(``[ASSUMIDO-ORACLE]`` R43).  This is the CONSERVATIVE choice for
avoidability claims: it never credits an allocation with a URLLC improvement
the data cannot demonstrate.

f_mMTC — ``MmtcConstantPdr``: PDR = 1.0 constant.  Measured: aggregate mMTC
PDR = 1.0 in ALL 30 runs up to lambda=7 (offered 0.0016 Mbps/UE, R35 —
orders of magnitude below capacity; R38 criterion never approached).

Support region + out-of-support handling are explicit (protocol section 6 /
gate of section 16): every lookup records its per-cell support counts;
predictions outside the calibrated support fall back to the nearest supported
cell (flat extrapolation, ``[ASSUMIDO-ORACLE]``) and the caller can query
``in_support`` to flag feasibility claims that rest on extrapolated cells.

Validation: leave-one-seed-out (train on 2 seeds, predict the 3rd; ALL loads
and BOTH arms pooled), reporting global error and error NEAR the SLA
thresholds (protocol section 6), plus the executed-outcome consistency rate
(no false infeasible on windows the executed arm actually satisfied — the
offline analogue of the section 16 confusion matrix).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nsoran.sqr.oracle.dataset import SLA_EMBB_MBPS, SLA_MMTC_PDR, SLA_URLLC_MS

# ---------------------------------------------------------------------------
# Stratum edges (frozen, documented in the model card)
# ---------------------------------------------------------------------------
#: Per-UE share s = a_e / N_e (RBG/UE) bin EDGES (left-closed).
EMMB_SHARE_EDGES: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 27.0, 40.0, 1e9)
#: CQI bins (raw wideband 1-15; the campaign sits at 13-15).
CQI_EDGES: tuple[float, ...] = (0.0, 12.5, 13.5, 14.5, 15.5)
#: N_urllc bins: 1, 2, 3+ (the tail decomp convention).
NU_EDGES: tuple[float, ...] = (0.5, 1.5, 2.5, 1e9)
#: N_embb bins: 0, 1, 2+ (demand-active eMBB UEs).
NE_EDGES: tuple[float, ...] = (-0.5, 0.5, 1.5, 1e9)


def _bin_index(values: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    return np.digitize(values, edges[1:-1], right=False)


@dataclass
class EmmbShareLookup:
    """f_eMBB: median per-UE throughput lookup on (CQI bin, share s = a_e/N_e).

    Fit target: the window's MEDIAN ``windowThroughputMbps`` over eMBB
    UE-windows with demand (the R36 judgment quantity).  Prediction for an
    arbitrary action: ``predict(a_e, n_embb, cqi)``.

    Out-of-support: a (cqi_bin, s_bin) cell with no training windows falls
    back to the s-bin MARGINAL median (over cqi); an s beyond the calibrated
    range is CLAMPED to the edge bin (flat extrapolation — capacity
    saturation [ASSUMIDO-ORACLE]); ``in_support`` reports both cases.
    """

    share_edges: tuple[float, ...] = EMMB_SHARE_EDGES
    cqi_edges: tuple[float, ...] = CQI_EDGES
    table: pd.DataFrame | None = None
    marginals: dict | None = None
    support: pd.DataFrame | None = None
    fallback_cqi: float = 14.0

    #: per-cell quantile columns served by :meth:`predict_quantile`
    QUANTILES = (0.25, 0.5, 0.75)

    def fit(self, windows: pd.DataFrame) -> "EmmbShareLookup":
        d = windows[(windows["n_embb"] >= 1) & windows["thr_embb_med"].notna()].copy()
        d = d[d["a_e"] > 0]
        d = d[d["cqi_embb"].notna()]
        d["s"] = d["a_e"] / d["n_embb"]
        d["sb"] = _bin_index(d["s"].values, self.share_edges)
        d["cb"] = _bin_index(d["cqi_embb"].values, self.cqi_edges)
        g = d.groupby(["cb", "sb"])["thr_embb_med"].agg(
            thr="median",
            thr_p25=lambda x: float(np.percentile(x, 25)),
            thr_p75=lambda x: float(np.percentile(x, 75)),
            n="size",
        )
        self.table = g.reset_index()
        nb = len(self.share_edges) - 1
        marg = {}
        for col in ("thr", "thr_p25", "thr_p75"):
            m = d.groupby("sb")["thr_embb_med"].agg(
                median="median",
                p25=lambda x: float(np.percentile(x, 25)),
                p75=lambda x: float(np.percentile(x, 75)),
            )
            arr = m["median"].reindex(range(nb)).values.astype(float)
            if col == "thr_p25":
                arr = m["p25"].reindex(range(nb)).values.astype(float)
            elif col == "thr_p75":
                arr = m["p75"].reindex(range(nb)).values.astype(float)
            marg[col] = self._fill_nearest(arr)
        self.marginals = marg
        self.support = g.reset_index().rename(columns={"n": "n_windows"})
        self.fallback_cqi = float(d["cqi_embb"].median())
        return self

    def _fill_nearest(self, arr: np.ndarray) -> np.ndarray:
        """Flat-fill share bins without support from the nearest supported bin
        (capacity-saturation extrapolation, [ASSUMIDO-ORACLE R42])."""
        arr = np.asarray(arr, dtype=float)
        if np.isnan(arr).any():
            valid = np.nonzero(np.isfinite(arr))[0]
            if len(valid):
                centers = 0.5 * (np.asarray(self.share_edges[:-1])
                                 + np.asarray(self.share_edges[1:]))
                centers[-1] = self.share_edges[-2] + 1.0
                arr = np.interp(centers, centers[valid], arr[valid])
            else:  # pragma: no cover — fit() with no data at all
                arr = np.zeros_like(arr)
        return arr

    def _cells(self, a_e: np.ndarray, n_embb: np.ndarray, cqi: np.ndarray):
        cqi = np.where(np.isfinite(cqi), cqi, self.fallback_cqi)
        s = a_e / np.maximum(n_embb, 1)
        sb = _bin_index(s, self.share_edges)
        cb = _bin_index(cqi, self.cqi_edges)
        return s, sb, cb

    def predict(self, a_e, n_embb, cqi) -> np.ndarray:
        """Predicted per-UE eMBB throughput (Mbps), median quantile; 0 if a_e==0."""
        vals, _ = self.predict_with_support(a_e, n_embb, cqi)
        return vals

    def predict_quantile(self, a_e, n_embb, cqi, q: float = 0.5) -> np.ndarray:
        """Predicted per-UE eMBB throughput at quantile q of the stratum.

        q=0.5 == :meth:`predict`; q=0.25/0.75 serve the gate-sensitivity
        analysis (optimistic/pessimistic feasibility, §16 false-infeasible).
        """
        col = {0.25: "thr_p25", 0.5: "thr", 0.75: "thr_p75"}[q]
        a_e = np.asarray(a_e, dtype=float)
        n_embb = np.asarray(n_embb, dtype=float)
        cqi = np.asarray(cqi, dtype=float)
        _, sb, cb = self._cells(a_e, n_embb, cqi)
        key = {(int(r.cb), int(r.sb)): float(getattr(r, col))
               for r in self.table.itertuples()}
        marg = self.marginals[col]
        vals = np.zeros(len(sb), dtype=float)
        for i in range(len(sb)):
            if n_embb[i] <= 0:
                vals[i] = np.nan  # no eMBB demand: KPI undefined
                continue
            if a_e[i] <= 0:
                continue
            v = key.get((int(cb[i]), int(sb[i])))
            vals[i] = v if v is not None else marg[int(sb[i])]
        return vals

    def predict_with_support(self, a_e, n_embb, cqi):
        """Like :meth:`predict` but also returns the in-support boolean mask.

        A prediction is IN support when its (cqi_bin, s_bin) cell had training
        windows AND the raw share s lies inside the calibrated share range.
        """
        a_e = np.asarray(a_e, dtype=float)
        n_embb = np.asarray(n_embb, dtype=float)
        cqi = np.asarray(cqi, dtype=float)
        s, sb, cb = self._cells(a_e, n_embb, cqi)
        key = {(int(r.cb), int(r.sb)): float(r.thr) for r in self.table.itertuples()}
        marg = self.marginals["thr"]
        vals = np.zeros(s.shape, dtype=float)
        supp = np.zeros(s.shape, dtype=bool)
        smin = self.share_edges[0]
        smax = self.share_edges[-2]
        for i in range(len(s)):
            if n_embb[i] <= 0:
                vals[i] = np.nan          # no eMBB demand: KPI undefined
                supp[i] = True            # vacuously supported (no claim made)
                continue
            if a_e[i] <= 0:
                vals[i] = 0.0
                supp[i] = True
                continue
            v = key.get((int(cb[i]), int(sb[i])))
            in_rng = (s[i] >= smin) and (s[i] <= smax)
            if v is not None:
                vals[i] = v
                supp[i] = in_rng
            else:
                vals[i] = marg[int(sb[i])]
                supp[i] = False
        return vals, supp


@dataclass
class UrllcStrataLookup:
    """f_URLLC: window p95 (ms) and miss% lookup on (N_urllc, N_embb) strata.

    Primary target: window p95 of URLLC packet delays (ms); the frozen R37
    window judgment is ``p95 <= 10 ms``.  The per-window miss% lookup rides
    along (same strata) for reporting.  Windows with no URLLC demand are
    vacuously satisfied.  Action-independent by design — see module docstring.
    """

    nu_edges: tuple[float, ...] = NU_EDGES
    ne_edges: tuple[float, ...] = NE_EDGES
    table: pd.DataFrame | None = None
    nub_marginal: pd.Series | None = None
    p95_fallback: float = 10.0

    def fit(self, windows: pd.DataFrame) -> "UrllcStrataLookup":
        d = windows[(windows["n_urllc"] >= 1) & windows["latency_urllc_p95_ms"].notna()]
        d = d.copy()
        d["nub"] = _bin_index(d["n_urllc"].values, self.nu_edges)
        d["neb"] = _bin_index(d["n_embb"].values, self.ne_edges)
        g = d.groupby(["nub", "neb"]).agg(
            p95_med=("latency_urllc_p95_ms", "median"),
            p95_p25=("latency_urllc_p95_ms", lambda x: float(np.percentile(x, 25))),
            p95_p75=("latency_urllc_p95_ms", lambda x: float(np.percentile(x, 75))),
            miss_med=("latency_urllc_miss_pct", "median"),
            n_windows=("latency_urllc_p95_ms", "size"),
        )
        self.table = g.reset_index()
        self.nub_marginal = d.groupby("nub")["latency_urllc_p95_ms"].median()
        self.p95_fallback = float(d["latency_urllc_p95_ms"].median())
        return self

    def cells(self, n_urllc, n_embb):
        nub = _bin_index(np.asarray(n_urllc, dtype=float), self.nu_edges)
        neb = _bin_index(np.asarray(n_embb, dtype=float), self.ne_edges)
        return nub, neb

    def predict(self, n_urllc, n_embb, q: float = 0.5):
        """(p95_hat ms, miss_hat %) per window; vacuous (0, 0) without demand.

        ``q`` selects the stratum quantile of the p95 (0.5 median primary;
        0.25/0.75 for the gate-sensitivity analysis).
        """
        col = {0.25: "p95_p25", 0.5: "p95_med", 0.75: "p95_p75"}[q]
        n_urllc = np.asarray(n_urllc, dtype=float)
        n_embb = np.asarray(n_embb, dtype=float)
        nub, neb = self.cells(n_urllc, n_embb)
        key = {(int(r.nub), int(r.neb)): float(getattr(r, col))
               for r in self.table.itertuples()}
        key_miss = {(int(r.nub), int(r.neb)): float(r.miss_med)
                    for r in self.table.itertuples()}
        marg = self.nub_marginal if self.nub_marginal is not None else pd.Series(dtype=float)
        p95 = np.zeros(len(nub), dtype=float)
        miss = np.zeros(len(nub), dtype=float)
        for i in range(len(nub)):
            if n_urllc[i] < 1:
                continue  # no URLLC demand -> vacuously satisfied
            v = key.get((int(nub[i]), int(neb[i])))
            if v is None:
                # unseen stratum: fall back to the N_urllc-bin MARGINAL median,
                # else the global median (documented out-of-support handling)
                m = marg.get(int(nub[i]))
                p95[i] = float(m) if m is not None and np.isfinite(m) else self.p95_fallback
            else:
                p95[i] = v
                miss[i] = key_miss.get((int(nub[i]), int(neb[i])), 0.0)
        return p95, miss


@dataclass
class MmtcConstantPdr:
    """f_mMTC: PDR = 1.0 constant (measured in all 30 runs; R38 never approached)."""

    pdr: float = 1.0
    measured_min_pdr: float = 1.0

    def fit(self, windows: pd.DataFrame) -> "MmtcConstantPdr":
        p = windows["pdr_mmtc"].dropna()
        if len(p):
            self.measured_min_pdr = float(p.min())
        # The SLA model stays 1.0 even if a stray window dipped: the R38
        # criterion is the SESSION/aggregate PDR, and the aggregate of every
        # run is 1.0 (frozen 2.3 table).  The minimum observed window PDR is
        # carried for honesty in the model card.
        self.pdr = 1.0
        return self

    def predict(self, n_mmtc) -> np.ndarray:
        n = np.asarray(n_mmtc, dtype=float)
        return np.where(n >= 1, self.pdr, np.nan)  # vacuous without demand


# ---------------------------------------------------------------------------
# Leave-one-seed-out validation (protocol section 6)
# ---------------------------------------------------------------------------

def loo_validate(windows: pd.DataFrame) -> dict:
    """Leave-one-seed-out validation of the KPI models at the EXECUTED action.

    Returns a dict with, per model and per held-out seed:
    - global MAE / RMSE of the KPI prediction;
    - MAE NEAR the SLA threshold (eMBB: observed within [2.8, 4.2] Mbps;
      URLLC: observed p95 within [8, 12] ms);
    - gate agreement at the executed action (accuracy / false-infeasible /
      false-feasible of the per-window SLA predicate);
    - executed-outcome consistency: of the windows where the EXECUTED arm
      satisfied ALL THREE frozen SLAs, the fraction the oracle classifies
      jointly feasible (the offline analogue of the section 16 TP rate; a
      low value means systematic FALSE-INFEASIBLE — the section 16 STOP gate).
    """
    seeds = sorted(windows["seed"].unique())
    rows = []
    consistency = []
    loo_pred_frames = []
    for held in seeds:
        tr = windows[windows["seed"] != held]
        te = windows[windows["seed"] == held]
        me = EmmbShareLookup().fit(tr)
        mu = UrllcStrataLookup().fit(tr)
        mm = MmtcConstantPdr().fit(tr)
        te = te.copy()
        te["thr_hat"], te["thr_supp"] = me.predict_with_support(
            te["a_e"].values, te["n_embb"].values, te["cqi_embb"].values)
        te["p95_hat"], te["miss_hat"] = mu.predict(te["n_urllc"].values,
                                                   te["n_embb"].values)
        te["pdr_hat"] = mm.predict(te["n_mmtc"].values)

        em = te[te["n_embb"] >= 1]
        near = em[em["thr_embb_med"].between(2.8, 4.2)]
        gate_obs = em["thr_embb_med"] >= SLA_EMBB_MBPS
        gate_hat = em["thr_hat"] >= SLA_EMBB_MBPS
        ur = te[te["n_urllc"] >= 1]
        near_u = ur[ur["latency_urllc_p95_ms"].between(8.0, 12.0)]
        gate_obs_u = ur["latency_urllc_p95_ms"] <= SLA_URLLC_MS
        gate_hat_u = ur["p95_hat"] <= SLA_URLLC_MS

        def _agree(obs, hat):
            obs = obs.astype(bool)
            hat = hat.astype(bool)
            acc = float((obs == hat).mean()) if len(obs) else np.nan
            fp = float((~obs & hat).mean()) if len(obs) else np.nan  # false feasible
            fn = float((obs & ~hat).mean()) if len(obs) else np.nan  # false infeasible
            return acc, fp, fn

        acc_e, fp_e, fn_e = _agree(gate_obs, gate_hat)
        acc_u, fp_u, fn_u = _agree(gate_obs_u, gate_hat_u)

        loo_pred_frames.append(te[[
            "arm", "lam", "seed", "win", "a_e", "a_u", "a_m",
            "n_embb", "n_urllc", "n_mmtc", "cqi_embb",
            "thr_embb_med", "thr_hat", "thr_supp",
            "latency_urllc_p95_ms", "p95_hat", "miss_hat", "pdr_mmtc"]].copy())

        row = {
            "held_seed": int(held),
            "embb_n": int(len(em)),
            "embb_mae": float(np.abs(em["thr_hat"] - em["thr_embb_med"]).mean()),
            "embb_rmse": float(np.sqrt(((em["thr_hat"] - em["thr_embb_med"]) ** 2).mean())),
            "embb_near_n": int(len(near)),
            "embb_near_mae": float(np.abs(near["thr_hat"] - near["thr_embb_med"]).mean())
            if len(near) else np.nan,
            "embb_gate_acc": acc_e, "embb_gate_false_feasible": fp_e,
            "embb_gate_false_infeasible": fn_e,
            "embb_out_of_support_frac": float(1.0 - em["thr_supp"].mean()),
            "urllc_n": int(len(ur)),
            "urllc_mae": float(np.abs(ur["p95_hat"] - ur["latency_urllc_p95_ms"]).mean()),
            "urllc_near_n": int(len(near_u)),
            "urllc_near_mae": float(np.abs(near_u["p95_hat"] - near_u["latency_urllc_p95_ms"]).mean())
            if len(near_u) else np.nan,
            "urllc_gate_acc": acc_u, "urllc_gate_false_feasible": fp_u,
            "urllc_gate_false_infeasible": fn_u,
        }
        rows.append(row)

        # executed-outcome consistency (all three SLAs observed satisfied)
        v = te[["observed_embb_violation", "observed_urllc_violation",
                "observed_mmtc_violation"]]
        all_ok = v.fillna(0).eq(0).all(axis=1) & v.notna().any(axis=1)
        embb_hat_ok = np.where(te["n_embb"] >= 1, te["thr_hat"] >= SLA_EMBB_MBPS, True)
        urllc_hat_ok = np.where(te["n_urllc"] >= 1, te["p95_hat"] <= SLA_URLLC_MS, True)
        mmtc_hat_ok = np.where(te["n_mmtc"] >= 1, te["pdr_hat"] >= SLA_MMTC_PDR, True)
        feasible_hat = embb_hat_ok & urllc_hat_ok & mmtc_hat_ok
        if all_ok.sum():
            consistency.append({
                "held_seed": int(held),
                "windows_all_ok_executed": int(all_ok.sum()),
                "predicted_feasible": int((all_ok & feasible_hat).sum()),
                "false_infeasible": int((all_ok & ~feasible_hat).sum()),
                "consistency_rate": float((all_ok & feasible_hat).sum() / all_ok.sum()),
            })
    loo = pd.DataFrame(rows)
    cons = pd.DataFrame(consistency)
    preds = (pd.concat(loo_pred_frames, ignore_index=True)
             if loo_pred_frames else pd.DataFrame())
    return {"loo": loo, "consistency": cons, "loo_windows": preds}


def fit_models(windows: pd.DataFrame) -> dict:
    """Fit the three models on the full window dataset (final calibration)."""
    return {
        "embb": EmmbShareLookup().fit(windows),
        "urllc": UrllcStrataLookup().fit(windows),
        "mmtc": MmtcConstantPdr().fit(windows),
    }
