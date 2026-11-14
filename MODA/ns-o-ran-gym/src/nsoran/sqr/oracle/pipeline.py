"""Despacho-I pipeline of the Etapa-2.4 offline oracle.

Orchestrates (protocol section 33, items 1-6):
1. per-window dataset over the 30 frozen 2.3 runs (:mod:`dataset`);
2. KPI models + leave-one-seed-out validation (:mod:`models`);
3. action simplex + F_t per window (:mod:`simplex`);
4. metrics P_inf / AVR / UVR / |F_t| / D_min, the section 19/20/28 tables,
   the section 21 figures (with the exact section 28 names) and the section
   15 candidate list of critical windows for the Despacho-II replay;
5. sanity checks SC1-SC4 (section 22) and hypotheses H2.4-1..4 (section 23).

NO ns-3 simulation is launched; ``executable_validation/`` belongs to
Despacho II.  All artifacts land under
``results/sqr_characterization_round2/oracle/`` (models/ predictions/ plots/
tables/ reports/).
"""

from __future__ import annotations

import glob
import os
import re
import time

import numpy as np
import pandas as pd

from nsoran.sqr.oracle import dataset as ds
from nsoran.sqr.oracle.models import CQI_EDGES, _bin_index, fit_models, loo_validate
from nsoran.sqr.oracle.simplex import (C_PRB, enumeration_timing,
                                       window_feasibility)

#: Witness URLLC quota: the observed support floor (prop arm floor, R39) —
#: keeps the representative feasible action inside the explored action region.
WITNESS_URLLC_RBGS = 13


# ---------------------------------------------------------------------------
# Classification of every window (cached per unique state key — exact)
# ---------------------------------------------------------------------------

def classify_windows(windows: pd.DataFrame, models: dict, c: int = C_PRB) -> pd.DataFrame:
    """Compute F_t / a* / saveability / avoidability for every window.

    Windows sharing the exact model state key (n_embb, n_urllc, n_mmtc,
    cqi_bin of the eMBB median CQI) share the same F_t BY CONSTRUCTION of the
    calibrated models; the per-window routine is exact, so computing it once
    per unique key and fanning out is an exact cache — every window is still
    classified from its own z_t.  The total exhaustive-enumeration cost is
    reported separately (``enumeration_timing``).
    """
    d = windows.copy().reset_index(drop=True)
    cqi = d["cqi_embb"].fillna(d.groupby("tag")["cqi_embb"].transform("median"))
    d["cqi_eff"] = cqi.fillna(14.0)
    d["cqi_bin"] = _bin_index(d["cqi_eff"].values, CQI_EDGES)
    key_frame = d[["n_embb", "n_urllc", "n_mmtc"]].fillna(0).astype("int64")
    key_frame["cqi_bin"] = d["cqi_bin"]
    d["_key"] = key_frame.astype(str).agg("|".join, axis=1)
    d[["n_embb", "n_urllc", "n_mmtc"]] = key_frame[["n_embb", "n_urllc", "n_mmtc"]]

    cache: dict[str, object] = {}
    for key, g in d.groupby("_key", sort=True):
        f = g.iloc[0]
        cache[key] = window_feasibility(
            models["embb"], models["urllc"], models["mmtc"],
            n_embb=int(f["n_embb"]), n_urllc=int(f["n_urllc"]),
            n_mmtc=int(f["n_mmtc"]), cqi=float(f["cqi_eff"]), c=c)

    res = [cache[k] for k in d["_key"]]
    out = pd.DataFrame(index=d.index)
    out["joint_feasible"] = [r.joint_feasible for r in res]
    out["num_feasible_actions"] = [r.num_feasible_actions for r in res]
    out["e_min"] = [r.e_min for r in res]
    out["urllc_gate"] = [r.urllc_gate for r in res]
    # model prediction of the eMBB KPI AT THE EXECUTED action (SC3 breakdown;
    # the URLLC prediction is action-independent -> urllc_gate covers it)
    key_thr_exec: dict[str, dict[int, float]] = {}
    for key, g in d.groupby("_key", sort=True):
        f = g.iloc[0]
        grid = models["embb"].predict(
            np.arange(C_PRB + 1, dtype=float),
            np.full(C_PRB + 1, float(f["n_embb"])),
            np.full(C_PRB + 1, float(f["cqi_eff"])))
        key_thr_exec[key] = {int(a): float(grid[a])
                             for a in g["a_e"].fillna(0).astype(int).unique()}
    out["thr_hat_exec"] = [
        key_thr_exec[k][int(a)] for k, a in zip(d["_key"], d["a_e"].fillna(0).astype(int))]
    out[["witness_a_e", "witness_a_u", "witness_a_m"]] = [
        (r.witness if r.witness else (-1, -1, -1)) for r in res]
    out[["a_star_e", "a_star_u", "a_star_m"]] = [r.best_deg_action for r in res]
    out["thr_hat_witness"] = [r.witness_kpis[0] if r.witness_kpis else np.nan for r in res]
    out["p95_hat_witness"] = [r.witness_kpis[1] if r.witness_kpis else np.nan for r in res]
    out["pdr_hat_witness"] = [r.witness_kpis[2] if r.witness_kpis else np.nan for r in res]
    out["thr_hat_star"] = [r.best_deg_kpis[0] for r in res]
    out["p95_hat_star"] = [r.best_deg_kpis[1] for r in res]
    out["pdr_hat_star"] = [r.best_deg_kpis[2] for r in res]
    out["min_total_deficit"] = [r.min_total_deficit for r in res]
    out["min_deficit_embb"] = [r.deficits[0] for r in res]
    out["min_deficit_urllc"] = [r.deficits[1] for r in res]
    out["min_deficit_mmtc"] = [r.deficits[2] for r in res]
    out["best_deg_n_ties"] = [r.n_ties_best_deg for r in res]

    # saveability is a property of INFEASIBLE states (§13): mask feasible ones
    feas = out["joint_feasible"].values
    for i, sl in enumerate(("embb", "urllc", "mmtc")):
        vals = [r.saveable[i] for r in res]
        out[f"{sl}_individually_saveable"] = np.where(feas, np.nan,
                                                      np.asarray(vals, dtype=float))

    # avoidability (§3, §13): observed slice violation AND F_t != empty
    for sl in ("embb", "urllc", "mmtc"):
        v = d[f"observed_{sl}_violation"]
        out[f"avoidable_{sl}"] = ((v == 1) & feas).astype(float).where(v.notna())
    vany = d["observed_any_violation"]
    out["avoidable_any"] = ((vany == 1) & feas).astype(float).where(vany.notna())

    # gate sensitivity (§16 false-infeasible concern): joint feasibility under
    # optimistic (p25) and pessimistic (p75) predicted-KPI quantiles — same
    # exact per-key cache; feasibility flag only
    for col, (eq, uq) in {"joint_feasible_p25": (0.25, 0.25),
                          "joint_feasible_p75": (0.75, 0.75)}.items():
        cq: dict[str, bool] = {}
        for key, g in d.groupby("_key", sort=True):
            f = g.iloc[0]
            cq[key] = window_feasibility(
                models["embb"], models["urllc"], models["mmtc"],
                n_embb=int(f["n_embb"]), n_urllc=int(f["n_urllc"]),
                n_mmtc=int(f["n_mmtc"]), cqi=float(f["cqi_eff"]),
                embb_q=eq, urllc_q=uq).joint_feasible
        out[col] = [cq[k] for k in d["_key"]]

    return pd.concat([d, out], axis=1)


# ---------------------------------------------------------------------------
# Aggregations: protocol §12 metrics + §20 tables
# ---------------------------------------------------------------------------

def _mean_or_nan(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean()) if len(x) else np.nan


def aggregate_by_lambda(classified: pd.DataFrame) -> pd.DataFrame:
    """§12 metrics + §20 'Tabela geral' (one row per lambda, both arms inside)."""
    rows = []
    for lam, g in classified.groupby("lam"):
        row = {"lambda_total": float(lam), "n_windows": int(len(g)),
               "P_inf": float((~g["joint_feasible"]).mean()),
               "P_feas": float(g["joint_feasible"].mean()),
               "P_inf_gate_p25": float((~g["joint_feasible_p25"]).mean()),
               "P_inf_gate_p75": float((~g["joint_feasible_p75"]).mean())}
        for arm, ga in g.groupby("arm"):
            row[f"AVR_{arm}"] = _mean_or_nan(ga["avoidable_any"])
            for sl in ("embb", "urllc", "mmtc"):
                row[f"AVR_{arm}_{sl}"] = _mean_or_nan(ga[f"avoidable_{sl}"])
            row[f"n_windows_{arm}"] = int(len(ga))
        feas = g.loc[g["joint_feasible"], "num_feasible_actions"]
        row["fset_median"] = float(feas.median()) if len(feas) else np.nan
        row["fset_p25"] = float(feas.quantile(0.25)) if len(feas) else np.nan
        row["fset_p75"] = float(feas.quantile(0.75)) if len(feas) else np.nan
        inf = g[~g["joint_feasible"]]
        row["n_infeasible"] = int(len(inf))
        row["Dmin_median_infeasible"] = (float(inf["min_total_deficit"].median())
                                         if len(inf) else np.nan)
        row["Dmin_mean_infeasible"] = (float(inf["min_total_deficit"].mean())
                                       if len(inf) else np.nan)
        # UVR (§12.4): P(V_t=1 | F_t=∅), per arm
        for arm, ga in inf.groupby("arm"):
            row[f"UVR_{arm}"] = _mean_or_nan(ga["observed_any_violation"])
        # regime label (§18); 'normal' vs 'marginal' by the |F_t| median inside
        # the lambda's own feasible distribution (upper tercile = comfortable).
        pinf = row["P_inf"]
        if pinf <= 0.05:
            comfy = float(feas.quantile(2.0 / 3.0)) if len(feas) else np.nan
            row["regime"] = "normal" if (np.isfinite(comfy) and row["fset_median"] >= comfy) else "marginal"
        elif pinf >= 0.95:
            row["regime"] = "overload_severo"
        else:
            row["regime"] = "overload_transitorio"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("lambda_total").reset_index(drop=True)


def aggregate_by_slice(classified: pd.DataFrame) -> pd.DataFrame:
    """§20 'Tabela por slice' (+ saveable rates among infeasible windows)."""
    rows = []
    for lam, g in classified.groupby("lam"):
        row = {"lambda_total": float(lam)}
        inf = g[~g["joint_feasible"]]
        for sl in ("embb", "urllc", "mmtc"):
            row[f"AVR_{sl}"] = _mean_or_nan(g[f"avoidable_{sl}"])
            sv = inf[f"{sl}_individually_saveable"].dropna()
            row[f"saveable_{sl}_rate_infeasible"] = float(sv.mean()) if len(sv) else np.nan
            row[f"n_infeasible_saveable_den_{sl}"] = int(len(sv))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("lambda_total").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sanity checks (§22) and hypotheses (§23)
# ---------------------------------------------------------------------------

def sanity_checks(classified: pd.DataFrame, by_lambda: pd.DataFrame) -> dict:
    sc = {}
    bl = by_lambda.set_index("lambda_total")
    pins = bl["P_inf"]
    spear = float(pins.corr(pd.Series(pins.index, index=pins.index), method="spearman"))
    sc["SC1"] = {
        "P_inf_by_lambda": {float(k): round(float(v), 4) for k, v in pins.items()},
        "spearman_pinf_vs_lambda": round(spear, 4),
        "passed": bool(spear > 0.8),
        "note": "tendência crescente exigida; monotonicidade perfeita não exigida (§22)",
    }
    p2 = float(bl.loc[2.0, "P_inf"]) if 2.0 in bl.index else float("nan")
    sc["SC2"] = {
        "P_inf_lambda2": round(p2, 4),
        "threshold": 0.05,
        "passed": bool(np.isfinite(p2) and p2 <= 0.05),
        "note": ("'≈0' operacionalizado como <= 0.05; SLAs/cenário congelados — "
                 "se alto, o diagnóstico deve identificar a slice dominante"),
    }
    v = classified[["observed_embb_violation", "observed_urllc_violation",
                    "observed_mmtc_violation"]]
    all_ok = v.fillna(0).eq(0).all(axis=1) & v.notna().any(axis=1)
    n_ok = int(all_ok.sum())
    infeas = ~classified["joint_feasible"].values
    n_bad = int((all_ok & infeas).sum())
    bad_mask = all_ok.values & infeas
    # breakdown: which model gate closes F_t despite the executed OK outcome?
    thr_hat_exec = classified["thr_hat_exec"]
    embb_model_closes = (classified["n_embb"] >= 1) & (thr_hat_exec < 3.5)
    urllc_model_closes = ~classified["urllc_gate"].values.astype(bool)
    n_e = int((bad_mask & embb_model_closes.values & ~urllc_model_closes).sum())
    n_u = int((bad_mask & urllc_model_closes & ~embb_model_closes.values).sum())
    n_b = int((bad_mask & embb_model_closes.values & urllc_model_closes).sum())
    bad_idx = classified.index[bad_mask][:5].tolist()
    sc["SC3"] = {
        "windows_executed_all_sla_ok": n_ok,
        "classified_joint_feasible": n_ok - n_bad,
        "inconsistencies": n_bad,
        "inconsistency_rate": round(n_bad / n_ok, 4) if n_ok else np.nan,
        "driven_by_embb_gate_only": n_e,
        "driven_by_urllc_gate_only": n_u,
        "driven_by_both": n_b,
        "diagnosis": ("portas de KPI previstas pela MEDIANA do estrato: janelas com "
                      "resultado executado OK mas estrato mediano acima do limiar "
                      "(dispersão janela-a-janela em torno da mediana; exatamente o "
                      "false-infeasible do §16) — sensibilidade p25/p75 em "
                      "P_inf_gate_p25/p75"),
        "passed": bool(n_bad == 0),
        "inconsistent_example_rows": bad_idx,
        "note": "janela com resultado executado OK em todos os SLAs deve ter F_t != vazio (§22)",
    }
    inf = classified[~classified["joint_feasible"]]
    frac_low_snr = (float((inf["snr_embb"] < 10.0).mean()) if len(inf) else np.nan)
    sc["SC4"] = {
        "infeasible_windows": int(len(inf)),
        "median_snr_infeasible": (None if not len(inf) or pd.isna(inf["snr_embb"].median())
                                  else round(float(inf["snr_embb"].median()), 2)),
        "median_snr_feasible": (None if pd.isna(classified["snr_embb"].median()) else
                                round(float(classified["snr_embb"].median()), 2)),
        "frac_infeasible_snr_lt_10dB": None if not np.isfinite(frac_low_snr) else round(frac_low_snr, 4),
        "passed": bool(np.isfinite(frac_low_snr) and frac_low_snr < 0.05),
        "note": "a fronteira deve ser de recurso/demanda, não cobertura (SNR saudável nas inviáveis)",
    }
    return sc


def hypotheses(classified: pd.DataFrame, by_lambda: pd.DataFrame,
               by_slice: pd.DataFrame) -> dict:
    bl = by_lambda.set_index("lambda_total")
    h = {}
    pins = bl["P_inf"]
    grew = float(pins.iloc[-1]) > float(pins.iloc[0])
    mono = bool(pins.is_monotonic_increasing)
    h["H2.4_1"] = {
        "claim": "P_inf cresce com a carga: P_inf(2)<P_inf(4)<P_inf(5)<P_inf(6)<P_inf(7) como tendência",
        "P_inf_by_lambda": {float(k): round(float(v), 4) for k, v in pins.items()},
        "verdict": ("CONFIRMADA (estritamente monótona)" if mono else
                    ("CONFIRMADA (tendência crescente, não estritamente monótona)" if grew
                     else "REFUTADA")),
    }
    avr_cols = ["AVR_equal", "AVR_prop"]
    coex = bl[(bl["P_inf"] > 0) & (bl["P_inf"] < 1)]
    avr_max = float(coex[avr_cols].max().max()) if len(coex) else float("nan")
    h["H2.4_2"] = {
        "claim": "existe região com AVR>0 e 0<P_inf<1 (região de justificativa do SQR-JC)",
        "lambdas_na_regiao": [float(x) for x in coex.index],
        "AVR_max_na_regiao": None if not np.isfinite(avr_max) else round(avr_max, 4),
        "verdict": ("CONFIRMADA" if len(coex) and np.isfinite(avr_max) and avr_max > 0
                    else "REFUTADA"),
    }
    avr_e = {float(l): bl.loc[l, "AVR_equal_embb"] for l in bl.index}
    avr_p = {float(l): bl.loc[l, "AVR_prop_embb"] for l in bl.index}
    avr_u_e = {float(l): bl.loc[l, "AVR_equal_urllc"] for l in bl.index}
    avr_u_p = {float(l): bl.loc[l, "AVR_prop_urllc"] for l in bl.index}
    pairs = [(avr_e[l], avr_p[l]) for l in bl.index
             if np.isfinite(avr_e[l]) and np.isfinite(avr_p[l])]
    upairs = [(avr_u_e[l], avr_u_p[l]) for l in bl.index
              if np.isfinite(avr_u_e[l]) and np.isfinite(avr_u_p[l])]
    embb_ok = all(p <= e for e, p in pairs)
    urllc_worse_somewhere = any(p > e for e, p in upairs)
    h["H2.4_3"] = {
        "claim": "prop reduz AVR eMBB vs equal, mas não necessariamente AVR URLLC",
        "AVR_embb_equal": {k: round(v, 4) for k, v in avr_e.items()},
        "AVR_embb_prop": {k: round(v, 4) for k, v in avr_p.items()},
        "AVR_urllc_equal": {k: round(v, 4) for k, v in avr_u_e.items()},
        "AVR_urllc_prop": {k: round(v, 4) for k, v in avr_u_p.items()},
        "verdict": "CONFIRMADA" if (embb_ok and urllc_worse_somewhere) else "REFUTADA",
    }
    hi = classified[classified["lam"] >= 6.0]
    inf_hi = hi[~hi["joint_feasible"]]
    frac = float((inf_hi["min_total_deficit"] > 0).mean()) if len(inf_hi) else float("nan")
    h["H2.4_4"] = {
        "claim": "em alta carga (λ>=6), D_min>0 em parte significativa das janelas",
        "n_windows_lambda_ge6": int(len(hi)),
        "n_infeasible": int(len(inf_hi)),
        "frac_Dmin_positive_among_infeasible": None if not np.isfinite(frac) else round(frac, 4),
        "frac_Dmin_positive_among_all": (round(float((hi["min_total_deficit"] > 0).mean()), 4)
                                         if len(hi) else None),
        "verdict": ("CONFIRMADA" if np.isfinite(frac) and frac > 0.5 else "REFUTADA"),
    }
    return h


# ---------------------------------------------------------------------------
# §14 arm-actions + §15 critical windows for the Despacho-II replay
# ---------------------------------------------------------------------------

def critical_windows(classified: pd.DataFrame, per_class_per_seed: int = 6) -> pd.DataFrame:
    """Candidate windows for executable replay (protocol §15).

    Priority classes — evaluated PER (lambda, seed) so the equal/prop arms
    stay PAIRED on the same traffic realization (same seed => same Poisson
    arrivals/channels across arms, the pairing premise of the 2.3 analysis):
      A: equal violates & prop doesn't (§15 'equal viola e prop não');
      B: prop violates URLLC (§15 'prop viola URLLC');
      C: both violate;
      D: oracle classifies infeasible;
      E: |F_t| small (lowest quartile among feasible windows of the seed/load).
    Deterministic: first ``per_class_per_seed`` windows of each class in time
    order, then one line per (class, lam, seed, win, arm).
    """
    picks = []
    for (lam, seed), g in classified.groupby(["lam", "seed"]):
        piv_any = g.pivot_table(index="win", columns="arm",
                                values="observed_any_violation", aggfunc="first")
        piv_u = g.pivot_table(index="win", columns="arm",
                              values="observed_urllc_violation", aggfunc="first")
        eq = piv_any.get("equal", pd.Series(dtype=float))
        pr = piv_any.get("prop", pd.Series(dtype=float))
        pru = piv_u.get("prop", pd.Series(dtype=float))
        pools = {
            "A_equal_viola_prop_nao": eq.index[(eq == 1) & (pr == 0)],
            "B_prop_viola_urllc": pru.index[pru == 1],
            "C_ambos_violam": eq.index[(eq == 1) & (pr == 1)],
            "D_oracle_inviavel": g.loc[~g["joint_feasible"], "win"].unique(),
        }
        fe = g[g["joint_feasible"]]
        if len(fe):
            thr = fe["num_feasible_actions"].quantile(0.25)
            pools["E_fset_pequeno"] = fe.loc[
                fe["num_feasible_actions"] <= thr, "win"].unique()
        else:
            pools["E_fset_pequeno"] = []
        for cls, wins in pools.items():
            for w in sorted(int(x) for x in wins)[:per_class_per_seed]:
                picks.append({"lam": lam, "seed": seed, "win": w,
                              "priority_class": cls})
    cand = pd.DataFrame(picks)
    meta_cols = ["arm", "lam", "seed", "tag", "win", "time_ms", "n_embb", "n_urllc",
                 "n_mmtc", "queue_embb", "queue_urllc", "thr_embb_med",
                 "latency_urllc_p95_ms", "a_e", "a_u", "a_m",
                 "joint_feasible", "num_feasible_actions", "e_min",
                 "witness_a_e", "witness_a_u", "witness_a_m",
                 "a_star_e", "a_star_u", "a_star_m", "observed_any_violation"]
    if cand.empty:
        return pd.DataFrame(columns=meta_cols + ["priority_class"])
    out = classified[meta_cols].merge(cand, on=["lam", "seed", "win"], how="inner")
    keep = ["priority_class", "lam", "seed", "win", "arm", "tag", "time_ms",
            "n_embb", "n_urllc", "n_mmtc", "queue_embb", "queue_urllc",
            "thr_embb_med", "latency_urllc_p95_ms", "a_e", "a_u", "a_m",
            "joint_feasible", "num_feasible_actions", "e_min",
            "witness_a_e", "witness_a_u", "witness_a_m",
            "a_star_e", "a_star_u", "a_star_m", "observed_any_violation"]
    return out[keep].sort_values(["priority_class", "lam", "seed", "win", "arm"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# §28 table writers (exact column orders where the protocol fixes them)
# ---------------------------------------------------------------------------

def oracle_windows_table(cl: pd.DataFrame) -> pd.DataFrame:
    """§19 EXACT column order of oracle_windows.csv."""
    t = pd.DataFrame(index=cl.index)
    t["seed"] = cl["seed"]
    t["time"] = cl["time_ms"]
    t["lambda_total"] = cl["lam"]
    t["arm"] = cl["arm"]
    t["n_embb"] = cl["n_embb"].fillna(0).astype(int)
    t["n_urllc"] = cl["n_urllc"].fillna(0).astype(int)
    t["n_mmtc"] = cl["n_mmtc"].fillna(0).astype(int)
    t["queue_embb"] = cl["queue_embb"]
    t["queue_urllc"] = cl["queue_urllc"]
    t["queue_mmtc"] = cl["queue_mmtc"]
    t["util_active_dl"] = cl["util_active_dl"]
    t["util_all_dl"] = cl["util_all_dl"]
    for col in ("observed_embb_violation", "observed_urllc_violation",
                "observed_mmtc_violation", "observed_any_violation"):
        t[col] = cl[col]
    t["joint_feasible"] = cl["joint_feasible"].astype(bool)
    t["num_feasible_actions"] = cl["num_feasible_actions"]
    for col in ("avoidable_any", "avoidable_embb", "avoidable_urllc",
                "avoidable_mmtc"):
        t[col] = cl[col]
    for col in ("embb_individually_saveable", "urllc_individually_saveable",
                "mmtc_individually_saveable"):
        t[col] = cl[col]
    t["best_feasible_embb"] = cl["thr_hat_witness"]
    t["best_feasible_urllc"] = cl["p95_hat_witness"]
    t["best_feasible_mmtc"] = cl["pdr_hat_witness"]
    t["best_degradation_embb"] = cl["thr_hat_star"]
    t["best_degradation_urllc"] = cl["p95_hat_star"]
    t["best_degradation_mmtc"] = cl["pdr_hat_star"]
    t["min_total_deficit"] = cl["min_total_deficit"]
    t["min_deficit_embb"] = cl["min_deficit_embb"]
    t["min_deficit_urllc"] = cl["min_deficit_urllc"]
    t["min_deficit_mmtc"] = cl["min_deficit_mmtc"]
    order = list(ds.ORACLE_WINDOWS_COLUMNS_19)
    assert list(t.columns) == order, "section-19 column order violated"
    return t


def oracle_windows_extra_table(cl: pd.DataFrame) -> pd.DataFrame:
    """All per-window quantities BEYOND §19 (extras live in a separate file)."""
    keep = ["arm", "lam", "seed", "tag", "win", "time_ms",
            "n_embb_present", "n_urllc_present", "n_mmtc_present",
            "n_embb_backlog", "offered_load_embb", "offered_load_urllc",
            "offered_load_mmtc",
            "thr_embb_med", "thr_embb_mean", "thr_embb_min", "thr_embb_uewins",
            "thr_embb_violfrac", "observed_embb_violation_anyue",
            "latency_urllc_p95_ms", "latency_urllc_med_ms",
            "latency_urllc_miss_pct", "urllc_n_pkts",
            "pdr_urllc", "pdr_mmtc", "pdr_urllc_1s", "pdr_mmtc_1s",
            "snr_embb", "snr_urllc", "snr_mmtc",
            "cqi_embb", "cqi_urllc", "cqi_mmtc", "cqi_eff", "cqi_bin",
            "total_rbg", "dl_slots", "action_source",
            "prb_requested", "prb_scheduled", "prb_executed", "prb_budget",
            "prb_idle",
            "prb_requested_pct_0", "prb_requested_pct_1", "prb_requested_pct_2",
            "prb_scheduled_rbg_0", "prb_scheduled_rbg_1", "prb_scheduled_rbg_2",
            "prb_budget_rbg_0", "prb_budget_rbg_1", "prb_budget_rbg_2",
            "prb_executed_rbg_0", "prb_executed_rbg_1", "prb_executed_rbg_2",
            "a_e", "a_u", "a_m",
            "queue_embb", "queue_urllc", "queue_mmtc",
            "e_min", "urllc_gate", "best_deg_n_ties", "cqi_embb"]
    keep = list(dict.fromkeys(keep))  # dedupe, preserve order
    return cl[keep]


def oracle_actions_table(cl: pd.DataFrame) -> pd.DataFrame:
    """§14: executed arm actions + oracle witness/best-degradation per window."""
    t = pd.DataFrame(index=cl.index)
    t["arm"] = cl["arm"]
    t["lam"] = cl["lam"]
    t["seed"] = cl["seed"]
    t["win"] = cl["win"]
    t["time_ms"] = cl["time_ms"]
    t["action_source"] = cl["action_source"]
    t["equal_action_a_e"] = np.where(cl["arm"] == "equal", cl["a_e"], np.nan)
    t["equal_action_a_u"] = np.where(cl["arm"] == "equal", cl["a_u"], np.nan)
    t["equal_action_a_m"] = np.where(cl["arm"] == "equal", cl["a_m"], np.nan)
    t["prop_action_a_e"] = np.where(cl["arm"] == "prop", cl["a_e"], np.nan)
    t["prop_action_a_u"] = np.where(cl["arm"] == "prop", cl["a_u"], np.nan)
    t["prop_action_a_m"] = np.where(cl["arm"] == "prop", cl["a_m"], np.nan)
    t["joint_feasible"] = cl["joint_feasible"]
    t["oracle_feasible_a_e"] = cl["witness_a_e"]
    t["oracle_feasible_a_u"] = cl["witness_a_u"]
    t["oracle_feasible_a_m"] = cl["witness_a_m"]
    t["oracle_best_a_e"] = cl["a_star_e"]
    t["oracle_best_a_u"] = cl["a_star_u"]
    t["oracle_best_a_m"] = cl["a_star_m"]
    t["oracle_feasible_thr_embb_hat"] = cl["thr_hat_witness"]
    t["oracle_feasible_p95_urllc_hat"] = cl["p95_hat_witness"]
    t["oracle_best_thr_embb_hat"] = cl["thr_hat_star"]
    t["oracle_best_p95_urllc_hat"] = cl["p95_hat_star"]
    t["min_total_deficit"] = cl["min_total_deficit"]
    t["observed_any_violation"] = cl["observed_any_violation"]
    return t


def tradeoff_sample_windows(cl: pd.DataFrame, per_lam: int = 2) -> pd.DataFrame:
    """Select representative windows for the §11 trade-off (Pareto fig 6).

    ``per_lam`` windows per lambda, chosen deterministically as the ones whose
    observed eMBB median throughput sits at evenly spaced quantiles of the
    lambda's distribution (covering the near-SLA region), restricted to
    windows with eMBB AND URLLC demand.  The sweep over candidate allocations
    for these windows is produced by the driver with the fitted models
    (:func:`examples.sqr_oracle_despacho1.tradeoff_sweep`).
    """
    rows = []
    for lam, g in cl.groupby("lam"):
        cand = g[(g["n_embb"] >= 1) & (g["n_urllc"] >= 1) & g["thr_embb_med"].notna()]
        if not len(cand):
            continue
        picks = np.linspace(0, len(cand) - 1, per_lam).astype(int)
        for i in picks:
            r = cand.iloc[int(i)]
            rows.append({"lam": float(lam), "seed": int(r["seed"]),
                         "tag": r["tag"], "arm": r["arm"], "win": int(r["win"]),
                         "n_embb": int(r["n_embb"]), "n_urllc": int(r["n_urllc"]),
                         "n_mmtc": int(r["n_mmtc"]), "cqi_eff": float(r["cqi_eff"]),
                         "thr_embb_obs": float(r["thr_embb_med"]),
                         "p95_urllc_obs": float(r["latency_urllc_p95_ms"])
                         if np.isfinite(r["latency_urllc_p95_ms"]) else np.nan,
                         "a_e_exec": int(r["a_e"]), "a_u_exec": int(r["a_u"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

def build_all(round2_dir: str):
    """Items 1-6 of the dispatch; returns every in-memory artifact."""
    t_start = time.perf_counter()
    windows = ds.build_window_dataset(round2_dir)
    t_ds = time.perf_counter() - t_start

    # 1 s rolling PDRs from raw packet counts
    kpm_cache = {}
    for arm, sub in ds._ARMS.items():
        for kpm_path in sorted(glob.glob(
                os.path.join(round2_dir, sub, "A_lam*", "**", "sqr_kpm_history.csv"),
                recursive=True)):
            tag_dir = os.path.dirname(kpm_path).rsplit(
                os.sep + "results_sqr_network_only", 1)[0]
            m = re.search(r"A_lam(\d+)_s(\d+)$", os.path.basename(tag_dir))
            if not m:
                continue
            kpm = pd.read_csv(kpm_path, usecols=["timestamp", "sliceId",
                                                 "dTxPackets", "dRxPackets"])
            kpm_cache[(arm, float(m.group(1)), int(m.group(2)))] = \
                ds.per_run_packet_counts(kpm, np.arange(0, 2999))
    windows = ds.add_rolling_pdr(windows, kpm_cache)

    models = fit_models(windows)
    loo = loo_validate(windows)

    enum = enumeration_timing(C_PRB, 1)
    t0 = time.perf_counter()
    classified = classify_windows(windows, models)
    t_cls = time.perf_counter() - t0

    by_lambda = aggregate_by_lambda(classified)
    by_slice = aggregate_by_slice(classified)
    sc = sanity_checks(classified, by_lambda)
    hyp = hypotheses(classified, by_lambda, by_slice)
    crit = critical_windows(classified)
    tradeoff = tradeoff_sample_windows(classified)

    timing = {"dataset_build_s": t_ds, "classify_s": t_cls,
              "n_windows": int(len(classified)),
              "n_unique_state_keys": int(classified["_key"].nunique()),
              "enumeration": enum}
    return {"windows": windows, "classified": classified, "models": models,
            "loo": loo, "by_lambda": by_lambda, "by_slice": by_slice,
            "critical": crit, "tradeoff_windows": tradeoff,
            "sc": sc, "hyp": hyp, "timing": timing}


def main():  # pragma: no cover — thin CLI wrapper
    raise SystemExit("use examples/sqr_oracle_despacho1.py")


if __name__ == "__main__":  # pragma: no cover
    main()
