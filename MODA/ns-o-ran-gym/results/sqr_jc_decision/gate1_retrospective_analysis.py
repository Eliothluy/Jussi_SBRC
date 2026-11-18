#!/usr/bin/env python3
"""Gate 1 Retrospective Diagnostic & Restricted Oracle Analysis (NO NEW REPLAYS).

Reads strictly committed Gate 1 artifacts:
  - A5_GATE1_METRICS.csv
  - A5_GATE1_DECISIONS.json
  - A5_GATE1_DIAGNOSTICS.json
  - b0_traced_gate1/*/causal_trace.jsonl.gz

Produces (strictly read-only retrospective analysis):
  - A5_GATE1_RETROSPECTIVE_ANALYSIS.json
  - A5_GATE1_RETROSPECTIVE_TABLES.csv
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = "/home/eliothluy/Documentos/Jussi_SBRC"
GYM_DIR = os.path.join(REPO_ROOT, "MODA", "ns-o-ran-gym")
SRC_DIR = os.path.join(GYM_DIR, "src")
DEC_DIR = os.path.join(GYM_DIR, "results", "sqr_jc_decision")
SAMPLING_CSV = os.path.join(DEC_DIR, "A5_GATE1_SAMPLING_PLAN.csv")
METRICS_CSV = os.path.join(DEC_DIR, "A5_GATE1_METRICS.csv")
DECISIONS_JSON = os.path.join(DEC_DIR, "A5_GATE1_DECISIONS.json")
DIAGNOSTICS_JSON = os.path.join(DEC_DIR, "A5_GATE1_DIAGNOSTICS.json")

OUT_JSON = os.path.join(DEC_DIR, "A5_GATE1_RETROSPECTIVE_ANALYSIS.json")
OUT_CSV = os.path.join(DEC_DIR, "A5_GATE1_RETROSPECTIVE_TABLES.csv")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from nsoran.sqr_jc.a5_joint_selector import (  # noqa: E402
    A5JointSelector,
    A5SelectorConfig,
    causal_active_slices,
)
from nsoran.sqr_jc.sla_margin import embb_margin, mmtc_margin, urllc_margin  # noqa: E402

ARMS = ("b0", "eq", "prop", "b1_raw", "b1_act", "b1_sup")
STATIC_REF_ARMS = ("b0", "eq", "prop")
SLICE_NAMES = {0: "eMBB", 1: "URLLC", 2: "mMTC"}


def _parse_bool_opt(val: Any) -> bool | None:
    if val is None or val == "" or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip()
    if s in ("True", "true", "1"):
        return True
    if s in ("False", "false", "0"):
        return False
    return None


def _parse_float_opt(val: Any) -> float | None:
    if val is None or val == "" or str(val).strip().lower() == "nan":
        return None
    return float(val)


def _load_traces_for_sampling_plan() -> dict[str, dict[str, Any]]:
    plan_rows = list(csv.DictReader(open(SAMPLING_CSV, encoding="utf-8")))
    needed: dict[str, dict[int, str]] = defaultdict(dict)
    for r in plan_rows:
        needed[r["b0_traced_dir"]][int(r["governed_window_index"])] = r["state_id"]

    traces_by_state: dict[str, dict[str, Any]] = {}
    for run_dir, w_map in needed.items():
        path = os.path.join(run_dir, "causal_trace.jsonl.gz")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if idx == 0 or not line.strip():
                    continue
                tw = json.loads(line)
                gw = int(tw["governed_window_index"])
                if gw in w_map:
                    traces_by_state[w_map[gw]] = tw
    return traces_by_state


def _auroc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                wins += 1.0
            elif sp == sn:
                wins += 0.5
    return float(wins / (len(pos) * len(neg)))


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    sx = pd.Series(x, dtype=float)
    sy = pd.Series(y, dtype=float)
    if sx.nunique() <= 1 or sy.nunique() <= 1:
        return None
    val = sx.corr(sy, method="spearman")
    return None if pd.isna(val) else float(val)


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    sx = pd.Series(x, dtype=float)
    sy = pd.Series(y, dtype=float)
    if sx.nunique() <= 1 or sy.nunique() <= 1:
        return None
    val = sx.corr(sy, method="pearson")
    return None if pd.isna(val) else float(val)


def _classification_stats(preds: list[bool], labels: list[bool], scores: list[float], reals: list[float], taus: list[float], sigmas: list[float]) -> dict[str, Any]:
    n = len(labels)
    if n == 0:
        return {"n": 0}
    tp = sum(1 for p, y in zip(preds, labels) if p and y)
    fp = sum(1 for p, y in zip(preds, labels) if p and not y)
    tn = sum(1 for p, y in zip(preds, labels) if not p and not y)
    fn = sum(1 for p, y in zip(preds, labels) if not p and y)
    acc = (tp + tn) / n
    tpr = tp / (tp + fn) if (tp + fn) > 0 else None
    tnr = tn / (tn + fp) if (tn + fp) > 0 else None
    ppv = tp / (tp + fp) if (tp + fp) > 0 else None
    npv = tn / (tn + fn) if (tn + fn) > 0 else None
    bal_acc = ((tpr + tnr) / 2.0) if (tpr is not None and tnr is not None) else None
    errs = [f - r for f, r in zip(scores, reals)]
    abs_errs = [abs(e) for e in errs]
    pass_sigmas = [sg for sg, y in zip(sigmas, labels) if y]
    fail_sigmas = [sg for sg, y in zip(sigmas, labels) if not y]
    return {
        "n": n,
        "actual_pass": tp + fn,
        "actual_fail": tn + fp,
        "actual_pass_rate": round((tp + fn) / n, 4),
        "pred_feasible_rate": round((tp + fp) / n, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4) if bal_acc is not None else None,
        "tpr_sensitivity": round(tpr, 4) if tpr is not None else None,
        "tnr_specificity": round(tnr, 4) if tnr is not None else None,
        "ppv": round(ppv, 4) if ppv is not None else None,
        "npv": round(npv, 4) if npv is not None else None,
        "auroc_f": round(v, 4) if (v := _auroc(scores, [int(y) for y in labels])) is not None else None,
        "spearman_f_vs_real_margin": round(v, 4) if (v := _spearman(scores, reals)) is not None else None,
        "pearson_f_vs_real_margin": round(v, 4) if (v := _pearson(scores, reals)) is not None else None,
        "spearman_sigma_vs_abs_err": round(v, 4) if (v := _spearman(sigmas, abs_errs)) is not None else None,
        "mae_margin": round(float(np.mean(abs_errs)), 4),
        "median_abs_err": round(float(np.median(abs_errs)), 4),
        "mean_signed_bias_f_minus_real": round(float(np.mean(errs)), 4),
        "median_f": round(float(np.median(scores)), 4),
        "median_real_margin": round(float(np.median(reals)), 4),
        "mean_tau": round(float(np.mean(taus)), 4),
        "mean_sigma": round(float(np.mean(sigmas)), 4),
        "mean_sigma_when_pass": round(float(np.mean(pass_sigmas)), 4) if pass_sigmas else None,
        "mean_sigma_when_fail": round(float(np.mean(fail_sigmas)), 4) if fail_sigmas else None,
    }


def _oracle_summary(states: list[dict[str, Any]], pool: tuple[str, ...], outcome_key: str) -> dict[str, Any]:
    pass_count = 0
    b0_pass = 0
    rescued = 0
    b0_fail = 0
    harmed = 0
    combos: Counter[str] = Counter()
    by_lam: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "pass": 0, "b0_pass": 0})
    by_seed: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "pass": 0, "b0_pass": 0})

    for st in states:
        b0_ok = st["arms"]["b0"][outcome_key] == "PASS"
        if b0_ok:
            b0_pass += 1
        else:
            b0_fail += 1
        passing_arms = [a for a in pool if st["arms"][a][outcome_key] == "PASS"]
        oracle_ok = len(passing_arms) > 0
        if oracle_ok:
            pass_count += 1
            if not b0_ok:
                rescued += 1
        elif b0_ok:
            harmed += 1
        combos[",".join(passing_arms) if passing_arms else "NONE"] += 1
        by_lam[str(st["lam"])]["n"] += 1
        by_lam[str(st["lam"])]["pass"] += int(oracle_ok)
        by_lam[str(st["lam"])]["b0_pass"] += int(b0_ok)
        by_seed[str(st["seed"])]["n"] += 1
        by_seed[str(st["seed"])]["pass"] += int(oracle_ok)
        by_seed[str(st["seed"])]["b0_pass"] += int(b0_ok)

    n = len(states)
    return {
        "warning": "NON_CAUSAL_HINDSIGHT_UPPER_BOUND_REQUIRES_ORACLE_OUTCOMES",
        "pool": list(pool),
        "outcome_key": outcome_key,
        "n_states": n,
        "oracle_pass": pass_count,
        "oracle_rate": round(pass_count / n, 4) if n else None,
        "b0_pass": b0_pass,
        "b0_rate": round(b0_pass / n, 4) if n else None,
        "rescued_from_b0_fail": rescued,
        "b0_fail_total": b0_fail,
        "rescue_rate": round(rescued / b0_fail, 4) if b0_fail else None,
        "harmed_from_b0_pass": harmed,
        "passing_arm_subsets": dict(sorted(combos.items())),
        "by_lam": {k: dict(v) for k, v in sorted(by_lam.items())},
        "by_seed": {k: dict(v) for k, v in sorted(by_seed.items())},
    }


def _curve_diagnostics(tw: dict[str, Any], active_mask: tuple[bool, bool, bool]) -> list[dict[str, Any]]:
    cap = int(tw["capacity_C"])
    out = []
    for sid in range(3):
        psm = tw["per_slice_model"][sid]
        fs = np.asarray(psm["f_candidates"], dtype=float)
        sigs = np.asarray(psm["sigma_candidates"], dtype=float)
        diffs = np.diff(fs)
        non_neg_steps = int(np.sum(diffs >= -1e-9))
        gs_beta1 = fs - 1.0 * sigs
        argmax_f = int(np.argmax(fs))
        argmax_g1 = int(np.argmax(gs_beta1))
        out.append({
            "slice_id": sid,
            "slice": SLICE_NAMES[sid],
            "active_causal": bool(active_mask[sid]),
            "tau": round(float(psm["tau"]), 4),
            "n_support": int(psm["n_support"]),
            "knn_mode": str(psm["knn_mode"]),
            "knn_raw": int(psm["knn_action_raw"]),
            "f_at_0": round(float(fs[0]), 4),
            "f_at_1": round(float(fs[1]), 4),
            "f_at_18": round(float(fs[18]), 4),
            "f_at_cap": round(float(fs[cap]), 4),
            "f_min": round(float(np.min(fs)), 4),
            "f_max": round(float(np.max(fs)), 4),
            "f_span": round(float(np.max(fs) - np.min(fs)), 4),
            "f_end_minus_start": round(float(fs[cap] - fs[0]), 4),
            "monotonic_nondec_frac": round(non_neg_steps / cap, 4),
            "argmax_f": argmax_f,
            "sigma_min": round(float(np.min(sigs)), 4),
            "sigma_max": round(float(np.max(sigs)), 4),
            "sigma_span": round(float(np.max(sigs) - np.min(sigs)), 4),
            "argmax_g_beta1": argmax_g1,
            "n_f_ge_0": int(np.sum(fs >= 0.0)),
            "n_sigma_le_015": int(np.sum(sigs <= 0.15)),
            "n_sigma_le_055": int(np.sum(sigs <= 0.55)),
        })
    return out


def _failing_slice_tag(e_ok: bool | None, u_ok: bool | None, m_ok: bool | None) -> str:
    tags = []
    if e_ok is False:
        tags.append("e")
    if u_ok is False:
        tags.append("u")
    if m_ok is False:
        tags.append("m")
    if e_ok is None or u_ok is None or m_ok is None:
        if any(x is None for x in (e_ok, u_ok, m_ok)) and not tags:
            return "UNK"
    return "+".join(tags) if tags else "PASS"


def main() -> None:
    metrics_rows = list(csv.DictReader(open(METRICS_CSV, encoding="utf-8")))
    decisions = json.loads(open(DECISIONS_JSON, encoding="utf-8").read())
    diagnostics = json.loads(open(DIAGNOSTICS_JSON, encoding="utf-8").read())
    traces = _load_traces_for_sampling_plan()

    dec_by_state = {s["state_id"]: s for s in decisions["states"]}
    rows_by_state_arm: dict[tuple[str, str], dict[str, Any]] = {}
    for r in metrics_rows:
        rows_by_state_arm[(r["state_id"], r["arm"])] = r

    # Offline trace counterfactual selectors isolating beta=0 vs beta=1.0 (NO ns-3 execution)
    off_raw_b0 = A5JointSelector(A5SelectorConfig(variant="b1_raw", sigma_max=0.15, sigma_soft=0.55, beta_uncertainty=0.0, enforce_active_floor_one=False))
    off_act_b0 = A5JointSelector(A5SelectorConfig(variant="b1_active", sigma_max=0.15, sigma_soft=0.55, beta_uncertainty=0.0, enforce_active_floor_one=False))
    off_sup_b0 = A5JointSelector(A5SelectorConfig(variant="b1_support", sigma_max=0.15, sigma_soft=0.55, beta_uncertainty=0.0, enforce_active_floor_one=True))
    offline_selectors = {
        "off_raw_beta0": off_raw_b0,
        "off_act_beta0": off_act_b0,
        "off_sup_beta0": off_sup_b0,
    }

    state_records: list[dict[str, Any]] = []
    table_csv_rows: list[dict[str, Any]] = []

    for s_dec in decisions["states"]:
        sid = s_dec["state_id"]
        tw = traces[sid]
        act_mask = tuple(bool(x) for x in s_dec["active_mask"])
        live_mask = tuple(bool(x) for x in s_dec["live_floor_mask"])
        psm_list = tw["per_slice_model"]

        # Map executed unique actions -> outcomes already known at this state
        known_action_outcome: dict[tuple[int, int, int], dict[str, Any]] = {}
        arms_data: dict[str, Any] = {}
        for arm in ARMS:
            row = rows_by_state_arm[(sid, arm)]
            act = (int(row["action_e"]), int(row["action_u"]), int(row["action_m"]))
            e_ok = _parse_bool_opt(row["embb_ok"])
            u_ok = _parse_bool_opt(row["urllc_ok"])
            m_ok = _parse_bool_opt(row["mmtc_ok"])
            e_thr = _parse_float_opt(row["embb_thr_mbps"])
            u_p95 = _parse_float_opt(row["urllc_p95_ms"])
            u_miss = _parse_float_opt(row["urllc_miss_pct"])
            m_pdr = _parse_float_opt(row["mmtc_pdr_rolling"])

            real_m = [
                embb_margin(e_thr) if e_ok is not None else None,
                urllc_margin(u_p95) if u_ok is not None else None,
                mmtc_margin(m_pdr) if m_ok is not None else None,
            ]
            chosen_f = [float(psm_list[i]["f_candidates"][act[i]]) for i in range(3)]
            chosen_sigma = [float(psm_list[i]["sigma_candidates"][act[i]]) for i in range(3)]
            chosen_g = [chosen_f[i] - 1.0 * chosen_sigma[i] for i in range(3)]
            taus = [float(psm_list[i]["tau"]) for i in range(3)]

            arm_rec = {
                "arm": arm,
                "replay_tag": row["replay_tag"],
                "reused_action": _parse_bool_opt(row["reused_action"]),
                "action": list(act),
                "action_sum": int(row["action_sum"]),
                "idle_prbs_lt53": 53 - int(row["action_sum"]),
                "inactive_prbs_allocated": int(row["inactive_prbs_allocated"]),
                "starved_live_slices": int(row["starved_live_slices"]),
                "tier": row["tier"],
                "bottleneck_slice": int(float(row["bottleneck_slice"])) if row["bottleneck_slice"] else None,
                "fallback_used": _parse_bool_opt(row["fallback_used"]),
                "support_violated_gt015": _parse_bool_opt(row["support_violated_gt015"]),
                "support_relaxed_gt055": _parse_bool_opt(row["support_relaxed_gt055"]),
                "max_chosen_sigma": _parse_float_opt(row["max_chosen_sigma"]),
                "chosen_f": [round(v, 5) for v in chosen_f],
                "chosen_sigma": [round(v, 5) for v in chosen_sigma],
                "chosen_g_beta1": [round(v, 5) for v in chosen_g],
                "taus": [round(v, 4) for v in taus],
                "strict_all3_r19": row["strict_all3_r19"],
                "strict_demand_r19": row["strict_demand_r19"],
                "embb_ok": e_ok,
                "urllc_ok": u_ok,
                "mmtc_ok": m_ok,
                "fail_tag": _failing_slice_tag(e_ok, u_ok, m_ok),
                "embb_thr_mbps": e_thr,
                "urllc_p95_ms": u_p95,
                "urllc_miss_pct": u_miss,
                "urllc_pool_tag": row["urllc_pool_tag"],
                "mmtc_pdr_rolling": m_pdr,
                "real_margins": [round(v, 5) if v is not None else None for v in real_m],
                "util_dl_pct": _parse_float_opt(row["util_dl_pct"]),
                "delta_embb_thr_vs_b0": _parse_float_opt(row["delta_embb_thr_vs_b0"]),
                "delta_urllc_p95_vs_b0": _parse_float_opt(row["delta_urllc_p95_vs_b0"]),
            }
            arms_data[arm] = arm_rec
            known_action_outcome.setdefault(act, {
                "strict_all3_r19": row["strict_all3_r19"],
                "strict_demand_r19": row["strict_demand_r19"],
                "embb_ok": e_ok,
                "urllc_ok": u_ok,
                "mmtc_ok": m_ok,
                "matched_arm": arm,
            })

            table_csv_rows.append({
                "state_id": sid,
                "universe": s_dec["universe"],
                "lam": int(s_dec["lam"]),
                "seed": int(s_dec["seed"]),
                "n_active_causal": int(s_dec["n_active_causal"]),
                "active_mask": "".join("1" if x else "0" for x in act_mask),
                "arm": arm,
                "action_e": act[0],
                "action_u": act[1],
                "action_m": act[2],
                "action_sum": sum(act),
                "idle_prbs": 53 - sum(act),
                "strict_all3_r19": row["strict_all3_r19"],
                "strict_demand_r19": row["strict_demand_r19"],
                "fail_tag": arm_rec["fail_tag"],
                "embb_thr_mbps": e_thr,
                "urllc_p95_ms": u_p95,
                "mmtc_pdr_rolling": m_pdr,
                "f_e": chosen_f[0],
                "f_u": chosen_f[1],
                "f_m": chosen_f[2],
                "sig_e": chosen_sigma[0],
                "sig_u": chosen_sigma[1],
                "sig_m": chosen_sigma[2],
                "real_m_e": real_m[0],
                "real_m_u": real_m[1],
                "real_m_m": real_m[2],
                "tau_e": taus[0],
                "tau_u": taus[1],
                "tau_m": taus[2],
            })

        # Offline beta=0 trace runs (NOT executed unless action matches existing replay)
        offline_beta0_data: dict[str, Any] = {}
        for ok_name, sel in offline_selectors.items():
            dec = sel.select_from_trace_record(tw)
            act_t = tuple(int(v) for v in dec.action)
            eval_idxs = [0, 1, 2] if ok_name == "off_raw_beta0" else [i for i in range(3) if act_mask[i]]
            bn = min(eval_idxs, key=lambda i: (dec.chosen_g[i], dec.chosen_f[i], i)) if eval_idxs else None
            matched = known_action_outcome.get(act_t)
            offline_beta0_data[ok_name] = {
                "execution_status": "EXECUTED_MATCH_EXISTING_REPLAY" if matched else "OFFLINE_TRACE_ONLY_UNEXECUTED",
                "matched_replay": matched,
                "action": list(act_t),
                "action_sum": int(dec.sum_prbs),
                "idle_prbs_lt53": 53 - int(dec.sum_prbs),
                "tier": dec.tier,
                "bottleneck_slice": bn,
                "fallback_used": bool(dec.fallback_used),
                "support_violated_gt015": any(dec.chosen_sigma[i] > 0.15 for i in range(3)) if ok_name == "off_raw_beta0" else any(dec.chosen_sigma[i] > 0.15 for i in range(3) if act_mask[i]),
                "max_chosen_sigma": round(float(dec.max_chosen_sigma), 5),
                "chosen_f": [round(float(v), 5) for v in dec.chosen_f],
                "chosen_sigma": [round(float(v), 5) for v in dec.chosen_sigma],
                "action_e_le_1_while_active": bool(act_mask[0] and act_t[0] <= 1),
            }

        state_records.append({
            "state_id": sid,
            "universe": s_dec["universe"],
            "stratum_id": s_dec["stratum_id"],
            "lam": int(s_dec["lam"]),
            "seed": int(s_dec["seed"]),
            "gov_w": int(s_dec["governed_window_index"]),
            "n_active_causal": int(s_dec["n_active_causal"]),
            "active_mask": list(act_mask),
            "live_floor_mask": list(live_mask),
            "prop_excess_gt53": bool(s_dec["prop_excess_gt53"]),
            "b0_raw": list(s_dec["b0_a_raw"]),
            "b0_prop": list(s_dec["b0_a_prop"]),
            "b0_exec": list(s_dec["b0_a_exec"]),
            "curves": _curve_diagnostics(tw, act_mask),
            "arms": arms_data,
            "offline_beta0_trace_checks": offline_beta0_data,
        })

    # Partition universes
    univ_A = [s for s in state_records if s["universe"] == "ALL3_COMPLETE"]
    univ_B = [s for s in state_records if s["universe"] == "LT3_DEMAND_COMPLETE"]
    univ_C = [s for s in state_records if s["universe"] == "UNKNOWN_INTEGRITY"]

    # 1. Beta=1.0 vs Beta=0.0 counterfactual decomposition across all 25 states + Univ A
    beta_decomp = {
        "warning": "OFFLINE_TRACE_ONLY unless matched to one of the 127 executed Gate 1 replays",
        "executed_b1_raw_sum_lt53": sum(1 for s in state_records if s["arms"]["b1_raw"]["action_sum"] < 53),
        "executed_b1_act_sum_lt53": sum(1 for s in state_records if s["arms"]["b1_act"]["action_sum"] < 53),
        "executed_b1_sup_sum_lt53": sum(1 for s in state_records if s["arms"]["b1_sup"]["action_sum"] < 53),
        "offline_raw_beta0_sum_lt53": sum(1 for s in state_records if s["offline_beta0_trace_checks"]["off_raw_beta0"]["action_sum"] < 53),
        "offline_act_beta0_sum_lt53": sum(1 for s in state_records if s["offline_beta0_trace_checks"]["off_act_beta0"]["action_sum"] < 53),
        "offline_sup_beta0_sum_lt53": sum(1 for s in state_records if s["offline_beta0_trace_checks"]["off_sup_beta0"]["action_sum"] < 53),
        "executed_b1_act_urllc_bn_25": sum(1 for s in state_records if s["arms"]["b1_act"]["bottleneck_slice"] == 1),
        "executed_b1_sup_urllc_bn_25": sum(1 for s in state_records if s["arms"]["b1_sup"]["bottleneck_slice"] == 1),
        "offline_act_beta0_urllc_bn_25": sum(1 for s in state_records if s["offline_beta0_trace_checks"]["off_act_beta0"]["bottleneck_slice"] == 1),
        "offline_sup_beta0_urllc_bn_25": sum(1 for s in state_records if s["offline_beta0_trace_checks"]["off_sup_beta0"]["bottleneck_slice"] == 1),
        "offline_act_beta0_embb_starved_A18": sum(1 for s in univ_A if s["offline_beta0_trace_checks"]["off_act_beta0"]["action"][0] <= 1),
        "offline_sup_beta0_embb_starved_A18": sum(1 for s in univ_A if s["offline_beta0_trace_checks"]["off_sup_beta0"]["action"][0] <= 1),
        "shrinkage_states_detail": [],
        "beta0_vs_beta1_matches_with_known_replay": [],
    }
    for s in state_records:
        b1s = s["arms"]["b1_sup"]
        ob0s = s["offline_beta0_trace_checks"]["off_sup_beta0"]
        if b1s["action_sum"] < 53 or ob0s["action_sum"] < 53:
            beta_decomp["shrinkage_states_detail"].append({
                "state_id": s["state_id"],
                "universe": s["universe"],
                "active_mask": s["active_mask"],
                "b1_sup_action": b1s["action"],
                "b1_sup_idle": b1s["idle_prbs_lt53"],
                "b1_sup_outcome": b1s["strict_demand_r19"] if s["universe"] == "LT3_DEMAND_COMPLETE" else b1s["strict_all3_r19"],
                "off_sup_beta0_action": ob0s["action"],
                "off_sup_beta0_idle": ob0s["idle_prbs_lt53"],
                "off_sup_beta0_matched_replay": ob0s["matched_replay"],
            })
        if ob0s["matched_replay"] is not None:
            beta_decomp["beta0_vs_beta1_matches_with_known_replay"].append({
                "state_id": s["state_id"],
                "universe": s["universe"],
                "off_sup_beta0_action": ob0s["action"],
                "b1_sup_action": b1s["action"],
                "matched_arm": ob0s["matched_replay"]["matched_arm"],
                "matched_all3": ob0s["matched_replay"]["strict_all3_r19"],
                "matched_demand": ob0s["matched_replay"]["strict_demand_r19"],
            })

    # Also evaluate exact known outcome rate on Univ A when off_sup_beta0 happens to hit an executed action vs differs
    off_sup_matches_A = [
        (s["state_id"], s["offline_beta0_trace_checks"]["off_sup_beta0"])
        for s in univ_A
        if s["offline_beta0_trace_checks"]["off_sup_beta0"]["matched_replay"] is not None
    ]
    beta_decomp["univ_A_off_sup_beta0_matched_count"] = len(off_sup_matches_A)
    beta_decomp["univ_A_off_sup_beta0_matched_pass"] = sum(
        1 for _, o in off_sup_matches_A if o["matched_replay"]["strict_all3_r19"] == "PASS"
    )

    # 2. Curve behavior across 25 snapshots
    curve_summary = {}
    for sid_idx, sname in SLICE_NAMES.items():
        all_c = [s["curves"][sid_idx] for s in state_records]
        act_c = [s["curves"][sid_idx] for s in state_records if s["active_mask"][sid_idx]]
        curve_summary[sname] = {
            "n_total": len(all_c),
            "n_active": len(act_c),
            "active_nondecreasing_frac_mean": round(float(np.mean([c["monotonic_nondec_frac"] for c in act_c])), 4),
            "active_f_end_minus_start_mean": round(float(np.mean([c["f_end_minus_start"] for c in act_c])), 4),
            "active_f_end_minus_start_min": round(float(np.min([c["f_end_minus_start"] for c in act_c])), 4),
            "active_f_end_minus_start_max": round(float(np.max([c["f_end_minus_start"] for c in act_c])), 4),
            "active_f_span_mean": round(float(np.mean([c["f_span"] for c in act_c])), 4),
            "active_sigma_span_mean": round(float(np.mean([c["sigma_span"] for c in act_c])), 4),
            "active_frac_f_span_lt_sigma_span": round(
                sum(1 for c in act_c if c["f_span"] < c["sigma_span"]) / len(act_c), 4
            ),
            "active_frac_zero_feasible_cands": round(
                sum(1 for c in act_c if c["n_f_ge_0"] == 0) / len(act_c), 4
            ),
            "active_argmax_f_median": float(np.median([c["argmax_f"] for c in act_c])),
            "active_argmax_g_beta1_median": float(np.median([c["argmax_g_beta1"] for c in act_c])),
        }

    # 3. Preservation & rescue tables in Universe A (N=18) and Universe B (N=5)
    def _rescue_harm_table(subset: list[dict[str, Any]], outcome_key: str) -> dict[str, Any]:
        b0_safe = [s for s in subset if s["arms"]["b0"][outcome_key] == "PASS"]
        b0_fail = [s for s in subset if s["arms"]["b0"][outcome_key] == "FAIL"]
        per_arm_rh = {}
        for arm in ARMS:
            pres = [s["state_id"] for s in b0_safe if s["arms"][arm][outcome_key] == "PASS"]
            harmed = [
                {
                    "state_id": s["state_id"],
                    "action": s["arms"][arm]["action"],
                    "fail_tag": s["arms"][arm]["fail_tag"],
                    "b0_action": s["arms"]["b0"]["action"],
                    "kpis": [s["arms"][arm]["embb_thr_mbps"], s["arms"][arm]["urllc_p95_ms"], s["arms"][arm]["mmtc_pdr_rolling"]],
                }
                for s in b0_safe if s["arms"][arm][outcome_key] == "FAIL"
            ]
            rescued = [
                {
                    "state_id": s["state_id"],
                    "b0_fail_tag": s["arms"]["b0"]["fail_tag"],
                    "b0_action": s["arms"]["b0"]["action"],
                    "action": s["arms"][arm]["action"],
                    "kpis": [s["arms"][arm]["embb_thr_mbps"], s["arms"][arm]["urllc_p95_ms"], s["arms"][arm]["mmtc_pdr_rolling"]],
                }
                for s in b0_fail if s["arms"][arm][outcome_key] == "PASS"
            ]
            still_fail = [
                {
                    "state_id": s["state_id"],
                    "b0_fail_tag": s["arms"]["b0"]["fail_tag"],
                    "arm_fail_tag": s["arms"][arm]["fail_tag"],
                    "action": s["arms"][arm]["action"],
                }
                for s in b0_fail if s["arms"][arm][outcome_key] == "FAIL"
            ]
            per_arm_rh[arm] = {
                "pass_total": sum(1 for s in subset if s["arms"][arm][outcome_key] == "PASS"),
                "preserved_count": len(pres),
                "preserved_states": pres,
                "harmed_count": len(harmed),
                "harmed_cases": harmed,
                "rescued_count": len(rescued),
                "rescued_cases": rescued,
                "still_fail_count": len(still_fail),
                "still_fail_cases": still_fail,
                "net_gain_count": len(rescued) - len(harmed),
            }
        state_overview = []
        for s in subset:
            state_overview.append({
                "state_id": s["state_id"],
                "lam": s["lam"],
                "seed": s["seed"],
                "active_mask": s["active_mask"],
                "b0_status": s["arms"]["b0"][outcome_key],
                "b0_fail_tag": s["arms"]["b0"]["fail_tag"],
                "outcomes": {a: s["arms"][a][outcome_key] for a in ARMS},
                "fail_tags": {a: s["arms"][a]["fail_tag"] for a in ARMS},
                "actions": {a: s["arms"][a]["action"] for a in ARMS},
            })
        return {"n_b0_safe": len(b0_safe), "n_b0_fail": len(b0_fail), "per_arm": per_arm_rh, "per_state": state_overview}

    rh_A = _rescue_harm_table(univ_A, "strict_all3_r19")
    rh_B = _rescue_harm_table(univ_B, "strict_demand_r19")

    # 4. Hindsight oracles
    oracles = {
        "univ_A_O3_b0_eq_prop": _oracle_summary(univ_A, ("b0", "eq", "prop"), "strict_all3_r19"),
        "univ_A_O2_eq_prop": _oracle_summary(univ_A, ("eq", "prop"), "strict_all3_r19"),
        "univ_A_O6_all_arms": _oracle_summary(univ_A, ARMS, "strict_all3_r19"),
        "univ_B_O3_b0_eq_prop": _oracle_summary(univ_B, ("b0", "eq", "prop"), "strict_demand_r19"),
        "univ_B_O6_all_arms": _oracle_summary(univ_B, ARMS, "strict_demand_r19"),
    }

    # 5. Pre-decision f_i(a) & sigma_i(a) empirical separability on executed replays
    # Exclude Universe C from outcome evaluation! Include Universe A (all 3 active) & active demand windows in Universe B.
    # Evaluate BOTH deduplicated by (state_id, slice_id, a_slice) within known outcome states (cleanest, avoids counting eq==b1_act or b1_act==b1_raw multiple times)
    # AND deduplicated by unique (state_id, replay_tag) replay.
    separability: dict[str, Any] = {}
    for dedup_mode in ("unique_state_slice_action", "unique_replay", "all_138_known_rows"):
        per_slice_sep: dict[str, Any] = {}
        for sid_idx, sname in SLICE_NAMES.items():
            seen: set[tuple[Any, ...]] = set()
            preds, labels, scores, reals, taus, sigmas = [], [], [], [], [], []
            trusted_items = []
            untrusted_items = []

            for s in state_records:
                if s["universe"] == "UNKNOWN_INTEGRITY":
                    continue
                for arm in ARMS:
                    r = s["arms"][arm]
                    ok_val = [r["embb_ok"], r["urllc_ok"], r["mmtc_ok"]][sid_idx]
                    real_m = r["real_margins"][sid_idx]
                    if ok_val is None or real_m is None:
                        continue  # vacuous window
                    # In Universe B, enforce causal active or demand-evaluated match cleanly
                    if dedup_mode == "unique_state_slice_action":
                        key = (s["state_id"], sid_idx, r["action"][sid_idx])
                    elif dedup_mode == "unique_replay":
                        key = (s["state_id"], r["replay_tag"])
                    else:
                        key = (s["state_id"], arm)
                    if key in seen:
                        continue
                    seen.add(key)

                    f_v = r["chosen_f"][sid_idx]
                    sig_v = r["chosen_sigma"][sid_idx]
                    tau_v = r["taus"][sid_idx]
                    pred_ok = f_v >= 0.0
                    preds.append(pred_ok)
                    labels.append(bool(ok_val))
                    scores.append(f_v)
                    reals.append(float(real_m))
                    taus.append(tau_v)
                    sigmas.append(sig_v)
                    tup = (pred_ok, bool(ok_val), f_v, float(real_m), tau_v, sig_v)
                    if sig_v <= 0.15:
                        trusted_items.append(tup)
                    else:
                        untrusted_items.append(tup)

            entry = {
                "overall": _classification_stats(preds, labels, scores, reals, taus, sigmas),
                "trusted_sigma_le_015": _classification_stats(
                    [x[0] for x in trusted_items],
                    [x[1] for x in trusted_items],
                    [x[2] for x in trusted_items],
                    [x[3] for x in trusted_items],
                    [x[4] for x in trusted_items],
                    [x[5] for x in trusted_items],
                ),
                "untrusted_sigma_gt_015": _classification_stats(
                    [x[0] for x in untrusted_items],
                    [x[1] for x in untrusted_items],
                    [x[2] for x in untrusted_items],
                    [x[3] for x in untrusted_items],
                    [x[4] for x in untrusted_items],
                    [x[5] for x in untrusted_items],
                ),
            }
            per_slice_sep[sname] = entry

        # Joint-level binary separation (all active known slices predicted f_i >= 0 vs joint PASS)
        seen_j: set[tuple[Any, ...]] = set()
        j_preds, j_labels, j_minf, j_minreal, j_maxsig, j_taus = [], [], [], [], [], []
        for s in state_records:
            if s["universe"] == "UNKNOWN_INTEGRITY":
                continue
            outcome_key = "strict_all3_r19" if s["universe"] == "ALL3_COMPLETE" else "strict_demand_r19"
            for arm in ARMS:
                r = s["arms"][arm]
                if dedup_mode == "unique_replay":
                    key = (s["state_id"], r["replay_tag"])
                elif dedup_mode == "unique_state_slice_action":
                    key = (s["state_id"], tuple(r["action"]))
                else:
                    key = (s["state_id"], arm)
                if key in seen_j:
                    continue
                seen_j.add(key)

                # Active indices with non-null real outcome in this state
                act_idxs = [
                    i for i in range(3)
                    if s["active_mask"][i] and r["real_margins"][i] is not None
                ]
                if not act_idxs:
                    act_idxs = [i for i in range(3) if r["real_margins"][i] is not None]
                pred_joint = all(r["chosen_f"][i] >= 0.0 for i in act_idxs)
                actual_joint = r[outcome_key] == "PASS"
                j_preds.append(pred_joint)
                j_labels.append(actual_joint)
                j_minf.append(min(r["chosen_f"][i] for i in act_idxs))
                j_minreal.append(min(float(r["real_margins"][i]) for i in act_idxs))
                j_maxsig.append(max(r["chosen_sigma"][i] for i in act_idxs))
                j_taus.append(float(np.mean([r["taus"][i] for i in act_idxs])))

        per_slice_sep["JOINT_ACTIVE"] = _classification_stats(j_preds, j_labels, j_minf, j_minreal, j_taus, j_maxsig)
        separability[dedup_mode] = per_slice_sep

    # 6. Test causal candidate-ranking skill strictly over pre-evaluated {b0, eq, prop} AND all unique executed replays
    # (Using ONLY pre-decision trace information available before W; outcomes ONLY used to score accuracy!)
    def _eval_causal_rankers(subset: list[dict[str, Any]], outcome_key: str) -> dict[str, Any]:
        pools = {
            "pool_3_b0_eq_prop": ("b0", "eq", "prop"),
            "pool_all_executed": ARMS,
        }
        res = {}
        for pname, pool in pools.items():
            scores_rule: dict[str, dict[str, Any]] = {}
            rules = {
                "R_maxmin_f_beta0": lambda r, act_idxs: (
                    min(r["chosen_f"][i] for i in act_idxs),
                    sum(r["chosen_f"][i] for i in act_idxs),
                ),
                "R_feas_count_then_sum_clip": lambda r, act_idxs: (
                    sum(1 for i in act_idxs if r["chosen_f"][i] >= 0.0),
                    sum(min(0.0, r["chosen_f"][i]) for i in act_idxs),
                    min(r["chosen_f"][i] for i in act_idxs),
                ),
                "R_feas_count_lcb_1sig": lambda r, act_idxs: (
                    sum(1 for i in act_idxs if (r["chosen_f"][i] - r["chosen_sigma"][i]) >= 0.0),
                    sum(min(0.0, r["chosen_f"][i] - r["chosen_sigma"][i]) for i in act_idxs),
                    min(r["chosen_f"][i] - r["chosen_sigma"][i] for i in act_idxs),
                ),
                "R_stay_b0_if_all_f_ge_0_else_feas_count": None,
                "R_max_embb_f_subject_urllc_ge_neg05": lambda r, act_idxs: (
                    1 if (1 not in act_idxs or r["chosen_f"][1] >= -0.5) else 0,
                    sum(1 for i in act_idxs if r["chosen_f"][i] >= 0.0),
                    r["chosen_f"][0] if 0 in act_idxs else 0.0,
                ),
                "R_min_sigma_then_feas": lambda r, act_idxs: (
                    1 if all(r["chosen_sigma"][i] <= 0.15 for i in act_idxs) else 0,
                    sum(1 for i in act_idxs if r["chosen_f"][i] >= 0.0),
                    min(r["chosen_f"][i] for i in act_idxs),
                ),
            }
            for rname in rules:
                passes = 0
                rescued = 0
                harmed = 0
                chosen_arms = Counter()
                for s in subset:
                    act_idxs = [i for i in range(3) if s["active_mask"][i]]
                    if not act_idxs:
                        act_idxs = [0, 1, 2]
                    b0_ok = s["arms"]["b0"][outcome_key] == "PASS"
                    if rname == "R_stay_b0_if_all_f_ge_0_else_feas_count":
                        b0_r = s["arms"]["b0"]
                        if all(b0_r["chosen_f"][i] >= 0.0 for i in act_idxs):
                            pick_arm = "b0"
                        else:
                            # deterministic order eq, prop, b0 on ties
                            order_pref = {a: -idx for idx, a in enumerate(pool)}
                            pick_arm = max(
                                pool,
                                key=lambda a: (
                                    sum(1 for i in act_idxs if s["arms"][a]["chosen_f"][i] >= 0.0),
                                    sum(min(0.0, s["arms"][a]["chosen_f"][i]) for i in act_idxs),
                                    order_pref[a],
                                ),
                            )
                    else:
                        fn = rules[rname]
                        order_pref = {a: -idx for idx, a in enumerate(pool)}
                        pick_arm = max(pool, key=lambda a: (*fn(s["arms"][a], act_idxs), order_pref[a]))
                    chosen_arms[pick_arm] += 1
                    pick_ok = s["arms"][pick_arm][outcome_key] == "PASS"
                    if pick_ok:
                        passes += 1
                        if not b0_ok:
                            rescued += 1
                    elif b0_ok:
                        harmed += 1
                n = len(subset)
                scores_rule[rname] = {
                    "pass": passes,
                    "n": n,
                    "rate": round(passes / n, 4) if n else None,
                    "rescued": rescued,
                    "harmed": harmed,
                    "net_gain": rescued - harmed,
                    "chosen_arm_counts": dict(chosen_arms),
                }
            res[pname] = scores_rule
        return res

    causal_ranking_test = {
        "note": "Evaluated WITHOUT outcome peeking inside decision rule, strictly ranking pre-existing executed replays via causal trace predictions f_i(a) & sigma_i(a).",
        "univ_A": _eval_causal_rankers(univ_A, "strict_all3_r19"),
        "univ_B": _eval_causal_rankers(univ_B, "strict_demand_r19"),
    }

    # 7. Exact capacity, fallback, support, inactive waste, starvation table across all 6 arms (all 25 + A + B)
    ops_summary: dict[str, Any] = {}
    for arm in ARMS:
        rows_all = [s["arms"][arm] for s in state_records]
        rows_A = [s["arms"][arm] for s in univ_A]
        rows_B = [s["arms"][arm] for s in univ_B]
        ops_summary[arm] = {
            "all25_n": len(rows_all),
            "all25_tiers": dict(Counter(r["tier"] for r in rows_all)),
            "all25_fallback_count": sum(1 for r in rows_all if r["fallback_used"]),
            "all25_support_violated_gt015": sum(1 for r in rows_all if r["support_violated_gt015"]),
            "all25_support_relaxed_gt055": sum(1 for r in rows_all if r["support_relaxed_gt055"]),
            "all25_sum_lt53_count": sum(1 for r in rows_all if r["action_sum"] < 53),
            "all25_total_idle_prbs": sum(r["idle_prbs_lt53"] for r in rows_all),
            "all25_mean_sum_prbs": round(float(np.mean([r["action_sum"] for r in rows_all])), 2),
            "all25_inactive_waste_states": sum(1 for r in rows_all if r["inactive_prbs_allocated"] > 0),
            "all25_total_inactive_prbs": sum(r["inactive_prbs_allocated"] for r in rows_all),
            "all25_starved_live_states": sum(1 for r in rows_all if r["starved_live_slices"] > 0),
            "all25_total_starved_live_slices": sum(r["starved_live_slices"] for r in rows_all),
            "all25_bottleneck_slice_counts": dict(Counter(str(r["bottleneck_slice"]) for r in rows_all)),
            "univ_A_pass": sum(1 for r in rows_A if r["strict_all3_r19"] == "PASS"),
            "univ_A_n": len(rows_A),
            "univ_B_pass": sum(1 for r in rows_B if r["strict_demand_r19"] == "PASS"),
            "univ_B_n": len(rows_B),
            "univ_B_inactive_waste_states": sum(1 for r in rows_B if r["inactive_prbs_allocated"] > 0),
            "univ_B_starved_live_states": sum(1 for r in rows_B if r["starved_live_slices"] > 0),
        }

    report = {
        "provenance": {
            "source_metrics_csv": METRICS_CSV,
            "source_decisions_json": DECISIONS_JSON,
            "source_diagnostics_json": DIAGNOSTICS_JSON,
            "n_states": len(state_records),
            "n_metrics_rows": len(metrics_rows),
            "n_unique_replays": int(decisions["n_unique_replays"]),
            "new_replays_executed": 0,
        },
        "universe_counts": {
            "ALL3_COMPLETE": len(univ_A),
            "LT3_DEMAND_COMPLETE": len(univ_B),
            "UNKNOWN_INTEGRITY": len(univ_C),
        },
        "ops_support_fallback_waste_summary": ops_summary,
        "beta_vs_urllc_decomposition": beta_decomp,
        "snapshot_curve_diagnostics_25": curve_summary,
        "rescue_harm_attribution": {
            "universe_A_ALL3_COMPLETE": rh_A,
            "universe_B_LT3_DEMAND_COMPLETE": rh_B,
            "universe_C_UNKNOWN_states": [
                {
                    "state_id": s["state_id"],
                    "urllc_tx_pkts": s["arms"]["b0"]["urllc_tx_pkts"] if "urllc_tx_pkts" in s["arms"]["b0"] else None,
                    "pool_tags": {a: s["arms"][a]["urllc_pool_tag"] for a in ARMS},
                    "outcomes": {a: s["arms"][a]["strict_all3_r19"] for a in ARMS},
                }
                for s in univ_C
            ],
        },
        "hindsight_oracles": oracles,
        "pre_decision_predictive_separability": separability,
        "causal_ranking_diagnostic_on_existing_replays": causal_ranking_test,
        "per_state_full": state_records,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    if table_csv_rows:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table_csv_rows[0].keys()))
            w.writeheader()
            w.writerows(table_csv_rows)

    print(f"Wrote {OUT_JSON} and {OUT_CSV}")
    print("\n=== ORACLE SUMMARY (NON-CAUSAL HINDSIGHT UPPER BOUND) ===")
    for k, v in oracles.items():
        print(f"  {k}: {v['oracle_pass']}/{v['n_states']} ({v['oracle_rate']}) vs B0={v['b0_pass']}/{v['n_states']} ({v['b0_rate']}), subsets={v['passing_arm_subsets']}")
    print("\n=== BETA=0 vs BETA=1 DECOMPOSITION (25 STATES) ===")
    print(json.dumps({k: v for k, v in beta_decomp.items() if k not in ("shrinkage_states_detail", "beta0_vs_beta1_matches_with_known_replay")}, indent=2))
    print("\n=== SEPARABILITY (UNIQUE STATE-SLICE-ACTION) ===")
    for sl, d in separability["unique_state_slice_action"].items():
        ov = d.get("overall", d)
        print(f"  {sl}: n={ov.get('n')} pass_rate={ov.get('actual_pass_rate')} pred_feas={ov.get('pred_feasible_rate')} acc={ov.get('accuracy')} bal_acc={ov.get('balanced_accuracy')} tpr={ov.get('tpr_sensitivity')} tnr={ov.get('tnr_specificity')} auroc={ov.get('auroc_f')} rho={ov.get('spearman_f_vs_real_margin')} mae={ov.get('mae_margin')} bias={ov.get('mean_signed_bias_f_minus_real')} mean_tau={ov.get('mean_tau')}")
    print("\n=== CAUSAL RANKING DIAGNOSTIC ON EXISTING REPLAYS (UNIV A N=18) ===")
    for pname, rmap in causal_ranking_test["univ_A"].items():
        print(f"  [{pname}]")
        for rname, rv in rmap.items():
            print(f"    {rname}: {rv['pass']}/{rv['n']} ({rv['rate']}) R={rv['rescued']} H={rv['harmed']} Net={rv['net_gain']} picks={rv['chosen_arm_counts']}")


if __name__ == "__main__":
    main()
