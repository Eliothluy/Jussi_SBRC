#!/usr/bin/env python3
"""Gate 1 Step 3: Evaluate A5 policies (B1-raw, B1-active, B1-support) plus
static references (equal, prop) strictly on pre-registered B0 causal trace
snapshots and generate the single-window counterfactual replay plan.

Preserves frozen Gate 1 exploration hyperparameters verbatim:
  sigma_max = 0.15, sigma_soft = 0.55, beta_uncertainty = 1.0, gamma = 1.0.

Outputs in MODA/ns-o-ran-gym/results/sqr_jc_decision/:
  - A5_GATE1_DECISIONS.json
  - A5_GATE1_REPLAY_PLAN.tsv
  - A5_GATE1_ARM_MAP.json
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
import sys
import time
from collections import Counter
from typing import Any

import numpy as np

REPO_ROOT = "/home/eliothluy/Documentos/Jussi_SBRC"
GYM_DIR = os.path.join(REPO_ROOT, "MODA", "ns-o-ran-gym")
SRC_DIR = os.path.join(GYM_DIR, "src")
DEC_DIR = os.path.join(GYM_DIR, "results", "sqr_jc_decision")
REPLAY_OUT_ROOT = os.path.join(DEC_DIR, "a5_gate1_replays")
SAMPLING_CSV = os.path.join(DEC_DIR, "A5_GATE1_SAMPLING_PLAN.csv")
SAMPLING_MANIFEST = os.path.join(DEC_DIR, "A5_GATE1_SAMPLING_MANIFEST.json")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from nsoran.sqr_jc.a5_joint_selector import (  # noqa: E402
    A5JointSelector,
    A5SelectorConfig,
    causal_active_slices,
    load_b0_rest_het_from_checkpoint,
    prop_active_split,
)
from nsoran.sqr_jc.freeze_contract import sha256_file  # noqa: E402
from nsoran.sqr_jc.joint_state import JointState  # noqa: E402


def _latency_summary(vals: list[float]) -> dict[str, float]:
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(a.size),
        "mean_ms": float(np.mean(a)) if a.size else float("nan"),
        "p50_ms": float(np.percentile(a, 50.0)) if a.size else float("nan"),
        "p95_ms": float(np.percentile(a, 95.0)) if a.size else float("nan"),
        "p99_ms": float(np.percentile(a, 99.0)) if a.size else float("nan"),
        "max_ms": float(np.max(a)) if a.size else float("nan"),
    }


def _reconstruct_state(tw: dict[str, Any]) -> JointState:
    st = tw["state"]
    return JointState(
        window_index=int(tw["window_index"]),
        n_e=int(st["n_e"]),
        n_u=int(st["n_u"]),
        n_m=int(st["n_m"]),
        queue_e=float(st["queue_e"]),
        queue_u=float(st["queue_u"]),
        queue_m=float(st["queue_m"]),
        kpi_e_raw=st["kpi_e_raw"],
        kpi_u_raw=st["kpi_u_raw"],
        kpi_m_raw=st["kpi_m_raw"],
        snr_e=float(st["snr_e"]),
        snr_u=float(st["snr_u"]),
        snr_m=float(st["snr_m"]),
        prev_a_e=int(st["prev_a_e"]),
        prev_a_u=int(st["prev_a_u"]),
        prev_a_m=int(st["prev_a_m"]),
        viol_e=int(st["viol_e"]),
        viol_u=int(st["viol_u"]),
        viol_m=int(st["viol_m"]),
        offered_e=float(st.get("offered_e", 0.0)),
        offered_u=float(st.get("offered_u", 0.0)),
        offered_m=float(st.get("offered_m", 0.0)),
        extras=dict(st.get("extras") or {}),
    )


def main() -> None:
    manifest = json.loads(open(SAMPLING_MANIFEST, encoding="utf-8").read())
    assert sha256_file(SAMPLING_CSV) == manifest["sampling_plan_csv_sha256"], (
        "Sampling plan CSV hash mismatch vs pre-registered manifest!"
    )

    plan_rows = list(csv.DictReader(open(SAMPLING_CSV, encoding="utf-8")))
    by_tag: dict[str, dict[int, dict[str, Any]]] = {}
    needed_by_tag: dict[str, set[int]] = {}
    for row in plan_rows:
        needed_by_tag.setdefault(row["b0_traced_dir"], set()).add(int(row["governed_window_index"]))

    for run_dir, gov_ws in needed_by_tag.items():
        by_tag[run_dir] = {}
        with gzip.open(os.path.join(run_dir, "causal_trace.jsonl.gz"), "rt", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if idx == 0 or not line.strip():
                    continue
                tw = json.loads(line)
                gw = int(tw["governed_window_index"])
                if gw in gov_ws:
                    by_tag[run_dir][gw] = tw

    sel_raw = A5JointSelector(A5SelectorConfig(variant="b1_raw", sigma_max=0.15, sigma_soft=0.55, beta_uncertainty=1.0, enforce_active_floor_one=False))
    sel_act = A5JointSelector(A5SelectorConfig(variant="b1_active", sigma_max=0.15, sigma_soft=0.55, beta_uncertainty=1.0, enforce_active_floor_one=False))
    sel_sup = A5JointSelector(A5SelectorConfig(variant="b1_support", sigma_max=0.15, sigma_soft=0.55, beta_uncertainty=1.0, enforce_active_floor_one=True))

    selectors = {
        "b1_raw": sel_raw,
        "b1_act": sel_act,
        "b1_sup": sel_sup,
    }

    state_diagnostics: list[dict[str, Any]] = []
    replay_plan_rows: list[dict[str, Any]] = []
    arm_map: dict[str, dict[str, Any]] = {}
    lat_trace_ms: dict[str, list[float]] = {k: [] for k in selectors}
    tier_counts: dict[str, Counter[str]] = {k: Counter() for k in selectors}

    for row in plan_rows:
        sid_str = row["state_id"]
        run_dir = row["b0_traced_dir"]
        gov_w = int(row["governed_window_index"])
        lam = int(row["lam"])
        seed = int(row["seed"])
        tw = by_tag[run_dir][gov_w]
        cap = int(tw["capacity_C"])
        st_dict = tw["state"]
        act_mask, live_mask = causal_active_slices(st_dict)

        b0_act = tuple(int(v) for v in tw["a_exec_per_slice"])
        eq_act = (18, 17, 18)  # A5_ESPECIFICACAO_CONCEITUAL §3.1: static blind equal baseline
        prop_act = tuple(int(v) for v in prop_active_split(st_dict, cap, act_mask))

        # Per-slice candidate uncertainty & prediction overview from snapshot tw
        per_slice_overview = []
        for s_idx in range(3):
            psm = tw["per_slice_model"][s_idx]
            sigs = np.asarray(psm["sigma_candidates"], dtype=float)
            fs = np.asarray(psm["f_candidates"], dtype=float)
            per_slice_overview.append({
                "slice_id": s_idx,
                "active_causal": bool(act_mask[s_idx]),
                "live_floor": bool(live_mask[s_idx]),
                "n_support": int(psm["n_support"]),
                "tau": float(psm["tau"]),
                "margin_m": float(psm["margin_m"]),
                "knn_action_raw": int(psm["knn_action_raw"]),
                "knn_mode": str(psm["knn_mode"]),
                "n_trusted_sigma_le_015": int(np.sum(sigs <= 0.15)),
                "n_soft_sigma_le_055": int(np.sum(sigs <= 0.55)),
                "sigma_min": float(np.min(sigs)),
                "sigma_max": float(np.max(sigs)),
                "f_min": float(np.min(fs)),
                "f_max": float(np.max(fs)),
                "f_at_0": float(fs[0]),
                "f_at_C": float(fs[cap]),
                "b0_f": float(fs[b0_act[s_idx]]),
                "b0_sigma": float(sigs[b0_act[s_idx]]),
            })

        # Evaluate all 3 A5 variants purely on snapshot tw
        a5_outs: dict[str, Any] = {}
        for arm_key, sel in selectors.items():
            dec = sel.select_from_trace_record(tw)
            lat_trace_ms[arm_key].append(float(dec.decision_time_ms))
            tier_counts[arm_key][dec.tier] += 1
            act_tuple = tuple(int(v) for v in dec.action)
            # Determine bottleneck active slice for chosen_g
            eval_idxs = [0, 1, 2] if arm_key == "b1_raw" else [i for i in range(3) if act_mask[i]]
            bn_slice = (
                min(eval_idxs, key=lambda i: (dec.chosen_g[i], dec.chosen_f[i], i))
                if eval_idxs
                else None
            )
            a5_outs[arm_key] = {
                "action": list(act_tuple),
                "tier": dec.tier,
                "critical_score": float(dec.critical_score),
                "efficient_score": float(dec.efficient_score),
                "sum_prbs": int(dec.sum_prbs),
                "n_feasible_candidates": int(dec.n_feasible_candidates),
                "max_chosen_sigma": float(dec.max_chosen_sigma),
                "chosen_sigma": [float(v) for v in dec.chosen_sigma],
                "chosen_f": [float(v) for v in dec.chosen_f],
                "chosen_g": [float(v) for v in dec.chosen_g],
                "bottleneck_slice": bn_slice,
                "fallback_used": bool(dec.fallback_used),
                "degenerate_avoided": bool(dec.degenerate_avoided),
                "support_violated_gt_015": any(
                    float(dec.chosen_sigma[i]) > 0.15 for i in range(3) if act_mask[i]
                ),
                "support_relaxed_gt_055": any(
                    float(dec.chosen_sigma[i]) > 0.55 for i in range(3) if act_mask[i]
                ),
                "decision_time_ms": float(dec.decision_time_ms),
            }

        arm_actions = {
            "b0": b0_act,
            "eq": eq_act,
            "prop": prop_act,
            "b1_raw": tuple(a5_outs["b1_raw"]["action"]),
            "b1_act": tuple(a5_outs["b1_act"]["action"]),
            "b1_sup": tuple(a5_outs["b1_sup"]["action"]),
        }

        sim_time = min(300, int(math.ceil((gov_w + 1) / 10.0)) + 3)
        # Group identical actions within the same state into unique replay executions
        # (b0 always keeps canonical tag <state_id>__b0 for explicit A/A verification).
        seen_actions_in_state: dict[tuple[int, int, int], str] = {}
        arm_map[sid_str] = {}

        # Always register b0 first so identical actions map to b0's A/A run
        for arm_name in ("b0", "eq", "prop", "b1_raw", "b1_act", "b1_sup"):
            act_t = arm_actions[arm_name]
            if act_t in seen_actions_in_state:
                replay_tag = seen_actions_in_state[act_t]
                reused = True
            else:
                suffix = "b0" if arm_name == "b0" else f"a{act_t[0]}_{act_t[1]}_{act_t[2]}"
                replay_tag = f"{sid_str}__{suffix}"
                seen_actions_in_state[act_t] = replay_tag
                reused = False
                replay_plan_rows.append({
                    "tag": replay_tag,
                    "state_id": sid_str,
                    "lam": lam,
                    "seed": seed,
                    "gov_w": gov_w,
                    "sim_time": sim_time,
                    "ae": act_t[0],
                    "au": act_t[1],
                    "am": act_t[2],
                    "b0_traced_dir": run_dir,
                    "out_dir": os.path.join(REPLAY_OUT_ROOT, replay_tag),
                    "is_b0_aa": int(arm_name == "b0"),
                })
            arm_map[sid_str][arm_name] = {
                "action": list(act_t),
                "replay_tag": replay_tag,
                "out_dir": os.path.join(REPLAY_OUT_ROOT, replay_tag),
                "reused_action": reused,
            }

        state_diagnostics.append({
            "state_id": sid_str,
            "universe": row["universe"],
            "stratum_id": row["stratum_id"],
            "lam": lam,
            "seed": seed,
            "step_index": int(row["step_index"]),
            "state_window_index": int(row["state_window_index"]),
            "governed_window_index": gov_w,
            "sim_time_ms": int(row["sim_time_ms"]),
            "replay_sim_time_s": sim_time,
            "capacity_C": cap,
            "n_active_causal": int(row["n_active_causal"]),
            "active_mask": list(act_mask),
            "live_floor_mask": list(live_mask),
            "prop_excess_gt53": row["prop_excess_gt53"] == "True",
            "b0_a_raw": json.loads(row["b0_a_raw"]),
            "b0_a_prop": json.loads(row["b0_a_prop"]),
            "b0_a_exec": list(b0_act),
            "b0_modes": json.loads(row["b0_modes"]),
            "b0_all3_r19": row["b0_all3_r19"],
            "b0_demand_r19": row["b0_demand_r19"],
            "b0_u_status": row["b0_u_status"],
            "state": st_dict,
            "per_slice_model_overview": per_slice_overview,
            "arm_actions": {k: list(v) for k, v in arm_actions.items()},
            "a5_details": a5_outs,
        })

    # Benchmark end-to-end online select_from_het latency (3x score_candidates + joint optimizer)
    het_ckpt_path = os.path.join(DEC_DIR, "b0_traced_gate1", "het_lam6_s2103", "sqr_checkpoint.pkl")
    scaler_manifest = os.path.join(REPO_ROOT, "results", "sqr_jc_final", "03_calibration", "feature_scaler_manifest.yaml")
    het_loaded = load_b0_rest_het_from_checkpoint(
        het_ckpt_path,
        scaler_manifest_path=scaler_manifest,
        target_window_index=2995,
    )
    het_bench = het_loaded[1] if isinstance(het_loaded, tuple) else het_loaded
    e2e_lat_ms: dict[str, list[float]] = {k: [] for k in selectors}
    st_sample = _reconstruct_state(by_tag[plan_rows[0]["b0_traced_dir"]][int(plan_rows[0]["governed_window_index"])])
    for arm_key, sel in selectors.items():
        # warm up numpy arrays once, then measure 25 realistic online invocations
        sel.select_from_het(het_bench, st_sample)
        for sd_row in plan_rows:
            tw_s = by_tag[sd_row["b0_traced_dir"]][int(sd_row["governed_window_index"])]
            d_e2e = sel.select_from_het(het_bench, _reconstruct_state(tw_s))
            e2e_lat_ms[arm_key].append(float(d_e2e.decision_time_ms))

    # Write replay plan TSV + decisions JSON + arm map
    plan_tsv = os.path.join(DEC_DIR, "A5_GATE1_REPLAY_PLAN.tsv")
    tsv_cols = ["tag", "state_id", "lam", "seed", "gov_w", "sim_time", "ae", "au", "am", "b0_traced_dir", "out_dir", "is_b0_aa"]
    with open(plan_tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=tsv_cols, delimiter="\t")
        w.writeheader()
        for pr in replay_plan_rows:
            w.writerow(pr)

    dec_json = os.path.join(DEC_DIR, "A5_GATE1_DECISIONS.json")
    with open(dec_json, "w", encoding="utf-8") as fh:
        json.dump({
            "sampling_plan_csv_sha256": manifest["sampling_plan_csv_sha256"],
            "n_states": len(state_diagnostics),
            "n_unique_replays": len(replay_plan_rows),
            "n_logical_arm_evals": len(state_diagnostics) * 6,
            "hyperparameters_frozen": {
                "sigma_max": 0.15,
                "sigma_soft": 0.55,
                "beta_uncertainty": 1.0,
                "gamma": 1.0,
                "enforce_active_floor_one": True,
            },
            "tier_counts": {k: dict(v) for k, v in tier_counts.items()},
            "latency_trace_table_ms": {k: _latency_summary(v) for k, v in lat_trace_ms.items()},
            "latency_end_to_end_online_het_ms": {k: _latency_summary(v) for k, v in e2e_lat_ms.items()},
            "states": state_diagnostics,
        }, fh, indent=2)

    arm_map_path = os.path.join(DEC_DIR, "A5_GATE1_ARM_MAP.json")
    with open(arm_map_path, "w", encoding="utf-8") as fh:
        json.dump(arm_map, fh, indent=2)

    print(f"Planned {len(replay_plan_rows)} unique ns-3 replays across {len(state_diagnostics)} states (6*25=150 logical evaluations).")
    print("Tier counts:", {k: dict(v) for k, v in tier_counts.items()})
    for k in selectors:
        lt = _latency_summary(lat_trace_ms[k])
        le = _latency_summary(e2e_lat_ms[k])
        print(f"  {k}: trace_opt p50={lt['p50_ms']:.2f} p95={lt['p95_ms']:.2f} p99={lt['p99_ms']:.2f} max={lt['max_ms']:.2f} ms | e2e_het p50={le['p50_ms']:.2f} p95={le['p95_ms']:.2f} max={le['max_ms']:.2f} ms")


if __name__ == "__main__":
    main()
