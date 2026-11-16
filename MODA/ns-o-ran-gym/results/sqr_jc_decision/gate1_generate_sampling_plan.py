#!/usr/bin/env python3
"""Gate 1 Step 2: Deterministic, pre-registered stratified sampling plan.

Reads EXCLUSIVELY frozen B0-rest traced telemetry/decisions/traces strictly
before any IPC 50ms timeout step (`cut_step`), WITHOUT importing or evaluating
A5JointSelector (zero A5 look-ahead or cherry-picking).

Outputs (committed BEFORE running A5 decisions or replays):
  - A5_GATE1_SAMPLING_PLAN.csv
  - A5_GATE1_SAMPLING_MANIFEST.json
"""
from __future__ import annotations

import csv
import glob
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Any

import pandas as pd

REPO_ROOT = "/home/eliothluy/Documentos/Jussi_SBRC"
GYM_DIR = os.path.join(REPO_ROOT, "MODA", "ns-o-ran-gym")
SRC_DIR = os.path.join(GYM_DIR, "src")
BASE_DEC_DIR = os.path.join(GYM_DIR, "results", "sqr_jc_decision")
TRACED_DIR = os.path.join(BASE_DEC_DIR, "b0_traced_gate1")
PARITY_JSON = os.path.join(TRACED_DIR, "b0_traced_parity_report.json")
DOCS_DIR = os.path.join(REPO_ROOT, "MODA", "docs", "sqr", "jc_decision")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from nsoran.sqr_jc.a5_joint_selector import causal_active_slices  # noqa: E402
from nsoran.sqr_jc.freeze_contract import sha256_file  # noqa: E402


def _find_single(run_dir: str, fname: str) -> str:
    hits = sorted(glob.glob(os.path.join(run_dir, "**", fname), recursive=True))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected 1 {fname} under {run_dir}, got {len(hits)}")
    return hits[0]


def _pick_uniform_indices(n: int, k: int) -> list[int]:
    """Deterministic quantile-stride indices in sorted order without RNG."""
    if n < k:
        raise ValueError(f"Pool size {n} < requested k={k}")
    if k == 1:
        return [n // 2]
    return [int((2 * i + 1) * n // (2 * k)) for i in range(k)]


def main() -> None:
    parity = json.loads(open(PARITY_JSON, encoding="utf-8").read())
    if not parity.get("all_ok"):
        raise RuntimeError("Parity check did not pass; aborting sampling plan generation.")
    cut_by_tag = {r["tag"]: int(r["clean_cut_step"]) for r in parity["runs"]}

    records: list[dict[str, Any]] = []
    for run_rep in parity["runs"]:
        tag = run_rep["tag"]
        lam = int(run_rep["lam"])
        seed = int(run_rep["seed"])
        cut_step = int(run_rep["clean_cut_step"])
        run_dir = os.path.join(TRACED_DIR, tag)

        prb_df = pd.read_csv(_find_single(run_dir, "sqr_prb_windows.csv"))
        prb_by_w = {
            int(r.timestamp_ms) // 100 - 1: (
                int(r.prb_executed_rbg_0),
                int(r.prb_executed_rbg_1),
                int(r.prb_executed_rbg_2),
            )
            for r in prb_df.itertuples()
        }
        kpm_df = pd.read_csv(_find_single(run_dir, "sqr_kpm_history.csv"))
        ku = kpm_df[kpm_df["sliceId"] == 1].copy()
        ku["win"] = ku["timestamp"] // 100 - 1
        tx_u_by_w = ku.groupby("win")["dTxPackets"].sum().to_dict()

        ul_by_w: dict[int, dict[str, Any]] = {}
        with open(os.path.join(run_dir, "urllc_delayed_updates.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    ul_by_w[int(rec["w"])] = rec

        with open(os.path.join(run_dir, "decisions.jsonl"), encoding="utf-8") as fh:
            het_decs = [json.loads(line) for line in fh if line.strip()]
        het_decs = [d for d in het_decs if d["policy"] == "SQR_HET"]

        with gzip.open(os.path.join(run_dir, "causal_trace.jsonl.gz"), "rt", encoding="utf-8") as fh:
            trace_rows = [json.loads(line) for line in fh if line.strip()][1:]

        assert len(het_decs) == len(trace_rows)

        for tw, dec in zip(trace_rows, het_decs):
            step_idx = int(tw["step_index"])
            if step_idx >= cut_step or dec.get("missing_reason"):
                continue
            gov_w = int(tw["governed_window_index"])
            if gov_w != int(dec["outcome_window_index"]):
                continue
            act_exec = tuple(int(v) for v in tw["a_exec_per_slice"])
            if prb_by_w.get(gov_w) != act_exec:
                continue

            st = tw["state"]
            act_mask, _live_mask = causal_active_slices(st)
            n_act = sum(1 for x in act_mask if x)
            n_ue_nz = sum(1 for k in ("n_e", "n_u", "n_m") if int(st[k]) > 0)
            sum_prop = sum(int(v) for v in tw["a_prop_per_slice"])
            prop_excess = sum_prop > int(tw["capacity_C"])

            margins = dec.get("sla_outcome_margins") or [None, None, None]
            e_ok = (float(margins[0]) >= 0.0) if margins[0] is not None else None
            m_ok = (float(margins[2]) >= 0.0) if margins[2] is not None else None
            tx_w = int(tx_u_by_w.get(gov_w, 0))
            u_ok: bool | None = None
            u_status = "ok"
            if margins[1] is not None:
                u_ok = float(margins[1]) >= 0.0
            elif tx_w == 0:
                u_status = "u_no_demand"
            else:
                u_rec = ul_by_w.get(gov_w)
                if u_rec and u_rec.get("tag") == "CLOSED" and u_rec.get("p95_ms") is not None:
                    u_ok = float(u_rec["p95_ms"]) <= 10.0
                else:
                    u_status = "u_unclosed_online"

            if e_ok is not None and u_ok is not None and m_ok is not None:
                all3 = "PASS" if (e_ok and u_ok and m_ok) else "FAIL"
            else:
                all3 = "UNKNOWN"

            dem_oks = []
            if margins[0] is not None:
                dem_oks.append(e_ok)
            if (tx_w > 0) or (margins[1] is not None):
                dem_oks.append(u_ok)
            if margins[2] is not None:
                dem_oks.append(m_ok)
            if not dem_oks:
                dem_lbl = "NO_DEMAND"
            elif any(x is None for x in dem_oks):
                dem_lbl = "UNKNOWN"
            else:
                dem_lbl = "PASS" if all(x is True for x in dem_oks) else "FAIL"

            # Exclude late boundary T where +3s tail reaches EOF (3000) just in case
            if gov_w >= 2960:
                continue

            records.append({
                "state_id": f"lam{lam}_s{seed}_w{gov_w}",
                "b0_traced_tag": tag,
                "b0_traced_dir": run_dir,
                "lam": lam,
                "seed": seed,
                "step_index": step_idx,
                "state_window_index": int(tw["window_index"]),
                "governed_window_index": gov_w,
                "sim_time_ms": int(tw["sim_time_ms"]),
                "capacity_C": int(tw["capacity_C"]),
                "n_active_causal": n_act,
                "n_ue_nonzero": n_ue_nz,
                "active_mask": "".join("1" if x else "0" for x in act_mask),
                "prop_excess_gt53": prop_excess,
                "sum_a_prop": sum_prop,
                "b0_a_raw": json.dumps(tw["a_raw_per_slice"]),
                "b0_a_prop": json.dumps(tw["a_prop_per_slice"]),
                "b0_a_exec": json.dumps(list(act_exec)),
                "b0_modes": json.dumps(tw["modes_per_slice"]),
                "b0_all3_r19": all3,
                "b0_demand_r19": dem_lbl,
                "b0_u_status": u_status,
                "b0_margins_online": json.dumps(margins),
            })

    # Pre-registered sampling quotas across the 3 explicit universes:
    # Universe A (ALL3_COMPLETE, 18 states): valid STRICT_ALL3_R19 evaluation
    #   - 16 states across (lam in {5,6} x seed in {2102,2103} x {PASS,FAIL} x {excess=False,True}) with n_act=3
    #   -  2 states with n_act=2 & ALL3=FAIL (lam5_s2102 + lam6_s2103)
    # Universe B (LT3_DEMAND_COMPLETE, 5 states): 1 & 2 active slices across lam in {2,5,6}, s2102 & s2103
    # Universe C (UNKNOWN_INTEGRITY, 2 states): isolated URLLC u_unclosed_online coverage/integrity states
    specs: list[tuple[str, str, int, dict[str, Any]]] = [
        # Universe A: ALL3 COMPLETE (PASS vs FAIL across 2 seeds x 2 loads x excess T/F)
        ("ALL3_COMPLETE", "A_lam5_s2102_PASS_noexc_act3", 1, {"lam": 5, "seed": 2102, "b0_all3_r19": "PASS", "n_active_causal": 3, "prop_excess_gt53": False}),
        ("ALL3_COMPLETE", "A_lam5_s2102_PASS_exc_act3",   1, {"lam": 5, "seed": 2102, "b0_all3_r19": "PASS", "n_active_causal": 3, "prop_excess_gt53": True}),
        ("ALL3_COMPLETE", "A_lam5_s2102_FAIL_noexc_act3", 1, {"lam": 5, "seed": 2102, "b0_all3_r19": "FAIL", "n_active_causal": 3, "prop_excess_gt53": False}),
        ("ALL3_COMPLETE", "A_lam5_s2102_FAIL_exc_act3",   1, {"lam": 5, "seed": 2102, "b0_all3_r19": "FAIL", "n_active_causal": 3, "prop_excess_gt53": True}),
        ("ALL3_COMPLETE", "A_lam5_s2102_FAIL_act2",       1, {"lam": 5, "seed": 2102, "b0_all3_r19": "FAIL", "n_active_causal": 2}),

        ("ALL3_COMPLETE", "A_lam5_s2103_PASS_noexc_act3", 1, {"lam": 5, "seed": 2103, "b0_all3_r19": "PASS", "n_active_causal": 3, "prop_excess_gt53": False}),
        ("ALL3_COMPLETE", "A_lam5_s2103_PASS_exc_act3",   1, {"lam": 5, "seed": 2103, "b0_all3_r19": "PASS", "n_active_causal": 3, "prop_excess_gt53": True}),
        ("ALL3_COMPLETE", "A_lam5_s2103_FAIL_noexc_act3", 1, {"lam": 5, "seed": 2103, "b0_all3_r19": "FAIL", "n_active_causal": 3, "prop_excess_gt53": False}),
        ("ALL3_COMPLETE", "A_lam5_s2103_FAIL_exc_act3",   1, {"lam": 5, "seed": 2103, "b0_all3_r19": "FAIL", "n_active_causal": 3, "prop_excess_gt53": True}),

        ("ALL3_COMPLETE", "A_lam6_s2102_PASS_noexc_act3", 1, {"lam": 6, "seed": 2102, "b0_all3_r19": "PASS", "n_active_causal": 3, "prop_excess_gt53": False}),
        ("ALL3_COMPLETE", "A_lam6_s2102_PASS_exc_act3",   1, {"lam": 6, "seed": 2102, "b0_all3_r19": "PASS", "n_active_causal": 3, "prop_excess_gt53": True}),
        ("ALL3_COMPLETE", "A_lam6_s2102_FAIL_noexc_act3", 1, {"lam": 6, "seed": 2102, "b0_all3_r19": "FAIL", "n_active_causal": 3, "prop_excess_gt53": False}),
        ("ALL3_COMPLETE", "A_lam6_s2102_FAIL_exc_act3",   1, {"lam": 6, "seed": 2102, "b0_all3_r19": "FAIL", "n_active_causal": 3, "prop_excess_gt53": True}),

        ("ALL3_COMPLETE", "A_lam6_s2103_PASS_noexc_act3", 1, {"lam": 6, "seed": 2103, "b0_all3_r19": "PASS", "n_active_causal": 3, "prop_excess_gt53": False}),
        ("ALL3_COMPLETE", "A_lam6_s2103_PASS_exc_act3",   1, {"lam": 6, "seed": 2103, "b0_all3_r19": "PASS", "n_active_causal": 3, "prop_excess_gt53": True}),
        ("ALL3_COMPLETE", "A_lam6_s2103_FAIL_noexc_act3", 1, {"lam": 6, "seed": 2103, "b0_all3_r19": "FAIL", "n_active_causal": 3, "prop_excess_gt53": False}),
        ("ALL3_COMPLETE", "A_lam6_s2103_FAIL_exc_act3",   1, {"lam": 6, "seed": 2103, "b0_all3_r19": "FAIL", "n_active_causal": 3, "prop_excess_gt53": True}),
        ("ALL3_COMPLETE", "A_lam6_s2103_FAIL_act2",       1, {"lam": 6, "seed": 2103, "b0_all3_r19": "FAIL", "n_active_causal": 2}),

        # Universe B: <3 active slices (STRICT_DEMAND_R19 complete, STRICT_ALL3 structurally UNKNOWN)
        ("LT3_DEMAND_COMPLETE", "B_lam2_s2102_act1_FAIL",     1, {"lam": 2, "seed": 2102, "b0_all3_r19": "UNKNOWN", "b0_demand_r19": "FAIL", "n_active_causal": 1}),
        ("LT3_DEMAND_COMPLETE", "B_lam2_s2102_act2_PASS_exc", 1, {"lam": 2, "seed": 2102, "b0_all3_r19": "UNKNOWN", "b0_demand_r19": "PASS", "n_active_causal": 2, "prop_excess_gt53": True, "b0_u_status": "ok"}),
        ("LT3_DEMAND_COMPLETE", "B_lam2_s2102_act1_PASS",     1, {"lam": 2, "seed": 2102, "b0_all3_r19": "UNKNOWN", "b0_demand_r19": "PASS", "n_active_causal": 1}),
        ("LT3_DEMAND_COMPLETE", "B_lam5_s2103_act1_FAIL",     1, {"lam": 5, "seed": 2103, "b0_all3_r19": "UNKNOWN", "b0_demand_r19": "FAIL", "n_active_causal": 1, "b0_u_status": "ok"}),
        ("LT3_DEMAND_COMPLETE", "B_lam6_s2102_act2_FAIL_exc", 1, {"lam": 6, "seed": 2102, "b0_all3_r19": "UNKNOWN", "b0_demand_r19": "FAIL", "n_active_causal": 2, "prop_excess_gt53": True}),

        # Universe C: UNKNOWN integrity / coverage verification ONLY
        ("UNKNOWN_INTEGRITY", "C_lam2_s2102_unk_unclosed", 1, {"lam": 2, "seed": 2102, "b0_u_status": "u_unclosed_online"}),
        ("UNKNOWN_INTEGRITY", "C_lam5_s2103_unk_unclosed", 1, {"lam": 5, "seed": 2103, "b0_u_status": "u_unclosed_online"}),
    ]

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    stratum_summary: list[dict[str, Any]] = []

    for universe, stratum_id, k, filt in specs:
        pool = [
            r for r in records
            if all(r[fk] == fv for fk, fv in filt.items())
            and r["state_id"] not in seen_ids
        ]
        pool.sort(key=lambda r: (r["lam"], r["seed"], r["governed_window_index"]))
        idxs = _pick_uniform_indices(len(pool), k)
        for idx in idxs:
            row = dict(pool[idx])
            row["universe"] = universe
            row["stratum_id"] = stratum_id
            row["stratum_pool_size"] = len(pool)
            row["picked_pool_rank"] = idx
            seen_ids.add(row["state_id"])
            selected.append(row)
        stratum_summary.append({
            "universe": universe,
            "stratum_id": stratum_id,
            "requested_k": k,
            "pool_size": len(pool),
            "filter": filt,
            "picked_indices": idxs,
            "picked_state_ids": [pool[idx]["state_id"] for idx in idxs],
        })

    # Write CSV plan to both results/sqr_jc_decision and docs/sqr/jc_decision
    csv_cols = [
        "state_id", "universe", "stratum_id", "b0_traced_tag", "lam", "seed",
        "step_index", "state_window_index", "governed_window_index", "sim_time_ms",
        "capacity_C", "n_active_causal", "n_ue_nonzero", "active_mask",
        "prop_excess_gt53", "sum_a_prop", "b0_a_raw", "b0_a_prop", "b0_a_exec",
        "b0_modes", "b0_all3_r19", "b0_demand_r19", "b0_u_status", "b0_margins_online",
        "stratum_pool_size", "picked_pool_rank", "b0_traced_dir",
    ]
    out_csv = os.path.join(BASE_DEC_DIR, "A5_GATE1_SAMPLING_PLAN.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_cols)
        w.writeheader()
        for row in selected:
            w.writerow({c: row[c] for c in csv_cols})

    manifest = {
        "gate": "A5_GATE1_EXPLORATORY",
        "status": "PRE_REGISTERED_BEFORE_A5_EVALUATION",
        "n_total_states": len(selected),
        "n_by_universe": {
            u: sum(1 for r in selected if r["universe"] == u)
            for u in ("ALL3_COMPLETE", "LT3_DEMAND_COMPLETE", "UNKNOWN_INTEGRITY")
        },
        "selection_rule": (
            "Stratified deterministic quantile-stride selection over pre-IPC-timeout B0-rest "
            "traced snapshots (step < clean_cut_step, executed_mac == a_exec, governed_window < 2960). "
            "Zero access to A5JointSelector or replay outcomes prior to freezing this plan."
        ),
        "clean_cut_steps_by_run": cut_by_tag,
        "sampling_plan_csv_sha256": sha256_file(out_csv),
        "strata": stratum_summary,
    }
    out_manifest = os.path.join(BASE_DEC_DIR, "A5_GATE1_SAMPLING_MANIFEST.json")
    with open(out_manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Registered {len(selected)} stratified states -> {out_csv}")
    print(f"Manifest sha256={manifest['sampling_plan_csv_sha256']} -> {out_manifest}")
    df = pd.DataFrame(selected)
    print(df[["state_id", "universe", "stratum_id", "n_active_causal", "prop_excess_gt53", "b0_all3_r19", "b0_demand_r19", "b0_a_exec", "stratum_pool_size"]].to_string(index=False))


if __name__ == "__main__":
    main()
