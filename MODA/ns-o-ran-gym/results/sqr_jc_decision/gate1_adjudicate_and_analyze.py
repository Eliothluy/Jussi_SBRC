#!/usr/bin/env python3
"""Gate 1 Step 4: Replay prefix/A-A integrity check, R19 counterfactual adjudication,
and scientific diagnostics (denominators, Rescue/Harm/NetGain, scale dominance).

Usage:
  .venv/bin/python results/sqr_jc_decision/gate1_adjudicate_and_analyze.py [--smoke-tag TAG]

Outputs (full mode):
  - A5_GATE1_REPLAY_INTEGRITY.json
  - A5_GATE1_METRICS.csv          (150 logical arm-state rows + outcomes)
  - A5_GATE1_DIAGNOSTICS.json     (aggregated metrics, denominators, transitions, scale dominance)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
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

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from types import SimpleNamespace

from nsoran.sqr.sqr_kpis import SqrKpmWindow  # noqa: E402
from nsoran.sqr_jc.het_runtime import HetRuntime  # noqa: E402
from nsoran.sqr_jc.sla_margin import SLA_MMTC_PDR, SLA_URLLC_MS  # noqa: E402


def _find_single(run_dir: str, fname: str) -> str:
    hits = sorted(glob.glob(os.path.join(run_dir, "**", fname), recursive=True))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected 1 {fname} under {run_dir}, got {len(hits)}")
    return hits[0]


def _check_replay_integrity(
    *,
    rep_dir: str,
    base_dir: str,
    gov_w: int,
    expected_action: tuple[int, int, int],
    is_aa_b0: bool,
) -> dict[str, Any]:
    """Verify monotonic stamps, C=53, override window, executed quota, and
    bit-exact pre-override parity across KPM, PRB, and delivered packet delays."""
    stamp_max = gov_w * 100
    gov_stamp = (gov_w + 1) * 100
    out: dict[str, Any] = {
        "done": os.path.isfile(os.path.join(rep_dir, ".done")),
        "valid": False,
        "aa_window_exact": None,
        "failure": None,
    }
    if not out["done"]:
        out["failure"] = "missing .done"
        return out

    log_txt = open(os.path.join(rep_dir, "run.log"), encoding="utf-8", errors="ignore").read()
    out["ns3_rc0"] = "ns3_exit_code: 0" in log_txt
    if not out["ns3_rc0"]:
        out["failure"] = "ns3_exit_code != 0"
        return out

    rep_kpm = pd.read_csv(_find_single(rep_dir, "sqr_kpm_history.csv"))
    rep_prb = pd.read_csv(_find_single(rep_dir, "sqr_prb_windows.csv"))
    rep_del = pd.read_csv(_find_single(rep_dir, "sqr_packet_delays.csv"))
    rep_step = pd.read_csv(os.path.join(rep_dir, "step_metrics.csv"))

    base_kpm = pd.read_csv(_find_single(base_dir, "sqr_kpm_history.csv"))
    base_prb = pd.read_csv(_find_single(base_dir, "sqr_prb_windows.csv"))
    base_del = pd.read_csv(_find_single(base_dir, "sqr_packet_delays.csv"))

    out["kpm_missing_total"] = int(rep_step["kpm_missing"].sum()) if "kpm_missing" in rep_step else -1
    out["total_rbg_all_53"] = bool((rep_prb["total_rbg"] == 53).all())
    out["monotonic_kpm"] = bool(np.all(np.diff(rep_kpm["timestamp"].values) >= 0))
    out["monotonic_prb"] = bool(np.all(np.diff(rep_prb["timestamp_ms"].values) >= 0))

    # Override step check & executed quota check at gov_w
    ov = rep_step[rep_step["mode"] == "replay_override"]
    out["override_steps"] = sorted(int(s) for s in ov["step"].unique().tolist())
    out["override_at_gov_w"] = bool(len(ov) == 3 and set(ov["step"]) == {gov_w})

    prb_t = rep_prb[rep_prb["timestamp_ms"] == gov_stamp]
    if len(prb_t) != 1:
        out["failure"] = f"missing prb row at stamp {gov_stamp}"
        return out
    exec_t = (
        int(prb_t.iloc[0]["prb_executed_rbg_0"]),
        int(prb_t.iloc[0]["prb_executed_rbg_1"]),
        int(prb_t.iloc[0]["prb_executed_rbg_2"]),
    )
    out["executed_at_gov_w"] = list(exec_t)
    out["executed_matches_plan"] = (exec_t == tuple(expected_action))

    # Bit-exact prefix checks (windows strictly before gov_w -> stamps <= gov_w * 100)
    rk_p = rep_kpm[rep_kpm["timestamp"] <= stamp_max].reset_index(drop=True)
    bk_p = base_kpm[base_kpm["timestamp"] <= stamp_max].reset_index(drop=True)
    out["kpm_prefix_exact"] = bool(rk_p.equals(bk_p))
    out["kpm_prefix_rows"] = int(len(rk_p))

    rp_p = rep_prb[rep_prb["timestamp_ms"] <= stamp_max].reset_index(drop=True)
    bp_p = base_prb[base_prb["timestamp_ms"] <= stamp_max].reset_index(drop=True)
    # Physical & executed quota columns; note window 0 (stamp 100) keeps initial sentinel
    # in base IPC while seq replay uses action 0 for window 0 if identical; compare all
    # columns for stamps >= 200 (and physical columns >= 100).
    phys_cols = [
        "timestamp_ms", "total_rbg", "total_prb", "dl_slots",
        "prb_scheduled_rbg_0", "prb_scheduled_rbg_1", "prb_scheduled_rbg_2",
        "prb_budget_rbg_0", "prb_budget_rbg_1", "prb_budget_rbg_2", "prb_idle_rbg",
    ]
    prb_phys_exact = bool(rp_p[phys_cols].equals(bp_p[phys_cols]))
    rp_1 = rp_p[rp_p["timestamp_ms"] >= 200].reset_index(drop=True)
    bp_1 = bp_p[bp_p["timestamp_ms"] >= 200].reset_index(drop=True)
    exec_cols = ["prb_executed_rbg_0", "prb_executed_rbg_1", "prb_executed_rbg_2"]
    prb_exec_exact = bool(rp_1[exec_cols].equals(bp_1[exec_cols]))
    out["prb_prefix_phys_exact"] = prb_phys_exact
    out["prb_prefix_exec_exact"] = prb_exec_exact
    out["prb_prefix_rows"] = int(len(rp_p))

    rd_p = (
        rep_del[(rep_del["txSec"] + rep_del["delaySec"]) * 1000.0 < stamp_max]
        .sort_values(["txSec", "delaySec"])
        .reset_index(drop=True)
    )
    bd_p = (
        base_del[(base_del["txSec"] + base_del["delaySec"]) * 1000.0 < stamp_max]
        .sort_values(["txSec", "delaySec"])
        .reset_index(drop=True)
    )
    out["delay_prefix_exact"] = bool(rd_p.equals(bd_p))
    out["delay_prefix_rows"] = int(len(rd_p))

    if is_aa_b0:
        rk_w = rep_kpm[rep_kpm["timestamp"] == gov_stamp].reset_index(drop=True)
        bk_w = base_kpm[base_kpm["timestamp"] == gov_stamp].reset_index(drop=True)
        rp_w = rep_prb[rep_prb["timestamp_ms"] == gov_stamp].reset_index(drop=True)
        bp_w = base_prb[base_prb["timestamp_ms"] == gov_stamp].reset_index(drop=True)
        rd_w = (
            rep_del[(rep_del["sliceId"] == 1) & (np.floor(rep_del["txSec"] * 10.0) == gov_w)]
            .sort_values(["txSec", "delaySec"])
            .reset_index(drop=True)
        )
        bd_w = (
            base_del[(base_del["sliceId"] == 1) & (np.floor(base_del["txSec"] * 10.0) == gov_w)]
            .sort_values(["txSec", "delaySec"])
            .reset_index(drop=True)
        )
        out["aa_window_kpm_exact"] = bool(rk_w.equals(bk_w))
        out["aa_window_prb_exact"] = bool(rp_w[phys_cols + exec_cols].equals(bp_w[phys_cols + exec_cols]))
        out["aa_window_delay_exact"] = bool(rd_w.equals(bd_w))
        out["aa_window_exact"] = bool(
            out["aa_window_kpm_exact"] and out["aa_window_prb_exact"] and out["aa_window_delay_exact"]
        )

    all_checks = (
        out["done"]
        and out["ns3_rc0"]
        and out["kpm_missing_total"] == 0
        and out["total_rbg_all_53"]
        and out["monotonic_kpm"]
        and out["monotonic_prb"]
        and out["override_at_gov_w"]
        and out["executed_matches_plan"]
        and out["kpm_prefix_exact"]
        and out["prb_prefix_phys_exact"]
        and out["prb_prefix_exec_exact"]
        and out["delay_prefix_exact"]
        and (out["aa_window_exact"] is True if is_aa_b0 else True)
    )
    out["valid"] = bool(all_checks)
    if not out["valid"]:
        reasons = [
            k for k in (
                "override_at_gov_w", "executed_matches_plan", "kpm_prefix_exact",
                "prb_prefix_phys_exact", "prb_prefix_exec_exact", "delay_prefix_exact",
                "total_rbg_all_53", "monotonic_kpm", "monotonic_prb",
            )
            if not out.get(k)
        ]
        if is_aa_b0 and not out["aa_window_exact"]:
            reasons.append("aa_window_exact")
        if out["kpm_missing_total"] != 0:
            reasons.append("kpm_missing_total")
        out["failure"] = ",".join(reasons)
    return out


def _evaluate_window_outcome(rep_dir: str, gov_w: int) -> dict[str, Any]:
    """Compute exact R19 + HetRuntime 1s rolling PDR KPIs on governed window gov_w."""
    kpm_path = _find_single(rep_dir, "sqr_kpm_history.csv")
    prb_path = _find_single(rep_dir, "sqr_prb_windows.csv")
    sim_root = os.path.dirname(os.path.dirname(os.path.dirname(kpm_path)))

    kpm_df = pd.read_csv(kpm_path)
    prb_df = pd.read_csv(prb_path)
    gov_stamp = (gov_w + 1) * 100

    # Run HetRuntime cleanly sequentially over all KPM windows <= gov_stamp to mirror
    # exact mMTC rolling deque & DelayStream closing over the +3s tail.
    rt = HetRuntime(sim_root)
    tmp_csv_rows = kpm_df[kpm_df["timestamp"] <= gov_stamp]
    last_w = None
    for ts, grp in tmp_csv_rows.groupby("timestamp", sort=True):
        rows_ns = tuple(
            SimpleNamespace(
                slice_id=int(r.sliceId),
                d_tx_bytes=float(r.dTxBytes),
                window_throughput_mbps=float(r.windowThroughputMbps),
                d_tx_packets=float(getattr(r, "dTxPackets", 0)),
                d_rx_packets=float(getattr(r, "dRxPackets", 0)),
            )
            for r in grp.itertuples(index=False)
        )
        w_obj = SqrKpmWindow(
            timestamp=int(ts),
            slices={},
            total_ues=len(rows_ns),
            cell_snr_median=None,
            rows=rows_ns,
        )
        rt.absorb(w_obj)
        last_w = int(ts) // 100 - 1

    # Poll ALL flushed URLLC delays through the full +3s drain horizon
    rt.delays.poll()
    u_tag, u_stats = rt.urllc_pool_status(gov_w)
    oc = rt.outcome_for_window(gov_w)
    margins = oc.margins

    kpm_w = kpm_df[kpm_df["timestamp"] == gov_stamp]
    tx_e = int(kpm_w[kpm_w["sliceId"] == 0]["dTxBytes"].sum())
    tx_u = int(kpm_w[kpm_w["sliceId"] == 1]["dTxPackets"].sum())
    tx_m_win = int(kpm_w[kpm_w["sliceId"] == 2]["dTxPackets"].sum())
    rx_m_win = int(kpm_w[kpm_w["sliceId"] == 2]["dRxPackets"].sum())
    u_rx = len(rt.delays.by_win.get(gov_w, ()))

    # Strict R19: URLLC complete ONLY when tx_u > 0 AND CLOSED (u_rx >= tx_u)
    e_ok = (float(margins[0]) >= 0.0) if oc.embb_throughput_mbps is not None else None
    u_ok = (float(u_stats["p95_ms"]) <= SLA_URLLC_MS) if (u_tag == "CLOSED" and u_stats is not None) else None
    m_ok = (float(margins[2]) >= 0.0) if oc.mmtc_pdr_rolling is not None else None

    if e_ok is not None and u_ok is not None and m_ok is not None:
        strict_all3 = "PASS" if (e_ok and u_ok and m_ok) else "FAIL"
    else:
        strict_all3 = "UNKNOWN"

    e_dem = len(kpm_w[(kpm_w["sliceId"] == 0) & (kpm_w["dTxBytes"] > 0)]) > 0
    u_dem = tx_u > 0
    m_dem = oc.mmtc_pdr_rolling is not None

    dem_oks = []
    if e_dem:
        dem_oks.append(e_ok)
    if u_dem:
        dem_oks.append(u_ok)
    if m_dem:
        dem_oks.append(m_ok)
    if not dem_oks:
        strict_demand = "NO_DEMAND"
    elif any(x is None for x in dem_oks):
        strict_demand = "UNKNOWN"
    else:
        strict_demand = "PASS" if all(x is True for x in dem_oks) else "FAIL"

    prb_w = prb_df[prb_df["timestamp_ms"] == gov_stamp].iloc[0]
    dl_slots = int(prb_w["dl_slots"])
    cap = int(prb_w["total_rbg"])
    sched = [int(prb_w[f"prb_scheduled_rbg_{i}"]) for i in range(3)]
    exec_rbg = [int(prb_w[f"prb_executed_rbg_{i}"]) for i in range(3)]
    util_dl_pct = (100.0 * sum(sched) / (cap * dl_slots)) if dl_slots > 0 else float("nan")

    # Sensitivity all_rx URLLC p95 even if unclosed loss
    all_rx_ms = rt.delays.by_win.get(gov_w, ())
    u_p95_allrx = float(np.percentile(all_rx_ms, 95.0)) if all_rx_ms else None
    u_miss_allrx = float(100.0 * np.mean(np.asarray(all_rx_ms) > SLA_URLLC_MS)) if all_rx_ms else None

    return {
        "gov_w": int(gov_w),
        "last_absorbed_w": last_w,
        "embb_thr_med_mbps": oc.embb_throughput_mbps,
        "embb_tx_bytes": tx_e,
        "embb_margin": margins[0],
        "embb_ok": e_ok,
        "urllc_tx_pkts": tx_u,
        "urllc_rx_pkts": u_rx,
        "urllc_pool_tag": u_tag,
        "urllc_p95_ms": (u_stats["p95_ms"] if (u_tag == "CLOSED" and u_stats) else None),
        "urllc_miss_pct": (100.0 * u_stats["miss_rate"] if (u_tag == "CLOSED" and u_stats) else None),
        "urllc_p95_allrx_ms": u_p95_allrx,
        "urllc_miss_allrx_pct": u_miss_allrx,
        "urllc_margin_r19": (float(10.0 - u_stats["p95_ms"]) / 10.0 if (u_tag == "CLOSED" and u_stats) else None),
        "urllc_ok": u_ok,
        "mmtc_pdr_rolling": oc.mmtc_pdr_rolling,
        "mmtc_win_tx_pkts": tx_m_win,
        "mmtc_win_rx_pkts": rx_m_win,
        "mmtc_margin": margins[2],
        "mmtc_ok": m_ok,
        "strict_all3_r19": strict_all3,
        "strict_demand_r19": strict_demand,
        "dl_slots": dl_slots,
        "exec_rbg": exec_rbg,
        "exec_sum_rbg": int(sum(exec_rbg)),
        "sched_rbg": sched,
        "util_dl_pct": float(util_dl_pct),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-tag", type=str, default=None, help="Verify a single replay tag (smoke mode)")
    args = ap.parse_args()

    plan_rows = list(csv.DictReader(open(os.path.join(DEC_DIR, "A5_GATE1_REPLAY_PLAN.tsv"), encoding="utf-8"), delimiter="\t"))
    if args.smoke_tag:
        hits = [r for r in plan_rows if r["tag"] == args.smoke_tag]
        if len(hits) != 1:
            raise ValueError(f"Unknown smoke tag: {args.smoke_tag}")
        r = hits[0]
        act = (int(r["ae"]), int(r["au"]), int(r["am"]))
        chk = _check_replay_integrity(
            rep_dir=r["out_dir"],
            base_dir=r["b0_traced_dir"],
            gov_w=int(r["gov_w"]),
            expected_action=act,
            is_aa_b0=(int(r["is_b0_aa"]) == 1),
        )
        oc = _evaluate_window_outcome(r["out_dir"], int(r["gov_w"]))
        print(json.dumps({"integrity": chk, "outcome": oc}, indent=2))
        if not chk["valid"]:
            sys.exit(1)
        return

    # Full Gate 1 verification + adjudication
    decisions_doc = json.load(open(os.path.join(DEC_DIR, "A5_GATE1_DECISIONS.json"), encoding="utf-8"))
    arm_map = json.load(open(os.path.join(DEC_DIR, "A5_GATE1_ARM_MAP.json"), encoding="utf-8"))
    state_by_id = {s["state_id"]: s for s in decisions_doc["states"]}

    replay_integrity: dict[str, dict[str, Any]] = {}
    replay_outcomes: dict[str, dict[str, Any]] = {}
    aa_ok_by_state: dict[str, bool] = {}

    # 1. Integrity & A/A verification on every unique replay directory
    for r in plan_rows:
        tag = r["tag"]
        sid = r["state_id"]
        act = (int(r["ae"]), int(r["au"]), int(r["am"]))
        is_aa = int(r["is_b0_aa"]) == 1
        chk = _check_replay_integrity(
            rep_dir=r["out_dir"],
            base_dir=r["b0_traced_dir"],
            gov_w=int(r["gov_w"]),
            expected_action=act,
            is_aa_b0=is_aa,
        )
        replay_integrity[tag] = chk
        if is_aa:
            aa_ok_by_state[sid] = bool(chk["valid"])
        if chk["valid"]:
            replay_outcomes[tag] = _evaluate_window_outcome(r["out_dir"], int(r["gov_w"]))
        else:
            replay_outcomes[tag] = {}

    n_valid = sum(1 for v in replay_integrity.values() if v["valid"])
    n_aa_ok = sum(1 for v in aa_ok_by_state.values() if v)
    print(f"Replay integrity: {n_valid}/{len(replay_integrity)} valid | A/A B0 parity: {n_aa_ok}/{len(aa_ok_by_state)}")
    if n_valid != len(replay_integrity) or n_aa_ok != len(aa_ok_by_state):
        bad = {k: v["failure"] for k, v in replay_integrity.items() if not v["valid"]}
        print("INTEGRITY FAILURES:", json.dumps(bad, indent=2))
        sys.exit(1)

    # 2. Build flat 150-row table (25 states x 6 arms)
    flat_rows: list[dict[str, Any]] = []
    arms_order = ("b0", "eq", "prop", "b1_raw", "b1_act", "b1_sup")
    for sid, st_meta in state_by_id.items():
        if not aa_ok_by_state.get(sid):
            raise RuntimeError(f"State {sid} failed A/A B0 check")
        b0_tag = arm_map[sid]["b0"]["replay_tag"]
        b0_out = replay_outcomes[b0_tag]
        for arm in arms_order:
            am = arm_map[sid][arm]
            rtag = am["replay_tag"]
            oc = replay_outcomes[rtag]
            act = am["action"]
            a5_det = st_meta["a5_details"].get(arm) or {}
            act_mask = st_meta["active_mask"]
            inactive_prbs = sum(act[i] for i in range(3) if not act_mask[i])
            live_mask = st_meta["live_floor_mask"]
            starved_live = sum(1 for i in range(3) if live_mask[i] and act[i] == 0)

            flat_rows.append({
                "state_id": sid,
                "universe": st_meta["universe"],
                "stratum_id": st_meta["stratum_id"],
                "lam": st_meta["lam"],
                "seed": st_meta["seed"],
                "gov_w": st_meta["governed_window_index"],
                "n_active_causal": st_meta["n_active_causal"],
                "active_mask": "".join("1" if x else "0" for x in act_mask),
                "prop_excess_gt53": st_meta["prop_excess_gt53"],
                "b0_online_all3": st_meta["b0_all3_r19"],
                "b0_online_demand": st_meta["b0_demand_r19"],
                "arm": arm,
                "replay_tag": rtag,
                "reused_action": am["reused_action"],
                "aa_prefix_valid": replay_integrity[rtag]["valid"],
                "action_e": act[0],
                "action_u": act[1],
                "action_m": act[2],
                "action_sum": sum(act),
                "inactive_prbs_allocated": inactive_prbs,
                "starved_live_slices": starved_live,
                "tier": a5_det.get("tier", "b0_or_static"),
                "bottleneck_slice": a5_det.get("bottleneck_slice"),
                "fallback_used": a5_det.get("fallback_used", False),
                "degenerate_avoided": a5_det.get("degenerate_avoided", False),
                "support_violated_gt015": a5_det.get("support_violated_gt015", False),
                "support_relaxed_gt055": a5_det.get("support_relaxed_gt055", False),
                "max_chosen_sigma": a5_det.get("max_chosen_sigma"),
                "critical_score": a5_det.get("critical_score"),
                "strict_all3_r19": oc["strict_all3_r19"],
                "strict_demand_r19": oc["strict_demand_r19"],
                "embb_ok": oc["embb_ok"],
                "urllc_ok": oc["urllc_ok"],
                "mmtc_ok": oc["mmtc_ok"],
                "embb_thr_mbps": oc["embb_thr_med_mbps"],
                "urllc_p95_ms": oc["urllc_p95_ms"],
                "urllc_miss_pct": oc["urllc_miss_pct"],
                "urllc_pool_tag": oc["urllc_pool_tag"],
                "urllc_tx_pkts": oc["urllc_tx_pkts"],
                "urllc_rx_pkts": oc["urllc_rx_pkts"],
                "mmtc_pdr_rolling": oc["mmtc_pdr_rolling"],
                "util_dl_pct": oc["util_dl_pct"],
                "b0_replay_all3": b0_out["strict_all3_r19"],
                "b0_replay_demand": b0_out["strict_demand_r19"],
                "delta_embb_thr_vs_b0": (
                    (oc["embb_thr_med_mbps"] - b0_out["embb_thr_med_mbps"])
                    if (oc["embb_thr_med_mbps"] is not None and b0_out["embb_thr_med_mbps"] is not None)
                    else None
                ),
                "delta_urllc_p95_vs_b0": (
                    (oc["urllc_p95_ms"] - b0_out["urllc_p95_ms"])
                    if (oc["urllc_p95_ms"] is not None and b0_out["urllc_p95_ms"] is not None)
                    else None
                ),
            })

    df = pd.DataFrame(flat_rows)
    metrics_csv = os.path.join(DEC_DIR, "A5_GATE1_METRICS.csv")
    df.to_csv(metrics_csv, index=False)

    # 3. Compute strict Universe-separated Rescue/Harm/NetGain tables
    def _eval_universe(sub: pd.DataFrame, col: str, b0_col: str) -> dict[str, Any]:
        out_u: dict[str, Any] = {}
        for arm in arms_order:
            a_df = sub[sub["arm"] == arm]
            n_tot = int(len(a_df))
            p_cnt = int((a_df[col] == "PASS").sum())
            f_cnt = int((a_df[col] == "FAIL").sum())
            u_cnt = int((a_df[col].isin(["UNKNOWN", "NO_DEMAND"])).sum())
            known = p_cnt + f_cnt

            # Comparable pairs where BOTH B0 and Arm are COMPLETE
            comp_fail_b0 = a_df[(a_df[b0_col] == "FAIL") & (a_df[col].isin(["PASS", "FAIL"]))]
            comp_pass_b0 = a_df[(a_df[b0_col] == "PASS") & (a_df[col].isin(["PASS", "FAIL"]))]
            rescued = int((comp_fail_b0[col] == "PASS").sum())
            harmed = int((comp_pass_b0[col] == "FAIL").sum())
            n_b0_fail_comp = int(len(comp_fail_b0))
            n_b0_pass_comp = int(len(comp_pass_b0))
            n_comp_all = n_b0_fail_comp + n_b0_pass_comp

            rescue_rate = (rescued / n_b0_fail_comp) if n_b0_fail_comp > 0 else float("nan")
            harm_rate = (harmed / n_b0_pass_comp) if n_b0_pass_comp > 0 else float("nan")
            net_gain_overall = ((rescued - harmed) / n_comp_all) if n_comp_all > 0 else float("nan")

            out_u[arm] = {
                "n_states": n_tot,
                "pass": p_cnt,
                "fail": f_cnt,
                "unknown": u_cnt,
                "known": known,
                "coverage_pct": (100.0 * known / n_tot) if n_tot else float("nan"),
                "cond_pass_pct": (100.0 * p_cnt / known) if known else float("nan"),
                "b0_fail_comparable_denom": n_b0_fail_comp,
                "rescued_count": rescued,
                "rescue_rate": rescue_rate,
                "b0_pass_comparable_denom": n_b0_pass_comp,
                "harmed_count": harmed,
                "harm_rate": harm_rate,
                "net_rescued_minus_harmed": rescued - harmed,
                "net_gain_comparable_denom": n_comp_all,
                "net_gain_overall": net_gain_overall,
                "net_rate_diff_rescue_minus_harm": (
                    (rescue_rate - harm_rate)
                    if (np.isfinite(rescue_rate) and np.isfinite(harm_rate))
                    else float("nan")
                ),
                "inactive_prb_waste_events": int((a_df["inactive_prbs_allocated"] > 0).sum()),
                "starved_live_events": int((a_df["starved_live_slices"] > 0).sum()),
                "mean_util_dl_pct": float(a_df["util_dl_pct"].mean()),
                "mean_action_sum": float(a_df["action_sum"].mean()),
            }
        return out_u

    u_a = _eval_universe(df[df["universe"] == "ALL3_COMPLETE"], "strict_all3_r19", "b0_replay_all3")
    u_a_act3 = _eval_universe(
        df[(df["universe"] == "ALL3_COMPLETE") & (df["n_active_causal"] == 3)],
        "strict_all3_r19", "b0_replay_all3",
    )
    u_a_lam5 = _eval_universe(df[(df["universe"] == "ALL3_COMPLETE") & (df["lam"] == 5)], "strict_all3_r19", "b0_replay_all3")
    u_a_lam6 = _eval_universe(df[(df["universe"] == "ALL3_COMPLETE") & (df["lam"] == 6)], "strict_all3_r19", "b0_replay_all3")
    u_a_s2102 = _eval_universe(df[(df["universe"] == "ALL3_COMPLETE") & (df["seed"] == 2102)], "strict_all3_r19", "b0_replay_all3")
    u_a_s2103 = _eval_universe(df[(df["universe"] == "ALL3_COMPLETE") & (df["seed"] == 2103)], "strict_all3_r19", "b0_replay_all3")
    u_b = _eval_universe(df[df["universe"] == "LT3_DEMAND_COMPLETE"], "strict_demand_r19", "b0_replay_demand")

    # Unknown integrity universe (Universe C)
    u_c_rows = df[df["universe"] == "UNKNOWN_INTEGRITY"].to_dict("records")

    # Detailed scale/bottleneck dominance across A5 arms
    scale_dom: dict[str, Any] = {}
    for arm in ("b1_raw", "b1_act", "b1_sup"):
        sub_a = df[df["arm"] == arm]
        bn_counts = Counter(int(x) for x in sub_a["bottleneck_slice"].dropna().tolist())
        collapsed_e_1 = int(((sub_a["action_e"] <= 1) & (sub_a["active_mask"].str[0] == "1")).sum())
        collapsed_m_1 = int(((sub_a["action_m"] <= 1) & (sub_a["active_mask"].str[2] == "1")).sum())
        idle_left = int((sub_a["action_sum"] < 53).sum())
        harm_rows = sub_a[(sub_a["universe"] == "ALL3_COMPLETE") & (sub_a["b0_replay_all3"] == "PASS") & (sub_a["strict_all3_r19"] == "FAIL")][
            ["state_id", "lam", "seed", "action_e", "action_u", "action_m", "embb_ok", "urllc_ok", "mmtc_ok", "embb_thr_mbps", "urllc_p95_ms", "mmtc_pdr_rolling"]
        ].to_dict("records")
        rescue_rows = sub_a[(sub_a["universe"] == "ALL3_COMPLETE") & (sub_a["b0_replay_all3"] == "FAIL") & (sub_a["strict_all3_r19"] == "PASS")][
            ["state_id", "lam", "seed", "action_e", "action_u", "action_m", "embb_thr_mbps", "urllc_p95_ms", "mmtc_pdr_rolling"]
        ].to_dict("records")
        scale_dom[arm] = {
            "bottleneck_slice_counts": {str(k): v for k, v in sorted(bn_counts.items())},
            "active_embb_le_1_count": collapsed_e_1,
            "active_mmtc_le_1_count": collapsed_m_1,
            "action_sum_lt_53_count": idle_left,
            "harm_cases_all3": harm_rows,
            "rescue_cases_all3": rescue_rows,
        }

    diag = {
        "integrity_summary": {
            "total_unique_replays": len(replay_integrity),
            "valid_replays": n_valid,
            "aa_states_verified": n_aa_ok,
            "total_states": len(aa_ok_by_state),
        },
        "universe_A_ALL3_COMPLETE": u_a,
        "universe_A_by_stratum": {
            "act3_only": u_a_act3,
            "lam5": u_a_lam5,
            "lam6": u_a_lam6,
            "seed_2102": u_a_s2102,
            "seed_2103": u_a_s2103,
        },
        "universe_B_LT3_DEMAND_COMPLETE": u_b,
        "universe_C_UNKNOWN_INTEGRITY": u_c_rows,
        "scale_dominance_and_cases": scale_dom,
        "latency_trace_table_ms": decisions_doc["latency_trace_table_ms"],
        "latency_end_to_end_online_het_ms": decisions_doc["latency_end_to_end_online_het_ms"],
    }
    diag_json = os.path.join(DEC_DIR, "A5_GATE1_DIAGNOSTICS.json")
    with open(diag_json, "w", encoding="utf-8") as fh:
        json.dump(diag, fh, indent=2, default=str)
    with open(os.path.join(DEC_DIR, "A5_GATE1_REPLAY_INTEGRITY.json"), "w", encoding="utf-8") as fh:
        json.dump(replay_integrity, fh, indent=2)

    print("\n=== UNIVERSE A (ALL3_COMPLETE, STRICT_ALL3_R19) ===")
    for arm, r in u_a.items():
        print(
            f"  {arm:7s}: PASS={r['pass']:2d}/{r['known']:2d} ({r['cond_pass_pct']:5.1f}%) unk={r['unknown']} "
            f"| Rescue={r['rescued_count']}/{r['b0_fail_comparable_denom']} ({100*r['rescue_rate']:5.1f}%) "
            f"| Harm={r['harmed_count']}/{r['b0_pass_comparable_denom']} ({100*r['harm_rate']:5.1f}%) "
            f"| Net={r['net_rescued_minus_harmed']:+d}/{r['net_gain_comparable_denom']} "
            f"| util={r['mean_util_dl_pct']:.1f}% sum_a={r['mean_action_sum']:.1f}"
        )
    print("\n=== UNIVERSE B (LT3_DEMAND_COMPLETE, STRICT_DEMAND_R19) ===")
    for arm, r in u_b.items():
        print(
            f"  {arm:7s}: PASS={r['pass']}/{r['known']} ({r['cond_pass_pct']:5.1f}%) unk={r['unknown']} "
            f"| Rescue={r['rescued_count']}/{r['b0_fail_comparable_denom']} "
            f"| Harm={r['harmed_count']}/{r['b0_pass_comparable_denom']} "
            f"| inactive_waste={r['inactive_prb_waste_events']} starved={r['starved_live_events']}"
        )


if __name__ == "__main__":
    main()
