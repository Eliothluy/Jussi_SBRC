#!/usr/bin/env python3
"""Gate Contextual (G0 + G1) Read-Only Inventory, Causal State Table & Signal Analysis.

Zero new simulations, zero policy training, zero post-hoc threshold optimization.
Reads strictly existing audited Gate 1, B0-rest, D1-v9, and D2-v2 artifacts.

Outputs:
  - MODA/docs/sqr/jc_decision/GATE_CONTEXTUAL_ESTADOS.csv
  - MODA/ns-o-ran-gym/results/sqr_jc_decision/GATE_CONTEXTUAL_ANALYSIS.json
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = "/home/eliothluy/Documentos/Jussi_SBRC"
DOCS_DIR = os.path.join(REPO_ROOT, "MODA", "docs", "sqr", "jc_decision")
DEC_DIR = os.path.join(REPO_ROOT, "MODA", "ns-o-ran-gym", "results", "sqr_jc_decision")
SRC_DIR = os.path.join(REPO_ROOT, "MODA", "ns-o-ran-gym", "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from nsoran.sqr_jc.a5_joint_selector import (  # noqa: E402
    causal_active_slices,
    equal_active_split,
    prop_active_split,
)

SAMPLING_CSV = os.path.join(DEC_DIR, "A5_GATE1_SAMPLING_PLAN.csv")
SAMPLING_MANIFEST = os.path.join(DEC_DIR, "A5_GATE1_SAMPLING_MANIFEST.json")
METRICS_CSV = os.path.join(DEC_DIR, "A5_GATE1_METRICS.csv")
DECISIONS_JSON = os.path.join(DEC_DIR, "A5_GATE1_DECISIONS.json")
DIAGNOSTICS_JSON = os.path.join(DEC_DIR, "A5_GATE1_DIAGNOSTICS.json")
INTEGRITY_JSON = os.path.join(DEC_DIR, "A5_GATE1_REPLAY_INTEGRITY.json")
PARITY_JSON = os.path.join(DEC_DIR, "A5_GATE1_B0_PARITY_REPORT.json")
B0_MANIFEST = os.path.join(DOCS_DIR, "B0_FROZEN_MANIFEST.json")

OUT_CSV = os.path.join(DOCS_DIR, "GATE_CONTEXTUAL_ESTADOS.csv")
OUT_JSON = os.path.join(DEC_DIR, "GATE_CONTEXTUAL_ANALYSIS.json")


def _sha256_file(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


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


def _pair_label(b0_status: str | None, k_status: str | None) -> str:
    if b0_status not in ("PASS", "FAIL") or k_status not in ("PASS", "FAIL"):
        return "UNKNOWN"
    if b0_status == "FAIL" and k_status == "PASS":
        return "RESCUE"
    if b0_status == "PASS" and k_status == "FAIL":
        return "HARM"
    if b0_status == "PASS" and k_status == "PASS":
        return "BENIGN"
    return "NO_RESCUE"


def _eu_status(e_ok: bool | None, u_ok: bool | None) -> str:
    if e_ok is None or u_ok is None:
        return "UNKNOWN"
    return "PASS" if (e_ok and u_ok) else "FAIL"


def _auroc_binary(scores: list[float], labels: list[int]) -> float | None:
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


def main() -> None:
    # 1. Artifact inventory & hashes
    inv_paths = {
        "gate1_exploratory_report": os.path.join(DOCS_DIR, "A5_GATE1_RELATORIO_EXPLORATORIO.md"),
        "gate1_retrospective_report": os.path.join(DOCS_DIR, "A5_GATE1_DIAGNOSTICO_RETROSPECTIVO.md"),
        "gate1_metrics_csv": METRICS_CSV,
        "gate1_decisions_json": DECISIONS_JSON,
        "gate1_diagnostics_json": DIAGNOSTICS_JSON,
        "gate1_sampling_plan_csv": SAMPLING_CSV,
        "gate1_sampling_manifest_json": SAMPLING_MANIFEST,
        "gate1_replay_integrity_json": INTEGRITY_JSON,
        "gate1_b0_parity_report_json": PARITY_JSON,
        "b0_frozen_manifest_json": B0_MANIFEST,
        "b0_readiness_md": os.path.join(DOCS_DIR, "B0_PRONTIDAO_E_CONGELAMENTO.md"),
        "d2v2_report_md": os.path.join(DEC_DIR, "D2v2_report.md"),
        "d2v2_states_csv": os.path.join(DEC_DIR, "D2v2_states.csv"),
        "d2v2_results_csv": os.path.join(DEC_DIR, "D2v2_results.csv"),
        "d3_retificacao_md": os.path.join(DOCS_DIR, "D3_RETIFICACAO_E_PLANO_ABLACAO.md"),
        "d1v9_report_md": os.path.join(DEC_DIR, "D1v9_report.md"),
        "amendment4_md": os.path.join(DOCS_DIR, "PROTOCOL_FROZEN_AMENDMENT_4_DELAYED_FEEDBACK.md"),
        "a5_spec_v1_md": os.path.join(DOCS_DIR, "A5_ESPECIFICACAO_CONCEITUAL.md"),
        "a5_selector_py": os.path.join(SRC_DIR, "nsoran", "sqr_jc", "a5_joint_selector.py"),
        "sqr_direcionamento_absent": "/home/eliothluy/Downloads/Telegram Desktop/SQR_DIRECIONAMENTO_CIENTIFICO_E_REVISAO_LITERATURA_2026-11-16.md",
    }
    inventory = {}
    for k, p in inv_paths.items():
        exists = os.path.isfile(p)
        inventory[k] = {
            "path": p,
            "exists": exists,
            "size_bytes": os.path.getsize(p) if exists else None,
            "sha256": _sha256_file(p),
        }

    # Check B0 manifest internal SHA-256 consistency
    b0_m = json.loads(open(B0_MANIFEST, encoding="utf-8").read())
    b0_hash_mismatches = []
    for p_rel, expected in b0_m["git_provenance"]["code_sha256"].items():
        full_p = os.path.join(REPO_ROOT, p_rel)
        got = _sha256_file(full_p)
        if got != expected:
            b0_hash_mismatches.append({"path": p_rel, "expected": expected, "got": got})

    plan_rows = list(csv.DictReader(open(SAMPLING_CSV, encoding="utf-8")))
    metrics_rows = list(csv.DictReader(open(METRICS_CSV, encoding="utf-8")))
    decisions = json.loads(open(DECISIONS_JSON, encoding="utf-8").read())
    integrity = json.loads(open(INTEGRITY_JSON, encoding="utf-8").read())
    parity = json.loads(open(PARITY_JSON, encoding="utf-8").read())

    # Index replay integrity and metrics by (state_id, arm)
    metrics_map = {(r["state_id"], r["arm"]): r for r in metrics_rows}
    dec_by_state = {s["state_id"]: s for s in decisions["states"]}
    parity_by_tag = {r["tag"]: r for r in parity["runs"]}

    # Load per-run causal logs strictly up to decision instant
    by_dir_gws: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in plan_rows:
        by_dir_gws[row["b0_traced_dir"]].append(row)

    traced_cache: dict[str, dict[str, Any]] = {}
    for d_str, rows_dir in by_dir_gws.items():
        wanted_gws = {int(r["governed_window_index"]) for r in rows_dir}
        trace_by_gw: dict[int, dict[str, Any]] = {}
        with gzip.open(os.path.join(d_str, "causal_trace.jsonl.gz"), "rt", encoding="utf-8") as gz:
            for idx, line in enumerate(gz):
                if idx == 0 or not line.strip():
                    continue
                tw = json.loads(line)
                gw = int(tw["governed_window_index"])
                if gw in wanted_gws:
                    trace_by_gw[gw] = tw

        dec_lines = [json.loads(l) for l in open(os.path.join(d_str, "decisions.jsonl"), encoding="utf-8") if l.strip()]
        udel_lines = [json.loads(l) for l in open(os.path.join(d_str, "urllc_delayed_updates.jsonl"), encoding="utf-8") if l.strip()]
        sm_df = pd.read_csv(os.path.join(d_str, "step_metrics.csv"))
        traced_cache[d_str] = {
            "trace_by_gw": trace_by_gw,
            "dec_lines": dec_lines,
            "udel_lines": udel_lines,
            "sm_df": sm_df,
        }

    # Build strict causal single-row-per-state table (Bloco A <= decision_at; Bloco B offline outcomes)
    estado_rows: list[dict[str, Any]] = []
    temporal_violations: list[str] = []

    for plan_r in plan_rows:
        sid = plan_r["state_id"]
        d_str = plan_r["b0_traced_dir"]
        run_tag = plan_r["b0_traced_tag"]
        gw = int(plan_r["governed_window_index"])
        sw = int(plan_r["state_window_index"])
        step_idx = int(plan_r["step_index"])
        dec_at_ms = int(plan_r["sim_time_ms"])
        assert sw == gw - 1, f"{sid}: state_window_index {sw} != gov_w-1 {gw-1}"
        assert dec_at_ms == gw * 100, f"{sid}: sim_time_ms {dec_at_ms} != gov_w*100"

        rc = traced_cache[d_str]
        tw = rc["trace_by_gw"][gw]
        st = tw["state"]
        act_mask, live_mask = causal_active_slices(st)
        cap = int(tw["capacity_C"])

        # Pre-decision decision record matching sw (only causal fields BEFORE outcome!)
        dec_matches = [d for d in rc["dec_lines"] if int(d["state_window_index"]) == sw]
        assert len(dec_matches) == 1, f"{sid}: expected 1 dec row for sw={sw}, got {len(dec_matches)}"
        d_cur = dec_matches[0]

        # Strictly prior step_metrics rows (step <= step_idx - 1 -> timestamp_ms <= dec_at_ms)
        sm_df = rc["sm_df"]
        sm_prev = sm_df[sm_df["step"] == (step_idx - 1)]
        if len(sm_prev) == 3 and int(sm_prev["timestamp_ms"].max()) <= dec_at_ms:
            sm_prev_ts = int(sm_prev["timestamp_ms"].max())
            prev_sched = [int(sm_prev[sm_prev["slice_id"] == i]["prb_scheduled"].iloc[0]) for i in range(3)]
            prev_idle_est = int(sm_prev["prb_idle_est"].iloc[0])
        else:
            sm_prev_ts = -1
            prev_sched = [None, None, None]
            prev_idle_est = None
        if sm_prev_ts > dec_at_ms:
            temporal_violations.append(f"{sid}: sm_prev_ts={sm_prev_ts} > {dec_at_ms}")

        # Strictly prior decisions with outcome_window_index <= sw (observed strictly <= dec_at_ms)
        # Note: in decisions.jsonl, row k (state_window_index=k, outcome_window_index=k+1) has
        # online outcome evaluated BEFORE decide(k+2), at sim_time_ms=(k+2)*100.
        # So for decision of gw (decided at sw = gw-1), ONLY rows with outcome_window_index <= sw
        # had their immediate outcome observed at/before dec_at_ms!
        prior_decs = [
            d for d in rc["dec_lines"]
            if d.get("outcome_window_index") is not None and int(d["outcome_window_index"]) <= sw
        ]
        last10_decs = prior_decs[-10:]
        hist10_embb_viol = sum(1 for d in last10_decs if d["sla_outcome_margins"][0] is not None and float(d["sla_outcome_margins"][0]) < 0.0)
        hist10_embb_obs = sum(1 for d in last10_decs if d["sla_outcome_margins"][0] is not None)
        hist10_mmtc_viol = sum(1 for d in last10_decs if d["sla_outcome_margins"][2] is not None and float(d["sla_outcome_margins"][2]) < 0.0)
        hist10_proj_cnt = sum(1 for d in last10_decs if bool(d["projection_triggered"]))
        # Prior decisions where URLLC had immediate outcome without delayed wait OR had pending wait
        hist10_urllc_imm_viol = sum(1 for d in last10_decs if d["sla_outcome_margins"][1] is not None and float(d["sla_outcome_margins"][1]) < 0.0)

        # Strictly prior URLLC delayed updates closed at observed_at_window <= sw
        prior_udels = [
            u for u in rc["udel_lines"]
            if int(u["observed_at_window"]) <= sw
        ]
        closed_udels = [u for u in prior_udels if u.get("updated") is True and u.get("p95_ms") is not None]
        recent_closed_udels = [u for u in closed_udels if int(u["w"]) >= sw - 10]
        last_udel = closed_udels[-1] if closed_udels else None
        if last_udel is not None and int(last_udel["observed_at_window"]) > sw:
            temporal_violations.append(f"{sid}: udel observed_at_window > sw")

        # Causal candidates (100% pre-decision)
        b0_exec = tuple(int(v) for v in tw["a_exec_per_slice"])
        b0_raw = tuple(int(v) for v in tw["a_raw_per_slice"])
        b0_prop = tuple(int(v) for v in tw["a_prop_per_slice"])
        eq_blind = (18, 17, 18)
        eq_act = equal_active_split(cap, act_mask)
        prop_act = prop_active_split(st, cap, act_mask)

        # Pre-decision model f & sigma on b0_exec, eq_blind, prop_act
        psm = tw["per_slice_model"]
        b0_f_exec = [float(psm[i]["f_candidates"][b0_exec[i]]) for i in range(3)]
        b0_sig_exec = [float(psm[i]["sigma_candidates"][b0_exec[i]]) for i in range(3)]
        eq_f = [float(psm[i]["f_candidates"][eq_blind[i]]) for i in range(3)]
        eq_sig = [float(psm[i]["sigma_candidates"][eq_blind[i]]) for i in range(3)]
        prop_f = [float(psm[i]["f_candidates"][prop_act[i]]) for i in range(3)]
        prop_sig = [float(psm[i]["sigma_candidates"][prop_act[i]]) for i in range(3)]

        # Offered & demand features
        tx_pkts = st.get("extras", {}).get("kpm_tx_packets", [0, 0, 0])
        modes = tw["modes_per_slice"]

        # Bloco B (offline outcomes ONLY)
        rb0 = metrics_map[(sid, "b0")]
        req = metrics_map[(sid, "eq")]
        rpr = metrics_map[(sid, "prop")]

        def _arm_vals(r: dict[str, Any]) -> dict[str, Any]:
            e_ok = _parse_bool_opt(r["embb_ok"])
            u_ok = _parse_bool_opt(r["urllc_ok"])
            m_ok = _parse_bool_opt(r["mmtc_ok"])
            tx_u = int(r["urllc_tx_pkts"])
            rx_u = int(r["urllc_rx_pkts"])
            return {
                "all3": r["strict_all3_r19"],
                "dem": r["strict_demand_r19"],
                "eu": _eu_status(e_ok, u_ok),
                "e_ok": e_ok,
                "u_ok": u_ok,
                "m_ok": m_ok,
                "e_thr": _parse_float_opt(r["embb_thr_mbps"]),
                "u_p95": _parse_float_opt(r["urllc_p95_ms"]),
                "u_miss": _parse_float_opt(r["urllc_miss_pct"]),
                "u_tx": tx_u,
                "u_rx": rx_u,
                "u_unrx": tx_u - rx_u,
                "u_pool": r["urllc_pool_tag"],
                "m_pdr": _parse_float_opt(r["mmtc_pdr_rolling"]),
                "util": _parse_float_opt(r["util_dl_pct"]),
                "inact_prbs": int(r["inactive_prbs_allocated"]),
                "replay_tag": r["replay_tag"],
            }

        vb0 = _arm_vals(rb0)
        veq = _arm_vals(req)
        vpr = _arm_vals(rpr)

        lbl_eq_strict = _pair_label(vb0["all3"], veq["all3"])
        lbl_prop_strict = _pair_label(vb0["all3"], vpr["all3"])
        lbl_eq_dem = _pair_label(vb0["dem"], veq["dem"])
        lbl_prop_dem = _pair_label(vb0["dem"], vpr["dem"])
        lbl_eq_eu = _pair_label(vb0["eu"], veq["eu"])
        lbl_prop_eu = _pair_label(vb0["eu"], vpr["eu"])

        par_run = parity_by_tag[run_tag]
        aa_b0_valid = _parse_bool_opt(rb0["aa_prefix_valid"])
        aa_eq_valid = _parse_bool_opt(req["aa_prefix_valid"])
        aa_pr_valid = _parse_bool_opt(rpr["aa_prefix_valid"])

        row_out = {
            # Keys & integrity
            "state_id": sid,
            "run_tag": run_tag,
            "lam": int(plan_r["lam"]),
            "seed": int(plan_r["seed"]),
            "step_index": step_idx,
            "state_window_index": sw,
            "governed_window_index": gw,
            "decision_at_ms": dec_at_ms,
            "universe": plan_r["universe"],
            "stratum_id": plan_r["stratum_id"],
            "aa_parity_b0_eq_prop": bool(aa_b0_valid and aa_eq_valid and aa_pr_valid),
            "clean_cut_step": int(par_run["clean_cut_step"]),
            # Bloco A: strictly pre-decision features (observed_at <= dec_at_ms)
            "n_e": int(st["n_e"]),
            "n_u": int(st["n_u"]),
            "n_m": int(st["n_m"]),
            "n_ues_total": int(st["n_e"]) + int(st["n_u"]) + int(st["n_m"]),
            "queue_e": float(st["queue_e"]),
            "queue_u": float(st["queue_u"]),
            "queue_m": float(st["queue_m"]),
            "offered_e_mbps": round(float(st["offered_e"]), 4),
            "offered_u_mbps": round(float(st["offered_u"]), 4),
            "offered_m_mbps": round(float(st["offered_m"]), 4),
            "snr_e_db": round(float(st["snr_e"]), 3),
            "snr_u_db": round(float(st["snr_u"]), 3),
            "snr_m_db": round(float(st["snr_m"]), 3),
            "n_active_causal": int(sum(act_mask)),
            "active_mask": "".join("1" if x else "0" for x in act_mask),
            "live_floor_mask": "".join("1" if x else "0" for x in live_mask),
            "tx_pkts_e_wprev": int(tx_pkts[0]),
            "tx_pkts_u_wprev": int(tx_pkts[1]),
            "tx_pkts_m_wprev": int(tx_pkts[2]),
            "prev_a_e": int(st["prev_a_e"]),
            "prev_a_u": int(st["prev_a_u"]),
            "prev_a_m": int(st["prev_a_m"]),
            "prev_prb_sched_e": prev_sched[0],
            "prev_prb_sched_u": prev_sched[1],
            "prev_prb_sched_m": prev_sched[2],
            "prev_prb_idle_est": prev_idle_est,
            "prev_kpi_e_mbps": st["kpi_e_raw"],
            "prev_kpi_u_p95_imm": st["kpi_u_raw"],
            "prev_kpi_m_pdr": st["kpi_m_raw"],
            "urllc_pool_open_at_dec": bool(st.get("extras", {}).get("urllc_pool_open", False)),
            "mmtc_rolling_active_at_dec": bool(st.get("extras", {}).get("mmtc_rolling_active", False)),
            "viol_e_prev": int(st["viol_e"]),
            "viol_u_prev_imm": int(st["viol_u"]),
            "viol_m_prev": int(st["viol_m"]),
            "hist10_embb_viol_cnt": int(hist10_embb_viol),
            "hist10_embb_obs_cnt": int(hist10_embb_obs),
            "hist10_mmtc_viol_cnt": int(hist10_mmtc_viol),
            "hist10_proj_cnt": int(hist10_proj_cnt),
            "hist10_urllc_imm_viol_cnt": int(hist10_urllc_imm_viol),
            "hist10_urllc_closed_delayed_cnt": len(recent_closed_udels),
            "hist10_urllc_closed_delayed_viol_cnt": sum(1 for u in recent_closed_udels if float(u["p95_ms"]) > 10.0),
            "last_closed_urllc_win": int(last_udel["w"]) if last_udel else None,
            "last_closed_urllc_obs_win": int(last_udel["observed_at_window"]) if last_udel else None,
            "last_closed_urllc_lag_from_dec": (sw - int(last_udel["w"])) if last_udel else None,
            "last_closed_urllc_p95_ms": float(last_udel["p95_ms"]) if last_udel else None,
            "b0_a_raw_e": b0_raw[0],
            "b0_a_raw_u": b0_raw[1],
            "b0_a_raw_m": b0_raw[2],
            "b0_a_prop_e": b0_prop[0],
            "b0_a_prop_u": b0_prop[1],
            "b0_a_prop_m": b0_prop[2],
            "b0_a_exec_e": b0_exec[0],
            "b0_a_exec_u": b0_exec[1],
            "b0_a_exec_m": b0_exec[2],
            "b0_sum_exec": sum(b0_exec),
            "b0_idle_prbs": cap - sum(b0_exec),
            "b0_proj_triggered": bool(d_cur["projection_triggered"]),
            "b0_proj_delta_l1": int(d_cur["projection_delta_l1"]),
            "b0_sum_proposals_over_C": int(d_cur["sum_proposals_over_C"]),
            "b0_prop_excess_gt53": bool(d_cur["sum_proposals_over_C"] > 0),
            "b0_mode_e": modes[0],
            "b0_mode_u": modes[1],
            "b0_mode_m": modes[2],
            "b0_tau_e": float(psm[0]["tau"]),
            "b0_tau_u": float(psm[1]["tau"]),
            "b0_tau_m": float(psm[2]["tau"]),
            "b0_f_exec_e": round(b0_f_exec[0], 5),
            "b0_f_exec_u": round(b0_f_exec[1], 5),
            "b0_f_exec_m": round(b0_f_exec[2], 5),
            "b0_sig_exec_e": round(b0_sig_exec[0], 5),
            "b0_sig_exec_u": round(b0_sig_exec[1], 5),
            "b0_sig_exec_m": round(b0_sig_exec[2], 5),
            "eq_e": eq_blind[0],
            "eq_u": eq_blind[1],
            "eq_m": eq_blind[2],
            "eq_act_e": eq_act[0],
            "eq_act_u": eq_act[1],
            "eq_act_m": eq_act[2],
            "prop_e": prop_act[0],
            "prop_u": prop_act[1],
            "prop_m": prop_act[2],
            "delta_eq_minus_b0_e": eq_blind[0] - b0_exec[0],
            "delta_eq_minus_b0_u": eq_blind[1] - b0_exec[1],
            "delta_eq_minus_b0_m": eq_blind[2] - b0_exec[2],
            "l1_b0_eq": sum(abs(eq_blind[i] - b0_exec[i]) for i in range(3)),
            "delta_prop_minus_b0_e": prop_act[0] - b0_exec[0],
            "delta_prop_minus_b0_u": prop_act[1] - b0_exec[1],
            "delta_prop_minus_b0_m": prop_act[2] - b0_exec[2],
            "l1_b0_prop": sum(abs(prop_act[i] - b0_exec[i]) for i in range(3)),
            "b0_f_eq_e": round(eq_f[0], 5),
            "b0_f_eq_u": round(eq_f[1], 5),
            "b0_f_eq_m": round(eq_f[2], 5),
            "b0_f_prop_e": round(prop_f[0], 5),
            "b0_f_prop_u": round(prop_f[1], 5),
            "b0_f_prop_m": round(prop_f[2], 5),
            "delta_f_eq_minus_b0_e": round(eq_f[0] - b0_f_exec[0], 5),
            "delta_f_eq_minus_b0_u": round(eq_f[1] - b0_f_exec[1], 5),
            "delta_f_prop_minus_b0_e": round(prop_f[0] - b0_f_exec[0], 5),
            "delta_f_prop_minus_b0_u": round(prop_f[1] - b0_f_exec[1], 5),
            # Bloco B: STRICTLY OFFLINE REPLAY OUTCOMES (prohibited as features!)
            "off_b0_strict_all3": vb0["all3"],
            "off_b0_strict_demand": vb0["dem"],
            "off_b0_eu": vb0["eu"],
            "off_b0_embb_thr": vb0["e_thr"],
            "off_b0_urllc_p95": vb0["u_p95"],
            "off_b0_urllc_miss": vb0["u_miss"],
            "off_b0_urllc_tx": vb0["u_tx"],
            "off_b0_urllc_rx": vb0["u_rx"],
            "off_b0_urllc_unrx": vb0["u_unrx"],
            "off_b0_urllc_pool": vb0["u_pool"],
            "off_b0_mmtc_pdr": vb0["m_pdr"],
            "off_eq_strict_all3": veq["all3"],
            "off_eq_strict_demand": veq["dem"],
            "off_eq_eu": veq["eu"],
            "off_eq_embb_thr": veq["e_thr"],
            "off_eq_urllc_p95": veq["u_p95"],
            "off_eq_urllc_miss": veq["u_miss"],
            "off_eq_urllc_tx": veq["u_tx"],
            "off_eq_urllc_rx": veq["u_rx"],
            "off_eq_urllc_unrx": veq["u_unrx"],
            "off_eq_urllc_pool": veq["u_pool"],
            "off_eq_mmtc_pdr": veq["m_pdr"],
            "off_eq_inact_prbs": veq["inact_prbs"],
            "off_prop_strict_all3": vpr["all3"],
            "off_prop_strict_demand": vpr["dem"],
            "off_prop_eu": vpr["eu"],
            "off_prop_embb_thr": vpr["e_thr"],
            "off_prop_urllc_p95": vpr["u_p95"],
            "off_prop_urllc_miss": vpr["u_miss"],
            "off_prop_urllc_tx": vpr["u_tx"],
            "off_prop_urllc_rx": vpr["u_rx"],
            "off_prop_urllc_unrx": vpr["u_unrx"],
            "off_prop_urllc_pool": vpr["u_pool"],
            "off_prop_mmtc_pdr": vpr["m_pdr"],
            "off_prop_inact_prbs": vpr["inact_prbs"],
            "off_label_eq_strict_all3": lbl_eq_strict,
            "off_label_prop_strict_all3": lbl_prop_strict,
            "off_label_eq_strict_demand": lbl_eq_dem,
            "off_label_prop_strict_demand": lbl_prop_dem,
            "off_label_eq_eu": lbl_eq_eu,
            "off_label_prop_eu": lbl_prop_eu,
        }
        estado_rows.append(row_out)

    # Write GATE_CONTEXTUAL_ESTADOS.csv
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(estado_rows[0].keys()))
        w.writeheader()
        w.writerows(estado_rows)

    # Analyze G1 discriminative power across ALL candidate pre-decision signals (no threshold tuning!)
    df = pd.DataFrame(estado_rows)
    df_A = df[df["universe"] == "ALL3_COMPLETE"].copy()

    def _label_table(sub: pd.DataFrame, lbl_col: str) -> dict[str, Any]:
        c = Counter(sub[lbl_col].tolist())
        eval_n = len(sub) - c.get("UNKNOWN", 0)
        by_seed = {}
        for sd, g in sub.groupby("seed"):
            cs = Counter(g[lbl_col].tolist())
            by_seed[str(sd)] = {k: int(cs.get(k, 0)) for k in ("RESCUE", "HARM", "BENIGN", "NO_RESCUE", "UNKNOWN")}
        by_lam = {}
        for lm, g in sub.groupby("lam"):
            cl = Counter(g[lbl_col].tolist())
            by_lam[str(lm)] = {k: int(cl.get(k, 0)) for k in ("RESCUE", "HARM", "BENIGN", "NO_RESCUE", "UNKNOWN")}
        return {
            "n_total": len(sub),
            "n_evaluable": eval_n,
            "counts": {k: int(c.get(k, 0)) for k in ("RESCUE", "HARM", "BENIGN", "NO_RESCUE", "UNKNOWN")},
            "by_seed": by_seed,
            "by_lam": by_lam,
        }

    num_candidate_features = [
        "n_e", "n_u", "n_m", "queue_e", "queue_u", "queue_m",
        "offered_e_mbps", "offered_u_mbps", "n_active_causal",
        "b0_a_exec_e", "b0_a_exec_u", "b0_a_exec_m", "b0_sum_exec", "b0_idle_prbs",
        "b0_proj_delta_l1", "b0_sum_proposals_over_C",
        "prev_kpi_e_mbps", "last_closed_urllc_p95_ms", "last_closed_urllc_lag_from_dec",
        "hist10_embb_viol_cnt", "hist10_urllc_closed_delayed_viol_cnt", "hist10_proj_cnt",
        "b0_f_exec_e", "b0_f_exec_u", "b0_sig_exec_e", "b0_sig_exec_u",
        "delta_eq_minus_b0_e", "delta_eq_minus_b0_u", "l1_b0_eq",
        "delta_prop_minus_b0_e", "delta_prop_minus_b0_u", "l1_b0_prop",
        "delta_f_eq_minus_b0_e", "delta_f_eq_minus_b0_u",
        "delta_f_prop_minus_b0_e", "delta_f_prop_minus_b0_u",
    ]

    signal_scan: dict[str, Any] = {}
    for opt, lbl_col in (("equal", "off_label_eq_strict_all3"), ("prop", "off_label_prop_strict_all3")):
        opt_scan: dict[str, Any] = {}
        rh = df_A[df_A[lbl_col].isin(["RESCUE", "HARM"])].copy()
        rb_all = df_A[df_A[lbl_col].isin(["RESCUE", "HARM", "BENIGN", "NO_RESCUE"])].copy()
        y_rh = [1 if v == "RESCUE" else 0 for v in rh[lbl_col]]
        y_harm_vs_safe = [1 if v == "HARM" else 0 for v in df_A[df_A[lbl_col].isin(["HARM", "BENIGN"])][lbl_col]]
        sub_harm_safe = df_A[df_A[lbl_col].isin(["HARM", "BENIGN"])]

        for feat in num_candidate_features:
            # Summary by group
            grp_stats = {}
            for lbl in ("RESCUE", "HARM", "BENIGN", "NO_RESCUE"):
                vals = pd.to_numeric(rb_all[rb_all[lbl_col] == lbl][feat], errors="coerce").dropna().tolist()
                grp_stats[lbl] = {
                    "n": len(vals),
                    "median": round(float(np.median(vals)), 4) if vals else None,
                    "min": round(float(np.min(vals)), 4) if vals else None,
                    "max": round(float(np.max(vals)), 4) if vals else None,
                }
            # AUROC RESCUE vs HARM
            rh_pairs = [(float(v), y) for v, y in zip(pd.to_numeric(rh[feat], errors="coerce"), y_rh) if pd.notna(v)]
            au_rh = _auroc_binary([p[0] for p in rh_pairs], [p[1] for p in rh_pairs]) if rh_pairs else None
            # AUROC HARM vs BENIGN (given B0=PASS, can we predict harm?)
            hs_pairs = [(float(v), y) for v, y in zip(pd.to_numeric(sub_harm_safe[feat], errors="coerce"), y_harm_vs_safe) if pd.notna(v)]
            au_hs = _auroc_binary([p[0] for p in hs_pairs], [p[1] for p in hs_pairs]) if hs_pairs else None
            # Symmetrized separation |AUC - 0.5| + 0.5
            opt_scan[feat] = {
                "auroc_rescue_vs_harm": round(au_rh, 4) if au_rh is not None else None,
                "auroc_harm_vs_benign": round(au_hs, 4) if au_hs is not None else None,
                "groups": grp_stats,
            }
        signal_scan[opt] = opt_scan

    # Per-state concise summary for ALL 25 states under G1
    per_state_g1 = []
    for r in estado_rows:
        per_state_g1.append({
            "state_id": r["state_id"],
            "lam": r["lam"],
            "seed": r["seed"],
            "universe": r["universe"],
            "n_active_causal": r["n_active_causal"],
            "active_mask": r["active_mask"],
            "n_e_u_m": [r["n_e"], r["n_u"], r["n_m"]],
            "q_e_u": [round(r["queue_e"], 0), round(r["queue_u"], 0)],
            "b0_exec": [r["b0_a_exec_e"], r["b0_a_exec_u"], r["b0_a_exec_m"]],
            "prop_act": [r["prop_e"], r["prop_u"], r["prop_m"]],
            "b0_proj": r["b0_proj_triggered"],
            "b0_modes": [r["b0_mode_e"], r["b0_mode_u"], r["b0_mode_m"]],
            "urllc_open_dec": r["urllc_pool_open_at_dec"],
            "last_urllc_p95": r["last_closed_urllc_p95_ms"],
            "last_urllc_lag_w": r["last_closed_urllc_lag_from_dec"],
            "b0_all3": r["off_b0_strict_all3"],
            "eq_all3": r["off_eq_strict_all3"],
            "prop_all3": r["off_prop_strict_all3"],
            "lbl_eq_all3": r["off_label_eq_strict_all3"],
            "lbl_prop_all3": r["off_label_prop_strict_all3"],
            "lbl_eq_eu": r["off_label_eq_eu"],
            "lbl_prop_eu": r["off_label_prop_eu"],
            "kpis_b0_eq_prop_e": [r["off_b0_embb_thr"], r["off_eq_embb_thr"], r["off_prop_embb_thr"]],
            "kpis_b0_eq_prop_u": [r["off_b0_urllc_p95"], r["off_eq_urllc_p95"], r["off_prop_urllc_p95"]],
        })

    out = {
        "inventory": inventory,
        "b0_manifest_hash_mismatches": b0_hash_mismatches,
        "temporal_violations": temporal_violations,
        "g0_summary": {
            "n_total_states": len(df),
            "universe_counts": dict(Counter(df["universe"])),
            "by_lam_seed": {
                f"lam{lam}_s{seed}": dict(Counter(g["universe"]))
                for (lam, seed), g in df.groupby(["lam", "seed"])
            },
            "aa_parity_valid_count": int(df["aa_parity_b0_eq_prop"].sum()),
            "b0_eq_prop_present_count": len(df),
            "valid_replays_total": sum(1 for v in integrity.values() if isinstance(v, dict) and v.get("valid")),
        },
        "g1_labels": {
            "ALL3_COMPLETE_equal_STRICT_ALL3": _label_table(df_A, "off_label_eq_strict_all3"),
            "ALL3_COMPLETE_prop_STRICT_ALL3": _label_table(df_A, "off_label_prop_strict_all3"),
            "ALL25_equal_STRICT_ALL3": _label_table(df, "off_label_eq_strict_all3"),
            "ALL25_prop_STRICT_ALL3": _label_table(df, "off_label_prop_strict_all3"),
            "ALL25_equal_EU": _label_table(df, "off_label_eq_eu"),
            "ALL25_prop_EU": _label_table(df, "off_label_prop_eu"),
            "LT3_DEMAND_equal_STRICT_DEMAND": _label_table(df[df["universe"] == "LT3_DEMAND_COMPLETE"], "off_label_eq_strict_demand"),
            "LT3_DEMAND_prop_STRICT_DEMAND": _label_table(df[df["universe"] == "LT3_DEMAND_COMPLETE"], "off_label_prop_strict_demand"),
        },
        "g1_signal_scan_univ_A": signal_scan,
        "per_state_g1": per_state_g1,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("B0 hash mismatches:", len(b0_hash_mismatches))
    print("Temporal violations:", len(temporal_violations))
    print("G0 summary:", json.dumps(out["g0_summary"], indent=2))
    print("\nG1 labels summary:")
    for k, v in out["g1_labels"].items():
        print(f"  {k}: n={v['n_total']} eval={v['n_evaluable']} counts={v['counts']} by_seed={v['by_seed']} by_lam={v['by_lam']}")

    print("\nTop univariate AUROCs in ALL3_COMPLETE (RESCUE vs HARM and HARM vs BENIGN):")
    for opt in ("equal", "prop"):
        print(f"\n  [{opt}]")
        ranked_rh = sorted(
            signal_scan[opt].items(),
            key=lambda kv: max(kv[1]["auroc_rescue_vs_harm"] or 0.5, 1.0 - (kv[1]["auroc_rescue_vs_harm"] or 0.5)),
            reverse=True,
        )[:12]
        for f, d in ranked_rh:
            print(f"    {f:36s}: AUC(R vs H)={d['auroc_rescue_vs_harm']} AUC(H vs B)={d['auroc_harm_vs_benign']} | med R={d['groups']['RESCUE']['median']} H={d['groups']['HARM']['median']} B={d['groups']['BENIGN']['median']} NR={d['groups']['NO_RESCUE']['median']}")


if __name__ == "__main__":
    main()
