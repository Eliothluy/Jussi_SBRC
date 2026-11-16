#!/usr/bin/env python3
"""Gate 1 Step 1: Verify bit-exact parity of traced vs untraced B0-rest runs
and inspect valid candidate strata across the 5 traced dev runs.

Outputs:
  - b0_traced_gate1/b0_traced_parity_report.json
  - b0_traced_gate1/candidate_pool_summary.json
"""
from __future__ import annotations

import glob
import gzip
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
BASE_DEC_DIR = os.path.join(GYM_DIR, "results", "sqr_jc_decision")
TRACED_DIR = os.path.join(BASE_DEC_DIR, "b0_traced_gate1")

UNTRACED_MAP = {
    (2, 2102): os.path.join(BASE_DEC_DIR, "a1_a2a_ascaler_runs_v9_va0_guard_eum", "het_lam2_s2102"),
    (5, 2102): os.path.join(BASE_DEC_DIR, "a1_a2a_ascaler_runs_v9_va0_guard_eum", "het_lam5_s2102"),
    (6, 2102): os.path.join(BASE_DEC_DIR, "a1_a2a_ascaler_runs_v9_va0_guard_eum", "het_lam6_s2102"),
    (5, 2103): os.path.join(BASE_DEC_DIR, "b0_robustness_s2103_v9_va0_guard_eum", "het_lam5_s2103"),
    (6, 2103): os.path.join(BASE_DEC_DIR, "b0_robustness_s2103_v9_va0_guard_eum", "het_lam6_s2103"),
}

EXPECTED_B0_CONFIG_SHA = "a2b40861ace7ba87ce2d321ccc62f42f22e6b3798c4432171875ae55d7040d7c"

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from nsoran.sqr.checkpoint import load_checkpoint  # noqa: E402
from nsoran.sqr.oracle.dataset import _urllc_packet_windows  # noqa: E402
from nsoran.sqr_jc.a5_joint_selector import causal_active_slices  # noqa: E402
from nsoran.sqr_jc.freeze_contract import (  # noqa: E402
    CAUSAL_TRACE_VERSION,
    checkpoint_payload_sha256,
    sha256_file,
    verify_causal_replay_no_future_checkpoint,
)


def _find_single(run_dir: str, fname: str) -> str:
    hits = sorted(glob.glob(os.path.join(run_dir, "**", fname), recursive=True))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected 1 {fname} under {run_dir}, got {len(hits)}")
    return hits[0]


def _reconstruct_seq_actions(prb_df: pd.DataFrame) -> tuple[list[tuple[int, int, int]], dict[int, tuple[int, int, int]]]:
    """Exact mirror of examples/sqr_run.py replay_base=='seq' action reconstruction."""
    seq: list[tuple[int, int, int]] = []
    by_win: dict[int, tuple[int, int, int]] = {}
    for row in prb_df.itertuples(index=False):
        w = int(row.timestamp_ms) // 100 - 1
        a = (
            int(row.prb_executed_rbg_0),
            int(row.prb_executed_rbg_1),
            int(row.prb_executed_rbg_2),
        )
        by_win[w] = a
        if w <= 0:
            continue
        while len(seq) < w - 1:
            seq.append(seq[-1] if seq else a)
        if len(seq) == w - 1:
            seq.append(a)
        else:
            seq[w - 1] = a
    return seq, by_win


def main() -> None:
    parity_reports: list[dict[str, Any]] = []
    pool_summary: list[dict[str, Any]] = []
    all_ok = True

    for (lam, seed), orig_dir in sorted(UNTRACED_MAP.items()):
        tag = f"het_lam{lam}_s{seed}"
        traced_dir = os.path.join(TRACED_DIR, tag)

        # 1. File-level bit-exact parity + IPC 50ms timeout boundary analysis
        file_checks: dict[str, dict[str, Any]] = {}
        for rel in ("decisions.jsonl", "step_metrics.csv", "urllc_delayed_updates.jsonl"):
            p_orig = os.path.join(orig_dir, rel)
            p_tr = os.path.join(traced_dir, rel)
            h_orig = sha256_file(p_orig)
            h_tr = sha256_file(p_tr)
            file_checks[rel] = {"sha256_orig": h_orig, "sha256_traced": h_tr, "exact": (h_orig == h_tr)}

        for tel in ("sqr_kpm_history.csv", "sqr_prb_windows.csv", "sqr_packet_delays.csv", "sqr_summary.csv"):
            p_orig = _find_single(orig_dir, tel)
            p_tr = _find_single(traced_dir, tel)
            h_orig = sha256_file(p_orig)
            h_tr = sha256_file(p_tr)
            file_checks[tel] = {"sha256_orig": h_orig, "sha256_traced": h_tr, "exact": (h_orig == h_tr)}

        # Compute exact pre-timeout parity up to first IPC 50ms wall-clock shift
        with open(os.path.join(orig_dir, "decisions.jsonl"), encoding="utf-8") as fh:
            orig_decs = [json.loads(line) for line in fh if line.strip()]
        with open(os.path.join(traced_dir, "decisions.jsonl"), encoding="utf-8") as fh:
            decs = [json.loads(line) for line in fh if line.strip()]
        prb_orig_df = pd.read_csv(_find_single(orig_dir, "sqr_prb_windows.csv"))
        prb_df = pd.read_csv(_find_single(traced_dir, "sqr_prb_windows.csv"))

        def _first_lag(prb_tab: pd.DataFrame, dec_rows: list[dict[str, Any]]) -> int | None:
            pmap = {
                int(r.timestamp_ms) // 100 - 1: (
                    int(r.prb_executed_rbg_0),
                    int(r.prb_executed_rbg_1),
                    int(r.prb_executed_rbg_2),
                )
                for r in prb_tab.itertuples()
            }
            for d in dec_rows:
                if d.get("policy") == "SQR_HET" and not d.get("missing_reason"):
                    if pmap.get(int(d["outcome_window_index"])) != tuple(int(v) for v in d["action"]):
                        return int(d["decision_index"])
            return None

        orig_first_lag = _first_lag(prb_orig_df, orig_decs)
        traced_first_lag = _first_lag(prb_df, decs)
        cut_step = min(
            [x for x in (orig_first_lag, traced_first_lag) if x is not None],
            default=999999,
        )

        pre_dec_eq = all(
            a == b
            for a, b in zip(orig_decs, decs)
            if int(a["decision_index"]) < cut_step and int(b["decision_index"]) < cut_step
        ) and (
            [int(a["decision_index"]) for a in orig_decs if int(a["decision_index"]) < cut_step]
            == [int(b["decision_index"]) for b in decs if int(b["decision_index"]) < cut_step]
        )
        kpm_orig_df = pd.read_csv(_find_single(orig_dir, "sqr_kpm_history.csv"))
        kpm_df = pd.read_csv(_find_single(traced_dir, "sqr_kpm_history.csv"))
        pre_kpm_eq = (
            kpm_orig_df[kpm_orig_df["timestamp"] <= cut_step * 100]
            .reset_index(drop=True)
            .equals(kpm_df[kpm_df["timestamp"] <= cut_step * 100].reset_index(drop=True))
        )
        pre_prb_eq = (
            prb_orig_df[prb_orig_df["timestamp_ms"] <= cut_step * 100]
            .reset_index(drop=True)
            .equals(prb_df[prb_df["timestamp_ms"] <= cut_step * 100].reset_index(drop=True))
        )
        del_orig_df = pd.read_csv(_find_single(orig_dir, "sqr_packet_delays.csv"))
        del_df = pd.read_csv(_find_single(traced_dir, "sqr_packet_delays.csv"))
        pre_del_eq = (
            del_orig_df[(del_orig_df["txSec"] + del_orig_df["delaySec"]) * 1000.0 < cut_step * 100.0]
            .sort_values(["txSec", "delaySec"])
            .reset_index(drop=True)
            .equals(
                del_df[(del_df["txSec"] + del_df["delaySec"]) * 1000.0 < cut_step * 100.0]
                .sort_values(["txSec", "delaySec"])
                .reset_index(drop=True)
            )
        )
        st_orig_df = pd.read_csv(os.path.join(orig_dir, "step_metrics.csv"))
        st_tr_df = pd.read_csv(os.path.join(traced_dir, "step_metrics.csv"))
        st_cols = [c for c in st_orig_df.columns if c not in ("decision_time_ms", "prb_scheduled", "prb_idle_est")]
        pre_step_eq = (
            st_orig_df[st_orig_df["step"] < cut_step][st_cols]
            .reset_index(drop=True)
            .equals(st_tr_df[st_tr_df["step"] < cut_step][st_cols].reset_index(drop=True))
        )
        pre_timeout_parity_ok = bool(pre_dec_eq and pre_kpm_eq and pre_prb_eq and pre_del_eq and pre_step_eq)

        # 2. Checkpoint payload_sha256 parity + horizon metadata
        ckpt_orig = load_checkpoint(os.path.join(orig_dir, "sqr_checkpoint.pkl"))
        ckpt_tr = load_checkpoint(os.path.join(traced_dir, "sqr_checkpoint.pkl"))
        pay_orig = checkpoint_payload_sha256(ckpt_orig)
        pay_tr = checkpoint_payload_sha256(ckpt_tr)
        meta_tr = ckpt_tr.get("meta") or {}
        ckpt_match = (pay_orig == pay_tr)
        ckpt_self_valid = (meta_tr.get("payload_sha256") == pay_tr)
        cfg_match = meta_tr.get("config_sha256") == EXPECTED_B0_CONFIG_SHA
        horizon = meta_tr.get("checkpoint_horizon") or {}
        if not (pre_timeout_parity_ok and ckpt_self_valid and cfg_match and "max_window_index" in horizon):
            all_ok = False

        # 3. Causal trace completeness, version/horizon invariants, and zero-future-checkpoint rule
        trace_path = os.path.join(traced_dir, "causal_trace.jsonl.gz")
        with gzip.open(trace_path, "rt", encoding="utf-8") as fh:
            trace_recs = [json.loads(line) for line in fh if line.strip()]
        header = trace_recs[0]
        windows = trace_recs[1:]

        het_decs = [d for d in decs if d["policy"] == "SQR_HET"]

        trace_len_match = len(windows) == len(het_decs)
        header_ok = (
            header.get("record_type") == "causal_trace_header"
            and header.get("causal_trace_version") == CAUSAL_TRACE_VERSION
            and (header.get("metadata") or {}).get("config_sha256") == EXPECTED_B0_CONFIG_SHA
        )

        # Verify causal invariants + future checkpoint block on every traced decision
        causal_invariants_ok = trace_len_match and header_ok
        future_ckpt_blocked = 0
        future_ckpt_boundary_allowed = 0
        max_f_err = 0.0
        prev_sim = -1
        horizon_w = int(horizon.get("max_window_index", -1))
        for tw, dec in zip(windows, het_decs):
            w_idx = int(tw["window_index"])
            gov_idx = int(tw["governed_window_index"])
            hor_idx = int(tw["causal_horizon_max_window_index"])
            hor_ms = int(tw["causal_horizon_max_sim_time_ms"])
            sim_ms = int(tw["sim_time_ms"])
            if not (
                tw.get("causal_trace_version") == CAUSAL_TRACE_VERSION
                and gov_idx == w_idx + 1
                and hor_idx == w_idx
                and hor_idx < gov_idx
                and hor_ms == sim_ms
                and sim_ms > prev_sim
                and tw["step_index"] == dec["decision_index"]
                and tw["a_raw_per_slice"] == dec["a_raw_per_slice"]
                and tw["a_prop_per_slice"] == dec["a_after_uncertainty_margin"]
                and tw["a_exec_per_slice"] == dec["action"]
            ):
                causal_invariants_ok = False
            prev_sim = sim_ms
            for sid in range(3):
                psm = tw["per_slice_model"][sid]
                max_f_err = max(
                    max_f_err,
                    abs(float(psm["f_candidates"][0]) - float(dec["f_at_a0"][sid])),
                    abs(float(psm["f_candidates"][37]) - float(dec["f_at_a37"][sid])),
                    abs(float(psm["f_candidates"][53]) - float(dec["f_at_amax"][sid])),
                )
            if gov_idx <= horizon_w:
                try:
                    verify_causal_replay_no_future_checkpoint(
                        ckpt_tr, target_window_index=gov_idx, target_sim_time_ms=(gov_idx + 1) * 100
                    )
                    causal_invariants_ok = False
                except ValueError:
                    future_ckpt_blocked += 1
            else:
                try:
                    verify_causal_replay_no_future_checkpoint(
                        ckpt_tr, target_window_index=gov_idx, target_sim_time_ms=(gov_idx + 1) * 100
                    )
                    future_ckpt_boundary_allowed += 1
                except ValueError:
                    causal_invariants_ok = False

        if max_f_err > 1e-11:
            causal_invariants_ok = False
        if not causal_invariants_ok:
            all_ok = False

        # 4. Candidate strata summary (for Gate 1 sampling design)
        prb_df = pd.read_csv(_find_single(traced_dir, "sqr_prb_windows.csv"))
        kpm_df = pd.read_csv(_find_single(traced_dir, "sqr_kpm_history.csv"))
        wins_df = _urllc_packet_windows(_find_single(traced_dir, "sqr_packet_delays.csv"))
        wins_by_w = wins_df.set_index("win").to_dict("index") if len(wins_df) else {}
        ku = kpm_df[kpm_df["sliceId"] == 1].copy()
        ku["win"] = ku["timestamp"] // 100 - 1
        tx_u_by_w = ku.groupby("win")["dTxPackets"].sum().to_dict()
        seq_actions, prb_exec_by_win = _reconstruct_seq_actions(prb_df)

        ul_path = os.path.join(traced_dir, "urllc_delayed_updates.jsonl")
        ul_by_w: dict[int, dict[str, Any]] = {}
        with open(ul_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    ul_by_w[int(r["w"])] = r

        strata_counts: Counter[tuple[Any, ...]] = Counter()
        seq_mismatch = 0
        ipc_lag_or_end = 0
        # Also evaluate causal active slices and B0 online outcome status at T (strictly pre-timeout!)
        for tw, dec in zip(windows, het_decs):
            gov_w = int(tw["governed_window_index"])
            if int(tw["step_index"]) >= cut_step:
                ipc_lag_or_end += 1
                continue
            if dec.get("missing_reason"):
                ipc_lag_or_end += 1
                continue
            dec_out_w = int(dec["outcome_window_index"])
            exec_mac = prb_exec_by_win.get(gov_w)
            act_dec = tuple(int(v) for v in tw["a_exec_per_slice"])
            if exec_mac != act_dec or gov_w != dec_out_w:
                ipc_lag_or_end += 1
                continue
            if gov_w - 1 >= len(seq_actions) or tuple(seq_actions[gov_w - 1]) != act_dec:
                seq_mismatch += 1
                continue

            act_mask, _live_mask = causal_active_slices(tw["state"])
            n_act = sum(1 for x in act_mask if x)
            n_ue_nz = sum(1 for k in ("n_e", "n_u", "n_m") if int(tw["state"][k]) > 0)
            prop_excess = sum(int(v) for v in tw["a_prop_per_slice"]) > int(tw["capacity_C"])

            margins = dec.get("sla_outcome_margins") or [None, None, None]
            e_ok = (float(margins[0]) >= 0.0) if margins[0] is not None else None
            m_ok = (float(margins[2]) >= 0.0) if margins[2] is not None else None
            tx_w = int(tx_u_by_w.get(gov_w, 0))
            u_ok: bool | None = None
            u_reason = "ok"
            if margins[1] is not None:
                u_ok = float(margins[1]) >= 0.0
            elif tx_w == 0:
                u_reason = "u_no_demand"
            else:
                u_rec = ul_by_w.get(gov_w)
                if u_rec and u_rec.get("tag") == "CLOSED" and u_rec.get("p95_ms") is not None:
                    u_ok = float(u_rec["p95_ms"]) <= 10.0
                else:
                    u_reason = "u_unclosed_online"

            if e_ok is not None and u_ok is not None and m_ok is not None:
                all3_lbl = "PASS" if (e_ok and u_ok and m_ok) else "FAIL"
            else:
                all3_lbl = "UNKNOWN"

            dem_oks = []
            e_dem = margins[0] is not None
            u_dem = (tx_w > 0) or (margins[1] is not None)
            m_dem = margins[2] is not None
            if e_dem:
                dem_oks.append(e_ok)
            if u_dem:
                dem_oks.append(u_ok)
            if m_dem:
                dem_oks.append(m_ok)
            if not dem_oks:
                dem_lbl = "NO_DEMAND"
            elif any(x is None for x in dem_oks):
                dem_lbl = "UNKNOWN"
            else:
                dem_lbl = "PASS" if all(x is True for x in dem_oks) else "FAIL"

            strata_counts[(all3_lbl, dem_lbl, n_act, n_ue_nz, prop_excess, u_reason)] += 1

        parity_reports.append({
            "tag": tag,
            "lam": lam,
            "seed": seed,
            "orig_first_ipc_lag_step": orig_first_lag,
            "traced_first_ipc_lag_step": traced_first_lag,
            "clean_cut_step": cut_step,
            "pre_timeout_parity_ok": pre_timeout_parity_ok,
            "pre_checks": {
                "dec": pre_dec_eq,
                "kpm": pre_kpm_eq,
                "prb": pre_prb_eq,
                "del_delivered": pre_del_eq,
                "step_excl_dt": pre_step_eq,
            },
            "file_checks": file_checks,
            "checkpoint_payload_sha256_orig": pay_orig,
            "checkpoint_payload_sha256_traced": pay_tr,
            "checkpoint_payload_match_full_ep": ckpt_match,
            "checkpoint_self_hash_valid": ckpt_self_valid,
            "config_sha256_match": cfg_match,
            "checkpoint_horizon": horizon,
            "trace_header_ok": header_ok,
            "trace_windows": len(windows),
            "het_decisions": len(het_decs),
            "causal_invariants_ok": causal_invariants_ok,
            "future_ckpt_blocked_count": future_ckpt_blocked,
            "max_f_candidate_err": max_f_err,
            "seq_mismatch_excluding_lag": seq_mismatch,
            "ipc_lag_or_end_excluded": ipc_lag_or_end,
        })

        for key, cnt in sorted(strata_counts.items()):
            pool_summary.append({
                "tag": tag,
                "lam": lam,
                "seed": seed,
                "all3_r19": key[0],
                "demand_r19": key[1],
                "n_active_causal": key[2],
                "n_ue_nonzero": key[3],
                "prop_excess_gt53": key[4],
                "u_status": key[5],
                "count": cnt,
            })

    out_parity = os.path.join(TRACED_DIR, "b0_traced_parity_report.json")
    with open(out_parity, "w", encoding="utf-8") as fh:
        json.dump({"all_ok": all_ok, "runs": parity_reports}, fh, indent=2)

    out_pool = os.path.join(TRACED_DIR, "candidate_pool_summary.json")
    with open(out_pool, "w", encoding="utf-8") as fh:
        json.dump({"strata": pool_summary}, fh, indent=2)

    print(f"PARITY ALL_OK={all_ok}")
    for rep in parity_reports:
        fc_ok = all(v["exact"] for v in rep["file_checks"].values())
        print(
            f"  {rep['tag']}: pre_parity_ok={rep['pre_timeout_parity_ok']} cut={rep['clean_cut_step']} "
            f"(orig_lag={rep['orig_first_ipc_lag_step']}, tr_lag={rep['traced_first_ipc_lag_step']}) "
            f"full_files={fc_ok} ckpt_full={rep['checkpoint_payload_match_full_ep']} "
            f"cfg_ok={rep['config_sha256_match']} causal_ok={rep['causal_invariants_ok']} "
            f"seq_mis={rep['seq_mismatch_excluding_lag']} lag_excl={rep['ipc_lag_or_end_excluded']}"
        )
    print("\nCandidate strata (PASS/FAIL ALL3 + DEMAND + active counts):")
    df_pool = pd.DataFrame(pool_summary)
    print(df_pool.to_string(index=False))
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
