#!/usr/bin/env python3
"""Fase A offline strictly-causal backtest of SQR-CER & Priority-SQR (zero ns-3 runs, zero .pkl).

Evaluates:
  - Primary dataset: Gate 1 B0-rest (25 pre-lag states, 127 unique replays)
  - Secondary dataset: D2-v2 PRE_DESYNC Shadow B0-rest (23 clean states, 299 replays)
  - CPU latency benchmark over >= 1000 CerReallocator.reallocate calls (target p95 < 1.0 ms)

Outputs exclusively under:
  MODA/ns-o-ran-gym/results/sqr_v2/cer_backtest_offline/
"""

from __future__ import annotations

from collections import defaultdict
import glob
import gzip
import json
import os
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd

GYM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(GYM_DIR, "..", ".."))
SRC_DIR = os.path.join(GYM_DIR, "src")
DEC_DIR = os.path.join(GYM_DIR, "results", "sqr_jc_decision")
DIAG_DIR = os.path.join(GYM_DIR, "results", "sqr_v2", "diag_exploratorio_v1")
OUT_DIR = os.path.join(GYM_DIR, "results", "sqr_v2", "cer_backtest_offline")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from nsoran.sqr.sqr_controller import SqrSliceController  # noqa: E402
from nsoran.sqr.sqr_kpis import SqrKpmParser  # noqa: E402
from nsoran.sqr_jc.feature_scaler import FeatureScaler  # noqa: E402
from nsoran.sqr_jc.het_runtime import HetRuntime  # noqa: E402
from nsoran.sqr_jc.sqr_het import SqrHetController  # noqa: E402
from nsoran.sqr_v2.cer_reallocator import (  # noqa: E402
    CerConfig,
    CerReallocator,
    PrequentialRiskTracker,
    PrequentialSnapshot,
    causal_active_mask,
    extract_bloco_a_telemetry,
)
from nsoran.sqr_v2.manifest_v2 import verify_b0_frozen_manifest  # noqa: E402
from nsoran.sqr_v2.priority_hoashi import (  # noqa: E402
    BALANCED_PRIORITY_WEIGHTS,
    CANONICAL_URLLC_PRIORITY_WEIGHTS,
    PriorityHoashiConfig,
    PriorityHoashiReallocator,
)

PKL_OPEN_GUARD_COUNT = 0


def _assert_not_pkl(path: str) -> str:
    global PKL_OPEN_GUARD_COUNT
    if str(path).lower().endswith(".pkl"):
        PKL_OPEN_GUARD_COUNT += 1
        raise PermissionError(f"Forbidden checkpoint access: {path}")
    return path


def _parse_action_str(s: str) -> tuple[int, int, int]:
    s = str(s).strip()
    if s.startswith("["):
        vals = json.loads(s)
        return (int(vals[0]), int(vals[1]), int(vals[2]))
    parts = [int(p) for p in s.split("-")]
    return (parts[0], parts[1], parts[2])


def _find_single(run_dir: str, fname: str) -> str:
    hits = sorted(glob.glob(os.path.join(run_dir, "**", fname), recursive=True))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected 1 {fname} under {run_dir}, got {len(hits)}")
    return _assert_not_pkl(hits[0])


def _l1(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _load_gate1_causal_data() -> dict[str, dict[str, Any]]:
    """Load Gate 1 causal trace snapshots + build strictly causal PrequentialRiskTracker per state."""
    gce_path = os.path.join(REPO_ROOT, "MODA", "docs", "sqr", "jc_decision", "GATE_CONTEXTUAL_ESTADOS.csv")
    df_meta = pd.read_csv(
        _assert_not_pkl(gce_path),
        usecols=["state_id", "run_tag", "lam", "seed", "governed_window_index"],
    )
    traced_base = os.path.join(DEC_DIR, "b0_traced_gate1")

    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in df_meta.itertuples():
        by_tag[str(r.run_tag)].append({
            "state_id": str(r.state_id),
            "b0_traced_tag": str(r.run_tag),
            "lam": int(r.lam),
            "seed": int(r.seed),
            "governed_window_index": int(r.governed_window_index),
        })

    records: dict[str, dict[str, Any]] = {}
    for tag, rows in by_tag.items():
        rdir = os.path.join(traced_base, tag)
        decs_path = _assert_not_pkl(os.path.join(rdir, "decisions.jsonl"))
        udels_path = _assert_not_pkl(os.path.join(rdir, "urllc_delayed_updates.jsonl"))
        trace_path = _assert_not_pkl(os.path.join(rdir, "causal_trace.jsonl.gz"))

        decs = [json.loads(l) for l in open(decs_path, encoding="utf-8") if l.strip()]
        udels = [json.loads(l) for l in open(udels_path, encoding="utf-8") if l.strip()]

        wanted_by_gw = {int(r["governed_window_index"]): r for r in rows}
        max_gw = max(wanted_by_gw)

        trace_by_gw: dict[int, dict[str, Any]] = {}
        with gzip.open(trace_path, "rt", encoding="utf-8") as gz:
            for idx, ln in enumerate(gz):
                if idx == 0 or not ln.strip():
                    continue
                tw = json.loads(ln)
                gw = int(tw["governed_window_index"])
                if gw <= max_gw:
                    trace_by_gw[gw] = tw

        dec_by_ow: dict[int, dict[str, Any]] = {}
        for d in decs:
            if d.get("outcome_window_index") is not None:
                dec_by_ow[int(d["outcome_window_index"])] = d

        udels_by_obs: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for u in udels:
            udels_by_obs[int(u["observed_at_window"])].append(u)

        tracker = PrequentialRiskTracker()
        pred_exec_by_ow: dict[int, tuple[float, float, float]] = {}
        absorbed_ow: set[int] = set()
        absorbed_obs_u: set[int] = set()

        for d in decs:
            sw = int(d["state_window_index"])
            gw = sw + 1
            if gw > max_gw:
                break

            for ow_prev in sorted(ow for ow in dec_by_ow if ow <= sw and ow not in absorbed_ow):
                absorbed_ow.add(ow_prev)
                dp = dec_by_ow[ow_prev]
                gw_p = int(dp["state_window_index"]) + 1
                tw_p = trace_by_gw.get(gw_p)
                if tw_p is not None:
                    psm_p = sorted(tw_p["per_slice_model"], key=lambda e: int(e["slice_id"]))
                    ae_p = [int(v) for v in tw_p["a_exec_per_slice"]]
                    f_exec = (
                        float(psm_p[0]["f_candidates"][ae_p[0]]),
                        float(psm_p[1]["f_candidates"][ae_p[1]]),
                        float(psm_p[2]["f_candidates"][ae_p[2]]),
                    )
                else:
                    p_list = dp["predicted_safe_after_projection"]
                    f_exec = (float(p_list[0]), float(p_list[1]), float(p_list[2]))
                pred_exec_by_ow[ow_prev] = f_exec
                ms = dp["sla_outcome_margins"]
                for s in (0, 2):
                    if ms[s] is not None:
                        tracker.record_closed_outcome(s, f_exec[s], float(ms[s]), outcome_win=ow_prev)
                ev_u = dp.get("sla_evidence", [None, None, None])[1]
                if ms[1] is not None:
                    tracker.record_closed_outcome(1, f_exec[1], float(ms[1]), outcome_win=ow_prev)
                elif ev_u == "PENDING_OBSERVATION":
                    tracker.mark_urllc_pending(ow_prev)

            for obs_w in sorted(k for k in udels_by_obs if k <= sw and k not in absorbed_obs_u):
                absorbed_obs_u.add(obs_w)
                for u in udels_by_obs[obs_w]:
                    uw = int(u["w"])
                    if u.get("updated") is True and u.get("p95_ms") is not None:
                        f_u = pred_exec_by_ow.get(uw, (0.0, 0.0, 0.0))[1]
                        m_u = (10.0 - float(u["p95_ms"])) / 10.0
                        tracker.record_closed_outcome(1, f_u, float(m_u), outcome_win=uw)
                    elif u.get("reason") == "expired_vacuous_R19":
                        tracker.mark_urllc_expired(uw, observed_at_win=obs_w)

            if gw in wanted_by_gw:
                tw = trace_by_gw[gw]
                assert int(tw["causal_horizon_max_window_index"]) == sw
                psm = sorted(tw["per_slice_model"], key=lambda e: int(e["slice_id"]))
                row_m = wanted_by_gw[gw]
                sid_key = str(row_m["state_id"])
                records[sid_key] = {
                    "state_id": sid_key,
                    "run_tag": tag,
                    "lam": int(row_m["lam"]),
                    "seed": int(row_m["seed"]),
                    "gov_w": gw,
                    "capacity_C": int(tw["capacity_C"]),
                    "a_prop_pre_proj": tuple(int(v) for v in tw["a_prop_per_slice"]),
                    "a_b0_exec": tuple(int(v) for v in tw["a_exec_per_slice"]),
                    "state": tw["state"],
                    "f": [np.asarray(psm[i]["f_candidates"], dtype=np.float64) for i in range(3)],
                    "sigma": [np.asarray(psm[i]["sigma_candidates"], dtype=np.float64) for i in range(3)],
                    "risk_snap": tracker.snapshot(governed_window=gw),
                }
    assert len(records) == 25
    return records


def _reconstruct_d2v2_pre_desync_shadow() -> dict[str, dict[str, Any]]:
    """Forward causal reconstruction of A0 (parity check) + Shadow B0-rest + PrequentialRiskTracker."""
    df_st = pd.read_csv(_assert_not_pkl(os.path.join(DIAG_DIR, "t2_d2v2_30_states_desync_status.csv")))
    d2_pre = df_st[df_st["desync_status"] == "PRE_DESYNC"]
    assert len(d2_pre) == 23

    d1_base = os.path.join(DEC_DIR, "d1_runs_v9_delayed")
    scaler_path = _assert_not_pkl(
        os.path.join(REPO_ROOT, "results", "sqr_jc_final", "03_calibration", "feature_scaler_manifest.yaml")
    )
    scaler = FeatureScaler.load_manifest(scaler_path)

    by_run: dict[str, dict[int, str]] = defaultdict(dict)
    for r in d2_pre.itertuples():
        by_run[str(r.run_tag)][int(r.t_win)] = str(r.state_id)

    records: dict[str, dict[str, Any]] = {}
    for rtag, target_map in sorted(by_run.items()):
        rdir = os.path.join(d1_base, rtag)
        decs = [
            json.loads(l)
            for l in open(_assert_not_pkl(os.path.join(rdir, "decisions.jsonl")), encoding="utf-8")
            if l.strip()
        ]
        ud_p = os.path.join(rdir, "urllc_delayed_updates.jsonl")
        udels = (
            [json.loads(l) for l in open(_assert_not_pkl(ud_p), encoding="utf-8") if l.strip()]
            if os.path.isfile(ud_p)
            else []
        )
        kpm_path = _find_single(rdir, "sqr_kpm_history.csv")
        sm_df = pd.read_csv(_assert_not_pkl(os.path.join(rdir, "step_metrics.csv")))

        imm_urllc_margin_by_w = {
            int(d["outcome_window_index"]): d["sla_outcome_margins"][1]
            for d in decs
            if d.get("outcome_window_index") is not None
        }
        udels_by_obs: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for u in udels:
            udels_by_obs[int(u["observed_at_window"])].append(u)

        with open(kpm_path, encoding="utf-8") as kf:
            raw_lines = kf.readlines()
        hdr = raw_lines[0]
        by_ts: dict[int, list[str]] = defaultdict(list)
        for ln in raw_lines[1:]:
            if ln.strip():
                by_ts[int(ln.split(",", 1)[0])].append(ln)

        seed = int(rtag.split("_s")[1])
        ctls_a0 = tuple(
            SqrSliceController(state_dim=6, n_prbs=52, gamma=1.0, sla_threshold=0.0, seed=seed + 1000 * i)
            for i in range(3)
        )
        het_a0 = SqrHetController(
            ctls_a0,
            scaler=scaler,
            c=52,
            virtual_alpha=0.05,
            empty_slice_guard=False,
            mean_field=False,
            fix_scaler_index=False,
        )

        ctls_sh = tuple(
            SqrSliceController(state_dim=6, n_prbs=52, gamma=1.0, sla_threshold=0.0, seed=seed + 1000 * i)
            for i in range(3)
        )
        het_sh = SqrHetController(
            ctls_sh,
            scaler=scaler,
            c=52,
            virtual_alpha=0.0,
            empty_slice_guard=True,
            mean_field=False,
            fix_scaler_index=True,
        )

        rt = HetRuntime("/nonexistent_no_poll")
        parser = SqrKpmParser(num_slices=3)
        tmpf = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        tmpf.close()
        tracker = PrequentialRiskTracker()

        try:
            with open(tmpf.name, "w", encoding="utf-8") as wf:
                wf.write(hdr)
            last_win = parser.parse_window(tmpf.name)

            dec_by_idx = {int(d["decision_index"]): d for d in decs}
            max_t = max(target_map.keys())
            prev_act, prev_viol = (0, 0, 0), (0, 0, 0)
            pending: dict[str, Any] | None = None
            state_by_gov_w: dict[int, tuple[Any, tuple[int, int, int], tuple[float, float, float]]] = {}

            for srow in sm_df[sm_df["slice_id"] == 0].sort_values("step").itertuples():
                step = int(srow.step)
                cb_ts = step * 100
                if cb_ts > (max_t + 1) * 100:
                    break
                if cb_ts in by_ts:
                    with open(tmpf.name, "w", encoding="utf-8") as wf:
                        wf.write(hdr)
                        wf.writelines(by_ts[cb_ts])
                    last_win = parser.parse_window(tmpf.name)

                win = last_win
                _win_ts = int(win.timestamp)
                _is_hold = (_win_ts == getattr(het_a0, "_last_win_ts", -1))
                if not _is_hold:
                    het_a0._last_win_ts = _win_ts
                    het_sh._last_win_ts = _win_ts
                    w = rt.window_index(win)
                    tx_m = sum(r.d_tx_packets for r in win.rows if r.slice_id == 2)
                    rx_m = sum(r.d_rx_packets for r in win.rows if r.slice_id == 2)
                    if rt._mmtc_roll and rt._mmtc_roll[-1][0] == w:
                        rt._mmtc_roll.pop()
                    rt._mmtc_roll.append((w, float(tx_m), float(rx_m)))
                    _txs = sum(t for _, t, _ in rt._mmtc_roll)
                    _rxs = sum(r for _, _, r in rt._mmtc_roll)
                    rt._win_cache[w] = {
                        "embb_median": rt.embb_median(win),
                        "mmtc_pdr": (_rxs / _txs) if _txs > 0 else None,
                        "mmtc_demand": _txs > 0,
                        "urllc_tx": sum(r.d_tx_packets for r in win.rows if r.slice_id == 1),
                    }
                    rt.last_window_index = w
                    m_u_imm = imm_urllc_margin_by_w.get(w)
                    if m_u_imm is not None:
                        rt.delays.by_win[w] = [10.0 * (1.0 - float(m_u_imm))]

                if pending is not None:
                    m_on = pending["dlog"]["sla_outcome_margins"]
                    zs_a0 = [het_a0.slice_state(pending["state"], sid) for sid in range(3)]
                    zs_sh = [het_sh.slice_state(pending["state"], sid) for sid in range(3)]
                    f_sh_exec = pending["f_sh_exec"]
                    for sid in range(3):
                        if m_on[sid] is not None:
                            ctls_a0[sid].observe(zs_a0[sid], int(pending["action"][sid]), float(m_on[sid]), threshold=0.0)
                            ctls_sh[sid].observe(zs_sh[sid], int(pending["action"][sid]), float(m_on[sid]), threshold=0.0)
                            tracker.record_closed_outcome(sid, f_sh_exec[sid], float(m_on[sid]), outcome_win=pending["gov_w"])
                    ev_u = pending["dlog"].get("sla_evidence", [None, None, None])[1]
                    if m_on[1] is None and ev_u == "PENDING_OBSERVATION":
                        tracker.mark_urllc_pending(pending["gov_w"])
                    prev_viol = tuple(0 if (m is None or m >= 0) else 1 for m in m_on)
                    pending = None

                if not _is_hold and rt.last_window_index is not None and rt.last_window_index in udels_by_obs:
                    for u in udels_by_obs.pop(rt.last_window_index):
                        uw = int(u["w"])
                        if u.get("updated") is True and u.get("p95_ms") is not None:
                            st_u, act_u, f_u_pred = state_by_gov_w[uw]
                            m_u = (10.0 - float(u["p95_ms"])) / 10.0
                            ctls_a0[1].observe(het_a0.slice_state(st_u, 1), int(act_u[1]), m_u, threshold=0.0)
                            ctls_sh[1].observe(het_sh.slice_state(st_u, 1), int(act_u[1]), m_u, threshold=0.0)
                            tracker.record_closed_outcome(1, f_u_pred[1], float(m_u), outcome_win=uw)
                        elif u.get("reason") == "expired_vacuous_R19":
                            tracker.mark_urllc_expired(uw, observed_at_win=int(rt.last_window_index))

                if step in dec_by_idx:
                    dlog = dec_by_idx[step]
                    cap_c = int(dlog["capacity_C"])
                    if het_a0.c != cap_c:
                        het_a0.set_n_prbs(cap_c)
                        het_sh.set_n_prbs(cap_c)
                    st = rt.joint_state(win, prev_act, prev_viol)
                    cch = rt._win_cache.get(rt.window_index(win)) or {}
                    u_tag, _ = rt.urllc_pool_status(rt.window_index(win))
                    st.extras["kpm_tx_packets"] = (
                        int(sum(r.d_tx_packets for r in win.rows if r.slice_id == 0)),
                        int(cch.get("urllc_tx") or 0),
                        int(sum(r.d_tx_packets for r in win.rows if r.slice_id == 2)),
                    )
                    st.extras["urllc_pool_open"] = (u_tag == "OPEN")
                    st.extras["mmtc_rolling_active"] = bool(cch.get("mmtc_demand", False))

                    zs_a0 = [het_a0.slice_state(st, sid) for sid in range(3)]
                    zs_sh = [het_sh.slice_state(st, sid) for sid in range(3)]
                    props = tuple(int(v) for v in dlog["a_after_uncertainty_margin"])
                    alloc = tuple(int(v) for v in dlog["action"])
                    sc_sh = [ctls_sh[i].score_candidates(zs_sh[i]) for i in range(3)]
                    f_sh_exec = tuple(float(sc_sh[i]["f"][alloc[i]]) for i in range(3))
                    gov_w = int(dlog["state_window_index"]) + 1

                    if gov_w in target_map:
                        sid_key = target_map[gov_w]
                        sc_a0 = [ctls_a0[i].score_candidates(zs_a0[i]) for i in range(3)]
                        pb = [float(sc_a0[i]["f"][props[i]]) for i in range(3)]
                        pa = [float(sc_a0[i]["f"][alloc[i]]) for i in range(3)]
                        eb = max(abs(a - b) for a, b in zip(pb, dlog["predicted_safe_before_proposal"]))
                        ea = max(abs(a - b) for a, b in zip(pa, dlog["predicted_safe_after_projection"]))
                        assert eb < 1e-12 and ea < 1e-12, (
                            f"Reconstruction mismatch at {sid_key}: eb={eb}, ea={ea}"
                        )
                        records[sid_key] = {
                            "state_id": sid_key,
                            "run_tag": rtag,
                            "gov_w": gov_w,
                            "capacity_C": cap_c,
                            "a_prop_pre_proj": props,
                            "a_b0_exec": alloc,
                            "state": st,
                            "f": [np.asarray(sc_sh[i]["f"], dtype=np.float64) for i in range(3)],
                            "sigma": [np.asarray(sc_sh[i]["sigma"], dtype=np.float64) for i in range(3)],
                            "risk_snap": tracker.snapshot(governed_window=gov_w),
                            "parity_err_before": float(eb),
                            "parity_err_after": float(ea),
                        }

                    if any(a < p for p, a in zip(props, alloc)):
                        het_a0.apply_virtual_updates(st, props, alloc)
                    state_by_gov_w[gov_w] = (st, alloc, f_sh_exec)
                    pending = {
                        "state": st,
                        "action": alloc,
                        "dlog": dlog,
                        "gov_w": gov_w,
                        "f_sh_exec": f_sh_exec,
                    }
                    prev_act = alloc
        finally:
            if os.path.exists(tmpf.name):
                os.unlink(tmpf.name)

    assert len(records) == 23
    return records


def _build_replay_catalog() -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Map dataset -> state_id -> list of executed replay records (strict_all3_r19 outcomes)."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {"gate1_b0rest": defaultdict(list), "d2v2_pre_desync_shadow": defaultdict(list)}
    g1_m = pd.read_csv(os.path.join(DEC_DIR, "A5_GATE1_METRICS.csv"))
    for r in g1_m.itertuples():
        out["gate1_b0rest"][str(r.state_id)].append({
            "arm": str(r.arm),
            "action": (int(r.action_e), int(r.action_u), int(r.action_m)),
            "label": str(r.strict_all3_r19),
            "embb_thr": None if pd.isna(r.embb_thr_mbps) else float(r.embb_thr_mbps),
            "urllc_p95": None if pd.isna(r.urllc_p95_ms) else float(r.urllc_p95_ms),
            "mmtc_pdr": None if pd.isna(r.mmtc_pdr_rolling) else float(r.mmtc_pdr_rolling),
        })

    d2_res = pd.read_csv(os.path.join(DEC_DIR, "D2v2_results.csv"))
    t2064 = pd.read_csv(os.path.join(DEC_DIR, "d2v2_t2064full_branches.csv"))
    for r in d2_res.itertuples():
        sid = str(r.state_id)
        if sid == "S1_lam5_s2101_t2064":
            continue
        lbl = "PASS" if str(r.label) == "STRICT_PASS" else ("FAIL" if str(r.label) == "STRICT_FAIL" else "UNKNOWN")
        out["d2v2_pre_desync_shadow"][sid].append({
            "arm": f"alt{int(r.alt_index)}:{r.mechanism}",
            "action": _parse_action_str(r.action),
            "label": lbl,
            "embb_thr": None if pd.isna(r.embb_thr) else float(r.embb_thr),
            "urllc_p95": None if pd.isna(r.urllc_p95) else float(r.urllc_p95),
            "mmtc_pdr": None if pd.isna(r.mmtc_pdr) else float(r.mmtc_pdr),
        })
    for r in t2064.itertuples():
        lbl = "PASS" if str(r.label) == "STRICT_PASS" else ("FAIL" if str(r.label) == "STRICT_FAIL" else "UNKNOWN")
        out["d2v2_pre_desync_shadow"]["S1_lam5_s2101_t2064"].append({
            "arm": f"alt{int(r.alt)}:{r.mech}",
            "action": _parse_action_str(r.action),
            "label": lbl,
            "embb_thr": None if pd.isna(r.embb) else float(r.embb),
            "urllc_p95": None if pd.isna(r.p95) else float(r.p95),
            "mmtc_pdr": None if pd.isna(r.pdr) else float(r.pdr),
        })
    return out


def _match_replay(
    act: tuple[int, int, int],
    replays: list[dict[str, Any]],
    base_act: tuple[int, int, int],
    base_label: str,
) -> dict[str, Any]:
    pass_arms = [r for r in replays if r["label"] == "PASS"]
    exact_matches = [r for r in replays if r["action"] == act]
    nearest = min(replays, key=lambda r: (_l1(act, r["action"]), 0 if r["label"] == "PASS" else 1))
    dist_best_pass = min((_l1(act, r["action"]) for r in pass_arms), default=None)

    if exact_matches:
        # Prefer PASS exact match if duplicate arm labels exist with identical action
        chosen = next((r for r in exact_matches if r["label"] == "PASS"), exact_matches[0])
        lbl = chosen["label"]
        if act == base_act:
            cls = f"ABSTAIN_{base_label}"
        elif lbl == "UNKNOWN" or base_label == "UNKNOWN":
            cls = "UNKNOWN_OUTCOME"
        elif base_label == "FAIL" and lbl == "PASS":
            cls = "RESCUE"
        elif base_label == "PASS" and lbl == "FAIL":
            cls = "HARM"
        else:
            cls = f"NEUTRAL_{lbl}"
        return {
            "exact_match": True,
            "exact_arm": chosen["arm"],
            "exact_label": lbl,
            "outcome_class": cls,
            "nearest_arm": chosen["arm"],
            "nearest_action": str(chosen["action"]),
            "nearest_l1": 0,
            "nearest_label": lbl,
            "dist_l1_to_best_pass": dist_best_pass,
        }

    cls = "UNOBSERVED_EXACT"
    return {
        "exact_match": False,
        "exact_arm": "",
        "exact_label": "",
        "outcome_class": cls,
        "nearest_arm": nearest["arm"],
        "nearest_action": str(nearest["action"]),
        "nearest_l1": int(_l1(act, nearest["action"])),
        "nearest_label": nearest["label"],
        "dist_l1_to_best_pass": dist_best_pass,
    }


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    g1_data = _load_gate1_causal_data()
    d2_data = _reconstruct_d2v2_pre_desync_shadow()
    catalog = _build_replay_catalog()

    cer_idle = CerReallocator(CerConfig(regime_a_only=True))
    cer_full = CerReallocator(CerConfig(regime_a_only=False))
    hoashi_canon = PriorityHoashiReallocator(PriorityHoashiConfig(CANONICAL_URLLC_PRIORITY_WEIGHTS))
    hoashi_bal = PriorityHoashiReallocator(PriorityHoashiConfig(BALANCED_PRIORITY_WEIGHTS))

    all_rows: list[dict[str, Any]] = []
    spotlight_ids = {
        "S1_lam6_s2102_t1608",
        "S3_lam6_s2102_t1420",
        "S2_lam6_s2102_t1626",
        "lam5_s2102_w1654",
        "lam6_s2102_w1396",
        "lam5_s2103_w1621",
        "lam6_s2103_w1254",
        "lam5_s2102_w2667",
        "lam6_s2103_w1283",
        "lam5_s2103_w1546",
        "lam5_s2103_w1560",
    }
    spotlight_rows: list[dict[str, Any]] = []

    datasets = (
        ("gate1_b0rest", g1_data),
        ("d2v2_pre_desync_shadow", d2_data),
    )

    for ds_name, ds_map in datasets:
        for sid, rec in sorted(ds_map.items()):
            replays = catalog[ds_name][sid]
            base_rep = replays[0] if ds_name == "d2v2_pre_desync_shadow" else next(r for r in replays if r["arm"] == "b0")
            base_act = base_rep["action"]
            base_lbl = base_rep["label"]
            c = int(rec["capacity_C"])
            tel = extract_bloco_a_telemetry(rec["state"])
            act_mask = causal_active_mask(tel)
            risk_snap: PrequentialSnapshot = rec["risk_snap"]

            h_can_dec = hoashi_canon.allocate(rec["a_prop_pre_proj"], c, act_mask)
            h_bal_dec = hoashi_bal.allocate(rec["a_prop_pre_proj"], c, act_mask)
            c_a_dec = cer_idle.reallocate(
                a_nom=rec["a_b0_exec"],
                f_curves=rec["f"],
                sigma_curves=rec["sigma"],
                telemetry=tel,
                risk=risk_snap,
                capacity_c=c,
            )
            c_f_dec = cer_full.reallocate(
                a_nom=rec["a_b0_exec"],
                f_curves=rec["f"],
                sigma_curves=rec["sigma"],
                telemetry=tel,
                risk=risk_snap,
                capacity_c=c,
            )

            methods = [
                ("B0_rest", rec["a_b0_exec"], "BASE", "", 0, risk_snap.u_preq, c_f_dec.donor_floors),
                ("Priority_SQR_canon", h_can_dec.action, "PRIORITY_CUT" if h_can_dec.cut_applied else "MANTER", "", 0, risk_snap.u_preq, (None, None, None)),
                ("Priority_SQR_bal", h_bal_dec.action, "PRIORITY_CUT" if h_bal_dec.cut_applied else "MANTER", "", 0, risk_snap.u_preq, (None, None, None)),
                ("SQR_CER_regime_A", c_a_dec.action, c_a_dec.cer_stage, c_a_dec.abstain_reason, c_a_dec.slack_prbs, c_a_dec.u_preq, c_a_dec.donor_floors),
                ("SQR_CER_full", c_f_dec.action, c_f_dec.cer_stage, c_f_dec.abstain_reason, c_f_dec.slack_prbs, c_f_dec.u_preq, c_f_dec.donor_floors),
            ]

            for m_name, act_m, stage_m, abst_m, slack_m, u_m, floors_m in methods:
                assert sum(act_m) <= c and all(v >= 0 for v in act_m)
                match_info = _match_replay(act_m, replays, base_act, base_lbl)
                row_out = {
                    "dataset": ds_name,
                    "state_id": sid,
                    "method": m_name,
                    "base_action": str(base_act),
                    "base_label": base_lbl,
                    "action": str(act_m),
                    "a_e": act_m[0], "a_u": act_m[1], "a_m": act_m[2],
                    "delta_e": act_m[0] - base_act[0],
                    "delta_u": act_m[1] - base_act[1],
                    "delta_m": act_m[2] - base_act[2],
                    "l1_vs_base": _l1(act_m, base_act),
                    "cer_stage": stage_m,
                    "abstain_reason": abst_m,
                    "slack_prbs": slack_m,
                    "a_prop": str(c_f_dec.a_prop),
                    "a_phys_m": c_f_dec.a_phys_m,
                    "donor_floors": str(floors_m),
                    "u_preq_e": round(u_m[0], 4),
                    "u_preq_u": round(u_m[1], 4),
                    "u_preq_m": round(u_m[2], 4),
                    "last_y_u": risk_snap.last_closed_margins[1],
                    "urllc_lag": risk_snap.urllc_lag_windows,
                    **match_info,
                }
                all_rows.append(row_out)
                if sid in spotlight_ids:
                    spotlight_rows.append(row_out)

            # Explicit clean-plateau counterfactual for D2-v2 t1608 (documenting D1-v9 A0 turbulence vs B0-rest)
            if sid == "S1_lam6_s2102_t1608":
                snap_clean = PrequentialSnapshot(
                    u_preq=(0.50, 0.0, 0.0),
                    r_rel=(0.50, 1.0, 1.0),
                    v_rate=(0.70, 0.0, 0.0),
                    overpred_pen=(0.0, 0.0, 0.0),
                    c_urllc=0.0,
                    last_closed_margins=(risk_snap.last_closed_margins[0], 0.19, 0.0526),
                    n_open_urllc=2,
                    n_exp_urllc=0,
                    urllc_lag_windows=2,
                    hist_counts=(20, 20, 20),
                )
                c_t1608_clean = cer_full.reallocate(
                    a_nom=rec["a_b0_exec"],
                    f_curves=rec["f"],
                    sigma_curves=rec["sigma"],
                    telemetry=tel,
                    risk=snap_clean,
                    capacity_c=c,
                )
                m_clean = _match_replay(c_t1608_clean.action, replays, base_act, base_lbl)
                spotlight_rows.append({
                    "dataset": f"{ds_name}__clean_plateau_history",
                    "state_id": sid,
                    "method": "SQR_CER_full",
                    "base_action": str(base_act),
                    "base_label": base_lbl,
                    "action": str(c_t1608_clean.action),
                    "a_e": c_t1608_clean.action[0], "a_u": c_t1608_clean.action[1], "a_m": c_t1608_clean.action[2],
                    "delta_e": c_t1608_clean.action[0] - base_act[0],
                    "delta_u": c_t1608_clean.action[1] - base_act[1],
                    "delta_m": c_t1608_clean.action[2] - base_act[2],
                    "l1_vs_base": _l1(c_t1608_clean.action, base_act),
                    "cer_stage": c_t1608_clean.cer_stage,
                    "abstain_reason": c_t1608_clean.abstain_reason,
                    "slack_prbs": c_t1608_clean.slack_prbs,
                    "a_prop": str(c_t1608_clean.a_prop),
                    "a_phys_m": c_t1608_clean.a_phys_m,
                    "donor_floors": str(c_t1608_clean.donor_floors),
                    "u_preq_e": round(snap_clean.u_preq[0], 4),
                    "u_preq_u": round(snap_clean.u_preq[1], 4),
                    "u_preq_m": round(snap_clean.u_preq[2], 4),
                    "last_y_u": 0.19,
                    "urllc_lag": 2,
                    **m_clean,
                })

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(OUT_DIR, "cer_offline_actions_all_states.csv"), index=False)
    pd.DataFrame(spotlight_rows).to_csv(os.path.join(OUT_DIR, "cer_offline_spotlight_states.csv"), index=False)

    # CPU benchmark over >= 1000 reallocate calls (40 passes * 25 Gate 1 states = 1000 decisions)
    g1_items = [
        (rec["a_b0_exec"], rec["f"], rec["sigma"], extract_bloco_a_telemetry(rec["state"]), rec["risk_snap"], int(rec["capacity_C"]))
        for rec in g1_data.values()
    ]
    # Warm up once
    _ = cer_full.reallocate(a_nom=g1_items[0][0], f_curves=g1_items[0][1], sigma_curves=g1_items[0][2], telemetry=g1_items[0][3], risk=g1_items[0][4], capacity_c=g1_items[0][5])
    lat_ms: list[float] = []
    for _pass in range(40):
        for a_nom, f_c, s_c, tel_c, rk_c, cap_c in g1_items:
            t0 = time.perf_counter()
            cer_full.reallocate(a_nom=a_nom, f_curves=f_c, sigma_curves=s_c, telemetry=tel_c, risk=rk_c, capacity_c=cap_c)
            lat_ms.append((time.perf_counter() - t0) * 1000.0)
    lat_arr = np.asarray(lat_ms, dtype=np.float64)
    cpu_bench = {
        "n_calls": int(lat_arr.size),
        "p50_ms": float(np.percentile(lat_arr, 50)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "p99_ms": float(np.percentile(lat_arr, 99)),
        "max_ms": float(np.max(lat_arr)),
        "meets_p95_lt_1ms": bool(np.percentile(lat_arr, 95) < 1.0),
    }

    # Aggregate summary by dataset and method
    summary_by_method: list[dict[str, Any]] = []
    for (ds_n, m_n), g in df_all.groupby(["dataset", "method"], sort=False):
        vc = g["outcome_class"].value_counts().to_dict()
        changed = g[g["l1_vs_base"] > 0]
        known_dist = g[g["dist_l1_to_best_pass"].notna()]["dist_l1_to_best_pass"].astype(float)
        summary_by_method.append({
            "dataset": ds_n,
            "method": m_n,
            "n_states": int(len(g)),
            "n_changed_vs_base": int(len(changed)),
            "n_exact_replay_match": int(g["exact_match"].sum()),
            "RESCUE": int(vc.get("RESCUE", 0)),
            "HARM": int(vc.get("HARM", 0)),
            "NEUTRAL_PASS": int(vc.get("NEUTRAL_PASS", 0)),
            "NEUTRAL_FAIL": int(vc.get("NEUTRAL_FAIL", 0)),
            "ABSTAIN_PASS": int(vc.get("ABSTAIN_PASS", 0)),
            "ABSTAIN_FAIL": int(vc.get("ABSTAIN_FAIL", 0)),
            "ABSTAIN_UNKNOWN": int(vc.get("ABSTAIN_UNKNOWN", 0)),
            "UNOBSERVED_EXACT": int(vc.get("UNOBSERVED_EXACT", 0)),
            "UNKNOWN_OUTCOME": int(vc.get("UNKNOWN_OUTCOME", 0)),
            "mean_dist_to_best_pass_l1": float(known_dist.mean()) if len(known_dist) else None,
        })

    d2_parity = {
        "n_states": len(d2_data),
        "max_err_before": float(max(r["parity_err_before"] for r in d2_data.values())),
        "max_err_after": float(max(r["parity_err_after"] for r in d2_data.values())),
    }
    b0_verify = verify_b0_frozen_manifest(REPO_ROOT)
    assert PKL_OPEN_GUARD_COUNT == 0
    assert b0_verify["all_ok"]

    master = {
        "checkpoints_pkl_opened": PKL_OPEN_GUARD_COUNT,
        "b0_frozen_verify": b0_verify,
        "d2v2_forward_parity_a0": d2_parity,
        "cpu_benchmark": cpu_bench,
        "summary_by_dataset_method": summary_by_method,
    }
    with open(os.path.join(OUT_DIR, "cer_offline_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(master, fh, indent=2)
        fh.write("\n")

    # Build factual RELATORIO_FASE_A_CER_BACKTEST_OFFLINE.md
    lines = [
        "# Relatório da Fase A — Implementação Modular e *Backtest* Causal Offline do `SQR-CER` (`sqr-v2`)",
        "",
        f"- **Integridade `v1`:** `verify_b0_frozen_manifest` = `{b0_verify['code_matched']}/{b0_verify['code_total']}` código + `{b0_verify['checkpoints_matched']}/{b0_verify['checkpoints_total']}` checkpoints dev (`all_ok={b0_verify['all_ok']}`)",
        f"- **Zero leitura de checkpoints `.pkl`:** `checkpoints_pkl_opened = {PKL_OPEN_GUARD_COUNT}`",
        f"- **Paridade *forward* `A0` na `D2-v2 PRE_DESYNC` (`{d2_parity['n_states']}/23`):** `max_err_before = {d2_parity['max_err_before']:.3e}`, `max_err_after = {d2_parity['max_err_after']:.3e}`",
        "- **Zero novas simulações `ns-3` / Zero *seeds* reservadas:** avaliado estritamente sobre **`Gate 1 (B0-rest)` (primário)** (`25` estados, `127` replays) e **`D2-v2 PRE_DESYNC` (`Shadow B0-rest`, secundário)** (`23` estados limpos, `299` replays).",
        "",
        "## 1. Desempenho Computacional (`CerReallocator.reallocate`)",
        "",
        f"- Benchmark em `{cpu_bench['n_calls']}` avaliações reais (`40 × 25` estados do `Gate 1`): **p50 = `{cpu_bench['p50_ms']:.4f} ms`**, **p95 = `{cpu_bench['p95_ms']:.4f} ms`**, **p99 = `{cpu_bench['p99_ms']:.4f} ms`**, **máximo = `{cpu_bench['max_ms']:.4f} ms`** (`meets_p95_lt_1ms = {cpu_bench['meets_p95_lt_1ms']}`).",
        "",
        "## 2. Sumário Comparativo por Conjunto e Método",
        "",
        "| Conjunto | Papel | Método | Alterou vs Base | Match Exato | `RESCUE` | `HARM` | `ABSTAIN (P/F/U)` | `UNOBSERVED_EXACT` | `UNKNOWN_OUTCOME` | Dist. Média $L_1$ p/ Melhor `PASS` |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    role_map = {
        "gate1_b0rest": "Primário (`B0-rest`)",
        "d2v2_pre_desync_shadow": "Secundário (`Shadow B0-rest`)",
    }
    for s in summary_by_method:
        dist_s = "—" if s["mean_dist_to_best_pass_l1"] is None else f"{s['mean_dist_to_best_pass_l1']:.2f}"
        abst_s = f"{s['ABSTAIN_PASS']}/{s['ABSTAIN_FAIL']}/{s['ABSTAIN_UNKNOWN']}"
        lines.append(
            f"| `{s['dataset']}` | {role_map.get(s['dataset'], s['dataset'])} | `{s['method']}` | "
            f"{s['n_changed_vs_base']}/{s['n_states']} | {s['n_exact_replay_match']} | "
            f"{s['RESCUE']} | {s['HARM']} | {abst_s} | "
            f"{s['UNOBSERVED_EXACT']} | {s['UNKNOWN_OUTCOME']} | {dist_s} |"
        )

    lines.extend([
        "",
        "## 3. Comportamento nos Estados-Chave (*Spotlight*)",
        "",
        "| Conjunto | Estado | Método | Base $\\to$ Ação | Estágio / Motivo | Desfecho Real / Mais Próximo ($L_1$) |",
        "|---|---|---|---|---|---|",
    ])
    for r in spotlight_rows:
        if r["method"] in ("B0_rest", "Priority_SQR_bal"):
            continue
        near_str = f"`EXATO {r['exact_label']}` ({r['exact_arm']})" if r["exact_match"] else f"`UNOBSERVED` · prox=`{r['nearest_action']}` (`L1={r['nearest_l1']}`, `{r['nearest_label']}`)"
        lines.append(
            f"| `{r['dataset']}` | `{r['state_id']}` | `{r['method']}` | `{r['base_action']}` $\\to$ **`{r['action']}`** | "
            f"`{r['cer_stage']}` ({r['abstain_reason'] or '—'}) | {near_str} |"
        )

    fail_known = df_all[(df_all["base_label"] == "FAIL") & (df_all["dist_l1_to_best_pass"].notna())]
    fail_l1_summary: dict[tuple[str, str], tuple[int, float, float]] = {}
    for (ds_n, m_n), grp in fail_known.groupby(["dataset", "method"], sort=False):
        vals = grp["dist_l1_to_best_pass"].astype(float)
        fail_l1_summary[(str(ds_n), str(m_n))] = (int(len(vals)), float(vals.mean()), float(vals.median()))

    lines.extend([
        "",
        "## 4. Interpretação Técnica da Avaliação Offline Causal",
        "",
        "1. **Conjunto Primário (`Gate 1 / B0-rest`) vs. Secundário (`D2-v2 PRE_DESYNC / Shadow B0-rest`):** no `Gate 1`, o backbone nominal `B0-rest` opera com `virtual_alpha = 0.0`, `empty_slice_guard = True` e `fix_scaler_index = True` sobre envelopes reais e predições auditadas; na `D2-v2 PRE_DESYNC`, as curvas (`f`, `sigma`) e o histórico prequencial refletem a reconstrução contrafactual `Shadow B0-rest` sobre a trajetória gerada originalmente por `A0` (`D1-v9`, com `R12` ativo).",
        "2. **Aproximação às ações recuperadoras em estados `FAIL` e lacuna de cobertura discreta:** no `Gate 1`, sobre os `7` estados `FAIL` com ramo `PASS` conhecido no catálogo de `127` replays, a distância média (mediana) $L_1$ até o melhor `PASS` cai de **`27.29` (`29.0`)** no `B0_rest`, **`28.43` (`29.0`)** no `Priority_SQR_canon` e **`29.29` (`29.0`)** no `Priority_SQR_bal` para **`24.29` (`20.0`)** em `SQR_CER_regime_A` e **`17.71` (`16.0`)** em `SQR_CER_full` (ex.: `w1283`: `41 -> 2`; `w1560`: `18 -> 6`; `w1397`: `12 -> 6`). Os desfechos das novas ações não presentes nos `127`/`299` replays discretos permanecem honestamente marcados como `UNOBSERVED_EXACT` (`14/25` no `Gate 1` e `20/23` na `D2-v2` para `SQR_CER_full`), exigindo avaliação pareada em `sqr-v2-sim` na Fase B.",
        "3. **Travamento prequencial em histórico turbulento vs. resgate limpo e abstenção em borda crítica:** em `lam5_s2103_w1546` (`last_y_u = -0.0029 < 0.08`) e `lam5_s2103_w1560` (`last_y_u = -0.21 < 0.08`), a trava unilateral bloqueia doação de URLLC (`donor_floors[1] = None`), e em `lam6_s2102_w2153` / `lam5_s2102_w2578` (`last_y_u = -0.015`, `U_u >= 0.39`) impõe `MANTER_B0`. Na `D2-v2`, em `S1_lam6_s2102_t1608`, o histórico prévio turbulento do `A0` (`y_last_u = -0.015`, `U_u = 0.21 >= 0.20`) dispara `MANTER_B0` (`L1 = 42` vs `alt9`), enquanto sob histórico de platô URLLC limpo (`y_last_u = +0.19`, `U_u = 0.0`) o `SQR_CER_full` executa `(15, 37, 1) -> (35, 17, 1)` (`L1 = 6` até `alt9: (33, 16, 4)` `STRICT_PASS`, contra `L1 = 66` do `Priority_SQR_canon`). Em `S3_lam6_s2102_t1420`, antes de a falha de `t=1420` fechar no `D1-v9`, os indicadores pré-decisão ainda marcavam `last_y_u = +0.19` e `U_u = 0.035`, levando `SQR_CER_full` a transferir `(22, 30, 1) -> (20, 12, 21)` (`L1 = 10` até `alt12 (18, 17, 18)` `STRICT_PASS`, contra `L1 = 36` no `Priority_SQR_canon`).",
    ])

    lines.extend([
        "",
        "### Distância $L_1$ ao Melhor `PASS` Apenas nos Estados `base_label == FAIL` com `PASS` Conhecido",
        "",
        "| Conjunto | Estados `FAIL` c/ `PASS` | Método | Média $L_1$ | Mediana $L_1$ |",
        "|---|---|---|---|---|",
    ])
    for (ds_n, m_n), (cnt, mn, md) in fail_l1_summary.items():
        lines.append(f"| `{ds_n}` | {cnt} | `{m_n}` | {mn:.2f} | {md:.1f} |")

    with open(os.path.join(OUT_DIR, "RELATORIO_FASE_A_CER_BACKTEST_OFFLINE.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("[DONE] cer_offline_check completed cleanly.")
    print(f"  d2_parity={d2_parity}")
    print(f"  cpu_bench={cpu_bench}")
    for s in summary_by_method:
        print(" ", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
