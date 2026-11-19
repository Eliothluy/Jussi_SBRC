#!/usr/bin/env python3
"""Phase 3 Step 2: Fit clean FeatureScaler, calibrate L3 (ell_a), and calibrate U_preq.

Uses strictly dev seeds {101, 102, 103} on the sanitized Etapa 3A + Etapa 3B (TDD 5 ms)
simulator. Never reads reserved confirmatory seeds (MAIN {3001..3010}, OOS >=3101).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GYM_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(GYM_ROOT, "src")
NS3_PATH = os.path.abspath(os.path.join(GYM_ROOT, "..", "ns-3-dev"))

for p in (SRC_DIR, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from nsoran.sqr.kernel import MaternKernel
from nsoran.sqr.sqr_controller import MARGIN_BOUNDS, MARGIN_INIT, TAU_INIT, TAU_STEP, VIOLATION_WINDOW
from nsoran.sqr_jc.feature_scaler import CAL_RULE_FEATURES, FeatureScaler
from nsoran.sqr_jc.het_runtime import DelayStream, HetRuntime
from nsoran.sqr_v2.cer_reallocator import CerConfig, PrequentialRiskTracker
from nsoran.sqr_v2.controller_v2 import SqrSliceControllerV2, build_controllers_v2
from nsoran.sqr_v2.env_v2 import SqrV2Env
from nsoran.sqr_v2.manifest_v2 import validate_dev_seed
import sqr_v2_run

DEFAULT_OUT_DIR = os.path.join(GYM_ROOT, "results", "sqr_v2", "fase3_calibration")
ELL_CANDIDATES = (1.0, 0.50, 0.35, 0.25, 0.20, 0.15, 0.10)


def _auroc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute exact AUROC via average-rank Mann-Whitney U without external deps."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(s, dtype=np.float64)
    n = s.size
    i = 0
    while i < n:
        j = i + 1
        while j < n and s[order[j]] == s[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos_ranks = float(np.sum(ranks[y == 1]))
    u = sum_pos_ranks - 0.5 * pos * (pos + 1)
    return float(u / float(pos * neg))


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 5:
        return None
    xv, yv = x[m], y[m]
    if np.std(xv) <= 1e-12 or np.std(yv) <= 1e-12:
        return 0.0
    rx = pd.Series(xv).rank(method="average").to_numpy()
    ry = pd.Series(yv).rank(method="average").to_numpy()
    c = float(np.corrcoef(rx, ry)[0, 1])
    return 0.0 if math.isnan(c) else c


def _collect_one_calibration_rollout(task: tuple[str, int, float, str, str]) -> dict[str, Any]:
    mode, seed, sim_time_s, slice_lambdas, work_root = task
    validate_dev_seed(seed, allow_reserved=False)
    out_dir = os.path.join(work_root, f"{mode}_s{seed}")
    os.makedirs(out_dir, exist_ok=True)

    args = sqr_v2_run.parse_args(
        [
            "--controller",
            "prop" if mode == "prop" else "sqr",
            "--gamma",
            "1",
            "--het",
            "--empty-slice-guard",
            "--urllc-delayed-update",
            "--seed",
            str(seed),
            "--simTime",
            str(sim_time_s),
            "--cellRadius",
            "300",
            "--sliceLambdas",
            slice_lambdas,
            "--sliceApps",
            "greedy,urllc,mmtc",
            "--warmup_s",
            "20.0",
            "--ipc-mode",
            "strict_rt",
            "--ipc-timeout-ms",
            "50",
            "--checkpoint_interval",
            "0",
            "--output",
            out_dir,
            "--run-id",
            f"cal_{mode}_s{seed}",
        ]
    )

    config: dict[str, Any] = {
        "simTime": [args.simTime],
        "seed": [args.seed],
        "periodMs": [args.periodMs],
        "sliceLambdas": [slice_lambdas],
        "sliceApps": ["greedy,urllc,mmtc"],
        "cellRadius": [300.0],
    }
    env = SqrV2Env(
        ns3_path=os.path.abspath(NS3_PATH),
        scenario_configuration=config,
        output_folder=out_dir,
        total_prbs=args.c_prbs,
        num_slices=args.num_slices,
        sla_thresholds={0: 0.5, 1: 1.0, 2: 1.5},
        ipc_mode="strict_rt",
        ipc_timeout_ms=50,
        run_id=f"cal_{mode}_s{seed}",
    )
    ctls = build_controllers_v2(args, env)
    scaler = FeatureScaler.load_manifest(sqr_v2_run.DEFAULT_HET_SCALER_MANIFEST, freeze=True)

    obs, reset_info = env.reset(seed=seed)
    total_prbs = int(reset_info.get("total_prbs", 53))
    sqr_v2_run._sync_controllers_capacity(ctls, total_prbs)

    runtime = HetRuntime(str(env.sim_path))
    init_win = env.builder.last_window
    if init_win is not None:
        runtime.absorb(init_win)
    last_ts = int(init_win.timestamp) if init_win is not None else -1

    prev_action = (0, 0, 0)
    prev_viol = (0, 0, 0)
    trajectories: list[dict[str, Any]] = []

    try:
        step_idx = 0
        while True:
            win = env.builder.last_window
            st = runtime.joint_state(win, prev_action, prev_viol)
            gov_win = int(st.window_index) + 1
            raw_vec = st.raw_vector()

            if mode == "prop":
                decs = [ctls[sid].act(np.zeros(6)) for sid in range(3)]
                props = [int(d.action) for d in decs]
            else:
                z_list = sqr_v2_run._build_het_slice_states(st, scaler, total_prbs, 3)
                decs = [ctls[sid].act(z_list[sid]) for sid in range(3)]
                props = [int(d.action) for d in decs]

            env.set_next_decision_wall_ms(0.2)
            _next_obs, _r, terminated, truncated, info = env.step(np.array(props, dtype=np.int64))
            new_c = int(info.get("total_prbs", total_prbs))
            if new_c > 0 and new_c != total_prbs:
                total_prbs = new_c
                sqr_v2_run._sync_controllers_capacity(ctls, total_prbs)

            if info.get("episode_end_no_next_window"):
                break

            receipt = info.get("ipc_receipt") or {}
            ipc_status = str(receipt.get("ipc_status", "APPLIED"))
            proj = info.get("projection", {})
            a_exec = tuple(int(v) for v in receipt.get("prb_executed_mac", proj.get("prb_executed", props)))

            new_win = env.builder.last_window
            if new_win is not None and int(new_win.timestamp) != last_ts:
                runtime.absorb(new_win)
                last_ts = int(new_win.timestamp)

            online_oc = runtime.outcome_for_window(gov_win)
            if mode == "sqr" and receipt.get("window_gap", 0) == 0:
                z_list = sqr_v2_run._build_het_slice_states(st, scaler, total_prbs, 3)
                for sid in range(3):
                    m = online_oc.margins[sid]
                    if m is not None:
                        ctls[sid].observe(z_list[sid], a_exec[sid], float(m))

            prev_viol = tuple(0 if (m is None or m >= 0.0) else 1 for m in online_oc.margins)
            prev_action = a_exec

            trajectories.append({
                "mode": mode,
                "seed": seed,
                "step": step_idx + 1,
                "window_index": gov_win,
                "total_prbs": total_prbs,
                "ipc_status": ipc_status,
                "raw_vec": raw_vec.tolist(),
                "a_exec": list(a_exec),
                "online_margins": list(online_oc.margins),
                "embb_thr_mbps": online_oc.embb_throughput_mbps,
                "mmtc_pdr_rolling": online_oc.mmtc_pdr_rolling,
            })
            step_idx += 1
            if terminated or truncated:
                break
        env.wait_sim_exit(timeout_s=120.0)
        full_delays = DelayStream(os.path.join(str(env.sim_path), "**", "sqr_packet_delays.csv"))
        full_delays.poll()
        for tr in trajectories:
            u_st = full_delays.stats(tr["window_index"])
            oc = sqr_v2_run.SliceOutcome(
                embb_throughput_mbps=tr["embb_thr_mbps"],
                urllc_p95_ms=float(u_st["p95_ms"]) if u_st is not None else None,
                urllc_miss_pct=float(100.0 * u_st["miss_rate"]) if u_st is not None else None,
                mmtc_pdr_rolling=tr["mmtc_pdr_rolling"],
            )
            tr["uncensored_margins"] = list(oc.margins)
            tr["urllc_p95_ms"] = oc.urllc_p95_ms
    finally:
        env.close()

    return {"mode": mode, "seed": seed, "trajectories": trajectories}


def _evaluate_l3_and_collect_risk_samples(
    rollouts: list[dict[str, Any]],
    scaler: FeatureScaler,
    ell_candidates: tuple[float, ...],
    warmup_steps: int = 200,
) -> tuple[pd.DataFrame, dict[int, float]]:
    rows: list[dict[str, Any]] = []
    best_ell: dict[int, float] = {}

    for sid in range(3):
        slice_name = ("eMBB", "URLLC", "mMTC")[sid]
        best_score = -1e9
        best_e = 0.25

        for ell_a in ell_candidates:
            length_scale = 1.0 if abs(ell_a - 1.0) <= 1e-12 else np.array([1.0] * 6 + [ell_a], dtype=np.float64)
            in_support = 0
            total_support_steps = 0
            flat_d8_cnt = 0
            d8_total = 0
            inv_cnt = 0
            adj_total = 0
            delta_tot_vals: list[float] = []
            pred_exec_vals: list[float] = []
            real_m_vals: list[float] = []

            for roll in rollouts:
                ctl = SqrSliceControllerV2(
                    state_dim=6,
                    n_prbs=53,
                    gamma=1.0,
                    sla_threshold=0.0,
                    kernel=MaternKernel(dim=7, nu=1.5, length_scale=length_scale),
                    tau=TAU_INIT,
                    tau_step=TAU_STEP,
                    margin=MARGIN_INIT,
                    margin_bounds=MARGIN_BOUNDS,
                    violation_window=VIOLATION_WINDOW,
                    seed=int(roll["seed"]) + 1000 * sid,
                )
                for tr in roll["trajectories"]:
                    c = int(tr["total_prbs"])
                    if ctl.n_prbs != c:
                        ctl.n_prbs = c
                    z_raw = np.asarray(tr["raw_vec"], dtype=np.float64)
                    z_norm = scaler.transform_joint(z_raw)
                    # Slice local state indices per JointState + fix_scaler_index=True
                    # [n_i, queue_i, prev_kpi_i, snr_i, prev_a_i/C, viol_i]
                    z_i = np.array(
                        [
                            z_norm[sid],
                            z_norm[3 + sid],
                            z_norm[6 + sid],
                            z_norm[12 + sid],
                            z_norm[15 + sid],
                            float(tr["raw_vec"][18 + sid]),
                        ],
                        dtype=np.float64,
                    )
                    a_ex = int(tr["a_exec"][sid])
                    m_real = tr["uncensored_margins"][sid]

                    if int(tr["step"]) > warmup_steps and m_real is not None:
                        sc = ctl.score_candidates(z_i)
                        f = sc["f"]
                        sig = sc["sigma"]
                        sup = sig <= 0.15
                        in_support += int(np.sum(sup))
                        total_support_steps += int(f.size)
                        delta_tot_vals.append(float(f[-1] - f[0]))
                        # Adjacent inversions in support
                        adj_mask = sup[:-1] & sup[1:]
                        if np.any(adj_mask):
                            adj_total += int(np.sum(adj_mask))
                            inv_cnt += int(np.sum((f[1:] < (f[:-1] - 1e-6))[adj_mask]))
                        # Flattening at step delta=8 inside support vs gamma*8/C
                        if f.size > 8:
                            d8_mask = sup[:-8] & sup[8:]
                            if np.any(d8_mask):
                                d8_total += int(np.sum(d8_mask))
                                flat_d8_cnt += int(
                                    np.sum((np.abs(f[8:] - f[:-8]) < (8.0 / float(c)))[d8_mask])
                                )
                        pred_exec_vals.append(float(f[min(a_ex, f.size - 1)]))
                        real_m_vals.append(float(m_real))

                    m_obs = tr["online_margins"][sid] if sid != 1 else tr["uncensored_margins"][sid]
                    if m_obs is not None:
                        ctl.observe(z_i, a_ex, float(m_obs))

            flat_pct = 100.0 * flat_d8_cnt / max(1, d8_total)
            inv_pct = 100.0 * inv_cnt / max(1, adj_total)
            sup_pct = 100.0 * in_support / max(1, total_support_steps)
            med_df = float(np.median(delta_tot_vals)) if delta_tot_vals else 0.0
            sp_rho = _spearman_corr(np.asarray(pred_exec_vals), np.asarray(real_m_vals)) or 0.0

            # Selection objective: reward Spearman alignment + curve responsiveness, penalize flattening & excess inversions
            inv_penalty = max(0.0, inv_pct - 25.0) * 0.02
            score = sp_rho + 0.5 * (1.0 - flat_pct / 100.0) + 0.25 * min(1.0, max(0.0, med_df)) - inv_penalty
            rows.append({
                "slice_id": sid,
                "slice_name": slice_name,
                "ell_a": ell_a,
                "in_support_pct": round(sup_pct, 2),
                "flattening_d8_pct": round(flat_pct, 2),
                "inversion_pct": round(inv_pct, 2),
                "median_delta_f_tot": round(med_df, 4),
                "spearman_f_vs_m": round(sp_rho, 4),
                "l3_score": round(score, 4),
            })
            if ell_a < 1.0 and score > best_score:
                best_score = score
                best_e = float(ell_a)

        best_ell[sid] = best_e

    return pd.DataFrame(rows), best_ell


def _calibrate_upreq_thresholds(
    rollouts: list[dict[str, Any]],
    scaler: FeatureScaler,
    best_ell: dict[int, float],
    warmup_steps: int = 200,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Replay once with selected L3 ell_a to capture exact (f_exec, sigma_exec, m_real, pending/exp)
    beta_grid = [
        (0.70, 0.30, 0.30),
        (0.75, 0.25, 0.25),
        (0.80, 0.20, 0.20),
        (0.60, 0.40, 0.25),
        (0.85, 0.15, 0.15),
    ]

    # Pre-collect f_exec & sigma_exec trajectories for L3 controller
    enriched_runs: list[list[dict[str, Any]]] = []
    for roll in rollouts:
        ctls = {
            sid: SqrSliceControllerV2(
                state_dim=6,
                n_prbs=53,
                gamma=1.0,
                sla_threshold=0.0,
                kernel=MaternKernel(
                    dim=7,
                    nu=1.5,
                    length_scale=np.array([1.0] * 6 + [best_ell[sid]], dtype=np.float64),
                ),
                tau=TAU_INIT,
                tau_step=TAU_STEP,
                margin=MARGIN_INIT,
                margin_bounds=MARGIN_BOUNDS,
                violation_window=VIOLATION_WINDOW,
                seed=int(roll["seed"]) + 1000 * sid,
            )
            for sid in range(3)
        }
        run_steps: list[dict[str, Any]] = []
        for tr in roll["trajectories"]:
            c = int(tr["total_prbs"])
            for sid in range(3):
                ctls[sid].n_prbs = c
            z_norm = scaler.transform_joint(np.asarray(tr["raw_vec"], dtype=np.float64))
            z_by_slice = [
                np.array(
                    [z_norm[s], z_norm[3 + s], z_norm[6 + s], z_norm[12 + s], z_norm[15 + s], float(tr["raw_vec"][18 + s])],
                    dtype=np.float64,
                )
                for s in range(3)
            ]
            f_ex = [0.0, 0.0, 0.0]
            sig_ex = [1.0, 1.0, 1.0]
            for sid in range(3):
                sc = ctls[sid].score_candidates(z_by_slice[sid])
                idx = min(int(tr["a_exec"][sid]), len(sc["f"]) - 1)
                f_ex[sid] = float(sc["f"][idx])
                sig_ex[sid] = float(sc["sigma"][idx])
                m_obs = tr["online_margins"][sid] if sid != 1 else tr["uncensored_margins"][sid]
                if m_obs is not None:
                    ctls[sid].observe(z_by_slice[sid], int(tr["a_exec"][sid]), float(m_obs))
            run_steps.append({
                "step": int(tr["step"]),
                "win": int(tr["window_index"]),
                "f_ex": f_ex,
                "sig_ex": sig_ex,
                "m_real": tr["uncensored_margins"],
                "m_online": tr["online_margins"],
            })
        enriched_runs.append(run_steps)

    grid_rows: list[dict[str, Any]] = []
    best_weights = (0.70, 0.30, 0.30)
    best_comb_auroc = -1.0
    best_thresh = {
        "theta_u_def_e": 0.25,
        "theta_u_def_u": 0.30,
        "theta_u_don_e": 0.15,
        "theta_u_don_u": 0.20,
        "theta_u_don_m": 0.15,
    }

    for bv, bo, bc in beta_grid:
        u_vals = {0: [], 1: [], 2: []}
        sig_vals = {0: [], 1: [], 2: []}
        res_vals = {0: [], 1: [], 2: []}
        viol_vals = {0: [], 1: [], 2: []}

        for run_steps in enriched_runs:
            tracker = PrequentialRiskTracker(w_preq=20, beta_v=bv, beta_o=bo, beta_c=bc, l_max=10)
            for item in run_steps:
                w = item["win"]
                snap = tracker.snapshot(governed_window=w)
                if item["step"] > warmup_steps:
                    for sid in range(3):
                        m = item["m_real"][sid]
                        if m is not None:
                            u_vals[sid].append(float(snap.u_preq[sid]))
                            sig_vals[sid].append(float(item["sig_ex"][sid]))
                            res_vals[sid].append(abs(float(m) - float(item["f_ex"][sid])))
                            viol_vals[sid].append(1 if float(m) < 0.0 else 0)
                for sid in range(3):
                    m = item["m_real"][sid]
                    if m is not None:
                        tracker.record_closed_outcome(sid, float(item["f_ex"][sid]), float(m), outcome_win=w)
                    elif sid == 1 and item["m_online"][1] is None:
                        tracker.mark_urllc_pending(w)

        aurocs_u = {}
        aurocs_sig = {}
        spearman_u = {}
        for sid, sname in enumerate(("eMBB", "URLLC", "mMTC")):
            au_u = _auroc_mann_whitney(np.asarray(u_vals[sid]), np.asarray(viol_vals[sid]))
            au_s = _auroc_mann_whitney(np.asarray(sig_vals[sid]), np.asarray(viol_vals[sid]))
            sp_u = _spearman_corr(np.asarray(u_vals[sid]), np.asarray(res_vals[sid]))
            aurocs_u[sid] = au_u
            aurocs_sig[sid] = au_s
            spearman_u[sid] = sp_u
            grid_rows.append({
                "beta_v": bv,
                "beta_o": bo,
                "beta_c": bc,
                "slice_id": sid,
                "slice_name": sname,
                "n_samples": len(viol_vals[sid]),
                "n_violations": int(sum(viol_vals[sid])),
                "auroc_u_preq": round(au_u, 4) if au_u is not None else None,
                "auroc_sigma_kernel": round(au_s, 4) if au_s is not None else None,
                "spearman_u_vs_abs_err": round(sp_u, 4) if sp_u is not None else None,
            })

        valid_scores = [aurocs_u[s] for s in (0, 1) if aurocs_u[s] is not None]
        comb = float(np.mean(valid_scores)) if valid_scores else 0.0
        if comb > best_comb_auroc:
            best_comb_auroc = comb
            best_weights = (bv, bo, bc)
            # Empirical Youden/quantile midpoint separation capped safely inside operational bounds
            for sid, key_don, lo_cap, hi_cap in (
                (0, "theta_u_don_e", 0.12, 0.25),
                (1, "theta_u_don_u", 0.15, 0.30),
                (2, "theta_u_don_m", 0.12, 0.25),
            ):
                uv = np.asarray(u_vals[sid], dtype=np.float64)
                yv = np.asarray(viol_vals[sid], dtype=np.int64)
                if np.sum(yv == 1) >= 5 and np.sum(yv == 0) >= 5:
                    q_safe = float(np.percentile(uv[yv == 0], 75))
                    q_viol = float(np.percentile(uv[yv == 1], 25))
                    mid = 0.5 * (q_safe + q_viol)
                    best_thresh[key_don] = round(float(np.clip(mid, lo_cap, hi_cap)), 3)
                    if sid == 0:
                        best_thresh["theta_u_def_e"] = round(float(np.clip(best_thresh[key_don] + 0.08, 0.20, 0.35)), 3)
                    elif sid == 1:
                        best_thresh["theta_u_def_u"] = round(float(np.clip(best_thresh[key_don] + 0.08, 0.22, 0.40)), 3)

    calibrated_cfg = {
        "beta_v": best_weights[0],
        "beta_o": best_weights[1],
        "beta_c": best_weights[2],
        "dl_slots_per_win": 80,
        **best_thresh,
    }
    return pd.DataFrame(grid_rows), calibrated_cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Calibrate FeatureScaler, L3 ell_a, and U_preq on clean dev seeds")
    p.add_argument("--seeds", type=str, default="101,102,103")
    p.add_argument("--simTime", type=float, default=120.0)
    p.add_argument("--sliceLambdas", type=str, default="5,5,5")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--outDir", type=str, default=DEFAULT_OUT_DIR)
    args = p.parse_args(argv)

    seeds = [validate_dev_seed(int(x.strip()), allow_reserved=False) for x in args.seeds.split(",") if x.strip()]
    os.makedirs(args.outDir, exist_ok=True)
    work_root = tempfile.mkdtemp(prefix="fase3_cal_runs_", dir="/tmp")

    tasks = [(mode, s, float(args.simTime), args.sliceLambdas, work_root) for mode in ("prop", "sqr") for s in seeds]
    print(f"[Calib] Running {len(tasks)} clean TDD-5ms dev episodes (seeds={seeds}, simTime={args.simTime}s, lambdas={args.sliceLambdas})...")
    rollouts: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            futs = [pool.submit(_collect_one_calibration_rollout, t) for t in tasks]
            for fut in as_completed(futs):
                res = fut.result()
                rollouts.append(res)
                print(f"[Calib] Collected {res['mode']} seed={res['seed']} windows={len(res['trajectories'])}")
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    # Sort deterministically by (mode, seed)
    rollouts.sort(key=lambda r: (r["mode"], r["seed"]))

    # (a) Fit & freeze FeatureScaler for sanitized TDD 5 ms regime
    raw_vecs = np.asarray(
        [tr["raw_vec"] for r in rollouts for tr in r["trajectories"]],
        dtype=np.float64,
    )
    scaler_new = FeatureScaler(c=53, frozen=False)
    scaler_new.fit(raw_vecs)
    scaler_new.frozen = True
    scaler_path = os.path.join(args.outDir, "scaler_manifest_v2_tdd5ms.yaml")
    scaler_new.save_manifest(scaler_path, source="DEV_ETAPA3A_3B_TDD5MS")

    scaler_legacy = FeatureScaler.load_manifest(sqr_v2_run.DEFAULT_HET_SCALER_MANIFEST, freeze=True)
    scaler_cmp_rows = []
    for feat in CAL_RULE_FEATURES:
        scaler_cmp_rows.append({
            "feature": feat,
            "legacy_p1": scaler_legacy._p1.get(feat),
            "legacy_p99": scaler_legacy._p99.get(feat),
            "v2_tdd5ms_p1": scaler_new._p1.get(feat),
            "v2_tdd5ms_p99": scaler_new._p99.get(feat),
        })
    pd.DataFrame(scaler_cmp_rows).to_csv(os.path.join(args.outDir, "scaler_comparison.csv"), index=False)

    # (b) Evaluate L3 anisotropic action length-scale sweep
    l3_df, best_ell = _evaluate_l3_and_collect_risk_samples(rollouts, scaler_new, ELL_CANDIDATES)
    l3_df.to_csv(os.path.join(args.outDir, "l3_calibration_sweep.csv"), index=False)

    # (c) Calibrate asymmetric U_preq risk parameters
    upreq_df, cal_cer_cfg = _calibrate_upreq_thresholds(rollouts, scaler_new, best_ell)
    upreq_df.to_csv(os.path.join(args.outDir, "upreq_calibration_grid.csv"), index=False)

    cer_cfg_path = os.path.join(args.outDir, "cer_config_v2_calibrated.json")
    with open(cer_cfg_path, "w", encoding="utf-8") as cf:
        json.dump(cal_cer_cfg, cf, indent=2)
        cf.write("\n")

    summary = {
        "schema": "fase3_calibration_summary_1.0",
        "seeds": seeds,
        "sim_time_s": float(args.simTime),
        "slice_lambdas": args.sliceLambdas,
        "n_rollouts": len(rollouts),
        "n_cal_samples": int(raw_vecs.shape[0]),
        "scaler_manifest_path": scaler_path,
        "selected_action_length_scales": [best_ell[0], best_ell[1], best_ell[2]],
        "cer_config_calibrated": cal_cer_cfg,
        "cer_config_path": cer_cfg_path,
    }
    with open(os.path.join(args.outDir, "calibration_summary.json"), "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)
        sf.write("\n")

    print("\n=== FeatureScaler Refit (p1/p99) ===")
    print(pd.DataFrame(scaler_cmp_rows).to_string(index=False))
    print("\n=== L3 Anisotropic Action Length-Scale Sweep ===")
    print(l3_df.to_string(index=False))
    print(f"Selected L3 ell_a = ({best_ell[0]}, {best_ell[1]}, {best_ell[2]})")
    print("\n=== U_preq Asymmetric Risk Calibration ===")
    print(upreq_df.to_string(index=False))
    print("Calibrated CER config:", json.dumps(cal_cer_cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
