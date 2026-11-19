#!/usr/bin/env python3
"""Phase 3 Step 3: Closed-loop Fase B evaluation (MAIN + OOS) & sanitized radius sweep.

Compares SQR-CER against Priority-SQR, SQR (gamma=1), SQR (gamma=0), KBRL,
Equal, and Prop in full closed-loop over IPC Handshake v2 on reserved
confirmatory seeds MAIN {3001..3010} and OOS {3101..3105} (authorized for Fase B),
plus a sanitized multi-radius sweep {300, 600, 1000, 1500, 2000} m on dev seeds
{101, 102, 103} without BUG-01 and with Patch 1 restored.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
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

for p in (SRC_DIR, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from nsoran.sqr_v2.campaign_analysis_v2 import compute_seed_scalars_v2, load_prb_windows_df
from nsoran.sqr_v2.manifest_v2 import validate_dev_seed
import sqr_v2_run

DEFAULT_CAL_SUMMARY = os.path.join(
    GYM_ROOT, "results", "sqr_v2", "fase3_calibration", "calibration_summary.json"
)
DEFAULT_OUT_DIR = os.path.join(GYM_ROOT, "results", "sqr_v2", "fase3_closed_loop")

MAIN_SEEDS = tuple(range(3001, 3011))
OOS_SEEDS = tuple(range(3101, 3106))
RADII_M = (300, 600, 1000, 1500, 2000)
DEV_RAD_SEEDS = (101, 102, 103)


def _method_cli_flags(method: str, cal: dict[str, Any]) -> list[str]:
    scaler_path = str(cal["scaler_manifest_path"])
    cer_cfg_path = str(cal["cer_config_path"])
    l3_str = ",".join(str(x) for x in cal["selected_action_length_scales"])
    base = [
        "--het",
        "--het-scaler-manifest",
        scaler_path,
    ]
    if method == "sqr_cer":
        return base + [
            "--controller", "sqr",
            "--gamma", "1",
            "--reallocator", "sqr_cer",
            "--empty-slice-guard",
            "--urllc-delayed-update",
            "--action-length-scales", l3_str,
            "--cer-dl-slots", "80",
            "--cer-config-json", cer_cfg_path,
            "--learning-ablation", "L3_anisotropic_upreq_cer",
        ]
    if method == "priority_sqr":
        return base + [
            "--controller", "sqr",
            "--gamma", "1",
            "--reallocator", "priority_hoashi",
            "--empty-slice-guard",
            "--urllc-delayed-update",
            "--priority-weights", "1,1000000,1",
        ]
    if method == "sqr_g1":
        return base + [
            "--controller", "sqr",
            "--gamma", "1",
            "--reallocator", "none",
            "--empty-slice-guard",
            "--urllc-delayed-update",
        ]
    if method == "sqr_g0":
        return base + [
            "--controller", "sqr",
            "--gamma", "0",
            "--reallocator", "none",
            "--empty-slice-guard",
            "--urllc-delayed-update",
        ]
    if method == "kbrl":
        return base + [
            "--controller", "kbrl",
            "--gamma", "1",
            "--reallocator", "none",
            "--urllc-delayed-update",
        ]
    if method == "equal":
        return base + ["--controller", "equal", "--gamma", "1", "--reallocator", "none"]
    if method == "prop":
        return base + ["--controller", "prop", "--gamma", "1", "--reallocator", "none"]
    raise ValueError(f"Unknown method: {method!r}")


def _run_one_het_job(job: dict[str, Any]) -> dict[str, Any]:
    split = job["split"]
    method = job["method"]
    seed = int(job["seed"])
    out_dir = os.path.join(job["work_root"], f"{split}_{method}_s{seed}")
    os.makedirs(out_dir, exist_ok=True)

    argv = _method_cli_flags(method, job["cal"]) + [
        "--seed", str(seed),
        "--simTime", str(job["sim_time_s"]),
        "--warmup_s", str(job["warmup_s"]),
        "--cellRadius", "300",
        "--sliceLambdas", str(job["slice_lambdas"]),
        "--sliceApps", "greedy,urllc,mmtc",
        "--ipc-mode", "strict_rt",
        "--ipc-timeout-ms", "50",
        "--checkpoint_interval", "0",
        "--phy-qos-profile", "P1_tdd5ms_etapa3a_3b",
        "--allow-reserved-seed",
        "--output", out_dir,
        "--run-id", f"fase3_{split}_{method}_s{seed}",
    ]
    rc = sqr_v2_run.main(argv)
    if rc != 0:
        raise RuntimeError(f"Run failed for {split} {method} seed={seed}")

    with open(os.path.join(out_dir, "run_done.json"), "r", encoding="utf-8") as df:
        done = json.load(df)

    prb_win = load_prb_windows_df(out_dir)
    df_step = pd.read_csv(os.path.join(out_dir, "step_metrics.csv"))
    scalars = compute_seed_scalars_v2(df_step, {0: 0.5, 1: 1.0, 2: 1.5}, 100, prb_win)

    het_df = pd.read_csv(os.path.join(out_dir, "het_window_metrics.csv"))
    post = het_df[het_df["warmup"] == 0].copy() if (het_df["warmup"] == 0).any() else het_df.copy()
    post_slim = post[
        ["window_index", "cer_stage", "a_nom_0", "a_nom_1", "a_nom_2",
         "a_exec_0", "a_exec_1", "a_exec_2", "sla_ok_all3", "sla_ok_e", "sla_ok_u", "sla_ok_m"]
    ].copy()

    # Keep only compact files; remove raw simulation scratch dirs before returning
    for sub in ("results_sqr_network_only",):
        p = os.path.join(out_dir, sub)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)

    ep = done["episode_summary"]
    hs = done["het_summary"]
    return {
        "split": split,
        "method": method,
        "seed": seed,
        "ns3_exitcode": int(done["ns3_exitcode"]),
        "steps": int(ep["steps"]),
        "applied_steps": int(ep["applied_steps"]),
        "timeout_hold_steps": int(ep["timeout_hold_steps"]),
        "window_gap_steps": int(ep["window_gap_steps"]),
        "rb_win_conservation_abs_err": float(scalars["rb_win_conservation_abs_err"]),
        "sla_viol_all3_pct": float(hs["sla_viol_all3_pct"]),
        "sla_viol_embb_pct": float(hs["sla_viol_embb_pct"]),
        "sla_viol_urllc_pct": float(hs["sla_viol_urllc_pct"]),
        "sla_viol_mmtc_pct": float(hs["sla_viol_mmtc_pct"]),
        "embb_thr_mean_mbps": hs["embb_thr_mean_mbps"],
        "urllc_p95_ms_mean": hs["urllc_p95_ms_mean"],
        "urllc_p95_ms_p95": hs["urllc_p95_ms_p95"],
        "urllc_miss_pct_mean": hs["urllc_miss_pct_mean"],
        "mmtc_pdr_mean": hs["mmtc_pdr_mean"],
        "prb_util_pct": hs["prb_util_pct"],
        "agent_ms_p50": float(hs.get("post_warmup_agent_ms_p50", ep["agent_decision_ms_p50"])),
        "agent_ms_p95": float(hs.get("post_warmup_agent_ms_p95", ep["agent_decision_ms_p95"])),
        "cer_ms_p50": float(ep.get("cer_reallocate_ms_p50", 0.0)),
        "cer_ms_p95": float(ep.get("cer_reallocate_ms_p95", 0.0)),
        "all_steps_agent_ms_p50": float(ep["agent_decision_ms_p50"]),
        "all_steps_agent_ms_p95": float(ep["agent_decision_ms_p95"]),
        "cer_interventions": int(hs.get("cer_interventions", 0)),
        "post_windows": post_slim.to_dict(orient="list"),
    }


def _run_one_radius_job(job: dict[str, Any]) -> dict[str, Any]:
    radius_m = int(job["radius_m"])
    seed = validate_dev_seed(int(job["seed"]), allow_reserved=False)
    profile = str(job["profile"])
    out_dir = os.path.join(job["work_root"], f"rad_{profile}_r{radius_m}_s{seed}")
    os.makedirs(out_dir, exist_ok=True)

    if profile == "embb_article_05_10_15":
        argv = [
            "--controller", "sqr",
            "--gamma", "1",
            "--slas", "0.5,1.0,1.5",
            "--seed", str(seed),
            "--simTime", str(job["sim_time_s"]),
            "--warmup_s", "10.0",
            "--cellRadius", str(radius_m),
            "--sliceLambdas", str(job["slice_lambdas"]),
            "--ipc-mode", "strict_rt",
            "--ipc-timeout-ms", "50",
            "--checkpoint_interval", "0",
            "--phy-qos-profile", "P1_tdd5ms_etapa3a_3b",
            "--output", out_dir,
            "--run-id", f"rad_{profile}_r{radius_m}_s{seed}",
        ]
    else:
        argv = _method_cli_flags("sqr_cer", job["cal"]) + [
            "--seed", str(seed),
            "--simTime", str(job["sim_time_s"]),
            "--warmup_s", "15.0",
            "--cellRadius", str(radius_m),
            "--sliceLambdas", str(job["slice_lambdas"]),
            "--sliceApps", "greedy,urllc,mmtc",
            "--ipc-mode", "strict_rt",
            "--ipc-timeout-ms", "50",
            "--checkpoint_interval", "0",
            "--phy-qos-profile", "P1_tdd5ms_etapa3a_3b",
            "--output", out_dir,
            "--run-id", f"rad_{profile}_r{radius_m}_s{seed}",
        ]

    rc = sqr_v2_run.main(argv)
    if rc != 0:
        raise RuntimeError(f"Radius job failed: {profile} r={radius_m} s={seed}")

    with open(os.path.join(out_dir, "run_done.json"), "r", encoding="utf-8") as df:
        done = json.load(df)
    df_step = pd.read_csv(os.path.join(out_dir, "step_metrics.csv"))
    prb_win = load_prb_windows_df(out_dir)
    scalars = compute_seed_scalars_v2(df_step, {0: 0.5, 1: 1.0, 2: 1.5}, 100, prb_win)

    shutil.rmtree(out_dir, ignore_errors=True)
    if profile == "embb_article_05_10_15":
        post_df = df_step[df_step["warmup"] == 0] if "warmup" in df_step.columns else df_step
        cap_m = float(scalars.get("rb_slot_capacity_c_mean", 53.0))
        ex_m = float(scalars.get("rb_slot_executed_cell_mean", np.nan))
        return {
            "profile": profile,
            "radius_m": radius_m,
            "seed": seed,
            "ns3_exitcode": int(done["ns3_exitcode"]),
            "sla_viol_all_pct": float(scalars["slice_time_violation_pct"]),
            "thr_p5_mean_mbps": float(post_df["y_t_p5_mbps"].dropna().mean()) if post_df["y_t_p5_mbps"].notna().any() else np.nan,
            "rb_util_pct": 100.0 * ex_m / max(1.0, cap_m) if np.isfinite(ex_m) else np.nan,
            "cons_err": float(scalars["rb_win_conservation_abs_err"]),
            "urllc_p95_ms": np.nan,
        }
    hs = done["het_summary"]
    return {
        "profile": profile,
        "radius_m": radius_m,
        "seed": seed,
        "ns3_exitcode": int(done["ns3_exitcode"]),
        "sla_viol_all_pct": float(hs["sla_viol_all3_pct"]),
        "thr_p5_mean_mbps": float(hs["embb_thr_mean_mbps"]) if hs["embb_thr_mean_mbps"] is not None else np.nan,
        "rb_util_pct": float(hs["prb_util_pct"]) if hs["prb_util_pct"] is not None else np.nan,
        "cons_err": float(scalars["rb_win_conservation_abs_err"]),
        "urllc_p95_ms": float(hs["urllc_p95_ms_mean"]) if hs["urllc_p95_ms_mean"] is not None else np.nan,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 3 Step 3: Fase B Closed-Loop & Radius Sweep")
    p.add_argument("--calSummary", type=str, default=DEFAULT_CAL_SUMMARY)
    p.add_argument("--outDir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--simTime", type=float, default=120.0)
    p.add_argument("--warmup_s", type=float, default=30.0)
    p.add_argument("--sliceLambdas", type=str, default="5,5,5")
    p.add_argument("--radSimTime", type=float, default=60.0)
    p.add_argument("--skip-radius-sweep", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)

    with open(args.calSummary, "r", encoding="utf-8") as cf:
        cal = json.load(cf)

    os.makedirs(args.outDir, exist_ok=True)
    work_root = tempfile.mkdtemp(prefix="fase3_eval_runs_", dir="/tmp")

    methods = ("sqr_cer", "priority_sqr", "sqr_g1", "sqr_g0", "kbrl", "equal", "prop")
    het_jobs: list[dict[str, Any]] = []
    for split, seeds in (("MAIN", MAIN_SEEDS), ("OOS", OOS_SEEDS)):
        for seed in seeds:
            for method in methods:
                het_jobs.append({
                    "split": split,
                    "method": method,
                    "seed": seed,
                    "sim_time_s": float(args.simTime),
                    "warmup_s": float(args.warmup_s),
                    "slice_lambdas": args.sliceLambdas,
                    "cal": cal,
                    "work_root": work_root,
                })

    het_results: list[dict[str, Any]] = []
    rad_raw_csv = os.path.join(args.outDir, "fase3_radius_sweep_raw.csv")
    rad_jobs: list[dict[str, Any]] = []
    if not (args.skip_radius_sweep and os.path.isfile(rad_raw_csv)):
        for profile in ("embb_article_05_10_15", "het_sqr_cer"):
            for r_m in RADII_M:
                for s in DEV_RAD_SEEDS:
                    rad_jobs.append({
                        "profile": profile,
                        "radius_m": r_m,
                        "seed": s,
                        "sim_time_s": float(args.radSimTime),
                        "slice_lambdas": args.sliceLambdas,
                        "cal": cal,
                        "work_root": work_root,
                    })

    rad_results: list[dict[str, Any]] = []
    try:
        print(f"[Fase3] Launching {len(het_jobs)} Fase B closed-loop episodes ({args.workers} workers)...")
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futs = [pool.submit(_run_one_het_job, j) for j in het_jobs]
            done_cnt = 0
            for fut in as_completed(futs):
                res = fut.result()
                het_results.append(res)
                done_cnt += 1
                print(
                    f"[Fase3 {done_cnt}/{len(het_jobs)}] {res['split']} {res['method']:12s} s={res['seed']} | "
                    f"viol={res['sla_viol_all3_pct']:5.2f}% | u_p95={res['urllc_p95_ms_mean'] or 0:6.2f}ms | "
                    f"e_thr={res['embb_thr_mean_mbps'] or 0:5.2f} | ms_p95={res['agent_ms_p95']:.3f}ms | "
                    f"holds={res['timeout_hold_steps']} | cons={res['rb_win_conservation_abs_err']:.1f}",
                    flush=True,
                )

        if rad_jobs:
            print(f"[Fase3] Launching {len(rad_jobs)} sanitized radius sweep runs ({args.workers} workers)...")
            with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
                futs = [pool.submit(_run_one_radius_job, j) for j in rad_jobs]
                for fut in as_completed(futs):
                    rad_results.append(fut.result())
        else:
            print(f"[Fase3] Reusing existing radius sweep results from {rad_raw_csv}")
            rad_results = pd.read_csv(rad_raw_csv).to_dict(orient="records")
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    # Save per-seed summary without bulky per-window dicts
    seed_rows = [{k: v for k, v in r.items() if k != "post_windows"} for r in het_results]
    df_seed = pd.DataFrame(seed_rows).sort_values(["split", "method", "seed"])
    df_seed.to_csv(os.path.join(args.outDir, "fase3_per_seed_summary.csv"), index=False)

    # Save slim per-window paired log for future auditability without retaining sim_runs
    post_win_path = os.path.join(args.outDir, "fase3_post_windows.jsonl")
    with open(post_win_path, "w", encoding="utf-8") as pwf:
        for r in het_results:
            pwf.write(
                json.dumps({
                    "split": r["split"],
                    "method": r["method"],
                    "seed": int(r["seed"]),
                    "post_windows": r["post_windows"],
                })
                + "\n"
            )

    # Aggregate by (split, method)
    agg_cols = [
        "sla_viol_all3_pct", "sla_viol_embb_pct", "sla_viol_urllc_pct", "sla_viol_mmtc_pct",
        "embb_thr_mean_mbps", "urllc_p95_ms_mean", "urllc_p95_ms_p95", "urllc_miss_pct_mean",
        "mmtc_pdr_mean", "prb_util_pct", "agent_ms_p50", "agent_ms_p95",
        "cer_ms_p50", "cer_ms_p95",
        "timeout_hold_steps", "window_gap_steps", "rb_win_conservation_abs_err", "cer_interventions",
    ]
    df_method = df_seed.groupby(["split", "method"], as_index=False)[agg_cols].agg(["mean", "std"])
    df_method.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0] for c in df_method.columns.to_flat_index()
    ]
    df_method.to_csv(os.path.join(args.outDir, "fase3_method_summary.csv"), index=False)

    # Paired RESCUE vs HARM against sqr_g1 on matched (split, seed, window_index)
    by_key = {(r["split"], r["seed"], r["method"]): pd.DataFrame(r["post_windows"]) for r in het_results}
    rh_rows: list[dict[str, Any]] = []
    for split, seeds in (("MAIN", MAIN_SEEDS), ("OOS", OOS_SEEDS)):
        for cand_m in ("sqr_cer", "priority_sqr", "sqr_g0", "kbrl", "equal", "prop"):
            n_paired = 0
            n_rescue_all = 0
            n_harm_all = 0
            n_intervened = 0
            n_rescue_int = 0
            n_harm_int = 0
            for s in seeds:
                df_b = by_key[(split, s, "sqr_g1")].drop_duplicates(subset=["window_index"]).set_index("window_index")
                df_c = by_key[(split, s, cand_m)].drop_duplicates(subset=["window_index"]).set_index("window_index")
                common = df_b.index.intersection(df_c.index)
                if len(common) == 0:
                    continue
                b_ok = df_b.loc[common, "sla_ok_all3"].to_numpy()
                c_ok = df_c.loc[common, "sla_ok_all3"].to_numpy()
                n_paired += int(len(common))
                rescue_mask = (b_ok == 0) & (c_ok == 1)
                harm_mask = (b_ok == 1) & (c_ok == 0)
                n_rescue_all += int(np.sum(rescue_mask))
                n_harm_all += int(np.sum(harm_mask))
                if cand_m == "sqr_cer":
                    st_arr = df_c.loc[common, "cer_stage"].fillna("").astype(str).to_numpy()
                    int_mask = st_arr != "MANTER_B0"
                else:
                    int_mask = np.ones_like(rescue_mask, dtype=bool)
                n_intervened += int(np.sum(int_mask))
                n_rescue_int += int(np.sum(rescue_mask & int_mask))
                n_harm_int += int(np.sum(harm_mask & int_mask))

            rh_rows.append({
                "split": split,
                "method_vs_sqr_g1": cand_m,
                "n_paired_windows": n_paired,
                "rescue_all": n_rescue_all,
                "harm_all": n_harm_all,
                "rescue_rate_all_pct": round(100.0 * n_rescue_all / max(1, n_paired), 3),
                "harm_rate_all_pct": round(100.0 * n_harm_all / max(1, n_paired), 3),
                "net_gain_pp": round(100.0 * (n_rescue_all - n_harm_all) / max(1, n_paired), 3),
                "n_intervened_windows": n_intervened,
                "rescue_intervened": n_rescue_int,
                "harm_intervened": n_harm_int,
                "rescue_vs_harm_win_ratio": round(n_rescue_int / max(1, n_rescue_int + n_harm_int), 4),
            })
    df_rh = pd.DataFrame(rh_rows)
    df_rh.to_csv(os.path.join(args.outDir, "fase3_rescue_harm_summary.csv"), index=False)

    # Save radius sweep results
    df_rad = pd.DataFrame(rad_results).sort_values(["profile", "radius_m", "seed"])
    df_rad.to_csv(os.path.join(args.outDir, "fase3_radius_sweep_raw.csv"), index=False)
    df_rad_agg = df_rad.groupby(["profile", "radius_m"], as_index=False)[
        ["sla_viol_all_pct", "thr_p5_mean_mbps", "rb_util_pct", "cons_err", "urllc_p95_ms"]
    ].mean()
    df_rad_agg.to_csv(os.path.join(args.outDir, "fase3_radius_sweep_summary.csv"), index=False)

    summary_payload = {
        "schema": "fase3_closed_loop_eval_1.0",
        "calibration_summary": args.calSummary,
        "sim_time_s": float(args.simTime),
        "warmup_s": float(args.warmup_s),
        "slice_lambdas": args.sliceLambdas,
        "main_seeds": list(MAIN_SEEDS),
        "oos_seeds": list(OOS_SEEDS),
        "n_het_runs": len(het_results),
        "n_rad_runs": len(rad_results),
        "method_summary": df_method.to_dict(orient="records"),
        "rescue_harm_summary": rh_rows,
        "radius_sweep_summary": df_rad_agg.to_dict(orient="records"),
    }
    with open(os.path.join(args.outDir, "fase3_summary.json"), "w", encoding="utf-8") as sf:
        json.dump(summary_payload, sf, indent=2)
        sf.write("\n")

    print("\n=== FASE B METHOD SUMMARY ===")
    print(df_method.to_string(index=False))
    print("\n=== PAIRED RESCUE vs HARM (vs SQR gamma=1) ===")
    print(df_rh.to_string(index=False))
    print("\n=== SANITIZED MULTI-RADIUS SWEEP (NO BUG-01, PATCH 1 RESTORED) ===")
    print(df_rad_agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
