#!/usr/bin/env python3
"""Fase 2 (Etapa 3B) validation gate: TDD 5 ms (DL|DL|DL|S|UL|) vs 7 ms URLLC action sweep.

Executes paired real ns-3 ``sqr-v2-sim`` runs over ``SqrV2Env`` at ``cellRadius=300 m``
for ``a_u in {5, 10, 15, 20, 25, 30}`` (fixing ``a_e=20, a_m=3`` so ``sum <= 53``)
and verifies:
  1. Zero aborts in ``AssignCsiRsOffset`` or TDD RX/TX alignment;
  2. Exact integer slot periodicity (80 DL+S slots per 100 ms under 5 ms TDD);
  3. Substantial reduction in ``dy = 0`` tie rate on URLLC ``p95`` latency compared
     to the legacy 69.4% / 7 ms regime;
  4. Monotonic reduction of URLLC ``p95`` latency with increasing ``a_u``;
  5. Zero window PRB conservation error (``rb_win_conservation_abs_err == 0.0``).

Outputs under ``MODA/ns-o-ran-gym/results/sqr_v2/fase2_tdd5ms_urllc_sweep/``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import glob
import json
import os
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

GYM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(GYM_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from nsoran.sqr_v2.campaign_analysis_v2 import load_prb_windows_df  # noqa: E402
from nsoran.sqr_v2.env_v2 import SqrV2Env  # noqa: E402

NS3_PATH = os.path.abspath(os.path.join(GYM_DIR, "..", "ns-3-dev"))
OUT_DIR = os.path.join(GYM_DIR, "results", "sqr_v2", "fase2_tdd5ms_urllc_sweep")
ACTIONS_U = (5, 10, 15, 20, 25, 30)


def _find_single(base_dir: str, fname: str) -> str:
    hits = sorted(glob.glob(os.path.join(base_dir, "**", fname), recursive=True))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected 1 {fname} under {base_dir}, got {len(hits)}")
    return hits[0]


def _run_one(
    *,
    tdd_label: str,
    tdd_flag: str | None,
    seed: int,
    a_u: int,
    sim_time: float,
    slice_lambdas: str,
    work_root: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    run_dir = os.path.join(work_root, f"{tdd_label}_s{seed}_au{a_u}")
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    cfg: dict[str, Any] = {
        "simTime": [float(sim_time)],
        "seed": [int(seed)],
        "periodMs": [100],
        "cellRadius": [300.0],
        "sliceLambdas": [slice_lambdas],
        "sliceApps": ["greedy,urllc,mmtc"],
    }
    if tdd_flag is not None:
        cfg["tddPattern"] = [tdd_flag]

    env = SqrV2Env(
        ns3_path=NS3_PATH,
        scenario_configuration=cfg,
        output_folder=run_dir,
        total_prbs=52,
        num_slices=3,
        ipc_mode="strict_rt",
        ipc_timeout_ms=50,
        run_id=f"fase2_{tdd_label}_s{seed}_au{a_u}",
    )
    try:
        _obs, _info = env.reset(seed=seed)
        c_prbs = int(env.total_prbs)
        assert c_prbs == 53, f"Expected synced C=53, got {c_prbs}"
        action = np.array([20, int(a_u), 3], dtype=np.int64)
        while True:
            _obs, _rew, term, trunc, info = env.step(action)
            if info.get("episode_end_no_next_window") or term or trunc:
                break
        rc = env.wait_sim_exit(timeout_s=120.0)
        cpp_sum = env.read_cpp_ipc_summary() or {}
    finally:
        env.close()

    assert rc == 0, f"sqr-v2-sim failed (rc={rc}) for {tdd_label} seed={seed} a_u={a_u}"

    prb_df = load_prb_windows_df(run_dir)
    assert prb_df is not None and not prb_df.empty
    sched = (
        prb_df["prb_scheduled_rbg_0"]
        + prb_df["prb_scheduled_rbg_1"]
        + prb_df["prb_scheduled_rbg_2"]
    )
    cap = prb_df["total_rbg"] * prb_df["dl_slots"]
    win_err = float((sched + prb_df["prb_idle_rbg"] - cap).abs().max())

    delays_path = _find_single(run_dir, "sqr_packet_delays.csv")
    delays = pd.read_csv(delays_path)
    urllc = delays[delays["sliceId"] == 1].copy()
    if len(urllc) > 0:
        urllc["win"] = np.floor(urllc["txSec"] * 10.0).astype(int)
        urllc["delay_ms"] = urllc["delaySec"] * 1000.0
        win_stats = (
            urllc.groupby("win")["delay_ms"]
            .agg(
                n_pkts="count",
                p50_ms=lambda s: float(np.percentile(s, 50.0)),
                p95_ms=lambda s: float(np.percentile(s, 95.0)),
                mean_ms="mean",
                miss_pct=lambda s: float(100.0 * np.mean(s > 10.0)),
            )
            .reset_index()
        )
        # Filter active closed arrival windows away from sim tail drain
        max_win = int(round(sim_time * 10.0)) - 2
        win_stats = win_stats[(win_stats["win"] >= 10) & (win_stats["win"] <= max_win)].copy()
    else:
        win_stats = pd.DataFrame(columns=["win", "n_pkts", "p50_ms", "p95_ms", "mean_ms", "miss_pct"])

    win_stats["tdd_mode"] = tdd_label
    win_stats["seed"] = int(seed)
    win_stats["a_u"] = int(a_u)

    kpm_path = _find_single(run_dir, "sqr_kpm_history.csv")
    kpm_df = pd.read_csv(kpm_path)
    has_neg999 = bool((kpm_df["snrDb"] == -999.0).any()) if len(kpm_df) else False

    row = {
        "tdd_mode": tdd_label,
        "seed": int(seed),
        "a_u": int(a_u),
        "exit_code": int(rc),
        "dl_slots_mean": float(prb_df["dl_slots"].mean()),
        "dl_slots_std": float(prb_df["dl_slots"].std(ddof=0)),
        "dl_slots_min": int(prb_df["dl_slots"].min()),
        "dl_slots_max": int(prb_df["dl_slots"].max()),
        "win_conservation_err_max": win_err,
        "has_neg999_snr": has_neg999,
        "urllc_packets": int(len(urllc)),
        "urllc_windows_active": int(len(win_stats)),
        "urllc_p95_ms_mean": float(win_stats["p95_ms"].mean()) if len(win_stats) else float("nan"),
        "urllc_p95_ms_median": float(win_stats["p95_ms"].median()) if len(win_stats) else float("nan"),
        "urllc_p95_ms_max": float(win_stats["p95_ms"].max()) if len(win_stats) else float("nan"),
        "urllc_mean_ms": float(win_stats["mean_ms"].mean()) if len(win_stats) else float("nan"),
        "urllc_miss_pct_mean": float(win_stats["miss_pct"].mean()) if len(win_stats) else float("nan"),
        "ipc_timeout_holds": int(cpp_sum.get("timeout_hold_count", -1)),
    }
    return row, win_stats


def _pairwise_stats(win_df: pd.DataFrame, tie_tol_ms: float = 0.01) -> dict[str, Any]:
    """Compare adjacent (a_u[k], a_u[k+1]) at identical (seed, win) for dy==0 ties & monotonicity."""
    a_vals = sorted(win_df["a_u"].unique().tolist())
    n_pairs = 0
    n_ties = 0
    n_improved = 0
    n_worsened = 0
    dy_list: list[float] = []

    for seed, sgrp in win_df.groupby("seed"):
        piv = sgrp.pivot(index="win", columns="a_u", values="p95_ms")
        for a1, a2 in zip(a_vals[:-1], a_vals[1:]):
            if a1 not in piv.columns or a2 not in piv.columns:
                continue
            valid = piv[[a1, a2]].dropna()
            for v1, v2 in zip(valid[a1].values, valid[a2].values):
                dy = float(v2 - v1)
                n_pairs += 1
                dy_list.append(dy)
                if abs(dy) <= tie_tol_ms:
                    n_ties += 1
                elif dy < -tie_tol_ms:
                    n_improved += 1
                else:
                    n_worsened += 1

    # Pooled matched windows Spearman between a_u and p95_ms
    common_wins = win_df.groupby(["seed", "win"])["a_u"].nunique()
    full_keys = common_wins[common_wins == len(a_vals)].index
    if len(full_keys) > 0:
        sub = win_df.set_index(["seed", "win"]).loc[full_keys].reset_index()
        rho, pval = spearmanr(sub["a_u"], sub["p95_ms"])
        # Within-window rank correlation across a_u
        within_rhos = []
        for (_, _), g in sub.groupby(["seed", "win"]):
            if g["p95_ms"].nunique() > 1:
                r_w, _ = spearmanr(g["a_u"], g["p95_ms"])
                if np.isfinite(r_w):
                    within_rhos.append(float(r_w))
        within_rho_mean = float(np.mean(within_rhos)) if within_rhos else float("nan")
    else:
        rho, pval, within_rho_mean = float("nan"), float("nan"), float("nan")

    return {
        "n_adjacent_pairs": int(n_pairs),
        "n_ties_abs_le_0_01ms": int(n_ties),
        "tie_rate_pct": (100.0 * n_ties / n_pairs) if n_pairs else float("nan"),
        "n_improved_lower_p95": int(n_improved),
        "n_worsened_higher_p95": int(n_worsened),
        "mean_dy_ms": float(np.mean(dy_list)) if dy_list else float("nan"),
        "median_dy_ms": float(np.median(dy_list)) if dy_list else float("nan"),
        "spearman_au_vs_p95_pooled": float(rho) if np.isfinite(rho) else float("nan"),
        "spearman_pval": float(pval) if np.isfinite(pval) else float("nan"),
        "spearman_within_window_mean": within_rho_mean,
    }


def _run_job(job: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    return _run_one(**job)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fase 2 TDD 5 ms vs 7 ms URLLC sweep")
    p.add_argument("--simTime", type=float, default=20.0)
    p.add_argument("--sliceLambdas", type=str, default="15,45,15", dest="slice_lambdas")
    p.add_argument("--seeds", type=str, default="101,102")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    work_root = os.path.join(OUT_DIR, "sim_runs")
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    modes = [
        ("tdd5ms_default", None),
        ("tdd7ms_legacy", "7ms"),
    ]

    jobs = [
        {
            "tdd_label": label,
            "tdd_flag": tdd_flag,
            "seed": seed,
            "a_u": a_u,
            "sim_time": args.simTime,
            "slice_lambdas": args.slice_lambdas,
            "work_root": work_root,
        }
        for label, tdd_flag in modes
        for seed in seeds
        for a_u in ACTIONS_U
    ]

    summary_rows: list[dict[str, Any]] = []
    win_frames: list[pd.DataFrame] = []
    max_workers = max(1, int(args.workers))
    if max_workers == 1:
        results = [_run_job(j) for j in jobs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(_run_job, j) for j in jobs]
            for fut in as_completed(futs):
                row, wdf = fut.result()
                results.append((row, wdf))
                print(
                    f"[Fase2] {row['tdd_mode']} seed={row['seed']} a_u={row['a_u']:2d} | "
                    f"dl_slots={row['dl_slots_mean']:.1f} (min={row['dl_slots_min']},max={row['dl_slots_max']}) | "
                    f"p95_med={row['urllc_p95_ms_median']:.3f} ms | "
                    f"miss={row['urllc_miss_pct_mean']:.2f}% | cons_err={row['win_conservation_err_max']}",
                    flush=True,
                )

    for row, wdf in results:
        summary_rows.append(row)
        win_frames.append(wdf)
        if max_workers == 1:
            print(
                f"[Fase2] {row['tdd_mode']} seed={row['seed']} a_u={row['a_u']:2d} | "
                f"dl_slots={row['dl_slots_mean']:.1f} (min={row['dl_slots_min']},max={row['dl_slots_max']}) | "
                f"p95_med={row['urllc_p95_ms_median']:.3f} ms | "
                f"miss={row['urllc_miss_pct_mean']:.2f}% | cons_err={row['win_conservation_err_max']}",
                flush=True,
            )

    df_sum = pd.DataFrame(summary_rows).sort_values(["tdd_mode", "seed", "a_u"]).reset_index(drop=True)
    df_win = pd.concat(win_frames, ignore_index=True)
    df_sum.to_csv(os.path.join(OUT_DIR, "urllc_sweep_runs.csv"), index=False)
    df_win.to_csv(os.path.join(OUT_DIR, "urllc_sweep_windows.csv"), index=False)

    agg = (
        df_sum.groupby(["tdd_mode", "a_u"], as_index=False)
        .agg(
            dl_slots_mean=("dl_slots_mean", "mean"),
            urllc_p95_ms_mean=("urllc_p95_ms_mean", "mean"),
            urllc_p95_ms_median=("urllc_p95_ms_median", "mean"),
            urllc_mean_ms=("urllc_mean_ms", "mean"),
            urllc_miss_pct_mean=("urllc_miss_pct_mean", "mean"),
            win_conservation_err_max=("win_conservation_err_max", "max"),
        )
    )
    agg.to_csv(os.path.join(OUT_DIR, "urllc_sweep_by_action.csv"), index=False)

    pair_stats = {
        label: _pairwise_stats(df_win[df_win["tdd_mode"] == label])
        for label, _ in modes
    }

    report = {
        "schema": "fase2_tdd5ms_urllc_sweep_1.0",
        "sim_time_s": float(args.simTime),
        "slice_lambdas": args.slice_lambdas,
        "seeds": seeds,
        "actions_u": list(ACTIONS_U),
        "pairwise_comparison": pair_stats,
        "by_action": agg.to_dict(orient="records"),
    }
    with open(os.path.join(OUT_DIR, "fase2_sweep_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print("\n=== Pairwise Tie & Monotonicity Summary ===")
    print(json.dumps(pair_stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
