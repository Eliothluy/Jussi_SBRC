#!/usr/bin/env python3
"""Auditoria matematica 3GPP (TS 38.306, TS 38.104, TS 38.214, TS 38.331, TR 38.901)
sobre os resultados reais da campanha Fase B (results/sqr_v2/fase3_closed_loop/).

Verifica sem alterar o simulador ou o baseline congelado B0/v1:
  1. Teto teorico de vazao DL da 3GPP TS 38.306 sec. 4.1.2 (N_PRB=52 e 53,
     DDDSU 5 ms) e cross-check exato MAC/PHY TBS do 5G-LENA (NrLteMiErrorModel),
     mais conservacao de espectro em 100% das janelas, episodios e medias.
  2. Orcamento de enlace 3GPP TR 38.901 UMa (300 m .. 2000 m) frente aos limiares
     RLF Q_out = -5.0 dB / Q_in = -3.9 dB e ao sweep real multi-raio.
  3. Piso fisico deterministico de latencia E2E (backhaul P2P + alinhamento TDD
     DDDSU + pipeline L1/L2 + decodificacao TB) frente a urllc_p95_ms medido.

Saida JSON:
  MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/3gpp_physical_audit_summary.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
GYM_DIR = os.path.join(REPO_ROOT, "MODA", "ns-o-ran-gym")
SRC_DIR = os.path.join(GYM_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from nsoran.sqr_jc.yaml_subset import load as load_yaml  # noqa: E402

CLOSED_LOOP_DIR = os.path.join(GYM_DIR, "results", "sqr_v2", "fase3_closed_loop")
SCALER_PATH = os.path.join(
    GYM_DIR, "results", "sqr_v2", "fase3_calibration", "scaler_manifest_v2_tdd5ms.yaml"
)

METHOD_SUMMARY_CSV = os.path.join(CLOSED_LOOP_DIR, "fase3_method_summary.csv")
PER_SEED_CSV = os.path.join(CLOSED_LOOP_DIR, "fase3_per_seed_summary.csv")
POST_WINDOWS_JSONL = os.path.join(CLOSED_LOOP_DIR, "fase3_post_windows.jsonl")
RADIUS_SUMMARY_CSV = os.path.join(CLOSED_LOOP_DIR, "fase3_radius_sweep_summary.csv")
RADIUS_RAW_CSV = os.path.join(CLOSED_LOOP_DIR, "fase3_radius_sweep_raw.csv")
OUT_JSON = os.path.join(CLOSED_LOOP_DIR, "3gpp_physical_audit_summary.json")


def norm_cdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    """Standard/Gaussian CDF via math.erf."""
    z = (x - mean) / (std * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def ts38_306_dl_peak_mbps(
    *,
    n_prb: int,
    j: int = 1,
    v_layers: int = 1,
    qm: int = 6,
    f_scaling: float = 1.0,
    r_max: float = 948.0 / 1024.0,
    mu: int = 0,
    oh: float = 0.14,
    tdd_factor: float = 1.0,
) -> float:
    """Compute 3GPP TS 38.306 Sec. 4.1.2 max DL data rate (Mbps)."""
    t_s_mu = 1e-3 / (14.0 * (2**mu))
    fdd_mbps = (
        1e-6
        * j
        * v_layers
        * qm
        * f_scaling
        * r_max
        * ((n_prb * 12.0) / t_s_mu)
        * (1.0 - oh)
    )
    return float(fdd_mbps * tdd_factor)


def lena_5g_tb_bytes(
    *,
    n_prb: int,
    n_pdsch_sym: int,
    useful_sc_per_rb: int = 11,
    qm: int = 6,
    r_code: float = 0.92,
    rank: int = 1,
    k_cb_bytes: int = 768,
    crc_bytes: int = 3,
) -> int:
    """Replicate NrAmc::CalculateTbSize + NrLteMiErrorModel::GetPayloadSize."""
    rsc_element = useful_sc_per_rb * (n_prb * n_pdsch_sym)
    payload_bytes = int((rsc_element * qm * r_code * rank) / 8.0)
    tb_bytes = payload_bytes - crc_bytes if payload_bytes >= crc_bytes else payload_bytes
    if tb_bytes > k_cb_bytes:
        n_cb = math.ceil(float(tb_bytes) / float(k_cb_bytes))
        tb_bytes = payload_bytes - int(n_cb * crc_bytes)
    return int(tb_bytes)


def lena_5g_tdd_peak_mbps(
    *,
    n_prb: int,
    n_dl_slots: int = 3,
    n_s_slots: int = 1,
    dl_pdsch_sym: int = 13,
    s_pdsch_sym: int = 12,
    period_ms: float = 5.0,
) -> dict[str, Any]:
    """Compute exact 5G-LENA MAC/PHY ceiling under NrLteMiErrorModel MCS 28."""
    tb_dl = lena_5g_tb_bytes(n_prb=n_prb, n_pdsch_sym=dl_pdsch_sym)
    tb_s = lena_5g_tb_bytes(n_prb=n_prb, n_pdsch_sym=s_pdsch_sym)
    total_bytes = n_dl_slots * tb_dl + n_s_slots * tb_s
    mbps = (total_bytes * 8.0) / (period_ms * 1e-3) / 1e6
    return {
        "n_prb": int(n_prb),
        "tb_dl_slot_bytes": tb_dl,
        "tb_s_slot_bytes": tb_s,
        "total_bytes_per_period": total_bytes,
        "period_ms": period_ms,
        "peak_mbps": float(mbps),
    }


def uma_plos(d_2d: float, h_ut: float = 1.5) -> float:
    """3GPP TR 38.901 Table 7.4.2-1 UMa P_LOS (channel-condition-model.cc:692-714)."""
    if d_2d <= 18.0:
        return 1.0
    c = 0.0 if h_ut <= 13.0 else ((h_ut - 13.0) / 10.0) ** 1.5
    p_los = (18.0 / d_2d + math.exp(-d_2d / 63.0) * (1.0 - 18.0 / d_2d)) * (
        1.0 + c * 1.25 * ((d_2d / 100.0) ** 3.0) * math.exp(-d_2d / 150.0)
    )
    return float(min(1.0, max(0.0, p_los)))


def uma_pl_los_nlos_db(
    d_2d: float,
    *,
    f_c_hz: float = 2.0e9,
    h_bs: float = 25.0,
    h_ut: float = 1.5,
    h_e: float = 1.0,
) -> dict[str, float]:
    """3GPP TR 38.901 Table 7.4.1-1 UMa PL_LOS and PL_NLOS (deterministic median)."""
    c_light = 3.0e8
    f_ghz = f_c_hz / 1e9
    d_3d = math.hypot(d_2d, h_bs - h_ut)
    d_bp = 4.0 * (h_bs - h_e) * (h_ut - h_e) * f_c_hz / c_light
    if d_2d <= d_bp:
        pl_los = 28.0 + 22.0 * math.log10(d_3d) + 20.0 * math.log10(f_ghz)
    else:
        pl_los = (
            28.0
            + 40.0 * math.log10(d_3d)
            + 20.0 * math.log10(f_ghz)
            - 9.0 * math.log10((d_bp**2) + ((h_bs - h_ut) ** 2))
        )
    pl_nlos_raw = (
        13.54 + 39.08 * math.log10(d_3d) + 20.0 * math.log10(f_ghz) - 0.6 * (h_ut - 1.5)
    )
    pl_nlos = max(pl_los, pl_nlos_raw)
    return {
        "d_2d_m": float(d_2d),
        "d_3d_m": float(d_3d),
        "d_bp_m": float(d_bp),
        "p_los": uma_plos(d_2d, h_ut),
        "pl_los_db": float(pl_los),
        "pl_nlos_raw_db": float(pl_nlos_raw),
        "pl_nlos_db": float(pl_nlos),
    }


def cosine_antenna_gain_db(
    d_2d: float,
    *,
    h_bs: float = 25.0,
    h_ut: float = 1.5,
    max_gain_dbi: float = 15.0,
    v_bw_deg: float = 65.0,
) -> float:
    """Exact CosineAntennaModel (H=360 deg, V=65 deg, tilt=0 deg) gain from ns-3."""
    theta_el_rad = math.atan2(h_bs - h_ut, d_2d)
    v_exp = -3.0 / (20.0 * math.log10(math.cos(math.radians(v_bw_deg / 4.0))))
    gain_lin = math.cos(-theta_el_rad / 2.0) ** (2.0 * v_exp)
    return float(max_gain_dbi + 10.0 * math.log10(gain_lin))


def audit_module_1() -> dict[str, Any]:
    """Module 1: TS 38.306 throughput ceilings & 100% spectrum conservation."""
    with open(SCALER_PATH, "r", encoding="utf-8") as fh:
        scaler = load_yaml(fh.read())

    tdd_factor = 54.0 / 70.0
    rate_52_fdd = ts38_306_dl_peak_mbps(n_prb=52, tdd_factor=1.0)
    rate_52_tdd = ts38_306_dl_peak_mbps(n_prb=52, tdd_factor=tdd_factor)
    rate_53_fdd = ts38_306_dl_peak_mbps(n_prb=53, tdd_factor=1.0)
    rate_53_tdd = ts38_306_dl_peak_mbps(n_prb=53, tdd_factor=tdd_factor)

    lena_52 = lena_5g_tdd_peak_mbps(n_prb=52)
    lena_53 = lena_5g_tdd_peak_mbps(n_prb=53)
    lena_53_7ms = lena_5g_tdd_peak_mbps(n_prb=53, period_ms=7.0)

    # 1) Inspect 100% of windows in fase3_post_windows.jsonl
    ep_alloc_means: dict[tuple[str, str, int], dict[str, float]] = {}
    total_episodes = 0
    total_windows = 0
    neg_prb_violations = 0
    cap_53_violations = 0
    max_a_exec_sum = -1

    with open(POST_WINDOWS_JSONL, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            total_episodes += 1
            pw = obj["post_windows"]
            a0 = np.asarray(pw["a_exec_0"], dtype=np.int64)
            a1 = np.asarray(pw["a_exec_1"], dtype=np.int64)
            a2 = np.asarray(pw["a_exec_2"], dtype=np.int64)
            ok_e = np.asarray(pw["sla_ok_e"], dtype=np.int64)
            asum = a0 + a1 + a2
            n_w = int(a0.size)
            n_vacuous_e = int(np.sum((ok_e == 1) & (a0 == 0)))
            n_non_vacuous_e = n_w - n_vacuous_e
            a0_pos = a0[a0 > 0]
            total_windows += n_w
            neg_prb_violations += int(np.sum((a0 < 0) | (a1 < 0) | (a2 < 0)))
            cap_53_violations += int(np.sum(asum > 53))
            max_a_exec_sum = max(max_a_exec_sum, int(np.max(asum)))
            ep_alloc_means[(obj["split"], obj["method"], int(obj["seed"]))] = {
                "mean_a_exec_0": float(np.mean(a0)),
                "mean_a_exec_0_non_vacuous": float(np.sum(a0) / max(1, n_non_vacuous_e)),
                "mean_a_exec_0_pos": float(np.mean(a0_pos)) if a0_pos.size else 0.0,
                "n_vacuous_embb": n_vacuous_e,
                "n_non_vacuous_embb": n_non_vacuous_e,
                "n_a_exec_0_pos": int(a0_pos.size),
                "mean_a_exec_1": float(np.mean(a1)),
                "mean_a_exec_2": float(np.mean(a2)),
                "mean_a_exec_sum": float(np.mean(asum)),
                "n_post_windows": n_w,
            }

    assert total_episodes == 105, f"Expected 105 episodes, got {total_episodes}"
    assert neg_prb_violations == 0, f"Negative PRB allocations: {neg_prb_violations}"
    assert cap_53_violations == 0, f"PRB sum > 53 violations: {cap_53_violations}"

    # Auxiliary CBR bounds from sqr-v2-sim.cc & scaler_manifest_v2_tdd5ms.yaml
    # Little's Law expected active UEs per slice: lambda = 3 UE/min, mean session = 60 s -> E[N] = 3.0
    urllc_nominal_mbps = 3.0 * (32.0 * 8.0 / 1e-3 / 1e6)  # 3 * 0.256 = 0.768 Mbps
    mmtc_nominal_per_ue_mbps = 200.0 * 8.0 / 1.0 / 1e6     # 0.0016 Mbps
    offered_u_p99 = float(scaler["offered_u_p99"])
    offered_m_p99 = float(scaler["offered_m_p99"])
    aux_p99_mbps = offered_u_p99 + offered_m_p99

    # 2) Check 14/14 rows in fase3_method_summary.csv
    df_m = pd.read_csv(METHOD_SUMMARY_CSV)
    assert len(df_m) == 14, f"Expected 14 method summary rows, got {len(df_m)}"
    method_checks: list[dict[str, Any]] = []
    for _, r in df_m.iterrows():
        embb_thr = float(r["embb_thr_mean_mbps_mean"])
        mmtc_pdr_eff = min(1.0, max(0.0, float(r["mmtc_pdr_mean_mean"])))
        mmtc_nom = 3.0 * mmtc_nominal_per_ue_mbps * mmtc_pdr_eff
        tot_nom = embb_thr + urllc_nominal_mbps + mmtc_nom
        tot_p99 = embb_thr + aux_p99_mbps
        cons_err = float(r["rb_win_conservation_abs_err_mean"])
        prb_util = float(r["prb_util_pct_mean"])
        assert cons_err == 0.0, f"Method {r['split']} {r['method']} cons_err={cons_err}"
        assert prb_util <= 100.0 + 1e-9, f"Method {r['method']} prb_util={prb_util}"
        assert tot_nom <= rate_52_tdd, f"Method {r['method']} tot_nom={tot_nom} > {rate_52_tdd}"
        assert tot_p99 <= rate_52_tdd, f"Method {r['method']} tot_p99={tot_p99} > {rate_52_tdd}"
        method_checks.append(
            {
                "split": str(r["split"]),
                "method": str(r["method"]),
                "embb_thr_mean_mbps": embb_thr,
                "total_nominal_mbps": tot_nom,
                "total_p99_worst_mbps": tot_p99,
                "headroom_vs_3gpp_52_tdd_nom_mbps": rate_52_tdd - tot_nom,
                "headroom_vs_3gpp_52_tdd_p99_mbps": rate_52_tdd - tot_p99,
                "prb_util_pct": prb_util,
                "cons_err": cons_err,
            }
        )

    # 3) Check 105/105 episodes in fase3_per_seed_summary.csv + join per-PRB eMBB check
    df_s = pd.read_csv(PER_SEED_CSV)
    assert len(df_s) == 105, f"Expected 105 per-seed rows, got {len(df_s)}"
    ceiling_per_prb_mbps = rate_53_tdd / 53.0  # == rate_52_tdd / 52.0
    max_ep_embb_thr = -1.0
    max_ep_embb_key = ""
    max_ep_tot_nom = -1.0
    max_ep_tot_p99 = -1.0
    max_ep_per_prb_rate_non_vac = -1.0
    max_ep_per_prb_key_non_vac = ""
    max_ep_per_prb_rate_pos = -1.0
    max_ep_per_prb_key_pos = ""
    max_ep_per_prb_rate_naive = -1.0
    max_ep_per_prb_key_naive = ""
    naive_gt_ceiling_episodes: list[dict[str, Any]] = []
    ep_cons_violations = 0
    ep_prb_util_violations = 0
    ep_ceiling_violations = 0
    ep_per_prb_non_vac_violations = 0
    ep_per_prb_pos_violations = 0

    for _, r in df_s.iterrows():
        key = (str(r["split"]), str(r["method"]), int(r["seed"]))
        embb_thr = float(r["embb_thr_mean_mbps"])
        mmtc_pdr_eff = min(1.0, max(0.0, float(r["mmtc_pdr_mean"])))
        tot_nom = embb_thr + urllc_nominal_mbps + 3.0 * mmtc_nominal_per_ue_mbps * mmtc_pdr_eff
        tot_p99 = embb_thr + aux_p99_mbps
        cons_err = float(r["rb_win_conservation_abs_err"])
        prb_util = float(r["prb_util_pct"])
        alloc_info = ep_alloc_means[key]
        mean_a0_all = alloc_info["mean_a_exec_0"]
        mean_a0_non_vac = alloc_info["mean_a_exec_0_non_vacuous"]
        mean_a0_pos = alloc_info["mean_a_exec_0_pos"]

        per_prb_non_vac = embb_thr / mean_a0_non_vac
        per_prb_pos = embb_thr / mean_a0_pos
        per_prb_naive = embb_thr / mean_a0_all

        if cons_err != 0.0:
            ep_cons_violations += 1
        if prb_util > 100.0 + 1e-9:
            ep_prb_util_violations += 1
        if tot_p99 > rate_52_tdd:
            ep_ceiling_violations += 1
        if per_prb_non_vac > ceiling_per_prb_mbps:
            ep_per_prb_non_vac_violations += 1
        if per_prb_pos > ceiling_per_prb_mbps:
            ep_per_prb_pos_violations += 1
        if per_prb_naive > ceiling_per_prb_mbps:
            naive_gt_ceiling_episodes.append(
                {
                    "episode": f"{key[0]}/{key[1]}/s{key[2]}",
                    "embb_thr_mean_mbps": embb_thr,
                    "mean_a_exec_0_all_windows": mean_a0_all,
                    "naive_ratio_mbps_per_prb": per_prb_naive,
                    "n_vacuous_embb_windows": alloc_info["n_vacuous_embb"],
                    "n_post_windows": alloc_info["n_post_windows"],
                    "mean_a_exec_0_non_vacuous": mean_a0_non_vac,
                    "non_vacuous_ratio_mbps_per_prb": per_prb_non_vac,
                    "mean_a_exec_0_pos": mean_a0_pos,
                    "pos_ratio_mbps_per_prb": per_prb_pos,
                }
            )

        if embb_thr > max_ep_embb_thr:
            max_ep_embb_thr = embb_thr
            max_ep_embb_key = f"{key[0]}/{key[1]}/s{key[2]}"
        max_ep_tot_nom = max(max_ep_tot_nom, tot_nom)
        max_ep_tot_p99 = max(max_ep_tot_p99, tot_p99)
        if per_prb_non_vac > max_ep_per_prb_rate_non_vac:
            max_ep_per_prb_rate_non_vac = per_prb_non_vac
            max_ep_per_prb_key_non_vac = (
                f"{key[0]}/{key[1]}/s{key[2]} (a0_non_vac={mean_a0_non_vac:.2f}, thr={embb_thr:.3f})"
            )
        if per_prb_pos > max_ep_per_prb_rate_pos:
            max_ep_per_prb_rate_pos = per_prb_pos
            max_ep_per_b_tag = (
                f"{key[0]}/{key[1]}/s{key[2]} (a0_pos={mean_a0_pos:.2f}, thr={embb_thr:.3f})"
            )
            max_ep_per_prb_key_pos = max_ep_per_b_tag
        if per_prb_naive > max_ep_per_prb_rate_naive:
            max_ep_per_prb_rate_naive = per_prb_naive
            max_ep_per_prb_key_naive = (
                f"{key[0]}/{key[1]}/s{key[2]} (a0_all={mean_a0_all:.2f}, thr={embb_thr:.3f})"
            )

    assert ep_cons_violations == 0
    assert ep_prb_util_violations == 0
    assert ep_ceiling_violations == 0
    assert ep_per_prb_non_vac_violations == 0
    assert ep_per_prb_pos_violations == 0

    # 4) Check radius sweep tables
    df_rs = pd.read_csv(RADIUS_SUMMARY_CSV)
    df_rr = pd.read_csv(RADIUS_RAW_CSV)
    assert len(df_rs) == 10 and len(df_rr) == 30
    assert float(df_rs["cons_err"].abs().max()) == 0.0
    assert float(df_rr["cons_err"].abs().max()) == 0.0
    assert float(df_rs["rb_util_pct"].max()) <= 100.0 + 1e-9
    assert float(df_rr["rb_util_pct"].max()) <= 100.0 + 1e-9
    assert float(df_rs["thr_p5_mean_mbps"].max()) < rate_52_tdd
    assert float(df_rr["thr_p5_mean_mbps"].max()) < rate_52_tdd

    return {
        "ts38_306_ceilings_mbps": {
            "N_PRB_52_FDD": rate_52_fdd,
            "N_PRB_52_TDD_DDDSU_54_70": rate_52_tdd,
            "N_PRB_53_FDD": rate_53_fdd,
            "N_PRB_53_TDD_DDDSU_54_70": rate_53_tdd,
            "tdd_duty_factor_54_70": tdd_factor,
            "ceiling_per_prb_mbps": ceiling_per_prb_mbps,
        },
        "lena_5g_tb_cross_check": {
            "tdd5ms_52_prb": lena_52,
            "tdd5ms_53_prb": lena_53,
            "tdd7ms_legacy_53_prb": lena_53_7ms,
        },
        "auxiliary_cbr_bounds_mbps": {
            "urllc_nominal_3_ues": urllc_nominal_mbps,
            "mmtc_nominal_3_ues_full_pdr": 3.0 * mmtc_nominal_per_ue_mbps,
            "offered_u_p99": offered_u_p99,
            "offered_m_p99": offered_m_p99,
            "aux_p99_worst_case": aux_p99_mbps,
        },
        "windows_checked": total_windows,
        "episodes_checked": total_episodes,
        "window_prb_sum_max": max_a_exec_sum,
        "window_prb_violations": cap_53_violations + neg_prb_violations,
        "method_summary_rows_checked": len(df_m),
        "method_summary_all_pass": True,
        "method_checks": method_checks,
        "per_seed_summary_rows_checked": len(df_s),
        "max_episode_embb_thr_mbps": max_ep_embb_thr,
        "max_episode_embb_key": max_ep_embb_key,
        "max_episode_total_nominal_mbps": max_ep_tot_nom,
        "max_episode_total_p99_worst_mbps": max_ep_tot_p99,
        "max_episode_embb_per_prb_non_vacuous_mbps": max_ep_per_prb_rate_non_vac,
        "max_episode_embb_per_prb_non_vacuous_key": max_ep_per_prb_key_non_vac,
        "max_episode_embb_per_prb_pos_mbps": max_ep_per_prb_rate_pos,
        "max_episode_embb_per_prb_pos_key": max_ep_per_prb_key_pos,
        "diagnostic_naive_per_prb_max_mbps": max_ep_per_prb_rate_naive,
        "diagnostic_naive_per_prb_max_key": max_ep_per_prb_key_naive,
        "diagnostic_naive_gt_ceiling_episodes": naive_gt_ceiling_episodes,
        "radius_summary_max_thr_p5_mbps": float(df_rs["thr_p5_mean_mbps"].max()),
        "radius_raw_max_thr_p5_mbps": float(df_rr["thr_p5_mean_mbps"].max()),
        "all_pass": True,
    }


def audit_module_2() -> dict[str, Any]:
    """Module 2: TR 38.901 UMa link budget & RLF thresholds vs radius sweep."""
    k_b = 1.38064852e-23
    t_0 = 290.0
    kbt = k_b * t_0  # nr-ue-phy.cc:1720
    p_tx_dbm = 30.0
    g_rx_dbi = 0.0
    nf_db = 9.0
    q_out_db = -5.0  # nr-ue-phy.cc:224
    q_in_db = -3.9   # nr-ue-phy.cc:231
    sigma_sf_los = 4.0   # three-gpp-propagation-loss-model.cc:1156
    sigma_sf_nlos = 6.0  # three-gpp-propagation-loss-model.cc:1160

    bw_10m = 10.0e6
    bw_53prb = 53.0 * 12.0 * 15.0e3  # 9.54 MHz usable band
    bw_52prb = 52.0 * 12.0 * 15.0e3  # 9.36 MHz nominal TS 38.104 band

    def noise_dbm(bw_hz: float) -> float:
        return float(10.0 * math.log10(kbt * bw_hz * 1000.0) + nf_db)

    n_10m_dbm = noise_dbm(bw_10m)
    n_53_dbm = noise_dbm(bw_53prb)
    n_52_dbm = noise_dbm(bw_52prb)

    df_rs = pd.read_csv(RADIUS_SUMMARY_CSV)
    radii = (300.0, 600.0, 1000.0, 1500.0, 2000.0)
    rows: list[dict[str, Any]] = []

    for r_m in radii:
        pl = uma_pl_los_nlos_db(r_m)
        g_tx = cosine_antenna_gain_db(r_m)
        eirp_dbm = p_tx_dbm + g_tx
        rx_los_dbm = eirp_dbm + g_rx_dbi - pl["pl_los_db"]
        rx_nlos_dbm = eirp_dbm + g_rx_dbi - pl["pl_nlos_db"]

        snr_los_10m = rx_los_dbm - n_10m_dbm
        snr_nlos_10m = rx_nlos_dbm - n_10m_dbm
        snr_los_53 = rx_los_dbm - n_53_dbm
        snr_nlos_53 = rx_nlos_dbm - n_53_dbm

        # Outage probability under NLOS / LOS log-normal shadowing (evaluated on active 53-PRB band)
        p_nlos_lt_qout = norm_cdf(q_out_db, mean=snr_nlos_53, std=sigma_sf_nlos)
        p_nlos_lt_qin = norm_cdf(q_in_db, mean=snr_nlos_53, std=sigma_sf_nlos)
        p_los_lt_qout = norm_cdf(q_out_db, mean=snr_los_53, std=sigma_sf_los)
        p_outage_mixed = (1.0 - pl["p_los"]) * p_nlos_lt_qout + pl["p_los"] * p_los_lt_qout

        sub = df_rs[df_rs["radius_m"] == int(r_m)]
        embb_row = sub[sub["profile"] == "embb_article_05_10_15"].iloc[0]
        het_row = sub[sub["profile"] == "het_sqr_cer"].iloc[0]

        rows.append(
            {
                **pl,
                "g_tx_eff_dbi": g_tx,
                "rx_los_dbm": rx_los_dbm,
                "rx_nlos_dbm": rx_nlos_dbm,
                "snr_los_10mhz_db": snr_los_10m,
                "snr_nlos_10mhz_db": snr_nlos_10m,
                "snr_los_53prb_db": snr_los_53,
                "snr_nlos_53prb_db": snr_nlos_53,
                "margin_nlos_vs_qout_53prb_db": snr_nlos_53 - q_out_db,
                "margin_nlos_vs_qin_53prb_db": snr_nlos_53 - q_in_db,
                "p_nlos_snr_lt_qout_pct": 100.0 * p_nlos_lt_qout,
                "p_nlos_snr_lt_qin_pct": 100.0 * p_nlos_lt_qin,
                "p_edge_rlf_mixed_pct": 100.0 * p_outage_mixed,
                "sim_embb_article_viol_pct": float(embb_row["sla_viol_all_pct"]),
                "sim_embb_article_thr_p5_mbps": float(embb_row["thr_p5_mean_mbps"]),
                "sim_het_cer_viol_pct": float(het_row["sla_viol_all_pct"]),
                "sim_het_cer_thr_p5_mbps": float(het_row["thr_p5_mean_mbps"]),
                "sim_het_cer_urllc_p95_ms": float(het_row["urllc_p95_ms"]),
            }
        )

    # Monotonic sanity checks vs physics
    assert rows[-1]["snr_nlos_53prb_db"] < q_out_db, "2000m NLOS edge SNR must be below Qout=-5 dB"
    assert rows[0]["snr_nlos_53prb_db"] > q_in_db + 15.0, "300m NLOS edge SNR must exceed Qin with >15 dB margin"
    assert rows[-1]["sim_het_cer_viol_pct"] > rows[0]["sim_het_cer_viol_pct"]
    assert rows[-1]["sim_embb_article_viol_pct"] > rows[0]["sim_embb_article_viol_pct"]

    return {
        "constants": {
            "f_c_ghz": 2.0,
            "h_bs_m": 25.0,
            "h_ut_m": 1.5,
            "d_bp_m": 320.0,
            "p_tx_dbm": p_tx_dbm,
            "g_tx_max_dbi": 15.0,
            "nf_db": nf_db,
            "noise_10mhz_dbm": n_10m_dbm,
            "noise_53prb_9_54mhz_dbm": n_53_dbm,
            "noise_52prb_9_36mhz_dbm": n_52_dbm,
            "q_out_db": q_out_db,
            "q_in_db": q_in_db,
            "sigma_sf_los_db": sigma_sf_los,
            "sigma_sf_nlos_db": sigma_sf_nlos,
        },
        "radii_profiles": rows,
        "all_pass": True,
    }


def audit_module_3() -> dict[str, Any]:
    """Module 3: Deterministic E2E latency floor vs measured URLLC p95."""
    backhaul_p2p_ms = 1.0          # sqr-v2-sim.cc:3259
    s1u_delay_ms = 0.0             # nr-point-to-point-epc-helper.cc:66
    serial_60b_100g_ms = (60.0 * 8.0) / 100e9 * 1e3
    l1_l2_ctrl_latency_ms = 2.0    # nr-phy.cc:703 (2 slots @ mu=0)
    n0_delay_ms = 0.0              # nr-gnb-phy.cc:135
    n1_delay_ms = 2.0              # nr-gnb-phy.cc:140
    n2_delay_ms = 2.0              # nr-gnb-phy.cc:146
    air_pdsch_min_ms = 13.0 / 14.0 # 13 PDSCH symbols of 14-symbol slot
    air_slot_full_ms = 1.0
    tb_decode_latency_ms = 0.10    # nr-ue-phy.cc:128 (100 us)

    # DDDSU 5 ms TDD alignment wait (4 DL/S slots + 1 UL slot) under slot-grid quantization
    tdd_wait_min_ms = 0.0
    tdd_wait_mean_ms = (4.0 / 5.0) * 0.5 + (1.0 / 5.0) * 1.5  # 0.70 ms
    tdd_wait_p95_ms = 1.75
    tdd_wait_max_ms = 2.0

    # Legacy 7 ms DDDSUUU comparison
    tdd7_wait_mean_ms = (4.0 / 7.0) * 0.5 + (3.0 / 7.0) * 2.5  # 1.35714 ms
    tdd7_wait_max_ms = 4.0

    floor_abs_min_ms = (
        backhaul_p2p_ms
        + s1u_delay_ms
        + tdd_wait_min_ms
        + l1_l2_ctrl_latency_ms
        + air_pdsch_min_ms
        + tb_decode_latency_ms
    )
    floor_full_slot_best_phase_ms = (
        backhaul_p2p_ms
        + s1u_delay_ms
        + tdd_wait_min_ms
        + l1_l2_ctrl_latency_ms
        + air_slot_full_ms
        + tb_decode_latency_ms
    )
    floor_unloaded_mean_ms = (
        backhaul_p2p_ms
        + s1u_delay_ms
        + tdd_wait_mean_ms
        + l1_l2_ctrl_latency_ms
        + air_pdsch_min_ms
        + tb_decode_latency_ms
    )
    floor_unloaded_worst_phase_ms = (
        backhaul_p2p_ms
        + s1u_delay_ms
        + tdd_wait_max_ms
        + l1_l2_ctrl_latency_ms
        + air_slot_full_ms
        + tb_decode_latency_ms
    )

    df_m = pd.read_csv(METHOD_SUMMARY_CSV)
    df_s = pd.read_csv(PER_SEED_CSV)

    assert float(df_m["urllc_p95_ms_mean_mean"].min()) >= floor_abs_min_ms
    assert float(df_m["urllc_p95_ms_p95_mean"].min()) >= floor_abs_min_ms
    assert float(df_s["urllc_p95_ms_mean"].min()) >= floor_abs_min_ms
    assert float(df_s["urllc_p95_ms_p95"].min()) >= floor_abs_min_ms

    min_ep_mean_row = df_s.loc[df_s["urllc_p95_ms_mean"].idxmin()]
    min_ep_p95_row = df_s.loc[df_s["urllc_p95_ms_p95"].idxmin()]

    method_latency_rows: list[dict[str, Any]] = []
    for _, r in df_m.iterrows():
        u_mean = float(r["urllc_p95_ms_mean_mean"])
        u_p95 = float(r["urllc_p95_ms_p95_mean"])
        method_latency_rows.append(
            {
                "split": str(r["split"]),
                "method": str(r["method"]),
                "urllc_p95_ms_mean": u_mean,
                "urllc_p95_ms_p95": u_p95,
                "excess_above_abs_min_ms": u_mean - floor_abs_min_ms,
                "excess_above_worst_phase_6_10_ms": u_mean - floor_unloaded_worst_phase_ms,
            }
        )

    return {
        "components_ms": {
            "backhaul_p2p_ms": backhaul_p2p_ms,
            "s1u_link_delay_ms": s1u_delay_ms,
            "serialization_60b_100g_ms": serial_60b_100g_ms,
            "tdd5ms_wait_min_ms": tdd_wait_min_ms,
            "tdd5ms_wait_mean_ms": tdd_wait_mean_ms,
            "tdd5ms_wait_p95_ms": tdd_wait_p95_ms,
            "tdd5ms_wait_max_ms": tdd_wait_max_ms,
            "tdd7ms_wait_mean_ms": tdd7_wait_mean_ms,
            "tdd7ms_wait_max_ms": tdd7_wait_max_ms,
            "l1_l2_ctrl_latency_ms": l1_l2_ctrl_latency_ms,
            "n0_delay_ms": n0_delay_ms,
            "n1_delay_ms": n1_delay_ms,
            "n2_delay_ms": n2_delay_ms,
            "air_pdsch_13of14_ms": air_pdsch_min_ms,
            "air_slot_full_ms": air_slot_full_ms,
            "tb_decode_latency_ms": tb_decode_latency_ms,
        },
        "floors_ms": {
            "abs_min_13of14_ms": floor_abs_min_ms,
            "full_slot_best_phase_ms": floor_full_slot_best_phase_ms,
            "unloaded_mean_ms": floor_unloaded_mean_ms,
            "unloaded_worst_phase_ms": floor_unloaded_worst_phase_ms,
            "harq_retx_1_cycle_approx_ms": [
                floor_unloaded_worst_phase_ms + 5.0,
                floor_unloaded_worst_phase_ms + 7.0,
            ],
        },
        "observed_min_method_urllc_p95_mean_ms": float(df_m["urllc_p95_ms_mean_mean"].min()),
        "observed_min_episode_urllc_p95_mean_ms": float(min_ep_mean_row["urllc_p95_ms_mean"]),
        "observed_min_episode_urllc_p95_mean_id": (
            f"{min_ep_mean_row['split']}/{min_ep_mean_row['method']}/s{int(min_ep_mean_row['seed'])}"
        ),
        "observed_min_episode_urllc_p95_p95_ms": float(min_ep_p95_row["urllc_p95_ms_p95"]),
        "observed_min_episode_urllc_p95_p95_id": (
            f"{min_ep_p95_row['split']}/{min_ep_p95_row['method']}/s{int(min_ep_p95_row['seed'])}"
        ),
        "method_rows": method_latency_rows,
        "all_pass": True,
    }


def main() -> int:
    m1 = audit_module_1()
    m2 = audit_module_2()
    m3 = audit_module_3()
    summary = {
        "schema": "sqr_v2_3gpp_physical_audit_1.0",
        "module_1_ts38_306_and_conservation": m1,
        "module_2_tr38_901_link_budget": m2,
        "module_3_latency_physical_floor": m3,
        "overall_pass": bool(m1["all_pass"] and m2["all_pass"] and m3["all_pass"]),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("=== 3GPP PHYSICAL AUDIT SUMMARY ===")
    c52 = m1["ts38_306_ceilings_mbps"]["N_PRB_52_TDD_DDDSU_54_70"]
    c53 = m1["ts38_306_ceilings_mbps"]["N_PRB_53_TDD_DDDSU_54_70"]
    print(
        f"[Module 1 PASS] TS 38.306 DL TDD ceiling: N_PRB=52 -> {c52:.6f} Mbps | "
        f"N_PRB=53 -> {c53:.6f} Mbps | 5G-LENA TB ceiling(52/53) -> "
        f"{m1['lena_5g_tb_cross_check']['tdd5ms_52_prb']['peak_mbps']:.3f}/"
        f"{m1['lena_5g_tb_cross_check']['tdd5ms_53_prb']['peak_mbps']:.3f} Mbps"
    )
    print(
        f"  Windows checked: {m1['windows_checked']} (max_prb_sum={m1['window_prb_sum_max']}, "
        f"violations={m1['window_prb_violations']})"
    )
    print(
        f"  Episodes checked: {m1['episodes_checked']} | max eMBB={m1['max_episode_embb_thr_mbps']:.3f} Mbps "
        f"({m1['max_episode_embb_key']}) | max total_p99={m1['max_episode_total_p99_worst_mbps']:.3f} Mbps <= {c52:.3f} Mbps | "
        f"max eMBB/PRB (non-vacuous)={m1['max_episode_embb_per_prb_non_vacuous_mbps']:.4f} "
        f"<= {m1['ts38_306_ceilings_mbps']['ceiling_per_prb_mbps']:.4f} Mbps/PRB "
        f"({m1['max_episode_embb_per_prb_non_vacuous_key']})"
    )
    if m1["diagnostic_naive_gt_ceiling_episodes"]:
        ex = m1["diagnostic_naive_gt_ceiling_episodes"][0]
        print(
            f"  [Diagnostic denom mismatch note] {len(m1['diagnostic_naive_gt_ceiling_episodes'])} ep "
            f"({ex['episode']}): naive a0_all={ex['mean_a_exec_0_all_windows']:.3f} -> "
            f"{ex['naive_ratio_mbps_per_prb']:.4f} (includes {ex['n_vacuous_embb_windows']} empty eMBB "
            f"windows); non-vacuous={ex['non_vacuous_ratio_mbps_per_prb']:.4f}, pos={ex['pos_ratio_mbps_per_prb']:.4f} Mbps/PRB"
        )
    r300, r2000 = m2["radii_profiles"][0], m2["radii_profiles"][-1]
    print(
        f"[Module 2 PASS] TR 38.901 UMa (53 PRB BW=9.54 MHz, N={m2['constants']['noise_53prb_9_54mhz_dbm']:.2f} dBm):"
    )
    print(
        f"  300 m : P_LOS={r300['p_los']:.4f}, PL_LOS={r300['pl_los_db']:.2f} dB, PL_NLOS={r300['pl_nlos_db']:.2f} dB, "
        f"SNR_NLOS={r300['snr_nlos_53prb_db']:.2f} dB (outage={r300['p_edge_rlf_mixed_pct']:.3f}%), sim het viol={r300['sim_het_cer_viol_pct']:.2f}%"
    )
    print(
        f"  2000 m: P_LOS={r2000['p_los']:.4f}, PL_LOS={r2000['pl_los_db']:.2f} dB, PL_NLOS={r2000['pl_nlos_db']:.2f} dB, "
        f"SNR_NLOS={r2000['snr_nlos_53prb_db']:.2f} dB (outage={r2000['p_edge_rlf_mixed_pct']:.2f}%), sim het viol={r2000['sim_het_cer_viol_pct']:.2f}%"
    )
    fl = m3["floors_ms"]
    print(
        f"[Module 3 PASS] Latency Floor (abs_min={fl['abs_min_13of14_ms']:.3f} ms, "
        f"mean_unloaded={fl['unloaded_mean_ms']:.3f} ms, worst_phase_unloaded={fl['unloaded_worst_phase_ms']:.3f} ms) | "
        f"observed min method mean={m3['observed_min_method_urllc_p95_mean_ms']:.3f} ms | "
        f"observed min episode mean={m3['observed_min_episode_urllc_p95_mean_ms']:.3f} ms ({m3['observed_min_episode_urllc_p95_mean_id']}) | "
        f"observed min episode p95={m3['observed_min_episode_urllc_p95_p95_ms']:.3f} ms ({m3['observed_min_episode_urllc_p95_p95_id']})"
    )
    print(f"Wrote: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
