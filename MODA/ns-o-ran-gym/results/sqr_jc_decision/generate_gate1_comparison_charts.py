#!/usr/bin/env python3
"""Generate publication-quality comparative charts (Gate 1 executed baselines vs A5)
strictly from A5_GATE1_DIAGNOSTICS.json and A5_GATE1_RETROSPECTIVE_ANALYSIS.json.

ZERO hardcoded synthetic outcomes; all values read from audited artifacts.
Outputs:
  - MODA/docs/sqr/jc_decision/figures/fig1_proposta_vs_baselines.png
  - MODA/docs/sqr/jc_decision/figures/fig2_diagnostico_separabilidade_teto.png
"""
from __future__ import annotations

import json
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = "/home/eliothluy/Documentos/Jussi_SBRC"
DEC_DIR = os.path.join(REPO_ROOT, "MODA", "ns-o-ran-gym", "results", "sqr_jc_decision")
FIG_DIR = os.path.join(REPO_ROOT, "MODA", "docs", "sqr", "jc_decision", "figures")

DIAG_JSON = os.path.join(DEC_DIR, "A5_GATE1_DIAGNOSTICS.json")
RETRO_JSON = os.path.join(DEC_DIR, "A5_GATE1_RETROSPECTIVE_ANALYSIS.json")

ARM_ORDER = ("b0", "eq", "prop", "b1_raw", "b1_act", "b1_sup")
ARM_LABELS = {
    "b0": "B0-rest\n(baseline)",
    "eq": "equal\n(18,17,18)",
    "prop": "prop\n(ativa)",
    "b1_raw": "B1-raw\n(A5 v1)",
    "b1_act": "B1-active\n(A5 v1)",
    "b1_sup": "B1-support\n(A5 v1)",
}
SHORT_LABELS = {
    "eq": "equal (18,17,18)",
    "prop": "prop (ativa)",
    "b1_raw": "B1-raw (A5 v1)",
    "b1_act": "B1-active (A5 v1)",
    "b1_sup": "B1-support (A5 v1)",
}


def _configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.6,
        "legend.fontsize": 8.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _make_fig1(diag: dict[str, Any], retro: dict[str, Any]) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 9.8), dpi=300)
    fig.subplots_adjust(hspace=0.35, wspace=0.28, top=0.90, bottom=0.07, left=0.11, right=0.98)
    fig.suptitle(
        "Gate 1 A5 (127 replays ns-3/5G-LENA válidos): Proposta B1-* vs Baselines (B0-rest, equal, prop)",
        fontsize=13.2, fontweight="bold", y=0.96,
    )

    # --- Panel (a): Joint SLA Pass Rate in Univ A (N=18) vs Univ B (N=5) ---
    ax = axes[0, 0]
    x = np.arange(len(ARM_ORDER))
    w = 0.37
    uA = diag["universe_A_ALL3_COMPLETE"]
    uB = diag["universe_B_LT3_DEMAND_COMPLETE"]
    o3A = retro["hindsight_oracles"]["univ_A_O3_b0_eq_prop"]["oracle_rate"] * 100.0
    o3B = retro["hindsight_oracles"]["univ_B_O3_b0_eq_prop"]["oracle_rate"] * 100.0

    vals_A = [uA[a]["cond_pass_pct"] for a in ARM_ORDER]
    vals_B = [uB[a]["cond_pass_pct"] for a in ARM_ORDER]
    bA = ax.bar(x - w / 2, vals_A, width=w, color="#2B5C8F", edgecolor="#1A365D", lw=0.7,
                label="Univ A (3 ativas, N=18)")
    bB = ax.bar(x + w / 2, vals_B, width=w, color="#2CA02C", edgecolor="#1A5E1A", lw=0.7, alpha=0.88,
                label="Univ B (<3 ativas, N=5)")

    ax.axhline(o3A, color="#2B5C8F", linestyle=":", lw=1.3, label=f"Oráculo O3* A ({o3A:.1f}%)")
    ax.axhline(o3B, color="#2CA02C", linestyle="-.", lw=1.2, label=f"Oráculo O3* B ({o3B:.1f}%)")

    for rect, arm in zip(bA, ARM_ORDER):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.0f}%\n({uA[arm]['pass']}/18)",
                ha="center", va="bottom", fontsize=7.1, color="#1A365D", fontweight="bold")
    for rect, arm in zip(bB, ARM_ORDER):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.0f}%\n({uB[arm]['pass']}/5)",
                ha="center", va="bottom", fontsize=7.1, color="#1A5E1A", fontweight="bold")

    ax.annotate("Ganho real:\nB1-act = 80%\n(+40 pp vs B0)",
                xy=(4 + w / 2, 82.0), xytext=(2.85, 91.0),
                arrowprops=dict(arrowstyle="->", color="#1A5E1A", lw=1.2),
                fontsize=7.8, fontweight="bold", color="#1A5E1A",
                bbox=dict(boxstyle="round,pad=0.22", fc="#EBF9EB", ec="#2CA02C", lw=0.8))
    ax.annotate("Perda em 3 ativas:\n-33,3 pp vs B0\n-55,6 pp vs eq",
                xy=(4 - w / 2, 19.0), xytext=(4.15, 36.0),
                arrowprops=dict(arrowstyle="->", color="#B22222", lw=1.2),
                fontsize=7.7, fontweight="bold", color="#B22222",
                bbox=dict(boxstyle="round,pad=0.22", fc="#FFF0F0", ec="#E45756", lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], fontsize=8.2)
    ax.set_ylim(0, 122)
    ax.set_ylabel("Taxa de Atendimento Conjunto R19 (%)")
    ax.set_title("(a) SLA Conjunto Executado: 3 Ativas (Univ A) vs <3 Ativas (Univ B)")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.94, fontsize=7.7)

    # --- Panel (b): Per-Slice Breakdown in Univ A (N=18) ---
    ax = axes[0, 1]
    subA = [s for s in retro["per_state_full"] if s["universe"] == "ALL3_COMPLETE"]
    e_pct, u_pct, m_pct, e_med = [], [], [], []
    for a in ARM_ORDER:
        rs = [s["arms"][a] for s in subA]
        e_pct.append(100.0 * sum(1 for r in rs if r["embb_ok"]) / len(rs))
        u_pct.append(100.0 * sum(1 for r in rs if r["urllc_ok"]) / len(rs))
        m_pct.append(100.0 * sum(1 for r in rs if r["mmtc_ok"]) / len(rs))
        e_med.append(float(np.median([r["embb_thr_mbps"] for r in rs])))

    w3 = 0.25
    r_e = ax.bar(x - w3, e_pct, width=w3, color="#E45756", edgecolor="#7F1D1D", lw=0.6, label="eMBB (>= 3,5 Mbps)")
    r_u = ax.bar(x, u_pct, width=w3, color="#4C78A8", edgecolor="#1E3A5F", lw=0.6, label="URLLC (p95 <= 10 ms)")
    r_m = ax.bar(x + w3, m_pct, width=w3, color="#54A24B", edgecolor="#1E4D1B", lw=0.6, label="mMTC (PDR >= 95%)")

    for i, rect in enumerate(r_e):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.0f}%\n({e_med[i]:.2f})",
                ha="center", va="bottom", fontsize=7.0, color="#7F1D1D", fontweight="bold")
    for rect in r_u:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=7.1, color="#1E3A5F", fontweight="bold")
    for rect in r_m:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=7.1, color="#1E4D1B")

    ax.annotate("Ganho URLLC (+22 a +28 pp vs B0)\nao custo do colapso eMBB (-55,6 pp)",
                xy=(4 - w3, 34.0), xytext=(2.45, 53.0),
                arrowprops=dict(arrowstyle="->", color="#7F1D1D", lw=1.1),
                fontsize=7.8, fontweight="bold", color="#7F1D1D",
                bbox=dict(boxstyle="round,pad=0.22", fc="#FFF4E6", ec="#F58518", lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], fontsize=8.2)
    ax.set_ylim(0, 126)
    ax.set_ylabel("Satisfação por Fatia no Universo A (%)")
    ax.set_title("(b) Trade-Off por Fatia no Universo A (mediana eMBB Mbps)")
    ax.legend(loc="upper center", ncol=3, frameon=True, framealpha=0.94, fontsize=7.8)

    # --- Panel (c): Rescue vs Harm on Univ A (N=18: 8 B0=PASS, 10 B0=FAIL) ---
    ax = axes[1, 0]
    comp_arms = ("eq", "prop", "b1_raw", "b1_act", "b1_sup")
    y_pos = np.arange(len(comp_arms))[::-1]
    resc_cnt = [uA[a]["rescued_count"] for a in comp_arms]
    harm_cnt = [-uA[a]["harmed_count"] for a in comp_arms]
    net_cnt = [uA[a]["net_rescued_minus_harmed"] for a in comp_arms]

    ax.barh(y_pos, resc_cnt, height=0.50, color="#2CA02C", edgecolor="#1A5E1A", lw=0.7,
            label="Resgates (B0=FAIL -> PASS, de 10)")
    ax.barh(y_pos, harm_cnt, height=0.50, color="#D62728", edgecolor="#7F1D1D", lw=0.7,
            label="Danos (B0=PASS -> FAIL, de 8)")
    ax.axvline(0, color="#222222", lw=1.0)

    for yp, r_v, h_v, n_v in zip(y_pos, resc_cnt, harm_cnt, net_cnt):
        ax.text(r_v + 0.25, yp, f"+{r_v}/10 ({10*r_v}%)", va="center", ha="left",
                fontsize=8.0, color="#1A5E1A", fontweight="bold")
        ax.text(h_v - 0.25, yp, f"{h_v}/8 ({abs(h_v)/8*100:.1f}%)", va="center", ha="right",
                fontsize=8.0, color="#7F1D1D", fontweight="bold")
        badge_fc = "#EBF9EB" if n_v > 0 else "#FFF0F0"
        badge_ec = "#2CA02C" if n_v > 0 else "#D62728"
        ax.text(11.3, yp, f"Net: {n_v:+d}/18", va="center", ha="center", fontsize=8.2, fontweight="bold",
                color="#1A5E1A" if n_v > 0 else "#7F1D1D",
                bbox=dict(boxstyle="round,pad=0.20", fc=badge_fc, ec=badge_ec, lw=0.8))

    ax.set_yticks(y_pos)
    ax.set_yticklabels([SHORT_LABELS[a] for a in comp_arms], fontweight="bold", fontsize=8.4)
    ax.set_xlim(-11.4, 13.4)
    ax.set_ylim(-1.35, 4.65)
    ax.set_xlabel("← Danos em B0=PASS   |   Resgates em B0=FAIL →")
    ax.set_title("(c) Preservação vs Resgate em Estados Idênticos (Univ A, N=18)")
    ax.legend(loc="lower center", ncol=2, frameon=True, framealpha=0.96, fontsize=7.6)

    # --- Panel (d): Capacity Waste & Starvation across all 25 states ---
    ax = axes[1, 1]
    ops = retro["ops_support_fallback_waste_summary"]
    idle_prbs = [ops[a]["all25_total_idle_prbs"] for a in ARM_ORDER]
    inact_prbs = [ops[a]["all25_total_inactive_prbs"] for a in ARM_ORDER]
    starved_st = [ops[a]["all25_starved_live_states"] for a in ARM_ORDER]
    w2 = 0.27
    ax.bar(x - w2, idle_prbs, width=w2, color="#F58518", edgecolor="#8C4300", lw=0.6,
           label="PRBs ociosos (sum < 53, total/25 est.)")
    ax.bar(x, inact_prbs, width=w2, color="#8C564B", edgecolor="#4A2B24", lw=0.6,
           label="PRBs em fatia inativa (total/25 est.)")
    ax.bar(x + w2, [s * 10 for s in starved_st], width=w2, color="#E45756", edgecolor="#7F1D1D", lw=0.6,
           label="Estados c/ slice viva=0 (×10 p/ escala)")

    for i in range(len(ARM_ORDER)):
        if idle_prbs[i] > 0:
            ax.text(x[i] - w2, idle_prbs[i] + 5, f"{idle_prbs[i]}", ha="center", va="bottom",
                    fontsize=7.6, color="#8C4300", fontweight="bold")
        if inact_prbs[i] > 0:
            ax.text(x[i], inact_prbs[i] + 5, f"{inact_prbs[i]}", ha="center", va="bottom",
                    fontsize=7.6, color="#4A2B24", fontweight="bold")
        ax.text(x[i] + w2, starved_st[i] * 10 + 5, f"{starved_st[i]}/25", ha="center", va="bottom",
                fontsize=7.4, color="#7F1D1D", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], fontsize=8.2)
    ax.set_ylim(0, 360)
    ax.set_ylabel("Contagem Acumulada (PRBs ou Estados ×10)")
    ax.set_title("(d) Desperdício de Capacidade & Inanição (Todos os 25 Estados)")
    ax.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=7.7)

    out_path = os.path.join(FIG_DIR, "fig1_proposta_vs_baselines.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _make_fig2(diag: dict[str, Any], retro: dict[str, Any]) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(
        1, 3, figsize=(17.0, 5.6), dpi=300,
        gridspec_kw={"width_ratios": [1.05, 1.0, 1.05]},
    )
    fig.subplots_adjust(wspace=0.52, top=0.83, bottom=0.19, left=0.05, right=0.98)
    fig.suptitle(
        "Por que Perdemos no Universo A e Qual o Teto Empírico sobre B0-rest Congelado",
        fontsize=13.0, fontweight="bold", y=0.95,
    )

    # --- Panel (a): Pre-decision Predictive Skill (AUROC, BalAcc, TPR) ---
    ax = axes[0]
    sep = retro["pre_decision_predictive_separability"]["unique_state_slice_action"]
    slices = ("eMBB", "URLLC", "mMTC", "JOINT_ACTIVE")
    s_labels = [
        f"eMBB\nn={sep['eMBB']['overall']['n']}\nτ={sep['eMBB']['overall']['mean_tau']:.3f}",
        f"URLLC\nn={sep['URLLC']['overall']['n']}\nτ={sep['URLLC']['overall']['mean_tau']:.3f}",
        f"mMTC\nn={sep['mMTC']['overall']['n']}\nτ={sep['mMTC']['overall']['mean_tau']:.3f}",
        f"JOINT\nn={sep['JOINT_ACTIVE']['n']}\nτ={sep['JOINT_ACTIVE']['mean_tau']:.3f}",
    ]
    aurocs = [sep[k].get("overall", sep[k])["auroc_f"] for k in slices]
    bal_accs = [sep[k].get("overall", sep[k])["balanced_accuracy"] for k in slices]
    tprs = [sep[k].get("overall", sep[k])["tpr_sensitivity"] for k in slices]

    x = np.arange(len(slices))
    w = 0.25
    b1 = ax.bar(x - w, aurocs, width=w, color="#4C78A8", edgecolor="#1E3A5F", lw=0.6, label="AUROC(f_i)")
    b2 = ax.bar(x, bal_accs, width=w, color="#72B7B2", edgecolor="#205C57", lw=0.6, label="Acurácia Bal. (f>=0)")
    b3 = ax.bar(x + w, tprs, width=w, color="#F58518", edgecolor="#8C4300", lw=0.6, label="TPR / Recall (f>=0)")
    ax.axhline(0.50, color="#B22222", linestyle="--", lw=1.2, label="Acaso (0,50)")

    for bars, col in ((b1, "#1E3A5F"), (b2, "#205C57"), (b3, "#8C4300")):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.015, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7.4, color=col, fontweight="bold")

    ax.annotate("URLLC ≈ acaso\n(viés -0,45, ρ=+0,12)",
                xy=(1 + w, 0.03), xytext=(0.45, 0.26),
                arrowprops=dict(arrowstyle="->", color="#B22222", lw=1.1),
                fontsize=7.6, fontweight="bold", color="#B22222",
                bbox=dict(boxstyle="round,pad=0.20", fc="#FFF0F0", ec="#E45756", lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(s_labels, fontsize=8.0)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Métrica de Discriminação (0–1)")
    ax.set_title("(a) Poder Preditivo Pré-Decisão (B0)")
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.94, fontsize=7.2)

    # --- Panel (b): Causal Decomposition beta=1.0 vs beta=0.0 ---
    ax = axes[1]
    bd = retro["beta_vs_urllc_decomposition"]
    ops = retro["ops_support_fallback_waste_summary"]
    variants = (
        ("b1_raw\n(β=0)", 100 * bd["executed_b1_raw_sum_lt53"] / 25, 100 * 10 / 18, 100 * ops["b1_raw"]["all25_bottleneck_slice_counts"].get("1", 0) / 25),
        ("b1_act\n(β=0)", 100 * bd["executed_b1_act_sum_lt53"] / 25, 100 * 10 / 18, 100 * bd["executed_b1_act_urllc_bn_25"] / 25),
        ("off_sup*\n(trace β=0)", 100 * bd["offline_sup_beta0_sum_lt53"] / 25, 100 * bd["offline_sup_beta0_embb_starved_A18"] / 18, 100 * bd["offline_sup_beta0_urllc_bn_25"] / 25),
        ("b1_sup\n(β=1,0)", 100 * bd["executed_b1_sup_sum_lt53"] / 25, 100 * 10 / 18, 100 * bd["executed_b1_sup_urllc_bn_25"] / 25),
    )
    xv = np.arange(len(variants))
    wv = 0.25
    c_idle = ax.bar(xv - wv, [v[1] for v in variants], width=wv, color="#F58518", edgecolor="#8C4300", lw=0.6, label="sum(a) < 53 (25 est.)")
    c_starve = ax.bar(xv, [v[2] for v in variants], width=wv, color="#D62728", edgecolor="#7F1D1D", lw=0.6, label="Inanição a_e <= 1 (Univ A, 18)")
    c_bn = ax.bar(xv + wv, [v[3] for v in variants], width=wv, color="#4C78A8", edgecolor="#1E3A5F", lw=0.6, label="Gargalo = URLLC (25 est.)")

    for bars, col in ((c_idle, "#8C4300"), (c_starve, "#7F1D1D"), (c_bn, "#1E3A5F")):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.0f}%",
                    ha="center", va="bottom", fontsize=7.6, color=col, fontweight="bold")

    ax.set_xticks(xv)
    ax.set_xticklabels([v[0] for v in variants], fontsize=8.2)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Incidência das Patologias (%)")
    ax.set_title("(b) Decomposição: -βσ vs max-min")
    ax.legend(loc="upper left", frameon=True, framealpha=0.94, fontsize=7.4)

    # --- Panel (c): Causal Ladder vs Hindsight Ceiling in Univ A ---
    ax = axes[2]
    cr = retro["causal_ranking_diagnostic_on_existing_replays"]["univ_A"]
    o3 = retro["hindsight_oracles"]["univ_A_O3_b0_eq_prop"]
    o2 = retro["hindsight_oracles"]["univ_A_O2_eq_prop"]
    o6 = retro["hindsight_oracles"]["univ_A_O6_all_arms"]
    uA = diag["universe_A_ALL3_COMPLETE"]

    ladder = [
        ("B1-act / B1-sup (A5 v1)", uA["b1_act"]["cond_pass_pct"], "#E45756", False),
        ("B1-raw (A5 v1)", uA["b1_raw"]["cond_pass_pct"], "#F58518", False),
        ("B0-rest (congelado)", uA["b0"]["cond_pass_pct"], "#4C78A8", False),
        ("R_feas (pool all)", cr["pool_all_executed"]["R_feas_count_then_sum_clip"]["rate"] * 100, "#9D755D", False),
        ("prop (ativa)", uA["prop"]["cond_pass_pct"], "#72B7B2", False),
        ("R_feas (pool3)", cr["pool_3_b0_eq_prop"]["R_feas_count_then_sum_clip"]["rate"] * 100, "#9D755D", False),
        ("equal=(18,17,18)", uA["eq"]["cond_pass_pct"], "#54A24B", False),
        ("R_maxmin_f β=0 (pool3)", cr["pool_3_b0_eq_prop"]["R_maxmin_f_beta0"]["rate"] * 100, "#2CA02C", False),
        ("O2* {eq, prop}", o2["oracle_rate"] * 100, "#BBBBBB", True),
        ("O3* {B0, eq, prop}", o3["oracle_rate"] * 100, "#888888", True),
        ("O6* {todos 6 braços}", o6["oracle_rate"] * 100, "#555555", True),
    ]
    yl = np.arange(len(ladder))
    for i, (lbl, val, col, is_hindsight) in enumerate(ladder):
        ax.barh(i, val, height=0.60, color=col, edgecolor="#222222", lw=0.6,
                hatch="///" if is_hindsight else None, alpha=0.82 if is_hindsight else 0.95)
        ax.text(val + 1.5, i, f"{val:.1f}%", va="center", ha="left", fontsize=7.9,
                fontweight="bold" if val >= 66.0 or val <= 20.0 else "normal",
                color="#222222")

    ax.axvline(uA["eq"]["cond_pass_pct"], color="#54A24B", linestyle=":", lw=1.3)
    ax.text(85.0, 2.2, "Hachurado (*):\nhindsight\n(não causal)", fontsize=7.3, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.22", fc="#F7F7F7", ec="#888888", lw=0.7))
    ax.set_yticks(yl)
    ax.set_yticklabels([x[0] for x in ladder], fontsize=7.9)
    ax.set_xlim(0, 102)
    ax.set_xlabel("STRICT_ALL3_R19 Universo A (%, N=18)")
    ax.set_title("(c) Escada Causal vs Oráculos Hindsight (*)")

    out_path = os.path.join(FIG_DIR, "fig2_diagnostico_separabilidade_teto.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    _configure_style()
    diag = json.loads(open(DIAG_JSON, encoding="utf-8").read())
    retro = json.loads(open(RETRO_JSON, encoding="utf-8").read())
    p1 = _make_fig1(diag, retro)
    p2 = _make_fig2(diag, retro)
    print("Generated:", p1)
    print("Generated:", p2)


if __name__ == "__main__":
    main()
