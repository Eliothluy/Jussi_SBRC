#!/usr/bin/env python
"""Driver of the Etapa-2.4 offline oracle — Despacho I (protocol §33 items 1-6).

Builds the per-window dataset over the 30 frozen Etapa-2.3 runs, fits the KPI
models (leave-one-seed-out validated), enumerates the action simplex (C=53,
step 1, timed), classifies every window (F_t, |F_t|, best degradation a*),
produces the §12 metrics, the §20 tables, the §21 figures with the §28 names,
the SC1-SC4 sanity checks, the H2.4-1..4 hypotheses and the §15 candidate
list of critical windows, and writes the partial report.

NO ns-3 simulation is launched. ``executable_validation/`` stays for
Despacho II.  Usage:

    .venv/bin/python examples/sqr_oracle_despacho1.py [--round2 DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_GYM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_GYM_ROOT, "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nsoran.sqr.oracle import dataset as ds  # noqa: E402
from nsoran.sqr.oracle.models import EMMB_SHARE_EDGES  # noqa: E402
from nsoran.sqr.oracle.pipeline import (  # noqa: E402
    WITNESS_URLLC_RBGS,
    build_all,
    oracle_actions_table,
    oracle_windows_extra_table,
    oracle_windows_table,
)
from nsoran.sqr.oracle.simplex import C_PRB  # noqa: E402

DEFAULT_ROUND2 = os.path.join(
    _GYM_ROOT, "results", "sqr_characterization_round2")
DEFAULT_OUT = os.path.join(DEFAULT_ROUND2, "oracle")

ARM_LABEL = {"equal": "equal-split", "prop": "prop"}


# ---------------------------------------------------------------------------
# figures (protocol §21; file names exactly §28)
# ---------------------------------------------------------------------------

def _mark(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def fig_pinf_full(classified: pd.DataFrame, path: str) -> None:
    """Figura 1 (final) — P_inf × carga por braço e agregado, com IC binomial 95%."""
    from math import sqrt

    def wilson(p, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan)
        den = 1 + z**2 / n
        c = (p + z**2 / (2 * n)) / den
        hw = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
        return max(0.0, c - hw), min(1.0, c + hw)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    series = {"agregado": ("k", "o-")}
    for arm, color in [("equal", "tab:blue"), ("prop", "tab:red")]:
        series[ARM_LABEL[arm]] = (color, "s--")
    for label, (color, style) in series.items():
        xs, ys, lo, hi = [], [], [], []
        for lam, g in (classified.groupby("lam") if label == "agregado"
                       else classified[classified["arm"] ==
                                       {v: k for k, v in ARM_LABEL.items()}[label]].groupby("lam")):
            n = len(g)
            p = float((~g["joint_feasible"]).mean())
            l, h = wilson(p, n)
            xs.append(lam); ys.append(p); lo.append(l); hi.append(h)
        ax.plot(xs, ys, style, color=color, label=label)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
    ax.set_ylim(-0.02, 1.02)
    _mark(ax, "Figura 1 — $P_{inf}(\\lambda)$: probabilidade de estado conjuntamente inviável",
          "$\\lambda_{total}$ [UE/min]", "$P(\\mathcal{F}_t=\\emptyset)$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_fset(classified: pd.DataFrame, path: str) -> None:
    """Figura 2 — |F_t| × carga (mediana, p25/p75)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    lams = sorted(classified["lam"].unique())
    med, p25, p75 = [], [], []
    for lam in lams:
        f = classified[(classified["lam"] == lam) & classified["joint_feasible"]][
            "num_feasible_actions"]
        med.append(f.median() if len(f) else np.nan)
        p25.append(f.quantile(0.25) if len(f) else np.nan)
        p75.append(f.quantile(0.75) if len(f) else np.nan)
    ax.plot(lams, med, "o-", color="tab:green", label="mediana de $|\\mathcal{F}_t|$")
    ax.fill_between(lams, p25, p75, color="tab:green", alpha=0.2, label="p25–p75")
    ax.set_yscale("log")
    _mark(ax, "Figura 2 — Tamanho do conjunto factível $|\\mathcal{F}_t|$ × carga",
          "$\\lambda_{total}$ [UE/min]",
          f"$|\\mathcal{{F}}_t|$ (simplex $\\leq {C_PRB}$, passo 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_avr(by_lambda: pd.DataFrame, path: str) -> None:
    """Figura 3 — AVR × carga, por braço (global + por slice)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    ax = axes[0]
    for arm, color in [("equal", "tab:blue"), ("prop", "tab:red")]:
        ax.plot(by_lambda["lambda_total"], by_lambda[f"AVR_{arm}"], "o-",
                color=color, label=f"AVR {ARM_LABEL[arm]}")
    _mark(ax, "Figura 3a — AVR global × carga", "$\\lambda_{total}$ [UE/min]",
          "AVR = P($V_t$=1 | $\\mathcal{F}_t\\neq\\emptyset$)")
    ax.legend()
    ax = axes[1]
    styles = {"embb": ("o-", 1.0), "urllc": ("s--", 1.0), "mmtc": ("^:", 1.0)}
    for arm, color in [("equal", "tab:blue"), ("prop", "tab:red")]:
        for sl, (style, _) in styles.items():
            ax.plot(by_lambda["lambda_total"], by_lambda[f"AVR_{arm}_{sl}"],
                    style, color=color, alpha=0.9,
                    label=f"{ARM_LABEL[arm]} · {sl}" if sl != "mmtc" else None)
    _mark(ax, "Figura 3b — AVR por slice × carga", "$\\lambda_{total}$ [UE/min]", "")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_deficit(classified: pd.DataFrame, path: str) -> None:
    """Figura 4 — severidade mínima D_min nas janelas inviáveis."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    lams = sorted(classified["lam"].unique())
    data = [classified[(classified["lam"] == lam) & ~classified["joint_feasible"]][
        "min_total_deficit"].values for lam in lams]
    bp = ax.boxplot(data, tick_labels=[str(int(l)) for l in lams],
                    showfliers=False, patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor("tab:orange")
        box.set_alpha(0.6)
    for lam, d in zip(lams, data):
        if len(d):
            ax.scatter([lam] * len(d), d, s=4, color="tab:orange", alpha=0.25, zorder=1)
    frac = [float((~classified[classified["lam"] == lam]["joint_feasible"]).mean())
            for lam in lams]
    for lam, f in zip(lams, frac):
        ax.text(lam, 1.03, f"{f:.0%}", ha="center", fontsize=8, color="gray")
    ax.set_ylim(-0.02, 1.12)
    _mark(ax, "Figura 4 — Severidade mínima possível $D_{min}$ (janelas inviáveis;\n"
              "% = fração de janelas inviáveis no λ)",
          "$\\lambda_{total}$ [UE/min]", "$D_{min} = \\min_a \\sum_i D_i(a)$ (w=1)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_state_map(classified: pd.DataFrame, path: str) -> None:
    """Figura 5 — mapa de estados N_embb × N_urllc (e N_total × fila)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    rng = np.random.default_rng(0)
    for ax, xcol, ycol, xlabel, ylabel in [
            (axes[0], "n_embb", "n_urllc", "$N_{eMBB}$ (com demanda)",
             "$N_{URLLC}$ (com demanda)"),
            (axes[1], "n_total", "queue_total", "$N_{total}$ (com demanda)",
             "fila total [B] (escala log)")]:
        d = classified.copy()
        if xcol == "n_total":
            d["n_total"] = d["n_embb"].fillna(0) + d["n_urllc"].fillna(0) + d["n_mmtc"].fillna(0)
            d["queue_total"] = d["queue_embb"].fillna(0) + d["queue_urllc"].fillna(0) \
                + d["queue_mmtc"].fillna(0)
        x = d[xcol].values.astype(float)
        y = d[ycol].values.astype(float)
        if ycol.startswith("queue"):
            y = np.log10(np.maximum(y, 1.0))
        if xcol == "n_embb":
            jit = rng.uniform(-0.18, 0.18, len(d))
        else:
            jit = np.zeros(len(d))
        cls = np.where(~d["joint_feasible"], "inviável",
                       np.where(d["num_feasible_actions"] <=
                                d.loc[d["joint_feasible"], "num_feasible_actions"]
                                .quantile(1 / 3), "marginal", "viável"))
        for cat, color in [("viável", "tab:green"), ("marginal", "tab:blue"),
                           ("inviável", "tab:red")]:
            m = cls == cat
            ax.scatter(x[m] + jit[m], y[m], s=3, alpha=0.25, color=color, label=cat)
        _mark(ax, "Figura 5 — Mapa de estados (cor = classe do oracle)",
              xlabel, ylabel)
        ax.legend(fontsize=8, markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_pareto(sweep: pd.DataFrame, path: str) -> None:
    """Figura 6 — trade-off eMBB × URLLC para janelas representativas.

    Um ponto por alocação candidata (varredura de a_e; URLLC constante no
    modelo calibrado), destacando as ações executadas equal/prop e a testemunha
    do oracle. O contraste observado entre as classes executadas (equal vs
    prop no mesmo estrato) entra como referência empírica.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    for ax, lam in zip(axes, sorted(sweep["lam"].unique())[-2:]):
        s = sweep[sweep["lam"] == lam]
        for sample_id, gs in s.groupby("sample_id"):
            gs = gs.sort_values("thr_embb_hat")
            ax.plot(gs["thr_embb_hat"], gs["deadline_miss_urllc_hat_pct"], "-",
                    alpha=0.65, label=f"janela {int(gs['win'].iloc[0])} ({gs['arm_sampled'].iloc[0]})")
            ex = gs[gs["point_kind"] == "executed"]
            if len(ex):
                ax.scatter(ex["thr_embb_hat"], ex["deadline_miss_urllc_hat_pct"],
                           marker="s", color="tab:red", zorder=5,
                           label="executada (modelo)")
            ax.scatter(gs["thr_embb_obs"].iloc[0], gs["p95_urllc_obs"].iloc[0],
                       marker="x", color="gray", zorder=5)
        _mark(ax, f"Figura 6 — Trade-off eMBB × URLLC (λ={int(lam)}; × = ponto observado;\n"
                  "curva do modelo: URLLC plano (ação-independente), eMBB cresce com a_e)",
              "throughput eMBB previsto por UE [Mbps]",
              "deadline-miss URLLC previsto [%]")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_episode(classified: pd.DataFrame, tag: str, t0: float, t1: float,
                path: str, arm: str = "prop") -> None:
    """Figura 7 — episódio temporal (N, fila, alocação executada, SLA, F_t).

    ``arm`` selects the run: the tag alone exists in BOTH arms' campaigns
    (same tag naming), and overlaying them would zigzag the time series.
    """
    g = classified[(classified["tag"] == tag) & (classified["arm"] == arm) &
                   (classified["time_ms"] >= t0 * 1000) &
                   (classified["time_ms"] < t1 * 1000)].sort_values("time_ms")
    if not len(g):
        return
    t = g["time_ms"] / 1000.0
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    ax = axes[0]
    ax.step(t, g["n_embb"].fillna(0), where="post", label="$N_{eMBB}$", color="tab:blue")
    ax.step(t, g["n_urllc"].fillna(0), where="post", label="$N_{URLLC}$", color="tab:red")
    ax.step(t, g["n_mmtc"].fillna(0), where="post", label="$N_{mMTC}$", color="tab:green")
    ax.legend(fontsize=8, ncol=3)
    _mark(ax, f"Episódio ({tag})", "", "UEs com demanda")
    ax = axes[1]
    ax.step(t, g["queue_embb"].fillna(0), where="post", color="tab:blue", label="fila eMBB")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    _mark(ax, "", "", "fila [B]")
    ax = axes[2]
    ax.step(t, g["a_e"], where="post", color="tab:blue", label="$a_{eMBB}$ executado")
    ax.step(t, g["a_u"], where="post", color="tab:red", label="$a_{URLLC}$")
    ax.step(t, g["e_min"], where="post", color="k", ls="--", label="$e_{min}$ (oracle)")
    ax.set_ylim(0, C_PRB + 1)
    ax.legend(fontsize=8, ncol=3)
    _mark(ax, "", "", "cota [RBG]")
    ax = axes[3]
    ax.step(t, g["observed_embb_violation"].fillna(0), where="post",
            color="tab:blue", label="viol. eMBB obs.")
    ax.step(t, g["observed_urllc_violation"].fillna(0), where="post",
            color="tab:red", label="viol. URLLC obs.")
    ax.step(t, ~g["joint_feasible"], where="post", color="k", ls="--",
            label="$\\mathcal{F}_t=\\emptyset$ (oracle)")
    ax.set_ylim(-0.1, 1.4)
    ax.legend(fontsize=8, ncol=3)
    _mark(ax, "", "", "indicador")
    ax = axes[4]
    ax.step(t, g["num_feasible_actions"], where="post", color="tab:purple")
    ax.set_yscale("log")
    _mark(ax, "", "tempo [s]", "$|\\mathcal{F}_t|$")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# trade-off sweep (§11 / fig 6 / oracle_tradeoff_samples.csv)
# ---------------------------------------------------------------------------

def tradeoff_sweep(models: dict, sample_windows: pd.DataFrame) -> pd.DataFrame:
    """Candidate-allocation sweep for the sampled windows (protocol §11 list)."""
    rows = []
    for _, w in sample_windows.iterrows():
        n_e, n_u, n_m = int(w["n_embb"]), int(w["n_urllc"]), int(w["n_mmtc"])
        cqi = float(w["cqi_eff"])
        p95_hat, miss_hat = models["urllc"].predict([n_u], [n_e])
        p95_hat, miss_hat = float(p95_hat[0]), float(miss_hat[0])
        for a_e in range(0, C_PRB + 1):
            a_u = min(WITNESS_URLLC_RBGS, C_PRB - a_e)
            a_m = C_PRB - a_e - a_u
            thr_hat, _supp = models["embb"].predict_with_support(
                np.array([a_e], dtype=float), np.array([n_e], dtype=float),
                np.array([cqi], dtype=float))
            thr_hat = float(thr_hat[0])
            kind = "candidate"
            if (a_e, a_u) == (int(w["a_e_exec"]), int(w["a_u_exec"])):
                kind = "executed"
            rows.append({
                "lam": w["lam"], "sample_id": f"{w['tag']}_w{int(w['win'])}",
                "arm_sampled": w["arm"], "seed": w["seed"], "win": int(w["win"]),
                "tag": w["tag"], "n_embb": n_e, "n_urllc": n_u, "n_mmtc": n_m,
                "allocation": f"({a_e},{a_u},{a_m})",
                "thr_embb_hat": thr_hat,
                "deadline_miss_urllc_hat_pct": miss_hat,
                "pdr_mmtc_hat": 1.0,
                "thr_embb_obs": w["thr_embb_obs"],
                "p95_urllc_obs": w["p95_urllc_obs"],
                "point_kind": kind,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Markdown table WITHOUT tabulate (the frozen venv has no tabulate)."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
        else:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else str(v))
    cols = [str(c) for c in d.columns]
    rows = d.astype(str).values.tolist()
    def _row(cells):
        return "| " + " | ".join(cells) + " |"
    out = [_row(cols), _row(["---"] * len(cols))]
    out += [_row(r) for r in rows]
    return "\n".join(out)


def write_report(art: dict, out_dir: str, args) -> None:
    """reports/relatorio_oracle_2_4_despacho1.md (Despacho I, §33 itens 1-6)."""
    cl = art["classified"]
    bl = art["by_lambda"]
    bs = art["by_slice"]
    sc, hyp, loo = art["sc"], art["hyp"], art["loo"]
    timing = art["timing"]

    bl_show = bl[["lambda_total", "P_inf", "P_feas", "AVR_equal", "AVR_prop",
                  "fset_median", "fset_p25", "fset_p75", "Dmin_median_infeasible",
                  "regime"]].rename(columns={
        "lambda_total": "λ", "P_inf": "P_inf", "P_feas": "P_feas",
        "AVR_equal": "AVR equal", "AVR_prop": "AVR prop",
        "fset_median": "|F_t| med", "fset_p25": "|F_t| p25",
        "fset_p75": "|F_t| p75", "Dmin_median_infeasible": "D_min med (inviáveis)",
        "regime": "Regime"})
    bs_show = bs.rename(columns={
        "lambda_total": "λ", "AVR_embb": "AVR eMBB", "AVR_urllc": "AVR URLLC",
        "AVR_mmtc": "AVR mMTC",
        "saveable_embb_rate_infeasible": "saveable eMBB",
        "saveable_urllc_rate_infeasible": "saveable URLLC",
        "saveable_mmtc_rate_infeasible": "saveable mMTC"})

    crit = art["critical"]
    crit_counts = crit.groupby(["priority_class", "lam"]).size().unstack(fill_value=0) \
        if len(crit) else pd.DataFrame()

    lines = []
    add = lines.append
    sc2_p25 = float(bl.loc[bl["lambda_total"] == 2.0, "P_inf_gate_p25"].iloc[0]) \
        if (bl["lambda_total"] == 2.0).any() else float("nan")
    add("# Etapa 2.4 — Oracle de viabilidade conjunta (DESPACHO I — relatório parcial)")
    add("")
    add(f"**Data:** 22/10/2026 · **Executor:** Agente B (oracle offline, sem ns-3) · "
        f"**Protocolo:** `docs/artigo/rodada_2_etapa_2_4_oracle_viabilidade.md` "
        f"(itens 1–6 do §33; item 7 = validação executável → DESPACHO II)")
    add("")
    add("**SLAs congelados (R36–R39, NÃO recalibrados):** eMBB `windowThroughputMbps` "
        "por UE-janela com demanda ≥ 3,5 Mbps; URLLC deadline por pacote 10 ms "
        "(`sqr_packet_delays.csv`); mMTC PDR ≥ 95%. **C = 53 PRB** (R01).")
    add("")
    add("---")
    add("## 0. Sumário executivo")
    add("")
    p2 = sc["SC2"]["P_inf_lambda2"]
    add(f"- **{timing['n_windows']:,} janelas** de 100 ms classificadas (30 runs × 2999), "
        f"simplex de ações enumerado com **passo 1** ({timing['enumeration']['n_actions']:,} ações, "
        f"Σ≤53; {timing['enumeration']['seconds_per_enumeration']*1000:.1f} ms por enumeração — "
        "o passo 1 cabe com folga, não há necessidade de passo 2).")
    add(f"- **P_inf(λ)** (agregado): " +
        ", ".join(f"λ{int(r.lambda_total)}={r.P_inf:.3f}" for r in bl.itertuples()) + ".")
    add(f"- **SC1 (tendência crescente): {'PASSA' if sc['SC1']['passed'] else 'FALHA'}**; "
        f"**SC2 (P_inf≈0 em λ2): {'PASSA' if sc['SC2']['passed'] else 'FALHA'}** "
        f"(P_inf(2)={p2:.3f} na porta mediana; {sc2_p25:.3f} na porta p25 — ver §4/§5: "
        "a inviabilidade remanescente em λ2 é URLLC-mecanismo + dispersão de mediana, "
        "não capacidade eMBB); **SC3 (coerência com braços): "
        f"{'PASSA' if sc['SC3']['passed'] else 'FALHA'}** ({sc['SC3']['inconsistencies']} "
        f"inconsistências em {sc['SC3']['windows_executed_all_sla_ok']} janelas OK, "
        f"{100*sc['SC3']['inconsistency_rate']:.1f}% — "
        f"{sc['SC3']['driven_by_urllc_gate_only']} pela porta URLLC mediana, "
        f"{sc['SC3']['driven_by_embb_gate_only']} só eMBB, "
        f"{sc['SC3']['driven_by_both']} ambas; ver §5); "
        f"**SC4 (rádio saudável nas inviáveis): {'PASSA' if sc['SC4']['passed'] else 'FALHA'}**.")
    for k in ("H2.4_1", "H2.4_2", "H2.4_3", "H2.4_4"):
        add(f"- **{k.replace('_', '-')}**: {hyp[k]['verdict']}.")
    add("")
    add("> **Leitura de abertura (importante):** sob os SLAs congelados, a fronteira "
        "conjunta é fechada em parte pelo **mecanismo URLLC do scheduler congelado** "
        "(a cauda de deadline cresce com N_urllc e responde pouco à cota dentro do suporte "
        "explorado — evidência 2.2/2.3), e em parte pela **capacidade eMBB** (cota vs "
        "N_eMBB×3,5 Mbps). O relatório separa os dois eixos e quantifica a evitabilidade "
        "condicional a cada um.")
    add("")
    add("---")
    add("## 1. Dataset por janela (item 1)")
    add("")
    add(f"- Fonte: 30 runs congelados da 2.3 (`hetero_sweep` = equal-split network-only; "
        "`hetero_sweep_prop` = prop IPC), 5 cargas × 3 seeds × 2 braços, 300 s, período 100 ms.")
    add(f"- Janela w = linha carimbada em (w+1)·100 ms descreve [100w, 100(w+1)) ms "
        "(alinhamento verificado no dreno de telemetria do `sqr-sim.cc`, ~1540–1640). "
        "Pacotes URLLC atribuídos por janela de CHEGADA (`win = floor(txSec·10)`), "
        "convenção do `analyze_urllc_tail.py`.")
    add("- **N_i (com demanda)**: UEs com `dTxBytes>0` na janela (a MESMA população de "
        "UE-janela do julgamento 2.3). Extras: `n_*_present` (linhas KPM) e "
        "`n_embb_backlog` (`queueBytes>0`).")
    add("- **Cargas oferecidas (proxies documentadas [ASSUMIDO-ORACLE R40])**: eMBB greedy "
        "com fila ⇒ oferece o alvo R36 (3,5 Mbps/UE); URLLC 32 B/1 ms ⇒ 0,256 Mbps/UE; "
        "mMTC 200 B/1 s ⇒ 0,0016 Mbps/UE (R35).")
    add("- **Ações executadas**: `prb_executed_rbg_i` quando houve ação (IPC; braço prop) e "
        "`prb_budget_rbg_i / dl_slots` no modo network-only (sentinela −1/0 sem ação; "
        "integralidade 100% verificada nos 15 runs equal — a cota é recuperada EXATA). "
        "equal executa **(18,17,18)** constante; prop executa **(37,13,3)** com demanda eMBB "
        "(36859 janelas) e **(3,13,3)** sem (24 janelas) — o suporte de ações observado é "
        "MINÚSCULO (2 classes), ver §3.")
    add("- **util_active_dl (D2)** = 100·Σ_i sched_i/(C·dl_slots) nas janelas com slots; "
        "**util_all_dl (D1)** = 100·Σ_i sched_i/(C·(400/7)) — definições congeladas da "
        "análise 2.3. `prb_scheduled/budget/idle` são ACUMULADORES RBG·slot da janela "
        "(não contagens de um slot).")
    add(f"- Integridade: as agregadas por run deste dataset REPRODUZEM o CSV da 2.3 "
        f"(`round2_hetero_sweep.csv`) — ver teste `test_sqr_oracle.py::test_dataset_reproduces_23_run_level`.")
    add("")
    add("## 2. Modelos de KPI (item 2) — formas, calibração, validação e suporte")
    add("")
    add("### 2.1 f_eMBB — lookup por estrato (CQI × participação por UE)")
    add("")
    add("```")
    add("thr_por_UE ≈ T(cqi_bin, s_bin),  s = a_e / max(N_e,1)  [Mbps/UE]")
    add("```")
    add("- **Justificativa semi-analítica**: o scheduler serve a cota inteira da slice "
        "backlogged (utilização quota = 1,0 medida no braço equal), logo o throughput "
        "celular entregue é ditado pela cota; o por-UE segue a participação s; e o RENDIMENTO "
        "por RBG degrada quando s colapsa (medido: yield 0,41→0,18 Mbps/RBG ao ir s 9→4,5 "
        "no equal — daí LOOKUP em s, não eficiência constante).")
    add("- **Features**: (cqi_bin do CQI wideband mediano da janela; s_bin com bordas "
        f"{[e if e < 1e8 else '∞' for e in EMMB_SHARE_EDGES]}). "
        "**Target**: `thr_embb_med` (mediana por UE-janela com demanda = quantidade do "
        "julgamento R36). **Split**: leave-one-seed-out (treino nas 2 seeds restantes, "
        "todas as cargas, 2 braços agregados).")
    add("- **Extrapolação**: célula (cqi,s) sem treino → mediana marginal do s_bin; s fora "
        "da faixa calibrada → clamp na borda (saturação de capacidade) [ASSUMIDO-ORACLE R42]; "
        "flag `thr_supp` marca janelas preditas fora do suporte.")
    add("")
    add("### 2.2 f_URLLC — lookup por estrato (N_urllc × N_embb), INDEPENDENTE de ação")
    add("")
    add("```")
    add("p95_janela ≈ G[N_u_bin, N_e_bin];  miss% ≈ idem  (miss>10 ms por pacote, R37)")
    add("```")
    add("- **Justificativa empírica**: a cauda de deadline é dominada pela contenção "
        "intra-slice (N_urllc: p95 mediano 8–9 ms com 1 UE → 11–12 com 2 → 13–14 com 3+); "
        "o efeito de classe de ação (equal (18,17,18) vs prop (37,13,3)) a estratos fixos é "
        "≤ ~2 ms de p95; e o experimento pareado do gate v2 (R39 v1.7-cal2) mostrou a cauda "
        "NÃO limitada por cota em baixa carga. **Logo o modelo primário não credita melhoria "
        "URLLC a nenhuma alocação** — escolha CONSERVADORA para evitabilidade "
        "[ASSUMIDO-ORACLE R43]. Consequência formal: o eixo URLLC entra como um GATE do "
        "estado z (p95_hat ≤ 10), não da ação.")
    add("")
    add("### 2.3 f_mMTC — PDR = 1,0 constante")
    add("")
    add("- Medido: PDR agregado mMTC = 1,0 em TODOS os 30 runs até λ7 (0,0016 Mbps/UE, "
        "ordens abaixo da capacidade; R38 nunca próximo). O modelo nunca liga F_t. "
        "Menor PDR de janela observado é exportado no model card para honestidade.")
    add("")
    add("### 2.4 Validação leave-one-seed-out (erro global, erro perto do limiar, coerência)")
    add("")
    add(_md_table(loo["loo"], "{:.4f}"))
    add("")
    cons = art["loo"]["consistency"]
    add(f"Coerência executado→oracle nos modelos LOO (janelas em que o braço executado "
        f"satisfez TODOS os SLAs e o modelo da seed retida confirma F_t≠∅ — análogo "
        f"offline da matriz §16; a mesma métrica para o oracle FINAL calibrado em toda "
        f"a base é o SC3 do §5): "
        f"{int(cons['predicted_feasible'].sum())}/{int(cons['windows_all_ok_executed'].sum())} "
        f"({100*cons['consistency_rate'].mean():.2f}%; false-infeasible total: "
        f"{int(cons['false_infeasible'].sum())}).")
    add("")
    add("### 2.5 Região de suporte das ações (§6/§16)")
    add("")
    add("- **Suporte observado**: equal ≈ constante (18,17,18); prop ∈ {(37,13,3) com demanda "
        "eMBB; (3,13,3) sem}. Combinações do simplex como (25,20,8) ou (10,40,3) estão "
        "TOTALMENTE FORA do suporte de AÇÕES (só 2 classes executadas existem nos dados).")
    add("- **Como o modelo lida**: (i) f_eMBB extrapola por s = a_e/N_e (variável física "
        "coberta de ~0,4 a 37 RBG/UE pelos dois braços) com clamp nas bordas; (ii) f_URLLC "
        "mantém o nível do estrato (não credita nem pune cota); (iii) toda predição carrega "
        "flag de suporte e o model card lista a cobertura por célula; (iv) a classificação "
        "de viabilidade em células fora do suporte fica MARCADA para a validação executável "
        "do Despacho II (§15/§16 do protocolo).")
    add("")
    add("## 3. Simplex + F_t (item 3)")
    add("")
    add(f"- Enumeração **passo 1**: {timing['enumeration']['n_actions']:,} ações "
        f"(combinações C(56,3)=27720 ✓, Σa≤53 ✓, ≥0 ✓; "
        f"{timing['enumeration']['seconds_per_enumeration']*1000:.1f} ms/enumeração → passo 1 mantido).")
    add("- Para cada janela: **F_t** = {a : SLA_e(a,z) ∧ SLA_u(z) ∧ SLA_m(a,z)}; "
        "|F_t| contado sobre o simplex 3D completo; **testemunha factível** = menor a_e "
        "factível (e_min) + URLLC no piso do suporte (13 RBG, R39) + resto em mMTC "
        "[ASSUMIDO-ORACLE R44]; quando F_t=∅: **a*** = argmin Σ_i D_i com **w=1 NEUTRO** "
        "(desempate: primeira ação em ordem lexicográfica; a degenerescência ao longo de "
        "(a_u,a_m) é inerente ao modelo URLLC-ação-independente e é reportada por janela "
        "em `best_deg_n_ties`), com **D_min** e déficits por slice (§9).")
    add("")
    add("## 4. Métricas e tabelas (item 4)")
    add("")
    add("### Tabela geral (§20-1)")
    add("")
    add(_md_table(bl_show))
    add("")
    add("### Sensibilidade da porta de KPI (§16 — dispersão mediana vs janela)")
    add("")
    add("As portas de SLA usam a MEDIANA do estrato (primário). A banda p25/p75 quantifica "
        "a incerteza da fronteira face à dispersão janela-a-janela dos KPIs (o "
        "false-infeasible do §16 mora aqui; a validação executável do Despacho II "
        "dirá qual extremo a realidade favorece):")
    add("")
    sens = bl[["lambda_total", "P_inf_gate_p25", "P_inf", "P_inf_gate_p75"]]
    sens.columns = ["λ", "P_inf (porta p25, otimista)", "P_inf (porta mediana, PRIMÁRIO)",
                    "P_inf (porta p75, pessimista)"]
    add(_md_table(sens))
    add("")
    add("### Tabela por slice (§20-2)")
    add("")
    add(_md_table(bs_show))
    add("")
    add("### UVR (violação em estados inviáveis, §12.4) e D_min")
    add("")
    uvr = bl[["lambda_total", "UVR_equal", "UVR_prop", "n_infeasible",
              "Dmin_mean_infeasible", "Dmin_median_infeasible"]]
    add(_md_table(uvr))
    add("")
    add("## 5. Sanity checks SC1–SC4 (item 5a)")
    add("")
    for k in ("SC1", "SC2", "SC3", "SC4"):
        d = sc[k]
        verdict = "APROVADO" if d["passed"] else "REFUTADO"
        add(f"- **{k}: {verdict}** — " + "; ".join(
            f"{kk}={vv}" for kk, vv in d.items()
            if kk not in ("passed", "note", "inconsistent_example_rows")
            and not isinstance(vv, dict)) + f". {d.get('note','')}")
    add("")
    add("## 6. Hipóteses H2.4-1..4 (item 5b)")
    add("")
    for k in ("H2.4_1", "H2.4_2", "H2.4_3", "H2.4_4"):
        add(f"- **{k.replace('_', '-')} — {hyp[k]['verdict']}** — {hyp[k]['claim']}")
        for kk, vv in hyp[k].items():
            if kk not in ("claim", "verdict") and not isinstance(vv, dict):
                add(f"  - {kk}: {vv}")
            elif isinstance(vv, dict):
                add(f"  - {kk}: " + ", ".join(f"{a}={b}" for a, b in vv.items()))
    add("")
    add("## 7. Figuras (item 4; nomes exatos do §28)")
    add("")
    add("Todos na RAIZ de `oracle/` (caminhos relativos à lista plana do §28): "
        "`fig_oracle_pinf.png` (F1), `fig_oracle_fset.png` (F2), `fig_oracle_avr.png` "
        "(F3), `fig_oracle_deficit.png` (F4), `fig_oracle_state_map.png` (F5), "
        "`fig_oracle_pareto.png` (F6, amostra representativa do trade-off eMBB×URLLC), "
        "`fig_oracle_episode.png` (F7 — episódio temporal; nome estendido pois o §28 não "
        "lista arquivo para a figura 7). Os CSVs §28 (`oracle_windows.csv` na ordem exata "
        "do §19, `oracle_actions.csv`, `oracle_by_lambda.csv`, `oracle_by_slice.csv`, "
        "`oracle_tradeoff_samples.csv`) também estão na raiz; os EXTRAS ficam em "
        "`tables/` (`oracle_windows_extra.csv`, `oracle_critical_windows.csv`).")
    add("")
    add("## 8. Janelas críticas candidatas à validação executável (item 7 → Despacho II, §15)")
    add("")
    add(f"CSV: `tables/oracle_critical_windows.csv` ({len(crit)} linhas, classes de "
        "prioridade por (λ, seed) com braços PAREADOS pela seed):")
    add("")
    if len(crit_counts):
        add(_md_table(crit_counts.reset_index().rename(columns={"index": "priority_class"})))
    else:
        add("(sem candidatos)")
    add("")
    add("Classes: `A_equal_viola_prop_nao` (§15 'equal viola e prop não'); "
        "`B_prop_viola_urllc`; `C_ambos_violam`; `D_oracle_inviavel` (F_t=∅); "
        "`E_fset_pequeno` (p25 de |F_t|). Cada linha traz o estado z_t, as ações "
        "executadas, a testemunha do oracle e a* — receita de replay do §15 "
        "(snapshot → restaurar → executar → observar; limitações documentadas: o "
        "cenário não tem snapshot de estado; alternativas §15.1–5 a decidir no Despacho II).")
    add("")
    add("## 9. Premissas novas registradas (todas [ASSUMIDO-ORACLE], pendentes de ratificação)")
    add("")
    add("| ID | Premissa | Onde |")
    add("|---|---|---|")
    add("| R40 | Proxies de carga oferecida (greedy→alvo R36; URLLC 0,256; mMTC 0,0016 Mbps/UE) e N_i = UEs com dTxBytes>0 | dataset.py / §1 |")
    add("| R41 | Julgamento de janela: eMBB = mediana dos UE-janelas < 3,5; URLLC = p95 da janela > 10 ms (≡ miss>5%); mMTC = PDR de janela < 0,95 (NaN sem pacotes) | dataset.py / R36–R38 |")
    add("| R42 | f_eMBB = lookup (cqi, s) com clamp nas bordas; f_URLLC = lookup (N_u, N_e) ação-independente (conservador); f_mMTC = 1,0 | models.py / §2 |")
    add("| R43 | Gate URLLC ação-independente (cauda não limitada por cota no suporte explorado — R39 v1.7-cal2 + estratos 2.3) | models.py / §2.2 |")
    add("| R44 | Testemunha factível = (e_min, min(13, C−e_min), resto); a* com w=1 neutro, desempate lexicográfico, degenerescência reportada | simplex.py / §3 |")
    add("| R45 | Alinhamento de janela ((w+1)·100 ms ⇒ [100w,100(w+1))) e ação executada (prb_executed IPC; budget/dl_slots network-only) | dataset.py / §1 |")
    add("| R46 | Limiar de regime: P_inf≤5% ⇒ normal/marginal (tercil superior de |F_t|); P_inf≥95% ⇒ severo; caso contrário transitório | pipeline.py / §18 |")
    add("")
    add("## 10. Escopo e limites deste despacho")
    add("")
    add("- **Nenhuma simulação ns-3 foi lançada** (oracle offline apenas sobre os dados "
        "congelados da 2.3); `executable_validation/` permanece vazio para o Despacho II.")
    add("- O gate de confiança do §16 (STOP se false-infeasible frequente / validação "
        "executável contradizer) é fechado no Despacho II; os insumos de decisão já "
        "produzidos aqui: erros LOO (§2.4), coerência executado→oracle, flags de suporte.")
    add("- Degenerescência conhecida do modelo URLLC (ação-independente): o eixo URLLC "
        "de F_t é um gate do estado; a leitura conjunta correta da fronteira exige os "
        "dois eixos separados (§0).")
    add("")
    header = "\n".join(lines)

    path = os.path.join(out_dir, "reports", "relatorio_oracle_2_4_despacho1.md")
    with open(path, "w") as f:
        f.write(header)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round2", default=DEFAULT_ROUND2)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    out = args.out
    for sub in ("models", "predictions", "tables", "reports"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    # executable_validation belongs to Despacho II — deliberately NOT created.

    art = build_all(args.round2)
    cl = art["classified"]

    # -- §28-named artifacts: FLAT at the oracle/ root. The protocol's file
    #    list ("Arquivos:") is root-relative and the campaign gate checks
    #    these exact paths; the subdirectories keep the NON-§28 extras
    #    (models/, predictions/, reports/, tables/*extra*).
    oracle_windows_table(cl).to_csv(os.path.join(out, "oracle_windows.csv"),
                                    index=False)
    oracle_windows_extra_table(cl).to_csv(
        os.path.join(out, "tables", "oracle_windows_extra.csv"), index=False)
    oracle_actions_table(cl).to_csv(os.path.join(out, "oracle_actions.csv"),
                                    index=False)
    art["by_lambda"].to_csv(os.path.join(out, "oracle_by_lambda.csv"), index=False)
    art["by_slice"].to_csv(os.path.join(out, "oracle_by_slice.csv"), index=False)
    art["critical"].to_csv(os.path.join(out, "tables", "oracle_critical_windows.csv"),
                           index=False)

    # -- models / predictions ------------------------------------------------
    m = art["models"]
    m["embb"].table.to_csv(os.path.join(out, "models", "embb_lookup.csv"), index=False)
    m["embb"].support.to_csv(os.path.join(out, "models", "embb_support.csv"), index=False)
    m["urllc"].table.to_csv(os.path.join(out, "models", "urllc_strata.csv"), index=False)
    with open(os.path.join(out, "models", "kpi_model_card.md"), "w") as f:
        f.write(model_card(art))
    art["loo"]["loo"].to_csv(os.path.join(out, "predictions", "oracle_loo_summary.csv"),
                             index=False)
    art["loo"]["consistency"].to_csv(
        os.path.join(out, "predictions", "oracle_loo_consistency.csv"), index=False)
    art["loo"]["loo_windows"].to_csv(
        os.path.join(out, "predictions", "oracle_loo_predictions.csv"), index=False)

    # -- trade-off sweep + fig 6 ---------------------------------------------
    sweep = tradeoff_sweep(m, art["tradeoff_windows"])
    sweep.to_csv(os.path.join(out, "oracle_tradeoff_samples.csv"), index=False)

    # -- figures (§28 names, flat at the root) --------------------------------
    fig_pinf_full(cl, os.path.join(out, "fig_oracle_pinf.png"))
    fig_fset(cl, os.path.join(out, "fig_oracle_fset.png"))
    fig_avr(art["by_lambda"], os.path.join(out, "fig_oracle_avr.png"))
    fig_deficit(cl, os.path.join(out, "fig_oracle_deficit.png"))
    fig_state_map(cl, os.path.join(out, "fig_oracle_state_map.png"))
    fig_pareto(sweep, os.path.join(out, "fig_oracle_pareto.png"))
    ep_tag = "A_lam6_s1"
    fig_episode(cl, ep_tag, t0=120, t1=180, arm="prop",
                path=os.path.join(out, "fig_oracle_episode.png"))

    # -- report ---------------------------------------------------------------
    rpt = write_report(art, out, args)

    print(f"[oracle] windows: {timing_line(art)}")
    print(f"[oracle] SC: " + ", ".join(
        f"{k}={'PASS' if v['passed'] else 'FAIL'}" for k, v in art["sc"].items()))
    print(f"[oracle] hyp: " + ", ".join(
        f"{k}={v['verdict']}" for k, v in art["hyp"].items()))
    print(f"[oracle] report: {rpt}")


def timing_line(art: dict) -> str:
    t = art["timing"]
    return (f"{t['n_windows']:,} janelas; dataset {t['dataset_build_s']:.1f}s; "
            f"classificação {t['classify_s']:.1f}s; "
            f"{t['n_unique_state_keys']} chaves de estado únicas; "
            f"enumeração {t['enumeration']['n_actions']:,} ações "
            f"({t['enumeration']['seconds_per_enumeration']*1000:.1f} ms)")


def _pivot_md(table: pd.DataFrame, index: str, columns: str, values: str) -> str:
    """Pivot + markdown without tabulate (model card helper)."""
    p = table.pivot(index=index, columns=columns, values=values)
    p = p.reset_index()
    p.columns = [f"{c}" for c in p.columns]
    return _md_table(p)


def model_card(art: dict) -> str:
    m = art["models"]
    loo = art["loo"]["loo"]
    lines = [
        "# Model card — oracle offline 2.4 (Despacho I)",
        "",
        "## f_eMBB (EmmbShareLookup)",
        f"- Células calibradas: {len(m['embb'].table)} (cqi_bin × s_bin); cobertura abaixo.",
        "- Extrapolação: marginal do s_bin para célula ausente (bins vazios recebem o "
        "vizinho suportado mais próximo — saturação plana); clamp de s fora da faixa.",
        "",
        "### Tabela (thr, Mbps por UE) — linhas = s_bin, colunas = cqi_bin",
        _pivot_md(m["embb"].table, "sb", "cb", "thr"),
        "",
        "### Suporte (n janelas de treino por célula)",
        _pivot_md(m["embb"].support, "sb", "cb", "n_windows"),
        "",
        "## f_URLLC (UrllcStrataLookup) — ação-independente",
        _md_table(m["urllc"].table),
        "",
        "## f_mMTC (MmtcConstantPdr)",
        f"- PDR = {m['mmtc'].pdr}; menor PDR de janela OBSERVADO: "
        f"{m['mmtc'].measured_min_pdr} (o modelo permanece 1.0; R38 julga PDR agregado).",
        "",
        "## Validação leave-one-seed-out (por seed retida)",
        _md_table(loo),
        "",
        "## Premissas [ASSUMIDO-ORACLE]: R40–R46 (ver assumptions_register.md e relatório §9).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
