#!/usr/bin/env python3
"""gera_figuras_tese.py — 8 figuras que defendem a tese do artigo SBRC 2027.

Tese: (i) adjudicação executável de violações; (ii) fronteira de viabilidade
sob estresse de demanda com rádio sadio; (iii) evitabilidade é função do
espaço de ações (achado central); (iv) cauda URLLC scheduler/TDD-dominada e o
piso do aprendizado seguro (G4→G4-A→G4-B, gates pré-registrados).

Fontes (artefatos congelados):
  R2      = MODA/ns-o-ran-gym/results/sqr_characterization_round2
  g2      = R2/oracle/adjudication/g2_deltas.csv                (5 evitáveis, II-c)
  e1w     = R2/oracle/oracle_recalibrated_v2/adjudication/confusion_witnesses_v2.csv (E1)
  g4c     = R2/oracle/adjudication/g4_urllc_curve.csv           (causal a_u, II-c G4)
  r1b     = MODA/ns-o-ran-gym/results/sqr_characterization/characterization_round1b_by_lambda.csv
  hetero  = R2/round2_hetero_sweep_by_lambda.csv
  g4bval  = results/sqr_jc_final/03_calibration/g4b/g4b_validation_windows.csv
  fronteira v2 (candidatas): oracle_recalibrated_v2/reports/relatorio_encerramento_rodada2.md
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = "/home/eliothluy/Documentos/Jussi_SBRC"
R2 = f"{ROOT}/MODA/ns-o-ran-gym/results/sqr_characterization_round2"
R1 = f"{ROOT}/MODA/ns-o-ran-gym/results/sqr_characterization"
OUT = f"{ROOT}/docs/artigo/figuras_tese"
os.makedirs(OUT, exist_ok=True)

C_F, C_M, C_I = "#2e7d32", "#f9a825", "#c62828"   # FEASIBLE/MARGINAL/INFEASIBLE
C_BLUE, C_GRAY = "#1565c0", "#616161"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def footer(fig, text):
    fig.text(0.01, 0.012, text, fontsize=6.5, color="#757575", va="bottom")


# ============================================================ Fig 1 — Fronteira
def fig1():
    lam = np.array([2, 4, 5, 6, 7], float)
    p_f = np.array([0.699, 0.446, 0.274, 0.241, 0.141])
    p_m = np.array([0.134, 0.220, 0.191, 0.175, 0.138])
    p_i = np.array([0.167, 0.334, 0.535, 0.584, 0.721])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.stackplot(lam, p_f, p_m, p_i,
                 labels=["FEASIBLE", "MARGINAL", "INFEASIBLE"],
                 colors=[C_F, C_M, C_I], alpha=0.85)
    ax.plot(lam, p_i, color=C_I, lw=2, marker="o", ms=4,
            label="P_I (inviável) — monotônica 0,17→0,72")
    ax.set_xlim(2, 7); ax.set_ylim(0, 1)
    ax.set_xticks(lam); ax.set_xlabel("carga λ (UE/min, total das 3 slices)")
    ax.set_ylabel("fração de janelas de decisão")
    ax.set_title("Fronteira de viabilidade sob estresse de demanda (candidatas v2)")
    ax.text(4.35, 0.06,
            "MARGINAL estável (13–22%): a incerteza\nconcentra-se na fronteira, não se espalha",
            fontsize=8, color="#8d6e00",
            bbox=dict(fc="#fff8e1", ec="#f9a825", alpha=0.9, pad=4))
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    footer(fig, "Fonte: oracle 3-way recalibrado v2 (89.970 janelas reclassificadas; P_F/P_M/P_I candidatos — gate v2 STOP "
                "por C3/C7).\nDuro e congelado: FIR=FFR=0 nas 27 testemunhas confiantes (replay determinístico, "
                "confusion_witnesses_v2). SLAs R36–R38; rádio congelado (300 m, 30 dBm, 10 MHz).")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(f"{OUT}/fig1_fronteira_viabilidade.png"); plt.close(fig)


# ==================================== Fig 2 — Evitabilidade × espaço de ações
def fig2():
    g2 = pd.read_csv(f"{R2}/oracle/adjudication/g2_deltas.csv")
    g2 = g2[g2.sla_todos_ok_no_override == True]           # as 5 evitáveis confirmadas
    e1 = pd.read_csv(f"{R2}/oracle/oracle_recalibrated_v2/adjudication/"
                     "confusion_witnesses_v2.csv")
    e1 = e1[e1.group == "E1"]
    e1 = e1.rename(columns={"lambda_total": "lam", "window_id": "win"})
    key = lambda d: list(zip(d.lam, d.seed, d.win))
    assert sorted(key(g2)) == sorted(key(e1)), "casos E1 ≠ G2"
    g2 = g2.sort_values(["lam", "seed", "win"]).reset_index(drop=True)
    e1 = e1.sort_values(["lam", "seed", "win"]).reset_index(drop=True)
    cases = [f"λ{int(r.lam)}s{int(r.seed)}\nt{int(r.win)}" for r in g2.itertuples()]
    x = np.arange(len(g2)); w = 0.27
    fig, (ax, axm) = plt.subplots(2, 1, figsize=(7.6, 5.6),
                                  gridspec_kw={"height_ratios": [2.1, 1]})
    ax.bar(x - w, g2.thr_embb_original, w, color=C_GRAY, label="original (18,17,18)")
    ax.bar(x, e1.observed_thr, w, color=C_I, label="in-suporte (37,13,3) — pisos a_u≥13, a_m≥3")
    ax.bar(x + w, g2.thr_embb_override, w, color=C_F, label="fora-de-piso (48,5,0)")
    ax.axhline(3.5, color="k", ls="--", lw=1.2)
    ax.text(4.45, 3.62, "SLA eMBB 3,5 Mbps", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(cases, fontsize=8)
    ax.set_ylabel("throughput eMBB observado (Mbps)")
    ax.set_title("Achado central: a evitabilidade depende do espaço de ações permitido")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    # matriz de resultado conjunto
    mat = np.zeros((3, len(g2)))
    for j in range(len(g2)):
        mat[:, j] = [0, 0, 1]        # original viola (por construção), in-suporte viola, fora-de-piso OK
    axm.imshow(mat, cmap=matplotlib.colors.ListedColormap([C_I, C_F]),
               vmin=0, vmax=1, aspect="auto")
    for j in range(len(g2)):
        for i, s in enumerate(("✗", "✗", "✓")):
            axm.text(j, i, s, ha="center", va="center", fontsize=13,
                     color="white", fontweight="bold")
    axm.set_xticks(range(len(g2))); axm.set_xticklabels(cases, fontsize=8)
    axm.set_yticks([0, 1, 2])
    axm.set_yticklabels(["(18,17,18)", "(37,13,3)\nin-suporte", "(48,5,0)\nfora-de-piso"],
                        fontsize=8)
    axm.set_title("SLA conjunto (3 slices) observado — 0/5 recuperam in-suporte; 5/5 fora-de-piso",
                  fontsize=9)
    for s in axm.spines.values():
        s.set_visible(False)
    footer(fig, "Fonte: replays determinísticos (prefixo bit-idêntico + override único na janela t; KPIs na estampa (t+1)·100 ms). "
                "II-c G2 (g2_deltas.csv) e II-d-v2 E1 (confusion_witnesses_v2.csv).\nMesmo estado, três ações: só a que viola os "
                "pisos estáticos (a_u≥13, a_m≥3) recupera o SLA conjunto — pisos conservadores convertem violações evitáveis "
                "em inevitáveis-por-desenho.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(f"{OUT}/fig2_evitabilidade_espaco_acoes.png"); plt.close(fig)


# ==================================== Fig 3 — Cauda URLLC causal (a_u)
def fig3():
    c = pd.read_csv(f"{R2}/oracle/adjudication/g4_urllc_curve.csv")
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    marks = {"replay_override": "o", "base_equal_original": "s"}
    for (lam, seed, win), grp in c.groupby(["lam", "seed", "win"]):
        g = grp.sort_values("a_u")
        base = grp[grp.fonte == "base_equal_original"]
        ax.plot(g.a_u, g.deadline_miss_pct, marker=marks["replay_override"],
                lw=1.6, ms=6, color=C_BLUE,
                label=f"λ{int(lam)} s{int(seed)} t{int(win)} (override executado)")
        if len(base):
            ax.scatter(base.a_u, base.deadline_miss_pct, marker="s", s=55,
                       color=C_GRAY, zorder=5,
                       label=f"λ{int(lam)} s{int(seed)} base (18,17,18)")
    ax.axhline(5.0, color=C_I, ls="--", lw=1.2)
    ax.text(24.6, 5.12, "alvo miss ≤ 5%", fontsize=8, color=C_I)
    ax.set_xlabel("a_u (RBGs dedicados à URLLC)")
    ax.set_ylabel("deadline miss URLLC na janela (%)")
    ax.set_title("Cauda URLLC é scheduler/TDD-dominada: alocar mais não compra latência")
    ax.annotate("a_u = 0 ⇒ colapso\n(miss 98–100%, G3/II-c)",
                xy=(13, 5.55), xytext=(17.5, 7.6), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=C_I), color=C_I)
    ax.set_ylim(4.2, 8.6)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    footer(fig, "Fonte: replays causais II-c G4 (g4_urllc_curve.csv) — override de a_u∈{13,25,35} com a_e compensatório; "
                "base (18,17,18).\nmiss ~5–5,5% plano em a_u 13→35 (dentro do suporte), sempre colado no alvo de 5%: a cauda "
                "não é recurso-escalável; é decisão de scheduler/TDD.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(f"{OUT}/fig3_cauda_urllc_causal.png"); plt.close(fig)


# ==================================== Fig 4 — Piso do aprendizado (G4→G4-B)
def fig4():
    stages = ["G4-A\n(s1003; 100 ms)", "réplica A\n(seeds novas; 100 ms)",
              "G4-B\n(seeds novas; rolling 1 s\n+ CAL_EXTRA λ5/λ6)"]
    ffr = [21.4, 17.4, 11.1]
    fir = [36.9, 12.7, 9.5]
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    b1 = ax.bar(x - w/2, ffr, w, color=C_I, label="FFR — falsas-FEASIBLE")
    b2 = ax.bar(x + w/2, fir, w, color=C_F, label="FIR — falsas-INFEASIBLE")
    ax.axhline(10, color="k", ls="--", lw=1.3)
    ax.text(2.42, 10.4, "gate pré-registrado ≤ 10%", fontsize=8)
    # CI 95% do FFR do G4-B (binomial, 79/713)
    ax.errorbar([2 - w/2], [11.1], yerr=[[11.1 - 8.8], [13.3 - 11.1]],
                fmt="none", ecolor="k", capsize=4, lw=1.2)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.1f%%", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=8.5)
    ax.set_ylabel("taxa de falsa classificação (%)")
    ax.set_ylim(0, 42)
    ax.set_title("O piso do aprendizado seguro: o que as correções autorizadas resolveram — e o que não")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    footer(fig, "Fonte: G4A_validation_report.md e G4B_validation_report.md (gates pré-registrados; classes confiantes; "
                "outcomes completos R19).\nFIR −74% (36,9→9,5%: era carência de dados de alta carga — CAL_EXTRA λ5/λ6; λ5 "
                "cai a 1,3%). FFR −48% (21,4→11,1%) mas acima do gate: cauda rasa URLLC miss 5–6% imprevisível pela "
                "representação estado-ação (72/79 das falsas). CI 95% do FFR G4-B: [8,8%, 13,3%].")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{OUT}/fig4_piso_aprendizado.png"); plt.close(fig)


# ==================================== Fig 5 — Ordenação e decidibilidade
def fig5():
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(7.6, 3.8),
                                  gridspec_kw={"width_ratios": [1.7, 1]})
    classes = ["FEASIBLE", "MARGINAL", "INFEASIBLE"]
    a = [0.826, 0.523, 0.127]
    b = [0.889, 0.591, 0.095]
    x = np.arange(3); w = 0.36
    r1 = ax.bar(x - w/2, a, w, color=C_GRAY, label="réplica A (100 ms)")
    r2 = ax.bar(x + w/2, b, w, color=C_BLUE, label="G4-B (rolling 1 s)")
    for rr, vals in ((r1, a), (r2, b)):
        ax.bar_label(rr, labels=[f"{v:.2f}" for v in vals], fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("P(all-SLA-OK | classe prevista)")
    ax.set_ylim(0, 1.05)
    ax.set_title("O detector ordena corretamente as classes")
    ax.legend(fontsize=8, frameon=False)
    # decidibilidade
    axd.bar(["réplica A", "G4-B"], [4617, 1491], 0.5, color=[C_GRAY, C_BLUE])
    axd.bar_label(axd.containers[0], fmt="%d", fontsize=9, fontweight="bold")
    axd.set_ylabel("janelas MARGINAL (de 12.075 completas)")
    axd.set_title("…e fica muito\nmais decidido", fontsize=9)
    footer(fig, "Fonte: g4b_verdict.json (validação G4B_VALIDATION, 9 seeds nunca vistas; classes confiantes; R19). "
                "Ordenação estrita P(all-OK|F) > P(all-OK|M) > P(all-OK|I) nas duas configurações.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{OUT}/fig5_ordenacao_decidibilidade.png"); plt.close(fig)


# ==================================== Fig 6 — R19: contabilidade de vacuous
def fig6():
    df = pd.read_csv(f"{ROOT}/results/sqr_jc_final/03_calibration/g4b/"
                     "g4b_validation_windows.csv")
    df = df[df.detector == "B_1s"]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    lam_tot, vac, comp = [], [], []
    for lam in (2, 5, 6):
        d = df[df.lam == lam]
        lam_tot.append(f"λ{lam}\n(n={len(d):,})".replace(",", "."))
        vac.append(100 * d.obs_partial.astype(bool).mean())
        comp.append(100 - 100 * d.obs_partial.astype(bool).mean())
    x = np.arange(3)
    ax.bar(x, comp, 0.55, color=C_F, label="outcome completo (3 margens observadas)")
    ax.bar(x, vac, 0.55, bottom=comp, color="#eceff1", edgecolor=C_GRAY,
           hatch="//", label="vacuous/insuficiente (sem demanda ou <1 s histórico)")
    for i, (cc, vv) in enumerate(zip(comp, vac)):
        ax.text(i, cc / 2, f"{cc:.0f}%", ha="center", color="white", fontweight="bold")
        if vv > 8:
            ax.text(i, cc + vv / 2, f"{vv:.0f}%", ha="center", color=C_GRAY, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(lam_tot, fontsize=9)
    ax.set_ylabel("janelas de decisão (%)")
    ax.set_ylim(0, 100)
    ax.set_title("R19: a maioria das janelas não constitui evidência de SLA")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    footer(fig, "Fonte: g4b_validation_windows.csv (detector B; 24.482 janelas → 12.075 completas). Janela vacuous = slice sem "
                "demanda/pacotes: contá-la como “SLA satisfeito” (prática padrão, inclusive no código auditado) infla a "
                "segurança reportada — 51% das janelas desta validação são vacuous ou insuficientes.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{OUT}/fig6_vacuous_r19.png"); plt.close(fig)


# ==================================== Fig 7 — Rádio sadio sob estresse
def fig7():
    d = pd.read_csv(f"{R1}/characterization_round1b_by_lambda.csv")
    d = d[d.lam >= 1]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(d.lam, d.util_dem_D2, marker="o", color=C_BLUE, lw=2,
            label="utilização demandada D2 (%)")
    ax.plot(d.lam, d.util_pot_D1, marker="s", color="#64b5f6", lw=1.4, ls="-.",
            label="utilização potencial D1 (%)")
    ax.set_xlabel("carga λ (UE/min)")
    ax.set_ylabel("utilização da célula (%)", color=C_BLUE)
    ax.tick_params(axis="y", labelcolor=C_BLUE)
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(d.lam, d.viol_pct, marker="^", color=C_I, lw=2, label="violação UE-tempo (%)")
    ax2.set_ylabel("violação de SLA (%)", color=C_I)
    ax2.tick_params(axis="y", labelcolor=C_I)
    ax.axvspan(4, 5, color="#f9a825", alpha=0.12)
    ax.text(4.5, 92, "joelho\nλ4–5", ha="center", fontsize=8, color="#8d6e00")
    ax.text(5.1, 8, "RLF = 0 em todos os pontos · SNR p5 ≈ 20–24 dB (violadores ≈ não-violadores)",
            fontsize=7.5, color="#424242",
            bbox=dict(fc="white", ec="#bdbdbd", alpha=0.9, pad=3))
    ax.set_title("Estresse por demanda com rádio sadio: a degradação não é de rádio")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="center right")
    footer(fig, "Fonte: characterization_round1b_by_lambda.csv (Rodada 1b; físicos congelados: 300 m, 30 dBm, 10 MHz/C=53, "
                "UMa, TDD 3DL/7). A correção da utilização (denominador 53×DL-slots) revelou o joelho λ4–5 com RLF 0 — "
                "pré-condição para atribuir inviabilidade a demanda/espaço de ações, não a SINR.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{OUT}/fig7_radio_sadio_demanda.png"); plt.close(fig)


# ==================================== Fig 8 — Coordenação (equal × prop)
def fig8():
    d = pd.read_csv(f"{R2}/round2_hetero_sweep_by_lambda.csv")
    d = d.groupby(["arm", "lam"], as_index=False).mean(numeric_only=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.0, 3.9), sharex=True)
    lam = d[d.arm == "equal"].lam
    for arm, col, mk in (("equal", C_GRAY, "s"), ("prop", C_BLUE, "o")):
        dd = d[d.arm == arm]
        a1.plot(dd.lam, dd.viol_embb_pct, marker=mk, color=col, lw=2,
                label=f"{arm} (util D2: {dd.util_D2.mean():.0f}%)")
        a2.plot(dd.lam, dd.urllc_miss_pct, marker=mk, color=col, lw=2, label=arm)
    a1.set_xlabel("λ"); a2.set_xlabel("λ")
    a1.set_ylabel("violação eMBB (%)")
    a2.set_ylabel("deadline miss URLLC (%)")
    a2.axhline(5, color=C_I, ls="--", lw=1)
    a1.set_title("eMBB: prop recupera 18–40 pp\n(mas com célula ociosa no equal)")
    a2.set_title("URLLC: prop PIORA sob pressão\n— nenhum braço domina")
    a1.legend(fontsize=8, frameon=False)
    footer(fig, "Fonte: round2_hetero_sweep_by_lambda.csv (Rodada 2.3; 30 runs equal×prop × λ∈{2,4,5,6,7} × 3 seeds; SLAs "
                "heterogêneos R36–R38). O trade-off eMBB×URLLC entre políticas estáticas motivou o oracle: a fronteira é "
                "conjunta e a pergunta certa é evitável×inevitável, não equal×prop.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{OUT}/fig8_coordenacao_equal_prop.png"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        f()
        print(f"[ok] {f.__name__}")
    print(f"Figuras em {OUT}")
