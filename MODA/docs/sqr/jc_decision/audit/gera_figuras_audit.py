#!/usr/bin/env python3
"""fig2 AUDITADA (D0-A) + cópias tight de fig4/6/7 — protocolo v2 §5.8.

fig2 é a única com MUDANÇA DE DADO: a linha A1 (37,13,3) era "✗✗✗✗✗" (label
do stage1, bug de métrica) e passa a ser "✓V ✓ ✓ ✗ ✗" (julgamento congelado
R36–R38; ✓V = vacuous no estrito). mMTC: tx in-window = 0 nas 15 células
(rolling portado por histórico) — revelado no rodapé. Originais preservados.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/eliothluy/Documentos/Jussi_SBRC"
GYM = f"{ROOT}/MODA/ns-o-ran-gym"
sys.path.insert(0, f"{GYM}/src")
OUT = f"{ROOT}/docs/artigo/figuras_tese/audit_v2"
import os
os.makedirs(OUT, exist_ok=True)

C_F, C_M, C_I, C_GRAY, C_BLUE = "#2e7d32", "#f9a825", "#c62828", "#616161", "#1565c0"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv(f"{ROOT}/MODA/docs/sqr/jc_decision/audit/D0A_replay_5x3.csv")
order = ["lam6_s3_t298", "lam4_s3_t1812", "lam5_s1_t1955", "lam4_s1_t2210",
         "lam5_s3_t1677"]
acts = ["A0_ORIGINAL", "A1_FLOORS_37-13-3", "A2_RELAXED_48-5-0"]
alab = ["original (18,17,18)", "in-suporte (37,13,3)", "fora-de-piso (48,5,0)"]
acol = [C_GRAY, C_I, C_F]

fig, (ax, axm) = plt.subplots(2, 1, figsize=(7.6, 5.8),
                              gridspec_kw={"height_ratios": [2.1, 1.25]})
x = np.arange(5)
w = 0.27
for i, a in enumerate(acts):
    vals = [df[(df.state_id == s) & (df.action_id == a)].eMBB_value.iloc[0]
            for s in order]
    ax.bar(x + (i - 1) * w, vals, w, color=acol[i], label=alab[i])
ax.axhline(3.5, color="k", ls="--", lw=1.2)
ax.text(4.42, 3.62, "SLA eMBB 3,5 Mbps", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([s.replace("_s", "s").replace("_t", "\nt") for s in order],
                   fontsize=8)
ax.set_ylabel("throughput eMBB mediana (Mbps)")
ax.set_title("Evitabilidade × espaço de ações — VERSÃO AUDITADA (D0-A)")
ax.legend(fontsize=8, frameon=False, loc="upper left")

# matriz: julgamento congelado (vacuous=OK) — A1 corrigido 0/5 → 3/5
frozen = np.array([[df[(df.state_id == s) & (df.action_id == a)]
                    .joint_frozen_vacuous_ok.iloc[0] for s in order]
                   for a in acts], dtype=float)
axm.imshow(frozen, cmap=matplotlib.colors.ListedColormap([C_I, C_F]),
           vmin=0, vmax=1, aspect="auto")
sym = [["✗", "✗", "✗", "✗", "✗"],
       ["✓V", "✓", "✓", "✗", "✗"],
       ["✓V", "✓", "✓", "✓", "✓"]]
for i in range(3):
    for j in range(5):
        axm.text(j, i, sym[i][j], ha="center", va="center", fontsize=12,
                 color="white", fontweight="bold")
axm.set_xticks(range(5))
axm.set_xticklabels([s.replace("_s", "s").replace("_t", "\nt") for s in order],
                    fontsize=8)
axm.set_yticks([0, 1, 2])
axm.set_yticklabels(["(18,17,18)", "(37,13,3)\nin-suporte", "(48,5,0)\nfora-de-piso"],
                    fontsize=8)
axm.set_title("SLA conjunto (julgamento congelado R36–R38): in-suporte 3/5 (era "
              "“0/5” — artefato de métrica) · fora-de-piso 5/5 · ✓V = vacuous no estrito",
              fontsize=8.5)
for sp in axm.spines.values():
    sp.set_visible(False)
fig.text(0.01, 0.008,
         "AUDITADO D0-A (08/11): recomputado dos traces brutos com funções congeladas; prefixos A/A bit-idênticos 10/10.\n"
         "Correção: E1 '0/5' usava eMBB por thr_min (não mediana R36) — corrigido para 3/5 frozen / 2/5 estrito.\n"
         "mMTC: tx in-window = 0 nas 15 células (tráfego 1 pkt/s) — PDR rolling portado por histórico pré-t; estrito = UNKNOWN.\n"
         "Fonte: MODA/docs/sqr/jc_decision/audit/D0A_replay_5x3.csv.",
         fontsize=6.5, color="#757575", va="bottom")
fig.tight_layout(rect=(0, 0.085, 1, 1))
fig.savefig(f"{OUT}/fig2_evitabilidade_AUDITADA.png", bbox_inches="tight")
plt.close(fig)
print("[ok] fig2_evitabilidade_AUDITADA.png")

# ---- cópias tight de fig4/6/7 (cosmético §5.8: clipping de rodapé) --------
sys.path.insert(0, f"{ROOT}/docs/artigo/figuras_tese")
import gera_figuras_tese as G
from PIL import Image, ImageChops
G.OUT = OUT
for f in (G.fig4, G.fig6, G.fig7):
    f()
for name in ("fig4_piso_aprendizado.png", "fig6_vacuous_r19.png",
             "fig7_radio_sadio_demanda.png"):
    p = f"{OUT}/{name}"
    im = Image.open(p).convert("RGB")
    bg = Image.new("RGB", im.size, (250, 250, 250))
    diff = ImageChops.difference(im, bg)
    im.crop(ImageChops.add(diff, diff, 2.0, -12).getbbox()).save(p)
    print(f"[ok] tight {name}")
