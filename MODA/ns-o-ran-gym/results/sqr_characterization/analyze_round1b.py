import glob, re, sys
import pandas as pd, numpy as np

R = "/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_characterization"
SLA = {0: 0.5, 1: 1.0, 2: 1.5}

def load_runs():
    for d in sorted(x for x in glob.glob(f"{R}/sweep/lam*_s*") if not x.endswith(".log")):
        if not glob.glob(f"{d}/.done"):  # run incompleto (em curso) — ignorar
            continue
        m = re.search(r"lam([\d.]+)_s(\d+)$", d)
        ks = glob.glob(f"{d}/results_sqr_network_only/*/sqr_kpm_history.csv")
        pw = glob.glob(f"{d}/results_sqr_network_only/*/sqr_prb_windows.csv")
        if not m or not ks or not pw:
            continue
        lam, seed = float(m.group(1)), int(m.group(2))
        k = pd.read_csv(ks[0]); k["sla"] = k.sliceId.map(SLA); k["lam"] = lam; k["seed"] = seed
        w = pd.read_csv(pw[0])
        # integridade (guarda contra runs corrompidos, ex. colisão de processos)
        ts_ok = k.timestamp.is_monotonic_increasing and w.timestamp_ms.is_monotonic_increasing
        full = len(w) >= 2900 and w.timestamp_ms.max() >= 295_000  # w: 1 linha por janela
        if not (ts_ok and full):
            print(f"[WARN] {d}: integridade suspeita (mono={ts_ok}, janelas={len(w)}, "
                  f"t_max={k.timestamp.max():.0f}) — verifique antes de usar")
        yield lam, seed, k, w

rows, fine_rows = [], []
for lam, seed, k, w in load_runs():
    dm = k[k.dTxBytes > 0].copy()  # UE-janela com demanda
    dm["viol"] = dm.windowThroughputMbps < dm.sla
    dm["ratio"] = dm.windowThroughputMbps / dm.sla

    # ---- throughput por UE: percentis da mediana-por-UE e da pool UE-janela ----
    per_ue = dm.groupby("ueImsi").windowThroughputMbps.median()
    per_ue_ratio = dm.groupby("ueImsi").apply(lambda g: (g.windowThroughputMbps / g.sla).median(),
                                              include_groups=False)
    # ---- violação por slice ----
    viol_slice = dm.groupby("sliceId").viol.mean() * 100
    # ---- quem viola: UEs que violam >=1 janela, e fração do tempo deles ----
    ue_viol = dm.groupby("ueImsi").viol.sum()
    ue_windows = dm.groupby("ueImsi").size()
    n_violators = int((ue_viol > 0).sum()); n_ues = len(ue_windows)
    viol_share = (ue_viol.sum() / ue_windows.sum()) * 100  # == viol_pct global
    # violação condicionada aos UEs violadores (concentração temporal)
    viol_cond = (ue_viol.sum() / ue_windows[ue_viol > 0].sum()) * 100 if n_violators else 0.0

    # ---- fase da sessão em que viola (s desde a 1ª janela com demanda do UE) ----
    dm = dm.sort_values(["ueImsi", "timestamp"])
    dm["t_since_first"] = dm.groupby("ueImsi").timestamp.transform(lambda t: t - t.iloc[0])
    def phase_bin(t):
        if t < 1000: return "0-1s"
        if t < 3000: return "1-3s"
        if t < 10000: return "3-10s"
        return ">10s"
    dm["phase"] = dm.t_since_first.map(phase_bin)
    ph = dm.groupby("phase", observed=True).agg(viol_pct=("viol", lambda v: 100 * v.mean()),
                                                n=("viol", "size"))
    for ph_name, r in ph.iterrows():
        fine_rows.append({"lam": lam, "seed": seed, "phase": ph_name,
                          "viol_pct": r.viol_pct, "n_windows": int(r.n)})

    # ---- utilização: denominadores auditados ----
    tot = int(w.total_rbg.iloc[0])
    sch = w[[c for c in w.columns if c.startswith("prb_scheduled_rbg_")]].sum(axis=1)
    bud = w[[c for c in w.columns if c.startswith("prb_budget_rbg_")]].sum(axis=1)
    dl = w.dl_slots
    has_dem = dl > 0
    util_dem = 100 * sch[has_dem].sum() / (tot * dl[has_dem].sum())          # D2: slots c/ demanda
    util_pot = 100 * sch.sum() / (tot * (100.0 * 4 / 7) * len(w))            # D1: teto 4/7 slots
    idle_in_quota = 100 * (1 - (sch[has_dem] / bud[has_dem].replace(0, np.nan)).dropna().clip(upper=1).mean())
    # janelas com demanda e cota subutilizada (scheduled < 80% do budget)
    frac_under = 100 * ((sch[has_dem] < 0.8 * bud[has_dem]) & (bud[has_dem] > 0)).mean() if has_dem.any() else np.nan

    # violação por terço do run (início/meio/fim) — tendência temporal
    w_end = dm.timestamp.max() if len(dm) else 0
    dm["terco"] = pd.cut(dm.timestamp, bins=[0, w_end / 3, 2 * w_end / 3, w_end + 1],
                         labels=["t1", "t2", "t3"])
    ter = dm.groupby("terco", observed=True).viol.mean() * 100

    # SNR dos violadores vs não-violadores (rádio igual?)
    snr_ok = dm[~dm.viol & (dm.snrDb != -999)].snrDb
    snr_v = dm[dm.viol & (dm.snrDb != -999)].snrDb

    rows.append({
        "lam": lam, "seed": seed,
        "n_ues": n_ues, "n_violators": n_violators,
        "pct_ues_violators": 100 * n_violators / n_ues if n_ues else 0,
        "viol_pct": viol_share, "viol_cond_violators": viol_cond,
        **{f"viol_s{s}": viol_slice.get(s, np.nan) for s in SLA},
        "thr_ue_p5": per_ue.quantile(.05), "thr_ue_p10": per_ue.quantile(.10),
        "thr_ue_p25": per_ue.quantile(.25), "thr_ue_med": per_ue.median(),
        "ratio_p5": per_ue_ratio.quantile(.05), "ratio_p10": per_ue_ratio.quantile(.10),
        "ratio_p25": per_ue_ratio.quantile(.25),
        "util_dem_D2": util_dem, "util_pot_D1": util_pot,
        "idle_in_quota_pct": idle_in_quota, "win_quota_under80": frac_under,
        "viol_t1": ter.get("t1", np.nan), "viol_t2": ter.get("t2", np.nan), "viol_t3": ter.get("t3", np.nan),
        "snr_viol_med": snr_v.median() if len(snr_v) else np.nan,
        "snr_ok_med": snr_ok.median() if len(snr_ok) else np.nan,
    })

df = pd.DataFrame(rows).sort_values(["lam", "seed"])
df.to_csv(f"{R}/characterization_round1b.csv", index=False)
g = df.groupby("lam").mean(numeric_only=True).round(2)
g.to_csv(f"{R}/characterization_round1b_by_lambda.csv")

fine = pd.DataFrame(fine_rows)
fine.pivot_table(index=["lam", "seed"], columns="phase", values="viol_pct").round(2)\
    .to_csv(f"{R}/characterization_round1b_viol_by_phase.csv")

# ---------------------------- figuras (Rodada 1b) ----------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(14, 9))

# (A) joelho: violação x λ (pontos por seed + média) com utilização D2
a = ax[0, 0]
gb = df.groupby("lam")
for s, sub in df.groupby("seed"):
    a.scatter(sub.lam, sub.viol_pct, color="#d62728", alpha=.45, s=28, zorder=2)
m = gb.viol_pct.mean(); e = [m - gb.viol_pct.min(), gb.viol_pct.max() - m]
a.errorbar(m.index, m.values, yerr=e, color="#d62728", marker="o", capsize=4,
           label="violação UE-janela c/ demanda (%)", zorder=3)
a.set_xlabel("λ (UEs/min)"); a.set_ylabel("violação (%)", color="#d62728")
a.grid(alpha=.3)
a2 = a.twinx()
mu = gb.util_dem_D2.mean()
a2.plot(mu.index, mu.values, color="#1f77b4", marker="s", label="utilização D2 (%)")
a2.set_ylabel("utilização D2: RBGs sched/(53×dl_slots) (%)", color="#1f77b4")
a2.set_ylim(0, 100)
a.set_title("(A) Joelho da degradação: violação × utilização × λ")

# (B) percentis do throughput por UE × λ
a = ax[0, 1]
for q, c in [("thr_ue_p5", "#d62728"), ("thr_ue_p10", "#ff7f0e"),
             ("thr_ue_p25", "#2ca02c"), ("thr_ue_med", "#1f77b4")]:
    mm = gb[q].mean()
    a.plot(mm.index, mm.values, marker="o", color=c, label=q.replace("thr_ue_", "UE med "))
for sla, ls in [(0.5, ":"), (1.0, "--"), (1.5, "-.")]:
    a.axhline(sla, ls=ls, color="gray", lw=.8)
    a.text(8.1, sla, f"SLA {sla}", fontsize=7, color="gray", va="bottom")
a.set_xlabel("λ (UEs/min)"); a.set_ylabel("throughput/UE (Mbps)")
a.legend(fontsize=7); a.grid(alpha=.3)
a.set_title("(B) Percentis do throughput mediano por UE")

# (C) violação × fase da sessão do UE
a = ax[1, 0]
ph_order = ["0-1s", "1-3s", "3-10s", ">10s"]
pvt = fine.pivot_table(index="lam", columns="phase", values="viol_pct")[ph_order]
for lam_ in pvt.index:
    a.plot(ph_order, pvt.loc[lam_].values, marker="o", label=f"λ={lam_:g}")
a.set_xlabel("fase da sessão do UE (s desde a 1ª janela c/ demanda)")
a.set_ylabel("violação (%)"); a.legend(fontsize=7); a.grid(alpha=.3)
a.set_title("(C) Onde na sessão a violação acontece")

# (D) violação por slice (pontos por seed — variância estocástica)
a = ax[1, 1]
for s, c in [(0, "#1f77b4"), (1, "#ff7f0e"), (2, "#2ca02c")]:
    sub = df[df[f"viol_s{s}"].notna()]
    a.scatter(sub.lam, sub[f"viol_s{s}"], color=c, alpha=.45, s=28)
    gbs = sub.groupby("lam")[f"viol_s{s}"].mean()
    a.plot(gbs.index, gbs.values, marker="o", color=c, label=f"s{s} (SLA {SLA[s]} Mbps)")
a.set_xlabel("λ (UEs/min)"); a.set_ylabel("violação (%)")
a.legend(fontsize=8); a.grid(alpha=.3)
a.set_title("(D) Violação por slice (pontos = seeds)")

fig.suptitle("Rodada 1b — refinamento da transição (300 m congelado, greedy, equal-split, 3 seeds)")
fig.tight_layout()
fig.savefig(f"{R}/figures_round1b.png", dpi=130)

pd.set_option("display.width", 250)
cols = ["viol_pct", "viol_s0", "viol_s1", "viol_s2", "pct_ues_violators",
        "viol_cond_violators", "thr_ue_p5", "thr_ue_p10", "thr_ue_p25", "thr_ue_med",
        "ratio_p5", "ratio_p10", "util_dem_D2", "util_pot_D1", "win_quota_under80",
        "viol_t1", "viol_t2", "viol_t3", "snr_viol_med", "snr_ok_med"]
print(g[cols].to_string())
print("\n[ok] CSVs: characterization_round1b{,_by_lambda,_viol_by_phase}.csv + figures_round1b.png")
