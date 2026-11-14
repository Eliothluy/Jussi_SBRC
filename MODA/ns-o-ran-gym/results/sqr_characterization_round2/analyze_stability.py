import glob, re
import pandas as pd, numpy as np

R = "/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_characterization_round2"
SW = f"{R}/stability"
SLA = {0: 0.5, 1: 1.0, 2: 1.5}
WARMUP_MS = 180_000  # MD §4.2: 0–180 s warm-up, 180–600 s medição

rows = []
for d in sorted(x for x in glob.glob(f"{SW}/lam*_s*") if not x.endswith(".log")):
    if not glob.glob(f"{d}/.done"):
        continue
    m = re.search(r"lam(\d+)_s(\d+)$", d)
    ks = glob.glob(f"{d}/results_sqr_network_only/*/sqr_kpm_history.csv")
    pw = glob.glob(f"{d}/results_sqr_network_only/*/sqr_prb_windows.csv")
    if not m or not ks or not pw:
        continue
    lam, seed = float(m.group(1)), int(m.group(2))
    k = pd.read_csv(ks[0]); k["sla"] = k.sliceId.map(SLA)
    w = pd.read_csv(pw[0])
    # integridade: 600 s → ~5999 janelas
    if len(w) < 5900 or not w.timestamp_ms.is_monotonic_increasing:
        print(f"[WARN] {d}: janelas={len(w)} mono={w.timestamp_ms.is_monotonic_increasing}")
    dm = k[k.dTxBytes > 0].copy(); dm["viol"] = dm.windowThroughputMbps < dm.sla

    def seg_stats(df_k, df_w, lo, hi, name):
        kk = df_k[(df_k.timestamp >= lo) & (df_k.timestamp < hi)]
        ww = df_w[(df_w.timestamp_ms >= lo) & (df_w.timestamp_ms < hi)]
        tot = int(ww.total_rbg.iloc[0]) if len(ww) else 53
        sch = ww[[c for c in ww.columns if c.startswith("prb_scheduled_rbg_")]].sum(axis=1)
        dl = ww.dl_slots; has = dl > 0
        per_ue = kk.groupby("ueImsi").windowThroughputMbps.median()
        return {
            "seg": name,
            "viol_pct": 100 * kk.viol.mean() if len(kk) else np.nan,
            "thr_p5": per_ue.quantile(.05) if len(per_ue) else np.nan,
            "thr_med": per_ue.median() if len(per_ue) else np.nan,
            "n_ue_mean": kk.groupby("timestamp").ueImsi.nunique().mean() if len(kk) else np.nan,
            "util_D2": 100 * sch[has].sum() / (tot * dl[has].sum()) if has.any() else np.nan,
            "queue_p95": kk.queueBytes.quantile(.95) if len(kk) else np.nan,
            "snr_p5": kk[kk.snrDb != -999].snrDb.quantile(.05) if len(kk) else np.nan,
        }

    end_ms = int(w.timestamp_ms.max())
    for name, lo, hi in [("warmup_0-180", 0, WARMUP_MS),
                         ("main_180-390", WARMUP_MS, 390_000),
                         ("main_390-600", 390_000, 600_000),
                         ("main_full_180-600", WARMUP_MS, 600_000)]:
        if lo >= end_ms:
            continue
        r = {"lam": lam, "seed": seed, **seg_stats(dm, w, lo, hi, name)}
        rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(f"{R}/round2_stability.csv", index=False)
pvt = df.pivot_table(index=["lam", "seed"], columns="seg",
                     values=["viol_pct", "thr_p5", "thr_med", "n_ue_mean", "util_D2"])
pvt.round(2).to_csv(f"{R}/round2_stability_by_seg.csv")

# média por λ, foco no critério §4.4: ordenação + estabilidade entre metades
g = df[df.seg != "warmup_0-180"].pivot_table(index="lam", columns="seg",
                                            values=["viol_pct", "thr_med", "util_D2", "n_ue_mean"])
g.round(2).to_csv(f"{R}/round2_stability_summary.csv")
pd.set_option("display.width", 220)
print("=== Etapa 2.0 — por segmento (média das 3 seeds) ===")
print(g.round(2).to_string())

stab = df[df.seg.isin(["main_180-390", "main_390-600"])].pivot_table(
    index=["lam", "seed"], columns="seg", values="viol_pct")
stab["delta_pp"] = stab["main_390-600"] - stab["main_180-390"]
print("\n=== Estabilidade pós-warm-up: violação 180–390 vs 390–600 (pp) ===")
print(stab.round(2).to_string())
print("\n[ok] CSVs em", R)
