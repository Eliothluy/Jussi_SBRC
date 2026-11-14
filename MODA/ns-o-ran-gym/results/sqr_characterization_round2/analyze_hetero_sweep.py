import glob, re, sys
import pandas as pd, numpy as np

# Etapa 2.3 (MD §12–16): sweep heterogêneo — Perfil A (balanceado), DOIS BRAÇOS
# (mandato 21/10): equal (baseline estático, network-only) e prop (witness
# adaptativo, IPC). Alvos congelados R36 (3,5 Mbps) / R37 (10 ms) / R38 (PDR 95%).

SLA_EMBB = 3.5
SLA_URLLC_MS = 10.0
SLA_MMTC_PDR = 0.95

R = "/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_characterization_round2"
ARMS = {"equal": f"{R}/hetero_sweep", "prop": f"{R}/hetero_sweep_prop"}


def load_run(kp):
    base = kp.rsplit("results_sqr_network_only/", 1)[0] + "results_sqr_network_only/"
    pw = sorted(glob.glob(f"{base}*/sqr_prb_windows.csv"))
    ds = sorted(glob.glob(f"{base}*/sqr_packet_delays.csv"))
    if not pw:
        return None
    return pd.read_csv(kp), pd.read_csv(pw[0]), (pd.read_csv(ds[0]) if ds else None)


rows = []
for arm, SW in ARMS.items():
    for kp in sorted(glob.glob(f"{SW}/**/results_sqr_network_only/*/sqr_kpm_history.csv", recursive=True)):
        m = re.search(r"A_lam(\d+)_s(\d+)", kp)
        if not m:
            continue
        loaded = load_run(kp)
        if loaded is None:
            continue
        k, w, pdl = loaded
        lam, seed = float(m.group(1)), int(m.group(2))
        tot = int(w.total_rbg.iloc[0])
        sch = w[[c for c in w.columns if c.startswith("prb_scheduled_rbg_")]].sum(axis=1)
        dl = w.dl_slots
        has = dl > 0
        utilD2 = 100 * sch[has].sum() / (tot * dl[has].sum())
        utilD1 = 100 * sch.sum() / (tot * (400 / 7) * len(w))

        e = k[(k.sliceId == 0) & (k.dTxBytes > 0)]
        thr_med = e.groupby("ueImsi").windowThroughputMbps.median()
        u_lat = pd.Series(dtype=float)
        if pdl is not None and len(pdl):
            u_lat = pdl[pdl.sliceId == 1].delaySec * 1000
        m_ = k[k.sliceId == 2]
        _mtx = m_.dTxPackets.sum()
        m_pdr = (m_.dRxPackets.sum() / _mtx) if _mtx > 0 else np.nan

        rows.append({
            "arm": arm, "lam": lam, "seed": seed,
            "n_embb_mean": k[k.sliceId == 0].groupby("timestamp").ueImsi.nunique().mean(),
            "n_urllc_mean": k[k.sliceId == 1].groupby("timestamp").ueImsi.nunique().mean(),
            "n_mmtc_mean": k[k.sliceId == 2].groupby("timestamp").ueImsi.nunique().mean(),
            "viol_embb_pct": 100 * (e.windowThroughputMbps < SLA_EMBB).mean() if len(e) else np.nan,
            "thr_embb_p5": thr_med.quantile(.05) if len(thr_med) else np.nan,
            "thr_embb_med": thr_med.median() if len(thr_med) else np.nan,
            "urllc_miss_pct": 100 * (u_lat > SLA_URLLC_MS).mean() if len(u_lat) else np.nan,
            "urllc_p95_ms": u_lat.quantile(.95) if len(u_lat) else np.nan,
            "urllc_med_ms": u_lat.median() if len(u_lat) else np.nan,
            "mmtc_pdr": m_pdr,
            "util_D2": utilD2, "util_D1": utilD1,
            "snr_p5": k[k.snrDb != -999].snrDb.quantile(.05),
            "queue_embb_p95": k[k.sliceId == 0].queueBytes.quantile(.95),
        })

df = pd.DataFrame(rows).sort_values(["arm", "lam", "seed"])
df.to_csv(f"{R}/round2_hetero_sweep.csv", index=False)
g = df.groupby(["arm", "lam"]).mean(numeric_only=True)
g.round(2).to_csv(f"{R}/round2_hetero_sweep_by_lambda.csv")
pd.set_option("display.width", 240)
for arm in df.arm.unique():
    print(f"\n=== Etapa 2.3 — Perfil A, braço {arm} (média das seeds disponíveis) ===")
    print(g.loc[arm][["n_embb_mean", "n_urllc_mean", "viol_embb_pct", "thr_embb_med",
                      "urllc_miss_pct", "urllc_p95_ms", "mmtc_pdr", "util_D2", "util_D1",
                      "snr_p5"]].round(2).to_string())
print("\n[ok] CSVs: round2_hetero_sweep{,_by_lambda}.csv")
