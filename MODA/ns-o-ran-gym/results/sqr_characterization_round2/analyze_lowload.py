import glob, sys
import pandas as pd, numpy as np

# Etapa 2.2 (MD §11): sanity em baixa carga com slices heterogêneos.
# ALVOS CONGELADOS (assumptions_register v1.6, R36–R38, orquestrador 20/10/2026):
SLA_EMBB_MBPS = 3.5        # R36 emendada: operacional 3,5 (7 Mbps = DORA, escalado 53/106 PRBs [ASSUMIDO])
SLA_URLLC_MS = 10.0        # R37 [ASSUMIDO, decisão do orquestrador]
SLA_MMTC_PDR = 0.95        # R38 [ASSUMIDO, definido antes do run]
# GATE STOP pré-declarado (sistemático = recorrente, não esporádico):
#   eMBB  : > 5% das UE-janelas c/ demanda com windowThroughput < 7 Mbps
#   URLLC : > 5% dos pacotes com delay E2E > 10 ms
#   mMTC  : PDR agregado do slice < 95%

R = "/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_characterization_round2"
SW = sys.argv[1] if len(sys.argv) > 1 else f"{R}/lowload"

rows, per_run = [], []
# runs network-only: {SW}/<tag>_s<N>/results_sqr_network_only/seed=N_run=1/...
# runs IPC (prop):    {SW}/<tag>/<uuid>/results_sqr_network_only/seed=N_run=1/...
kpaths = sorted(glob.glob(f"{SW}/**/results_sqr_network_only/*/sqr_kpm_history.csv", recursive=True))
for kp in kpaths:
    base = kp.rsplit("results_sqr_network_only/", 1)[0] + "results_sqr_network_only/"
    d = base.rstrip("/").rsplit("/", 1)[0]
    ds = sorted(glob.glob(f"{base}*/sqr_packet_delays.csv"))
    pw = sorted(glob.glob(f"{base}*/sqr_prb_windows.csv"))
    if not pw:
        continue
    k = pd.read_csv(kp)
    w = pd.read_csv(pw[0])
    tot = int(w.total_rbg.iloc[0])
    sch = w[[c for c in w.columns if c.startswith("prb_scheduled_rbg_")]].sum(axis=1)
    dl = w.dl_slots; has = dl > 0
    util = 100 * sch[has].sum() / (tot * dl[has].sum())

    # ---- eMBB (slice 0): throughput por UE-janela + suavizado MM 1s ----
    e = k[(k.sliceId == 0) & (k.dTxBytes > 0)].sort_values(["ueImsi", "timestamp"])
    e = e.assign(viol=e.windowThroughputMbps < SLA_EMBB_MBPS)
    e["thr_smooth"] = e.groupby("ueImsi").windowThroughputMbps.transform(
        lambda s: s.rolling(10, min_periods=3, center=True).mean())
    e = e.assign(viol_smooth=e.thr_smooth < SLA_EMBB_MBPS)

    # ---- URLLC (slice 1): delay por pacote ----
    u_lat = pd.Series(dtype=float)
    u_kpm = k[k.sliceId == 1]
    if ds:
        pdl = pd.read_csv(ds[0])
        if len(pdl):
            u_lat = pdl[pdl.sliceId == 1].delaySec * 1000
    u_lat_kpm = np.where(u_kpm.get("dRxPackets", 0) > 0,
                         1000 * u_kpm.get("dDelaySumSec", 0) / u_kpm.get("dRxPackets", 1), np.nan)
    u_lat_kpm = pd.Series(u_lat_kpm).dropna()

    # ---- mMTC (slice 2): PDR de sessão por UE + agregado ----
    m = k[k.sliceId == 2].groupby("ueImsi").agg(tx=("dTxPackets", "sum"), rx=("dRxPackets", "sum"))
    m_sess = (m.rx / m.tx.replace(0, np.nan)).dropna()
    _mtx = k[k.sliceId == 2].dTxPackets.sum()
    m_agg = (k[k.sliceId == 2].dRxPackets.sum() / _mtx) if _mtx > 0 else np.nan  # sem sessão mMTC => sem PDR a violar

    run = d.split("/")[-1]
    per_run.append({
        "run": run,
        "embb_viol_pct": 100 * e.viol.mean() if len(e) else np.nan,
        "embb_viol_smooth_pct": 100 * e.viol_smooth.mean() if len(e) else np.nan,
        "embb_thr_p5": e.windowThroughputMbps.quantile(.05) if len(e) else np.nan,
        "embb_thr_med": e.windowThroughputMbps.median() if len(e) else np.nan,
        "urllc_miss_pct": 100 * (u_lat > SLA_URLLC_MS).mean() if len(u_lat) else np.nan,
        "urllc_p95_ms": u_lat.quantile(.95) if len(u_lat) else np.nan,
        "urllc_med_ms": u_lat.median() if len(u_lat) else np.nan,
        "urllc_n_pkts": int(len(u_lat)),
        "mmtc_pdr_agg": m_agg,
        "mmtc_pdr_sess_p5": m_sess.quantile(.05) if len(m_sess) else np.nan,
        "mmtc_n_sess": int(len(m_sess)),
        "util_D2": util,
        "snr_p5": k[(k.snrDb != -999)].snrDb.quantile(.05),
    })

df = pd.DataFrame(per_run)
OUT = f"{R}/round2_lowload_{SW.split(chr(47))[-1]}.csv"
df.to_csv(OUT, index=False)
pd.set_option("display.width", 240)
print("=== Etapa 2.2 — sanity em baixa carga (por run) ===")
print(df.round(3).to_string(index=False))

g = df.mean(numeric_only=True)
print("\n=== Média ===")
print(g[["embb_viol_pct", "embb_viol_smooth_pct", "embb_thr_med", "urllc_miss_pct",
         "urllc_p95_ms", "mmtc_pdr_agg", "util_D2"]].round(3).to_string())

print("\n=== GATE §11.3 (pré-declarado; sistemático = >5% / PDR<95%) ===")
stops = []
if (df.embb_viol_pct > 5).any():
    stops.append(f"eMBB: violação {df.embb_viol_pct.max():.1f}% > 5% (thr < {SLA_EMBB_MBPS} Mbps)")
if (df.urllc_miss_pct > 5).any():
    stops.append(f"URLLC: deadline-miss {df.urllc_miss_pct.max():.1f}% > 5% (> {SLA_URLLC_MS} ms)")
if (df.mmtc_pdr_agg.dropna() < SLA_MMTC_PDR).any():
    stops.append(f"mMTC: PDR agregado {df.mmtc_pdr_agg.min():.3f} < {SLA_MMTC_PDR}")
if stops:
    print("STOP — violação sistemática em baixa carga:")
    for s in stops:
        print("  -", s)
    print("NÃO iniciar o sweep 2.3. Investigar: SLA, KPI, tráfego, scheduler, física, métrica, janela (MD §11.3).")
    sys.exit(1)
else:
    print("OK — nenhum slice viola sistematicamente em baixa carga. Sweep 2.3 liberado.")
    sys.exit(0)
print("\n[ok] CSV:", OUT)
