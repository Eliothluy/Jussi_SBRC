import glob, sys
import pandas as pd, numpy as np

# Decomposição da cauda URLLC (mandato 21/10): miss por N_urllc(t) × atividade
# concorrente (N_embb(t)), SEM nova campanha — dados do sanity 2.2 final.
# Braços: prop (lowload_prop_v2, 7 seeds) e equal (lowload, 3 seeds).

R = "/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_characterization_round2"
ARMS = {"prop": f"{R}/lowload_prop_v2", "equal": f"{R}/lowload"}
SLA_MS = 10.0

rows = []
for arm, SW in ARMS.items():
    for kp in sorted(glob.glob(f"{SW}/**/results_sqr_network_only/*/sqr_kpm_history.csv", recursive=True)):
        base = kp.rsplit("results_sqr_network_only/", 1)[0] + "results_sqr_network_only/"
        dp = sorted(glob.glob(f"{base}*/sqr_packet_delays.csv"))
        if not dp:
            continue
        k = pd.read_csv(kp)
        pdl = pd.read_csv(dp[0])
        u = pdl[pdl.sliceId == 1].copy()
        if not len(u):
            continue
        # N por slice por janela (UEs com demanda)
        nw = (k[k.dTxBytes > 0]
              .assign(win=k[k.dTxBytes > 0].timestamp // 100)
              .groupby("win")
              .agg(n_u=("sliceId", lambda s: int((s == 1).sum())),
                   n_e=("sliceId", lambda s: int((s == 0).sum())))
              .reset_index())
        u["win"] = (u.txSec * 10).astype(int)
        u = u.merge(nw, on="win", how="left")
        u["ms"] = u.delaySec * 1000
        u["miss"] = u.ms > SLA_MS
        u["n_u_bin"] = u.n_u.fillna(0).clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"})
        u["n_e_bin"] = np.where(u.n_e.fillna(0) == 0, "0", np.where(u.n_e.fillna(0) == 1, "1", "2+"))
        for (nu, ne), g in u.groupby(["n_u_bin", "n_e_bin"]):
            rows.append({
                "arm": arm, "n_urllc": nu, "n_embb": ne,
                "n_pkts": len(g), "miss_pct": 100 * g.miss.mean(),
                "med_ms": g.ms.median(), "p95_ms": g.ms.quantile(.95),
            })

df = pd.DataFrame(rows)
ordem_nu = ["0", "1", "2", "3+"]
ordem_ne = ["0", "1", "2+"]
df["o1"] = df.n_urllc.map({v: i for i, v in enumerate(ordem_nu)})
df["o2"] = df.n_embb.map({v: i for i, v in enumerate(ordem_ne)})
df = df.sort_values(["arm", "o1", "o2"])
df.to_csv(f"{R}/round2_urllc_tail_decomp.csv", index=False)
pd.set_option("display.width", 200)
for arm in ["prop", "equal"]:
    print(f"\n=== Braço {arm}: miss% por N_urllc × N_embb (pacotes, miss%, med, p95 ms) ===")
    print(df[df.arm == arm][["n_urllc", "n_embb", "n_pkts", "miss_pct", "med_ms", "p95_ms"]]
          .round(2).to_string(index=False))
print("\n[ok] CSV:", f"{R}/round2_urllc_tail_decomp.csv")
