import glob, re, subprocess, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_characterization"
SLA={0:0.5,1:1.0,2:1.5}; rows=[]
for d in sorted(x for x in glob.glob(f"{R}/sweep/lam*_s*") if not x.endswith(".log")):
    lam=float(re.search(r"lam([\d.]+)_s(\d)",d).group(1)); seed=int(re.search(r"_s(\d)",d).group(1))
    k=pd.read_csv(glob.glob(f"{d}/results_sqr_network_only/*/sqr_kpm_history.csv")[0]); k["sla"]=k.sliceId.map(SLA)
    dm=k[k.dTxBytes>0]; nw=k.groupby("timestamp").ueImsi.nunique()
    log=open(f"{d}.log").read() if glob.glob(f"{d}.log") else ""
    rlf=len(re.findall(r"UE_RNTI_CHANGED",log))
    pw=glob.glob(f"{d}/results_sqr_network_only/*/sqr_prb_windows.csv")
    util=idle=np.nan
    if pw:
        w=pd.read_csv(pw[0]); tot=w["total_rbg"].iloc[0]
        sch=w[[c for c in w.columns if c.startswith("prb_scheduled_rbg_")]].sum(axis=1)
        idl=w[[c for c in w.columns if c.startswith("prb_idle")]].sum(axis=1) if any(c.startswith("prb_idle") for c in w.columns) else tot-sch
        util=sch.mean()/tot; idle=idl.mean()/tot
    r={"lam":lam,"seed":seed,"n_mean":nw.mean(),"n_p95":nw.quantile(.95),
       "viol_pct":100*(dm.windowThroughputMbps<dm.sla).mean(),
       **{f"viol_s{s}_pct":100*(dm[dm.sliceId==s].windowThroughputMbps<dm[dm.sliceId==s].sla).mean() for s in SLA},
       "thr_ue_med":dm.groupby("ueImsi").windowThroughputMbps.median().median(),
       "snr_p5":dm[dm.snrDb!=-999].snrDb.quantile(.05),"snr_mean":dm[dm.snrDb!=-999].snrDb.mean(),
       "queue_p95":dm.queueBytes.quantile(.95),"rlf":rlf,"util":util,"idle":idle}
    rows.append(r)
df=pd.DataFrame(rows).sort_values(["lam","seed"]); df.to_csv(f"{R}/characterization_round1.csv",index=False)
g=df.groupby("lam").agg(["mean","min","max"]).round(3); g.to_csv(f"{R}/characterization_round1_by_lambda.csv")
fig,ax=plt.subplots(1,3,figsize=(15,4))
for m,c in [("util","#1f77b4"),("viol_pct","#d62728"),("snr_p5","#2ca02c")]:
    a=ax[["util","viol_pct","snr_p5"].index(m)]
    if m in df and df[m].notna().any():
        gb=df.groupby("lam")[m]; a.errorbar(gb.mean().index,gb.mean().values,yerr=[gb.mean()-gb.min(),gb.max()-gb.mean()],marker="o",capsize=4,color=c)
        a.set_xlabel("λ (UEs/min)"); a.set_ylabel(m); a.grid(alpha=.3)
ax[0].axhline(0.8,ls="--",c="gray",lw=.8); ax[1].axhline(5,ls="--",c="gray",lw=.8)
fig.suptitle("Rodada 1 — varredura de carga (300 m congelado, greedy, equal-split, 3 seeds)"); fig.tight_layout()
fig.savefig(f"{R}/figures_round1.png",dpi=130)
print(g.to_string()); print(f"\n[ok] CSV + figures_round1.png em {R}")
