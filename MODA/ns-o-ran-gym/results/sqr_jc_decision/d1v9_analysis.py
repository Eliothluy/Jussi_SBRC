#!/usr/bin/env python
"""d1v9_analysis.py — indicadores da D1-v9 (Emenda 4: delayed feedback).

Por run: decisões bootstrap/B0, P_projection (B0 c/ outcome), updates por
slice (estágio 1 = eMBB/mMTC normais; estágio 2 = URLLC delayed com age),
latência das atualizações URLLC, e contagem de categorias R19
(NO_DEMAND / INCOMPLETE-expired / CLOSED). Universo explícito em todo
indicador (regra da errata D1-v8).
"""
import glob
import json
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE, sys.argv[1] if len(sys.argv) > 1
                   else "d1_runs_v9_delayed")


def load(run):
    dec = [json.loads(l) for l in open(os.path.join(run, "decisions.jsonl"))]
    df = pd.DataFrame(dec)
    ups = []
    f = os.path.join(run, "urllc_delayed_updates.jsonl")
    if os.path.exists(f):
        ups = [json.loads(l) for l in open(f)]
    return df, pd.DataFrame(ups)


rows = []
tot = {"dec_bs": 0, "dec_b0": 0, "b0_out": 0}
for run in sorted(glob.glob(os.path.join(RUNS, "het_*"))):
    name = os.path.basename(run).replace("het_", "")
    df, ups = load(run)
    if df.empty:
        continue
    bs = df[df.policy == "SQR_HET_BOOTSTRAP_PROP"]
    b0 = df[df.policy == "SQR_HET"]
    b0_out = b0[b0.get("outcome_window_index").notna()] if "outcome_window_index" in b0 else b0
    # margens não-null por slice no estágio 1 (proxy de updates eMBB/mMTC)
    def n_upd(i):
        if "sla_outcome_margins" not in df or df["sla_outcome_margins"].isna().all():
            return 0
        return int(sum(1 for m in df["sla_outcome_margins"].dropna()
                       if m[i] is not None))
    u_closed = ups[ups["tag"] == "CLOSED"] if not ups.empty else pd.DataFrame()
    u_exp = ups[ups.get("tag") == "OPEN"] if not ups.empty else pd.DataFrame()
    r = {
        "run": name,
        "dec_bootstrap": len(bs), "dec_B0": len(b0),
        "B0_c_outcome": len(b0_out),
        "p_proj_pct": round(100 * b0["projection_triggered"].mean(), 1)
                      if len(b0) else None,
        "upd_eMBB_s1": n_upd(0), "upd_mMTC_s1": n_upd(2),
        "upd_URLLC_s2": len(u_closed),
        "urllc_age_med": u_closed["age"].median() if len(u_closed) else None,
        "urllc_age_p95": u_closed["age"].quantile(.95) if len(u_closed) else None,
        "urllc_no_demand": int((ups["tag"] == "NO_DEMAND").sum())
                           if not ups.empty else None,
        "urllc_expired": int((ups.get("reason") == "expired_vacuous_R19").sum())
                         if not ups.empty else None,
    }
    tot["dec_bs"] += len(bs); tot["dec_b0"] += len(b0); tot["b0_out"] += len(b0_out)
    rows.append(r)

out = pd.DataFrame(rows)
print(out.to_string(index=False))
print(f"\nTOTAL decisões: {tot['dec_bs']} bootstrap + {tot['dec_b0']} B0 "
      f"({tot['b0_out']} c/ outcome)")
print(f"TOTAL updates URLLC delayed (estágio 2): {out['upd_URLLC_s2'].sum()}")
out.to_csv(os.path.join(BASE, "d1v9_summary.csv"), index=False)
print("-> d1v9_summary.csv")
