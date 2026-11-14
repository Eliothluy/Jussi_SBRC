#!/usr/bin/env python
"""d3_adjudicate.py — adjudicação pareada D3 (B0 × B1 × equal).

Para cada estado da amostra confirmatória mínima (d3_replay_plan.csv):
  1. A/A: prefixo KPM bit-exact até t·100 do ramo vs run base (por ramo);
  2. labels OFFLINE da janela t por ramo (window_kpis/_sla_oks congelados):
     STRICT primário + e∧u + UNKNOWN;
  3. comparação pareada: B0 (run base, offline) × B1 (replay) × equal.
Saída: d3_adjudication.csv + sumário de gate §5.7.
"""
import glob
import json
import os
import sys

import ast

import numpy as np
import pandas as pd

GYM = "/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym"
sys.path.insert(0, os.path.join(GYM, "src"))
from nsoran.sqr.oracle.adjudication import _sla_oks, window_kpis
from nsoran.sqr.oracle.replay_sources import replay_paths

HERE = os.path.join(GYM, "results/sqr_jc_decision")
REPL = os.path.join(HERE, "d3_replays")
D1 = os.path.join(HERE, "d1_runs_v9_delayed")


def label_of(replay_dir: str, t: int) -> dict:
    k = window_kpis(replay_dir, t)
    ev = ["COMPLETE" if k["n_embb"] >= 1 else "NO_ACTIVE_DEMAND",
          "COMPLETE" if k["urllc_n_pkts"] >= 1 else "NO_ACTIVE_DEMAND",
          "COMPLETE" if k["mmtc_demand"] else "NO_ACTIVE_DEMAND"]
    sla = _sla_oks(k)
    frozen = all((sla["observed_embb_ok"], sla["observed_urllc_ok"],
                  sla["observed_mmtc_ok"]))
    eu = bool(sla["observed_embb_ok"]) and bool(sla["observed_urllc_ok"])
    if "NO_ACTIVE_DEMAND" in ev:
        lab = "UNKNOWN"
    else:
        lab = "STRICT_PASS" if frozen else "STRICT_FAIL"
    eu_lab = ("EU_UNKNOWN" if "NO_ACTIVE_DEMAND" in ev[:2]
              else ("EU_PASS" if eu else "EU_FAIL"))
    return {"label": lab, "eu": eu_lab, "ev": "|".join(ev),
            "embb": k["embb_throughput"], "p95": k["urllc_latency_p95"],
            "pdr": k["mmtc_pdr"]}


def main() -> int:
    plan = pd.read_csv(os.path.join(HERE, "d3_replay_plan.csv"))
    rows = []
    for _, r in plan.iterrows():
        sid = f"D3_{r.kind}_{r['run'].split('_lam')[1]}_t{int(r.win)}"
        t = int(r.win)
        base_kpm = glob.glob(os.path.join(D1, r["run"], "**",
                                          "sqr_kpm_history.csv"),
                             recursive=True)[0]
        ko = pd.read_csv(base_kpm)
        pref_b = ko[ko.timestamp <= t * 100].reset_index(drop=True)
        branches = {}
        for branch in ("b1", "eq"):
            d = os.path.join(REPL, f"D3_{r.kind}_{r[chr(114)+chr(117)+chr(110)].split(chr(95)+chr(108)+chr(97)+chr(109))[1]}_t{int(r.win)}_{branch}")
            if not os.path.isdir(d):
                continue
            ka = pd.read_csv(glob.glob(os.path.join(
                d, "**", "sqr_kpm_history.csv"), recursive=True)[0])
            pa = ka[ka.timestamp <= t * 100].reset_index(drop=True)
            aa = (len(pa) == len(pref_b)
                  and pa.astype(str).equals(pref_b.astype(str)))
            lab = label_of(d, t)
            branches[branch] = lab
            rows.append({"kind": r.kind, "sid": sid, "run": r["run"],
                         "lam": int(r.lam), "win": t,
                         "branch": branch,
                         "action": (list(map(int, ast.literal_eval(r.b1))) if branch == "b1"
                                    else [18, 17, 18]),
                         "aa_ok": bool(aa), **lab})
        # B0 do run base (janela t): strict do par offline
        pr = [json.loads(l) for l in open(os.path.join(
            HERE, "d3_test", r["run"], "decisions_paired.jsonl"))]
        b0 = [x for x in pr if x["win"] == t]
        rows.append({"kind": r.kind, "sid": sid, "run": r["run"],
                     "lam": int(r.lam), "win": t, "branch": "B0",
                     "action": list(map(int, ast.literal_eval(r.a_B0))),
                     "aa_ok": True,
                     "label": (b0[0]["outcome_strict"] if b0 else "?"),
                     "eu": ("EU_PASS" if (b0 and b0[0]["outcome_strict"]
                                          == "STRICT_PASS")
                            else ("EU_FAIL" if (b0 and b0[0]["outcome_strict"]
                                                == "STRICT_FAIL")
                                 else "EU_UNKNOWN")),
                     "ev": "", "embb": None, "p95": None, "pdr": None})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "d3_adjudication.csv"), index=False)
    piv = df.pivot_table(index=["kind", "sid", "win"], columns="branch",
                         values="label", aggfunc="first")
    print(piv.to_string())
    print("\nA/A por ramo:", df[df.branch != "B0"].aa_ok.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
