#!/usr/bin/env python3
"""d2_analysis.py — adjudicação dos replays D2 (Emendas 1+2; PRONTO p/ rodar
quando run_d2_replays.sh completar os 390).

Para cada estado (30):
  1. carrega os 13 ramos (a_B0 + alt1..12) de d2_replays/<state_id>_alt*;
  2. CRITÉRIO 1 (Emenda 2): prefixo KPM bit-idêntico ENTRE ramos até t·100;
     falha ⇒ estado INVALID (reportado);
  3. KPIs da janela t por ramo (window_kpis congelado) + evidência R19 por
     slice (COMPLETE/NO_ACTIVE_DEMAND) ⇒ rótulo PRIMÁRIO STRICT
     (3×COMPLETE ∧ all-ok), FROZEN_OK (só no congelado) e UNKNOWN separados;
  4. rótulo do estado: RECOVERABLE_STRICT (≥1 alt STRICT_PASS ∧ a_B0 não),
     NOT_RECOVERED_IN_TESTED_SET, INVALID; a_B0-vs-B0: quotas (obrigatório
     0 divergências) + drift KPM reportado;
  5. relatório por estrato×λ×seed COM denominadores e fração amostrada
     (ajuste 6 — sem extrapolação p/ população).
Saídas: D2_results.csv (por ramo), D2_states.csv (por estado),
D2_report.md, 3 figuras.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

GYM = "/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym"
sys.path.insert(0, os.path.join(GYM, "src"))
from nsoran.sqr.oracle.adjudication import _sla_oks, window_kpis  # frozen
from nsoran.sqr.oracle.replay_sources import replay_paths

HERE = os.path.dirname(os.path.abspath(__file__))
REPL = os.path.join(HERE, "d2_v2_replays")
D1 = os.path.join(HERE, "d1_runs_v9_delayed")


def kpis_strict(replay_dir: str, t: int) -> dict:
    """KPIs da janela t + evidência R19 por slice (strict, Emenda 2)."""
    k = window_kpis(replay_dir, t)
    ev = ["COMPLETE" if k["n_embb"] >= 1 else "NO_ACTIVE_DEMAND",
          "COMPLETE" if k["urllc_n_pkts"] >= 1 else "NO_ACTIVE_DEMAND",
          "COMPLETE" if k["mmtc_demand"] else "NO_ACTIVE_DEMAND"]
    sla = _sla_oks(k)
    frozen = all((sla["observed_embb_ok"], sla["observed_urllc_ok"],
                  sla["observed_mmtc_ok"]))
    eu = bool(sla["observed_embb_ok"]) and bool(sla["observed_urllc_ok"])
    if "NO_ACTIVE_DEMAND" in ev[:2]:
        eu_label = "EU_UNKNOWN"
    else:
        eu_label = "EU_PASS" if eu else "EU_FAIL" 
    if "NO_ACTIVE_DEMAND" in ev:
        label = "UNKNOWN"
    else:
        label = "STRICT_PASS" if frozen else "STRICT_FAIL"
    return {"k": k, "ev": ev, "frozen": frozen, "label": label, "eu": eu_label}


def main() -> int:
    plan = pd.read_csv(f"{HERE}/D2_sampling_FINAL_V9.csv")
    rows, states = [], []
    for sid, g in plan.groupby("state_id", sort=True):
        t = int(g["win"].iloc[0])
        branches = {}
        for _, r in g.iterrows():
            a = "-".join(str(int(v)) for v in
                         (r["action"] if isinstance(r["action"], list)
                          else json.loads(r["action"])))
            d = f"{REPL}/{sid}_alt{int(r['alt_index'])}_{a}"
            if not os.path.isdir(d):
                continue
            branches[int(r["alt_index"])] = (d, a, r)
        if 0 not in branches or len(branches) < 13:
            states.append({"state_id": sid, "label": "INCOMPLETE",
                           "n_branches": len(branches)})
            continue
        # critério 1: prefixo bit-idêntico entre ramos
        pref = {}
        for i, (d, a, _) in branches.items():
            kpm = pd.read_csv(replay_paths(d)["kpm"])
            pref[i] = kpm[kpm.timestamp <= t * 100].reset_index(drop=True)
        base = pref[0]
        ok = all(pref[i].astype(str).equals(base.astype(str)) for i in pref)
        b0_run = os.path.join(D1, g["run"].iloc[0])
        labels = {}
        for i, (d, a, r) in branches.items():
            res = kpis_strict(d, t)
            labels[i] = res
            rows.append({
                "state_id": sid, "lam": int(r["lam"]), "seed": int(r["seed"]),
                "stratum": r["stratum"], "vslice": r["vslice"],
                "alt_index": i, "action": a, "mechanism": r["mechanism"],
                "substituted": bool(r["substituted"]),
                "label": res["label"], "frozen_ok": res["frozen"],
                "evidence": "|".join(res["ev"]), "eu": res["eu"],
                "embb_thr": res["k"]["embb_throughput"],
                "urllc_p95": res["k"]["urllc_latency_p95"],
                "mmtc_pdr": res["k"]["mmtc_pdr"],
            })
        alt_pass = [i for i in labels if i > 0
                    and labels[i]["label"] == "STRICT_PASS"]
        b0_pass = labels[0]["label"] == "STRICT_PASS"
        b0_fail = labels[0]["label"] == "STRICT_FAIL"
        if not ok:
            lab = "INVALID_PREFIX"
        elif b0_pass:
            lab = "B0_ALREADY_PASS"
        elif alt_pass and b0_fail:
            # recuperação STRICT genuína: violação observada na a_B0 e
            # alternativa com ALL3_COMPLETE passando (ajuste 2 do orientador)
            lab = "RECOVERABLE_STRICT"
        elif alt_pass:
            # a_B0 não-avaliável (UNKNOWN: vacuous) mas alternativas passam —
            # NÃO é recuperação (não havia violação); caso reportado à parte
            lab = "B0_UNKNOWN_ALTS_PASS"
        else:
            lab = "NOT_RECOVERED_IN_TESTED_SET"
        # a_B0 vs B0 run: quotas + drift
        q_ok, drift = True, None
        try:
            pr = pd.read_csv(replay_paths(branches[0][0])["prb"])
            po = pd.read_csv(replay_paths(b0_run)["prb"])
            # prefixo EXCLUI a janela do override (a ação candidata é a
            # única diferença permitida lá — INV-08; (t+1)*100 incluía a
            # linha da intervenção e dava False em 30/30 por desenho)
            f = lambda d: d[(d.timestamp_ms <= t * 100) & (d.dl_slots > 0)]
            # comparação robusta (valores; .equals é estrito a dtype/índice
            # e deu False esporro com drift=0 — bug de comparação, não de dado)
            q_ok = (f(pr).reset_index(drop=True).astype(str).equals(
                    f(po).reset_index(drop=True).astype(str)))
            ka = pd.read_csv(replay_paths(branches[0][0])["kpm"])
            ko = pd.read_csv(replay_paths(b0_run)["kpm"])
            m = (ka[ka.timestamp <= t * 100].reset_index(drop=True)
                 .astype(str).values !=
                 ko[ko.timestamp <= t * 100].reset_index(drop=True)
                 .astype(str).values).any(axis=1)
            drift = int(m.sum())
        except Exception:
            q_ok = False
        # A/A formal: janela t do alt0 (mesma a_B0) vs janela t do run B0
        aa_ok = None
        try:
            ka = pd.read_csv(replay_paths(branches[0][0])["kpm"])
            ko = pd.read_csv(replay_paths(b0_run)["kpm"])
            fa = ka[ka.timestamp == t * 100].reset_index(drop=True).astype(str)
            fo = ko[ko.timestamp == t * 100].reset_index(drop=True).astype(str)
            aa_ok = bool(fa.equals(fo)) if (len(fa) and len(fo)) else None
        except Exception:
            aa_ok = None
        states.append({"state_id": sid, "lam": int(g["lam"].iloc[0]),
                       "seed": int(g["seed"].iloc[0]),
                       "stratum": g["stratum"].iloc[0],
                       "label": lab, "prefix_ok": ok, "quotas_ok": q_ok, "aa_ok": aa_ok,
                       "kpm_drift_rows_vs_B0": drift,
                       "n_alt_strict_pass": len(alt_pass),
                       "first_pass_alt": (sorted(alt_pass)[0]
                                          if alt_pass else None)})
        print(f"[d2] {sid}: {lab} (prefix ok={ok}, quotas ok={q_ok}, "
              f"drift={drift}, alt PASS={len(alt_pass)})", flush=True)
    pd.DataFrame(rows).to_csv(f"{HERE}/D2v2_results.csv", index=False)
    st = pd.DataFrame(states)
    st.to_csv(f"{HERE}/D2v2_states.csv", index=False)
    # ajuste 6: por estrato×λ com denominadores (sem extrapolação)
    print("\n[d2] por estrato×λ (contagens da AMOSTRA):")
    print(st.groupby(["stratum", "lam"])["label"].value_counts()
            .unstack(fill_value=0).to_string())
    print("\n[d2] por seed:")
    print(st.groupby("seed")["label"].value_counts().unstack(fill_value=0)
          .to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
