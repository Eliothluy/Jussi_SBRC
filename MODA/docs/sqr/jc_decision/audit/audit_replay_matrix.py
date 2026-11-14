"""D0A audit_replay_matrix — recompute the 5×3 matrix from RAW traces.

Protocolo SQR_JC_Protocolo_Implementacao_Auditoria_v2.md §5.3 (GATE
REPLAY_5x3). Para cada um dos 5 estados × 3 ações:

  A0_ORIGINAL (18,17,18): hetero_sweep/A_lam{λ}_s{seed} (run congelado 2.3)
  A1_FLOORS   (37,13,3):  oracle_recalibrated_v2/support_extension/replays_v2/E1_*
  A2_RELAXED  (48,5,0):   oracle/executable_validation/replays/G2_*

Recomputa por slice, com as FUNÇÕES CONGELADAS do adjudicador II-c
(window_kpis/_sla_oks — mesmas convenções R41/R45), e acrescenta:
  - status de evidência R19 por slice (COMPLETE / NO_ACTIVE_DEMAND);
  - julgamento conjunto em DUAS semânticas explicitadas:
      joint_frozen : julgamento de janela do Gate II-d (vacuous = satisfeito;
                     a regra PRÉ-REGISTRADA sob a qual os replays foram
                     adjudicados na Rodada 2 — adjudication.py:404-412);
      joint_strict : ALL3_COMPLETE do protocolo v2 §4.3/§5.10 (qualquer
                     NO_ACTIVE_DEMAND => UNKNOWN; jamais PASS por vacuidade);
  - re-validação A/A OFFLINE: prefixo KPM (timestamp < t·100 ms) e prefixo
    de delays (txSec < t/10 s) do replay vs run original — idênticos?
  - ação efetivamente executada na janela t (sqr_prb_windows.csv).

Não altera nenhum artefato original; escreve D0A_replay_5x3.csv em
MODA/docs/sqr/jc_decision/audit/.
"""
import os
import sys

import numpy as np
import pandas as pd

GYM = "/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym"
sys.path.insert(0, os.path.join(GYM, "src"))

from nsoran.sqr.oracle.adjudication import _sla_oks, window_kpis  # frozen
from nsoran.sqr.oracle.replay_sources import replay_paths

R2 = f"{GYM}/results/sqr_characterization_round2"
OUT_DIR = "/home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/audit"

STATES = [  # (lam, seed, t) — os 5 evitáveis confirmados (g2_deltas sla True)
    (6, 3, 298), (4, 3, 1812), (5, 1, 1955), (4, 1, 2210), (5, 3, 1677),
]
ACTIONS = {
    "A0_ORIGINAL": (18, 17, 18),
    "A1_FLOORS_37-13-3": (37, 13, 3),
    "A2_RELAXED_48-5-0": (48, 5, 0),
}


def dir_of(lam: int, seed: int, t: int, action_id: str) -> str:
    if action_id == "A0_ORIGINAL":
        return f"{R2}/hetero_sweep/A_lam{lam}_s{seed}"
    if action_id.startswith("A1"):
        return (f"{R2}/oracle/oracle_recalibrated_v2/support_extension/"
                f"replays_v2/E1_equal_lam{lam}_s{seed}_t{t}_a37-13-3")
    return (f"{R2}/oracle/executable_validation/replays/"
            f"G2_equal_lam{lam}_s{seed}_t{t}_a48-5-0")


def prefix_checks(replay_dir: str, orig_dir: str, t: int) -> dict:
    """A/A offline: prefixos KPM/delays do replay idênticos ao original."""
    rp, op = replay_paths(replay_dir), replay_paths(orig_dir)
    stamp_cut = t * 100  # KPM rows strictly before the override window
    sec_cut = t / 10.0
    out = {}
    rk = pd.read_csv(rp["kpm"])
    ok = pd.read_csv(op["kpm"])
    r_pre = rk[rk["timestamp"] < stamp_cut].reset_index(drop=True)
    o_pre = ok[ok["timestamp"] < stamp_cut].reset_index(drop=True)
    kpm_eq = r_pre.equals(o_pre)
    out["kpm_prefix_rows_r_o"] = f"{len(r_pre)}/{len(o_pre)}"
    out["kpm_prefix_identical"] = bool(kpm_eq)
    if not kpm_eq and len(r_pre) == len(o_pre):
        neq = (r_pre.astype(str).values != o_pre.astype(str).values).any(axis=1)
        out["kpm_prefix_first_diff"] = str(r_pre.index[np.flatnonzero(neq)[:1]])
    rd = pd.read_csv(rp["delays"])
    od = pd.read_csv(op["delays"])
    # REGRA CONGELADA II-b (adjudication.py:307-325): pre-override =
    # ENTREGA (txSec+delaySec)·1000 < t·100 ms — pacotes entregues dentro da
    # janela t podem deslocar ±1 slot legítimo sob o override e ficam fora
    # da alegação de bit-exatidão. Ordenar por (txSec, delaySec) e .equals().
    r_ms = (rd["txSec"] + rd["delaySec"]) * 1000.0
    o_ms = (od["txSec"] + od["delaySec"]) * 1000.0
    a = rd[r_ms < stamp_cut].sort_values(["txSec", "delaySec"]).reset_index(drop=True)
    b = od[o_ms < stamp_cut].sort_values(["txSec", "delaySec"]).reset_index(drop=True)
    out["delays_prefix_rows_r_o"] = f"{len(a)}/{len(b)}"
    out["delays_prefix_identical"] = bool(a.equals(b))
    if not out["delays_prefix_identical"] and len(a) == len(b):
        neq = (a.astype(str).values != b.astype(str).values).any(axis=1)
        out["delays_prefix_first_diff"] = str(a.index[np.flatnonzero(neq)[:1]])
    return out


def main() -> int:
    rows = []
    for lam, seed, t in STATES:
        for aid, avec in ACTIONS.items():
            d = dir_of(lam, seed, t, aid)
            k = window_kpis(d, t)
            sla = _sla_oks(k)
            # coorte estrita mMTC (100 ms da janela do override)
            _p = replay_paths(d)
            _k = pd.read_csv(_p["kpm"])
            _g = _k[_k["timestamp"] == (t + 1) * 100]
            _m = _g[_g["sliceId"] == 2]
            mmtc_tx = float(_m["dTxPackets"].sum())
            mmtc_rx = float(_m["dRxPackets"].sum())
            ev_e = "COMPLETE" if k["n_embb"] >= 1 else "NO_ACTIVE_DEMAND"
            ev_u = "COMPLETE" if k["urllc_n_pkts"] >= 1 else "NO_ACTIVE_DEMAND"
            ev_m = "COMPLETE" if k["mmtc_demand"] else "NO_ACTIVE_DEMAND"
            joint_frozen = all((sla["observed_embb_ok"], sla["observed_urllc_ok"],
                                sla["observed_mmtc_ok"]))
            if "NO_ACTIVE_DEMAND" in (ev_e, ev_u, ev_m):
                joint_strict = "UNKNOWN"
            else:
                joint_strict = "PASS" if joint_frozen else "FAIL"
            row = {
                "state_id": f"lam{lam}_s{seed}_t{t}",
                "lambda": lam, "seed": seed, "decision_index": t,
                "action_id": aid, "action_vector": str(avec),
                "executed_action": str(k["executed_action"]),
                "action_matches": str(k["executed_action"]) == str(list(avec)),
                "eMBB_value": k["embb_throughput"],
                "eMBB_n_uewins": k["n_embb"],
                "eMBB_status": ev_e,
                "eMBB_outcome": ("PASS" if k["n_embb"] < 1 else
                                 ("PASS" if sla["observed_embb_ok"] else "FAIL")),
                # §5.3 atenção crítica mMTC: coorte estrita da janela (100 ms)
                # vs rolling congelado 1 s (que carrega histórico pré-t).
                "mMTC_tx_100ms": mmtc_tx, "mMTC_rx_100ms": mmtc_rx,
                "mMTC_pdr_100ms": (mmtc_rx / mmtc_tx) if mmtc_tx > 0 else np.nan,
                "mMTC_evidence_in_window": bool(mmtc_tx > 0),
                "URLLC_p95_ms": k["urllc_latency_p95"],
                "URLLC_miss_pct": k["urllc_deadline_miss"],
                "URLLC_n_pkts": k["urllc_n_pkts"],
                "URLLC_status": ev_u,
                "URLLC_outcome": ("PASS" if k["urllc_n_pkts"] < 1 else
                                  ("PASS" if sla["observed_urllc_ok"] else "FAIL")),
                "mMTC_pdr": k["mmtc_pdr"],
                "mMTC_demand": k["mmtc_demand"],
                "mMTC_status": ev_m,
                "mMTC_outcome": ("PASS" if not k["mmtc_demand"] else
                                 ("PASS" if sla["observed_mmtc_ok"] else "FAIL")),
                "joint_frozen_vacuous_ok": joint_frozen,
                "joint_strict_ALL3_COMPLETE": joint_strict,
            }
            if aid != "A0_ORIGINAL":
                row.update(prefix_checks(d, dir_of(lam, seed, t, "A0_ORIGINAL"), t))
            rows.append(row)
            print(f"[5x3] λ{lam}s{seed}t{t} {aid}: exec={k['executed_action']} "
                  f"eMBB={k['embb_throughput']:.2f}({ev_e}) "
                  f"URLLC p95={k['urllc_latency_p95']:.2f} n={k['urllc_n_pkts']}"
                  f"({ev_u}) mMTC pdr={k['mmtc_pdr']:.3f}({ev_m}) "
                  f"frozen={joint_frozen} strict={joint_strict}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/D0A_replay_5x3.csv", index=False)
    print(f"\n[5x3] -> {OUT_DIR}/D0A_replay_5x3.csv ({len(df)} linhas)")

    # resumo por ação (as duas semânticas)
    for aid in ACTIONS:
        sub = df[df.action_id == aid]
        frozen = sub.joint_frozen_vacuous_ok.sum()
        strict = (sub.joint_strict_ALL3_COMPLETE == "PASS").sum()
        unk = (sub.joint_strict_ALL3_COMPLETE == "UNKNOWN").sum()
        print(f"  {aid}: frozen OK {frozen}/5 | strict ALL3 PASS {strict}/5 "
              f"(UNKNOWN {unk}) | action_match {sub.action_matches.all()}")
    pre = df[df.action_id != "A0_ORIGINAL"]
    print(f"  prefixos A/A: kpm idênticos "
          f"{pre.kpm_prefix_identical.sum()}/{len(pre)} · delays idênticos "
          f"{pre.delays_prefix_identical.sum()}/{len(pre)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
