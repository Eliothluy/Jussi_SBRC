"""Etapa 2.4-II-c driver — executable adjudication of the oracle (§4–§11).

Validates the 28 Despacho-II-b replays (§4), builds
``oracle_replay_results.csv`` (§5), adjudicates G1–G4 (§6–§9), builds the
confusion matrix with INFEASIBLE as the positive class (§10) and applies the
confidence gate (§11).  On PASS the caller runs
``examples/sqr_oracle_finalize.py`` (§12–§21); on STOP nothing is frozen
(protocol §11: NÃO congelar P_inf, NÃO escolher L_O).

Usage (from ``MODA/ns-o-ran-gym``)::

    .venv/bin/python examples/sqr_oracle_adjudicate.py \
        [--oracle_dir results/sqr_characterization_round2/oracle] \
        [--round2_dir results/sqr_characterization_round2]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from nsoran.sqr.oracle.adjudication import (
    adjudicate_g1,
    adjudicate_g2,
    adjudicate_g3,
    adjudicate_g4,
    build_replay_results,
    confidence_gate,
    confusion_counts,
    g4_verdict,
)

ORACLE = "results/sqr_characterization_round2/oracle"
ROUND2 = "results/sqr_characterization_round2"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Etapa 2.4-II-c adjudication")
    ap.add_argument("--oracle_dir", default=os.path.abspath(ORACLE))
    ap.add_argument("--round2_dir", default=os.path.abspath(ROUND2))
    args = ap.parse_args(argv)
    od = args.oracle_dir

    # ---- §23 artifact layout ------------------------------------------------
    rep_dir = os.path.join(od, "replay")
    adj_dir = os.path.join(od, "adjudication")
    for d in (os.path.join(rep_dir, "raw"), os.path.join(rep_dir, "validated"),
              adj_dir):
        os.makedirs(d, exist_ok=True)

    # pointer for §23 replay/raw (the runs live in executable_validation/)
    with open(os.path.join(rep_dir, "raw", "REPLAYS_LOCATION.txt"), "w") as fh:
        fh.write(
            "Os 28 runs de replay (saída bruta ns-3 + telemetria) estão em\n"
            f"{os.path.join(od, 'executable_validation', 'replays')}/<tag>/\n"
            "(Despacho II-b, runner oracle/run_replays.sh; um diretório por\n"
            "linha de executable_validation/replay_plan.csv).\n")

    # ---- §4 + §5 ------------------------------------------------------------
    from nsoran.sqr.oracle.adjudication import RESULTS_COLUMNS
    res, validations = build_replay_results(od, args.round2_dir)
    # §5 EXACT column order on the main CSV; the two adjudication auxiliaries
    # (util_active_dl / executed_action) ride in a side file.
    res[list(RESULTS_COLUMNS)].to_csv(
        os.path.join(rep_dir, "oracle_replay_results.csv"), index=False)
    res[["replay_id", "util_active_dl", "executed_action"]].to_csv(
        os.path.join(rep_dir, "validated", "replay_window_aux.csv"), index=False)
    for v in validations:
        tag = v["replay_id"]
        with open(os.path.join(rep_dir, "validated", f"{tag}.json"), "w") as fh:
            json.dump(v, fh, indent=2, ensure_ascii=False, default=str)
    n_valid = int(res["replay_valid"].sum())
    print(f"[II-c] §4: {n_valid}/{len(res)} replays válidos")
    for v in validations:
        if not v["replay_valid"]:
            print(f"[II-c]   INVÁLIDO {v['replay_id']}: "
                  f"{v['checks'].get('failure')}")

    # ---- §6–§9 --------------------------------------------------------------
    ow = pd.read_csv(os.path.join(od, "oracle_windows.csv"))
    ex = pd.read_csv(os.path.join(od, "tables", "oracle_windows_extra.csv"))

    g1 = adjudicate_g1(res)
    g2 = adjudicate_g2(res, ex, ow)
    g3 = adjudicate_g3(res, ow, ex)
    curve = adjudicate_g4(res, ex)
    verdict4 = g4_verdict(curve)

    # ---- §10 ----------------------------------------------------------------
    cm = confusion_counts(res)
    pd.DataFrame([{"classe": "TP_feas (oracle ok, replay ok)",
                   "n": cm["TP_feas"]},
                  {"classe": "FP_feas (oracle ok, replay violou)",
                   "n": cm["FP_feas"]},
                  {"classe": "FN_feas / FALSE-INFEASIBLE (oracle inf, replay ok)",
                   "n": cm["FN_feas_false_infeasible"]},
                  {"classe": "TN_infeas (oracle inf, replay violou)",
                   "n": cm["TN_infeas"]}]).to_csv(
        os.path.join(adj_dir, "confusion_matrix.csv"), index=False)

    # ---- group summary (§17 input) ------------------------------------------
    g2_acr = float(g2["sla_todos_ok_no_override"].mean()) if len(g2) else float("nan")
    g3_lit = (float(g3["corrobora_a_star"].mean()) if len(g3) else float("nan"))
    g3_strict = (float(g3["corrobora_estrita"].mean()) if len(g3)
                 else float("nan"))
    n_counter = int(g3["contraprova_executavel"].sum()) if len(g3) else 0
    summary = pd.DataFrame([
        {"group": "G1", "alvo": "false-infeasible (critério AUXILIAR p25)",
         "n": g1["n_executed"],
         "confirmados": g1["n_executed"] - g1["false_infeasible_count"],
         "falhas": g1["false_infeasible_count"],
         "taxa": g1["false_infeasible_rate"],
         "leitura": ("FN = janelas em que a testemunha p25 satisfaz todos os "
                     "SLAs no replay — false-infeasible da porta mediana")},
        {"group": "G2", "alvo": "violação evitável (ACR)", "n": len(g2),
         "confirmados": int(g2["sla_todos_ok_no_override"].sum())
         if len(g2) else 0,
         "falhas": int((~g2["sla_todos_ok_no_override"]).sum()) if len(g2) else 0,
         "taxa": g2_acr,
         "leitura": "ACR = SLA recuperado com mesma seed/estado, outra ação"},
        {"group": "G3", "alvo": "inviabilidade corroborada (ICR)", "n": len(g3),
         "confirmados": int(g3["corrobora_estrita"].sum()) if len(g3) else 0,
         "falhas": int((~g3["corrobora_estrita"]).sum()) if len(g3) else 0,
         "taxa": g3_strict,
         "leitura": (f"ICR estrita (a* viola E sem contraprova equal/prop); "
                     f"ICR literal do a* = {g3_lit:.3f}; "
                     f"{n_counter} contraprovas executáveis")},
        {"group": "G4", "alvo": "causalidade URLLC a_u", "n": len(curve[
            curve["fonte"] == "replay_override"]),
         "confirmados": verdict4["interp_A_resource_dependent"],
         "falhas": verdict4["interp_B_scheduler_tdd"],
         "taxa": (verdict4["interp_A_resource_dependent"] /
                  max(1, verdict4["interp_A_resource_dependent"]
                      + verdict4["interp_B_scheduler_tdd"])),
         "leitura": ("janelas RESOURCE-DEPENDENT vs SCHEDULER/TDD-DOMINATED "
                     "(a_u fora do suporte, R43/R48)")},
    ])
    summary.to_csv(os.path.join(adj_dir, "group_summary.csv"), index=False)
    curve.to_csv(os.path.join(adj_dir, "g4_urllc_curve.csv"), index=False)
    g2.to_csv(os.path.join(adj_dir, "g2_deltas.csv"), index=False)
    g3.to_csv(os.path.join(adj_dir, "g3_deficits.csv"), index=False)

    # ---- §11 gate -----------------------------------------------------------
    gate = confidence_gate(res, g1, g2, g3, cm)
    with open(os.path.join(adj_dir, "gate_verdict.json"), "w") as fh:
        json.dump({"gate": gate, "g4_verdict": verdict4,
                   "g1": g1}, fh, indent=2, ensure_ascii=False, default=str)

    # ---- adjudication notes (numbers the report cites) -----------------------
    with open(os.path.join(adj_dir, "adjudication_notes.md"), "w") as fh:
        fh.write(f"# Notas de adjudicação II-c (geradas por "
                 f"examples/sqr_oracle_adjudicate.py)\n\n")
        fh.write(f"- §4 validação: {n_valid}/{len(res)} replays válidos.\n")
        fh.write(f"- G1 (§6): FIR sob critério AUXILIAR p25 = "
                 f"{g1['false_infeasible_count']}/{g1['n_executed']} = "
                 f"{g1['false_infeasible_rate']:.3f} — NÃO é a taxa da porta "
                 "principal (mediana); a amostra G1 foi SELECIONADA como "
                 "suspeita (R48).\n")
        fh.write(f"- G2 (§7): ACR = {g2_acr:.3f} ({int(g2['sla_todos_ok_no_override'].sum())}/{len(g2)}).\n")
        fh.write(f"- G3 (§8): ICR literal do a* = {g3_lit:.3f}; ICR estrita "
                 f"(sem contraprova equal/prop) = {g3_strict:.3f} "
                 f"({n_counter} contraprovas executáveis); D_min obs vs "
                 "predito em g3_deficits.csv.\n")
        fh.write(f"- G4 (§9): {verdict4['interp_A_resource_dependent']} "
                 "janela(s) RESOURCE-DEPENDENT, "
                 f"{verdict4['interp_B_scheduler_tdd']} SCHEDULER/TDD-DOMINATED.\n")
        fh.write(f"- §10 confusion (classe positiva INFEASIBLE, porta "
                 f"mediana): {json.dumps(cm, ensure_ascii=False)}\n")
        fh.write(f"- §11 gate: **{gate['verdict']}**\n")
        for c in gate["checks"]:
            fh.write(f"  - {c['id']}: value={c['value']} "
                     f"({c['threshold']}) -> "
                     f"{'PASS' if c['passed'] else 'FAIL'} — {c['note']}\n")

    print(f"[II-c] G1 FIR(p25 aux)={g1['false_infeasible_rate']:.3f} "
          f"({g1['false_infeasible_count']}/{g1['n_executed']}) | "
          f"G2 ACR={g2_acr:.3f} | G3 ICR lit={g3_lit:.3f} "
          f"estrita={g3_strict:.3f}")
    print(f"[II-c] G4: A={verdict4['interp_A_resource_dependent']} "
          f"B={verdict4['interp_B_scheduler_tdd']}")
    print(f"[II-c] confusion: {json.dumps(cm)}")
    for c in gate["checks"]:
        print(f"[II-c] gate {c['id']}: "
              f"{'PASS' if c['passed'] else 'FAIL'} ({c['note']})")
    print(f"[II-c] §11 GATE = {gate['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
