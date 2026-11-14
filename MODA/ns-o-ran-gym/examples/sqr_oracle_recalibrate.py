#!/usr/bin/env python
"""Driver da Etapa 2.4-II-d parte 1 — recalibracao do oracle (banda 3-way).

Pipeline (protocolo ``docs/artigo/fase_2_4_II_d_recalibracao_oracle.md``,
comando §40 itens 1-8; o GATE de replay em si e a parte 2 do despacho):

1. carrega o dataset congelado de janelas da 2.3 (89.970 janelas);
2. exporta o desenho CONGELADO (frozen_thresholds.json, regra §23) ANTES do
   gate;
3. refit eMBB/URLLC quantilico (p25/p50/p75) + gates leave-one-seed-out
   (§9/§11/§16);
4. reprocessamento 3-way completo no ESPACO SUPORTADO (§14) com
   D_min_support em envelopes (§15) — nomes PROVISORIOS P_F/M/I_candidate
   (§17; nenhum P_inf_final);
5. planos de replay: D6 (microcalibracao mMTC, §13) e D1-D5 + controle G4
   (§18-§20) — formato do executor run_replays.sh;
6. model cards, placeholders mMTC, figuras e relatorio parcial II-d-1.

Artefatos em results/sqr_characterization_round2/oracle/oracle_recalibrated/
(árvore §32).  NENHUMA simulacao ns-3 e lancada aqui.

Uso:
    .venv/bin/python examples/sqr_oracle_recalibrate.py [--am-min 3]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_GYM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_GYM_ROOT, "src"))

from nsoran.sqr.oracle import quantile as qm  # noqa: E402
from nsoran.sqr.oracle import three_way as tw  # noqa: E402
from nsoran.sqr.oracle import replay_plan_v2 as rp  # noqa: E402

R2 = os.path.join(_GYM_ROOT, "results", "sqr_characterization_round2")
ORACLE = os.path.join(R2, "oracle")
OUT = os.path.join(ORACLE, "oracle_recalibrated")


# ---------------------------------------------------------------------------
# Frozen dataset loader (identical conventions to the II-c oracle)
# ---------------------------------------------------------------------------

def load_windows() -> pd.DataFrame:
    """oracle_windows.csv (§19) + extras, + previous-window features."""
    w = pd.read_csv(os.path.join(ORACLE, "oracle_windows.csv")).rename(
        columns={"lambda_total": "lam", "time": "time_ms"})
    x = pd.read_csv(os.path.join(ORACLE, "tables", "oracle_windows_extra.csv"))
    drop = [c for c in ("n_embb", "n_urllc", "n_mmtc", "queue_embb",
                        "queue_urllc", "queue_mmtc", "util_active_dl",
                        "util_all_dl", "observed_embb_violation",
                        "observed_urllc_violation", "observed_mmtc_violation",
                        "observed_any_violation") if c in x.columns]
    d = w.merge(x.drop(columns=drop), on=["arm", "lam", "seed", "time_ms"],
                how="left")
    d = d.sort_values(["arm", "lam", "seed", "win"]).reset_index(drop=True)
    g = d.groupby(["arm", "lam", "seed"], sort=False)
    d["prev_thr_embb_med"] = g["thr_embb_med"].shift(1)
    d["prev_p95"] = g["latency_urllc_p95_ms"].shift(1)
    d["prev_miss_pct"] = g["latency_urllc_miss_pct"].shift(1)
    assert len(d) == 89970, f"expected 89970 windows, got {len(d)}"
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--am-min", type=int, default=3,
                    help="piso mMTC (placeholder 3; reprocessar com o valor "
                         "medido pelo D6 quando o orquestrador fornecer)")
    ap.add_argument("--skip-loo", action="store_true",
                    help="pula os gates leave-one-seed-out (dev apenas)")
    ap.add_argument("--plans-only", action="store_true",
                    help="recarrega o reprocessamento salvo e regenera "
                         "apenas os planos de replay (dev)")
    args = ap.parse_args()
    am_min = int(args.am_min)

    for sub in ("models", "support", "predictions", "replay_v2",
                "adjudication", "plots", "tables", "reports"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    print("[II-d-1] carregando dataset congelado ...")
    d = load_windows()
    if args.plans_only:
        cls = pd.read_csv(os.path.join(OUT, "predictions",
                                       "oracle_3way_windows.csv"))
        run_plans(d, cls, am_min)
        return

    # ---- frozen design (BEFORE the gate; §23) ---------------------------
    frozen = dict(qm.FROZEN_RECAL)
    frozen["am_min_used"] = am_min
    frozen["support_space"] = {
        "a_e": "[18, 37] (faixa executada da 2.3 — so interpolacao)",
        "a_u": ">= 13 (piso R50, protocolo §12)",
        "a_m": f">= {am_min} (placeholder ate o D6 medir; PARAMETRO)",
        "sum": "<= 53 (R01; RBGs ociosos permitidos, R27)",
    }
    frozen["d_min_envelopes"] = {
        "p25": "pessimista (thr p25, p95 p75)",
        "p50": "mediana (p50, p50)",
        "p75": "otimista (thr p75, p95 p25)",
        "uncertain": "D_min_otimista == 0 < D_min_pessimista",
    }
    frozen["mmtc_placeholder"] = ("PDR = 1.0 dentro do suporte; acoes com "
                                  f"a_m < {am_min} estao FORA do espaco "
                                  "admissivel (sem claim de modelo)")
    with open(os.path.join(OUT, "models", "frozen_thresholds.json"), "w") as f:
        json.dump(frozen, f, indent=2, ensure_ascii=False)
    ae_g, au_g, comp = tw.supported_grids(am_min)
    with open(os.path.join(OUT, "support", "support_space.json"), "w") as f:
        json.dump({
            "a_e_grid": ae_g.tolist(), "a_u_grid": au_g.tolist(),
            "n_pairs": int((comp > 0).sum()),
            "n_supported_actions_3d": int(comp.sum()),
            "am_min": am_min,
            "comp_matrix": comp.tolist(),
        }, f, indent=2)
    print(f"[II-d-1] suporte: |A| = {int(comp.sum())} acoes "
          f"({int((comp > 0).sum())} pares a_e x a_u), a_m_min = {am_min}")

    # ---- final models (fit on ALL seeds; LOO validates generalization) ---
    print("[II-d-1] fit final eMBB (LP pinball 3 quantis) ...")
    embb = qm.EmmbQuantileRecal().fit(d)
    print("[II-d-1] fit final URLLC ...")
    urllc = qm.UrllcQuantileRecal().fit(d)
    betas_rows = []
    for name, mdl in (("embb", embb), ("urllc", urllc)):
        feats = (qm.FROZEN_RECAL["embb_features"] if name == "embb"
                 else qm.FROZEN_RECAL["urllc_features"])
        for q, b in mdl.model.betas.items():
            for k, (f, coef) in enumerate(zip(feats, b)):
                betas_rows.append({"model": name, "quantile": q,
                                   "feature": f, "beta_std_space": coef,
                                   "sd_train": float(mdl.model.sd[k])})
    pd.DataFrame(betas_rows).to_csv(
        os.path.join(OUT, "models", "quantile_betas.csv"), index=False)

    # ---- LOO gates (§9/§11/§16) -----------------------------------------
    if not args.skip_loo:
        print("[II-d-1] gates leave-one-seed-out (3 folds x 2 modelos) ...")
        gates = qm.loo_gates(d)
        gates["embb"].to_csv(os.path.join(
            OUT, "tables", "oracle_3way_gates_loo_embb.csv"), index=False)
        gates["urllc"].to_csv(os.path.join(
            OUT, "tables", "oracle_3way_gates_loo_urllc.csv"), index=False)
        print("[II-d-1] gates LOO eMBB:\n",
              gates["embb"][["held_seed", "n", "mae", "near_mae",
                             "cqi_low_mae", "coverage_p25_p75",
                             "band_absorbed_marginal",
                             "band_false_infeasible",
                             "band_false_feasible"]].to_string(index=False))
        print("[II-d-1] gates LOO URLLC:\n",
              gates["urllc"][["held_seed", "n", "mae", "near_mae",
                              "coverage_p25_p75", "band_absorbed_marginal",
                              "band_false_infeasible",
                              "band_false_feasible"]].to_string(index=False))
    else:
        print("[II-d-1] --skip-loo: gates NAO executados (dev)")

    # ---- full 3-way reprocessing (§17; provisional names) ----------------
    print("[II-d-1] reprocessamento 3-way das 89.970 janelas ...")
    cls = tw.classify_windows_3way(d, embb, urllc, am_min=am_min)
    # CANONICAL location of the per-window table: tables/ (§32 minimal list;
    # the orchestrator's gate checks tables/).  A BYTE-IDENTICAL copy is
    # also written under predictions/ as gate compatibility (the gate chain
    # tests BOTH resolutions; discovered on the second gate run).
    win_csv = os.path.join(OUT, "tables", "oracle_3way_windows.csv")
    cls.to_csv(win_csv, index=False)
    cls.to_csv(os.path.join(OUT, "predictions", "oracle_3way_windows.csv"),
               index=False)
    by_lam = tw.aggregate_3way_by_lambda(cls)
    by_lam.to_csv(os.path.join(OUT, "tables", "oracle_3way_by_lambda.csv"),
                  index=False)
    cons = tw.consistency_candidates(cls)
    with open(os.path.join(OUT, "tables",
                           "consistency_candidates.json"), "w") as f:
        json.dump(cons, f, indent=2, ensure_ascii=False)
    print("[II-d-1] P_F/M/I_candidate por lambda:\n",
          by_lam[["lambda_total", "P_F_candidate", "P_M_candidate",
                  "P_I_candidate", "frac_D_min_uncertain"]].to_string(index=False))
    print("[II-d-1] consistencia (candidatos, NAO e o gate):", cons)

    run_plans(d, cls, am_min)

    print(f"[II-d-1] CONCLUIDO — artefatos em {OUT}")


def run_plans(d: pd.DataFrame, cls: pd.DataFrame, am_min: int) -> None:
    """Planos D6 e D1-D5 (+ checagens); usa o reprocessamento `cls`."""
    print("[II-d-1] planos de replay D6 e D1-D5 ...")
    plan_d6, d6_meta = rp.build_plan_d6(d)
    plan_d6[["grupo", "arm_base", "lam", "seed", "win_t", "action_e",
             "action_u", "action_m", "alvo"]].to_csv(
        os.path.join(OUT, "replay_v2", "plan_d6_mmtc.csv"), index=False)
    d6_meta.to_csv(os.path.join(OUT, "replay_v2", "plan_d6_meta.csv"),
                   index=False)
    plan = rp.build_plan_v2(cls, d, am_min=am_min)
    plan[["grupo", "arm_base", "lam", "seed", "win_t", "action_e",
          "action_u", "action_m", "alvo"]].to_csv(
        os.path.join(OUT, "replay_v2", "plan_d1d5.csv"), index=False)
    plan.to_csv(os.path.join(OUT, "replay_v2", "plan_d1d5_metadata.csv"),
                index=False)
    chk = rp.validate_plan(plan, am_min=am_min)
    chk_d6 = rp.validate_plan(plan_d6, am_min=am_min)
    with open(os.path.join(OUT, "replay_v2", "plan_checks.json"), "w") as f:
        json.dump({"d1d5": chk, "d6": chk_d6}, f, indent=2)
    print("[II-d-1] checagem do plano D1-D5:", chk)
    print("[II-d-1] checagem do plano D6 (calibracao, fora do piso de "
          "proposito):", {k: chk_d6[k] for k in
                          ("n_rows", "n_control_or_calibration", "sum_le_c")})

    # ---- figures (subset of §33 available before the gate) ---------------
    by_lam = pd.read_csv(os.path.join(OUT, "tables",
                                      "oracle_3way_by_lambda.csv"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(by_lam["lambda_total"], by_lam["P_F_candidate"], "o-",
                label="P_F_candidate")
        ax.plot(by_lam["lambda_total"], by_lam["P_M_candidate"], "s-",
                label="P_M_candidate")
        ax.plot(by_lam["lambda_total"], by_lam["P_I_candidate"], "^-",
                label="P_I_candidate")
        ax.set_xlabel("lambda total (UEs/min)")
        ax.set_ylabel("fracao de janelas (PROVISORIO — §17)")
        ax.set_title("Oracle 3-way recalibrado: P_F/P_M/P_I x carga")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "plots", "fig_pf_pm_pi_lambda.png"),
                    dpi=150)
        plt.close(fig)

        g = cls.groupby(["lam"]).agg(
            F=("num_feasible_actions", "median"),
            M=("num_marginal_actions", "median")).reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(g["lam"], g["F"], "o-", label="|F_t| (mediana)")
        ax.plot(g["lam"], g["M"], "s-", label="|M_t| (mediana)")
        ax.set_xlabel("lambda total"); ax.set_ylabel("acoes no suporte")
        ax.set_title("Contagens de acao por janela x carga")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(OUT, "plots", "fig_fset_counts_lambda.png"),
                    dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover — figures are best-effort
        print(f"[II-d-1] AVISO: figuras nao geradas ({exc})")


if __name__ == "__main__":
    main()
