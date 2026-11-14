#!/usr/bin/env python
"""Driver da Etapa 2.4-II-d parte 2 (FINAL) — adjudicação executável 3-way.

Consumo dos 49 replays (31 D1–D5 + 18 D6) produzidos pelo executor
``replay_v2/run_d15.sh`` / ``run_d6`` sobre o plano congelado, e saída:

1. §4 (mesmo critério do II-c): validação por replay — .done, telemetria
   parseável, stamps monótonos, C=53, override ÚNICO na janela t, cota
   executada == plano, prefixo bit-idêntico ao run congelado 2.3;
2. ``adjudication/oracle_3way_replay_results.csv``: predição do oracle 3-way
   (estado da janela + banda por eixo na ação do replay, modelos congelados)
   vs o observado na janela t+1, com o metadata do plano;
3. adjudicação por grupo D1–D6 (§20);
4. ``adjudication/confusion_matrix_3way.csv`` (§21) + duas visões
   conservadoras + métricas;
5. ``adjudication/gate_verdict_II_d.json`` (§22, C1–C7, limiares congelados).

PASS  ⇒ ``--finalize`` gera ``final/`` (P_F/P_M/P_I congelados, P_inf=P_I,
AVR, boundary, L_N/L_O, decisão SQR-JC — §24-§31) e as figuras §33.
STOP  ⇒ nada é congelado (o driver termina no passo 5).

Uso (a partir de MODA/ns-o-ran-gym)::

    .venv/bin/python examples/sqr_oracle_adjudicate_d.py                  # gate + finalize se PASS
    .venv/bin/python examples/sqr_oracle_adjudicate_d.py --skip-finalize  # só o gate
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

from nsoran.sqr.oracle import adjudication_3way as a3  # noqa: E402
from nsoran.sqr.oracle import quantile as qm           # noqa: E402
from nsoran.sqr.oracle import three_way as tw          # noqa: E402

R2 = os.path.join(_GYM_ROOT, "results", "sqr_characterization_round2")
ORACLE = os.path.join(R2, "oracle")
OUT = os.path.join(ORACLE, "oracle_recalibrated")


def load_windows() -> pd.DataFrame:
    """Frozen window dataset — IDENTICAL loader to the recalibration driver
    (import-time duplication avoided by re-implementing the 12-line recipe
    verbatim; asserted same row count)."""
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


def refit_frozen_models(d: pd.DataFrame):
    """Re-fit the FROZEN recalibrated models (deterministic pinball LP on the
    same frozen dataset — the same objects that produced the part-1
    reprocessing; rule §23: nothing about the design changes)."""
    embb = qm.EmmbQuantileRecal().fit(d)
    urllc = qm.UrllcQuantileRecal().fit(d)
    return embb, urllc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-finalize", action="store_true",
                    help="não invoca o finalize mesmo com PASS (o "
                         "orquestrador roda o finalize explicitamente)")
    args = ap.parse_args(argv)

    adj = os.path.join(OUT, "adjudication")
    os.makedirs(adj, exist_ok=True)

    # ---- frozen inputs ----------------------------------------------------
    print("[II-d-2] carregando dataset congelado e refit dos modelos ...")
    d = load_windows()
    embb, urllc = refit_frozen_models(d)
    cls_saved = pd.read_csv(os.path.join(OUT, "predictions",
                                         "oracle_3way_windows.csv"))
    # determinism guard: the refit (LP determinístico) must REPRODUCE the
    # frozen reprocessing of part 1 — same models, same state classes
    print("[II-d-2] guard de determinismo: reclassificando 89.970 janelas ...")
    cls_refit = tw.classify_windows_3way(d, embb, urllc, am_min=3)
    key_cols = ["arm", "lam", "seed", "win", "state_class",
                "num_feasible_actions", "num_marginal_actions",
                "num_infeasible_actions", "best_feasible_action",
                "best_marginal_action", "best_degradation_action",
                "D_min_support_p25_pessimistic", "D_min_support_p50",
                "D_min_support_p75_optimistic"]
    a = cls_refit[key_cols].sort_values(
        ["arm", "lam", "seed", "win"]).reset_index(drop=True)
    b = cls_saved[key_cols].sort_values(
        ["arm", "lam", "seed", "win"]).reset_index(drop=True)
    for c in ("best_feasible_action", "best_marginal_action",
              "best_degradation_action"):
        # CSV round-trip turns the empty-string sentinel into NaN
        a[c] = a[c].fillna("")
        b[c] = b[c].fillna("")
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    print("[II-d-2] guard OK: refit reproduz o reprocessamento congelado "
          "bit-a-bit (§23 respeitado — nada mudou no desenho)")

    # ---- §4 validation + §5-analog results --------------------------------
    print("[II-d-2] validação §4 + resultados dos 49 replays ...")
    res, validations, d6 = a3.build_replay_results_3way(
        OUT, R2, d, embb, urllc, cls_saved)
    res.to_csv(os.path.join(adj, "oracle_3way_replay_results.csv"),
               index=False)
    os.makedirs(os.path.join(adj, "replay_validations"), exist_ok=True)
    for v in validations:
        with open(os.path.join(adj, "replay_validations",
                               f"{v['replay_id']}.json"), "w") as fh:
            json.dump(v, fh, indent=2, ensure_ascii=False, default=str)
    n_valid = int(res["replay_valid"].sum())
    n_d1d5 = int((res["group"] != "D6").sum())
    print(f"[II-d-2] §4: {n_valid}/{len(res)} replays válidos "
          f"(D1–D5+G4: {n_d1d5}, D6: {len(res) - n_d1d5})")
    for v in validations:
        if not v["replay_valid"]:
            print(f"[II-d-2]   INVÁLIDO {v['replay_id']}: "
                  f"{v['checks'].get('failure')}")

    # ---- §20 group adjudication -------------------------------------------
    print("[II-d-2] adjudicação por grupo (§20) ...")
    groups = {
        "D1": a3.adjudicate_d1(res), "D2": a3.adjudicate_d2(res),
        "D3": a3.adjudicate_d3(res), "D4": a3.adjudicate_d4(res),
        "D5": a3.adjudicate_d5(res),
    }
    d6_curve, d6_summary = a3.adjudicate_d6(d6, None)
    pd.concat([df.assign(group=g) for g, df in groups.items()]).to_csv(
        os.path.join(adj, "group_adjudication_d1_d5.csv"), index=False)
    d6_curve.to_csv(os.path.join(adj, "d6_pdr_vs_am.csv"), index=False)
    with open(os.path.join(adj, "d6_floor_summary.json"), "w") as fh:
        json.dump(d6_summary, fh, indent=2, ensure_ascii=False)
    for g, df in groups.items():
        if len(df):
            col = "verdict" if "verdict" in df.columns \
                else "binding_in_band_or_near"
            vc = df[col].value_counts().to_dict()
            print(f"[II-d-2]   {g}: {vc}")
    print(f"[II-d-2]   D6: a_m_min_medido={d6_summary['a_m_min_measured']} "
          f"decisão={d6_summary['a_m_min_decision']}")

    # ---- §21 confusion ------------------------------------------------------
    conf = a3.confusion_matrix_3way(res)
    rows = [{"predicted": k.split("_")[0], "observed": "_".join(k.split("_")[1:]),
             "n": v} for k, v in conf["matrix"].items()]
    pd.DataFrame(rows).to_csv(os.path.join(adj, "confusion_matrix_3way.csv"),
                              index=False)
    with open(os.path.join(adj, "confusion_metrics_3way.json"), "w") as fh:
        json.dump(conf, fh, indent=2, ensure_ascii=False, default=str)
    print(f"[II-d-2] §21 confusion: {conf['matrix']} | FIR="
          f"{conf['false_infeasible_rate']} FFR={conf['false_feasible_rate']} "
          f"marginal_rate={conf['marginal_rate']} macro_f1={conf['macro_f1']}")

    # ---- §22 gate -----------------------------------------------------------
    gate = a3.gate_ii_d(res, cls_saved, d6_summary, conf)
    with open(os.path.join(adj, "gate_verdict_II_d.json"), "w") as fh:
        json.dump(gate, fh, indent=2, ensure_ascii=False, default=str)
    for c in gate["checks"]:
        print(f"[II-d-2] gate {c['id']}: "
              f"{'PASS' if c['passed'] else 'FAIL'} — {c['note']}")
    print(f"[II-d-2] §22 GATE = {gate['verdict']}")

    # ---- notes for the report ----------------------------------------------
    with open(os.path.join(adj, "adjudication_notes_3way.md"), "w") as fh:
        fh.write("# Notas de adjudicação II-d parte 2 (geradas por "
                 "examples/sqr_oracle_adjudicate_d.py)\n\n")
        fh.write(f"- §4: {n_valid}/{len(res)} replays válidos "
                 f"(31 D1–D5+G4, 18 D6).\n")
        for g, df in groups.items():
            if len(df):
                col = "verdict" if "verdict" in df.columns \
                    else "binding_in_band_or_near"
                fh.write(f"- §20 {g}: "
                         + "; ".join(f"{r.replay_id}: {getattr(r, col)}"
                                     for r in df.itertuples()) + "\n")
        fh.write(f"- §20 D6: a_m_min_medido="
                 f"{d6_summary['a_m_min_measured']}; decisão=3 "
                 "(orquestrador, conservador — ver d6_floor_summary.json)\n")
        fh.write(f"- §21 confusion: `{json.dumps(conf['matrix'])}`; FIR="
                 f"{conf['false_infeasible_rate']}; FFR="
                 f"{conf['false_feasible_rate']}; marginal_rate="
                 f"{conf['marginal_rate']}; macro_f1={conf['macro_f1']}\n")
        fh.write(f"- §21 visão conservadora F: "
                 f"`{json.dumps(conf['view_feasible'])}`\n")
        fh.write(f"- §21 visão conservadora I: "
                 f"`{json.dumps(conf['view_infeasible'])}`\n")
        fh.write(f"- §22 GATE = **{gate['verdict']}**\n")
        for c in gate["checks"]:
            fh.write(f"  - {c['id']}: value={c['value']} "
                     f"({c['threshold']}) -> "
                     f"{'PASS' if c['passed'] else 'FAIL'} — {c['note']}\n")

    # ---- PASS branch ---------------------------------------------------------
    if gate["verdict"] == "PASS":
        if args.skip_finalize:
            print("[II-d-2] GATE PASS — finalize NÃO executado "
                  "(--skip-finalize)")
        else:
            finalize(OUT, cls_saved)
    else:
        print("[II-d-2] GATE STOP — NADA é congelado (§24+ bloqueado); "
              "diagnóstico em adjudication/gate_verdict_II_d.json")
        make_figures_stop(OUT)
    return 0


def make_figures_stop(out: str) -> None:
    """§33 figures under STOP — everything labeled CANDIDATE/PROVISÓRIO;
    no final/ artifact is produced (§24+ blocked)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = os.path.join(out, "plots")
    os.makedirs(plots, exist_ok=True)
    adj = os.path.join(out, "adjudication")

    # (1) P_F/P_M/P_I × λ — CANDIDATE (the freeze is blocked by STOP)
    by_lam = pd.read_csv(os.path.join(out, "tables",
                                      "oracle_3way_by_lambda.csv"))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(by_lam["lambda_total"], by_lam["P_F_candidate"], "o-",
            label="P_F_candidate")
    ax.plot(by_lam["lambda_total"], by_lam["P_M_candidate"], "s-",
            label="P_M_candidate")
    ax.plot(by_lam["lambda_total"], by_lam["P_I_candidate"], "^-",
            label="P_I_candidate")
    ax.set_xlabel("lambda total (UEs/min)")
    ax.set_ylabel("fração de janelas (CANDIDATA — gate II-d = STOP)")
    ax.set_title("Oracle 3-way: P_F/P_M/P_I × carga\n(NÃO congelado — §24 "
                 "bloqueado pelo STOP)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "fig_pf_pm_pi_lambda.png"), dpi=150)
    plt.close(fig)

    # (2) |F_t| and |M_t| × load
    w = pd.read_csv(os.path.join(out, "predictions",
                                 "oracle_3way_windows.csv"))
    g = w[w["arm"] == "equal"].groupby("lam").agg(
        F=("num_feasible_actions", "median"),
        M=("num_marginal_actions", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g["lam"], g["F"], "o-", label="|F_t| (mediana)")
    ax.plot(g["lam"], g["M"], "s-", label="|M_t| (mediana)")
    ax.set_xlabel("lambda total")
    ax.set_ylabel("ações no suporte")
    ax.set_title("Contagens de ação por janela × carga")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "fig_fset_counts_lambda.png"), dpi=150)
    plt.close(fig)

    # (3)+(4) near-threshold LOO gates
    for name, fname, near_col, mae_col in (
            ("embb", "fig_embb_near_threshold.png", "near_mae", "mae"),
            ("urllc", "fig_urllc_near_threshold.png", "near_mae", "mae")):
        p = os.path.join(out, "tables",
                         f"oracle_3way_gates_loo_{name}.csv")
        if not os.path.isfile(p):
            continue
        t = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(t))
        ax.bar(x - 0.2, t[mae_col], 0.4, label="MAE geral")
        ax.bar(x + 0.2, t[near_col], 0.4, label="MAE near-threshold")
        ax2 = ax.twinx()
        ax2.plot(x, t["coverage_p25_p75"], "gD--", label="cobertura [p25,p75]")
        ax2.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax2.set_ylabel("cobertura")
        ax2.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {s}" for s in t["held_seed"]])
        unit = "Mbps" if name == "embb" else "ms"
        ax.set_ylabel(f"erro ({unit})")
        ax.set_title(f"Gate {name} near-threshold (LOO, refit II-d)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(plots, fname), dpi=150)
        plt.close(fig)

    # (5) PDR mMTC × a_m (D6)
    d6c = pd.read_csv(os.path.join(adj, "d6_pdr_vs_am.csv"))
    if len(d6c):
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        markers = {5.0: "o", 6.0: "s", 7.0: "^"}
        for (lam, win), gg in d6c.groupby(["lambda_total", "win"]):
            gg = gg.sort_values("a_m")
            mk = markers.get(float(lam), "o")
            ax.plot(gg["a_m"] - 0.06, gg["pdr_mmtc_1s_rolling"], mk + "-",
                    label=f"PDR 1 s rolling λ{lam:g} t{win}")
            ax.plot(gg["a_m"] + 0.06, gg["mmtc_pdr_100ms"], mk + "--",
                    alpha=0.55,
                    label=f"PDR 100 ms λ{lam:g} t{win}")
        ax.axhline(0.95, color="red", lw=1, ls=":", label="SLA PDR 0.95")
        ax.axvline(3, color="green", lw=1.2, ls="-.",
                   label="a_m_min DECIDIDO = 3 (conservador)")
        ax.axvline(1, color="orange", lw=1, ls=":",
                   label="a_m_min MEDIDO (D6) = 1")
        ax.set_xlabel("a_m (RBGs)")
        ax.set_ylabel("PDR mMTC")
        ax.set_ylim(-0.05, 1.08)
        ax.set_title("D6: PDR mMTC × a_m (microcalibração do piso, §13)\n"
                     "a_m=0 quebra o PDR (λ7 t205); a_m ≥ 1 entrega 1.0")
        ax.legend(fontsize=6.5, loc="center right")
        fig.tight_layout()
        fig.savefig(os.path.join(plots, "fig_pdr_mmtc_vs_am.png"), dpi=150)
        plt.close(fig)

    # (6) confusion matrix 3-way
    conf = json.load(open(os.path.join(adj, "confusion_metrics_3way.json")))
    m = conf["matrix"]
    tab = np.array([[m["FEASIBLE_all_ok"], m["FEASIBLE_violation"]],
                    [m["MARGINAL_all_ok"], m["MARGINAL_violation"]],
                    [m["INFEASIBLE_all_ok"], m["INFEASIBLE_violation"]]])
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    im = ax.imshow(tab, cmap="Blues")
    for r in range(3):
        for c in range(2):
            ax.text(c, r, str(tab[r, c]), ha="center", va="center",
                    color="white" if tab[r, c] > tab.max() / 2 else "black",
                    fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["replay all-OK", "replay violou"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["FEASIBLE", "MARGINAL", "INFEASIBLE"])
    ax.set_title(f"Matriz de confusão 3-way ({conf['n_witness']} testemunhas "
                 "válidas)\ngate II-d = STOP (C1, C3, C5, C7 FAIL)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "fig_confusion_matrix_3way.png"), dpi=150)
    plt.close(fig)

    # (7) state map N_embb × N_urllc
    eq = w[w["arm"] == "equal"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = {"FEASIBLE": "tab:green", "MARGINAL": "tab:orange",
              "INFEASIBLE": "tab:red"}
    rng = np.random.default_rng(0)
    for sc, c in colors.items():
        s = eq[eq["state_class"] == sc]
        if not len(s):
            continue
        ax.scatter(s["n_embb"] + rng.uniform(-0.15, 0.15, len(s)),
                   s["n_urllc"] + rng.uniform(-0.15, 0.15, len(s)),
                   s=4, alpha=0.25, c=c, label=sc)
    ax.set_xlabel("N_embb (UEs eMBB com demanda)")
    ax.set_ylabel("N_urllc (UEs URLLC)")
    ax.set_title("Mapa de estados do oracle 3-way (braço equal)\n"
                 "classes CANDIDATAS (gate STOP — não congelado)")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "fig_state_map_nembb_nurllc.png"),
                dpi=150)
    plt.close(fig)

    print(f"[II-d-2] figuras §33 (rótulos PROVISÓRIOS/candidatos, STOP) em "
          f"{plots}")


def finalize(out: str, cls_saved: pd.DataFrame) -> None:
    """§24-§31 freezing + §33 figures (PASS only)."""
    fin = os.path.join(out, "final")
    plots = os.path.join(out, "plots")
    os.makedirs(fin, exist_ok=True)
    os.makedirs(plots, exist_ok=True)

    # windows table frozen (§17 provisional names are now final; the frame is
    # the SAME frozen reprocessing the gate adjudicated)
    w = cls_saved.copy()
    w.to_csv(os.path.join(fin, "oracle_3way_windows_final.csv"), index=False)

    by_lambda = a3.freeze_by_lambda(w)
    by_lambda.to_csv(os.path.join(fin, "oracle_3way_by_lambda_final.csv"),
                     index=False)
    by_slice = a3.avr_by_slice(w)
    by_slice.to_csv(os.path.join(fin, "oracle_3way_by_slice_final.csv"),
                    index=False)
    avr = a3.avr_table(w)
    avr.to_csv(os.path.join(fin, "avr_final.csv"), index=False)

    # merge AVR into the boundary frame
    avr_ins = avr[avr["scope"] == "exec_in_support"][
        ["lambda_total", "AVR", "AVR_FM", "n"]].rename(
        columns={"AVR": "AVR_exec_in_support",
                 "AVR_FM": "AVR_FM_exec_in_support",
                 "n": "n_exec_in_support"})
    boundary = by_lambda.merge(avr_ins, on="lambda_total", how="left")
    boundary.to_csv(os.path.join(fin, "feasibility_boundary.csv"), index=False)

    lnlo = a3.select_ln_lo(by_lambda)
    jc = a3.sqr_jc_decision(by_lambda, avr, lnlo, lnlo)
    with open(os.path.join(fin, "ln_lo_selection.json"), "w") as fh:
        json.dump({**lnlo, "sqr_jc_decision": jc}, fh, indent=2,
                  ensure_ascii=False, default=str)
    with open(os.path.join(fin, "ln_lo_selection.md"), "w") as fh:
        fh.write("# Seleção de L_N e L_O (§26/§27) + decisão SQR-JC (§28–§31)\n\n")
        fh.write("Congelado APÓS o PASS do gate II-d (§24). Números de "
                 "`final/feasibility_boundary.csv`.\n\n")
        fh.write(f"## L_N = λ{lnlo['L_N']['lambda_total']}\n\n")
        fh.write(f"- P_F={lnlo['L_N']['P_F']:.3f} "
                 f"P_M={lnlo['L_N']['P_M']:.3f} P_I={lnlo['L_N']['P_I']:.3f} "
                 f"(regime {lnlo['L_N']['regime']}) — regra: {lnlo['rule_L_N']}\n")
        fh.write(f"## L_O = λ{lnlo['L_O']['lambda_total']}\n\n")
        fh.write(f"- P_F={lnlo['L_O']['P_F']:.3f} "
                 f"P_M={lnlo['L_O']['P_M']:.3f} P_I={lnlo['L_O']['P_I']:.3f} "
                 f"(regime {lnlo['L_O']['regime']}) — regra: {lnlo['rule_L_O']}\n")
        fh.write("\n## Decisão SQR-JC (especificação; implementação é fase "
                 "posterior)\n\n")
        fh.write(f"- AVR(L_N)={jc['avr_L_N']:.3f}; AVR(L_O)={jc['avr_L_O']:.3f}\n")
        for c in jc["components"]:
            fh.write(f"- {c}\n")
        for inv in jc["invariants"]:
            fh.write(f"- INVARIANTE: {inv}\n")
    print(f"[II-d-2] final/ congelado: P_F/P_M/P_I, P_inf=P_I, AVR, "
          f"L_N=λ{lnlo['L_N']['lambda_total']}, "
          f"L_O=λ{lnlo['L_O']['lambda_total']}")

    # ---- §33 figures ---------------------------------------------------------
    make_figures(out, fin, plots, w, by_lambda)


def make_figures(out: str, fin: str, plots: str, w: pd.DataFrame,
                 by_lambda: pd.DataFrame) -> None:
    """The 7 mandatory figures of §33 (the 2 pre-gate ones are re-issued with
    the FROZEN names)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (1) P_F/P_M/P_I × λ (final names)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(by_lambda["lambda_total"], by_lambda["P_F"], "o-",
            label="P_F (congelado)")
    ax.plot(by_lambda["lambda_total"], by_lambda["P_M"], "s-",
            label="P_M (congelado)")
    ax.plot(by_lambda["lambda_total"], by_lambda["P_I"], "^-",
            label="P_I = P_inf (congelado)")
    ax.set_xlabel("lambda total (UEs/min)")
    ax.set_ylabel("fração de janelas")
    ax.set_title("Oracle 3-way: P_F/P_M/P_I × carga (congelado pós-PASS)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "fig_pf_pm_pi_lambda_final.png"), dpi=150)
    plt.close(fig)

    # (2) |F_t| and |M_t| × load
    g = w.groupby("lam").agg(F=("num_feasible_actions", "median"),
                             M=("num_marginal_actions", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g["lam"], g["F"], "o-", label="|F_t| (mediana)")
    ax.plot(g["lam"], g["M"], "s-", label="|M_t| (mediana)")
    ax.set_xlabel("lambda total")
    ax.set_ylabel("ações no suporte")
    ax.set_title("Contagens de ação por janela × carga")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "fig_fset_counts_lambda_final.png"),
                dpi=150)
    plt.close(fig)

    # (3)+(4) near-threshold error gates (LOO tables from part 1)
    for name, fname, near_col, mae_col in (
            ("embb", "fig_embb_near_threshold.png", "near_mae", "mae"),
            ("urllc", "fig_urllc_near_threshold.png", "near_mae", "mae")):
        p = os.path.join(out, "tables",
                         f"oracle_3way_gates_loo_{name}.csv")
        if not os.path.isfile(p):
            continue
        t = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(t))
        ax.bar(x - 0.2, t[mae_col], 0.4, label="MAE geral")
        ax.bar(x + 0.2, t[near_col], 0.4, label="MAE near-threshold")
        ax2 = ax.twinx()
        ax2.plot(x, t["coverage_p25_p75"], "gD--", label="cobertura [p25,p75]")
        ax2.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax2.set_ylabel("cobertura")
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {s}" for s in t["held_seed"]])
        unit = "Mbps" if name == "embb" else "ms"
        ax.set_ylabel(f"erro ({unit})")
        ax.set_title(f"Gate {name} near-threshold (LOO, refit II-d)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(plots, fname), dpi=150)
        plt.close(fig)

    # (5) PDR mMTC × a_m (D6)
    d6c = pd.read_csv(os.path.join(out, "adjudication", "d6_pdr_vs_am.csv"))
    if len(d6c):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for (lam, win), gg in d6c.groupby(["lambda_total", "win"]):
            gg = gg.sort_values("a_m")
            ax.plot(gg["a_m"], gg["pdr_mmtc_1s_rolling"], "o-",
                    label=f"PDR 1 s rolling λ{lam:g} t{win}")
            ax.plot(gg["a_m"], gg["mmtc_pdr_100ms"], "x--", alpha=0.6,
                    label=f"PDR 100 ms λ{lam:g} t{win}")
        ax.axhline(0.95, color="red", lw=1, ls=":", label="SLA PDR 0.95")
        ax.axvline(3, color="green", lw=1, ls="-.",
                   label="a_m_min DECIDIDO = 3 (conservador)")
        ax.set_xlabel("a_m (RBGs)")
        ax.set_ylabel("PDR mMTC")
        ax.set_title("D6: PDR mMTC × a_m (microcalibração do piso, §13)")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(plots, "fig_pdr_mmtc_vs_am.png"), dpi=150)
        plt.close(fig)

    # (6) confusion matrix 3-way
    conf_p = os.path.join(out, "adjudication", "confusion_metrics_3way.json")
    if os.path.isfile(conf_p):
        conf = json.load(open(conf_p))
        m = conf["matrix"]
        tab = np.array([[m["FEASIBLE_all_ok"], m["FEASIBLE_violation"]],
                        [m["MARGINAL_all_ok"], m["MARGINAL_violation"]],
                        [m["INFEASIBLE_all_ok"], m["INFEASIBLE_violation"]]])
        fig, ax = plt.subplots(figsize=(5.6, 4))
        im = ax.imshow(tab, cmap="Blues")
        for r in range(3):
            for c in range(2):
                ax.text(c, r, str(tab[r, c]), ha="center", va="center",
                        color="white" if tab[r, c] > tab.max() / 2 else "black",
                        fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["replay all-OK", "replay violou"])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["FEASIBLE", "MARGINAL", "INFEASIBLE"])
        ax.set_title("Matriz de confusão 3-way (25 testemunhas válidas)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(plots, "fig_confusion_matrix_3way.png"),
                    dpi=150)
        plt.close(fig)

    # (7) state map N_embb × N_urllc
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = {"FEASIBLE": "tab:green", "MARGINAL": "tab:orange",
              "INFEASIBLE": "tab:red"}
    eq = w[w["arm"] == "equal"]
    rng = np.random.default_rng(0)
    for sc, c in colors.items():
        s = eq[eq["state_class"] == sc]
        if not len(s):
            continue
        jitter_x = s["n_embb"] + rng.uniform(-0.15, 0.15, len(s))
        jitter_y = s["n_urllc"] + rng.uniform(-0.15, 0.15, len(s))
        ax.scatter(jitter_x, jitter_y, s=4, alpha=0.25, c=c, label=sc)
    ax.set_xlabel("N_embb (UEs eMBB com demanda)")
    ax.set_ylabel("N_urllc (UEs URLLC)")
    ax.set_title("Mapa de estados do oracle 3-way (braço equal, N×N)")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "fig_state_map_nembb_nurllc.png"), dpi=150)
    plt.close(fig)

    print(f"[II-d-2] figuras §33 em {plots}")


if __name__ == "__main__":
    raise SystemExit(main())
