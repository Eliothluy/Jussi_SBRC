#!/usr/bin/env python
"""Driver da Etapa 2.4-II-d-v2 parte b (FINAL) — refit eMBB + gate v2.

Ordem obrigatória §18 (a partir do passo 2 — os 13 replays já foram
executados pelo orquestrador em
``results/sqr_characterization_round2/oracle/oracle_recalibrated_v2/
support_extension/replays_v2/``):

2. validação de integridade (§4 do II-c + banner-λ, lição R56);
3. anexar dados — ``support_extension_replays.csv`` (colunas EXATAS §10,
   extração t+1 congelada); curva executável eMBB (§11);
4. refit eMBB (§12–§16) + gates de erro antes×depois (LOO);
5. reprocessar as 89.970 janelas (§18.6) com o eMBB v2 (URLLC/mMTC
   congelados, guard de determinismo vs a tabela II-d);
6. GATE v2 (§19–§20): R1–R4 + confusion 3-way (27 testemunhas) + C1–C7;
7. PASS ⇒ congelamento §23–§31 + ``ln_lo_selection_v2.md``;
   STOP ⇒ NADA congelado; artefatos rotulados CANDIDATE/PROVISÓRIO;
   caracterização ENCERRA reportando MARGINAL explicitamente (§22).

MEMÓRIA: o kernel OOM-killed este pipeline num processo único (12,9 GB
anon-rss em 24/10 20:45 — a retenção de arena dos LPs HiGHS-IPM acumula
entre os ~18 solves + 2 classificações de 89.970 janelas).  Cada fase
pesada roda num SUBPROCESSO próprio (``--phase``; o SO devolve toda a
memória entre fases) com artefatos em disco entre elas — nenhum resultado
depende disso (as fases são determinísticas e os modelos são serializados
bit-a-bit).

Artefatos §32 em results/sqr_characterization_round2/oracle/
oracle_recalibrated_v2/.  NENHUMA simulação ns-3 é lançada aqui.

Uso (a partir de MODA/ns-o-ran-gym)::

    .venv/bin/python examples/sqr_oracle_adjudicate_d_v2.py            # tudo
    .venv/bin/python examples/sqr_oracle_adjudicate_d_v2.py --phase prep
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

_GYM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_GYM_ROOT, "src"))

from nsoran.sqr.oracle import adjudication_3way as a3  # noqa: E402
from nsoran.sqr.oracle import quantile as qm           # noqa: E402
from nsoran.sqr.oracle import recal_v2 as rv           # noqa: E402
from nsoran.sqr.oracle import three_way as tw          # noqa: E402
from nsoran.sqr.oracle.adjudication import WINDOW_MS   # noqa: E402

R2 = os.path.join(_GYM_ROOT, "results", "sqr_characterization_round2")
ORACLE = os.path.join(R2, "oracle")
V2 = os.path.join(ORACLE, "oracle_recalibrated_v2")
SUPPORT = os.path.join(V2, "support_extension")
MODELS = os.path.join(V2, "models")
PHASES = ("prep", "guard", "fit_v2", "loo", "reclass", "gate")


# ---------------------------------------------------------------------------
# model serialization (bit-exact; avoids LP refits in later phases)
# ---------------------------------------------------------------------------

def save_pinball(path: str, model: qm.PinballQuantileModel,
                 train_frame: pd.DataFrame | None = None) -> None:
    arr: dict = {"quantiles": np.array(sorted(model.quantiles))}
    for q in model.quantiles:
        arr[f"beta_{q}"] = model.betas[q]
    arr["mu"], arr["sd"], arr["keep"] = model.mu, model.sd, model._keep
    np.savez(path, **arr)
    if train_frame is not None:
        pd.to_pickle(train_frame, path + ".trainframe.pkl")


def load_pinball(path: str, cls_type: str = "embb_v2"):
    z = np.load(path)
    m = rv.EmmbQuantileV2() if cls_type == "embb_v2" else \
        (qm.EmmbQuantileRecal() if cls_type == "embb" else
         qm.UrllcQuantileRecal())
    m.model.betas = {float(q): z[f"beta_{q}"] for q in z["quantiles"]}
    m.model.mu, m.model.sd, m.model._keep = z["mu"], z["sd"], z["keep"]
    if cls_type == "embb_v2":
        m.train_frame = pd.read_pickle(path + ".trainframe.pkl")
    return m


def _plan_and_replays():
    plan = rv.load_plan_v2(SUPPORT)
    replays = pd.read_csv(os.path.join(SUPPORT,
                                       "support_extension_replays.csv"))
    return plan, replays


# ---------------------------------------------------------------------------
# phase: prep — §18.2 validation + §10 + §11 (no LP)
# ---------------------------------------------------------------------------

def phase_prep() -> None:
    for sub in ("support_extension", "models", "predictions", "adjudication",
                "plots", "tables", "reports"):
        os.makedirs(os.path.join(V2, sub), exist_ok=True)
    print("[II-d-v2] carregando dataset congelado e plano ...")
    d = rv.load_windows(ORACLE)
    plan = rv.load_plan_v2(SUPPORT)

    print("[II-d-v2] §18.2 validação dos 13 replays (§4 + banner-λ R56) ...")
    plan, validations = rv.validate_replays_v2(SUPPORT, R2)
    vdir = os.path.join(SUPPORT, "replay_validations_v2")
    os.makedirs(vdir, exist_ok=True)
    for v in validations:
        with open(os.path.join(vdir, f"{v['replay_id']}.json"), "w") as fh:
            json.dump(v, fh, indent=2, ensure_ascii=False, default=str)
    n_valid = sum(bool(v["replay_valid"]) for v in validations)
    print(f"[II-d-v2] §4: {n_valid}/{len(validations)} replays válidos")
    for v in validations:
        if not v["replay_valid"]:
            print(f"[II-d-v2]   INVÁLIDO {v['replay_id']}: "
                  f"{v['checks'].get('failure')}")

    print("[II-d-v2] §10 support_extension_replays.csv ...")
    replays = rv.build_support_extension_rows(SUPPORT, ORACLE, R2, plan,
                                              validations)
    replays.to_csv(os.path.join(SUPPORT, "support_extension_replays.csv"),
                   index=False)

    print("[II-d-v2] §11 curva executável eMBB ...")
    pts, curve, disp = rv.build_support_curve(ORACLE, replays, plan, d)
    pts.to_csv(os.path.join(SUPPORT, "embb_support_curve_points.csv"),
               index=False)
    curve.to_csv(os.path.join(SUPPORT, "embb_support_curve.csv"), index=False)
    disp.to_csv(os.path.join(SUPPORT, "embb_support_curve_dispersion.csv"),
                index=False)
    with open(os.path.join(V2, "models", "prep_summary.json"), "w") as fh:
        json.dump({"n_replays_valid": n_valid,
                   "n_replays_total": int(len(replays))}, fh)


# ---------------------------------------------------------------------------
# phase: guard — frozen II-d path reproduces the saved table (URLLC/mMTC
# frozen); models serialized for later phases
# ---------------------------------------------------------------------------

def phase_guard() -> None:
    d = rv.load_windows(ORACLE)
    print("[II-d-v2] guard: fit URLLC congelado + eMBB II-d ...")
    urllc = qm.UrllcQuantileRecal().fit(d)
    embb_ii_d = qm.EmmbQuantileRecal().fit(d)
    save_pinball(os.path.join(MODELS, "urllc_frozen.npz"), urllc.model)
    save_pinball(os.path.join(MODELS, "embb_iid.npz"), embb_ii_d.model)
    print("[II-d-v2] guard: classificando 89.970 janelas (II-d path) ...")
    cls_guard = tw.classify_windows_3way(d, embb_ii_d, urllc, am_min=3)
    saved = pd.read_csv(os.path.join(ORACLE, "oracle_recalibrated",
                                     "predictions",
                                     "oracle_3way_windows.csv"))
    key_cols = ["arm", "lam", "seed", "win", "state_class",
                "num_feasible_actions", "num_marginal_actions",
                "num_infeasible_actions", "best_feasible_action",
                "best_marginal_action", "best_degradation_action",
                "D_min_support_p25_pessimistic", "D_min_support_p50",
                "D_min_support_p75_optimistic"]
    a = cls_guard[key_cols].sort_values(["arm", "lam", "seed", "win"]
                                        ).reset_index(drop=True)
    b = saved[key_cols].sort_values(["arm", "lam", "seed", "win"]
                                    ).reset_index(drop=True)
    for c in ("best_feasible_action", "best_marginal_action",
              "best_degradation_action"):
        a[c] = a[c].fillna("")
        b[c] = b[c].fillna("")
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    with open(os.path.join(MODELS, "guard_ok.json"), "w") as fh:
        json.dump({"ok": True,
                   "note": ("refit II-d reproduz a tabela congelada "
                            "bit-a-bit (URLLC/mMTC inalterados; qualquer "
                            "diferença da v2 vem SOMENTE do refit eMBB)")},
                  fh, ensure_ascii=False)
    print("[II-d-v2] guard OK")


# ---------------------------------------------------------------------------
# phase: fit_v2 — the §12 refit (3 LPs, own process)
# ---------------------------------------------------------------------------

def phase_fit_v2() -> None:
    d = rv.load_windows(ORACLE)
    plan, replays = _plan_and_replays()
    e1_rows = rv.e1_training_rows(d, replays, plan)
    print(f"[II-d-v2] §12 refit eMBB v2 ({len(e1_rows)} linhas E1 "
          "in-suporte) ...")
    embb_v2 = rv.EmmbQuantileV2().fit(d, e1_rows)
    save_pinball(os.path.join(MODELS, "embb_v2.npz"), embb_v2.model,
                 train_frame=embb_v2.train_frame)
    betas_rows = []
    for qq, bb in embb_v2.model.betas.items():
        for k, (f, coef) in enumerate(zip(qm.EMMB_FEATURES, bb)):
            betas_rows.append({"model": "embb_v2", "quantile": qq,
                               "feature": f, "beta_std_space": coef,
                               "sd_train": float(embb_v2.model.sd[k])})
    pd.DataFrame(betas_rows).to_csv(
        os.path.join(MODELS, "quantile_betas_v2.csv"), index=False)
    print("[II-d-v2] §12 refit concluído "
          f"({len(embb_v2.train_frame)} linhas de treino)")


# ---------------------------------------------------------------------------
# phase: loo — 3 folds, ONE SUBPROCESS PER FOLD (memory isolation)
# ---------------------------------------------------------------------------

def phase_loo() -> None:
    folds = []
    for held in (1, 2, 3):
        out = os.path.join(MODELS, f"embb_loo_fold_{held}.csv")
        subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--phase", "loo_fold", "--held", str(held),
                        "--out", out], check=True)
        folds.append(pd.read_csv(out))
    loo_v2 = pd.concat(folds, ignore_index=True)
    loo_v2.to_csv(os.path.join(MODELS, "embb_loo_gates_v2.csv"), index=False)
    loo_before = pd.read_csv(os.path.join(ORACLE, "oracle_recalibrated",
                                          "tables",
                                          "oracle_3way_gates_loo_embb.csv"))
    cols = ["held_seed", "mae", "near_mae", "coverage_p25_p75"]
    cmp = loo_before[cols].merge(loo_v2[cols], on="held_seed",
                                 suffixes=("_antes_II_d", "_depois_v2"))
    cmp.to_csv(os.path.join(MODELS, "embb_loo_antes_depois_v2.csv"),
               index=False)
    print(cmp.to_string(index=False))


def phase_loo_fold(held: int, out: str) -> None:
    d = rv.load_windows(ORACLE)
    plan, replays = _plan_and_replays()
    e1_rows = rv.e1_training_rows(d, replays, plan)
    gates = rv.loo_gates_v2(d, e1_rows, held_seeds=(held,))
    gates.to_csv(out, index=False)
    print(f"[II-d-v2] LOO fold seed={held} concluído")


# ---------------------------------------------------------------------------
# phase: reclass — §18.6 reprocessing with the saved v2 model (no LP)
# ---------------------------------------------------------------------------

def phase_reclass() -> None:
    d = rv.load_windows(ORACLE)
    plan, replays = _plan_and_replays()
    embb_v2 = load_pinball(os.path.join(MODELS, "embb_v2.npz"), "embb_v2")
    urllc = load_pinball(os.path.join(MODELS, "urllc_frozen.npz"), "urllc")
    embb_ii_d = load_pinball(os.path.join(MODELS, "embb_iid.npz"), "embb")
    print("[II-d-v2] §18.6 reprocessamento 3-way v2 (89.970 janelas) ...")
    cls_v2 = rv.classify_windows_3way_v2(d, embb_v2, urllc, am_min=3)
    win_csv = os.path.join(V2, "tables", "oracle_3way_windows_v2.csv")
    cls_v2.to_csv(win_csv, index=False)
    os.makedirs(os.path.join(V2, "predictions"), exist_ok=True)
    cls_v2.to_csv(os.path.join(V2, "predictions",
                               "oracle_3way_windows_v2.csv"), index=False)
    by_lam = rv.aggregate_3way_by_lambda_v2(cls_v2)
    by_lam.to_csv(os.path.join(V2, "tables", "oracle_3way_by_lambda_v2.csv"),
                  index=False)
    print(by_lam[["lambda_total", "P_F_candidate", "P_M_candidate",
                  "P_I_candidate"]].to_string(index=False))
    # band movement before × after at the 13 replay actions (predict-only)
    move_rows = []
    for _, pr in plan.iterrows():
        t = int(pr["win_t"])
        hit = d[(d["arm"] == "equal") & (d["lam"] == float(pr["lam"]))
                & (d["seed"] == int(pr["seed"]))
                & (d["time_ms"] == (t + 1) * WINDOW_MS)]
        if not len(hit):
            continue
        s = hit.iloc[0]
        args = (np.array([float(s["n_embb"])]),
                np.array([float(s["cqi_eff"])]),
                np.array([float(s["queue_embb"])]),
                np.array([float(s["n_embb_backlog"])]),
                np.array([float(s["offered_load_embb"])]),
                np.array([float(s["prev_thr_embb_med"])]),
                np.array([float(pr["action_e"])]))
        q0 = embb_ii_d.predict_grid_bands(*args)
        q1 = embb_v2.predict_grid_bands(*args)
        move_rows.append({
            "replay_id": f"{pr['grupo']}_lam{pr['lam']}s{pr['seed']}"
                         f"t{t}_a{pr['action_e']}-{pr['action_u']}"
                         f"-{pr['action_m']}",
            "a_e": int(pr["action_e"]),
            "p25_antes": float(q0[0.25][0, 0]),
            "p25_depois": float(q1[0.25][0, 0]),
            "p50_antes": float(q0[0.5][0, 0]),
            "p50_depois": float(q1[0.5][0, 0]),
            "p75_antes": float(q0[0.75][0, 0]),
            "p75_depois": float(q1[0.75][0, 0]),
            "max_abs_delta": float(max(
                abs(float(q1[k][0, 0]) - float(q0[k][0, 0]))
                for k in (0.25, 0.5, 0.75))),
        })
    move = pd.DataFrame(move_rows)
    move.to_csv(os.path.join(MODELS, "embb_band_movement_witness_v2.csv"),
                index=False)
    print(f"[II-d-v2]   movimento máximo de banda nas 13 ações: "
          f"{move['max_abs_delta'].max():.6f} Mbps")


# ---------------------------------------------------------------------------
# phase: gate — replay results v2 + confusion + gate + figures + freeze
# ---------------------------------------------------------------------------

def phase_gate() -> None:
    d = rv.load_windows(ORACLE)
    plan, replays = _plan_and_replays()
    embb_v2 = load_pinball(os.path.join(MODELS, "embb_v2.npz"), "embb_v2")
    urllc = load_pinball(os.path.join(MODELS, "urllc_frozen.npz"), "urllc")
    cls_v2 = pd.read_csv(os.path.join(V2, "tables",
                                      "oracle_3way_windows_v2.csv"))
    by_lam = pd.read_csv(os.path.join(V2, "tables",
                                      "oracle_3way_by_lambda_v2.csv"))
    e1_rows = rv.e1_training_rows(d, replays, plan)

    res_v2 = rv.build_replay_results_v2(plan, _validations(plan), cls_v2, d,
                                        embb_v2, urllc)
    res_v2.to_csv(os.path.join(V2, "adjudication",
                               "oracle_3way_replay_results_v2.csv"),
                  index=False)

    print("[II-d-v2] §19–§20 gate v2 ...")
    ver = pd.read_csv(os.path.join(ORACLE, "oracle_recalibrated",
                                   "adjudication",
                                   "oracle_3way_replay_results_verified.csv"))
    confusion = rv.confusion_matrix_3way_v2(cls_v2, ver, replays)
    wit = confusion.pop("witnesses")
    wit.to_csv(os.path.join(V2, "adjudication",
                            "confusion_witnesses_v2.csv"), index=False)
    rows = [{"predicted": k.split("_")[0],
             "observed": "_".join(k.split("_")[1:]), "n": v}
            for k, v in confusion["matrix"].items()]
    pd.DataFrame(rows).to_csv(os.path.join(V2, "adjudication",
                                           "confusion_matrix_3way_v2.csv"),
                              index=False)
    with open(os.path.join(V2, "adjudication",
                           "confusion_metrics_3way_v2.json"), "w") as fh:
        json.dump(confusion, fh, indent=2, ensure_ascii=False, default=str)
    confusion["witnesses"] = wit  # pop() acima removeu; gate_v2 precisa
    print(f"[II-d-v2] confusion 27 testemunhas: {confusion['matrix']} | "
          f"FIR={confusion['false_infeasible_rate']} "
          f"FFR={confusion['false_feasible_rate']}")

    gate = rv.gate_v2(cls_v2, replays, ver, confusion, embb_v2, d, e1_rows)
    with open(os.path.join(V2, "adjudication",
                           "gate_verdict_II_d_v2.json"), "w") as fh:
        json.dump(gate, fh, indent=2, ensure_ascii=False, default=str)
    for c in gate["checks"]:
        print(f"[II-d-v2] gate {c['id']}: "
              f"{'PASS' if c['passed'] else 'FAIL'} — {c['note']}")
    print(f"[II-d-v2] GATE v2 = {gate['verdict']}")

    make_figures(ver, confusion, gate, cls_v2, by_lam)

    lnlo_path = os.path.join(V2, "reports", "ln_lo_selection_v2.md")
    if gate["verdict"] == "PASS":
        by_lambda, by_slice, avr, boundary, lnlo = rv.freeze_all(cls_v2)
        by_lambda.to_csv(os.path.join(V2, "tables",
                                      "pf_pm_pi_freeze_v2.csv"), index=False)
        by_slice.to_csv(os.path.join(V2, "tables",
                                     "avr_by_slice_freeze_v2.csv"),
                        index=False)
        avr.to_csv(os.path.join(V2, "tables", "avr_freeze_v2.csv"),
                   index=False)
        boundary.to_csv(os.path.join(V2, "tables",
                                     "feasibility_boundary_v2.csv"),
                        index=False)
        with open(os.path.join(V2, "reports", "final_freeze_v2.json"),
                  "w") as fh:
            json.dump(lnlo, fh, indent=2, ensure_ascii=False, default=str)
        write_ln_lo(lnlo_path, lnlo, frozen=True)
        print(f"[II-d-v2] §23–§31 CONGELADO: "
              f"L_N=λ{lnlo['L_N']['lambda_total']} "
              f"L_O=λ{lnlo['L_O']['lambda_total']}")
    else:
        cand = a3.freeze_by_lambda(cls_v2)
        cand.to_csv(os.path.join(V2, "tables",
                                 "pf_pm_pi_candidate_v2.csv"), index=False)
        write_ln_lo(lnlo_path, None, frozen=False)
        print("[II-d-v2] GATE STOP — NADA congelado (§23); P_F/P_M/P_I "
              "gravados como CANDIDATE")

    move = pd.read_csv(os.path.join(MODELS,
                                    "embb_band_movement_witness_v2.csv"))
    with open(os.path.join(V2, "reports", "gate_numbers_v2.json"), "w") as fh:
        json.dump({
            "verdict": gate["verdict"],
            "confusion": confusion,
            "checks": [{k: c[k] for k in ("id", "passed", "threshold",
                                          "note")} for c in gate["checks"]],
            "by_lambda_candidate": by_lam.to_dict(orient="records"),
            "n_replays_valid": int(replays["replay_valid"].sum()),
            "band_max_abs_delta_witness": float(move["max_abs_delta"].max()),
        }, fh, indent=2, ensure_ascii=False, default=str)


def _validations(plan: pd.DataFrame) -> list[dict]:
    """Reload the §4 validation JSONs written by phase prep."""
    out = []
    for _, pr in plan.iterrows():
        tag = (f"{pr['grupo']}_{pr['arm_base']}_lam{pr['lam']}_s{pr['seed']}"
               f"_t{pr['win_t']}_a{pr['action_e']}-{pr['action_u']}"
               f"-{pr['action_m']}")
        with open(os.path.join(SUPPORT, "replay_validations_v2",
                               f"{tag}.json")) as fh:
            out.append(json.load(fh))
    return out


def write_ln_lo(path: str, lnlo: dict | None, frozen: bool) -> None:
    with open(path, "w") as fh:
        if frozen:
            fh.write("# Seleção de L_N e L_O (§28/§29) + decisão SQR-JC "
                     "(§30–§31) — v2\n\nCONGELADO após o PASS do gate v2 "
                     "(§23). Números de `tables/pf_pm_pi_freeze_v2.csv`.\n\n")
            fh.write(f"## L_N = λ{lnlo['L_N']['lambda_total']}\n\n- "
                     f"P_F={lnlo['L_N']['P_F']:.4f} "
                     f"P_M={lnlo['L_N']['P_M']:.4f} "
                     f"P_I={lnlo['L_N']['P_I']:.4f} "
                     f"(regime {lnlo['L_N']['regime']})\n")
            fh.write(f"## L_O = λ{lnlo['L_O']['lambda_total']}\n\n- "
                     f"P_F={lnlo['L_O']['P_F']:.4f} "
                     f"P_M={lnlo['L_O']['P_M']:.4f} "
                     f"P_I={lnlo['L_O']['P_I']:.4f} "
                     f"(regime {lnlo['L_O']['regime']})\n")
            fh.write("\n## Decisão SQR-JC\n\n")
            for c in lnlo["sqr_jc_decision"]["components"]:
                fh.write(f"- {c}\n")
        else:
            fh.write("# Seleção de L_N e L_O — v2\n\n"
                     "**NÃO CONGELADO — gate v2 = STOP (§23 bloqueia o "
                     "congelamento; §22: caracterização encerra reportando "
                     "MARGINAL explicitamente; nova rodada só por decisão "
                     "do orientador).**\n\n"
                     "Este arquivo existe para satisfazer a lista mínima "
                     "de artefatos (§32) SEM congelar nada: a escolha de "
                     "L_N/L_O e a decisão SQR-JC permanecem em aberto.\n")


def make_figures(ver, confusion, gate, cls_v2, by_lam) -> None:
    """§33 mandatory figures (labels CANDIDATE when the gate is STOP)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = pd.read_csv(os.path.join(SUPPORT, "embb_support_curve_points.csv"))
    disp = pd.read_csv(os.path.join(SUPPORT,
                                    "embb_support_curve_dispersion.csv"))
    loo_before = pd.read_csv(os.path.join(ORACLE, "oracle_recalibrated",
                                          "tables",
                                          "oracle_3way_gates_loo_embb.csv"))
    loo_v2 = pd.read_csv(os.path.join(MODELS, "embb_loo_gates_v2.csv"))
    plots = os.path.join(V2, "plots")
    stop = gate["verdict"] == "STOP"
    tag = "NÃO CONGELADO — gate v2 = STOP" if stop else "congelado pós-PASS"

    # (1) thr × a_e (support curve) + SLA line
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    mk = {"v2_replay": ("o", 60), "ii_c_g2_deltas": ("^", 50),
          "ii_d_replay_verified": ("s", 50), "original_equal": ("x", 40)}
    for src, (m, s) in mk.items():
        g = pts[pts["source"] == src]
        if len(g):
            ax.scatter(g["a_e"], g["thr_embb"], marker=m, s=s,
                       alpha=0.75, label=src)
    ax.axhline(3.5, color="red", lw=1, ls=":", label="SLA eMBB 3,5 Mbps")
    ax.axvline(37, color="gray", lw=1, ls="-."),
    ax.text(37.3, 1.0, "borda do suporte\n(pisos R58)", fontsize=7)
    ax.set_xlabel("a_e (RBGs)"); ax.set_ylabel("thr eMBB (Mbps, mediana UE)")
    ax.set_title("Curva executável eMBB × a_e (13 replays v2 + referências "
                 "II-c/II-d)\na_e>37 = diagnóstico fora-de-piso (cauda "
                 "(5,0), R59)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "fig_embb_support_curve_v2.png"),
                dpi=150)
    plt.close(fig)

    # (1b) dispersion by a_e
    fig, ax = plt.subplots(figsize=(6.4, 4))
    mid = (disp["thr_min_at_ae"] + disp["thr_max_at_ae"]) / 2
    err = (disp["thr_max_at_ae"] - disp["thr_min_at_ae"]) / 2
    ax.errorbar(disp["a_e"], mid, yerr=err, fmt="o-", capsize=4)
    ax.axhline(3.5, color="red", lw=1, ls=":")
    ax.axvline(37, color="gray", lw=1, ls="-.") 
    ax.set_xlabel("a_e (RBGs)"); ax.set_ylabel("thr eMBB (Mbps) min–max")
    ax.set_title("Dispersão entre janelas por nível de a_e (§11)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "fig_embb_dispersion_v2.png"), dpi=150)
    plt.close(fig)

    # (3) near-threshold error before × after (LOO)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(len(loo_v2))
    ax.bar(x - 0.2, loo_before["mae"], 0.4, label="MAE geral (antes II-d)")
    ax.bar(x + 0.2, loo_v2["mae"], 0.4, label="MAE geral (depois v2)")
    ax.bar(x - 0.2, loo_before["near_mae"], 0.4,
           label="MAE near (antes II-d)", hatch="//")
    ax.bar(x + 0.2, loo_v2["near_mae"], 0.4,
           label="MAE near (depois v2)", hatch="//")
    ax2 = ax.twinx()
    ax2.plot(x, loo_before["coverage_p25_p75"], "gD--",
             label="cobertura antes")
    ax2.plot(x, loo_v2["coverage_p25_p75"], "gD-", label="cobertura depois")
    ax2.axhline(0.5, color="gray", lw=0.8, ls=":"); ax2.set_ylim(0, 1)
    ax2.set_ylabel("cobertura [p25,p75]")
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s
                                          in loo_v2["held_seed"]])
    ax.set_ylabel("erro (Mbps)")
    ax.set_title("Gate eMBB near-threshold: ANTES (II-d) × DEPOIS (v2)\n"
                 "Δ máx de banda nas 13 ações de replay: 0,0053 Mbps")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "fig_embb_near_threshold_v2.png"),
                dpi=150)
    plt.close(fig)

    # (4) confusion 3-way v2
    m = confusion["matrix"]
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
    ax.set_xticks([0, 1]); ax.set_xticklabels(["replay all-OK",
                                               "replay violou"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["FEASIBLE", "MARGINAL", "INFEASIBLE"])
    ax.set_title(f"Matriz de confusão 3-way v2 (27 testemunhas: 22 "
                 f"verificadas + 5 E1)\ngate v2 = {gate['verdict']} "
                 f"({tag})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "fig_confusion_matrix_3way_v2.png"),
                dpi=150)
    plt.close(fig)

    # (5) P_F/P_M/P_I × λ (candidate names on STOP)
    cols = ("P_F", "P_M", "P_I") if not stop else \
        ("P_F_candidate", "P_M_candidate", "P_I_candidate")
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(by_lam["lambda_total"], by_lam[cols[0]], "o-", label=cols[0])
    ax.plot(by_lam["lambda_total"], by_lam[cols[1]], "s-", label=cols[1])
    ax.plot(by_lam["lambda_total"], by_lam[cols[2]], "^-", label=cols[2])
    ax.set_xlabel("lambda total (UEs/min)")
    ax.set_ylabel(f"fração de janelas ({tag})")
    ax.set_title("Oracle 3-way v2: P_F/P_M/P_I × carga")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "fig_pf_pm_pi_lambda_v2.png"), dpi=150)
    plt.close(fig)

    # (6) |F_t|/|M_t| × λ
    g = cls_v2.groupby("lam").agg(F=("num_feasible_actions", "median"),
                                  M=("num_marginal_actions", "median")
                                  ).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g["lam"], g["F"], "o-", label="|F_t| (mediana, suporte)")
    ax.plot(g["lam"], g["M"], "s-", label="|M_t| (mediana, suporte)")
    ax.set_xlabel("lambda total"); ax.set_ylabel("ações no suporte")
    ax.set_title(f"Contagens de ação por janela × carga (v2; {tag})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "fig_fset_counts_lambda_v2.png"),
                dpi=150)
    plt.close(fig)

    # (7) state map
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = {"FEASIBLE": "tab:green", "MARGINAL": "tab:orange",
              "INFEASIBLE": "tab:red"}
    eq = cls_v2[cls_v2["arm"] == "equal"]
    rng = np.random.default_rng(0)
    for sc, c in colors.items():
        s = eq[eq["state_class"] == sc]
        if not len(s):
            continue
        ax.scatter(s["n_embb"] + rng.uniform(-0.15, 0.15, len(s)),
                   s["n_urllc"] + rng.uniform(-0.15, 0.15, len(s)),
                   s=4, alpha=0.25, c=c, label=sc)
    ax.set_xlabel("N_embb"); ax.set_ylabel("N_urllc")
    ax.set_title(f"Mapa de estados do oracle 3-way v2 (equal; {tag})")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "fig_state_map_v2.png"), dpi=150)
    plt.close(fig)
    print(f"[II-d-v2] figuras §33 em {plots}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=PHASES + ("loo_fold", "all"),
                    default="all")
    ap.add_argument("--held", type=int, default=1)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args(argv)

    if args.phase == "prep":
        phase_prep()
    elif args.phase == "guard":
        phase_guard()
    elif args.phase == "fit_v2":
        phase_fit_v2()
    elif args.phase == "loo":
        phase_loo()
    elif args.phase == "loo_fold":
        phase_loo_fold(args.held, args.out)
    elif args.phase == "reclass":
        phase_reclass()
    elif args.phase == "gate":
        phase_gate()
    else:  # all — each heavy phase in its own process (OOM isolation)
        for ph in ("prep", "guard", "fit_v2", "loo", "reclass", "gate"):
            print(f"[II-d-v2] ===== fase {ph} =====", flush=True)
            subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--phase", ph], check=True)
        print("[II-d-v2] TODAS as fases concluídas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
