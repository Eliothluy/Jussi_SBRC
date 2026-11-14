"""d1_analysis.py — estatísticas do diagnóstico de projeção (protocolo v2 §7.2).

Lê os decisions.jsonl dos 9 runs B0 (SQR-HET) e produz:
  projection_summary.csv / projection_per_seed.csv / projection_events.parquet
  fig_projection_rate_by_lambda.png / fig_before_after_prediction.png /
  fig_outcomes_by_projection_and_load.png
  (+ números para D1_projection_diagnostic.md)

Semânticas (congeladas):
  joint_frozen_ok .... julgamento de janela R36–R38 (vacuous = satisfeito)
  joint_strict ...... ALL3_COMPLETE (qualquer NO_ACTIVE_DEMAND => UNKNOWN)
  projection_triggered = coordenador cortou (Σ propostas pós-margem > C)
Inferência por seed/run (janelas autocorrelacionadas não são amostras).
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "d1_runs")
OUT = HERE


def load() -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(f"{RUNS}/het_lam*_s*/decisions.jsonl")):
        tag = os.path.basename(os.path.dirname(p))
        lam = int(tag.split("_lam")[1].split("_")[0])
        seed = int(tag.split("_s")[1])
        for line in open(p):
            r = json.loads(line)
            r["lam"], r["seed"], r["tag"] = lam, seed, tag
            rows.append(r)
    df = pd.DataFrame(rows)
    df["valid"] = df.get("sla_outcome_margins").notna() if "sla_outcome_margins" in df else False
    if "sla_outcome_margins" in df:
        df["valid"] = df["sla_outcome_margins"].apply(lambda v: isinstance(v, list))
    df["proj"] = df.get("projection_triggered", False).astype(bool)
    df["sum_prop"] = df["a_after_uncertainty_margin"].apply(lambda v: sum(v) if isinstance(v, list) else np.nan)
    df["C"] = df.get("capacity_C")
    df["over_ratio"] = df["sum_prop"] / df["C"]
    df["delta_l1"] = df.get("projection_delta_l1", 0)
    # predicted min-margin before/after the projection (§7.2.3 — diagnóstico)
    def _min(col):
        return df[col].apply(lambda v: min(v) if isinstance(v, list) else np.nan)
    df["pred_min_before"] = _min("predicted_safe_before_proposal")
    df["pred_min_after"] = _min("predicted_safe_after_projection")
    df["pred_degraded"] = df["pred_min_after"] < df["pred_min_before"] - 1e-12
    # strict evidence/outcome
    if "sla_evidence" in df:
        ev = df["sla_evidence"].apply(lambda v: v if isinstance(v, list) else [])
        df["n_complete"] = ev.apply(lambda v: sum(1 for e in v if e == "COMPLETE"))
        df["joint_strict"] = np.where(
            df["n_complete"] < 3, "UNKNOWN",
            np.where(df["joint_frozen_ok"], "PASS", "FAIL"))
    # §3.4: fase bootstrap-PROP (decisões do prop R39) separada das
    # decisões do próprio B0 — §7.2 caracteriza a seleção/projeção do B0.
    df["bootstrap"] = (df.get("policy") == "SQR_HET_BOOTSTRAP_PROP")
    df["sent_eq_applied"] = df.apply(
        lambda r: r.get("a_sent_to_mac") == r.get("a_effectively_applied")
        if isinstance(r.get("a_sent_to_mac"), list) else np.nan, axis=1)
    return df


def main() -> int:
    df = load()
    val = df[df.valid].copy()
    n_bs = int(val.bootstrap.sum())
    val = val[~val.bootstrap].copy()   # §7.2: apenas decisões do próprio B0
    print(f"[d1] {len(df)} decisões totais | bootstrap-PROP com outcome "
          f"{n_bs} | B0 {len(df)-n_bs} | B0 com outcome {len(val)} "
          f"({100*len(val)/max(1,len(df)-n_bs):.1f}%)")
    print(f"[d1] total vs com-outcome: a ÚLTIMA decisão de cada run governa "
          f"janela após o fim do sim -> {len(df)-int(df.valid.sum())} linhas "
          f"'end_of_run' sem outcome (missing_reason; 1 por run)")
    # --- §7.2.1/2 ------------------------------------------------------
    print(f"[d1] P_projection = {100*val.proj.mean():.2f}% "
          f"(Σpropostas/C: mediana {val.over_ratio.median():.2f}, "
          f"p95 {val.over_ratio.quantile(.95):.2f}; "
          f"||Δ||₁ quando projeta: mediana "
          f"{val[val.proj].delta_l1.median():.0f})")
    per = val.groupby(["lam", "seed"]).agg(
        n=("proj", "size"), p_proj=("proj", "mean"),
        over_med=("over_ratio", "median"),
        viol_joint=("joint_frozen_ok", lambda s: 1 - s.mean()),
        n_strict_pass=("joint_strict", lambda s: (s == "PASS").mean()),
        n_strict_unknown=("joint_strict", lambda s: (s == "UNKNOWN").mean()),
    ).reset_index()
    per.to_csv(f"{OUT}/projection_per_seed.csv", index=False)
    print(per.to_string(index=False))
    # denominadores dos SLAs por λ (por slice: observadas vs R19-excluídas)
    den = []
    for lam in sorted(val.lam.unique()):
        s = val[val.lam == lam]
        row = {"lam": lam, "n_decisions_B0": len(s)}
        for si, name in ((0, "embb"), (1, "urllc"), (2, "mmtc")):
            m = s["sla_outcome_margins"].apply(
                lambda v: v[si] if isinstance(v, list) else None)
            row[f"{name}_observed"] = int(m.notna().sum())
            row[f"{name}_excluded_R19"] = int(m.isna().sum())
            row[f"{name}_violation_pct"] = (float(100 * (m.dropna() < 0).mean())
                                            if m.notna().any() else None)
        row["all3_complete"] = int((s["n_complete"] == 3).sum())
        den.append(row)
    den = pd.DataFrame(den)
    den.to_csv(f"{OUT}/sla_denominators_by_lambda.csv", index=False)
    print("[d1] denominadores dos SLAs por λ:\n", den.to_string(index=False))
    # --- §7.2.3 --------------------------------------------------------
    pr = val[val.proj]
    print(f"[d1] §7.2.3 margem prevista degradada pela correção: "
          f"{100*pr.pred_degraded.mean():.1f}% das decisões com projeção "
          f"(diagnóstico; sem atribuição causal)")
    # --- §7.2.4/5 matriz projeção × outcome -----------------------------
    for sem, col in (("frozen", "joint_frozen_ok"), ("strict", "joint_strict")):
        m = val.pivot_table(index="proj", columns=col,
                            values="tag", aggfunc="count", fill_value=0)
        print(f"[d1] matriz projeção×outcome ({sem}):\n{m}")
    # §7.2.6 estratos semelhantes (proxy: mesmos n_ues por slice)
    val["dem_key"] = val["slice_n_ues"].apply(lambda v: tuple(v) if isinstance(v, list) else ())
    strat = val.groupby(["dem_key", "proj"]).joint_frozen_ok.agg(["mean", "size"])
    strat.to_csv(f"{OUT}/projection_strata_demanda.csv")
    # divergência sent vs applied (§7.2.7)
    bad = val[val.sent_eq_applied == False]
    print(f"[d1] §7.2.7 sent≠applied: {len(bad)} decisões")
    try:
        val.to_parquet(f"{OUT}/projection_events.parquet", index=False)
    except Exception:
        val.to_pickle(f"{OUT}/projection_events.pkl")
        val.to_csv(f"{OUT}/projection_events.csv", index=False)

    # --- figuras --------------------------------------------------------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    # fig 1: taxa de projeção por λ (por seed)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for lam in sorted(val.lam.unique()):
        s = per[per.lam == lam]
        ax.scatter([lam]*len(s), 100*s.p_proj, color="#1565c0", s=28,
                   zorder=3, label="seeds" if lam == sorted(val.lam.unique())[0] else None)
    g = per.groupby("lam").p_proj
    ax.plot(sorted(val.lam.unique()), 100*g.mean().sort_index(), "o--",
            color="#c62828", label="média")
    ax.set_xlabel("carga λ (UE/min total)"); ax.set_ylabel("decisões com projeção (%)")
    ax.set_title("D1 §7.2.1 — P(projeção) por carga (SQR-HET, 3 seeds)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_projection_rate_by_lambda.png"); plt.close(fig)

    # fig 2: predições antes/depois (§7.2.3)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    sub = val[val.proj]
    ax.scatter(sub.pred_min_before, sub.pred_min_after, s=9, alpha=.35,
               color="#1565c0", label="projetadas")
    lims = [np.nanmin([sub.pred_min_before.min(), sub.pred_min_after.min()]),
            np.nanmax([sub.pred_min_before.max(), sub.pred_min_after.max()])]
    ax.plot(lims, lims, "k--", lw=1)
    ax.axhline(0, color="#c62828", lw=.8); ax.axvline(0, color="#c62828", lw=.8)
    ax.set_xlabel("mín_i f_τ(z_i, proposta) (margem prevista, pré-projeção)")
    ax.set_ylabel("mín_i f_τ(z_i, alocada) (pós-projeção)")
    ax.set_title(f"D1 §7.2.3 — efeito previsto da projeção nas margens "
                 f"({100*sub.pred_degraded.mean():.0f}% degradadas)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_before_after_prediction.png"); plt.close(fig)

    # fig 3: outcome por projeção × carga (frozen + strict)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    for ax, col, name in ((axes[0], "joint_frozen_ok", "frozen (vacuous=OK)"),
                          (axes[1], "joint_strict", "strict ALL3")):
        for i, (pr_, color) in enumerate(((False, "#90a4ae"), (True, "#c62828"))):
            bots, labs = [], []
            for lam in sorted(val.lam.unique()):
                s = val[(val.lam == lam) & (val.proj == pr_)]
                if col == "joint_frozen_ok":
                    okr = 100 * s.joint_frozen_ok.mean()
                else:
                    okr = 100 * (s.joint_strict == "PASS").mean()
                bars = [okr, 100 - okr]
                ax.bar(lam, bars[0], .5, color=color if i else "#2e7d32")
                ax.bar(lam, bars[1], .5, bottom=bars[0],
                       color="#eceff1", edgecolor="#90a4ae", hatch="//")
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("λ")
        axes[0].set_ylabel("outcome da janela (%)")
    axes[-1].text(.98, .04, "verde=all-OK · hachurado=viola/UNKNOWN",
                  transform=axes[-1].transAxes, ha="right", fontsize=7, color="#616161")
    fig.suptitle("D1 §7.2.5 — outcome × projeção × carga (vermelho=projetada)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(f"{OUT}/fig_outcomes_by_projection_and_load.png"); plt.close(fig)

    summary = {"n_decisions": int(len(df)), "n_valid": int(len(val)),
               "P_projection_pct": float(100 * val.proj.mean()),
               "per_seed": per.to_dict("records")}
    pd.DataFrame([summary]).to_csv(f"{OUT}/projection_summary.csv", index=False)
    print(f"[d1] artefatos em {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
