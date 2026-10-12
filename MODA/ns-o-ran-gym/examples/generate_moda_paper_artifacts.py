#!/usr/bin/env python3
r"""Generate IEEE-ready figures and LaTeX tables for the MODA offline-DRL line.

MODA = Metaheuristic-Offline Distilled Allocation for RAN slicing.
We distill the offline experience produced by the metaheuristic campaign
(GA/PSO/SA/hybrid ns-3 evaluations of the slice PRB-weight simplex) into fast
neural slice-weight policies deployable as an O-RAN xApp:

  MODA-Q    -> offline DDQN     : Q-value regression over the discretized simplex
  MODA-BC   -> offline SAC      : behavioral cloning of top-quantile candidates
  MODA-RWR  -> offline PPO      : reward-weighted regression (RWR)

Honest naming: the logged data has no next-state, so none of these use
TD-learning or policy-gradient rollouts; the ddqn/sac/ppo tags are kept only
as training-recipe identifiers. Reference methods: Best metaheuristic
candidate (re-run in the current environment), RSLAQ fixed [0.33,0.40,0.27]
and Equal [1/3,1/3,1/3].

Reads the closed-loop evaluation CSVs (all runs executed in the SAME
environment, hence mutually comparable) and writes everything into
paper_moda_offline_drl/{figures,tables,data}.

Usage:
    cd ns-o-ran-gym
    python3 examples/generate_moda_paper_artifacts.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
OUT_ROOT = REPO_ROOT / "paper_moda_offline_drl"
FIG_DIR = OUT_ROOT / "figures"
TAB_DIR = OUT_ROOT / "tables"
DATA_DIR = OUT_ROOT / "data"
for d in (FIG_DIR, TAB_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Naming: map raw evaluation labels -> paper labels
# ---------------------------------------------------------------------------
PAPER_NAME = {
    "DDQN-offline": "MODA-Q",
    "SAC-offline": "MODA-BC",
    "PPO-offline": "MODA-RWR",
    "Best meta (rerun)": "Best meta.",
    "RSLAQ fixed [0.33,0.40,0.27]": "RSLAQ fixed",
    "Equal [0.33,0.33,0.33]": "Equal",
}
# radio-efficiency CSV uses slightly different labels
RADIO_NAME = {
    "DDQN-offline": "MODA-Q",
    "SAC-offline": "MODA-BC",
    "PPO-offline": "MODA-RWR",
    "Best meta (rerun)": "Best meta.",
    "RSLAQ-fixed": "RSLAQ fixed",
    "Equal": "Equal",
}
# Fixed display order (proposals first, then references)
ORDER = ["MODA-Q", "MODA-BC", "MODA-RWR", "Best meta.", "RSLAQ fixed", "Equal"]
PROPOSALS = {"MODA-Q", "MODA-BC", "MODA-RWR"}

SCENARIOS = ["low_traffic", "normal", "congestion", "stressed"]
SCEN_LABEL = {"low_traffic": "Low", "normal": "Normal",
              "congestion": "Congestion", "stressed": "Stressed"}
SLICES = ["eMBB", "URLLC", "MTC"]

# Colorblind-safe (Wong) palette; proposals in blues/greens, references greys/orange
COLOR = {
    "MODA-Q":      "#0072B2",   # blue
    "MODA-BC":     "#009E73",   # green
    "MODA-RWR":    "#56B4E9",   # light blue
    "Best meta.":  "#D55E00",   # vermillion
    "RSLAQ fixed": "#999999",   # grey
    "Equal":       "#CCCCCC",   # light grey
}
SLICE_COLOR = {"eMBB": "#0072B2", "URLLC": "#E69F00", "MTC": "#009E73"}


def set_ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  figure: {name}.png / .pdf")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_scores() -> pd.DataFrame:
    df = pd.read_csv(MODELS_DIR / "evaluation_closedloop.csv")
    df = df[df["method"] != "Best meta (campaign env)"].copy()  # not comparable
    df["method"] = df["method"].map(PAPER_NAME).fillna(df["method"])
    return df


def load_radio_slice() -> pd.DataFrame:
    df = pd.read_csv(MODELS_DIR / "closedloop_radio_efficiency_per_slice.csv")
    df["method"] = df["method"].map(RADIO_NAME).fillna(df["method"])
    return df


def agg_scores(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["scenario", "method"])["score"]
         .agg(["mean", "std", "min", "max", "count"]).reset_index())
    g["std"] = g["std"].fillna(0.0)
    return g


def cell_radio() -> pd.DataFrame:
    """Cell-level radio metrics from the properly aggregated cell CSV.

    Uses the seed mean of each metric; goodput_b_per_rbg_cell is the true cell
    ratio (sum rx_bytes / sum allocated_rbg per run, then averaged over seeds).
    """
    c = pd.read_csv(MODELS_DIR / "closedloop_radio_efficiency_cell.csv",
                    header=[0, 1], index_col=[0, 1])
    rows = []
    for (sc, m) in c.index:
        rows.append({
            "scenario": sc, "method": RADIO_NAME.get(m, m),
            "served_mbps": c.loc[(sc, m), ("served_mbps_total", "mean")],
            "goodput_b_per_rbg": c.loc[(sc, m), ("goodput_b_per_rbg_cell", "mean")],
            "sla_min": c.loc[(sc, m), ("sla_min", "mean")],
            "satisf_min": c.loc[(sc, m), ("offered_satisfaction_min", "mean")],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def grouped_bars(ax, agg: pd.DataFrame, value: str, err: str | None, ylabel: str):
    methods = [m for m in ORDER if m in agg["method"].unique()]
    x = np.arange(len(SCENARIOS))
    n = len(methods)
    width = 0.8 / n
    for i, m in enumerate(methods):
        vals, errs = [], []
        for sc in SCENARIOS:
            r = agg[(agg["scenario"] == sc) & (agg["method"] == m)]
            vals.append(r[value].values[0] if len(r) else np.nan)
            errs.append(r[err].values[0] if (err and len(r)) else 0.0)
        offset = (i - (n - 1) / 2) * width
        hatch = "///" if m in PROPOSALS else None
        ax.bar(x + offset, vals, width, yerr=errs, label=m,
               color=COLOR[m], edgecolor="black", linewidth=0.4,
               error_kw={"elinewidth": 0.5, "capsize": 1.5}, hatch=hatch)
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_LABEL[s] for s in SCENARIOS])
    ax.set_ylabel(ylabel)


def fig_scores(agg: pd.DataFrame):
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(7.16, 2.8))
    grouped_bars(ax, agg, "mean", "std", "Composite score")
    ax.set_ylim(0, 105)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              frameon=False, columnspacing=1.0, handletextpad=0.4)
    save(fig, "fig1_composite_score")


def fig_goodput(cell: pd.DataFrame):
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    grouped_bars(ax, cell.rename(columns={"goodput_b_per_rbg": "mean"}),
                 "mean", None, "Goodput per RBG (B)")
    ax.set_ylim(0, 52)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.20),
              frameon=False, columnspacing=0.8, handletextpad=0.3)
    save(fig, "fig2_goodput_per_rbg")


def fig_worst_sla(cell: pd.DataFrame):
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    grouped_bars(ax, cell.rename(columns={"sla_min": "mean"}),
                 "mean", None, "Worst-slice SLA satisf. (%)")
    ax.set_ylim(0, 108)
    ax.legend(ncol=2, loc="lower left", frameon=False,
              columnspacing=0.8, handletextpad=0.3)
    save(fig, "fig3_worst_slice_sla")


def fig_congestion_alloc(df_slice: pd.DataFrame):
    """Stacked RBG share per slice, per method, in congestion (the mechanism)."""
    set_ieee_style()
    sc = "congestion"
    sub = df_slice[df_slice["scenario"] == sc]
    methods = [m for m in ORDER if m in sub["method"].unique()]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    for sl in SLICES:
        vals = []
        for m in methods:
            r = sub[(sub["method"] == m) & (sub["slice"] == sl)]
            vals.append(r["rbg_share_pct"].values[0] if len(r) else 0.0)
        ax.bar(x, vals, 0.7, bottom=bottom, label=sl,
               color=SLICE_COLOR[sl], edgecolor="black", linewidth=0.4)
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("RBG share (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14),
              frameon=False, columnspacing=1.0, handletextpad=0.4)
    save(fig, "fig4_congestion_rbg_share")


def fig_robustness(agg: pd.DataFrame):
    """Mean score vs std (dispersion across seeds) — upper-left = robust+good."""
    set_ieee_style()
    # constrained layout + "outside" legends: matplotlib reserves the space for
    # the legend bands below the axes automatically, so nothing gets clipped
    # (in-axes legends overlapped points once n=3 spread congestion to std~44).
    fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
    markers = {"low_traffic": "o", "normal": "s", "congestion": "^", "stressed": "D"}
    for m in [mm for mm in ORDER if mm in agg["method"].unique()]:
        for sc in SCENARIOS:
            r = agg[(agg["scenario"] == sc) & (agg["method"] == m)]
            if not len(r):
                continue
            ax.scatter(r["std"], r["mean"], s=28, color=COLOR[m],
                       marker=markers[sc], edgecolor="black", linewidth=0.4,
                       zorder=3)
    # legends: colors = methods, markers = scenarios (both outside the axes)
    from matplotlib.lines import Line2D
    meth_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR[m],
                           markeredgecolor="black", markersize=6, label=m)
                    for m in ORDER if m in agg["method"].unique()]
    scen_handles = [Line2D([0], [0], marker=markers[s], color="w", markerfacecolor="grey",
                           markeredgecolor="black", markersize=6, label=SCEN_LABEL[s])
                    for s in SCENARIOS]
    # side-by-side outside legends: two legends sharing the same
    # "outside lower center" loc are stacked on top of each other by
    # constrained layout (bbox_to_anchor is ignored for outside locs),
    # so place methods on the lower left and scenarios on the lower right
    fig.legend(handles=meth_handles, loc="outside lower left", ncol=2,
               frameon=False, fontsize=6, title="Method", title_fontsize=6.5,
               columnspacing=0.9, handletextpad=0.3)
    fig.legend(handles=scen_handles, loc="outside lower right", ncol=2,
               frameon=False, fontsize=6, title="Scenario", title_fontsize=6.5,
               columnspacing=0.9, handletextpad=0.3)
    ax.margins(y=0.10)
    ax.set_xlabel("Score std. across seeds (lower = robust)")
    ax.set_ylabel("Mean composite score")
    save(fig, "fig5_robustness_frontier")


# ---------------------------------------------------------------------------
# LaTeX tables (booktabs)
# ---------------------------------------------------------------------------
def _fmt(v, p=1):
    return "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{p}f}"


def tex_header(caption: str, label: str, colspec: str, header_row: str) -> list[str]:
    return [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header_row, r"\midrule",
    ]


def tex_footer() -> list[str]:
    return [r"\bottomrule", r"\end{tabular}", r"\end{table}"]


def bold_max_per_scenario(agg: pd.DataFrame, value: str) -> dict:
    best = {}
    for sc in SCENARIOS:
        sub = agg[agg["scenario"] == sc]
        if len(sub):
            best[sc] = sub.loc[sub[value].idxmax(), "method"]
    return best


def table_scores(agg: pd.DataFrame):
    best = bold_max_per_scenario(agg, "mean")
    lines = tex_header(
        "Closed-loop composite score (mean\\,$\\pm$\\,std over 3 seeds, all runs "
        "executed in the same ns-3 environment). Best per scenario in bold. "
        "MODA variants are the proposed offline-distilled policies.",
        "tab:moda_scores", "l" + "c" * 4,
        "Method & " + " & ".join(SCEN_LABEL[s] for s in SCENARIOS) + r" \\")
    for m in ORDER:
        if m not in agg["method"].unique():
            continue
        cells = []
        for sc in SCENARIOS:
            r = agg[(agg["scenario"] == sc) & (agg["method"] == m)]
            if not len(r):
                cells.append("--"); continue
            txt = f"{r['mean'].values[0]:.1f}$\\pm${r['std'].values[0]:.1f}"
            if best.get(sc) == m:
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        row = f"{m} & " + " & ".join(cells) + r" \\"
        if m == "MODA-RWR":  # rule between proposals and references
            row += "\n\\midrule"
        lines.append(row)
    lines += tex_footer()
    (TAB_DIR / "table1_composite_scores.tex").write_text("\n".join(lines) + "\n")
    print("  table: table1_composite_scores.tex")


def table_radio(cell: pd.DataFrame):
    """Radio efficiency: goodput/RBG and worst-slice SLA, congestion + stressed."""
    scen = ["congestion", "stressed"]
    hdr = ("Method & " +
           " & ".join([f"\\multicolumn{{2}}{{c}}{{{SCEN_LABEL[s]}}}" for s in scen]) + r" \\")
    subhdr = " & " + " & ".join(["Gp/RBG & SLA$_{\\min}$"] * len(scen)) + r" \\"
    lines = tex_header(
        "Radio-resource efficiency under scarcity: goodput per allocated RBG "
        "(bytes) and worst-slice SLA satisfaction (\\%). Higher is better; best "
        "per column in bold.",
        "tab:moda_radio", "l" + "cc" * len(scen), hdr)
    lines.pop()  # drop the plain \midrule from the header helper
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    lines.append(subhdr)
    lines.append(r"\midrule")
    # find best per column
    best_gp = {s: cell[cell.scenario == s].loc[cell[cell.scenario == s]["goodput_b_per_rbg"].idxmax(), "method"] for s in scen}
    best_sla = {s: cell[cell.scenario == s].loc[cell[cell.scenario == s]["sla_min"].idxmax(), "method"] for s in scen}
    for m in ORDER:
        if m not in cell["method"].unique():
            continue
        cells = []
        for s in scen:
            r = cell[(cell.scenario == s) & (cell.method == m)]
            if not len(r):
                cells += ["--", "--"]; continue
            gp = f"{r['goodput_b_per_rbg'].values[0]:.1f}"
            sla = f"{r['sla_min'].values[0]:.1f}"
            if best_gp[s] == m: gp = r"\textbf{" + gp + "}"
            if best_sla[s] == m: sla = r"\textbf{" + sla + "}"
            cells += [gp, sla]
        row = f"{m} & " + " & ".join(cells) + r" \\"
        if m == "MODA-RWR":
            row += "\n\\midrule"
        lines.append(row)
    lines += tex_footer()
    (TAB_DIR / "table2_radio_efficiency.tex").write_text("\n".join(lines) + "\n")
    print("  table: table2_radio_efficiency.tex")


def table_congestion_slices(df_slice: pd.DataFrame):
    """Per-slice allocation & QoS in congestion — shows the starvation mechanism."""
    sc = "congestion"
    sub = df_slice[df_slice["scenario"] == sc]
    lines = tex_header(
        "Per-slice allocation and QoS under \\emph{congestion}. MODA-Q/BC keep "
        "MTC alive; the raw metaheuristic optimum starves it (delay $>$1\\,s). "
        "RBG share (\\%), offered-load satisfaction (\\%), mean delay (ms).",
        "tab:moda_congestion", "llccc",
        r"Method & Slice & RBG\% & Satisf.\% & Delay (ms) \\")
    for m in ORDER:
        if m not in sub["method"].unique():
            continue
        for j, sl in enumerate(SLICES):
            r = sub[(sub["method"] == m) & (sub["slice"] == sl)]
            if not len(r):
                continue
            r = r.iloc[0]
            mname = m if j == 0 else ""
            lines.append(f"{mname} & {sl} & {r['rbg_share_pct']:.1f} & "
                         f"{r['offered_load_satisfaction_pct']:.1f} & "
                         f"{r['delay_ms_mean']:.0f}" + r" \\")
        lines.append(r"\addlinespace[2pt]")
    lines += tex_footer()
    (TAB_DIR / "table3_congestion_per_slice.tex").write_text("\n".join(lines) + "\n")
    print("  table: table3_congestion_per_slice.tex")


# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading closed-loop results...")
    scores = load_scores()
    agg = agg_scores(scores)
    df_slice = load_radio_slice()
    cell = cell_radio()

    # Persist tidy data next to the paper artifacts
    agg.to_csv(DATA_DIR / "scores_aggregate.csv", index=False)
    scores.to_csv(DATA_DIR / "scores_per_seed.csv", index=False)
    df_slice.to_csv(DATA_DIR / "radio_per_slice.csv", index=False)
    cell.to_csv(DATA_DIR / "radio_cell.csv", index=False)

    print("Figures:")
    fig_scores(agg)
    fig_goodput(cell)
    fig_worst_sla(cell)
    fig_congestion_alloc(df_slice)
    fig_robustness(agg)

    print("Tables:")
    table_scores(agg)
    table_radio(cell)
    table_congestion_slices(df_slice)

    print(f"\nAll artifacts under: {OUT_ROOT}")


if __name__ == "__main__":
    main()
