"""Replay-plan selector for the executable oracle validation (Etapa 2.4, Despacho II-a).

Reads the Despacho-I oracle artifacts and emits the deterministic replay plan
``executable_validation/replay_plan.csv`` executed by ``oracle/run_replays.sh``
with ``examples/sqr_run.py --controller replay`` (one single-window action
override per run, same seed as the original 2.3 run — MESMA seed ⇒ MESMO
estado até t).  Protocol: ``docs/artigo/rodada_2_etapa_2_4_oracle_viabilidade.md``
§15 (replay amostral, preferir |F_t| pequeno e estado perto do limiar) and §16
(false-infeasible concern), adjudicated by groups:

- **G1 false-infeasible (8)** — λ2 windows classified joint-INFEASIBLE by the
  MEDIAN gate (joint_feasible=0) whose EXECUTED outcome was all-SLA-OK (the
  §16 suspects).  Override = the oracle witness built at the OPTIMISTIC p25
  gate ``(e_min_p25, min(13, C−e_min_p25), resto)`` — the same R44 recipe at
  the §16 sensitivity port, because on the median port these windows have
  e_min = −1 (F_t = ∅, no witness exists).  Windows infeasible even at p25
  are skipped (no witness to test — the oracle is consistent there).
  Alvo: if every replayed KPI is OK, the offline "infeasible" label was a
  REAL false-infeasible.
- **G2 avoidable (8)** — λ4–6 windows joint-FEASIBLE on the equal arm WITH an
  observed violation (avoidable_any = 1).  Override = the oracle witness
  ``(e_min, min(13, C−e_min), resto)`` (R44; the witness IS the oracle's
  representative feasible action).  Alvo: the violation must disappear ⇒
  real Avoidable Violation Rate.
- **G3 infeasible-by-capacity (6)** — λ6–7 windows joint-infeasible with the
  URLLC gate OPEN (urllc_gate = True ⇒ the closure is eMBB CAPACITY, not the
  URLLC port; seed 2 has none — a population fact, not a filter choice).
  Override = the oracle best-degradation action ``a* = (a_star_e, 0, 0)``
  (R44 neutral-weight argmin ΣD_i, degenerate along (a_u, a_m)).  a* is
  recomputed per window with the SAME oracle code
  (:func:`nsoran.sqr.oracle.models.fit_models` +
  :func:`nsoran.sqr.oracle.simplex.window_feasibility`) refit on the merged
  §19+extra tables; the refit is CROSS-CHECKED against the exported
  ``oracle_critical_windows.csv`` witness/a* columns (hard-fails on
  divergence).  Alvo: measure the EXECUTED deficit vs the predicted D_min.
- **G4 causal URLLC (6)** — λ4–6 equal-arm windows with n_urllc ≥ 2 (where
  the R43 action-independence claim says the URLLC tail cannot move):
  2 distinct windows × 3 overrides ``a_u ∈ {13, 25, 35}`` with
  ``a_e = 53 − a_u − 1``, ``a_m = 1`` — deliberately OUTSIDE the executed
  action support (equal (18,17,18) / prop (37,13,3)).  Alvo: the
  miss% × a_u response curve adjudicates R43 causally.

Deterministic selection: within each group candidates are ranked by the §15
preference (small |F_t|, state near the SLA threshold), then by (win, seed);
a per-(lam, seed) cap forces seed/load spread.  Windows picked by an earlier
group are excluded from later overlapping pools (G2 before G4).  Output has
EXACTLY the 9 positional columns the runner reads:
``grupo,arm_base,lam,seed,win_t,action_e,action_u,action_m,alvo``.

Usage (from ``MODA/ns-o-ran-gym``)::

    .venv/bin/python examples/sqr_oracle_select_replays.py \
        [--oracle-dir results/sqr_characterization_round2/oracle] \
        [--out <oracle-dir>/executable_validation/replay_plan.csv]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from nsoran.sqr.oracle.models import fit_models
from nsoran.sqr.oracle.simplex import C_PRB, window_feasibility

#: Plan row counts per group (the ask's ~28 replays: 8 + 8 + 6 + 6; G4 is
#: 2 windows x 3 override actions).
GROUP_SIZES = {"G1": 8, "G2": 8, "G3": 6, "G4_WINDOWS": 2}
#: Per-(lam, seed) cap of picked windows, forcing the seed/load spread.
CAPS_PER_RUN = {"G1": 3, "G2": 1, "G3": 2, "G4": 1}

#: G4 override grid: a_u values, with a_e = C − a_u − 1 and a_m = 1.
G4_AU_VALUES = (13, 25, 35)
G4_AM_RBGS = 1

#: Exactly the 9 positional columns the executor (oracle/run_replays.sh) reads.
PLAN_COLUMNS = (
    "grupo", "arm_base", "lam", "seed", "win_t",
    "action_e", "action_u", "action_m", "alvo",
)

#: Alvo (target of the adjudication) per group — comma-free (CSV column 9).
GROUP_ALVOS = {
    "G1": "kpi_todos_ok_reprova_inviabilidade_offline",
    "G2": "violacao_deve_sumir_prova_avr_real",
    "G3": "deficit_executado_vs_Dmin_predito",
    "G4": "curva_miss_x_au_fora_do_suporte_R43",
}

ALVO = GROUP_ALVOS  # short alias


# ---------------------------------------------------------------------------
# Loading and model refit (exact same oracle code paths as Despacho I)
# ---------------------------------------------------------------------------

def load_merged_windows(oracle_dir: str) -> pd.DataFrame:
    """§19 table joined 1:1 with the extras on (arm, lam, seed, time_ms).

    The §19 CSV carries the violations / joint_feasible / D_min; the extras
    carry win, executed (a_e, a_u, a_m), e_min, urllc_gate and every model
    input (n_* demand counts via §19; thr/p95/miss/pdr/cqi_eff).  lam is
    normalized to ``lam``; the merge is validated 1:1 (the two tables are
    row-aligned by construction).
    """
    w19 = pd.read_csv(os.path.join(oracle_dir, "oracle_windows.csv")).rename(
        columns={"lambda_total": "lam", "time": "time_ms"}
    )
    extra_cols = [
        "arm", "lam", "seed", "win", "time_ms", "n_embb_backlog",
        "thr_embb_med", "latency_urllc_p95_ms", "latency_urllc_miss_pct",
        "urllc_n_pkts", "pdr_mmtc", "cqi_embb", "cqi_eff",
        "a_e", "a_u", "a_m", "e_min", "urllc_gate",
    ]
    ex = pd.read_csv(
        os.path.join(oracle_dir, "tables", "oracle_windows_extra.csv"),
        usecols=extra_cols,
    )
    merged = w19.merge(ex, on=["arm", "lam", "seed", "time_ms"], validate="1:1")
    if len(merged) != len(w19) or len(merged) != len(ex):
        raise RuntimeError(
            f"§19/extras join lost rows: {len(w19)} x {len(ex)} -> {len(merged)}"
        )
    return merged


def refit_oracle_models(merged: pd.DataFrame, critical: pd.DataFrame) -> dict:
    """Refit the three KPI models with the Despacho-I code and CROSS-CHECK.

    The fit inputs (n_embb, a_e, thr_embb_med, cqi_embb, n_urllc,
    latency_urllc_p95_ms, latency_urllc_miss_pct, pdr_mmtc) are exactly the
    columns the Despacho-I pipeline fitted on; the export
    ``oracle_critical_windows.csv`` carries the final-model witness/a* for
    900 windows, so every row with ``witness_a_e >= 0`` must reproduce its
    witness and every row its a* — any divergence means the refit is NOT the
    Despacho-I calibration and the plan would be unfaithful (hard fail).
    """
    models = fit_models(merged)

    n_bad_w = n_bad_a = n_checked = 0
    for r in critical.itertuples():
        row = merged[(merged["arm"] == r.arm) & (merged["lam"] == r.lam)
                     & (merged["seed"] == r.seed) & (merged["win"] == r.win)]
        if len(row) != 1:
            raise RuntimeError(f"critical row not in merged: {r!r}")
        f = row.iloc[0]
        res = window_feasibility(
            models["embb"], models["urllc"], models["mmtc"],
            n_embb=int(f["n_embb"]), n_urllc=int(f["n_urllc"]),
            n_mmtc=int(f["n_mmtc"]), cqi=float(f["cqi_eff"]),
        )
        n_checked += 1
        if r.witness_a_e >= 0 and tuple(res.witness or (-1, -1, -1)) != \
                (int(r.witness_a_e), int(r.witness_a_u), int(r.witness_a_m)):
            n_bad_w += 1
        if tuple(res.best_deg_action) != (int(r.a_star_e), int(r.a_star_u),
                                          int(r.a_star_m)):
            n_bad_a += 1
    if n_checked and (n_bad_w or n_bad_a):
        raise RuntimeError(
            "refit cross-check FAILED: witness mismatches "
            f"{n_bad_w}, a* mismatches {n_bad_a} of {n_checked} critical rows "
            "(refit is not the Despacho-I calibration)"
        )
    return models


def _witness_from_e_min(e_min: int, c: int = C_PRB) -> tuple[int, int, int]:
    """The R44 witness recipe: (e_min, min(13, C−e_min), resto) — Σ = c."""
    if e_min < 0:
        raise ValueError(f"no witness for e_min = {e_min}")
    a_u = min(13, c - int(e_min))
    a_m = c - int(e_min) - a_u
    return int(e_min), int(a_u), int(a_m)


def _nearness(df: pd.DataFrame) -> pd.Series:
    """Distance of the executed KPIs to their frozen thresholds (smaller =
    closer to the SLA boundary, §15 'estados próximos ao SLA').  NaN KPI
    (vacuous gate) contributes 0."""
    d_e = (df["thr_embb_med"] - 3.5).abs()
    d_u = (df["latency_urllc_p95_ms"] - 10.0).abs()
    return d_e.fillna(0.0) + d_u.fillna(0.0)


def _pick_spread(cand: pd.DataFrame, n: int, cap_per_run: int,
                 rank_cols: list[str]) -> pd.DataFrame:
    """Top-``n`` rows of ``cand`` ranked by ``rank_cols`` (ascending, then
    win, then seed), with at most ``cap_per_run`` windows per (lam, seed).
    Deterministic: stable sort on explicit keys."""
    cand = cand.sort_values(rank_cols + ["win", "seed"], kind="mergesort")
    picked, used = [], {}
    for r in cand.itertuples():
        key = (float(r.lam), int(r.seed))
        if used.get(key, 0) >= cap_per_run:
            continue
        picked.append(r.Index)
        used[key] = used.get(key, 0) + 1
        if len(picked) >= n:
            break
    return cand.loc[picked]


# ---------------------------------------------------------------------------
# Group selection (each returns plan rows with actions resolved)
# ---------------------------------------------------------------------------

def select_g1(merged: pd.DataFrame, models: dict) -> pd.DataFrame:
    """λ2 median-gate-infeasible windows whose EXECUTED outcome was all-OK;
    witness at the optimistic p25 port ((0.25, 0.25), the §16 sensitivity
    actually used by classify_windows' joint_feasible_p25)."""
    eq = merged[merged["arm"] == "equal"]
    cand = eq[(eq["lam"] == 2.0) & (~eq["joint_feasible"].astype(bool))].copy()
    viol = cand[["observed_embb_violation", "observed_urllc_violation",
                 "observed_mmtc_violation"]]
    executed_ok = viol.fillna(0).eq(0).all(axis=1) & viol.notna().any(axis=1)
    cand["executed_ok"] = executed_ok

    # p25-port witness (None when infeasible even at p25 — skip those).
    p25_e, p25_u, p25_m = [], [], []
    for r in cand.itertuples():
        res = window_feasibility(
            models["embb"], models["urllc"], models["mmtc"],
            n_embb=int(r.n_embb), n_urllc=int(r.n_urllc),
            n_mmtc=int(r.n_mmtc), cqi=float(r.cqi_eff),
            embb_q=0.25, urllc_q=0.25,
        )
        wit = _witness_from_e_min(res.e_min) if res.e_min >= 0 else None
        p25_e.append(wit[0] if wit else -1)
        p25_u.append(wit[1] if wit else -1)
        p25_m.append(wit[2] if wit else -1)
    cand["w25_e"], cand["w25_u"], cand["w25_m"] = p25_e, p25_u, p25_m
    cand = cand[(cand["w25_e"] >= 0) & (cand["w25_u"] >= 0)
                & (cand["w25_m"] >= 0)]
    cand["near"] = _nearness(cand)

    # Rank: executed-OK first (cleanest false-infeasible evidence), then the
    # §15 near-threshold preference.
    cand["ok_rank"] = (~cand["executed_ok"]).astype(int)
    picked = _pick_spread(cand, GROUP_SIZES["G1"], CAPS_PER_RUN["G1"],
                          ["ok_rank", "near"])
    return pd.DataFrame({
        "grupo": "G1",
        "arm_base": "equal",
        "lam": picked["lam"].astype(float).astype(int).values,
        "seed": picked["seed"].astype(int).values,
        "win_t": picked["win"].astype(int).values,
        "action_e": picked["w25_e"].astype(int).values,
        "action_u": picked["w25_u"].astype(int).values,
        "action_m": picked["w25_m"].astype(int).values,
        "alvo": ALVO["G1"],
    })


def select_g2(merged: pd.DataFrame) -> pd.DataFrame:
    """λ4–6 equal-arm feasible windows with an OBSERVED violation
    (avoidable_any = 1); override = the median-port oracle witness."""
    eq = merged[merged["arm"] == "equal"]
    cand = eq[(eq["lam"].between(4.0, 6.0)) & eq["joint_feasible"].astype(bool)
              & (eq["avoidable_any"] == 1.0)].copy()
    cand["near"] = _nearness(cand)
    picked = _pick_spread(cand, GROUP_SIZES["G2"], CAPS_PER_RUN["G2"],
                          ["num_feasible_actions", "near"])
    wit = [_witness_from_e_min(int(e)) for e in picked["e_min"]]
    return pd.DataFrame({
        "grupo": "G2",
        "arm_base": "equal",
        "lam": picked["lam"].astype(float).astype(int).values,
        "seed": picked["seed"].astype(int).values,
        "win_t": picked["win"].astype(int).values,
        "action_e": [w[0] for w in wit],
        "action_u": [w[1] for w in wit],
        "action_m": [w[2] for w in wit],
        "alvo": ALVO["G2"],
    })


def select_g3(merged: pd.DataFrame, models: dict) -> pd.DataFrame:
    """λ6–7 equal-arm infeasible windows with the URLLC gate OPEN (capacity-
    driven closure); override = the oracle a* recomputed per window (R44
    neutral argmin, degenerate (a_star_e, 0, 0))."""
    eq = merged[merged["arm"] == "equal"]
    cand = eq[(eq["lam"].isin([6.0, 7.0])) & (~eq["joint_feasible"].astype(bool))
              & (eq["urllc_gate"].astype(bool))].copy()
    a_star = []
    for r in cand.itertuples():
        res = window_feasibility(
            models["embb"], models["urllc"], models["mmtc"],
            n_embb=int(r.n_embb), n_urllc=int(r.n_urllc),
            n_mmtc=int(r.n_mmtc), cqi=float(r.cqi_eff),
        )
        a_star.append(tuple(res.best_deg_action))
    cand["as_e"] = [a[0] for a in a_star]
    cand["as_u"] = [a[1] for a in a_star]
    cand["as_m"] = [a[2] for a in a_star]
    # §15 near-threshold: the SMALLEST predicted D_min is the closest to the
    # boundary (a near-feasible state), so deficit deficits rank ascending.
    picked = _pick_spread(cand, GROUP_SIZES["G3"], CAPS_PER_RUN["G3"],
                          ["min_total_deficit"])
    return pd.DataFrame({
        "grupo": "G3",
        "arm_base": "equal",
        "lam": picked["lam"].astype(float).astype(int).values,
        "seed": picked["seed"].astype(int).values,
        "win_t": picked["win"].astype(int).values,
        "action_e": picked["as_e"].astype(int).values,
        "action_u": picked["as_u"].astype(int).values,
        "action_m": picked["as_m"].astype(int).values,
        "alvo": ALVO["G3"],
    })


def select_g4(merged: pd.DataFrame, exclude: set[tuple]) -> pd.DataFrame:
    """2 distinct λ4–6 equal-arm windows with n_urllc ≥ 2 (URLLC packets
    present), × 3 overrides a_u ∈ {13, 25, 35}, a_e = C − a_u − 1, a_m = 1."""
    eq = merged[merged["arm"] == "equal"]
    cand = eq[(eq["lam"].between(4.0, 6.0)) & (eq["n_urllc"] >= 2)
              & eq["latency_urllc_p95_ms"].notna()].copy()
    cand = cand[~cand.apply(lambda r: (r["lam"], int(r["seed"]), int(r["win"])),
                            axis=1).isin(exclude)]
    cand["near_u"] = (cand["latency_urllc_p95_ms"] - 10.0).abs()
    windows = _pick_spread(cand, GROUP_SIZES["G4_WINDOWS"], CAPS_PER_RUN["G4"],
                           ["near_u"])
    rows = []
    for r in windows.itertuples():
        for au in G4_AU_VALUES:
            ae = C_PRB - au - G4_AM_RBGS
            rows.append({
                "grupo": "G4",
                "arm_base": "equal",
                "lam": int(r.lam),
                "seed": int(r.seed),
                "win_t": int(r.win),
                "action_e": ae,
                "action_u": au,
                "action_m": G4_AM_RBGS,
                "alvo": ALVO["G4"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plan assembly + CLI
# ---------------------------------------------------------------------------

def build_plan(oracle_dir: str) -> tuple[pd.DataFrame, dict]:
    """Full deterministic plan (4 groups) + selection stats for the report."""
    merged = load_merged_windows(oracle_dir)
    critical = pd.read_csv(
        os.path.join(oracle_dir, "tables", "oracle_critical_windows.csv")
    )
    models = refit_oracle_models(merged, critical)

    g1 = select_g1(merged, models)
    g2 = select_g2(merged)
    g3 = select_g3(merged, models)
    used = {(float(r.lam), int(r.seed), int(r.win_t))
            for r in pd.concat([g1, g2, g3]).itertuples()}
    g4 = select_g4(merged, exclude=used)

    plan = pd.concat([g1, g2, g3, g4], ignore_index=True)[list(PLAN_COLUMNS)]
    for col in ("lam", "seed", "win_t", "action_e", "action_u", "action_m"):
        plan[col] = plan[col].astype(int)
    sums = plan[["action_e", "action_u", "action_m"]].sum(axis=1)
    if (sums > C_PRB).any() or (plan[["action_e", "action_u", "action_m"]]
                                < 0).any().any():
        raise RuntimeError("plan contains an invalid action (Σ > C or negative)")
    if plan.duplicated(subset=["grupo", "lam", "seed", "win_t",
                               "action_e", "action_u", "action_m"]).any():
        raise RuntimeError("plan contains duplicate (group, window, action)")
    stats = {
        "rows": int(len(plan)),
        "by_group": {k: int(v) for k, v in
                     plan.groupby("grupo").size().items()},
        "windows_by_group": {
            k: int(plan[plan.grupo == k]
                   .drop_duplicates(["lam", "seed", "win_t"]).shape[0])
            for k in ("G1", "G2", "G3", "G4")
        },
        "sum_le_c": bool((sums <= C_PRB).all()),
        "models_refit_crosscheck": "passed (witness + a* on 900 critical rows)",
    }
    return plan, stats


def main(argv=None) -> int:
    default_oracle = os.path.join(
        os.path.dirname(__file__), "..", "results",
        "sqr_characterization_round2", "oracle",
    )
    parser = argparse.ArgumentParser(
        description="Select the Despacho-II replay plan (oracle §15-§16)"
    )
    parser.add_argument("--oracle_dir", dest="oracle_dir",
                        default=os.path.abspath(default_oracle),
                        help="Despacho-I oracle artifact directory")
    parser.add_argument("--out", dest="out", default="",
                        help="output plan CSV (default <oracle_dir>/"
                             "executable_validation/replay_plan.csv)")
    args = parser.parse_args(argv)

    plan, stats = build_plan(args.oracle_dir)
    out = args.out or os.path.join(
        args.oracle_dir, "executable_validation", "replay_plan.csv"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plan.to_csv(out, index=False)

    print(f"[select_replays] plan: {out}")
    print(f"[select_replays] rows: {stats['rows']} "
          f"by group: {stats['by_group']} "
          f"(distinct windows: {stats['windows_by_group']})")
    print(f"[select_replays] Σa ≤ {C_PRB}: {stats['sum_le_c']} | "
          f"model refit: {stats['models_refit_crosscheck']}")
    print(plan.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
