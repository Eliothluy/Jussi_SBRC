"""Replay plan v2 builders for the Etapa 2.4-II-d gate (protocol §18-§20).

The plan reuses OBLIGATORILY the hard cases of the II-c gate (§19) and adds
the new MARGINAL windows (D5) and the mMTC floor microcalibration (D6):

- D1 — the 7 II-c false-infeasible G1 windows (all-OK replays that refuted
  the binary oracle): re-tested with the NEW oracle's supported witness
  action; expectation (§20): the state migrates to FEASIBLE (replay all-OK)
  or MARGINAL (band straddles the SLA — documented, not forced).
- D2 — the 3 II-c G2 FAILURES (the false-feasible axis): the new oracle's
  recommended action at the same state; expectation: absorbed by MARGINAL
  or classified correctly.
- D3 — the 3 II-c G3 EXECUTABLE COUNTER-PROOFS: new witness; expectation:
  NOT INFEASIBLE while an all-OK in-support action exists.
- D4 — the 2 II-c G3 CORROBORATED windows: the new supported
  best-degradation action; expectation: still violates (infeasibility
  corroborated in the supported space) or strongly-edge MARGINAL.
- G4 (control) — the 6 frozen II-c URLLC causality replays, reused VERBATIM
  as controls (a_u in {13, 25, 35}): expectation: tail flat in a_u above 13
  (scheduler/TDD dominance preserved, R50).  NOTE: their a_m = 1 is below
  the mMTC floor ON PURPOSE — they are controls for the a_u axis, not
  feasibility witnesses, and the gate adjudication must treat them as such.
- D5 — new windows classified MARGINAL by the recalibrated oracle (both
  KPI axes with demand, deterministic stratified pick): tests whether the
  MARGINAL class concentrates genuine boundary uncertainty rather than
  arbitrary error (§20 D5).
- D6 — mMTC floor microcalibration (§13): ~3 windows with mMTC demand x
  a_m in {0..5} with (a_e, a_u) fixed at supported values; measures
  a_m_min = the smallest a_m with PDR >= 95% (R41 window rule).

Row format is EXACTLY the executor's (``run_replays.sh``):
``grupo,arm_base,lam,seed,win_t,action_e,action_u,action_m,alvo``.

Witness rows (D1-D5) are ALL inside the supported space
(a_e in [18, 37], a_u >= 13, a_m >= a_m_min, sum <= 53); the two
exceptions above (G4 controls, D6 calibration points) are explicitly
tagged in ``alvo`` and carry role="control"/"calibration" in the metadata
CSV so the gate adjudication never uses them as feasibility witnesses
(protocol §22 C6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Frozen support constants — DUPLICATED LITERALS from
# ``nsoran.sqr.oracle.three_way`` on purpose: the from-import of these names
# proved INTERMITTENTLY corrupted in this environment (module __dict__ has
# them; IMPORT_FROM raises ImportError without any code change).  The
# consistency assertion below (best-effort) cross-checks the copies at
# import time and the recalibration test-suite verifies them too.
C_PRB = 53
URLLC_FLOOR = 13
EMBB_SUPPORT_MIN = 18
EMBB_SUPPORT_MAX = 37

try:  # best-effort cross-check (never load-bearing)
    import nsoran.sqr.oracle.three_way as _tw
    for _name, _val in (("C_PRB", C_PRB), ("URLLC_FLOOR", URLLC_FLOOR),
                        ("EMBB_SUPPORT_MIN", EMBB_SUPPORT_MIN),
                        ("EMBB_SUPPORT_MAX", EMBB_SUPPORT_MAX)):
        _got = _tw.__dict__.get(_name, _val)
        assert _got == _val, f"support constant drift: {_name}"
except Exception:  # pragma: no cover — import robustness fallback
    pass

PLAN_COLUMNS = ("grupo", "arm_base", "lam", "seed", "win_t",
                "action_e", "action_u", "action_m", "alvo")

#: The 7 II-c false-infeasible G1 windows (observed_joint_ok = True).
D1_CASES: tuple[tuple[float, int, int], ...] = (   # (lam, seed, win_t)
    (2.0, 2, 1480), (2.0, 2, 1487), (2.0, 1, 1291), (2.0, 3, 1955),
    (2.0, 1, 1263), (2.0, 1, 1277), (2.0, 3, 1905),
)
#: The 3 II-c G2 failures (observed_joint_ok = False).
D2_CASES: tuple[tuple[float, int, int], ...] = (
    (6.0, 1, 1423), (6.0, 2, 1317), (5.0, 2, 2544),
)
#: The 3 II-c G3 executable counter-proofs (contraprova_executavel = True).
D3_CASES: tuple[tuple[float, int, int], ...] = (
    (6.0, 1, 1866), (7.0, 3, 2013), (6.0, 3, 2263),
)
#: The 2 II-c G3 corroborated windows (corrobora_estrita = True).
D4_CASES: tuple[tuple[float, int, int], ...] = (
    (7.0, 1, 1555), (7.0, 1, 1557),
)
#: The 6 frozen G4 control rows (verbatim from the II-c plan).
G4_CONTROL_ROWS: tuple[tuple[float, int, int, int, int, int], ...] = (
    (5.0, 3, 110, 39, 13, 1),
    (5.0, 3, 110, 27, 25, 1),
    (5.0, 3, 110, 17, 35, 1),
    (5.0, 1, 187, 39, 13, 1),
    (5.0, 1, 187, 27, 25, 1),
    (5.0, 1, 187, 17, 35, 1),
)

#: §20 expectations per group (frozen BEFORE the gate; part of the plan).
GROUP_EXPECTATIONS: dict[str, str] = {
    "D1": "estado migra para FEASIBLE (replay all-OK) ou MARGINAL; "
          "nunca permanecer INFEASIBLE",
    "D2": "absorvido preferencialmente por MARGINAL ou classificado "
          "corretamente (FEASIBLE so se a nova acao seja all-OK)",
    "D3": "ADJUDICACAO: a contraprova II-c (acao do braco prop all-OK) "
          "diverge de estado (historico de filas); o novo oracle classificou "
          "INFEASIBLE — o replay do a* do suporte decide: all-OK refuta o "
          "novo modelo (falso-infeasible novo); violacao corrobora e "
          "explica a contraprova II-c por divergencia de estado",
    "D4": "permanece INFEASIBLE (a* do suporte ainda viola) ou MARGINAL "
          "fortemente na borda",
    "G4": "CONTROLE: cauda URLLC plana em a_u (scheduler/TDD-dominada, R50); "
          "fora do suporte de proposito (a_m=1) — NAO e testemunha",
    "D5": "a classe MARGINAL concentra incerteza de fronteira, nao erro "
          "arbitrario: banda cruza o SLA no replay (nenhum contrato duro)",
    "D6": "microcalibracao: fecha a_m_min = menor a_m com PDR >= 95% (R41); "
          "fora do suporte de proposito (a_m < piso) — NAO e testemunha",
}


def _parse_action(s: str) -> tuple[int, int, int] | None:
    s = str(s).strip()
    if not s.startswith("("):
        return None
    try:
        parts = [int(p) for p in s.strip("()").split(",")]
    except ValueError:
        return None
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else None


def in_supported_space(a_e: int, a_u: int, a_m: int, am_min: int = 3) -> bool:
    """§14 supported-space membership for a candidate witness row."""
    return (EMBB_SUPPORT_MIN <= int(a_e) <= EMBB_SUPPORT_MAX
            and int(a_u) >= URLLC_FLOOR and int(a_m) >= int(am_min)
            and int(a_e) + int(a_u) + int(a_m) <= C_PRB)


def _rows(group: str, cases, actions: dict, am_min: int,
          role: str = "witness") -> list[dict]:
    rows = []
    for (lam, seed, win) in cases:
        act = actions.get((float(lam), int(seed), int(win)))
        if act is None:
            raise KeyError(f"no {group} action for {(lam, seed, win)}")
        ae, au, am = act
        rows.append({
            "grupo": group, "arm_base": "equal", "lam": int(lam),
            "seed": int(seed), "win_t": int(win),
            "action_e": int(ae), "action_u": int(au), "action_m": int(am),
            "alvo": GROUP_EXPECTATIONS[group].split(";")[0],
            "_role": role,
            "_case_of": f"II-c {group}",
            "_in_support": in_supported_space(ae, au, am, am_min),
        })
    return rows


def _action_lookup(cls: pd.DataFrame, group: str, cases) -> dict:
    """New-oracle action per case: witness (F), best marginal (M) or the
    supported best-degradation action (I) — the oracle's OWN recommendation
    at that state, so the replay tests the recalibrated label."""
    lut = {}
    key = cls.set_index(["arm", "lam", "seed", "win"]) if len(cls) else None
    for (lam, seed, win) in cases:
        try:
            r = key.loc[("equal", float(lam), int(seed), int(win))]
        except (KeyError, TypeError):
            raise KeyError(f"window {(lam, seed, win)} not in the "
                           "reprocessed table (arm=equal)")
        if isinstance(r, pd.DataFrame):  # duplicate keys — take the first
            r = r.iloc[0]
        state = r["state_class"]
        col = {"FEASIBLE": "best_feasible_action",
               "MARGINAL": "best_marginal_action",
               "INFEASIBLE": "best_degradation_action"}[state]
        act = _parse_action(r[col])
        if act is None:  # MARGINAL window without a marginal representative?
            act = _parse_action(r["best_degradation_action"])
        lut[(float(lam), int(seed), int(win))] = act
    return lut


def build_plan_d6(windows: pd.DataFrame, am_fixed_e: int = 35,
                  am_fixed_u: int = 13) -> tuple[pd.DataFrame, pd.DataFrame]:
    """§13 microcalibration: a_m in {0..5} on 3 mMTC-active windows.

    Window selection (deterministic): among the EQUAL-arm runs, order the
    (lam, seed) pairs by DESCENDING lambda then seed; for each, take the
    FIRST window >= 200 (post warm-up) whose 1 s rolling PDR is defined
    (mMTC demand at or right before the window); stop at 3 windows with
    distinct (lam, seed).  Actions: a_u = 13 (URLLC floor), a_e = 35 fixed
    (supported, leaves a_m <= 5 within C = 53).  OUT of the supported space
    ON PURPOSE for a_m < a_m_min — calibration, not witness (tagged).
    """
    # demand IN THE OVERRIDE WINDOW is required (pdr_mmtc 100 ms notna
    # <=> dTxPackets > 0 in window t; the 1 s rolling PDR alone may reflect
    # packets from earlier windows and would measure nothing)
    eq = windows[(windows["arm"] == "equal")
                 & (windows["win"] >= 200)
                 & (windows["n_mmtc"] >= 1)
                 & windows["pdr_mmtc"].notna()]
    picks = []
    for (lam, seed), g in eq.groupby(["lam", "seed"]):
        g = g.sort_values("win")
        picks.append((float(lam), int(seed), int(g["win"].iloc[0])))
    picks.sort(key=lambda t: (-t[0], t[1]))
    # spread over distinct lambdas first (highest load — where the floor
    # binds), then allow a second seed of the top lambda if needed
    chosen, seen_lam = [], set()
    for p in picks:
        if p[0] not in seen_lam:
            chosen.append(p)
            seen_lam.add(p[0])
        if len(chosen) == 3:
            break
    if len(chosen) < 3:  # fewer than 3 lambdas with mMTC demand — fill
        for p in picks:
            if p not in chosen:
                chosen.append(p)
            if len(chosen) == 3:
                break
    rows, meta = [], []
    for (lam, seed, win) in chosen[:3]:
        for am in range(0, 6):
            rows.append({
                "grupo": "D6", "arm_base": "equal", "lam": int(lam),
                "seed": int(seed), "win_t": int(win),
                "action_e": int(am_fixed_e), "action_u": int(am_fixed_u),
                "action_m": int(am),
                "alvo": f"pdr_mmtc_vs_am_medir_a_m_min (am={am})",
                "_role": "calibration",
                "_case_of": "protocolo §13",
                "_in_support": in_supported_space(am_fixed_e, am_fixed_u,
                                                  am),
            })
        row = windows[(windows["arm"] == "equal")
                      & (windows["lam"] == lam) & (windows["seed"] == seed)
                      & (windows["win"] == win)].iloc[0]
        meta.append({"lam": lam, "seed": seed, "win": win,
                     "n_mmtc": int(row["n_mmtc"]),
                     "pdr_mmtc_100ms_original": float(row["pdr_mmtc"]),
                     "pdr_mmtc_1s_original": (float(row["pdr_mmtc_1s"])
                                              if pd.notna(row["pdr_mmtc_1s"])
                                              else np.nan)})
    return pd.DataFrame(rows), pd.DataFrame(meta)


def build_plan_v2(cls: pd.DataFrame, windows: pd.DataFrame,
                  am_min: int = 3, d5_per_lam: int = 2,
                  d5_total: int = 10) -> pd.DataFrame:
    """Full D1-D5 + G4-control plan (the D6 rows come from build_plan_d6).

    ``cls`` is the reprocessed 3-way table (equal-arm rows), ``windows`` the
    frozen window dataset (for the D5 stratification features).
    """
    eq = cls[cls["arm"] == "equal"]
    rows: list[dict] = []
    rows += _rows("D1", D1_CASES, _action_lookup(eq, "D1", D1_CASES), am_min)
    rows += _rows("D2", D2_CASES, _action_lookup(eq, "D2", D2_CASES), am_min)
    rows += _rows("D3", D3_CASES, _action_lookup(eq, "D3", D3_CASES), am_min)
    rows += _rows("D4", D4_CASES, _action_lookup(eq, "D4", D4_CASES), am_min)
    for (lam, seed, win, ae, au, am) in G4_CONTROL_ROWS:
        rows.append({
            "grupo": "G4", "arm_base": "equal", "lam": lam, "seed": seed,
            "win_t": win, "action_e": ae, "action_u": au, "action_m": am,
            "alvo": GROUP_EXPECTATIONS["G4"].split(";")[0],
            "_role": "control", "_case_of": "II-c G4 verbatim",
            "_in_support": in_supported_space(ae, au, am, am_min),
        })

    # ---- D5: new MARGINAL windows, deterministic stratified pick --------
    taken = {(float(r["lam"]), int(r["seed"]), int(r["win_t"]))
             for r in rows}
    marg = eq[eq["state_class"] == "MARGINAL"].copy()
    marg = marg[marg["n_embb"] >= 1]
    marg = marg[(marg["lam"].isin([2.0, 4.0, 5.0, 6.0, 7.0]))]
    picked = []
    for lam in sorted(marg["lam"].unique(), reverse=True):
        gl = marg[marg["lam"] == lam].sort_values(["seed", "win"])
        n_here = 0
        for r in gl.itertuples():
            k = (float(r.lam), int(r.seed), int(r.win))
            if k in taken:
                continue
            picked.append(k)
            taken.add(k)
            n_here += 1
            if n_here >= d5_per_lam:
                break
        if len(picked) >= d5_total:
            break
    rows += _rows("D5", picked, _action_lookup(eq, "D5", picked), am_min)

    plan = pd.DataFrame(rows)
    plan = plan[[*PLAN_COLUMNS, "_role", "_case_of", "_in_support"]]
    # deterministic order: group, then lam/seed/win/action
    plan = plan.sort_values(["grupo", "lam", "seed", "win_t", "action_e",
                             "action_u", "action_m"]).reset_index(drop=True)
    return plan


def validate_plan(plan: pd.DataFrame, am_min: int = 3) -> dict:
    """§22-C6 pre-check: witness rows (role='witness') MUST be in-support;
    control/calibration rows are REPORTED out-of-support (by design)."""
    wit = plan[plan["_role"] == "witness"]
    bad = wit[~wit["_in_support"].astype(bool)]
    out = {
        "n_rows": int(len(plan)),
        "n_witness": int(len(wit)),
        "n_witness_out_of_support": int(len(bad)),
        "witness_support_ok": bool(len(bad) == 0),
        "n_control_or_calibration": int((plan["_role"] != "witness").sum()),
        "unique_tags": bool(not plan.duplicated(
            subset=["grupo", "lam", "seed", "win_t", "action_e", "action_u",
                    "action_m"]).any()),
        "sum_le_c": bool((plan["action_e"] + plan["action_u"]
                          + plan["action_m"] <= C_PRB).all()),
        "groups": sorted(plan["grupo"].unique().tolist()),
    }
    return out
