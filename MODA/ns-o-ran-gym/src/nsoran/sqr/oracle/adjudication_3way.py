"""Executable 3-way adjudication of the recalibrated oracle (Etapa 2.4-II-d,
part 2; protocol ``docs/artigo/fase_2_4_II_d_recalibracao_oracle.md`` §18-§22,
§24-§31; Despacho II-d parte 2 — FINAL).

Layers (one per protocol section):

§4 (II-c, MESMO critério) :func:`validate_replay` is REUSED verbatim from
``nsoran.sqr.oracle.adjudication`` (the replay executor and telemetry
conventions are identical); this module only relocates the plan/directories.

§5-analog :func:`build_replay_results_3way` — one row per replay (31 D1-D5 +
18 D6) with the frozen 3-way oracle PREDICTION vs the OBSERVED outcome of the
override window t (telemetry stamped (t+1)·100 ms):
- ``oracle_state_class``: the state classification of the SOURCE window
  (equal arm, frozen reprocessing ``oracle_3way_windows.csv``);
- per-axis predicted band at the OVERRIDE action (p25/p50/p75 from the FROZEN
  recalibrated models re-fit on the same frozen dataset — deterministic LP,
  verified against the saved reprocessing), with the joint band code of the
  action itself;
- observed per-axis SLA oks (frozen R36-R38 rules, R41 window judgments,
  vacuous = OK) and the joint all-OK.

§20 :func:`adjudicate_d1`..:func:`adjudicate_d6` — group verdicts against the
expectations frozen BEFORE the gate (``replay_plan_v2.GROUP_EXPECTATIONS``).

§21 :func:`confusion_matrix_3way` — the 3-way confusion (predicted
FEASIBLE/MARGINAL/INFEASIBLE × observed all-OK/violation) over the valid
WITNESS replays only (role=witness — G4 controls and D6 calibration points do
not vote, replay_v2/README_planos.md), the two conservative views
(FEASIBLE vs NOT-CONFIDENTLY-FEASIBLE; INFEASIBLE vs
NOT-CONFIDENTLY-INFEASIBLE) and the §21 metrics.

§22 :func:`gate_ii_d` — the C1-C7 confidence gate with the thresholds frozen
in the protocol/ask (C1 FIR <= 0.25, C2 FFR <= 0.25, C3 the 5 II-c confirmed
avoidable windows must NOT be state INFEASIBLE, C4 the 2 corroborated G3 stay
coherent, C5 the 3 G3 counter-proofs judged via the D3 replay, C6 witnesses
in support, C7 near-threshold errors absorbed by MARGINAL).  Nothing here is
tuned after seeing results (rule §23).

§24-§27 :func:`freeze_by_lambda` / :func:`avr_table` — POST-PASS freezing of
P_F/P_M/P_I(λ) (renamed from the provisional *_candidate columns),
P_inf = P_I, the AVR of §25 and the L_N/L_O selection of §26/§27.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from nsoran.sqr.oracle.adjudication import (
    WINDOW_MS,
    _replay_tag,
    validate_replay,
    window_kpis,
)
from nsoran.sqr.oracle.dataset import (
    SLA_EMBB_MBPS,
    SLA_MMTC_PDR,
    SLA_URLLC_MS,
    _read_actions,
    per_run_packet_counts,
)
from nsoran.sqr.oracle.quantile import (
    BAND_NAMES,
    EMMB_NEAR_HALF_WIDTH_MBPS,
    URLLC_NEAR_HALF_WIDTH_MS,
    embb_band,
    urllc_band,
)

#: The 5 II-c confirmed-avoidable G2 windows (``adjudication/g2_deltas.csv``,
#: sla_todos_ok_no_override = True; R50).  §22-C3: the new oracle must NOT
#: classify their states INFEASIBLE (a runnable all-OK action existed at the
#: same state).
CONFIRMED_AVOIDABLE_II_C: tuple[tuple[float, int, int], ...] = (
    (6.0, 3, 298), (4.0, 3, 1812), (5.0, 1, 1955), (4.0, 1, 2210),
    (5.0, 3, 1677),
)

#: Witness groups that vote in the §21 confusion (controls G4 and calibration
#: D6 are excluded BY DESIGN — replay_v2/README_planos.md).
WITNESS_GROUPS = ("D1", "D2", "D3", "D4", "D5")


# ---------------------------------------------------------------------------
# Plan / directory plumbing
# ---------------------------------------------------------------------------

def load_plan_d1d5(oracle_dir: str) -> pd.DataFrame:
    """The 31-row D1-D5 plan joined with its frozen metadata."""
    base = pd.read_csv(os.path.join(oracle_dir, "replay_v2", "plan_d1d5.csv"))
    meta = pd.read_csv(os.path.join(oracle_dir, "replay_v2",
                                    "plan_d1d5_metadata.csv"))
    key = ["grupo", "lam", "seed", "win_t", "action_e", "action_u",
           "action_m"]
    plan = base.merge(meta[key + ["_role", "_case_of", "_in_support"]],
                      on=key, how="left", validate="one_to_one")
    for col in ("lam", "seed", "win_t", "action_e", "action_u", "action_m"):
        plan[col] = plan[col].astype(int)
    return plan


def load_plan_d6(oracle_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The 18-row D6 microcalibration plan + the chosen-window metadata."""
    from nsoran.sqr.oracle.replay_plan_v2 import in_supported_space
    plan = pd.read_csv(os.path.join(oracle_dir, "replay_v2",
                                    "plan_d6_mmtc.csv"))
    meta = pd.read_csv(os.path.join(oracle_dir, "replay_v2",
                                    "plan_d6_meta.csv"))
    for col in ("lam", "seed", "win_t", "action_e", "action_u", "action_m"):
        plan[col] = plan[col].astype(int)
    plan["_role"] = "calibration"
    plan["_case_of"] = "protocolo §13"
    plan["_in_support"] = [
        in_supported_space(int(r.action_e), int(r.action_u), int(r.action_m))
        for r in plan.itertuples()]
    return plan, meta


def replay_dir_3way(oracle_dir: str, row) -> str:
    """Replay directory of one plan row (the .done dirs use the INT lam tag —
    the leftover ``lam<f>`` directories are aborted earlier attempts without
    .done and are never read)."""
    sub = "replays_d6" if str(row["grupo"]) == "D6" else "replays_d1d5"
    return os.path.join(oracle_dir, "replay_v2", sub, _replay_tag(row))


# ---------------------------------------------------------------------------
# §5-analog — observed window + frozen oracle prediction per replay
# ---------------------------------------------------------------------------

def _mmtc_window_pdr(kpm_path: str, win_t: int) -> tuple[float, float, float]:
    """(pdr_100ms, tx, rx) of the mMTC demand packets IN window ``win_t``.

    The 1 s rolling PDR of R41 dilutes a single-window override (~1/10 of the
    packets); README_planos.md mandates reporting the override-window PDR too.
    """
    kpm = pd.read_csv(kpm_path)
    wins = np.arange(max(0, win_t - 1), win_t + 2)
    counts = per_run_packet_counts(kpm, wins)
    tx = float(counts["tx_mmtc"].get(win_t, 0.0) or 0.0)
    rx = float(counts["rx_mmtc"].get(win_t, 0.0) or 0.0)
    return (rx / tx if tx > 0 else np.nan), tx, rx


def window_kpis_3way(replay_dir: str, win_t: int,
                     orig_kpm: str | None = None) -> dict:
    """Observed KPIs of the override window — II-c conventions (frozen R41)
    PLUS the D6 resalva columns (100 ms PDR of the window; Δ vs the original
    run's same-window PDR when ``orig_kpm`` is given)."""
    k = window_kpis(replay_dir, win_t)
    hits = sorted(glob.glob(os.path.join(replay_dir, "**",
                                         "sqr_kpm_history.csv"),
                            recursive=True))
    pdr100, tx, rx = _mmtc_window_pdr(hits[0], win_t) if hits else (np.nan,
                                                                    0.0, 0.0)
    k["mmtc_pdr_100ms"] = pdr100
    k["mmtc_tx_100ms"] = tx
    k["mmtc_rx_100ms"] = rx
    if orig_kpm:
        opdr, otx, orx = _mmtc_window_pdr(orig_kpm, win_t)
        k["mmtc_pdr_100ms_original"] = opdr
        k["mmtc_pdr_100ms_delta"] = (pdr100 - opdr
                                     if np.isfinite(pdr100)
                                     and np.isfinite(opdr) else np.nan)
    # mMTC demand AT the override window (not only the rolling 1 s support)
    k["mmtc_demand_100ms"] = bool(tx > 0)
    return k


def _axis_sla_ok(k: dict) -> dict:
    """Per-axis observed SLA oks (frozen R36-R38/R41, vacuous = OK)."""
    n_e = k.get("n_embb", 0)
    embb_ok = True if n_e is None or (isinstance(n_e, float)
                                      and not np.isfinite(n_e)) or n_e < 1 \
        else bool(k["embb_throughput"] >= SLA_EMBB_MBPS)
    urllc_ok = True if k.get("urllc_n_pkts", 0) < 1 else bool(
        k["urllc_latency_p95"] <= SLA_URLLC_MS)
    mmtc_ok = True if not k.get("mmtc_demand", False) else bool(
        k["mmtc_pdr"] >= SLA_MMTC_PDR)
    return {"observed_embb_ok": embb_ok, "observed_urllc_ok": urllc_ok,
            "observed_mmtc_ok": mmtc_ok}


def predict_action_bands(embb_model, urllc_model, wrow: pd.Series,
                         a_e: int, a_u: int) -> dict:
    """Frozen-model band of ONE override action at ONE source-window state.

    The state features are those of the ORIGINAL equal-arm run at window t
    (the replay prefix is bit-identical up to t; the counterfactual semantics
    of the oracle — the same ones that chose the plan's witness actions —
    evaluate the action on that state).  Axes without demand are VACUOUS
    FEASIBLE with NaN bands (frozen convention).
    """
    out: dict = {"band_embb_p25": np.nan, "band_embb_p50": np.nan,
                 "band_embb_p75": np.nan, "band_embb_class": "",
                 "band_urllc_p25": np.nan, "band_urllc_p50": np.nan,
                 "band_urllc_p75": np.nan, "band_urllc_class": "",
                 "action_class_override": ""}
    ne = float(wrow["n_embb"]) if pd.notna(wrow["n_embb"]) else 0.0
    nu = float(wrow["n_urllc"]) if pd.notna(wrow["n_urllc"]) else 0.0
    e_code, u_code = 0, 0
    if ne >= 1:
        q = embb_model.predict_grid_bands(
            np.array([ne]), np.array([float(wrow["cqi_eff"])]),
            np.array([float(wrow["queue_embb"]) if pd.notna(
                wrow["queue_embb"]) else 0.0]),
            np.array([float(wrow["n_embb_backlog"]) if pd.notna(
                wrow["n_embb_backlog"]) else 0.0]),
            np.array([float(wrow["offered_load_embb"]) if pd.notna(
                wrow["offered_load_embb"]) else 0.0]),
            np.array([float(wrow["prev_thr_embb_med"])]),
            np.array([float(a_e)]))
        for qk, name in ((0.25, "p25"), (0.5, "p50"), (0.75, "p75")):
            out[f"band_embb_{name}"] = float(q[qk][0, 0])
        e_code = int(embb_band(q[0.25], q[0.75])[0, 0])
        out["band_embb_class"] = BAND_NAMES[e_code]
    if nu >= 1:
        q = urllc_model.predict_grid_bands(
            np.array([nu]), np.array([float(wrow["n_embb"])]),
            np.array([float(wrow["queue_urllc"]) if pd.notna(
                wrow["queue_urllc"]) else 0.0]),
            np.array([float(wrow["queue_embb"]) if pd.notna(
                wrow["queue_embb"]) else 0.0]),
            np.array([float(wrow["urllc_n_pkts"]) if pd.notna(
                wrow["urllc_n_pkts"]) else 0.0]),
            np.array([float(wrow["dl_slots"])]),
            np.array([float(wrow["prev_p95"])]),
            np.array([float(wrow["prev_miss_pct"])
                      if pd.notna(wrow["prev_miss_pct"]) else 0.0]),
            np.array([float(a_u)]))
        for qk, name in ((0.25, "p25"), (0.5, "p50"), (0.75, "p75")):
            out[f"band_urllc_{name}"] = float(q[qk][0, 0])
        u_code = int(urllc_band(q[0.25], q[0.75])[0, 0])
        out["band_urllc_class"] = BAND_NAMES[u_code]
    # joint band of the action (three_way._COMBINE rule, §6)
    if e_code == 2 or u_code == 2:
        out["action_class_override"] = "INFEASIBLE"
    elif e_code == 0 and u_code == 0:
        out["action_class_override"] = "FEASIBLE"
    else:
        out["action_class_override"] = "MARGINAL"
    return out


def build_replay_results_3way(oracle_dir: str, round2_dir: str,
                              windows: pd.DataFrame, embb_model,
                              urllc_model, cls: pd.DataFrame) -> \
        tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """One row per replay (D1-D5 witnesses + G4 controls + D6 calibration)
    with prediction vs observation, plus the §4 (II-c criterion) validation
    records and the D6 frame.

    ``windows`` is the frozen dataset (``load_windows`` of the recalibration
    driver); ``cls`` the frozen reprocessing (state classes per window).
    """
    plan = load_plan_d1d5(oracle_dir)
    plan_d6, d6_meta = load_plan_d6(oracle_dir)

    wkey = windows.set_index(
        [windows["arm"], windows["lam"], windows["seed"], windows["time_ms"]])
    ckey = cls.set_index(
        [cls["arm"], cls["lam"], cls["seed"], cls["time_ms"]])

    def _src_row(lam: int, seed: int, win_t: int) -> pd.Series | None:
        k = ("equal", float(lam), int(seed), (int(win_t) + 1) * WINDOW_MS)
        try:
            hit = wkey.index.get_indexer_for([k])
        except (TypeError, KeyError):
            return None
        return wkey.iloc[hit[0]] if len(hit) and hit[0] >= 0 else None

    def _cls_row(lam: int, seed: int, win_t: int) -> pd.Series | None:
        k = ("equal", float(lam), int(seed), (int(win_t) + 1) * WINDOW_MS)
        hit = ckey.index.get_indexer_for([k])
        return ckey.iloc[hit[0]] if len(hit) and hit[0] >= 0 else None

    def _orig_kpm(lam: int, seed: int) -> str | None:
        hits = sorted(glob.glob(os.path.join(
            round2_dir, "hetero_sweep", f"A_lam{int(lam)}_s{int(seed)}",
            "**", "sqr_kpm_history.csv"), recursive=True))
        return hits[0] if hits else None

    rows, validations = [], []
    for plan_df in (plan, plan_d6):
        for _, pr in plan_df.iterrows():  # iterrows: _role etc. keep their names
            tag = _replay_tag(pr)
            rdir = replay_dir_3way(oracle_dir, pr)
            val = validate_replay(rdir, plan_row=pr, round2_dir=round2_dir)
            validations.append(val)

            t = int(pr["win_t"])
            row = {
                "group": pr["grupo"],
                "replay_id": tag,
                "role": pr["_role"],
                "case_of": pr["_case_of"],
                "in_support_plan": bool(pr["_in_support"]),
                "lambda_total": float(pr["lam"]),
                "seed": int(pr["seed"]),
                "arm_original": "equal",
                "window_id": t,
                "time_start_ms": t * WINDOW_MS,
                "time_end_ms": (t + 1) * WINDOW_MS,
                "base_action_embb": 18, "base_action_urllc": 17,
                "base_action_mmtc": 18,
                "override_embb": int(pr["action_e"]),
                "override_urllc": int(pr["action_u"]),
                "override_mmtc": int(pr["action_m"]),
                "executed_action": str(val["checks"].get("executed_action",
                                                         "")),
                "override_matches_plan": bool(
                    val["checks"].get("executed_matches_plan", False)),
                "override_only_at_t": bool(
                    val["checks"].get("override_at_target", False)),
                "total_rbg_all_53": bool(
                    val["checks"].get("total_rbg_all_53", False)),
                "replay_valid": bool(val["replay_valid"]),
            }

            crow = _cls_row(pr["lam"], pr["seed"], t)
            wrow = _src_row(pr["lam"], pr["seed"], t)
            if crow is not None:
                row["oracle_state_class"] = crow["state_class"]
                row["oracle_num_feasible_actions"] = int(
                    crow["num_feasible_actions"])
                row["oracle_num_marginal_actions"] = int(
                    crow["num_marginal_actions"])
                row["oracle_num_infeasible_actions"] = int(
                    crow["num_infeasible_actions"])
            else:
                row["oracle_state_class"] = ""
                row["oracle_num_feasible_actions"] = np.nan
                row["oracle_num_marginal_actions"] = np.nan
                row["oracle_num_infeasible_actions"] = np.nan
            if wrow is not None and val["replay_valid"]:
                row.update(predict_action_bands(embb_model, urllc_model,
                                                wrow, int(pr["action_e"]),
                                                int(pr["action_u"])))
            else:
                for c in ("band_embb_p25", "band_embb_p50", "band_embb_p75",
                          "band_urllc_p25", "band_urllc_p50", "band_urllc_p75"):
                    row[c] = np.nan
                for c in ("band_embb_class", "band_urllc_class",
                          "action_class_override"):
                    row[c] = ""

            if val["replay_valid"]:
                k = window_kpis_3way(rdir, t, orig_kpm=_orig_kpm(pr["lam"],
                                                                 pr["seed"]))
                oks = _axis_sla_ok(k)
                row.update(oks)
                row["observed_joint_ok"] = bool(
                    oks["observed_embb_ok"] and oks["observed_urllc_ok"]
                    and oks["observed_mmtc_ok"])
                for c in ("embb_throughput", "urllc_latency_p95",
                          "urllc_deadline_miss", "urllc_n_pkts", "mmtc_pdr",
                          "mmtc_pdr_100ms", "mmtc_pdr_100ms_original",
                          "mmtc_pdr_100ms_delta", "n_embb", "snr_p05",
                          "util_active_dl", "rlf_count"):
                    row[c] = k.get(c, np.nan)
                row["mmtc_demand"] = bool(k.get("mmtc_demand", False))
            else:
                row["observed_joint_ok"] = np.nan
                for c in ("observed_embb_ok", "observed_urllc_ok",
                          "observed_mmtc_ok"):
                    row[c] = np.nan
            rows.append(row)

    res = pd.DataFrame(rows)
    d6 = res[res["group"] == "D6"].reset_index(drop=True)
    return res, validations, d6


# ---------------------------------------------------------------------------
# §20 — group adjudication
# ---------------------------------------------------------------------------

def _violated(r) -> bool:
    return not bool(r.observed_joint_ok)


def adjudicate_d1(res: pd.DataFrame) -> pd.DataFrame:
    """§20-D1: the 7 ex false-infeasible windows migrate to FEASIBLE or
    MARGINAL; INFEASIBLE = model failure (the II-c replay already refuted it,
    here under the NEW witness)."""
    rows = []
    for r in res[res["group"] == "D1"].itertuples():
        ok = bool(r.observed_joint_ok)
        verdict = {
            ("FEASIBLE", True): "CONFIRMED_FEASIBLE (replay all-OK)",
            ("MARGINAL", True): "ABSORVED_MARGINAL (replay all-OK; banda "
                                "cruzava o SLA — estado marginal, sem "
                                "contrato duro)",
            ("MARGINAL", False): "ABSORVED_MARGINAL (replay violou; banda "
                                 "cruzava o SLA — incerteza declarada)",
            ("FEASIBLE", False): "FALSE_FEASIBLE (falha do modelo)",
            ("INFEASIBLE", True): "FALSE_INFEASIBLE_NOVO (falha do modelo)",
            ("INFEASIBLE", False): "CORRECT_INFEASIBLE",
        }.get((r.oracle_state_class, ok), "INVALID_REPLAY")
        rows.append({"replay_id": r.replay_id, "lam": r.lambda_total,
                     "seed": r.seed, "win": r.window_id,
                     "pred": r.oracle_state_class, "replay_all_ok": ok,
                     "verdict": verdict})
    return pd.DataFrame(rows)


def adjudicate_d2(res: pd.DataFrame) -> pd.DataFrame:
    """§20-D2: the 3 G2 failures absorbed by MARGINAL or correctly classified;
    the replay of the NEW recommended action decides."""
    rows = []
    for r in res[res["group"] == "D2"].itertuples():
        ok = bool(r.observed_joint_ok)
        verdict = {
            ("MARGINAL", True): "ABSORVED_MARGINAL (nova acao all-OK; o "
                                "estado era viavel e a banda declarou "
                                "incerteza)",
            ("MARGINAL", False): "ABSORVED_MARGINAL (nova acao violou; "
                                 "incerteza declarada, sem falso-FEASIBLE)",
            ("FEASIBLE", True): "CORRECT_FEASIBLE",
            ("FEASIBLE", False): "FALSE_FEASIBLE (falha do modelo)",
            ("INFEASIBLE", False): "CORRECT_INFEASIBLE (o a* do suporte "
                                   "tambem nao salva)",
            ("INFEASIBLE", True): "FALSE_INFEASIBLE_NOVO (falha do modelo)",
        }.get((r.oracle_state_class, ok), "INVALID_REPLAY")
        rows.append({"replay_id": r.replay_id, "lam": r.lambda_total,
                     "seed": r.seed, "win": r.window_id,
                     "pred": r.oracle_state_class, "replay_all_ok": ok,
                     "verdict": verdict})
    return pd.DataFrame(rows)


def adjudicate_d3(res: pd.DataFrame) -> pd.DataFrame:
    """§20-D3 (THE KEY CONFLICT): the 3 II-c counter-proofs were classified
    INFEASIBLE by the new oracle (the II-c counter-action came from the PROP
    arm whose state diverges by queue history).  The replay of the SUPPORTED
    a* (37,13,3) at the EQUAL-arm state ADJUDICATES:
    - replay VIOLATES  ⇒ the new INFEASIBLE label is corroborated and the
      II-c counter-proof is EXPLAINED by state divergence;
    - replay ALL-OK   ⇒ new false-infeasible ⇒ II-d-v2 required (no freeze).
    """
    rows = []
    for r in res[res["group"] == "D3"].itertuples():
        ok = bool(r.observed_joint_ok)
        verdict = ("CORROBORATED (a* do suporte viola no estado equal; a "
                   "contraprova II-c explica-se por divergencia de estado do "
                   "braco prop)" if not ok
                   else "REFUTED (all-OK do a* do suporte — falso-INFEASIBLE "
                        "novo; II-d-v2 NECESSARIO, nao congelar)")
        rows.append({"replay_id": r.replay_id, "lam": r.lambda_total,
                     "seed": r.seed, "win": r.window_id,
                     "pred": r.oracle_state_class, "replay_all_ok": ok,
                     "corroborates_new_label": not ok, "verdict": verdict})
    return pd.DataFrame(rows)


def adjudicate_d4(res: pd.DataFrame) -> pd.DataFrame:
    """§20-D4: the 2 corroborated G3 windows stay INFEASIBLE (or extreme
    MARGINAL at the edge) and the supported a* must VIOLATE."""
    rows = []
    for r in res[res["group"] == "D4"].itertuples():
        ok = bool(r.observed_joint_ok)
        if r.oracle_state_class == "INFEASIBLE":
            verdict = ("COHERENT (permanece INFEASIBLE e o a* do suporte "
                       "violou)" if not ok else
                       "INCOHERENT (INFEASIBLE mas o a* all-OK — falso-"
                       "INFEASIBLE novo)")
        elif r.oracle_state_class == "MARGINAL" and not ok:
            verdict = "COHERENT (MARGINAL na borda e o a* violou)"
        else:
            verdict = "INCOHERENT"
        rows.append({"replay_id": r.replay_id, "lam": r.lambda_total,
                     "seed": r.seed, "win": r.window_id,
                     "pred": r.oracle_state_class, "replay_all_ok": ok,
                     "verdict": verdict})
    return pd.DataFrame(rows)


def _near_embb(thr: float) -> bool:
    return np.isfinite(thr) and abs(thr - SLA_EMBB_MBPS) \
        <= EMMB_NEAR_HALF_WIDTH_MBPS


def _near_urllc(p95: float) -> bool:
    return np.isfinite(p95) and abs(p95 - SLA_URLLC_MS) \
        <= URLLC_NEAR_HALF_WIDTH_MS


def _in_band(v: float, lo: float, hi: float) -> bool:
    return bool(np.isfinite(v) and np.isfinite(lo) and np.isfinite(hi)
                and lo <= v <= hi)


def _boundary_row(r) -> dict:
    """Boundary-uncertainty evidence of ONE MARGINAL-predicted witness: the
    BINDING axis (the one the model declared marginal at the override action)
    realized inside [p25, p75] (band covers the realization) OR inside the
    frozen near-threshold window.  No hard contract (§20-D5) — evidence."""
    e_marg = r.band_embb_class == "MARGINAL"
    u_marg = r.band_urllc_class == "MARGINAL"
    e_ok = (_in_band(r.embb_throughput, r.band_embb_p25,
                     r.band_embb_p75) or _near_embb(r.embb_throughput))
    u_ok = (_in_band(r.urllc_latency_p95, r.band_urllc_p25,
                     r.band_urllc_p75) or _near_urllc(r.urllc_latency_p95))
    binding = "embb" if e_marg and not u_marg else \
        ("urllc" if u_marg and not e_marg else
         ("both" if e_marg and u_marg else "none"))
    binding_ok = {"embb": e_ok, "urllc": u_ok, "both": e_ok or u_ok,
                  "none": True}[binding]
    return {"replay_id": r.replay_id, "lam": r.lambda_total,
            "seed": r.seed, "win": r.window_id, "binding_axis": binding,
            "obs_thr": r.embb_throughput,
            "band_e": (f"[{r.band_embb_p25:.2f}, {r.band_embb_p75:.2f}]"
                       if np.isfinite(r.band_embb_p25) else ""),
            "obs_thr_in_band_or_near": bool(e_ok),
            "obs_p95": r.urllc_latency_p95,
            "band_u": (f"[{r.band_urllc_p25:.2f}, {r.band_urllc_p75:.2f}]"
                       if np.isfinite(r.band_urllc_p25) else ""),
            "obs_p95_in_band_or_near": bool(u_ok),
            "binding_in_band_or_near": bool(binding_ok),
            "replay_all_ok": bool(r.observed_joint_ok)}


def adjudicate_d5(res: pd.DataFrame) -> pd.DataFrame:
    """§20-D5: does MARGINAL concentrate genuine boundary uncertainty (the
    band crosses the SLA in the replay) rather than arbitrary error?"""
    rows = [_boundary_row(r) for r in
            res[res["group"] == "D5"].itertuples()]
    return pd.DataFrame(rows)


def adjudicate_d6(d6: pd.DataFrame, d6_meta: pd.DataFrame) -> tuple[
        pd.DataFrame, dict]:
    """§20-D6 / §13: close the mMTC floor — PDR × a_m curve; measured
    ``a_m_min_meas`` = smallest a_m with rolling PDR >= 0.95 on ALL windows at
    that a_m.  The ORCHESTRATOR'S DECISION (pre-gate) keeps a_m_min = 3
    conservatively: the Profile-A runs show PDR = 1.0 even at a_m = 0 because
    the IDLE/SURPLUS RBGs serve the mMTC — the oracle must not depend on the
    surplus, so the floor is set by decision and documented, not by the
    measured minimum alone."""
    d6 = d6[d6["replay_valid"]].copy()
    d6["a_m"] = d6["override_mmtc"].astype(int)
    curve = d6.sort_values(["lambda_total", "seed", "a_m"])[[
        "replay_id", "lambda_total", "seed", "window_id", "a_m",
        "mmtc_pdr", "mmtc_pdr_100ms", "mmtc_pdr_100ms_original",
        "mmtc_pdr_100ms_delta", "observed_mmtc_ok"]].rename(
        columns={"mmtc_pdr": "pdr_mmtc_1s_rolling",
                 "window_id": "win"})
    measured = np.nan
    ok_levels = []
    for am, g in d6.groupby("a_m"):
        all_ok = bool(len(g) and g["observed_mmtc_ok"].astype(bool).all())
        ok_levels.append({"a_m": int(am), "n": int(len(g)),
                          "pdr_1s_rolling_min": float(g["mmtc_pdr"].min()),
                          "pdr_100ms_min": (float(g["mmtc_pdr_100ms"].min())
                                            if g["mmtc_pdr_100ms"].notna()
                                            .any() else np.nan),
                          "pdr_ge_095_all_windows": all_ok})
    okdf = pd.DataFrame(ok_levels)
    good = okdf[okdf["pdr_ge_095_all_windows"]]
    if len(good):
        measured = int(good["a_m"].min())
    summary = {
        "a_m_min_measured": (int(measured) if np.isfinite(measured)
                             else None),
        "a_m_min_decision": 3,
        "decision_rule": "ORQUESTRADOR (pre-gate): manter 3 conservador — "
                         "PDR_run=1.0 ate a_m=0 no Perfil A porque o "
                         "sobrante (RBGs ociosos) serve o mMTC; o oracle NAO "
                         "deve depender do sobrante (R27: cota absoluta). "
                         "a_m >= a_m_min = 3 preservado no espaco suportado.",
        "levels": ok_levels,
    }
    return curve, summary


# ---------------------------------------------------------------------------
# §21 — confusion 3-way + conservative views
# ---------------------------------------------------------------------------

def confusion_matrix_3way(res: pd.DataFrame) -> dict:
    """§21 confusion over the valid WITNESS replays (role=witness, valid,
    observed defined): predicted 3-way × observed 2-way (all-OK / violation),
    the two conservative binary views and the §21 metrics.

    MARGINAL predictions carry NO hard contract — they are excluded from the
    FIR/FFR denominators BY DESIGN (that is what 'absorving near-threshold
    errors' means); their breakdown is reported separately.
    """
    wit = res[(res["role"] == "witness") & res["replay_valid"]
              & res["observed_joint_ok"].notna()].copy()
    ok = wit["observed_joint_ok"].astype(bool)
    viol = ~ok
    f = wit["oracle_state_class"] == "FEASIBLE"
    m = wit["oracle_state_class"] == "MARGINAL"
    i = wit["oracle_state_class"] == "INFEASIBLE"
    tp_f, fp_f = int((f & ok).sum()), int((f & viol).sum())
    n_m = int(m.sum())
    m_ok, m_v = int((m & ok).sum()), int((m & viol).sum())
    tn_i, fn_i = int((i & viol).sum()), int((i & ok).sum())
    n_f, n_i = int(f.sum()), int(i.sum())

    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (2 * prec * rec / (prec + rec)
              if np.isfinite(prec) and np.isfinite(rec)
              and (prec + rec) > 0 else np.nan)
        return prec, rec, f1

    prec_f, rec_f, f1_f = _prf(tp_f, fp_f, fn_i + m_ok)   # FN_feas: all-OK
    # ... not confidently predicted FEASIBLE (MARGINAL or INFEASIBLE)
    prec_i, rec_i, f1_i = _prf(tn_i, fn_i, fp_f + m_v)    # FN_inf: violations
    # ... not confidently predicted INFEASIBLE (MARGINAL or FEASIBLE)
    macro_f1 = float(np.nanmean([f1_f, f1_i])) if np.isfinite(f1_f) \
        and np.isfinite(f1_i) else np.nan
    n = int(len(wit))
    out = {
        "n_witness": n,
        "matrix": {  # predicted × observed
            "FEASIBLE_all_ok": tp_f, "FEASIBLE_violation": fp_f,
            "MARGINAL_all_ok": m_ok, "MARGINAL_violation": m_v,
            "INFEASIBLE_all_ok": fn_i, "INFEASIBLE_violation": tn_i,
        },
        "n_pred_feasible": n_f, "n_pred_marginal": n_m,
        "n_pred_infeasible": n_i, "n_obs_all_ok": int(ok.sum()),
        "n_obs_violation": int(viol.sum()),
        "precision_feasible": prec_f, "recall_feasible": rec_f,
        "f1_feasible": f1_f,
        "precision_infeasible": prec_i, "recall_infeasible": rec_i,
        "f1_infeasible": f1_i, "macro_f1": macro_f1,
        "false_infeasible_rate": (fn_i / n_i) if n_i else np.nan,
        "false_feasible_rate": (fp_f / n_f) if n_f else np.nan,
        "marginal_rate": (n_m / n) if n else np.nan,
        "accuracy_confident": ((tp_f + tn_i) / (n_f + n_i))
        if (n_f + n_i) else np.nan,
    }
    # ---- conservative view 1: FEASIBLE vs NOT-CONFIDENTLY-FEASIBLE --------
    nf = ~f                                  # predicted NOT-confidently-F
    out["view_feasible"] = {
        "TP (pred F, all-OK)": tp_f, "FP (pred F, violated)": fp_f,
        "FN (pred ¬F, all-OK)": int((nf & ok).sum()),
        "TN (pred ¬F, violated)": int((nf & viol).sum()),
        "precision": prec_f, "recall": rec_f, "f1": f1_f,
    }
    # ---- conservative view 2: INFEASIBLE vs NOT-CONFIDENTLY-INFEASIBLE ----
    ni = ~i
    out["view_infeasible"] = {
        "TP (pred I, violated)": tn_i, "FP (pred I, all-OK)": fn_i,
        "FN (pred ¬I, violated)": int((ni & viol).sum()),
        "TN (pred ¬I, all-OK)": int((ni & ok).sum()),
        "precision": prec_i, "recall": rec_i, "f1": f1_i,
    }
    out["witness_replay_ids"] = wit["replay_id"].tolist()
    return out


# ---------------------------------------------------------------------------
# §22 — gate C1–C7
# ---------------------------------------------------------------------------

def gate_ii_d(res: pd.DataFrame, cls: pd.DataFrame, d6_summary: dict,
              confusion: dict) -> dict:
    """§22 confidence gate — C1..C7, thresholds FROZEN (protocol §22 + ask).

    C7 operationalization (registered BEFORE the verdict as R57, no model
    quantity touched): (i) every HARD error (predicted INFEASIBLE + replay
    all-OK, or predicted FEASIBLE + replay violation) must be NEAR-threshold
    (|thr-3.5|<=1 Mbps / |p95-10|<=2 ms — the frozen §9/§11 windows) on its
    binding axis — a FAR hard error would be systematic, not marginal noise;
    (ii) every MARGINAL witness must have its binding axis realized inside the
    predicted band OR the near-threshold window (the declared band genuinely
    straddles the SLA — 'estado marginal', §35).
    """
    wit = res[(res["role"] == "witness") & res["replay_valid"]
              & res["observed_joint_ok"].notna()]
    fir = confusion["false_infeasible_rate"]
    ffr = confusion["false_feasible_rate"]

    # ---- C3: the 5 II-c confirmed avoidable must NOT be state INFEASIBLE --
    c3_rows = []
    for (lam, seed, win) in CONFIRMED_AVOIDABLE_II_C:
        hit = cls[(cls["arm"] == "equal") & (cls["lam"] == lam)
                  & (cls["seed"] == seed) & (cls["time_ms"] == (win + 1)
                                             * WINDOW_MS)]
        sc = hit["state_class"].iloc[0] if len(hit) else "MISSING"
        c3_rows.append({"lam": lam, "seed": seed, "win": win,
                        "state_class_new_oracle": sc})
    c3_bad = [r for r in c3_rows if r["state_class_new_oracle"] == "INFEASIBLE"]

    # ---- C4: the 2 corroborated G3 coherent ------------------------------
    d4 = adjudicate_d4(res)
    c4_coherent = int(d4["verdict"].str.startswith("COHERENT").sum()) \
        if len(d4) else 0

    # ---- C5: the 3 counter-proofs judged via D3 --------------------------
    d3 = adjudicate_d3(res)
    n_refuted = int((~d3["corroborates_new_label"]).sum()) if len(d3) else 0

    # ---- C6: witnesses in support ----------------------------------------
    wit_rows = res[res["role"] == "witness"]
    c6_bad = wit_rows[~wit_rows["in_support_plan"].astype(bool)][
        "replay_id"].tolist()

    # ---- C7: near-threshold absorbed by MARGINAL -------------------------
    # (i) hard errors (predicted INFEASIBLE + all-OK, or predicted FEASIBLE +
    # violation) must be NEAR-threshold on every demand axis with a frozen
    # near window (eMBB ±1 Mbps / URLLC ±2 ms) — a FAR hard error would be
    # systematic, not marginal noise;
    # (ii) EVERY MARGINAL witness (D1 2 + D2 2 + D5 10) must have its binding
    # axis realized inside the predicted band or the near window.
    hard = wit[(wit["oracle_state_class"] == "INFEASIBLE")
               & wit["observed_joint_ok"].astype(bool)]
    hard = pd.concat([hard, wit[(wit["oracle_state_class"] == "FEASIBLE")
                                & (~wit["observed_joint_ok"].astype(bool))]])
    far_errors = []
    for r in hard.itertuples():
        axes_far = []
        if r.oracle_state_class == "FEASIBLE":
            # false-FEASIBLE: only the VIOLATING axes count (a comfortable
            # pass on the other axis is not part of the error)
            if (np.isfinite(r.embb_throughput)
                    and r.embb_throughput < SLA_EMBB_MBPS
                    and not _near_embb(r.embb_throughput)):
                axes_far.append(f"embb={r.embb_throughput:.2f}")
            if (np.isfinite(r.urllc_latency_p95)
                    and r.urllc_latency_p95 > SLA_URLLC_MS
                    and not _near_urllc(r.urllc_latency_p95)):
                axes_far.append(f"urllc={r.urllc_latency_p95:.2f}")
        else:
            # false-INFEASIBLE (all-OK): EVERY demand axis must be near — a
            # comfortably-OK window refutes the model structurally
            if (np.isfinite(r.embb_throughput)
                    and not _near_embb(r.embb_throughput)):
                axes_far.append(f"embb={r.embb_throughput:.2f}")
            if (np.isfinite(r.urllc_latency_p95)
                    and not _near_urllc(r.urllc_latency_p95)):
                axes_far.append(f"urllc={r.urllc_latency_p95:.2f}")
        if axes_far:
            far_errors.append({"replay_id": r.replay_id, "far_axes":
                               "; ".join(axes_far)})
    mwit = wit[wit["oracle_state_class"] == "MARGINAL"]
    boundary = pd.DataFrame([_boundary_row(r) for r in mwit.itertuples()])
    c7_bad_m = boundary[~boundary["binding_in_band_or_near"]][
        "replay_id"].tolist() if len(boundary) else []
    d5 = adjudicate_d5(res)

    checks = [
        {"id": "C1_FIR", "value": fir, "threshold": "<= 0.25",
         "passed": bool(np.isfinite(fir) and fir <= 0.25),
         "note": (f"FIR = {confusion['matrix']['INFEASIBLE_all_ok']}/"
                  f"{confusion['n_pred_infeasible']} testemunhas previstas "
                  "INFEASIBLE que vieram all-OK no replay (MARGINAL fora do "
                  "denominador por desenho)")},
        {"id": "C2_FFR", "value": ffr, "threshold": "<= 0.25",
         "passed": bool(np.isfinite(ffr) and ffr <= 0.25),
         "note": (f"FFR = {confusion['matrix']['FEASIBLE_violation']}/"
                  f"{confusion['n_pred_feasible']} testemunhas previstas "
                  "FEASIBLE que violaram no replay")},
        {"id": "C3_avoidable_not_infeasible", "value": len(c3_bad),
         "threshold": "== 0 (as 5 evitáveis confirmadas do II-c NÃO podem "
                      "virar INFEASIBLE)",
         "passed": bool(len(c3_bad) == 0),
         "note": ("classes do novo oracle nas 5 janelas G2 all-OK do II-c: "
                  + ", ".join(f"λ{r['lam']}s{r['seed']}t{r['win']}"
                              f"={r['state_class_new_oracle']}"
                              for r in c3_rows))},
        {"id": "C4_corroborated_coherent", "value": c4_coherent,
         "threshold": "== 2 (as 2 G3 corroboradas coerentes com INFEASIBLE "
                      "ou marginal extremo)",
         "passed": bool(c4_coherent == 2),
         "note": "; ".join(f"{r.replay_id}: {r.verdict}"
                           for r in d4.itertuples())},
        {"id": "C5_counterproofs_via_D3", "value": n_refuted,
         "threshold": "== 0 (nenhuma contraprova G3 com a* do suporte "
                      "all-OK; julgada pelo replay D3)",
         "passed": bool(n_refuted == 0),
         "note": "; ".join(f"{r.replay_id}: {'VIOLOU (corrobora)' if r.corroborates_new_label else 'ALL-OK (refuta → II-d-v2)'}"
                           for r in d3.itertuples())},
        {"id": "C6_witnesses_in_support", "value": len(c6_bad),
         "threshold": "== 0 (nenhuma testemunha fora do suporte; controles "
                      "G4 e calibração D6 não são testemunhas)",
         "passed": bool(len(c6_bad) == 0 and not c6_bad),
         "note": (f"25/25 testemunhas no suporte (plan_checks.json "
                  f"n_witness_out_of_support=0); fora: {c6_bad or 'nenhuma'}")},
        {"id": "C7_near_threshold_absorbed",
         "value": {"far_hard_errors": far_errors,
                   "marginal_not_boundary": c7_bad_m},
         "threshold": ("0 erros duros FAR do limiar E toda testemunha "
                       "MARGINAL com eixo binding dentro da banda ou da "
                       "janela near-threshold congelada"),
         "passed": bool(len(far_errors) == 0 and len(c7_bad_m) == 0),
         "note": (f"erros duros (I∧OK ∪ F∧V) far do limiar: {far_errors or '0'}; "
                  f"MARGINAL sem realização na banda/near: "
                  f"{c7_bad_m or '0'}")},
    ]
    passed = all(c["passed"] for c in checks)
    return {"verdict": "PASS" if passed else "STOP", "checks": checks,
            "confusion": confusion, "c3_rows": c3_rows,
            "d6_summary": d6_summary, "d5_boundary": d5,
            "boundary_all_marginal": boundary,
            "far_hard_errors": far_errors,
            "n_witness": int(confusion["n_witness"]),
            "n_replays_valid": int(res["replay_valid"].sum()),
            "n_replays_total": int(len(res))}


# ---------------------------------------------------------------------------
# §24-§27 — POST-PASS freezing (only called on PASS)
# ---------------------------------------------------------------------------

def freeze_by_lambda(cls: pd.DataFrame) -> pd.DataFrame:
    """§24: freeze P_F/P_M/P_I(λ) (renamed from the PROVISIONAL *_candidate
    columns of §17 — the gate PASSED), P_inf = P_I, plus the §24 regime
    reading per load."""
    agg = cls.groupby("lam").agg(
        n_windows=("state_class", "size"),
        P_F=("state_class", lambda s: float((s == "FEASIBLE").mean())),
        P_M=("state_class", lambda s: float((s == "MARGINAL").mean())),
        P_I=("state_class", lambda s: float((s == "INFEASIBLE").mean())),
        frac_D_min_uncertain=("D_min_uncertain", "mean"),
        median_num_feasible=("num_feasible_actions", "median"),
        median_num_marginal=("num_marginal_actions", "median"),
    ).reset_index().rename(columns={"lam": "lambda_total"})

    def _regime(p_f: float, p_m: float, p_i: float) -> str:
        # §24 reading: NORMAL P_F alto, P_M baixo, P_I ~0; MARGINAL P_M alto;
        # OVERLOAD TRANSITÓRIO 0<P_I<1; OVERLOAD SEVERO P_I -> 1.
        if p_i <= 0.05 and p_f >= 0.60:
            return "NORMAL"
        if p_i >= 0.95:
            return "OVERLOAD_SEVERO"
        if p_i > 0.05:
            return "OVERLOAD_TRANSITORIO"
        return "MARGINAL"

    agg["P_inf"] = agg["P_I"]          # §24: P_inf(lambda) = P_I(lambda)
    agg["regime"] = [_regime(p_f, p_m, p_i) for p_f, p_m, p_i
                     in zip(agg["P_F"], agg["P_M"], agg["P_I"])]
    s = agg["P_F"] + agg["P_M"] + agg["P_I"]
    assert np.allclose(s, 1.0), "P_F + P_M + P_I must be 1"
    return agg


def avr_table(cls: pd.DataFrame) -> pd.DataFrame:
    """§25 revised AVR: P(V=1 | state) with the observed violation of the
    EXECUTED action.  Reported on ALL windows and restricted to windows whose
    executed action is IN the supported space (the honest denominator — the
    supported-space claim cannot be tested by out-of-support executions)."""
    rows = []
    for lam, g in cls.groupby("lam"):
        insup = g["exec_in_support"] == 1
        for label, gg in (("all", g), ("exec_in_support", g[insup])):
            f = gg[gg["state_class"] == "FEASIBLE"]
            fm = gg[gg["state_class"].isin(["FEASIBLE", "MARGINAL"])]
            rows.append({
                "lambda_total": float(lam), "scope": label,
                "n": int(len(gg)),
                "n_state_feasible": int(len(f)),
                "AVR": (float(f["observed_any_violation"].mean())
                        if len(f) else np.nan),
                "AVR_FM": (float(fm["observed_any_violation"].mean())
                           if len(fm) else np.nan),
            })
    return pd.DataFrame(rows)


def avr_by_slice(cls: pd.DataFrame) -> pd.DataFrame:
    """by_slice analog of the II-c table under the 3-way states: per-λ per-axis
    AVR on FEASIBLE states + the saveable rate (axis violation among INFEASIBLE
    windows whose deficit on that axis is zero — an axis-wise slack view)."""
    rows = []
    for lam, g in cls.groupby("lam"):
        f = g[g["state_class"] == "FEASIBLE"]
        i = g[g["state_class"] == "INFEASIBLE"]
        row = {"lambda_total": float(lam), "n_windows": int(len(g)),
               "n_state_feasible": int(len(f)),
               "n_state_infeasible": int(len(i))}
        for ax in ("embb", "urllc", "mmtc"):
            vcol = f"observed_{ax}_violation"
            row[f"AVR_{ax}"] = (float(f[vcol].mean()) if len(f) else np.nan)
            if len(i):
                # saveable: INFEASIBLE windows with NO violation on this axis
                # (the infeasibility came from OTHER axes)
                row[f"infeasible_not_{ax}"] = float((i[vcol] == 0).mean())
            else:
                row[f"infeasible_not_{ax}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def select_ln_lo(by_lambda: pd.DataFrame) -> dict:
    """§26/§27: L_N = the load CLOSEST to the ideal regime (P_I ~ 0, P_M
    baixo, P_F alto) — a RANKING choice among the frozen loads, not an
    absolute filter (if no load reaches P_I≈0 the closest is chosen and the
    gap is REPORTED, §37 'suporte insuficiente é relatado'); L_O = load with
    0 < P_I < 1 AND relevant P_M, preferring transitory overload over
    collapse (λ7 NOT the default — P_I too close to 1)."""
    # stable two-key ranking: primary P_I asc, secondary P_M asc, P_F desc
    cand_n = by_lambda.sort_values(["P_I", "P_M", "P_F"],
                                   ascending=[True, True, False])
    ln = cand_n.iloc[0] if len(cand_n) else None
    cand_o = by_lambda[(by_lambda["P_I"] > 0) & (by_lambda["P_I"] < 1)
                       & (by_lambda["P_M"] >= 0.10)]
    # transitory first: smallest P_I that still shows a relevant P_M
    cand_o = cand_o.sort_values(["P_I"], ascending=True)
    lo = cand_o.iloc[0] if len(cand_o) else None
    return {
        "L_N": (None if ln is None else
                {"lambda_total": float(ln["lambda_total"]),
                 "P_F": float(ln["P_F"]), "P_M": float(ln["P_M"]),
                 "P_I": float(ln["P_I"]), "regime": ln["regime"]}),
        "L_O": (None if lo is None else
                {"lambda_total": float(lo["lambda_total"]),
                 "P_F": float(lo["P_F"]), "P_M": float(lo["P_M"]),
                 "P_I": float(lo["P_I"]), "regime": lo["regime"]}),
        "rule_L_N": ("menor P_I; desempate por menor P_M e maior P_F "
                     "(ideal §26: P_I ~ 0, P_M baixo, P_F alto; se nenhum "
                     "load chega a P_I≈0, o mais próximo é escolhido e a "
                     "distância é RELATADA)"),
        "rule_L_O": ("0 < P_I < 1 com P_M >= 0.10; MENOR P_I (sobrecarga "
                     "transitória, não colapso — λ7 evitado por padrão, §27)"),
    }


def sqr_jc_decision(by_lambda: pd.DataFrame, avr: pd.DataFrame,
                    ln: dict, lo: dict) -> dict:
    """§28-§31: the SQR-JC architecture decision from AVR / P_I / P_M
    (specification ONLY — implementation is a later phase)."""
    lam_n, lam_o = ln["L_N"]["lambda_total"], lo["L_O"]["lambda_total"]
    a_n = avr[(avr["lambda_total"] == lam_n)
              & (avr["scope"] == "exec_in_support")]
    a_o = avr[(avr["lambda_total"] == lam_o)
              & (avr["scope"] == "exec_in_support")]
    avr_n = float(a_n["AVR"].iloc[0]) if len(a_n) else np.nan
    avr_o = float(a_o["AVR"].iloc[0]) if len(a_o) else np.nan
    p_i_n = ln["L_N"]["P_I"]
    p_m_n = ln["L_N"]["P_M"]
    p_i_o = lo["L_O"]["P_I"]
    p_m_o = lo["L_O"]["P_M"]
    components = []
    # §28: AVR alto e P_I baixo -> priorizar Joint Coordination
    jc_first = bool(np.isfinite(avr_n) and avr_n >= 0.10 and p_i_n <= 0.10)
    if jc_first:
        components.append("Joint_Coordination (prioridade: AVR_LN alto e "
                          "P_I(L_N) baixo)")
    if p_i_o > 0.10:
        components += ["Feasibility_Awareness (P_I relevante em L_O)",
                       "Graceful_Degradation (INFEASIBLE em L_O)"]
    if p_m_n >= 0.10 or p_m_o >= 0.10:
        components.append("Uncertainty_Aware_Coordination (MARGINAL "
                          "frequente — §28: o controlador NÃO trata "
                          "MARGINAL como INFEASIBLE)")
    return {
        "avr_L_N": avr_n, "avr_L_O": avr_o,
        "P_I_L_N": p_i_n, "P_M_L_N": p_m_n,
        "P_I_L_O": p_i_o, "P_M_L_O": p_m_o,
        "joint_coordination_first": jc_first,
        "components": components,
        "invariants": [
            "MARGINAL != INFEASIBLE no controlador (§28: estados marginais "
            "não declaram inviabilidade; preferir robustez)",
            "URLLC tratado como GATE DE ESTADO acima de a_u >= 13 (§29; "
            "scheduler/TDD-dominado, R50 — não corrigir a cauda com cota)",
            "a_m >= a_m_min = 3 preservado antes de redistribuir slack (§30)",
            "Estado FEASIBLE: preservar pisos + distribuir slack (§31); "
            "INFEASIBLE: graceful degradation (§31)",
        ],
    }
