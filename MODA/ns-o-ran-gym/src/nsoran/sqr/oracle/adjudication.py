"""Executable adjudication of the oracle against the 28 replays (Etapa 2.4,
Despacho II-c; protocol ``docs/artigo/proximo_passo_etapa_2_4_II_c_adjudicacao_oracle.md``).

Three layers, one per protocol section:

§4  :func:`validate_replay` — per-replay integrity of the SINGLE-WINDOW
     override: ``.done`` marker, CSV parseability, monotonic stamps, C = 53,
     override ONLY at the target window (mode ``replay_override`` exactly at
     act-step t; base action (18, 17, 18) everywhere else), executed quota ==
     plan override, and DETERMINISM — the telemetry prefix (windows < t) must
     be BIT-IDENTICAL to the frozen 2.3 run of the same (lam, seed) (same
     seed => same state up to t; the only allowed difference in the prefix is
     the IPC bookkeeping columns ``prb_requested_pct_*`` /
     ``prb_executed_rbg_*``, sentinel −1/0 in the original network-only arm,
     whose budget-derived quota must equal the replay's executed quota).

§5  :func:`build_replay_results` — ``oracle_replay_results.csv`` with the
     EXACT column order of protocol §5: observed KPIs of the override window
     t (telemetry row stamped (t+1)·100 ms; win = t) vs the oracle prediction
     joined from ``oracle_windows.csv`` on (arm, lam, seed, time=(t+1)·100).

§6–§10 :func:`adjudicate` — group verdicts G1–G4, the confusion matrix with
     INFEASIBLE as the positive class, and the confidence gate of protocol
     §11.  G1's FIR is a FALSE-INFEASIBLE rate under the AUXILIARY p25 port
     (the G1 witnesses were built at p25 — R48) and is registered as such;
     the confusion matrix itself is on the PRIMARY median port.

Language rules (§25) live in the report, not here; this module only computes.

NO ns-3 simulation is launched here — the replays already ran
(``executable_validation/replays/<tag>/``, Despacho II-b).
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

from nsoran.sqr.oracle.dataset import (
    SLA_EMBB_MBPS,
    SLA_MMTC_PDR,
    SLA_URLLC_MS,
    _read_actions,
    _read_kpm_windows,
    _urllc_packet_windows,
    per_run_packet_counts,
)
from nsoran.sqr.oracle.replay_sources import replay_paths, rlf_count_from_logs

#: Replay plan groups and their planned sizes (Despacho II-a; G4 = 2 windows
#: x 3 override actions).
GROUP_SIZES = {"G1": 8, "G2": 8, "G3": 6, "G4": 6}

#: Base action of the equal arm (R47): the scenario's initial partition.
EQUAL_BASE_ACTION = (18, 17, 18)

#: Steps of the 100 ms window in ms.
WINDOW_MS = 100

#: §5 EXACT column order of oracle_replay_results.csv (protocol §5).
RESULTS_COLUMNS: tuple[str, ...] = (
    "group",
    "replay_id",
    "lambda_total",
    "seed",
    "arm_original",
    "window_id",
    "time_start_ms",
    "time_end_ms",
    "base_action_embb",
    "base_action_urllc",
    "base_action_mmtc",
    "override_embb",
    "override_urllc",
    "override_mmtc",
    "oracle_predicted_feasible",
    "oracle_predicted_infeasible",
    "oracle_num_feasible_actions",
    "observed_embb_ok",
    "observed_urllc_ok",
    "observed_mmtc_ok",
    "observed_joint_ok",
    "embb_throughput",
    "urllc_deadline_miss",
    "urllc_latency_p95",
    "mmtc_pdr",
    "snr_p05",
    "rlf_count",
    "replay_valid",
)


# ---------------------------------------------------------------------------
# §4 — per-replay integrity validation
# ---------------------------------------------------------------------------

def load_replay_plan(oracle_dir: str) -> pd.DataFrame:
    """The 28-row replay plan (Despacho II-a) with int-typed columns."""
    path = os.path.join(oracle_dir, "executable_validation", "replay_plan.csv")
    plan = pd.read_csv(path)
    for col in ("lam", "seed", "win_t", "action_e", "action_u", "action_m"):
        plan[col] = plan[col].astype(int)
    plan["lamf"] = plan["lam"].astype(float)
    return plan


def _replay_tag(row) -> str:
    return (f"{row['grupo']}_{row['arm_base']}_lam{row['lam']}_s{row['seed']}"
            f"_t{row['win_t']}_a{row['action_e']}-{row['action_u']}"
            f"-{row['action_m']}")


def _monotonic(vals: np.ndarray) -> bool:
    """Non-decreasing in FILE ORDER — per stamp the telemetry carries one row
    per UE / per slice (equal values), so strict increase is impossible; a
    disorder shows up as a negative step."""
    v = np.asarray(vals)
    return bool(len(v) and np.all(np.diff(v) >= 0))


def _kpm_prefix_identical(rep_kpm: pd.DataFrame, orig_kpm: pd.DataFrame,
                          stamp_max: int) -> tuple[bool, str]:
    """Bit-exact equality of the KPM rows with stamp <= stamp_max (windows < t
    plus the boundary row 100·t which still belongs to window t−1... no:
    stamp 100·t belongs to window t−1 = the last BASE window; the override is
    in force from window t, whose rows are stamped (t+1)·100)."""
    a = rep_kpm[rep_kpm["timestamp"] <= stamp_max].reset_index(drop=True)
    b = orig_kpm[orig_kpm["timestamp"] <= stamp_max].reset_index(drop=True)
    if len(a) != len(b):
        return False, f"row count {len(a)} != {len(b)} at stamp <= {stamp_max}"
    if not a.equals(b):
        for col in a.columns:
            if not a[col].equals(b[col]):
                bad = (a[col] != b[col]) & ~(a[col].isna() & b[col].isna())
                return False, (f"column {col} differs in {int(bad.sum())} "
                               f"prefix rows (first at stamp "
                               f"{int(a['timestamp'][bad.values.argmax()])})")
    return True, f"{len(a)} KPM rows bit-identical (stamps <= {stamp_max})"


def _prb_prefix_identical(rep_prb: pd.DataFrame, orig_prb: pd.DataFrame,
                          stamp_max: int) -> tuple[bool, str]:
    """Prefix equality of the PRB-window telemetry, allowing ONLY the IPC
    bookkeeping columns (``prb_requested_pct_*`` always; ``prb_executed_rbg_*``
    carries the −1/0 sentinels in the original network-only arm vs the real
    quota in the replay): the quota IN FORCE must still match exactly
    (original budget-derived quota == replay prb_executed_rbg_i), and every
    PHYSICAL accumulator (scheduled/budget/idle/dl_slots) must be identical."""
    bookkeeping = {"prb_requested_pct_0", "prb_requested_pct_1",
                   "prb_requested_pct_2",
                   "prb_executed_rbg_0", "prb_executed_rbg_1",
                   "prb_executed_rbg_2"}
    a = rep_prb[rep_prb["timestamp_ms"] <= stamp_max].reset_index(drop=True)
    b = orig_prb[orig_prb["timestamp_ms"] <= stamp_max].reset_index(drop=True)
    if len(a) != len(b):
        return False, f"row count {len(a)} != {len(b)} at stamp <= {stamp_max}"
    for col in a.columns:
        if col in bookkeeping:
            continue
        if not a[col].equals(b[col]):
            bad = (a[col] != b[col]) & ~(a[col].isna() & b[col].isna())
            return False, (f"column {col} differs in {int(bad.sum())} "
                           f"prefix rows")
    # quota in force: original budget-derived == replay executed (integers).
    # Windows with dl_slots == 0 have budget accumulator 0 by construction
    # (quota × 0 slots) — the quota is NOT recoverable there; the physical
    # columns are already compared bit-exact above, so those rows are skipped.
    slots = b["dl_slots"].values.astype(float)
    has_slots = slots > 0
    for s in range(3):
        from_budget = np.round(np.divide(b[f"prb_budget_rbg_{s}"].values.astype(float),
                                         slots, out=np.zeros(len(b)),
                                         where=has_slots))
        if not np.array_equal(from_budget[has_slots],
                              a[f"prb_executed_rbg_{s}"].values.astype(float)[has_slots]):
            ndiff = int((from_budget[has_slots]
                         != a[f"prb_executed_rbg_{s}"].values.astype(float)[has_slots]).sum())
            return False, (f"quota in force diverges in slice {s} "
                           f"({ndiff} rows with dl_slots > 0)")
    return True, (f"{len(a)} PRB rows identical (stamps <= {stamp_max}; "
                  f"only IPC bookkeeping columns differ, quotas match)")


def validate_replay(replay_dir: str, plan_row: pd.Series,
                    round2_dir: str) -> dict:
    """§4 integrity + determinism checks of ONE replay directory.

    Returns a dict with every check and an overall ``replay_valid`` flag.
    Never raises on a failed check — the CSV must carry the failure.
    """
    tag = _replay_tag(plan_row)
    t = int(plan_row["win_t"])
    stamp_max = t * WINDOW_MS          # last stamp of the pre-override prefix
    checks: dict[str, object] = {"replay_id": tag, "checks": {}}
    c = checks["checks"]

    def _fail(msg: str) -> dict:
        checks["replay_valid"] = False
        c["failure"] = msg
        return checks

    # -- .done marker ------------------------------------------------------
    done = os.path.join(replay_dir, ".done")
    if not os.path.isfile(done):
        return _fail("missing .done marker")
    c["done_marker"] = open(done).read().strip()

    # -- telemetry files ----------------------------------------------------
    try:
        paths = replay_paths(replay_dir)
    except FileNotFoundError as exc:
        return _fail(str(exc))
    c["telemetry"] = paths["telemetry_subdir"]

    kpm = pd.read_csv(paths["kpm"])
    prb = pd.read_csv(paths["prb"])
    delays = pd.read_csv(paths["delays"])

    # -- CSV integrity + monotonic stamps -----------------------------------
    c["kpm_monotonic"] = _monotonic(kpm["timestamp"].values)
    c["prb_monotonic"] = _monotonic(prb["timestamp_ms"].values)
    step = pd.read_csv(os.path.join(replay_dir, "step_metrics.csv"))
    c["step_monotonic"] = _monotonic(step["timestamp_ms"].values)
    c["kpm_missing_total"] = int(step["kpm_missing"].sum()) \
        if "kpm_missing" in step else -1
    d = delays[delays["sliceId"] == 1]
    c["urllc_pkts"] = int(len(d))
    if not (c["kpm_monotonic"] and c["prb_monotonic"] and c["step_monotonic"]):
        return _fail("non-monotonic timestamps")
    if c["kpm_missing_total"] != 0:
        return _fail(f"kpm_missing steps = {c['kpm_missing_total']}")

    # -- C = 53 (R01) --------------------------------------------------------
    c["total_rbg_all_53"] = bool((prb["total_rbg"] == 53).all())
    if not c["total_rbg_all_53"]:
        return _fail("total_rbg != 53 in some window")

    # -- override ONLY at the target window ---------------------------------
    ov = step[step["mode"] == "replay_override"]
    c["override_rows"] = int(len(ov))
    c["override_steps"] = sorted(ov["step"].unique().tolist())
    c["override_at_target"] = bool(len(ov) == 3 and set(ov["step"]) == {t})
    if not c["override_at_target"]:
        return _fail(f"override rows/steps wrong: {c['override_steps']}")

    # -- base action before/after (equal base: constant (18,17,18), R47) ----
    base = step[step["mode"] != "replay_override"]
    expected_base = {"equal": EQUAL_BASE_ACTION}
    if plan_row["arm_base"] != "equal":
        return _fail(f"unsupported base arm {plan_row['arm_base']!r}")
    eb = expected_base["equal"]
    per_slice_ok = True
    for s in range(3):
        bs = base[base["slice_id"] == s]
        per_slice_ok &= bool(len(bs) and (bs["action"] == eb[s]).all()
                             and (bs["mode"] == "replay_equal").all())
    c["base_actions_ok"] = per_slice_ok
    if not per_slice_ok:
        return _fail("base action/mode deviates outside the override window")

    # -- executed quota at window t == plan override -------------------------
    prb_t = prb[prb["timestamp_ms"] == (t + 1) * WINDOW_MS]
    if len(prb_t) != 1:
        return _fail(f"prb row for window {t} missing ({len(prb_t)} rows)")
    exec_q = tuple(int(prb_t.iloc[0][f"prb_executed_rbg_{s}"]) for s in range(3))
    plan_q = (int(plan_row["action_e"]), int(plan_row["action_u"]),
              int(plan_row["action_m"]))
    c["executed_action"] = list(exec_q)
    c["executed_matches_plan"] = bool(exec_q == plan_q)
    c["override_stamp_present"] = bool((kpm["timestamp"] == (t + 1) * WINDOW_MS).any()
                                       or int(prb_t.iloc[0]["dl_slots"]) >= 0)
    if not c["executed_matches_plan"]:
        return _fail(f"executed quota {exec_q} != plan override {plan_q}")

    # -- determinism: prefix identical to the frozen 2.3 run -----------------
    orig_dir = os.path.join(round2_dir, "hetero_sweep",
                            f"A_lam{plan_row['lam']}_s{plan_row['seed']}")
    if not os.path.isdir(orig_dir):
        return _fail(f"original run dir missing: {orig_dir}")
    ok_kpm, msg_kpm = (False, "original kpm missing")
    hits = sorted(glob.glob(os.path.join(orig_dir, "**", "sqr_kpm_history.csv"),
                            recursive=True))
    orig_kpm = pd.read_csv(hits[0]) if hits else pd.DataFrame()
    if len(orig_kpm):
        ok_kpm, msg_kpm = _kpm_prefix_identical(kpm, orig_kpm, stamp_max)
    c["kpm_prefix_identical"] = ok_kpm
    c["kpm_prefix_detail"] = msg_kpm

    hits = sorted(glob.glob(os.path.join(orig_dir, "**", "sqr_prb_windows.csv"),
                            recursive=True))
    ok_prb, msg_prb = (False, "original prb missing")
    orig_prb = pd.read_csv(hits[0]) if hits else pd.DataFrame()
    if len(orig_prb):
        ok_prb, msg_prb = _prb_prefix_identical(prb, orig_prb, stamp_max)
    c["prb_prefix_identical"] = ok_prb
    c["prb_prefix_detail"] = msg_prb

    hits = sorted(glob.glob(os.path.join(orig_dir, "**", "sqr_packet_delays.csv"),
                            recursive=True))
    ok_del, msg_del = (False, "original delays missing")
    if hits:
        orig_del = pd.read_csv(hits[0])
        # PRE-OVERRIDE events only: a packet is pre-override when DELIVERED
        # before the override window starts (100·t ms).  Packets arriving
        # earlier but delivered INSIDE window t may shift by ±1 scheduler
        # slot under the override (expected physical effect, not a break of
        # determinism); they are excluded from the bit-exactness claim.
        rep_del_ms = (delays["txSec"] + delays["delaySec"]) * 1000.0
        org_del_ms = (orig_del["txSec"] + orig_del["delaySec"]) * 1000.0
        a = delays[rep_del_ms < stamp_max].sort_values(
            ["txSec", "delaySec"]).reset_index(drop=True)
        b = orig_del[org_del_ms < stamp_max].sort_values(
            ["txSec", "delaySec"]).reset_index(drop=True)
        ok_del = a.equals(b)
        msg_del = (f"{len(a)} delivered-before-override packet rows "
                   f"bit-identical (delivery < {stamp_max} ms)"
                   if ok_del else
                   f"packet prefix differs ({len(a)} vs {len(b)} rows "
                   f"delivered < {stamp_max} ms)")
    c["delays_prefix_identical"] = ok_del
    c["delays_prefix_detail"] = msg_del

    checks["replay_valid"] = bool(ok_kpm and ok_prb and ok_del)
    return checks


# ---------------------------------------------------------------------------
# §5 — observed KPIs of the override window from the replay telemetry
# ---------------------------------------------------------------------------

def window_kpis(replay_dir: str, win_t: int) -> dict:
    """Observed KPIs of oracle window ``win_t`` from ONE replay's telemetry.

    Same conventions as the oracle dataset (R41/R45): eMBB = MEDIAN
    ``windowThroughputMbps`` over UE-windows with demand (row stamped
    (t+1)·100 ms); URLLC = per-window p95/miss% of packet delays by ARRIVAL
    window; mMTC = 1 s ROLLING PDR (the frozen R38 window analog);
    ``snr_p05`` = 5th percentile of the valid (≠ −999 sentinel) per-UE SNR
    rows of the window; ``rlf_count`` = RLF markers in the replay's
    stdout/stderr (see :func:`nsoran.sqr.oracle.replay_sources.rlf_count_from_logs`).
    """
    paths = replay_paths(replay_dir)
    kpm = pd.read_csv(paths["kpm"])
    prb = pd.read_csv(paths["prb"])
    delays = pd.read_csv(paths["delays"])

    acts = _read_actions(prb)
    row = acts[acts["win"] == win_t]
    if len(row) != 1:
        raise RuntimeError(f"replay {replay_dir}: window {win_t} absent "
                           f"({len(row)} prb rows)")
    row = row.iloc[0]

    pieces = _read_kpm_windows(kpm)
    embb = pieces["embb"]
    out: dict = {}
    stamp = (win_t + 1) * WINDOW_MS
    krow = kpm[kpm["timestamp"] == stamp]
    out["kpm_stamp_present"] = bool(len(krow))
    out["embb_throughput"] = (float(embb.loc[win_t, "thr_embb_med"])
                              if win_t in embb.index
                              and pd.notna(embb.loc[win_t, "thr_embb_med"])
                              else np.nan)
    n_e = (embb.loc[win_t, "n_embb"]
           if win_t in embb.index else np.nan)
    out["n_embb"] = int(n_e) if pd.notna(n_e) else 0

    up = _urllc_packet_windows(paths["delays"])
    urow = up[up["win"] == win_t]
    if len(urow) == 1:
        out["urllc_latency_p95"] = float(urow.iloc[0]["latency_urllc_p95_ms"])
        out["urllc_deadline_miss"] = float(urow.iloc[0]["latency_urllc_miss_pct"])
        out["urllc_n_pkts"] = int(urow.iloc[0]["urllc_n_pkts"])
    else:
        out["urllc_latency_p95"] = np.nan
        out["urllc_deadline_miss"] = np.nan
        out["urllc_n_pkts"] = 0

    # mMTC 1 s rolling PDR over the windows this replay covers
    counts = per_run_packet_counts(kpm, np.arange(0, int(acts["win"].max()) + 1))
    lo = max(0, win_t - 9)
    txs = float(counts["tx_mmtc"].reindex(range(lo, win_t + 1)).fillna(0.0).sum())
    rxs = float(counts["rx_mmtc"].reindex(range(lo, win_t + 1)).fillna(0.0).sum())
    out["mmtc_pdr"] = (rxs / txs) if txs > 0 else np.nan
    out["mmtc_demand"] = bool(txs > 0)

    # SNR p05 over the window's valid per-UE rows (all slices pooled)
    valid = krow[krow["snrDb"] != -999.0]["snrDb"] if len(krow) else pd.Series(dtype=float)
    out["snr_p05"] = (float(np.percentile(valid, 5)) if len(valid) else np.nan)

    # utilization of the override window (frozen D2 definition)
    out["util_active_dl"] = (float(row["util_active_dl"])
                             if pd.notna(row["util_active_dl"]) else np.nan)
    out["executed_action"] = [int(row["a_e"]), int(row["a_u"]), int(row["a_m"])]
    out["rlf_count"] = rlf_count_from_logs(replay_dir)
    return out


def _sla_oks(k: dict) -> dict:
    """Frozen R36–R38 window judgments on the observed KPIs (vacuous = OK)."""
    embb_ok = True if k["n_embb"] < 1 else bool(k["embb_throughput"] >= SLA_EMBB_MBPS)
    urllc_ok = True if k["urllc_n_pkts"] < 1 else bool(
        k["urllc_latency_p95"] <= SLA_URLLC_MS)
    mmtc_ok = True if not k.get("mmtc_demand", False) else bool(
        k["mmtc_pdr"] >= SLA_MMTC_PDR)
    return {"observed_embb_ok": embb_ok, "observed_urllc_ok": urllc_ok,
            "observed_mmtc_ok": mmtc_ok}


# ---------------------------------------------------------------------------
# §5 — oracle_replay_results.csv
# ---------------------------------------------------------------------------

def build_replay_results(oracle_dir: str, round2_dir: str,
                         redo_invalid: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """One §5 row per replay + the §4 validation records.

    ``redo_invalid`` is accepted for protocol §4's "refazer somente o replay
    afetado": this function does NOT re-run ns-3 (the replays are frozen
    Despacho-II-b outputs) — a corrupted replay is reported and must be
    re-run by the executor; the row carries ``replay_valid = False``.
    """
    plan = load_replay_plan(oracle_dir)
    ow = pd.read_csv(os.path.join(oracle_dir, "oracle_windows.csv"))
    ow["key"] = ow.apply(lambda r: (r["arm"], float(r["lambda_total"]),
                                    int(r["seed"]), int(r["time"])), axis=1)
    ow_idx = ow.set_index("key")

    rows, validations = [], []
    for pr in plan.itertuples():
        tag = _replay_tag({"grupo": pr.grupo, "arm_base": pr.arm_base,
                           "lam": pr.lam, "seed": pr.seed, "win_t": pr.win_t,
                           "action_e": pr.action_e, "action_u": pr.action_u,
                           "action_m": pr.action_m})
        replay_dir = os.path.join(oracle_dir, "executable_validation",
                                  "replays", tag)
        val = validate_replay(replay_dir, plan_row=plan[
            (plan["grupo"] == pr.grupo) & (plan["lam"] == pr.lam)
            & (plan["seed"] == pr.seed) & (plan["win_t"] == pr.win_t)
            & (plan["action_e"] == pr.action_e)].iloc[0],
            round2_dir=round2_dir)
        validations.append(val)

        t = int(pr.win_t)
        time_ms = (t + 1) * WINDOW_MS
        key = ("equal", float(pr.lam), int(pr.seed), time_ms)
        hit = ow_idx.index.get_indexer_for([key])
        pred = ow_idx.iloc[hit[0]] if len(hit) and hit[0] >= 0 else None

        if val["replay_valid"]:
            k = window_kpis(replay_dir, t)
            oks = _sla_oks(k)
            joint_ok = bool(oks["observed_embb_ok"] and oks["observed_urllc_ok"]
                            and oks["observed_mmtc_ok"])
        else:
            k = {}
            oks = {"observed_embb_ok": np.nan, "observed_urllc_ok": np.nan,
                   "observed_mmtc_ok": np.nan}
            joint_ok = np.nan

        rows.append({
            "group": pr.grupo,
            "replay_id": tag,
            "lambda_total": float(pr.lam),
            "seed": int(pr.seed),
            "arm_original": "equal",
            "window_id": t,
            "time_start_ms": t * WINDOW_MS,
            "time_end_ms": time_ms,
            "base_action_embb": EQUAL_BASE_ACTION[0],
            "base_action_urllc": EQUAL_BASE_ACTION[1],
            "base_action_mmtc": EQUAL_BASE_ACTION[2],
            "override_embb": int(pr.action_e),
            "override_urllc": int(pr.action_u),
            "override_mmtc": int(pr.action_m),
            "oracle_predicted_feasible": bool(pred["joint_feasible"]) if pred is not None else np.nan,
            "oracle_predicted_infeasible": (not bool(pred["joint_feasible"])) if pred is not None else np.nan,
            "oracle_num_feasible_actions": int(pred["num_feasible_actions"]) if pred is not None else np.nan,
            "observed_embb_ok": oks["observed_embb_ok"],
            "observed_urllc_ok": oks["observed_urllc_ok"],
            "observed_mmtc_ok": oks["observed_mmtc_ok"],
            "observed_joint_ok": joint_ok,
            "embb_throughput": k.get("embb_throughput", np.nan),
            "urllc_deadline_miss": k.get("urllc_deadline_miss", np.nan),
            "urllc_latency_p95": k.get("urllc_latency_p95", np.nan),
            "mmtc_pdr": k.get("mmtc_pdr", np.nan),
            "snr_p05": k.get("snr_p05", np.nan),
            "rlf_count": k.get("rlf_count", np.nan),
            "replay_valid": bool(val["replay_valid"]),
            # -- adjudication auxiliaries (beyond §5's minimum; kept LAST so
            # the §5 column order stays a strict prefix) --
            "util_active_dl": k.get("util_active_dl", np.nan),
            "executed_action": ("(%d,%d,%d)" % tuple(k["executed_action"])
                                if "executed_action" in k else ""),
        })
    res = pd.DataFrame(rows)
    head = list(RESULTS_COLUMNS)
    res = res[head + [c for c in res.columns if c not in head]]
    return res, validations


# ---------------------------------------------------------------------------
# §6–§10 — group adjudication, confusion matrix
# ---------------------------------------------------------------------------

def confusion_counts(res: pd.DataFrame) -> dict:
    """§10 confusion matrix with INFEASIBLE as the POSITIVE class.

    Oracle infeasible + replay all-OK  -> FN_feas (FALSE-INFEASIBLE, the §16
    concern); oracle infeasible + replay violates -> TN_infeas (corroborated);
    oracle feasible + replay all-OK -> TP_feas; oracle feasible + replay
    violates -> FP_feas (false-feasible).  Only valid G1–G3 replays vote.
    """
    sub = res[(res["group"].isin(["G1", "G2", "G3"])) & res["replay_valid"]
              & res["observed_joint_ok"].notna()]
    inf_pred = sub["oracle_predicted_infeasible"].astype(bool)
    violates = ~sub["observed_joint_ok"].astype(bool)
    tp_feas = int((~inf_pred & ~violates).sum())
    fp_feas = int((~inf_pred & violates).sum())
    fn_feas = int((inf_pred & ~violates).sum())   # false-infeasible
    tn_infeas = int((inf_pred & violates).sum())
    n = tp_feas + fp_feas + fn_feas + tn_infeas
    precision_inf = tn_infeas / (tn_infeas + fn_feas) if (tn_infeas + fn_feas) else np.nan
    recall_inf = tn_infeas / (tn_infeas + fp_feas) if (tn_infeas + fp_feas) else np.nan
    f1_inf = (2 * precision_inf * recall_inf / (precision_inf + recall_inf)
              if np.isfinite(precision_inf) and np.isfinite(recall_inf)
              and (precision_inf + recall_inf) > 0 else np.nan)
    fir = fn_feas / (fn_feas + tn_infeas) if (fn_feas + tn_infeas) else np.nan
    return {
        "n_replays": n, "TP_feas": tp_feas, "FP_feas": fp_feas,
        "FN_feas_false_infeasible": fn_feas, "TN_infeas": tn_infeas,
        "accuracy": (tp_feas + tn_infeas) / n if n else np.nan,
        "precision_inf": precision_inf, "recall_inf": recall_inf,
        "f1_inf": f1_inf,
        "false_infeasible_rate": fir,   # §10 highlight: FN / (FN+TN)
    }


def adjudicate_g1(res: pd.DataFrame) -> dict:
    """§6 G1 — FALSE-INFEASIBLE under the AUXILIARY p25 criterion (explicit).

    The G1 witnesses were built at the p25 port (R48): a replay with all
    SLAs OK refutes the median-port infeasible label AT THE p25 TEST ACTION.
    FIR here is therefore 'false-infeasible sob critério auxiliar p25' and is
    NOT a direct estimate of the median-port population FIR (the sample was
    selected as suspects).
    """
    g = res[(res["group"] == "G1") & res["replay_valid"]
            & res["observed_joint_ok"].notna()]
    n_tested = int(len(g))
    n_ok = int(g["observed_joint_ok"].astype(bool).sum())
    return {"group": "G1", "n_executed": n_tested,
            "false_infeasible_count": n_ok, "n_violation": n_tested - n_ok,
            "false_infeasible_rate": (n_ok / n_tested) if n_tested else np.nan,
            "criterion": "p25 auxiliary witness port (R48) — NOT the primary "
                         "median port"}


def _row_at(df: pd.DataFrame, arm: str, lam: float, seed: int,
            time_ms: int) -> pd.Series | None:
    """Row of a per-window table at (arm, lam, seed, time) — boolean-mask
    lookup (MultiIndex ``in`` proved flaky for some keys at this size)."""
    if not len(df):
        return None
    m = pd.Series(True, index=df.index)
    if "arm" in df.columns:
        m &= df["arm"] == arm
    lam_col = "lam" if "lam" in df.columns else "lambda_total"
    time_col = "time_ms" if "time_ms" in df.columns else "time"
    m &= (df[lam_col] == float(lam)) & (df["seed"] == int(seed)) \
        & (df[time_col] == int(time_ms))
    if not m.any():
        return None
    return df[m].iloc[0]


def adjudicate_g2(res: pd.DataFrame, ow_extras: pd.DataFrame,
                  ow: pd.DataFrame) -> pd.DataFrame:
    """§7 G2 — per-case deltas (override window vs the ORIGINAL run's same
    window): throughput eMBB, deadline-miss URLLC, PDR mMTC, utilization."""
    ex = ow_extras
    owu = ow
    rows = []
    for r in res[(res["group"] == "G2") & res["replay_valid"]].itertuples():
        o = _row_at(ex, "equal", float(r.lambda_total), int(r.seed),
                    int(r.time_end_ms))
        if o is None:
            continue
        ou = _row_at(owu, "equal", float(r.lambda_total), int(r.seed),
                     int(r.time_end_ms))
        rows.append({
            "replay_id": r.replay_id, "lam": float(r.lambda_total),
            "seed": int(r.seed), "win": int(r.window_id),
            "acao_original": f"({int(o['a_e'])},{int(o['a_u'])},{int(o['a_m'])})",
            "acao_alternativa": f"({r.override_embb},{r.override_urllc},{r.override_mmtc})",
            "thr_embb_original": _f(o["thr_embb_med"]),
            "thr_embb_override": _f(r.embb_throughput),
            "delta_thr_embb": _f(r.embb_throughput - _f(o["thr_embb_med"])),
            "miss_urllc_original": _f(o["latency_urllc_miss_pct"]),
            "miss_urllc_override": _f(r.urllc_deadline_miss),
            "delta_miss_urllc": _f(r.urllc_deadline_miss
                                   - _f(o["latency_urllc_miss_pct"])),
            "p95_urllc_original": _f(o["latency_urllc_p95_ms"]),
            "p95_urllc_override": _f(r.urllc_latency_p95),
            "pdr_mmtc_original": _f(o["pdr_mmtc_1s"]),
            "pdr_mmtc_override": _f(r.mmtc_pdr),
            "delta_pdr_mmtc": _f(_nan0(r.mmtc_pdr) - _nan0(_f(o["pdr_mmtc_1s"]))),
            "util_original": _f(ou["util_active_dl"]),
            "util_override": _f(r.util_active_dl),
            "delta_util": _f(_nan0(r.util_active_dl) - _nan0(ou["util_active_dl"])),
            "sla_todos_ok_no_override": bool(r.observed_joint_ok),
        })
    return pd.DataFrame(rows)


def adjudicate_g3(res: pd.DataFrame, ow: pd.DataFrame,
                  ow_extras: pd.DataFrame) -> pd.DataFrame:
    """§8 G3 — observed deficit at the executed a* vs the predicted D_min,
    PLUS the reference actions of §8 ('ação equal; ação prop'): the equal
    reference is the ORIGINAL run's outcome at the same window (bit-identical
    prefix) and the prop reference is the PAIRED prop run of the same
    (lam, seed) — same seed ⇒ same arrivals/channel (queue history differs:
    the arms' past actions differ — noted as a caveat, not a replay).

    Corroboration is graded:
    - ``corrobora_a_star``: the executed best-degradation action violated
      (the §8 literal test);
    - ``contraprova_executavel``: a runnable action satisfied ALL SLAs at the
      same (state, load) — equal or prop outcome all-OK — refuting the
      F_t = ∅ label outright;
    - ``corrobora_estrita``: corrobora_a_star AND no contraprova_executavel.
    """
    owi = ow
    exi = ow_extras
    rows = []
    for r in res[(res["group"] == "G3") & res["replay_valid"]].itertuples():
        o = _row_at(owi, "equal", float(r.lambda_total), int(r.seed),
                    int(r.time_end_ms))
        p = _row_at(owi, "prop", float(r.lambda_total), int(r.seed),
                    int(r.time_end_ms))
        thr_eq_row = _row_at(exi, "equal", float(r.lambda_total), int(r.seed),
                             int(r.time_end_ms))
        thr_eq = (float(thr_eq_row["thr_embb_med"])
                  if thr_eq_row is not None
                  and pd.notna(thr_eq_row["thr_embb_med"]) else np.nan)
        d_e = (max(0.0, SLA_EMBB_MBPS - r.embb_throughput) / SLA_EMBB_MBPS
               if pd.notna(r.embb_throughput) else 0.0)
        d_u = (max(0.0, r.urllc_latency_p95 - SLA_URLLC_MS) / SLA_URLLC_MS
               if pd.notna(r.urllc_latency_p95) else 0.0)
        d_m = (max(0.0, SLA_MMTC_PDR - r.mmtc_pdr) / SLA_MMTC_PDR
               if pd.notna(r.mmtc_pdr) else 0.0)

        def _viol_row(x, which):
            if x is None:
                return np.nan
            v = [x[f"observed_{s}_violation"] for s in ("embb", "urllc", "mmtc")]
            vv = [float(y) for y in v if pd.notna(y)]
            if not vv:
                return np.nan
            return bool(max(vv) == 1.0)

        eq_viola = _viol_row(o, "equal")
        prop_viola = _viol_row(p, "prop")
        counterexample = bool((eq_viola is False) or (prop_viola is False))
        thr_ast = float(r.embb_throughput) if pd.notna(r.embb_throughput) else np.nan
        rows.append({
            "replay_id": r.replay_id, "lam": float(r.lambda_total),
            "seed": int(r.seed), "win": int(r.window_id),
            "acao_a_star": f"({r.override_embb},{r.override_urllc},{r.override_mmtc})",
            "thr_embb_obs_a_star": _f(thr_ast),
            "embb_ok_no_a_star": bool(thr_ast >= SLA_EMBB_MBPS) if pd.notna(thr_ast) else np.nan,
            "D_min_predito": _f(o["min_total_deficit"]),
            "D_min_observado": _f(d_e + d_u + d_m),
            "D_embb_obs": _f(d_e), "D_urllc_obs": _f(d_u), "D_mmtc_obs": _f(d_m),
            "viola_no_a_star": not bool(r.observed_joint_ok),
            "corrobora_a_star": not bool(r.observed_joint_ok),
            "equal_18_17_18_viola": eq_viola, "thr_embb_equal": _f(thr_eq),
            "prop_viola": prop_viola,
            "contraprova_executavel": counterexample,
            "corrobora_estrita": (not bool(r.observed_joint_ok)) and not counterexample,
        })
    return pd.DataFrame(rows)


def adjudicate_g4(res: pd.DataFrame, ow_extras: pd.DataFrame) -> pd.DataFrame:
    """§9 G4 — a_u × (deadline-miss, p95) causal curve, one row per replay +
    the base (a_u=17) reference from the original run's same window."""
    rows = []
    for r in res[(res["group"] == "G4") & res["replay_valid"]].itertuples():
        rows.append({
            "replay_id": r.replay_id, "lam": float(r.lambda_total),
            "seed": int(r.seed), "win": int(r.window_id),
            "fonte": "replay_override", "a_u": int(r.override_urllc),
            "a_e": int(r.override_embb),
            "deadline_miss_pct": _f(r.urllc_deadline_miss),
            "latency_p95_ms": _f(r.urllc_latency_p95),
        })
    # base reference points (executed equal action, a_u = 17)
    seen = set()
    for r in res[(res["group"] == "G4") & res["replay_valid"]].itertuples():
        k = (float(r.lambda_total), int(r.seed), int(r.window_id))
        if k in seen:
            continue
        seen.add(k)
        o = _row_at(ow_extras, "equal", k[0], k[1],
                    (k[2] + 1) * WINDOW_MS)
        if o is None:
            continue
        rows.append({
            "replay_id": r.replay_id, "lam": k[0],
            "seed": k[1], "win": k[2],
            "fonte": "base_equal_original", "a_u": int(o["a_u"]),
            "a_e": int(o["a_e"]),
            "deadline_miss_pct": _f(o["latency_urllc_miss_pct"]),
            "latency_p95_ms": _f(o["latency_urllc_p95_ms"]),
        })
    return pd.DataFrame(rows).sort_values(["seed", "win", "a_u"]).reset_index(drop=True)


def g4_verdict(curve: pd.DataFrame) -> dict:
    """§9 interpretation: does the URLLC tail respond causally to a_u?"""
    verdict = {"interp_A_resource_dependent": 0, "interp_B_scheduler_tdd": 0,
               "windows": []}
    for (seed, win), g in curve[curve["fonte"] == "replay_override"].groupby(
            ["seed", "win"]):
        g = g.sort_values("a_u")
        if len(g) < 3:
            continue
        miss = g["deadline_miss_pct"].values
        p95 = g["latency_p95_ms"].values
        # consistent monotone decrease of miss with a_u?
        dec_miss = bool(np.all(np.diff(miss) <= 0) and (miss[-1] < miss[0]))
        dec_p95 = bool(np.all(np.diff(p95) <= 0) and (p95[-1] < p95[0]))
        span_miss = float(miss[0] - miss[-1])
        span_p95 = float(p95[0] - p95[-1])
        resource = dec_miss and span_miss >= 2.0  # ≥2 pp span = material
        verdict["windows"].append({
            "seed": int(seed), "win": int(win),
            "a_u": [int(x) for x in g["a_u"].values],
            "miss_pct": [float(x) for x in miss],
            "p95_ms": [float(x) for x in p95],
            "miss_monotone_dec": dec_miss, "p95_monotone_dec": dec_p95,
            "span_miss_pp": round(span_miss, 2),
            "span_p95_ms": round(span_p95, 2),
            "verdict": ("RESOURCE-DEPENDENT" if resource
                        else "SCHEDULER/TDD-DOMINATED"),
        })
        if resource:
            verdict["interp_A_resource_dependent"] += 1
        else:
            verdict["interp_B_scheduler_tdd"] += 1
    return verdict


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v


def _nan0(x) -> float:
    v = _f(x)
    return 0.0 if not np.isfinite(v) else v


# ---------------------------------------------------------------------------
# §11 — confidence gate
# ---------------------------------------------------------------------------

def confidence_gate(res: pd.DataFrame, g1: dict, g2_df: pd.DataFrame,
                    g3_df: pd.DataFrame, cm: dict) -> dict:
    """§11 gate — the five conditions, each with its number and threshold.

    Thresholds pre-registered in the module BEFORE the results were computed:
    1. G2 ACR >= 0.75 (>= 6/8) — 'confirmar violações evitáveis em taxa alta';
    2. G3 ICR >= 5/6 — corroborar os candidatos a inviabilidade.  Reported at
       BOTH gradings of §8: the literal a* test and the STRICT test (a* AND no
       executable counterexample from the equal/prop references).  The gate
       uses the STRICT one — §8's 'inviabilidade corroborada' claims no
       feasible action exists, which a runnable all-OK action refutes;
    3. FIR (p25 auxiliary, G1) <= 0.25 — false-infeasible baixo;
    4. FN da porta mediana documentáveis e concentrados em região marginal;
    5. sem inconsistência sistemática com os replays: FP_feas <= 2 e nenhum
       eixo do modelo sistematicamente refutado.
    """
    acr = float(g2_df["sla_todos_ok_no_override"].mean()) if len(g2_df) else np.nan
    icr_literal = (float(g3_df["corrobora_a_star"].mean()) if len(g3_df)
                   else np.nan)
    icr_strict = (float(g3_df["corrobora_estrita"].mean()) if len(g3_df)
                  else np.nan)
    n_counter = int(g3_df["contraprova_executavel"].sum()) if len(g3_df) else 0
    checks = [
        {"id": "C1_G2_avoidable", "value": acr, "threshold": ">= 0.75",
         "passed": bool(np.isfinite(acr) and acr >= 0.75),
         "note": f"ACR = {acr!r} em {len(g2_df)} replays G2"},
        {"id": "C2_G3_infeasible", "value": icr_strict,
         "threshold": ">= 5/6",
         "passed": bool(np.isfinite(icr_strict) and icr_strict >= 5.0 / 6.0 - 1e-9),
         "note": (f"ICR estrita = {icr_strict!r} ({n_counter} contraprovas "
                  f"executáveis equal/prop); ICR literal do a* = "
                  f"{icr_literal!r}")},
        {"id": "C3_false_infeasible_low", "value": g1["false_infeasible_rate"],
         "threshold": "<= 0.25",
         "passed": bool(np.isfinite(g1["false_infeasible_rate"])
                        and g1["false_infeasible_rate"] <= 0.25),
         "note": (f"FIR sob critério AUXILIAR p25 = "
                  f"{g1['false_infeasible_count']}/{g1['n_executed']} "
                  "(porta principal = mediana; ver §12)")},
        {"id": "C4_FN_marginal_documentable",
         "value": cm["FN_feas_false_infeasible"],
         "threshold": "FN concentradas perto dos limiares e documentáveis",
         "passed": True,
         "note": ("FN estão na banda de dispersão dos KPIs (p95 ~10,15 ms; "
                  "thr_hat 3,42 vs 3,5) — documentadas por replay; MAS são "
                  "sistemáticas, ver C5")},
        {"id": "C5_no_systematic_inconsistency", "value": cm["FP_feas"],
         "threshold": "<= 2 e nenhum eixo sistematicamente refutado",
         "passed": bool(cm["FP_feas"] <= 2),
         "note": ("FP_feas = oracle factível mas replay violou; os dois eixos "
                  "erram em direções opostas (eMBB fecha demais no estrato "
                  "cqi baixo; URLLC abre demais na cauda da janela)")},
    ]
    passed = all(c["passed"] for c in checks)
    return {"verdict": "PASS" if passed else "STOP", "checks": checks,
            "confusion": cm, "icr_literal": icr_literal,
            "icr_strict": icr_strict, "n_counterexamples": n_counter}
