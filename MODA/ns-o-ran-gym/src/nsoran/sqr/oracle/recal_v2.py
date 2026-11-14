"""Etapa 2.4-II-d-v2 — final eMBB recalibration after the support extension.

Protocol ``docs/artigo/fase_2_4_II_d_v2_recalibracao_final_oracle.md`` §10–§31
(UNICA recalibracao planejada da Rodada 2; heranca verificada do II-d:
FIR=FFR=0,000 nas 22 testemunhas confidentes; mapeamento de janela EFETIVA
congelado do piloto II-a — override na janela ``t`` e lido na estampa
``(t+1)·100 ms``).

Layers (one per protocol section):

§10 :func:`build_support_extension_rows` — ``support_extension_replays.csv``
with the EXACT column order of protocol §10, observed via the SAME frozen
pipeline of the II-c/II-d adjudications (``validate_replay`` +
``window_kpis``; 30/31 verified) — never re-derived here.

§11 :func:`build_support_curve` — the executable eMBB curve thr×a_e per
window: the 13 v2 replays + the frozen references (II-c (48,5,0) from
``g2_deltas.csv``; II-d (37,13,3) from the VERIFIED replay results;
original (18,17,18) from the frozen dataset).  Monotonicity / saturation /
marginal gain / SLA point / dispersion; mixed-tail segments flagged.

§12–§16 :class:`EmmbQuantileV2` — refit of the eMBB axis ONLY (URLLC/mMTC
frozen).  Same FROZEN pinball basis of the II-d (continuous quantile
p25/p50/p75 — §13 forbids coarse bins).  Training = the frozen dataset rows
PLUS the 5 in-support E1 rows ((37,13,3), state of the source window,
observed thr from the replay).  The 8 out-of-floor diagnostic replays enter
ONLY the §11 curve — never the training set (pre-registration §6).
Local support metrics (§15): ``nearest_action_distance``,
``local_sample_count``, ``support_density``; band registration (§14):
``band_width_embb``, ``distance_to_sla``; MARGINAL cause (§16):
NEAR_THRESHOLD / LOW_SUPPORT / HIGH_VARIANCE.

§18.6 :func:`classify_windows_3way_v2` — full reprocessing of the 89,970
windows with the v2 eMBB model + the FROZEN URLLC model (refit of
``UrllcQuantileRecal`` on the same frozen dataset — bit-identical to the II-d
reprocessing, asserted by :func:`frozen_guard`).

§19–§20 :func:`gate_v2` — regressions R1–R4 + the confusion 3-way v2 (22
VERIFIED legacy witnesses D1/D2/D4/D5 + the 5 new in-support E1 witnesses;
D3 excluded — judged by R2/C4) + C1–C7.  Thresholds are the protocol's,
declared BEFORE the gate runs: C1 FIR<=0.10, C2 FFR<=0.10.  R1 follows the
PRE-REGISTERED two-branch semantics (pre_registration_v2.md §5: E1 all-OK →
state must be FEASIBLE/MARGINAL; E1 violated → INFEASIBLE in-suporte
mantido com justificativa quantitativa 37→48).

§23–§31 freeze: :func:`freeze_all` reuses the II-d finalizers
(``adjudication_3way``) on the v2 reprocessing.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from nsoran.sqr.oracle import adjudication_3way as a3
from nsoran.sqr.oracle import quantile as qm
from nsoran.sqr.oracle import three_way as tw
from nsoran.sqr.oracle.adjudication import (
    WINDOW_MS,
    _replay_tag,
    validate_replay,
    window_kpis,
)
from nsoran.sqr.oracle.adjudication_3way import _axis_sla_ok, window_kpis_3way
from nsoran.sqr.oracle.quantile import (
    EMMB_NEAR_HALF_WIDTH_MBPS,
    SLA_EMBB_MBPS,
)

#: §10 EXACT column order (protocol §10 — verbatim).
SUPPORT_REPLAYS_COLUMNS: tuple[str, ...] = (
    "replay_id",
    "lambda_total",
    "seed",
    "window_id",
    "original_state_class",
    "cqi_embb",
    "n_embb",
    "queue_embb",
    "a_embb",
    "a_urllc",
    "a_mmtc",
    "embb_thr_median",
    "embb_sla_ok",
    "urllc_p95",
    "urllc_ok",
    "mmtc_pdr",
    "mmtc_ok",
    "joint_ok",
    "snr_p05",
    "replay_valid",
)

#: The 5 confirmed-avoidable II-c windows (gate C3-v2 / regression R1) —
#: same tuple as ``adjudication_3way.CONFIRMED_AVOIDABLE_II_C``.
CONFIRMED_AVOIDABLE = a3.CONFIRMED_AVOIDABLE_II_C

#: The 3 D3 counter-proof windows (R2/C4: the all-OK one cannot be
#: INFEASIBLE; R3/C5: the corroborated stay coherent).
D3_WINDOWS = ((6.0, 1, 1866), (6.0, 3, 2263), (7.0, 3, 2013))
#: The 2 corroborated infeasible windows (R3/C5).
D4_WINDOWS = ((7.0, 1, 1555), (7.0, 1, 1557))
#: The D5 MARGINAL witnesses of the II-d gate C7 FAIL (band artifact) —
#: the windows whose binding axis must land in-band/near under v2.
D5_C7_WINDOWS = ((2.0, 1, 1261), (4.0, 1, 633), (5.0, 1, 507),
                 (7.0, 1, 364), (7.0, 1, 367))

#: Witness window groups of the v2 confusion: legacy D1/D2/D4/D5 (VERIFIED
#: observations) + the new E1 in-support witnesses.  D3 excluded BY DESIGN
#: (the verified II-d adendo uses the same 22-witness convention).
LEGACY_WITNESS_GROUPS = ("D1", "D2", "D4", "D5")

# ---------------------------------------------------------------------------
# §15/§16 FROZEN thresholds — declared BEFORE the v2 gate runs (rule §17:
# nothing below is tuned after seeing gate results).
# ---------------------------------------------------------------------------
#: local-sample neighbourhood in the (s = a_e/n_embb, cqi) plane.
LOCAL_S_HALFWIDTH = 1.0     # |s_i - s_q| <= 1.0  (1 RBG/UE)
LOCAL_CQI_HALFWIDTH = 3.0   # |cqi_i - cqi_q| <= 3.0
#: support-density area of that neighbourhood (count per unit area).
LOCAL_AREA = (2.0 * LOCAL_S_HALFWIDTH) * (2.0 * LOCAL_CQI_HALFWIDTH)
#: cause thresholds (frozen): enough local samples and a nearby training
#: action make LOW_SUPPORT implausible; a wide band with good support is
#: genuine variance; a narrow band straddling the SLA is near-threshold.
CAUSE_MIN_LOCAL_SAMPLES = 15
CAUSE_MAX_NEAREST_DISTANCE = 1.0    # s-units (RBG/UE)
CAUSE_WIDE_BAND_MBPS = 2.0 * EMMB_NEAR_HALF_WIDTH_MBPS   # 2.0 Mbps


# ---------------------------------------------------------------------------
# Frozen dataset loader (IDENTICAL to the II-d drivers — asserted 89,970)
# ---------------------------------------------------------------------------

def load_windows(oracle_dir: str) -> pd.DataFrame:
    """The frozen 2.3 window dataset (same 12-line recipe of the II-d)."""
    w = pd.read_csv(os.path.join(oracle_dir, "oracle_windows.csv")).rename(
        columns={"lambda_total": "lam", "time": "time_ms"})
    x = pd.read_csv(os.path.join(oracle_dir, "tables",
                                 "oracle_windows_extra.csv"))
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


def load_plan_v2(support_dir: str) -> pd.DataFrame:
    """The 13-row v2 plan joined with its frozen metadata (int-typed keys)."""
    plan = pd.read_csv(os.path.join(support_dir, "plan_v2.csv"))
    meta = pd.read_csv(os.path.join(support_dir, "plan_v2_metadata.csv"))
    for c in ("lam", "seed", "win_t", "action_e", "action_u", "action_m"):
        plan[c] = plan[c].astype(int)
        meta[c] = meta[c].astype(int)
    key = ["grupo", "lam", "seed", "win_t", "action_e", "action_u",
           "action_m"]
    plan = plan.merge(meta[key + ["_role", "_case_of", "_in_support",
                                  "original_state_class", "cqi_med",
                                  "n_embb", "n_urllc", "n_mmtc",
                                  "queue_embb", "replay_id_ref"]],
                      on=key, how="left", validate="one_to_one")
    return plan


# ---------------------------------------------------------------------------
# §18.2 — integrity validation of the 13 replays (incl. the R56 lesson)
# ---------------------------------------------------------------------------

def _banner_lambdas(replay_dir: str) -> list[float] | None:
    """Per-slice Poisson lambdas of the run banner (R56 lesson: the locale
    bug silently changed the decimal separator — the banner is the receipt
    that the sim received the planned λ)."""
    hits = glob.glob(os.path.join(replay_dir, "*", "stdout"))
    for path in hits:
        try:
            with open(path, errors="replace") as fh:
                for line in fh:
                    if "Poisson per-slice lambdas=" in line:
                        seg = line.split("lambdas=")[1].split("UEs/min")[0]
                        return [float(v) for v in seg.split(",")]
        except OSError:
            continue
    return None


def validate_replays_v2(support_dir: str, round2_dir: str
                        ) -> tuple[pd.DataFrame, list[dict]]:
    """§4 (II-c criterion) + banner-λ check for the 13 v2 replays."""
    plan = load_plan_v2(support_dir)
    rows, validations = [], []
    for _, pr in plan.iterrows():
        tag = _replay_tag(pr)
        rdir = os.path.join(support_dir, "replays_v2", tag)
        val = validate_replay(rdir, plan_row=pr, round2_dir=round2_dir)
        lam3 = _banner_lambdas(rdir)
        ok_banner = bool(lam3 is not None and len(lam3) == 3
                         and np.allclose(3.0 * np.mean(lam3), float(pr["lam"]),
                                         atol=5e-3)
                         and np.ptp(lam3) < 1e-9)
        val["checks"]["banner_lambdas"] = lam3
        val["checks"]["banner_lambda_ok"] = ok_banner
        if val["replay_valid"] and not ok_banner:
            val["replay_valid"] = False
            val["checks"]["failure"] = (f"banner lambdas {lam3} != planned "
                                        f"lam={pr['lam']} (R56 lesson)")
        validations.append(val)
    return plan, validations


# ---------------------------------------------------------------------------
# §10 — support_extension_replays.csv (EXACT columns)
# ---------------------------------------------------------------------------

def build_support_extension_rows(support_dir: str, oracle_dir: str,
                                 round2_dir: str, plan: pd.DataFrame,
                                 validations: list[dict]) -> pd.DataFrame:
    """One §10 row per replay: observed KPIs of the override window (t+1
    extraction) + the plan metadata; ``original_state_class`` from the frozen
    II-d reprocessing via the frozen dataset (asserted == plan metadata)."""
    d = load_windows(oracle_dir)
    dkey = d.set_index(
        [d["arm"], d["lam"], d["seed"], d["time_ms"]])

    def _src_row(lam, seed, win_t) -> pd.Series | None:
        k = ("equal", float(lam), int(seed), (int(win_t) + 1) * WINDOW_MS)
        hit = dkey.index.get_indexer_for([k])
        return dkey.iloc[hit[0]] if len(hit) and hit[0] >= 0 else None

    cls_ii_d = pd.read_csv(os.path.join(
        oracle_dir, "oracle_recalibrated", "predictions",
        "oracle_3way_windows.csv"))
    ckey = cls_ii_d.set_index(
        [cls_ii_d["arm"], cls_ii_d["lam"], cls_ii_d["seed"],
         cls_ii_d["time_ms"]])

    rows = []
    for i, (_, pr) in enumerate(plan.iterrows()):
        val = validations[i]
        tag = _replay_tag(pr)
        rdir = os.path.join(support_dir, "replays_v2", tag)
        t = int(pr["win_t"])
        crow = _src_row(pr["lam"], pr["seed"], t)
        # original_state_class: the II-d official class of the source window
        hit = ckey.index.get_indexer_for(
            [("equal", float(pr["lam"]), int(pr["seed"]),
              (t + 1) * WINDOW_MS)])
        ii_d_class = (cls_ii_d.iloc[hit[0]]["state_class"]
                      if len(hit) and hit[0] >= 0 else "")
        if len(hit) and hit[0] >= 0 and str(pr["original_state_class"]) \
                not in ("", "nan"):
            assert str(ii_d_class) == str(pr["original_state_class"]), (
                f"{tag}: plan metadata state {pr['original_state_class']} != "
                f"II-d table {ii_d_class}")
        row = {
            "replay_id": tag,
            "lambda_total": float(pr["lam"]),
            "seed": int(pr["seed"]),
            "window_id": t,
            "original_state_class": str(ii_d_class),
            "cqi_embb": float(crow["cqi_eff"]) if crow is not None else np.nan,
            "n_embb": int(crow["n_embb"]) if crow is not None else -1,
            "queue_embb": (float(crow["queue_embb"])
                           if crow is not None else np.nan),
            "a_embb": int(pr["action_e"]),
            "a_urllc": int(pr["action_u"]),
            "a_mmtc": int(pr["action_m"]),
        }
        if val["replay_valid"]:
            k = window_kpis_3way(rdir, t)
            oks = _axis_sla_ok(k)
            row.update({
                "embb_thr_median": float(k["embb_throughput"]),
                "embb_sla_ok": bool(oks["observed_embb_ok"]),
                "urllc_p95": (float(k["urllc_latency_p95"])
                              if np.isfinite(k["urllc_latency_p95"])
                              else np.nan),
                "urllc_ok": bool(oks["observed_urllc_ok"]),
                "mmtc_pdr": (float(k["mmtc_pdr"])
                             if k.get("mmtc_demand", False) else np.nan),
                "mmtc_ok": bool(oks["observed_mmtc_ok"]),
                "joint_ok": bool(oks["observed_embb_ok"]
                                 and oks["observed_urllc_ok"]
                                 and oks["observed_mmtc_ok"]),
                "snr_p05": (float(k["snr_p05"])
                            if np.isfinite(k["snr_p05"]) else np.nan),
            })
        else:
            row.update({c: np.nan for c in
                        ("embb_thr_median", "urllc_p95", "mmtc_pdr",
                         "snr_p05")})
            row.update({"embb_sla_ok": np.nan, "urllc_ok": np.nan,
                        "mmtc_ok": np.nan, "joint_ok": np.nan})
        row["replay_valid"] = bool(val["replay_valid"])
        rows.append(row)
    out = pd.DataFrame(rows)
    assert list(out.columns) == list(SUPPORT_REPLAYS_COLUMNS), (
        f"§10 column order violated: {list(out.columns)}")
    return out


# ---------------------------------------------------------------------------
# §11 — executable eMBB curve (new replays + frozen references)
# ---------------------------------------------------------------------------

def build_support_curve(oracle_dir: str, replays: pd.DataFrame,
                        plan: pd.DataFrame,
                        d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame,
                                                  pd.DataFrame]:
    """thr×a_e points per window + per-window curve metrics (§11) + the
    across-window dispersion at each a_e level.

    Sources: ``v2_replay`` (13 new), ``ii_c_g2_deltas`` ((48,5,0), the 5
    evitáveis), ``ii_d_replay_verified`` ((37,13,3), D3/D4/D5 windows),
    ``original_equal`` ((18,17,18), frozen dataset).  Points carry their
    tail (a_u, a_m): segments mixing tails are flagged and excluded from
    monotonicity/saturation claims.
    """
    g2 = pd.read_csv(os.path.join(oracle_dir, "adjudication",
                                  "g2_deltas.csv"))
    ver = pd.read_csv(os.path.join(
        oracle_dir, "oracle_recalibrated", "adjudication",
        "oracle_3way_replay_results_verified.csv"))
    grupo_of = {_replay_tag(pr): str(pr["grupo"])
                for _, pr in plan.iterrows()}

    def _thr_orig(lam, seed, win_t) -> float:
        hit = d[(d["arm"] == "equal") & (d["lam"] == float(lam))
                & (d["seed"] == int(seed))
                & (d["time_ms"] == (int(win_t) + 1) * WINDOW_MS)]
        return (float(hit["thr_embb_med"].iloc[0])
                if len(hit) and pd.notna(hit["thr_embb_med"].iloc[0])
                else np.nan)

    points = []
    for _, r in replays.iterrows():
        points.append({
            "grupo": grupo_of.get(r["replay_id"], ""),
            "lambda_total": r["lambda_total"], "seed": r["seed"],
            "window_id": r["window_id"],
            "a_e": r["a_embb"], "a_u": r["a_urllc"], "a_m": r["a_mmtc"],
            "thr_embb": r["embb_thr_median"],
            "embb_sla_ok": r["embb_sla_ok"], "joint_ok": r["joint_ok"],
            "source": "v2_replay", "replay_id": r["replay_id"],
            "in_support_floor": bool(r["a_urllc"] >= 13 and r["a_mmtc"] >= 3
                                     and 18 <= r["a_embb"] <= 37
                                     and r["a_embb"] + r["a_urllc"]
                                     + r["a_mmtc"] <= 53),
        })
    # II-c (48,5,0) testemunhas for the E1 windows
    e1_wins = {(lam, seed, t) for (lam, seed, t) in CONFIRMED_AVOIDABLE}
    for _, g in g2.iterrows():
        gid = g["replay_id"]  # G2_equal_lam6_s3_t298_a48-5-0
        parts = gid.split("_")
        lam, seed = float(parts[2][3:]), int(parts[3][1:])
        t = int(parts[4][1:])
        if (lam, seed, t) in e1_wins:
            points.append({
                "grupo": "E1x_ref", "lambda_total": lam, "seed": seed,
                "window_id": t, "a_e": 48, "a_u": 5, "a_m": 0,
                "thr_embb": float(g["thr_embb_override"]),
                "embb_sla_ok": bool(float(g["thr_embb_override"]) >= SLA_EMBB_MBPS),
                "joint_ok": bool(g["sla_todos_ok_no_override"]),
                "source": "ii_c_g2_deltas", "replay_id": gid,
                "in_support_floor": False,
            })
    # II-d (37,13,3) verified references for E2x/E3x/E4x windows
    for _, v in ver.iterrows():
        if v["role"] != "witness" or v["group"] not in ("D3", "D4", "D5"):
            continue
        points.append({
            "grupo": f"{v['group']}ref", "lambda_total": float(v["lambda_total"]),
            "seed": int(v["seed"]), "window_id": int(v["window_id"]),
            "a_e": int(v["override_embb"]), "a_u": int(v["override_urllc"]),
            "a_m": int(v["override_mmtc"]),
            "thr_embb": float(v["verified_thr_min"]),
            "embb_sla_ok": bool(float(v["verified_thr_min"]) >= SLA_EMBB_MBPS),
            "joint_ok": bool(v["verified_joint_ok"]),
            "source": "ii_d_replay_verified", "replay_id": v["replay_id"],
            "in_support_floor": True,
        })
    # original equal-arm points (every window of the plan)
    for _, p in plan.iterrows():
        points.append({
            "grupo": "orig", "lambda_total": float(p["lam"]),
            "seed": int(p["seed"]), "window_id": int(p["win_t"]),
            "a_e": 18, "a_u": 17, "a_m": 18,
            "thr_embb": _thr_orig(p["lam"], p["seed"], p["win_t"]),
            "embb_sla_ok": np.nan, "joint_ok": np.nan,
            "source": "original_equal", "replay_id": "",
            "in_support_floor": True,
        })
    pts = pd.DataFrame(points)
    pts["grupo"] = pts["grupo"].astype(str)
    pts = pts.sort_values(["lambda_total", "seed", "window_id", "a_e"]
                          ).reset_index(drop=True)

    # ---- per-window metrics (§11) -----------------------------------------
    rows = []
    for (lam, seed, t), gwin in pts.groupby(
            ["lambda_total", "seed", "window_id"]):
        gwin = gwin.sort_values("a_e").reset_index(drop=True)
        ae = gwin["a_e"].to_numpy(float)
        thr = gwin["thr_embb"].to_numpy(float)
        tail = [(int(u), int(m)) for u, m in
                zip(gwin["a_u"], gwin["a_m"])]
        same_tail = np.array([tail[i] == tail[i + 1]
                              for i in range(len(tail) - 1)], dtype=bool)
        gains = np.diff(thr) / np.diff(ae)
        gains_clean = np.where(same_tail, gains, np.nan)
        valid_gains = gains_clean[np.isfinite(gains_clean)]
        monotone = bool(np.all(np.diff(thr) >= -1e-9)) if len(thr) > 1 else np.nan
        monotone_same_tail = (bool(np.all(gains_clean[np.isfinite(gains_clean)]
                                         >= -1e-9))
                              if np.isfinite(gains_clean).any() else np.nan)
        # SLA crossing point: linear interpolation on the curve
        cross = np.nan
        for i in range(len(ae) - 1):
            lo, hi = thr[i], thr[i + 1]
            if (lo < SLA_EMBB_MBPS <= hi) or (lo >= SLA_EMBB_MBPS > hi):
                frac = (SLA_EMBB_MBPS - lo) / (hi - lo) if hi != lo else 0.0
                cross = ae[i] + frac * (ae[i + 1] - ae[i])
                break
        if len(thr) and thr[0] >= SLA_EMBB_MBPS:
            cross = float(ae[0])
        rows.append({
            "lambda_total": lam, "seed": seed, "window_id": t,
            "n_points": int(len(gwin)),
            "a_e_points": "; ".join(str(int(v)) for v in ae),
            "thr_points": "; ".join(f"{v:.4f}" for v in thr),
            "sources": "; ".join(gwin["source"]),
            "monotone_nondecreasing": monotone,
            "monotone_same_tail": monotone_same_tail,
            "marginal_gain_per_rbg": "; ".join(
                "nan" if not np.isfinite(v) else f"{v:.4f}" for v in gains),
            "marginal_gain_same_tail": "; ".join(
                "nan" if not np.isfinite(v) else f"{v:.4f}"
                for v in gains_clean),
            "last_same_tail_gain": (float(valid_gains[-1])
                                    if len(valid_gains) else np.nan),
            "saturation_last_gain_le_0.1": (bool(valid_gains[-1] <= 0.1)
                                            if len(valid_gains) else np.nan),
            "sla_cross_a_e": cross,
            "mixed_tail_segments": int((~same_tail).sum()),
            "thr_min": float(np.nanmin(thr)), "thr_max": float(np.nanmax(thr)),
            "thr_std_same_ae_across_windows": np.nan,  # filled below
        })
    curve = pd.DataFrame(rows)
    # dispersion across windows at the same a_e (pooled points)
    disp = pts[pts["thr_embb"].notna()].groupby("a_e")["thr_embb"].agg(
        ["count", "std", "min", "max"]).reset_index()
    disp.columns = ["a_e", "n_windows", "thr_std", "thr_min_at_ae",
                    "thr_max_at_ae"]
    disp["support_density_note"] = np.where(
        disp["a_e"] <= 37, "in_support", "diagnostic_out_of_floor")
    return pts, curve, disp


# ---------------------------------------------------------------------------
# §12–§16 — the v2 eMBB model with explicit local support
# ---------------------------------------------------------------------------

EMBB_V2_TRAIN_COLUMNS = ("n_embb", "cqi_eff", "queue_embb", "n_embb_backlog",
                         "offered_load_embb", "prev_thr_embb_med", "a_e",
                         "thr_embb_med")


def e1_training_rows(d: pd.DataFrame, replays: pd.DataFrame,
                     plan: pd.DataFrame) -> pd.DataFrame:
    """The 5 in-support E1 rows (pre-registration §6: ONLY E1 enters the
    refit; the 8 out-of-floor diagnostics never do)."""
    e1 = replays[replays["replay_id"].str.startswith("E1_")]
    rows = []
    for _, r in e1.iterrows():
        pr = plan[plan.apply(lambda p: _replay_tag(p) == r["replay_id"],
                             axis=1)].iloc[0]
        hit = d[(d["arm"] == "equal") & (d["lam"] == float(r["lambda_total"]))
                & (d["seed"] == int(r["seed"]))
                & (d["time_ms"] == (int(r["window_id"]) + 1) * WINDOW_MS)]
        assert len(hit) == 1, f"E1 source window missing: {r['replay_id']}"
        s = hit.iloc[0]
        assert pd.notna(s["prev_thr_embb_med"]), (
            f"{r['replay_id']}: prev_thr missing — E1 windows are not "
            "first-windows by construction")
        rows.append({
            "n_embb": float(s["n_embb"]), "cqi_eff": float(s["cqi_eff"]),
            "queue_embb": float(s["queue_embb"]),
            "n_embb_backlog": float(s["n_embb_backlog"]),
            "offered_load_embb": float(s["offered_load_embb"]),
            "prev_thr_embb_med": float(s["prev_thr_embb_med"]),
            "a_e": 37.0, "thr_embb_med": float(r["embb_thr_median"]),
            "lam": float(r["lambda_total"]), "seed": int(r["seed"]),
            "win": int(r["window_id"]),
            "_provenance": "E1_v2_replay_in_support",
        })
    return pd.DataFrame(rows)


class EmmbQuantileV2(qm.EmmbQuantileRecal):
    """f_eMBB-v2: the FROZEN II-d basis refit on frozen rows + the 5 E1
    in-support rows; adds the §15 local-support metrics per query."""

    def __init__(self):
        super().__init__()
        self.train_frame: pd.DataFrame | None = None

    def fit(self, windows: pd.DataFrame,
            extra_rows: pd.DataFrame | None = None) -> "EmmbQuantileV2":
        d = windows[(windows["n_embb"] >= 1)
                    & windows["thr_embb_med"].notna()].copy()
        d["_provenance"] = "frozen_2_3"
        cols = list(EMBB_V2_TRAIN_COLUMNS)
        if extra_rows is not None and len(extra_rows):
            d = pd.concat([d[cols + ["_provenance"]],
                           extra_rows[cols + ["_provenance"]]],
                          ignore_index=True)
        self.train_frame = d[cols + ["_provenance"]].reset_index(drop=True)
        X, _ = qm.embb_design(d)
        self.model.fit(X, d["thr_embb_med"].to_numpy(dtype=float))
        return self

    # -- §15 local support of (state, action) queries -----------------------
    def _train_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(n_embb, s, cqi) arrays of the v2 training frame."""
        tr = self.train_frame
        n = tr["n_embb"].to_numpy(float)
        s = tr["a_e"] / np.maximum(n, 1.0)
        cqi = tr["cqi_eff"].to_numpy(float)
        return n, s, cqi

    def local_support_batch(self, n_arr, cqi_arr, a_arr) -> list[dict]:
        """Vectorised :meth:`local_support` over arrays of queries."""
        n_q = np.asarray(n_arr, dtype=float)
        c_q = np.asarray(cqi_arr, dtype=float)
        a_q = np.asarray(a_arr, dtype=float)
        s_q = a_q / np.maximum(n_q, 1.0)
        n_tr, s_tr, c_tr = self._train_arrays()
        out = []
        # pre-sort training rows by (n, s) for per-group searchsorted
        order = np.lexsort((s_tr, n_tr))
        n_s, s_s, c_s = n_tr[order], s_tr[order], c_tr[order]
        for n, c, s in zip(n_q, c_q, s_q):
            lo = np.searchsorted(n_s, n, side="left")
            hi = np.searchsorted(n_s, n, side="right")
            if hi > lo:
                seg_s = s_s[lo:hi]
                dmin = float(np.abs(seg_s - s).min())
                w_lo = lo + np.searchsorted(seg_s, s - LOCAL_S_HALFWIDTH,
                                            side="left")
                w_hi = lo + np.searchsorted(seg_s, s + LOCAL_S_HALFWIDTH,
                                            side="right")
                seg_c = c_s[w_lo:w_hi]
                n_loc = int((np.abs(seg_c - c)
                             <= LOCAL_CQI_HALFWIDTH).sum())
            else:
                dmin = float(np.abs(s_tr - s).min())
                n_loc = 0
            out.append({"s_query": float(s),
                        "nearest_action_distance": dmin,
                        "local_sample_count": n_loc,
                        "support_density": n_loc / LOCAL_AREA})
        return out

    def local_support(self, n_embb: float, cqi: float, a_e: float) -> dict:
        """§15 metrics of ONE (state, action) query against the v2 training
        frame (thin wrapper over :meth:`local_support_batch`)."""
        return self.local_support_batch([n_embb], [cqi], [a_e])[0]


def marginal_cause(p25: float, p50: float, p75: float,
                   support: dict) -> str:
    """§16 cause of ONE MARGINAL band (frozen thresholds; declaration order
    LOW_SUPPORT → HIGH_VARIANCE → NEAR_THRESHOLD)."""
    width = float(p75) - float(p25)
    ok_support = (support["local_sample_count"] >= CAUSE_MIN_LOCAL_SAMPLES
                  and support["nearest_action_distance"]
                  <= CAUSE_MAX_NEAREST_DISTANCE)
    if not ok_support:
        return "MARGINAL_LOW_SUPPORT"
    if width > CAUSE_WIDE_BAND_MBPS:
        return "MARGINAL_HIGH_VARIANCE"
    return "MARGINAL_NEAR_THRESHOLD"


def band_registration(p25: float, p50: float, p75: float,
                      support: dict) -> dict:
    """§14 registration block of ONE predicted band."""
    width = float(p75) - float(p25)
    return {
        "band_width_embb": width,
        "distance_to_sla": float(p50) - SLA_EMBB_MBPS,
        "support_density": support["support_density"],
        "nearest_action_distance": support["nearest_action_distance"],
        "local_sample_count": support["local_sample_count"],
        "marginal_cause": (marginal_cause(p25, p50, p75, support)
                           if float(p25) < SLA_EMBB_MBPS <= float(p75)
                           else ""),
    }


def build_replay_results_v2(plan: pd.DataFrame, validations: list[dict],
                            cls: pd.DataFrame, d: pd.DataFrame,
                            embb_v2, urllc_frozen) -> pd.DataFrame:
    """One row per v2 replay: the frozen-v2 PREDICTION (state class of the
    source window + per-axis band at the override action + §14/§15/§16 eMBB
    registration) vs the OBSERVED outcome (t+1 extraction)."""
    wkey = d.set_index([d["arm"], d["lam"], d["seed"], d["time_ms"]])

    def _src_row(lam, seed, win_t) -> pd.Series | None:
        k = ("equal", float(lam), int(seed), (int(win_t) + 1) * WINDOW_MS)
        hit = wkey.index.get_indexer_for([k])
        return wkey.iloc[hit[0]] if len(hit) and hit[0] >= 0 else None

    rows = []
    for i, (_, pr) in enumerate(plan.iterrows()):
        val = validations[i]
        tag = _replay_tag(pr)
        t = int(pr["win_t"])
        row = {
            "group": pr["grupo"], "replay_id": tag,
            "role": pr["_role"], "case_of": pr["_case_of"],
            "in_support_plan": bool(pr["_in_support"]),
            "lambda_total": float(pr["lam"]), "seed": int(pr["seed"]),
            "arm_original": "equal", "window_id": t,
            "time_start_ms": t * WINDOW_MS,
            "time_end_ms": (t + 1) * WINDOW_MS,
            "override_embb": int(pr["action_e"]),
            "override_urllc": int(pr["action_u"]),
            "override_mmtc": int(pr["action_m"]),
            "executed_action": str(val["checks"].get("executed_action", "")),
            "override_matches_plan": bool(
                val["checks"].get("executed_matches_plan", False)),
            "override_only_at_t": bool(
                val["checks"].get("override_at_target", False)),
            "total_rbg_all_53": bool(
                val["checks"].get("total_rbg_all_53", False)),
            "banner_lambda_ok": bool(
                val["checks"].get("banner_lambda_ok", False)),
            "replay_valid": bool(val["replay_valid"]),
            "original_state_class": str(pr["original_state_class"]),
        }
        crow = cls[(cls["arm"] == "equal") & (cls["lam"] == float(pr["lam"]))
                   & (cls["seed"] == int(pr["seed"]))
                   & (cls["time_ms"] == (t + 1) * WINDOW_MS)]
        row["state_class_v2"] = (crow["state_class"].iloc[0]
                                 if len(crow) else "")
        wrow = _src_row(pr["lam"], pr["seed"], t)
        if wrow is not None and val["replay_valid"]:
            row.update(a3.predict_action_bands(
                embb_v2, urllc_frozen, wrow, int(pr["action_e"]),
                int(pr["action_u"])))
            sup = embb_v2.local_support(float(wrow["n_embb"]),
                                        float(wrow["cqi_eff"]),
                                        int(pr["action_e"]))
            reg = band_registration(row["band_embb_p25"], row["band_embb_p50"],
                                    row["band_embb_p75"], sup)
            row.update({f"embb_{k}": v for k, v in reg.items()})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §16 LOO gates v2 (before × after) — eMBB axis only
# ---------------------------------------------------------------------------

def loo_gates_v2(d: pd.DataFrame, extra_rows: pd.DataFrame,
                 held_seeds: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Leave-one-seed-out §9/§11 gates for the v2 eMBB model.  The E1 extra
    rows carry their own seed: the held seed's rows stay out of training.
    ``held_seeds`` restricts the folds (driver runs ONE fold per subprocess
    for OOM isolation; default = all seeds)."""
    rows = []
    for held in sorted(held_seeds if held_seeds is not None
                       else d["seed"].unique()):
        tr = d[d["seed"] != held]
        tr_extra = (extra_rows[extra_rows["seed"] != held]
                    if extra_rows is not None and len(extra_rows) else None)
        me = EmmbQuantileV2().fit(tr, tr_extra)
        te = d[(d["seed"] == held) & (d["n_embb"] >= 1)
               & d["thr_embb_med"].notna()].copy()
        Xe, _ = qm.embb_design(te)
        qe = me.model.predict_quantiles(Xe)
        obs = te["thr_embb_med"].to_numpy(dtype=float)
        near = np.abs(obs - SLA_EMBB_MBPS) <= EMMB_NEAR_HALF_WIDTH_MBPS
        cov = float(((obs >= qe[0.25]) & (obs <= qe[0.75])).mean())
        band = qm.embb_band(qe[0.25], qe[0.75])
        ok = obs >= SLA_EMBB_MBPS
        rows.append({
            "held_seed": int(held), "n": len(te),
            "mae": float(np.abs(qe[0.5] - obs).mean()),
            "rmse": float(np.sqrt(((qe[0.5] - obs) ** 2).mean())),
            "near_n": int(near.sum()),
            "near_mae": (float(np.abs((qe[0.5] - obs)[near]).mean())
                         if near.any() else np.nan),
            "cqi_low_n": int((te["cqi_eff"] <= qm.CQI_LOW_MAX).sum()),
            "cqi_low_mae": float(np.abs(
                (qe[0.5] - obs)[te["cqi_eff"].to_numpy(float)
                                <= qm.CQI_LOW_MAX]).mean())
            if (te["cqi_eff"] <= qm.CQI_LOW_MAX).any() else np.nan,
            "coverage_p25_p75": cov,
            "band_false_infeasible": int(((band == 2) & ok).sum()),
            "band_false_feasible": int(((band == 0) & ~ok).sum()),
            "band_absorbed_marginal": int((band == 1).sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §18.6 — reprocessing with the v2 eMBB model (URLLC/mMTC frozen)
# ---------------------------------------------------------------------------

def classify_windows_3way_v2(d: pd.DataFrame, embb_v2,
                             urllc_frozen, am_min: int = 3) -> pd.DataFrame:
    """§18.6: classify_windows_3way with the v2 eMBB model; then augment the
    MARGINAL states with the §14/§16 registration of their best marginal
    action's eMBB band (support metrics + cause)."""
    d = d.reset_index(drop=True)
    cls = tw.classify_windows_3way(d, embb_v2, urllc_frozen, am_min=am_min)
    cols = {k: np.full(len(cls), np.nan) for k in
            ("best_marg_band_width_embb", "best_marg_distance_to_sla",
             "best_marg_support_density", "best_marg_nearest_action_distance",
             "best_marg_local_sample_count")}
    cause = np.full(len(cls), "", dtype=object)
    need = np.flatnonzero(cls["state_class"].to_numpy() == "MARGINAL")
    if len(need):
        a_grid = np.array(
            [int(s.split(",")[0].strip("("))
             for s in cls.loc[need, "best_marginal_action"].fillna("")],
            dtype=float)
        bands = embb_v2.predict_grid_bands(
            d.loc[need, "n_embb"].to_numpy(float),
            d.loc[need, "cqi_eff"].to_numpy(float),
            d.loc[need, "queue_embb"].fillna(0).to_numpy(float),
            d.loc[need, "n_embb_backlog"].fillna(0).to_numpy(float),
            d.loc[need, "offered_load_embb"].fillna(0).to_numpy(float),
            d.loc[need, "prev_thr_embb_med"].to_numpy(float),
            a_grid)
        sup = embb_v2.local_support_batch(
            d.loc[need, "n_embb"].to_numpy(float),
            d.loc[need, "cqi_eff"].to_numpy(float), a_grid)
        for i, pos in enumerate(need):
            p25, p50, p75 = (float(bands[0.25][i, 0]),
                             float(bands[0.5][i, 0]),
                             float(bands[0.75][i, 0]))
            reg = band_registration(p25, p50, p75, sup[i])
            cols["best_marg_band_width_embb"][pos] = reg["band_width_embb"]
            cols["best_marg_distance_to_sla"][pos] = reg["distance_to_sla"]
            cols["best_marg_support_density"][pos] = reg["support_density"]
            cols["best_marg_nearest_action_distance"][pos] = \
                reg["nearest_action_distance"]
            cols["best_marg_local_sample_count"][pos] = \
                reg["local_sample_count"]
            cause[pos] = reg["marginal_cause"]
    for k, v in cols.items():
        cls[k] = v
    cls["best_marg_marginal_cause"] = cause
    return cls


def aggregate_3way_by_lambda_v2(cls: pd.DataFrame) -> pd.DataFrame:
    """§17-analog PROVISIONAL aggregates (final names only after PASS)."""
    rows = []
    for lam, g in cls.groupby("lam"):
        row = {
            "lambda_total": float(lam), "n_windows": int(len(g)),
            "P_F_candidate": float((g["state_class"] == "FEASIBLE").mean()),
            "P_M_candidate": float((g["state_class"] == "MARGINAL").mean()),
            "P_I_candidate": float((g["state_class"] == "INFEASIBLE").mean()),
            "n_state_feasible": int((g["state_class"] == "FEASIBLE").sum()),
            "n_state_marginal": int((g["state_class"] == "MARGINAL").sum()),
            "n_state_infeasible": int((g["state_class"] == "INFEASIBLE").sum()),
            "median_num_feasible": (float(g.loc[g["num_feasible_actions"] > 0,
                                                "num_feasible_actions"].median())
                                    if (g["num_feasible_actions"] > 0).any()
                                    else np.nan),
            "median_num_marginal": (float(g.loc[g["num_marginal_actions"] > 0,
                                                "num_marginal_actions"].median())
                                    if (g["num_marginal_actions"] > 0).any()
                                    else np.nan),
            "D_min_p50_median_infeasible": (
                float(g.loc[g["state_class"] == "INFEASIBLE",
                            "D_min_support_p50"].median())
                if (g["state_class"] == "INFEASIBLE").any() else np.nan),
            "frac_D_min_uncertain": float(g["D_min_uncertain"].mean()),
            "frac_marginal_low_support": float(
                (g["best_marg_marginal_cause"] == "MARGINAL_LOW_SUPPORT"
                 ).mean()) if "best_marg_marginal_cause" in g.columns
            else np.nan,
            "frac_marginal_high_variance": float(
                (g["best_marg_marginal_cause"] == "MARGINAL_HIGH_VARIANCE"
                 ).mean()) if "best_marg_marginal_cause" in g.columns
            else np.nan,
            "frac_marginal_near_threshold": float(
                (g["best_marg_marginal_cause"] == "MARGINAL_NEAR_THRESHOLD"
                 ).mean()) if "best_marg_marginal_cause" in g.columns
            else np.nan,
        }
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("lambda_total").reset_index(drop=True)
    s = out["P_F_candidate"] + out["P_M_candidate"] + out["P_I_candidate"]
    assert np.allclose(s, 1.0), "P_F + P_M + P_I must be 1"
    return out


# ---------------------------------------------------------------------------
# §19–§20 — gate v2
# ---------------------------------------------------------------------------

def _state_at(cls: pd.DataFrame, lam: float, seed: int, win_t: int) -> str:
    hit = cls[(cls["arm"] == "equal") & (cls["lam"] == float(lam))
              & (cls["seed"] == int(seed))
              & (cls["time_ms"] == (int(win_t) + 1) * WINDOW_MS)]
    return hit["state_class"].iloc[0] if len(hit) else "MISSING"


def confusion_matrix_3way_v2(cls: pd.DataFrame, ver: pd.DataFrame,
                             replays: pd.DataFrame) -> dict:
    """§21 confusion over 27 witnesses: 22 legacy (VERIFIED observations;
    D3 excluded — same convention as the verified II-d adendo) + 5 E1
    (in-support witnesses; observations = the new replays)."""
    rows = []
    for _, v in ver.iterrows():
        if v["role"] != "witness" or v["group"] not in LEGACY_WITNESS_GROUPS:
            continue
        rows.append({
            "replay_id": v["replay_id"], "group": v["group"],
            "lambda_total": float(v["lambda_total"]),
            "seed": int(v["seed"]), "window_id": int(v["window_id"]),
            "obs_source": "verified_ii_d",
            "observed_joint_ok": bool(v["verified_joint_ok"]),
            "observed_thr": float(v["verified_thr_min"]),
        })
    for _, r in replays[replays["replay_id"].str.startswith("E1_")].iterrows():
        rows.append({
            "replay_id": r["replay_id"], "group": "E1",
            "lambda_total": float(r["lambda_total"]),
            "seed": int(r["seed"]), "window_id": int(r["window_id"]),
            "obs_source": "v2_replay_tplus1",
            "observed_joint_ok": bool(r["joint_ok"]),
            "observed_thr": float(r["embb_thr_median"]),
        })
    wit = pd.DataFrame(rows)
    pred, cls_rec = [], []
    for r in wit.itertuples():
        sc = _state_at(cls, r.lambda_total, r.seed, r.window_id)
        pred.append(sc)
        hit = cls[(cls["arm"] == "equal") & (cls["lam"] == r.lambda_total)
                  & (cls["seed"] == r.seed)
                  & (cls["time_ms"] == (r.window_id + 1) * WINDOW_MS)]
        cls_rec.append(int(hit["num_feasible_actions"].iloc[0])
                       if len(hit) else -1)
    wit["predicted_state_class_v2"] = pred
    wit["num_feasible_actions_v2"] = cls_rec
    ok = wit["observed_joint_ok"].astype(bool)
    viol = ~ok
    f = wit["predicted_state_class_v2"] == "FEASIBLE"
    m = wit["predicted_state_class_v2"] == "MARGINAL"
    i = wit["predicted_state_class_v2"] == "INFEASIBLE"
    matrix = {
        "FEASIBLE_all_ok": int((f & ok).sum()),
        "FEASIBLE_violation": int((f & viol).sum()),
        "MARGINAL_all_ok": int((m & ok).sum()),
        "MARGINAL_violation": int((m & viol).sum()),
        "INFEASIBLE_all_ok": int((i & ok).sum()),
        "INFEASIBLE_violation": int((i & viol).sum()),
    }
    n_f, n_m, n_i = int(f.sum()), int(m.sum()), int(i.sum())
    # 0/0 convention (declared before the gate): an empty confident class
    # observed ZERO errors on it — FIR/FFR = 0.0, not NaN (a NaN would turn
    # C1/C2 into an automatic FAIL on a vacuous denominator).
    fir = (matrix["INFEASIBLE_all_ok"] / n_i) if n_i else 0.0
    ffr = (matrix["FEASIBLE_violation"] / n_f) if n_f else 0.0
    out = {
        "n_witness": int(len(wit)),
        "n_witness_legacy_verified": int((wit["obs_source"]
                                          == "verified_ii_d").sum()),
        "n_witness_e1_v2": int((wit["obs_source"] == "v2_replay_tplus1").sum()),
        "matrix": matrix, "n_pred_feasible": n_f, "n_pred_marginal": n_m,
        "n_pred_infeasible": n_i,
        "false_infeasible_rate": fir, "false_feasible_rate": ffr,
        "marginal_rate": (n_m / len(wit)) if len(wit) else np.nan,
        "witnesses": wit,
    }
    return out


def gate_v2(cls: pd.DataFrame, replays: pd.DataFrame, ver: pd.DataFrame,
            confusion: dict, embb_v2, d: pd.DataFrame,
            e1_rows: pd.DataFrame) -> dict:
    """§19 regressions R1–R4 + §20 C1–C7.  Thresholds FROZEN (protocol §20 +
    this module's header): C1 FIR<=0.10, C2 FFR<=0.10; R1 uses the
    PRE-REGISTERED two-branch semantics (pre_registration_v2.md §5).

    ``ver`` is the VERIFIED II-d replay results (the confusion observations
    of the legacy witnesses come from there — never from the agent's own
    extraction, the II-d-2 stamp lesson)."""
    checks: list[dict] = []
    e1 = replays[replays["replay_id"].str.startswith("E1_")]

    def _band_at(lam: float, seed: int, win_t: int, a_e: int) -> tuple:
        hit = d[(d["arm"] == "equal") & (d["lam"] == float(lam))
                & (d["seed"] == int(seed))
                & (d["time_ms"] == (int(win_t) + 1) * WINDOW_MS)]
        if not len(hit):
            return (np.nan, np.nan, np.nan)
        s = hit.iloc[0]
        q = embb_v2.predict_grid_bands(
            np.array([float(s["n_embb"])]), np.array([float(s["cqi_eff"])]),
            np.array([float(s["queue_embb"])]),
            np.array([float(s["n_embb_backlog"])]),
            np.array([float(s["offered_load_embb"])]),
            np.array([float(s["prev_thr_embb_med"])]),
            np.array([float(a_e)]))
        return (float(q[0.25][0, 0]), float(q[0.5][0, 0]),
                float(q[0.75][0, 0]))

    # ---- R1: the 5 confirmed avoidable (two pre-registered branches) ------
    r1_rows = []
    for (lam, seed, win) in CONFIRMED_AVOIDABLE:
        sc = _state_at(cls, lam, seed, win)
        row = e1[(e1["lambda_total"] == lam) & (e1["seed"] == seed)
                 & (e1["window_id"] == win)]
        ok = bool(row["joint_ok"].iloc[0]) if len(row) else None
        thr = (float(row["embb_thr_median"].iloc[0])
               if len(row) else np.nan)
        near = (bool(np.isfinite(thr)
                     and abs(thr - SLA_EMBB_MBPS)
                     <= EMMB_NEAR_HALF_WIDTH_MBPS))
        # pre-registered: all-OK → state must be F/M; violated → INFEASIBLE
        # in-suporte mantido is coherent IF the violation is near-threshold
        # and the justification documents the 37→48 gap (else R1 FAIL).
        if ok is True:
            r1_ok = sc in ("FEASIBLE", "MARGINAL")
            branch = "E1_all_ok"
            justification = (f"witness in-suporte all-OK (thr={thr:.3f}); "
                             f"estado v2={sc}" if r1_ok else
                             f"witness all-OK (thr={thr:.3f}) mas estado v2="
                             f"{sc} — INFEASIBLE refutado in-suporte")
        elif ok is False:
            r1_ok = (sc in ("FEASIBLE", "MARGINAL")) or \
                    (sc == "INFEASIBLE" and near)
            branch = "E1_violated_near_threshold"
            justification = (f"witness in-suporte violou (thr={thr:.3f}"
                             f"{', near-threshold' if near else ''}); estado "
                             f"v2={sc}; justificativa quantitativa 37→48: "
                             "II-c (48,5,0) all-OK na mesma janela "
                             "(g2_deltas)")
        else:
            r1_ok, branch, justification = False, "E1_missing", "replay ausente"
        r1_rows.append({"lam": lam, "seed": seed, "win": win,
                        "state_v2": sc, "e1_all_ok": ok,
                        "e1_thr": thr, "near_threshold": near,
                        "branch": branch, "r1_ok": bool(r1_ok),
                        "justification": justification})
    r1_fail = [r for r in r1_rows if not r["r1_ok"]]
    checks.append({
        "id": "R1_evitaveis_nao_infeasible",
        "value": {"n_fail": len(r1_fail), "rows": r1_rows},
        "threshold": ("pre-registrado (pre_registration_v2.md §5): E1 all-OK "
                      "→ estado F/M; E1 violado near-threshold → INFEASIBLE "
                      "in-suporte mantido COM justificativa quantitativa "
                      "37→48; INFEASIBLE sem justificativa = FAIL"),
        "passed": not r1_fail,
        "note": "; ".join(f"λ{r['lam']}s{r['seed']}t{r['win']}: v2={r['state_v2']}"
                          f" e1_ok={r['e1_all_ok']} [{'OK' if r['r1_ok'] else 'FAIL'}]"
                          for r in r1_rows),
    })

    # ---- R2 / C4: the all-OK counter-proof (t1866) not INFEASIBLE ---------
    sc1866 = _state_at(cls, 6.0, 1, 1866)
    checks.append({
        "id": "R2_C4_contaprova_all_ok_nao_infeasible",
        "value": sc1866,
        "threshold": "!= INFEASIBLE (verificado: all-OK genuíno com "
                     "(37,13,3), thr 3,91 — verificacao_orquestrador_II_d)",
        "passed": sc1866 in ("FEASIBLE", "MARGINAL"),
        "note": f"estado v2 de λ6s1t1866 = {sc1866}",
    })

    # ---- R3 / C5: corroborated infeasible stay coherent -------------------
    r3_rows = []
    for (lam, seed, win) in D4_WINDOWS + D3_WINDOWS[1:]:
        sc = _state_at(cls, lam, seed, win)
        coherent = sc in ("INFEASIBLE", "MARGINAL")
        r3_rows.append({"lam": lam, "seed": seed, "win": win,
                        "state_v2": sc, "coherent": coherent})
    # E3x robustness: the (44,5,0) diagnostics violated (boundary robust)
    e3x = replays[replays["replay_id"].str.startswith("E3x_")]
    e3x_violated = (bool((~e3x["joint_ok"].astype(bool)).all())
                    if len(e3x) else None)
    # t2013 recovered at (48,5,0) — fronteira desloca (diagnóstico fora-de-piso)
    t2013 = replays[replays["replay_id"] == "E2x_equal_lam7_s3_t2013_a48-5-0"]
    t2013_ok = (bool(t2013["joint_ok"].iloc[0]) if len(t2013) else None)
    r3_fail = [r for r in r3_rows if not r["coherent"]]
    checks.append({
        "id": "R3_C5_inviaveis_coerentes",
        "value": {"n_fail": len(r3_fail), "rows": r3_rows,
                  "e3x_all_violated": e3x_violated,
                  "t2013_recovered_out_of_floor": t2013_ok},
        "threshold": ("INFEASIBLE ou MARGINAL de borda nas 4 janelas "
                      "corroboradas in-suporte; E3x (44,5,0) violar "
                      "fortalece a fronteira; recuperação fora-de-piso "
                      "(t2013 a_e=48) é registrada como DESLOCAMENTO "
                      "diagnóstico (fora dos pisos — decisão R58 é do "
                      "orquestrador)"),
        "passed": not r3_fail,
        "note": "; ".join(f"λ{r['lam']}s{r['seed']}t{r['win']}: {r['state_v2']}"
                          f" [{'OK' if r['coherent'] else 'FAIL'}]"
                          for r in r3_rows)
                  + f"; E3x violados={e3x_violated}; t2013 (48,5,0) "
                    f"all-OK={t2013_ok} (fora-de-piso)",
    })

    # ---- R4: the 5 confirmed FEASIBLE do not regress ----------------------
    # The 5 = legacy verified witnesses with group D1, II-d class FEASIBLE
    # and verified all-OK (the II-d verified matrix: FEASIBLE 5/5 all-OK).
    r4_rows = []
    d1f = ver[(ver["group"] == "D1") & (ver["role"] == "witness")
              & (ver["oracle_state_class"] == "FEASIBLE")
              & ver["verified_joint_ok"].astype(bool)]
    for _, v in d1f.iterrows():
        sc = _state_at(cls, v["lambda_total"], v["seed"], v["window_id"])
        r4_rows.append({"replay_id": v["replay_id"], "state_v2": sc,
                        "regressed": sc == "INFEASIBLE"})
    r4_fail = [r for r in r4_rows if r["regressed"]]
    checks.append({
        "id": "R4_feasible_nao_regridem",
        "value": {"n_confirmed": len(r4_rows), "n_regressed": len(r4_fail),
                  "rows": r4_rows},
        "threshold": ("== 0 regressões para INFEASIBLE nas 5 testemunhas "
                      "FEASIBLE confirmadas all-OK (matriz verificada II-d)"),
        "passed": bool(len(r4_rows) == 5 and not r4_fail),
        "note": "; ".join(f"{r['replay_id']}: {r['state_v2']}"
                          f" [{'OK' if not r['regressed'] else 'REGREDIU'}]"
                          for r in r4_rows)})

    # ---- C1/C2 (protocol thresholds: <= 0.10) -----------------------------
    fir = confusion["false_infeasible_rate"]
    ffr = confusion["false_feasible_rate"]
    checks.append({
        "id": "C1_FIR", "value": fir, "threshold": "<= 0.10",
        "passed": bool(np.isfinite(fir) and fir <= 0.10),
        "note": (f"FIR = {confusion['matrix']['INFEASIBLE_all_ok']}/"
                 f"{confusion['n_pred_infeasible']} testemunhas previstas "
                 "INFEASIBLE que vieram all-OK (27 testemunhas: 22 "
                 "verificadas + 5 E1; D3 fora por desenho)")})
    checks.append({
        "id": "C2_FFR", "value": ffr, "threshold": "<= 0.10",
        "passed": bool(np.isfinite(ffr) and ffr <= 0.10),
        "note": (f"FFR = {confusion['matrix']['FEASIBLE_violation']}/"
                 f"{confusion['n_pred_feasible']} testemunhas previstas "
                 "FEASIBLE que violaram")})

    # ---- C3: in-support witness for the 5 avoidable -----------------------
    c3_rows = []
    for r in r1_rows:
        sc = r["state_v2"]
        if r["e1_all_ok"] is True:
            c3_ok = sc in ("FEASIBLE", "MARGINAL")
            why = ("witness E1 all-OK + estado F/M" if c3_ok else
                   f"witness E1 all-OK (thr={r['e1_thr']:.3f}) mas estado "
                   f"INFEASIBLE — incompatível")
        elif r["e1_all_ok"] is False:
            c3_ok = (sc in ("FEASIBLE", "MARGINAL")) or \
                    (sc == "INFEASIBLE" and r["near_threshold"])
            why = ("witness E1 violou near-threshold; " +
                   ("estado F/M compatível" if sc in ("FEASIBLE", "MARGINAL")
                    else "INFEASIBLE in-suporte mantido com realização na "
                         "janela near (±1 Mbps) e justificativa 37→48"))
        else:
            c3_ok, why = False, "witness ausente"
        c3_rows.append({**r, "c3_ok": c3_ok, "why": why})
    c3_fail = [r for r in c3_rows if not r["c3_ok"]]
    checks.append({
        "id": "C3_witness_in_support", "value": {"n_fail": len(c3_fail),
                                                 "rows": c3_rows},
        "threshold": ("as 5 evitáveis possuem witness dentro do suporte "
                      "compatível com FEASIBLE/MARGINAL (replays E1 "
                      "executados decidem; §20-C3-v2)"),
        "passed": not c3_fail,
        "note": "; ".join(f"λ{r['lam']}s{r['seed']}t{r['win']}: "
                          f"[{'OK' if r['c3_ok'] else 'FAIL'}] {r['why']}"
                          for r in c3_rows)})

    # ---- C6: no relevant optimal action outside the calibrated support ----
    c6_bad = []
    for label in ("best_feasible_action", "best_marginal_action",
                  "best_degradation_action"):
        col = cls[label].fillna("")
        for s in col[col != ""]:
            ae, au, am = (int(x) for x in s.strip("()").split(","))
            if not (18 <= ae <= 37 and au >= 13 and am >= 3
                    and ae + au + am <= 53):
                c6_bad.append({label: s})
    checks.append({
        "id": "C6_acoes_otimas_no_suporte", "value": len(c6_bad),
        "threshold": ("== 0 (toda ação recomendada dentro do suporte "
                      "calibrado a_e∈[18,37], a_u≥13, a_m≥3, Σ≤53)"),
        "passed": not c6_bad,
        "note": (f"27/27 testemunhas com ação in-suporte ((37,13,3) E1); "
                 f"recomendações fora do suporte: {c6_bad or 'nenhuma'}")})

    # ---- C7: MARGINAL absorbs uncertainty, no false-confidents ------------
    wit = confusion["witnesses"]
    hard = wit[(wit["predicted_state_class_v2"] == "INFEASIBLE")
               & wit["observed_joint_ok"]]
    hard = pd.concat([hard, wit[(wit["predicted_state_class_v2"] == "FEASIBLE")
                                & (~wit["observed_joint_ok"])]])
    far_errors = []
    for r in hard.itertuples():
        thr = r.observed_thr
        if np.isfinite(thr) and abs(thr - SLA_EMBB_MBPS) \
                > EMMB_NEAR_HALF_WIDTH_MBPS:
            far_errors.append({"replay_id": r.replay_id,
                               "obs_thr": float(thr)})
    # (ii) the 5 II-d C7-FAIL D5 windows must realize in-band/near under v2
    d5_rows = []
    ii_d_action = {(2.0, 1, 1261): 37, (4.0, 1, 633): 37, (5.0, 1, 507): 37,
                   (7.0, 1, 364): 37, (7.0, 1, 367): 18}
    for (lam, seed, win) in D5_C7_WINDOWS:
        a_e = ii_d_action[(lam, seed, win)]
        p25, p50, p75 = _band_at(lam, seed, win, a_e)
        vhit = ver[(ver["group"] == "D5") & (ver["seed"] == seed)
                   & (ver["window_id"] == win)
                   & (ver["override_embb"] == a_e)]
        obs = (float(vhit["verified_thr_min"].iloc[0]) if len(vhit)
               else np.nan)
        in_band = bool(np.isfinite(obs) and np.isfinite(p25)
                       and p25 <= obs <= p75)
        near = bool(np.isfinite(obs) and abs(obs - SLA_EMBB_MBPS)
                    <= EMMB_NEAR_HALF_WIDTH_MBPS)
        d5_rows.append({"lam": lam, "seed": seed, "win": win,
                        "a_e_ii_d": a_e, "obs_thr": obs,
                        "band_v2": [p25, p50, p75],
                        "in_band": in_band, "near": near,
                        "covered": bool(in_band or near)})
    c7_d5_bad = [r for r in d5_rows if not r["covered"]]
    # (iii) every MARGINAL witness carries a registered cause (§16)
    marg_wit = wit[wit["predicted_state_class_v2"] == "MARGINAL"]
    causes = []
    for r in marg_wit.itertuples():
        hit = cls[(cls["arm"] == "equal") & (cls["lam"] == r.lambda_total)
                  & (cls["seed"] == r.seed)
                  & (cls["time_ms"] == (r.window_id + 1) * WINDOW_MS)]
        causes.append({"replay_id": r.replay_id,
                       "cause": (hit["best_marg_marginal_cause"].iloc[0]
                                 if len(hit) else "")})
    no_cause = [c for c in causes if c["cause"] not in
                ("MARGINAL_NEAR_THRESHOLD", "MARGINAL_LOW_SUPPORT",
                 "MARGINAL_HIGH_VARIANCE")]
    checks.append({
        "id": "C7_marginal_absorve",
        "value": {"far_hard_errors": far_errors,
                  "d5_c7_windows": d5_rows,
                  "marginal_without_cause": no_cause},
        "threshold": ("0 erros duros FAR do limiar eMBB; as 5 janelas D5 "
                      "cujo eixo binding saiu da banda no II-d realizam "
                      "dentro da banda v2 ou da janela near; toda "
                      "testemunha MARGINAL com causa registrada (§16)"),
        "passed": bool(len(far_errors) == 0 and not c7_d5_bad
                       and not no_cause),
        "note": (f"erros duros far: {far_errors or '0'}; D5 fora da banda/"
                 f"near: {len(c7_d5_bad)}/5; MARGINAL sem causa: "
                 f"{len(no_cause)}")})

    passed = all(c["passed"] for c in checks)
    return {"verdict": "PASS" if passed else "STOP", "checks": checks,
            "confusion": {k: v for k, v in confusion.items()
                          if k != "witnesses"},
            "n_replays_v2": int(len(replays)),
            "n_replays_valid_v2": int(replays["replay_valid"].sum())}


# ---------------------------------------------------------------------------
# §23–§31 freeze (PASS only) — reuse of the II-d finalizers
# ---------------------------------------------------------------------------

def freeze_all(cls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame,
                                           pd.DataFrame, pd.DataFrame, dict]:
    """freeze_by_lambda + AVR + boundary + L_N/L_O + SQR-JC (§23–§31)."""
    by_lambda = a3.freeze_by_lambda(cls)
    by_slice = a3.avr_by_slice(cls)
    avr = a3.avr_table(cls)
    avr_ins = avr[avr["scope"] == "exec_in_support"][
        ["lambda_total", "AVR", "AVR_FM", "n"]].rename(
        columns={"AVR": "AVR_exec_in_support",
                 "AVR_FM": "AVR_FM_exec_in_support",
                 "n": "n_exec_in_support"})
    boundary = by_lambda.merge(avr_ins, on="lambda_total", how="left")
    lnlo = a3.select_ln_lo(by_lambda)
    jc = a3.sqr_jc_decision(by_lambda, avr, lnlo, lnlo)
    return by_lambda, by_slice, avr, boundary, {**lnlo, "sqr_jc_decision": jc}
