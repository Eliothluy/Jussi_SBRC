"""Per-window dataset builder for the Etapa-2.4 offline oracle (Despacho I).

Builds, for EACH of the 30 frozen Etapa-2.3 runs (5 loads x 3 seeds x 2 arms),
one row per 100 ms decision window with the state z_t, the executed action,
the observed KPIs and the observed SLA violations (protocol section 4 list;
the protocol section 19 column order is produced by
:func:`oracle_windows_table`).

Window convention (verified against ``scratch/sqr/sqr-sim.cc`` ~1540-1640,
"Telemetria SQR por janela" drain comment): the ``sqr_prb_windows.csv`` /
``sqr_kpm_history.csv`` row stamped at t ms describes the window
(t-100, t].  We index windows by ``win = t//100 - 1`` (0-based, window w
covers [100w, 100(w+1)) ms).  URLLC packets from ``sqr_packet_delays.csv``
are assigned to windows by ARRIVAL time ``win = floor(txSec * 10)`` — the
same convention of ``analyze_urllc_tail.py`` — so the window state pairs with
the service the packets actually received.

Demand convention ("UEs com demanda", R35/R36): a UE row is WITH DEMAND in a
window when it carries traffic in that window — ``dTxBytes > 0`` — the exact
UE-window population the frozen 2.3 judgment uses (``analyze_hetero_sweep.py``
``e = k[(k.sliceId==0) & (k.dTxBytes>0)]``).  Offered-load proxies
(``[ASSUMIDO-ORACLE]``, registered): a BACKLOGGED eMBB UE (``queueBytes>0``,
greedy full-buffer R04) offers its R36 service target (3.5 Mbps/UE); CBR
profiles offer their nominal cadence (R35: URLLC 32 B/1 ms = 0.256 Mbps/UE;
mMTC 200 B/1000 ms = 0.0016 Mbps/UE).

Executed action ("ação executada"): the quota actually in force in the
window.  ``prb_executed_rbg_i`` carries it when an agent action arrived
(IPC/E2, ``prb_requested_pct_0 >= 0``); in the network-only equal arm those
stay 0/-1 (sqr-sim.cc:1556 "Sentinela -1/0 = janela sem ação do agente") and
the quota is recovered exactly from the budget accumulator
``prb_budget_rbg_i / dl_slots`` (integer by construction — the accumulator
multiplies the integer quota by the window's DL slots).  ``prb_scheduled_rbg_i``
and ``prb_budget_rbg_i`` are RBG-slot ACCUMULATORS over the window (sum of
per-slot counters), NOT single-slot counts; ``prb_idle_rbg`` likewise
(``total_rbg*dl_slots - sum_i scheduled_i``, sqr-sim.cc drain).

util definitions are the FROZEN 2.3 ones (``analyze_hetero_sweep.py``):
``util_active_dl`` (D2) = 100 * sum_i sched_i / (C * dl_slots) over windows
with dl_slots > 0 (per-window: NaN when dl_slots == 0); ``util_all_dl`` (D1)
= 100 * sum_i sched_i / (C * (400/7)) per window — the same nominal
400/7 DL-slots-per-window constant of the 2.3 script.

SLA rules are the FROZEN R36-R39 (NEVER recalibrated here):
- eMBB (R36): ``windowThroughputMbps < 3.5`` per UE-window with demand; the
  window-level judgment aggregates the per-UE-window values by their MEDIAN
  (``[ASSUMIDO-ORACLE]`` window operationalization, consistent with the
  equal-share model; the any-UE and violation-fraction variants are exported
  as extras);
- URLLC (R37): per-packet delay > 10 ms is a miss (same packet_delays
  instrument); the window-level judgment is the window p95 of packet delays
  > 10 ms (``[ASSUMIDO-ORACLE]``; equivalent to deadline-miss ratio > 5% in
  the window; the per-window miss% is exported alongside);
- mMTC (R38): window PDR = sum dRxPackets / sum dTxPackets < 0.95 (windows
  with zero offered packets carry no information -> NaN, R19-style
  exclusion); a 1 s rolling PDR is exported as an extra for the sparse profile.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Frozen SLAs (R36-R39 — DO NOT touch; same constants as analyze_hetero_sweep.py)
# ---------------------------------------------------------------------------
SLA_EMBB_MBPS = 3.5
SLA_URLLC_MS = 10.0
SLA_MMTC_PDR = 0.95

#: Offered-load proxies [ASSUMIDO-ORACLE] (R36 target; R35 nominal cadences).
OFFERED_EMBB_MBPS_PER_UE = 3.5      # backlogged greedy UE offers the R36 target
OFFERED_URLLC_MBPS_PER_UE = 32 * 8.0 / 1e3 / 1e-3 / 1e6   # 32 B / 1 ms = 0.256
OFFERED_MMTC_MBPS_PER_UE = 200 * 8.0 / 1e3 / 1.0 / 1e6    # 200 B / 1000 ms = 0.0016

#: Nominal DL slots per 100 ms window of the frozen 2.3 D1 formula.
D1_SLOTS_PER_WINDOW = 400.0 / 7.0

#: Number of slices in the canonical sqr-sim layout (0=eMBB, 1=URLLC, 2=mMTC).
NUM_SLICES = 3

_ARMS = {"equal": "hetero_sweep", "prop": "hetero_sweep_prop"}
_TAG_RE = re.compile(r"A_lam(\d+)_s(\d+)$")


def _prb_windows_for_run(tag_dir: str) -> str:
    hits = sorted(glob.glob(os.path.join(tag_dir, "**", "sqr_prb_windows.csv"),
                            recursive=True))
    if not hits:
        raise FileNotFoundError(f"sqr_prb_windows.csv under {tag_dir}")
    return hits[0]


def _packet_delays_for_run(tag_dir: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(tag_dir, "**", "sqr_packet_delays.csv"),
                            recursive=True))
    return hits[0] if hits else None


def _kpm_for_run(tag_dir: str) -> str:
    hits = sorted(glob.glob(os.path.join(tag_dir, "**", "sqr_kpm_history.csv"),
                            recursive=True))
    if not hits:
        raise FileNotFoundError(f"sqr_kpm_history.csv under {tag_dir}")
    return hits[0]


def _read_actions(prb: pd.DataFrame) -> pd.DataFrame:
    """Executed quota (a_e, a_u, a_m) per window + per-slice PRB telemetry.

    IPC/E2 windows (``prb_requested_pct_0 >= 0``) use ``prb_executed_rbg_i``
    verbatim; network-only windows recover the integer quota from the budget
    accumulator ``prb_budget_rbg_i / dl_slots`` (exact integer — verified in
    the 15 equal runs, integrality 100%).  ``prb_requested`` is exported in
    RBG-equivalent (sum of pct/100 * C, 0 when the -1 sentinel).
    """
    c = int(prb["total_rbg"].iloc[0])
    ipc = prb["prb_requested_pct_0"].values >= 0
    slots = prb["dl_slots"].values.astype(float)
    out = pd.DataFrame({
        "win": prb["timestamp_ms"].values // 100 - 1,
        "time_ms": prb["timestamp_ms"].values,
        "total_rbg": c,
        "dl_slots": prb["dl_slots"].values,
        "action_source": np.where(ipc, "ipc", "budget_derived"),
    })
    sched = np.zeros(len(prb))
    ex = np.zeros(len(prb))
    bud = np.zeros(len(prb))
    idle = prb["prb_idle_rbg"].values.astype(float)
    req = np.zeros(len(prb))
    for s in range(NUM_SLICES):
        sc = prb[f"prb_scheduled_rbg_{s}"].values.astype(float)
        bd = prb[f"prb_budget_rbg_{s}"].values.astype(float)
        rq = prb[f"prb_requested_pct_{s}"].values.astype(float)
        exq = prb[f"prb_executed_rbg_{s}"].values.astype(float)
        from_budget = np.round(np.divide(bd, slots, out=np.zeros_like(bd),
                                         where=slots > 0))
        quota = np.where(ipc, exq, from_budget)
        out[f"a_{s}"] = quota
        out[f"prb_scheduled_rbg_{s}"] = sc          # RBG-slot accumulator
        out[f"prb_budget_rbg_{s}"] = bd             # RBG-slot accumulator
        out[f"prb_executed_rbg_{s}"] = quota        # integer quota in force
        out[f"prb_requested_pct_{s}"] = np.where(rq >= 0, rq, np.nan)
        req += np.where(rq >= 0, rq / 100.0 * c, 0.0)
        sched += sc
        ex += quota
        bud += bd
    out["prb_requested"] = req
    out["prb_scheduled"] = sched
    out["prb_executed"] = ex
    out["prb_budget"] = bud
    out["prb_idle"] = idle
    # canonical slice aliases (0=eMBB, 1=URLLC, 2=mMTC)
    out = out.rename(columns={"a_0": "a_e", "a_1": "a_u", "a_2": "a_m"})
    with np.errstate(divide="ignore", invalid="ignore"):
        out["util_active_dl"] = np.where(
            out["dl_slots"].values > 0,
            100.0 * sched / (c * np.maximum(out["dl_slots"].values, 1)),
            np.nan,
        )
    out["util_all_dl"] = 100.0 * sched / (c * D1_SLOTS_PER_WINDOW)
    return out


def _read_kpm_windows(kpm: pd.DataFrame) -> pd.DataFrame:
    """Per-window state/KPI aggregates from the per-UE KPM rows."""
    k = kpm.copy()
    k["win"] = k["timestamp"].values // 100 - 1
    k["has_demand"] = k["dTxBytes"] > 0
    pieces = {}
    for s, name in [(0, "embb"), (1, "urllc"), (2, "mmtc")]:
        ks = k[k["sliceId"] == s]
        dem = ks[ks["has_demand"]]
        g = pd.DataFrame(index=sorted(k["win"].unique()))
        g.index.name = "win"
        g[f"n_{name}"] = dem.groupby("win")["ueImsi"].nunique()
        g[f"n_{name}_present"] = ks.groupby("win")["ueImsi"].nunique()
        if s == 0:
            g["thr_embb_med"] = dem.groupby("win")["windowThroughputMbps"].median()
            g["thr_embb_mean"] = dem.groupby("win")["windowThroughputMbps"].mean()
            g["thr_embb_min"] = dem.groupby("win")["windowThroughputMbps"].min()
            g["thr_embb_uewins"] = dem.groupby("win")["windowThroughputMbps"].size()
            g["thr_embb_violfrac"] = dem.groupby("win")["windowThroughputMbps"].apply(
                lambda x: float((x < SLA_EMBB_MBPS).mean()))
            g["n_embb_backlog"] = ks[ks["queueBytes"] > 0].groupby("win")["ueImsi"].nunique()
        g[f"queue_{name}"] = ks.groupby("win")["queueBytes"].sum()
        tx = dem.groupby("win")["dTxPackets"].sum()
        rx = dem.groupby("win")["dRxPackets"].sum()
        g[f"pdr_{name}"] = rx / tx.where(tx > 0)
        valid_snr = ks[ks["snrDb"] != -999.0]
        g[f"snr_{name}"] = valid_snr.groupby("win")["snrDb"].median()
        valid_cqi = ks[ks["cqiWideband"] != -1]
        g[f"cqi_{name}"] = valid_cqi.groupby("win")["cqiWideband"].median()
        if s == 0:
            pieces["embb"] = g
        elif s == 1:
            pieces["urllc"] = g
        else:
            pieces["mmtc"] = g
    return pieces


def _urllc_packet_windows(delays_path: str) -> pd.DataFrame:
    """Per-slice-window packet-delay aggregates (slice 1), arrival-window convention."""
    d = pd.read_csv(delays_path)
    u = d[d["sliceId"] == 1].copy()
    if not len(u):
        return pd.DataFrame(columns=["win", "latency_urllc_p95_ms",
                                     "latency_urllc_miss_pct", "urllc_n_pkts",
                                     "latency_urllc_med_ms"])
    u["win"] = (u["txSec"] * 10).astype(int)
    ms = u["delaySec"] * 1000.0
    u = u.assign(ms=ms, miss=ms > SLA_URLLC_MS)
    g = u.groupby("win").agg(
        latency_urllc_p95_ms=("ms", lambda x: float(np.percentile(x, 95))),
        latency_urllc_med_ms=("ms", "median"),
        latency_urllc_miss_pct=("miss", lambda x: 100.0 * float(x.mean())),
        urllc_n_pkts=("ms", "size"),
    )
    return g.reset_index()


def build_window_dataset(round2_dir: str) -> pd.DataFrame:
    """Build the full per-window dataset over the 30 frozen 2.3 runs.

    Parameters
    ----------
    round2_dir:
        ``.../results/sqr_characterization_round2`` (contains ``hetero_sweep``
        and ``hetero_sweep_prop``).

    Returns
    -------
    pd.DataFrame
        One row per (arm, lam, seed, win) — 30 runs x 2999 windows — with
        run identity, state z_t, executed action, observed KPIs and observed
        SLA violations (median/any variants).  See module docstring for every
        convention; new premises are tagged ``[ASSUMIDO-ORACLE]`` in
        ``MODA/docs/sqr/assumptions_register.md`` (R40 ff.).
    """
    frames = []
    for arm, sub in _ARMS.items():
        sw = os.path.join(round2_dir, sub)
        for tag_dir in sorted(glob.glob(os.path.join(sw, "A_lam*"))):
            if not os.path.isdir(tag_dir):
                continue
            m = _TAG_RE.search(os.path.basename(tag_dir))
            if not m:
                continue
            lam, seed = float(m.group(1)), int(m.group(2))
            kpm_path = _kpm_for_run(tag_dir)
            prb_path = _prb_windows_for_run(tag_dir)
            delays_path = _packet_delays_for_run(tag_dir)
            kpm = pd.read_csv(kpm_path)
            prb = pd.read_csv(prb_path)
            acts = _read_actions(prb)
            pieces = _read_kpm_windows(kpm)
            base = pieces["embb"].join(pieces["urllc"], how="outer").join(
                pieces["mmtc"], how="outer")
            df = acts.set_index("win").join(base, how="left").reset_index()
            if delays_path is not None:
                up = _urllc_packet_windows(delays_path)
                df = df.merge(up, on="win", how="left")
            else:  # pragma: no cover — all 30 runs have packet delays
                for col in ("latency_urllc_p95_ms", "latency_urllc_med_ms",
                            "latency_urllc_miss_pct", "urllc_n_pkts"):
                    df[col] = np.nan
            df["arm"] = arm
            df["lam"] = lam
            df["seed"] = seed
            df["tag"] = os.path.basename(tag_dir)
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    # ---- observed window violations (frozen rules; NaN = no information) ----
    out["observed_embb_violation"] = (out["thr_embb_med"] < SLA_EMBB_MBPS).astype(
        float).where(out["thr_embb_med"].notna())
    out["observed_embb_violation_anyue"] = (out["thr_embb_min"] < SLA_EMBB_MBPS).astype(
        float).where(out["thr_embb_min"].notna())
    out["observed_urllc_violation"] = (out["latency_urllc_p95_ms"] > SLA_URLLC_MS).astype(
        float).where(out["latency_urllc_p95_ms"].notna())
    # mMTC: R38 judges the SESSION PDR; the window analog is the 1 s ROLLING
    # PDR (the R38 rationale itself warns the 100 ms PDR is binary for the
    # sparse profile) — set here provisionally and OVERWRITTEN with the real
    # rolling value in add_rolling_pdr; the 100 ms variant kept as an extra.
    out["observed_mmtc_violation_100ms"] = (out["pdr_mmtc"] < SLA_MMTC_PDR).astype(
        float).where(out["pdr_mmtc"].notna())
    out["observed_mmtc_violation"] = out["observed_mmtc_violation_100ms"]
    viol = out[["observed_embb_violation", "observed_urllc_violation",
                "observed_mmtc_violation"]]
    out["observed_any_violation"] = viol.any(axis=1).astype(float).where(
        viol.notna().any(axis=1))

    # ---- offered-load proxies [ASSUMIDO-ORACLE] ----
    out["offered_load_embb"] = OFFERED_EMBB_MBPS_PER_UE * out["n_embb_backlog"].fillna(0)
    out["offered_load_urllc"] = OFFERED_URLLC_MBPS_PER_UE * out["n_urllc"].fillna(0)
    out["offered_load_mmtc"] = OFFERED_MMTC_MBPS_PER_UE * out["n_mmtc"].fillna(0)

    # 1 s rolling PDRs are added downstream by :func:`add_rolling_pdr` (they
    # need the raw per-window packet counts; the ratio column alone cannot be
    # rolled).  Sort once for the rolling window to be causal in time.
    out = out.sort_values(["arm", "lam", "seed", "win"]).reset_index(drop=True)
    out["pdr_urllc_1s"] = np.nan
    out["pdr_mmtc_1s"] = np.nan
    return out


def add_rolling_pdr(df: pd.DataFrame, kpm_cache: dict[tuple, pd.DataFrame]) -> pd.DataFrame:
    """Add 1 s (10-window) rolling PDR columns using raw per-window packet counts.

    ``kpm_cache`` maps (arm, lam, seed) -> per-window (tx, rx) DataFrames for
    the URLLC and mMTC slices, as produced by :func:`per_run_packet_counts`.
    """
    for (arm, lam, seed), g_idx in df.groupby(["arm", "lam", "seed"]).groups.items():
        counts = kpm_cache.get((arm, lam, seed))
        if counts is None:
            continue
        idx = df.loc[g_idx].index
        rows = df.loc[idx]
        for prof in ("urllc", "mmtc"):
            tx = counts[f"tx_{prof}"].reindex(rows["win"].values).fillna(0.0)
            rx = counts[f"rx_{prof}"].reindex(rows["win"].values).fillna(0.0)
            t10 = tx.rolling(10, min_periods=1).sum()
            r10 = rx.rolling(10, min_periods=1).sum()
            vals = (r10 / t10).where(t10 > 0)
            df.loc[idx, f"pdr_{prof}_1s"] = vals.to_numpy()
        # R38 window analog: mMTC violation = 1 s rolling PDR < 95%
        roll = df.loc[idx, "pdr_mmtc_1s"]
        df.loc[idx, "observed_mmtc_violation"] = (roll < SLA_MMTC_PDR).astype(
            float).where(roll.notna())
        if {"observed_embb_violation", "observed_urllc_violation"} \
                <= set(df.columns):
            v = df.loc[idx, ["observed_embb_violation", "observed_urllc_violation",
                             "observed_mmtc_violation"]]
            anyv = v.any(axis=1).astype(float).where(
                v.notna().any(axis=1)).to_numpy()
            df.loc[idx, "observed_any_violation"] = anyv
    return df


def per_run_packet_counts(kpm: pd.DataFrame, wins: np.ndarray) -> pd.DataFrame:
    """Per-window offered/delivered packet counts for the URLLC/mMTC slices."""
    k = kpm.copy()
    k["win"] = k["timestamp"].values // 100 - 1
    k = k[k["win"].isin(set(wins))]
    out = pd.DataFrame(index=sorted(wins))
    out.index.name = "win"
    for s, name in [(1, "urllc"), (2, "mmtc")]:
        ks = k[k["sliceId"] == s]
        out[f"tx_{name}"] = ks.groupby("win")["dTxPackets"].sum().reindex(out.index).fillna(0.0)
        out[f"rx_{name}"] = ks.groupby("win")["dRxPackets"].sum().reindex(out.index).fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Protocol section 19 — EXACT column order of the main oracle CSV
# ---------------------------------------------------------------------------
ORACLE_WINDOWS_COLUMNS_19: tuple[str, ...] = (
    "seed",
    "time",
    "lambda_total",
    "arm",
    "n_embb",
    "n_urllc",
    "n_mmtc",
    "queue_embb",
    "queue_urllc",
    "queue_mmtc",
    "util_active_dl",
    "util_all_dl",
    "observed_embb_violation",
    "observed_urllc_violation",
    "observed_mmtc_violation",
    "observed_any_violation",
    "joint_feasible",
    "num_feasible_actions",
    "avoidable_any",
    "avoidable_embb",
    "avoidable_urllc",
    "avoidable_mmtc",
    "embb_individually_saveable",
    "urllc_individually_saveable",
    "mmtc_individually_saveable",
    "best_feasible_embb",
    "best_feasible_urllc",
    "best_feasible_mmtc",
    "best_degradation_embb",
    "best_degradation_urllc",
    "best_degradation_mmtc",
    "min_total_deficit",
    "min_deficit_embb",
    "min_deficit_urllc",
    "min_deficit_mmtc",
)
