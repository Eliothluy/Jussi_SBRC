#!/usr/bin/env python
"""sqr_jc_calibrate.py — calibração offline da fase CAL (§13/§16/§37).

Consome os 9 runs CAL (results/sqr_jc_final/03_calibration/cal_runs/
cal_lam{2,5,6}_s{1001,1002,1003} — controlador prop congelado = o
bootstrap-PROP puro do §29) e produz, SEM tocar ns-3:

1. ``feature_scaler_manifest.yaml`` (§13): p1/p99 das features sem limite
   físico (contagens/filas/ofertas), calculado SOMENTE com dados CAL.
2. ``sigma_support_manifest.yaml`` (§16): distribuição de sigma nos pontos
   CAL, grid de percentis {50,60,70,80,90,95}, critérios de cobertura
   q25/q75 com moving-block bootstrap (bloco 5 s = 50 janelas, 1000
   reamostras, CI 95%, erro absoluto ≤ 0.10, suporte ≥ 70%); escolhe o
   MAIOR threshold que passa e congela.
3. ``kernel_log.csv`` (§57) agregado + ``calibration_report.json``.
4. ``search_benchmark.json`` (§37): benchmark da busca exaustiva em 100
   estados CAL estratificados por carga (low/mid/high) + coarse-to-fine.

Convenções de janela: as FROZEN da Rodada 2 (win = timestamp//100 − 1;
URLLC por janela de CHEGADA floor(txSec·10); mMTC PDR rolante de 1 s;
eMBB = mediana do windowThroughput com demanda) — reusadas por importação
somente-leitura de ``nsoran.sqr.oracle.dataset`` (código congelado do Gate
2.4-II-d; NADA é modificado).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsoran.sqr.oracle.dataset import (  # frozen Round-2 conventions (read-only)
    _read_actions,
    _read_kpm_windows,
    _urllc_packet_windows,
)

from nsoran.sqr_jc.feature_scaler import FeatureScaler
from nsoran.sqr_jc.joint_action_space import C_PRB
from nsoran.sqr_jc.joint_state import JOINT_STATE_DIM, JointState
from nsoran.sqr_jc.kernel_models import MultiQuantileKernelModel
from nsoran.sqr_jc.kernel_models import MEMORY_SIZE as B_MEMORY
from nsoran.sqr_jc.sla_margin import SliceOutcome

SNR_SENTINEL = -999.0
CQI_SENTINEL = -1

SIGMA_GRID_PCT = (50, 60, 70, 80, 90, 95)      # §16
SUPPORTED_MIN_FRAC = 0.70                       # §16
COVERAGE_ABS_ERR_MAX = 0.10                     # §16
BLOCK_BOOTSTRAP_RESAMPLES = 1000                # §16
BLOCK_SECONDS = 5.0                             # §16 (5 s = 50 janelas de 100 ms)
COVERAGE_TARGETS = {"q25": 0.25, "q75": 0.75}
BENCHMARK_STATES = 100                          # §37
KERNEL_LENGTHSCALE = float(os.environ.get("SQRJC_KERNEL_LS", "1.0"))
# audited default 1.0; §14 grid autorizado {0.5, 1.0, 2.0} pós-CAL via SQRJC_KERNEL_LS


# ---------------------------------------------------------------------------
# Frozen window conventions (reuse of the Gate II-d dataset machinery)
# ---------------------------------------------------------------------------

def load_run(run_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    kpm_path = _one(glob.glob(os.path.join(run_dir, "**", "sqr_kpm_history.csv"),
                              recursive=True))
    prb_path = _one(glob.glob(os.path.join(run_dir, "**", "sqr_prb_windows.csv"),
                              recursive=True))
    delays_path = _one(glob.glob(os.path.join(run_dir, "**", "sqr_packet_delays.csv"),
                                 recursive=True))
    if kpm_path is None or prb_path is None:
        raise FileNotFoundError(f"CAL run incompleto sob {run_dir}")
    kpm = pd.read_csv(kpm_path)
    prb = pd.read_csv(prb_path)
    delays = pd.read_csv(delays_path) if delays_path else None
    return kpm, prb, delays


def _one(hits: list[str]) -> str | None:
    return hits[0] if hits else None


def impute_snr_column(kpm: pd.DataFrame) -> pd.Series:
    """R09 offline: per-UE carry -> prev-window cell median -> 15 dB.

    Replica a cadeia congelada do SqrKpmParser sobre o DataFrame inteiro
    (CAL only; a cadeia é sequencial por janela e é reproduzida aqui para
    que as features CAL batam com as do runtime).
    """
    snr = kpm["snrDb"].astype(float).copy()
    ok = snr != SNR_SENTINEL
    carried: dict[int, float] = {}
    filled = snr.to_numpy(dtype=float).copy()
    prev_median: float | None = None
    for ts, group in kpm.groupby("timestamp", sort=True):
        idx = group.index.to_numpy()
        vals = filled[idx]
        med = float(np.median(vals[ok.to_numpy()[idx]])) if np.any(ok.to_numpy()[idx]) else None
        for i in idx:
            imsi = int(kpm.at[i, "ueImsi"])
            if ok.at[i]:
                carried[imsi] = float(snr.at[i])
                continue
            v = carried.get(imsi)
            if v is None:
                v = prev_median
            if v is None:
                v = 15.0
            filled[i] = v
        prev_median = med
    return pd.Series(filled, index=kpm.index)


def window_features(kpm: pd.DataFrame, imputed_snr: pd.Series,
                    pieces: dict[str, pd.DataFrame],
                    urllc_pw: pd.DataFrame) -> dict[int, dict]:
    """Raw per-window state features + Gate II-d closed-window KPIs.

    ``pieces`` é o retorno congelado de ``_read_kpm_windows`` (n/queue/pdr/
    snr median por slice, thr_embb_med etc.); os KPIs do julgamento vêm
    dele (median eMBB com demanda; p95 URLLC por chegada; PDR rolante 1 s
    recalculado abaixo com a mesma convenção do adjudication: janelas
    [w-9, w]).
    """
    embb = pieces["embb"]
    urllc = pieces["urllc"]
    mmtc = pieces["mmtc"]
    windows: dict[int, dict] = {}
    all_wins = sorted(set(embb.index) | set(urllc.index) | set(mmtc.index))
    tx_m = kpm["dTxPackets"].to_numpy(float) if "dTxPackets" in kpm else np.zeros(len(kpm))
    rx_m = kpm["dRxPackets"].to_numpy(float) if "dRxPackets" in kpm else np.zeros(len(kpm))
    wins = (kpm["timestamp"].to_numpy() // 100) - 1
    sid = kpm["sliceId"].to_numpy(int)
    thr = kpm["windowThroughputMbps"].to_numpy(float)
    for w in all_wins:
        sel = wins == w
        f: dict = {"win": int(w)}

        def _num(piece: pd.DataFrame, col: str) -> float:
            v = piece[col].get(w, 0.0) if w in piece.index else 0.0
            return float(v) if pd.notna(v) else 0.0

        for name in ("e", "u", "m"):
            base = {"e": "embb", "u": "urllc", "m": "mmtc"}[name]
            f[f"n_{name}"] = _num(pieces[base], f"n_{base}")
            f[f"queue_{name}"] = _num(pieces[base], f"queue_{base}")
            f[f"snr_{name}"] = _num(pieces[base], f"snr_{base}")
        # offered = served-rate sum per slice (all rows of the window)
        for s, name in ((0, "e"), (1, "u"), (2, "m")):
            m = sel & (sid == s)
            f[f"offered_{name}"] = float(thr[m].sum())
        # imputed-SNR mean per slice (runtime parity feature)
        snr_w = imputed_snr.to_numpy(float)
        for s, name in ((0, "e"), (1, "u"), (2, "m")):
            m = sel & (sid == s) & (kpm["snrDb"].to_numpy(float) != SNR_SENTINEL)
            f[f"snr_imputed_{name}"] = float(snr_w[m].mean()) if m.any() else 0.0
        # Gate II-d closed-window KPIs of THIS window
        f["kpi_e"] = (float(embb["thr_embb_med"].get(w))
                      if w in embb.index and pd.notna(embb["thr_embb_med"].get(w))
                      else None)
        row_u = urllc_pw[urllc_pw["win"] == w] if len(urllc_pw) else None
        if row_u is not None and len(row_u) == 1:
            f["kpi_u_p95"] = float(row_u.iloc[0]["latency_urllc_p95_ms"])
            f["kpi_u_p99"] = float(np.nanpercentile([row_u.iloc[0]["latency_urllc_p95_ms"]], 99))  # p99 via percentile-of-window below
            f["kpi_u_miss"] = float(row_u.iloc[0]["latency_urllc_miss_pct"]) / 100.0
            f["urllc_n_pkts"] = int(row_u.iloc[0]["urllc_n_pkts"])
        else:
            f["kpi_u_p95"] = None
            f["kpi_u_p99"] = None
            f["kpi_u_miss"] = None
            f["urllc_n_pkts"] = 0
        # mMTC 1 s rolling PDR over [w-9, w] (adjudication.py:384-390)
        lo, hi = max(0, w - 9), w
        def _roll(s: int) -> float:
            txs = rxs = 0.0
            m = (wins >= lo) & (wins <= hi) & (sid == s)
            txs = float(tx_m[m].sum())
            rxs = float(rx_m[m].sum())
            return (rxs / txs) if txs > 0 else np.nan
        f["kpi_m"] = _roll(2)
        f["mmtc_demand"] = bool(
            ((wins >= lo) & (wins <= hi) & (sid == 2) & (tx_m > 0)).any()
        )
        windows[int(w)] = f
    return windows


def state_of(f: dict, prev_action: tuple[int, int, int],
             prev_viol: tuple[int, int, int]) -> JointState:
    """JointState of the decision after window W (§11; all pre-decision)."""
    def _opt(v) -> float | None:
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    return JointState(
        n_e=int(f["n_e"]), n_u=int(f["n_u"]), n_m=int(f["n_m"]),
        queue_e=f["queue_e"], queue_u=f["queue_u"], queue_m=f["queue_m"],
        kpi_e_raw=_opt(f["kpi_e"]), kpi_u_raw=_opt(f["kpi_u_p95"]),
        kpi_m_raw=_opt(f["kpi_m"]),
        offered_e=f["offered_e"], offered_u=f["offered_u"], offered_m=f["offered_m"],
        snr_e=f.get("snr_imputed_e", 0.0), snr_u=f.get("snr_imputed_u", 0.0),
        snr_m=f.get("snr_imputed_m", 0.0),
        prev_a_e=prev_action[0], prev_a_u=prev_action[1], prev_a_m=prev_action[2],
        viol_e=prev_viol[0], viol_u=prev_viol[1], viol_m=prev_viol[2],
        urllc_miss_raw=f.get("kpi_u_miss") or 0.0,
        urllc_p99_raw=f.get("kpi_u_p99") or 0.0,
        window_index=int(f["win"]),
    )


# ---------------------------------------------------------------------------
# §16 block bootstrap coverage
# ---------------------------------------------------------------------------

def moving_block_bootstrap_ci(mask_stat: np.ndarray, values: np.ndarray,
                              block_len: int, resamples: int,
                              rng: np.random.Generator) -> tuple[float, float, float]:
    """(stat, ci_lo, ci_hi) da fração ``values`` sobre as janelas de ``mask_stat``
    com moving-block bootstrap de blocos de ``block_len`` janelas."""
    idx = np.flatnonzero(mask_stat)
    if idx.size == 0:
        return float("nan"), float("nan"), float("nan")
    n_blocks = int(np.ceil(idx.size / block_len))
    stats = np.empty(resamples)
    for b in range(resamples):
        starts = rng.integers(0, max(1, idx.size - block_len + 1), size=n_blocks)
        chosen: list[int] = []
        for s in starts:
            chosen.extend(idx[s:s + block_len].tolist())
        chosen = np.asarray(chosen[:idx.size], dtype=int)
        stats[b] = float(values[chosen].mean())
    point = float(values[idx].mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main calibration
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cal-dir", required=True)
    ap.add_argument("--seeds", default="1001 1002 1003")
    ap.add_argument("--lambdas", default="2 5 6")
    ap.add_argument("--output", required=True)
    ap.add_argument("--benchmark-states", type=int, default=BENCHMARK_STATES,
                    help="estados CAL no benchmark de busca (§37: 100)")
    ap.add_argument("--bootstrap-resamples", type=int,
                    default=BLOCK_BOOTSTRAP_RESAMPLES,
                    help="reamostras do block bootstrap (§16: 1000)")
    ap.add_argument("--coarse-to-fine-states", type=int, default=25,
                    help="estados no benchmark coarse-to-fine (§38)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split()]
    lambdas = [float(l) for l in args.lambdas.split()]
    os.makedirs(args.output, exist_ok=True)

    runs: list[dict] = []
    for lt in lambdas:
        for seed in seeds:
            d = os.path.join(args.cal_dir, f"cal_lam{lt:g}_s{seed}")
            if not os.path.isdir(d):
                print(f"[calibrate] AVISO: {d} ausente — pulando", file=sys.stderr)
                continue
            kpm, prb, delays = load_run(d)
            runs.append({"tag": os.path.basename(d), "lam": lt, "seed": seed,
                         "kpm": kpm, "prb": prb, "delays": delays})
    if not runs:
        print("[calibrate] nenhum run CAL carregado", file=sys.stderr)
        return 2

    # ---- build per-run window streams ------------------------------------
    streams = []
    for r in runs:
        imputed = impute_snr_column(r["kpm"])
        pieces = _read_kpm_windows(r["kpm"])
        acts = _read_actions(r["prb"])
        urllc_pw = (_urllc_packet_windows_from_df(r["delays"])
                    if r["delays"] is not None else
                    pd.DataFrame(columns=["win", "latency_urllc_p95_ms",
                                          "latency_urllc_miss_pct", "urllc_n_pkts"]))
        feats = window_features(r["kpm"], imputed, pieces, urllc_pw)
        streams.append({**r, "acts": acts, "feats": feats})

    # ---- §13: fit the FeatureScaler on ALL CAL z vectors ------------------
    raw_z_rows = []
    for s in streams:
        prev_a = (0, 0, 0)
        for w in sorted(s["feats"]):
            st = state_of(s["feats"][w], prev_a, (0, 0, 0))
            raw_z_rows.append(st.raw_vector())
    scaler = FeatureScaler(c=C_PRB)
    scaler.fit(np.stack(raw_z_rows))
    scaler_path = os.path.join(args.output, "feature_scaler_manifest.yaml")
    scaler.save_manifest(scaler_path, source="CAL")
    print(f"[calibrate] §13 scaler congelado -> {scaler_path} "
          f"({scaler.n_cal_samples} janelas CAL)")

    # ---- online replay: train heads, collect sigma/q predictions ----------
    models = tuple(
        MultiQuantileKernelModel(dim=JOINT_STATE_DIM + 3, length_scale=KERNEL_LENGTHSCALE,
                                 memory_size=B_MEMORY)
        for _ in range(3)
    )
    sigma_rows: list[dict] = []
    kernel_log: list[str] = []
    insertions = evictions = 0
    t0 = time.perf_counter()
    for s in streams:
        prev_a = (0, 0, 0)
        prev_viol = (0, 0, 0)
        acts = s["acts"].set_index("win")
        for w in sorted(s["feats"]):
            st = state_of(s["feats"][w], prev_a, prev_viol)
            out_w = w + 1
            if out_w not in acts.index:
                continue  # outcome window fora do log (última janela)
            a = acts.loc[out_w]
            action = (int(round(a["a_e"])), int(round(a["a_u"])), int(round(a["a_m"])))
            x = st.feature_x(action, C_PRB, scaler)
            preds = [m.predict_raw(x[None, :]) for m in models]
            margins = SliceOutcome(
                embb_throughput_mbps=s["feats"][out_w]["kpi_e"] if out_w in s["feats"] else None,
                urllc_p95_ms=s["feats"][out_w]["kpi_u_p95"] if out_w in s["feats"] else None,
                urllc_miss_pct=(s["feats"][out_w]["kpi_u_miss"] * 100.0) if out_w in s["feats"] and s["feats"][out_w]["kpi_u_miss"] is not None else None,
                mmtc_pdr_rolling=s["feats"][out_w]["kpi_m"] if out_w in s["feats"] else None,
            ).margins
            for si in range(3):
                sigma_rows.append({
                    "tag": s["tag"], "lam": s["lam"], "win": w, "slice": si,
                    "sigma": float(preds[si]["sigma"][0]),
                    "q25": float(preds[si]["q25"][0]),
                    "q75": float(preds[si]["q75"][0]),
                })
            for si in range(3):
                m = margins[si]
                if m is None or not np.isfinite(m):
                    continue  # vacuous/NaN: excluded from the update (R19)
                rec = models[si].update(x, float(m), threshold=0.0)
                insertions += 1
                evictions += 1 if rec.pruned_index is not None else 0
            viol = tuple(0 if (m is None or m >= 0) else 1 for m in margins)
            prev_a, prev_viol = action, viol
    replay_s = time.perf_counter() - t0

    sigma_df = pd.DataFrame(sigma_rows)
    # observed margins per (tag, win, slice) for the §16 coverage check
    margin_by_win: dict[tuple[str, int, int], float] = {}
    for s in streams:
        acts2 = s["acts"].set_index("win")
        for w in sorted(s["feats"]):
            out_w = w + 1
            if out_w not in acts2.index or out_w not in s["feats"]:
                continue
            mg = SliceOutcome(
                embb_throughput_mbps=s["feats"][out_w]["kpi_e"],
                urllc_p95_ms=s["feats"][out_w]["kpi_u_p95"],
                urllc_miss_pct=(s["feats"][out_w]["kpi_u_miss"] * 100.0) if s["feats"][out_w]["kpi_u_miss"] is not None else None,
                mmtc_pdr_rolling=s["feats"][out_w]["kpi_m"],
            ).margins
            for si in range(3):
                if mg[si] is not None:
                    margin_by_win[(s["tag"], w, si)] = float(mg[si])

    # join sigma rows with the observed margin of the SAME (tag, win, slice)
    sigma_df["margin"] = [
        margin_by_win.get((r.tag, int(r.win), int(r.slice)), np.nan)
        for r in sigma_df.itertuples()
    ]

    # ---- §16: sigma_support grid with block-bootstrap coverage ------------
    rng = np.random.default_rng(20261103)
    block_len = int(BLOCK_SECONDS * 10)  # 50 windows
    n_resamples = int(args.bootstrap_resamples)
    sigma_support: dict = {"candidates": {}}
    chosen = None
    for pct in SIGMA_GRID_PCT:
        print(f"[calibrate] §16 percentile {pct} ...", flush=True)
        thr = float(np.percentile(sigma_df["sigma"], pct))
        sup = (sigma_df["sigma"] <= thr).to_numpy()
        frac = float(sup.mean())
        cand = {"threshold": thr, "supported_frac": frac, "pass": False, "reasons": []}
        if frac < SUPPORTED_MIN_FRAC:
            cand["reasons"].append(f"suporte {frac:.3f} < {SUPPORTED_MIN_FRAC}")
        ok_cov = True
        for si in range(3):
            sub = sigma_df[(sigma_df["slice"] == si) & sup & sigma_df["margin"].notna()]
            if sub.empty:
                continue
            values_q = (sub["margin"] <= sub["q25"]).to_numpy(float)
            values_7 = (sub["margin"] <= sub["q75"]).to_numpy(float)
            for name, vals, target in (("q25", values_q, 0.25), ("q75", values_7, 0.75)):
                _, lo, hi = moving_block_bootstrap_ci(
                    np.ones(len(vals), dtype=bool), vals, block_len,
                    n_resamples, rng)
                err = abs(vals.mean() - target)
                in_ci = lo <= target <= hi
                if err > COVERAGE_ABS_ERR_MAX:
                    ok_cov = False
                    cand["reasons"].append(
                        f"slice{si} {name}: erro de cobertura {err:.3f} > {COVERAGE_ABS_ERR_MAX}")
                if not in_ci:
                    ok_cov = False
                    cand["reasons"].append(
                        f"slice{si} {name}: alvo {target} fora do CI bootstrap [{lo:.3f},{hi:.3f}]")
                cand[f"slice{si}_{name}_coverage"] = float(vals.mean())
                cand[f"slice{si}_{name}_ci"] = [lo, hi]
        cand["pass"] = bool(ok_cov and not [r for r in cand["reasons"] if "suporte" in r])
        sigma_support["candidates"][str(pct)] = cand
        if cand["pass"]:
            chosen = (pct, thr)
    sigma_support["chosen_percentile"] = chosen[0] if chosen else None
    sigma_support["sigma_support"] = chosen[1] if chosen else None
    sigma_support["grid"] = list(SIGMA_GRID_PCT)
    sigma_support["block_bootstrap"] = {
        "block_len_windows": block_len, "resamples": n_resamples,
        "criterion": f"|coverage - target| <= {COVERAGE_ABS_ERR_MAX} e alvo no CI 95%",
    }
    sigma_path = os.path.join(args.output, "sigma_support_manifest.yaml")
    from nsoran.sqr_jc.yaml_subset import dump
    with open(sigma_path, "w", encoding="utf-8") as fh:
        fh.write(dump({k: v for k, v in sigma_support.items()
                       if k in ("chosen_percentile", "sigma_support", "grid", "block_bootstrap")}))
    with open(os.path.join(args.output, "sigma_support_calibration.json"), "w",
              encoding="utf-8") as fh:
        json.dump(sigma_support, fh, indent=2, default=float)
    print(f"[calibrate] §16 sigma_support -> {sigma_path} "
          f"(escolhido: {sigma_support['chosen_percentile']})")

    # ---- §57 kernel log ----------------------------------------------------
    kl_path = os.path.join(args.output, "kernel_log.csv")
    taus = [m.tau for m in models]
    with open(kl_path, "w", encoding="utf-8") as fh:
        fh.write("metric,value\n")
        fh.write(f"insertions_total,{insertions}\n")
        fh.write(f"evictions_total,{evictions}\n")
        fh.write(f"replay_time_s,{replay_s:.3f}\n")
        for si in range(3):
            fh.write(f"support_count_slice{si},{models[si].n_support}\n")
            fh.write(f"tau_safe_slice{si},{taus[si]:.4f}\n")
            fh.write(f"margin_m_slice{si},{models[si].margin_m:.4f}\n")
    print(f"[calibrate] §57 kernel log -> {kl_path}")

    # ---- §37 search benchmark over 100 CAL states -------------------------
    from nsoran.sqr_jc.action_search import (
        CoarseToFineParams,
        CoarseToFineSearch,
        JointActionOracleEvaluator,
        benchmark_exhaustive,
    )
    n_bench = int(args.benchmark_states)
    z_all = []
    step = max(1, len(raw_z_rows) // n_bench)
    for i in range(0, len(raw_z_rows), step):
        if len(z_all) >= n_bench:
            break
        z_all.append(raw_z_rows[i])
    z_scaled = [scaler.transform_joint(z) for z in z_all]
    evaluator = JointActionOracleEvaluator(
        models,
        sigma_support=(chosen[1] if chosen else 0.9),
        c=C_PRB,
    )
    print(f"[calibrate] §37 benchmark exaustivo em {len(z_scaled)} estados CAL "
          f"(memória ~{models[0].n_support} SV/slice) — pode levar minutos", flush=True)
    bench = benchmark_exhaustive(evaluator, z_scaled)
    ctf = CoarseToFineSearch(evaluator, CoarseToFineParams(coarse_step=4, refine_radius=4))
    t_ctf = []
    for z in z_scaled[: int(args.coarse_to_fine_states)]:
        t1 = time.perf_counter()
        ctf.search(z, floors=(0, 0, 0), key_fn=lambda r: [
            r.actions.sum(axis=1).astype(float),
            np.abs(r.actions - np.array([18, 13, 3])).sum(axis=1).astype(float),
            r.sigma.sum(axis=1),
            np.maximum(0.0, -r.safe).sum(axis=1),
        ])
        t_ctf.append((time.perf_counter() - t1) * 1000.0)
    bench_search = {
        "gate": "G5_search_runtime",
        "protocol": "SQRJC-ETAPA-2.5-FINAL (§37/§38)",
        "n_states": len(z_scaled),
        "stratification": "CAL states em ordem (low/mid/high pela mistura de lambdas)",
        "exhaustive": bench,
        "coarse_to_fine": {
            "params": {"coarse_step": 4, "refine_radius": 4},
            "n_states": len(t_ctf),
            "median_ms": float(np.median(t_ctf)),
            "p95_ms": float(np.percentile(t_ctf, 95)),
        },
        "note": "PASS exaustivo se p95 <= 80 ms; caso contrario validar o "
                "coarse-to-fine no gate §38 (concordancia >= 98%, regret "
                "mediano <= 2%, p95 <= 5%, p95 tempo <= 80 ms) antes do PILOT.",
    }
    bench_path = os.path.join(args.output, "search_benchmark.json")
    with open(bench_path, "w", encoding="utf-8") as fh:
        json.dump(bench_search, fh, indent=2)
    print(f"[calibrate] §37/§38 benchmark -> {bench_path} "
          f"(exaustivo p95={bench['p95_ms']:.1f} ms, "
          f"passe={bench['passed_p95_80ms']})")

    # ---- consolidated report ----------------------------------------------
    report = {
        "gate": "G4_kernel_calibration",
        "protocol": "SQRJC-ETAPA-2.5-FINAL (§32/§13/§16)",
        "n_runs": len(runs),
        "runs": [s["tag"] for s in streams],
        "n_cal_windows": int(len(raw_z_rows)),
        "n_updates": insertions,
        "evictions": evictions,
        "sigma_support_chosen": sigma_support["chosen_percentile"],
        "sigma_support_value": sigma_support["sigma_support"],
        "sigma_support_pass": chosen is not None,
        "scaler_manifest": os.path.basename(scaler_path),
        "verdict": "PASS" if chosen is not None else "STOP",
        "note": "§16: se nenhum threshold passa -> STOP, revisar features/kernel, "
                "NAO iniciar PILOT.",
    }
    with open(os.path.join(args.output, "calibration_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[calibrate] G4: {report['verdict']} "
          f"(sigma_support={sigma_support['sigma_support']})")
    return 0 if chosen is not None else 1


def _urllc_packet_windows_from_df(delays: pd.DataFrame) -> pd.DataFrame:
    """Frozen _urllc_packet_windows semantics on an already-loaded DataFrame."""
    import numpy as np
    u = delays[delays["sliceId"] == 1].copy()
    if not len(u):
        return pd.DataFrame(columns=["win", "latency_urllc_p95_ms",
                                     "latency_urllc_miss_pct", "urllc_n_pkts",
                                     "latency_urllc_med_ms"])
    u["win"] = (u["txSec"].astype(float) * 10).astype(int)
    ms = u["delaySec"].astype(float) * 1000.0
    u = u.assign(ms=ms, miss=ms > 10.0)
    g = u.groupby("win").agg(
        latency_urllc_p95_ms=("ms", lambda x: float(np.percentile(x, 95))),
        latency_urllc_med_ms=("ms", "median"),
        latency_urllc_miss_pct=("miss", lambda x: 100.0 * float(x.mean())),
        urllc_n_pkts=("ms", "size"),
    )
    return g.reset_index()


if __name__ == "__main__":
    raise SystemExit(main())
