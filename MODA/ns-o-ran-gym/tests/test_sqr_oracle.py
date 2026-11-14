"""Tests for the Etapa-2.4 offline oracle (Despacho I) — no ns-3.

Covers ``nsoran.sqr.oracle`` (protocol
``docs/artigo/rodada_2_etapa_2_4_oracle_viabilidade.md``, §33 items 1-6):

- action simplex enumeration (§7): combinatorial count, Σ ≤ C, ordering, timing;
- ``window_feasibility`` (§2/§9/§10/§13): F_t closed form vs BRUTE-FORCE over
  the enumerated simplex, best-degradation a* vs brute-force argmin, §9
  deficit arithmetic, per-slice saveability;
- KPI models (§6): hand-checked stratum lookups, quantile gates, out-of-support
  fallback/clamp, vacuous demand;
- leave-one-seed-out validation plumbing;
- per-window dataset builder (§4): window alignment, demand counting, offered
  proxies, executed-action extraction (IPC vs network-only budget accumulator),
  util D1/D2 frozen formulas, packet-delay arrival-window binning, observed
  violations — all on a hand-computed synthetic mini-run;
- §19 column order of the main oracle CSV;
- integrity: the full 30-run dataset REPRODUCES the frozen Etapa-2.3 run-level
  aggregates (``round2_hetero_sweep.csv``) — skipped automatically if the
  campaign data is absent.
"""

import glob
import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_TEST_DIR, "..", "src")
sys.path.insert(0, _SRC)

from nsoran.sqr.oracle import dataset as ds  # noqa: E402
from nsoran.sqr.oracle.models import (  # noqa: E402
    EMMB_SHARE_EDGES,
    EmmbShareLookup,
    MmtcConstantPdr,
    UrllcStrataLookup,
    loo_validate,
)
from nsoran.sqr.oracle.pipeline import (  # noqa: E402
    aggregate_by_lambda,
    classify_windows,
    critical_windows,
    hypotheses,
    oracle_actions_table,
    oracle_windows_extra_table,
    oracle_windows_table,
    sanity_checks,
)
from nsoran.sqr.oracle.simplex import (  # noqa: E402
    C_PRB,
    action_count,
    enumeration_timing,
    enumerate_actions,
    window_feasibility,
)

ROUND2_DIR = os.path.join(_TEST_DIR, "..", "results", "sqr_characterization_round2")
HAS_CAMPAIGN = os.path.isdir(os.path.join(ROUND2_DIR, "hetero_sweep")) and \
    os.path.isdir(os.path.join(ROUND2_DIR, "hetero_sweep_prop"))


# ---------------------------------------------------------------------------
# §7 — action simplex
# ---------------------------------------------------------------------------

def test_simplex_enumeration_count_and_constraints():
    acts = enumerate_actions(C_PRB, 1)
    assert acts.shape == (27720, 3)               # C(56,3)
    assert action_count(C_PRB, 1) == 27720
    assert (acts.sum(axis=1) <= C_PRB).all()
    assert (acts >= 0).all()
    # lexicographic order by (a_e, a_u, a_m)
    order = np.lexsort((acts[:, 2], acts[:, 1], acts[:, 0]))
    assert (order == np.arange(len(acts))).all()


def test_simplex_enumeration_step2_matches_combinatorics():
    assert len(enumerate_actions(53, 2)) == action_count(53, 2)


def test_enumeration_timing_step1_is_reported():
    t = enumeration_timing(C_PRB, 1, repeats=1)
    assert t["n_actions"] == 27720
    assert t["sum_le_c_ok"] and t["min_ok"]
    assert t["seconds_per_enumeration"] < 5.0  # protocol §7: step 1 fits


# ---------------------------------------------------------------------------
# models — hand-checked lookups (§6)
# ---------------------------------------------------------------------------

@pytest.fixture()
def mini_windows():
    return pd.DataFrame({
        "arm": ["equal"] * 16, "lam": [4.0] * 16, "seed": [1] * 16,
        "win": np.arange(16), "a_e": [18.0] * 16, "a_u": [17.0] * 16,
        "a_m": [18.0] * 16,
        "n_embb": [1, 2, 3, 4] * 4,
        "thr_embb_med": [7.2, 6.0, 4.2, 1.6] * 4,
        "cqi_embb": [14.0] * 16,
        "n_urllc": [1, 2, 3, 1] * 4,
        "n_mmtc": [1] * 16,
        "latency_urllc_p95_ms": [8.0, 11.0, 13.0, 9.0] * 4,
        "latency_urllc_miss_pct": [1.0, 10.0, 15.0, 2.0] * 4,
        "observed_embb_violation": [0.0] * 16,
        "observed_urllc_violation": [0.0] * 16,
        "observed_mmtc_violation": [0.0] * 16,
        "pdr_mmtc": [1.0] * 16,
    })


def test_emmb_lookup_median_and_clamp(mini_windows):
    me = EmmbShareLookup().fit(mini_windows)
    # s = 18/2 = 9 -> trained cell (cqi 14, s in [8,12)) -> median of {6.0 x4}
    assert me.predict([18.0], [2.0], [14.0])[0] == pytest.approx(6.0)
    # beyond-range share (a_e=53, N=1 -> s=53, bin [40,inf) empty in training)
    # falls back to the NEAREST supported bin (flat saturation extrapolation)
    assert me.predict([53.0], [1.0], [14.0])[0] == pytest.approx(7.2)
    # no demand -> KPI undefined (NaN), whatever the action
    assert np.isnan(me.predict([53.0], [0.0], [14.0])[0])


def test_emmb_lookup_fallback_and_support_flag(mini_windows):
    me = EmmbShareLookup().fit(mini_windows)
    # unseen (cqi=13, s) cell -> marginal of the s bin
    vals, supp = me.predict_with_support([18.0], [2.0], [13.0])
    assert vals[0] == pytest.approx(6.0)      # same s bin marginal (only cqi 14 data)
    assert not supp[0]                        # flagged out-of-support
    vals2, supp2 = me.predict_with_support([18.0], [2.0], [14.0])
    assert supp2[0]


def test_urllc_strata_median_and_vacuous(mini_windows):
    mu = UrllcStrataLookup().fit(mini_windows)
    p95, miss = mu.predict([2], [2])
    assert p95[0] == pytest.approx(11.0)      # median of {11.0 x4}
    assert miss[0] == pytest.approx(10.0)
    p95v, missv = mu.predict([0], [3])
    assert p95v[0] == 0.0 and missv[0] == 0.0  # vacuous without demand


def test_urllc_strata_quantile_gate(mini_windows):
    mu = UrllcStrataLookup().fit(mini_windows)
    p25, _ = mu.predict([1], [1], q=0.25)
    p75, _ = mu.predict([1], [1], q=0.75)
    # all values 8/9 -> p25 = p75 = 8.0..9.0 range endpoints by interpolation
    assert p25[0] <= 9.0 and p75[0] >= 8.0


def test_mmtc_constant_pdr(mini_windows):
    mm = MmtcConstantPdr().fit(mini_windows)
    assert mm.predict([1])[0] == 1.0
    assert mm.measured_min_pdr == 1.0


# ---------------------------------------------------------------------------
# window_feasibility — F_t / a* / deficits vs brute force (§2/§9/§10/§13)
# ---------------------------------------------------------------------------

def _brute_force(me, mu, mm, n_e, n_u, n_m, cqi):
    acts = enumerate_actions(C_PRB, 1)
    thr = me.predict(acts[:, 0].astype(float), np.full(len(acts), float(n_e)),
                     np.full(len(acts), cqi))
    p95, _ = mu.predict(np.full(len(acts), n_u), np.full(len(acts), n_e))
    ok_e = (thr >= 3.5) if n_e >= 1 else np.ones(len(acts), bool)
    ok_u = (p95 <= 10.0) if n_u >= 1 else np.ones(len(acts), bool)
    ok_m = np.ones(len(acts), bool)  # PDR model == 1.0
    feas = ok_e & ok_u & ok_m
    d_e = (np.maximum(0.0, 3.5 - thr) / 3.5) if n_e >= 1 else np.zeros(len(acts))
    d_u = (np.maximum(0.0, p95 - 10.0) / 10.0) if n_u >= 1 else np.zeros(len(acts))
    total = d_e + d_u
    a_star = int(np.argmin(total))
    return int(feas.sum()), tuple(acts[a_star]), float(total[a_star])


@pytest.mark.parametrize("n_e,n_u", [(1, 1), (2, 2), (3, 3), (6, 3)])
def test_window_feasibility_matches_brute_force(mini_windows, n_e, n_u):
    me = EmmbShareLookup().fit(mini_windows)
    mu = UrllcStrataLookup().fit(mini_windows)
    mm = MmtcConstantPdr().fit(mini_windows)
    r = window_feasibility(me, mu, mm, n_embb=n_e, n_urllc=n_u, n_mmtc=1, cqi=14.0)
    bf_n, bf_star, bf_dmin = _brute_force(me, mu, mm, n_e, n_u, 1, 14.0)
    assert r.num_feasible_actions == bf_n
    assert r.best_deg_action == bf_star
    assert r.min_total_deficit == pytest.approx(bf_dmin, abs=1e-9)
    # witness consistency: e_min is the smallest feasible a_e and is feasible
    if r.joint_feasible:
        assert r.e_min >= 0
        assert r.witness[0] == r.e_min
        assert sum(r.witness) <= C_PRB


def test_deficit_math_section9(mini_windows):
    """§9 arithmetic on a hand-computed state: D = [Λ − y]+/Λ (throughput)."""
    me = EmmbShareLookup().fit(mini_windows)
    mu = UrllcStrataLookup().fit(mini_windows)
    mm = MmtcConstantPdr().fit(mini_windows)
    # N_e = 3, a_e = 18 -> s = 6 -> thr_hat = 4.2 (median of the s bin [6,8))
    r = window_feasibility(me, mu, mm, n_embb=3, n_urllc=3, n_mmtc=1, cqi=14.0)
    # URLLC stratum (N_u=3+, N_e=2+) median p95 = 13.0 -> D_u = 0.3 (infeasible)
    assert r.deficits[1] == pytest.approx(0.3)
    assert not r.joint_feasible
    assert r.saveable == (True, False, True)  # eMBB saveable alone, URLLC not
    assert r.deficits[2] == 0.0


def test_vacuous_window_fully_feasible(mini_windows):
    me = EmmbShareLookup().fit(mini_windows)
    mu = UrllcStrataLookup().fit(mini_windows)
    mm = MmtcConstantPdr().fit(mini_windows)
    r = window_feasibility(me, mu, mm, n_embb=0, n_urllc=0, n_mmtc=0, cqi=14.0)
    assert r.joint_feasible
    assert r.num_feasible_actions == 27720  # whole simplex


# ---------------------------------------------------------------------------
# LOO plumbing (§6 validation split)
# ---------------------------------------------------------------------------

def test_loo_validate_runs_and_covers(mini_windows):
    w = mini_windows.copy()
    w["seed"] = np.repeat([1, 2], 8)
    w["win"] = np.tile(np.arange(8), 2)
    out = loo_validate(w)
    assert set(out["loo"]["held_seed"]) == {1, 2}
    assert len(out["loo_windows"]) == len(w)
    # in the per-window LOO export, `seed` IS the held-out seed (each window is
    # predicted by models trained on the OTHER seeds)
    assert set(out["loo_windows"]["seed"]) == {1, 2}
    assert (out["loo"]["embb_n"] > 0).all()


# ---------------------------------------------------------------------------
# dataset builder — synthetic mini-run with hand-computed expectations (§4)
# ---------------------------------------------------------------------------

KPM_HEADER = ("timestamp,ueImsi,sliceId,dTxBytes,dRxBytes,plr,resourceSharePct,"
              "dLostPackets,throughputMbps,bufferBytes,windowThroughputMbps,snrDb,"
              "queueBytes,dTxPackets,dRxPackets,dDelaySumSec,cqiWideband")


def _mini_run(root, lam=4, seed=1):
    tag = f"A_lam{lam}_s{seed}"
    # build_window_dataset expects {root}/hetero_sweep/A_lamX_sY/...
    run = os.path.join(root, "hetero_sweep", tag, "results_sqr_network_only",
                       f"seed={seed}_run=1")
    os.makedirs(run, exist_ok=True)
    # --- KPM: win 1 (ts 200) and win 2 (ts 300) ---
    rows = [
        # ts, ue, slice, dTx, dRx, plr, rsh, lost, thr, buf, winThr, snr, q, txP, rxP, dSum, cqi
        (200, 1, 0, 1000, 900, 10.0, 30.0, 2, 4.0, 500, 4.0, 20.0, 500, 10, 9, 0.05, 14),
        (200, 2, 1, 320, 320, 0.0, 30.0, 0, 0.26, 0, 0.26, 21.0, 0, 100, 100, 0.6, 14),
        (300, 1, 0, 1000, 500, 50.0, 30.0, 5, 2.0, 900, 3.0, 20.0, 900, 10, 5, 0.09, 13),
        (300, 3, 2, 200, 200, 0.0, 5.0, 0, 0.0016, 0, 0.0016, 19.0, 0, 1, 1, 0.01, 14),
    ]
    with open(os.path.join(run, "sqr_kpm_history.csv"), "w") as f:
        f.write(KPM_HEADER + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    # --- prb_windows: win 1 network-only, win 2 IPC ---
    with open(os.path.join(run, "sqr_prb_windows.csv"), "w") as f:
        f.write("timestamp_ms,total_rbg,total_prb,dl_slots,"
                "prb_requested_pct_0,prb_requested_pct_1,prb_requested_pct_2,"
                "prb_executed_rbg_0,prb_executed_rbg_1,prb_executed_rbg_2,"
                "prb_scheduled_rbg_0,prb_scheduled_rbg_1,prb_scheduled_rbg_2,"
                "prb_budget_rbg_0,prb_budget_rbg_1,prb_budget_rbg_2,prb_idle_rbg\n")
        # win 1 (ts 200): network-only, budget = quota x 56 slots
        f.write("200,53,53,56,-1.00,-1.00,-1.00,0,0,0,900,100,5,"
                f"{18*56},{17*56},{18*56},{53*56-1005}\n")
        # win 2 (ts 300): IPC action (37,13,3)
        f.write("300,53,53,56,84.91,24.53,5.66,37,13,3,1800,200,10,"
                f"{37*56},{13*56},{3*56},{53*56-2010}\n")
    # --- packet delays: URLLC pkts in win 1 (txSec 0.10x) and win 2 ---
    with open(os.path.join(run, "sqr_packet_delays.csv"), "w") as f:
        f.write("ueImsi,sliceId,txSec,delaySec\n")
        f.write("2,1,0.100,0.005\n")     # win 1, ok
        f.write("2,1,0.105,0.012\n")     # win 1, miss
        f.write("2,1,0.200,0.006\n")     # win 2, ok
    return tag


def test_dataset_builder_mini_run_hand_checks(tmp_path):
    tag = _mini_run(str(tmp_path))
    df = ds.build_window_dataset(str(tmp_path))
    assert len(df) == 2  # two prb_windows rows -> wins 0 and 1
    df = df.set_index("win")

    # ---- win 1 (ts 200): network-only action recovered EXACTLY from budget ----
    w1 = df.loc[1]
    assert w1["a_e"] == 18.0 and w1["a_u"] == 17.0 and w1["a_m"] == 18.0
    assert w1["action_source"] == "budget_derived"
    # state: eMBB UE1 + URLLC UE2 with demand; mMTC none in this window
    assert w1["n_embb"] == 1 and w1["n_urllc"] == 1
    assert pd.isna(w1["n_mmtc"])
    assert w1["thr_embb_med"] == pytest.approx(4.0)
    assert w1["queue_embb"] == 500
    assert w1["pdr_embb"] == pytest.approx(0.9)
    assert w1["n_embb_backlog"] == 1
    assert w1["offered_load_embb"] == pytest.approx(3.5)          # R36 proxy
    assert w1["offered_load_urllc"] == pytest.approx(32 * 8 / 1e3 / 1e-3 / 1e6)
    # packet delays (arrival window 1): 5 ms ok + 12 ms miss
    assert w1["urllc_n_pkts"] == 2
    assert w1["latency_urllc_miss_pct"] == pytest.approx(50.0)
    assert w1["latency_urllc_p95_ms"] == pytest.approx(np.percentile([5.0, 12.0], 95))
    assert w1["observed_urllc_violation"] == 1.0                  # p95 > 10
    assert w1["observed_embb_violation"] == 0.0                   # 4.0 >= 3.5
    # utils (frozen 2.3 formulas)
    assert w1["util_active_dl"] == pytest.approx(100 * 1005 / (53 * 56))
    assert w1["util_all_dl"] == pytest.approx(100 * 1005 / (53 * (400 / 7)))
    assert w1["prb_idle"] == 53 * 56 - 1005

    # ---- win 2 (ts 300): IPC action taken verbatim; eMBB 3.0 -> violation ----
    w2 = df.loc[2]
    assert w2["a_e"] == 37.0 and w2["a_u"] == 13.0 and w2["a_m"] == 3.0
    assert w2["action_source"] == "ipc"
    assert w2["n_embb"] == 1 and w2["n_mmtc"] == 1
    assert w2["thr_embb_med"] == pytest.approx(3.0)
    assert w2["observed_embb_violation"] == 1.0
    assert w2["observed_urllc_violation"] == 0.0                  # 6 ms <= 10
    # mMTC: the R38 window analog uses the 1 s rolling PDR — computed by
    # add_rolling_pdr (called by the pipeline); exercise it here explicitly
    kpm_path = sorted(glob.glob(os.path.join(str(tmp_path), "**", "sqr_kpm_history.csv"),
                                recursive=True))[0]
    kpm = pd.read_csv(kpm_path)
    cache = {("equal", 4.0, 1): ds.per_run_packet_counts(kpm, df.index.values)}
    df = ds.add_rolling_pdr(df.reset_index(), cache).set_index("win")
    w1 = df.loc[1]
    w2 = df.loc[2]
    assert w2["pdr_mmtc_1s"] == pytest.approx(1.0)
    assert w2["observed_mmtc_violation"] == 0.0
    # any-violation OR of the three
    assert w2["observed_any_violation"] == 1.0
    assert w1["observed_any_violation"] == 1.0


def test_add_rolling_pdr_weighted_10_window():
    wins = pd.DataFrame({
        "arm": ["equal"] * 12, "lam": [4.0] * 12, "seed": [1] * 12,
        "win": np.arange(12),
    })
    wins["pdr_urllc_1s"] = np.nan
    wins["pdr_mmtc_1s"] = np.nan
    counts = pd.DataFrame({
        "tx_urllc": np.ones(12) * 100, "rx_urllc":
            np.array([100.] * 9 + [50.] + [100.] * 2),
        "tx_mmtc": np.zeros(12), "rx_mmtc": np.zeros(12),
    })
    counts.index.name = "win"
    out = ds.add_rolling_pdr(wins, {("equal", 4.0, 1): counts})
    # window 9: rolling 10-window rx = 9*100 + 50 = 950 / tx 1000 = 0.95
    assert out.set_index("win").loc[9, "pdr_urllc_1s"] == pytest.approx(0.95)
    # window 8: only 9 windows so far (0..8), all delivered -> 1.0
    assert out.set_index("win").loc[8, "pdr_urllc_1s"] == pytest.approx(1.0)
    # window 10: the lossy window 9 is still inside the 10-window fold -> 0.95
    assert out.set_index("win").loc[10, "pdr_urllc_1s"] == pytest.approx(0.95)
    # window 11: covers 2..11 -> still contains window 9 -> 0.95
    assert out.set_index("win").loc[11, "pdr_urllc_1s"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# §19 column order + pipeline table writers
# ---------------------------------------------------------------------------

def _minimal_classified(n=3):
    cl = pd.DataFrame({
        "seed": [1] * n, "time_ms": (np.arange(n) + 1) * 100, "lam": [4.0] * n,
        "arm": ["equal"] * n, "tag": ["A_lam4_s1"] * n, "win": np.arange(n),
        "n_embb": [1] * n, "n_urllc": [1] * n, "n_mmtc": [1] * n,
        "n_embb_present": [1] * n, "n_urllc_present": [1] * n,
        "n_mmtc_present": [1] * n, "n_embb_backlog": [1] * n,
        "offered_load_embb": [3.5] * n, "offered_load_urllc": [0.256] * n,
        "offered_load_mmtc": [0.0016] * n,
        "queue_embb": [100.0] * n, "queue_urllc": [0.0] * n, "queue_mmtc": [0.0] * n,
        "thr_embb_med": [4.0] * n, "thr_embb_mean": [4.0] * n, "thr_embb_min": [4.0] * n,
        "thr_embb_uewins": [1] * n, "thr_embb_violfrac": [0.0] * n,
        "observed_embb_violation": [0.0] * n, "observed_embb_violation_anyue": [0.0] * n,
        "latency_urllc_p95_ms": [8.0] * n, "latency_urllc_med_ms": [5.0] * n,
        "latency_urllc_miss_pct": [1.0] * n, "urllc_n_pkts": [100] * n,
        "observed_urllc_violation": [0.0] * n, "observed_mmtc_violation": [0.0] * n,
        "observed_any_violation": [0.0] * n,
        "pdr_urllc": [1.0] * n, "pdr_mmtc": [1.0] * n,
        "pdr_urllc_1s": [1.0] * n, "pdr_mmtc_1s": [1.0] * n,
        "snr_embb": [20.0] * n, "snr_urllc": [21.0] * n, "snr_mmtc": [19.0] * n,
        "cqi_embb": [14.0] * n, "cqi_urllc": [14.0] * n, "cqi_mmtc": [14.0] * n,
        "cqi_eff": [14.0] * n, "cqi_bin": [3] * n,
        "total_rbg": [53] * n, "dl_slots": [56] * n, "action_source": ["ipc"] * n,
        "prb_requested": [53.0] * n, "prb_scheduled": [2010.0] * n,
        "prb_executed": [53.0] * n, "prb_budget": [53 * 56] * n,
        "prb_idle": [53 * 56 - 2010.0] * n,
        "prb_requested_pct_0": [84.9] * n, "prb_requested_pct_1": [24.5] * n,
        "prb_requested_pct_2": [5.7] * n,
        "prb_scheduled_rbg_0": [1800.0] * n, "prb_scheduled_rbg_1": [200.0] * n,
        "prb_scheduled_rbg_2": [10.0] * n,
        "prb_budget_rbg_0": [37 * 56] * n, "prb_budget_rbg_1": [13 * 56] * n,
        "prb_budget_rbg_2": [3 * 56] * n,
        "prb_executed_rbg_0": [37] * n, "prb_executed_rbg_1": [13] * n,
        "prb_executed_rbg_2": [3] * n,
        "a_e": [37] * n, "a_u": [13] * n, "a_m": [3] * n,
        "util_active_dl": [67.0] * n, "util_all_dl": [53.0] * n,
        "joint_feasible": [True] * n, "num_feasible_actions": [27720] * n,
        "e_min": [8] * n, "urllc_gate": [True] * n,
        "witness_a_e": [8] * n, "witness_a_u": [13] * n, "witness_a_m": [32] * n,
        "a_star_e": [8] * n, "a_star_u": [0] * n, "a_star_m": [0] * n,
        "thr_hat_witness": [4.1] * n, "p95_hat_witness": [8.1] * n,
        "pdr_hat_witness": [1.0] * n,
        "thr_hat_star": [4.1] * n, "p95_hat_star": [8.1] * n, "pdr_hat_star": [1.0] * n,
        "min_total_deficit": [0.0] * n, "min_deficit_embb": [0.0] * n,
        "min_deficit_urllc": [0.0] * n, "min_deficit_mmtc": [0.0] * n,
        "best_deg_n_ties": [1] * n,
        "embb_individually_saveable": [np.nan] * n,
        "urllc_individually_saveable": [np.nan] * n,
        "mmtc_individually_saveable": [np.nan] * n,
        "avoidable_any": [0.0] * n, "avoidable_embb": [0.0] * n,
        "avoidable_urllc": [0.0] * n, "avoidable_mmtc": [0.0] * n,
        "joint_feasible_p25": [True] * n, "joint_feasible_p75": [True] * n,
        "thr_hat_exec": [4.0] * n,
    })
    return cl


def test_oracle_windows_table_exact_section19_order():
    t = oracle_windows_table(_minimal_classified())
    assert list(t.columns) == list(ds.ORACLE_WINDOWS_COLUMNS_19)


def test_oracle_actions_and_extras_tables_shapes():
    cl = _minimal_classified()
    ta = oracle_actions_table(cl)
    assert {"equal_action_a_e", "prop_action_a_e", "oracle_feasible_a_e",
            "oracle_best_a_e"} <= set(ta.columns)
    assert ta["prop_action_a_e"].isna().all()          # arm == equal
    assert (ta["equal_action_a_e"] == 37).all()
    te = oracle_windows_extra_table(cl)
    assert {"a_e", "prb_scheduled_rbg_0", "offered_load_embb", "pdr_mmtc_1s"} \
        <= set(te.columns)


# ---------------------------------------------------------------------------
# pipeline aggregates + SC/hypotheses plumbing on a synthetic classified frame
# ---------------------------------------------------------------------------

def test_aggregate_and_sanity_plumbing():
    # 12 windows: BOTH arms per lambda (the arms stay paired per (lam, seed),
    # the premise of the §15 pairing); 2 windows infeasible per lambda
    rows = []
    for lam in (2.0, 7.0):
        for arm in ("equal", "prop"):
            for k in range(3):
                rows.append({"lam": lam, "arm": arm, "k": k})
    cl = _minimal_classified(12)
    cl["lam"] = [r["lam"] for r in rows]
    cl["arm"] = [r["arm"] for r in rows]
    cl["seed"] = [1] * 12
    # feasibility pattern per (lam, arm): T, T, F
    cl["joint_feasible"] = ([True, True, False] * 4)
    cl["observed_any_violation"] = [0.0, 1.0, 1.0] * 4
    cl["avoidable_any"] = ([0.0, 1.0, np.nan] * 4)
    cl["min_total_deficit"] = [0.0, 0.0, 0.4] * 2 + [0.0, 0.0, 0.25] * 2
    bl = aggregate_by_lambda(cl)
    r2 = bl[bl["lambda_total"] == 2.0].iloc[0]
    assert r2["P_inf"] == pytest.approx(1 / 3)
    assert r2["AVR_equal"] == pytest.approx(0.5)   # 1 avoidable of 2 feasible
    assert r2["UVR_equal"] == pytest.approx(1.0)   # the infeasible window violates
    assert r2["Dmin_median_infeasible"] == pytest.approx(0.4)
    bs = pd.DataFrame({"lambda_total": [2.0, 7.0]})
    sc = sanity_checks(cl, bl)
    hyp = hypotheses(cl, bl, bs)
    assert set(sc) == {"SC1", "SC2", "SC3", "SC4"}
    assert set(hyp) == {"H2.4_1", "H2.4_2", "H2.4_3", "H2.4_4"}
    # SC4 must see a healthy SNR in the synthetic infeasible windows
    assert sc["SC4"]["passed"]
    # critical windows pairing: same (lam, seed) across arms
    cw = critical_windows(cl)
    assert set(cw.columns) >= {"priority_class", "lam", "seed", "win", "arm"}


# ---------------------------------------------------------------------------
# integrity against the frozen campaign (skipped when data is absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CAMPAIGN, reason="campanha 2.3 não presente neste checkout")
def test_dataset_reproduces_23_run_level():
    """The per-window builder must reproduce the FROZEN 2.3 run aggregates.

    viol_embb_pct / urllc_miss_pct / util D2 recomputed from the window
    dataset match ``round2_hetero_sweep.csv`` (tolerance 0.05 pp).
    """
    w = ds.build_window_dataset(ROUND2_DIR)
    ref = pd.read_csv(os.path.join(ROUND2_DIR, "round2_hetero_sweep.csv"))
    w = w.copy()
    w["uewins"] = w["thr_embb_uewins"].fillna(0)
    w["viol_uewins"] = (w["thr_embb_violfrac"].fillna(0) * w["uewins"])
    w["pkts"] = w["urllc_n_pkts"].fillna(0)
    w["miss_pkts"] = (w["latency_urllc_miss_pct"].fillna(0) * w["pkts"])
    g = w.groupby(["arm", "lam", "seed"])
    rec = pd.DataFrame({
        "viol_embb_pct": 100.0 * g["viol_uewins"].sum() / g["uewins"].sum(),
        "urllc_miss_pct": g["miss_pkts"].sum() / g["pkts"].sum(),
        "sched": g["prb_scheduled"].sum(), "slots": g["dl_slots"].sum(),
    }).reset_index()
    m = ref.merge(rec, on=["arm", "lam", "seed"], how="outer",
                  suffixes=("_ref", "_rec"), validate="1:1")
    assert len(m) == 30
    assert np.abs(m["viol_embb_pct_ref"] - m["viol_embb_pct_rec"]).max() < 0.05
    assert np.abs(m["urllc_miss_pct_ref"] - m["urllc_miss_pct_rec"]).max() < 0.05
    d2 = 100.0 * m["sched"] / (53 * m["slots"])
    assert np.abs(d2.values - m["util_D2"].values).max() < 0.05


@pytest.mark.skipif(not HAS_CAMPAIGN, reason="campanha 2.3 não presente neste checkout")
def test_full_pipeline_smoke_real_data():
    """Classify the real windows end-to-end on a REDUCED grid (2 runs) and
    check artifact-table invariants (fast smoke; the full 30-run pass is the
    driver's job, executed in the dispatch)."""
    from nsoran.sqr.oracle.models import fit_models
    w = ds.build_window_dataset(ROUND2_DIR)
    w = w[(w["lam"] == 7.0) & (w["seed"] == 1)]
    models = fit_models(w)
    cl = classify_windows(w, models)
    assert cl["joint_feasible"].notna().all()
    assert (cl["num_feasible_actions"] >= 0).all()
    assert cl["num_feasible_actions"].max() <= 27720
    # infeasible windows carry saveability; feasible ones do not
    inf = cl[~cl["joint_feasible"]]
    feas = cl[cl["joint_feasible"]]
    assert inf["embb_individually_saveable"].notna().all()
    assert feas["embb_individually_saveable"].isna().all()
    # avoidable ⊆ observed violation ∧ feasible
    av = cl["avoidable_any"].dropna()
    assert ((av == 1) <= (cl.loc[av.index, "observed_any_violation"] == 1)).all()
