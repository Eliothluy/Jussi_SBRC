"""Tests for the Etapa 2.4-II-d oracle recalibration — no ns-3.

Covers ``nsoran.sqr.oracle.quantile`` (pinball LP, frozen bases, band
gates, LOO plumbing), ``nsoran.sqr.oracle.three_way`` (supported space,
3-way classification vs BRUTE FORCE, D_min envelopes, best actions,
candidate aggregates) and ``nsoran.sqr.oracle.replay_plan_v2`` (D1-D6 plan
format, support of witnesses, reuse of the II-c hard cases).
"""

import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_TEST_DIR, "..", "src")
sys.path.insert(0, _SRC)

from nsoran.sqr.oracle import quantile as qm  # noqa: E402
from nsoran.sqr.oracle import replay_plan_v2 as rp  # noqa: E402
from nsoran.sqr.oracle import three_way as tw  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _synth_windows(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic window table obeying the frozen dataset columns."""
    rng = np.random.default_rng(seed)
    ne = rng.integers(1, 5, n)
    nu = rng.integers(1, 4, n)
    a_e = rng.choice([18, 37], n)
    a_u = rng.integers(13, 18, n)
    a_m = rng.integers(3, 6, n)
    s = a_e / ne
    thr = 1.2 * s + rng.normal(0, 0.3, n)          # share-driven truth
    p95 = 6.0 + 1.8 * nu + rng.normal(0, 0.8, n)   # contention-driven truth
    return pd.DataFrame({
        "arm": "equal", "lam": 5.0, "seed": 1, "win": np.arange(n),
        "time_ms": 100 * (np.arange(n) + 1),
        "n_embb": ne, "n_urllc": nu, "n_mmtc": 0,
        "n_embb_backlog": ne,
        "queue_embb": rng.uniform(0, 1e6, n),
        "queue_urllc": rng.uniform(0, 1e5, n),
        "offered_load_embb": 3.5 * ne,
        "cqi_eff": rng.uniform(13.5, 15.0, n),
        "a_e": a_e, "a_u": a_u, "a_m": a_m,
        "prev_thr_embb_med": thr + rng.normal(0, 0.1, n),
        "prev_p95": p95 + rng.normal(0, 0.2, n),
        "prev_miss_pct": np.where(p95 > 10, 20.0, 0.0),
        "urllc_n_pkts": 100 * nu,
        "dl_slots": rng.integers(40, 59, n),
        "thr_embb_med": np.clip(thr, 0, 24),
        "latency_urllc_p95_ms": p95,
        "observed_embb_violation": (thr < 3.5).astype(float),
        "observed_urllc_violation": (p95 > 10).astype(float),
        "observed_mmtc_violation": np.full(n, np.nan),
        "observed_any_violation": ((thr < 3.5) | (p95 > 10)).astype(float),
    })


@pytest.fixture(scope="module")
def fitted():
    d = _synth_windows()
    return d, qm.EmmbQuantileRecal().fit(d), qm.UrllcQuantileRecal().fit(d)


# ---------------------------------------------------------------------------
# pinball LP
# ---------------------------------------------------------------------------

def test_pinball_lp_recovers_true_quantiles():
    rng = np.random.default_rng(7)
    n = 4000
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(0, 1.0, n)
    X = np.column_stack([np.ones(n), x])
    Xs = X.copy()
    b25 = qm.pinball_lp(Xs, y, 0.25)
    b75 = qm.pinball_lp(Xs, y, 0.75)
    # pinball loss of the LP solution must not be worse than a perturbed one
    def pb(b, q):
        r = y - X @ b
        return float(np.mean(np.maximum(q * r, 0) + np.maximum((1 - q) * (-r), 0)))
    for b, q, truth in ((b25, 0.25, 1.0 - 0.674), (b75, 0.75, 1.0 + 0.674)):
        assert pb(b, q) <= pb(b + np.array([0.05, 0.05]), q) + 1e-9
        assert abs(b[1] - 2.0) < 0.15          # slope recovered
        assert abs(b[0] - truth) < 0.2         # intercept at the quantile


def test_pinball_lp_intercept_not_destroyed_by_standardization():
    """Regression guard: constant columns must SURVIVE the standardization
    (centering them deleted the intercept and collapsed every prediction)."""
    rng = np.random.default_rng(3)
    n = 600
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = 5.0 + 0 * X[:, 1] + rng.normal(0, 0.1, n)
    m = qm.PinballQuantileModel(quantiles=(0.5,)).fit(X, y)
    pred = m.predict_median(X)
    assert np.abs(pred - 5.0).max() < 0.5


def test_rearrange_fixes_crossing():
    qs = {0.25: np.array([3.0]), 0.5: np.array([1.0]), 0.75: np.array([2.0])}
    out = qm.rearrange(qs)
    assert out[0.25][0] <= out[0.5][0] <= out[0.75][0]


# ---------------------------------------------------------------------------
# frozen bases + grid consistency
# ---------------------------------------------------------------------------

def test_emmb_grid_matches_point_evaluation(fitted):
    d, me, _ = fitted
    samp = d.iloc[:120]
    pt = me.predict_point(samp)
    gr = me.predict_grid_bands(
        samp["n_embb"].to_numpy(float), samp["cqi_eff"].to_numpy(float),
        samp["queue_embb"].to_numpy(float), samp["n_embb_backlog"].to_numpy(float),
        samp["offered_load_embb"].to_numpy(float),
        samp["prev_thr_embb_med"].to_numpy(float),
        samp["a_e"].to_numpy(float))
    # the grid applies the frozen capacity clip; the point path does not —
    # compare point predictions AFTER the same clip (a_e in {18, 37}, so no
    # zero-anchor column involved)
    clip = lambda v: np.clip(v, 0.0, qm.EMMB_THR_CAP_MBPS)
    # factorization identity: evaluating the FULL grid at a SINGLE action a
    # must equal the point evaluation at a (the frozen shape constraints
    # — clip/monotone/anchor — are single-column no-ops for a != 0)
    for a in (18, 37):
        ga = me.predict_grid_bands(
            samp["n_embb"].to_numpy(float), samp["cqi_eff"].to_numpy(float),
            samp["queue_embb"].to_numpy(float),
            samp["n_embb_backlog"].to_numpy(float),
            samp["offered_load_embb"].to_numpy(float),
            samp["prev_thr_embb_med"].to_numpy(float), np.array([float(a)]))
        rows_a = samp[samp["a_e"] == a]
        idx = rows_a.index.to_numpy()
        assert np.allclose(clip(pt[0.5][idx]), ga[0.5][idx, 0], atol=1e-8)
        assert np.allclose(clip(pt[0.25][idx]), ga[0.25][idx, 0], atol=1e-8)


def test_emmb_grid_monotone_in_quota_and_capped(fitted):
    d, me, _ = fitted
    g = me.predict_grid_bands(
        np.array([2.0]), np.array([15.0]), np.array([1e5]), np.array([2.0]),
        np.array([7.0]), np.array([4.0]), np.arange(0, 38))
    med = g[0.5][0]
    assert med[0] == 0.0                       # zero-quota anchor (R27)
    assert np.all(np.diff(med) >= -1e-9)       # monotone nondecreasing
    assert med.max() <= qm.EMMB_THR_CAP_MBPS + 1e-9   # capacity cap


def test_urllc_grid_flat_above_floor_and_bounded(fitted):
    _, _, mu = fitted
    g = mu.predict_grid_bands(
        np.array([2.0]), np.array([2.0]), np.array([1e4]), np.array([1e6]),
        np.array([200.0]), np.array([53.0]), np.array([9.0]), np.array([0.0]),
        np.arange(13, 51))
    for q in g:
        # |effect(a_u) - effect(13)| bounded by the frozen +/- 2 ms clip
        assert np.abs(g[q][0] - g[q][0, 0]).max() <= 2.0 + 1e-9


def test_frozen_design_records():
    f = qm.FROZEN_RECAL
    assert f["band_quantiles"] == [0.25, 0.5, 0.75]
    assert f["urllc_floor_rbg"] == 13
    assert "cqi" in f["embb_features"] and "s" in f["embb_features"]
    assert "burst" in f["urllc_features"] and "dl_slots" in f["urllc_features"]
    assert "au_over" in f["urllc_action_cols"]
    # duplicated support literals in replay_plan_v2 match three_way
    assert rp.C_PRB == tw.C_PRB
    assert rp.URLLC_FLOOR == tw.URLLC_FLOOR
    assert rp.EMBB_SUPPORT_MIN == tw.EMBB_SUPPORT_MIN
    assert rp.EMBB_SUPPORT_MAX == tw.EMBB_SUPPORT_MAX


# ---------------------------------------------------------------------------
# band rules (§5)
# ---------------------------------------------------------------------------

def test_band_rules_higher_better():
    b = qm.embb_band(np.array([3.6, 3.4, 3.0, 4.0]),
                     np.array([3.8, 3.6, 3.2, 4.1]))
    # p25 >= SLA -> F; p75 < SLA -> I; else M
    assert b.tolist() == [tw.FEASIBLE, tw.MARGINAL, tw.INFEASIBLE, tw.FEASIBLE]


def test_band_rules_lower_better():
    b = qm.urllc_band(np.array([9.0, 11.0, 9.5, 8.0]),
                      np.array([9.5, 13.0, 12.0, 8.5]))
    # p75 <= SLA -> F; p25 > SLA -> I; else M
    assert b.tolist() == [tw.FEASIBLE, tw.INFEASIBLE, tw.MARGINAL, tw.FEASIBLE]


# ---------------------------------------------------------------------------
# supported space + 3-way classification vs brute force
# ---------------------------------------------------------------------------

def test_supported_space_counts_match_brute_force():
    for am_min in (3, 5):
        ae_g, au_g, comp = tw.supported_grids(am_min)
        brute = {(a, u, m) for a in range(0, 54) for u in range(0, 54)
                 for m in range(0, 54)
                 if 18 <= a <= 37 and u >= 13 and m >= am_min and a + u + m <= 53}
        total = int(comp.sum())
        assert total == len(brute)
        # per-pair completions
        for i, a in enumerate(ae_g):
            for j, u in enumerate(au_g):
                cnt = sum(1 for m in range(am_min, 54) if a + u + m <= 53)
                assert comp[i, j] == cnt


def test_classify_3way_matches_brute_force():
    """Full per-window classification vs an independent brute force."""
    d = _synth_windows(n=60, seed=1)
    me = qm.EmmbQuantileRecal().fit(d)
    mu = qm.UrllcQuantileRecal().fit(d)
    am_min = 3
    cls = tw.classify_windows_3way(d, me, mu, am_min=am_min)
    ae_g, au_g, comp = tw.supported_grids(am_min)

    # independent re-derivation of the bands from the same grid predictions
    q_e = me.predict_grid_bands(
        d["n_embb"].to_numpy(float), d["cqi_eff"].to_numpy(float),
        d["queue_embb"].to_numpy(float), d["n_embb_backlog"].to_numpy(float),
        d["offered_load_embb"].to_numpy(float),
        d["prev_thr_embb_med"].to_numpy(float), ae_g)
    q_u = mu.predict_grid_bands(
        d["n_urllc"].to_numpy(float), d["n_embb"].to_numpy(float),
        d["queue_urllc"].to_numpy(float), d["queue_embb"].to_numpy(float),
        d["urllc_n_pkts"].to_numpy(float), d["dl_slots"].to_numpy(float),
        d["prev_p95"].to_numpy(float), d["prev_miss_pct"].to_numpy(float),
        au_g)
    rng = np.random.default_rng(0)
    for r in rng.choice(len(d), 10, replace=False):
        eb = qm.embb_band(q_e[0.25][r], q_e[0.75][r])
        ub = qm.urllc_band(q_u[0.25][r], q_u[0.75][r])
        nf = nm = ni = 0
        best_f = None
        for i in range(len(ae_g)):
            for j in range(len(au_g)):
                if comp[i, j] == 0:
                    continue
                jc = tw._COMBINE[eb[i], ub[j]]
                nf += comp[i, j] * (jc == tw.FEASIBLE)
                nm += comp[i, j] * (jc == tw.MARGINAL)
                ni += comp[i, j] * (jc == tw.INFEASIBLE)
        row = cls.iloc[r]
        assert row["num_feasible_actions"] == nf
        assert row["num_marginal_actions"] == nm
        assert row["num_infeasible_actions"] == ni
        assert nf + nm + ni == int(comp.sum())
        expect_state = ("FEASIBLE" if nf > 0 else
                        "MARGINAL" if nm > 0 else "INFEASIBLE")
        assert row["state_class"] == expect_state
        # best actions are inside the supported space when present
        for col in ("best_feasible_action", "best_marginal_action",
                    "best_degradation_action"):
            act = rp._parse_action(row[col])
            if act is not None:
                assert rp.in_supported_space(*act, am_min)


def test_state_class_rule_and_counts_sum():
    d = _synth_windows(n=40, seed=2)
    me = qm.EmmbQuantileRecal().fit(d)
    mu = qm.UrllcQuantileRecal().fit(d)
    cls = tw.classify_windows_3way(d, me, mu, am_min=3)
    tot = (cls["num_feasible_actions"] + cls["num_marginal_actions"]
           + cls["num_infeasible_actions"])
    assert (tot == cls["n_supported_actions"]).all()
    m = cls["state_class"] == "FEASIBLE"
    assert (cls.loc[m, "num_feasible_actions"] > 0).all()
    m = cls["state_class"] == "MARGINAL"
    assert (cls.loc[m, "num_feasible_actions"] == 0).all()
    assert (cls.loc[m, "num_marginal_actions"] > 0).all()
    m = cls["state_class"] == "INFEASIBLE"
    assert (cls.loc[m, ["num_feasible_actions", "num_marginal_actions"]] == 0).all().all()


def test_dmin_envelopes_ordering_and_uncertain_flag():
    d = _synth_windows(n=40, seed=3)
    me = qm.EmmbQuantileRecal().fit(d)
    mu = qm.UrllcQuantileRecal().fit(d)
    cls = tw.classify_windows_3way(d, me, mu, am_min=3)
    pess = cls["D_min_support_p25_pessimistic"].to_numpy()
    med = cls["D_min_support_p50"].to_numpy()
    opt = cls["D_min_support_p75_optimistic"].to_numpy()
    ok = np.isfinite(pess)
    assert (opt[ok] <= med[ok] + 1e-12).all()
    assert (med[ok] <= pess[ok] + 1e-12).all()
    unc = cls["D_min_uncertain"].to_numpy()
    expect = (opt == 0) & (pess > 0)      # NaN -> False, same rule as code
    assert (unc == expect).all()
    # deficits are non-negative
    assert (np.nan_to_num(opt) >= 0).all()


def test_aggregate_p_candidates_sum_to_one():
    d = _synth_windows(n=80, seed=4)
    d["lam"] = np.where(d["win"] < 40, 2.0, 5.0)
    me = qm.EmmbQuantileRecal().fit(d)
    mu = qm.UrllcQuantileRecal().fit(d)
    cls = tw.classify_windows_3way(d, me, mu, am_min=3)
    by = tw.aggregate_3way_by_lambda(cls)
    assert np.allclose(by["P_F_candidate"] + by["P_M_candidate"]
                       + by["P_I_candidate"], 1.0)
    assert set(by["lambda_total"]) == {2.0, 5.0}
    assert "candidate" in "P_F_candidate"  # provisional naming rule (§17)


def test_mmtc_placeholder_band():
    mp = qm.MmtcFloorPlaceholder(floor=3)
    band = mp.band(np.array([1.0, 1.0, 0.0]), np.array([3.0, 2.0, 5.0]))
    assert (band[0] == 1.0).all()          # at the floor -> PDR 1.0
    assert np.isnan(band[1]).all()         # below floor -> OUT of support
    assert np.isnan(band[2]).all()         # no demand -> vacuous (NaN)


# ---------------------------------------------------------------------------
# LOO gates plumbing
# ---------------------------------------------------------------------------

def test_loo_gates_runs_and_metrics_bounded():
    d = _synth_windows(n=90, seed=5)
    d["seed"] = np.where(d["win"] < 30, 1, np.where(d["win"] < 60, 2, 3))
    gates = qm.loo_gates(d)
    for name, g in gates.items():
        assert sorted(g["held_seed"]) == [1, 2, 3]
        assert (g["mae"] >= 0).all()
        assert ((g["coverage_p25_p75"] >= 0) & (g["coverage_p25_p75"] <= 1)).all()
        assert (g["near_n"] >= 0).all()


# ---------------------------------------------------------------------------
# replay plans (D1-D6)
# ---------------------------------------------------------------------------

def test_plan_v2_reuses_ii_c_hard_cases_and_support():
    d = _synth_windows(n=300, seed=6)
    # cover the required (lam, seed, win) identities of the II-c cases
    cases = list(rp.D1_CASES) + list(rp.D2_CASES) + list(rp.D3_CASES) \
        + list(rp.D4_CASES)
    extra = []
    for k, (lam, seed, win) in enumerate(cases):
        r = d.iloc[k % len(d)].copy()
        r["lam"], r["seed"], r["win"] = lam, seed, win
        r["time_ms"] = 100 * (win + 1)
        extra.append(r)
    d2 = pd.concat([d] + [pd.DataFrame([e]) for e in extra],
                   ignore_index=True)
    me = qm.EmmbQuantileRecal().fit(d2)
    mu = qm.UrllcQuantileRecal().fit(d2)
    cls = tw.classify_windows_3way(d2, me, mu, am_min=3)
    plan = rp.build_plan_v2(cls, d2, am_min=3)
    chk = rp.validate_plan(plan, am_min=3)
    assert chk["witness_support_ok"], chk
    assert chk["sum_le_c"]
    assert chk["unique_tags"]
    got = {(r.grupo, float(r.lam), int(r.seed), int(r.win_t))
           for r in plan.itertuples()}
    for group, cases_g in (("D1", rp.D1_CASES), ("D2", rp.D2_CASES),
                           ("D3", rp.D3_CASES), ("D4", rp.D4_CASES)):
        for (lam, seed, win) in cases_g:
            assert (group, lam, seed, win) in got, (group, lam, seed, win)
    for (lam, seed, win, ae, au, am) in rp.G4_CONTROL_ROWS:
        assert ("G4", lam, seed, win) in got          # verbatim controls
    # D5 rows exist and are witnesses inside the support
    d5 = plan[plan["grupo"] == "D5"]
    assert len(d5) >= 1
    assert d5["_in_support"].all()


def test_plan_d6_format_and_grid():
    d = _synth_windows(n=300, seed=7)
    d["arm"] = "equal"
    # three distinct runs, each spanning windows >= 200 (D6 pick window),
    # all with mMTC demand IN the window (pdr 100 ms notna + n_mmtc >= 1)
    trip = np.where(d["win"] % 3 == 0, 0, np.where(d["win"] % 3 == 1, 1, 2))
    d["lam"] = np.where(trip == 0, 7.0, np.where(trip == 1, 6.0, 5.0))
    d["seed"] = np.where(trip == 1, 3, 1)
    d["n_mmtc"] = 1
    d["pdr_mmtc"] = 1.0
    d["pdr_mmtc_1s"] = 1.0
    plan, meta = rp.build_plan_d6(d)
    assert list(plan.columns[:9]) == list(rp.PLAN_COLUMNS)
    assert len(plan) == 18                             # 3 janelas x a_m 0..5
    assert (plan["grupo"] == "D6").all()
    assert set(plan["action_m"]) == {0, 1, 2, 3, 4, 5}
    assert plan.groupby("action_m").size().eq(3).all()   # 3 janelas por a_m
    assert plan.groupby(["lam", "seed", "win_t"]).size().eq(6).all()
    assert (plan["action_e"] + plan["action_u"] + plan["action_m"] <= 53).all()
    assert (plan["action_u"] == 13).all()              # URLLC at the floor
    assert len(meta) == 3
    assert plan["grupo"].eq("D6").all() and (plan["_role"] == "calibration").all()
    # a_m < floor rows are flagged OUT of the support (by design)
    below = plan[plan["action_m"] < 3]
    assert (~below["_in_support"]).all()


def test_expectations_frozen_for_every_group():
    for g in ("D1", "D2", "D3", "D4", "G4", "D5", "D6"):
        assert g in rp.GROUP_EXPECTATIONS and rp.GROUP_EXPECTATIONS[g].strip()
