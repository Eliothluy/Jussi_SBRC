"""Tests for the Etapa 2.4-II-d-v2 final recalibration — no ns-3.

Covers ``nsoran.sqr.oracle.recal_v2``: the §10 EXACT column contract, the
§15 local-support metrics, the §16 MARGINAL cause rule (frozen thresholds),
the §11 curve builder (monotonicity / SLA crossing / mixed tails), the
EmmbQuantileV2 refit (frozen basis + in-support extra rows, band-anchor and
monotone shape preserved) and the §19–§20 gate v2 semantics on synthetic
frames (R1 two-branch pre-registered rule, R4 no-regression, C1 FIR<=0.10,
C3 witness-in-support, C7 far-hard-error rule).
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
from nsoran.sqr.oracle import recal_v2 as rv  # noqa: E402


# ---------------------------------------------------------------------------
# §10 — EXACT column contract
# ---------------------------------------------------------------------------

def test_support_replays_columns_exact_order():
    assert list(rv.SUPPORT_REPLAYS_COLUMNS) == [
        "replay_id", "lambda_total", "seed", "window_id",
        "original_state_class", "cqi_embb", "n_embb", "queue_embb",
        "a_embb", "a_urllc", "a_mmtc", "embb_thr_median", "embb_sla_ok",
        "urllc_p95", "urllc_ok", "mmtc_pdr", "mmtc_ok", "joint_ok",
        "snr_p05", "replay_valid"]


# ---------------------------------------------------------------------------
# §15 — local support metrics
# ---------------------------------------------------------------------------

def _tiny_model_frame() -> pd.DataFrame:
    # 4 rows at n=2/a=18 (s=9), cqi 14..15; 2 rows at n=2/a=37 (s=18.5);
    # 1 row at n=1/a=10 (s=10).
    return pd.DataFrame({
        "n_embb": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0],
        "a_e": [18.0, 18.0, 18.0, 18.0, 37.0, 37.0, 10.0],
        "cqi_eff": [14.0, 14.5, 15.0, 15.0, 15.0, 13.0, 15.0],
        "queue_embb": [0.0] * 7, "n_embb_backlog": [0.0] * 7,
        "offered_load_embb": [0.0] * 7, "prev_thr_embb_med": [1.0] * 7,
        "thr_embb_med": [4.0, 4.5, 5.0, 5.0, 7.0, 6.0, 3.0],
        "_provenance": ["frozen"] * 7,
    })


def test_local_support_batch_counts_and_distance():
    m = rv.EmmbQuantileV2()
    m.train_frame = _tiny_model_frame()
    out = m.local_support_batch([2.0], [14.8], [18.0])[0]
    # s_q = 9; the four s=9 rows with |cqi-14.8| <= 3 all count
    assert out["local_sample_count"] == 4
    assert out["nearest_action_distance"] == 0.0
    assert out["support_density"] == pytest.approx(4 / rv.LOCAL_AREA)
    # far query: same n, s=30 → nearest s is 18.5 → distance 11.5, no locals
    far = m.local_support_batch([2.0], [15.0], [60.0])[0]
    assert far["nearest_action_distance"] == pytest.approx(11.5)
    assert far["local_sample_count"] == 0
    # n with no training rows falls back to the global nearest s (10 at n=1)
    miss = m.local_support_batch([5.0], [15.0], [50.0])[0]
    assert miss["local_sample_count"] == 0
    assert miss["nearest_action_distance"] == pytest.approx(abs(10.0 - 10.0))
    # batch == single-query wrapper
    one = m.local_support(2.0, 14.8, 18.0)
    assert one == out


# ---------------------------------------------------------------------------
# §16 — MARGINAL cause (frozen thresholds)
# ---------------------------------------------------------------------------

def test_marginal_cause_frozen_thresholds():
    good = {"local_sample_count": 100, "nearest_action_distance": 0.0}
    poor = {"local_sample_count": 3, "nearest_action_distance": 0.0}
    far = {"local_sample_count": 100, "nearest_action_distance": 2.0}
    # narrow band straddling the SLA + good support → NEAR_THRESHOLD
    assert rv.marginal_cause(3.2, 3.5, 3.8, good) == "MARGINAL_NEAR_THRESHOLD"
    # wide band + good support → HIGH_VARIANCE (width 9.0 > 2.0)
    assert rv.marginal_cause(1.0, 5.0, 10.0, good) == "MARGINAL_HIGH_VARIANCE"
    # poor local support dominates the label
    assert rv.marginal_cause(3.2, 3.5, 3.8, poor) == "MARGINAL_LOW_SUPPORT"
    assert rv.marginal_cause(1.0, 5.0, 10.0, far) == "MARGINAL_LOW_SUPPORT"
    # boundary widths: exactly 2.0 Mbps is NOT wide
    edge = {"local_sample_count": 100, "nearest_action_distance": 1.0}
    assert rv.marginal_cause(2.5, 3.5, 4.5, edge) == "MARGINAL_NEAR_THRESHOLD"


def test_band_registration_fields():
    reg = rv.band_registration(3.2, 3.6, 4.4,
                               {"support_density": 7.5,
                                "nearest_action_distance": 0.0,
                                "local_sample_count": 90})
    assert reg["band_width_embb"] == pytest.approx(1.2)
    assert reg["distance_to_sla"] == pytest.approx(0.1)
    assert reg["marginal_cause"] == "MARGINAL_NEAR_THRESHOLD"


# ---------------------------------------------------------------------------
# §12 — EmmbQuantileV2 refit
# ---------------------------------------------------------------------------

def _synth_windows(n: int = 360, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ne = rng.integers(1, 5, n)
    nu = rng.integers(1, 4, n)
    a_e = rng.choice([18, 37], n)
    s = a_e / ne
    thr = 0.9 * s + rng.normal(0, 0.3, n)
    p95 = 6.0 + 1.8 * nu + rng.normal(0, 0.8, n)
    return pd.DataFrame({
        "arm": "equal", "lam": 5.0, "seed": 1, "win": np.arange(n),
        "time_ms": 100 * (np.arange(n) + 1),
        "n_embb": ne, "n_urllc": nu, "n_mmtc": 0,
        "n_embb_backlog": ne.astype(float),
        "queue_embb": rng.uniform(0, 1e6, n),
        "queue_urllc": rng.uniform(0, 1e5, n),
        "offered_load_embb": 3.5 * ne,
        "cqi_eff": rng.uniform(13.5, 15.0, n),
        "a_e": a_e, "a_u": rng.integers(13, 18, n),
        "a_m": rng.integers(3, 6, n),
        "prev_thr_embb_med": thr + rng.normal(0, 0.1, n),
        "prev_p95": p95 + rng.normal(0, 0.2, n),
        "prev_miss_pct": np.where(p95 > 10, 20.0, 0.0),
        "urllc_n_pkts": 100 * nu,
        "dl_slots": rng.integers(40, 59, n).astype(float),
        "thr_embb_med": np.clip(thr, 0, 24),
        "latency_urllc_p95_ms": p95,
        "observed_embb_violation": (thr < 3.5).astype(float),
        "observed_urllc_violation": (p95 > 10).astype(float),
        "observed_mmtc_violation": np.full(n, np.nan),
        "observed_any_violation": ((thr < 3.5) | (p95 > 10)).astype(float),
    })


def test_emmb_v2_fit_deterministic_and_shapes():
    d = _synth_windows()
    extra = pd.DataFrame([{
        "n_embb": 4.0, "cqi_eff": 15.0, "queue_embb": 1e6,
        "n_embb_backlog": 4.0, "offered_load_embb": 14.0,
        "prev_thr_embb_med": 2.0, "a_e": 37.0, "thr_embb_med": 4.9,
        "lam": 6.0, "seed": 3, "win": 298,
        "_provenance": "E1_v2_replay_in_support"}])
    m1 = rv.EmmbQuantileV2().fit(d, extra)
    m2 = rv.EmmbQuantileV2().fit(d, extra)
    for qq in (0.25, 0.5, 0.75):
        assert np.allclose(m1.model.betas[qq], m2.model.betas[qq])
    assert len(m1.train_frame) == len(d[(d.n_embb >= 1)
                                        & d.thr_embb_med.notna()]) + 1
    assert (m1.train_frame["_provenance"] == "E1_v2_replay_in_support"
            ).sum() == 1
    # grid bands keep the frozen shape: monotone in a_e, capped, anchored
    grid = np.array([0, 18, 24, 37], dtype=float)
    bands = m1.predict_grid_bands(np.array([4.0]), np.array([15.0]),
                                  np.array([1e6]), np.array([4.0]),
                                  np.array([14.0]), np.array([2.0]), grid)
    for qq in (0.25, 0.5, 0.75):
        assert np.all(np.diff(bands[qq][0]) >= -1e-9)
        assert bands[qq][0, 0] == 0.0          # a_e=0 anchor (R27)
        assert np.all(bands[qq] <= qm.EMMB_THR_CAP_MBPS + 1e-9)


def test_loo_gates_v2_structure():
    d = _synth_windows(n=600)
    # make 3 seeds so the LOO has 3 folds
    d = pd.concat([d.assign(seed=s) for s in (1, 2, 3)], ignore_index=True)
    extra = pd.DataFrame([{
        "n_embb": 4.0, "cqi_eff": 15.0, "queue_embb": 1e6,
        "n_embb_backlog": 4.0, "offered_load_embb": 14.0,
        "prev_thr_embb_med": 2.0, "a_e": 37.0, "thr_embb_med": 4.9,
        "lam": 6.0, "seed": 3, "win": 298,
        "_provenance": "E1_v2_replay_in_support"}])
    gates = rv.loo_gates_v2(d, extra)
    assert list(gates["held_seed"]) == [1, 2, 3]
    assert gates[["mae", "near_mae", "coverage_p25_p75"]].notna().all().all()
    assert (gates["coverage_p25_p75"] >= 0).all()


# ---------------------------------------------------------------------------
# §11 — support curve builder
# ---------------------------------------------------------------------------

@pytest.fixture()
def curve_env(tmp_path):
    """oracle_dir skeleton + plan/replays/d frames for build_support_curve."""
    (tmp_path / "adjudication").mkdir()
    (tmp_path / "oracle_recalibrated" / "adjudication").mkdir(parents=True)
    g2 = pd.DataFrame([
        {"replay_id": "G2_equal_lam6_s3_t298_a48-5-0",
         "thr_embb_override": 5.623, "sla_todos_ok_no_override": True},
    ])
    g2.to_csv(tmp_path / "adjudication" / "g2_deltas.csv", index=False)
    ver = pd.DataFrame([
        {"group": "D3", "role": "witness", "lambda_total": 6.0, "seed": 1,
         "window_id": 1866, "override_embb": 37, "override_urllc": 13,
         "override_mmtc": 3, "verified_thr_min": 3.9117,
         "verified_joint_ok": True, "replay_id":
             "D3_equal_lam6_s1_t1866_a37-13-3"},
    ])
    ver.to_csv(tmp_path / "oracle_recalibrated" / "adjudication" /
               "oracle_3way_replay_results_verified.csv", index=False)
    plan = pd.DataFrame([
        {"grupo": "E1", "arm_base": "equal", "lam": 6, "seed": 3,
         "win_t": 298, "action_e": 37, "action_u": 13, "action_m": 3},
        {"grupo": "E2x", "arm_base": "equal", "lam": 6, "seed": 1,
         "win_t": 1866, "action_e": 42, "action_u": 5, "action_m": 0},
    ])
    replays = pd.DataFrame([
        {"replay_id": "E1_equal_lam6_s3_t298_a37-13-3", "lambda_total": 6.0,
         "seed": 3, "window_id": 298, "a_embb": 37, "a_urllc": 13,
         "a_mmtc": 3, "embb_thr_median": 4.9507, "embb_sla_ok": True,
         "joint_ok": True, "replay_valid": True},
        {"replay_id": "E2x_equal_lam6_s1_t1866_a42-5-0", "lambda_total": 6.0,
         "seed": 1, "window_id": 1866, "a_embb": 42, "a_urllc": 5,
         "a_mmtc": 0, "embb_thr_median": 4.4006, "embb_sla_ok": True,
         "joint_ok": False, "replay_valid": True},
    ])
    d = pd.DataFrame([
        {"arm": "equal", "lam": 6.0, "seed": 3, "win": 298,
         "time_ms": 29900, "thr_embb_med": 2.567},
        {"arm": "equal", "lam": 6.0, "seed": 1, "win": 1866,
         "time_ms": 186700, "thr_embb_med": 2.3226},
    ])
    return tmp_path, plan, replays, d


def test_build_support_curve_points_and_metrics(curve_env):
    tmp, plan, replays, d = curve_env
    pts, curve, disp = rv.build_support_curve(str(tmp), replays, plan, d)
    # E1 window: original (18) + v2 (37) + II-c (48) = 3 points
    w1 = curve[(curve["lambda_total"] == 6.0) & (curve["window_id"] == 298)]
    assert w1["n_points"].iloc[0] == 3
    assert w1["monotone_nondecreasing"].iloc[0] is True or \
        bool(w1["monotone_nondecreasing"].iloc[0])
    assert w1["mixed_tail_segments"].iloc[0] == 2   # 18→37 and 37→48
    # SLA crossing between a_e=18 (2.567) and 37 (4.9507)
    cross = float(w1["sla_cross_a_e"].iloc[0])
    assert 18 < cross < 37
    # E2x window: original (18) + II-d verified (37) + v2 (42) = 3 points
    w2 = curve[(curve["lambda_total"] == 6.0) & (curve["window_id"] == 1866)]
    assert w2["n_points"].iloc[0] == 3
    srcs = set(w2["sources"].iloc[0].split("; "))
    assert srcs == {"original_equal", "ii_d_replay_verified", "v2_replay"}
    # dispersion table: a_e=37 pools 2 windows
    row37 = disp[disp["a_e"] == 37]
    assert int(row37["n_windows"].iloc[0]) == 2
    assert row37["support_density_note"].iloc[0] == "in_support"


# ---------------------------------------------------------------------------
# §19–§20 — gate semantics on synthetic frames
# ---------------------------------------------------------------------------

class _FakeEmbb:
    """Constant-band stub for gate_v2's C7 band lookups."""
    def __init__(self, p25=3.0, p75=4.0):
        self.p25, self.p75 = p25, p75

    def predict_grid_bands(self, n, cqi, q, bl, off, prev, a_grid):
        shape = (len(np.atleast_1d(n)), len(np.atleast_1d(a_grid)))
        return {0.25: np.full(shape, self.p25),
                0.5: np.full(shape, (self.p25 + self.p75) / 2),
                0.75: np.full(shape, self.p75)}


AVOID = rv.CONFIRMED_AVOIDABLE
WIDE = rv.D5_C7_WINDOWS


def _synth_cls(states: dict, n_extra: int = 0) -> pd.DataFrame:
    """cls frame with one equal row per (lam, seed, win) → state."""
    rows = []
    for (lam, seed, win), (state, cause) in states.items():
        rows.append({
            "arm": "equal", "lam": float(lam), "seed": int(seed),
            "win": int(win), "time_ms": (int(win) + 1) * 100,
            "state_class": state,
            "num_feasible_actions": 5 if state == "FEASIBLE" else 0,
            "num_marginal_actions": 5 if state == "MARGINAL" else 0,
            "num_infeasible_actions": 5 if state == "INFEASIBLE" else 0,
            "best_feasible_action": "(37,13,3)" if state == "FEASIBLE"
            else "",
            "best_marginal_action": "(37,13,3)" if state == "MARGINAL"
            else "",
            "best_degradation_action": "(37,13,3)",
            "best_marg_marginal_cause": cause,
        })
    return pd.DataFrame(rows)


def _synth_gate_inputs(e1_states, d5_state="MARGINAL",
                       d5_obs=(12.6, 11.5, 10.3, 14.3, 6.8),
                       fake_band=(3.0, 24.0)):
    """Build (cls, replays, ver, confusion, model, d) with controlled labels.

    e1_states: dict window → (v2 state, e1_all_ok, e1_thr)."""
    states = {}
    for (lam, seed, win) in rv.CONFIRMED_AVOIDABLE:
        st = e1_states[(lam, seed, win)][0]
        states[(lam, seed, win)] = (st, "MARGINAL_NEAR_THRESHOLD"
                                    if st == "MARGINAL" else "")
    states[(6.0, 1, 1866)] = ("INFEASIBLE", "")     # R2/C4 window
    for (lam, seed, win) in rv.D4_WINDOWS + rv.D3_WINDOWS[1:]:
        states[(lam, seed, win)] = ("INFEASIBLE", "")   # R3/C5 coherent
    for (lam, seed, win) in WIDE:
        states[(lam, seed, win)] = (d5_state, "MARGINAL_HIGH_VARIANCE")
    # the 5 confirmed FEASIBLE (R4 set) — same windows as the ver fixture
    for win in (1263, 1277, 1291, 1480, 1487):
        states[(2.0, 1, win)] = ("FEASIBLE", "")
    cls = _synth_cls(states)

    e1_rows = []
    for i, (lam, seed, win) in enumerate(rv.CONFIRMED_AVOIDABLE):
        state, ok, thr = e1_states[(lam, seed, win)]
        e1_rows.append({
            "replay_id": f"E1_equal_lam{int(lam)}_s{seed}_t{win}_a37-13-3",
            "lambda_total": lam, "seed": seed, "window_id": win,
            "a_embb": 37, "a_urllc": 13, "a_mmtc": 3,
            "embb_thr_median": thr, "embb_sla_ok": thr >= 3.5,
            "joint_ok": ok, "replay_valid": True})
    diag = [
        {"replay_id": "E3x_equal_lam7_s1_t1555_a44-5-0", "lambda_total": 7.0,
         "seed": 1, "window_id": 1555, "joint_ok": False,
         "embb_thr_median": 1.83, "replay_valid": True},
        {"replay_id": "E2x_equal_lam7_s3_t2013_a48-5-0", "lambda_total": 7.0,
         "seed": 3, "window_id": 2013, "joint_ok": True,
         "embb_thr_median": 5.62, "replay_valid": True},
    ]
    replays = pd.DataFrame(e1_rows + diag)

    ver_rows = []
    # 5 confirmed FEASIBLE (R4 set)
    for i, win in enumerate((1263, 1277, 1291, 1480, 1487)):
        ver_rows.append({
            "group": "D1", "role": "witness", "lambda_total": 2.0, "seed": 1,
            "window_id": win, "oracle_state_class": "FEASIBLE",
            "verified_thr_min": 7.5, "verified_joint_ok": True,
            "override_embb": 18, "replay_id": f"D1_t{win}"})
    # D5 rows for the C7 band-coverage lookups
    for (lam, seed, win), obs in zip(WIDE, d5_obs):
        ver_rows.append({
            "group": "D5", "role": "witness", "lambda_total": lam,
            "seed": seed, "window_id": win,
            "oracle_state_class": "MARGINAL", "verified_thr_min": obs,
            "verified_joint_ok": obs >= 3.5,
            "override_embb": 37 if win != 367 else 18,
            "replay_id": f"D5_t{win}"})
    ver = pd.DataFrame(ver_rows)

    d_rows = []
    for (lam, seed, win) in WIDE:
        d_rows.append({
            "arm": "equal", "lam": lam, "seed": seed, "win": win,
            "time_ms": (win + 1) * 100, "n_embb": 1.0, "cqi_eff": 15.0,
            "queue_embb": 1e6, "n_embb_backlog": 1.0,
            "offered_load_embb": 3.5, "prev_thr_embb_med": 5.0})
    d = pd.DataFrame(d_rows)
    model = _FakeEmbb(*fake_band)
    confusion = rv.confusion_matrix_3way_v2(cls, ver, replays)
    return cls, replays, ver, confusion, model, d


def test_gate_v2_all_ok_e1_with_infeasible_states_fails():
    """The actual v2 outcome: 5 E1 all-OK... we use 3 all-OK + 2 violated —
    states INFEASIBLE → R1/C1/C3/C7 FAIL; R3/R4/C2/C6 PASS."""
    e1_states = {}
    oks = [True, True, True, False, False]
    thrs = [4.95, 4.52, 3.55, 3.18, 3.42]
    for (lam, seed, win), ok, thr in zip(rv.CONFIRMED_AVOIDABLE, oks, thrs):
        e1_states[(lam, seed, win)] = ("INFEASIBLE", ok, thr)
    cls, replays, ver, confusion, model, d = _synth_gate_inputs(e1_states)
    gate = rv.gate_v2(cls, replays, ver, confusion, model, d, None)
    assert gate["verdict"] == "STOP"
    by_id = {c["id"]: c for c in gate["checks"]}
    assert not by_id["R1_evitaveis_nao_infeasible"]["passed"]
    assert not by_id["R2_C4_contaprova_all_ok_nao_infeasible"]["passed"]
    assert not by_id["C1_FIR"]["passed"]
    assert not by_id["C3_witness_in_support"]["passed"]
    assert not by_id["C7_marginal_absorve"]["passed"]
    assert by_id["R3_C5_inviaveis_coerentes"]["passed"]
    assert by_id["R4_feasible_nao_regridem"]["passed"]
    assert by_id["C2_FFR"]["passed"]
    assert by_id["C6_acoes_otimas_no_suporte"]["passed"]
    # FIR = 3 INFEASIBLE∧all-OK among the 5 predicted-INFEASIBLE witnesses
    # (the fixture's ver has only D1+D5+E1; D2/D4 are not in it)
    assert by_id["C1_FIR"]["value"] == pytest.approx(3 / 5)
    # the 2 near-threshold violations keep C3 alive for those two windows
    c3 = by_id["C3_witness_in_support"]
    assert c3["value"]["n_fail"] == 3


def test_gate_v2_passes_when_states_flip_to_marginal():
    """Same E1 observations but the v2 oracle classifies the 5 states (and
    t1866) MARGINAL/FEASIBLE: every regression closes; C1 FIR = 0."""
    e1_states = {}
    oks = [True, True, True, False, False]
    thrs = [4.95, 4.52, 3.55, 3.18, 3.42]
    for (lam, seed, win), ok, thr in zip(rv.CONFIRMED_AVOIDABLE, oks, thrs):
        e1_states[(lam, seed, win)] = ("MARGINAL", ok, thr)
    cls, replays, ver, confusion, model, d = _synth_gate_inputs(e1_states)
    # R2/C4 window becomes MARGINAL as well
    cls.loc[(cls["lam"] == 6.0) & (cls["seed"] == 1)
            & (cls["win"] == 1866), "state_class"] = "MARGINAL"
    gate = rv.gate_v2(cls, replays, ver, confusion, model, d, None)
    by_id = {c["id"]: c for c in gate["checks"]}
    assert by_id["R1_evitaveis_nao_infeasible"]["passed"]
    assert by_id["R2_C4_contaprova_all_ok_nao_infeasible"]["passed"]
    assert by_id["C1_FIR"]["passed"]
    assert by_id["C3_witness_in_support"]["passed"]
    assert by_id["R3_C5_inviaveis_coerentes"]["passed"]
    assert by_id["R4_feasible_nao_regridem"]["passed"]
    # C7: the D5 far observation (12.6, band [3,24] covers it) passes here
    assert by_id["C7_marginal_absorve"]["passed"]
    assert gate["verdict"] == "PASS"


def test_gate_v2_r4_fails_on_regression():
    e1_states = {(lam, seed, win): ("MARGINAL", True, 4.0)
                 for (lam, seed, win) in rv.CONFIRMED_AVOIDABLE}
    cls, replays, ver, confusion, model, d = _synth_gate_inputs(e1_states)
    # regress the 5 confirmed FEASIBLE to INFEASIBLE
    d1 = cls[(cls["lam"] == 2.0) & (cls["win"].isin([1263, 1277, 1291,
                                                     1480, 1487]))]
    cls.loc[d1.index, "state_class"] = "INFEASIBLE"
    confusion = rv.confusion_matrix_3way_v2(cls, ver, replays)
    gate = rv.gate_v2(cls, replays, ver, confusion, model, d, None)
    by_id = {c["id"]: c for c in gate["checks"]}
    assert not by_id["R4_feasible_nao_regridem"]["passed"]
    assert by_id["R4_feasible_nao_regridem"]["value"]["n_regressed"] == 5


def test_confusion_v2_27_witnesses_and_excludes_d3():
    e1_states = {(lam, seed, win): ("MARGINAL", True, 4.0)
                 for (lam, seed, win) in rv.CONFIRMED_AVOIDABLE}
    cls, replays, ver, confusion, model, d = _synth_gate_inputs(e1_states)
    wit = confusion["witnesses"] if "witnesses" in confusion else None
    # confusion_matrix_3way_v2 returns witnesses separately in the driver;
    # call it directly to inspect
    conf = rv.confusion_matrix_3way_v2(cls, ver, replays)
    assert conf["n_witness"] == 15     # 5 D1 + 5 D5 (legacy) + 5 E1 (v2)
    assert conf["n_witness_legacy_verified"] == 10
    assert conf["n_witness_e1_v2"] == 5
    assert not conf["witnesses"]["replay_id"].str.contains("D3").any()
    assert set(conf["witnesses"]["obs_source"]) == {
        "verified_ii_d", "v2_replay_tplus1"}
