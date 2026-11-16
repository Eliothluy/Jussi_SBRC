"""Unit and integration tests for modular Phase A5 joint selectors (B1-raw, B1-active, B1-support).

Verifies:
1. Exact agreement with brute-force lexicographic optimization & deterministic tie-breaking.
2. Action legality (0 <= a_i <= C, sum(a) <= C) across edge cases and capacities.
3. Null demand & pending-observation parity with B0-rest (``empty_slice_guard``).
4. Avoidance of D3-v1 extrapolative degeneracies ``(0,0,0)`` / ``(0,53,0)``.
5. Hierarchical envelopes & deterministic OOD fallback tiers (Tier 1 -> 2 -> 3).
6. Determinism, zero RNG / controller mutation, causality & non-anticipation parity.
7. Single-window A/A counterfactual replay comparison (B0-rest, equal, prop, B1-*) + latency < 100 ms.
"""
from __future__ import annotations

import copy
import gzip
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.dirname(__file__))

import sqr_run  # noqa: E402
from nsoran.sqr.replay_controller import ReplayController, initial_equal_split  # noqa: E402
from nsoran.sqr.sqr_controller import SqrSliceController  # noqa: E402
from nsoran.sqr.static_controllers import PropDemandController  # noqa: E402
from nsoran.sqr_jc.a5_joint_selector import (  # noqa: E402
    A5_VARIANTS,
    A5JointSelector,
    A5SelectorConfig,
    bounded_project_active,
    causal_active_slices,
    equal_active_split,
    load_b0_rest_het_from_checkpoint,
    prop_active_split,
)
from nsoran.sqr_jc.joint_state import JointState  # noqa: E402
from nsoran.sqr_jc.het_runtime import HetRuntime  # noqa: E402
from nsoran.sqr_jc.sqr_het import SqrHetController  # noqa: E402
from test_sqr_jc_freeze_contract import (  # noqa: E402
    REPO_SCALER_MANIFEST,
    _DivergentFutureHetEnv,
    _ScriptedHetEnv,
    _make_controllers,
    _make_scaler,
)


def _bruteforce_winner(
    g_tbl: np.ndarray,
    u_tbl: np.ndarray,
    c: int,
    active_mask: tuple[bool, bool, bool],
    scope_sids: tuple[int, ...],
) -> tuple[int, int, int]:
    """Reference O(C^3) pure-Python tuple sort matching PROTOCOL_FROZEN §5.4."""
    best = None
    best_act = None
    r0 = range(c + 1) if active_mask[0] else (0,)
    r1 = range(c + 1) if active_mask[1] else (0,)
    r2 = range(c + 1) if active_mask[2] else (0,)
    for a0 in r0:
        for a1 in r1:
            if a0 + a1 > c:
                break
            for a2 in r2:
                if a0 + a1 + a2 > c:
                    break
                act = (a0, a1, a2)
                crit = min(float(g_tbl[s, act[s]]) for s in scope_sids)
                eff = sum(float(u_tbl[s, act[s]]) for s in scope_sids)
                key = (-crit, -eff, a0 + a1 + a2, a0, a1, a2)
                if best is None or key < best:
                    best = key
                    best_act = act
    assert best_act is not None
    return best_act


def test_vectorized_lexicographic_exact_tie_breaking_matches_bruteforce():
    """Vectorized B1-raw & B1-active match exact pure-Python sort including ties."""
    rng = np.random.default_rng(12345)
    c = 15  # fast grid for thorough multi-seed brute-force parity

    # Deliberately quantized scores forcing frequent exact ties at crit, eff, sum_a
    for trial in range(20):
        f_tbl = rng.choice([-0.4, -0.2, 0.0, 0.1, 0.3], size=(3, c + 1))
        actions = np.arange(c + 1, dtype=np.float64) / float(c)
        u_tbl = f_tbl - 1.0 * actions[None, :]
        sig_tbl = np.full((3, c + 1), 0.05, dtype=np.float64)

        active = (
            bool(trial % 2 == 0 or trial % 3 == 0),
            bool(trial % 3 != 1),
            bool(trial % 5 != 0),
        )
        if not any(active):
            active = (True, False, True)
        st = JointState(
            n_e=1 if active[0] else 0,
            n_u=1 if active[1] else 0,
            n_m=1 if active[2] else 0,
        )

        # B1-raw over all 3 slices
        sel_raw = A5JointSelector(A5SelectorConfig(variant="b1_raw", capacity_c=c))
        dec_raw = sel_raw.select_from_tables(st, f_tbl, u_tbl, sig_tbl)
        exp_raw = _bruteforce_winner(f_tbl, u_tbl, c, (True, True, True), (0, 1, 2))
        assert dec_raw.action == exp_raw

        # B1-active restricted to active slices
        sel_act = A5JointSelector(A5SelectorConfig(variant="b1_active", capacity_c=c))
        dec_act = sel_act.select_from_tables(st, f_tbl, u_tbl, sig_tbl)
        scope = tuple(s for s in (0, 1, 2) if active[s])
        exp_act = _bruteforce_winner(f_tbl, u_tbl, c, active, scope)
        assert dec_act.action == exp_act


def test_causal_active_demand_and_pending_parity_with_het_guard(tmp_path):
    """B1-active/B1-support active mask matches SqrHetController.is_slice_provably_empty."""
    ctls = _make_controllers()
    scaler = _make_scaler(tmp_path)
    het = SqrHetController(
        tuple(ctls[sid] for sid in range(3)),
        scaler,
        53,
        virtual_alpha=0.0,
        empty_slice_guard=True,
        mean_field=False,
        fix_scaler_index=True,
    )

    states = [
        JointState(),  # all provably empty
        JointState(n_e=2, queue_e=1000.0),  # eMBB only
        JointState(extras={"urllc_pool_open": True}),  # URLLC pool open pending
        JointState(kpi_m_raw=0.98, extras={"mmtc_rolling_active": True}),  # mMTC 1s rolling window
        JointState(offered_u=0.256, extras={"kpm_tx_packets": (0, 5, 0)}),  # closed-win TX
        JointState(n_e=1, n_u=2, n_m=3, queue_e=500.0, queue_u=128.0, queue_m=200.0),  # all 3
    ]
    for st in states:
        expected_act = tuple(not het.is_slice_provably_empty(st, sid) for sid in range(3))
        st_dict = {
            "n_e": st.n_e, "n_u": st.n_u, "n_m": st.n_m,
            "queue_e": st.queue_e, "queue_u": st.queue_u, "queue_m": st.queue_m,
            "offered_e": st.offered_e, "offered_u": st.offered_u, "offered_m": st.offered_m,
            "kpi_m_raw": st.kpi_m_raw, "extras": st.extras,
        }
        act_obj, live_obj = causal_active_slices(st)
        act_dic, live_dic = causal_active_slices(st_dict)
        assert act_obj == expected_act
        assert act_dic == expected_act
        assert live_obj == live_dic


def test_action_legality_all_variants_capacities_and_empty_grid():
    """All A5 variants always emit legal integers in [0, C] summing to <= C."""
    rng = np.random.default_rng(77)
    for c in (10, 52, 53):
        selectors = [
            A5JointSelector(A5SelectorConfig(variant=v, capacity_c=c))
            for v in A5_VARIANTS
        ]
        for active_bits in range(8):
            a_e = bool(active_bits & 1)
            a_u = bool(active_bits & 2)
            a_m = bool(active_bits & 4)
            st = JointState(
                n_e=2 if a_e else 0,
                n_u=1 if a_u else 0,
                n_m=3 if a_m else 0,
                queue_e=1e5 if a_e else 0.0,
                queue_u=64.0 if a_u else 0.0,
                queue_m=200.0 if a_m else 0.0,
            )
            f_tbl = rng.uniform(-0.8, 0.3, size=(3, c + 1))
            sig_tbl = rng.uniform(0.0, 1.0, size=(3, c + 1))
            actions = np.arange(c + 1, dtype=np.float64) / float(c)
            u_tbl = f_tbl - actions[None, :]
            for sel in selectors:
                dec = sel.select_from_tables(st, f_tbl, u_tbl, sig_tbl, knn_actions=(18, 17, 18))
                assert len(dec.action) == 3
                assert all(isinstance(v, int) and 0 <= v <= c for v in dec.action)
                assert sum(dec.action) <= c
                if sel.config.variant in ("b1_active", "b1_support"):
                    if not any((a_e, a_u, a_m)):
                        assert dec.action == (0, 0, 0)
                        assert dec.tier == "all_empty"
                    else:
                        if not a_e:
                            assert dec.action[0] == 0
                        if not a_u:
                            assert dec.action[1] == 0
                        if not a_m:
                            assert dec.action[2] == 0
                    if sel.config.variant == "b1_support" and any((a_e, a_u, a_m)):
                        if a_e:
                            assert dec.action[0] >= 1
                        if a_u:
                            assert dec.action[1] >= 1
                        if a_m:
                            assert dec.action[2] >= 1


def test_b1_support_prevents_d3v1_extrapolation_degeneracy(tmp_path):
    """Reproduces the D3-v1 failure mode under Lambda=0 (f_tau->0 away from support)
    and verifies B1-support selects inside empirical support instead of (0,0,0)."""
    ctls = _make_controllers()
    scaler = _make_scaler(tmp_path)
    het = SqrHetController(
        tuple(ctls[sid] for sid in range(3)),
        scaler,
        53,
        virtual_alpha=0.0,
        empty_slice_guard=True,
        mean_field=False,
        fix_scaler_index=True,
    )
    st = JointState(
        n_e=2, n_u=1, n_m=2,
        queue_e=500_000.0, queue_u=128.0, queue_m=200.0,
        kpi_e_raw=2.8, kpi_u_raw=12.0, kpi_m_raw=0.98,
        snr_e=15.0, snr_u=14.0, snr_m=12.0,
        prev_a_e=25, prev_a_u=20, prev_a_m=8,
        viol_e=1, viol_u=1, viol_m=0,
    )
    zs = [het.slice_state(st, sid) for sid in range(3)]
    # Feed observations at realistic congestion allocations with slightly negative margins
    # so supported region predicts ~ -0.15..-0.25, while unvisited a=0 and a=53 extrapolate near 0.0
    for _ in range(12):
        ctls[0].observe(zs[0], 25, -0.18, threshold=0.0)
        ctls[1].observe(zs[1], 20, -0.22, threshold=0.0)
        ctls[2].observe(zs[2], 8, 0.03, threshold=0.0)

    sc0 = ctls[0].score_candidates(zs[0])
    # Sanity check: unsupported a=0 extrapolates higher (closer to 0) than supported a=25
    # Under Matern nu=1.5, length_scale=1.0, |25/53 - 0/53| = 0.472 gives sigma ~ 0.197 > 0.15
    assert sc0["f"][0] > sc0["f"][25]
    assert sc0["sigma"][0] > 0.15
    assert sc0["sigma"][25] < 0.05

    dec_raw = A5JointSelector(A5SelectorConfig(variant="b1_raw", capacity_c=53)).select_from_het(het, st)
    dec_act = A5JointSelector(A5SelectorConfig(variant="b1_active", capacity_c=53)).select_from_het(het, st)
    dec_sup = A5JointSelector(A5SelectorConfig(variant="b1_support", capacity_c=53)).select_from_het(het, st)

    # Without support awareness, D3-v1's extrapolated zero defect manifests:
    assert dec_raw.action[0] == 0 or dec_raw.action == (0, 0, 0)
    assert dec_act.action[0] == 0 or dec_act.action == (0, 0, 0)

    # B1-support selects well-supported positive allocations avoiding starvation
    assert dec_sup.tier == "joint_support"
    assert dec_sup.fallback_used is False
    assert dec_sup.max_chosen_sigma <= 0.15
    assert dec_sup.action[0] > 0 and dec_sup.action[1] > 0 and dec_sup.action[2] >= 1
    assert dec_sup.action[:2] == (25, 20)


def test_support_hierarchy_relaxed_and_ood_fallbacks():
    """Exercises Tier 2 (relaxed LCB when tight supports conflict > C) and all Tier 3 OOD fallbacks."""
    c = 53
    st = JointState(n_e=2, n_u=2, n_m=1, queue_e=400_000.0, queue_u=200.0, queue_m=100.0)
    actions = np.arange(c + 1, dtype=np.float64) / float(c)

    # Case A: Tier 1 infeasible (supported e>=30, u>=30, m>=5 => sum>=65 > 53),
    # while relaxed envelope sigma <= sigma_soft=0.55 exists around (24, 24, 5)
    f_tbl = np.full((3, c + 1), -0.3, dtype=np.float64)
    u_tbl = f_tbl - actions[None, :]
    sig_tbl = np.full((3, c + 1), 0.75, dtype=np.float64)
    sig_tbl[0, 30:] = 0.05
    sig_tbl[1, 30:] = 0.05
    sig_tbl[2, 5:12] = 0.05
    sig_tbl[0, 20:30] = 0.35  # inside sigma_soft=0.55, outside sigma_max=0.15
    sig_tbl[1, 20:30] = 0.35
    f_tbl[0, 24] = -0.10
    f_tbl[1, 24] = -0.10
    f_tbl[2, 5] = 0.02

    sel = A5JointSelector(A5SelectorConfig(variant="b1_support", capacity_c=c))
    dec_t2 = sel.select_from_tables(st, f_tbl, u_tbl, sig_tbl)
    assert dec_t2.tier == "joint_lcb_relaxed"
    assert dec_t2.fallback_used is True
    assert dec_t2.max_chosen_sigma <= 0.55
    assert sum(dec_t2.action) <= c and all(v >= 1 for v in dec_t2.action)

    # Case B: Pure OOD (all sigma > sigma_soft=0.55) -> Tier 3 deterministic fallbacks
    sig_ood = np.full((3, c + 1), 0.85, dtype=np.float64)
    # B1: full positive kNN exceeding C=53 gets bounded-projected preserving live floors >=1
    dec_knn = sel.select_from_tables(st, f_tbl, u_tbl, sig_ood, knn_actions=(40, 20, 10))
    assert dec_knn.tier == "fallback_knn"
    assert dec_knn.fallback_used is True
    assert sum(dec_knn.action) == c and all(v >= 1 for v in dec_knn.action)

    # B2: incomplete/zero-starving kNN on live slice -> delegates to prop_active rather than starving
    dec_prop = sel.select_from_tables(st, f_tbl, u_tbl, sig_ood, knn_actions=(45, 0, 5))
    assert dec_prop.tier == "fallback_prop_active"
    assert dec_prop.action == prop_active_split(st, c, (True, True, True), (1, 1, 1))
    assert sum(dec_prop.action) == c and all(v >= 1 for v in dec_prop.action)

    # B3: explicit equal_active fallback mirrors initial_equal_split for 3 active slices
    sel_eq = A5JointSelector(A5SelectorConfig(variant="b1_support", capacity_c=c, fallback_policy="equal_active"))
    dec_eq = sel_eq.select_from_tables(st, f_tbl, u_tbl, sig_ood)
    assert dec_eq.tier == "fallback_equal_active"
    assert dec_eq.action == tuple(int(v) for v in initial_equal_split(53, 3))
    assert equal_active_split(53, (True, False, True)) == (27, 0, 26)
    assert bounded_project_active((52, 1, 1), 53, (True, True, True), (1, 1, 1)) == (51, 1, 1)


def test_determinism_zero_rng_and_causal_trace_non_anticipation(tmp_path):
    """A5 consumes zero RNG, never mutates controllers, matches CausalTraceWriter parity,
    and is invariant to post-W futures and blocks future checkpoints."""
    def _run_trace(mode: str):
        sim_dir = os.path.join(str(tmp_path), f"sim_{mode}")
        run_dir = os.path.join(str(tmp_path), f"run_{mode}")
        os.makedirs(sim_dir, exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)
        env = _DivergentFutureHetEnv(
            sim_dir, n_steps=35, diverge_after_win=22, future_mode=mode
        )
        ctls = _make_controllers()
        scaler = _make_scaler(tmp_path)
        het = SqrHetController(
            tuple(ctls[sid] for sid in range(3)),
            scaler,
            53,
            virtual_alpha=0.0,
            empty_slice_guard=True,
            mean_field=False,
            fix_scaler_index=True,
        )
        boot = {
            sid: PropDemandController(
                slice_id=sid, n_prbs=53, num_slices=3,
                window_provider=lambda e=env: e.builder.last_window,
            )
            for sid in range(3)
        }
        trace_path = os.path.join(run_dir, "causal_trace.jsonl.gz")
        ckpt_path = os.path.join(run_dir, "ckpt_end.pkl")
        sqr_run.run_episode(
            env,
            ctls,
            het=het,
            het_runtime=HetRuntime(sim_dir),
            het_bootstrap=boot,
            decision_jsonl=os.path.join(run_dir, "decisions.jsonl"),
            het_meta={"run_id": f"a5_{mode}"},
            gamma=1.0,
            virtual_alpha=0.0,
            freeze=False,
            warmup_steps=15,
            seed=2102,
            checkpoint_path=ckpt_path,
            checkpoint_meta={
                "controller": "sqr_het",
                "seed": 2102,
                "gamma": 1.0,
                "virtual_alpha": 0.0,
                "empty_slice_guard": True,
                "mean_field": False,
                "fix_scaler_index": True,
                "margin_init": 10.0,
                "margin_bounds": [2.0, 50.0],
                "urllc_delayed_update": False,
            },
            causal_trace_path=trace_path,
        )
        with gzip.open(trace_path, "rt", encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh if line.strip()][1:]
        return recs, ctls, het, ckpt_path

    recs_b, ctls_b, het_b, ckpt_b = _run_trace("baseline")
    recs_s, _, _, _ = _run_trace("shock")
    rec_22_b = next(r for r in recs_b if r["window_index"] == 22)
    rec_22_s = next(r for r in recs_s if r["window_index"] == 22)

    for var in A5_VARIANTS:
        sel = A5JointSelector(A5SelectorConfig(variant=var, capacity_c=53))
        db1 = sel.select_from_trace_record(rec_22_b)
        db2 = sel.select_from_trace_record(rec_22_b)
        ds = sel.select_from_trace_record(rec_22_s)
        assert db1.action == db2.action == ds.action
        assert db1.tier == ds.tier

    # Calling select_from_het on live frozen B0-rest mutates ZERO RNG and ZERO controller state
    rng_before = copy.deepcopy([ctls_b[sid]._rng.bit_generator.state for sid in range(3)])
    sd_before = copy.deepcopy({sid: ctls_b[sid].state_dict() for sid in range(3)})
    dummy_state = JointState(n_e=1, n_u=1, n_m=1, queue_e=1024.0, queue_u=128.0, queue_m=200.0)
    for var in A5_VARIANTS:
        sel = A5JointSelector(A5SelectorConfig(variant=var, capacity_c=53))
        d_a = sel.select_from_het(het_b, dummy_state)
        d_b = sel.select_from_het(het_b, dummy_state)
        assert d_a.action == d_b.action
    rng_after = [ctls_b[sid]._rng.bit_generator.state for sid in range(3)]
    assert rng_before == rng_after
    for sid in range(3):
        b_sd, a_sd = sd_before[sid], ctls_b[sid].state_dict()
        assert b_sd["margin_m"] == a_sd["margin_m"]
        assert b_sd["regressor"]["tau"] == a_sd["regressor"]["tau"]
        assert np.array_equal(b_sd["regressor"]["alphas"], a_sd["regressor"]["alphas"])

    # Loader blocks future checkpoint reuse on past window W=22 and restores valid fix_scaler_index=True
    with pytest.raises(ValueError, match="future/same-window information leakage"):
        load_b0_rest_het_from_checkpoint(
            ckpt_b, scaler_manifest_path=REPO_SCALER_MANIFEST, target_window_index=22
        )
    _, het_loaded, _ = load_b0_rest_het_from_checkpoint(
        ckpt_b, scaler_manifest_path=REPO_SCALER_MANIFEST, target_window_index=None
    )
    assert het_loaded.fix_scaler_index is True
    assert het_loaded.empty_slice_guard is True
    assert het_loaded.mean_field is False
    assert het_loaded.virtual_alpha == 0.0


def test_single_window_aa_replay_plan_across_all_arms_and_latency(tmp_path):
    """End-to-end single-window counterfactual replay plan comparing B0-rest, equal, prop,
    and B1-{raw, active, support} with bit-identical A/A physical prefix + latency << 100 ms."""
    sim_dir = os.path.join(str(tmp_path), "b0_sim")
    run_dir = os.path.join(str(tmp_path), "b0_run")
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    env0 = _ScriptedHetEnv(sim_dir, n_steps=30, force_heavy=True)
    ctls0 = _make_controllers()
    scaler = _make_scaler(tmp_path)
    het0 = SqrHetController(
        tuple(ctls0[sid] for sid in range(3)),
        scaler,
        53,
        virtual_alpha=0.0,
        empty_slice_guard=True,
        mean_field=False,
        fix_scaler_index=True,
    )
    boot0 = {
        sid: PropDemandController(
            slice_id=sid, n_prbs=53, num_slices=3,
            window_provider=lambda e=env0: e.builder.last_window,
        )
        for sid in range(3)
    }
    trace_path = os.path.join(run_dir, "causal_trace.jsonl.gz")
    dec_jsonl = os.path.join(run_dir, "decisions.jsonl")
    sqr_run.run_episode(
        env0,
        ctls0,
        het=het0,
        het_runtime=HetRuntime(sim_dir),
        het_bootstrap=boot0,
        decision_jsonl=dec_jsonl,
        het_meta={"run_id": "b0_base"},
        gamma=1.0,
        virtual_alpha=0.0,
        freeze=False,
        warmup_steps=15,
        seed=2102,
        causal_trace_path=trace_path,
    )
    with open(dec_jsonl, "r", encoding="utf-8") as fh:
        base_decs = [json.loads(ln) for ln in fh if ln.strip()]
    seq_actions = [tuple(int(v) for v in d["action"]) for d in base_decs]
    with gzip.open(trace_path, "rt", encoding="utf-8") as fh:
        trace_recs = [json.loads(ln) for ln in fh if ln.strip()][1:]

    # Choose causal target window w=20 (decision_index=21) -> replay_at_win=21
    snap_w20 = next(r for r in trace_recs if r["window_index"] == 20)
    replay_at_win = int(snap_w20["step_index"])
    a_b0 = tuple(snap_w20["a_exec_per_slice"])
    a_equal = tuple(int(v) for v in initial_equal_split(53, 3))
    a_prop = prop_active_split(snap_w20["state"], 53, (True, True, True), (1, 1, 1))
    arms = {
        "B0_rest": a_b0,
        "equal": a_equal,
        "prop": a_prop,
    }
    for var in A5_VARIANTS:
        sel = A5JointSelector(A5SelectorConfig(variant=var, capacity_c=53))
        dec = sel.select_from_trace_record(snap_w20)
        assert dec.decision_time_ms < 100.0
        arms[var] = dec.action

    # Run bit-exact single-window replay per arm on base="seq"
    logs_by_arm: dict[str, list[str]] = {}
    for arm_name, override_action in arms.items():
        r_sim = os.path.join(str(tmp_path), f"rep_{arm_name}")
        os.makedirs(r_sim, exist_ok=True)
        r_env = _ScriptedHetEnv(r_sim, n_steps=25, force_heavy=True)
        r_ctls = {
            sid: ReplayController(
                slice_id=sid,
                n_prbs=53,
                num_slices=3,
                base="seq",
                replay_at_win=replay_at_win,
                replay_action=override_action,
                seq_actions=seq_actions,
            )
            for sid in range(3)
        }
        csv_p = os.path.join(str(tmp_path), f"step_{arm_name}.csv")
        sqr_run.run_episode(
            r_env,
            r_ctls,
            gamma=1.0,
            virtual_alpha=0.0,
            freeze=False,
            warmup_steps=0,
            seed=2102,
            step_logger=sqr_run.StepMetricsLogger(csv_p),
        )
        with open(csv_p, "r", encoding="utf-8") as fh:
            logs_by_arm[arm_name] = fh.read().splitlines()

    # Prefix strictly before replay_at_win (1 header + (replay_at_win - 1) * 3 lines) MUST match B0_rest identically
    prefix_len = 1 + (replay_at_win - 1) * 3
    ref_prefix = logs_by_arm["B0_rest"][:prefix_len]
    for arm_name, lines in logs_by_arm.items():
        assert lines[:prefix_len] == ref_prefix, f"prefix drift on arm {arm_name}"

    # Verify runtime complexity with B=2000 support points per slice stays << 100 ms decision period
    full_ctls = _make_controllers()
    scaler_full = _make_scaler(tmp_path)
    het_full = SqrHetController(
        tuple(full_ctls[sid] for sid in range(3)),
        scaler_full,
        53,
        virtual_alpha=0.0,
        empty_slice_guard=True,
        mean_field=False,
        fix_scaler_index=True,
    )
    rng_fast = np.random.default_rng(99)
    for sid in range(3):
        for k in range(2000):
            z_rnd = rng_fast.uniform(0.0, 1.0, size=6)
            full_ctls[sid].regressor.insert_support_vector(
                np.concatenate([z_rnd, [float((k % 53) + 1) / 53.0]]),
                alpha=-0.01 if k % 3 == 0 else 0.005,
            )
    st_bench = JointState(n_e=2, n_u=2, n_m=2, queue_e=3e5, queue_u=128.0, queue_m=200.0)
    sel_bench = A5JointSelector(A5SelectorConfig(variant="b1_support", capacity_c=53))
    lat_samples = [sel_bench.select_from_het(het_full, st_bench).decision_time_ms for _ in range(5)]
    assert min(lat_samples) < 100.0
