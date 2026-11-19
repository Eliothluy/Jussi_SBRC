"""Etapa 2 regression, fault-injection, A/A parity, BUG-05, and BUG-04 tests for sqr-v2."""

from __future__ import annotations

import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
GYM_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(GYM_ROOT, "..", ".."))
NS3_PATH = os.path.join(REPO_ROOT, "MODA", "ns-3-dev")
EXAMPLES_DIR = os.path.join(GYM_ROOT, "examples")
SRC_DIR = os.path.join(GYM_ROOT, "src")

for p in (SRC_DIR, EXAMPLES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from nsoran.sqr.kernel import MaternKernel
from nsoran.sqr.quantile_regressor import KernelQuantileRegressor
from nsoran.sqr.sqr_controller import SqrSliceController
from nsoran.sqr.static_controllers import EqualSplitController
from nsoran.sqr_v2.campaign_analysis_v2 import (
    compute_seed_scalars_v2,
    compute_seed_series_v2,
    run_analysis_v2,
)
from nsoran.sqr_v2.controller_v2 import (
    KernelQuantileRegressorV2,
    SqrSliceControllerV2,
)
from nsoran.sqr_v2.env_v2 import SqrV2Env
from nsoran.sqr_v2.ipc_v2 import write_action_v2_atomic
from nsoran.sqr_v2.manifest_v2 import (
    build_v2_manifest,
    validate_dev_seed,
    verify_b0_frozen_manifest,
)
import sqr_v2_run


def _strict_rt_env(
    tmp_path,
    *,
    seed: int = 101,
    sim_time: float = 3.0,
    cell_radius: float = 300.0,
    slice_lambdas: str | None = None,
    ipc_timeout_ms: int = 50,
    ipc_mode: str = "strict_rt",
    run_id: str | None = None,
) -> SqrV2Env:
    cfg = {
        "simTime": [sim_time],
        "seed": [seed],
        "periodMs": [100],
        "cellRadius": [cell_radius],
    }
    if slice_lambdas:
        cfg["sliceLambdas"] = [slice_lambdas]
    return SqrV2Env(
        ns3_path=NS3_PATH,
        scenario_configuration=cfg,
        output_folder=str(tmp_path),
        total_prbs=52,
        num_slices=3,
        ipc_mode=ipc_mode,
        ipc_timeout_ms=ipc_timeout_ms,
        run_id=run_id,
    )


def test_b0_frozen_manifest_and_seed_governance():
    status = verify_b0_frozen_manifest(REPO_ROOT)
    assert status["code_total"] == 17
    assert status["code_matched"] == 17
    assert status["checkpoints_total"] == 5
    assert status["checkpoints_matched"] == 5
    assert status["all_ok"] is True

    # Permitted dev/smoke seeds pass
    for ok_seed in (1, 21, 101, 102, 103, 2101, 2102, 2103):
        assert validate_dev_seed(ok_seed) == ok_seed

    # Reserved confirmatory seeds blocked
    for bad_seed in (1201, 1205, 1209, 3001, 3005, 3010, 3101, 3200):
        with pytest.raises(ValueError, match="reserved confirmatory set"):
            validate_dev_seed(bad_seed)


def test_predict_and_uncertainty_numerical_equivalence_and_speed():
    rng = np.random.default_rng(42)
    X_eval = rng.uniform(0.0, 1.0, size=(54, 7))

    for n_mem in (0, 1, 50, 2000):
        k1 = MaternKernel(dim=7, nu=1.5, length_scale=1.0)
        k2 = MaternKernel(dim=7, nu=1.5, length_scale=1.0)
        r1 = KernelQuantileRegressor(kernel=k1, memory_size=2000)
        r2 = KernelQuantileRegressorV2(kernel=k2, memory_size=2000)
        if n_mem > 0:
            sx = rng.uniform(0.0, 1.0, size=(n_mem, 7))
            al = rng.normal(0.0, 0.1, size=(n_mem,))
            r1.set_memory(sx, al)
            r2.set_memory(sx, al)
        f1 = r1.predict(X_eval)
        s1 = r1.uncertainty(X_eval)
        f2, s2 = r2.predict_and_uncertainty(X_eval)
        assert float(np.max(np.abs(f1 - f2))) < 1e-14
        assert float(np.max(np.abs(s1 - s2))) < 1e-14

    # Decision equivalence between v1 SqrSliceController and v2 SqrSliceControllerV2
    z = rng.uniform(0.0, 1.0, size=(6,))
    c1 = SqrSliceController(state_dim=6, n_prbs=53, gamma=1.0, sla_threshold=1.0, seed=7)
    c2 = SqrSliceControllerV2(state_dim=6, n_prbs=53, gamma=1.0, sla_threshold=1.0, seed=7)
    for _ in range(100):
        zz = rng.uniform(0.0, 1.0, size=(6,))
        aa = int(rng.integers(0, 54))
        yy = float(rng.uniform(0.0, 2.5))
        c1.observe(zz, aa, yy)
        c2.observe(zz, aa, yy)
    assert c1.act(z) == c2.act(z)


def test_causal_vs_rt_separation_in_manifest(tmp_path):
    m_causal = build_v2_manifest(
        run_id="r_causal",
        controller="sqr",
        seed=101,
        ipc_mode="causal_sim",
        ipc_timeout_ms=2000,
        repo_root=REPO_ROOT,
    )
    assert m_causal["ipc"]["ipc_mode"] == "causal_sim"
    assert m_causal["ipc"]["ipc_timeout_ms"] == 2000
    assert m_causal["ipc"]["rt_50ms_deadline_enforced"] is False
    assert m_causal["b0_frozen_integrity"]["all_ok"] is True

    m_rt = build_v2_manifest(
        run_id="r_rt",
        controller="sqr",
        seed=101,
        ipc_mode="strict_rt",
        ipc_timeout_ms=50,
        repo_root=REPO_ROOT,
    )
    assert m_rt["ipc"]["ipc_mode"] == "strict_rt"
    assert m_rt["ipc"]["ipc_timeout_ms"] == 50
    assert m_rt["ipc"]["rt_50ms_deadline_enforced"] is True


def test_bug05_step1_capacity_sync_equal_controller_real_ns3(tmp_path):
    env = _strict_rt_env(
        tmp_path,
        seed=101,
        sim_time=2.5,
        cell_radius=300.0,
        ipc_timeout_ms=50,
    )
    ctls = {sid: EqualSplitController(slice_id=sid, n_prbs=52, num_slices=3) for sid in range(3)}
    logger = sqr_v2_run.StepMetricsV2Logger(str(tmp_path / "step_metrics.csv"))
    try:
        summary = sqr_v2_run.run_v2_episode(
            env, ctls, gamma=1.0, seed=101, step_logger=logger
        )
        ns3_exit = env.wait_sim_exit(timeout_s=60.0)
        cpp = env.read_cpp_ipc_summary()
    finally:
        env.close()

    assert ns3_exit == 0
    assert summary["total_prbs_final"] == 53
    assert summary["timeout_hold_steps"] == 0
    assert cpp is not None
    assert cpp["timeout_hold_count"] == 0
    assert cpp["applied_count"] == cpp["windows_total"]

    df = pd.read_csv(tmp_path / "step_metrics.csv")
    s1 = df[df["step"] == 1].sort_values("slice_id")
    assert s1["prb_requested_ipc"].tolist() == [18, 18, 17]
    assert s1["prb_executed_mac"].tolist() == [18, 18, 17]
    assert int(s1["window_seq"].iloc[0]) == 1
    assert int(s1["applied_action_seq"].iloc[0]) == 1


def test_ipc_v2_fault_injection_four_non_delay_classes_real_ns3(tmp_path):
    """Test wrong run_id, malformed (truncated/<3 slices/NaN), obsolete seq<w, and duplicate seq==w."""
    env = _strict_rt_env(
        tmp_path,
        seed=102,
        sim_time=2.2,
        cell_radius=300.0,
        ipc_timeout_ms=50,
    )
    ctls = {sid: EqualSplitController(slice_id=sid, n_prbs=52, num_slices=3) for sid in range(3)}

    def _inject(e: SqrV2Env, seq: int, _proposals: list[int]) -> None:
        t_ms = e.current_sim_time_ms
        c = e.total_prbs
        rows_valid = [
            (0, 18, 18, 18, 100.0 * 18 / c, 1),
            (1, 18, 18, 18, 100.0 * 18 / c, 1),
            (2, 17, 17, 17, 100.0 * 17 / c, 1),
        ]
        if seq == 2:
            # Class 5: wrong run_id (must be rejected; valid seq=2 arrives right after)
            write_action_v2_atomic(
                e.sim_path,
                run_id="foreign_run_999",
                window_seq=seq,
                sim_time_ms=t_ms,
                slice_rows=rows_valid,
                agent_decision_wall_ms=1.0,
                append_history=False,
            )
            e.controlSemaphore.release()
            time.sleep(0.004)
        elif seq == 4:
            # Class 4a: truncated payload missing # END_V2
            bad_path = os.path.join(e.sim_path, "sqr_ipc_action_v2.csv")
            with open(bad_path, "w", encoding="utf-8") as bf:
                bf.write("run_id,window_seq,sim_time_ms,slice_id,prb_proposed,prb_requested,prb_allocated,dedicated_pct,algorithm,agent_decision_wall_ms\n")
                bf.write(f"{e.run_id},{seq},{t_ms},0,18,18,18,33.96,1,1.0\n")
            e.controlSemaphore.release()
            time.sleep(0.003)
            # Class 4b: only 2 slices even with # END_V2
            write_action_v2_atomic(
                e.sim_path,
                run_id=e.run_id,
                window_seq=seq,
                sim_time_ms=t_ms,
                slice_rows=rows_valid[:2],
                agent_decision_wall_ms=1.0,
                append_history=False,
            )
            e.controlSemaphore.release()
            time.sleep(0.003)
            # Class 4c: NaN float value
            with open(bad_path, "w", encoding="utf-8") as bf:
                bf.write("run_id,window_seq,sim_time_ms,slice_id,prb_proposed,prb_requested,prb_allocated,dedicated_pct,algorithm,agent_decision_wall_ms\n")
                bf.write(f"{e.run_id},{seq},{t_ms},0,18,18,18,nan,1,1.0\n")
                bf.write(f"{e.run_id},{seq},{t_ms},1,18,18,18,33.96,1,1.0\n")
                bf.write(f"{e.run_id},{seq},{t_ms},2,17,17,17,32.08,1,1.0\n")
                bf.write("# END_V2\n")
            e.controlSemaphore.release()
            time.sleep(0.003)
        elif seq == 6:
            # Class 1: obsolete old response (window_seq = 5 < 6, already != lastApplied or < w)
            # Note: use seq=4 (< lastApplied=5, < current=6) so ParseAndValidateActionCsvV2 hits OBSOLETE
            write_action_v2_atomic(
                e.sim_path,
                run_id=e.run_id,
                window_seq=4,
                sim_time_ms=400,
                slice_rows=rows_valid,
                agent_decision_wall_ms=1.0,
                append_history=False,
            )
            e.controlSemaphore.release()
            time.sleep(0.004)
        elif seq == 8:
            # Class 2a: duplicate response carrying last applied window_seq == 7
            write_action_v2_atomic(
                e.sim_path,
                run_id=e.run_id,
                window_seq=7,
                sim_time_ms=700,
                slice_rows=rows_valid,
                agent_decision_wall_ms=1.0,
                append_history=False,
            )
            e.controlSemaphore.release()
            time.sleep(0.004)
        elif seq == 10:
            # Class 2b: duplicate delivery of the EXACT same window seq==10 after valid apply
            orig_sem = e.controlSemaphore

            class _DoubleSendSameSeqProxy:
                def __init__(self, inner):
                    self._inner = inner

                def release(self):
                    e.controlSemaphore = self._inner
                    self._inner.release()
                    time.sleep(0.003)
                    write_action_v2_atomic(
                        e.sim_path,
                        run_id=e.run_id,
                        window_seq=seq,
                        sim_time_ms=t_ms,
                        slice_rows=rows_valid,
                        agent_decision_wall_ms=1.0,
                        append_history=False,
                    )
                    self._inner.release()
                    time.sleep(0.003)

                def __getattr__(self, name):
                    return getattr(self._inner, name)

            e.controlSemaphore = _DoubleSendSameSeqProxy(orig_sem)

    logger = sqr_v2_run.StepMetricsV2Logger(str(tmp_path / "step_metrics.csv"))
    try:
        summary = sqr_v2_run.run_v2_episode(
            env, ctls, gamma=1.0, seed=102, step_logger=logger, step_hook=_inject
        )
        ns3_exit = env.wait_sim_exit(timeout_s=60.0)
        cpp = env.read_cpp_ipc_summary()
    finally:
        env.close()

    assert ns3_exit == 0
    assert cpp is not None
    assert cpp["wrong_run_discarded_count"] >= 1
    assert cpp["malformed_discarded_count"] >= 3
    assert cpp["obsolete_discarded_count"] >= 1
    assert cpp["duplicate_discarded_count"] >= 2
    assert cpp["timeout_hold_count"] == 0
    assert cpp["applied_count"] == cpp["windows_total"]

    df = pd.read_csv(tmp_path / "step_metrics.csv")
    # All windows still applied with exact window_seq == applied_action_seq
    assert (df["window_seq"] == df["applied_action_seq"]).all()
    assert (df["ipc_status"] == "APPLIED").all()


def test_ipc_v2_delays_single_alternating_burst_and_mac_learning_only(tmp_path):
    """Class 3 (>50ms delays): single, alternating, 3-step burst -> zero permanent shift + MAC-only learning."""
    env = _strict_rt_env(
        tmp_path,
        seed=103,
        sim_time=7.0,
        cell_radius=300.0,
        slice_lambdas="24,24,24",
        ipc_timeout_ms=50,
    )
    ctls = {
        sid: SqrSliceControllerV2(
            state_dim=6, n_prbs=52, gamma=1.0, sla_threshold=0.5, seed=103 + 1000 * sid
        )
        for sid in range(3)
    }
    # Force an unmistakable proposal (41 PRBs on slice 0 -> sum > 53) on delayed windows
    # across single (25), alternating (30, 32), and 3-step burst (40, 41, 42) patterns.
    delayed_seqs = {25, 30, 32, 40, 41, 42}
    orig_act_0 = ctls[0].act
    active_seq_box = {"seq": 0}

    def _act0(z):
        d = orig_act_0(z)
        if active_seq_box["seq"] in delayed_seqs:
            from nsoran.sqr.sqr_controller import SqrDecision
            return SqrDecision(
                action=41,
                raw_action=41,
                mode="confident",
                sigma=d.sigma,
                margin_applied=0,
                a_safe_size=1,
                n_candidates=d.n_candidates,
                tau=d.tau,
                margin_m=d.margin_m,
            )
        return d

    ctls[0].act = _act0

    def _delay_hook(e: SqrV2Env, seq: int, _proposals: list[int]) -> None:
        if seq in delayed_seqs:
            time.sleep(0.075)  # 75 ms > 50 ms ipcTimeoutMs

    orig_reset = env.reset
    orig_step = env.step

    def _tracked_reset(*args, **kwargs):
        out = orig_reset(*args, **kwargs)
        active_seq_box["seq"] = env.current_window_seq
        return out

    def _tracked_step(action):
        res = orig_step(action)
        active_seq_box["seq"] = env.current_window_seq
        return res

    env.reset = _tracked_reset
    env.step = _tracked_step

    logger = sqr_v2_run.StepMetricsV2Logger(str(tmp_path / "step_metrics.csv"))
    try:
        sqr_v2_run.run_v2_episode(
            env,
            ctls,
            gamma=1.0,
            virtual_alpha=0.05,
            seed=103,
            step_logger=logger,
            step_hook=_delay_hook,
        )
        ns3_exit = env.wait_sim_exit(timeout_s=60.0)
        cpp = env.read_cpp_ipc_summary()
    finally:
        env.close()

    assert ns3_exit == 0
    assert cpp is not None
    assert cpp["timeout_hold_count"] >= len(delayed_seqs)

    df = pd.read_csv(tmp_path / "step_metrics.csv")
    df0 = df[df["slice_id"] == 0].set_index("window_seq")

    learned_on_hold = 0
    for dseq in delayed_seqs:
        assert dseq in df0.index
        row = df0.loc[dseq]
        assert row["ipc_status"] == "TIMEOUT_HOLD"
        assert int(row["prb_requested_ipc"]) == 41
        assert int(row["prb_executed_mac"]) != 41
        assert int(row["virtual_updates"]) == 0
        if int(row["learned"]) == 1:
            learned_on_hold += 1
    assert learned_on_hold >= 1  # non-vacuous proof that observe() actually ran on TIMEOUT_HOLD

    # Immediately following non-delayed windows after single, alternating, and 3-step burst recover with ZERO shift
    for clean_next in (26, 31, 33, 43):
        row = df0.loc[clean_next]
        assert row["ipc_status"] == "APPLIED"
        assert int(row["applied_action_seq"]) == int(clean_next)

    # Regressor memory must contain ONLY MAC-executed actions, never the rejected 41
    assert ctls[0].regressor.n_support > 0
    learned_actions_norm = ctls[0].regressor.support_points[:, -1]
    assert not np.any(np.isclose(learned_actions_norm, 41.0 / 53.0, atol=1e-6))


def test_closed_loop_aa_parity_strict_rt_real_ns3(tmp_path):
    """Execute two independent closed-loop SQR runs (A1, A2) under strict 50ms RT budget and verify bit-exact parity."""
    dir_a = tmp_path / "run_A1"
    dir_b = tmp_path / "run_A2"

    common = [
        "--controller", "sqr",
        "--gamma", "1",
        "--seed", "101",
        "--simTime", "10.0",
        "--cellRadius", "300",
        "--sliceLambdas", "12,12,12",
        "--warmup_s", "2.0",
        "--ipc-mode", "strict_rt",
        "--ipc-timeout-ms", "50",
        "--checkpoint_interval", "0",
    ]
    rc_a = sqr_v2_run.main(common + ["--output", str(dir_a), "--run-id", "aa_parity_A1"])
    rc_b = sqr_v2_run.main(common + ["--output", str(dir_b), "--run-id", "aa_parity_A2"])
    assert rc_a == 0 and rc_b == 0

    df_a = pd.read_csv(dir_a / "step_metrics.csv")
    df_b = pd.read_csv(dir_b / "step_metrics.csv")
    assert len(df_a) == len(df_b) > 0
    assert df_a["n_ues"].sum() > 0
    assert df_a["learned"].sum() > 0

    det_cols = [
        c for c in df_a.columns
        if c not in {"run_id", "decision_time_ms", "agent_decision_wall_ms", "ns3_wait_wall_ms", "over_50ms"}
    ]
    pd.testing.assert_frame_equal(df_a[det_cols], df_b[det_cols], check_exact=True)

    with open(dir_a / "sqr_checkpoint.pkl", "rb") as fa:
        ck_a = pickle.load(fa)
    with open(dir_b / "sqr_checkpoint.pkl", "rb") as fb:
        ck_b = pickle.load(fb)
    assert ck_a["meta"]["payload_sha256"] == ck_b["meta"]["payload_sha256"]

    import json as _json
    done_a = _json.loads((dir_a / "run_done.json").read_text())
    done_b = _json.loads((dir_b / "run_done.json").read_text())
    assert done_a["rt_50ms_deadline_enforced"] is True
    assert done_a["cpp_ipc_v2_summary"]["timeout_hold_count"] == 0
    assert done_b["cpp_ipc_v2_summary"]["timeout_hold_count"] == 0
    assert done_a["cpp_ipc_v2_summary"]["applied_count"] == done_a["cpp_ipc_v2_summary"]["windows_total"]
    assert done_b["cpp_ipc_v2_summary"]["applied_count"] == done_b["cpp_ipc_v2_summary"]["windows_total"]


def test_bug04_invariants_synthetic_and_v1_reanalysis(tmp_path):
    """Synthetic 3-slice x 40-step test verifying .first() idle_est and slot vs window separation."""
    c_prbs = 53
    rows = []
    win_rows = []
    for step in range(1, 41):
        exec_vec = [18, 12, 10] if step % 2 == 0 else [18, 18, 17]
        unalloc = c_prbs - sum(exec_vec)
        dl_slots = 56 if step % 2 == 0 else 58
        sched_vec = [e * dl_slots - 2 for e in exec_vec]
        budget_vec = [e * dl_slots for e in exec_vec]
        idle_win = c_prbs * dl_slots - sum(sched_vec)
        win_rows.append({
            "timestamp_ms": step * 100,
            "total_rbg": c_prbs,
            "total_prb": c_prbs,
            "dl_slots": dl_slots,
            "prb_executed_rbg_0": exec_vec[0],
            "prb_executed_rbg_1": exec_vec[1],
            "prb_executed_rbg_2": exec_vec[2],
            "prb_scheduled_rbg_0": sched_vec[0],
            "prb_scheduled_rbg_1": sched_vec[1],
            "prb_scheduled_rbg_2": sched_vec[2],
            "prb_budget_rbg_0": budget_vec[0],
            "prb_budget_rbg_1": budget_vec[1],
            "prb_budget_rbg_2": budget_vec[2],
            "prb_idle_rbg": idle_win,
        })
        for sid in range(3):
            rows.append({
                "episode": 1,
                "step": step,
                "timestamp_ms": step * 100,
                "seed": 101,
                "gamma": 1.0,
                "slice_id": sid,
                "mode": "confident",
                "raw_action": exec_vec[sid],
                "action": exec_vec[sid],
                "prb_requested": exec_vec[sid],
                "prb_executed": exec_vec[sid],
                "prb_scheduled": sched_vec[sid],
                "prb_idle_est": unalloc,
                "sigma": 0.1,
                "margin_m": 10.0,
                "margin_applied": 0,
                "tau": 0.05,
                "y_t_p5_mbps": 1.6,
                "lowest_thr_mbps": 1.6,
                "n_ues": 2,
                "violation_rate": 0.0,
                "n_snr_imputed": 0,
                "kpm_missing": 0,
                "learned": 1,
                "virtual_updates": 0,
                "decision_time_ms": 12.0,
                "reward": 0.0,
                "warmup": 0,
                "terminated": False,
                "truncated": False,
            })

    df = pd.DataFrame(rows)
    pw = pd.DataFrame(win_rows)
    sc = compute_seed_scalars_v2(df, {0: 0.5, 1: 1.0, 2: 1.5}, 100, pw)
    assert abs((sc["rb_slot_executed_cell_mean"] + sc["rb_slot_unallocated_quota_mean"]) - 53.0) < 1e-9
    assert sc["rb_slot_conservation_abs_err"] < 1e-9
    assert sc["rb_win_conservation_abs_err"] < 1e-9
    assert abs(sc["rb_slot_unallocated_quota_mean"] - 6.5) < 1e-9  # half 13, half 0 (not 3x!)

    ser = compute_seed_series_v2(df, pw)
    assert "slot_scheduled_active_dl" in ser
    assert "win_capacity_rbg_slots" in ser

    # Verify v1 reanalysis outputs exist and satisfy slot-level conservation
    re_tab = os.path.join(
        GYM_ROOT, "results", "sqr_v2", "reanalysis_v1_bug04", "tables", "errata_bug04_v1_vs_v2.csv"
    )
    if os.path.isfile(re_tab):
        err_df = pd.read_csv(re_tab)
        assert len(err_df) == 4
        assert (err_df["v2_slot_conservation_err"] < 1e-9).all()
        g0 = err_df[err_df["method"] == "sqr-g0"].iloc[0]
        assert abs(g0["v1_buggy_idle_est_3x_sum"] / 3.0 - g0["v2_fixed_idle_est_first_slot"]) < 1e-4
