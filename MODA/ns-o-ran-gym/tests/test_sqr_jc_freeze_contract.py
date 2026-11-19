"""Tests for the B0 freeze/resume contract, manifest/payload hashing,
and non-invasive CausalTraceWriter parity (Item 6 / D3-v2 prep).
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

import sqr_run  # noqa: E402
from nsoran.sqr.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint  # noqa: E402
from nsoran.sqr.replay_controller import ReplayController  # noqa: E402
from nsoran.sqr.sqr_controller import SqrSliceController  # noqa: E402
from nsoran.sqr.sqr_env import SqrObservationBuilder  # noqa: E402
from nsoran.sqr.static_controllers import PropDemandController  # noqa: E402
from nsoran.sqr_jc.feature_scaler import FeatureScaler  # noqa: E402
from nsoran.sqr_jc.freeze_contract import (  # noqa: E402
    CAUSAL_TRACE_VERSION,
    FREEZE_CONTRACT_VERSION,
    build_b0_config_descriptor,
    build_version_manifest,
    canonical_json_sha256,
    checkpoint_payload_sha256,
    sha256_file,
    verify_causal_replay_no_future_checkpoint,
    verify_causal_snapshot_matches_controllers,
    verify_freeze_no_learning,
    verify_resume_contract,
)
from nsoran.sqr_jc.het_runtime import HetRuntime  # noqa: E402
from nsoran.sqr_jc.sqr_het import SqrHetController  # noqa: E402


REPO_SCALER_MANIFEST = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "results",
        "sqr_jc_final",
        "03_calibration",
        "feature_scaler_manifest.yaml",
    )
)


class _ScriptedHetEnv:
    """Minimal self-contained closed-loop mock exercising SqrObservationBuilder + delays."""

    def __init__(self, sim_dir: str, n_steps: int = 50, force_heavy: bool = True) -> None:
        self.sim_path = sim_dir
        self.n_steps = n_steps
        self.force_heavy = force_heavy
        self.total_prbs = 53
        self.scenario_configuration: dict = {}
        self.builder = SqrObservationBuilder(num_slices=3, sla_thresholds={0: 3.5, 1: 3.5, 2: 3.5})
        self._step = 0
        self._kpm_path = os.path.join(sim_dir, "sqr-kpms.txt")
        self._delay_path = os.path.join(sim_dir, "sqr_packet_delays.csv")

    def _emit_window(self, w: int) -> None:
        ts_ms = (w + 1) * 100
        q_e = 500000 if self.force_heavy else 1024
        t_e = 2.6 if (w % 4 == 0) else 4.5
        rows = [
            "timestamp,ueImsi,sliceId,dTxBytes,dRxBytes,plr,resourceSharePct,dLostPackets,throughputMbps,bufferBytes,windowThroughputMbps,snrDb,queueBytes,dTxPackets,dRxPackets",
            f"{ts_ms},1,0,12000,12000,0.0,40.0,0,{t_e},{q_e}, {t_e},15.0,{q_e},10,10",
            f"{ts_ms},2,1,3200,3200,0.0,30.0,0,0.256,128,0.256,14.0,128,100,100",
            f"{ts_ms},3,2,200,200,0.0,30.0,0,0.0016,200,0.0016,12.0,200,1,1",
        ]
        with open(self._kpm_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(rows) + "\n")
        if w == 0:
            with open(self._delay_path, "w", encoding="utf-8") as fh:
                fh.write("ueImsi,sliceId,txSec,delaySec\n")
        delay_s = 0.013 if (w % 5 == 0) else 0.005
        with open(self._delay_path, "a", encoding="utf-8") as fh:
            for k in range(10):
                tx_sec = w * 0.1 + 0.004 * k
                fh.write(f"2,1,{tx_sec:.6f},{delay_s:.6f}\n")

    def reset(self):
        self._step = 0
        self._emit_window(0)
        self.builder.update_from_kpm(self._kpm_path)
        return self.builder.build_obs(), {}

    def step(self, proposals):
        proposals = [int(v) for v in proposals]
        total = sum(proposals)
        if total > self.total_prbs and total > 0:
            alloc = [int(v * self.total_prbs // total) for v in proposals]
            alloc[0] += self.total_prbs - sum(alloc)
            cut = [p > a for p, a in zip(proposals, alloc)]
        else:
            alloc = list(proposals)
            cut = [False, False, False]

        self._step += 1
        self._emit_window(self._step)
        win = self.builder.update_from_kpm(self._kpm_path)
        info = {
            "kpm_timestamp": win.timestamp,
            "total_ues": win.total_ues,
            "total_prbs": self.total_prbs,
            "pooled_violation_rate": self.builder.pooled_violation_rate(),
            "kpm_missing": False,
            "kpm_missing_steps": 0,
            "projection": {
                "prb_requested": proposals,
                "prb_executed": alloc,
                "cut": cut,
                "over_capacity": any(cut),
                "virtual_updates": [
                    {"slice_id": i, "proposed": proposals[i], "allocated": alloc[i], "deficit": proposals[i] - alloc[i]}
                    for i in range(3) if cut[i]
                ],
            },
        }
        for sid in range(3):
            k = win.slices[sid]
            info[f"slice_{sid}"] = {
                "n_ues": k.n_ues,
                "p5_thr_mbps": k.p5_thr_mbps,
                "lowest_thr_mbps": k.lowest_thr_mbps,
                "mean_thr_mbps": k.mean_thr_mbps,
                "queue_bytes_sum": k.queue_bytes_sum,
                "violation_rate": k.violation_rate,
                "n_snr_imputed": k.n_snr_imputed,
                "snr_imputations": dict(k.snr_imputations),
            }
        return self.builder.build_obs(), 0.0, False, self._step >= self.n_steps, info


def _make_controllers(seed: int = 2102) -> dict[int, SqrSliceController]:
    return {
        sid: SqrSliceController(
            n_prbs=53,
            sla_threshold=0.0,
            gamma=1.0,
            margin=10.0,
            margin_bounds=(2.0, 50.0),
            seed=seed + 1000 * sid,
            state_dim=6,
        )
        for sid in range(3)
    }


def _make_scaler(tmp_path) -> FeatureScaler:
    if os.path.isfile(REPO_SCALER_MANIFEST):
        return FeatureScaler.load_manifest(REPO_SCALER_MANIFEST)
    rng = np.random.default_rng(7)
    data = np.column_stack([rng.uniform(0, 10, 100)] * 23)
    return FeatureScaler.fit(data)


def test_version_manifest_and_config_descriptor_hashes():
    manifest_arg = REPO_SCALER_MANIFEST if os.path.isfile(REPO_SCALER_MANIFEST) else None
    vm = build_version_manifest(scaler_manifest_path=manifest_arg)
    assert vm["freeze_contract_version"] == FREEZE_CONTRACT_VERSION
    assert vm["causal_trace_version"] == CAUSAL_TRACE_VERSION
    assert vm["sla_manifest_version"] == "1.1-g4b"
    assert isinstance(vm["sla_manifest_sha256"], str) and len(vm["sla_manifest_sha256"]) == 64
    if manifest_arg is not None:
        assert isinstance(vm["scaler_manifest_sha256"], str) and len(vm["scaler_manifest_sha256"]) == 64

    desc_b0, sha_b0 = build_b0_config_descriptor(
        controller="sqr_het",
        gamma=1.0,
        c_prbs=53,
        virtual_alpha=0.0,
        empty_slice_guard=True,
        fix_scaler_index=True,
        mean_field=False,
        margin_init=10.0,
        margin_bounds=(2.0, 50.0),
        scaler_manifest_path=manifest_arg,
    )
    assert desc_b0["mean_field"] is False
    assert desc_b0["fix_scaler_index"] is True
    _, sha_mf = build_b0_config_descriptor(
        controller="sqr_het",
        gamma=1.0,
        c_prbs=53,
        virtual_alpha=0.0,
        empty_slice_guard=True,
        fix_scaler_index=True,
        mean_field=True,
        margin_init=10.0,
        margin_bounds=(2.0, 50.0),
        scaler_manifest_path=manifest_arg,
    )
    assert sha_b0 != sha_mf


def test_checkpoint_payload_sha256_and_tamper_detection(tmp_path):
    ctls = _make_controllers()
    ctls[0].observe(np.array([0.2, 0.4, 0.5, 0.6, 0.0, 0.0]), action=20, y=0.3)
    ckpt_path = str(tmp_path / "ckpt.pkl")
    meta = {
        "controller": "sqr_het",
        "gamma": 1.0,
        "virtual_alpha": 0.0,
        "empty_slice_guard": True,
        "mean_field": False,
        "fix_scaler_index": True,
        "margin_init": 10.0,
        "margin_bounds": [2.0, 50.0],
        "config_sha256": "abc123",
    }
    sqr_run._save_checkpoint_with_hash(ckpt_path, ctls, meta)
    loaded = load_checkpoint(ckpt_path)
    assert "payload_sha256" in loaded["meta"]
    assert loaded["meta"]["payload_sha256"] == checkpoint_payload_sha256(loaded)
    verify_resume_contract(loaded, meta, _make_controllers())

    tampered = copy.deepcopy(loaded)
    tampered["slices"][0]["margin_m"] = 99.0
    with pytest.raises(ValueError, match="payload_sha256 mismatch"):
        verify_resume_contract(tampered, meta, _make_controllers())


def test_resume_contract_rejects_mismatches_and_keeps_legacy_compat(tmp_path):
    ctls = _make_controllers()
    base_meta = {
        "controller": "sqr_het",
        "gamma": 1.0,
        "virtual_alpha": 0.0,
        "empty_slice_guard": True,
        "mean_field": False,
        "fix_scaler_index": True,
        "margin_init": 10.0,
        "margin_bounds": [2.0, 50.0],
        "urllc_delayed_update": False,
        "config_sha256": "hash-b0",
    }
    payload = build_checkpoint(ctls, meta=base_meta)
    verify_resume_contract(payload, base_meta, _make_controllers())

    for key, bad_val in (
        ("mean_field", True),
        ("fix_scaler_index", False),
        ("virtual_alpha", 0.05),
        ("empty_slice_guard", False),
        ("margin_bounds", [1.0, 10.0]),
        ("urllc_delayed_update", True),
        ("config_sha256", "other-hash"),
    ):
        req = dict(base_meta)
        req[key] = bad_val
        with pytest.raises(ValueError, match="mismatch"):
            verify_resume_contract(payload, req, _make_controllers())

    # Legacy checkpoint lacking extended keys still resumes when matching base flags
    legacy_payload = build_checkpoint(ctls, meta={"gamma": 1.0, "fix_scaler_index": True})
    restored = verify_resume_contract(legacy_payload, base_meta, _make_controllers())
    assert restored["gamma"] == 1.0


def test_freeze_no_learning_and_resume_source_preservation(tmp_path, monkeypatch):
    sim_dir = str(tmp_path / "sim")
    out_train = str(tmp_path / "train")
    out_eval = str(tmp_path / "eval")
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_eval, exist_ok=True)

    # First run: train for 30 windows and emit checkpoint with full contract metadata
    env_train = _ScriptedHetEnv(sim_dir, n_steps=30)
    monkeypatch.setattr(sqr_run, "SqrEnv", lambda *a, **kw: env_train)
    train_argv = [
        "--controller", "sqr_het",
        "--seed", "2102",
        "--c_prbs", "53",
        "--gamma", "1",
        "--virtual_alpha", "0.0",
        "--empty_slice_guard",
        "--fix_scaler_index",
        "--margin_init", "10.0",
        "--margin_bounds", "2,50",
        "--warmup_s", "1.0",
        "--output", out_train,
    ]
    if os.path.isfile(REPO_SCALER_MANIFEST):
        train_argv += ["--het_scaler_manifest", REPO_SCALER_MANIFEST]
    sqr_run.main(train_argv)

    ckpt_file = os.path.join(out_train, "sqr_checkpoint.pkl")
    assert os.path.isfile(ckpt_file)
    ckpt_sha_before = sha256_file(ckpt_file)
    ckpt_payload_before = load_checkpoint(ckpt_file)
    assert "config_sha256" in ckpt_payload_before["meta"]
    assert "payload_sha256" in ckpt_payload_before["meta"]

    # Second run: --resume --freeze must keep source checkpoint untouched & state unmutated
    sim_eval = str(tmp_path / "sim_eval")
    os.makedirs(sim_eval, exist_ok=True)
    env_eval = _ScriptedHetEnv(sim_eval, n_steps=30)
    monkeypatch.setattr(sqr_run, "SqrEnv", lambda *a, **kw: env_eval)
    eval_argv = [
        "--controller", "sqr_het",
        "--seed", "2102",
        "--c_prbs", "53",
        "--gamma", "1",
        "--virtual_alpha", "0.0",
        "--empty_slice_guard",
        "--fix_scaler_index",
        "--margin_init", "10.0",
        "--margin_bounds", "2,50",
        "--warmup_s", "1.0",
        "--resume",
        "--freeze",
        "--checkpoint_path", ckpt_file,
        "--output", out_eval,
    ]
    if os.path.isfile(REPO_SCALER_MANIFEST):
        eval_argv += ["--het_scaler_manifest", REPO_SCALER_MANIFEST]
    sqr_run.main(eval_argv)

    assert sha256_file(ckpt_file) == ckpt_sha_before
    # Also explicitly verify run_episode freeze invariance on loaded controllers
    fresh = _make_controllers()
    for sid in range(3):
        fresh[sid].load_state_dict(ckpt_payload_before["slices"][sid])
    before_states = {sid: copy.deepcopy(fresh[sid].state_dict()) for sid in range(3)}
    sim_eval2 = str(tmp_path / "sim_eval2")
    os.makedirs(sim_eval2, exist_ok=True)
    env_eval2 = _ScriptedHetEnv(sim_eval2, n_steps=30)
    scaler = _make_scaler(tmp_path)
    het = SqrHetController(
        tuple(fresh[sid] for sid in range(3)),
        scaler,
        53,
        virtual_alpha=0.0,
        empty_slice_guard=True,
        mean_field=False,
        fix_scaler_index=True,
    )
    boot = {
        sid: PropDemandController(slice_id=sid, n_prbs=53, num_slices=3, window_provider=lambda: env_eval2.builder.last_window)
        for sid in range(3)
    }
    sqr_run.run_episode(
        env_eval2,
        fresh,
        het=het,
        het_runtime=HetRuntime(sim_eval2),
        het_bootstrap=boot,
        gamma=1.0,
        virtual_alpha=0.0,
        freeze=True,
        warmup_steps=10,
        seed=2102,
    )
    after_states = {sid: fresh[sid].state_dict() for sid in range(3)}
    verify_freeze_no_learning(before_states, after_states)


def test_causal_trace_strict_causality_and_bit_exact_parity(tmp_path):
    scaler = _make_scaler(tmp_path)

    def _run(use_trace: bool) -> tuple[str, list[dict], list[list[str]], str, str | None]:
        run_dir = str(tmp_path / ("with_trace" if use_trace else "no_trace"))
        sim_dir = os.path.join(run_dir, "sim")
        os.makedirs(sim_dir, exist_ok=True)
        env = _ScriptedHetEnv(sim_dir, n_steps=50)
        ctls = _make_controllers(seed=2102)
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
                slice_id=sid,
                n_prbs=53,
                num_slices=3,
                window_provider=lambda e=env: e.builder.last_window,
            )
            for sid in range(3)
        }
        csv_path = os.path.join(run_dir, "step_metrics.csv")
        jsonl_path = os.path.join(run_dir, "decisions.jsonl")
        trace_path = os.path.join(run_dir, "causal_trace.jsonl.gz") if use_trace else None
        logger = sqr_run.StepMetricsLogger(csv_path)
        sqr_run.run_episode(
            env,
            ctls,
            het=het,
            het_runtime=HetRuntime(sim_dir),
            het_bootstrap=boot,
            decision_jsonl=jsonl_path,
            het_meta={"run_id": "parity_test"},
            gamma=1.0,
            virtual_alpha=0.0,
            freeze=False,
            warmup_steps=15,
            seed=2102,
            step_logger=logger,
            causal_trace_path=trace_path,
            causal_trace_meta=(
                {
                    "run_id": "parity_test",
                    "versions": {
                        "freeze_contract_version": FREEZE_CONTRACT_VERSION,
                        "causal_trace_version": CAUSAL_TRACE_VERSION,
                    },
                    "config_sha256": "parity-cfg-hash",
                }
                if use_trace
                else None
            ),
        )
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            dec_text = fh.read()
        dec_rows = [json.loads(line) for line in dec_text.splitlines() if line.strip()]
        with open(csv_path, "r", encoding="utf-8") as fh:
            csv_text = fh.read()
        final_hash = canonical_json_sha256({sid: ctls[sid].state_dict() for sid in range(3)})
        return dec_text, dec_rows, csv_text, final_hash, trace_path

    dec_no, rows_no, csv_no, hash_no, _ = _run(use_trace=False)
    dec_tr, rows_tr, csv_tr, hash_tr, trace_path = _run(use_trace=True)

    # Bit-exact parity across decisions, step metrics, and final controller state
    assert dec_no == dec_tr
    assert csv_no == csv_tr
    assert hash_no == hash_tr

    with gzip.open(trace_path, "rt", encoding="utf-8") as fh:
        trace_lines = [json.loads(line) for line in fh if line.strip()]
    assert trace_lines[0]["record_type"] == "causal_trace_header"
    assert trace_lines[0]["causal_trace_version"] == CAUSAL_TRACE_VERSION
    assert trace_lines[0]["metadata"]["config_sha256"] == "parity-cfg-hash"
    assert trace_lines[0]["metadata"]["versions"]["freeze_contract_version"] == FREEZE_CONTRACT_VERSION

    trace_windows = [r for r in trace_lines[1:] if r["record_type"] == "causal_window"]
    het_decs = [r for r in rows_tr if r["policy"] == "SQR_HET"]
    assert len(trace_windows) == len(het_decs)
    assert len(trace_windows) == 35  # 50 steps - 15 warmup steps

    # Strict causality and exact instantaneous model prediction correspondence
    prev_ts = -1
    for tw, dec in zip(trace_windows, het_decs):
        assert tw["sim_time_ms"] > prev_ts
        prev_ts = tw["sim_time_ms"]
        assert tw["causal_trace_version"] == CAUSAL_TRACE_VERSION
        assert tw["governed_window_index"] == tw["window_index"] + 1
        assert tw["causal_horizon_max_window_index"] == tw["window_index"]
        assert tw["causal_horizon_max_window_index"] < tw["governed_window_index"]
        assert tw["causal_horizon_max_sim_time_ms"] == tw["sim_time_ms"]
        assert tw["step_index"] == dec["decision_index"]
        assert tw["a_raw_per_slice"] == dec["a_raw_per_slice"]
        assert tw["a_prop_per_slice"] == dec["a_after_uncertainty_margin"]
        assert tw["a_exec_per_slice"] == dec["action"]
        assert len(tw["per_slice_model"]) == 3
        for sid, psm in enumerate(tw["per_slice_model"]):
            assert "knn_action_raw" in psm and "knn_mode" in psm
            assert len(psm["f_candidates"]) == 54  # 0..53 inclusive
            assert len(psm["scores_candidates"]) == 54
            assert len(psm["sigma_candidates"]) == 54
            assert psm["f_candidates"][0] == pytest.approx(dec["f_at_a0"][sid], abs=1e-12)
            assert psm["f_candidates"][37] == pytest.approx(dec["f_at_a37"][sid], abs=1e-12)
            assert psm["f_candidates"][53] == pytest.approx(dec["f_at_amax"][sid], abs=1e-12)

    # Also verify deferred trace flush (_wait_data_availability hook after control release)
    # yields bit-identical trace records to immediate flush.
    class _DeferredHookEnv(_ScriptedHetEnv):
        def __init__(self, sim_dir: str, n_steps: int = 25) -> None:
            super().__init__(sim_dir, n_steps=n_steps, force_heavy=True)
            self.wait_calls = 0

        def _wait_data_availability(self) -> None:
            self.wait_calls += 1

        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            self._wait_data_availability()
            return obs, reward, term, trunc, info

    sim_def = os.path.join(str(tmp_path), "sim_def")
    run_def = os.path.join(str(tmp_path), "run_def")
    os.makedirs(sim_def, exist_ok=True)
    os.makedirs(run_def, exist_ok=True)
    env_def = _DeferredHookEnv(sim_def, n_steps=50)
    ctls_def = _make_controllers()
    het_def = SqrHetController(
        tuple(ctls_def[sid] for sid in range(3)),
        _make_scaler(tmp_path),
        53,
        virtual_alpha=0.0,
        empty_slice_guard=True,
        mean_field=False,
        fix_scaler_index=True,
    )
    boot_def = {
        sid: PropDemandController(
            slice_id=sid, n_prbs=53, num_slices=3,
            window_provider=lambda e=env_def: e.builder.last_window,
        )
        for sid in range(3)
    }
    trace_def = os.path.join(run_def, "causal_trace_deferred.jsonl.gz")
    sqr_run.run_episode(
        env_def,
        ctls_def,
        het=het_def,
        het_runtime=HetRuntime(sim_def),
        het_bootstrap=boot_def,
        decision_jsonl=os.path.join(run_def, "decisions.jsonl"),
        het_meta={"run_id": "parity_test"},
        gamma=1.0,
        virtual_alpha=0.0,
        freeze=False,
        warmup_steps=15,
        seed=2102,
        causal_trace_path=trace_def,
        causal_trace_meta={
            "run_id": "parity_test",
            "versions": {
                "freeze_contract_version": FREEZE_CONTRACT_VERSION,
                "causal_trace_version": CAUSAL_TRACE_VERSION,
            },
            "config_sha256": "parity-cfg-hash",
        },
    )
    assert env_def.wait_calls == 50
    with gzip.open(trace_def, "rt", encoding="utf-8") as fh:
        trace_def_lines = [json.loads(line) for line in fh if line.strip()]
    assert trace_def_lines == trace_lines


class _DivergentFutureHetEnv(_ScriptedHetEnv):
    """Identical physical windows for w <= diverge_after_win, shocked future afterwards."""

    def __init__(
        self,
        sim_dir: str,
        *,
        n_steps: int = 45,
        diverge_after_win: int = 25,
        future_mode: str = "baseline",
    ) -> None:
        super().__init__(sim_dir, n_steps=n_steps, force_heavy=True)
        self.diverge_after_win = int(diverge_after_win)
        self.future_mode = str(future_mode)
        self._last_alloc = (18, 17, 18)

    def _emit_window(self, w: int) -> None:
        if w <= self.diverge_after_win or self.future_mode == "baseline":
            super()._emit_window(w)
            return
        ts_ms = (w + 1) * 100
        a_e, a_u, a_m = self._last_alloc
        t_e = round(0.08 * a_e + 0.1, 4)
        q_e = 9_999_999
        delay_s = 0.048 if a_u < 20 else 0.004
        rows = [
            "timestamp,ueImsi,sliceId,dTxBytes,dRxBytes,plr,resourceSharePct,dLostPackets,throughputMbps,bufferBytes,windowThroughputMbps,snrDb,queueBytes,dTxPackets,dRxPackets",
            f"{ts_ms},1,0,24000,6000,0.5,40.0,1,{t_e},{q_e},{t_e},9.0,{q_e},20,10",
            f"{ts_ms},2,1,3200,3200,0.0,30.0,0,0.256,4096,0.256,11.0,4096,100,100",
            f"{ts_ms},3,2,400,200,0.0,30.0,0,0.0005,800,0.0005,8.0,800,2,1",
        ]
        with open(self._kpm_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(rows) + "\n")
        if w == 0:
            with open(self._delay_path, "w", encoding="utf-8") as fh:
                fh.write("ueImsi,sliceId,txSec,delaySec\n")
        with open(self._delay_path, "a", encoding="utf-8") as fh:
            for k in range(10):
                tx_sec = w * 0.1 + 0.004 * k
                fh.write(f"2,1,{tx_sec:.6f},{delay_s:.6f}\n")

    def step(self, proposals):
        proposals = [int(v) for v in proposals]
        total = sum(proposals)
        if total > self.total_prbs and total > 0:
            alloc = [int(v * self.total_prbs // total) for v in proposals]
            alloc[0] += self.total_prbs - sum(alloc)
        else:
            alloc = list(proposals)
        self._last_alloc = tuple(alloc)
        return super().step(proposals)


def test_causal_trace_non_anticipation_and_future_divergence(tmp_path):
    """Snapshot at window W is produced pre-action and unaffected by future windows > W."""

    def _run_future(mode: str, capture_at_win: int = 25):
        sim_dir = os.path.join(str(tmp_path), f"sim_{mode}")
        run_dir = os.path.join(str(tmp_path), f"run_{mode}")
        os.makedirs(sim_dir, exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)
        env = _DivergentFutureHetEnv(
            sim_dir, n_steps=45, diverge_after_win=capture_at_win, future_mode=mode
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
                slice_id=sid,
                n_prbs=53,
                num_slices=3,
                window_provider=lambda e=env: e.builder.last_window,
            )
            for sid in range(3)
        }
        trace_path = os.path.join(run_dir, "causal_trace.jsonl.gz")
        jsonl_path = os.path.join(run_dir, "decisions.jsonl")
        ckpt_path = os.path.join(run_dir, "ckpt_end.pkl")
        sqr_run.run_episode(
            env,
            ctls,
            het=het,
            het_runtime=HetRuntime(sim_dir),
            het_bootstrap=boot,
            decision_jsonl=jsonl_path,
            het_meta={"run_id": f"future_{mode}"},
            gamma=1.0,
            virtual_alpha=0.0,
            freeze=False,
            warmup_steps=15,
            seed=2102,
            checkpoint_path=ckpt_path,
            checkpoint_interval=25,
            checkpoint_meta={"controller": "sqr_het"},
            causal_trace_path=trace_path,
        )
        with gzip.open(trace_path, "rt", encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh if line.strip()][1:]
        return recs, ctls, ckpt_path

    recs_base, ctls_base, ckpt_base = _run_future("baseline", capture_at_win=25)
    recs_shock, _, _ = _run_future("shock", capture_at_win=25)

    assert len(recs_base) == len(recs_shock) == 30  # 45 - 15 warmup
    # Every causal snapshot through W=25 (including exact decision snapshot at W=25)
    # MUST be bit-identical even though physical traffic after W=25 diverges wildly.
    for rb, rs in zip(recs_base, recs_shock):
        if rb["window_index"] <= 25:
            assert rb == rs
    # Post-divergence window > 25 must diverge in state/metrics.
    late_b = [r for r in recs_base if r["window_index"] > 26]
    late_s = [r for r in recs_shock if r["window_index"] > 26]
    assert late_b != late_s

    # Snapshot verifier rejects end-of-episode controllers when compared to W=25 snapshot
    snap_25 = next(r for r in recs_base if r["window_index"] == 25)
    with pytest.raises(ValueError, match="state hash mismatch"):
        verify_causal_snapshot_matches_controllers(snap_25, ctls_base, capacity_c=53)
    # And passes against an exact frozen snapshot at step=0 with zero warmup
    fresh_ctls = _make_controllers()
    sim_p = os.path.join(str(tmp_path), "sim_pos")
    run_p = os.path.join(str(tmp_path), "run_pos")
    os.makedirs(sim_p, exist_ok=True)
    os.makedirs(run_p, exist_ok=True)
    env_p = _ScriptedHetEnv(sim_p, n_steps=1, force_heavy=True)
    scaler_p = _make_scaler(tmp_path)
    het_p = SqrHetController(
        tuple(fresh_ctls[sid] for sid in range(3)),
        scaler_p,
        53,
        virtual_alpha=0.0,
        empty_slice_guard=True,
        mean_field=False,
        fix_scaler_index=True,
    )
    snap_trace_p = os.path.join(run_p, "snap.jsonl.gz")
    # Run in freeze=True so controllers remain identical to the pre-decision snapshot at W=0
    before_ctls = copy.deepcopy(fresh_ctls)
    sqr_run.run_episode(
        env_p,
        fresh_ctls,
        het=het_p,
        het_runtime=HetRuntime(sim_p),
        het_bootstrap=None,
        decision_jsonl=os.path.join(run_p, "decisions.jsonl"),
        het_meta={"run_id": "snap_pos"},
        gamma=1.0,
        virtual_alpha=0.0,
        freeze=True,
        warmup_steps=0,
        seed=2102,
        causal_trace_path=snap_trace_p,
    )
    with gzip.open(snap_trace_p, "rt", encoding="utf-8") as fh:
        pos_rec = json.loads(fh.read().splitlines()[1])
    verify_causal_snapshot_matches_controllers(pos_rec, before_ctls, capacity_c=53)
    with pytest.raises(ValueError, match="capacity mismatch"):
        verify_causal_snapshot_matches_controllers(pos_rec, before_ctls, capacity_c=52)


def test_causal_replay_refuses_future_or_missing_horizon_checkpoints(tmp_path):
    """Forbids reusing future/same-window or un-stamped checkpoints on past window W,
    while allowing distinct out-of-sample --freeze --resume evaluation from step=0."""
    sim_dir = os.path.join(str(tmp_path), "sim")
    run_dir = os.path.join(str(tmp_path), "run")
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    env = _ScriptedHetEnv(sim_dir, n_steps=40, force_heavy=True)
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
            slice_id=sid,
            n_prbs=53,
            num_slices=3,
            window_provider=lambda e=env: e.builder.last_window,
        )
        for sid in range(3)
    }
    ckpt_path = os.path.join(run_dir, "sqr_ckpt.pkl")
    meta = {
        "controller": "sqr_het",
        "gamma": 1.0,
        "virtual_alpha": 0.0,
        "empty_slice_guard": True,
        "mean_field": False,
        "fix_scaler_index": True,
        "margin_init": 10.0,
        "margin_bounds": [2.0, 50.0],
        "urllc_delayed_update": False,
        "config_sha256": "b0-frozen-test",
    }
    sqr_run.run_episode(
        env,
        ctls,
        het=het,
        het_runtime=HetRuntime(sim_dir),
        het_bootstrap=boot,
        decision_jsonl=os.path.join(run_dir, "decisions.jsonl"),
        het_meta={"run_id": "horizon_test"},
        gamma=1.0,
        virtual_alpha=0.0,
        freeze=False,
        warmup_steps=15,
        seed=2102,
        checkpoint_path=ckpt_path,
        checkpoint_meta=meta,
    )

    end_payload = load_checkpoint(ckpt_path)
    horizon = end_payload["meta"]["checkpoint_horizon"]
    assert horizon["max_step_index"] == 40
    assert horizon["max_window_index"] == 39
    assert horizon["max_sim_time_ms"] == 4000

    # End-of-episode checkpoint (W_max=39) MUST be refused for counterfactual replay at W=20 or W=39
    with pytest.raises(ValueError, match="future/same-window information leakage"):
        verify_causal_replay_no_future_checkpoint(
            end_payload, target_window_index=20, target_sim_time_ms=2100, target_step_index=21
        )
    with pytest.raises(ValueError, match="future/same-window information leakage"):
        verify_causal_replay_no_future_checkpoint(
            end_payload, target_window_index=39, target_sim_time_ms=4000
        )

    # Checkpoint lacking checkpoint_horizon provenance MUST be refused for causal window replay
    no_horizon_payload = copy.deepcopy(end_payload)
    del no_horizon_payload["meta"]["checkpoint_horizon"]
    with pytest.raises(ValueError, match="lacks 'checkpoint_horizon'"):
        verify_causal_replay_no_future_checkpoint(no_horizon_payload, target_window_index=20)

    # Strictly pre-decision checkpoint (max_window_index < target_window_index) IS accepted
    accepted = verify_causal_replay_no_future_checkpoint(
        end_payload, target_window_index=40, target_sim_time_ms=4100, target_step_index=41
    )
    assert accepted["max_window_index"] == 39

    # Contrast: closed-loop out-of-sample evaluation (--resume --freeze from step=0) uses end_payload
    # on an independent trajectory and mutates zero parameters.
    frozen_ctls = _make_controllers()
    verify_resume_contract(end_payload, meta, frozen_ctls)
    from nsoran.sqr.checkpoint import restore_controllers

    restore_controllers(end_payload, frozen_ctls)
    before_state = copy.deepcopy({sid: c.state_dict() for sid, c in frozen_ctls.items()})
    eval_sim = os.path.join(str(tmp_path), "eval_sim")
    os.makedirs(eval_sim, exist_ok=True)
    eval_env = _DivergentFutureHetEnv(eval_sim, n_steps=25, diverge_after_win=5, future_mode="shock")
    eval_scaler = _make_scaler(tmp_path)
    eval_het = SqrHetController(
        tuple(frozen_ctls[sid] for sid in range(3)),
        eval_scaler,
        53,
        virtual_alpha=0.0,
        empty_slice_guard=True,
        mean_field=False,
        fix_scaler_index=True,
    )
    sqr_run.run_episode(
        eval_env,
        frozen_ctls,
        het=eval_het,
        het_runtime=HetRuntime(eval_sim),
        het_bootstrap=None,
        decision_jsonl=os.path.join(str(tmp_path), "eval_decisions.jsonl"),
        het_meta={"run_id": "oos_eval"},
        gamma=1.0,
        virtual_alpha=0.0,
        freeze=True,
        warmup_steps=0,
        seed=3101,
    )
    after_state = {sid: c.state_dict() for sid, c in frozen_ctls.items()}
    verify_freeze_no_learning(before_state, after_state)


def test_physical_state_identity_in_aa_replay_branches(tmp_path):
    """A/A replay produces bit-identical physical states across the whole run;
    A/B replay shares a bit-identical prefix strictly before override window t."""

    class _AllocSensitiveEnv(_ScriptedHetEnv):
        def __init__(self, sim_dir: str, n_steps: int = 30) -> None:
            super().__init__(sim_dir, n_steps=n_steps, force_heavy=True)
            self._last_alloc = (18, 17, 18)

        def _emit_window(self, w: int) -> None:
            ts_ms = (w + 1) * 100
            a_e, a_u, a_m = self._last_alloc
            t_e = round(0.15 * a_e + 0.5, 4)
            q_e = 100000 + 1000 * max(0, 30 - a_e)
            delay_s = 0.004 if a_u >= 15 else 0.025
            rows = [
                "timestamp,ueImsi,sliceId,dTxBytes,dRxBytes,plr,resourceSharePct,dLostPackets,throughputMbps,bufferBytes,windowThroughputMbps,snrDb,queueBytes,dTxPackets,dRxPackets",
                f"{ts_ms},1,0,12000,12000,0.0,40.0,0,{t_e},{q_e},{t_e},15.0,{q_e},10,10",
                f"{ts_ms},2,1,3200,3200,0.0,30.0,0,0.256,{128 + a_u},0.256,14.0,{128 + a_u},100,100",
                f"{ts_ms},3,2,200,200,0.0,30.0,0,0.0016,{200 + a_m},0.0016,12.0,{200 + a_m},1,1",
            ]
            with open(self._kpm_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(rows) + "\n")
            if w == 0:
                with open(self._delay_path, "w", encoding="utf-8") as fh:
                    fh.write("ueImsi,sliceId,txSec,delaySec\n")
            with open(self._delay_path, "a", encoding="utf-8") as fh:
                for k in range(5):
                    fh.write(f"2,1,{w * 0.1 + 0.004 * k:.6f},{delay_s:.6f}\n")

        def step(self, proposals):
            proposals = [int(v) for v in proposals]
            total = sum(proposals)
            if total > self.total_prbs and total > 0:
                alloc = [int(v * self.total_prbs // total) for v in proposals]
                alloc[0] += self.total_prbs - sum(alloc)
            else:
                alloc = list(proposals)
            self._last_alloc = tuple(alloc)
            return super().step(proposals)

    def _run_replay(tag: str, override_action: tuple[int, int, int], t_win: int = 12):
        sim_dir = os.path.join(str(tmp_path), f"sim_{tag}")
        os.makedirs(sim_dir, exist_ok=True)
        env = _AllocSensitiveEnv(sim_dir, n_steps=25)
        ctls = {
            sid: ReplayController(
                slice_id=sid,
                n_prbs=53,
                num_slices=3,
                base="prop",
                replay_at_win=t_win,
                replay_action=override_action,
                window_provider=lambda e=env: e.builder.last_window,
            )
            for sid in range(3)
        }
        csv_path = os.path.join(str(tmp_path), f"step_{tag}.csv")
        logger = sqr_run.StepMetricsLogger(csv_path)
        sqr_run.run_episode(
            env,
            ctls,
            gamma=1.0,
            virtual_alpha=0.0,
            freeze=False,
            warmup_steps=0,
            seed=2102,
            step_logger=logger,
        )
        with open(csv_path, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        return lines

    lines_a1 = _run_replay("A1", (18, 17, 18), t_win=12)
    lines_a2 = _run_replay("A2", (18, 17, 18), t_win=12)
    lines_b = _run_replay("B", (35, 10, 8), t_win=12)

    # Exact A/A replay yields bit-identical physical trajectories across the full run
    assert lines_a1 == lines_a2
    # 1 header + 25 steps * 3 slices = 76 lines
    assert len(lines_a1) == 76
    # Pre-override prefix (header + steps 1..11 = 1 + 33 lines) is bit-identical between A and B;
    # at step 12 decision differs, and step 13+ physical KPMs diverge.
    prefix_rows = 1 + 11 * 3
    assert lines_a1[:prefix_rows] == lines_b[:prefix_rows]
    assert lines_a1[prefix_rows:] != lines_b[prefix_rows:]
