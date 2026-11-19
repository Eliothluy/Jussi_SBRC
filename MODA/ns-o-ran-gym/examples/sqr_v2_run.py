"""SQR v2 closed-loop runner over IPC Handshake v2 (Etapa 2: BUG-03 + BUG-05)."""

from __future__ import annotations

from collections import deque
import argparse
import csv
from dataclasses import replace
import json
import os
import sys
import time
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsoran.compute_accounting import ComputeAccounting
from nsoran.sqr.checkpoint import build_checkpoint, save_checkpoint
from nsoran.sqr.coordinator import DEFAULT_VIRTUAL_ALPHA, apply_virtual_updates, project_prbs
from nsoran.sqr.sqr_controller import (
    MARGIN_BOUNDS,
    MARGIN_INIT,
    TAU_INIT,
    VIOLATION_WINDOW,
)
from nsoran.sqr.static_controllers import PropDemandController
from nsoran.sqr_jc.feature_scaler import FeatureScaler
from nsoran.sqr_jc.freeze_contract import checkpoint_payload_sha256
from nsoran.sqr_jc.het_runtime import DelayStream, HetRuntime
from nsoran.sqr_jc.joint_state import JointState
from nsoran.sqr_jc.sla_margin import SliceOutcome
from nsoran.sqr_v2.cer_reallocator import (
    CerBlocoATelemetry,
    CerConfig,
    CerReallocator,
    DL_SLOTS_PER_WIN,
    PrequentialRiskTracker,
    causal_active_mask,
    extract_bloco_a_telemetry,
)
from nsoran.sqr_v2.controller_v2 import (
    build_controllers_v2,
    parse_action_length_scales_v2,
    parse_margin_bounds_v2,
    parse_slas_v2,
)
from nsoran.sqr_v2.env_v2 import Ns3PrbWindowTelemetryV2, SqrV2Env
from nsoran.sqr_v2.manifest_v2 import (
    DEFAULT_INFRA_REV,
    DEFAULT_LEARNING_ABLATION,
    DEFAULT_PHY_QOS_PROFILE,
    build_v2_manifest,
    validate_dev_seed,
    write_v2_manifest,
)
from nsoran.sqr_v2.priority_hoashi import (
    PriorityHoashiConfig,
    PriorityHoashiReallocator,
    parse_priority_weights,
)

DEFAULT_NS3_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ns-3-dev")
)
DEFAULT_OUTPUT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results", "sqr_v2", "run")
)
DEFAULT_HET_SCALER_MANIFEST = os.path.abspath(
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
_URLLC_DELAYED_MAX_AGE = 25
_HET_KPI_FEATURE = ("kpi_e", "kpi_u", "kpi_m")

STEP_METRICS_V2_COLUMNS = [
    "episode",
    "step",
    "timestamp_ms",
    "seed",
    "gamma",
    "slice_id",
    "mode",
    "raw_action",
    "action",
    "prb_requested",
    "prb_executed",
    "prb_scheduled",
    "prb_idle_est",
    "sigma",
    "margin_m",
    "margin_applied",
    "tau",
    "y_t_p5_mbps",
    "lowest_thr_mbps",
    "n_ues",
    "violation_rate",
    "n_snr_imputed",
    "kpm_missing",
    "learned",
    "virtual_updates",
    "decision_time_ms",
    "reward",
    "warmup",
    "terminated",
    "truncated",
    # Additive v2 causal & timing columns
    "run_id",
    "window_seq",
    "applied_action_seq",
    "ipc_status",
    "prb_requested_ipc",
    "prb_executed_mac",
    "agent_decision_wall_ms",
    "ns3_wait_wall_ms",
    "over_50ms",
]


class StepMetricsV2Logger:
    """Append-safe writer for v2 step_metrics.csv."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._initialized = False

    def write(self, rows: list[list[Any]]) -> None:
        if not self._initialized:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(STEP_METRICS_V2_COLUMNS)
            self._initialized = True
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)


def _sync_controllers_capacity(controllers: dict[int, Any], new_c: int) -> None:
    if int(new_c) <= 0:
        return
    for ctl in controllers.values():
        setter = getattr(ctl, "set_n_prbs", None)
        if setter is not None:
            setter(int(new_c))
        else:
            ctl.n_prbs = int(new_c)


def _projection_for_virtual(proj: dict[str, Any], total_prbs: int):
    from nsoran.sqr.coordinator import ProjectionResult, VirtualUpdate

    updates = tuple(
        VirtualUpdate(
            slice_id=int(u["slice_id"]),
            proposed=int(u["proposed"]),
            allocated=int(u["allocated"]),
            deficit=int(u["deficit"]),
        )
        for u in proj.get("virtual_updates", [])
    )
    return ProjectionResult(
        allocated=np.asarray(proj.get("prb_executed", []), dtype=np.int64),
        proposals=np.asarray(proj.get("prb_requested", []), dtype=np.int64),
        total_prbs=int(total_prbs),
        cut=np.asarray(proj.get("cut", []), dtype=bool),
        virtual_updates=updates,
    )


def _extract_online_bloco_a(env: SqrV2Env, num_slices: int) -> CerBlocoATelemetry:
    """Build causal Bloco A telemetry from last closed KPM window (safe fallback pre-step-1)."""
    builder = getattr(env, "builder", None)
    last_win = getattr(builder, "last_window", None) if builder is not None else None
    if last_win is not None:
        return extract_bloco_a_telemetry(last_win)
    return CerBlocoATelemetry(
        n_ues=(1, 1, 1)[:num_slices] + (0,) * max(0, 3 - num_slices),  # type: ignore[arg-type]
        queue_bytes=(0.0, 0.0, 0.0),
        offered_mbps=(0.0, 0.0, 0.0),
        snr_db=(10.0, 10.0, 10.0),
        tx_pkts_prev=(0, 0, 0),
    )


def _load_cer_config(
    *,
    cer_mode: str,
    idle_empty_guard_prbs: int,
    cer_dl_slots: int | None = None,
    cer_config_json: str | None = None,
) -> CerConfig:
    overrides: dict[str, Any] = {}
    if cer_config_json:
        raw_arg = str(cer_config_json).strip()
        if os.path.isfile(raw_arg):
            with open(raw_arg, "r", encoding="utf-8") as cf:
                payload = json.load(cf)
        else:
            payload = json.loads(raw_arg)
        if not isinstance(payload, dict):
            raise ValueError("--cer-config-json must decode to a JSON object")
        overrides.update(payload)
    overrides["regime_a_only"] = cer_mode == "regime_a_only"
    overrides["idle_empty_guard_prbs"] = int(idle_empty_guard_prbs)
    if cer_dl_slots is not None:
        overrides["dl_slots_per_win"] = int(cer_dl_slots)
    return CerConfig(**overrides)


def _build_het_slice_states(
    state: JointState,
    scaler: FeatureScaler,
    total_prbs: int,
    num_slices: int,
) -> list[np.ndarray]:
    """Build SQR-HET canonical 6-dim normalized states z_i (with fix_scaler_index=True)."""
    c = max(1.0, float(total_prbs))
    letters = "eum"
    ns = (state.n_e, state.n_u, state.n_m)
    queues = (state.queue_e, state.queue_u, state.queue_m)
    kpis = (state.kpi_e_raw, state.kpi_u_raw, state.kpi_m_raw)
    snrs = (state.snr_e, state.snr_u, state.snr_m)
    viols = (state.viol_e, state.viol_u, state.viol_m)
    prev_as = (state.prev_a_e, state.prev_a_u, state.prev_a_m)
    out: list[np.ndarray] = []
    for sid in range(num_slices):
        letter = letters[sid]
        out.append(
            np.array(
                [
                    float(scaler.transform_feature(f"n_{letter}", float(ns[sid]))),
                    float(scaler.transform_feature(f"queue_{letter}", float(queues[sid]))),
                    float(scaler.transform_feature(_HET_KPI_FEATURE[sid], float(kpis[sid] or 0.0))),
                    float(scaler.transform_feature(f"snr_{letter}", float(snrs[sid]))),
                    float(prev_as[sid]) / c,
                    float(viols[sid]),
                ],
                dtype=np.float64,
            )
        )
    return out


class _BootstrapDecisionWrapper:
    """Wrapper matching SqrDecision fields during bootstrap-PROP."""

    def __init__(self, prop_action: int, alloc_action: int) -> None:
        self.mode = "bootstrap_prop"
        self.raw_action = int(prop_action)
        self.action = int(alloc_action)
        self.sigma = float("nan")
        self.margin_m = float("nan")
        self.margin_applied = 0
        self.tau = float("nan")


def _summarize_and_write_het_windows(
    sim_path: str | None,
    records: list[dict[str, Any]],
    *,
    output_csv_path: str | None = None,
) -> dict[str, Any]:
    """Reconcile final uncensored URLLC delays on disk after sim exit and summarize."""
    if not records:
        return {"n_windows": 0, "n_post_warmup": 0}
    final_delays = (
        DelayStream(os.path.join(str(sim_path), "**", "sqr_packet_delays.csv"))
        if sim_path
        else None
    )
    if final_delays is not None:
        final_delays.poll()

    rows_out: list[dict[str, Any]] = []
    for rec in records:
        gov_w = int(rec["window_index"])
        online_u_p95 = rec["urllc_p95_ms_online"]
        online_u_miss = rec["urllc_miss_pct_online"]
        u_stat = final_delays.stats(gov_w) if final_delays is not None else None
        if u_stat is not None:
            final_u_p95 = float(u_stat["p95_ms"])
            final_u_miss = float(100.0 * u_stat["miss_rate"])
            final_u_pkts = int(u_stat["n_pkts"])
        else:
            final_u_p95 = online_u_p95
            final_u_miss = online_u_miss
            final_u_pkts = 0
        final_oc = SliceOutcome(
            embb_throughput_mbps=rec["embb_thr_mbps"],
            urllc_p95_ms=final_u_p95,
            urllc_miss_pct=final_u_miss,
            mmtc_pdr_rolling=rec["mmtc_pdr_rolling"],
        )
        ok_e, ok_u, ok_m = final_oc.sla_ok
        m_e, m_u, m_m = final_oc.margins
        urllc_tx_pkts = int(rec.get("urllc_tx_pkts", 0) or 0)
        urllc_blackout = (final_u_p95 is None) and (final_u_pkts == 0) and (urllc_tx_pkts > 0)
        if urllc_blackout:
            final_u_miss = 100.0
            m_u = -1.0
            ok_u = False
        joint_ok = bool(ok_e and ok_u and ok_m)
        row = {
            **rec,
            "urllc_p95_ms": final_u_p95,
            "urllc_miss_pct": final_u_miss,
            "urllc_n_pkts": final_u_pkts,
            "margin_e": m_e,
            "margin_u": m_u,
            "margin_m": m_m,
            "sla_ok_e": int(ok_e),
            "sla_ok_u": int(ok_u),
            "sla_ok_m": int(ok_m),
            "sla_ok_all3": int(joint_ok),
        }
        rows_out.append(row)

    if output_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        cols = list(rows_out[0].keys())
        with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for r in rows_out:
                writer.writerow(
                    {
                        k: ("" if r[k] is None else r[k])
                        for k in cols
                    }
                )

    post = [r for r in rows_out if not int(r["warmup"])]
    target = post if post else rows_out
    embb_vals = [float(r["embb_thr_mbps"]) for r in target if r["embb_thr_mbps"] is not None]
    u_vals = [float(r["urllc_p95_ms"]) for r in target if r["urllc_p95_ms"] is not None]
    u_miss_vals = [float(r["urllc_miss_pct"]) for r in target if r["urllc_miss_pct"] is not None]
    m_vals = [float(r["mmtc_pdr_rolling"]) for r in target if r["mmtc_pdr_rolling"] is not None]
    util_vals = [
        100.0 * float(r["a_exec_0"] + r["a_exec_1"] + r["a_exec_2"]) / max(1.0, float(r["total_prbs"]))
        for r in target
    ]
    wall_vals = [float(r["agent_wall_ms"]) for r in target]

    return {
        "n_windows": len(rows_out),
        "n_post_warmup": len(post),
        "sla_viol_all3_pct": 100.0 * (1.0 - float(np.mean([r["sla_ok_all3"] for r in target]))),
        "sla_viol_embb_pct": 100.0 * (1.0 - float(np.mean([r["sla_ok_e"] for r in target]))),
        "sla_viol_urllc_pct": 100.0 * (1.0 - float(np.mean([r["sla_ok_u"] for r in target]))),
        "sla_viol_mmtc_pct": 100.0 * (1.0 - float(np.mean([r["sla_ok_m"] for r in target]))),
        "embb_thr_mean_mbps": float(np.mean(embb_vals)) if embb_vals else None,
        "embb_thr_p5_mbps": float(np.percentile(embb_vals, 5.0)) if embb_vals else None,
        "urllc_p95_ms_mean": float(np.mean(u_vals)) if u_vals else None,
        "urllc_p95_ms_p95": float(np.percentile(u_vals, 95.0)) if u_vals else None,
        "urllc_miss_pct_mean": float(np.mean(u_miss_vals)) if u_miss_vals else None,
        "mmtc_pdr_mean": float(np.mean(m_vals)) if m_vals else None,
        "prb_util_pct": float(np.mean(util_vals)) if util_vals else None,
        "post_warmup_agent_ms_p50": float(np.percentile(wall_vals, 50.0)) if wall_vals else 0.0,
        "post_warmup_agent_ms_p95": float(np.percentile(wall_vals, 95.0)) if wall_vals else 0.0,
        "cer_interventions": int(
            sum(1 for r in target if str(r.get("cer_stage", "")) not in ("", "MANTER_B0"))
        ),
    }


def run_v2_episode(
    env: SqrV2Env,
    controllers: dict[int, Any],
    *,
    gamma: float,
    virtual_alpha: float = DEFAULT_VIRTUAL_ALPHA,
    freeze: bool = False,
    learn_on_timeout_hold: bool = True,
    warmup_steps: int = 0,
    seed: int = 101,
    episode: int = 1,
    step_logger: StepMetricsV2Logger | None = None,
    decision_timer=None,
    checkpoint_path: str | None = None,
    checkpoint_interval: int = 0,
    checkpoint_meta: dict[str, Any] | None = None,
    max_steps: int | None = None,
    step_hook=None,
    reallocator: str = "none",
    cer_mode: str = "full",
    idle_empty_guard_prbs: int = 0,
    priority_weights: tuple[float, float, float] | None = None,
    cer_log_path: str | None = None,
    het: bool = False,
    het_scaler: FeatureScaler | None = None,
    het_scaler_manifest: str | None = None,
    empty_slice_guard: bool = False,
    urllc_delayed_update: bool = False,
    cer_dl_slots: int | None = None,
    cer_config_json: str | None = None,
    het_records_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one causal closed-loop episode over IPC Handshake v2."""
    if reallocator not in ("none", "priority_hoashi", "sqr_cer"):
        raise ValueError(f"Unsupported reallocator: {reallocator!r}")
    if reallocator != "none" and not all(hasattr(c, "score_candidates") for c in controllers.values()):
        raise ValueError(f"--reallocator={reallocator} requires SQR controllers")

    num_slices = len(controllers)
    obs, reset_info = env.reset(seed=seed)

    # BUG-05 fix: synchronize controller capacities BEFORE Step 1 decision
    total_prbs = int(reset_info.get("total_prbs", next(iter(controllers.values())).n_prbs))
    _sync_controllers_capacity(controllers, total_prbs)

    het_runtime: HetRuntime | None = None
    active_scaler: FeatureScaler | None = het_scaler
    het_bootstrap: dict[int, PropDemandController] | None = None
    last_absorbed_ts = -1
    if het:
        if num_slices != 3:
            raise ValueError(f"--het requires 3 slices (eMBB, URLLC, mMTC), got {num_slices}")
        if active_scaler is None:
            manifest_path = het_scaler_manifest or DEFAULT_HET_SCALER_MANIFEST
            active_scaler = FeatureScaler.load_manifest(manifest_path, freeze=True)
        het_runtime = HetRuntime(str(getattr(env, "sim_path", "")))
        init_win = getattr(getattr(env, "builder", None), "last_window", None)
        if init_win is not None:
            het_runtime.absorb(init_win)
            last_absorbed_ts = int(init_win.timestamp)
        is_learning_family = all(hasattr(c, "observe") for c in controllers.values()) and not any(
            getattr(c, "LEARNING", True) is False for c in controllers.values()
        )
        if is_learning_family and warmup_steps > 0:
            builder = getattr(env, "builder", None)
            w_prov = (lambda b: lambda: b.last_window)(builder) if builder is not None else None
            het_bootstrap = {
                sid: PropDemandController(
                    slice_id=sid,
                    n_prbs=total_prbs,
                    num_slices=num_slices,
                    window_provider=w_prov,
                )
                for sid in range(num_slices)
            }

    telemetry = Ns3PrbWindowTelemetryV2(getattr(env, "sim_path", None), num_slices=num_slices)

    hoashi_reallocator = (
        PriorityHoashiReallocator(
            PriorityHoashiConfig(weights=priority_weights)
            if priority_weights is not None
            else PriorityHoashiConfig()
        )
        if reallocator == "priority_hoashi"
        else None
    )
    cer_cfg = (
        _load_cer_config(
            cer_mode=cer_mode,
            idle_empty_guard_prbs=int(idle_empty_guard_prbs),
            cer_dl_slots=cer_dl_slots,
            cer_config_json=cer_config_json,
        )
        if reallocator == "sqr_cer"
        else None
    )
    cer_reallocator = CerReallocator(cer_cfg) if cer_cfg is not None else None
    cer_risk_tracker = (
        PrequentialRiskTracker(
            w_preq=cer_cfg.w_preq,
            beta_v=cer_cfg.beta_v,
            beta_o=cer_cfg.beta_o,
            beta_c=cer_cfg.beta_c,
            l_max=cer_cfg.l_max,
        )
        if cer_cfg is not None
        else None
    )
    cer_log_fh = None
    if cer_log_path and reallocator != "none":
        os.makedirs(os.path.dirname(os.path.abspath(cer_log_path)), exist_ok=True)
        cer_log_fh = open(cer_log_path, "w", encoding="utf-8")

    n_virtual = 0
    n_learned = {sid: 0 for sid in range(num_slices)}
    n_timeout_holds = 0
    n_applied = 0
    n_window_gaps = 0
    n_over_50ms = 0
    wall_samples_ms: list[float] = []
    cer_wall_samples_ms: list[float] = []

    het_prev_action: tuple[int, int, int] = (0, 0, 0)
    het_prev_viol: tuple[int, int, int] = (0, 0, 0)
    urllc_delayed: deque[dict[str, Any]] | None = deque() if (het and urllc_delayed_update) else None
    urllc_updated_wins: set[int] = set()
    local_het_records: list[dict[str, Any]] = het_records_out if het_records_out is not None else []

    step_idx = 0
    try:
        while True:
            current_seq = int(getattr(env, "current_window_seq", step_idx + 1))
            in_warmup = (step_idx + 1) <= warmup_steps

            st: JointState | None = None
            tel_for_step: CerBlocoATelemetry | None = None
            gov_win = current_seq
            if het_runtime is not None and active_scaler is not None:
                cur_win = getattr(getattr(env, "builder", None), "last_window", None)
                if cur_win is None:
                    raise RuntimeError("HetRuntime requires valid initial last_window")
                st = het_runtime.joint_state(cur_win, het_prev_action, het_prev_viol)
                st.extras["urllc_pool_open"] = bool(
                    st.extras.get("urllc_pool_open", False)
                    or (urllc_delayed is not None and len(urllc_delayed) > 0)
                )
                states = _build_het_slice_states(st, active_scaler, total_prbs, num_slices)
                tel_for_step = extract_bloco_a_telemetry(st)
                gov_win = int(st.window_index) + 1
            else:
                states = [np.asarray(obs[i], dtype=np.float64) for i in range(num_slices)]

            cer_dec = None
            hoashi_dec = None
            sc_list = None
            cer_step_ms = 0.0
            t0 = time.perf_counter()

            def _decide_once():
                nonlocal sc_list, cer_dec, hoashi_dec, cer_step_ms
                if het_bootstrap is not None and in_warmup:
                    prop_decs = [het_bootstrap[sid].act(states[sid]) for sid in range(num_slices)]
                    prop_raw = [int(d.action) for d in prop_decs]
                    presult = project_prbs(np.asarray(prop_raw, dtype=np.int64), total_prbs)
                    return [
                        _BootstrapDecisionWrapper(prop_raw[sid], int(presult.allocated[sid]))
                        for sid in range(num_slices)
                    ]

                decs = [controllers[sid].act(states[sid]) for sid in range(num_slices)]
                if het and empty_slice_guard and reallocator == "none":
                    tel_chk = tel_for_step or _extract_online_bloco_a(env, num_slices)
                    act_chk = causal_active_mask(tel_chk)
                    for sid in range(num_slices):
                        if not act_chk[sid] and hasattr(decs[sid], "raw_action"):
                            decs[sid] = replace(
                                decs[sid],
                                action=0,
                                raw_action=0,
                                margin_applied=0,
                            )
                if reallocator == "priority_hoashi" and hoashi_reallocator is not None:
                    tel = tel_for_step or _extract_online_bloco_a(env, num_slices)
                    act_m = causal_active_mask(tel)
                    tc0 = time.perf_counter()
                    hoashi_dec = hoashi_reallocator.allocate(
                        [int(d.action) for d in decs],
                        total_prbs,
                        act_m,
                    )
                    cer_step_ms = (time.perf_counter() - tc0) * 1000.0
                elif reallocator == "sqr_cer" and cer_reallocator is not None and cer_risk_tracker is not None:
                    tel = tel_for_step or _extract_online_bloco_a(env, num_slices)
                    sc_list = [controllers[sid].score_candidates(states[sid]) for sid in range(num_slices)]
                    f_norm_curves = []
                    for sid in range(num_slices):
                        sla = float(getattr(controllers[sid], "sla_threshold", 0.0))
                        f_raw = np.asarray(sc_list[sid]["f"], dtype=np.float64)
                        f_norm_curves.append((f_raw - sla) / sla if sla > 0.0 else f_raw)
                    risk_snap = cer_risk_tracker.snapshot(governed_window=gov_win)
                    tc0 = time.perf_counter()
                    cer_dec = cer_reallocator.reallocate(
                        a_nom=[int(d.action) for d in decs],
                        f_curves=f_norm_curves,
                        sigma_curves=[sc_list[sid]["sigma"] for sid in range(num_slices)],
                        telemetry=tel,
                        risk=risk_snap,
                        capacity_c=total_prbs,
                    )
                    cer_step_ms = (time.perf_counter() - tc0) * 1000.0
                return decs

            if decision_timer is not None:
                with decision_timer.measure():
                    decisions = _decide_once()
            else:
                decisions = _decide_once()
            agent_wall_ms = (time.perf_counter() - t0) * 1000.0
            wall_samples_ms.append(agent_wall_ms)
            if (cer_dec is not None) or (hoashi_dec is not None):
                cer_wall_samples_ms.append(cer_step_ms)
            if agent_wall_ms > 50.0:
                n_over_50ms += 1

            # Ensure pre-exec sc_list exists during warmup if SQR-CER tracks risk prequentially
            if (
                het_bootstrap is not None
                and in_warmup
                and cer_risk_tracker is not None
                and sc_list is None
            ):
                sc_list = [controllers[sid].score_candidates(states[sid]) for sid in range(num_slices)]

            if cer_dec is not None:
                proposals = list(cer_dec.action)
            elif hoashi_dec is not None:
                proposals = list(hoashi_dec.action)
            else:
                proposals = [int(d.action) for d in decisions]

            if cer_log_fh is not None and cer_dec is not None:
                cer_log_fh.write(
                    json.dumps({
                        "window_seq": current_seq,
                        "action": list(cer_dec.action),
                        "a_nom_guarded": list(cer_dec.a_nom_guarded),
                        "a_stage1_idle": list(cer_dec.a_stage1_idle),
                        "cer_stage": cer_dec.cer_stage,
                        "slack_prbs": cer_dec.slack_prbs,
                        "donor_floors": list(cer_dec.donor_floors),
                        "u_preq": [round(float(x), 6) for x in cer_dec.u_preq],
                        "delta_prbs": list(cer_dec.delta_prbs),
                        "abstain_reason": cer_dec.abstain_reason,
                    })
                    + "\n"
                )
            elif cer_log_fh is not None and hoashi_dec is not None:
                cer_log_fh.write(
                    json.dumps({
                        "window_seq": current_seq,
                        "reallocator": "priority_hoashi",
                        "action": list(hoashi_dec.action),
                        "proposals_guarded": list(hoashi_dec.proposals_guarded),
                        "excess_cut": hoashi_dec.excess_cut,
                        "cut_applied": hoashi_dec.cut_applied,
                    })
                    + "\n"
                )

            if step_hook is not None:
                # Test hook for fault injection / deterministic delay scenarios
                step_hook(env, current_seq, proposals)

            env.set_next_decision_wall_ms(agent_wall_ms)
            next_obs, reward, terminated, truncated, info = env.step(
                np.array(proposals, dtype=np.int64)
            )

            new_c = int(info.get("total_prbs", total_prbs))
            if new_c > 0 and new_c != total_prbs:
                total_prbs = new_c
                _sync_controllers_capacity(controllers, total_prbs)
                if het_bootstrap is not None:
                    _sync_controllers_capacity(het_bootstrap, total_prbs)

            if info.get("episode_end_no_next_window"):
                # Final slice of simTime exited cleanly via Simulator::Stop without a next KPM window
                break

            receipt = info.get("ipc_receipt") or {}
            ipc_status = str(receipt.get("ipc_status", "APPLIED"))
            applied_seq = int(receipt.get("applied_action_seq", current_seq))
            window_gap = int(receipt.get("window_gap", 0))
            ns3_wait_ms = float(receipt.get("ns3_wait_wall_ms", 0.0))

            proj = info.get("projection", {})
            prb_requested_ipc = [int(v) for v in proj.get("prb_requested", proposals)]
            prb_executed_proj = [int(v) for v in proj.get("prb_executed", proposals)]
            prb_executed_mac = [int(v) for v in receipt.get("prb_executed_mac", prb_executed_proj)]

            if ipc_status == "TIMEOUT_HOLD":
                n_timeout_holds += 1
            elif ipc_status == "APPLIED":
                n_applied += 1
            if window_gap != 0:
                n_window_gaps += 1

            # Absorb newly closed window into HetRuntime when enabled
            het_outcome: SliceOutcome | None = None
            urllc_tx_gov = 0
            urllc_blackout_online = False
            if het_runtime is not None:
                new_win = getattr(getattr(env, "builder", None), "last_window", None)
                if new_win is not None:
                    new_ts = int(new_win.timestamp)
                    if new_ts != last_absorbed_ts:
                        het_runtime.absorb(new_win)
                        last_absorbed_ts = new_ts
                het_outcome = het_runtime.outcome_for_window(gov_win)
                urllc_tx_gov = int((het_runtime._win_cache.get(gov_win) or {}).get("urllc_tx") or 0)
                urllc_blackout_online = (
                    het_outcome.margins[1] is None
                    and urllc_tx_gov > 0
                    and int(prb_executed_mac[1]) == 0
                )

            # Strictly learn ONLY from prb_executed_mac confirmed by ns-3 MAC for THIS state's window
            window_learned: list[int] = []
            v_updates_applied = 0
            can_learn = (not freeze) and (window_gap == 0) and (
                ipc_status == "APPLIED"
                or (ipc_status == "TIMEOUT_HOLD" and learn_on_timeout_hold)
            )
            if can_learn:
                for sid in range(num_slices):
                    if het_outcome is not None:
                        y_t = -1.0 if (sid == 1 and urllc_blackout_online) else het_outcome.margins[sid]
                    else:
                        y_t = info.get(f"slice_{sid}", {}).get("p5_thr_mbps")
                    if y_t is None:
                        continue
                    f_raw_exec = None
                    if sc_list is not None:
                        a_exec_i = min(int(prb_executed_mac[sid]), len(sc_list[sid]["f"]) - 1)
                        f_raw_exec = float(sc_list[sid]["f"][a_exec_i])
                    if cer_risk_tracker is not None and f_raw_exec is not None:
                        sla = float(getattr(controllers[sid], "sla_threshold", 0.0))
                        y_norm = (float(y_t) - sla) / sla if sla > 0.0 else float(y_t)
                        f_norm = (f_raw_exec - sla) / sla if sla > 0.0 else f_raw_exec
                        cer_risk_tracker.record_closed_outcome(sid, f_norm, y_norm, outcome_win=gov_win)
                    if not getattr(controllers[sid], "LEARNING", True):
                        continue
                    if hasattr(controllers[sid], "notify_projection"):
                        controllers[sid].notify_projection(bool(proj.get("over_capacity", False)))
                    controllers[sid].observe(states[sid], int(prb_executed_mac[sid]), float(y_t))
                    window_learned.append(sid)
                    n_learned[sid] += 1

                if (
                    het_runtime is not None
                    and het_outcome is not None
                    and urllc_delayed is not None
                    and (not urllc_blackout_online)
                    and het_outcome.margins[1] is None
                    and gov_win not in urllc_updated_wins
                ):
                    tag0, _ = het_runtime.urllc_pool_status(gov_win)
                    if tag0 == "OPEN":
                        if cer_risk_tracker is not None:
                            cer_risk_tracker.mark_urllc_pending(gov_win)
                        f_1_exec = None
                        if sc_list is not None:
                            a1 = min(int(prb_executed_mac[1]), len(sc_list[1]["f"]) - 1)
                            f_1_exec = float(sc_list[1]["f"][a1])
                        urllc_delayed.append({
                            "w": int(gov_win),
                            "age": 0,
                            "state_1": states[1].copy(),
                            "action_1": int(prb_executed_mac[1]),
                            "f_1_exec": f_1_exec,
                        })

                # Virtual updates only apply if THIS proposal actually entered the MAC (APPLIED)
                if (
                    ipc_status == "APPLIED"
                    and virtual_alpha > 0.0
                    and all(hasattr(c, "insert_virtual_support") for c in controllers.values())
                ):
                    proj_obj = _projection_for_virtual(proj, total_prbs)
                    if len(proj_obj.virtual_updates) > 0:
                        apply_virtual_updates(proj_obj, states, controllers, alpha=virtual_alpha)
                        v_updates_applied = len(proj_obj.virtual_updates)
                        n_virtual += v_updates_applied

            # Bounded causal drain of delayed URLLC outcomes (even on hold windows)
            if het_runtime is not None and urllc_delayed is not None:
                for _ in range(len(urllc_delayed)):
                    item = urllc_delayed.popleft()
                    w_del = int(item["w"])
                    if w_del in urllc_updated_wins:
                        continue
                    tag, _ = het_runtime.urllc_pool_status(w_del)
                    if tag == "CLOSED":
                        oc_u = het_runtime.urllc_only_outcome(w_del)
                        m_u = oc_u.margins[1] if oc_u is not None else None
                        if m_u is not None:
                            if cer_risk_tracker is not None and item["f_1_exec"] is not None:
                                cer_risk_tracker.record_closed_outcome(
                                    1, float(item["f_1_exec"]), float(m_u), outcome_win=w_del
                                )
                            if (not freeze) and getattr(controllers[1], "LEARNING", True):
                                controllers[1].observe(item["state_1"], int(item["action_1"]), float(m_u))
                                n_learned[1] += 1
                            urllc_updated_wins.add(w_del)
                    elif tag == "OPEN":
                        item["age"] += 1
                        if item["age"] >= _URLLC_DELAYED_MAX_AGE:
                            if cer_risk_tracker is not None:
                                cer_risk_tracker.mark_urllc_expired(w_del, observed_at_win=gov_win)
                        else:
                            urllc_delayed.append(item)

            if het_outcome is not None:
                m_eff = (
                    het_outcome.margins[0],
                    -1.0 if urllc_blackout_online else het_outcome.margins[1],
                    het_outcome.margins[2],
                )
                het_prev_viol = tuple(0 if (m is None or m >= 0.0) else 1 for m in m_eff)
                het_prev_action = tuple(int(v) for v in prb_executed_mac)

            # Telemetry keyed strictly to governed window sequence when available
            win_row = telemetry.for_window_seq(current_seq) or telemetry.latest() or {}
            info_sched = win_row.get("prb_scheduled", {})
            now_ms = info.get("kpm_timestamp", info.get("sim_time_ms", ""))
            unalloc_quota = int(total_prbs) - int(sum(prb_executed_mac))

            if het_outcome is not None:
                local_het_records.append({
                    "window_index": int(gov_win),
                    "window_seq": int(current_seq),
                    "step": int(step_idx + 1),
                    "warmup": int(in_warmup),
                    "ipc_status": ipc_status,
                    "total_prbs": int(total_prbs),
                    "cer_stage": cer_dec.cer_stage if cer_dec is not None else ("HOASHI" if hoashi_dec is not None else ""),
                    "abstain_reason": cer_dec.abstain_reason if cer_dec is not None else "",
                    "a_nom_0": int(decisions[0].action),
                    "a_nom_1": int(decisions[1].action),
                    "a_nom_2": int(decisions[2].action),
                    "a_exec_0": int(prb_executed_mac[0]),
                    "a_exec_1": int(prb_executed_mac[1]),
                    "a_exec_2": int(prb_executed_mac[2]),
                    "prb_sched_0": info_sched.get("0", ""),
                    "prb_sched_1": info_sched.get("1", ""),
                    "prb_sched_2": info_sched.get("2", ""),
                    "u_preq_0": round(float(cer_dec.u_preq[0]), 6) if cer_dec is not None else "",
                    "u_preq_1": round(float(cer_dec.u_preq[1]), 6) if cer_dec is not None else "",
                    "u_preq_2": round(float(cer_dec.u_preq[2]), 6) if cer_dec is not None else "",
                    "embb_thr_mbps": het_outcome.embb_throughput_mbps,
                    "urllc_tx_pkts": urllc_tx_gov,
                    "urllc_p95_ms_online": het_outcome.urllc_p95_ms,
                    "urllc_miss_pct_online": het_outcome.urllc_miss_pct,
                    "mmtc_pdr_rolling": het_outcome.mmtc_pdr_rolling,
                    "agent_wall_ms": round(float(agent_wall_ms), 4),
                })

            if step_logger is not None:
                ep_rows = []
                for sid in range(num_slices):
                    d = decisions[sid]
                    k = info.get(f"slice_{sid}", {})
                    ep_rows.append([
                        episode,
                        step_idx + 1,
                        now_ms,
                        seed,
                        gamma,
                        sid,
                        d.mode,
                        d.raw_action,
                        d.action,
                        prb_requested_ipc[sid],
                        prb_executed_mac[sid],
                        info_sched.get(str(sid), ""),
                        unalloc_quota,
                        f"{d.sigma:.6f}",
                        f"{d.margin_m:.3f}",
                        d.margin_applied,
                        f"{d.tau:.5f}",
                        "" if k.get("p5_thr_mbps") is None else f"{k['p5_thr_mbps']:.6f}",
                        "" if k.get("lowest_thr_mbps") is None else f"{k['lowest_thr_mbps']:.6f}",
                        k.get("n_ues", 0),
                        "" if k.get("violation_rate") is None else f"{k['violation_rate']:.6f}",
                        k.get("n_snr_imputed", 0),
                        int(info.get("kpm_missing_steps", 0)),
                        int(sid in window_learned),
                        v_updates_applied,
                        f"{agent_wall_ms:.4f}",
                        f"{reward:.6f}",
                        int(in_warmup),
                        bool(terminated),
                        bool(truncated),
                        str(info.get("run_id", "")),
                        int(current_seq),
                        int(applied_seq),
                        ipc_status,
                        int(prb_requested_ipc[sid]),
                        int(prb_executed_mac[sid]),
                        f"{agent_wall_ms:.4f}",
                        f"{ns3_wait_ms:.4f}",
                        int(agent_wall_ms > 50.0 or ns3_wait_ms > 50.0),
                    ])
                step_logger.write(ep_rows)

            obs = next_obs
            step_idx += 1
            if checkpoint_path and checkpoint_interval and step_idx % checkpoint_interval == 0:
                payload = build_checkpoint(controllers, meta=checkpoint_meta)
                if payload.get("meta") is not None:
                    payload["meta"]["payload_sha256"] = checkpoint_payload_sha256(payload)
                save_checkpoint(checkpoint_path, payload)

            if terminated or truncated or (max_steps is not None and step_idx >= max_steps):
                break
    finally:
        if cer_log_fh is not None:
            cer_log_fh.close()

    if checkpoint_path:
        payload = build_checkpoint(controllers, meta=checkpoint_meta)
        if payload.get("meta") is not None:
            payload["meta"]["payload_sha256"] = checkpoint_payload_sha256(payload)
        save_checkpoint(checkpoint_path, payload)

    arr = np.asarray(wall_samples_ms, dtype=np.float64)
    cer_arr = np.asarray(cer_wall_samples_ms, dtype=np.float64)
    return {
        "steps": step_idx,
        "run_id": getattr(env, "run_id", ""),
        "total_prbs_final": total_prbs,
        "applied_steps": n_applied,
        "timeout_hold_steps": n_timeout_holds,
        "window_gap_steps": n_window_gaps,
        "virtual_updates": n_virtual,
        "learned_windows": dict(n_learned),
        "kpm_missing_steps": int(getattr(env, "kpm_missing_steps", 0)),
        "over_50ms_steps": int(n_over_50ms),
        "agent_decision_ms_p50": float(np.percentile(arr, 50)) if arr.size else 0.0,
        "agent_decision_ms_p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "agent_decision_ms_max": float(np.max(arr)) if arr.size else 0.0,
        "cer_reallocate_ms_p50": float(np.percentile(cer_arr, 50)) if cer_arr.size else 0.0,
        "cer_reallocate_ms_p95": float(np.percentile(cer_arr, 95)) if cer_arr.size else 0.0,
        "cer_reallocate_ms_max": float(np.max(cer_arr)) if cer_arr.size else 0.0,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SQR v2 closed-loop driver over IPC Handshake v2")
    p.add_argument("--controller", type=str, default="sqr", choices=("sqr", "kbrl", "equal", "prop"))
    p.add_argument("--gamma", type=int, default=1, choices=(0, 1))
    p.add_argument("--ns3_path", type=str, default=DEFAULT_NS3_PATH)
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--simTime", type=float, default=15.0)
    p.add_argument("--periodMs", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--c_prbs", type=int, default=52)
    p.add_argument("--num_slices", type=int, default=3)
    p.add_argument("--slas", type=str, default="0.5,1.0,1.5")
    p.add_argument("--violation_window", type=int, default=VIOLATION_WINDOW)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--knn_k", type=int, default=10)
    p.add_argument("--virtual_alpha", type=float, default=DEFAULT_VIRTUAL_ALPHA)
    p.add_argument("--margin_init", type=float, default=MARGIN_INIT)
    p.add_argument("--margin_bounds", type=str, default=f"{MARGIN_BOUNDS[0]},{MARGIN_BOUNDS[1]}")
    p.add_argument("--warmup_s", type=float, default=10.0)
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--checkpoint_interval", type=int, default=500)
    p.add_argument("--sliceLambdas", type=str, default=None, dest="slice_lambdas")
    p.add_argument("--sliceApps", type=str, default=None, dest="slice_apps")
    p.add_argument("--cellRadius", type=float, default=None, dest="cell_radius")
    p.add_argument("--ipc-mode", type=str, default="strict_rt", choices=("strict_rt", "causal_sim"))
    p.add_argument("--ipc-timeout-ms", type=int, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--learn-on-timeout-hold", type=str, default="true", choices=("true", "false"))
    p.add_argument("--infra-rev", type=str, default=DEFAULT_INFRA_REV)
    p.add_argument("--phy-qos-profile", type=str, default=DEFAULT_PHY_QOS_PROFILE)
    p.add_argument("--learning-ablation", type=str, default=DEFAULT_LEARNING_ABLATION)
    p.add_argument(
        "--reallocator",
        type=str,
        default="none",
        choices=("none", "priority_hoashi", "sqr_cer"),
    )
    p.add_argument(
        "--cer-mode",
        type=str,
        default="full",
        choices=("full", "regime_a_only"),
    )
    p.add_argument(
        "--idleEmptyGuardPrbs",
        type=int,
        default=0,
        choices=(0, 2),
        dest="idle_empty_guard_prbs",
    )
    p.add_argument("--priority-weights", type=str, default="1,1000000,1")
    p.add_argument("--het", action="store_true", help="Use heterogeneous eMBB/URLLC/mMTC normalized margins & HetRuntime")
    p.add_argument("--het-scaler-manifest", type=str, default=None)
    p.add_argument("--empty-slice-guard", action="store_true")
    p.add_argument("--urllc-delayed-update", action="store_true")
    p.add_argument("--action-length-scales", type=str, default=None)
    p.add_argument("--allow-reserved-seed", action="store_true")
    p.add_argument("--cer-dl-slots", type=int, default=None)
    p.add_argument("--cer-config-json", type=str, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    allow_reserved = bool(getattr(args, "allow_reserved_seed", False))
    validate_dev_seed(args.seed, allow_reserved=allow_reserved)
    if args.reallocator != "none" and args.controller != "sqr":
        raise ValueError(f"--reallocator={args.reallocator} requires --controller=sqr")
    priority_weights = parse_priority_weights(args.priority_weights)
    action_scales = parse_action_length_scales_v2(args.action_length_scales, args.num_slices)
    os.makedirs(args.output, exist_ok=True)

    slas = parse_slas_v2(args.slas, args.num_slices)
    parse_margin_bounds_v2(args.margin_bounds)

    config: dict[str, Any] = {
        "simTime": [args.simTime],
        "seed": [args.seed],
        "periodMs": [args.periodMs],
    }
    if args.slice_lambdas:
        config["sliceLambdas"] = [args.slice_lambdas]
    if args.slice_apps:
        config["sliceApps"] = [args.slice_apps]
    if args.cell_radius is not None:
        config["cellRadius"] = [float(args.cell_radius)]

    max_steps = args.max_steps if args.max_steps > 0 else None
    env = SqrV2Env(
        ns3_path=os.path.abspath(args.ns3_path),
        scenario_configuration=config,
        output_folder=args.output,
        total_prbs=args.c_prbs,
        num_slices=args.num_slices,
        sla_thresholds=slas,
        max_steps=max_steps,
        allow_reserved_seed=allow_reserved,
        ipc_mode=args.ipc_mode,
        ipc_timeout_ms=args.ipc_timeout_ms,
        run_id=args.run_id,
    )
    controllers = build_controllers_v2(args, env)
    warmup_steps = int(round(args.warmup_s * 1000.0 / args.periodMs))
    ckpt_path = os.path.join(args.output, "sqr_checkpoint.pkl")
    step_logger = StepMetricsV2Logger(os.path.join(args.output, "step_metrics.csv"))
    accounting = ComputeAccounting(interaction_budget=env.max_steps, period_ms=args.periodMs)

    ckpt_meta = {
        "schema": "sqr_v2_checkpoint",
        "controller": args.controller,
        "reallocator": args.reallocator,
        "cer_mode": args.cer_mode,
        "idle_empty_guard_prbs": int(args.idle_empty_guard_prbs),
        "priority_weights": list(priority_weights),
        "het": bool(args.het),
        "empty_slice_guard": bool(args.empty_slice_guard),
        "urllc_delayed_update": bool(args.urllc_delayed_update),
        "action_length_scales": [action_scales[sid] for sid in range(args.num_slices)],
        "cer_dl_slots": args.cer_dl_slots,
        "gamma": args.gamma,
        "seed": args.seed,
        "c_prbs": args.c_prbs,
        "ipc_mode": env.ipc_mode,
        "ipc_timeout_ms": env.ipc_timeout_ms,
        "rt_50ms_deadline_enforced": env.rt_50ms_deadline_enforced,
        "infra_rev": args.infra_rev,
        "phy_qos_profile": args.phy_qos_profile,
        "learning_ablation": args.learning_ablation,
    }

    cer_log_path = (
        os.path.join(args.output, "cer_decisions.jsonl")
        if args.reallocator != "none"
        else None
    )
    het_records: list[dict[str, Any]] = []
    het_summary: dict[str, Any] | None = None

    try:
        summary = run_v2_episode(
            env,
            controllers,
            gamma=float(args.gamma),
            virtual_alpha=float(args.virtual_alpha),
            freeze=bool(args.freeze),
            learn_on_timeout_hold=(args.learn_on_timeout_hold == "true"),
            warmup_steps=warmup_steps,
            seed=int(args.seed),
            episode=1,
            step_logger=step_logger,
            decision_timer=accounting.decision_timer,
            checkpoint_path=ckpt_path,
            checkpoint_interval=int(args.checkpoint_interval),
            checkpoint_meta=ckpt_meta,
            max_steps=max_steps,
            reallocator=args.reallocator,
            cer_mode=args.cer_mode,
            idle_empty_guard_prbs=int(args.idle_empty_guard_prbs),
            priority_weights=priority_weights,
            cer_log_path=cer_log_path,
            het=bool(args.het),
            het_scaler_manifest=args.het_scaler_manifest,
            empty_slice_guard=bool(args.empty_slice_guard),
            urllc_delayed_update=bool(args.urllc_delayed_update),
            cer_dl_slots=args.cer_dl_slots,
            cer_config_json=args.cer_config_json,
            het_records_out=het_records if args.het else None,
        )
        ns3_exit = env.wait_sim_exit(timeout_s=120.0)
        cpp_summary = env.read_cpp_ipc_summary()
        if args.het:
            het_summary = _summarize_and_write_het_windows(
                getattr(env, "sim_path", None),
                het_records,
                output_csv_path=os.path.join(args.output, "het_window_metrics.csv"),
            )
    finally:
        env.close()

    finish = accounting.finish()
    manifest = build_v2_manifest(
        run_id=summary["run_id"],
        controller=args.controller,
        seed=int(args.seed),
        ipc_mode=env.ipc_mode,
        ipc_timeout_ms=env.ipc_timeout_ms,
        infra_rev=args.infra_rev,
        phy_qos_profile=args.phy_qos_profile,
        learning_ablation=args.learning_ablation,
        sim_binary_path=env.script_executable,
        extra={
            "reallocator": args.reallocator,
            "cer_mode": args.cer_mode,
            "idle_empty_guard_prbs": int(args.idle_empty_guard_prbs),
            "priority_weights": list(priority_weights),
            "het": bool(args.het),
            "action_length_scales": [action_scales[sid] for sid in range(args.num_slices)],
            "python_episode_summary": summary,
            "het_summary": het_summary,
            "cpp_ipc_v2_summary": cpp_summary,
            "ns3_exitcode": ns3_exit,
        },
    )
    write_v2_manifest(os.path.join(args.output, "SQR_V2_MANIFEST.json"), manifest)

    done_payload = {
        "schema_version": "sqr_v2_run_done_1.0",
        "run_id": summary["run_id"],
        "controller": args.controller,
        "seed": int(args.seed),
        "gamma": int(args.gamma),
        "het": bool(args.het),
        "ipc_mode": env.ipc_mode,
        "ipc_timeout_ms": env.ipc_timeout_ms,
        "rt_50ms_deadline_enforced": env.rt_50ms_deadline_enforced,
        "ns3_exitcode": ns3_exit,
        "episode_summary": summary,
        "het_summary": het_summary,
        "cpp_ipc_v2_summary": cpp_summary,
        "xapp_runtime": finish.get("xapp_runtime"),
    }
    with open(os.path.join(args.output, "run_done.json"), "w", encoding="utf-8") as df:
        json.dump(done_payload, df, indent=2)
        df.write("\n")

    print(
        f"[sqr_v2_run] run_id={summary['run_id']} ipc_mode={env.ipc_mode} "
        f"(timeout={env.ipc_timeout_ms}ms, rt_50ms_enforced={env.rt_50ms_deadline_enforced}) "
        f"steps={summary['steps']} applied={summary['applied_steps']} "
        f"holds={summary['timeout_hold_steps']} "
        f"agent_ms_p50={summary['agent_decision_ms_p50']:.3f} "
        f"p95={summary['agent_decision_ms_p95']:.3f} max={summary['agent_decision_ms_max']:.3f} "
        f"ns3_exit={ns3_exit}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
