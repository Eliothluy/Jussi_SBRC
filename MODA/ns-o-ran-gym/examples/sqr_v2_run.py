"""SQR v2 closed-loop runner over IPC Handshake v2 (Etapa 2: BUG-03 + BUG-05)."""

from __future__ import annotations

import argparse
import csv
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
from nsoran.sqr.coordinator import DEFAULT_VIRTUAL_ALPHA, apply_virtual_updates
from nsoran.sqr.sqr_controller import (
    MARGIN_BOUNDS,
    MARGIN_INIT,
    TAU_INIT,
    VIOLATION_WINDOW,
)
from nsoran.sqr_jc.freeze_contract import checkpoint_payload_sha256
from nsoran.sqr_v2.controller_v2 import (
    build_controllers_v2,
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

DEFAULT_NS3_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ns-3-dev")
)
DEFAULT_OUTPUT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results", "sqr_v2", "run")
)

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
) -> dict[str, Any]:
    """Run one causal closed-loop episode over IPC Handshake v2."""
    num_slices = len(controllers)
    obs, reset_info = env.reset(seed=seed)

    # BUG-05 fix: synchronize controller capacities BEFORE Step 1 decision
    total_prbs = int(reset_info.get("total_prbs", next(iter(controllers.values())).n_prbs))
    _sync_controllers_capacity(controllers, total_prbs)

    telemetry = Ns3PrbWindowTelemetryV2(getattr(env, "sim_path", None), num_slices=num_slices)

    n_virtual = 0
    n_learned = {sid: 0 for sid in range(num_slices)}
    n_timeout_holds = 0
    n_applied = 0
    n_window_gaps = 0
    n_over_50ms = 0
    wall_samples_ms: list[float] = []

    step_idx = 0
    while True:
        current_seq = int(getattr(env, "current_window_seq", step_idx + 1))
        states = [np.asarray(obs[i], dtype=np.float64) for i in range(num_slices)]

        t0 = time.perf_counter()
        if decision_timer is not None:
            with decision_timer.measure():
                decisions = [controllers[sid].act(states[sid]) for sid in range(num_slices)]
        else:
            decisions = [controllers[sid].act(states[sid]) for sid in range(num_slices)]
        agent_wall_ms = (time.perf_counter() - t0) * 1000.0
        wall_samples_ms.append(agent_wall_ms)
        if agent_wall_ms > 50.0:
            n_over_50ms += 1

        proposals = [int(d.action) for d in decisions]

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

        # Strictly learn ONLY from prb_executed_mac confirmed by ns-3 MAC for THIS state's window
        window_learned: list[int] = []
        v_updates_applied = 0
        can_learn = (not freeze) and (window_gap == 0) and (
            ipc_status == "APPLIED"
            or (ipc_status == "TIMEOUT_HOLD" and learn_on_timeout_hold)
        )
        if can_learn:
            for sid in range(num_slices):
                y_t = info.get(f"slice_{sid}", {}).get("p5_thr_mbps")
                if y_t is None:
                    continue
                if not getattr(controllers[sid], "LEARNING", True):
                    continue
                controllers[sid].observe(states[sid], int(prb_executed_mac[sid]), float(y_t))
                window_learned.append(sid)
                n_learned[sid] += 1

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

        # Telemetry keyed strictly to governed window sequence when available
        win_row = telemetry.for_window_seq(current_seq) or telemetry.latest() or {}
        info_sched = win_row.get("prb_scheduled", {})
        now_ms = info.get("kpm_timestamp", info.get("sim_time_ms", ""))
        unalloc_quota = int(total_prbs) - int(sum(prb_executed_mac))

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
                    int((step_idx + 1) <= warmup_steps),
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

    if checkpoint_path:
        payload = build_checkpoint(controllers, meta=checkpoint_meta)
        if payload.get("meta") is not None:
            payload["meta"]["payload_sha256"] = checkpoint_payload_sha256(payload)
        save_checkpoint(checkpoint_path, payload)

    arr = np.asarray(wall_samples_ms, dtype=np.float64)
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
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    validate_dev_seed(args.seed, allow_reserved=False)
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
        )
        ns3_exit = env.wait_sim_exit(timeout_s=120.0)
        cpp_summary = env.read_cpp_ipc_summary()
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
            "python_episode_summary": summary,
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
        "ipc_mode": env.ipc_mode,
        "ipc_timeout_ms": env.ipc_timeout_ms,
        "rt_50ms_deadline_enforced": env.rt_50ms_deadline_enforced,
        "ns3_exitcode": ns3_exit,
        "episode_summary": summary,
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
