"""SQR closed-loop run over the IPC contract (Fase 2/4 driver script).

Structure clones ``examples/rslaq_train_ddqn.py`` (same path bootstrap,
ComputeAccounting for the decision wall-time, per-step CSV, checkpoints)
but the agent is the model-based SQR stack — pure NumPy, no torch:

    obs (KPM window) -> per-slice SqrSliceController.act(z_i)
        -> integer proposals -> deterministic projection Pi (sum <= C)
        -> SqrEnv.step (writes rslaq_actions_for_ns3.csv, contract §3)
        -> y_t = p5 per slice (contract §4, register R18)
        -> REAL regressor update with the EXECUTED action (register R22)
        -> virtual negative update at the PRE-projection action for cut
           slices (register R12, weight --virtual-alpha, default eta/2)

gamma alternates ONLY the action-selection cost term (AGENTS.md §5 rule 7):
``--gamma 0`` maximizes the quantile alone, ``--gamma 1`` trades quantile
for PRB economy (Figs. 2-4 comparison).  P_penalty stays active in both.

Online learning happens DURING the episode (paper's Algorithm 1 cadence);
``--freeze`` disables all updates (evaluation mode).  A controller
checkpoint (support matrix, alphas, hyperparameters, tau, M, violation
window, experience memory, RNG state) is written every
``--checkpoint_interval`` steps and at the end of each episode.

Fase 3: ``--controller kbrl`` swaps the per-slice controllers for the KBRL
baseline port (Alcaraz et al., IEEE TWC 2022; ``nsoran.sqr.kbrl``) on the
SAME obs/action contract — same z, same integer proposals, same env
projection Pi, same y_t/Lambda labels (slice-time level) with the UE-time
violations still logged (register R19).  ``--gamma`` and the R12 virtual
updates are SQR-only and are ignored for KBRL.

NOTE (Fase 2 scope): this script is exercised in CI only against a MOCK of
the contract (tests/test_sqr_run.py, no ns-3).  The integration with the
real ``sqr-sim`` is validated in Fase 1/4 by the ns-3 agent.

Usage (requires the sqr-sim scenario built in MODA/ns-3-dev):
    python examples/sqr_run.py --gamma 1 --simTime 600 --seed 1 \
        --output results/sqr_run
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import csv
import glob
import time

import numpy as np

from nsoran.compute_accounting import ComputeAccounting
from nsoran.sqr.checkpoint import (
    build_checkpoint,
    load_checkpoint,
    restore_controllers,
    save_checkpoint,
)
from nsoran.sqr.coordinator import DEFAULT_VIRTUAL_ALPHA, apply_virtual_updates
from nsoran.sqr.sqr_controller import (
    MARGIN_BOUNDS,
    MARGIN_INIT,
    TAU_INIT,
    TAU_STEP,
    VIOLATION_WINDOW,
    SqrSliceController,
)
from nsoran.sqr.sqr_env import SqrEnv

#: Controller choices of --controller (Fase 3): the model-based SQR agent
#: or the KBRL baseline (Alcaraz et al., IEEE TWC 2022) ported as a
#: consumer of the SAME obs/action contract.  ``gamma`` and the R12 virtual
#: updates apply to SQR only (KBRL is resource-economical structurally and
#: has no virtual-update mechanism).
CONTROLLER_CHOICES = ("sqr", "kbrl")

DEFAULT_NS3_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ns-3-dev")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "results", "sqr_run")

STEP_METRICS_COLUMNS = [
    "episode",
    "step",
    "timestamp_ms",
    "seed",
    "gamma",
    "slice_id",
    "mode",                 # confident | softmax | knn | knn_nearest | fallback
    "raw_action",
    "action",               # final action (after the round(sigma*M) margin)
    "prb_requested",        # proposal, pre-projection (contract §3.5)
    "prb_executed",         # post-projection allocation (contract §3.5)
    "prb_scheduled",        # MAC RBGs actually served — ns-3 side (Fase 1)
    "prb_idle_est",         # C - sum(executed): RBGs left idle (R27 estimate)
    "sigma",
    "margin_m",             # M of this decision
    "margin_applied",
    "tau",
    "y_t_p5_mbps",          # p5 of per-UE window throughput (register R18)
    "lowest_thr_mbps",
    "n_ues",
    "violation_rate",       # per-UE rate of the window (register R19)
    "n_snr_imputed",
    "kpm_missing",          # cumulative count of windows that arrived
                            # WITHOUT a KPM at this step (contract §5;
                            # the per-window flag is the delta to the
                            # previous step)
    "learned",              # 1 if the real update ran (y_t defined, not frozen)
    "virtual_updates",      # number of cut slices penalized this window
    "decision_time_ms",
    "reward",
    "warmup",               # inside the 100 s warm-up (register R14)
    "terminated",
    "truncated",
]


def build_controllers(args) -> dict[int, SqrSliceController]:
    """One controller per slice with the frozen R11/R21 defaults.

    ``args.controller`` selects the method (default "sqr" for backward
    compatibility with older invocations/tests): "sqr" builds the
    hierarchical SQR stack; "kbrl" builds the KBRL baseline port (same
    obs/action contract; gamma and the SQR-only knobs are ignored).
    """
    slas = parse_slas(args.slas, args.num_slices)
    controller = getattr(args, "controller", "sqr")
    if controller == "kbrl":
        from nsoran.sqr.kbrl import KbrlSliceController

        return {
            sid: KbrlSliceController(
                state_dim=6,                       # R20: 4 local + 2 global
                n_prbs=args.c_prbs,
                sla_threshold=slas[sid],           # same Lambda as SQR
                seed=None if args.seed is None else int(args.seed) + 1000 * sid,
            )
            for sid in range(args.num_slices)
        }
    if controller != "sqr":
        raise ValueError(
            f"unknown controller {controller!r} (choices: {CONTROLLER_CHOICES})"
        )
    return {
        sid: SqrSliceController(
            state_dim=6,                       # R20: 4 local + 2 global
            n_prbs=args.c_prbs,
            gamma=float(args.gamma),
            sla_threshold=slas[sid],
            tau=TAU_INIT,                      # tau0 = 0.05 (R11)
            tau_step=TAU_STEP,                 # delta_tau = 0.005 (R11)
            margin=MARGIN_INIT,                # M0 = 10 PRBs (R11)
            margin_bounds=MARGIN_BOUNDS,       # [2, 50] (R11)
            violation_window=args.violation_window,   # W = 100 (R11)
            epsilon=args.epsilon,              # 0.1 [ARTIGO]
            temperature=args.temperature,      # T_softmax [ASSUMIDO]
            knn_k=args.knn_k,                  # K = 10 [ARTIGO]
            seed=None if args.seed is None else int(args.seed) + 1000 * sid,
        )
        for sid in range(args.num_slices)
    }


def parse_slas(slas_arg: str, num_slices: int) -> dict[int, float]:
    values = [float(v) for v in slas_arg.split(",")]
    if len(values) != num_slices:
        raise ValueError(
            f"--slas must have exactly {num_slices} comma-separated values"
        )
    if any(v <= 0 for v in values):
        raise ValueError("--slas values must be positive (Mbps per UE)")
    return dict(enumerate(values))


class StepMetricsLogger:
    """One row per slice per decision window (step_metrics.csv)."""

    def __init__(self, path: str):
        self.path = path
        self._initialized = False

    def write(self, rows: list[list]) -> None:
        if not self._initialized:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(STEP_METRICS_COLUMNS)
            self._initialized = True
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerows(rows)


class Ns3PrbWindowTelemetry:
    """Incremental reader of the ns-3 PRB telemetry (contract §3.5).

    The ``sqr-sim`` C++ scenario drains, at every window boundary, the MAC
    counters into ``<sim_path>/results_sqr_network_only/seed=<seed>_run=<run>/
    sqr_prb_windows.csv`` with the columns ``timestamp_ms, total_rbg,
    total_prb, dl_slots, prb_requested_pct_{0..2}, prb_executed_rbg_{0..2},
    prb_scheduled_rbg_{0..2}, prb_budget_rbg_{0..2}, prb_idle_rbg``.

    ``latest()`` returns the newest COMPLETE row so the step log can carry
    the MAC-side ``prb_scheduled`` (the Gym ``info`` does not transport
    it) plus ``total_prb`` — the effective cell capacity C (register R01
    as amended: 53, not the round 3GPP 52).

    The reader keeps a byte offset into the file and consumes only the
    appended portion on each call (O(new bytes), not O(file) per step —
    the file grows one row per 100 ms window for the whole episode).  A
    trailing line without ``\\n`` (torn write) stays unconsumed until the
    C++ finishes the line; a complete row whose column count mismatches
    the header is skipped.  Rows already consumed are cached, so repeated
    calls without new windows return the same row.  Inert (always
    ``None``) when there is no real sim folder — contract mocks (Fase 2
    tests) keep the column empty.
    """

    def __init__(self, sim_path: str | None, num_slices: int = 3):
        self._sim_path = sim_path
        self._num_slices = int(num_slices)
        self._path: str | None = None   # resolved telemetry CSV
        self._offset: int = 0           # bytes already consumed
        self._header: list[str] | None = None
        self._last: dict | None = None  # newest complete parsed row

    def _resolve(self) -> str | None:
        """Locate the telemetry CSV once; re-glob only if it disappears."""
        if self._path and os.path.isfile(self._path):
            return self._path
        matches = glob.glob(
            os.path.join(
                self._sim_path, "results_sqr_network_only", "seed=*",
                "sqr_prb_windows.csv",
            )
        )
        new_path = matches[0] if matches else None
        if new_path != self._path:  # first resolution OR the file moved
            self._path = new_path
            self._offset = 0
            self._header = None
        return self._path

    def _parse_row(self, line: str) -> dict | None:
        """One complete CSV line -> row dict (None: malformed, skip it)."""
        values = line.strip().split(",")
        if self._header is None or len(values) != len(self._header):
            return None  # torn/broken write of the C++; skip this sample
        row = dict(zip(self._header, values))
        try:
            out = {
                "timestamp_ms": int(float(row["timestamp_ms"])),
                "total_rbg": int(float(row["total_rbg"])),
                "total_prb": int(float(row["total_prb"])),
                "dl_slots": int(float(row["dl_slots"])),
                "prb_idle_rbg": int(float(row["prb_idle_rbg"])),
                "prb_scheduled": {},
                "prb_executed_rbg": {},
            }
            for sid in range(self._num_slices):
                out["prb_scheduled"][str(sid)] = int(
                    float(row[f"prb_scheduled_rbg_{sid}"])
                )
                out["prb_executed_rbg"][str(sid)] = int(
                    float(row[f"prb_executed_rbg_{sid}"])
                )
        except (KeyError, ValueError):
            return None
        return out

    def latest(self) -> dict | None:
        """Newest complete telemetry row (incremental, offset-based)."""
        if not self._sim_path:
            return None
        path = self._resolve()
        if path is None:
            return None
        try:
            with open(path, "r", newline="") as f:
                size = f.seek(0, os.SEEK_END)
                if size < self._offset:  # truncated/recreated: start over
                    self._offset = 0
                    self._header = None
                    self._last = None
                f.seek(self._offset)
                new_text = f.read()
        except OSError:
            return None
        if new_text and not new_text.endswith("\n"):
            # Torn write: consume only up to the last complete line.
            cut = new_text.rfind("\n")
            if cut == -1:
                return self._last
            new_text = new_text[: cut + 1]
        self._offset += len(new_text)
        for line in new_text.splitlines():
            if not line.strip():
                continue
            if self._header is None:
                self._header = line.strip().split(",")
                continue
            parsed = self._parse_row(line)
            if parsed is not None:
                self._last = parsed
        return self._last


def run_episode(
    env,
    controllers: dict[int, SqrSliceController],
    *,
    gamma: float,
    virtual_alpha: float = DEFAULT_VIRTUAL_ALPHA,
    freeze: bool = False,
    warmup_steps: int = 0,
    seed: int = 1,
    episode: int = 1,
    step_logger: StepMetricsLogger | None = None,
    decision_timer=None,
    checkpoint_path: str | None = None,
    checkpoint_interval: int = 0,
    checkpoint_meta: dict | None = None,
    max_steps: int | None = None,
    ns3_prb_telemetry: "Ns3PrbWindowTelemetry | None" = None,
) -> dict:
    """One closed-loop episode: decide -> act -> learn (paper cadence).

    Returns per-episode aggregates; each slice's y_t (p5) feeds the REAL
    regressor update with the EXECUTED action, and every cut slice gets the
    R12 virtual negative update at its PRE-projection proposal.  Windows
    where a slice has no UE are excluded from that slice's learning
    (register R19) but still logged.  Windows that arrive WITHOUT a KPM
    (contract §5 countable fallback) do not update anything and are
    counted in ``kpm_missing_steps``.
    """
    num_slices = len(controllers)
    total_prbs = next(iter(controllers.values())).n_prbs
    obs, _ = env.reset()
    if ns3_prb_telemetry is None:
        # Real runs: read the ns-3 side PRB telemetry (contract §3.5) so the
        # step log carries the MAC prb_scheduled; inert for contract mocks.
        ns3_prb_telemetry = Ns3PrbWindowTelemetry(
            getattr(env, "sim_path", None), num_slices=num_slices
        )
    ep_rows: list[list] = []
    n_virtual = 0
    n_kpm_missing = 0
    n_learned = {sid: 0 for sid in range(num_slices)}
    decisions_log = []

    step = 0
    while True:
        states = [np.asarray(obs[i], dtype=np.float64) for i in range(num_slices)]

        # -- decide (timed: xApp decision wall-time) -----------------------
        if decision_timer is not None:
            with decision_timer.measure():
                decisions = [controllers[sid].act(states[sid]) for sid in range(num_slices)]
        else:
            decisions = [controllers[sid].act(states[sid]) for sid in range(num_slices)]
        proposals = [d.action for d in decisions]

        # -- act (projection + CSV happen inside the env) ------------------
        obs, reward, terminated, truncated, info = env.step(np.array(proposals, dtype=np.int64))

        # Dynamic C (register R01 as amended): the env adopts the effective
        # cell capacity read from the ns-3 telemetry; sync the controllers
        # so the action set {0..C}, the cost term a/C and the candidate
        # grid follow the SAME C the env projects onto.
        new_c = info.get("total_prbs")
        if (
            isinstance(new_c, int)
            and not isinstance(new_c, bool)
            and new_c > 0
            and new_c != total_prbs
        ):
            total_prbs = new_c
            for ctl in controllers.values():
                setter = getattr(ctl, "set_n_prbs", None)
                if setter is not None:  # KBRL: rescale the accuracy table too
                    setter(int(new_c))
                else:                   # SQR: plain attribute sync
                    ctl.n_prbs = int(new_c)

        if info.get("kpm_missing"):
            n_kpm_missing += 1

        proj = info.get("projection", {})
        executed = proj.get("prb_executed", list(proposals))
        virtual_updates = proj.get("virtual_updates", [])
        n_virtual += len(virtual_updates)

        # KBRL bookkeeping (Fase 3): the reference gates the security-factor
        # update on the global "adjusted" flag — the projection outcome of
        # the window (kbrl_control.py:66-68,98).  SQR controllers have no
        # such hook and are unaffected.
        over_capacity = bool(proj.get("over_capacity", False))
        for ctl in controllers.values():
            hook = getattr(ctl, "notify_projection", None)
            if hook is not None:
                hook(over_capacity)

        # -- learn (online, with the EXECUTED action; register R22) --------
        window_learned = []
        if not freeze:
            for sid in range(num_slices):
                y_t = info.get(f"slice_{sid}", {}).get("p5_thr_mbps")
                if y_t is None:
                    continue  # empty slice: excluded from learning (R19)
                controllers[sid].observe(states[sid], int(executed[sid]), float(y_t))
                window_learned.append(sid)
            # R12: virtual negative update at the pre-projection proposal.
            # SQR-only mechanism (KBRL has no virtual-update analogue);
            # skipped when any controller does not implement it.
            if virtual_alpha > 0 and all(
                hasattr(ctl, "insert_virtual_support") for ctl in controllers.values()
            ):
                apply_virtual_updates(
                    _projection_from_info(info, total_prbs),
                    states,
                    controllers,
                    alpha=virtual_alpha,
                )
        for sid in window_learned:
            n_learned[sid] += 1

        # -- log ------------------------------------------------------------
        now_ms = info.get("kpm_timestamp", "")
        kpi = lambda sid: info.get(f"slice_{sid}", {})
        # MAC-side prb_scheduled (contract §3.5): info first (if the env
        # transports it), otherwise the ns-3 telemetry reader; "" in mocks.
        info_sched = info.get("prb_scheduled")
        if not isinstance(info_sched, dict):
            telemetry_row = (
                ns3_prb_telemetry.latest()
                if ns3_prb_telemetry is not None
                else None
            )
            info_sched = telemetry_row["prb_scheduled"] if telemetry_row else {}
        for sid in range(num_slices):
            d = decisions[sid]
            k = kpi(sid)
            row = [
                episode,
                step + 1,
                now_ms,
                seed,
                gamma,
                sid,
                d.mode,
                d.raw_action,
                d.action,
                int(proj.get("prb_requested", [0] * num_slices)[sid]),
                int(executed[sid]),
                info_sched.get(str(sid), ""),
                total_prbs - int(np.sum(executed)),
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
                len(virtual_updates),
                f"{decision_timer.samples_ms[-1]:.4f}" if decision_timer is not None and decision_timer.samples_ms else "",
                f"{reward:.6f}",
                int((step + 1) <= warmup_steps),
                bool(terminated),
                bool(truncated),
            ]
            ep_rows.append(row)
        decisions_log.append([(d.mode, d.action) for d in decisions])

        if step_logger is not None and ep_rows:
            step_logger.write(ep_rows)
        ep_rows = []

        step += 1
        if checkpoint_path and checkpoint_interval and step % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_path,
                build_checkpoint(controllers, meta=checkpoint_meta),
            )
        done = terminated or truncated or (max_steps is not None and step >= max_steps)
        if done:
            break

    if checkpoint_path:
        save_checkpoint(
            checkpoint_path, build_checkpoint(controllers, meta=checkpoint_meta)
        )

    return {
        "steps": step,
        "virtual_updates": n_virtual,
        "kpm_missing_steps": n_kpm_missing,
        "total_prbs_final": total_prbs,
        "learned_windows": dict(n_learned),
        "decisions": decisions_log,
        # tau/M are SQR quantities; KBRL controllers carry no analogue
        # (NaN keeps the summary shape for both methods).
        "final_tau": {
            sid: float(getattr(controllers[sid], "tau", float("nan")))
            for sid in controllers
        },
        "final_margin": {
            sid: float(getattr(controllers[sid], "margin", float("nan")))
            for sid in controllers
        },
    }


def _projection_from_info(info: dict, total_prbs: int):
    """Rebuild a minimal ProjectionResult-like view from the env info.

    ``apply_virtual_updates`` only needs ``.virtual_updates``; the env
    already serializes them.  For a feasible step the list is empty and no
    update is applied.
    """
    from nsoran.sqr.coordinator import ProjectionResult, VirtualUpdate

    proj = info.get("projection", {})
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
        total_prbs=total_prbs,
        cut=np.asarray(proj.get("cut", []), dtype=bool),
        virtual_updates=updates,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="SQR/KBRL closed-loop run over the IPC contract (NumPy only)"
    )
    parser.add_argument("--controller", type=str, default="sqr",
                        choices=CONTROLLER_CHOICES,
                        help="sqr = model-based SQR agent (Fase 2); "
                             "kbrl = KBRL baseline (Alcaraz et al., TWC 2022, "
                             "Fase 3 port) on the SAME obs/action contract")
    parser.add_argument("--gamma", type=int, default=1, choices=(0, 1),
                        help="0 = quantile only; 1 = quantile minus a/C "
                             "(cost term ONLY; P_penalty active in both). "
                             "SQR only — ignored by --controller kbrl")
    parser.add_argument("--ns3_path", type=str, default=DEFAULT_NS3_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--simTime", type=float, default=600.0,
                        help="simulated seconds per episode (R23: 600 s)")
    parser.add_argument("--appStart", type=float, default=0.5)
    parser.add_argument("--periodMs", type=int, default=100,
                        help="decision window (contract §2: 100 ms)")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="override steps per episode (0 = auto from simTime)")
    parser.add_argument("--c_prbs", type=int, default=52,
                        help="INITIAL C, cell capacity in PRBs (register R01 "
                             "prior); once the ns-3 telemetry appears the "
                             "effective C is read from sqr_prb_windows.csv "
                             "(total_prb column, R01 as amended — 53 on the "
                             "10 MHz scenario)")
    parser.add_argument("--num_slices", type=int, default=3)
    parser.add_argument("--slas", type=str, default="0.5,1.0,1.5",
                        help="per-slice per-UE SLA in Mbps [ARTIGO, Tabela I]")
    parser.add_argument("--violation_window", type=int, default=VIOLATION_WINDOW,
                        help="W periods for the tau/M controller (R11)")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="uncertain-mode exploration [ARTIGO]")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="T_softmax [ASSUMIDO, R07]")
    parser.add_argument("--knn_k", type=int, default=10,
                        help="safety-first k-NN neighbors [ARTIGO, Tabela I]")
    parser.add_argument("--virtual_alpha", type=float, default=DEFAULT_VIRTUAL_ALPHA,
                        help="eta_virt of the R12 virtual update (ablation "
                             "{0.025, 0.05, 0.1}; 0 disables)")
    parser.add_argument("--freeze", action="store_true",
                        help="evaluation mode: no regressor/memory updates")
    parser.add_argument("--warmup_s", type=float, default=100.0,
                        help="warm-up seconds excluded from metrics (R14)")
    parser.add_argument("--checkpoint_interval", type=int, default=500,
                        help="steps between checkpoints (0 = only at episode end)")
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="checkpoint file (default: <output>/sqr_checkpoint.pkl)")
    parser.add_argument("--resume", action="store_true",
                        help="restore the controllers from --checkpoint_path "
                             "before the first episode")
    parser.add_argument("--step_log_file", type=str, default="step_metrics.csv")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output, exist_ok=True)

    slas = parse_slas(args.slas, args.num_slices)
    config = {
        "simTime": [args.simTime],
        "appStart": [args.appStart],
        "scenario": ["sqr"],
        "seed": [args.seed],
        "seed_cycle": [100],
        "periodMs": [args.periodMs],
        "ues": [20],  # datalake hint; sqr-sim has dynamic UE churn
    }
    max_steps = args.max_steps if args.max_steps > 0 else None
    warmup_steps = int(round(args.warmup_s * 1000.0 / args.periodMs))

    interaction_budget = args.episodes * (max_steps or int(
        (args.simTime - args.appStart) * 1000 / args.periodMs
    ))
    accounting = ComputeAccounting(
        interaction_budget=interaction_budget, period_ms=args.periodMs
    )

    env = SqrEnv(
        ns3_path=os.path.abspath(args.ns3_path),
        scenario_configuration=config,
        output_folder=args.output,
        optimized=False,
        total_prbs=args.c_prbs,
        num_slices=args.num_slices,
        sla_thresholds=slas,
        max_steps=max_steps,
    )
    # Integração ns-O-RAN ↔ sqr-sim: o CommandLine do ns-3.46 ABORTA
    # (std::exit(1), src/core/model/command-line.cc HandleArgument::errorExit)
    # ao receber uma opção não registrada pelo cenário C++. O SqrEnv carrega
    # chaves de CONFIG do gym (scenario/appStart/ues) que o sqr-sim não
    # registra — são metadados do lado Python (o cenário SQR tem churn
    # dinâmico de UEs; o tráfego começa no attachGrace) e são removidas
    # ANTES do start_sim montar o comando --opt=valor.
    for key in ("scenario", "appStart", "ues"):
        env.scenario_configuration.pop(key, None)
    controllers = build_controllers(args)
    checkpoint_path = args.checkpoint_path or os.path.join(
        args.output, "sqr_checkpoint.pkl"
    )
    if args.resume:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"--resume: checkpoint not found at {checkpoint_path}"
            )
        restored_meta = restore_controllers(
            load_checkpoint(checkpoint_path), controllers
        )
        print(f"[sqr_run] resumed controllers from {checkpoint_path} (meta={restored_meta})")

    step_logger = StepMetricsLogger(os.path.join(args.output, args.step_log_file))
    meta = {
        "controller": args.controller,
        "gamma": args.gamma,
        "seed": args.seed,
        "c_prbs": args.c_prbs,
        "slas": slas,
        "virtual_alpha": args.virtual_alpha,
        "freeze": bool(args.freeze),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    total_kpm_missing = 0
    for episode in range(1, args.episodes + 1):
        summary = run_episode(
            env,
            controllers,
            gamma=float(args.gamma),
            virtual_alpha=args.virtual_alpha,
            freeze=args.freeze,
            warmup_steps=warmup_steps,
            seed=args.seed,
            episode=episode,
            step_logger=step_logger,
            decision_timer=accounting.decision_timer,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_meta=meta,
        )
        total_kpm_missing += summary["kpm_missing_steps"]
        print(
            f"[sqr_run] episode {episode}: steps={summary['steps']} "
            f"virtual_updates={summary['virtual_updates']} "
            f"kpm_missing={summary['kpm_missing_steps']} "
            f"total_prbs_final={summary['total_prbs_final']} "
            f"tau={ {k: round(v, 3) for k, v in summary['final_tau'].items()} } "
            f"M={ {k: round(v, 1) for k, v in summary['final_margin'].items()} }"
        )

    finish = accounting.finish()
    print(f"[sqr_run] xApp decision accounting: {finish['xapp_runtime']}")
    # Resumo final do fallback contável (contrato §5): janelas que chegaram
    # sem KPM — observação congelada na última janela válida, contadas por
    # passo em step_metrics.csv (coluna kpm_missing).
    print(f"[sqr_run] kpm_missing_steps total (contrato §5): {total_kpm_missing}")

    # Integração: esperar o término limpo do par de processos. O loop trunca
    # em max_steps (contrato §2) mas o ns-3 segue até --simTime e só então
    # escreve o relatório final e faz o cleanup dos semáforos POSIX. Sem esta
    # espera o C++ ficaria órfão e o stdout final (SQR RESULTS / "IPC
    # semaphores cleaned up") não seria drenado para os arquivos do run.
    sim_path = getattr(env, "sim_path", None)
    ns3_exit_code = None
    sim_process = getattr(env, "sim_process", None)
    if sim_path is not None and sim_process is not None:
        deadline = time.time() + 900.0
        while sim_process.poll() is None and time.time() < deadline:
            time.sleep(0.5)
        selector = getattr(env, "selector", None)
        for _ in range(100):  # drena o stdout/stderr remanescente do ns-3
            if selector is None:
                break
            try:
                events = selector.select(timeout=0.2)
                env.read_streams()
            except (OSError, ValueError):
                break
            if not events:
                break
        ns3_exit_code = sim_process.returncode
        print(f"[sqr_run] sim_path: {sim_path}")
        print(f"[sqr_run] ns3_exit_code: {ns3_exit_code}")

    print(f"[sqr_run] checkpoint: {checkpoint_path}")
    return finish


if __name__ == "__main__":
    main()
