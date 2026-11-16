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

Fase 4: ``--controller equal`` is the static baseline of the campaign
(ask item 4): a trivial Python controller proposing the FIXED equal split
of C across the slices (33.33%/slice; ``sqr_dummy_controller.py`` phase-C
pattern as deterministic integers) — no learning, no observation
dependence.  The method is registered in the run through the checkpoint
meta (``controller``) and the constant ``mode == "equal"`` column of
step_metrics.csv.

Rodada 2 Etapa 2.4 (Despacho II-a): ``--controller replay`` is the
deterministic SINGLE-WINDOW action-override replay of the executable
oracle validation (protocol §15-§16).  Same scenario/seed as a frozen
2.3 run (MESMA seed ⇒ MESMO estado até t), base action ``--replay_base``
(``equal`` = the scenario's initial partition (18, 17, 18) at C = 53,
reproducing the network-only arm; ``prop`` = the unchanged
PropDemandController sequence of the IPC prop arm) at every window
EXCEPT ``--replay_at_win T``, where the fixed ``--replay_action
'aE,aU,aM'`` (integer RBGs, Σ ≤ C) is applied instead; windows > T
return to the base.  The act() call k governs oracle window k
(window w covers [100w, 100(w+1)) ms — see
``nsoran.sqr.replay_controller`` for the verified mapping).  A SHORT
``--simTime`` (≈ T/10 + 2 s) stops the run soon after the override;
``--c_prbs 53`` is RECOMMENDED so the very first decision already uses
the effective telemetry C (equal base exact from window 1 on).

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
from collections import deque
import json
import time

import numpy as np

from nsoran.compute_accounting import ComputeAccounting
from nsoran.sqr.checkpoint import (
    build_checkpoint,
    load_checkpoint,
    restore_controllers,
    save_checkpoint,
)
from nsoran.sqr.coordinator import (
    DEFAULT_VIRTUAL_ALPHA,
    apply_virtual_updates,
    project_prbs,
)
from nsoran.sqr.sqr_controller import (
    MARGIN_BOUNDS,
    MARGIN_INIT,
    TAU_INIT,
    TAU_STEP,
    VIOLATION_WINDOW,
    SqrSliceController,
)
from nsoran.sqr.sqr_env import SqrEnv
from nsoran.sqr_jc.freeze_contract import (
    CausalTraceWriter,
    build_b0_config_descriptor,
    build_version_manifest,
    checkpoint_payload_sha256,
    verify_resume_contract,
)

#: Controller choices of --controller (Fase 3): the model-based SQR agent
#: or the KBRL baseline (Alcaraz et al., IEEE TWC 2022) ported as a
#: consumer of the SAME obs/action contract.  ``gamma`` and the R12 virtual
#: updates apply to SQR only (KBRL is resource-economical structurally and
#: has no virtual-update mechanism).  Rodada 2 Etapa 2.2 adds the sanity
#: baselines "equal" (fixed split) and "prop" (demand-proportional with
#: service floors, register R39).  Etapa 2.4 Despacho II-a adds "replay"
#: (deterministic single-window action override, oracle §15-§16).
CONTROLLER_CHOICES = ("sqr", "kbrl", "equal", "prop", "replay", "sqr_het")

#: Default frozen §13 feature scaler (CAL manifest) for --controller sqr_het.
DEFAULT_HET_SCALER = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "results", "sqr_jc_final",
    "03_calibration", "feature_scaler_manifest.yaml")   # repo-root results

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


def build_controllers(args, env=None):
    """One controller per slice with the frozen R11/R21 defaults.

    ``args.controller`` selects the method (default "sqr" for backward
    compatibility with older invocations/tests): "sqr" builds the
    hierarchical SQR stack; "kbrl" builds the KBRL baseline port (same
    obs/action contract; gamma and the SQR-only knobs are ignored);
    "equal" builds the Fase 4 static baseline (fixed 33.33%/slice split,
    ``nsoran.sqr.static_controllers.EqualSplitController`` — no learning,
    gamma and the SQR-only knobs are ignored); "prop" builds the Rodada 2
    Etapa 2.2 demand-proportional sanity baseline
    (``nsoran.sqr.static_controllers.PropDemandController``, register R39),
    which reads the parsed KPM window through ``env.builder.last_window``
    when an ``env`` with a builder is given (floor-only proposals in mocks
    without one — same no-learning contract as "equal").  "replay" builds
    the Etapa 2.4 single-window override controllers
    (``nsoran.sqr.replay_controller.ReplayController``): base
    ``args.replay_base`` everywhere, the fixed ``args.replay_action`` at
    window ``args.replay_at_win`` (validated: integer RBGs, Σ ≤ C;
    ValueError otherwise), ``LEARNING = False``.
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
    if controller == "equal":
        from nsoran.sqr.static_controllers import EqualSplitController

        return {
            sid: EqualSplitController(
                slice_id=sid,
                n_prbs=args.c_prbs,
                num_slices=args.num_slices,
            )
            for sid in range(args.num_slices)
        }
    if controller == "prop":
        from nsoran.sqr.static_controllers import PropDemandController

        builder = getattr(env, "builder", None)
        window_provider = (
            (lambda b: lambda: b.last_window)(builder)
            if builder is not None
            else None
        )
        return {
            sid: PropDemandController(
                slice_id=sid,
                n_prbs=args.c_prbs,
                num_slices=args.num_slices,
                window_provider=window_provider,
            )
            for sid in range(args.num_slices)
        }
    if controller == "replay":
        from nsoran.sqr.replay_controller import (
            ReplayController,
            parse_replay_action,
        )

        # D2 (PROTOCOL_FROZEN §4.2): --replay_base pode ser um PATH para um
        # run B0 (dir com decisions.jsonl) — base "seq" = a própria sequência
        # de ações gravada do B0 (continuação idêntica em todos os ramos).
        base_name = str(getattr(args, "replay_base", "equal"))
        seq_actions = None
        seq_log = None
        if base_name not in ("equal", "prop"):
            import json as _json
            jl = os.path.join(base_name, "decisions.jsonl")
            if not os.path.isfile(jl):
                raise ValueError(
                    f"--replay_base {base_name!r}: nem equal/prop nem um "
                    "run B0 com decisions.jsonl"
                )
            # A base seq reproduz o fluxo EXECUTADO do B0 (telemetria
            # prb: quota em vigor por janela = budget/dl_slots, inteira;
            # janelas sem slot DL preenchidas com a decisão pretendida —
            # cota irrelevante nessas janelas). O fluxo pretendido
            # (decisions.jsonl) diverge do executado em janelas de lag IPC
            # (incidente A/A D2: run B0 aplicou a decisão k-1 na janela k
            # em transições — corrida semáforo/callback que autocorrige);
            # só o fluxo EXECUTADO reproduz o prefixo bit-a-bit.
            import numpy as _np
            import pandas as pd
            pw = sorted(glob.glob(os.path.join(base_name, "**",
                                               "sqr_prb_windows.csv"),
                                  recursive=True))
            if not pw:
                raise ValueError(f"run B0 {base_name!r} sem telemetria prb")
            prb = pd.read_csv(pw[0])
            intended = []
            for line in open(jl):
                intended.append(tuple(int(v) for v in _json.loads(line)["action"]))
            # D2-v2 incident (13/11): the base universe is the CLOCK-WINDOW
            # span of the executed telemetry, NOT len(decisions.jsonl) —
            # v8/v9 base runs log no decision on cell-empty HOLD windows
            # (e.g. lam2_s2102: 848 decisions for ~2972 slot-DL windows),
            # so states with replay_at_win beyond the decision count were
            # rejected even though the EXECUTED flow exists for every
            # slot-DL window in the prb telemetry.
            n = int(prb.timestamp_ms.max() // 100)
            seq_actions = [None] * n
            dl = prb[prb.dl_slots > 0]
            for row in dl.itertuples():
                w = int(row.timestamp_ms // 100 - 1)
                if 1 <= w <= n:
                    q = [int(round(getattr(row, f"prb_budget_rbg_{i}")
                                   / row.dl_slots)) for i in range(3)]
                    seq_actions[w - 1] = tuple(q)
            # fill holes: windows without slot DL carry no physical effect
            # (nothing scheduled ⇒ KPM unaffected) — before the first
            # executed quota use the intended decisions in order (same as
            # the pre-v9 behaviour, bit-exact-validated in D0-B); after it,
            # hold the last executed quota (ns-3 keeps the previous
            # configuration on no-action windows).
            k_int = 0
            last_q = None
            for i in range(n):
                if seq_actions[i] is not None:
                    last_q = seq_actions[i]
                elif last_q is None and k_int < len(intended):
                    seq_actions[i] = intended[k_int]
                    k_int += 1
                else:
                    seq_actions[i] = (last_q if last_q is not None
                                      else intended[k_int])
            seq_actions = tuple(seq_actions)
            seq_log = f"seq(B0 {os.path.basename(base_name)}, {len(seq_actions)} janelas)"
            print(f"[sqr_run] replay base seq: {seq_log}")

        override = parse_replay_action(
            getattr(args, "replay_action", ""),
            num_slices=args.num_slices,
            n_prbs=args.c_prbs,
        )
        at_win = int(getattr(args, "replay_at_win", 0) or 0)
        if at_win < 1:
            raise ValueError(
                "--replay_at_win must be >= 1 (window 0 runs before any "
                f"controller decision), got {at_win}"
            )
        builder = getattr(env, "builder", None)
        window_provider = (
            (lambda b: lambda: b.last_window)(builder)
            if builder is not None and getattr(args, "replay_base", "") == "prop"
            else None
        )
        return {
            sid: ReplayController(
                slice_id=sid,
                n_prbs=args.c_prbs,
                num_slices=args.num_slices,
                base=("seq" if seq_actions is not None
                      else getattr(args, "replay_base", "equal")),
                replay_at_win=at_win,
                replay_action=override,
                window_provider=window_provider,
                seq_actions=seq_actions,
            )
            for sid in range(args.num_slices)
        }
    if controller == "sqr_het":
        # D1 (protocolo v2 §2.2/§27): the SAME audited per-slice stack on
        # the COMMON MARGIN scale (Lambda = 0).  The SqrHetController wrapper
        # (slice-wise decide + original projection) is built in main(),
        # where the env delay stream and the frozen §13 scaler live.
        raw_margin_init = getattr(args, "margin_init", None)
        margin_init = MARGIN_INIT if raw_margin_init is None else float(raw_margin_init)
        raw_margin_bounds = getattr(args, "margin_bounds", None)
        margin_bounds = (
            MARGIN_BOUNDS
            if raw_margin_bounds is None
            else parse_margin_bounds(raw_margin_bounds)
        )
        return {
            sid: SqrSliceController(
                state_dim=6,                     # R20: 4 local + 2 global
                n_prbs=args.c_prbs,
                gamma=float(args.gamma),
                sla_threshold=0.0,               # margin scale (§27)
                tau=TAU_INIT,
                tau_step=TAU_STEP,
                margin=margin_init,
                margin_bounds=margin_bounds,
                violation_window=args.violation_window,
                epsilon=args.epsilon,
                temperature=args.temperature,
                knn_k=args.knn_k,
                seed=None if args.seed is None else int(args.seed) + 1000 * sid,
            )
            for sid in range(args.num_slices)
        }
    if controller != "sqr":
        raise ValueError(
            f"unknown controller {controller!r} (choices: {CONTROLLER_CHOICES})"
        )
    raw_margin_init = getattr(args, "margin_init", None)
    margin_init = MARGIN_INIT if raw_margin_init is None else float(raw_margin_init)
    raw_margin_bounds = getattr(args, "margin_bounds", None)
    margin_bounds = (
        MARGIN_BOUNDS
        if raw_margin_bounds is None
        else parse_margin_bounds(raw_margin_bounds)
    )
    return {
        sid: SqrSliceController(
            state_dim=6,                       # R20: 4 local + 2 global
            n_prbs=args.c_prbs,
            gamma=float(args.gamma),
            sla_threshold=slas[sid],
            tau=TAU_INIT,                      # tau0 = 0.05 (R11)
            tau_step=TAU_STEP,                 # delta_tau = 0.005 (R11)
            margin=margin_init,                # M0 = 10 PRBs (R11)
            margin_bounds=margin_bounds,       # [2, 50] (R11)
            violation_window=args.violation_window,   # W = 100 (R11)
            epsilon=args.epsilon,              # 0.1 [ARTIGO]
            temperature=args.temperature,      # T_softmax [ASSUMIDO]
            knn_k=args.knn_k,                  # K = 10 [ARTIGO]
            seed=None if args.seed is None else int(args.seed) + 1000 * sid,
        )
        for sid in range(args.num_slices)
    }


def parse_margin_bounds(bounds_arg) -> tuple[float, float]:
    """Parse uncertainty-margin bounds ``(M_min, M_max)`` from ``'lo,hi'`` or tuple."""
    if isinstance(bounds_arg, (tuple, list)):
        if len(bounds_arg) != 2:
            raise ValueError(
                f"--margin_bounds must have 2 values, got {bounds_arg!r}"
            )
        lo, hi = float(bounds_arg[0]), float(bounds_arg[1])
    else:
        parts = [p.strip() for p in str(bounds_arg).split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(
                f"--margin_bounds must be 'lo,hi', got {bounds_arg!r}"
            )
        lo, hi = float(parts[0]), float(parts[1])
    if lo < 0.0 or hi < lo:
        raise ValueError(
            f"--margin_bounds requires 0 <= lo <= hi, got ({lo}, {hi})"
        )
    return (lo, hi)


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


def _save_checkpoint_with_hash(
    path: str,
    controllers: dict[int, SqrSliceController],
    meta: dict | None,
    *,
    horizon: dict | None = None,
) -> None:
    """Build checkpoint payload, inject payload_sha256 in meta, and persist."""
    meta_copy = dict(meta) if meta is not None else {}
    if horizon is not None:
        meta_copy["checkpoint_horizon"] = dict(horizon)
    payload = build_checkpoint(controllers, meta=meta_copy if (meta is not None or horizon is not None) else None)
    if payload.get("meta") is not None:
        payload["meta"]["payload_sha256"] = checkpoint_payload_sha256(payload)
    save_checkpoint(path, payload)


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
    het=None,
    het_runtime=None,
    decision_jsonl: str | None = None,
    het_meta: dict | None = None,
    het_bootstrap=None,
    urllc_delayed_update: bool = False,
    causal_trace_path: str | None = None,
    causal_trace_meta: dict | None = None,
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
    if het is not None and het_runtime is None:
        # LAZY (pós-reset): env.sim_path existe aqui — o delay stream é o
        # DESTE run (corrige o incidente de contaminação do D1)
        from nsoran.sqr_jc.het_runtime import HetRuntime
        het_runtime = HetRuntime(getattr(env, "sim_path", ""))
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
    # -- D1 SQR-HET joint-path state (protocolo v2 §7.1) ------------------
    _het_jsonl = (open(decision_jsonl, "a", encoding="utf-8")
                  if (het is not None and decision_jsonl) else None)
    _trace_meta = causal_trace_meta if causal_trace_meta is not None else het_meta
    _causal_writer = (
        CausalTraceWriter(causal_trace_path, metadata=_trace_meta)
        if (het is not None and causal_trace_path)
        else None
    )
    _deferred_trace_kwargs = None

    class _HoldDecision:
        """step-row placeholder while no KPM window exists yet."""
        mode = "hold"
        raw_action = 0
        action = 0
        sigma = 0.0
        margin_m = 0.0
        margin_applied = 0
        tau = 0.0

    _het_pending = None
    # Delayed feedback URLLC (flag --urllc_delayed_update — teste isolado,
    # Emenda candidata): fila de pendentes {w, state, action, age}; a cada
    # iteração, W cujo pool FECHOU online (delivered==tx, ambos conhecidos
    # no instante) recebe uma SEGUNDA observe_outcome só-URLLC (eMBB/mMTC
    # None ⇒ R19 skip ⇒ 0 duplicação); W sem demanda sai; pool que nunca
    # fecha (perda real) expira em _URLLC_MAX_AGE janelas como vacuous (R19:
    # sem-RX nunca é miss nem sucesso implícito). Log dedicado auditável.
    _urllc_delayed = deque() if urllc_delayed_update else None
    _urllc_updated_wins: set[int] = set()
    _URLLC_MAX_AGE = 30          # janelas (3 s) — só limite de fila
    _urllc_jsonl = None
    if _urllc_delayed is not None and decision_jsonl:
        _urllc_jsonl = open(os.path.join(
            os.path.dirname(os.path.abspath(decision_jsonl)),
            "urllc_delayed_updates.jsonl"), "a", encoding="utf-8")
    _het_prev_action = (0, 0, 0)
    _het_prev_viol = (0, 0, 0)
    _het_prev_decisions = None

    step = 0
    last_window_index = -1
    last_sim_time_ms = -1
    while True:
        states = [np.asarray(obs[i], dtype=np.float64) for i in range(num_slices)]

        # -- decide (timed: xApp decision wall-time) -----------------------
        if het is not None:
            # D1 §7.1: joint decide on the JUST-CLOSED KPM window W; the
            # outcome of the PREVIOUS decision (governed window W) is
            # observed here (one-window update latency, logged explicitly).
            win = getattr(getattr(env, "builder", None), "last_window", None)
            # Guard anti-repetição (bug 3, D1-v6 λ5s2102) + fix anti-spin
            # (bug 4, D1-v7: `continue` antes do env.step girava sem o
            # bloqueio do semáforo ⇒ decisions_log sem dreno ⇒ OOM 10,3 GB).
            # HOLD = timestamp do builder não avançou (célula vazia):
            # repete a última ação e SEGUE o fluxo normal até env.step
            # (contrato IPC nunca é pulado); NÃO absorve/observa/loga de
            # novo.  O pending pendente é observado contra a SUA janela
            # governada (outcome_for_window — Opção B: quanto mais tarde,
            # mais completo o outcome da janela governada).
            _win_ts = int(win.timestamp) if win is not None else None
            _is_hold = (win is not None
                        and _win_ts == getattr(het, "_last_win_ts", -1))
            if win is not None and not _is_hold:
                het._last_win_ts = _win_ts
                het_runtime.absorb(win)
            if _het_pending is not None:
                _gov = int(_het_pending["state"].window_index) + 1
                outcome = het_runtime.outcome_for_window(_gov)
                if not freeze:
                    het.observe_outcome(_het_pending["state"],
                                        _het_pending["action"], outcome)
                margins = outcome.margins
                if _het_jsonl is not None:
                    row = {k: v for k, v in _het_pending.items() if k != "state"}
                    row["sla_evidence"] = het_runtime.evidence(
                        _het_pending["state"].window_index + 1)
                    row["sla_outcome_margins"] = [
                        None if m is None else float(m) for m in margins]
                    row["sla_ok"] = [bool(b) for b in outcome.sla_ok]
                    row["joint_frozen_ok"] = bool(all(outcome.sla_ok))
                    row["outcome_window_index"] = _gov
                    row["outcome_kpm_stamp_ms"] = (_gov + 1) * 100
                    _het_jsonl.write(json.dumps(row, default=float) + "\n")
                    _het_jsonl.flush()
                if (_urllc_delayed is not None
                        and margins[1] is None
                        and _gov not in _urllc_updated_wins):
                    # stage-2 enqueue: margem URLLC ainda indisponível no
                    # instante do update imediato — re-perguntar quando o
                    # pool fechar ONLINE (sem dado futuro algum)
                    _tag0, _ = het_runtime.urllc_pool_status(_gov)
                    if _tag0 in ("OPEN",):
                        _urllc_delayed.append({
                            "w": _gov, "age": 0,
                            "state": _het_pending["state"],
                            "action": _het_pending["action"]})
                _het_prev_viol = tuple(
                    0 if (m is None or m >= 0) else 1 for m in margins)
                _het_pending = None
                del outcome, margins
            if _urllc_delayed is not None:
                # stage-2 drain: um passe FINITO por iteração (fila de
                # tamanho ≤ MAX_AGE; sem spin — env.step abaixo sempre roda)
                for _ in range(len(_urllc_delayed)):
                    _it = _urllc_delayed.popleft()
                    _w = int(_it["w"])
                    if _w in _urllc_updated_wins:
                        continue          # dup guard (defensivo)
                    _tag, _st = het_runtime.urllc_pool_status(_w)
                    _cch = het_runtime._win_cache.get(_w) or {}
                    _log = {"w": _w, "tag": _tag, "age": int(_it["age"]),
                            "observed_at_window":
                                getattr(het_runtime, "last_window_index",
                                        None),
                            "tx": int(_cch.get("urllc_tx") or 0),
                            "delivered": len(
                                het_runtime.delays.by_win.get(_w, ()))}
                    if _tag == "CLOSED":
                        _oc = het_runtime.urllc_only_outcome(_w)
                        if not freeze:
                            het.observe_outcome(_it["state"],
                                                _it["action"], _oc)
                        _urllc_updated_wins.add(_w)
                        _log.update({
                            "updated": True,
                            "p95_ms": (_st or {}).get("p95_ms"),
                            "miss_pct": (100.0 * _st["miss_rate"])
                                        if _st else None,
                            "n_pkts": (_st or {}).get("n_pkts")})
                    elif _tag in ("NO_DEMAND", "NO_KPM"):
                        _log.update({"updated": False,
                                     "reason": _tag})
                    else:               # OPEN
                        _it["age"] += 1
                        if _it["age"] >= _URLLC_MAX_AGE:
                            _log.update({"updated": False,
                                         "reason": "expired_vacuous_R19"})
                        else:
                            _urllc_delayed.append(_it)
                            _log = None      # ainda pendente: não loga
                    if _log is not None and _urllc_jsonl is not None:
                        _urllc_jsonl.write(json.dumps(_log,
                                                      default=float) + "\n")
                if _urllc_jsonl is not None:
                    _urllc_jsonl.flush()
            if win is None:
                proposals = list(_het_prev_action)          # hold (kpm missing)
                decisions = _het_prev_decisions or [_HoldDecision()
                                                    for _ in range(num_slices)]
            elif _is_hold and _het_prev_decisions is not None:
                # HOLD (célula vazia; bug 3): repete a última decisão — o
                # pending foi observado acima contra a janela governada; a
                # ação mantida é o que fisicamente governa as janelas do gap.
                decisions = list(_het_prev_decisions)
                proposals = list(_het_prev_action)
            elif (het_bootstrap is not None
                  and (step + 1) <= warmup_steps):
                # PROTOCOL_FROZEN §3.4 — bootstrap-PROP compartilhado
                # 0–100 s: decisões = controlador prop congelado (R39);
                # o learner SQR-HET OBSERVA os outcomes destas ações
                # (atualização normal no pending-outcome path abaixo).
                prop_decisions = [het_bootstrap[sid].act(states[sid])
                                  for sid in range(num_slices)]
                prop_proposals = [int(d.action) for d in prop_decisions]
                presult = project_prbs(
                    np.asarray(prop_proposals, dtype=np.int64), total_prbs)
                allocated = [int(v) for v in presult.allocated]
                class _PW:  # step-row placeholder com campos SqrDecision
                    def __init__(self, d, a):
                        self.mode, self.raw_action = "bootstrap_prop", int(d.action)
                        self.action = int(a)
                        self.sigma = float("nan")
                        self.margin_m = float("nan")
                        self.margin_applied = 0
                        self.tau = float("nan")
                decisions = [_PW(d, a) for d, a in
                             zip(prop_decisions, allocated)]
                proposals = allocated
                st = het_runtime.joint_state(win, _het_prev_action,
                                             _het_prev_viol)
                zs = [het.slice_state(st, sid) for sid in range(num_slices)]
                preds_b, preds_a = [], []
                f_a0, f_a37, f_amax = [], [], []
                for sid in range(num_slices):
                    sc = controllers[sid].score_candidates(zs[sid])
                    preds_b.append(float(sc["f"][prop_proposals[sid]]))
                    preds_a.append(float(sc["f"][allocated[sid]]))
                    f_a0.append(float(sc["f"][0]))
                    f_a37.append(float(sc["f"][min(37, len(sc["f"]) - 1)]))
                    f_amax.append(float(sc["f"][-1]))
                if (not freeze) and any(presult.cut):
                    het.apply_virtual_updates(st, tuple(prop_proposals),
                                              tuple(allocated))
                _het_pending = {
                    **(het_meta or {}),
                    "episode": episode, "decision_index": step + 1,
                    "sim_time_ms": int(win.timestamp),
                    "policy": "SQR_HET_BOOTSTRAP_PROP",
                    "state_window_index": st.window_index,
                    "action_unit": "RBG (=1 PRB per DL slot; resource_unit_manifest)",
                    "capacity_C": int(total_prbs),
                    "a_raw_per_slice": list(prop_proposals),
                    "a_after_uncertainty_margin": list(prop_proposals),
                    "action": list(allocated),
                    "a_after_projection": list(allocated),
                    "projection_triggered": bool(any(presult.cut)),
                    "projection_delta_l1": int(sum(
                        int(p) - int(a) for p, a in zip(prop_proposals,
                                                        allocated))),
                    "sum_proposals_over_C": int(sum(prop_proposals)
                                                - total_prbs),
                    "predicted_safe_before_proposal": preds_b,
                    "predicted_safe_after_projection": preds_a,
                    "f_at_a0": f_a0,
                    "f_at_a37": f_a37,
                    "f_at_amax": f_amax,
                    "slice_n_ues": [st.n_e, st.n_u, st.n_m],
                    "slice_backlog": [st.queue_e, st.queue_u, st.queue_m],
                    "mean_field": bool(het.mean_field),
                    "fix_scaler_index": bool(het.fix_scaler_index),
                    "z_dim0_dim1_per_slice": [
                        [round(float(z[0]), 6), round(float(z[1]), 6)]
                        for z in zs
                    ],
                    "z_dim4_dim5_per_slice": [
                        [round(float(z[4]), 6), round(float(z[5]), 6)]
                        for z in zs
                    ],
                    "state": st,
                }
                _het_prev_action = tuple(int(v) for v in proposals)
                _het_prev_decisions = list(decisions)
            else:
                st = het_runtime.joint_state(win, _het_prev_action,
                                             _het_prev_viol)
                st.extras["urllc_pool_open"] = bool(
                    st.extras.get("urllc_pool_open", False)
                    or (_urllc_delayed is not None and len(_urllc_delayed) > 0)
                )
                if decision_timer is not None:
                    with decision_timer.measure():
                        het_dec = het.decide(st)
                else:
                    het_dec = het.decide(st)
                decisions = list(het_dec.slice_decisions)
                proposals = list(het_dec.action)
                # §4.2: model predictions at the PRE-projection proposal vs
                # the POST-projection allocation (same models, same z_i).
                zs = [het.slice_state(st, sid) for sid in range(num_slices)]
                preds_b, preds_a = [], []
                f_a0, f_a37, f_amax = [], [], []
                _sc_cache = []
                for sid in range(num_slices):
                    sc = controllers[sid].score_candidates(zs[sid])
                    if _causal_writer is not None:
                        _sc_cache.append(sc)
                    preds_b.append(float(sc["f"][int(het_dec.proposals[sid])]))
                    preds_a.append(float(sc["f"][int(het_dec.action[sid])]))
                    f_a0.append(float(sc["f"][0]))
                    f_a37.append(float(sc["f"][min(37, len(sc["f"]) - 1)]))
                    f_amax.append(float(sc["f"][-1]))
                _deferred_trace_kwargs = None
                if _causal_writer is not None:
                    _trace_kwargs = {
                        "step_index": step + 1,
                        "sim_time_ms": int(win.timestamp),
                        "policy": "SQR_HET",
                        "state": st,
                        "zs": zs,
                        "controllers": controllers,
                        "capacity_c": int(total_prbs),
                        "a_raw_per_slice": [int(d.raw_action) for d in decisions],
                        "a_prop_per_slice": list(het_dec.proposals),
                        "a_exec_per_slice": list(het_dec.action),
                        "cut_per_slice": list(het_dec.cut),
                        "modes_per_slice": [d.mode for d in decisions],
                        "precomputed_scores": _sc_cache,
                    }
                    _will_apply_virt = (not freeze) and any(het_dec.cut) and (
                        float(getattr(het, "virtual_alpha", 0.0)) != 0.0
                    )
                    if _will_apply_virt:
                        _causal_writer.record_decision(**_trace_kwargs)
                    else:
                        # Defer serialization/flush until right AFTER controlSemaphore.release()
                        # while ns-3 simulates the next window (zero overhead inside the 50 ms IPC timeout).
                        _deferred_trace_kwargs = _trace_kwargs
                if (not freeze) and any(het_dec.cut):
                    het.apply_virtual_updates(st, het_dec.proposals,
                                              het_dec.action)
                _het_pending = {
                    **(het_meta or {}),
                    "episode": episode, "decision_index": step + 1,
                    "sim_time_ms": int(win.timestamp),
                    "policy": "SQR_HET",
                    "state_window_index": st.window_index,
                    "action_unit": "RBG (=1 PRB per DL slot; resource_unit_manifest)",
                    "capacity_C": int(total_prbs),
                    "a_raw_per_slice": [int(d.raw_action) for d in decisions],
                    "a_after_uncertainty_margin": list(het_dec.proposals),
                    "a_unconstrained_raw_per_slice": list(het_dec.unconstrained_raw),
                    "a_unconstrained_prop_per_slice": list(het_dec.unconstrained_proposals),
                    "guard_suppressed_per_slice": [bool(v) for v in het_dec.guard_suppressed],
                    "action": [int(v) for v in het_dec.action],
                    "a_after_projection": list(het_dec.action),
                    "projection_triggered": bool(any(het_dec.cut)),
                    "projection_delta_l1": int(sum(
                        int(p) - int(a) for p, a in zip(het_dec.proposals,
                                                        het_dec.action))),
                    "sum_proposals_over_C": int(sum(het_dec.proposals)
                                                - total_prbs),
                    "predicted_safe_before_proposal": preds_b,
                    "predicted_safe_after_projection": preds_a,
                    "f_at_a0": f_a0,
                    "f_at_a37": f_a37,
                    "f_at_amax": f_amax,
                    "margin_m_per_slice": [float(d.margin_m) for d in decisions],
                    "margin_applied_per_slice": [int(d.margin_applied) for d in decisions],
                    "sigma_per_slice": [float(d.sigma) for d in decisions],
                    "tau_per_slice": [float(d.tau) for d in decisions],
                    "mode_per_slice": [str(d.mode) for d in decisions],
                    "slice_n_ues": [st.n_e, st.n_u, st.n_m],
                    "slice_backlog": [st.queue_e, st.queue_u, st.queue_m],
                    "mean_field": bool(het.mean_field),
                    "fix_scaler_index": bool(het.fix_scaler_index),
                    "z_dim0_dim1_per_slice": [
                        [round(float(z[0]), 6), round(float(z[1]), 6)]
                        for z in zs
                    ],
                    "z_dim4_dim5_per_slice": [
                        [round(float(z[4]), 6), round(float(z[5]), 6)]
                        for z in zs
                    ],
                    "state": st,
                }
                _het_prev_action = tuple(int(v) for v in proposals)
                _het_prev_decisions = list(decisions)
        elif decision_timer is not None:
            with decision_timer.measure():
                decisions = [controllers[sid].act(states[sid]) for sid in range(num_slices)]
        else:
            decisions = [controllers[sid].act(states[sid]) for sid in range(num_slices)]
        if het is None:
            proposals = [d.action for d in decisions]

        # -- act (projection + CSV happen inside the env) ------------------
        _trace_payload = locals().get("_deferred_trace_kwargs")
        _orig_wait = getattr(env, "_wait_data_availability", None)
        _trace_flushed = [False]
        if _trace_payload is not None and _causal_writer is not None and callable(_orig_wait):
            def _post_release_wait():
                if not _trace_flushed[0]:
                    _trace_flushed[0] = True
                    _causal_writer.record_decision(**_trace_payload)
                return _orig_wait()
            env._wait_data_availability = _post_release_wait
        try:
            obs, reward, terminated, truncated, info = env.step(np.array(proposals, dtype=np.int64))
        finally:
            if _orig_wait is not None and getattr(env, "_wait_data_availability", None) is not _orig_wait:
                env._wait_data_availability = _orig_wait
            if _trace_payload is not None and _causal_writer is not None and not _trace_flushed[0]:
                _trace_flushed[0] = True
                _causal_writer.record_decision(**_trace_payload)
            _deferred_trace_kwargs = None

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
            if het is not None:
                het.set_n_prbs(int(new_c))
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
        if het is not None and _het_pending is not None:
            _het_pending["a_sent_to_mac"] = [
                int(v) for v in proj.get("prb_requested", list(proposals))]
            _het_pending["a_effectively_applied"] = [
                int(v) for v in executed]

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
        if not freeze and het is None:
            for sid in range(num_slices):
                y_t = info.get(f"slice_{sid}", {}).get("p5_thr_mbps")
                if y_t is None:
                    continue  # empty slice: excluded from learning (R19)
                if not getattr(controllers[sid], "LEARNING", True):
                    continue  # static baseline (Fase 4 "equal"): nothing to update
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
        if het is not None and _het_pending is not None:
            last_window_index = max(
                last_window_index, int(_het_pending["state"].window_index)
            )
            last_sim_time_ms = max(
                last_sim_time_ms, int(_het_pending["sim_time_ms"])
            )
        else:
            last_window_index = max(last_window_index, int(step))
            if isinstance(now_ms, (int, float)) and not isinstance(now_ms, bool):
                last_sim_time_ms = max(last_sim_time_ms, int(now_ms))
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
            _save_checkpoint_with_hash(
                checkpoint_path,
                controllers,
                checkpoint_meta,
                horizon={
                    "max_step_index": int(step),
                    "max_window_index": int(last_window_index),
                    "max_sim_time_ms": int(last_sim_time_ms),
                },
            )
        done = terminated or truncated or (max_steps is not None and step >= max_steps)
        if done:
            break

    if _het_jsonl is not None:
        if _het_pending is not None:
            row = {k: v for k, v in _het_pending.items() if k != "state"}
            row["missing_reason"] = "end_of_run"
            _het_jsonl.write(json.dumps(row, default=float) + "\n")
        _het_jsonl.close()
    if _causal_writer is not None:
        _causal_writer.close()

    if checkpoint_path:
        _save_checkpoint_with_hash(
            checkpoint_path,
            controllers,
            checkpoint_meta,
            horizon={
                "max_step_index": int(step),
                "max_window_index": int(last_window_index),
                "max_sim_time_ms": int(last_sim_time_ms),
            },
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
                             "Fase 3 port) on the SAME obs/action contract; "
                             "equal = static baseline, fixed 33.33 percent/split of "
                             "C (Fase 4; dummy-controller pattern); "
                             "prop = demand-proportional sanity baseline "
                             "(Rodada 2 Etapa 2.2; service floors + "
                             "demand-proportional remainder, register R39); "
                             "replay = deterministic single-window override "
                             "(Etapa 2.4 Despacho II-a, oracle §15-§16)")
    parser.add_argument(
        "--replay_base", type=str, default="equal",
        help="(equal | prop | PATH de run B0 com decisions.jsonl — base seq "
             "D2 §4.2): base action outside the override window")
    parser.add_argument("--replay_at_win", type=int, default=0,
                        help="replay ONLY: oracle window index t (0-based; "
                             "window w covers [100w,100(w+1)) ms) whose quota "
                             "is overridden; >= 1 (window 0 precedes any "
                             "decision).  Short runs: simTime ≈ t/10 + 2 s")
    parser.add_argument("--replay_action", type=str, default="",
                        help="replay ONLY: override allocation 'aE,aU,aM' in "
                             "integer RBGs with Σ ≤ C (e.g. '40,12,1'); "
                             "written to the action CSV as 100·a/C percent")
    parser.add_argument(
        "--het_scaler_manifest", default=DEFAULT_HET_SCALER,
        help="frozen §13 feature-scaler manifest (CAL) for --controller sqr_het")
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
    parser.add_argument("--empty_slice_guard", action="store_true",
                        help="A2a isolated guard: zero proposal (and margin) "
                             "in SqrHetController.decide() only when a slice "
                             "has no active UEs, no queue, no offered traffic, "
                             "and no pending pipeline.")
    parser.add_argument("--margin_init", type=float, default=MARGIN_INIT,
                        help="R11 initial uncertainty margin M_0 in PRB/RBG "
                             "(default: 10.0; A3a: 3.0)")
    parser.add_argument("--margin_bounds", type=str,
                        default=f"{MARGIN_BOUNDS[0]},{MARGIN_BOUNDS[1]}",
                        help="R11 uncertainty margin bounds 'M_min,M_max' "
                             "(default: '2.0,50.0'; A3a: '1.0,10.0')")
    parser.add_argument("--mean_field", action="store_true",
                        help="A4 ablation (SQR-HET): restore original paper's "
                             "causal global mean-field features [prb_others_norm, "
                             "total_ues_norm] in dims 4..5 of z_i")
    parser.add_argument("--fix_scaler_index", action="store_true",
                        help="A_scaler isolated variant (SQR-HET): fix "
                             "slice_state() scaler lookup from 'emu'[slice_id] "
                             "to 'eum'[slice_id] so eMBB/URLLC/mMTC use their "
                             "own n_{e,u,m} and queue_{e,u,m} scalers")
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
    # Rodada 2 Etapa 2.2: pass-through of the sqr-sim heterogeneous-scenario
    # flags (registered in scratch/sqr/sqr-sim.cc:2028-2046; contract v1.4
    # §4.2, registers R34/R35).  None/empty = NOT sent — the C++ keeps its
    # own defaults (lambdaUesPerMin/3 per slice, greedy x3, radius 2000 m).
    parser.add_argument("--sliceLambdas", type=str, default=None,
                        dest="slice_lambdas",
                        help="pass-through --sliceLambdas of sqr-sim: "
                             "per-slice Poisson arrival rates in UEs/min, "
                             "comma list of 3 (e.g. 1,1,1)")
    parser.add_argument("--sliceApps", type=str, default=None,
                        dest="slice_apps",
                        help="pass-through --sliceApps of sqr-sim: per-slice "
                             "traffic profile greedy|urllc|mmtc, comma list "
                             "of 3 (e.g. greedy,urllc,mmtc)")
    parser.add_argument("--cellRadius", type=float, default=None,
                        dest="cell_radius",
                        help="pass-through --cellRadius of sqr-sim: cell "
                             "radius for UE placement/walk in meters "
                             "(e.g. 300)")
    parser.add_argument("--urllc_delayed_update", action="store_true",
                        help="TESTE ISOLADO (Emenda candidata — NÃO é a "
                             "canônica): fila de pendentes URLLC; quando o "
                             "pool da janela W fecha ONLINE (delivered==tx, "
                             "ambos disponíveis no instante), uma segunda "
                             "observe_outcome só-URLLC atualiza o modelo "
                             "URLLC com margem completa (R19). Sem dados "
                             "futuros; 1 update por janela; eMBB/mMTC "
                             "intactos.")
    parser.add_argument("--causal_trace_path", type=str, default=None,
                        help="opt-in JSONL(.gz) causal trace per decision "
                             "window for D3-v2/A5 replay (pure read-only, "
                             "zero RNG/state side effects)")
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
    # Rodada 2 Etapa 2.2: optional pass-through of the heterogeneous-scenario
    # flags (absent -> C++ defaults; keys must NOT be popped below — they are
    # registered CommandLine options of sqr-sim).
    if args.slice_lambdas:
        config["sliceLambdas"] = [args.slice_lambdas]
    if args.slice_apps:
        config["sliceApps"] = [args.slice_apps]
    if args.cell_radius is not None:
        config["cellRadius"] = [float(args.cell_radius)]
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
    controllers = build_controllers(args, env)
    margin_bounds = list(parse_margin_bounds(args.margin_bounds))
    scaler_manifest_path = (
        args.het_scaler_manifest if args.controller == "sqr_het" else None
    )
    version_manifest = build_version_manifest(
        scaler_manifest_path=scaler_manifest_path,
    )
    b0_descriptor, b0_config_sha256 = build_b0_config_descriptor(
        controller=args.controller,
        gamma=float(args.gamma),
        c_prbs=int(args.c_prbs),
        virtual_alpha=float(args.virtual_alpha),
        empty_slice_guard=bool(args.empty_slice_guard),
        fix_scaler_index=bool(args.fix_scaler_index),
        mean_field=bool(args.mean_field),
        margin_init=float(args.margin_init),
        margin_bounds=margin_bounds,
        urllc_delayed_update=bool(getattr(args, "urllc_delayed_update", False)),
        warmup_s=float(args.warmup_s),
        scaler_manifest_path=scaler_manifest_path,
    )
    meta = {
        "controller": args.controller,
        "gamma": args.gamma,
        "seed": args.seed,
        "c_prbs": args.c_prbs,
        "slas": slas,
        "virtual_alpha": args.virtual_alpha,
        "empty_slice_guard": bool(args.empty_slice_guard),
        "margin_init": float(args.margin_init),
        "margin_bounds": margin_bounds,
        "mean_field": bool(args.mean_field),
        "fix_scaler_index": bool(args.fix_scaler_index),
        "urllc_delayed_update": bool(getattr(args, "urllc_delayed_update", False)),
        "warmup_s": float(args.warmup_s),
        "freeze": bool(args.freeze),
        "versions": version_manifest,
        "sla_manifest_sha256": version_manifest["sla_manifest_sha256"],
        "scaler_manifest_sha256": version_manifest["scaler_manifest_sha256"],
        "config_descriptor": b0_descriptor,
        "config_sha256": b0_config_sha256,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    checkpoint_path = args.checkpoint_path or os.path.join(
        args.output, "sqr_checkpoint.pkl"
    )
    if args.resume:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"--resume: checkpoint not found at {checkpoint_path}"
            )
        ckpt_payload = load_checkpoint(checkpoint_path)
        restored_meta = verify_resume_contract(ckpt_payload, meta, controllers)
        restore_controllers(ckpt_payload, controllers)
        print(f"[sqr_run] resumed controllers from {checkpoint_path} (meta={restored_meta})")

    step_logger = StepMetricsLogger(os.path.join(args.output, args.step_log_file))
    if args.controller == "replay":
        # Replay provenance (Etapa 2.4): the override plan row this run
        # executes (grupo/plan coordinates land in the analysis join).
        meta["replay"] = {
            "replay_base": args.replay_base,
            "replay_at_win": int(args.replay_at_win),
            "replay_action": args.replay_action,
        }
        total_steps = int((args.simTime - 0.5) * 1000 / args.periodMs)
        if args.replay_at_win >= total_steps:
            print(
                f"[sqr_run] WARNING: replay_at_win={args.replay_at_win} is "
                f"beyond the ~{total_steps} windows of simTime="
                f"{args.simTime}s — the override would never be applied; "
                "raise --simTime (≈ t/10 + 2 s)."
            )

    het = het_runtime = het_bootstrap = None
    if args.controller == "sqr_het":
        from nsoran.sqr_jc.feature_scaler import FeatureScaler
        from nsoran.sqr_jc.het_runtime import HetRuntime
        from nsoran.sqr_jc.sqr_het import SqrHetController

        scaler = FeatureScaler.load_manifest(args.het_scaler_manifest)
        het = SqrHetController(
            tuple(controllers[sid] for sid in range(args.num_slices)),
            scaler,
            args.c_prbs,
            virtual_alpha=args.virtual_alpha,
            empty_slice_guard=bool(args.empty_slice_guard),
            mean_field=bool(args.mean_field),
            fix_scaler_index=bool(args.fix_scaler_index),
        )
        # HetRuntime construído LAZY dentro de run_episode (após env.reset,
        # quando env.sim_path existe). Incidente 11/11: construído aqui com
        # sim_path="" o glob do delay stream era relativo ao CWD do GYM e
        # casou o sqr_packet_delays.csv do A_lam2_s1 (Rodada 2) para TODOS
        # os 9 runs — URLLC dos estados/rótulos contaminada (SIM_BUG §5.9).
        het_runtime = None
        # §3.4: bootstrap-PROP compartilhado 0–warmup_s (decisões prop R39;
        # learner observa; B0/B1 partem idênticos)
        from nsoran.sqr.static_controllers import PropDemandController
        builder = getattr(env, "builder", None)
        window_provider = ((lambda b: lambda: b.last_window)(builder)
                           if builder is not None else None)
        het_bootstrap = {
            sid: PropDemandController(
                slice_id=sid, n_prbs=args.c_prbs, num_slices=args.num_slices,
                window_provider=window_provider)
            for sid in range(args.num_slices)}

    ep_checkpoint_path = (
        None if (args.freeze and args.resume) else checkpoint_path
    )
    total_kpm_missing = 0
    _lambda_total_meta = (
        float(sum(
            float(v) for v in
            (getattr(args, "sliceLambdas", None)
             or getattr(args, "slice_lambdas", None)
             or "").split(",")))
        if (getattr(args, "sliceLambdas", None)
            or getattr(args, "slice_lambdas", None))
        else None
    )
    _run_id_meta = os.path.basename(os.path.normpath(args.output))
    for episode in range(1, args.episodes + 1):
        summary = run_episode(
            env,
            controllers,
            het=het,
            het_runtime=het_runtime,
            het_bootstrap=het_bootstrap,
            decision_jsonl=os.path.join(args.output, "decisions.jsonl"),
            het_meta={"run_id": _run_id_meta,
                      "lambda_total": _lambda_total_meta},
            gamma=float(args.gamma),
            virtual_alpha=args.virtual_alpha,
            freeze=args.freeze,
            warmup_steps=warmup_steps,
            seed=args.seed,
            episode=episode,
            step_logger=step_logger,
            decision_timer=accounting.decision_timer,
            checkpoint_path=ep_checkpoint_path,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_meta=meta,
            urllc_delayed_update=bool(getattr(args, "urllc_delayed_update",
                                              False)),
            causal_trace_path=getattr(args, "causal_trace_path", None),
            causal_trace_meta={
                "run_id": _run_id_meta,
                "lambda_total": _lambda_total_meta,
                "seed": int(args.seed),
                "c_prbs": int(args.c_prbs),
                "versions": version_manifest,
                "config_descriptor": b0_descriptor,
                "config_sha256": b0_config_sha256,
            },
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
