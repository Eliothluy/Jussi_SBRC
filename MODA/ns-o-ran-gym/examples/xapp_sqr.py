"""SQR xApp over the E2 control path (Fase 5.4 — PLANO §3-Fase 5.4).

The SAME SQR decision stack of ``examples/sqr_run.py`` (per-slice
``SqrSliceController`` + deterministic projection Pi + online y_t learning,
register R22) running as an O-RAN **xApp**: a separate RIC-side process that

    1. receives the E2SM-KPM indication of each 100 ms decision window,
    2. rebuilds the FROZEN contract §4 observation from it,
    3. decides with SqrController + Pi (identical seeds -> identical
       decisions to the IPC path — that identity is the PARITY gate of
       ``examples/run_sqr_e2_parity.sh``),
    4. answers with a RIC Control (RAN slicing style) carrying the PRB
       quotas.

Transport (DEGRADED vs FULL — decision logic is transport-agnostic):

- **DEGRADED (this machine)**: contrib/nori has no e2sim/SCTP, so the E2
  endpoint of ``sqr-sim`` (``--e2Endpoint=xapp``, ``scratch/sqr/
  sqr-e2-adapter.cc``) bridges each indication over a LOCAL UDP JSON
  datagram to THIS process (documented local transport of the DEGRADED
  mode). The xApp binds the UDP socket BEFORE spawning the ns-3 scenario
  and answers one RIC Control JSON per indication, same ``seq``. The 50 ms
  wall budget per window is the contract §5 budget (the same
  ``sem_timedwait(50 ms)`` of the IPC path); a timeout keeps the previous
  quota and is countable on the ns-3 side.
- **FULL (future, Fase 5.1)**: the same receive→decide→reply loop is
  driven by the e2sim/SCTP transport of the Near-RT RIC; only the
  transport object changes (``UdpLoopbackTransport`` vs the documented
  ``SctpE2simTransport`` stub below) — the decision core is untouched.

RAN-slicing quantization (transport divergence handling): the RIC Control
quotas ride on ``ns3::SlicePRBQuota`` whose ``dedicatePRBRatio`` is an ASN.1
INTEGER percent (0..100) — fractions like ``100*a_i/53`` are NOT expressible
(same constraint registered in Fase 5.3). The xApp therefore searches for
INTEGER percentages that reproduce EXACTLY the RBG quota vector the IPC
path would apply (replica of ``SqrMacScheduler::ComputeRbgQuotas``,
sqr-mac-scheduler.cc:57) and verifies the replica before sending. A window
where no integer vector reproduces the target is counted
(``quant_exact=0``) and sent best-effort — reported by the parity script.

Window alignment vs the IPC driver: the IPC loop logs iteration k's
``prb_scheduled`` from the telemetry row of boundary k+1 (its ``step``
blocks until the next boundary). The xApp defers its step_metrics row
flush to the NEXT indication so the SAME alignment holds; decisions
(timestamp, raw/executed quotas) are logged at decision time.

Usage (the parity script is the canonical entry point):
    python examples/xapp_sqr.py --gamma 1 --simTime 120 --seed 1 \
        --output results/sqr_e2_xapp --xappPort 5560
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sqr_run  # reuses build_controllers/parse_slas/loggers — SAME stack (also bootstraps src/)

import argparse
import csv
import itertools
import json
import math
import socket
import subprocess
import time

import numpy as np

from nsoran.compute_accounting import DecisionTimer
from nsoran.sqr.coordinator import apply_virtual_updates, ProjectionResult, VirtualUpdate
from nsoran.sqr.sqr_env import SqrObservationBuilder, read_effective_total_prbs
from nsoran.sqr.sqr_kpis import SQR_KPM_COLUMNS

DEFAULT_NS3_BIN = os.path.join(
    os.path.dirname(__file__), "..", "..", "ns-3-dev", "build", "scratch", "sqr",
    "ns3.46-sqr-sim-default",
)
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "results", "sqr_e2_xapp")

#: SQR-internal RAN Function ID of the control reply (E2SM-RC, Fase 5.3).
RAN_FUNCTION_ID = 300
REQUEST_TYPE_RAN_SLICING = 1003

XAPP_DECISION_COLUMNS = [
    "seq",
    "timestamp_ms",
    "kpm_ues",
    "total_rbg",
    "total_prbs_c",
    "mode",
    "raw_action_0",
    "raw_action_1",
    "raw_action_2",
    "action_0",
    "action_1",
    "action_2",
    "prb_executed_0",
    "prb_executed_1",
    "prb_executed_2",
    "pct_sent_0",
    "pct_sent_1",
    "pct_sent_2",
    "target_rbg_0",
    "target_rbg_1",
    "target_rbg_2",
    "quant_exact",
    "decision_ms",
    "handle_ms",
    "frozen_echo",
]


# ---------------------------------------------------------------------------
# C++ replica + integer-percent search (transport divergence handling)
# ---------------------------------------------------------------------------


def compute_rbg_quotas_py(pct, total_rbg: int, num_slices: int = 3) -> list[int]:
    """Exact replica of ``SqrMacScheduler::ComputeRbgQuotas``.

    Source of truth: ``scratch/sqr/sqr-mac-scheduler.cc:57-129`` —
    ``raw_i = clamp(pct_i,0,100)/100 * totalRbg``; ``target = llround(Σ raw)``
    clamped to totalRbg (llround on non-negative x = floor(x+0.5)); quotas
    start at ``floor(raw_i)`` and the ``target − Σ floor`` surplus is
    distributed 1 RBG per slice by LARGEST fractional remainder, ties to the
    LOWEST slice index (strict ``>`` comparison). No renormalization (R27).
    """
    raw = []
    for i in range(num_slices):
        p = float(pct[i]) if i < len(pct) else 0.0
        p = min(max(p, 0.0), 100.0)
        raw.append(p / 100.0 * float(total_rbg))
    # Naive left fold — MUST match the C++ accumulation (`rawSum += raw[i]`,
    # sqr-mac-scheduler.cc:87). Python's builtin sum() is compensated
    # (Neumaier) since 3.12 and can differ in the last ulp, which flips the
    # llround boundary (real case: (2,12,36)@53 -> 26.499999999999996 vs
    # 26.5 -> target 26 vs 27 -> quota 6 vs 7).
    raw_sum = 0.0
    for r in raw:
        raw_sum += r
    target = int(math.floor(raw_sum + 0.5))  # llround, values are >= 0
    if target > total_rbg:
        target = int(total_rbg)
    quotas: list[int] = []
    fracs: list[float] = []
    base = 0
    for r in raw:
        f = int(math.floor(r))
        quotas.append(f)
        fracs.append(r - f)
        base += f
    extra = target - base
    while extra > 0:
        best = -1
        for i, fr in enumerate(fracs):
            if fr >= 0.0 and (best < 0 or fr > fracs[best]):
                best = i
        if best < 0:
            break  # defensive; unreachable in the C++ too
        quotas[best] += 1
        fracs[best] = -1.0
        extra -= 1
    return quotas


def integer_pcts_for_quotas(
    pct_float: list[float], total_rbg: int, num_slices: int = 3, max_radius: int = 4
) -> tuple[list[int], list[int], bool]:
    """Integer percents that reproduce the float-percent RBG quotas exactly.

    Primary candidate (constructive): ``ceil(100*Q_i/totalRbg)`` puts every
    ``raw'_i`` inside ``[Q_i, Q_i + totalRbg/100)`` — with the primary grid
    (totalRbg=13, L=3) the summed slack stays < 0.5, so the replica target
    and the largest-remainder assignment are preserved exactly. If that (or
    the plain rounding) misses — possible for larger grids where the slack
    can exceed 0.5 — a deterministic local search around ``round(pct)``
    runs; failure is REPORTED (``exact=False``) and sent best-effort.
    """
    target = compute_rbg_quotas_py(pct_float, total_rbg, num_slices)
    ceil_base = [min(int(math.ceil(100.0 * q / float(total_rbg))), 100) for q in target]
    if compute_rbg_quotas_py(ceil_base, total_rbg, num_slices) == target:
        return ceil_base, target, True
    round_base = [min(max(int(math.floor(p + 0.5)), 0), 100) for p in pct_float]
    if compute_rbg_quotas_py(round_base, total_rbg, num_slices) == target:
        return round_base, target, True
    for radius in range(1, max_radius + 1):
        rng = range(-radius, radius + 1)
        for offs in itertools.product(rng, repeat=num_slices):
            cand = [
                min(max(round_base[i] + offs[i], 0), 100) for i in range(num_slices)
            ]
            if compute_rbg_quotas_py(cand, total_rbg, num_slices) == target:
                return cand, target, True
    return round_base, target, False


# ---------------------------------------------------------------------------
# Indication → contract §4 KPM materialization
# ---------------------------------------------------------------------------


def write_kpm_window(path: str, ind: dict) -> int:
    """Write one indication's UE records in the EXACT sqr-kpms.txt format.

    Column formatting mirrors ``sqr-sim.cc`` ``emitRow`` (plr/share %.2f,
    throughputs %.4f, snr %.2f) so ``SqrKpmParser`` sees byte-identical
    values on both paths. ``ueImsi`` carries the record IMSI — an opaque
    per-UE key for the parser (SNR-imputation carry); the observation and
    all per-slice derivations are key-independent, so parity holds.
    """
    rows = []
    for rec in ind.get("ue_records", []):
        (
            imsi, rnti, slice_id, d_tx, d_rx, plr, rsh, d_lost,
            thr, buf, wthr, snr, queue,
        ) = rec
        rows.append(
            f"{int(ind['timestamp_ms'])},{int(imsi)},{int(slice_id)},"
            f"{int(d_tx)},{int(d_rx)},{plr:.2f},{rsh:.2f},{int(d_lost)},"
            f"{thr:.4f},{int(buf)},{wthr:.4f},{snr:.2f},{int(queue)}"
        )
    with open(path, "w", newline="") as f:
        f.write(",".join(SQR_KPM_COLUMNS) + "\n")
        for r in rows:
            f.write(r + "\n")
    return len(rows)


def window_info(window, num_slices: int = 3) -> dict:
    """Rebuild the ``SqrEnv._build_info`` slice fields from a parsed window."""
    info: dict = {"kpm_timestamp": window.timestamp, "total_ues": window.total_ues}
    for sid in range(num_slices):
        k = window.slices.get(sid)
        info[f"slice_{sid}"] = {
            "n_ues": k.n_ues if k else 0,
            "p5_thr_mbps": k.p5_thr_mbps if k else None,
            "lowest_thr_mbps": k.lowest_thr_mbps if k else None,
            "mean_thr_mbps": k.mean_thr_mbps if k else None,
            "queue_bytes_sum": k.queue_bytes_sum if k else 0.0,
            "violation_rate": k.violation_rate if k else None,
            "n_snr_imputed": k.n_snr_imputed if k else 0,
            "snr_imputations": dict(k.snr_imputations) if k else {},
        }
    rate = pooled_violation_rate(window, num_slices)
    info["pooled_violation_rate"] = rate
    info["reward"] = 0.0 if rate is None else -float(rate)  # SqrEnv._compute_reward
    return info


def pooled_violation_rate(window, num_slices: int = 3) -> float | None:
    """R19/R26 pooled per-UE violation rate (slices with UEs only)."""
    num = 0
    den = 0
    for sid in range(num_slices):
        k = window.slices.get(sid)
        if k is None or k.n_ues == 0 or k.violation_rate is None:
            continue
        num += k.violation_rate * k.n_ues
        den += k.n_ues
    return None if den == 0 else float(num / den)


# ---------------------------------------------------------------------------
# Transport abstraction (DEGRADED concrete; FULL documented stub)
# ---------------------------------------------------------------------------


class UdpLoopbackTransport:
    """DEGRADED local transport: UDP JSON datagrams on localhost.

    Peer: ``SqrE2LoopbackTerm::HandleIndicationXappUdp`` (scratch/sqr/
    sqr-e2-adapter.cc). The xApp binds FIRST; sqr-sim is spawned afterwards
    so no indication is ever lost. ``seq`` deduplication makes the bridge's
    same-seq retries idempotent (a duplicate indication is answered with the
    cached control — it never re-decides and never re-learns).
    """

    def __init__(self, address: str, port: int, recv_timeout_s: float = 1.0):
        self._addr = (address, int(port))
        self._recv_timeout_s = float(recv_timeout_s)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(self._addr)
        self._sock.settimeout(self._recv_timeout_s)
        self._peer = None  # gNB bridge endpoint (ephemeral port of the sender)
        self._last_seq = -1
        self._last_reply: bytes | None = None

    @property
    def bound_port(self) -> int:
        return self._addr[1]

    def recv_indication(self) -> dict | None:
        """Next KPM indication (None on idle timeout — caller polls the sim)."""
        try:
            data, peer = self._sock.recvfrom(65536)
        except socket.timeout:
            return None
        self._peer = peer  # replies go to the SENDER, not to our bound addr
        ind = json.loads(data.decode("utf-8"))
        if ind.get("type") != "kpm_indication":
            return None
        seq = int(ind.get("seq", -1))
        if seq == self._last_seq:
            # Bridge retry of the same window: re-send the cached control.
            if self._last_reply is not None and self._peer is not None:
                self._sock.sendto(self._last_reply, self._peer)
            ind["duplicate"] = True
            return ind
        ind["duplicate"] = False
        return ind

    def send_ric_control(
        self, seq: int, pcts: list[int], decision_ms: float, frozen: bool = False
    ) -> int:
        if self._peer is None:
            raise RuntimeError("no KPM indication received yet — no bridge peer")
        reply = {
            "type": "ric_control",
            "seq": int(seq),
            "ran_function_id": RAN_FUNCTION_ID,
            "request_type": REQUEST_TYPE_RAN_SLICING,
            "ric_style_type": 1,
            "action_id": 1,
            "decision": 0,  # accept (nori convention)
            "prb_quotas": [
                {
                    "slice_id": sid,  # SST = slice index (Fase 5.3 mapping)
                    "dedicate_prb_ratio": int(pcts[sid]),
                    "min_prb_ratio": int(pcts[sid]),
                    "max_prb_ratio": int(pcts[sid]),
                }
                for sid in range(len(pcts))
            ],
            "decision_ms": float(decision_ms),
            "frozen_echo": bool(frozen),
        }
        # Compact wire format (no spaces) — the C++ side parses it, but the
        # parser is whitespace-tolerant; compact keeps the datagram small.
        payload = json.dumps(reply, separators=(",", ":")).encode("utf-8")
        self._sock.sendto(payload, self._peer)
        self._last_seq = int(seq)
        self._last_reply = payload
        return len(payload)


class SctpE2simTransport:
    """FULL-mode transport stub (Fase 5.1) — deliberately not implemented.

    The e2sim/ASN.1/SCTP stack does not exist on this machine (MODA-PORT §4,
    same caveat as Fase 5.3 §4). In FULL the xApp subscribes via E2SM-KPM and
    receives ``RicControlMessage`` requests over SCTP; the DECISION CORE of
    this script (observation builder, controllers, projection, integer-quota
    verification) is transport-agnostic and is meant to be reused unchanged
    behind this interface.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FULL mode requires the e2sim/SCTP stack (Fase 5.1). "
            "Use --transport udp_loopback (DEGRADED local bridge)."
        )


TRANSPORTS = {
    "udp_loopback": UdpLoopbackTransport,
    "sctp_e2sim": SctpE2simTransport,
}


# ---------------------------------------------------------------------------
# xApp decision/latency log
# ---------------------------------------------------------------------------


class XappDecisionLogger:
    """One row per indication in xapp_decisions.csv (latency + quotas)."""

    def __init__(self, path: str):
        self.path = path
        self._initialized = False

    def write(self, rows: list[list]) -> None:
        if not self._initialized:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(XAPP_DECISION_COLUMNS)
            self._initialized = True
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerows(rows)


def spawn_sqr_sim(args, sim_path: str):
    """Spawn the ns-3 scenario with --controlPath=e2 --e2Endpoint=xapp.

    Mirrors ``NsOranEnv.start_sim`` (direct binary + LD_LIBRARY_PATH,
    cwd = run dir) WITHOUT simId: the E2 path has no semaphores/CSV
    (sqr-sim ignores --simId with controlPath=e2, Fase 5.3).
    """
    os.makedirs(sim_path, exist_ok=True)
    ns3_path = os.path.abspath(args.ns3_path)
    env = {
        "LD_LIBRARY_PATH": f"{ns3_path}/build:{ns3_path}/build/lib",
        "DYLD_LIBRARY_PATH": f"{ns3_path}/build:{ns3_path}/build/lib",
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }
    command = [
        os.path.abspath(args.ns3_bin),
        f"--simTime={args.simTime}",
        f"--seed={args.seed}",
        "--run=1",
        f"--periodMs={args.periodMs}",
        "--controlPath=e2",
        "--e2Endpoint=xapp",
        f"--e2XappPort={args.xappPort}",
        f"--e2XappTimeoutMs={args.e2XappTimeoutMs}",
        f"--e2XappRetries={args.e2XappRetries}",
    ]
    stdout_f = open(os.path.join(sim_path, "stdout"), "w")
    stderr_f = open(os.path.join(sim_path, "stderr"), "w")
    print(f"[xapp_sqr] ns-3 command: {' '.join(command)}")
    proc = subprocess.Popen(
        command, cwd=sim_path, env=env, stdout=stdout_f, stderr=stderr_f
    )
    return proc, (stdout_f, stderr_f)


def _p_stats(samples: list[float]) -> dict:
    """p50/p95/max of a latency sample list (p95 = nearest-rank, as xApp KPI)."""
    if not samples:
        return {"n": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    s = sorted(samples)

    def p95() -> float:
        idx = max(math.ceil(0.95 * len(s)) - 1, 0)
        return round(s[idx], 4)

    return {
        "n": len(s),
        "p50_ms": round(float(np.median(s)), 4),
        "p95_ms": p95(),
        "max_ms": round(s[-1], 4),
    }


# ---------------------------------------------------------------------------
# xApp control loop
# ---------------------------------------------------------------------------


def run_xapp(args) -> dict:
    """xApp setup + supervised session (never leaks the ns-3 child)."""
    num_slices = args.num_slices
    slas = sqr_run.parse_slas(args.slas, num_slices)
    controllers = sqr_run.build_controllers(args)

    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    sim_path = os.path.join(out_dir, f"e2_seed={args.seed}")
    os.makedirs(sim_path, exist_ok=True)

    transport = TRANSPORTS[args.transport](args.xappAddress, args.xappPort)
    print(
        f"[xapp_sqr] transport={args.transport} bound on "
        f"{args.xappAddress}:{transport.bound_port}"
    )

    proc, files = spawn_sqr_sim(args, sim_path)
    try:
        return _xapp_session(args, proc, files, transport, controllers, slas,
                             out_dir, sim_path)
    except BaseException:
        # Never leak the ns-3 child: an orphan sqr-sim keeps sending
        # indications to the port and corrupts the next run that binds it.
        proc.kill()
        try:
            proc.wait()
        except OSError:
            pass
        raise


def _xapp_session(args, proc, files, transport, controllers, slas, out_dir,
                  sim_path) -> dict:
    """xApp main loop: indication → flush/learn → decide → RIC Control."""
    num_slices = args.num_slices

    builder = SqrObservationBuilder(
        num_slices=num_slices, total_prbs=args.c_prbs, sla_thresholds=slas
    )
    kpm_tmp = os.path.join(out_dir, "e2_kpm_indication.csv")
    step_logger = sqr_run.StepMetricsLogger(os.path.join(out_dir, args.step_log_file))
    xapp_logger = XappDecisionLogger(os.path.join(out_dir, "xapp_decisions.csv"))
    telemetry = sqr_run.Ns3PrbWindowTelemetry(sim_path, num_slices=num_slices)

    max_steps = int(round((args.simTime - args.appStart) * 1000 / args.periodMs))
    warmup_steps = int(round(args.warmup_s * 1000.0 / args.periodMs))
    decision_timer = DecisionTimer(period_ms=args.periodMs)
    handle_ms_samples: list[float] = []

    total_prbs = int(args.c_prbs)
    n_virtual = 0
    n_learns = 0
    n_quant_divergence = 0
    n_frozen = 0
    step = 0
    prev = None          # (states z_k, executed a_final_k, proj_k, k): learns at k+1
    pending = None       # step_metrics rows of iteration k, completed at k+1
    last_pcts: list[int] = [0] * num_slices

    #: step_metrics columns completed at the NEXT boundary (the IPC driver
    #: fills them from the window its step BLOCKED on = window k+1): the KPM
    #: derivations, the reward and the kpm timestamp (watermark 0 while no UE
    #: has ever appeared — identical on both paths).
    DEFERRED_COLS = {2, 17, 18, 19, 20, 21, 26}

    def flush_pending(window_info_next: dict | None, telemetry_row: dict | None):
        nonlocal pending
        if pending is None:
            return
        rows = pending
        for r in rows:
            sid = int(r[5])  # slice_id column of step_metrics
            r[11] = (telemetry_row or {}).get("prb_scheduled", {}).get(str(sid), "")
            if window_info_next is not None:
                k = window_info_next.get(f"slice_{sid}", {})
                r[2] = window_info_next.get("kpm_timestamp", "")
                r[17] = "" if k.get("p5_thr_mbps") is None else f"{k['p5_thr_mbps']:.6f}"
                r[18] = "" if k.get("lowest_thr_mbps") is None else f"{k['lowest_thr_mbps']:.6f}"
                r[19] = k.get("n_ues", 0)
                r[20] = "" if k.get("violation_rate") is None else f"{k['violation_rate']:.6f}"
                r[21] = k.get("n_snr_imputed", 0)
                r[26] = f"{window_info_next.get('reward', 0.0):.6f}"
        step_logger.write(rows)
        pending = None

    while True:
        if proc.poll() is not None:
            break
        ind = transport.recv_indication()
        if ind is None:
            continue
        if ind.get("duplicate"):
            continue  # cached control already re-sent; never re-decide/re-learn
        t_handle0 = time.perf_counter()
        seq = int(ind["seq"])
        ts = int(ind["timestamp_ms"])
        total_rbg = int(ind.get("total_rbg", 0))

        # (0b/1) parse the indication into the FROZEN contract §4 window
        #        (the SNR imputation state lives in the builder, as in
        #        SqrEnv) — the pending rows are flushed right after with
        #        THIS window's KPI columns (window k+1 of iteration k).
        n_ues = write_kpm_window(kpm_tmp, ind)
        new_c = read_effective_total_prbs(sim_path)
        if isinstance(new_c, int) and new_c > 0 and new_c != total_prbs:
            total_prbs = int(new_c)
            builder.total_prbs = total_prbs
            for ctl in controllers.values():
                setter = getattr(ctl, "set_n_prbs", None)
                if setter is not None:  # KBRL rescales its accuracy table
                    setter(int(new_c))
                else:                   # SQR: plain attribute sync
                    ctl.n_prbs = int(new_c)
        window = builder.update_from_kpm(kpm_tmp)
        info = window_info(window, num_slices)

        # flush AFTER the parse: the pending rows take THIS window's KPI
        # columns (window k+1 from the point of view of iteration k).
        if pending is not None:
            flush_pending(info, telemetry.latest())

        # (2) learn the PREVIOUS decision with the realized y_t of THIS
        #     window (register R22: (z at decision, executed a_final, y_t)).
        #     Runs BEFORE act — exactly the IPC iteration order (learn at the
        #     end of iteration k, act at the start of iteration k+1).
        if prev is not None and prev[3] <= max_steps and not args.freeze:
            states_prev, executed_prev, proj_prev = prev[0], prev[1], prev[2]
            for ctl in controllers.values():
                hook = getattr(ctl, "notify_projection", None)
                if hook is not None:
                    hook(bool(proj_prev.over_capacity))
            learned_any = False
            for sid in range(num_slices):
                y_t = info.get(f"slice_{sid}", {}).get("p5_thr_mbps")
                if y_t is None:
                    continue  # empty slice: excluded from learning (R19)
                if not getattr(controllers[sid], "LEARNING", True):
                    continue  # static baseline (equal): nothing to update
                controllers[sid].observe(
                    states_prev[sid], int(executed_prev[sid]), float(y_t)
                )
                learned_any = True
            if learned_any:
                n_learns += 1
            if args.virtual_alpha > 0 and all(
                hasattr(c, "insert_virtual_support") for c in controllers.values()
            ):
                # proj_prev.virtual_updates already carries VirtualUpdate
                # objects (same as sqr_run's _projection_from_info rebuild).
                updates = tuple(proj_prev.virtual_updates)
                view = ProjectionResult(
                    allocated=np.asarray(proj_prev.allocated, dtype=np.int64),
                    proposals=np.asarray(proj_prev.proposals, dtype=np.int64),
                    total_prbs=total_prbs,
                    cut=np.asarray(proj_prev.cut, dtype=bool),
                    virtual_updates=updates,
                )
                apply_virtual_updates(
                    view, list(states_prev), controllers, alpha=args.virtual_alpha
                )
                n_virtual += len(updates)

        # (3) decide (timed with the SAME DecisionTimer of sqr_run) + project
        #     + integer-percent verification + RIC Control reply.
        if seq <= max_steps:
            obs = builder.build_obs()  # float32 matrix, identical to _get_obs
            states = [np.asarray(obs[i], dtype=np.float64) for i in range(num_slices)]
            with decision_timer.measure():
                decisions = [
                    controllers[sid].act(states[sid]) for sid in range(num_slices)
                ]
            proposals = [d.action for d in decisions]
            actions, proj = builder.compute_control_tuples(
                np.array(proposals, dtype=np.int64)
            )
            executed = [int(a) for a in proj.allocated]
            pct_float = [100.0 * a / float(total_prbs) for a in executed]
            pcts, target_q, quant_exact = integer_pcts_for_quotas(
                pct_float, total_rbg, num_slices
            )
            if not quant_exact:
                n_quant_divergence += 1
            decision_ms = (
                decision_timer.samples_ms[-1] if decision_timer.samples_ms else 0.0
            )
            transport.send_ric_control(seq, pcts, decision_ms)
            last_pcts = list(pcts)
            # Queue the step_metrics rows: the KPI/reward/timestamp columns are
            # completed at the NEXT boundary with the window this action
            # governs (IPC row semantics — see DEFERRED_COLS).
            rows = []
            for sid in range(num_slices):
                d = decisions[sid]
                rows.append([
                    1, seq, "", args.seed, float(args.gamma), sid, d.mode,
                    d.raw_action, d.action,
                    int(proj.proposals[sid]), int(executed[sid]),
                    "",  # prb_scheduled: filled at the next boundary
                    total_prbs - int(np.sum(executed)),
                    f"{d.sigma:.6f}", f"{d.margin_m:.3f}", d.margin_applied,
                    f"{d.tau:.5f}",
                    "", "", 0, "", 0,  # y/lowest/n_ues/viol/snr_imp: deferred
                    0,  # kpm_missing: the E2 indication IS the KPM
                    int(not args.freeze),
                    0,  # virtual_updates counted at learn time (next window)
                    f"{decision_ms:.4f}", "",  # reward: deferred
                    int(seq <= warmup_steps), False, seq == max_steps,
                ])
            pending = rows
            xapp_row = [
                seq, ts, n_ues, total_rbg, total_prbs, decisions[0].mode,
                decisions[0].raw_action, decisions[1].raw_action,
                decisions[2].raw_action,
                decisions[0].action, decisions[1].action, decisions[2].action,
                executed[0], executed[1], executed[2],
                pcts[0], pcts[1], pcts[2],
                target_q[0], target_q[1], target_q[2],
                int(quant_exact), f"{decision_ms:.4f}", "", 0,
            ]
            prev = (states, executed, proj, seq)
            step = seq
        else:
            # Frozen echo (beyond max_steps): the IPC driver has already
            # exited its loop and sqr-sim's 50 ms timeout keeps the LAST
            # quota — echoing the last control is the exact E2 counterpart.
            transport.send_ric_control(seq, last_pcts, 0.0, frozen=True)
            n_frozen += 1
            xapp_row = [
                seq, ts, n_ues, total_rbg, total_prbs, "frozen_echo",
                "", "", "", "", "", "", "", "", "",
                last_pcts[0], last_pcts[1], last_pcts[2],
                "", "", "", 1, "0.0", "", 1,
            ]
        handle_ms = (time.perf_counter() - t_handle0) * 1000.0
        handle_ms_samples.append(handle_ms)
        xapp_row[-2] = f"{handle_ms:.4f}"
        xapp_logger.write([xapp_row])
        if seq >= max_steps + 1000:  # pure backstop; the loop exits with the sim
            break

    # Final flush with the last telemetry row and the LAST parsed window
    # (post-loop there is no next boundary; the columns keep the last values).
    if pending is not None:
        flush_pending(info, telemetry.latest())

    deadline = time.time() + 300.0
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.2)
    exit_code = proc.poll()
    for f in files:
        try:
            f.close()
        except OSError:
            pass

    if args.checkpoint_path:
        from nsoran.sqr.checkpoint import build_checkpoint, save_checkpoint

        save_checkpoint(
            args.checkpoint_path,
            build_checkpoint(
                controllers,
                meta={
                    "controller": args.controller,
                    "gamma": args.gamma,
                    "seed": args.seed,
                    "c_prbs": total_prbs,
                    "slas": slas,
                    "virtual_alpha": args.virtual_alpha,
                    "freeze": bool(args.freeze),
                    "transport": args.transport,
                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            ),
        )

    summary = {
        "steps": step,
        "learns": n_learns,
        "virtual_updates": n_virtual,
        "quant_divergences": n_quant_divergence,
        "frozen_echoes": n_frozen,
        "total_prbs_final": total_prbs,
        "ns3_exit_code": exit_code,
        "sim_path": sim_path,
        "decision": decision_timer.summary(),
        "handle_ms": _p_stats(handle_ms_samples),
        "checkpoint": args.checkpoint_path,
    }
    print(f"[xapp_sqr] summary: {json.dumps(summary, default=str)}")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="SQR xApp over the E2 path (Fase 5.4): KPM indication -> "
                    "SqrController + Pi -> RIC Control (DEGRADED UDP bridge)"
    )
    parser.add_argument("--controller", type=str, default="sqr",
                        choices=sqr_run.CONTROLLER_CHOICES)
    parser.add_argument("--gamma", type=int, default=1, choices=(0, 1))
    parser.add_argument("--ns3_path", type=str, default=sqr_run.DEFAULT_NS3_PATH)
    parser.add_argument("--ns3_bin", type=str, default=DEFAULT_NS3_BIN)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--simTime", type=float, default=120.0)
    parser.add_argument("--appStart", type=float, default=0.5)
    parser.add_argument("--periodMs", type=int, default=100)
    parser.add_argument("--c_prbs", type=int, default=52)
    parser.add_argument("--num_slices", type=int, default=3)
    parser.add_argument("--slas", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--violation_window", type=int,
                        default=sqr_run.VIOLATION_WINDOW)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--virtual_alpha", type=float,
                        default=sqr_run.DEFAULT_VIRTUAL_ALPHA)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--warmup_s", type=float, default=100.0)
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--step_log_file", type=str, default="step_metrics.csv")
    parser.add_argument("--transport", type=str, default="udp_loopback",
                        choices=tuple(TRANSPORTS),
                        help="udp_loopback = DEGRADED local bridge; "
                             "sctp_e2sim = FULL (requires e2sim, Fase 5.1)")
    parser.add_argument("--xappAddress", type=str, default="127.0.0.1")
    parser.add_argument("--xappPort", type=int, default=5560)
    parser.add_argument("--e2XappTimeoutMs", type=int, default=50,
                        help="bridge wall budget per attempt (contract §5; "
                             "must match the sqr-sim --e2XappTimeoutMs)")
    parser.add_argument("--e2XappRetries", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.checkpoint_path == "":
        args.checkpoint_path = os.path.join(
            os.path.abspath(args.output), "sqr_xapp_checkpoint.pkl"
        )
    return run_xapp(args)


if __name__ == "__main__":
    main()
