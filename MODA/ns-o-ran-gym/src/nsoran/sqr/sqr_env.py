"""Gymnasium environment for the SQR scenario (IPC contract, 100 ms windows).

Follows the ``RslaqEnv`` pattern (NsOranEnv subclass) but implements the
FROZEN SQR contract (``MODA/docs/sqr/contrato_interface.md``):

- **KPM**: ``sqr-kpms.txt``, 13 columns (contract §4), parsed by
  :class:`nsoran.sqr.sqr_kpis.SqrKpmParser` — the SNR sentinel −999 never
  reaches the observation (deterministic imputation, register R09).
- **Observation**: shape ``(num_slices, 6)`` float32; row i is the local
  state z_i of register R20 = ``[lowest_thr_norm, n_ues_norm, queue_norm,
  snr_norm, prb_outros_norm, total_ues_norm]`` — 4 per-slice features plus
  the 2 global ones (PRBs of the OTHER slices over C, cell UE count), each
  in [0, 1] with the frozen R09 scales.  y_t (p5 per slice) rides in
  ``info``, it is NOT an observation input.
- **Action**: integer PRBs per slice, ``a_i in {0..C}``; the environment
  runs the deterministic integer projection Pi (register R24) and writes
  the UNCHANGED action CSV (``rslaq_actions_for_ns3.csv``) with
  ``dedicatedPRB = 100 * a_i / C`` and ``minPRB = maxPRB = dedicatedPRB``
  (contract §3.2/§3.4), algorithm column fixed at PF = 1 (register R17).
  The absolute-quote execution semantics (Σa < C leaves RBGs idle, R27)
  live on the ns-3 side; this env never renormalizes.
  **C is DYNAMIC (register R01 as amended):** the constructor ``total_prbs``
  is only the pre-telemetry prior (52); as soon as the ns-3 side telemetry
  ``sqr_prb_windows.csv`` appears (contract §3.5), the effective cell
  capacity is read from its ``total_prb`` column (53 = the real C++ RBG
  grid on the 10 MHz scenario) and the action space, the projection and
  the ``dedicatedPRB`` conversion adopt it (:meth:`SqrEnv.refresh_effective_c`).
- **Decision period**: 100 ms (contract §2) — ``periodMs`` default 100.

Causality (contract §2): the observation returned by ``step`` reflects the
window that JUST CLOSED; the training loop (``examples/sqr_run.py``) uses
y_t of that window together with the action that was EXECUTED during it
(z_t, a_final, y_t — register R22).

The heavy ns-3 lifecycle is inherited from ``NsOranEnv``; everything that
is pure computation (KPM → observation, action → projection → CSV tuples)
is factored into :class:`SqrObservationBuilder` and module functions so it
can be tested against a MOCK of the contract without ns-3 (Fase 2 gate).
"""

from __future__ import annotations

import glob
import os
import uuid
import subprocess
import selectors
import fcntl
import time
import warnings
from typing import override

import numpy as np
from gymnasium import spaces
from posix_ipc import Semaphore, O_CREAT

from nsoran.ns_env import NsOranEnv
from nsoran.action_controller import ActionController
from nsoran.datalake import SQLiteDatabaseAPI

from nsoran.sqr.coordinator import project_prbs, ProjectionResult
from nsoran.sqr.sqr_kpis import SqrKpmParser, SqrKpmWindow

#: Files of the frozen contract.
SQR_KPM_FILENAME = "sqr-kpms.txt"          # contract §4 (NEW name, no rslaq clash)
SQR_CONTROL_FILENAME = "rslaq_actions_for_ns3.csv"  # contract §3.1 (unchanged)
SQR_LOG_FILENAME = "SqrActions.txt"
#: ns-3 side per-window telemetry (contract §3.5): source of the effective C.
SQR_PRB_TELEMETRY_GLOB = os.path.join(
    "results_sqr_network_only", "seed=*", "sqr_prb_windows.csv"
)

#: Scheduler column value (register R17: PF fixed, identical for all methods).
SQR_ALGORITHM_PF = 1

#: Frozen observation scales (register R09; provisory until the Fase 1 smoke).
MAX_THR_NORM = 1.5            # highest per-UE SLA (Mbps)
N_MAX_SLICE = 10              # UE count scale, per slice
QUEUE_CAP_BYTES = 2_000_000.0 # 2 MB queue scale (sum over the slice's UEs)
SNR_SCALE_DB = 30.0           # snr_norm = clip(snr/30, 0, 1)
N_MAX_CELL = 20               # UE count scale, whole cell

DEFAULT_SLAS = {0: 0.5, 1: 1.0, 2: 1.5}  # Mbps per UE [ARTIGO, Tabela I]


def clip_norm(value: float, cap: float) -> float:
    """value / cap clipped to [0, 1] (0 when the cap is not positive)."""
    if cap <= 0:
        return 0.0
    return float(np.clip(float(value) / cap, 0.0, 1.0))


def read_effective_total_prbs(sim_path: str | None) -> int | None:
    """Effective cell capacity C from the ns-3 telemetry (contract §3.5).

    Reads the last complete row of ``sqr_prb_windows.csv`` and returns its
    ``total_prb`` column — the C++ RBG grid count in PRBs, which is the REAL
    capacity the quota percentages are applied against (53 on the primary
    10 MHz scenario: 13 RBGs x 4 PRBs + 1, register R01 as amended — the
    round 3GPP number 52 is NOT the grid the scheduler actually holds).

    Returns None whenever there is no telemetry yet, the file has no
    complete row, the last row is a torn write (column-count mismatch),
    the column is malformed or the value is not positive — the caller then
    keeps the C it already has (fail-static, never fail-loud here: this
    runs inside the control loop).
    """
    if not sim_path:
        return None
    matches = glob.glob(os.path.join(sim_path, SQR_PRB_TELEMETRY_GLOB))
    if not matches:
        return None
    try:
        with open(matches[0], "r", newline="") as f:
            lines = [ln for ln in f if ln.strip()]
    except OSError:
        return None
    if len(lines) < 2:
        return None  # header only (or empty): no window closed yet
    header = lines[0].strip().split(",")
    values = lines[-1].strip().split(",")
    if len(values) != len(header):
        return None  # torn write of the C++; skip this sample
    try:
        c = int(float(values[header.index("total_prb")]))
    except (KeyError, ValueError):
        return None
    return c if c > 0 else None


class SqrObservationBuilder:
    """Pure KPM→observation / projection→CSV logic (mock-testable).

    Holds the stateful KPM parser (SNR imputation carries across windows)
    and the last allocation, needed for the ``prb_outros_norm`` feature.
    """

    def __init__(
        self,
        num_slices: int = 3,
        total_prbs: int = 52,
        sla_thresholds: dict[int, float] | None = None,
        max_thr_norm: float = MAX_THR_NORM,
        n_max_slice: int = N_MAX_SLICE,
        queue_cap_bytes: float = QUEUE_CAP_BYTES,
        snr_scale_db: float = SNR_SCALE_DB,
        n_max_cell: int = N_MAX_CELL,
    ) -> None:
        self.num_slices = int(num_slices)
        self.total_prbs = int(total_prbs)
        self.sla_thresholds = dict(
            sla_thresholds if sla_thresholds is not None else DEFAULT_SLAS
        )
        self.max_thr_norm = float(max_thr_norm)
        self.n_max_slice = float(n_max_slice)
        self.queue_cap_bytes = float(queue_cap_bytes)
        self.snr_scale_db = float(snr_scale_db)
        self.n_max_cell = float(n_max_cell)
        self.parser = SqrKpmParser(
            num_slices=self.num_slices, sla_thresholds=self.sla_thresholds
        )
        self.last_window: SqrKpmWindow | None = None
        self.last_allocation: np.ndarray | None = None

    # ------------------------------------------------------------------
    # KPM -> observation
    # ------------------------------------------------------------------

    def update_from_kpm(self, kpm_path: str) -> SqrKpmWindow:
        """Parse the KPM file and cache the window."""
        self.last_window = self.parser.parse_window(kpm_path)
        return self.last_window

    def build_obs(
        self,
        window: SqrKpmWindow | None = None,
        allocation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Observation matrix (num_slices, 6) float32, all entries in [0, 1].

        Slice with no UE: throughput/queue/SNR features are 0.0 (nothing on
        the air — documented encoding); the UE-count column carries the
        information that the slice is empty.
        """
        w = window if window is not None else self.last_window
        alloc = (
            np.asarray(allocation, dtype=np.float64)
            if allocation is not None
            else self.last_allocation
        )
        obs = np.zeros((self.num_slices, 6), dtype=np.float32)
        if w is None:
            return obs
        total_ues_norm = clip_norm(w.total_ues, self.n_max_cell)
        for sid in range(self.num_slices):
            k = w.slices.get(sid)
            if k is None or k.n_ues == 0:
                # prb_outros/total_ues are still meaningful for an empty slice.
                obs[sid, 4] = self._prb_outros_norm(sid, alloc)
                obs[sid, 5] = total_ues_norm
                continue
            snr_mean = (
                float(np.mean(k.snr_values)) if k.snr_values else 0.0
            )
            obs[sid, 0] = clip_norm(k.lowest_thr_mbps or 0.0, self.max_thr_norm)
            obs[sid, 1] = clip_norm(k.n_ues, self.n_max_slice)
            obs[sid, 2] = clip_norm(k.queue_bytes_sum, self.queue_cap_bytes)
            obs[sid, 3] = clip_norm(snr_mean, self.snr_scale_db)
            obs[sid, 4] = self._prb_outros_norm(sid, alloc)
            obs[sid, 5] = total_ues_norm
        return obs

    def _prb_outros_norm(self, slice_id: int, alloc: np.ndarray | None) -> float:
        """sum_{j != i} a_j / C; 0.0 before the first action."""
        if alloc is None:
            return 0.0
        others = float(np.sum(alloc)) - float(alloc[slice_id])
        return clip_norm(others, self.total_prbs)

    def slice_states(self, window: SqrKpmWindow | None = None) -> list[np.ndarray]:
        """Per-slice z_i (float64, dim 6) as consumed by the controllers."""
        obs = self.build_obs(window)
        return [obs[i].astype(np.float64) for i in range(self.num_slices)]

    # ------------------------------------------------------------------
    # Action -> projection -> CSV tuples
    # ------------------------------------------------------------------

    def compute_control_tuples(
        self, proposals: np.ndarray
    ) -> tuple[list[tuple], ProjectionResult]:
        """Project integer proposals and build the CSV rows (contract §3).

        Returns the ``(sliceId, dedicatedPRB, minPRB, maxPRB, algorithm)``
        tuples in slice order plus the :class:`ProjectionResult`
        (``prb_requested`` = proposals, ``prb_executed`` = allocated).
        """
        result = project_prbs(proposals, self.total_prbs)
        self.last_allocation = result.allocated.astype(np.float64)
        actions = []
        for sid in range(self.num_slices):
            a = int(result.allocated[sid])
            dedicated = 100.0 * a / float(self.total_prbs)
            actions.append(
                (sid, dedicated, dedicated, dedicated, SQR_ALGORITHM_PF)
            )
        return actions, result

    def pooled_violation_rate(self, window: SqrKpmWindow | None = None) -> float | None:
        """Pooled per-UE violation rate over slices WITH UEs (R19/R26).

        Slices without UEs are excluded from the denominator; returns None
        when no slice has UEs (period excluded everywhere).
        """
        w = window if window is not None else self.last_window
        if w is None:
            return None
        num = 0
        den = 0
        for sid in range(self.num_slices):
            k = w.slices.get(sid)
            if k is None or k.n_ues == 0 or k.violation_rate is None:
                continue
            num += k.violation_rate * k.n_ues
            den += k.n_ues
        if den == 0:
            return None
        return float(num / den)


class SqrEnv(NsOranEnv):
    """SQR environment over the frozen IPC contract (100 ms windows)."""

    def __init__(
        self,
        ns3_path: str,
        scenario_configuration: dict,
        output_folder: str,
        optimized: bool = False,
        total_prbs: int = 52,
        num_slices: int = 3,
        sla_thresholds: dict[int, float] | None = None,
        max_steps: int | None = None,
        enable_step_logging: bool = False,  # sqr_run.py owns step_metrics.csv
        builder_kwargs: dict | None = None,
    ):
        scenario_configuration.setdefault("simId", [""])
        scenario_configuration.setdefault("periodMs", [100])  # contract §2
        scenario_configuration.setdefault("scenario", ["sqr"])
        scenario_configuration.setdefault("seed", [1])
        scenario_configuration.setdefault("appStart", [0.5])
        scenario_configuration.setdefault("simTime", [600.0])

        sim_time = float(scenario_configuration.get("simTime", [600.0])[0])
        app_start = float(scenario_configuration.get("appStart", [0.5])[0])
        period_ms = int(scenario_configuration.get("periodMs", [100])[0])
        if max_steps is None:
            max_steps = int((sim_time - app_start) * 1000 / period_ms)
            if max_steps <= 0:
                max_steps = 50  # fallback safety
        elif isinstance(max_steps, str):
            max_steps = int(max_steps)
        self.max_steps = max_steps

        self.total_prbs = int(total_prbs)
        self.num_slices = int(num_slices)
        self.sla_thresholds = dict(
            sla_thresholds if sla_thresholds is not None else DEFAULT_SLAS
        )
        self.enable_step_logging = enable_step_logging

        observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_slices, 6), dtype=np.float32
        )
        action_space = spaces.MultiDiscrete(
            np.full(self.num_slices, self.total_prbs + 1, dtype=np.int64)
        )

        super().__init__(
            ns3_path=ns3_path,
            scenario="sqr-sim",
            scenario_configuration=scenario_configuration,
            output_folder=output_folder,
            optimized=optimized,
            skip_configuration=True,
            control_header=[
                "timestamp", "sliceId", "dedicatedPRB", "minPRB", "maxPRB",
                "algorithm",
            ],
            log_file=SQR_LOG_FILENAME,
            control_file=SQR_CONTROL_FILENAME,
        )

        # Override whatever spaces the base class set.
        self.observation_space = observation_space
        self.action_space = action_space

        self.builder = SqrObservationBuilder(
            num_slices=self.num_slices,
            total_prbs=self.total_prbs,
            sla_thresholds=self.sla_thresholds,
            **(builder_kwargs or {}),
        )
        if all(k in scenario_configuration for k in ("embbUes", "urllcUes", "mtcUes")):
            self.num_ues = sum(
                int(scenario_configuration[k][0]) for k in ("embbUes", "urllcUes", "mtcUes")
            )
        else:
            self.num_ues = int(scenario_configuration.get("ues", [20])[0])

        self._base_seed = int(scenario_configuration.get("seed", [1])[0])
        self._episode_count = 0
        self._seed_cycle = int(scenario_configuration.get("seed_cycle", [100])[0])
        self._current_seed_index = 0
        self.num_steps = 0
        self.last_projection: ProjectionResult | None = None
        self.last_decision_time_ms: float | None = None
        # Countable KPM-missing fallback (contract §5): windows where the
        # acquire returned but sqr-kpms.txt was absent — the observation
        # stays at the last valid window and every such step warns.
        self.kpm_missing_steps = 0
        self._last_kpm_missing = False
        # Where the current C came from: "config" (pre-telemetry prior) or
        # "telemetry" (effective C read from sqr_prb_windows.csv, R01).
        self.total_prbs_source = "config"
        # Defaults before start_sim() (which overwrites both); set here so
        # the pure-computation methods are usable against a contract mock.
        self.sim_result = {"params": {}, "meta": {}}
        self.last_timestamp = 0

    @override
    def reset(self, *, seed=None, options=None):
        self.num_steps = 0
        self.kpm_missing_steps = 0
        self._last_kpm_missing = False
        self._episode_count += 1
        self._current_seed_index = (self._episode_count - 1) // self._seed_cycle
        current_seed = self._base_seed + self._current_seed_index
        self.scenario_configuration["seed"] = [current_seed]
        return super().reset(seed=seed, options=options)

    def start_sim(self):
        """Start one ns-3 SQR simulation process and initialize IPC resources.

        Clone of ``RslaqEnv.start_sim`` (the base-class version requires a
        ``ues`` datalake parameter and no seed rotation); the SQR deltas are
        the scenario name and the 100 ms period.
        """
        if self.is_open:
            raise ValueError(
                "The environment is open and a new start_sim has been called."
            )

        self.is_open = True

        sim_uuid = str(uuid.uuid4())
        self.scenario_configuration["simId"] = [sim_uuid]

        parameters = self.scenario_configuration
        flat_params = {}
        for k, v in parameters.items():
            if k in ("seed_cycle", "_current_seed_index"):
                continue
            if isinstance(v, list) and len(v) == 1:
                flat_params[k] = v[0]
            else:
                flat_params[k] = v

        self.sim_result = {"params": {}, "meta": {}}
        self.sim_result["params"].update(flat_params)

        command = [self.script_executable] + [
            f"--{param}={value}" for param, value in flat_params.items()
        ]

        self.sim_result["meta"]["id"] = sim_uuid
        self.sim_path = os.path.join(self.output_folder, sim_uuid)
        os.makedirs(self.sim_path)

        self.action_controller = ActionController(
            self.sim_path, self.log_file, self.control_file, self.control_header
        )
        self.datalake = SQLiteDatabaseAPI(
            self.sim_path, num_ues_gnb=self.num_ues, debug=False
        )
        self._init_datalake_usecase()

        nameMetricsReadySemaphore = "/sem_metrics_" + sim_uuid
        nameControlSemaphore = "/sem_control_" + sim_uuid
        self.metricsReadySemaphore = Semaphore(
            nameMetricsReadySemaphore, O_CREAT, mode=0o660, initial_value=0
        )
        self.controlSemaphore = Semaphore(
            nameControlSemaphore, O_CREAT, mode=0o660, initial_value=0
        )
        self.last_timestamp = 0

        self.sim_result["meta"]["start_time"] = time.time()
        self.sim_process = subprocess.Popen(
            command,
            cwd=self.sim_path,
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.selector = selectors.DefaultSelector()
        self._set_nonblocking(self.sim_process.stdout)
        self._set_nonblocking(self.sim_process.stderr)
        self.selector.register(self.sim_process.stdout, selectors.EVENT_READ)
        self.selector.register(self.sim_process.stderr, selectors.EVENT_READ)

    def refresh_effective_c(self) -> int | None:
        """Adopt the effective C read from the ns-3 telemetry (register R01).

        The constructor ``total_prbs`` is only the pre-telemetry prior (52);
        the C++ scheduler grid may hold a different effective capacity (53
        on the primary scenario — see :func:`read_effective_total_prbs`).
        When the telemetry reports a new positive C, this env's
        ``total_prbs``, the action space ``MultiDiscrete(C+1)`` and the
        observation builder are updated IN PLACE, so from the next decision
        on, the action set {0..C}, the projection Pi and the
        ``dedicatedPRB = 100*a_i/C`` conversion all follow the real grid.

        Returns the new C when it changed, else None. Idempotent: re-reading
        the same telemetry is a no-op.
        """
        c = read_effective_total_prbs(getattr(self, "sim_path", None))
        if c is None or c == self.total_prbs:
            return None
        previous = self.total_prbs
        self.total_prbs = int(c)
        self.total_prbs_source = "telemetry"
        self.action_space = spaces.MultiDiscrete(
            np.full(self.num_slices, self.total_prbs + 1, dtype=np.int64)
        )
        self.builder.total_prbs = self.total_prbs
        warnings.warn(
            f"[SqrEnv] C efetivo atualizado pela telemetria ns-3 "
            f"(sqr_prb_windows.csv:total_prb, register R01 emendada): "
            f"{previous} -> {self.total_prbs} PRBs; espaço de ação, projeção "
            f"e conversão dedicatedPRB passam a usá-lo",
            RuntimeWarning,
            stacklevel=2,
        )
        return self.total_prbs

    @override
    def _compute_action(self, action) -> list[tuple]:
        """Integer PRBs per slice -> projection -> control tuples (contract §3)."""
        raw = np.asarray(action)
        if raw.shape != (self.num_slices,):
            raise ValueError(
                f"action must have shape ({self.num_slices},), got {raw.shape}"
            )
        if not np.all(raw == np.floor(raw)):
            raise ValueError(
                "SQR actions must be whole PRBs; got "
                f"{raw[raw != np.floor(raw)].tolist()}"
            )
        actions, result = self.builder.compute_control_tuples(raw.astype(np.int64))
        self.last_projection = result
        return actions

    @override
    def _get_obs(self):
        return self.builder.build_obs()

    @override
    def _compute_reward(self) -> float:
        """Log-only reward: negative pooled per-UE violation rate (R19/R26).

        SQR learning does NOT consume this (model-based updates on y_t);
        the value exists for compatibility with Gymnasium loops and logs.
        """
        rate = self.builder.pooled_violation_rate()
        return 0.0 if rate is None else -float(rate)

    @override
    def _init_datalake_usecase(self):
        pass

    @override
    def _fill_datalake_usecase(self):
        """Read sqr-kpms.txt (contract §4) and build the observation.

        Before the KPM, the effective C is refreshed from the ns-3
        telemetry (contract §3.5) so the very first decisions after the
        first window boundary already use the real grid.  A missing KPM is
        a COUNTABLE fallback (contract §5): the observation stays at the
        last valid window, ``kpm_missing_steps`` advances and each such
        step emits a warning — never a silent return.
        """
        self.refresh_effective_c()
        kpm_files = glob.glob(os.path.join(self.sim_path, SQR_KPM_FILENAME))
        if not kpm_files:
            self.kpm_missing_steps += 1
            self._last_kpm_missing = True
            warnings.warn(
                f"[SqrEnv] KPM ausente no acquire (contrato §5) no passo "
                f"{self.num_steps}: mantendo a última janela válida "
                f"(fallback contável; kpm_missing_steps="
                f"{self.kpm_missing_steps})",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        self._last_kpm_missing = False
        window = self.builder.update_from_kpm(kpm_files[0])
        self.last_timestamp = max(self.last_timestamp, window.timestamp)

    @override
    def step(self, action) -> tuple:
        """Advance one 100 ms decision window (clone of RslaqEnv.step)."""
        if not self.is_simulation_over():
            actions = self._compute_action(action)
            self.action_controller.create_control_action(self.last_timestamp, actions)
            self.controlSemaphore.release()
            self._wait_data_availability()
            self._fill_datalake()

        self.num_steps += 1
        if self.num_steps >= self.max_steps:
            self.truncated = True

        obs = self._get_obs()
        reward_val = self._compute_reward()
        info = self._build_info(action)
        return obs, reward_val, self.terminated, self.truncated, info

    def _build_info(self, raw_action) -> dict:
        info = {
            "is_open": self.is_open,
            "results": self.sim_result,
            "num_steps": self.num_steps,
            "max_steps": self.max_steps,
            "raw_action": np.asarray(raw_action).tolist(),
            "action_info": self._action_info(),
            # Effective C (register R01 as amended) — the run loop syncs the
            # controllers' n_prbs to it; source: "config" | "telemetry".
            "total_prbs": self.total_prbs,
            "total_prbs_source": self.total_prbs_source,
            # Countable KPM-missing fallback (contract §5).
            "kpm_missing": self._last_kpm_missing,
            "kpm_missing_steps": self.kpm_missing_steps,
        }
        w = self.builder.last_window
        if w is not None:
            info["kpm_timestamp"] = w.timestamp
            info["total_ues"] = w.total_ues
            info["pooled_violation_rate"] = self.builder.pooled_violation_rate()
            for sid in range(self.num_slices):
                k = w.slices.get(sid)
                info[f"slice_{sid}"] = {
                    "n_ues": k.n_ues if k else 0,
                    "p5_thr_mbps": k.p5_thr_mbps if k else None,       # y_t
                    "lowest_thr_mbps": k.lowest_thr_mbps if k else None,
                    "mean_thr_mbps": k.mean_thr_mbps if k else None,
                    "queue_bytes_sum": k.queue_bytes_sum if k else 0.0,
                    "violation_rate": k.violation_rate if k else None,
                    "n_snr_imputed": k.n_snr_imputed if k else 0,
                    "snr_imputations": dict(k.snr_imputations) if k else {},
                }
        if self.last_projection is not None:
            info["projection"] = {
                "prb_requested": self.last_projection.proposals.tolist(),
                "prb_executed": self.last_projection.allocated.tolist(),
                "cut": self.last_projection.cut.tolist(),
                "over_capacity": self.last_projection.over_capacity,
                "virtual_updates": [
                    {
                        "slice_id": u.slice_id,
                        "proposed": u.proposed,
                        "allocated": u.allocated,
                        "deficit": u.deficit,
                    }
                    for u in self.last_projection.virtual_updates
                ],
            }
        return info

    def _action_info(self) -> dict:
        proj = self.last_projection
        period = self.scenario_configuration.get("periodMs", 100)
        if isinstance(period, (list, tuple)):
            period = period[0]
        return {
            "prb_requested": proj.proposals.tolist() if proj else None,
            "prb_executed": proj.allocated.tolist() if proj else None,
            "algorithm": SQR_ALGORITHM_PF,
            "period_ms": int(period),
        }
