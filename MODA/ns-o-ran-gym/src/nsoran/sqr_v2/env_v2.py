"""SQR v2 Gymnasium environment over IPC Handshake v2 (BUG-03 + BUG-05)."""

from __future__ import annotations

import glob
import os
import selectors
import subprocess
import time
import uuid
import warnings
from typing import Any, override

import numpy as np
from gymnasium import spaces
from posix_ipc import BusyError, ExistentialError, O_CREAT, Semaphore

from nsoran.action_controller import ActionController
from nsoran.datalake import SQLiteDatabaseAPI
from nsoran.ns_env import NsOranEnv
from nsoran.sqr.coordinator import ProjectionResult
from nsoran.sqr.sqr_env import (
    DEFAULT_SLAS,
    SQR_ALGORITHM_PF,
    SQR_CONTROL_FILENAME,
    SQR_KPM_FILENAME,
    SQR_LOG_FILENAME,
    SqrObservationBuilder,
)
from nsoran.sqr_v2.ipc_v2 import (
    IPC_V2_INDICATION_FILENAME,
    IpcIndicationV2,
    read_indication_v2,
    reconcile_prev_receipt,
    write_action_v2_atomic,
)
from nsoran.sqr_v2.manifest_v2 import validate_dev_seed


def compute_expected_callbacks(sim_time_s: float, period_ms: int) -> int:
    """Exact number of periodic callbacks ``sqr-v2-sim`` executes before ``Stop(simTime)``.

    ``KpmAndControlCallback`` is scheduled at ``period_ms, 2*period_ms, ...`` strictly
    before ``Simulator::Stop(Seconds(simTime))`` terminates; e.g. ``simTime=2.0 s``
    at ``100 ms`` executes ``19`` boundaries (``100..1900 ms``).
    """
    total_ms = int(round(float(sim_time_s) * 1000.0))
    n = (total_ms - 1) // int(period_ms)
    return max(n, 1)


class Ns3PrbWindowTelemetryV2:
    """Incremental reader of v2 ``sqr_prb_windows.csv`` keyed by ``window_seq``."""

    def __init__(self, sim_path: str | None, num_slices: int = 3):
        self._sim_path = sim_path
        self._num_slices = int(num_slices)
        self._path: str | None = None
        self._offset: int = 0
        self._header: list[str] | None = None
        self._last: dict[str, Any] | None = None
        self._by_governed_seq: dict[int, dict[str, Any]] = {}

    def _resolve(self) -> str | None:
        if self._path and os.path.isfile(self._path):
            return self._path
        if not self._sim_path:
            return None
        matches = glob.glob(
            os.path.join(
                self._sim_path,
                "results_sqr_network_only",
                "seed=*",
                "sqr_prb_windows.csv",
            )
        )
        new_path = matches[0] if matches else None
        if new_path != self._path:
            self._path = new_path
            self._offset = 0
            self._header = None
        return self._path

    def _parse_row(self, line: str) -> dict[str, Any] | None:
        values = line.strip().split(",")
        if self._header is None or len(values) != len(self._header):
            return None
        row = dict(zip(self._header, values))
        try:
            out: dict[str, Any] = {
                "timestamp_ms": int(float(row["timestamp_ms"])),
                "total_rbg": int(float(row["total_rbg"])),
                "total_prb": int(float(row["total_prb"])),
                "dl_slots": int(float(row["dl_slots"])),
                "prb_idle_rbg": int(float(row["prb_idle_rbg"])),
                "prb_scheduled": {},
                "prb_executed_rbg": {},
                "prb_budget_rbg": {},
                "run_id": row.get("run_id", ""),
                "window_seq": int(float(row["window_seq"])) if "window_seq" in row else -1,
                "applied_action_seq": (
                    int(float(row["applied_action_seq"])) if "applied_action_seq" in row else -1
                ),
                "ipc_status": row.get("ipc_status", ""),
                "decision_wall_ms": (
                    float(row["decision_wall_ms"]) if "decision_wall_ms" in row else np.nan
                ),
            }
            for sid in range(self._num_slices):
                out["prb_scheduled"][str(sid)] = int(float(row[f"prb_scheduled_rbg_{sid}"]))
                out["prb_executed_rbg"][str(sid)] = int(float(row[f"prb_executed_rbg_{sid}"]))
                out["prb_budget_rbg"][str(sid)] = int(float(row[f"prb_budget_rbg_{sid}"]))
        except (KeyError, ValueError):
            return None
        return out

    def refresh(self) -> dict[str, Any] | None:
        path = self._resolve()
        if path is None:
            return None
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                size = f.seek(0, os.SEEK_END)
                if size < self._offset:
                    self._offset = 0
                    self._header = None
                    self._last = None
                    self._by_governed_seq.clear()
                f.seek(self._offset)
                new_text = f.read()
        except OSError:
            return self._last
        if new_text and not new_text.endswith("\n"):
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
                wseq = parsed.get("window_seq", -1)
                if wseq > 0:
                    self._by_governed_seq[wseq] = parsed
        return self._last

    def latest(self) -> dict[str, Any] | None:
        return self.refresh()

    def for_window_seq(self, window_seq: int) -> dict[str, Any] | None:
        self.refresh()
        return self._by_governed_seq.get(int(window_seq))


class SqrV2Env(NsOranEnv):
    """SQR v2 Gymnasium environment executing ``scratch/sqr_v2/sqr-v2-sim.cc`` over IPC v2."""

    def __init__(
        self,
        ns3_path: str,
        scenario_configuration: dict,
        output_folder: str,
        *,
        optimized: bool = False,
        total_prbs: int = 52,
        num_slices: int = 3,
        sla_thresholds: dict[int, float] | None = None,
        max_steps: int | None = None,
        ipc_mode: str = "strict_rt",
        ipc_timeout_ms: int | None = None,
        ipc_protocol_version: int = 2,
        run_id: str | None = None,
        allow_reserved_seed: bool = False,
        builder_kwargs: dict | None = None,
    ):
        if ipc_mode not in ("strict_rt", "causal_sim"):
            raise ValueError(f"ipc_mode must be 'strict_rt' or 'causal_sim', got {ipc_mode!r}")
        if int(ipc_protocol_version) not in (1, 2):
            raise ValueError(f"ipc_protocol_version must be 1 or 2, got {ipc_protocol_version}")

        self.ipc_mode = str(ipc_mode)
        self.ipc_timeout_ms = (
            int(ipc_timeout_ms)
            if ipc_timeout_ms is not None
            else (50 if self.ipc_mode == "strict_rt" else 2000)
        )
        if self.ipc_timeout_ms <= 0:
            raise ValueError(f"ipc_timeout_ms must be positive, got {self.ipc_timeout_ms}")
        self.ipc_protocol_version = int(ipc_protocol_version)
        self.rt_50ms_deadline_enforced = bool(
            self.ipc_mode == "strict_rt" and self.ipc_timeout_ms <= 50
        )
        self.allow_reserved_seed = bool(allow_reserved_seed)
        self._explicit_run_id = run_id

        cfg = dict(scenario_configuration)
        cfg.setdefault("simId", [""])
        cfg.setdefault("periodMs", [100])
        cfg.setdefault("seed", [1])
        cfg.setdefault("simTime", [600.0])

        sim_time = float(cfg["simTime"][0] if isinstance(cfg["simTime"], list) else cfg["simTime"])
        period_ms = int(cfg["periodMs"][0] if isinstance(cfg["periodMs"], list) else cfg["periodMs"])
        if max_steps is None or int(max_steps) <= 0:
            max_steps = compute_expected_callbacks(sim_time, period_ms)
        self.max_steps = int(max_steps)
        self.period_ms = period_ms

        self.total_prbs = int(total_prbs)
        self.num_slices = int(num_slices)
        self.sla_thresholds = dict(sla_thresholds if sla_thresholds is not None else DEFAULT_SLAS)

        base_seed = int(cfg["seed"][0] if isinstance(cfg["seed"], list) else cfg["seed"])
        validate_dev_seed(base_seed, allow_reserved=self.allow_reserved_seed)
        self._base_seed = base_seed
        self._seed_cycle = int(
            cfg.pop("seed_cycle", [100])[0]
            if isinstance(cfg.get("seed_cycle"), list)
            else cfg.get("seed_cycle", 100)
        )
        # Remove Python-only hints rejected by ns-3 CommandLine
        for py_only in ("scenario", "appStart", "ues"):
            cfg.pop(py_only, None)

        super().__init__(
            ns3_path=ns3_path,
            scenario="sqr-v2-sim",
            scenario_configuration=cfg,
            output_folder=output_folder,
            optimized=optimized,
            skip_configuration=True,
            control_header=["timestamp", "sliceId", "dedicatedPRB", "minPRB", "maxPRB", "algorithm"],
            log_file=SQR_LOG_FILENAME,
            control_file=SQR_CONTROL_FILENAME,
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_slices, 6), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            np.full(self.num_slices, self.total_prbs + 1, dtype=np.int64)
        )
        self.builder = SqrObservationBuilder(
            num_slices=self.num_slices,
            total_prbs=self.total_prbs,
            sla_thresholds=self.sla_thresholds,
            **(builder_kwargs or {}),
        )
        self.num_ues = 20
        self._episode_count = 0
        self._current_seed_index = 0
        self.num_steps = 0
        self.run_id: str = ""
        self.sim_uuid: str = ""
        self.current_indication: IpcIndicationV2 | None = None
        self.current_window_seq: int = 0
        self.current_sim_time_ms: int = 0
        self.last_projection: ProjectionResult | None = None
        self._next_decision_wall_ms: float = 0.0
        self._last_written_decision_wall_ms: float = 0.0
        self.kpm_missing_steps = 0
        self._last_kpm_missing = False
        self.total_prbs_source = "config"
        self.last_timestamp = 0
        self.terminated = False
        self.truncated = False
        self.sim_result = {"params": {}, "meta": {}}

    def set_next_decision_wall_ms(self, wall_ms: float) -> None:
        self._next_decision_wall_ms = float(wall_ms)

    def _sync_capacity_from_value(self, new_c: int, source: str = "envelope") -> bool:
        if new_c <= 0 or new_c == self.total_prbs:
            if new_c == self.total_prbs and new_c > 0:
                self.total_prbs_source = source
            return False
        self.total_prbs = int(new_c)
        self.total_prbs_source = source
        self.action_space = spaces.MultiDiscrete(
            np.full(self.num_slices, self.total_prbs + 1, dtype=np.int64)
        )
        self.builder.total_prbs = self.total_prbs
        return True

    def _drain_metrics_semaphore(self) -> int:
        drained = 0
        while True:
            try:
                self.metricsReadySemaphore.acquire(timeout=0)
                drained += 1
            except BusyError:
                break
        return drained

    def _read_current_indication(self) -> IpcIndicationV2:
        ind_path = os.path.join(self.sim_path, IPC_V2_INDICATION_FILENAME)
        return read_indication_v2(ind_path, expected_run_id=self.run_id)

    @override
    def start_sim(self) -> None:
        if self.is_open:
            raise ValueError("Environment already open when start_sim() called")
        self.is_open = True

        sim_uuid = uuid.uuid4().hex
        self.sim_uuid = sim_uuid
        current_seed = int(self.scenario_configuration["seed"])
        validate_dev_seed(current_seed, allow_reserved=self.allow_reserved_seed)
        self.run_id = self._explicit_run_id or f"sqrv2_s{current_seed}_{sim_uuid[:10]}"

        params = dict(self.scenario_configuration)
        params["simId"] = sim_uuid
        params["runId"] = self.run_id
        params["ipcTimeoutMs"] = int(self.ipc_timeout_ms)
        params["ipcProtocolVersion"] = int(self.ipc_protocol_version)

        self.sim_result = {"params": dict(params), "meta": {"id": sim_uuid, "run_id": self.run_id}}
        self.sim_path = os.path.join(self.output_folder, sim_uuid)
        os.makedirs(self.sim_path, exist_ok=True)

        self.action_controller = ActionController(
            self.sim_path, self.log_file, self.control_file, self.control_header
        )
        self.datalake = SQLiteDatabaseAPI(self.sim_path, num_ues_gnb=self.num_ues, debug=False)
        self.metricsReadySemaphore = Semaphore(
            "/sem_metrics_" + sim_uuid, O_CREAT, mode=0o660, initial_value=0
        )
        self.controlSemaphore = Semaphore(
            "/sem_control_" + sim_uuid, O_CREAT, mode=0o660, initial_value=0
        )
        self.last_timestamp = 0

        command = [self.script_executable] + [f"--{k}={v}" for k, v in params.items()]
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

    @override
    def _wait_data_availability(self, timeout: float = 0.2) -> bool:
        while not self.is_simulation_over():
            try:
                self.metricsReadySemaphore.acquire(timeout=timeout)
                self._drain_metrics_semaphore()
                return True
            except BusyError:
                continue
        # Process exited: drain any final indication token queued right before exit
        return self._drain_metrics_semaphore() > 0

    def _parse_kpm_window(self) -> None:
        kpm_path = os.path.join(self.sim_path, SQR_KPM_FILENAME)
        if not os.path.isfile(kpm_path):
            self.kpm_missing_steps += 1
            self._last_kpm_missing = True
            warnings.warn(
                f"[SqrV2Env] KPM ausente no passo {self.num_steps}: fallback contável",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        self._last_kpm_missing = False
        window = self.builder.update_from_kpm(kpm_path)
        self.last_timestamp = max(self.last_timestamp, window.timestamp)

    @override
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            validate_dev_seed(seed, allow_reserved=self.allow_reserved_seed)
            self._base_seed = int(seed)
        self.num_steps = 0
        self.kpm_missing_steps = 0
        self._last_kpm_missing = False
        self.current_indication = None
        self.current_window_seq = 0
        self.current_sim_time_ms = 0
        self.last_projection = None
        self.terminated = False
        self.truncated = False

        self.close()
        self._episode_count += 1
        self._current_seed_index = (self._episode_count - 1) // self._seed_cycle
        current_seed = self._base_seed + self._current_seed_index
        validate_dev_seed(current_seed, allow_reserved=self.allow_reserved_seed)
        self.scenario_configuration["seed"] = current_seed

        self.start_sim()
        got = self._wait_data_availability(timeout=0.2)
        if not got and self.is_simulation_over():
            raise RuntimeError(
                f"sqr-v2-sim exited before first IPC indication (exit={self.sim_process.returncode})"
            )

        ind = self._read_current_indication()
        self.current_indication = ind
        self.current_window_seq = ind.window_seq
        self.current_sim_time_ms = ind.sim_time_ms
        # BUG-05 fix: sync total_prbs BEFORE Step 1 decision
        if ind.total_prb > 0:
            self._sync_capacity_from_value(ind.total_prb, source="ipc_v2_envelope")
        self._parse_kpm_window()

        obs = self._get_obs()
        info = self._build_info(raw_action=np.zeros(self.num_slices, dtype=np.int64), receipt=None)
        return obs, info

    @override
    def _compute_action(self, action) -> list[tuple]:
        raw = np.asarray(action)
        if raw.shape != (self.num_slices,):
            raise ValueError(f"action must have shape ({self.num_slices},), got {raw.shape}")
        if not np.all(raw == np.floor(raw)):
            raise ValueError(f"SQR actions must be integer PRBs, got {raw.tolist()}")
        actions, result = self.builder.compute_control_tuples(raw.astype(np.int64))
        self.last_projection = result
        return actions

    @override
    def _get_obs(self):
        return self.builder.build_obs()

    @override
    def _compute_reward(self) -> float:
        rate = self.builder.pooled_violation_rate()
        return 0.0 if rate is None else -float(rate)

    @override
    def _init_datalake_usecase(self):
        pass

    @override
    def _fill_datalake_usecase(self):
        self._parse_kpm_window()

    @override
    def step(self, action) -> tuple:
        sent_seq = self.current_window_seq
        sent_time_ms = self.current_sim_time_ms
        receipt: dict[str, Any] | None = None
        episode_end_no_next_window = False
        new_ind: IpcIndicationV2 | None = None

        if not self.is_simulation_over():
            actions = self._compute_action(action)
            assert self.last_projection is not None
            proj = self.last_projection
            slice_rows = [
                (
                    int(sid),
                    int(proj.proposals[sid]),
                    int(proj.proposals[sid]),
                    int(proj.allocated[sid]),
                    float(100.0 * int(proj.allocated[sid]) / float(self.total_prbs)),
                    int(SQR_ALGORITHM_PF),
                )
                for sid in range(self.num_slices)
            ]
            self._last_written_decision_wall_ms = float(self._next_decision_wall_ms)
            write_action_v2_atomic(
                self.sim_path,
                run_id=self.run_id,
                window_seq=sent_seq,
                sim_time_ms=sent_time_ms,
                slice_rows=slice_rows,
                agent_decision_wall_ms=self._last_written_decision_wall_ms,
                append_history=True,
            )
            self._next_decision_wall_ms = 0.0
            self.controlSemaphore.release()

            self._wait_data_availability(timeout=0.2)
            new_ind = self._read_current_indication()
            if new_ind.window_seq > sent_seq:
                self.current_indication = new_ind
                self.current_window_seq = new_ind.window_seq
                self.current_sim_time_ms = new_ind.sim_time_ms
                if new_ind.total_prb > 0:
                    self._sync_capacity_from_value(new_ind.total_prb, source="ipc_v2_envelope")
                self._parse_kpm_window()
                receipt = reconcile_prev_receipt(
                    new_ind,
                    expected_run_id=self.run_id,
                    sent_window_seq=sent_seq,
                )
            else:
                episode_end_no_next_window = True
                self.terminated = True

        self.num_steps += 1
        if self.num_steps >= self.max_steps and not episode_end_no_next_window:
            self.truncated = True

        obs = self._get_obs()
        reward_val = self._compute_reward()
        info = self._build_info(
            raw_action=action,
            receipt=receipt,
            sent_seq=sent_seq,
            sent_time_ms=sent_time_ms,
            episode_end_no_next_window=episode_end_no_next_window,
        )
        return obs, reward_val, self.terminated, self.truncated, info

    def _build_info(
        self,
        *,
        raw_action,
        receipt: dict[str, Any] | None,
        sent_seq: int = 0,
        sent_time_ms: int = 0,
        episode_end_no_next_window: bool = False,
    ) -> dict[str, Any]:
        ind = self.current_indication
        info: dict[str, Any] = {
            "is_open": self.is_open,
            "results": self.sim_result,
            "num_steps": self.num_steps,
            "max_steps": self.max_steps,
            "run_id": self.run_id,
            "ipc_mode": self.ipc_mode,
            "ipc_timeout_ms": self.ipc_timeout_ms,
            "rt_50ms_deadline_enforced": self.rt_50ms_deadline_enforced,
            "window_seq": self.current_window_seq,
            "sim_time_ms": self.current_sim_time_ms,
            "sent_window_seq": sent_seq,
            "sent_sim_time_ms": sent_time_ms,
            "episode_end_no_next_window": bool(episode_end_no_next_window),
            "raw_action": np.asarray(raw_action).tolist(),
            "total_prbs": self.total_prbs,
            "total_prbs_source": self.total_prbs_source,
            "kpm_missing": self._last_kpm_missing,
            "kpm_missing_steps": self.kpm_missing_steps,
            "ipc_receipt": receipt,
            "agent_decision_wall_ms": self._last_written_decision_wall_ms,
            "indication_active_ues": ind.active_ues if ind is not None else 0,
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
                    "p5_thr_mbps": k.p5_thr_mbps if k else None,
                    "lowest_thr_mbps": k.lowest_thr_mbps if k else None,
                    "mean_thr_mbps": k.mean_thr_mbps if k else None,
                    "queue_bytes_sum": k.queue_bytes_sum if k else 0.0,
                    "violation_rate": k.violation_rate if k else None,
                    "n_snr_imputed": k.n_snr_imputed if k else 0,
                }
        if self.last_projection is not None:
            info["projection"] = {
                "prb_requested": self.last_projection.proposals.tolist(),
                "prb_executed_python_proj": self.last_projection.allocated.tolist(),
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

    def wait_sim_exit(self, timeout_s: float = 120.0) -> int | None:
        """Wait cleanly for ``sqr-v2-sim`` to write WriteFinalReport & drain stdout."""
        proc = getattr(self, "sim_process", None)
        if proc is None:
            return None
        deadline = time.time() + float(timeout_s)
        while proc.poll() is None and time.time() < deadline:
            self.read_streams()
            time.sleep(0.05)
        self.read_streams()
        if proc.returncode is not None:
            self.sim_result.setdefault("meta", {})["exitcode"] = proc.returncode
        return proc.returncode

    def read_cpp_ipc_summary(self) -> dict[str, Any] | None:
        if not getattr(self, "sim_path", None):
            return None
        matches = glob.glob(
            os.path.join(
                self.sim_path,
                "results_sqr_network_only",
                "seed=*",
                "sqr_ipc_v2_summary.json",
            )
        )
        if not matches:
            return None
        import json as _json

        with open(matches[0], "r", encoding="utf-8") as f:
            return _json.load(f)

    @override
    def close(self) -> None:
        if not self.is_open:
            return
        proc = getattr(self, "sim_process", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
        try:
            self.read_streams()
        except Exception:
            pass
        for sem_attr in ("controlSemaphore", "metricsReadySemaphore"):
            sem_obj = getattr(self, sem_attr, None)
            if sem_obj is not None:
                try:
                    sem_obj.close()
                except Exception:
                    pass
                try:
                    sem_obj.unlink()
                except (ExistentialError, Exception):
                    pass
        selector = getattr(self, "selector", None)
        if selector is not None:
            try:
                selector.close()
            except Exception:
                pass
        self.is_open = False
