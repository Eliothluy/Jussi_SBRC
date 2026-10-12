"""Predictive helpers for RSLAQ experiments.

This module keeps the predictive data pipeline independent from ``RslaqEnv``.
It consumes the existing per-step ``step_metrics.csv`` logs and builds compact
temporal windows that can be used both for offline forecaster training and for
online state augmentation during SAC training.
"""

from __future__ import annotations

import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - exercised only without optional ML deps
    torch = None
    nn = None

from .rslaq_kpis import (
    DEFAULT_MAX_BTX,
    DEFAULT_MAX_BUFFER_BYTES,
    DEFAULT_MAX_TDP,
)
from .rslaq_reward import compute_rslaq_reward


SCENARIOS = [
    "low_traffic",
    "normal",
    "congestion",
    "stressed",
    "insufficient_resources",
]

OBS_DIM = 16
ACTION_DIM = 3
RISK_DIM = 6
SCENARIO_DIM = len(SCENARIOS)
FEATURE_DIM = OBS_DIM + ACTION_DIM + 1 + RISK_DIM + SCENARIO_DIM
FORECAST_DIM = RISK_DIM + OBS_DIM

SLICE_NAME_TO_ID = {
    "embb": 0,
    "eMBB": 0,
    "urllc": 1,
    "URLLC": 1,
    "mtc": 2,
    "MTC": 2,
}

DEFAULT_RESOURCE_EFFICIENT_REWARD_CONFIG = {
    "reward_mode": "resource_efficient",
    "warmup_steps": 0,
    "consecutive_outage_steps": 1,
    "resource_efficiency_weight": 0.25,
    "need_match_weight": 0.30,
    "waste_penalty_weight": 0.35,
    "under_allocation_penalty_weight": 0.15,
    "embb_soft_guard_penalty_weight": 0.25,
    "embb_soft_guard_ratio": 0.80,
    "action_smoothness_weight": 0.05,
    "resource_dynamic_need_weight": 0.75,
    "resource_waste_deadband": 0.03,
}


def _require_torch():
    """Return torch modules or raise a clear error for optional ML features."""
    if torch is None or nn is None:
        raise ImportError(
            "rslaq_predictive requires PyTorch. Install it with `pip install nsoran[ml]`."
        )
    return torch, nn


@dataclass(frozen=True)
class StepFrame:
    """One environment step reconstructed from three per-slice log rows."""

    sim_id: str
    scenario: str
    seed: int
    episode: int
    step: int
    obs: np.ndarray
    action: np.ndarray
    reward: float
    outage_flags: np.ndarray
    soft_flags: np.ndarray


def _safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float, returning a default for malformed inputs."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    """Convert a value to int through float parsing, or return a default."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_bool(value) -> bool:
    """Parse common textual truthy values from CSV fields."""
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _slice_id(value) -> int:
    """Parse either canonical slice names or numeric slice IDs."""
    text = str(value).strip()
    if text in SLICE_NAME_TO_ID:
        return SLICE_NAME_TO_ID[text]
    lowered = text.lower()
    if lowered in SLICE_NAME_TO_ID:
        return SLICE_NAME_TO_ID[lowered]
    return _safe_int(text, -1)


def _clip_norm(value: float, cap: float) -> float:
    """Normalize a value by a positive cap and clip it to [0, 1]."""
    if cap <= 0:
        return 0.0
    return float(np.clip(value / cap, 0.0, 1.0))


def scenario_one_hot(scenario: str) -> np.ndarray:
    """Encode a scenario name as a one-hot vector."""
    vec = np.zeros(SCENARIO_DIM, dtype=np.float32)
    if scenario in SCENARIOS:
        vec[SCENARIOS.index(scenario)] = 1.0
    return vec


def rows_to_observation(rows: Sequence[dict]) -> np.ndarray:
    """Build the same 4x4 paper-style observation from step log rows."""
    obs = np.zeros((4, 4), dtype=np.float32)
    slice_rows = {int(row["slice_id"]): row for row in rows}

    for sid in range(3):
        row = slice_rows.get(sid, {})
        obs[0, sid] = _clip_norm(_safe_float(row.get("dTxBytes")), DEFAULT_MAX_BTX)
        obs[1, sid] = _clip_norm(
            _safe_float(row.get("bufferBytes_mean")), DEFAULT_MAX_BUFFER_BYTES
        )
        obs[2, sid] = _clip_norm(_safe_float(row.get("resourceSharePct")), 100.0)
        obs[3, sid] = _clip_norm(_safe_float(row.get("dLostPackets")), DEFAULT_MAX_TDP)

    dtx_total = sum(_safe_float(row.get("dTxBytes")) for row in slice_rows.values())
    buffer_mean = np.mean(
        [_safe_float(row.get("bufferBytes_mean")) for row in slice_rows.values()]
    )
    rsh_mean = np.mean(
        [_safe_float(row.get("resourceSharePct")) for row in slice_rows.values()]
    )
    lost_total = sum(_safe_float(row.get("dLostPackets")) for row in slice_rows.values())

    obs[0, 3] = _clip_norm(dtx_total, DEFAULT_MAX_BTX * 3)
    obs[1, 3] = _clip_norm(buffer_mean, DEFAULT_MAX_BUFFER_BYTES * 3)
    obs[2, 3] = _clip_norm(rsh_mean, 100.0)
    obs[3, 3] = _clip_norm(lost_total, DEFAULT_MAX_TDP * 3)
    return obs


def rows_to_frame(rows: Sequence[dict]) -> StepFrame:
    """Convert the three rows for one step into a compact frame."""
    if len(rows) < 3:
        raise ValueError(f"expected at least 3 slice rows, got {len(rows)}")
    first = rows[0]
    by_slice = {int(row["slice_id"]): row for row in rows}

    action = np.array(
        [
            _safe_float(first.get("action_embb")) / 100.0,
            _safe_float(first.get("action_urllc")) / 100.0,
            _safe_float(first.get("action_mtc")) / 100.0,
        ],
        dtype=np.float32,
    )
    outage = np.array(
        [_as_bool(by_slice.get(sid, {}).get("outage_flag")) for sid in range(3)],
        dtype=np.float32,
    )
    soft = np.array(
        [_as_bool(by_slice.get(sid, {}).get("soft_flag")) for sid in range(3)],
        dtype=np.float32,
    )

    return StepFrame(
        sim_id=str(first.get("sim_id", "")),
        scenario=str(first.get("scenario", "")),
        seed=_safe_int(first.get("seed")),
        episode=_safe_int(first.get("episode")),
        step=_safe_int(first.get("step")),
        obs=rows_to_observation(rows),
        action=action,
        reward=_safe_float(first.get("reward")),
        outage_flags=outage,
        soft_flags=soft,
    )


def _path_value(path: str, prefix: str, default: str = "") -> str:
    """Extract ``prefix=value`` metadata from a baseline output path."""
    for part in os.path.normpath(path).split(os.sep):
        if part.startswith(prefix):
            return part[len(prefix):]
    return default


def _seed_from_path(path: str) -> int:
    """Extract seed from baseline run directories like ``seed=1_run=1``."""
    match = re.search(r"seed=(\d+)", path)
    return int(match.group(1)) if match else 0


def _baseline_run_dir(timeseries_path: str) -> str:
    return os.path.dirname(timeseries_path)


def _load_slice_allocations(run_dir: str) -> dict[int, list[float]]:
    """Load per-timestamp slice allocation percentages from ``slice_alloc.csv``."""
    path = os.path.join(run_dir, "slice_alloc.csv")
    if not os.path.exists(path):
        return {}

    allocations: dict[int, list[float]] = {}
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            timestamp = _safe_int(row.get("timestamp_ms"), -1)
            sid = _slice_id(row.get("slice"))
            if timestamp < 0 or sid not in (0, 1, 2):
                continue
            pct = _safe_float(row.get("rsh_real_pct"), _safe_float(row.get("configured_weight")) * 100.0)
            allocations.setdefault(timestamp, [0.0, 0.0, 0.0])[sid] = max(pct, 0.0)
    return allocations


def _allocation_for_timestamp(
    timestamp: int,
    allocations: dict[int, list[float]],
    metrics: dict[int, dict],
) -> list[float]:
    """Return the most recent known baseline allocation for a timestamp."""
    if timestamp in allocations and sum(allocations[timestamp]) > 0.0:
        return allocations[timestamp]

    previous = [ts for ts in allocations if ts <= timestamp and sum(allocations[ts]) > 0.0]
    if previous:
        return allocations[max(previous)]

    observed = [
        max(float(metrics.get(sid, {}).get("resourceSharePct_mean", 0.0)), 0.0)
        for sid in range(3)
    ]
    if sum(observed) > 0.0:
        return observed
    return [100.0 / 3.0, 100.0 / 3.0, 100.0 / 3.0]


def _baseline_metrics_from_rows(rows: Sequence[dict], allocation_pct: list[float]) -> dict[int, dict]:
    """Aggregate per-UE baseline timeseries rows into per-slice reward metrics."""
    metrics: dict[int, dict] = {}
    for sid in range(3):
        slice_rows = [row for row in rows if _slice_id(row.get("slice")) == sid]
        valid_buffers = [
            _safe_float(row.get("buffer_bytes"))
            for row in slice_rows
            if _safe_float(row.get("buffer_bytes"), -1.0) >= 0.0
        ]
        metrics[sid] = {
            "throughputMbps_sum": sum(_safe_float(row.get("thr_mbps")) for row in slice_rows),
            "dTxBytes_sum": sum(_safe_float(row.get("tx_bytes_delta")) for row in slice_rows),
            "dRxBytes_sum": sum(_safe_float(row.get("rx_bytes_delta")) for row in slice_rows),
            "dLostPackets_sum": sum(_safe_float(row.get("dropped_packets_delta")) for row in slice_rows),
            "bufferBytes_mean": float(np.mean(valid_buffers)) if valid_buffers else 0.0,
            "bufferBytes_max": max(valid_buffers) if valid_buffers else 0.0,
            "resourceSharePct_mean": allocation_pct[sid] if sid < len(allocation_pct) else 0.0,
            "ue_count": float(len(slice_rows)),
        }
    return metrics


def _baseline_observation(metrics: dict[int, dict]) -> np.ndarray:
    """Build a paper-style 4x4 observation from baseline aggregate metrics."""
    rows = []
    for sid, name in enumerate(["eMBB", "URLLC", "MTC"]):
        values = metrics.get(sid, {})
        rows.append(
            {
                "slice_id": sid,
                "slice": name,
                "dTxBytes": values.get("dTxBytes_sum", 0.0),
                "bufferBytes_mean": values.get("bufferBytes_mean", 0.0),
                "resourceSharePct": values.get("resourceSharePct_mean", 0.0),
                "dLostPackets": values.get("dLostPackets_sum", 0.0),
            }
        )
    return rows_to_observation(rows)


def load_baseline_frames(
    timeseries_path: str,
    reward_config: dict | None = None,
) -> List[StepFrame]:
    """Load baseline ns-3 ``timeseries.csv`` data as predictive frames.

    The frame reward and SLA flags are recalculated with the resource-efficient
    RSLAQ reward so offline forecaster targets match the article contribution.
    """
    run_dir = _baseline_run_dir(timeseries_path)
    scenario = _path_value(timeseries_path, "scenario=", "normal")
    seed = _seed_from_path(timeseries_path)
    allocations = _load_slice_allocations(run_dir)
    config = dict(DEFAULT_RESOURCE_EFFICIENT_REWARD_CONFIG)
    if reward_config:
        config.update(reward_config)

    grouped: dict[int, list[dict]] = {}
    with open(timeseries_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            timestamp = _safe_int(row.get("timestamp_ms"), -1)
            if timestamp >= 0:
                grouped.setdefault(timestamp, []).append(row)

    frames: list[StepFrame] = []
    previous_action: list[float] | None = None
    for step, timestamp in enumerate(sorted(grouped), start=1):
        rows = grouped[timestamp]
        rough_metrics = _baseline_metrics_from_rows(rows, [0.0, 0.0, 0.0])
        allocation_pct = _allocation_for_timestamp(timestamp, allocations, rough_metrics)
        metrics = _baseline_metrics_from_rows(rows, allocation_pct)
        action_info = {"prb_pct": allocation_pct}
        if previous_action is not None:
            action_info["previous_prb_pct"] = previous_action
        reward = compute_rslaq_reward(
            metrics,
            scenario=scenario,
            action_info=action_info,
            config=config,
            step_count=step,
        )
        frames.append(
            StepFrame(
                sim_id=f"baseline:{scenario}:{_path_value(timeseries_path, 'mode=', 'unknown')}:{seed}",
                scenario=scenario,
                seed=seed,
                episode=1,
                step=step,
                obs=_baseline_observation(metrics),
                action=(np.asarray(allocation_pct, dtype=np.float32) / 100.0),
                reward=float(reward.reward),
                outage_flags=np.array(
                    [bool(reward.outage_flags.get(sid, False)) for sid in range(3)],
                    dtype=np.float32,
                ),
                soft_flags=np.array(
                    [bool(reward.soft_flags.get(sid, False)) for sid in range(3)],
                    dtype=np.float32,
                ),
            )
        )
        previous_action = allocation_pct
    return frames


def frame_to_feature(frame: StepFrame) -> np.ndarray:
    """Flatten one reconstructed step frame into a forecaster input feature."""
    reward_scaled = (np.clip(frame.reward, -1.0, 2.0) + 1.0) / 3.0
    return np.concatenate(
        [
            frame.obs.reshape(-1),
            frame.action,
            np.array([reward_scaled], dtype=np.float32),
            frame.outage_flags,
            frame.soft_flags,
            scenario_one_hot(frame.scenario),
        ]
    ).astype(np.float32)


def frames_to_target(future_frames: Sequence[StepFrame]) -> np.ndarray:
    """Build a future-risk and KPI target vector from forecast-horizon frames."""
    outage_risk = np.max([frame.outage_flags for frame in future_frames], axis=0)
    soft_risk = np.max([frame.soft_flags for frame in future_frames], axis=0)
    obs_mean = np.mean([frame.obs.reshape(-1) for frame in future_frames], axis=0)
    return np.concatenate([outage_risk, soft_risk, obs_mean]).astype(np.float32)


def load_step_frames(csv_path: str) -> List[StepFrame]:
    """Load all step frames from one ``step_metrics.csv`` file."""
    grouped: dict[tuple[int, int], list[dict]] = {}
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            key = (_safe_int(row.get("episode")), _safe_int(row.get("step")))
            grouped.setdefault(key, []).append(row)

    frames = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: _safe_int(row.get("slice_id")))
        if len(rows) >= 3:
            frames.append(rows_to_frame(rows[:3]))
    return frames


def find_step_metric_files(source_root: str, limit_files: int | None = None) -> list[str]:
    """Find step_metrics.csv files below a campaign results directory."""
    files = sorted(glob.glob(os.path.join(source_root, "**", "step_metrics.csv"), recursive=True))
    if limit_files is not None and limit_files > 0:
        files = files[:limit_files]
    return files


def find_baseline_timeseries_files(source_root: str, limit_files: int | None = None) -> list[str]:
    """Find ns-3 network-only baseline ``timeseries.csv`` files."""
    files = sorted(glob.glob(os.path.join(source_root, "**", "timeseries.csv"), recursive=True))
    if limit_files is not None and limit_files > 0:
        files = files[:limit_files]
    return files


def build_forecast_sequences(
    source_root: str,
    sequence_len: int = 8,
    horizon: int = 5,
    limit_files: int | None = None,
    source_format: str = "auto",
    reward_config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build ``(X, y, meta)`` arrays from a campaign result directory."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    meta: list[dict] = []

    source_format = source_format.strip().lower()
    sources: list[tuple[str, str]] = []
    if source_format in ("auto", "step_metrics"):
        sources.extend(("step_metrics", path) for path in find_step_metric_files(source_root))
    if source_format in ("auto", "baseline"):
        sources.extend(("baseline", path) for path in find_baseline_timeseries_files(source_root))
    if limit_files is not None and limit_files > 0:
        sources = sources[:limit_files]

    for source_type, path in sources:
        frames = (
            load_baseline_frames(path, reward_config=reward_config)
            if source_type == "baseline"
            else load_step_frames(path)
        )
        if len(frames) < sequence_len + horizon:
            continue
        features = [frame_to_feature(frame) for frame in frames]
        for end_idx in range(sequence_len - 1, len(frames) - horizon):
            future = frames[end_idx + 1 : end_idx + 1 + horizon]
            xs.append(np.stack(features[end_idx - sequence_len + 1 : end_idx + 1]))
            ys.append(frames_to_target(future))
            meta.append(
                {
                    "source": path,
                    "source_type": source_type,
                    "sim_id": frames[end_idx].sim_id,
                    "scenario": frames[end_idx].scenario,
                    "seed": frames[end_idx].seed,
                    "episode": frames[end_idx].episode,
                    "step": frames[end_idx].step,
                }
            )

    if not xs:
        return (
            np.zeros((0, sequence_len, FEATURE_DIM), dtype=np.float32),
            np.zeros((0, FORECAST_DIM), dtype=np.float32),
            meta,
        )
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), meta


class TemporalKpiForecaster(nn.Module if nn is not None else object):
    """Small GRU forecaster for near-future SLA risk and normalized KPIs."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 64,
        output_dim: int = FORECAST_DIM,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """Create the GRU encoder and sigmoid prediction head."""
        _, nn_mod = _require_torch()
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn_mod.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn_mod.Sequential(
            nn_mod.Linear(hidden_dim, hidden_dim),
            nn_mod.ReLU(),
            nn_mod.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict normalized future SLA risks and KPIs from temporal features."""
        torch_mod, _ = _require_torch()
        _, hidden = self.gru(x)
        logits = self.head(hidden[-1])
        return torch_mod.sigmoid(logits)


class TemporalKpiForecasterLSTM(nn.Module if nn is not None else object):
    """LSTM forecaster for near-future SLA risk and normalized KPIs."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 64,
        output_dim: int = FORECAST_DIM,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """Create the LSTM encoder and sigmoid prediction head."""
        _, nn_mod = _require_torch()
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn_mod.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn_mod.Sequential(
            nn_mod.Linear(hidden_dim, hidden_dim),
            nn_mod.ReLU(),
            nn_mod.Dropout(effective_dropout),
            nn_mod.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict normalized future SLA risks and KPIs from temporal features."""
        torch_mod, _ = _require_torch()
        _, (hidden, _) = self.lstm(x)
        logits = self.head(hidden[-1])
        return torch_mod.sigmoid(logits)


def create_forecaster(cell_type: str = "gru", **kwargs):
    """Factory to instantiate either GRU or LSTM forecaster."""
    cell = cell_type.strip().lower()
    if cell == "gru":
        return TemporalKpiForecaster(**kwargs)
    elif cell == "lstm":
        return TemporalKpiForecasterLSTM(**kwargs)
    else:
        raise ValueError(f"Unknown cell_type: {cell_type}. Use 'gru' or 'lstm'.")


def empty_feature(scenario: str = "normal") -> np.ndarray:
    """Return a neutral feature vector used for history left-padding."""
    frame = StepFrame(
        sim_id="",
        scenario=scenario,
        seed=0,
        episode=0,
        step=0,
        obs=np.zeros((4, 4), dtype=np.float32),
        action=np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32),
        reward=0.0,
        outage_flags=np.zeros(3, dtype=np.float32),
        soft_flags=np.zeros(3, dtype=np.float32),
    )
    return frame_to_feature(frame)


def load_forecaster(path: str, device: torch.device | str = "cpu"):
    """Load a forecaster checkpoint using the factory (GRU or LSTM)."""
    torch_mod, _ = _require_torch()
    ckpt = torch_mod.load(path, map_location=device, weights_only=False)
    cell_type = ckpt.get("cell_type", "gru")
    model = create_forecaster(
        cell_type=cell_type,
        input_dim=ckpt.get("input_dim", FEATURE_DIM),
        hidden_dim=ckpt.get("hidden_dim", 64),
        output_dim=ckpt.get("output_dim", FORECAST_DIM),
        num_layers=ckpt.get("num_layers", 1),
        dropout=ckpt.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, int(ckpt.get("sequence_len", 8)), cell_type


def forecast_from_history(
    model,
    history: Iterable[np.ndarray],
    sequence_len: int,
    scenario: str,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Run the forecaster using left-padding when the history is short."""
    torch_mod, _ = _require_torch()
    items = [np.asarray(item, dtype=np.float32) for item in history]
    pad = [empty_feature(scenario)] * max(0, sequence_len - len(items))
    window = (pad + items)[-sequence_len:]
    x = torch_mod.as_tensor(
        np.stack(window), dtype=torch_mod.float32, device=device
    ).unsqueeze(0)
    model.eval()
    with torch_mod.no_grad():
        return model(x).detach().cpu().numpy().reshape(-1).astype(np.float32)
