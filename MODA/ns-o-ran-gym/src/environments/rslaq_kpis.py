"""
RSLAQ KPI Parser and Observation Builder.

Reads per-UE KPMs produced by ns-3 (rslaq-kpms.txt) and builds the
observation matrix used by the DRL agent.
"""

import csv
import glob
import os
import warnings
from typing import Dict, Any
import numpy as np

from .rslaq_slice_ids import normalize_slice_id, get_num_slices

# Default normalization caps
DEFAULT_MAX_BTX = 200000.0   # bytes (per 10ms period)
DEFAULT_MAX_BUFFER_BYTES = 100000.0  # bytes (buffer real)
DEFAULT_MAX_TDP = 1000.0     # packets (proxy for dropped bytes)


def parse_kpm_file(
    kpm_path: str,
    last_timestamp: int = 0,
) -> Dict[int, Dict[str, Any]]:
    """
    Parse rslaq-kpms.txt and return aggregated metrics per slice + cell total.
    """
    slice_data: Dict[int, Dict[str, float]] = {
        0: {"dTxBytes_sum": 0.0, "dRxBytes_sum": 0.0, "plr_sum": 0.0,
            "bufferBytes_sum": 0.0, "bufferBytes_max": 0.0,
            "resourceSharePct_sum": 0.0, "dLostPackets_sum": 0.0,
            "throughputMbps_sum": 0.0, "ue_count": 0.0},
        1: {"dTxBytes_sum": 0.0, "dRxBytes_sum": 0.0, "plr_sum": 0.0,
            "bufferBytes_sum": 0.0, "bufferBytes_max": 0.0,
            "resourceSharePct_sum": 0.0, "dLostPackets_sum": 0.0,
            "throughputMbps_sum": 0.0, "ue_count": 0.0},
        2: {"dTxBytes_sum": 0.0, "dRxBytes_sum": 0.0, "plr_sum": 0.0,
            "bufferBytes_sum": 0.0, "bufferBytes_max": 0.0,
            "resourceSharePct_sum": 0.0, "dLostPackets_sum": 0.0,
            "throughputMbps_sum": 0.0, "ue_count": 0.0},
    }

    latest_ts = last_timestamp

    if not os.path.exists(kpm_path):
        return _build_return_dict(slice_data, latest_ts)

    rows_to_process = []
    raw_slice_ids = set()
    with open(kpm_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = int(row["timestamp"])
            except (KeyError, ValueError):
                continue
            if ts < last_timestamp:
                continue
            latest_ts = max(latest_ts, ts)

            try:
                raw_slice_id = int(row.get("sliceId", -1))
            except (ValueError, KeyError):
                continue

            rows_to_process.append((row, raw_slice_id))
            raw_slice_ids.add(raw_slice_id)

    legacy_one_based = 0 not in raw_slice_ids and 3 in raw_slice_ids
    if legacy_one_based:
        warnings.warn(
            "Detected legacy 1-based slice ids in KPM file. Converting them to 0-based ids.",
            stacklevel=2,
        )

    for row, raw_slice_id in rows_to_process:
        try:
            if legacy_one_based:
                slice_id = raw_slice_id - 1
                if slice_id not in range(get_num_slices()):
                    raise ValueError
            else:
                slice_id = normalize_slice_id(raw_slice_id)
        except ValueError:
            continue

        if slice_id not in slice_data:
            continue

        sd = slice_data[slice_id]
        sd["dTxBytes_sum"] += float(row.get("dTxBytes", 0.0))
        sd["dRxBytes_sum"] += float(row.get("dRxBytes", 0.0))
        sd["plr_sum"] += float(row.get("plr", 0.0))

        # Extração do Buffer e rastreio do Valor Máximo (Para Equação 17)
        ue_buffer = float(row.get("bufferBytes", 0.0))
        sd["bufferBytes_sum"] += ue_buffer
        sd["bufferBytes_max"] = max(sd["bufferBytes_max"], ue_buffer)

        sd["resourceSharePct_sum"] += float(row.get("resourceSharePct", 0.0))
        sd["dLostPackets_sum"] += float(row.get("dLostPackets", 0.0))
        sd["throughputMbps_sum"] += float(row.get("throughputMbps", 0.0))
        sd["ue_count"] += 1.0

    return _build_return_dict(slice_data, latest_ts)


def _build_return_dict(
    slice_data: Dict[int, Dict[str, float]],
    latest_ts: int,
) -> Dict[int, Dict[str, Any]]:
    """Build the aggregated return dict with averages and cell totals."""
    result: Dict[int, Dict[str, Any]] = {}
    cell = {
        "dTxBytes_sum": 0.0,
        "dRxBytes_sum": 0.0,
        "plr_mean": 0.0,
        "bufferBytes_mean": 0.0,
        "bufferBytes_max": 0.0,
        "resourceSharePct_mean": 0.0,
        "dLostPackets_sum": 0.0,
        "throughputMbps_sum": 0.0,
        "ue_count": 0.0,
    }

    for sid in range(get_num_slices()):
        sd = slice_data[sid]
        n = sd["ue_count"]
        if n > 0:
            agg = {
                "dTxBytes_sum": sd["dTxBytes_sum"],
                "dRxBytes_sum": sd["dRxBytes_sum"],
                "plr_mean": sd["plr_sum"] / n,
                "bufferBytes_mean": sd["bufferBytes_sum"] / n,
                "bufferBytes_max": sd["bufferBytes_max"],
                "resourceSharePct_mean": sd["resourceSharePct_sum"] / n,
                "dLostPackets_sum": sd["dLostPackets_sum"],
                "throughputMbps_sum": sd["throughputMbps_sum"],
                "ue_count": int(n),
            }
        else:
            agg = {
                "dTxBytes_sum": 0.0,
                "dRxBytes_sum": 0.0,
                "plr_mean": 0.0,
                "bufferBytes_mean": 0.0,
                "bufferBytes_max": 0.0,
                "resourceSharePct_mean": 0.0,
                "dLostPackets_sum": 0.0,
                "throughputMbps_sum": 0.0,
                "ue_count": 0,
            }
        result[sid] = agg

        cell["dTxBytes_sum"] += agg["dTxBytes_sum"]
        cell["dRxBytes_sum"] += agg["dRxBytes_sum"]
        cell["dLostPackets_sum"] += agg["dLostPackets_sum"]
        cell["throughputMbps_sum"] += agg["throughputMbps_sum"]
        cell["ue_count"] += agg["ue_count"]
        cell["bufferBytes_max"] = max(cell["bufferBytes_max"], agg["bufferBytes_max"])

    total_ue = cell["ue_count"]
    if total_ue > 0:
        cell_plr_weighted = sum(
            result[sid]["plr_mean"] * result[sid]["ue_count"] for sid in range(get_num_slices())
        ) / total_ue
        cell_rsh_weighted = sum(
            result[sid]["resourceSharePct_mean"] * result[sid]["ue_count"] for sid in range(get_num_slices())
        ) / total_ue
        cell_buffer_weighted = sum(
            result[sid]["bufferBytes_mean"] * result[sid]["ue_count"] for sid in range(get_num_slices())
        ) / total_ue

        cell["plr_mean"] = cell_plr_weighted
        cell["resourceSharePct_mean"] = cell_rsh_weighted
        cell["bufferBytes_mean"] = cell_buffer_weighted
    else:
        cell["plr_mean"] = 0.0
        cell["resourceSharePct_mean"] = 0.0
        cell["bufferBytes_mean"] = 0.0

    result[3] = cell
    result["_latest_timestamp"] = latest_ts
    return result


def build_observation(
    kpi_dict: Dict[int, Dict[str, Any]],
    mode: str = "paper",
    max_btx: float = DEFAULT_MAX_BTX,
    max_tdp: float = DEFAULT_MAX_TDP,
    max_buffer_bytes: float = DEFAULT_MAX_BUFFER_BYTES,
    use_proxy_bfs: bool = False,
) -> np.ndarray:
    """Build the observation matrix from parsed KPIs."""
    if mode == "paper":
        obs = np.zeros((4, 4), dtype=np.float32)
        for col, sid in enumerate(range(get_num_slices())):
            metrics = kpi_dict.get(sid, {})
            obs[0, col] = _clip_norm(metrics.get("dTxBytes_sum", 0.0), max_btx)
            if use_proxy_bfs:
                obs[1, col] = _clip_norm(metrics.get("plr_mean", 0.0), 100.0)
            else:
                obs[1, col] = _clip_norm(metrics.get("bufferBytes_mean", 0.0), max_buffer_bytes)
            obs[2, col] = _clip_norm(metrics.get("resourceSharePct_mean", 0.0), 100.0)
            obs[3, col] = _clip_norm(metrics.get("dLostPackets_sum", 0.0), max_tdp)

        cell = kpi_dict.get(3, {})
        obs[0, 3] = _clip_norm(cell.get("dTxBytes_sum", 0.0), max_btx * get_num_slices())
        if use_proxy_bfs:
            obs[1, 3] = _clip_norm(cell.get("plr_mean", 0.0), 100.0)
        else:
            obs[1, 3] = _clip_norm(cell.get("bufferBytes_mean", 0.0), max_buffer_bytes * get_num_slices())
        obs[2, 3] = _clip_norm(cell.get("resourceSharePct_mean", 0.0), 100.0)
        obs[3, 3] = _clip_norm(cell.get("dLostPackets_sum", 0.0), max_tdp * get_num_slices())
        return obs

    elif mode == "debug":
        obs = np.zeros((3, 5), dtype=np.float32)
        for row, sid in enumerate(range(get_num_slices())):
            metrics = kpi_dict.get(sid, {})
            thr = metrics.get("throughputMbps_sum", 0.0)
            obs[row, 0] = min(thr / 150.0, 1.0)
            obs[row, 1] = _clip_norm(metrics.get("dTxBytes_sum", 0.0), max_btx)
            obs[row, 2] = _clip_norm(metrics.get("plr_mean", 0.0), 100.0)
            obs[row, 3] = _clip_norm(metrics.get("resourceSharePct_mean", 0.0), 100.0)
            obs[row, 4] = _clip_norm(metrics.get("dLostPackets_sum", 0.0), max_tdp)
        return obs
    else:
        raise ValueError(f"Unknown observation mode: {mode}. Use 'paper' or 'debug'.")


def _clip_norm(value: float, cap: float) -> float:
    """Normalize value by cap and clip to [0, 1]."""
    if cap <= 0:
        return 0.0
    return float(np.clip(value / cap, 0.0, 1.0))
