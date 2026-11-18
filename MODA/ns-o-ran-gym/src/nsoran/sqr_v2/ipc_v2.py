"""Atomic IPC Handshake v2 envelope & action payload helpers (BUG-03 + BUG-05)."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any

IPC_V2_INDICATION_FILENAME = "sqr_ipc_indication_v2.json"
IPC_V2_ACTION_FILENAME = "sqr_ipc_action_v2.csv"
IPC_V2_HISTORY_FILENAME = "SqrActions_v2_history.csv"
IPC_V2_END_SENTINEL = "# END_V2"

ACTION_V2_HEADER = (
    "run_id",
    "window_seq",
    "sim_time_ms",
    "slice_id",
    "prb_proposed",
    "prb_requested",
    "prb_allocated",
    "dedicated_pct",
    "algorithm",
    "agent_decision_wall_ms",
)


@dataclass(frozen=True)
class IpcIndicationV2:
    """Validated snapshot of ``sqr_ipc_indication_v2.json``."""

    protocol_version: int
    run_id: str
    window_seq: int
    sim_time_ms: int
    period_ms: int
    total_rbg: int
    total_prb: int
    active_ues: int
    kpm_file: str
    governed_prev_window_seq: int
    prev_applied_action_seq: int
    prev_ipc_status: str
    prev_prb_executed_rbg: tuple[int, ...]
    prev_decision_wall_ms: float
    recent_windows_tail: tuple[dict[str, Any], ...]


def read_indication_v2(
    path: str, *, expected_run_id: str | None = None
) -> IpcIndicationV2:
    """Read and validate ``sqr_ipc_indication_v2.json``."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    proto = int(data["protocol_version"])
    if proto != 2:
        raise ValueError(f"Expected protocol_version=2, got {proto}")

    run_id = str(data["run_id"])
    if not run_id:
        raise ValueError("Indication envelope missing run_id")
    if expected_run_id is not None and run_id != str(expected_run_id):
        raise ValueError(
            f"Cross-run envelope mismatch: expected run_id={expected_run_id!r}, got {run_id!r}"
        )

    return IpcIndicationV2(
        protocol_version=proto,
        run_id=run_id,
        window_seq=int(data["window_seq"]),
        sim_time_ms=int(data["sim_time_ms"]),
        period_ms=int(data["period_ms"]),
        total_rbg=int(data["total_rbg"]),
        total_prb=int(data["total_prb"]),
        active_ues=int(data.get("active_ues", 0)),
        kpm_file=str(data.get("kpm_file", "sqr-kpms.txt")),
        governed_prev_window_seq=int(data.get("governed_prev_window_seq", 0)),
        prev_applied_action_seq=int(data.get("prev_applied_action_seq", 0)),
        prev_ipc_status=str(data.get("prev_ipc_status", "INIT")),
        prev_prb_executed_rbg=tuple(int(v) for v in data.get("prev_prb_executed_rbg", (0, 0, 0))),
        prev_decision_wall_ms=float(data.get("prev_decision_wall_ms", 0.0)),
        recent_windows_tail=tuple(data.get("recent_windows_tail", ())),
    )


def format_action_v2_csv(
    *,
    run_id: str,
    window_seq: int,
    sim_time_ms: int,
    slice_rows: list[tuple[int, int, int, int, float, int]],
    agent_decision_wall_ms: float,
    include_end_sentinel: bool = True,
) -> str:
    """Format the O(1) constant-size action CSV payload for one window."""
    if not run_id:
        raise ValueError("run_id must be non-empty")
    if int(window_seq) < 1:
        raise ValueError(f"window_seq must be >= 1, got {window_seq}")
    lines = [",".join(ACTION_V2_HEADER)]
    for sid, prb_prop, prb_req, prb_alloc, ded_pct, algo in slice_rows:
        lines.append(
            f"{run_id},{int(window_seq)},{int(sim_time_ms)},{int(sid)},"
            f"{int(prb_prop)},{int(prb_req)},{int(prb_alloc)},{float(ded_pct):.6f},"
            f"{int(algo)},{float(agent_decision_wall_ms):.4f}"
        )
    if include_end_sentinel:
        lines.append(IPC_V2_END_SENTINEL)
    return "\n".join(lines) + "\n"


def write_action_v2_atomic(
    sim_dir: str,
    *,
    run_id: str,
    window_seq: int,
    sim_time_ms: int,
    slice_rows: list[tuple[int, int, int, int, float, int]],
    agent_decision_wall_ms: float,
    append_history: bool = True,
) -> str:
    """Atomically write ``sqr_ipc_action_v2.csv`` (tmp -> replace) and log history."""
    payload = format_action_v2_csv(
        run_id=run_id,
        window_seq=window_seq,
        sim_time_ms=sim_time_ms,
        slice_rows=slice_rows,
        agent_decision_wall_ms=agent_decision_wall_ms,
        include_end_sentinel=True,
    )
    target = os.path.join(sim_dir, IPC_V2_ACTION_FILENAME)
    tmp = f"{target}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(payload)
        f.flush()
    os.replace(tmp, target)

    if append_history:
        hist = os.path.join(sim_dir, IPC_V2_HISTORY_FILENAME)
        need_header = not os.path.isfile(hist)
        with open(hist, "a", encoding="utf-8", newline="") as hf:
            writer = csv.writer(hf)
            if need_header:
                writer.writerow(ACTION_V2_HEADER)
            for sid, prb_prop, prb_req, prb_alloc, ded_pct, algo in slice_rows:
                writer.writerow([
                    run_id,
                    int(window_seq),
                    int(sim_time_ms),
                    int(sid),
                    int(prb_prop),
                    int(prb_req),
                    int(prb_alloc),
                    f"{float(ded_pct):.6f}",
                    int(algo),
                    f"{float(agent_decision_wall_ms):.4f}",
                ])
            hf.flush()
    return target


def reconcile_prev_receipt(
    ind: IpcIndicationV2,
    *,
    expected_run_id: str,
    sent_window_seq: int,
) -> dict[str, Any]:
    """Reconcile the physical MAC execution receipt after waiting for the next boundary."""
    if ind.run_id != str(expected_run_id):
        raise ValueError(
            f"run_id mismatch on reconciliation: expected={expected_run_id!r}, got={ind.run_id!r}"
        )

    expected_boundary = int(sent_window_seq) + 1
    window_gap = int(ind.window_seq) - expected_boundary

    # Locate the receipt explicitly governing ``sent_window_seq``.
    receipt: dict[str, Any] | None = None
    if ind.governed_prev_window_seq == int(sent_window_seq):
        receipt = {
            "governed_window_seq": ind.governed_prev_window_seq,
            "applied_action_seq": ind.prev_applied_action_seq,
            "ipc_status": ind.prev_ipc_status,
            "prb_executed_rbg": list(ind.prev_prb_executed_rbg),
            "decision_wall_ms": ind.prev_decision_wall_ms,
        }
    else:
        for tail_item in ind.recent_windows_tail:
            if int(tail_item.get("governed_window_seq", -1)) == int(sent_window_seq):
                receipt = {
                    "governed_window_seq": int(tail_item["governed_window_seq"]),
                    "applied_action_seq": int(tail_item["applied_action_seq"]),
                    "ipc_status": str(tail_item["ipc_status"]),
                    "prb_executed_rbg": [int(v) for v in tail_item["prb_executed_rbg"]],
                    "decision_wall_ms": float(tail_item["decision_wall_ms"]),
                }
                break

    if receipt is None:
        # Extreme multi-window jump > len(tail): report explicit JUMP status
        return {
            "sent_window_seq": int(sent_window_seq),
            "next_window_seq": int(ind.window_seq),
            "window_gap": window_gap,
            "governed_prev_window_seq": int(ind.governed_prev_window_seq),
            "applied_action_seq": int(ind.prev_applied_action_seq),
            "ipc_status": "MULTI_WINDOW_GAP",
            "prb_executed_mac": list(ind.prev_prb_executed_rbg),
            "ns3_wait_wall_ms": float(ind.prev_decision_wall_ms),
            "receipt_matched": False,
        }

    return {
        "sent_window_seq": int(sent_window_seq),
        "next_window_seq": int(ind.window_seq),
        "window_gap": window_gap,
        "governed_prev_window_seq": receipt["governed_window_seq"],
        "applied_action_seq": receipt["applied_action_seq"],
        "ipc_status": receipt["ipc_status"],
        "prb_executed_mac": list(receipt["prb_executed_rbg"]),
        "ns3_wait_wall_ms": receipt["decision_wall_ms"],
        "receipt_matched": True,
    }
