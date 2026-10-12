#!/usr/bin/env python3
"""Compare RSLAQ campaign outputs with SLA-first metrics."""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_run_name(name: str):
    m = re.match(r"sac_(paper|resource_efficient)_(.*)_seed(\d+)$", name)
    if m:
        return "sac", m.group(2), int(m.group(3)), m.group(1)
    m = re.match(r"ddqn_paper_(.*)_seed(\d+)$", name)
    if m:
        return "ddqn", m.group(1), int(m.group(2)), "paper"
    m = re.match(r"(predictive_sac|ddqn|sac)_(.*)_seed(\d+)$", name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3)), None


def training_log_summary(run_dir: Path, algo: str):
    if algo == "predictive_sac":
        path = run_dir / "predictive_sac_training_log.csv"
        reward_col = "base_reward"
        avg_col = "avg_shaped_reward"
    else:
        path = run_dir / f"{algo}_training_log.csv"
        reward_col = "total_reward"
        avg_col = "avg_reward"
    if not path.exists():
        return {}
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    tail = rows[-10:] if len(rows) >= 10 else rows
    return {
        "episodes": len(rows),
        "tail_reward_mean": sum(safe_float(r.get(reward_col)) for r in tail) / len(tail),
        "tail_steps_mean": sum(safe_float(r.get("steps")) for r in tail) / len(tail),
        "tail_outages_mean": sum(safe_float(r.get("outage_count")) for r in tail) / len(tail),
        "tail_soft_mean": sum(safe_float(r.get("soft_count")) for r in tail) / len(tail),
        "final_avg_reward": safe_float(rows[-1].get(avg_col)),
    }


def step_metrics_summary(run_dir: Path):
    totals = {
        "step_count": 0,
        "violation_steps": 0,
        "outage_steps": 0,
        "soft_steps": 0,
        "thr": [0.0, 0.0, 0.0],
        "buf": [0.0, 0.0, 0.0],
        "plr": [0.0, 0.0, 0.0],
        "lost": [0.0, 0.0, 0.0],
        "rsh": [0.0, 0.0, 0.0],
        "act": [0.0, 0.0, 0.0],
        "resource_efficiency": 0.0,
        "need_allocation_match": 0.0,
        "over_allocation": 0.0,
        "under_allocation": 0.0,
        "action_smoothness_penalty": 0.0,
        "resource_efficient_shaping": 0.0,
        "slice_rows": [0, 0, 0],
    }
    for path in run_dir.glob("*/step_metrics.csv"):
        grouped = defaultdict(list)
        with path.open("r", newline="") as f:
            for row in csv.DictReader(f):
                grouped[(row.get("sim_id"), row.get("step"))].append(row)
        for rows in grouped.values():
            totals["step_count"] += 1
            has_outage = any(str(r.get("outage_flag")) == "True" for r in rows)
            has_soft = any(str(r.get("soft_flag")) == "True" for r in rows)
            totals["violation_steps"] += int(has_outage or has_soft)
            totals["outage_steps"] += int(has_outage)
            totals["soft_steps"] += int(has_soft)
            first = rows[0]
            totals["act"][0] += safe_float(first.get("action_embb"))
            totals["act"][1] += safe_float(first.get("action_urllc"))
            totals["act"][2] += safe_float(first.get("action_mtc"))
            totals["resource_efficiency"] += safe_float(first.get("resource_efficiency"))
            totals["need_allocation_match"] += safe_float(first.get("need_allocation_match"))
            totals["over_allocation"] += safe_float(first.get("over_allocation"))
            totals["under_allocation"] += safe_float(first.get("under_allocation"))
            totals["action_smoothness_penalty"] += safe_float(first.get("action_smoothness_penalty"))
            totals["resource_efficient_shaping"] += safe_float(first.get("resource_efficient_shaping"))
            for row in rows:
                sid = safe_int(row.get("slice_id"), -1)
                if sid not in (0, 1, 2):
                    continue
                totals["slice_rows"][sid] += 1
                totals["thr"][sid] += safe_float(row.get("throughput_mbps"))
                totals["buf"][sid] += safe_float(row.get("bufferBytes_max"))
                totals["plr"][sid] += safe_float(row.get("plr_pct"))
                totals["lost"][sid] += safe_float(row.get("dLostPackets"))
                totals["rsh"][sid] += safe_float(row.get("resourceSharePct"))

    steps = max(totals["step_count"], 1)
    result = {
        "sla_reliability": 1.0 - totals["violation_steps"] / steps,
        "outage_step_rate": totals["outage_steps"] / steps,
        "soft_step_rate": totals["soft_steps"] / steps,
        "action_embb": totals["act"][0] / steps,
        "action_urllc": totals["act"][1] / steps,
        "action_mtc": totals["act"][2] / steps,
        "resource_efficiency": totals["resource_efficiency"] / steps,
        "need_allocation_match": totals["need_allocation_match"] / steps,
        "over_allocation": totals["over_allocation"] / steps,
        "under_allocation": totals["under_allocation"] / steps,
        "action_smoothness_penalty": totals["action_smoothness_penalty"] / steps,
        "resource_efficient_shaping": totals["resource_efficient_shaping"] / steps,
    }
    names = ["embb", "urllc", "mtc"]
    for sid, name in enumerate(names):
        n = max(totals["slice_rows"][sid], 1)
        result[f"{name}_thr_mean"] = totals["thr"][sid] / n
        result[f"{name}_buf_max_mean"] = totals["buf"][sid] / n
        result[f"{name}_plr_mean"] = totals["plr"][sid] / n
        result[f"{name}_lost_mean"] = totals["lost"][sid] / n
        result[f"{name}_rsh_mean"] = totals["rsh"][sid] / n
    return result


def compute_summary(run_dir: Path, algo: str):
    summary_file = {
        "ddqn": "ddqn_summary.json",
        "sac": "sac_summary.json",
        "predictive_sac": "predictive_sac_summary.json",
    }.get(algo)
    if not summary_file:
        return {}
    path = run_dir / summary_file
    if not path.exists():
        return {}
    try:
        compute = json.loads(path.read_text()).get("compute", {})
    except json.JSONDecodeError:
        return {}
    training = compute.get("training", {})
    xapp = compute.get("xapp_runtime", {})
    cost = compute.get("cost_estimate", {})
    return {
        "compute_wall_time_s": safe_float(training.get("wall_time_s")),
        "compute_cpu_time_s": safe_float(training.get("cpu_time_s")),
        "compute_child_cpu_time_s": safe_float(training.get("child_cpu_time_s")),
        "compute_peak_rss_mb": safe_float(training.get("peak_rss_mb")),
        "compute_steps_per_second": safe_float(training.get("steps_per_second")),
        "xapp_inference_ms_mean": safe_float(xapp.get("inference_ms_mean")),
        "xapp_inference_ms_p95": safe_float(xapp.get("inference_ms_p95")),
        "xapp_deadline_usage_pct": safe_float(xapp.get("deadline_usage_pct")),
        "compute_cloud_cost_usd": safe_float(cost.get("cloud_cost_usd")),
        "compute_cost_per_1000_steps_usd": safe_float(cost.get("cost_per_1000_steps_usd")),
        "compute_energy_kwh": safe_float(cost.get("energy_kwh")),
        "compute_energy_cost_usd": safe_float(cost.get("energy_cost_usd")),
    }


def collect(root: Path, line_name: str):
    rows = []
    if not root.exists():
        return rows
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        parsed = parse_run_name(run_dir.name)
        if not parsed:
            continue
        algo, scenario, seed, parsed_line = parsed
        row = {
            "line": parsed_line or line_name,
            "algo": algo,
            "scenario": scenario,
            "seed": seed,
            "run_dir": str(run_dir),
        }
        row.update(training_log_summary(run_dir, algo))
        row.update(step_metrics_summary(run_dir))
        row.update(compute_summary(run_dir, algo))
        rows.append(row)
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["line"], row["algo"], row["scenario"])].append(row)
    out = []
    numeric_keys = [
        "episodes", "tail_reward_mean", "tail_steps_mean", "tail_outages_mean",
        "tail_soft_mean", "final_avg_reward", "sla_reliability",
        "outage_step_rate", "soft_step_rate", "action_embb", "action_urllc",
        "action_mtc", "embb_thr_mean", "urllc_thr_mean", "mtc_thr_mean",
        "embb_buf_max_mean", "urllc_buf_max_mean", "mtc_buf_max_mean",
        "embb_plr_mean", "urllc_plr_mean", "mtc_plr_mean",
        "embb_lost_mean", "urllc_lost_mean", "mtc_lost_mean",
        "resource_efficiency", "need_allocation_match", "over_allocation",
        "under_allocation", "action_smoothness_penalty",
        "resource_efficient_shaping",
        "compute_wall_time_s", "compute_cpu_time_s", "compute_child_cpu_time_s",
        "compute_peak_rss_mb", "compute_steps_per_second",
        "xapp_inference_ms_mean", "xapp_inference_ms_p95",
        "xapp_deadline_usage_pct", "compute_cloud_cost_usd",
        "compute_cost_per_1000_steps_usd", "compute_energy_kwh",
        "compute_energy_cost_usd",
    ]
    for key, items in sorted(groups.items()):
        row = {"line": key[0], "algo": key[1], "scenario": key[2], "runs": len(items)}
        for name in numeric_keys:
            vals = [safe_float(item.get(name)) for item in items if name in item]
            if vals:
                row[name] = sum(vals) / len(vals)
        out.append(row)
    return out


def write_csv(path: Path, rows):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare RSLAQ controlled campaigns")
    parser.add_argument("--baseline_root", required=True)
    parser.add_argument("--predictive_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(collect(Path(args.baseline_root), "paper_faithful"))
    rows.extend(collect(Path(args.predictive_root), "predictive_forecaster_sac"))
    summary = aggregate(rows)

    write_csv(output / "comparison_runs.csv", rows)
    write_csv(output / "comparison_summary.csv", summary)
    with (output / "comparison_summary.json").open("w") as f:
        json.dump({"runs": rows, "summary": summary}, f, indent=2)
    print(f"Comparison written to {output}")


if __name__ == "__main__":
    main()
