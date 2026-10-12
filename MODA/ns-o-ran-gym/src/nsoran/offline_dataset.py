"""Dataset builder for the offline slice-weight policies (xApp line).

Consolidates logged simulation results into one flat contextual-bandit table
(context, action, reward), combining two sources:

  - "metaheuristic": one row per search evaluation (candidate.json sidecar) —
    action = searched weight vector, reward = checkpointed score.
  - "heuristic": one row per baseline run under results_rslaq_network_only —
    reward = nsoran.scoring.score_summary_rows on the run's own summary.csv
    (the exact same objective used to score the metaheuristic evals).

Heuristic action semantics depend on the mode's slice_weight_policy:
  - "static" (psta_equal, slice_weighted_*): the configured weight vector is
    the action (weight_provenance="configured").
  - dynamic policies (slice_aqps, slice_sla_greedy, ...): the weights change
    every window, so the action is the renormalized time-average of
    configured_weight from slice_alloc.csv (weight_provenance="time_averaged").
    Caveat: the reward was produced by the dynamic policy, so this pair is a
    noisy bandit sample, not the exact outcome of the averaged static vector.
  - pure schedulers (pure_rr/pf/bcqi, policy "NA"): excluded — no slice-weight
    enforcement exists in those runs, so attributing their reward to any weight
    vector would poison the regression targets.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd

from nsoran.scoring import normalize_weights, read_summary, score_summary_rows

SCENARIOS = ["low_traffic", "normal", "congestion", "stressed"]
SEEDS = [1, 2, 3]
SLICE_ORDER = ["eMBB", "URLLC", "MTC"]

# Scores at or below this value mark crash-tolerated evaluations.
CRASH_SCORE_THRESHOLD = -1e5


# ---------------------------------------------------------------------------
# Context (state) features
# ---------------------------------------------------------------------------
def _per_slice(meta_value, default=0) -> tuple:
    """Read a per-slice metadata field that may be a dict or a list."""
    if isinstance(meta_value, dict):
        return tuple(meta_value.get(name, default) for name in SLICE_ORDER)
    values = list(meta_value or [])
    values.extend([default] * (3 - len(values)))
    return tuple(values[:3])


def extract_state_features(meta: dict) -> dict:
    """Extract the bandit context from a parsed metadata.json dict."""
    nues = _per_slice(meta.get("num_ues_per_slice", {}))
    load = _per_slice(meta.get("offered_load_mbps_per_slice", {}))
    pkt = _per_slice(meta.get("packet_size_bytes_per_slice", {}))
    total_ues = meta.get("num_ues", sum(int(v) for v in nues))
    return {
        "num_ues_total": int(total_ues),
        "num_ues_embb": int(nues[0]),
        "num_ues_urllc": int(nues[1]),
        "num_ues_mtc": int(nues[2]),
        "offered_load_embb": float(load[0]),
        "offered_load_urllc": float(load[1]),
        "offered_load_mtc": float(load[2]),
        "pkt_size_embb": int(pkt[0]),
        "pkt_size_urllc": int(pkt[1]),
        "pkt_size_mtc": int(pkt[2]),
    }


def load_scenario_metadata(scenario: str, meta_root: Path, heuristics_root: Path | None) -> dict:
    """Find and parse any metadata.json for the scenario (same across seeds)."""
    candidates = list((meta_root / f"scenario={scenario}").rglob("metadata.json"))
    if not candidates and heuristics_root is not None:
        candidates = sorted((heuristics_root / f"scenario={scenario}").rglob("metadata.json"))
    if not candidates:
        raise FileNotFoundError(f"No metadata.json found for scenario {scenario}")
    return json.loads(candidates[0].read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Auxiliary KPI columns from a summary.csv
# ---------------------------------------------------------------------------
def summary_metrics_from_rows(rows: list[dict]) -> dict:
    by_slice = {r.get("slice", ""): r for r in rows}
    result: dict = {}
    for sl in SLICE_ORDER:
        r = by_slice.get(sl, {})
        key = sl.lower()
        result[f"delay_{key}_mean"] = float(r.get("delay_ms_mean", 0) or 0)
        result[f"pdr_{key}"] = float(r.get("pdr_pct", 0) or 0)
        result[f"sla_{key}"] = float(r.get("sla_satisfaction_pct", 0) or 0)
        result[f"thr_{key}"] = float(r.get("throughput_mbps_mean", 0) or 0)
    result["throughput_total"] = sum(result[f"thr_{sl.lower()}"] for sl in SLICE_ORDER)
    result["sla_min"] = min(result[f"sla_{sl.lower()}"] for sl in SLICE_ORDER)
    return result


# ---------------------------------------------------------------------------
# Source 1: metaheuristic search evaluations
# ---------------------------------------------------------------------------
def iter_metaheuristic_rows(
    meta_root: Path,
    scenarios: list[str],
    seeds: list[int],
    stats: dict | None = None,
) -> Iterator[dict]:
    stats = stats if stats is not None else {}
    for scenario in scenarios:
        for seed in seeds:
            evals_dir = meta_root / f"scenario={scenario}/seed={seed}/metaheuristic_search/evals"
            if not evals_dir.exists():
                continue
            for sidecar in sorted(evals_dir.glob("*/candidate.json")):
                try:
                    p = json.loads(sidecar.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    stats["skipped_missing"] = stats.get("skipped_missing", 0) + 1
                    continue
                score = float(p.get("score", -1e6))
                if score <= CRASH_SCORE_THRESHOLD:
                    stats["skipped_crash"] = stats.get("skipped_crash", 0) + 1
                    continue
                weights = list(p.get("weights", [])) + [0.0] * 3
                row = {
                    "scenario": scenario,
                    "seed": seed,
                    "source": "metaheuristic",
                    "method": p.get("method", "?"),
                    "mode": p.get("method", "?"),
                    "weight_provenance": "search",
                    "eval_id": p.get("evaluation_id", 0),
                    "w_embb": float(weights[0]),
                    "w_urllc": float(weights[1]),
                    "w_mtc": float(weights[2]),
                    "score": score,
                }
                summary_path = (sidecar.parent / "results_rslaq_network_only"
                                / f"scenario={scenario}" / "mode=slice_custom"
                                / f"seed={seed}_run=1" / "summary.csv")
                if summary_path.exists():
                    try:
                        row.update(summary_metrics_from_rows(read_summary(summary_path)))
                    except (OSError, ValueError):
                        pass
                yield row


# ---------------------------------------------------------------------------
# Source 2: heuristic / baseline runs
# ---------------------------------------------------------------------------
def _time_averaged_weights(slice_alloc_path: Path) -> list[float] | None:
    """Renormalized per-slice time-average of configured_weight."""
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    with slice_alloc_path.open("r", newline="") as handle:
        for r in csv.DictReader(handle):
            try:
                slice_id = int(r["slice"])
                weight = float(r["configured_weight"])
            except (KeyError, TypeError, ValueError):
                continue
            sums[slice_id] += weight
            counts[slice_id] += 1
    if not all(counts.get(i) for i in range(3)):
        return None
    return normalize_weights([sums[i] / counts[i] for i in range(3)])


def iter_heuristic_rows(
    heuristics_root: Path,
    scenarios: list[str],
    seeds: list[int],
    include_dynamic: bool = True,
    stats: dict | None = None,
) -> Iterator[dict]:
    stats = stats if stats is not None else {}
    for scenario in scenarios:
        scenario_dir = heuristics_root / f"scenario={scenario}"
        if not scenario_dir.exists():
            continue
        for mode_dir in sorted(scenario_dir.glob("mode=*")):
            mode = mode_dir.name.split("=", 1)[1]
            for seed in seeds:
                run_dir = mode_dir / f"seed={seed}_run=1"
                meta_path = run_dir / "metadata.json"
                summary_path = run_dir / "summary.csv"
                if not (meta_path.exists() and summary_path.exists()):
                    stats["skipped_incomplete"] = stats.get("skipped_incomplete", 0) + 1
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    stats["skipped_incomplete"] = stats.get("skipped_incomplete", 0) + 1
                    continue

                policy = str(meta.get("slice_weight_policy", "NA"))
                configured = list(meta.get("slice_weights_configured") or [])
                if policy in ("NA", "") or not configured:
                    # Pure scheduler: no slice-weight semantics — always excluded.
                    stats["skipped_pure"] = stats.get("skipped_pure", 0) + 1
                    continue
                if policy == "static":
                    weights = normalize_weights(configured)
                    provenance = "configured"
                else:
                    if not include_dynamic:
                        stats["skipped_dynamic"] = stats.get("skipped_dynamic", 0) + 1
                        continue
                    weights = _time_averaged_weights(run_dir / "slice_alloc.csv")
                    if weights is None:
                        stats["skipped_incomplete"] = stats.get("skipped_incomplete", 0) + 1
                        continue
                    provenance = "time_averaged"

                summary_rows = read_summary(summary_path)
                score = score_summary_rows(summary_rows)
                if score <= CRASH_SCORE_THRESHOLD:
                    stats["skipped_crash"] = stats.get("skipped_crash", 0) + 1
                    continue

                row = {
                    "scenario": scenario,
                    "seed": seed,
                    "source": "heuristic",
                    "method": mode,
                    "mode": mode,
                    "weight_provenance": provenance,
                    "eval_id": 0,
                    "w_embb": weights[0],
                    "w_urllc": weights[1],
                    "w_mtc": weights[2],
                    "score": score,
                }
                row.update(summary_metrics_from_rows(summary_rows))
                yield row


# ---------------------------------------------------------------------------
# Combined build
# ---------------------------------------------------------------------------
def build_dataset(
    meta_root: Path,
    heuristics_root: Path | None,
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
    include_heuristics: bool = True,
    include_dynamic_modes: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Build the combined dataset; returns (dataframe, skip-stats)."""
    scenarios = scenarios or SCENARIOS
    seeds = seeds or SEEDS
    stats: dict = {}

    rows: list[dict] = []
    for scenario in scenarios:
        state_features = extract_state_features(
            load_scenario_metadata(scenario, meta_root, heuristics_root))
        scenario_rows: list[dict] = []
        scenario_rows.extend(iter_metaheuristic_rows(meta_root, [scenario], seeds, stats))
        if include_heuristics and heuristics_root is not None:
            scenario_rows.extend(iter_heuristic_rows(
                heuristics_root, [scenario], seeds, include_dynamic_modes, stats))
        for row in scenario_rows:
            row.update(state_features)
        rows.extend(scenario_rows)

    df = pd.DataFrame(rows)
    if not df.empty:
        for sc in SCENARIOS:
            df[f"is_{sc}"] = (df["scenario"] == sc).astype(int)
    return df, stats
