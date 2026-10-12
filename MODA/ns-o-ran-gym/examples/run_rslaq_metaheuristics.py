#!/usr/bin/env python3
"""Offline meta-heuristic search for RSLAQ slice_custom PRB weights.

This runner keeps the ns-3 baselines intact and uses repeated network-only
simulations to optimize the slice PRB vector [eMBB, URLLC, MTC]. GA, PSO, SA
and the hybrid method operate only on the weight simplex; the ns-3 scheduler
still applies the selected intra-slice algorithm through the existing
slice_custom mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsoran.scoring import (
    SLICE_NAMES,
    clamp,
    format_weights,
    normalize_weights,
    read_summary,
    safe_float,
    score_summary_rows,
    score_summary_rows_v2,
)

# Selecionável via --score_version. v1 é o default (compatibilidade com a
# campanha e o cache antigos); a campanha v2 (correções da auditoria) usa v2.
SCORE_FUNCTIONS = {"v1": score_summary_rows, "v2": score_summary_rows_v2}


@dataclass
class Evaluation:
    method: str
    evaluation_id: int
    scenario: str
    seed: int
    run: int
    weights: list[float]
    score: float
    result_dir: Path
    log_file: Path
    # Camera-ready search-cost instrumentation (WPMC 2026): wall-clock of the
    # ns-3 execution (0.0 when served from cache) and whether the result was
    # recovered from the checkpoint cache. Cost analyses must only sum
    # elapsed_s over evaluations with cached=False.
    elapsed_s: float = 0.0
    cached: bool = False


@dataclass
class Particle:
    position: list[float]
    velocity: list[float]
    personal_best: list[float]
    personal_score: float = -math.inf


def build_sim_command(
    *,
    scenario: str,
    output_root: Path,
    weights: Sequence[float],
    intra_algo: str,
    sim_time: float,
    app_start: float,
    drain_time: float,
    period_ms: int,
    seed: int,
    run: int,
    tx_power: float = 43.0,
    tdd_pattern: str = "D|D|8D|4GB|4U|U|U",
    rlc_mode: str = "um",
) -> str:
    return (
        "scratch/rslaq/rslaq-sim "
        f"--scenario={scenario} "
        "--baselineMode=slice_custom "
        f"--weights={format_weights(weights)} "
        f"--intraAlgo={intra_algo} "
        f"--simTime={sim_time:g} "
        f"--appStart={app_start:g} "
        f"--drainTimeSec={drain_time:g} "
        f"--periodMs={period_ms} "
        f"--seed={seed} "
        f"--run={run} "
        f"--txPower={tx_power:g} "
        f"--tddPattern={tdd_pattern} "
        f"--rlcMode={rlc_mode} "
        f"--outputDir={output_root}"
    )


def random_weights(rng: random.Random) -> list[float]:
    return normalize_weights([rng.random(), rng.random(), rng.random()])


def mutate_weights(weights: Sequence[float], rng: random.Random, strength: float) -> list[float]:
    return normalize_weights([w + rng.gauss(0.0, strength) for w in weights])


def crossover_weights(a: Sequence[float], b: Sequence[float], rng: random.Random) -> list[float]:
    alpha = rng.random()
    return normalize_weights([alpha * x + (1.0 - alpha) * y for x, y in zip(a, b)])


def seed_weight_candidates(rng: random.Random, population_size: int) -> list[list[float]]:
    priors = [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.3333, 0.4000, 0.2667],
        [0.70, 0.05, 0.25],
        [0.60, 0.15, 0.25],
        [0.55, 0.25, 0.20],
        [0.45, 0.35, 0.20],
        [0.50, 0.10, 0.40],
        [0.25, 0.45, 0.30],
    ]
    candidates: list[list[float]] = []
    seen: set[tuple[float, float, float]] = set()
    for candidate in priors:
        normalized = normalize_weights(candidate)
        key = tuple(round(value, 6) for value in normalized)
        if key not in seen:
            candidates.append(normalized)
            seen.add(key)
        if len(candidates) >= population_size:
            return candidates

    while len(candidates) < population_size:
        candidates.append(random_weights(rng))
    return candidates


def summary_path(output_root: Path, scenario: str, seed: int, run: int) -> Path:
    return (
        output_root
        / "results_rslaq_network_only"
        / f"scenario={scenario}"
        / "mode=slice_custom"
        / f"seed={seed}_run={run}"
        / "summary.csv"
    )


def result_dir(output_root: Path, scenario: str, seed: int, run: int) -> Path:
    return summary_path(output_root, scenario, seed, run).parent


def load_cached_evaluation(
    candidate_root: Path,
    method: str,
    evaluation_id: int,
    requested_weights: Sequence[float],
    score_version: str = "v1",
) -> Evaluation | None:
    """Return a cached Evaluation for this (method, eval_id) if a valid
    candidate.json sidecar exists.

    The cache is valid only when the requested weights match the cached ones
    (within 1e-9) AND the cached score_version matches the requested one. A
    weight mismatch means the RNG trajectory has diverged; a score_version
    mismatch means the sidecar was produced by a different objective. In either
    case the cache is discarded and the caller re-runs the ns-3 evaluation.
    (Sidecars written before v2 lack the field and default to "v1".)
    """
    sidecar = candidate_root / "candidate.json"
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    cached_weights = payload.get("weights")
    if not isinstance(cached_weights, list) or len(cached_weights) != len(requested_weights):
        return None
    if any(abs(float(cached_weights[i]) - float(requested_weights[i])) > 1e-9
           for i in range(len(requested_weights))):
        return None
    if str(payload.get("method")) != method or int(payload.get("evaluation_id", -1)) != evaluation_id:
        return None
    if str(payload.get("score_version", "v1")) != score_version:
        return None
    return Evaluation(
        method=str(payload["method"]),
        evaluation_id=int(payload["evaluation_id"]),
        scenario=str(payload["scenario"]),
        seed=int(payload["seed"]),
        run=int(payload["run"]),
        weights=[float(w) for w in payload["weights"]],
        score=float(payload["score"]),
        result_dir=Path(payload["result_dir"]),
        log_file=Path(payload["log_file"]),
        elapsed_s=float(payload.get("elapsed_s", 0.0)),
        cached=True,
    )


def run_candidate(
    *,
    ns3_dir: Path,
    eval_root: Path,
    method: str,
    evaluation_id: int,
    scenario: str,
    weights: Sequence[float],
    intra_algo: str,
    sim_time: float,
    app_start: float,
    drain_time: float,
    period_ms: int,
    seed: int,
    run: int,
    tx_power: float,
    tdd_pattern: str,
    rlc_mode: str,
    score_fn=score_summary_rows,
    score_version: str = "v1",
) -> Evaluation:
    candidate_root = eval_root / f"{method}_eval_{evaluation_id:04d}"
    candidate_root.mkdir(parents=True, exist_ok=True)
    log_file = candidate_root / "ns3.log"
    command = build_sim_command(
        scenario=scenario,
        output_root=candidate_root,
        weights=weights,
        intra_algo=intra_algo,
        sim_time=sim_time,
        app_start=app_start,
        drain_time=drain_time,
        period_ms=period_ms,
        seed=seed,
        run=run,
        tx_power=tx_power,
        tdd_pattern=tdd_pattern,
        rlc_mode=rlc_mode,
    )

    with log_file.open("w") as log:
        t_start = time.perf_counter()
        completed = subprocess.run(
            ["./ns3", "run", command],
            cwd=ns3_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        elapsed_s = time.perf_counter() - t_start
    if completed.returncode != 0:
        # The ns-3 simulator occasionally hits an internal NS_ASSERT bug in the
        # RLC UM reassembly path (SIGABRT). Rather than aborting the whole
        # (scenario, seed) pair and discarding every other valid evaluation, we
        # treat a crashed simulation as a hard penalty: the candidate receives a
        # strongly negative score so the optimizer avoids that weight region,
        # and the result is checkpointed (failed=True) so a resume does not
        # retry the same crashing eval forever.
        return _failed_evaluation(
            eval_root=eval_root,
            candidate_root=candidate_root,
            log_file=log_file,
            method=method,
            evaluation_id=evaluation_id,
            scenario=scenario,
            seed=seed,
            run=run,
            weights=weights,
            reason=f"ns-3 exit code {completed.returncode}",
            score_version=score_version,
            elapsed_s=elapsed_s,
        )

    out_summary = summary_path(candidate_root, scenario, seed, run)
    if not out_summary.exists():
        return _failed_evaluation(
            eval_root=eval_root,
            candidate_root=candidate_root,
            log_file=log_file,
            method=method,
            evaluation_id=evaluation_id,
            scenario=scenario,
            seed=seed,
            run=run,
            weights=weights,
            reason="summary.csv missing after ns-3 run",
            score_version=score_version,
            elapsed_s=elapsed_s,
        )

    rows = read_summary(out_summary)
    evaluation = Evaluation(
        method=method,
        evaluation_id=evaluation_id,
        scenario=scenario,
        seed=seed,
        run=run,
        weights=normalize_weights(weights),
        score=score_fn(rows),
        result_dir=result_dir(candidate_root, scenario, seed, run),
        log_file=log_file,
        elapsed_s=elapsed_s,
        cached=False,
    )

    _write_sidecar(candidate_root, evaluation, score_version=score_version)
    return evaluation


def _write_sidecar(candidate_root: Path, evaluation: Evaluation, *,
                   failed: bool = False, score_version: str = "v1") -> None:
    """Persist a candidate.json sidecar for checkpointing.

    ``failed=True`` marks evaluations whose ns-3 run crashed; they carry a
    heavily penalized score so they are neither re-tried on resume nor
    mistaken for genuinely poor solutions during analysis. ``score_version``
    records which objective (v1/v2) produced ``score`` so mixed caches are
    never silently compared.
    """
    candidate_meta = candidate_root / "candidate.json"
    candidate_meta.write_text(json.dumps({
        "method": evaluation.method,
        "evaluation_id": evaluation.evaluation_id,
        "scenario": evaluation.scenario,
        "seed": evaluation.seed,
        "run": evaluation.run,
        "weights": list(evaluation.weights),
        "score": evaluation.score,
        "score_version": score_version,
        "result_dir": str(evaluation.result_dir),
        "log_file": str(evaluation.log_file),
        "failed": failed,
        "elapsed_s": evaluation.elapsed_s,
        "cached": evaluation.cached,
    }), encoding="utf-8")


# Score assigned to any candidate whose ns-3 run crashed. Strongly negative so
# the optimizer steers away, and below any legitimately-computed score.
FAILED_SCORE = -1e6


def _failed_evaluation(
    *,
    eval_root: Path,
    candidate_root: Path,
    log_file: Path,
    method: str,
    evaluation_id: int,
    scenario: str,
    seed: int,
    run: int,
    weights: Sequence[float],
    reason: str,
    score_version: str = "v1",
    elapsed_s: float = 0.0,
) -> Evaluation:
    """Build and checkpoint a penalized Evaluation for a crashed ns-3 run.

    Also appends a structured line to ``eval_root/.ns3_failures.log`` so the
    dashboard / analysis can find every tolerated crash in one place.

    ``score_version`` must be stamped on the sidecar even though FAILED_SCORE is
    a constant that no objective computed: the resume cache rejects any sidecar
    whose version differs from the requested one, so a sidecar left at the "v1"
    default would be re-simulated on every v2 resume — violating the
    never-retry contract and, if the crash turned out to be transient, silently
    diverging the optimizer trajectory from that evaluation onward.
    """
    normalized = normalize_weights(weights)
    evaluation = Evaluation(
        method=method,
        evaluation_id=evaluation_id,
        scenario=scenario,
        seed=seed,
        run=run,
        weights=normalized,
        score=FAILED_SCORE,
        result_dir=result_dir(candidate_root, scenario, seed, run),
        log_file=log_file,
        elapsed_s=elapsed_s,
        cached=False,
    )
    _write_sidecar(candidate_root, evaluation, failed=True)

    failures_log = eval_root / ".ns3_failures.log"
    with failures_log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{scenario}\tseed={seed}\t{method}\teval={evaluation_id}\t"
            f"weights={normalized}\treason={reason}\tlog={log_file}\n"
        )
    print(
        f"[META fail] scenario={scenario} seed={seed} {method} eval {evaluation_id} "
        f"crashed ({reason}); score={FAILED_SCORE} (continuing)"
    )
    return evaluation


def optimize_ga(
    rng: random.Random,
    evaluate: Callable[[str, int, Sequence[float]], Evaluation],
    iterations: int,
    population_size: int,
    mutation_strength: float,
) -> list[Evaluation]:
    population = [random_weights(rng) for _ in range(population_size)]
    evaluations: list[Evaluation] = []
    eval_id = 0

    for _ in range(iterations):
        generation: list[Evaluation] = []
        for weights in population:
            eval_id += 1
            result = evaluate("ga", eval_id, weights)
            generation.append(result)
            evaluations.append(result)

        generation.sort(key=lambda item: item.score, reverse=True)
        elite_count = max(2, population_size // 3)
        elites = [item.weights for item in generation[:elite_count]]
        next_population = list(elites)
        while len(next_population) < population_size:
            parent_a, parent_b = rng.sample(elites, 2)
            child = crossover_weights(parent_a, parent_b, rng)
            child = mutate_weights(child, rng, mutation_strength)
            next_population.append(child)
        population = next_population

    return evaluations


def optimize_pso(
    rng: random.Random,
    evaluate: Callable[[str, int, Sequence[float]], Evaluation],
    iterations: int,
    particles: int,
) -> list[Evaluation]:
    positions = [random_weights(rng) for _ in range(particles)]
    velocities = [[0.0, 0.0, 0.0] for _ in range(particles)]
    personal_best = [list(pos) for pos in positions]
    personal_scores = [-math.inf] * particles
    global_best = [1.0 / 3.0] * 3
    global_score = -math.inf
    evaluations: list[Evaluation] = []
    eval_id = 0

    for _ in range(iterations):
        for i, position in enumerate(positions):
            eval_id += 1
            result = evaluate("pso", eval_id, position)
            evaluations.append(result)
            if result.score > personal_scores[i]:
                personal_scores[i] = result.score
                personal_best[i] = list(result.weights)
            if result.score > global_score:
                global_score = result.score
                global_best = list(result.weights)

        for i in range(particles):
            for d in range(3):
                r1 = rng.random()
                r2 = rng.random()
                velocities[i][d] = (
                    0.55 * velocities[i][d]
                    + 1.30 * r1 * (personal_best[i][d] - positions[i][d])
                    + 1.30 * r2 * (global_best[d] - positions[i][d])
                )
            positions[i] = normalize_weights([positions[i][d] + velocities[i][d] for d in range(3)])

    return evaluations


def optimize_sa(
    rng: random.Random,
    evaluate: Callable[[str, int, Sequence[float]], Evaluation],
    iterations: int,
    mutation_strength: float,
) -> list[Evaluation]:
    current = [1.0 / 3.0] * 3
    evaluations: list[Evaluation] = []
    current_result: Evaluation | None = None

    for eval_id in range(1, iterations + 1):
        temperature = max(0.01, 1.0 - (eval_id - 1) / max(iterations, 1))
        candidate = mutate_weights(current, rng, mutation_strength * temperature)
        candidate_result = evaluate("sa", eval_id, candidate)
        evaluations.append(candidate_result)

        if current_result is None:
            current = list(candidate_result.weights)
            current_result = candidate_result
            continue

        delta = candidate_result.score - current_result.score
        if delta >= 0.0 or rng.random() < math.exp(delta / max(temperature * 20.0, 1e-9)):
            current = list(candidate_result.weights)
            current_result = candidate_result

    return evaluations


def optimize_grid(
    evaluate: Callable[[str, int, Sequence[float]], Evaluation],
    grid_step: float,
) -> list[Evaluation]:
    """Uniform grid search over the weight simplex (camera-ready reference).

    Deterministically enumerates every combination (i, j, k) of non-negative
    multiples of ``grid_step`` with i+j+k = 1/grid_step, in ascending (i, j)
    order. Consumes NO randomness, so running it alongside (or before/after)
    the metaheuristics does not perturb their seeded RNG trajectories. With
    step 0.05 this yields 231 combinations. Uses the exact same evaluate()
    pipeline (run_candidate -> ns-3 -> summary.csv -> score) as the other
    methods — no separate objective.
    """
    n = int(round(1.0 / grid_step))
    combinations: list[list[float]] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            combinations.append([i * grid_step, j * grid_step, k * grid_step])

    evaluations: list[Evaluation] = []
    for eval_id, weights in enumerate(combinations, start=1):
        evaluations.append(evaluate("grid", eval_id, weights))
    return evaluations


def optimize_hybrid(
    rng: random.Random,
    evaluate: Callable[[str, int, Sequence[float]], Evaluation],
    iterations: int,
    population_size: int,
    mutation_strength: float,
) -> list[Evaluation]:
    population_size = max(1, population_size)
    particles = [
        Particle(position=list(weights), velocity=[0.0, 0.0, 0.0], personal_best=list(weights))
        for weights in seed_weight_candidates(rng, population_size)
    ]
    global_best = [1.0 / 3.0] * 3
    global_score = -math.inf
    current_result: Evaluation | None = None
    archive: list[Evaluation] = []
    evaluations: list[Evaluation] = []
    eval_id = 0
    stagnant_iterations = 0

    for iteration in range(iterations):
        temperature = max(0.05, 1.0 - iteration / max(iterations, 1))
        generation: list[Evaluation] = []
        improved = False

        for particle in particles:
            eval_id += 1
            result = evaluate("hybrid", eval_id, particle.position)
            evaluations.append(result)
            generation.append(result)
            if result.score > particle.personal_score:
                particle.personal_score = result.score
                particle.personal_best = list(result.weights)
            if result.score > global_score:
                global_score = result.score
                global_best = list(result.weights)
                improved = True

        archive = sorted(archive + generation, key=lambda item: item.score, reverse=True)[: max(2, population_size)]
        if archive and current_result is None:
            current_result = archive[0]
        stagnant_iterations = 0 if improved else stagnant_iterations + 1

        adaptive_mutation = mutation_strength * (1.0 + 0.5 * min(stagnant_iterations, 3))
        eval_id += 1
        local_candidate = mutate_weights(global_best, rng, adaptive_mutation * temperature)
        local_result = evaluate("hybrid", eval_id, local_candidate)
        evaluations.append(local_result)
        archive = sorted(archive + [local_result], key=lambda item: item.score, reverse=True)[: max(2, population_size)]

        if current_result is None:
            current_result = local_result
        else:
            delta = local_result.score - current_result.score
            if delta >= 0.0 or rng.random() < math.exp(delta / max(temperature * 20.0, 1e-9)):
                current_result = local_result
        if current_result.score > global_score:
            global_score = current_result.score
            global_best = list(current_result.weights)

        elite_count = min(max(1, population_size // 3), len(archive))
        elites = [item.weights for item in archive[:elite_count]]
        next_particles = [
            Particle(
                position=list(item.weights),
                velocity=[0.0, 0.0, 0.0],
                personal_best=list(item.weights),
                personal_score=item.score,
            )
            for item in archive[:elite_count]
        ]

        inertia = 0.65 - 0.25 * (iteration / max(iterations, 1))
        pso_quota = max(1, (population_size - elite_count) // 2)
        ga_quota = max(0, population_size - elite_count - pso_quota)

        pso_sources = sorted(particles, key=lambda item: item.personal_score, reverse=True)
        for particle in pso_sources:
            if len(next_particles) >= elite_count + pso_quota:
                break
            updated_velocity = []
            for d in range(3):
                r1 = rng.random()
                r2 = rng.random()
                updated_velocity.append(
                    inertia * particle.velocity[d]
                    + 1.35 * r1 * (particle.personal_best[d] - particle.position[d])
                    + 1.35 * r2 * (global_best[d] - particle.position[d])
                )
            pso_position = normalize_weights([particle.position[d] + updated_velocity[d] for d in range(3)])
            next_particles.append(
                Particle(
                    position=pso_position,
                    velocity=updated_velocity,
                    personal_best=list(particle.personal_best),
                    personal_score=particle.personal_score,
                )
            )

        for _ in range(ga_quota):
            if len(elites) >= 2:
                parent_a, parent_b = rng.sample(elites, 2)
                child = crossover_weights(parent_a, parent_b, rng)
            else:
                child = list(global_best)
            child = mutate_weights(child, rng, adaptive_mutation)
            next_particles.append(Particle(position=child, velocity=[0.0, 0.0, 0.0], personal_best=child))

        while len(next_particles) < population_size:
            candidate = mutate_weights(global_best, rng, adaptive_mutation)
            next_particles.append(
                Particle(position=candidate, velocity=[0.0, 0.0, 0.0], personal_best=candidate)
            )

        particles = next_particles[:population_size]

    return evaluations


def write_results(path: Path, evaluations: Sequence[Evaluation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "evaluation_id",
                "scenario",
                "seed",
                "run",
                "score",
                "weight_embb",
                "weight_urllc",
                "weight_mtc",
                "result_dir",
                "log_file",
                # Camera-ready search-cost instrumentation (appended; parsers
                # reading by column name keep working). Cost analyses sum
                # elapsed_s only over rows with cached=False.
                "elapsed_s",
                "cached",
            ],
        )
        writer.writeheader()
        for item in evaluations:
            writer.writerow(
                {
                    "method": item.method,
                    "evaluation_id": item.evaluation_id,
                    "scenario": item.scenario,
                    "seed": item.seed,
                    "run": item.run,
                    "score": f"{item.score:.6f}",
                    "weight_embb": f"{item.weights[0]:.6f}",
                    "weight_urllc": f"{item.weights[1]:.6f}",
                    "weight_mtc": f"{item.weights[2]:.6f}",
                    "result_dir": str(item.result_dir),
                    "log_file": str(item.log_file),
                    "elapsed_s": f"{item.elapsed_s:.3f}",
                    "cached": item.cached,
                }
            )


def write_best(path: Path, evaluations: Sequence[Evaluation]) -> None:
    if not evaluations:
        return
    best = max(evaluations, key=lambda item: item.score)
    payload = {
        "method": best.method,
        "evaluation_id": best.evaluation_id,
        "scenario": best.scenario,
        "seed": best.seed,
        "run": best.run,
        "score": best.score,
        "weights": {
            "eMBB": best.weights[0],
            "URLLC": best.weights[1],
            "MTC": best.weights[2],
        },
        "ns3_args": {
            "baselineMode": "slice_custom",
            "weights": format_weights(best.weights),
        },
        "result_dir": str(best.result_dir),
        "log_file": str(best.log_file),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def best_candidate_filename(scenario: str, seed: int) -> str:
    return f"best_candidate_{scenario}_seed{seed}.json"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run metaheuristic optimization over RSLAQ slice_custom weights")
    parser.add_argument("--method", choices=["ga", "pso", "sa", "hybrid", "grid", "all"], default="ga")
    parser.add_argument("--grid_step", type=float, default=0.05,
                        help="Uniform grid step for --method grid (0.05 -> 231 simplex points)")
    parser.add_argument("--scenario", default="normal", help="Single scenario (backward compatible)")
    parser.add_argument("--scenarios", default=None, help="Space-separated list of scenarios (overrides --scenario)")
    parser.add_argument("--seed", type=int, default=1, help="Single seed (backward compatible)")
    parser.add_argument("--seeds", default=None, help="Space-separated list of seeds (overrides --seed)")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=2026)
    parser.add_argument("--score_version", choices=["v1", "v2"], default="v1",
                        help="Objective function: v1 (original) or v2 (audit reformulation)")
    parser.add_argument("--per_seed_search", action="store_true",
                        help="Vary the optimizer RNG per ns-3 seed (random_seed + seed) "
                             "so the search trajectory differs across seeds (v2 campaign)")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--mutation_strength", type=float, default=0.12)
    parser.add_argument("--intra_algo", choices=["RR", "PF", "BCQI"], default="PF")
    parser.add_argument("--sim_time", type=float, default=5.0)
    parser.add_argument("--app_start", type=float, default=0.4)
    parser.add_argument("--drain_time", type=float, default=0.2)
    parser.add_argument("--period_ms", type=int, default=10)
    parser.add_argument("--tx_power", type=float, default=43.0)
    parser.add_argument("--tdd_pattern", default="D|D|8D|4GB|4U|U|U")
    parser.add_argument("--rlc_mode", choices=["um", "am"], default="um")
    parser.add_argument("--ns3_dir", type=Path, default=repo_root / "ns-3-dev")
    parser.add_argument("--output_root", type=Path, default=repo_root / "ns-3-dev" / "results_rslaq_metaheuristics")
    parser.add_argument("--build_ns3", action="store_true")
    args = parser.parse_args()

    if args.scenarios is not None:
        args.scenarios = args.scenarios.split()
    else:
        args.scenarios = [args.scenario]

    if args.seeds is not None:
        args.seeds = [int(s) for s in args.seeds.split()]
    else:
        args.seeds = [args.seed]

    return args


def main() -> None:
    args = parse_args()

    if args.build_ns3:
        subprocess.run(["./ns3", "build", "rslaq-sim"], cwd=args.ns3_dir, check=True)

    methods = ["ga", "pso", "sa", "hybrid"] if args.method == "all" else [args.method]
    # Grid search shares the pair loop and evaluate() but is invoked as its own
    # --method grid run (kept out of "all" so it never interleaves with the
    # metaheuristics' seeded search in one process).

    score_fn = SCORE_FUNCTIONS[args.score_version]
    print(f"[META] score_version={args.score_version} ({score_fn.__name__})")

    for scenario in args.scenarios:
        for seed in args.seeds:
            # Trajetória de busca independente por seed do ns-3 (v2). Na v1 o
            # --random_seed era fixo para todas as seeds, tornando a busca
            # idêntica entre elas; somar a seed do ns-3 descorrelaciona as buscas
            # sem quebrar a reprodutibilidade (determinístico por par).
            search_random_seed = (args.random_seed + seed
                                  if args.per_seed_search else args.random_seed)
            rng = random.Random(search_random_seed)
            search_root = args.output_root / f"scenario={scenario}" / f"seed={seed}" / "metaheuristic_search"
            eval_root = search_root / "evals"
            eval_root.mkdir(parents=True, exist_ok=True)

            print(f"[META] scenario={scenario} seed={seed} methods={methods}")

            def evaluate(method: str, evaluation_id: int, weights: Sequence[float]) -> Evaluation:
                candidate_root = eval_root / f"{method}_eval_{evaluation_id:04d}"
                cached = load_cached_evaluation(candidate_root, method, evaluation_id, weights,
                                                score_version=args.score_version)
                if cached is not None:
                    print(f"[META cache] scenario={scenario} seed={seed} {method} eval {evaluation_id} "
                          f"score={cached.score:.4f} (skipped ns-3)")
                    return cached
                return run_candidate(
                    ns3_dir=args.ns3_dir,
                    eval_root=eval_root,
                    method=method,
                    evaluation_id=evaluation_id,
                    scenario=scenario,
                    weights=weights,
                    intra_algo=args.intra_algo,
                    sim_time=args.sim_time,
                    app_start=args.app_start,
                    drain_time=args.drain_time,
                    period_ms=args.period_ms,
                    seed=seed,
                    run=args.run,
                    tx_power=args.tx_power,
                    tdd_pattern=args.tdd_pattern,
                    rlc_mode=args.rlc_mode,
                    score_fn=score_fn,
                    score_version=args.score_version,
                )

            all_evaluations: list[Evaluation] = []
            results_filename = f"metaheuristic_results_{scenario}_seed{seed}.csv"
            # Budget equalization: GA/PSO evaluate (iterations x population) times,
            # while SA is a single-trajectory method (1 evaluation per step). To
            # make the comparison fair under an equal evaluation budget, the SA
            # chain length is set to the same number of evaluations the GA/PSO
            # population gets, not to the bare iteration count. The hybrid gets
            # iterations x (population + 1) naturally (population + 1 local SA
            # refinement per iteration).
            sa_steps = max(args.iterations, args.iterations * args.population)
            for method in methods:
                if method == "ga":
                    all_evaluations.extend(
                        optimize_ga(rng, evaluate, args.iterations, args.population, args.mutation_strength)
                    )
                elif method == "pso":
                    all_evaluations.extend(optimize_pso(rng, evaluate, args.iterations, args.population))
                elif method == "sa":
                    all_evaluations.extend(optimize_sa(rng, evaluate, sa_steps, args.mutation_strength))
                elif method == "grid":
                    # Deterministic sweep; does not consume rng, so the seeded
                    # trajectories of ga/pso/sa/hybrid are unaffected whether
                    # grid runs alone or in the same invocation.
                    all_evaluations.extend(optimize_grid(evaluate, args.grid_step))
                elif method == "hybrid":
                    all_evaluations.extend(
                        optimize_hybrid(rng, evaluate, args.iterations, args.population, args.mutation_strength)
                    )

                # Incremental persistence: after each method, rewrite the
                # results CSV and best-candidate JSON over everything seen so
                # far. A crash in the middle of a later method still leaves the
                # earlier methods' results and best candidate on disk.
                write_results(search_root / results_filename, all_evaluations)
                write_best(search_root / best_candidate_filename(scenario, seed), all_evaluations)
                if all_evaluations:
                    best = max(all_evaluations, key=lambda item: item.score)
                    print(f"  [{scenario}/seed{seed}] after {method}: "
                          f"best={best.method} score={best.score:.4f} "
                          f"weights={format_weights(best.weights)}")

            if all_evaluations:
                best = max(all_evaluations, key=lambda item: item.score)
                print(f"  [{scenario}/seed{seed}] DONE: best={best.method} "
                      f"score={best.score:.4f} weights={format_weights(best.weights)} "
                      f"over {len(all_evaluations)} evals -> {search_root}")


if __name__ == "__main__":
    main()
