#!/usr/bin/env bash
# Paper network-only campaign: schedulers, RSLAQ, AQPS, heuristics, and meta-heuristics.
#
# Parallel version: runs N simulations concurrently across all three phases.
#
# Phase 1 — Baselines: all (scenario, mode, seed, run) combos run in parallel
#   via the ns-3-dev/run_all_scenarios.sh job pool.
# Phase 2 — Meta-heuristics: each (scenario, seed) pair runs as an independent
#   Python process, up to PARALLEL_JOBS in parallel.
# Phase 3 — Meta-evaluation: each scenario runs as an independent Python
#   process, up to PARALLEL_JOBS in parallel.
#
# Usage:
#   cd /home/elioth/Documentos/artigo_jussi/ns-o-ran-gym
#   bash examples/run_all_scenarios.sh
#
# Useful overrides:
#   PARALLEL_JOBS=8 SIM_TIME=5 SEEDS="1 2 3" bash examples/run_all_scenarios.sh
#   RUN_BASELINES=0 RUN_METAHEURISTICS=1 META_ITERATIONS=8 bash examples/run_all_scenarios.sh
#   BASELINE_MODES="pure_rr pure_bcqi pure_pf slice_weighted_pf slice_aqps" bash examples/run_all_scenarios.sh
#   RESUME=0 bash examples/run_all_scenarios.sh   # re-run everything from scratch

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Default derivado da localização do script (…/ns-o-ran-gym/examples/x.sh ->
# repo root), corrigindo o antigo hardcode /home/elioth inexistente.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/../.." && pwd)}"
NS3_DIR="${NS3_DIR:-${REPO_ROOT}/ns-3-dev}"
GYM_DIR="${GYM_DIR:-${REPO_ROOT}/ns-o-ran-gym}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GYM_DIR}/results_controlled/heuristics_metaheuristics/${RUN_TAG}}"
BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-${HEURISTIC_OUTPUT_ROOT:-${OUTPUT_ROOT}/heuristics_ns3}}"
META_OUTPUT_ROOT="${META_OUTPUT_ROOT:-${OUTPUT_ROOT}/metaheuristics}"
META_EVAL_OUTPUT="${META_EVAL_OUTPUT:-${OUTPUT_ROOT}/meta_evaluation}"

# RSLAQ DDQN (paper-faithful reward) sub-campaign root. Writing to a dedicated
# sub-directory keeps the DRL outputs isolated from the network-only baselines
# and meta-heuristics.
RSLAQ_DDQN_PAPER_OUTPUT_ROOT="${RSLAQ_DDQN_PAPER_OUTPUT_ROOT:-${OUTPUT_ROOT}/rslaq_ddqn_paper}"

# ---------------------------------------------------------------------------
# Phase toggles
# ---------------------------------------------------------------------------
RUN_BASELINES="${RUN_BASELINES:-${RUN_HEURISTICS:-1}}"
RUN_METAHEURISTICS="${RUN_METAHEURISTICS:-1}"
RUN_META_EVALUATION="${RUN_META_EVALUATION:-1}"
RUN_RSLAQ_DDQN_PAPER="${RUN_RSLAQ_DDQN_PAPER:-0}"
BUILD_NS3="${BUILD_NS3:-1}"

# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------
PARALLEL_JOBS="${PARALLEL_JOBS:-6}"
RESUME="${RESUME:-1}"

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SCENARIOS="${SCENARIOS:-low_traffic normal congestion stressed insufficient_resources}"
SEEDS="${SEEDS:-1 2 3}"
RUNS="${RUNS:-1}"

SIM_TIME="${SIM_TIME:-5}"
APP_START="${APP_START:-0.4}"
DRAIN_TIME_SEC="${DRAIN_TIME_SEC:-0.2}"
PERIOD_MS="${PERIOD_MS:-10}"
TX_POWER="${TX_POWER:-43}"
TDD_PATTERN="${TDD_PATTERN:-D|D|8D|4GB|4U|U|U}"
RLC_MODE="${RLC_MODE:-um}"

# ---------------------------------------------------------------------------
# Baseline modes (all modes needed for the paper)
# ---------------------------------------------------------------------------
SCHEDULER_MODES="${SCHEDULER_MODES:-pure_rr pure_bcqi pure_pf}"
RSLAQ_MODES="${RSLAQ_MODES:-slice_weighted_pf slice_weighted_rr slice_weighted_bcqi psta_equal}"
AQPS_MODE="${AQPS_MODE:-slice_aqps}"
HEURISTIC_MODES="${HEURISTIC_MODES:-slice_demand_greedy slice_sla_greedy slice_least_waste slice_qos_mixed slice_random_vine slice_meta_risk_elastic}"

RSLAQ_REFERENCE_WEIGHTS="${RSLAQ_REFERENCE_WEIGHTS:-0.3333,0.4000,0.2667}"

# Fair-comparison grouping (contract: tests/test_all_scenarios_runner_contract.py).
# GROUP1: fixed baselines (plain schedulers + static split) — no adaptation.
# GROUP2: adaptive/learning methods (RSLAQ weighted variants, AQPS, heuristics).
# The default BASELINE_MODES is the union of the groups — identical to the
# previous composition (same 14 modes); GROUP1_MODES/GROUP2_MODES exist so
# campaigns can select a whole comparison group at once.
GROUP1_MODES="${GROUP1_MODES:-${SCHEDULER_MODES} psta_equal}"
GROUP2_MODES="${GROUP2_MODES:-slice_weighted_pf slice_weighted_rr slice_weighted_bcqi ${AQPS_MODE} ${HEURISTIC_MODES}}"

BASELINE_MODES="${BASELINE_MODES:-${GROUP1_MODES} ${GROUP2_MODES}}"

# ---------------------------------------------------------------------------
# Meta-heuristic parameters
# ---------------------------------------------------------------------------
META_METHOD="${META_METHOD:-all}"
META_ITERATIONS="${META_ITERATIONS:-12}"
META_POPULATION="${META_POPULATION:-6}"
META_MUTATION_STRENGTH="${META_MUTATION_STRENGTH:-0.12}"
META_INTRA_ALGO="${META_INTRA_ALGO:-PF}"
META_RANDOM_SEED="${META_RANDOM_SEED:-2026}"
META_SEED="${META_SEED:-1}"
META_RUN="${META_RUN:-1}"
# Score objective: v1 (original) or v2 (audit reformulation). PER_SEED_SEARCH=1
# decorrelates the optimizer trajectory across ns-3 seeds (v2 campaign).
META_SCORE_VERSION="${META_SCORE_VERSION:-v1}"
META_PER_SEED_SEARCH="${META_PER_SEED_SEARCH:-0}"
META_PER_SEED_FLAG=""
if [[ "${META_PER_SEED_SEARCH}" == "1" ]]; then
    META_PER_SEED_FLAG="--per_seed_search"
fi

mkdir -p "${OUTPUT_ROOT}" "${META_OUTPUT_ROOT}" "${META_EVAL_OUTPUT}"

if [[ "${RUN_RSLAQ_DDQN_PAPER}" == "1" ]]; then
    mkdir -p "${RSLAQ_DDQN_PAPER_OUTPUT_ROOT}"
fi

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "============================================"
echo "Paper campaign (parallel)"
echo "============================================"
echo "Output root       : ${OUTPUT_ROOT}"
echo "Parallel jobs      : ${PARALLEL_JOBS}"
echo "Resume             : ${RESUME}"
echo "Scenarios          : ${SCENARIOS}"
echo "Seeds              : ${SEEDS}"
echo "Runs               : ${RUNS}"
echo "SIM_TIME           : ${SIM_TIME}"
echo ""
echo "Run baselines      : ${RUN_BASELINES}"
echo "Run metaheur       : ${RUN_METAHEURISTICS}"
echo "Run meta-eval      : ${RUN_META_EVALUATION}"
echo "Run RSLAQ DDQN paper: ${RUN_RSLAQ_DDQN_PAPER}"
echo ""
echo "Scheduler modes    : ${SCHEDULER_MODES}"
echo "RSLAQ modes        : ${RSLAQ_MODES}"
echo "AQPS mode          : ${AQPS_MODE}"
echo "Heuristic modes    : ${HEURISTIC_MODES}"
echo "Baseline modes     : ${BASELINE_MODES}"
echo "RSLAQ ref weights  : ${RSLAQ_REFERENCE_WEIGHTS}"
echo ""
echo "Meta method        : ${META_METHOD}"
echo "Meta iterations    : ${META_ITERATIONS}"
echo "Meta population    : ${META_POPULATION}"
echo "Meta intra-algo    : ${META_INTRA_ALGO}"
echo "DRL                : disabled"
echo ""

# ---------------------------------------------------------------------------
# Build ns-3 (once, before any phase)
# ---------------------------------------------------------------------------
if [[ "${BUILD_NS3}" == "1" ]]; then
    echo "[BUILD] Building rslaq-sim..."
    cd "${NS3_DIR}"
    ./ns3 build rslaq-sim
    echo "[BUILD] Done."
    echo ""
else
    echo "[BUILD] Skipping (BUILD_NS3=${BUILD_NS3})"
fi

# ===========================================================================
# Phase 1: Baselines (parallel job pool)
# ===========================================================================
if [[ "${RUN_BASELINES}" == "1" ]]; then
    echo "============================================"
    echo "Phase 1: Baseline/comparator matrix"
    echo "============================================"
    echo "Modes: ${BASELINE_MODES}"
    echo "Parallel jobs: ${PARALLEL_JOBS}"
    echo ""

    cd "${NS3_DIR}"
    OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT}" \
    BUILD_NS3=0 \
    PARALLEL_JOBS="${PARALLEL_JOBS}" \
    RESUME="${RESUME}" \
    SCENARIOS="${SCENARIOS}" \
    BASELINE_MODES="${BASELINE_MODES}" \
    SEEDS="${SEEDS}" \
    RUNS="${RUNS}" \
    SIM_TIME="${SIM_TIME}" \
    APP_START="${APP_START}" \
    DRAIN_TIME_SEC="${DRAIN_TIME_SEC}" \
    PERIOD_MS="${PERIOD_MS}" \
    TX_POWER="${TX_POWER}" \
    TDD_PATTERN="${TDD_PATTERN}" \
    RLC_MODE="${RLC_MODE}" \
    ./run_all_scenarios.sh || {
        echo "[PHASE 1] ns-3 baseline batch reported non-zero exit (some jobs may have failed)."
        echo "[PHASE 1] Continuing to Phases 2-4 so the campaign can still produce meta-heuristics"
        echo "[PHASE 1] and RSLAQ DDQN paper-faithful outputs. Failed baseline jobs remain"
        echo "[PHASE 1] marked run_failed in the manifest and should be retried separately."
    }
else
    echo "[SKIP] Phase 1 (baselines) disabled"
fi

# ===========================================================================
# Phase 2: Meta-heuristics (parallel per scenario/seed pair)
# ===========================================================================
if [[ "${RUN_METAHEURISTICS}" == "1" ]]; then
    echo ""
    echo "============================================"
    echo "Phase 2: Offline meta-heuristic search"
    echo "============================================"
    echo "Methods: ${META_METHOD}"
    echo "Iterations: ${META_ITERATIONS}  Population: ${META_POPULATION}"
    echo "Intra-algo: ${META_INTRA_ALGO}"
    echo "Parallel pairs: up to ${PARALLEL_JOBS}"
    echo ""

    cd "${GYM_DIR}"

    # Generate (scenario, seed) pairs
    pairs=()
    for scenario in ${SCENARIOS}; do
        for seed in ${SEEDS}; do
            pairs+=("${scenario}:${seed}")
        done
    done

    total_pairs=${#pairs[@]}
    pair_idx=0

    for pair in "${pairs[@]}"; do
        scenario="${pair%%:*}"
        seed="${pair##*:}"
        pair_idx=$((pair_idx + 1))

        echo "[META ${pair_idx}/${total_pairs}] scenario=${scenario} seed=${seed}"

        (
            cd "${GYM_DIR}"
            python3 examples/run_rslaq_metaheuristics.py \
                --method "${META_METHOD}" \
                --scenarios "${scenario}" \
                --seeds "${seed}" \
                --run "${META_RUN}" \
                --random_seed "${META_RANDOM_SEED}" \
                --iterations "${META_ITERATIONS}" \
                --population "${META_POPULATION}" \
                --mutation_strength "${META_MUTATION_STRENGTH}" \
                --intra_algo "${META_INTRA_ALGO}" \
                --sim_time "${SIM_TIME}" \
                --app_start "${APP_START}" \
                --drain_time "${DRAIN_TIME_SEC}" \
                --period_ms "${PERIOD_MS}" \
                --tx_power "${TX_POWER}" \
                --tdd_pattern "${TDD_PATTERN}" \
                --rlc_mode "${RLC_MODE}" \
                --ns3_dir "${NS3_DIR}" \
                --output_root "${META_OUTPUT_ROOT}" \
                --score_version "${META_SCORE_VERSION}" \
                ${META_PER_SEED_FLAG} \
                > "${META_OUTPUT_ROOT}/.meta_${scenario}_seed${seed}.log" 2>&1
        ) &

        # Throttle
        while (( $(jobs -rp | wc -l) >= PARALLEL_JOBS )); do
            wait -n 2>/dev/null || sleep 0.1
        done
    done

    echo "Waiting for remaining meta-heuristic jobs..."
    wait

    # Report any failures
    meta_fail=0
    for pair in "${pairs[@]}"; do
        scenario="${pair%%:*}"
        seed="${pair##*:}"
        log="${META_OUTPUT_ROOT}/.meta_${scenario}_seed${seed}.log"
        if [[ -f "$log" ]] && grep -qi "traceback\|error\|failed" "$log" 2>/dev/null; then
            echo "[META ERROR] ${scenario}/seed${seed} — see ${log}"
            tail -5 "$log"
            meta_fail=$((meta_fail + 1))
        fi
    done
    if [[ "$meta_fail" -gt 0 ]]; then
        echo "[META] ${meta_fail} pair(s) had errors."
    else
        echo "[META] All pairs completed."
    fi
    echo "Meta-heuristics output: ${META_OUTPUT_ROOT}"
else
    echo "[SKIP] Phase 2 (meta-heuristics) disabled"
fi

# ===========================================================================
# Phase 3: Meta-evaluation (parallel per scenario)
# ===========================================================================
if [[ "${RUN_META_EVALUATION}" == "1" ]]; then
    echo ""
    echo "============================================"
    echo "Phase 3: Meta-heuristic evaluation"
    echo "============================================"
    echo "Parallel scenarios: up to ${PARALLEL_JOBS}"
    echo ""

    cd "${GYM_DIR}"

    eval_idx=0
    total_eval=$(echo "${SCENARIOS}" | wc -w)

    for scenario in ${SCENARIOS}; do
        eval_idx=$((eval_idx + 1))
        echo "[EVAL ${eval_idx}/${total_eval}] scenario=${scenario}"

        scenario_search_root="${META_OUTPUT_ROOT}/scenario=${scenario}"

        if [[ ! -d "$scenario_search_root" ]]; then
            echo "  SKIP: no search root found at ${scenario_search_root}"
            continue
        fi

        (
            cd "${GYM_DIR}"
            python3 examples/run_meta_evaluation.py \
                --search_root "${scenario_search_root}" \
                --output_root "${META_EVAL_OUTPUT}" \
                --ns3_dir "${NS3_DIR}" \
                --intra_algo "${META_INTRA_ALGO}" \
                --sim_time "${SIM_TIME}" \
                --app_start "${APP_START}" \
                --drain_time "${DRAIN_TIME_SEC}" \
                --period_ms "${PERIOD_MS}" \
                --tx_power "${TX_POWER}" \
                --tdd_pattern "${TDD_PATTERN}" \
                --rlc_mode "${RLC_MODE}" \
                --run "${META_RUN}" \
                > "${META_EVAL_OUTPUT}/.eval_${scenario}.log" 2>&1
        ) &

        # Throttle
        while (( $(jobs -rp | wc -l) >= PARALLEL_JOBS )); do
            wait -n 2>/dev/null || sleep 0.1
        done
    done

    echo "Waiting for remaining evaluation jobs..."
    wait

    # Report
    eval_fail=0
    for scenario in ${SCENARIOS}; do
        log="${META_EVAL_OUTPUT}/.eval_${scenario}.log"
        if [[ -f "$log" ]] && grep -qi "traceback\|error\|failed" "$log" 2>/dev/null; then
            echo "[EVAL ERROR] ${scenario} — see ${log}"
            tail -5 "$log"
            eval_fail=$((eval_fail + 1))
        fi
    done
    if [[ "$eval_fail" -gt 0 ]]; then
        echo "[EVAL] ${eval_fail} scenario(s) had errors."
    else
        echo "[EVAL] All scenarios completed."
    fi
    echo "Meta-evaluation output: ${META_EVAL_OUTPUT}"
else
    echo "[SKIP] Phase 3 (meta-evaluation) disabled"
fi

# ===========================================================================
# Phase 4: RSLAQ DDQN (paper-faithful reward)
# ===========================================================================
# Trains a DDQN agent with the paper-faithful RSLAQ reward on the same
# scenario/seed grid used by Phases 1-3, so that the optimized policy can be
# compared against the meta-heuristic best candidates and the scheduler
# baselines (RR, BCQI, PF, RSLAQ weighted, AQPS, heuristics).
#
# This phase is a thin wrapper over examples/run_controlled_rslaq_validation.sh
# with only the DDQN paper-faithful line enabled; every other DRL line
# (resource-efficient DDQN, paper SAC, resource-efficient SAC, predictive SAC)
# is disabled to keep the RSLAQ paper narrative intact.
if [[ "${RUN_RSLAQ_DDQN_PAPER}" == "1" ]]; then
    echo ""
    echo "============================================"
    echo "Phase 4: RSLAQ DDQN (paper-faithful reward)"
    echo "============================================"
    echo "Output root        : ${RSLAQ_DDQN_PAPER_OUTPUT_ROOT}"
    echo "Scenarios          : ${SCENARIOS}"
    echo "Seeds              : ${SEEDS}"
    echo "Interaction steps  : ${DDQN_INTERACTION_STEPS:-20000}"
    echo "Episode steps      : ${DDQN_EPISODE_STEPS:-100}"
    echo "Training replicates : ${DDQN_TRAINING_REPLICATES:-1}"
    echo "DRL jobs (parallel): ${DDQN_DRL_JOBS:-1}"
    echo ""

    cd "${GYM_DIR}"

    SCENARIOS="${SCENARIOS}" \
    SEEDS="${SEEDS}" \
    EXPERIMENT_LINE="rslaq_ddqn_paper" \
    OUTPUT_ROOT="${RSLAQ_DDQN_PAPER_OUTPUT_ROOT}" \
    BUILD_NS3="0" \
    RUN_BASELINES="0" \
    RUN_DDQN="1" \
    RUN_RESOURCE_EFFICIENT_DDQN="0" \
    RUN_PAPER_SAC="0" \
    RUN_RESOURCE_EFFICIENT_SAC="0" \
    RUN_PREDICTIVE_SAC="0" \
    RUN_PREDICTIVE_SAC_GRU="0" \
    RUN_PREDICTIVE_SAC_LSTM="0" \
    RUN_FORECASTER="0" \
    RUN_FORECASTER_GRU="0" \
    RUN_FORECASTER_LSTM="0" \
    RUN_FORECASTER_COMPARISON="0" \
    RUN_DRL_COMPARISON="0" \
    SIM_TIME="${SIM_TIME}" \
    APP_START="${APP_START}" \
    PERIOD_MS="${PERIOD_MS}" \
    INTERACTION_STEPS="${DDQN_INTERACTION_STEPS:-20000}" \
    EPISODE_STEPS="${DDQN_EPISODE_STEPS:-100}" \
    TRAINING_REPLICATES="${DDQN_TRAINING_REPLICATES:-1}" \
    DRL_JOBS="${DDQN_DRL_JOBS:-1}" \
    TRAINING_REPLICATE_SEED_STRIDE="${DDQN_REPLICATE_SEED_STRIDE:-1000}" \
    DDQN_BUFFER_SIZE="${DDQN_BUFFER_SIZE:-10000}" \
    DDQN_BATCH_SIZE="${DDQN_BATCH_SIZE:-64}" \
    DDQN_LR="${DDQN_LR:-0.001}" \
    DDQN_GAMMA="${DDQN_GAMMA:-0.80}" \
    DDQN_EPS_START="${DDQN_EPS_START:-1.0}" \
    DDQN_EPS_MIN="${DDQN_EPS_MIN:-0.05}" \
    DDQN_EPS_DECAY="${DDQN_EPS_DECAY:-0.998}" \
    DDQN_TARGET_UPDATE="${DDQN_TARGET_UPDATE:-200}" \
    REWARD_ALPHA="${REWARD_ALPHA:-0.3333}" \
    REWARD_BETA="${REWARD_BETA:-0.4000}" \
    REWARD_GAMMA="${REWARD_GAMMA:-0.2667}" \
    ENABLE_STEP_LOGGING="${ENABLE_STEP_LOGGING:-1}" \
        bash examples/run_controlled_rslaq_validation.sh \
            > "${RSLAQ_DDQN_PAPER_OUTPUT_ROOT}/.phase4_rslaq_ddqn_paper.log" 2>&1 \
            || echo "[PHASE 4] RSLAQ DDQN paper-faithful sub-campaign reported non-zero exit. See ${RSLAQ_DDQN_PAPER_OUTPUT_ROOT}/.phase4_rslaq_ddqn_paper.log"

    echo "[PHASE 4] RSLAQ DDQN paper-faithful sub-campaign done. Output: ${RSLAQ_DDQN_PAPER_OUTPUT_ROOT}"
else
    echo "[SKIP] Phase 4 (RSLAQ DDQN paper-faithful) disabled"
fi

# ===========================================================================
# Done
# ===========================================================================
echo ""
echo "============================================"
echo "Campaign complete at $(date)"
echo "============================================"
echo "Baselines output      : ${BASELINE_OUTPUT_ROOT}/results_rslaq_network_only"
echo "Meta-heuristics output: ${META_OUTPUT_ROOT}"
echo "Meta-evaluation output: ${META_EVAL_OUTPUT}"
echo "RSLAQ DDQN paper      : ${RSLAQ_DDQN_PAPER_OUTPUT_ROOT}"
echo ""

# Final summary
baseline_manifest="${BASELINE_OUTPUT_ROOT}/results_rslaq_network_only/batch_manifest.csv"
if [[ -f "$baseline_manifest" ]]; then
    total=$(tail -n +2 "$baseline_manifest" | wc -l)
    ok=$(grep -c ",ok," "$baseline_manifest" || true)
    failed=$(tail -n +2 "$baseline_manifest" | grep -vc ",ok," || true)
    echo "Baselines: ${ok} OK, ${failed} failed, ${total} total"
fi
