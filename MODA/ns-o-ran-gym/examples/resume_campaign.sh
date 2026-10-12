#!/usr/bin/env bash
# Resume the partial paper campaign at run tag 20260626_122326 and launch the
# RSLAQ DDQN paper-faithful sub-campaign.
#
# Recovery scope (run_tag=20260626_122326):
#   Phase 1 (baselines)         -> RESUME=1 re-runs the only failed job:
#                                   congestion / slice_aqps / seed=1
#   Phase 2 (meta-heuristics)   -> re-runs all (scenario, seed) pairs because
#                                   the previous run was killed before writing
#                                   best_candidate_*.json / aggregate CSVs
#                                   even for the pairs whose evals are present.
#                                   Re-runs are deterministic (META_RANDOM_SEED
#                                   is fixed) and overwrite partial eval dirs
#                                   cleanly.
#   Phase 3 (meta-evaluation)   -> currently empty; runs for all 5 scenarios.
#   Phase 4 (RSLAQ DDQN paper)  -> NEW: trains the DDQN agent with the
#                                   paper-faithful RSLAQ reward over the same
#                                   scenario/seed grid as Phases 1-3.
#
# This script is intended to run in the background. Two ways to launch:
#
#   nohup bash examples/resume_campaign.sh > /dev/null 2>&1 &
#   disown
#
#   (or, equivalently:)
#   setsid bash examples/resume_campaign.sh > /dev/null 2>&1 &
#
# All stdout/stderr is captured to RESUME_LOG_FILE so the caller may safely
# discard stdio. Monitor with:
#   tail -f "${RESUME_LOG_FILE}"
#   ps -p "$(cat "${RESUME_PID_FILE}")" -o pid,etime,cmd

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Paths and runtime tag
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/home/elioth/Documentos/artigo_jussi}"
GYM_DIR="${GYM_DIR:-${REPO_ROOT}/ns-o-ran-gym}"
NS3_DIR="${NS3_DIR:-${REPO_ROOT}/ns-3-dev}"

RUN_TAG="${RUN_TAG:-20260626_122326}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GYM_DIR}/results_controlled/heuristics_metaheuristics/${RUN_TAG}}"
RESUME_LOG_FILE="${RESUME_LOG_FILE:-${OUTPUT_ROOT}/resume_campaign.log}"
RESUME_PID_FILE="${RESUME_PID_FILE:-${OUTPUT_ROOT}/resume_campaign.pid}"

mkdir -p "${OUTPUT_ROOT}"

# ---------------------------------------------------------------------------
# Lock: refuse to run if a previous resume_campaign is still alive
# ---------------------------------------------------------------------------
if [[ -f "${RESUME_PID_FILE}" ]]; then
    prev_pid="$(cat "${RESUME_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${prev_pid}" ]] && kill -0 "${prev_pid}" 2>/dev/null; then
        echo "[resume_campaign] already running as PID ${prev_pid}; refusing to start a second instance." >&2
        echo "[resume_campaign] log: ${RESUME_LOG_FILE}" >&2
        exit 1
    fi
    rm -f "${RESUME_PID_FILE}"
fi

# Write our own pid so callers can monitor us.
echo $$ > "${RESUME_PID_FILE}"
trap 'rm -f "${RESUME_PID_FILE}"' EXIT INT TERM

# ---------------------------------------------------------------------------
# Campaign parameters (mirrors examples/run_all_scenarios.sh defaults but
# made explicit here so the resume is reproducible end-to-end).
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

PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

# Meta-heuristics (same as the original campaign)
META_METHOD="${META_METHOD:-all}"
META_ITERATIONS="${META_ITERATIONS:-12}"
META_POPULATION="${META_POPULATION:-6}"
META_MUTATION_STRENGTH="${META_MUTATION_STRENGTH:-0.12}"
META_INTRA_ALGO="${META_INTRA_ALGO:-PF}"
META_RANDOM_SEED="${META_RANDOM_SEED:-2026}"
META_SEED="${META_SEED:-1}"
META_RUN="${META_RUN:-1}"

# RSLAQ DDQN (paper-faithful reward) sub-campaign knobs. Defaults correspond
# to the controlled validation runner defaults so the agent is trained with
# the same budget used in the paper narrative.
DDQN_INTERACTION_STEPS="${DDQN_INTERACTION_STEPS:-20000}"
DDQN_EPISODE_STEPS="${DDQN_EPISODE_STEPS:-100}"
DDQN_TRAINING_REPLICATES="${DDQN_TRAINING_REPLICATES:-1}"
DDQN_DRL_JOBS="${DDQN_DRL_JOBS:-1}"
DDQN_REPLICATE_SEED_STRIDE="${DDQN_REPLICATE_SEED_STRIDE:-1000}"
DDQN_BUFFER_SIZE="${DDQN_BUFFER_SIZE:-10000}"
DDQN_BATCH_SIZE="${DDQN_BATCH_SIZE:-64}"
DDQN_LR="${DDQN_LR:-0.001}"
DDQN_GAMMA="${DDQN_GAMMA:-0.80}"
DDQN_EPS_START="${DDQN_EPS_START:-1.0}"
DDQN_EPS_MIN="${DDQN_EPS_MIN:-0.05}"
DDQN_EPS_DECAY="${DDQN_EPS_DECAY:-0.998}"
DDQN_TARGET_UPDATE="${DDQN_TARGET_UPDATE:-200}"
ENABLE_STEP_LOGGING="${ENABLE_STEP_LOGGING:-1}"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
{
    echo "============================================"
    echo "Resume campaign"
    echo "============================================"
    echo "Started at         : $(date)"
    echo "Run tag            : ${RUN_TAG}"
    echo "Output root        : ${OUTPUT_ROOT}"
    echo "Log file           : ${RESUME_LOG_FILE}"
    echo "Parallel jobs      : ${PARALLEL_JOBS}"
    echo "Scenarios          : ${SCENARIOS}"
    echo "Seeds              : ${SEEDS}"
    echo "Sim time           : ${SIM_TIME}"
    echo ""
    echo "Meta method        : ${META_METHOD}"
    echo "Meta iterations    : ${META_ITERATIONS}"
    echo "Meta population    : ${META_POPULATION}"
    echo "Meta intra-algo    : ${META_INTRA_ALGO}"
    echo "Meta random seed   : ${META_RANDOM_SEED}"
    echo ""
    echo "DDQN interaction   : ${DDQN_INTERACTION_STEPS}"
    echo "DDQN episode steps : ${DDQN_EPISODE_STEPS}"
    echo "DDQN replicates    : ${DDQN_TRAINING_REPLICATES}"
    echo "DDQN DRL jobs      : ${DDQN_DRL_JOBS}"
    echo "============================================"
} | tee -a "${RESUME_LOG_FILE}"

cd "${GYM_DIR}"

# ---------------------------------------------------------------------------
# Launch the patched campaign (Phase 1-4) into the existing output root.
# ---------------------------------------------------------------------------
env \
    RUN_TAG="${RUN_TAG}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    REPO_ROOT="${REPO_ROOT}" \
    NS3_DIR="${NS3_DIR}" \
    GYM_DIR="${GYM_DIR}" \
    RESUME="1" \
    BUILD_NS3="${BUILD_NS3:-1}" \
    RUN_BASELINES="${RUN_BASELINES:-1}" \
    RUN_METAHEURISTICS="${RUN_METAHEURISTICS:-1}" \
    RUN_META_EVALUATION="${RUN_META_EVALUATION:-1}" \
    RUN_RSLAQ_DDQN_PAPER="${RUN_RSLAQ_DDQN_PAPER:-1}" \
    PARALLEL_JOBS="${PARALLEL_JOBS}" \
    SCENARIOS="${SCENARIOS}" \
    SEEDS="${SEEDS}" \
    RUNS="${RUNS}" \
    SIM_TIME="${SIM_TIME}" \
    APP_START="${APP_START}" \
    DRAIN_TIME_SEC="${DRAIN_TIME_SEC}" \
    PERIOD_MS="${PERIOD_MS}" \
    TX_POWER="${TX_POWER}" \
    TDD_PATTERN="${TDD_PATTERN}" \
    RLC_MODE="${RLC_MODE}" \
    META_METHOD="${META_METHOD}" \
    META_ITERATIONS="${META_ITERATIONS}" \
    META_POPULATION="${META_POPULATION}" \
    META_MUTATION_STRENGTH="${META_MUTATION_STRENGTH}" \
    META_INTRA_ALGO="${META_INTRA_ALGO}" \
    META_RANDOM_SEED="${META_RANDOM_SEED}" \
    META_SEED="${META_SEED}" \
    META_RUN="${META_RUN}" \
    DDQN_INTERACTION_STEPS="${DDQN_INTERACTION_STEPS}" \
    DDQN_EPISODE_STEPS="${DDQN_EPISODE_STEPS}" \
    DDQN_TRAINING_REPLICATES="${DDQN_TRAINING_REPLICATES}" \
    DDQN_DRL_JOBS="${DDQN_DRL_JOBS}" \
    DDQN_REPLICATE_SEED_STRIDE="${DDQN_REPLICATE_SEED_STRIDE}" \
    DDQN_BUFFER_SIZE="${DDQN_BUFFER_SIZE}" \
    DDQN_BATCH_SIZE="${DDQN_BATCH_SIZE}" \
    DDQN_LR="${DDQN_LR}" \
    DDQN_GAMMA="${DDQN_GAMMA}" \
    DDQN_EPS_START="${DDQN_EPS_START}" \
    DDQN_EPS_MIN="${DDQN_EPS_MIN}" \
    DDQN_EPS_DECAY="${DDQN_EPS_DECAY}" \
    DDQN_TARGET_UPDATE="${DDQN_TARGET_UPDATE}" \
    ENABLE_STEP_LOGGING="${ENABLE_STEP_LOGGING}" \
    bash examples/run_all_scenarios.sh 2>&1 | tee -a "${RESUME_LOG_FILE}"

status="${PIPESTATUS[0]:-0}"
{
    echo ""
    echo "============================================"
    echo "Resume campaign finished at $(date)"
    echo "Status : ${status}"
    echo "Outputs:"
    echo "  Baselines       : ${OUTPUT_ROOT}/heuristics_ns3/results_rslaq_network_only"
    echo "  Meta-heuristics : ${OUTPUT_ROOT}/metaheuristics"
    echo "  Meta-evaluation : ${OUTPUT_ROOT}/meta_evaluation"
    echo "  RSLAQ DDQN paper: ${OUTPUT_ROOT}/rslaq_ddqn_paper"
    echo "============================================"
} | tee -a "${RESUME_LOG_FILE}"

exit "${status}"