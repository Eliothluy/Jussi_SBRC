#!/usr/bin/env bash
# RSLAQ DDQN paper-faithful campaign — 3 light scenarios x 5 seeds.
#
# Faithful to Yungaicela-Naula et al., IEEE TMC 2026 (Hyp-set3):
#   E≈3500 interaction steps, ntsr=100, nsut=200, L=500, btsz=350,
#   LR=0.001, gamma=0.80, eps_decay=0.998, eps_min=0.05, psta=0.5,
#   weights [0.3333, 0.4000, 0.2667], reward piecewise Eq. 12.
#
# congestion and insufficient_resources are deferred (cost: ~13-15 days/seed
# in congestion on this machine per AGENTS.md §14.7). This wrapper runs the
# 3 affordable scenarios first so incremental results come in days, not weeks.
#
# Usage:  bash examples/run_rslaq_ddqn_paper_20260811.sh
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/eliothluy/Documentos/artigo_jussi}"
GYM_DIR="${GYM_DIR:-${REPO_ROOT}/ns-o-ran-gym}"
RUNNER="${GYM_DIR}/examples/run_all_scenarios.sh"

# Separate tag — do NOT mix with the v2 meta-heuristics line.
RUN_TAG="20260811_rslaq_ddqn_paper"

echo "=========================================================="
echo "RSLAQ DDQN paper-faithful campaign — $(date)"
echo "RUN_TAG=${RUN_TAG}"
echo "Scenarios: low_traffic normal stressed (congestion/insufficient deferred)"
echo "Seeds: 1 2 3 4 5   SIM_TIME=5   Hyp-set3 (paper Table VI)"
echo "=========================================================="

# -------------------------------------------------------------------
# Phase 4 of run_all_scenarios.sh with paper-faithful overrides.
# The orchestrator forwards these to run_controlled_rslaq_validation.sh,
# which computes EPISODES = ceil(INTERACTION_STEPS / EPISODE_STEPS) = 35.
# -------------------------------------------------------------------
exec env \
    RUN_TAG="${RUN_TAG}" \
    RUN_BASELINES=0 \
    RUN_METAHEURISTICS=0 \
    RUN_META_EVALUATION=0 \
    RUN_RSLAQ_DDQN_PAPER=1 \
    BUILD_NS3=1 \
    SCENARIOS="low_traffic normal stressed" \
    SEEDS="1 2 3 4 5" \
    SIM_TIME=5 \
    APP_START=0.5 \
    PERIOD_MS=10 \
    TX_POWER=43 \
    TDD_PATTERN='D|D|8D|4GB|4U|U|U' \
    RLC_MODE=um \
    DDQN_INTERACTION_STEPS=3500 \
    DDQN_EPISODE_STEPS=100 \
    DDQN_BUFFER_SIZE=500 \
    DDQN_BATCH_SIZE=350 \
    DDQN_LR=0.001 \
    DDQN_GAMMA=0.80 \
    DDQN_EPS_START=1.0 \
    DDQN_EPS_MIN=0.05 \
    DDQN_EPS_DECAY=0.998 \
    DDQN_TARGET_UPDATE=200 \
    DDQN_TRAINING_REPLICATES=1 \
    DDQN_REPLICATE_SEED_STRIDE=1000 \
    DDQN_DRL_JOBS=5 \
    P_STA_STATIC_FRACTION=0.5 \
    P_STA_WEIGHTS="0.3333,0.4000,0.2667" \
    REWARD_ALPHA=0.3333 \
    REWARD_BETA=0.4000 \
    REWARD_GAMMA=0.2667 \
    ENABLE_STEP_LOGGING=1 \
    bash "${RUNNER}"
