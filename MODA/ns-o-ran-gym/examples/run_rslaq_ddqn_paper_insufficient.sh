#!/usr/bin/env bash
# RSLAQ DDQN paper-faithful campaign — insufficient_resources scenario, seeds 1-5.
#
# Same protocol as the other scenarios in the 20260811_rslaq_ddqn_paper tag
# (Hyp-set3, paper Table VI: E=3500, ntsr=100, L=500, btsz=350, LR=0.001,
# gamma=0.80, eps_decay=0.998). insufficient_resources is the heaviest load
# profile (70 UEs: 20 eMBB / 10 URLLC / 40 MTC, ~295 Mbps offered) — the last
# scenario deferred in AGENTS.md §14.8. With the paper-faithful 3500-step
# budget and the 4 defensive patches (PDCP, RLC-UM, NetDevice, scheduler
# renormalization) it is now affordable.
#
# Reuses the same RUN_TAG so ddqn_paper_insufficient_resources_seed{1..5}/
# sum to the existing campaign tree.
#
# Usage:  bash examples/run_rslaq_ddqn_paper_insufficient.sh
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/eliothluy/Documentos/artigo_jussi}"
GYM_DIR="${GYM_DIR:-${REPO_ROOT}/ns-o-ran-gym}"
RUNNER="${GYM_DIR}/examples/run_all_scenarios.sh"

# SAME tag — dirs sum into the same rslaq_ddqn_paper/ folder.
RUN_TAG="20260811_rslaq_ddqn_paper"

echo "=========================================================="
echo "RSLAQ DDQN paper-faithful — insufficient_resources seeds 1-5 — $(date)"
echo "RUN_TAG=${RUN_TAG} (soma aos dirs existentes)"
echo "Scenario: insufficient_resources (70 UEs: 20 eMBB / 10 URLLC / 40 MTC)"
echo "Seeds: 1 2 3 4 5   SIM_TIME=5   Hyp-set3 (paper Table VI)"
echo "=========================================================="

exec env \
    RUN_TAG="${RUN_TAG}" \
    RUN_BASELINES=0 \
    RUN_METAHEURISTICS=0 \
    RUN_META_EVALUATION=0 \
    RUN_RSLAQ_DDQN_PAPER=1 \
    BUILD_NS3=0 \
    SCENARIOS="insufficient_resources" \
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
