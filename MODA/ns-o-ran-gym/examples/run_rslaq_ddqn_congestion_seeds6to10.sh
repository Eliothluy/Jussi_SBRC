#!/usr/bin/env bash
# RSLAQ DDQN paper-faithful — congestion seeds 6-10 (completes n=10 parity
# with the metaheuristics v2 campaign, which uses 10 seeds everywhere).
# Same Hyp-set3 protocol; same RUN_TAG so dirs sum into the campaign tree.
set -Eeuo pipefail
REPO_ROOT="${REPO_ROOT:-/home/eliothluy/Documentos/artigo_jussi}"
RUNNER="${REPO_ROOT}/ns-o-ran-gym/examples/run_all_scenarios.sh"

echo "=========================================================="
echo "RSLAQ DDQN — congestion seeds 6-10 — $(date)"
echo "=========================================================="

exec env \
    RUN_TAG="20260811_rslaq_ddqn_paper" \
    RUN_BASELINES=0 RUN_METAHEURISTICS=0 RUN_META_EVALUATION=0 \
    RUN_RSLAQ_DDQN_PAPER=1 BUILD_NS3=0 \
    SCENARIOS="congestion" SEEDS="6 7 8 9 10" \
    SIM_TIME=5 APP_START=0.5 PERIOD_MS=10 TX_POWER=43 \
    TDD_PATTERN='D|D|8D|4GB|4U|U|U' RLC_MODE=um \
    DDQN_INTERACTION_STEPS=3500 DDQN_EPISODE_STEPS=100 \
    DDQN_BUFFER_SIZE=500 DDQN_BATCH_SIZE=350 \
    DDQN_LR=0.001 DDQN_GAMMA=0.80 \
    DDQN_EPS_START=1.0 DDQN_EPS_MIN=0.05 DDQN_EPS_DECAY=0.998 \
    DDQN_TARGET_UPDATE=200 DDQN_TRAINING_REPLICATES=1 \
    DDQN_REPLICATE_SEED_STRIDE=1000 DDQN_DRL_JOBS=5 \
    P_STA_STATIC_FRACTION=0.5 P_STA_WEIGHTS="0.3333,0.4000,0.2667" \
    REWARD_ALPHA=0.3333 REWARD_BETA=0.4000 REWARD_GAMMA=0.2667 \
    ENABLE_STEP_LOGGING=1 \
    bash "${RUNNER}"
