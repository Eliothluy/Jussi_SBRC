#!/usr/bin/env bash
# Predictive RSLAQ validation campaign.
#
# Trains a temporal KPI forecaster from ns-3 network-only baseline data,
# runs predictive SAC with the same interaction budget, and compares results.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/elioth/Documentos/artigo_jussi}"
NS3_DIR="${NS3_DIR:-${REPO_ROOT}/ns-3-dev}"
GYM_DIR="${GYM_DIR:-${REPO_ROOT}/ns-o-ran-gym}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_LINE="${EXPERIMENT_LINE:-predictive_forecaster_sac}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GYM_DIR}/results_controlled/${EXPERIMENT_LINE}/${RUN_TAG}}"

BASELINE_ROOT="${BASELINE_ROOT:-${GYM_DIR}/results_controlled/rslaq_sla_resource_efficiency/latest}"
FORECAST_SOURCE_ROOT="${FORECAST_SOURCE_ROOT:-${NS3_DIR}/results_rslaq_network_only}"
FORECAST_SOURCE_FORMAT="${FORECAST_SOURCE_FORMAT:-baseline}"

SCENARIOS_INPUT="${SCENARIOS:-low_traffic normal congestion stressed insufficient_resources}"
SEEDS_INPUT="${SEEDS:-1 2 3 4 5}"
read -r -a SCENARIOS <<< "${SCENARIOS_INPUT}"
read -r -a SEEDS <<< "${SEEDS_INPUT}"

INTERACTION_STEPS="${INTERACTION_STEPS:-20000}"
EPISODE_STEPS="${EPISODE_STEPS:-100}"
EPISODES="$(python3 -c "import math; print(math.ceil(${INTERACTION_STEPS}/${EPISODE_STEPS}))")"

APP_START="${APP_START:-0.5}"
PERIOD_MS="${PERIOD_MS:-10}"
SIM_TIME="${SIM_TIME:-$(python3 -c "print(${APP_START} + (${EPISODE_STEPS} * ${PERIOD_MS}) / 1000.0 + 0.5)")}"
CONSECUTIVE_OUTAGE_STEPS="${CONSECUTIVE_OUTAGE_STEPS:-5}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
SEED_CYCLE="${SEED_CYCLE:-999999}"
PREDICTIVE_TERMINATE_ON_SLA_VIOLATION="${PREDICTIVE_TERMINATE_ON_SLA_VIOLATION:-0}"

PREDICTIVE_P_STA_FRACTION="${PREDICTIVE_P_STA_FRACTION:-${P_STA_STATIC_FRACTION:-0.25}}"
P_STA_WEIGHTS="${P_STA_WEIGHTS:-0.3333,0.4000,0.2667}"
REWARD_ALPHA="${REWARD_ALPHA:-0.3333}"
REWARD_BETA="${REWARD_BETA:-0.4000}"
REWARD_GAMMA="${REWARD_GAMMA:-0.2667}"
PREDICTIVE_REWARD_MODE="${PREDICTIVE_REWARD_MODE:-${REWARD_MODE:-resource_efficient}}"
if [[ -z "${PREDICTIVE_DEMAND_AWARE_EMBB_OUTAGE:-}" ]]; then
    if [[ "${PREDICTIVE_REWARD_MODE}" == "paper" ]]; then
        PREDICTIVE_DEMAND_AWARE_EMBB_OUTAGE="0"
    else
        PREDICTIVE_DEMAND_AWARE_EMBB_OUTAGE="1"
    fi
fi
if [[ "${PREDICTIVE_DEMAND_AWARE_EMBB_OUTAGE}" == "1" ]]; then
    PREDICTIVE_DEMAND_AWARE_EMBB_OUTAGE_JSON="true"
else
    PREDICTIVE_DEMAND_AWARE_EMBB_OUTAGE_JSON="false"
fi
RESOURCE_EFFICIENCY_WEIGHT="${RESOURCE_EFFICIENCY_WEIGHT:-0.25}"
NEED_MATCH_WEIGHT="${NEED_MATCH_WEIGHT:-0.30}"
WASTE_PENALTY_WEIGHT="${WASTE_PENALTY_WEIGHT:-0.35}"
UNDER_ALLOCATION_PENALTY_WEIGHT="${UNDER_ALLOCATION_PENALTY_WEIGHT:-0.15}"
EMBB_SOFT_GUARD_PENALTY_WEIGHT="${EMBB_SOFT_GUARD_PENALTY_WEIGHT:-0.25}"
EMBB_SOFT_GUARD_RATIO="${EMBB_SOFT_GUARD_RATIO:-0.80}"
ACTION_SMOOTHNESS_WEIGHT="${ACTION_SMOOTHNESS_WEIGHT:-0.05}"
RESOURCE_DYNAMIC_NEED_WEIGHT="${RESOURCE_DYNAMIC_NEED_WEIGHT:-0.75}"
RESOURCE_WASTE_DEADBAND="${RESOURCE_WASTE_DEADBAND:-0.03}"

FORECAST_SEQUENCE_LEN="${FORECAST_SEQUENCE_LEN:-8}"
FORECAST_HORIZON="${FORECAST_HORIZON:-5}"
FORECASTER_EPOCHS="${FORECASTER_EPOCHS:-30}"
FORECASTER_BATCH_SIZE="${FORECASTER_BATCH_SIZE:-256}"
FORECASTER_LR="${FORECASTER_LR:-0.001}"
FORECASTER_HIDDEN_DIM="${FORECASTER_HIDDEN_DIM:-64}"
FORECASTER_LIMIT_FILES="${FORECASTER_LIMIT_FILES:-0}"

SAC_BUFFER_SIZE="${SAC_BUFFER_SIZE:-50000}"
SAC_BATCH_SIZE="${SAC_BATCH_SIZE:-256}"
SAC_LR="${SAC_LR:-0.001}"
SAC_GAMMA="${SAC_GAMMA:-0.99}"
SAC_TAU="${SAC_TAU:-0.005}"
SAC_ALPHA="${SAC_ALPHA:-0.1}"
RISK_PENALTY="${RISK_PENALTY:-0.5}"
SOFT_PENALTY="${SOFT_PENALTY:-0.2}"
COMPUTE_COST_PER_HOUR_USD="${COMPUTE_COST_PER_HOUR_USD:-0.0}"
COMPUTE_AVG_POWER_WATTS="${COMPUTE_AVG_POWER_WATTS:-0.0}"
COMPUTE_ELECTRICITY_COST_USD_PER_KWH="${COMPUTE_ELECTRICITY_COST_USD_PER_KWH:-0.0}"
BUILD_NS3="${BUILD_NS3:-1}"
RUN_FORECASTER="${RUN_FORECASTER:-1}"
RUN_PREDICTIVE_SAC="${RUN_PREDICTIVE_SAC:-1}"
RUN_COMPARISON="${RUN_COMPARISON:-1}"

mkdir -p "${OUTPUT_ROOT}"

echo "RSLAQ predictive validation"
echo "  line:              ${EXPERIMENT_LINE}"
echo "  output:            ${OUTPUT_ROOT}"
echo "  comparison base:   ${BASELINE_ROOT}"
echo "  forecast source:   ${FORECAST_SOURCE_ROOT}"
echo "  forecast format:   ${FORECAST_SOURCE_FORMAT}"
echo "  scenarios:         ${SCENARIOS[*]}"
echo "  seeds:             ${SEEDS[*]}"
echo "  budget:            ${EPISODES} episodes x ${EPISODE_STEPS} steps = $((EPISODES * EPISODE_STEPS))"
echo "  simTime:           ${SIM_TIME}s"
echo "  forecast:          seq=${FORECAST_SEQUENCE_LEN}, horizon=${FORECAST_HORIZON}"
echo "  penalties:         outage=${RISK_PENALTY}, soft=${SOFT_PENALTY}"
echo "  predictive reward: ${PREDICTIVE_REWARD_MODE}, P_STA=${PREDICTIVE_P_STA_FRACTION}"
echo "  demand-aware eMBB outage: ${PREDICTIVE_DEMAND_AWARE_EMBB_OUTAGE}"
echo "  predictive SLA terminal: ${PREDICTIVE_TERMINATE_ON_SLA_VIOLATION}"
echo "  contribution w:    eff=${RESOURCE_EFFICIENCY_WEIGHT}, match=${NEED_MATCH_WEIGHT}, waste=${WASTE_PENALTY_WEIGHT}, under=${UNDER_ALLOCATION_PENALTY_WEIGHT}, embb_guard=${EMBB_SOFT_GUARD_PENALTY_WEIGHT}, smooth=${ACTION_SMOOTHNESS_WEIGHT}"
echo "  compute cost:      hourly=${COMPUTE_COST_PER_HOUR_USD}, power=${COMPUTE_AVG_POWER_WATTS}W, electricity=${COMPUTE_ELECTRICITY_COST_USD_PER_KWH}/kWh"
echo "  run forecaster:    ${RUN_FORECASTER}"
echo "  run predictive SAC:${RUN_PREDICTIVE_SAC}"
echo "  run comparison:    ${RUN_COMPARISON}"

if [[ "${BUILD_NS3}" == "1" ]]; then
    cd "${NS3_DIR}"
    ./ns3 build rslaq-sim
else
    echo "[build] Skipping ns-3 build because BUILD_NS3=${BUILD_NS3}"
fi

cd "${GYM_DIR}"

FORECASTER_DIR="${OUTPUT_ROOT}/forecaster"
if [[ "${RUN_FORECASTER}" == "1" ]]; then
    python3 examples/rslaq_train_forecaster.py \
        --source_root "${FORECAST_SOURCE_ROOT}" \
        --source_format "${FORECAST_SOURCE_FORMAT}" \
        --output "${FORECASTER_DIR}" \
        --sequence_len "${FORECAST_SEQUENCE_LEN}" \
        --forecast_horizon "${FORECAST_HORIZON}" \
        --hidden_dim "${FORECASTER_HIDDEN_DIM}" \
        --epochs "${FORECASTER_EPOCHS}" \
        --batch_size "${FORECASTER_BATCH_SIZE}" \
        --lr "${FORECASTER_LR}" \
        --compute_cost_per_hour_usd "${COMPUTE_COST_PER_HOUR_USD}" \
        --compute_avg_power_watts "${COMPUTE_AVG_POWER_WATTS}" \
        --compute_electricity_cost_usd_per_kwh "${COMPUTE_ELECTRICITY_COST_USD_PER_KWH}" \
        --limit_files "${FORECASTER_LIMIT_FILES}"
else
    echo "[forecaster] Skipping forecaster training because RUN_FORECASTER=${RUN_FORECASTER}"
fi

FORECASTER_CHECKPOINT="${FORECASTER_DIR}/forecaster_best.pt"
predictive_terminal_args=()
predictive_outage_args=()
if [[ "${PREDICTIVE_TERMINATE_ON_SLA_VIOLATION}" == "1" ]]; then
    predictive_terminal_args+=(--terminate-on-sla-violation)
fi
if [[ "${PREDICTIVE_DEMAND_AWARE_EMBB_OUTAGE}" != "1" ]]; then
    predictive_outage_args+=(--paper-faithful-outage)
fi

if [[ "${RUN_PREDICTIVE_SAC}" == "1" ]]; then
    for scenario in "${SCENARIOS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "[Predictive SAC] scenario=${scenario} seed=${seed}"
            python3 examples/rslaq_train_predictive_sac.py \
                --scenario "${scenario}" \
                --episodes "${EPISODES}" \
                --seed "${seed}" \
                --seed_cycle "${SEED_CYCLE}" \
                --simTime "${SIM_TIME}" \
                --appStart "${APP_START}" \
                --periodMs "${PERIOD_MS}" \
                --max_steps "${EPISODE_STEPS}" \
                --observation_mode paper \
                --consecutive_outage_steps "${CONSECUTIVE_OUTAGE_STEPS}" \
                --warmup_steps "${WARMUP_STEPS}" \
                --buffer_size "${SAC_BUFFER_SIZE}" \
                --batch_size "${SAC_BATCH_SIZE}" \
                --lr "${SAC_LR}" \
                --gamma "${SAC_GAMMA}" \
                --tau "${SAC_TAU}" \
                --alpha "${SAC_ALPHA}" \
                --p_sta_static_fraction "${PREDICTIVE_P_STA_FRACTION}" \
                --p_sta_weights "${P_STA_WEIGHTS}" \
                --reward_alpha "${REWARD_ALPHA}" \
                --reward_beta "${REWARD_BETA}" \
                --reward_gamma "${REWARD_GAMMA}" \
                --reward_mode "${PREDICTIVE_REWARD_MODE}" \
                --resource_efficiency_weight "${RESOURCE_EFFICIENCY_WEIGHT}" \
                --need_match_weight "${NEED_MATCH_WEIGHT}" \
                --waste_penalty_weight "${WASTE_PENALTY_WEIGHT}" \
                --under_allocation_penalty_weight "${UNDER_ALLOCATION_PENALTY_WEIGHT}" \
                --embb_soft_guard_penalty_weight "${EMBB_SOFT_GUARD_PENALTY_WEIGHT}" \
                --embb_soft_guard_ratio "${EMBB_SOFT_GUARD_RATIO}" \
                --action_smoothness_weight "${ACTION_SMOOTHNESS_WEIGHT}" \
                --resource_dynamic_need_weight "${RESOURCE_DYNAMIC_NEED_WEIGHT}" \
                --resource_waste_deadband "${RESOURCE_WASTE_DEADBAND}" \
                --risk_penalty "${RISK_PENALTY}" \
                --soft_penalty "${SOFT_PENALTY}" \
                --forecaster_checkpoint "${FORECASTER_CHECKPOINT}" \
                --sequence_len "${FORECAST_SEQUENCE_LEN}" \
                --compute_cost_per_hour_usd "${COMPUTE_COST_PER_HOUR_USD}" \
                --compute_avg_power_watts "${COMPUTE_AVG_POWER_WATTS}" \
                --compute_electricity_cost_usd_per_kwh "${COMPUTE_ELECTRICITY_COST_USD_PER_KWH}" \
                "${predictive_terminal_args[@]}" \
                "${predictive_outage_args[@]}" \
                --output "${OUTPUT_ROOT}/predictive_sac_${scenario}_seed${seed}"
        done
    done
else
    echo "[Predictive SAC] Skipping policy training because RUN_PREDICTIVE_SAC=${RUN_PREDICTIVE_SAC}"
fi

if [[ "${RUN_COMPARISON}" == "1" ]]; then
    python3 examples/compare_rslaq_campaigns.py \
        --baseline_root "${BASELINE_ROOT}" \
        --predictive_root "${OUTPUT_ROOT}" \
        --output "${OUTPUT_ROOT}/comparison"
else
    echo "[comparison] Skipping comparison because RUN_COMPARISON=${RUN_COMPARISON}"
fi

cat > "${OUTPUT_ROOT}/campaign_config.json" <<EOF
{
  "experiment_line": "${EXPERIMENT_LINE}",
  "run_tag": "${RUN_TAG}",
  "baseline_root": "${BASELINE_ROOT}",
  "forecast_source_root": "${FORECAST_SOURCE_ROOT}",
  "forecast_source_format": "${FORECAST_SOURCE_FORMAT}",
  "scenarios": "${SCENARIOS[*]}",
  "seeds": "${SEEDS[*]}",
  "episodes": ${EPISODES},
  "episode_steps": ${EPISODE_STEPS},
  "interaction_budget": $((EPISODES * EPISODE_STEPS)),
  "sim_time": ${SIM_TIME},
  "app_start": ${APP_START},
  "period_ms": ${PERIOD_MS},
  "forecast_sequence_len": ${FORECAST_SEQUENCE_LEN},
  "forecast_horizon": ${FORECAST_HORIZON},
  "risk_penalty": ${RISK_PENALTY},
  "soft_penalty": ${SOFT_PENALTY},
  "p_sta_static_fraction": ${PREDICTIVE_P_STA_FRACTION},
  "p_sta_weights": "${P_STA_WEIGHTS}",
  "reward_mode": "${PREDICTIVE_REWARD_MODE}",
  "predictive_terminate_on_sla_violation": ${PREDICTIVE_TERMINATE_ON_SLA_VIOLATION},
  "reward_weights": "${REWARD_ALPHA},${REWARD_BETA},${REWARD_GAMMA}",
  "resource_efficient_reward": {
    "resource_efficiency_weight": ${RESOURCE_EFFICIENCY_WEIGHT},
    "need_match_weight": ${NEED_MATCH_WEIGHT},
    "waste_penalty_weight": ${WASTE_PENALTY_WEIGHT},
    "under_allocation_penalty_weight": ${UNDER_ALLOCATION_PENALTY_WEIGHT},
    "embb_soft_guard_penalty_weight": ${EMBB_SOFT_GUARD_PENALTY_WEIGHT},
    "embb_soft_guard_ratio": ${EMBB_SOFT_GUARD_RATIO},
    "action_smoothness_weight": ${ACTION_SMOOTHNESS_WEIGHT},
    "resource_dynamic_need_weight": ${RESOURCE_DYNAMIC_NEED_WEIGHT},
    "resource_waste_deadband": ${RESOURCE_WASTE_DEADBAND}
  },
  "compute_cost": {
    "cost_per_hour_usd": ${COMPUTE_COST_PER_HOUR_USD},
    "avg_power_watts": ${COMPUTE_AVG_POWER_WATTS},
    "electricity_cost_usd_per_kwh": ${COMPUTE_ELECTRICITY_COST_USD_PER_KWH}
  },
  "build_ns3": ${BUILD_NS3},
  "run_forecaster": ${RUN_FORECASTER},
  "run_predictive_sac": ${RUN_PREDICTIVE_SAC},
  "run_comparison": ${RUN_COMPARISON},
  "demand_aware_embb_outage": ${PREDICTIVE_DEMAND_AWARE_EMBB_OUTAGE_JSON}
}
EOF

echo "Predictive campaign complete: ${OUTPUT_ROOT}"
