#!/bin/bash
# Run all RSLAQ training scenarios with paper mode configuration

set -e

DEFAULT_NS3_PATH="../ns-3-dev"
OUTPUT_DIR="results_paper"
EPISODES=20
MAX_STEPS=150

echo "======================================"
echo "RSLAQ Training - Paper Mode"
echo "======================================"

# Parse command line arguments
SCENARIO_ARG="${1:-all}"
EPISODES_ARG="${2:-20}"
MAX_STEPS_ARG="${3:-150}"

if [ "$SCENARIO_ARG" != "all" ]; then
    echo "Single scenario: $SCENARIO_ARG"
fi

# Update from args if provided
if [ "$EPISODES_ARG" != "20" ]; then
    EPISODES=$EPISODES_ARG
fi
if [ "$MAX_STEPS_ARG" != "150" ]; then
    MAX_STEPS=$MAX_STEPS_ARG
fi

echo "Configuration:"
echo "  Output dir: $OUTPUT_DIR"
echo "  Episodes: $EPISODES"
echo "  Max steps: $MAX_STEPS"

mkdir -p "$OUTPUT_DIR"

# DDQN Training (with scheduler action)
echo ""
echo "--------------------------------------"
echo "Training DDQN (with scheduler)..."
echo "--------------------------------------"

python3 examples/rslaq_train_ddqn.py \
    --scenario "$SCENARIO_ARG" \
    --ns3_path "$DEFAULT_NS3_PATH" \
    --output "$OUTPUT_DIR" \
    --episodes "$EPISODES" \
    --max_steps "$MAX_STEPS" \
    --action_mode discrete \
    --observation_mode paper \
    --apply_p_sta False \
    --include_scheduler \
    --mtc_is_no_policy True \
    --use_real_bfs False \
    2>&1 | tee "$OUTPUT_DIR/ddqn_training.log"

# SAC Training (baseline with PF)
echo ""
echo "--------------------------------------"
echo "Training SAC (baseline)..."
echo "--------------------------------------"

python3 examples/rslaq_train_sac.py \
    --scenario "$SCENARIO_ARG" \
    --ns3_path "$DEFAULT_NS3_PATH" \
    --output "$OUTPUT_DIR" \
    --episodes "$EPISODES" \
    --max_steps "$MAX_STEPS" \
    --action_mode continuous \
    --observation_mode paper \
    --apply_p_sta False \
    --mtc_is_no_policy True \
    --use_real_bfs False \
    2>&1 | tee "$OUTPUT_DIR/sac_training.log"

echo ""
echo "======================================"
echo "Training complete!"
echo "======================================"
echo "Results saved to: $OUTPUT_DIR"
echo "Check logs for details:"
echo "  - $OUTPUT_DIR/ddqn_training.log"
echo "  - $OUTPUT_DIR/sac_training.log"
