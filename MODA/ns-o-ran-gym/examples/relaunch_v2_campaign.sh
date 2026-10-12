#!/usr/bin/env bash
# Relaunch the v2 campaign (run tag v2_all_scenarios) after a reboot or a kill.
#
# Safe to run at any time: every finished evaluation is checkpointed in a
# candidate.json sidecar and replayed from cache (the optimizer RNG is
# random_seed + ns3_seed, consumed in a fixed GA->PSO->SA->hybrid order, so the
# search trajectory is reproduced exactly). Baselines resume from
# batch_manifest.csv via RESUME=1. Only the evaluations that were in flight when
# the process died are re-simulated (at most PARALLEL_JOBS of them).
#
# The parameters below are the ones the original master process was launched
# with; changing META_POPULATION or the RNG seeding WOULD invalidate the cache.
#
#   bash examples/relaunch_v2_campaign.sh
set -euo pipefail

cd "$(dirname "$0")/.."

REPO_ROOT="${REPO_ROOT:-/home/eliothluy/Documentos/artigo_jussi}"
RUN_TAG="v2_all_scenarios"
OUTPUT_ROOT="$PWD/results_controlled/heuristics_metaheuristics_v2/$RUN_TAG"

if pgrep -f "run_rslaq_metaheuristics.py" > /dev/null; then
    echo "A campanha já está rodando ($(pgrep -cf run_rslaq_metaheuristics.py) runners). Nada a fazer."
    exit 0
fi

mkdir -p "$OUTPUT_ROOT"

REPO_ROOT="$REPO_ROOT" \
RUN_TAG="$RUN_TAG" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
SCENARIOS="low_traffic normal stressed insufficient_resources" \
SEEDS="1 2 3 4 5 6 7 8 9 10" \
SIM_TIME=5 \
META_ITERATIONS=4 \
META_POPULATION=6 \
META_SCORE_VERSION=v2 \
META_PER_SEED_SEARCH=1 \
RUN_BASELINES=1 \
RUN_METAHEURISTICS=1 \
RUN_META_EVALUATION=0 \
RUN_RSLAQ_DDQN_PAPER=0 \
PARALLEL_JOBS=10 \
RESUME=1 \
setsid nohup bash examples/run_all_scenarios.sh \
    >> "$OUTPUT_ROOT/../${RUN_TAG}.log" 2>&1 &

echo "Campanha relançada (PID $!). Log: $OUTPUT_ROOT/../${RUN_TAG}.log"
echo "Dashboard: setsid nohup python3 examples/v2_dashboard.py > /dev/null 2>&1 &"
