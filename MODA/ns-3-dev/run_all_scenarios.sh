#!/usr/bin/env bash
# Run every RSLAQ network-only baseline over every traffic scenario.
#
# Parallel version: runs N simulations concurrently using a bash job pool.
# Each simulation writes to its own output directory, so there are no file
# conflicts. Results are collected in per-job temp files and merged into
# the manifest after all jobs finish (no race conditions on the manifest).
#
# Resume support: if the manifest already exists and RESUME=1, jobs marked
# "ok" are skipped and their entries are preserved in the new manifest.
#
# Override examples:
#   PARALLEL_JOBS=6 SIM_TIME=5 SEEDS="1 2 3" ./run_all_scenarios.sh
#   RESUME=0 PARALLEL_JOBS=8 BASELINE_MODES="pure_pf slice_aqps" ./run_all_scenarios.sh

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SCENARIOS_INPUT="${SCENARIOS:-}"
BASELINE_MODES_INPUT="${BASELINE_MODES:-}"

NS3_DIR="${NS3_DIR:-/home/eliothluy/Documentos/artigo_jussi/ns-3-dev}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$NS3_DIR}"
RESULTS_DIR="$OUTPUT_ROOT/results_rslaq_network_only"
LOG_DIR="$RESULTS_DIR/_logs"
MANIFEST="$RESULTS_DIR/batch_manifest.csv"

SIM_TIME="${SIM_TIME:-5}"
APP_START="${APP_START:-0.4}"
DRAIN_TIME_SEC="${DRAIN_TIME_SEC:-0.2}"
PERIOD_MS="${PERIOD_MS:-10}"
TX_POWER="${TX_POWER:-43}"
TDD_PATTERN="${TDD_PATTERN:-D|D|8D|4GB|4U|U|U}"
RLC_MODE="${RLC_MODE:-um}"
SEEDS="${SEEDS:-1}"
RUNS="${RUNS:-1}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
BUILD_NS3="${BUILD_NS3:-1}"

# Parallel execution
PARALLEL_JOBS="${PARALLEL_JOBS:-6}"
RESUME="${RESUME:-1}"

# ---------------------------------------------------------------------------
# Default scenario/mode lists
# ---------------------------------------------------------------------------
SCENARIOS=(
    low_traffic
    normal
    congestion
    stressed
    insufficient_resources
)

BASELINE_MODES=(
    pure_rr
    pure_pf
    pure_bcqi
    slice_rr
    slice_pf
    slice_bcqi
    slice_weighted_pf
    slice_weighted_rr
    slice_weighted_bcqi
    psta_equal
    slice_demand_greedy
    slice_sla_greedy
    slice_least_waste
    slice_qos_mixed
    slice_random_vine
    slice_meta_risk_elastic
    slice_aqps
)

if [[ -n "$SCENARIOS_INPUT" ]]; then
    read -r -a SCENARIOS <<< "$SCENARIOS_INPUT"
fi

if [[ -n "$BASELINE_MODES_INPUT" ]]; then
    read -r -a BASELINE_MODES <<< "$BASELINE_MODES_INPUT"
fi

read -r -a SEED_LIST <<< "$SEEDS"
read -r -a RUN_LIST <<< "$RUNS"

mkdir -p "$LOG_DIR"
cd "$NS3_DIR"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "============================================"
echo "RSLAQ baseline matrix (parallel)"
echo "============================================"
echo "Started at      : $(date)"
echo "Results root    : $RESULTS_DIR"
echo "Logs            : $LOG_DIR"
echo "Scenarios       : ${SCENARIOS[*]}"
echo "Baselines       : ${BASELINE_MODES[*]}"
echo "Seeds           : ${SEED_LIST[*]}"
echo "Runs            : ${RUN_LIST[*]}"
echo "Sim config      : simTime=${SIM_TIME}s appStart=${APP_START}s drain=${DRAIN_TIME_SEC}s period=${PERIOD_MS}ms rlcMode=${RLC_MODE}"
echo "Parallel jobs   : ${PARALLEL_JOBS}"
echo "Resume mode     : ${RESUME}"
echo ""

# ---------------------------------------------------------------------------
# Build ns-3 (once, before launching any parallel jobs)
# ---------------------------------------------------------------------------
if [[ "$BUILD_NS3" == "1" ]]; then
    echo "Building rslaq-sim..."
    ./ns3 build rslaq-sim
else
    echo "Skipping rslaq-sim build (BUILD_NS3=$BUILD_NS3)"
fi
echo ""

# ---------------------------------------------------------------------------
# Validation function (same as original)
# ---------------------------------------------------------------------------
validate_outputs() {
    local result_dir="$1"
    local required=(
        "$result_dir/timeseries.csv"
        "$result_dir/slice_alloc.csv"
        "$result_dir/summary.csv"
        "$result_dir/metadata.json"
    )

    local path
    for path in "${required[@]}"; do
        if [[ ! -s "$path" ]]; then
            echo "missing_or_empty: $path"
            return 1
        fi
    done

    if ! head -n 1 "$result_dir/timeseries.csv" | grep -q "loss_pct_interval,buffer_bytes"; then
        echo "invalid timeseries.csv header"
        return 1
    fi

    if ! head -n 1 "$result_dir/slice_alloc.csv" | grep -q "rsh_real_pct"; then
        echo "invalid slice_alloc.csv header"
        return 1
    fi

    if ! head -n 1 "$result_dir/summary.csv" | grep -q "buffer_bytes_mean.*rsh_real_pct_mean"; then
        echo "invalid summary.csv header"
        return 1
    fi

    if ! grep -q '"baseline_mode"' "$result_dir/metadata.json"; then
        echo "invalid metadata.json"
        return 1
    fi

    echo "ok"
    return 0
}

# ---------------------------------------------------------------------------
# Worker function: runs a single ns-3 simulation in a subshell.
# Writes the result line to a per-job temp file (no manifest race).
# Always returns 0 — errors are captured in the result file.
# ---------------------------------------------------------------------------
run_worker() {
    local scenario="$1" mode="$2" seed="$3" run="$4"
    local result_dir="$RESULTS_DIR/scenario=${scenario}/mode=${mode}/seed=${seed}_run=${run}"
    local log_file="$LOG_DIR/${scenario}__${mode}__seed=${seed}_run=${run}.log"
    local result_file="$LOG_DIR/.result_${scenario}_${mode}_s${seed}_r${run}"

    local exit_code=0
    (
        cd "$NS3_DIR"
        ./ns3 run "scratch/rslaq/rslaq-sim \
            --scenario=${scenario} \
            --baselineMode=${mode} \
            --simTime=${SIM_TIME} \
            --appStart=${APP_START} \
            --drainTimeSec=${DRAIN_TIME_SEC} \
            --periodMs=${PERIOD_MS} \
            --seed=${seed} \
            --run=${run} \
            --txPower=${TX_POWER} \
            --tddPattern=${TDD_PATTERN} \
            --rlcMode=${RLC_MODE} \
            --outputDir=${OUTPUT_ROOT}"
    ) > "$log_file" 2>&1 || exit_code=$?

    if [[ "$exit_code" -eq 0 ]]; then
        local validation_message
        validation_message="$(validate_outputs "$result_dir")"
        if [[ "$validation_message" == "ok" ]]; then
            echo "${scenario},${mode},${seed},${run},ok,${result_dir},${log_file},${validation_message}" > "$result_file"
        else
            echo "${scenario},${mode},${seed},${run},validation_failed,${result_dir},${log_file},${validation_message}" > "$result_file"
        fi
    else
        echo "${scenario},${mode},${seed},${run},run_failed,${result_dir},${log_file},ns3_run_failed" > "$result_file"
    fi

    return 0
}

# ---------------------------------------------------------------------------
# Resume: collect already-"ok" entries from existing manifest
# ---------------------------------------------------------------------------
old_ok_entries=""
if [[ "$RESUME" == "1" ]] && [[ -f "$MANIFEST" ]]; then
    old_ok_entries="$(tail -n +2 "$MANIFEST" | grep ",ok," 2>/dev/null || true)"
    # Count only when there is something to count. `grep -c .` on an empty
    # string prints "0" *and* exits 1, so a `|| echo 0` fallback appends a
    # second "0" and the variable becomes the two-line string "0\n0", which
    # blows up the arithmetic below. That aborted every resume whose manifest
    # held no ",ok," rows -- silently skipping jobs the resume was meant to run.
    if [[ -n "$old_ok_entries" ]]; then
        old_ok_count="$(printf '%s\n' "$old_ok_entries" | wc -l)"
    else
        old_ok_count=0
    fi
    echo "Resume: found ${old_ok_count} already-completed job(s) in existing manifest."
else
    old_ok_count=0
fi

# Start fresh manifest with header + preserved "ok" entries
echo "scenario,baseline_mode,seed,run,status,result_dir,log_file,message" > "$MANIFEST"
if [[ -n "$old_ok_entries" ]]; then
    echo "$old_ok_entries" >> "$MANIFEST"
fi

# Build skip set from old "ok" entries
declare -A skip_set
if [[ -n "$old_ok_entries" ]]; then
    while IFS=, read -r sc md sd rn _; do
        skip_set["${sc}|${md}|${sd}|${rn}"]=1
    done <<< "$old_ok_entries"
fi

# ---------------------------------------------------------------------------
# Generate job list (excluding already-ok jobs)
# ---------------------------------------------------------------------------
jobs_list=()
for scenario in "${SCENARIOS[@]}"; do
    for mode in "${BASELINE_MODES[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
            for run in "${RUN_LIST[@]}"; do
                local_key="${scenario}|${mode}|${seed}|${run}"
                if [[ -z "${skip_set[$local_key]:-}" ]]; then
                    jobs_list+=("${scenario}|${mode}|${seed}|${run}")
                fi
            done
        done
    done
done

total_to_run=${#jobs_list[@]}
total=$(( total_to_run + old_ok_count ))
skipped=$old_ok_count

echo ""
echo "Total jobs      : ${total}"
echo "Already OK      : ${skipped}"
echo "To run          : ${total_to_run}"
echo "Parallel workers: ${PARALLEL_JOBS}"
echo ""

if [[ "$total_to_run" -eq 0 ]]; then
    echo "All jobs already completed. Nothing to do."
    echo "Manifest: $MANIFEST"
    exit 0
fi

# ---------------------------------------------------------------------------
# Launch parallel jobs with throttling
# ---------------------------------------------------------------------------
count=0
failed_count=0
abort=0

for job_spec in "${jobs_list[@]}"; do
    IFS='|' read -r scenario mode seed run <<< "$job_spec"
    count=$((count + 1))
    echo "[$count/$total_to_run] scenario=$scenario mode=$mode seed=$seed run=$run"

    run_worker "$scenario" "$mode" "$seed" "$run" &

    # Throttle: wait for a slot if at capacity
    while (( $(jobs -rp | wc -l) >= PARALLEL_JOBS )); do
        wait -n 2>/dev/null || sleep 0.1
    done

    # Check if any job failed and STOP_ON_FAILURE is set
    if [[ "$STOP_ON_FAILURE" == "1" ]]; then
        # Check result files for failures
        for rf in "$LOG_DIR"/.result_*; do
            [[ -f "$rf" ]] || continue
            if grep -qE ",(run_failed|validation_failed)," "$rf" 2>/dev/null; then
                echo "STOP_ON_FAILURE: a job failed. Waiting for running jobs to finish..."
                abort=1
                break
            fi
        done
        [[ "$abort" -eq 1 ]] && break
    fi
done

# Wait for all remaining jobs
echo ""
echo "Waiting for remaining jobs to finish..."
wait

# ---------------------------------------------------------------------------
# Merge per-job result files into the manifest
# ---------------------------------------------------------------------------
new_ok=0
new_failed=0

for result_file in "$LOG_DIR"/.result_*; do
    [[ -f "$result_file" ]] || continue
    cat "$result_file" >> "$MANIFEST"
    if grep -q ",ok," "$result_file" 2>/dev/null; then
        new_ok=$((new_ok + 1))
    else
        new_failed=$((new_failed + 1))
        # Show failed job details
        echo "FAILED: $(cat "$result_file")"
    fi
    rm -f "$result_file"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total_ok=$((new_ok + skipped))
total_failed=$new_failed

echo ""
echo "============================================"
echo "Batch complete at $(date)"
echo "============================================"
echo "Manifest         : $MANIFEST"
echo "Results          : $RESULTS_DIR"
echo "Total jobs       : ${total}"
echo "OK (preserved)   : ${skipped}"
echo "OK (new)         : ${new_ok}"
echo "Failed           : ${new_failed}"
echo "Total OK         : ${total_ok}"
echo "Total failed     : ${total_failed}"

if (( total_failed > 0 )); then
    echo ""
    echo "Failed jobs:"
    tail -n +2 "$MANIFEST" | grep -vE ",ok," | while IFS=, read -r sc md sd rn st _ _ _; do
        echo "  $sc / $md / seed=$sd run=$rn -> $st"
    done
    exit 1
fi

echo ""
echo "All runs completed and metric files validated."
