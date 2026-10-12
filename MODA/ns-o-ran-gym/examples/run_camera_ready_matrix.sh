#!/usr/bin/env bash
# WPMC 2026 camera-ready — full matrix (Fase 3), gated by the validated Fase 2.
#
# Order (per approved plan): baselines -> low_traffic -> normal -> congestion.
# Each scenario wave launches 5 (scenario,seed) pairs; each pair runs 2
# processes (meta + grid in SEPARATE output roots — fixes the CSV overwrite
# race found in the gate). Concurrency policy is CONSTANT across waves
# (10 processes) so wall-clock elapsed_s stays comparable between methods.
#
# V1 protocol frozen: iter=12 pop=6 (SA auto-equalized to 72 by the runner),
# mutation=0.12, random_seed=2026 (no per-seed search), intra=PF, score v1,
# SIM_TIME=5, APP_START=0.4. ns-3 seeds: 4 5 6 7 8.
#
# REQUIRES the v1 binary (scratch from backup_v1_scratch_rslaq_20260626).
# The v1 scratch files are NOT restored to v2 at the end automatically — the
# camera-ready may need follow-up runs. To restore manually when done:
#   cd ns-3-dev && git checkout -- scratch/rslaq/ && ./ns3 build rslaq-sim
#
# To stop: pkill -f run_rslaq_metaheuristics.py ; checkpoint sidecars survive.
set -uo pipefail

REPO_ROOT=/home/eliothluy/Documentos/artigo_jussi
GYM="${REPO_ROOT}/ns-o-ran-gym"
NS3_DIR="${REPO_ROOT}/ns-3-dev"
CR_ROOT="${GYM}/results_controlled/v1_camera_ready"
LOGS="${CR_ROOT}/_logs"
mkdir -p "${LOGS}"

ts() { date '+%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

run_pair() {  # scenario seed — appends the 2 child PIDs to $PID_FILE
  local sc="$1" seed="$2"
  local common="--scenarios ${sc} --seeds ${seed} --run 1 --random_seed 2026 \
    --mutation_strength 0.12 --intra_algo PF --score_version v1 \
    --sim_time 5 --app_start 0.4 --drain_time 0.2 --period_ms 10 --tx_power 43 \
    --tdd_pattern D|D|8D|4GB|4U|U|U --rlc_mode um \
    --ns3_dir ${NS3_DIR} --iterations 12 --population 6"

  # meta (GA72/PSO72/SA72/HYB84)
  nohup python3 "${GYM}/examples/run_rslaq_metaheuristics.py" \
    --method all ${common} \
    --output_root "${CR_ROOT}/metaheuristics" \
    > "${LOGS}/${sc}_seed${seed}_meta.log" 2>&1 &
  local m=$!
  # grid (231) — separate root (fixes CSV race found in the gate)
  nohup python3 "${GYM}/examples/run_rslaq_metaheuristics.py" \
    --method grid --grid_step 0.05 ${common} \
    --output_root "${CR_ROOT}/metaheuristics_grid" \
    > "${LOGS}/${sc}_seed${seed}_grid.log" 2>&1 &
  local g=$!
  echo "${m}" >> "${PID_FILE}"
  echo "${g}" >> "${PID_FILE}"
  log "  pair ${sc}/seed${seed}: meta pid=${m} grid pid=${g}"
}

wave() {  # scenario — runs run_pair in THIS shell so `wait` works
  local sc="$1"
  PID_FILE="${LOGS}/pids_${sc}.txt"
  rm -f "${PID_FILE}"
  log "WAVE ${sc}: launching 5 pairs (meta+grid each)..."
  for seed in 4 5 6 7 8; do
    run_pair "${sc}" "${seed}"
    sleep 2
  done
  log "WAVE ${sc}: waiting for $(wc -l < "${PID_FILE}") processes..."
  local fail=0
  while read -r pid; do
    wait "${pid}" 2>/dev/null
    local rc=$?
    if [ "${rc}" -ne 0 ]; then
      fail=1
      log "  [WARN] pid ${pid} exited rc=${rc}"
    fi
  done < "${PID_FILE}"
  log "WAVE ${sc} DONE (fail=${fail})"
}

# ---------- F3.1 baselines (RR/PF/BCQI x 3 scenarios x 5 seeds) ----------
log "F3.1 baselines (45 runs, pool=10)..."
cd "${NS3_DIR}"
OUTPUT_ROOT="${CR_ROOT}/baselines_ns3" \
SCENARIOS="low_traffic normal congestion" \
BASELINE_MODES="pure_rr pure_pf pure_bcqi" \
SEEDS="4 5 6 7 8" RUNS="1" \
SIM_TIME=5 APP_START=0.4 DRAIN_TIME_SEC=0.2 PERIOD_MS=10 TX_POWER=43 \
TDD_PATTERN='D|D|8D|4GB|4U|U|U' RLC_MODE=um \
BUILD_NS3=0 PARALLEL_JOBS=10 RESUME=1 \
bash ./run_all_scenarios.sh > "${LOGS}/baselines.log" 2>&1 \
  && log "F3.1 baselines OK" || log "[WARN] F3.1 baselines reported issues — see ${LOGS}/baselines.log"

# ---------- F3.2-F3.4 waves ----------
wave low_traffic
wave normal
wave congestion

log "============================================================"
log "CAMERA-READY MATRIX COMPLETE"
log "Reminders: (1) v1 binary still in place — restore v2 when done:"
log "  cd ns-3-dev && git checkout -- scratch/rslaq/ && ./ns3 build rslaq-sim"
log "(2) Consolidation: analyze_camera_ready (sidecars = source of truth)."
log "============================================================"
