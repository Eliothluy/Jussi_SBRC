#!/usr/bin/env bash
# V1 metaheuristics extension: seeds 4-8 on low_traffic, normal, congestion.
#
# Faithful v1 protocol: iter=12, pop=6, mutation=0.12, random_seed=2026,
# intra_algo=PF, score v1, SIM_TIME=5, APP_START=0.4 — one ns-3 run per
# candidate, 300 evals per (scenario, seed) pair (GA 72 + PSO 72 + SA 72 +
# hybrid 84). Requires the v1 binary (scratch restored from
# backup_v1_scratch_rslaq_20260626/ and rebuilt — done by the launch step).
#
# 15 pairs run as independent processes (searches are sequential within a
# pair; parallelism is across pairs). When ALL pairs finish, the v2 scratch
# files are restored from git and the v2 binary rebuilt automatically.
set -uo pipefail

REPO_ROOT=/home/eliothluy/Documentos/artigo_jussi
GYM="${REPO_ROOT}/ns-o-ran-gym"
NS3_DIR="${REPO_ROOT}/ns-3-dev"
OUT_ROOT="${GYM}/results_controlled/v1_extra_seeds_4to8"
LOG_DIR="${OUT_ROOT}/_logs"
mkdir -p "${LOG_DIR}"

PIDS=()
for sc in low_traffic normal congestion; do
  for seed in 4 5 6 7 8; do
    log_file="${LOG_DIR}/${sc}_seed${seed}.log"
    echo "[$(date '+%m-%d %H:%M:%S')] launching pair ${sc} seed=${seed}"
    nohup python3 "${GYM}/examples/run_rslaq_metaheuristics.py" \
        --method all \
        --scenarios "${sc}" \
        --seeds "${seed}" \
        --run 1 \
        --random_seed 2026 \
        --iterations 12 \
        --population 6 \
        --mutation_strength 0.12 \
        --intra_algo PF \
        --score_version v1 \
        --sim_time 5 \
        --app_start 0.4 \
        --drain_time 0.2 \
        --period_ms 10 \
        --tx_power 43 \
        --tdd_pattern 'D|D|8D|4GB|4U|U|U' \
        --rlc_mode um \
        --ns3_dir "${NS3_DIR}" \
        --output_root "${OUT_ROOT}/metaheuristics" \
        > "${log_file}" 2>&1 &
    PIDS+=($!)
    sleep 2   # stagger starts (build dirs, avoid thundering herd)
  done
done

echo "[$(date '+%m-%d %H:%M:%S')] 15 pairs launched: ${PIDS[*]}"

FAIL=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  wait "${pid}" || { echo "[WARN] pair process pid=${pid} exited non-zero"; FAIL=1; }
  echo "[$(date '+%m-%d %H:%M:%S')] pair pid=${pid} finished"
done

echo "[$(date '+%m-%d %H:%M:%S')] all pairs done (fail=${FAIL}). Restoring v2 scratch from git..."
cd "${REPO_ROOT}/ns-3-dev"
git checkout -- scratch/rslaq/rslaq-sim.cc scratch/rslaq/rslaq-mac-scheduler.cc scratch/rslaq/rslaq-mac-scheduler.h
./ns3 build rslaq-sim -j 10 >> "${LOG_DIR}/restore_v2.log" 2>&1 \
  && echo "[$(date '+%m-%d %H:%M:%S')] v2 binary restored+rebuilt OK" \
  || echo "[$(date '+%m-%d %H:%M:%S')] [WARN] v2 restore failed — rebuild manually (git checkout + ./ns3 build rslaq-sim)"

echo "[$(date '+%m-%d %H:%M:%S')] V1 EXTRA-SEEDS CAMPAIGN COMPLETE"
