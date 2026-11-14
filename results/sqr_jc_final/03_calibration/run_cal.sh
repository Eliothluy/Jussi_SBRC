#!/usr/bin/env bash
# run_cal.sh — FASE CAL independente (protocolo Etapa 2.5 §32; gates G4/G5).
#
# CAL_SEEDS = {1001,1002,1003} × lambda_total ∈ {2,5,6} × 300 s, com o
# SQR-JC em modo BOOTSTRAP-PROP PURO: as ações de CAL são exatamente as do
# controlador prop congelado (§29 — todos os controladores aprendentes usam
# o MESMO PROP no bootstrap), então as trajetórias ns-3 reusam as MESMAS
# flags congeladas da Rodada 2.  Os kernels do SQR-JC e os scalers são
# CALIBRADOS OFFLINE pelo replay determinístico do learner sobre as janelas
# logadas (examples/sqr_jc_calibrate.py), que produz:
#   - feature_scaler_manifest.yaml   (§13: p1/p99 de contagens/filas, CAL only)
#   - sigma_support_manifest.yaml    (§16: grid de percentis + block bootstrap)
#   - kernel_log.csv por run         (§57: suporte/evictions/tau/M/tempos)
#   - search_benchmark.json          (§37: p95 de decisão em 100 estados CAL)
# Dados CAL NÃO reportam desempenho principal (§32).
#
# Lições congeladas: LC_ALL=C (R56), .done idempotente, 3 tentativas,
# timeout 1800 s/run, lock flock, limpeza de semáforos órfãos.
#
# Uso (orquestrador; este agente NÃO executa ns-3):
#   LC_ALL=C bash -c 'results/sqr_jc_final/03_calibration/run_cal.sh'
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$_SCRIPT_DIR"                                             # saída junto à cópia invocada
# GYM resolve em QUALQUER cópia deste script: canônica (ns-o-ran-gym) ou
# espelho na raiz do projeto (Jussi_SBRC/results/sqr_jc_final).
GYM="$(cd "$_SCRIPT_DIR/../../.." && pwd)"
if [[ ! -x "${GYM}/.venv/bin/python" ]]; then
    GYM="$(cd "${GYM}/MODA/ns-o-ran-gym" && pwd)"
fi
if [[ ! -x "${GYM}/.venv/bin/python" ]]; then
    # último recurso: raiz do repositório Jussi_SBRC -> MODA/ns-o-ran-gym
    GYM="$(cd "$_SCRIPT_DIR/../../../.." && pwd)/MODA/ns-o-ran-gym"
fi
[[ -x "${GYM}/.venv/bin/python" ]] || {
    echo "[cal] ERRO: não achei o venv do ns-o-ran-gym a partir de $_SCRIPT_DIR" >&2
    exit 2
}
CAL="$OUT/cal_runs"
PYBIN="${GYM}/.venv/bin/python"
RUN_SCRIPT="${GYM}/examples/sqr_run.py"
CALIBRATE="${GYM}/examples/sqr_jc_calibrate.py"

CAL_SEEDS=(1001 1002 1003)
CAL_LAMBDAS=(2 5 6)
SIMTIME=300
TIMEOUT_S="${TIMEOUT_S:-1800}"
LOCK="${CAL}/.lock_runner"

mkdir -p "${CAL}"

run_one() {
  local lt="$1" seed="$2"
  local lam3
  lam3="$(awk -v l="${lt}" 'BEGIN{printf "%.3f", l/3}')"
  local tag="cal_lam${lt}_s${seed}"
  local dir="${CAL}/${tag}"
  if [[ -s "${dir}/.done" ]]; then
    echo "[cal] ${tag} ja completo, pulando"
    return 0
  fi
  local tent rc ok
  for tent in 1 2 3; do
    rm -rf "${dir}"
    mkdir -p "${dir}"
    echo "[cal] $(date +%H:%M:%S) iniciando ${tag} tent.${tent} (bootstrap-PROP IPC, lambda=${lam3}/slice, ${SIMTIME} s, timeout ${TIMEOUT_S} s)"
    set +e
    ( flock -x 9
      cd "${GYM}" && timeout -s KILL "${TIMEOUT_S}" \
        "${PYBIN}" "${RUN_SCRIPT}" \
        --controller prop --seed "${seed}" \
        --simTime "${SIMTIME}" --num_slices 3 --slas 3.5,3.5,3.5 \
        --periodMs 100 --c_prbs 52 --warmup_s 100 \
        --sliceLambdas="${lam3},${lam3},${lam3}" \
        --sliceApps=greedy,urllc,mmtc \
        --cellRadius=300 \
        --output "${dir}" ) 9>"${LOCK}" > "${dir}/run.log" 2>&1
    rc=$?
    set -e
    ok=0
    [[ -s "${dir}/step_metrics.csv" ]] && ok=$((ok+1))
    [[ -n "$(find "${dir}" -name sqr_kpm_history.csv -size +0 2>/dev/null)" ]] && ok=$((ok+1))
    [[ -n "$(find "${dir}" -name sqr_packet_delays.csv -size +0 2>/dev/null)" ]] && ok=$((ok+1))
    if [[ ${rc} -eq 0 && ${ok} -eq 3 ]]; then
      echo "${tag} rc=0 $(date +%H:%M:%S) tent.${tent}" > "${dir}/.done"
      echo "[cal] $(date +%H:%M:%S) OK ${tag} (tent.${tent})"
      return 0
    fi
    echo "[cal] $(date +%H:%M:%S) FALHOU ${tag} rc=${rc} (checks ${ok}/3) tent.${tent}"
    rm -f /dev/shm/sem_metrics_* /dev/shm/sem_control_* 2>/dev/null || true
  done
  echo "[cal] ${tag}: 3 tentativas falharam" >&2
  return 1
}

for lt in "${CAL_LAMBDAS[@]}"; do
  for seed in "${CAL_SEEDS[@]}"; do
    run_one "${lt}" "${seed}"
  done
done

# --- Calibração offline (§13/§16/§37) sobre os 9 runs ----------------------
echo "[cal] $(date +%H:%M:%S) calibrando (scalers §13, sigma_support §16, benchmark §37) via sqr_jc_calibrate.py"
"${PYBIN}" "${CALIBRATE}" \
  --cal-dir "${CAL}" \
  --seeds "${CAL_SEEDS[*]}" \
  --lambdas "${CAL_LAMBDAS[*]}" \
  --output "${OUT}"

echo "[cal] FIM $(date +%H:%M:%S) — artefatos em ${OUT}:"
ls -1 "${OUT}" | grep -E "feature_scaler_manifest.yaml|sigma_support_manifest.yaml|search_benchmark.json|calibration_report.json" || true
