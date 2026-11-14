#!/usr/bin/env bash
# run_radio_sanity.sh — GATE G3 (protocolo Etapa 2.5 §24): rádio saudável.
#
# Low-load sanity ANTES da aprendizagem: lambda_total = 2, 3 seeds, 300 s,
# controlador prop (bootstrap-PROP do §29 — as MESMAS flags congeladas do
# braço prop da Rodada 2: --slas 3.5,3.5,3.5 --sliceApps=greedy,urllc,mmtc
# --cellRadius=300, λ/3 por slice, IPC, 100 ms).
#
# Lições congeladas incorporadas (Rodada 2):
#   - LC_ALL=C em TODO o script (R56: locale quebrou o executor de replay);
#   - .done idempotente + 3 tentativas + limpeza de semáforos órfãos;
#   - timeout 1800 s/run com KILL (300 s de sim = 480–650 s de parela);
#   - lock flock (contenda de RAM/OOM de 21/10 com runs simultâneas).
#
# Uso (orquestrador; este agente NÃO executa ns-3):
#   LC_ALL=C bash -c 'results/sqr_jc_final/02_radio_sanity/run_radio_sanity.sh'
# Saída: radio_sanity/<tag>/{run,...} + radio_sanity_report.json +
#        radio_sanity_verdict.txt (PASS/STOP §24).
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
    echo "[sanity] ERRO: não achei o venv do ns-o-ran-gym a partir de $_SCRIPT_DIR" >&2
    exit 2
}
SAN="$OUT/radio_sanity"
PYBIN="${GYM}/.venv/bin/python"
RUN_SCRIPT="${GYM}/examples/sqr_run.py"
SANALYZE="${GYM}/examples/sqr_jc_radio_sanity.py"

LAMBDA_TOTAL=2
SEEDS=(1 2 3)
SIMTIME=300
TIMEOUT_S="${TIMEOUT_S:-1800}"
LOCK="${SAN}/.lock_runner"

mkdir -p "${SAN}"

run_one() {
  local seed="$1"
  local lam3
  lam3="$(awk -v l="${LAMBDA_TOTAL}" 'BEGIN{printf "%.3f", l/3}')"
  local tag="sanity_lam${LAMBDA_TOTAL}_s${seed}"
  local dir="${SAN}/${tag}"
  if [[ -s "${dir}/.done" ]]; then
    echo "[sanity] ${tag} ja completo, pulando"
    return 0
  fi
  local tent rc ok
  for tent in 1 2 3; do
    rm -rf "${dir}"
    mkdir -p "${dir}"
    echo "[sanity] $(date +%H:%M:%S) iniciando ${tag} tent.${tent} (prop IPC, lambda=${lam3}/slice, ${SIMTIME} s, timeout ${TIMEOUT_S} s)"
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
      echo "[sanity] $(date +%H:%M:%S) OK ${tag} (tent.${tent})"
      return 0
    fi
    echo "[sanity] $(date +%H:%M:%S) FALHOU ${tag} rc=${rc} (checks ${ok}/3) tent.${tent}"
    rm -f /dev/shm/sem_metrics_* /dev/shm/sem_control_* 2>/dev/null || true
  done
  echo "[sanity] ${tag}: 3 tentativas falharam" >&2
  return 1
}

for seed in "${SEEDS[@]}"; do
  run_one "${seed}"
done

# --- análise §24: RLF/SNR/CQI/PHY → veredicto PASS/STOP -------------------
echo "[sanity] $(date +%H:%M:%S) analizando (RLF/SNR/CQI/PHY) via sqr_jc_radio_sanity.py"
"${PYBIN}" "${SANALYZE}" \
  --sanity-dir "${SAN}" \
  --lambda-total "${LAMBDA_TOTAL}" \
  --seeds "${SEEDS[*]}" \
  --report "${SAN}/radio_sanity_report.json" \
  --verdict "${SAN}/radio_sanity_verdict.txt"

echo "[sanity] veredicto:"
cat "${SAN}/radio_sanity_verdict.txt"
n_ok="$(ls "${SAN}"/sanity_lam*/.done 2>/dev/null | wc -l)"
echo "[sanity] FIM $(date +%H:%M:%S) — ${n_ok}/${#SEEDS[@]} runs completos"
