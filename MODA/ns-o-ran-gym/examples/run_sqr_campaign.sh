#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_sqr_campaign.sh — campanha multi-seed da Fase 4 (espelho das Figs. 2–4
# do artigo SQR; PLANO §3-Fase 4).
#
# Roda examples/sqr_run.py para METODOS x SEEDS x SIMTIME, cada run num
# diretório próprio  <campaign>/<metodo>/seed_<seed>/  contendo
# step_metrics.csv, sqr_checkpoint.pkl, sqr_run.log, run_config.json e
# (após a verificação de conclusão) run_done.json.
#
# Uso:
#   run_sqr_campaign.sh [metodos] [seeds] [simTime]
#     metodos : lista separada por vírgulas, subconjunto de
#               {sqr-g0,sqr-g1,kbrl,equal}   (default: os 4)
#     seeds   : lista e/ou intervalos, ex. "1..10" ou "1,3,5" (default: 1..10)
#     simTime : segundos simulados por run (default: 600 — register R23)
#
# Variáveis de ambiente:
#   JOBS         runs ns-3 em paralelo (default 1 = sequencial). Seguro:
#                cada run tem sim_path/simId uuid próprio (SqrEnv.start_sim),
#                semáforos /sem_*_<uuid> exclusivos e SQLite por run.
#   CAMPAIGN_DIR default <gym>/results/sqr_campaign
#   C_PRBS       C INICIAL do agente (default 52, R01; o C efetivo 53 é
#                adotado da telemetria sqr_prb_windows.csv pelo próprio env)
#   TIMEOUT_S    timeout por run (default 7200; 600 s simulados ≈ 11 min de
#                parede na referência de 12/10/2026)
#   NS3_BIN      executável do cenário (default build/scratch/sqr/
#                ns3.46-sqr-sim-default; rebuild incremental via ns3_gate.sh
#                se fontes em scratch/sqr forem mais novas)
#
# Resume POR RUN: um run é pulado somente se COMPLETO — step_metrics.csv
# não-vazio + sqr_checkpoint.pkl não-vazio + último passo >= esperado para o
# simTime (fórmula do contrato §2) + ns3_exit_code 0 no log (o par de
# processos terminou limpo; o marcador run_done.json é escrito ao constatar).
# Run incompleto é refeito do zero no mesmo diretório (step_metrics.csv é
# truncado na 1ª escrita pelo próprio StepMetricsLogger).
#
# Ao final imprime "CAMPANHA OK" (todos os runs pedidos completos) ou
# "CAMPANHA FALHOU" com a lista dos runs que falharam. A análise/figuras é
# responsabilidade de run_sqr_analysis.sh (campaign_analysis.py).
# ---------------------------------------------------------------------------
set -euo pipefail

GYM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODA_ROOT="$(cd "${GYM_ROOT}/.." && pwd)"
NS3_ROOT="${MODA_ROOT}/ns-3-dev"
PYBIN="${GYM_ROOT}/.venv/bin/python"
RUN_SCRIPT="${GYM_ROOT}/examples/sqr_run.py"
NS3_GATE="${NS3_ROOT}/scratch/sqr/ns3_gate.sh"
NS3_BIN="${NS3_BIN:-${NS3_ROOT}/build/scratch/sqr/ns3.46-sqr-sim-default}"

METODOS="${1:-sqr-g0,sqr-g1,kbrl,equal}"
SEEDS_SPEC="${2:-1..10}"
SIMTIME="${3:-600}"
JOBS="${JOBS:-1}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-${GYM_ROOT}/results/sqr_campaign}"
C_PRBS="${C_PRBS:-52}"
TIMEOUT_S="${TIMEOUT_S:-7200}"
PERIOD_MS=100     # contrato §2 (congelado)
APP_START=0.5     # default do sqr_run.py/sqr-sim
WARMUP_S="${WARMUP_S:-100}"   # R14: aquecimento excluído das MÉTRICAS (a
                              # análise filtra pela coluna warmup com
                              # --exclude-warmup; o run loga a marcação)

fail() { echo "CAMPANHA FALHOU: $*" >&2; exit 1; }

[[ -x "${PYBIN}" ]] || fail "interpretador Python do projeto ausente: ${PYBIN}"
[[ -f "${RUN_SCRIPT}" ]] || fail "driver ausente: ${RUN_SCRIPT}"

# --- métodos válidos -> argumentos do sqr_run.py ---------------------------
method_args() {
    case "$1" in
        sqr-g0) echo "--controller sqr --gamma 0" ;;
        sqr-g1) echo "--controller sqr --gamma 1" ;;
        kbrl)   echo "--controller kbrl" ;;
        equal)  echo "--controller equal" ;;
        *) return 1 ;;
    esac
}

IFS=',' read -ra METHOD_LIST <<< "${METODOS}"
for m in "${METHOD_LIST[@]}"; do
    method_args "${m}" >/dev/null || fail "método desconhecido: '${m}' (válidos: sqr-g0, sqr-g1, kbrl, equal)"
done

# --- seeds: listas "1,3" e intervalos "1..10" ------------------------------
SEED_LIST=()
IFS=',' read -ra _parts <<< "${SEEDS_SPEC}"
for p in "${_parts[@]}"; do
    if [[ "${p}" == *..* ]]; then
        a="${p%%..*}"; b="${p##*..}"
        [[ "${a}" =~ ^[0-9]+$ && "${b}" =~ ^[0-9]+$ ]] || fail "intervalo de seeds inválido: '${p}'"
        (( a <= b )) || fail "intervalo de seeds vazio: '${p}'"
        for (( s=a; s<=b; s++ )); do SEED_LIST+=("${s}"); done
    else
        [[ "${p}" =~ ^[0-9]+$ ]] || fail "seed inválida: '${p}'"
        SEED_LIST+=("${p}")
    fi
done
((${#SEED_LIST[@]} > 0)) || fail "nenhuma seed em '${SEEDS_SPEC}'"

TOTAL_RUNS=$(( ${#METHOD_LIST[@]} * ${#SEED_LIST[@]} ))
echo "[campanha] ${TOTAL_RUNS} runs: métodos=[${METODOS}] seeds=[${SEEDS_SPEC}] simTime=${SIMTIME}s JOBS=${JOBS}"
echo "[campanha] diretório: ${CAMPAIGN_DIR}"
mkdir -p "${CAMPAIGN_DIR}"

# --- binário do cenário em dia? (uma única vez, antes de disparar os runs) -
if [[ ! -x "${NS3_BIN}" ]] \
   || [[ -n "$(find "${NS3_ROOT}/scratch/sqr" -maxdepth 1 \
                 \( -name '*.cc' -o -name '*.h' \) \
                 -newer "${NS3_BIN}" -print -quit 2>/dev/null)" ]]; then
    echo "[campanha] binário do sqr-sim ausente ou mais velho que as fontes — build"
    bash "${NS3_GATE}" build > "${CAMPAIGN_DIR}/ns3_build.log" 2>&1 \
        || { tail -30 "${CAMPAIGN_DIR}/ns3_build.log" >&2; fail "build do ns-3 falhou"; }
fi
[[ -x "${NS3_BIN}" ]] || fail "executável do cenário não encontrado: ${NS3_BIN}"
echo "[campanha] OK: binário do cenário em dia: ${NS3_BIN}"

# Passos esperados por episode (contrato §2: loop trunca em max_steps)
EXPECTED_STEPS=$("${PYBIN}" -c "print(int(round((${SIMTIME} - ${APP_START}) * 1000 / ${PERIOD_MS})))")

# --- run_complete <run_dir>: critério de resume ----------------------------
run_complete() {
    local out="$1"
    [[ -s "${out}/step_metrics.csv" ]] || return 1
    [[ -s "${out}/sqr_checkpoint.pkl" ]] || return 1
    # último passo >= esperado (sem buracos: colunas episode,step são 1..N)
    local last
    last=$(awk -F, 'NR>1 { if ($2+0 > m) m = $2+0 } END { print m+0 }' "${out}/step_metrics.csv")
    (( last >= EXPECTED_STEPS )) || return 1
    grep -q "^\[sqr_run\] ns3_exit_code: 0" "${out}/sqr_run.log" 2>/dev/null || return 1
    return 0
}

# --- run_one <metodo> <seed>: um run completo com verificação --------------
run_one() {
    local method="$1" seed="$2"
    local cargs out log
    cargs="$(method_args "${method}")" || return 9
    out="${CAMPAIGN_DIR}/${method}/seed_${seed}"
    log="${out}/sqr_run.log"

    if [[ -f "${out}/run_done.json" ]] && run_complete "${out}"; then
        echo "[campanha] ${method}/seed_${seed}: já completo — pulando (resume)"
        return 0
    fi
    if [[ -d "${out}" ]]; then
        echo "[campanha] ${method}/seed_${seed}: incompleto — refazendo do zero"
        rm -rf "${out}"
    fi
    mkdir -p "${out}"

    # run_config.json: registro do run (método registrado no run, ask item 4)
    cat > "${out}/run_config.json" <<JSON
{
  "method": "${method}",
  "controller_args": "${cargs}",
  "seed": ${seed},
  "simTime": ${SIMTIME},
  "periodMs": ${PERIOD_MS},
  "c_prbs_initial": ${C_PRBS},
  "warmup_s": ${WARMUP_S},
  "expected_steps": ${EXPECTED_STEPS},
  "started_at": "$(date -Is)",
  "driver": "run_sqr_campaign.sh"
}
JSON

    echo "[campanha] ${method}/seed_${seed}: iniciando (timeout ${TIMEOUT_S}s)"
    set +e
    timeout --signal=TERM --kill-after=120 "${TIMEOUT_S}" \
        "${PYBIN}" "${RUN_SCRIPT}" \
        ${cargs} \
        --simTime "${SIMTIME}" --seed "${seed}" \
        --periodMs "${PERIOD_MS}" --c_prbs "${C_PRBS}" \
        --warmup_s "${WARMUP_S}" \
        --output "${out}" > "${log}" 2>&1
    local rc=$?
    set -e

    if [[ ${rc} -ne 0 ]]; then
        echo "[campanha] ${method}/seed_${seed}: FALHOU (exit ${rc}; 124=timeout)" >&2
        tail -20 "${log}" >&2 || true
        { echo "{\"method\": \"${method}\", \"seed\": ${seed}, \"exit_code\": ${rc}, \"failed_at\": \"$(date -Is)\"}"; } \
            > "${out}/run_failed.json"
        return 1
    fi

    if ! run_complete "${out}"; then
        echo "[campanha] ${method}/seed_${seed}: exit 0 mas critérios de conclusão não atendidos" >&2
        { echo "{\"method\": \"${method}\", \"seed\": ${seed}, \"exit_code\": 0, \"reason\": \"incomplete artifacts\", \"failed_at\": \"$(date -Is)\"}"; } \
            > "${out}/run_failed.json"
        return 1
    fi

    cat > "${out}/run_done.json" <<JSON
{
  "method": "${method}",
  "controller_args": "${cargs}",
  "seed": ${seed},
  "simTime": ${SIMTIME},
  "expected_steps": ${EXPECTED_STEPS},
  "ns3_exit_code": 0,
  "done_at": "$(date -Is)"
}
JSON
    echo "[campanha] ${method}/seed_${seed}: OK ($(awk -F, 'NR>1{m=$2+0} END{print m+0}' "${out}/step_metrics.csv") passos)"
    return 0
}

# --- disparo (sequencial ou com JOBS slots) --------------------------------
FAILED=()
RUN_IDX=0
for m in "${METHOD_LIST[@]}"; do
    for s in "${SEED_LIST[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))
        if [[ "${JOBS}" -le 1 ]]; then
            run_one "${m}" "${s}" || FAILED+=("${m}/seed_${s}")
        else
            run_one "${m}" "${s}" &
            # limita slots: espera qualquer job liberar antes de disparar o próximo
            while (( $(jobs -rp | wc -l) >= JOBS )); do
                wait -n || true
            done
        fi
    done
done
if [[ "${JOBS}" -gt 1 ]]; then
    wait
fi

# --- veredito: re-checa TODOS os runs pedidos ------------------------------
MISSING=()
for m in "${METHOD_LIST[@]}"; do
    for s in "${SEED_LIST[@]}"; do
        out="${CAMPAIGN_DIR}/${m}/seed_${s}"
        if ! run_complete "${out}"; then
            MISSING+=("${m}/seed_${s}")
        fi
    done
done

if (( ${#MISSING[@]} > 0 )); then
    fail "runs incompletos/falhados: ${MISSING[*]} (logs em <run>/sqr_run.log)"
fi

echo ""
echo "CAMPANHA OK: ${TOTAL_RUNS} runs completos em ${CAMPAIGN_DIR}"
echo "Próximo passo: bash ${GYM_ROOT}/examples/run_sqr_analysis.sh"
