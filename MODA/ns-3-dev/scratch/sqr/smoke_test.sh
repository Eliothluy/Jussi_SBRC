#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Smoke test do cenário SQR (Fase 1) — sqr-sim em modo network-only
# (sem agente Python; padrão baseline do run_all_scenarios.sh).
#
# Valida (gate Fase 1):
#   1. exit 0 do ns-3;
#   2. arquivo de KPMs (rslaq-kpms.txt, contrato CWD do ns-3-dev) gerado com as
#      colunas novas do contrato SQR no header (windowThroughputMbps, snrDb,
#      queueBytes);
#   3. evidência de >= 1 chegada (UE_ARRIVAL) e >= 1 partida (UE_DEPARTURE)
#      de UE no log;
#   4. >= 1 attach real no rádio (UE_ATTACHED) — o RNTI pós-RACH foi registrado
#      no mapeamento RNTI→slice do scheduler;
#   5. >= 1 janela PRB_WINDOW com prb_scheduled > 0 (o MAC efetivamente
#      agendou RBGs);
#   6. >= 1 linha de KPM (histórico sqr_kpm_history.csv) com
#      windowThroughputMbps > 0 — tráfego entregue ponta a ponta.
#
# Uso: bash scratch/sqr/smoke_test.sh [simTime=120] [seed=1]
# Sempre invoca o ns-3 via ns3_gate.sh (Python 3.13; o wrapper ./ns3 quebra
# com o Python 3.14 do sistema).
# ---------------------------------------------------------------------------
set -u

NS3_ROOT="/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev"
SQR_DIR="${NS3_ROOT}/scratch/sqr"
NS3_GATE="${SQR_DIR}/ns3_gate.sh"
OUT_DIR="${SQR_DIR}/smoke_out"
RUN_DIR="${OUT_DIR}/run_$(date +%Y%m%d_%H%M%S)_s${2:-1}"

SIMTIME="${1:-120}"
SEED="${2:-1}"

mkdir -p "${RUN_DIR}"
LOG="${RUN_DIR}/sqr-sim.log"
KPM_FILE="${NS3_ROOT}/rslaq-kpms.txt"

fail() {
    echo "SMOKE FAILED: $1" >&2
    echo "Log: ${LOG}" >&2
    exit 1
}

[[ -x "${NS3_GATE}" ]] || fail "ns3_gate.sh não encontrado em ${NS3_GATE}"

echo "[smoke] Removendo KPMs de runs anteriores: ${KPM_FILE}"
rm -f "${KPM_FILE}"

echo "[smoke] Rodando sqr-sim network-only: simTime=${SIMTIME}s seed=${SEED} (ns3_gate.sh)"
cd "${NS3_ROOT}" || fail "não foi possível entrar em ${NS3_ROOT}"
bash "${NS3_GATE}" run sqr-sim -- \
    --simTime="${SIMTIME}" \
    --seed="${SEED}" \
    --run=1 \
    --periodMs=100 \
    --outputDir="${RUN_DIR}" > "${LOG}" 2>&1
RC=$?

echo "[smoke] Exit code: ${RC}"
if [[ ${RC} -ne 0 ]]; then
    tail -40 "${LOG}" >&2
    fail "ns-3 retornou código ${RC} (esperado 0)"
fi

# --- Check 1: KPM gerado com colunas novas no header ---
if [[ ! -s "${KPM_FILE}" ]]; then
    fail "arquivo de KPMs ausente/vazio: ${KPM_FILE}"
fi
HEADER="$(head -n1 "${KPM_FILE}")"
for col in windowThroughputMbps snrDb queueBytes; do
    if ! grep -q "${col}" <<< "${HEADER}"; then
        fail "header do KPM não contém a coluna '${col}' (header: ${HEADER})"
    fi
done
KPM_ROWS=$(( $(wc -l < "${KPM_FILE}") - 1 ))
echo "[smoke] OK: KPM gerado, colunas novas no header, ${KPM_ROWS} linhas de dados"

# --- Check 2: >= 1 chegada e >= 1 partida de UE no log ---
ARRIVALS=$(grep -c "UE_ARRIVAL" "${LOG}" || true)
DEPARTURES=$(grep -c "UE_DEPARTURE" "${LOG}" || true)
if [[ ${ARRIVALS} -lt 1 ]]; then
    fail "nenhuma chegada de UE (UE_ARRIVAL) no log"
fi
if [[ ${DEPARTURES} -lt 1 ]]; then
    fail "nenhuma partida de UE (UE_DEPARTURE) no log"
fi
echo "[smoke] OK: ${ARRIVALS} chegada(s) e ${DEPARTURES} partida(s) de UE"

# --- Check 3 (gate Fase 1): >= 1 attach real no rádio (UE_ATTACHED) ---
ATTACHED=$(grep -c "UE_ATTACHED" "${LOG}" || true)
if [[ ${ATTACHED} -lt 1 ]]; then
    fail "nenhum attach no rádio (UE_ATTACHED) — RACH/RRC não completou ou o RNTI não foi registrado no mapeamento slice"
fi
echo "[smoke] OK: ${ATTACHED} attach(s) de UE (UE_ATTACHED)"

# --- Check 4 (gate Fase 1): >= 1 janela PRB_WINDOW com prb_scheduled > 0 ---
PRB_WINDOWS=$(grep -c "PRB_WINDOW" "${LOG}" || true)
PRB_NONZERO=$(grep "PRB_WINDOW" "${LOG}" | grep -vc "prb_scheduled_rbg=\[0, 0, 0\]" || true)
if [[ ${PRB_NONZERO} -lt 1 ]]; then
    fail "nenhuma janela com prb_scheduled > 0 (${PRB_WINDOWS} janelas, todas zeradas — scheduler sem UEs mapeados ou sem demanda)"
fi
echo "[smoke] OK: ${PRB_NONZERO}/${PRB_WINDOWS} janelas com prb_scheduled > 0"

# --- Check 5 (gate Fase 1): >= 1 linha de KPM com throughput entregue > 0 ---
KPM_HIST="${RUN_DIR}/results_sqr_network_only/seed=${SEED}_run=1/sqr_kpm_history.csv"
if [[ ! -s "${KPM_HIST}" ]]; then
    fail "histórico de KPMs ausente/vazio: ${KPM_HIST}"
fi
KPM_HIST_ROWS=$(( $(wc -l < "${KPM_HIST}") - 1 ))
# windowThroughputMbps é a 11ª coluna do CSV
KPM_THR_POS=$(awk -F, 'NR>1 && $11+0 > 0' "${KPM_HIST}" | wc -l)
if [[ ${KPM_THR_POS} -lt 1 ]]; then
    fail "nenhuma linha de KPM com windowThroughputMbps > 0 em ${KPM_HIST_ROWS} janelas — nenhum byte entregue ponta a ponta"
fi
echo "[smoke] OK: ${KPM_THR_POS}/${KPM_HIST_ROWS} linhas de KPM com throughput > 0"

# --- Evidências para inspeção ---
cp "${KPM_FILE}" "${RUN_DIR}/rslaq-kpms.txt"
[[ -f "${RUN_DIR}/results_sqr_network_only/seed=${SEED}_run=1/sqr_prb_windows.csv" ]] && \
    echo "[smoke] prb CSV: ${RUN_DIR}/results_sqr_network_only/seed=${SEED}_run=1/sqr_prb_windows.csv"
echo "[smoke] kpm histórico: ${KPM_HIST}"
echo "[smoke] Log completo: ${LOG}"
echo "[smoke] KPM copiado:  ${RUN_DIR}/rslaq-kpms.txt"

echo "SMOKE OK"
exit 0
