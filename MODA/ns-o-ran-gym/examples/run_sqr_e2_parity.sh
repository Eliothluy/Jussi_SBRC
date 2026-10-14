#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_sqr_e2_parity.sh — paridade IPC x E2 do SQR xApp (Fase 5.4, PLANO
# §3-Fase 5.4).
#
# Roda o MESMO seed/simTime/gamma pelos DOIS caminhos de controle e compara:
#   - sequência de decisões/cotas (step_metrics.csv dos dois runs);
#   - telemetria prb_* aplicada no MAC (sqr_prb_windows.csv escrito pelo
#     sqr-sim nos dois caminhos — verdade de chão);
#   - KPMs/violações por UE (sqr_kpm_history.csv dos dois runs);
#   - latência de decisão p50/p95 por caminho vs orçamento de 50 ms
#     (contrato §5 — o mesmo budget do sem_timedwait do IPC).
#
# Caminhos:
#   IPC: examples/sqr_run.py   (semáforos POSIX + rslaq_actions_for_ns3.csv)
#   E2 : examples/xapp_sqr.py  (xApp: E2SM-KPM indication → SQR+Π → RIC
#        Control via ponte UDP local do modo DEGRADED; em FULL o transporte
#        vira e2sim/SCTP sem mudar a lógica do xApp — docs no cabeçalho)
#
# Uso:
#   bash examples/run_sqr_e2_parity.sh [simTime] [seed] [gamma]
#     defaults: 120 s, seed 1, γ=1 (pedido da Fase 5.4)
#   PARITY_DIR=<dir>  base de saída (default <gym>/results/sqr_e2_parity)
#   SKIP_BUILD=1      pula o build check do ns-3
#
# Saída: tabela comparativa + "PARITY OK" (decisões/cotas/KPMs coincidem)
# ou "PARITY DIVERGED" (com as janelas divergentes); relatório em
# <PARITY_DIR>/parity_report.md com a diferença de latência documentada.
# ---------------------------------------------------------------------------
set -euo pipefail

GYM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODA_ROOT="$(cd "${GYM_ROOT}/.." && pwd)"
NS3_ROOT="${MODA_ROOT}/ns-3-dev"
PYBIN="${GYM_ROOT}/.venv/bin/python"
NS3_GATE="${NS3_ROOT}/scratch/sqr/ns3_gate.sh"

SIMTIME="${1:-120}"
SEED="${2:-1}"
GAMMA="${3:-1}"
PERIOD_MS=100
APP_START=0.5
BUDGET_MS=50     # contrato §5 (sem_timedwait do IPC = budget da ponte E2)
XAPP_PORT="${XAPP_PORT:-5560}"

PARITY_DIR="${PARITY_DIR:-${GYM_ROOT}/results/sqr_e2_parity/run_$(date +%Y%m%d_%H%M%S)_sim${SIMTIME}s_seed${SEED}_g${GAMMA}}"
IPC_DIR="${PARITY_DIR}/ipc"
E2_DIR="${PARITY_DIR}/e2"

fail() { echo "PARITY FAILED: $*" >&2; exit 2; }

[[ -x "${PYBIN}" ]] || fail "interpretador Python do projeto ausente: ${PYBIN}"
mkdir -p "${PARITY_DIR}"

# --- 0. binário do cenário em dia? (fontes scratch/sqr mais novas => build)
NS3_BIN="${NS3_ROOT}/build/scratch/sqr/ns3.46-sqr-sim-default"
if [[ "${SKIP_BUILD:-0}" != "1" ]] && { [[ ! -x "${NS3_BIN}" ]] || \
     [[ -n "$(find "${NS3_ROOT}/scratch/sqr" -maxdepth 1 \
                   \( -name '*.cc' -o -name '*.h' \) \
                   -newer "${NS3_BIN}" -print -quit 2>/dev/null)" ]]; }; then
    echo "[parity] binário do sqr-sim ausente/obsoleto — build (ns3_gate.sh)"
    bash "${NS3_GATE}" build > "${PARITY_DIR}/ns3_build.log" 2>&1 \
        || { tail -30 "${PARITY_DIR}/ns3_build.log" >&2; fail "build do ns-3 falhou"; }
fi
[[ -x "${NS3_BIN}" ]] || fail "executável do cenário não encontrado: ${NS3_BIN}"

EXPECTED_STEPS=$("${PYBIN}" -c "print(int(round((${SIMTIME} - ${APP_START}) * 1000 / ${PERIOD_MS})))")
echo "[parity] seed=${SEED} simTime=${SIMTIME}s gamma=${GAMMA} steps esperados=${EXPECTED_STEPS}"
echo "[parity] diretório: ${PARITY_DIR}"

# --- 1. caminho IPC (referência): sqr_run.py com o agente SQR γ definido ---
echo "[parity] passo 1/3: run IPC (sqr_run.py)"
"${PYBIN}" "${GYM_ROOT}/examples/sqr_run.py" \
    --controller sqr --gamma "${GAMMA}" \
    --seed "${SEED}" --simTime "${SIMTIME}" --periodMs "${PERIOD_MS}" \
    --output "${IPC_DIR}" \
    > "${PARITY_DIR}/ipc_run.log" 2>&1 \
    || { tail -30 "${PARITY_DIR}/ipc_run.log" >&2; fail "run IPC falhou"; }
grep -E "ns3_exit_code|episode 1:" "${PARITY_DIR}/ipc_run.log" || true

# --- 2. caminho E2 (xApp): xapp_sqr.py sobre a ponte UDP do modo DEGRADED --
echo "[parity] passo 2/3: run E2 (xapp_sqr.py, ponte UDP local, budget ${BUDGET_MS} ms)"
"${PYBIN}" "${GYM_ROOT}/examples/xapp_sqr.py" \
    --controller sqr --gamma "${GAMMA}" \
    --seed "${SEED}" --simTime "${SIMTIME}" --periodMs "${PERIOD_MS}" \
    --output "${E2_DIR}" --xappPort "${XAPP_PORT}" \
    --e2XappTimeoutMs "${BUDGET_MS}" \
    > "${PARITY_DIR}/e2_run.log" 2>&1 \
    || { tail -30 "${PARITY_DIR}/e2_run.log" >&2; fail "run E2 (xApp) falhou"; }
grep -E "summary:" "${PARITY_DIR}/e2_run.log" || true

# --- 3. comparação + relatório ---------------------------------------------
echo "[parity] passo 3/3: comparação (decisões, prb_*, violações, latência)"
set +e
"${PYBIN}" "${GYM_ROOT}/examples/sqr_parity_report.py" \
    --ipc "${IPC_DIR}" --e2 "${E2_DIR}" \
    --budget-ms "${BUDGET_MS}" \
    --out "${PARITY_DIR}/parity_report.md"
PARITY_RC=$?
set -e

# Sanity: passos completos nos dois runs (contrato §2: loop trunca em max_steps)
IPC_STEPS=$("${PYBIN}" -c "
import csv,sys
rows=list(csv.DictReader(open('${IPC_DIR}/step_metrics.csv')))
print(len(set(int(r['step']) for r in rows)))")
E2_STEPS=$("${PYBIN}" -c "
import csv,sys
rows=list(csv.DictReader(open('${E2_DIR}/step_metrics.csv')))
print(len(set(int(r['step']) for r in rows)))")
echo "[parity] passos completos: IPC=${IPC_STEPS} E2=${E2_STEPS} (esperado ${EXPECTED_STEPS})"
if [[ "${IPC_STEPS}" != "${EXPECTED_STEPS}" || "${E2_STEPS}" != "${EXPECTED_STEPS}" ]]; then
    echo "PARITY FAILED: runs incompletos (IPC=${IPC_STEPS}, E2=${E2_STEPS}, esperado ${EXPECTED_STEPS})" >&2
    exit 2
fi

if [[ "${PARITY_RC}" -eq 0 ]]; then
    echo "PARITY OK"
else
    echo "PARITY DIVERGED (detalhes em ${PARITY_DIR}/parity_report.md)"
fi
exit "${PARITY_RC}"
