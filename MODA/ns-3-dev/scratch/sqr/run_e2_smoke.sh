#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_e2_smoke.sh — GATE de fumaça do caminho de controle E2 (Fase 5.3).
#
# Sequência DEFAULT (sem argumentos):
#   1. build check do ns-3 (via ns3_gate.sh — Python 3.13, nunca ./ns3);
#   2. fase e2 do smoke oficial (--controlPath=e2: RIC Control do contrib/nori
#      + E2SM-KPM indications com loopback em-processo; critérios R27 i–iv
#      idênticos aos do IPC — ver smoke_test.sh);
#   3. imprime "E2 SMOKE OK".
#
# Uso:
#   bash scratch/sqr/run_e2_smoke.sh [simTime] [seed]
#     simTime default 30 s (3 fases × 100 janelas de 100 ms, como o IPC);
#     seed default 1.
#
# Saída: exit 0 + "E2 SMOKE OK" em sucesso; exit 1 + "SMOKE FAILED: ..." na
# primeira falha (o build e a fase e2 são fatais).
# ---------------------------------------------------------------------------
set -u

SQR_DIR="/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr"
NS3_GATE="${SQR_DIR}/ns3_gate.sh"
SMOKE="${SQR_DIR}/smoke_test.sh"

SIMTIME="${1:-30}"
SEED="${2:-1}"

fail() {
    echo "SMOKE FAILED: $1" >&2
    exit 1
}

[[ -x "${NS3_GATE}" ]] || fail "ns3_gate.sh não encontrado em ${NS3_GATE}"
[[ -f "${SMOKE}" ]] || fail "smoke_test.sh não encontrado em ${SMOKE}"

echo "[e2-smoke] passo 1/2: build check"
bash "${NS3_GATE}" build || fail "build do ns-3 falhou (ns3_gate.sh build)"

echo "[e2-smoke] passo 2/2: fase e2 do smoke oficial (simTime=${SIMTIME}s seed=${SEED})"
bash "${SMOKE}" e2 "${SIMTIME}" "${SEED}" || fail "fase e2 do smoke falhou"

echo "E2 SMOKE OK"
exit 0
