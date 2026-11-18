#!/usr/bin/env bash
# Suite de regressão e integração da Etapa 2 (`sqr-v2`): IPC Handshake v2 + BUG-05 + BUG-04 + A/A
set -Eeuo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${GYM_ROOT}"

PYTHON="${SQR_PYTHON:-${GYM_ROOT}/.venv/bin/python}"
[[ -x "${PYTHON}" ]] || { echo "[run_sqr_v2_tests] ERRO: ${PYTHON} ausente" >&2; exit 2; }

echo "[run_sqr_v2_tests] interpretador: ${PYTHON}"
exec "${PYTHON}" -m pytest tests/test_sqr_v2_ipc_and_metrics.py "$@"
