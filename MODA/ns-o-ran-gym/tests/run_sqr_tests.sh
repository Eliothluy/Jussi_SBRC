#!/usr/bin/env bash
# Gate Fase 2 (go/no-go) — testes sintéticos do agente SQR, SEM ns-3.
#
# AGENTS.md §5 regra 5 / §7.2: o SQR só pode avançar para qualquer run ns-3
# depois que estes testes passarem. Este gate roda APENAS os test_sqr_*.
#
# Interpretador (fixado em 2026-10-11): o python3 do sistema (3.14.4) NÃO tem
# numpy/pytest; o interpretador que roda a suíte é o venv CPython 3.13.12 em
# MODA/ns-o-ran-gym/.venv (criado via uv; pyproject exige >=3.12 e os
# classifiers são 3.12/3.13). Override: SQR_PYTHON=/caminho/para/python.
#
# Uso:
#   bash tests/run_sqr_tests.sh            # a partir da raiz do repo ns-o-ran-gym
#   bash tests/run_sqr_tests.sh -v         # args extras vão para o pytest

set -Eeuo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${GYM_ROOT}"

# ---------------------------------------------------------------------
# Interpretador: SQR_PYTHON > .venv > python3 do PATH (com checagem numpy)
# ---------------------------------------------------------------------
if [[ -n "${SQR_PYTHON:-}" ]]; then
    PYTHON="${SQR_PYTHON}"
elif [[ -x "${GYM_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${GYM_ROOT}/.venv/bin/python"
else
    PYTHON="$(command -v python3)"
fi

echo "[run_sqr_tests] interpretador: ${PYTHON}"
"${PYTHON}" -c "import numpy, pytest" 2>/dev/null || {
    echo "[run_sqr_tests] ERRO: ${PYTHON} não tem numpy/pytest." >&2
    echo "  Recrie o venv fixado:" >&2
    echo "    cd ${GYM_ROOT} && uv venv --python python3.13 .venv && \\" >&2
    echo "    uv pip install --python .venv/bin/python 'gymnasium>=0.29' 'posix-ipc>=1.1.1' 'pandas>=2.2.3' 'numpy>=2.1.1' 'sem>=0.3.9.1' scipy pytest" >&2
    exit 2
}

# ---------------------------------------------------------------------
# Gate: somente os testes sintéticos do SQR (+ baseline KBRL, Fase 3)
# ---------------------------------------------------------------------
exec "${PYTHON}" -m pytest \
    tests/test_sqr_kernel.py \
    tests/test_sqr_quantile.py \
    tests/test_sqr_controller.py \
    tests/test_sqr_coordinator.py \
    tests/test_sqr_kpms.py \
    tests/test_sqr_env.py \
    tests/test_sqr_checkpoint.py \
    tests/test_sqr_run.py \
    tests/test_sqr_kbrl.py \
    "$@"
