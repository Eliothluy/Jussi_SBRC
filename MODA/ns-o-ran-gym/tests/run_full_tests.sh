#!/usr/bin/env bash
# Suíte completa de testes Python do ns-o-ran-gym (tests/).
#
# Interpretador: mesmo critério do run_sqr_tests.sh — venv CPython 3.13 em
# MODA/ns-o-ran-gym/.venv (o python3 do sistema 3.14 não tem as deps).
# Override: SQR_PYTHON=/caminho/para/python.
#
# BASELINE CONHECIDA (2026-10-11, anterior ao agente SQR — NÃO é regressão):
#   - 3 erros de COLEÇÃO: tests/test_check_env.py, tests/test_rslaq_predictive.py
#     e tests/test_time_ts.py importam 'nsoran.environments.*' (layout upstream
#     que não existe neste fork — os pacotes locais são 'nsoran' e 'environments').
#   - 2 FALHAS: tests/test_all_scenarios_runner_contract.py::
#       test_runner_defines_group1_and_group2_for_fair_comparison
#       test_runner_passes_multi_seed_to_metaheuristic_search
#     (deriva de contrato em examples/run_all_scenarios.sh).
#   Qualquer desvio além desse conjunto deve ser tratado como regressão.
#
# Uso:
#   bash tests/run_full_tests.sh           # suíte completa
#   bash tests/run_full_tests.sh -q        # args extras vão para o pytest

set -Eeuo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${GYM_ROOT}"

if [[ -n "${SQR_PYTHON:-}" ]]; then
    PYTHON="${SQR_PYTHON}"
elif [[ -x "${GYM_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${GYM_ROOT}/.venv/bin/python"
else
    PYTHON="$(command -v python3)"
fi

echo "[run_full_tests] interpretador: ${PYTHON}"
"${PYTHON}" -c "import numpy, pytest" 2>/dev/null || {
    echo "[run_full_tests] ERRO: ${PYTHON} não tem numpy/pytest; veja run_sqr_tests.sh." >&2
    exit 2
}

# A suíte COMPLETA. --continue-on-collection-errors é necessário porque os 3
# arquivos do baseline quebrado interromperiam a coleta e impediriam a
# execução dos demais testes; com a flag, TODA a suíte roda e os erros de
# coleção aparecem no resumo final (o gate continua saindo com erro).
status=0
"${PYTHON}" -m pytest tests/ --continue-on-collection-errors "$@" || status=$?

if [[ "${status}" -ne 0 ]]; then
    echo ""
    echo "[run_full_tests] Suíte terminou com status ${status}."
    echo "[run_full_tests] Confira se as falhas/erros estão dentro do baseline"
    echo "[run_full_tests] conhecido documentado no cabeçalho deste script"
    echo "[run_full_tests] (3 coleções quebradas + 2 falhas de contrato, pré-existentes)."
fi
exit "${status}"
