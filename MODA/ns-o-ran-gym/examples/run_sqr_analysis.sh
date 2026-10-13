#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_sqr_analysis.sh — wrapper da análise da campanha Fase 4 (espelho das
# Figs. 2–4). Agrega os step_metrics.csv de todos os runs de um diretório de
# campanha (produzido por run_sqr_campaign.sh), com IC 95% por bootstrap na
# unidade seed, e exporta figuras + tabelas CSV.
#
# Uso:
#   run_sqr_analysis.sh [campaign_dir] [args do campaign_analysis.py...]
#     campaign_dir : default <gym>/results/sqr_campaign
#
# Exemplos:
#   # campanha completa (exclui o aquecimento R14 dos 600 s runs):
#   run_sqr_analysis.sh results/sqr_campaign --exclude-warmup
#
#   # piloto (figuras com sufixo _pilot):
#   run_sqr_analysis.sh results/sqr_pilot --suffix _pilot
#
# Saídas (defaults, relativos ao campaign_dir):
#   figures/fig2_window_violation_rate[.png]  Fig. 2 (meta δ=1%)
#   figures/fig3_cumulative_violation[.png]   Fig. 3 (violação acumulada)
#   figures/fig4_prb_distribution[.png]       Fig. 4 (RBs requested/executed/
#                                             scheduled/idle)
#   figures/fig_summary_violation_bars[.png]  barras com IC bootstrap
#   figures/fig_modes[.png]                   diversidade de modos
#   tables/summary_by_method.csv              médias + IC por métrica/método
#   tables/violation_by_slice.csv             R26: por slice
#   tables/modes_by_method.csv                modos KBRL (scan/cold) e SQR
#   tables/per_seed_metrics.csv               unidade seed (transparência)
#   tables/series_<metodo>.csv                séries temporais (média seeds)
# ---------------------------------------------------------------------------
set -euo pipefail

GYM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="${GYM_ROOT}/.venv/bin/python"
ANALYSIS="${GYM_ROOT}/src/nsoran/sqr/campaign_analysis.py"
DEFAULT_DIR="${GYM_ROOT}/results/sqr_campaign"

[[ -x "${PYBIN}" ]] || { echo "interpretador ausente: ${PYBIN}" >&2; exit 1; }
[[ -f "${ANALYSIS}" ]] || { echo "análise ausente: ${ANALYSIS}" >&2; exit 1; }

CAMPAIGN_DIR="${1:-${DEFAULT_DIR}}"
if [[ $# -gt 0 ]]; then shift; fi

exec "${PYBIN}" "${ANALYSIS}" --campaign-dir "${CAMPAIGN_DIR}" "$@"
