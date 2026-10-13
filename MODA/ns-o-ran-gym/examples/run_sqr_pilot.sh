#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_sqr_pilot.sh — PILOTO rápido da campanha Fase 4 (gate local).
#
# Roda 3 MÉTODOS (sqr-g0, sqr-g1, kbrl) x SEED 1 x SIMTIME 60 s contra o
# sqr-sim real (IPC), VERIFICA que cada run está completo (step_metrics.csv +
# checkpoint + último passo esperado + término limpo do par ns-3/Python) e
# roda a análise (campaign_analysis.py) sobre o piloto com figuras geradas
# com o sufixo _pilot.
#
# Diferenças deliberadas em relação à campanha completa:
#   - simTime 60 s (vs 600 s, R23) — apenas sanidade do encadeamento;
#   - --warmup_s 0 (o aquecimento R14 de 100 s excederia o horizonte do
#     piloto; a marcação warmup=0 mantém as 595 janelas elegíveis);
#   - diretório próprio: results/sqr_pilot (a campanha de 600 s usa
#     results/sqr_campaign — runs do piloto nunca são misturados nela).
#
# Uso:
#   bash examples/run_sqr_pilot.sh
# Variáveis: METHODS (default sqr-g0,sqr-g1,kbrl), PILOT_SEED (1),
#            PILOT_SIMTIME (60), JOBS (1)
#
# Critério de aceite: terminar com "PILOTO OK" e as 5 figuras _pilot em
# results/sqr_pilot/figures/.
# ---------------------------------------------------------------------------
set -euo pipefail

GYM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="${GYM_ROOT}/.venv/bin/python"
DRIVER="${GYM_ROOT}/examples/run_sqr_campaign.sh"
ANALYSIS="${GYM_ROOT}/src/nsoran/sqr/campaign_analysis.py"
PILOT_DIR="${GYM_ROOT}/results/sqr_pilot"

METHODS="${METHODS:-sqr-g0,sqr-g1,kbrl}"
SEED="${PILOT_SEED:-1}"
SIMTIME="${PILOT_SIMTIME:-60}"
SFX="_pilot"

fail() { echo "PILOTO FALHOU: $*" >&2; exit 1; }

[[ -x "${PYBIN}" ]] || fail "interpretador Python ausente: ${PYBIN}"
[[ -f "${DRIVER}" ]] || fail "driver da campanha ausente: ${DRIVER}"
[[ -f "${ANALYSIS}" ]] || fail "análise ausente: ${ANALYSIS}"

echo "[piloto] 3 métodos x seed ${SEED} x ${SIMTIME}s em ${PILOT_DIR}"
CAMPAIGN_DIR="${PILOT_DIR}" WARMUP_S=0 JOBS="${JOBS:-1}" \
    bash "${DRIVER}" "${METHODS}" "${SEED}" "${SIMTIME}"

# --- verificação dos runs (o driver já escreveu run_done.json) -------------
IFS=',' read -ra METHOD_LIST <<< "${METHODS}"
for m in "${METHOD_LIST[@]}"; do
    out="${PILOT_DIR}/${m}/seed_${SEED}"
    [[ -f "${out}/run_done.json" ]] || fail "${m}/seed_${SEED}: run_done.json ausente"
    [[ -s "${out}/step_metrics.csv" ]] || fail "${m}/seed_${SEED}: step_metrics.csv vazio"
    [[ -s "${out}/sqr_checkpoint.pkl" ]] || fail "${m}/seed_${SEED}: checkpoint vazio"
    grep -q "^\[sqr_run\] ns3_exit_code: 0" "${out}/sqr_run.log" \
        || fail "${m}/seed_${SEED}: ns-3 sem término limpo"
    echo "[piloto] OK: ${m}/seed_${SEED} completo"
done

# --- análise sobre o piloto (figuras com sufixo _pilot) --------------------
"${PYBIN}" "${ANALYSIS}" --campaign-dir "${PILOT_DIR}" --suffix "${SFX}"

# --- figuras presentes? -----------------------------------------------------
for name in \
    fig2_window_violation_rate \
    fig3_cumulative_violation \
    fig4_prb_distribution \
    fig_summary_violation_bars \
    fig_modes; do
    fig="${PILOT_DIR}/figures/${name}${SFX}.png"
    [[ -s "${fig}" ]] || fail "figura ausente: ${fig}"
    echo "[piloto] figura: ${fig}"
done

echo ""
echo "PILOTO OK"
