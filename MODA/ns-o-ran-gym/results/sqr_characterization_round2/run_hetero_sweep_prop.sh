#!/bin/bash
# run_hetero_sweep_prop.sh — Etapa 2.3, BRAÇO ADAPTATIVO (mandato 21/10):
# mesmo grade do braço estático (Perfil A balanceado, λ_total ∈ {2,4,5,6,7} ÷ 3
# por slice, seeds 1–3, 300 s, rádio congelado), com o controlador 'prop'
# (witness adaptativo) no loop IPC. SLAs congelados R36–R39.
# Uso: setsid bash run_hetero_sweep_prop.sh > hetero_sweep_prop_runner.log 2>&1 &
set -u
GYM=/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym
BASE=$GYM/results/sqr_characterization_round2
SW="$BASE/hetero_sweep_prop"
JOBS=1
# Fix 21/10 (nota docs/sqr/nota_diagnostico_degradacao_2026-10-21.md): mesmo
# fix do braço estático — TIMEOUT 1800 s (420 s matava runs saudáveis: 300 s de
# sim levam ~480–650 s de parela) e LOCK flock compartilhado com o braço
# network-only (as versões de 06:29 rodavam 10 sims simultâneas → contenda de
# RAM/ritmo ~1% e OOM 08:04). Diagnóstico: NÃO há vazamento no binário atual —
# run solo IPC manteve py 177→178 MB / ns3 68→108 MB, exit 0.
TIMEOUT_S=${TIMEOUT_S:-1800}
LOCK="$SW/.lock_braço"

run_one() {
  local lt="$1" seed="$2"
  local lam3
  lam3=$(awk -v l="$lt" 'BEGIN{printf "%.3f", l/3}')
  local tag="A_lam${lt}_s${seed}"
  local dir="$SW/$tag"
  if [[ -s "$dir/.done" ]]; then echo "[runner] $tag já completo, pulando"; return 0; fi
  local tent rc ok
  for tent in 1 2 3; do
    rm -rf "$dir"; mkdir -p "$dir"
    echo "[runner] $(date +%H:%M:%S) iniciando $tag tent.$tent (prop IPC, λ=${lam3}/slice, 300 s, timeout ${TIMEOUT_S}s, lock)"
    ( flock -x 9
      cd "$GYM" && timeout -s KILL "${TIMEOUT_S}" .venv/bin/python examples/sqr_run.py --controller prop --seed "$seed" \
        --simTime 300 --num_slices 3 --slas 3.5,3.5,3.5 --output "$dir" \
        --sliceLambdas="${lam3},${lam3},${lam3}" --sliceApps=greedy,urllc,mmtc \
        --cellRadius=300 ) 9>"$LOCK" > "$SW/$tag.log" 2>&1
    rc=$?
    ok=0
    [[ -s "$dir/step_metrics.csv" ]] && ok=1
    if [[ -n $(find "$dir" -name sqr_kpm_history.csv -size +0 2>/dev/null) ]]; then ok=$((ok+1)); fi
    if [[ -n $(find "$dir" -name sqr_packet_delays.csv -size +0 2>/dev/null) ]]; then ok=$((ok+1)); fi
    if [[ $rc -eq 0 && $ok -eq 3 ]]; then
      echo "$tag rc=0 $(date +%H:%M:%S) tent.$tent" > "$dir/.done"
      echo "[runner] $(date +%H:%M:%S) OK $tag (tent.$tent)"
      return 0
    fi
    echo "[runner] $(date +%H:%M:%S) FALHOU $tag rc=$rc (checks: $ok/3) tent.$tent"
    # limpa semáforos IPC órfãos da tentativa morta (kill -9 não remove)
    rm -f /dev/shm/sem_*sqr* /dev/shm/sem_metrics_* /dev/shm/sem_control_* 2>/dev/null
  done
}

for lt in 2 4 5 6 7; do
  for seed in 1 2 3; do
    while [[ $(jobs -rp | wc -l) -ge $JOBS ]]; do sleep 5; done
    run_one "$lt" "$seed" &
  done
done
wait
n_ok=$(ls "$SW"/A_lam*/.done 2>/dev/null | wc -l)
echo "[runner] FIM $(date +%H:%M:%S) — $n_ok/15 completos (braço prop)"
