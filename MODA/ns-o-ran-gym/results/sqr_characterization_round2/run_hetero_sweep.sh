#!/bin/bash
# run_hetero_sweep.sh — Etapa 2.3 do MD da Rodada 2 (sweep heterogêneo, 1ª leva).
# PERFIL A (balanceado, MD §14): λ_total ∈ {2,4,5,6,7} UE/min distribuído igualmente
# por slice (λ/3 cada), --sliceApps=greedy,urllc,mmtc, rádio congelado das rodadas
# anteriores, 300 s, seeds 1–3, network-only, EQUAL-SPLIT como baseline estático
# (consistência com 1b/2.0; o controlador prop/oracle 2.4 quantifica o evitável).
# NADA além de λ muda entre pontos (mandato da caracterização).
# Uso: setsid bash run_hetero_sweep.sh > hetero_sweep_runner.log 2>&1 &
set -u
NS3_ROOT=/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev
GATE="$NS3_ROOT/scratch/sqr/ns3_gate.sh"
BASE=/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_characterization_round2
SW="$BASE/hetero_sweep"
JOBS=1
# Fix 21/10 (nota docs/sqr/nota_diagnostico_degradacao_2026-10-21.md):
# - TIMEOUT 420 s matava runs SAUDÁVEIS (run solo de 300 s leva ~480–650 s de
#   parela a 0,4–0,7×; rc=124/137 com artefatos completos nos logs de 09:59).
#   1800 s = folga ~3× sobre o pior caso observado.
# - LOCK compartilhado com o braço prop (flock): as versões de 06:29 lançavam
#   5 runs por braço + o outro braço em paralelo = 10 sims + 5 pythons →
#   contenda de RAM (zram thrashing, ritmo ~1%) e OOM global 08:04 (journalctl:
#   ns3 anon-rss 2,46 GB). Um lock por RUN serializa os dois braços.
# - O "vazamento de RSS por tempo de parela" do dossiê NÃO se reproduz no
#   binário 18:11 de 20/10: run solo 300 s manteve RSS 82→106 MB (network-only)
#   e py 177→178 / ns3 68→108 MB (prop IPC), e 5 simultâneos completaram
#   5/5 com RSS ≤ 120 MB e PSI-CPU 0. Retry mantido só para falha genuína.
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
    echo "[runner] $(date +%H:%M:%S) iniciando $tag tent.$tent (λ=${lam3},${lam3},${lam3}/slice, 300 s, timeout ${TIMEOUT_S}s, lock)"
    ( flock -x 9
      cd "$NS3_ROOT" && timeout -s KILL "${TIMEOUT_S}" bash "$GATE" run sqr-sim -- \
        --simTime=300 --seed="$seed" --run=1 --periodMs=100 \
        --sliceLambdas="${lam3},${lam3},${lam3}" --sliceApps=greedy,urllc,mmtc \
        --cellRadius=300 --outputDir="$dir" ) 9>"$LOCK" > "$SW/$tag.log" 2>&1
    rc=$?
    ok=0
    [[ -s "$dir/results_sqr_network_only/seed=${seed}_run=1/sqr_kpm_history.csv" ]] && ok=1
    [[ -s "$dir/results_sqr_network_only/seed=${seed}_run=1/sqr_packet_delays.csv" ]] || ok=0
    if [[ $rc -eq 0 && $ok -eq 1 ]]; then
      echo "$tag rc=0 $(date +%H:%M:%S) tent.$tent" > "$dir/.done"
      echo "[runner] $(date +%H:%M:%S) OK $tag (tent.$tent)"
      return 0
    fi
    echo "[runner] $(date +%H:%M:%S) FALHOU $tag rc=$rc (KPM/delays: $ok) tent.$tent"
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
echo "[runner] FIM $(date +%H:%M:%S) — $n_ok/15 completos (Perfil A, λ_total ∈ {2,4,5,6,7})"
