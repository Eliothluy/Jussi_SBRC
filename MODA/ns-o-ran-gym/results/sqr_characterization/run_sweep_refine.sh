#!/bin/bash
# run_sweep_refine.sh — refinamento da transição λ∈{5,6,7} (Rodada 1b).
# Mesmo rádio CONGELADO da Rodada 1 (300 m, 30 dBm, 10 MHz/C=53, UMa, omni
# 15 dBi, TDD DL|DL|DL|S|UL|UL|UL, μ=0, PF, RLC UM, greedy 100 Mbps/UE,
# sessão Exp(60 s), equal weights, network-only, 300 s) — só λ e seed mudam.
# Uso: setsid bash run_sweep_refine.sh > refine_runner.log 2>&1 &
set -u
NS3_ROOT=/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev
GATE="$NS3_ROOT/scratch/sqr/ns3_gate.sh"
BASE=/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_characterization
SW="$BASE/sweep"
JOBS=9

run_one() {
  local lam="$1" seed="$2"
  local tag="lam${lam}_s${seed}"
  local dir="$SW/$tag"
  # pula runs já completos (retomável)
  if [[ -s "$dir/.done" ]]; then echo "[runner] $tag já completo, pulando"; return 0; fi
  rm -rf "$dir"; mkdir -p "$dir"
  echo "[runner] $(date +%H:%M:%S) iniciando $tag"
  ( cd "$NS3_ROOT" && bash "$GATE" run sqr-sim -- \
      --simTime=300 --seed="$seed" --run=1 --periodMs=100 \
      --lambdaUesPerMin="$lam" --cellRadius=300 \
      --outputDir="$dir" ) > "$SW/$tag.log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 && -s "$dir/results_sqr_network_only/seed=${seed}_run=1/sqr_kpm_history.csv" ]]; then
    echo "$tag rc=0 $(date +%H:%M:%S)" > "$dir/.done"
    echo "[runner] $(date +%H:%M:%S) OK $tag"
  else
    echo "[runner] $(date +%H:%M:%S) FALHOU $tag rc=$rc"
  fi
}

pids=()
for lam in 5 6 7; do
  for seed in 1 2 3; do
    # no máximo JOBS paralelos
    while [[ $(jobs -rp | wc -l) -ge $JOBS ]]; do sleep 5; done
    run_one "$lam" "$seed" &
    pids+=($!)
  done
done
wait
n_ok=$(ls "$SW"/lam[567]_s*/.done 2>/dev/null | wc -l)
echo "[runner] FIM $(date +%H:%M:%S) — $n_ok/9 completos"
