#!/bin/bash
# run_lowload.sh — Etapa 2.2 do MD da Rodada 2 (sanity em baixa carga, gate STOP §11.3).
# Chegadas EXPLÍCITAS por slice (mandato do orquestrador 20/10): λ=1,1,1 UEs/min
# (E[N]≈1 UE/slice — baixa contenção; equal-split dá ~1/3 da célula à slice eMBB,
# 1 UE ≈ 8 Mbps vs SLA 7 Mbps). Perfis greedy,urllc,mmtc (contrato v1.4 §4.2).
# Rádio CONGELADO das rodadas anteriores (300 m, 30 dBm, 10 MHz/C=53, UMa, omni
# 15 dBi, TDD 3DL/7, μ=0, PF, RLC UM). 300 s, seeds 1–3. Artefatos (KPM + packet
# delays) em diretório PRÓPRIO da campanha (mandato: preservar latência por pacote).
# Uso: setsid bash run_lowload.sh > lowload_runner.log 2>&1 &
set -u
NS3_ROOT=/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev
GATE="$NS3_ROOT/scratch/sqr/ns3_gate.sh"
BASE=/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_characterization_round2
SW="$BASE/lowload"
JOBS=3

run_one() {
  local seed="$1"
  local tag="hetero_s${seed}"
  local dir="$SW/$tag"
  if [[ -s "$dir/.done" ]]; then echo "[runner] $tag já completo, pulando"; return 0; fi
  rm -rf "$dir"; mkdir -p "$dir"
  echo "[runner] $(date +%H:%M:%S) iniciando $tag (300 s, sliceLambdas=1,1,1, greedy/urllc/mmtc)"
  ( cd "$NS3_ROOT" && bash "$GATE" run sqr-sim -- \
      --simTime=300 --seed="$seed" --run=1 --periodMs=100 \
      --sliceLambdas=1,1,1 --sliceApps=greedy,urllc,mmtc --cellRadius=300 \
      --outputDir="$dir" ) > "$SW/$tag.log" 2>&1
  local rc=$?
  local ok=0
  [[ -s "$dir/results_sqr_network_only/seed=${seed}_run=1/sqr_kpm_history.csv" ]] && ok=1
  [[ -s "$dir/results_sqr_network_only/seed=${seed}_run=1/sqr_packet_delays.csv" ]] || ok=0
  if [[ $rc -eq 0 && $ok -eq 1 ]]; then
    echo "$tag rc=0 $(date +%H:%M:%S)" > "$dir/.done"
    echo "[runner] $(date +%H:%M:%S) OK $tag"
  else
    echo "[runner] $(date +%H:%M:%S) FALHOU $tag rc=$rc (KPM/delays presentes: $ok)"
  fi
}

for seed in 1 2 3; do
  while [[ $(jobs -rp | wc -l) -ge $JOBS ]]; do sleep 5; done
  run_one "$seed" &
done
wait
n_ok=$(ls "$SW"/hetero_s*/.done 2>/dev/null | wc -l)
echo "[runner] FIM $(date +%H:%M:%S) — $n_ok/3 completos (KPM + packet_delays preservados em $SW)"
