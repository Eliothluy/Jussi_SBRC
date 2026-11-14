#!/usr/bin/env bash
# run_d1.sh — D1: runs do B0 (SQR-HET) — protocolo v2 §7 / PROTOCOL_FROZEN §3.
# λ∈{2,5,6} × seeds {2101,2102,2103} (disjuntas de CAL/PILOT/MAIN/G4B),
# flags CONGELADAS idênticas às campanhas (300 s, slas 3,5, c_prbs 52→53,
# warmup 100 s, cellRadius 300, greedy/urllc/mmtc, λ/3 por slice).
# Lições: LC_ALL=C; .done idempotente; 3 tentativas; TIMEOUT_S generoso
# (crawl intermitente); NUNCA matar o filho ns-3 (lição s1102).
set -uo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM="/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym"
OUT="$D/d1_runs_v9_delayed"
TIMEOUT_S="${TIMEOUT_S:-3600}"
mkdir -p "$OUT"

run_one() {
  local lt="$1" seed="$2"
  local lam3; lam3="$(awk -v l="$lt" 'BEGIN{printf "%.3f", l/3}')"
  local tag="het_lam${lt}_s${seed}"; local dir="$OUT/$tag"
  if [[ -s "$dir/.done" ]]; then echo "[d1] $tag ok prévio"; return 0; fi
  local tent rc
  for tent in 1 2 3; do
    rm -rf "$dir"; mkdir -p "$dir"
    echo "[d1] $(date +%H:%M:%S) $tag tent.$tent (SQR-HET, λ=${lam3}/slice)"
    set +e
    ( cd "$GYM" && timeout -s KILL "$TIMEOUT_S" .venv/bin/python examples/sqr_run.py \
        --controller sqr_het --seed "$seed" --simTime 300 --num_slices 3 \
        --slas 3.5,3.5,3.5 --periodMs 100 --c_prbs 52 --warmup_s 100 \
        --sliceLambdas="${lam3},${lam3},${lam3}" --sliceApps=greedy,urllc,mmtc \
        --cellRadius=300 --urllc_delayed_update --output "$dir" ) > "$dir/run.log" 2>&1
    rc=$?
    set -e
    # completude: decisões registradas (~2900) + exit 0 + telemetry (lições D0-A)
    local ndec tel
    ndec=$( [ -f "$dir/decisions.jsonl" ] && wc -l < "$dir/decisions.jsonl" || echo 0 )
    tel=$(find "$dir" -name sqr_kpm_history.csv -size +0 2>/dev/null | wc -l)
    # Opção B: holds não geram linhas ⇒ decisões variam por run (λ2s2102 ~844);
    # completude = SIM inteiro (último KPM >= 290000) + exit 0 + telemetria
    # + >=600 decisões reais (lições D0-A)
    local kts
    kts=$(find "$dir" -name sqr_kpm_history.csv -exec sh -c 'tail -1 "$1" | cut -d, -f1' _ {} \; 2>/dev/null | head -1)
    if [[ $rc -eq 0 && ${ndec:-0} -ge 600 && ${tel:-0} -ge 1 \
          && ${kts:-0} -ge 290000 ]] \
       && grep -q "ns3_exit_code: 0" "$dir/run.log"; then
      echo "$tag rc=0 decisões=$ndec $(date +%H:%M:%S) tent.$tent" > "$dir/.done"
      echo "[d1] $(date +%H:%M:%S) OK $tag ($ndec decisões)"
      return 0
    fi
    echo "[d1] $(date +%H:%M:%S) FALHOU $tag rc=$rc dec=$ndec tel=$tel tent.$tent"
    rm -f /dev/shm/sem_metrics_* /dev/shm/sem_control_* 2>/dev/null || true
  done
  echo "[d1] $tag: 3 tentativas falharam" >&2
  return 1
}

for lt in 2 5 6; do for seed in 2101 2102 2103; do run_one "$lt" "$seed" || true; done; done
echo "[d1] FIM $(date +%H:%M:%S) — $(ls $OUT/*/.done 2>/dev/null | wc -l)/9"
