#!/usr/bin/env bash
# run_d2_replays.sh — executa os 390 replays D2 (D2_sampling_FINAL.csv),
# ordem do CSV (ajuste 5: sem ordenamento por oracle). Idempotente/.done.
set -uo pipefail
export LC_ALL=C
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM="/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym"
OUT="$D/d2_v2_t2064_full"; mkdir -p "$OUT"
PLAN="${PLAN:-$D/D2_plan_V9_t2064full_0.tsv}"
tail -n +2 "$PLAN" | while IFS=$'\t' read -r sid run lam seed win alt a mech sub; do
  tag="${sid}_alt${alt}_${a}"
  [[ -s "$OUT/$tag/.done" ]] && continue
  st=300
  lam3=$(awk -v l="$lam" 'BEGIN{printf "%.3f", l/3}')
  mkdir -p "$OUT/$tag"
  ( cd "$GYM" && timeout -s KILL 3600 .venv/bin/python examples/sqr_run.py \
      --controller replay --replay_base "$D/d1_runs_v9_delayed/$run" \
      --replay_at_win "$win" --replay_action "$(echo "$a" | tr '-' ',')" \
      --seed "$seed" --simTime "$st" --c_prbs 53 --num_slices 3 --slas 3.5,3.5,3.5 \
      --sliceLambdas="${lam3},${lam3},${lam3}" --sliceApps=greedy,urllc,mmtc \
      --cellRadius=300 --output "$OUT/$tag" ) > "$OUT/$tag/run.log" 2>&1
  rc=$?
  ok=0
  [[ -n "$(find "$OUT/$tag" -name sqr_kpm_history.csv -size +0 2>/dev/null)" ]] && ok=1
  if [[ $rc -eq 0 && $ok -eq 1 ]] && grep -q "ns3_exit_code: 0" "$OUT/$tag/run.log"; then
    echo "$tag rc=0 $(date +%H:%M:%S)" > "$OUT/$tag/.done"
    echo "[d2r] $(date +%H:%M:%S) OK $tag"
  else
    echo "[d2r] $(date +%H:%M:%S) FALHOU $tag rc=$rc"; # sem cleanup global (/dev/shm): semaforos sao por-simUuid; em paralelo o cleanup mataria as instancias irmas
  fi
done
echo "[d2r] FIM $(date +%H:%M:%S) — $(ls $OUT/*/.done 2>/dev/null | wc -l)/390 (V9)"
