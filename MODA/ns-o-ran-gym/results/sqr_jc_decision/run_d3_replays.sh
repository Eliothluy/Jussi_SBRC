#!/usr/bin/env bash
set -uo pipefail
export LC_ALL=C
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM="/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym"
OUT="$D/d3_replays"; mkdir -p "$OUT"
PLAN="${PLAN:-$D/D3_plan_shard0.tsv}"
tail -n +2 "$PLAN" | while IFS=$'\t' read -r sid run lam seed win branch a; do
  tag="${sid}_${branch}"
  [[ -s "$OUT/$tag/.done" ]] && continue
  st=$(( 10#$win / 10 + 2 ))
  mkdir -p "$OUT/$tag"
  ( cd "$GYM" && timeout -s KILL 3600 .venv/bin/python examples/sqr_run.py \
      --controller replay --replay_base "$D/d1_runs_v9_delayed/$run" \
      --replay_at_win "$win" --replay_action "$(echo "$a" | tr '-' ',')" \
      --seed "$seed" --simTime "$st" --c_prbs 53 --num_slices 3 --slas 3.5,3.5,3.5 \
      --sliceLambdas="$(awk -v l="$lam" 'BEGIN{printf "%.3f", l/3}'),$(awk -v l="$lam" 'BEGIN{printf "%.3f", l/3}'),$(awk -v l="$lam" 'BEGIN{printf "%.3f", l/3}')" --sliceApps=greedy,urllc,mmtc \
      --cellRadius=300 --output "$OUT/$tag" ) > "$OUT/$tag/run.log" 2>&1
  rc=$?
  ok=0
  [[ -n "$(find "$OUT/$tag" -name sqr_kpm_history.csv -size +0 2>/dev/null)" ]] && ok=1
  if [[ $rc -eq 0 && $ok -eq 1 ]] && grep -q "ns3_exit_code: 0" "$OUT/$tag/run.log"; then
    echo "$tag rc=0 $(date +%H:%M:%S)" > "$OUT/$tag/.done"
    echo "[d3r] $(date +%H:%M:%S) OK $tag"
  else
    echo "[d3r] $(date +%H:%M:%S) FALHOU $tag rc=$rc"
  fi
done
echo "[d3r] FIM $(date +%H:%M:%S)"
