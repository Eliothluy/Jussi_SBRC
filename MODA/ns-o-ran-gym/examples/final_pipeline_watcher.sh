#!/usr/bin/env bash
# Watcher: aguarda as 50 runs DDQN completarem e executa o pipeline final
# (extract → rescore paralelo → merge → analyze) para os 5 cenários × 10 seeds.
set -uo pipefail

GYM=/home/eliothluy/Documentos/artigo_jussi/ns-o-ran-gym
OUT="$GYM/results_controlled/heuristics_metaheuristics/20260811_rslaq_ddqn_paper/rslaq_ddqn_paper"
NS3=/home/eliothluy/Documentos/artigo_jussi/ns-3-dev
PAPER=/home/eliothluy/Documentos/artigo_jussi/paper_v2_campaign

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

# ---------------------------------------------------------------------------
# 1. Aguardar 50/50 summaries
# ---------------------------------------------------------------------------
log "aguardando 50/50 ddqn_summary.json..."
while true; do
    n=$(find "$OUT" -maxdepth 2 -name ddqn_summary.json 2>/dev/null | wc -l)
    [ "$n" -ge 50 ] && break
    log "  $n/50 — aguardando 5 min"
    sleep 300
done
log "50/50 runs completas. Iniciando pipeline final."

# ---------------------------------------------------------------------------
# 2. Extract (congestion + insufficient apenas; os outros 3 já foram extraídos)
# ---------------------------------------------------------------------------
log "[EXTRACT] congestion + insufficient_resources (20 runs, eval=3)..."
cd "$GYM"
python3 examples/extract_ddqn_weights.py \
    --ddqn-root "$OUT" \
    --output "$OUT/ddqn_extracted_weights_ci.csv" \
    --ns3-dir "$NS3" \
    --eval-episodes 3 \
    --scenarios "congestion,insufficient_resources" || log "EXTRACT FALHOU (continuando com o que houver)"

# ---------------------------------------------------------------------------
# 3. Rescore em 2 workers paralelos (1 por cenário)
# ---------------------------------------------------------------------------
log "[RESCORE] disparando 2 workers paralelos..."
python3 examples/rescore_ddqn_standalone.py \
    --weights-csv "$OUT/ddqn_extracted_weights_ci.csv" \
    --ns3-dir "$NS3" \
    --output-root "$OUT/ddqn_rescored" \
    --output-csv "$OUT/ddqn_scored_v2_congestion.csv" \
    --sim-time 5 --scenarios "congestion" &
PID_CONG=$!
python3 examples/rescore_ddqn_standalone.py \
    --weights-csv "$OUT/ddqn_extracted_weights_ci.csv" \
    --ns3-dir "$NS3" \
    --output-root "$OUT/ddqn_rescored" \
    --output-csv "$OUT/ddqn_scored_v2_insufficient.csv" \
    --sim-time 5 --scenarios "insufficient_resources" &
PID_INS=$!
log "  worker congestion=$PID_CONG, insufficient=$PID_INS (ETA ~5-9h)"
wait $PID_CONG; log "  congestion worker done (exit=$?)"
wait $PID_INS;  log "  insufficient worker done (exit=$?)"

# ---------------------------------------------------------------------------
# 4. Merge: 30 antigas (low/normal/stressed, já com retry) + 20 novas
# ---------------------------------------------------------------------------
log "[MERGE] consolidando ddqn_scored_v2_all.csv..."
python3 << 'PYEOF'
import csv

OUT = "/home/eliothluy/Documentos/artigo_jussi/ns-o-ran-gym/results_controlled/heuristics_metaheuristics/20260811_rslaq_ddqn_paper/rslaq_ddqn_paper"

rows = []
fieldnames = None

# base: low/normal/stressed (30 rows, retry stressed já mergeado)
with open(f"{OUT}/ddqn_scored_v2.csv") as f:
    r = csv.DictReader(f)
    fieldnames = r.fieldnames
    rows.extend(r)

# novas: congestion + insufficient (apenas status=ok)
for name in ("congestion", "insufficient"):
    try:
        with open(f"{OUT}/ddqn_scored_v2_{name}.csv") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    rows.append({k: row.get(k, "") for k in fieldnames})
    except FileNotFoundError:
        print(f"[WARN] ddqn_scored_v2_{name}.csv ausente")

with open(f"{OUT}/ddqn_scored_v2_all.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

from collections import Counter
c = Counter(r["scenario"] for r in rows)
print(f"total: {len(rows)} rows")
for sc, n in sorted(c.items()):
    print(f"  {sc}: {n}")
PYEOF

# ---------------------------------------------------------------------------
# 5. Análise final: 5 cenários × 10 seeds vs meta-heurísticas
# ---------------------------------------------------------------------------
log "[ANALYZE] comparação final 5 cenários..."
python3 examples/analyze_ddqn_vs_meta.py \
    --meta-csv "$PAPER/v2_meta_unified_5scenarios.csv" \
    --ddqn-csv "$OUT/ddqn_scored_v2_all.csv" \
    --baseline-csv "$GYM/results_controlled/heuristics_metaheuristics_v2/v2_opt_vs_baseline_paired.csv" \
    --output-dir "$PAPER"

log "===== PIPELINE FINAL COMPLETO ====="
