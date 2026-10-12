"""Shared KPI scoring for RSLAQ slice summary rows.

Used by both the offline meta-heuristic search and the post-search evaluation
so that baselines, heuristics, and meta-heuristics are ranked by the same
score function.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence


SLICE_NAMES = ("eMBB", "URLLC", "MTC")


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, "", "NA"):
            return default
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(max(value, lo), hi)


def normalize_weights(values: Sequence[float]) -> list[float]:
    clean = [max(float(v), 0.0) for v in values[:3]]
    if len(clean) < 3:
        clean.extend([0.0] * (3 - len(clean)))
    total = sum(clean)
    if total <= 0.0:
        return [1.0 / 3.0] * 3
    return [v / total for v in clean]


def format_weights(weights: Sequence[float]) -> str:
    normalized = normalize_weights(weights)
    first = round(normalized[0], 6)
    second = round(normalized[1], 6)
    third = round(1.0 - first - second, 6)
    if third < 0.0:
        rounded = normalize_weights([first, second, max(third, 0.0)])
    else:
        rounded = [first, second, third]
    return ",".join(f"{weight:.6f}" for weight in rounded)


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def score_summary_rows(rows: Iterable[dict[str, str]]) -> float:
    by_slice = {row.get("slice", ""): row for row in rows}
    if not all(name in by_slice for name in SLICE_NAMES):
        return -1e9

    sla = []
    offered = []
    pdr = []
    util = []
    for name in SLICE_NAMES:
        row = by_slice[name]
        sla.append(clamp(safe_float(row.get("sla_satisfaction_pct")) / 100.0))
        offered.append(clamp(safe_float(row.get("offered_load_satisfaction_pct")) / 100.0))
        pdr.append(clamp(safe_float(row.get("pdr_pct")) / 100.0))
        util_value = safe_float(row.get("budget_utilization_pct_mean"), 50.0)
        util.append(clamp(util_value / 100.0))

    mean_sla = sum(sla) / len(sla)
    min_sla = min(sla)
    mean_offered = sum(offered) / len(offered)
    mean_pdr = sum(pdr) / len(pdr)
    mean_util = sum(util) / len(util)

    urllc = by_slice["URLLC"]
    mtc = by_slice["MTC"]
    embb = by_slice["eMBB"]

    urllc_delay_ms = safe_float(urllc.get("delay_ms_mean"))
    # URLLC delay SLA violation. The URLLC latency budget is 10 ms; the penalty
    # saturates at 30 ms (well past the URLLC viability threshold), so the
    # divisor is 20 ms. The previous /200.0 made the penalty negligible across
    # the observed operating range (e.g. 14.864 ms -> -0.6 pts), letting the
    # optimizer accept solutions that openly violate the 10 ms target.
    urllc_delay_penalty = clamp((urllc_delay_ms - 10.0) / 20.0)

    mtc_thr = safe_float(mtc.get("throughput_mbps_mean"))
    mtc_starvation_penalty = max(0.0, 1.0 - sla[2])
    if mtc_thr <= 1e-9:
        mtc_starvation_penalty += 0.5

    embb_buffer = safe_float(embb.get("buffer_bytes_mean"))
    embb_buffer_penalty = 0.05 * clamp(embb_buffer / 5_000_000.0)

    score = 100.0 * (
        0.35 * mean_sla
        + 0.20 * min_sla
        + 0.20 * mean_offered
        + 0.15 * mean_pdr
        + 0.10 * mean_util
    )
    score -= 25.0 * urllc_delay_penalty
    score -= 40.0 * mtc_starvation_penalty
    score -= 100.0 * embb_buffer_penalty
    return score


# ---------------------------------------------------------------------------
# Score v2 (reformulação da auditoria): URLLC latency como ÚNICA restrição rígida
# ---------------------------------------------------------------------------
# Calibração decidida no piloto congestion (2026-07-18): dos alvos de SLA, apenas
# a latência de cauda URLLC (p99) é fisicamente atingível sob sobrecarga; PDR e
# throughput de MTC/eMBB têm teto abaixo do ideal quando a célula está saturada
# (245 Mbps ofertados vs. capacidade da célula), tornando qualquer região viável
# vazia se fossem restrições rígidas. Portanto:
#   - restrição RÍGIDA: p99_URLLC <= V2_URLLC_DELAY_MS_MAX (o SLA canônico URLLC);
#   - PDR/throughput entram GRADUADOS na satisfação composta por slice
#     (sla_satisfaction_pct do C++ = min(S_thr,S_pdr,S_delay), com alvos em
#     SLA_TARGETS do rslaq-sim.cc), não como gates.
# Assim todo cenário com p99 atingível tem região viável, e o score discrimina
# por quão bem a alocação serve seus slices (fairness max-min + média).
V2_URLLC_DELAY_MS_MAX = 10.0     # deadline p99 URLLC — ÚNICA restrição rígida
# URLLC morto (nenhum pacote entregue -> histograma vazio -> p99=NA) NÃO é
# "0 ms viável": é o pior outage possível. Sem esta guarda, o otimizador
# poderia satisfazer a restrição de latência simplesmente MATANDO o URLLC
# (PDR=0 -> p99=NA -> safe_float=0 <= 10ms -> "viável"), reintroduzindo o
# paradoxo de starvation da v1. Tratamos como latência sentinela alta.
V2_URLLC_DEAD_P99_MS = 10000.0   # p99 atribuído a URLLC morto (infeasível)
V2_URLLC_MIN_PDR_ALIVE = 1e-9    # PDR abaixo disto = slice morto
# Alvos de GRADAÇÃO (espelham SLA_TARGETS no C++; NÃO são gates em Python — o
# grading S_thr/S_pdr/S_delay já é feito no C++ e chega via sla_satisfaction_pct).
V2_URLLC_PDR_MIN_PCT = 90.0
V2_MTC_PDR_MIN_PCT = 80.0
V2_EMBB_THR_FRAC_OFFERED = 0.60
# Peso da componente max-min (fairness) vs. média na pontuação de soluções
# viáveis. 0.5 pune fortemente starvation (min domina) sem ignorar o serviço
# global (média).
V2_MINSLA_WEIGHT = 0.5


def score_summary_rows_v2(rows: Iterable[dict[str, str]]) -> float:
    """Score v2: URLLC tail latency é a única restrição rígida; fairness governa.

    - INFEASÍVEL (p99_URLLC > deadline): score < 0 proporcional ao excesso de
      latência (gradiente rumo à viabilidade); sempre perde para qualquer viável.
    - FEASÍVEL: score = 100 * (w*min_i SLA_i + (1-w)*mean_i SLA_i), onde SLA_i é
      a satisfação COMPOSTA por slice (min(S_thr,S_pdr,S_delay), vinda do C++).
      O termo min_i implementa fairness max-min e pune starvation de qualquer
      slice; a média recompensa servir bem todos os slices. Sem termo de
      utilização (a auditoria observou que premiar esgotar espectro é indevido).

    Assinatura list[dict]->float idêntica a score_summary_rows (plugável via
    --score_version=v2, sem tocar nos otimizadores).
    """
    by_slice = {row.get("slice", ""): row for row in rows}
    if not all(name in by_slice for name in SLICE_NAMES):
        return -1e9

    # Única restrição rígida: latência de cauda URLLC. Um URLLC morto (sem
    # pacotes entregues -> p99 ausente/NA, ou PDR~0) é outage, não 0 ms: tratado
    # como latência sentinela para ficar infeasível (fecha a brecha de starvation).
    urllc_row = by_slice["URLLC"]
    p99_raw = urllc_row.get("delay_ms_p99")
    urllc_pdr = safe_float(urllc_row.get("pdr_pct"))
    if p99_raw in (None, "", "NA") or urllc_pdr <= V2_URLLC_MIN_PDR_ALIVE:
        urllc_p99 = V2_URLLC_DEAD_P99_MS
    else:
        urllc_p99 = safe_float(p99_raw)

    if urllc_p99 > V2_URLLC_DELAY_MS_MAX and V2_URLLC_DELAY_MS_MAX > 0.0:
        return -100.0 * (urllc_p99 - V2_URLLC_DELAY_MS_MAX) / V2_URLLC_DELAY_MS_MAX

    # Feasível: fairness max-min + média da satisfação de SLA composta por slice.
    sla = [clamp(safe_float(by_slice[name].get("sla_satisfaction_pct")) / 100.0)
           for name in SLICE_NAMES]
    mean_sla = sum(sla) / len(sla)
    min_sla = min(sla)
    return 100.0 * (V2_MINSLA_WEIGHT * min_sla + (1.0 - V2_MINSLA_WEIGHT) * mean_sla)
