# D0B_baseline_audit.md — BASELINE, INSTRUMENTAÇÃO E CONGELAMENTO (PASS)

**Protocolo:** v2 §6 · **Data:** 08–09/11/2026 · **Pré-condição:** `D0A_GATE.json` = PASS (com revisões) ✓ · **Entregas:** este relatório, `INVENTORY.md`, `PROTOCOL_FROZEN.md`, A/A fresco em `results/sqr_jc_decision/d0b_aa/`.

## 1. Manifests, cenário e contrato (tarefa 1–2) — verificados, nada pressuposto

`INVENTORY.md` §1–2: caminhos reais verificados um a um. Cenário heterogêneo reconstruído dos runners congelados (flags idênticas: 300 s · slas 3,5/3,5/3,5 · periodMs 100 · c_prbs 52→telemetria 53 (R01) · warmup 100 s · cellRadius 300 · greedy/urllc/mmtc · λ/3 por slice R34). Contrato IPC/ação/KPM v1.4 (`contrato_interface.md`); `sla_manifest` v1.1-g4b; `resource_unit_manifest` (C=53=53RBGs=53PRBs — **errata aplicada no docstring `sqr_env.py` com texto original preservado**).

## 2. SQR-HET × Algoritmo 1 do CCNC 2026 (tarefa 3) — fidelidade CONFIRMADA por leitura

`sqr_jc/sqr_het.py` **reusa por import** (não reimplementa) o maquinário auditado: 3×`SqrSliceController` (kernel quantílico pinball; τ∈[0,01; 0,05] adaptativo R11 W=100; margem a←a+round(σ·M); exploração ε-softmax/kNN-safety-first/fallback) + **projeção original** `coordinator.project_prbs` (corte largest-remainder Σ≤C) + **updates virtuais negativos R12** na proposta pré-projeção + `Λ=0` na escala da margem normalizada comum + R19 (vacuous não atualiza). Guard explícito no construtor rejeita controlador que não seja o auditado e `Λ≠0`. Diferença vs Algoritmo 1: KPIs heterogêneos na margem comum (autorizado §27) — arquitetura de decisão idêntica.

## 3. Semântica dos KPIs / orientação URLLC (tarefa 4) — CORRETA por construção

O alvo de regressão é a **margem normalizada m_i** (maior=melhor para TODAS as slices; m_u=(10−p95)/10). O conjunto seguro f_τ(z,a) ≥ Λ usa τ **baixo** ⇒ quantil inferior da **margem** ⇒ limite conservador CORRETO (equivalente a limite superior conservador do p95/miss). A construção proibida pelo protocolo ("limiar − quantil inferior do miss") **não existe** no B0. Pinball verificado (`quantile_regressor.py`: subpred α=+ητ / overpred α=η(τ−1) / crítico ×P_penalty).

## 4. Replay — fidelidade (tarefa 5) — evidência máxima

**(novo) Re-execução A/A fresca** do replay E1 λ5s1 t1955 (37,13,3), args idênticos, executada hoje independente do original: **os 3 CSVs de telemetria bit-idênticos** — `sqr_kpm_history.csv` (10.631 linhas), `sqr_packet_delays.csv` (430.366), `sqr_prb_windows.csv` (1.970). Somado ao acervo: piloto 813/813 (II-a), 28/28 (II-b), 31 re-exec pós-locale (R56), 10/10 offline (D0-A). **Mesma ação + mesmo checkpoint ⇒ mesma saída, byte-a-byte, em re-execução.**

## 5. Dados para D1/D2 (tarefa 6) — gap de instrumentação identificado

- **B0 (SQR-HET) nunca rodou**: `CONTROLLER_CHOICES = (sqr, kbrl, equal, prop, replay)` — **pré-requisito D1**: wired `--controller sqr_het` + log por decisão (schema §4.2: a_raw → a_after_margin → a_after_projection → a_sent → a_applied + predições antes/depois). Instrumentação permitida (sem campanha de otimização); runs D1 com seeds próprias {2101–2103}×λ{2,5,6} (PROTOCOL_FROZEN §3).
- Replays base = {equal,prop} hoje; **extensão para base = run B0 arbitrário** é pré-requisito D2 (a sequência de ações B0 registrada vira a política de continuação).
- Reutilizáveis sem duplicação: 30 runs equal×prop (estados suplementares), 90+ replays, CAL/G4B (kernels ls=2.0/thr=90 congelados; G4-B STOP — 3-way só análise).

## 6. PROTOCOL_FROZEN.md (tarefa 7) — congelado ANTES da seleção de estados

Inclui o **esclarecimento estrutural pós-D0-A**: pisos a_u≥13/a_m≥3 pertenciam ao plano do oracle (R50/R58), **não ao controlador B0** ⇒ `A_common(B0)=A_common(B1)=simplex Σ≤53` (27.720 ações; B3 vazio por construção; (48,5,0) é admissível a ambos). Ranking do JointSelector (lexicográfico critical→efficient→minΣa→vetor), busca D2 (24 alternativas + exaustiva em 5 estados, seed de amostragem 20261108), métricas D3 pareadas (Δ_allOK principal; sem tautologia), gate D3 **exploratório declarado** (a–d), unidade de inferência = seed/run, 15 estados-ação do D0-A excluídos do teste D3.

## 7. Testes executados

`pytest tests/test_sqr_coordinator.py tests/test_sqr_jc_controller.py tests/test_sqr_replay.py` → **52 passed** (6,5 s). Suíte integral 342 passed (04/11). Divergências: nenhuma nova (as do D0-A propagadas com errata).

## GATE D0-B: **PASS**

Manifests/instrumentação confiáveis ✓ · SQR-HET bem definido e fiel ao Algoritmo 1 ✓ · replay repetível (bit-a-bit em re-execução) ✓ · PROTOCOL_FROZEN antes dos estados ✓. **D1 autorizado** (instrumentação sqr_het + runs B0 + estatísticas de projeção). D2/D3 gated; D4/D5 bloqueados.
