# AUDITORIA TÉCNICA, METODOLÓGICA E ESTATÍSTICA COMPLETA
## Campanha de Otimização de Slices 5G — Meta-heurísticas

**Run tag**: `20260626_122326`
**Data da auditoria**: 2026-07-14
**Cenários analisados**: low_traffic, normal, congestion (3 cenários completos)
**Fonte primária**: dados brutos (`summary.csv`, `candidate.json`, `scoring.py`)

---

## 1. Resumo Executivo

A auditoria revela que **a otimização por meta-heurísticas NÃO demonstra, com os dados atuais, melhoria estatisticamente significativa sobre schedulers puros**. Os 3 achados mais críticos são:

1. **O ruído entre seeds domina completamente o sinal entre métodos**: o coeficiente de variação entre seeds (9–32%) é 4 a 15× maior que a diferença entre GA/PSO/SA/Híbrida (1.7–5.5 pontos). Com n=3, é impossível declarar qualquer método superior.

2. **A função objetivo tem uma falha grave**: recompensa configurações que causam starvation de MTC. O caso `sa_eval_0021` (congestion/seed2) recebeu o **score mais alto (85.28)** com pesos `[0.816, 0.180, 0.004]` — essencialmente matando o slice MTC (PDR=2.4%, delay=4479ms) enquanto reporta paradoxalmente `sla_satisfaction=100%`.

3. **URLLC viola SLA sistematicamente**: 61% das células têm `delay_p99 > 10ms`, com caudas de até 1663ms. As meta-heurísticas **nunca** mantêm p99 ≤ 10ms em congestion (todas as 12 células excedem, 21–175ms).

**Conclusão objetiva**: Não é possível concluir que a otimização é eficiente com os dados apresentados. A função objetivo é inadequada, a validade estatística é insuficiente (n=3), e há inconsistências métricas graves (SLA=100% com PDR<10%; `delay_mean > delay_p95` em 20 células; `plr=0` universalmente).

---

## 2. Descrição Reconstruída do Experimento

| Elemento experimental | Valor encontrado | Fonte | Situação |
|---|---:|---|---|
| Simulador | ns-3 5G-LENA (5G-LENA nr module) | rslaq-sim.cc | confirmado |
| Duração da simulação | 5 segundos | metadata.json, run_all_scenarios.sh | confirmado |
| Warm-up | 0.4s (appStart) | run_all_scenarios.sh | confirmado |
| UEs (low_traffic) | 10 (eMBB=2, URLLC=2, MTC=6) | summary.csv | confirmado |
| UEs (normal) | 20 | metadata.json | confirmado |
| UEs (congestion) | 60 | metadata.json | confirmado |
| Largura de banda | 100 MHz | rslaq-sim.cc | confirmado |
| Frequência | 3.55 GHz (n78) | rslaq-sim.cc | confirmado |
| Numerologia | 1 (SCS 30 kHz) | rslaq-sim.cc | confirmado |
| RBGs totais | 16 (266 RBs / 16 RBs por RBG) | nr-gnb-mac.cc | confirmado |
| Potência TX | 43 dBm | run_all_scenarios.sh | confirmado |
| Mobilidade | Estacionário (sem mobilidade) | rslaq-sim.cc | confirmado |
| Deadline URLLC | 10 ms (alvo de latência) | scoring.py:91 | confirmado |
| Confiabilidade exigida URLLC | Não definida explicitamente | — | **AUSENTE** |
| Throughput mínimo eMBB | Não definido como restrição | — | **AUSENTE** |
| Requisitos mMTC | SLA mínimo 1 Mbps (trivialmente baixo) | metadata.json | confirmado |
| Número de seeds | 3 (1, 2, 3) | run_all_scenarios.sh | confirmado |
| Orçamento de avaliações | 300 por par (GA=72, PSO=72, SA=72, híbrida=84) | run_rslaq_metaheuristics.py | confirmado |
| Critério de parada | Iterações fixas (12) | run_rslaq_metaheuristics.py | confirmado |
| Hardware | PC local (Linux 6.17, 15 GB RAM) | uptime, free | confirmado |
| Tempo por avaliação | 47–160s (depende do cenário) | mtime de sidecars | confirmado |

---

## 3. Métricas Realmente Disponíveis

As seguintes métricas são calculadas e estão presentes no `summary.csv`:

**Por slice (eMBB, URLLC, MTC)**:
- throughput_mbps_mean, throughput_mbps_p50, throughput_mbps_p95
- delay_ms_mean, delay_ms_p95, delay_ms_p99
- jitter_ms_mean
- pdr_pct, plr_pct
- sla_satisfaction_pct
- buffer_bytes_mean, buffer_bytes_p95, buffer_bytes_p99
- budget_utilization_pct_mean, unused_budget_pct_mean
- offered_load_satisfaction_pct
- allocated_rbg_total, rx_bytes_total, tx_bytes_total

**Métricas ausentes mas necessárias** (ver seção 4).

---

## 4. Métricas Críticas Ausentes

| Métrica ausente | Impacto | Como coletar |
|---|---|---|
| **delay_ms_p99.9** | URLLC exige confiabilidade 99.999%; sem p99.9 não se pode validar | Modificar StatsCallback no C++ |
| **Taxa de violação de deadline** | Métrica central de URLLC | Contar pacotes entregues após D_max / total gerado |
| **Confiabilidade dentro do prazo** | R_URLLC = pacotes no prazo / total gerado | Mesmo que acima |
| **Retransmissões HARQ** | Afeta latência e confiabilidade | Instrumentar HARQ no C++ |
| **Pacotes gerados vs. recebidos** | Plr=0 universalmente é inconsistente com PDR<100 | Verificar definição de plr_pct |
| **throughput por usuário** | eMBB requer granularidade por UE | Já no ue_detail.csv, não agregado |
| **Eficiência espectral** | Métrica padrão 5G | Derivar de throughput/RBs |
| **Atraso de fila ao longo do tempo** | Diagnóstico de bufferbloat | timeseries.csv existe mas não agregado |

---

## 5. Auditoria da Função Objetivo

### Fórmula exata (scoring.py:58–110)

```
F(w) = 100 × (0.35·mean_sla + 0.20·min_sla + 0.20·mean_offered + 0.15·mean_pdr + 0.10·mean_util)
     − 25·P_URLLC − 40·P_MTC − 100·P_eMBB
```

**Termos positivos** (recompensa, máximo 100):
- 0.35 × mean_sla: satisfação média de SLA dos 3 slices, normalizada [0,1]
- 0.20 × min_sla: pior slice (proxy de justiça max-min)
- 0.20 × mean_offered: carga oferecida atendida
- 0.15 × mean_pdr: packet delivery ratio médio
- 0.10 × mean_util: utilização do orçamento de RBGs

**Termos de penalidade**:
- P_URLLC = clamp((delay_ms_mean − 10) / 20, 0, 1) → até −25 pts (satura em 30ms)
- P_MTC = max(0, 1 − sla[MTC]) + 0.5 se thr_MTC ≈ 0 → até −60 pts
- P_eMBB = 0.05 × clamp(buffer / 5e6, 0, 1) → até −5 pts

### Problemas identificados

| Problema | Severidade | Detalhe |
|---|---|---|
| **Pesos ad-hoc não justificados** | Crítica | 0.35/0.20/0.20/0.15/0.10 não derivados de AHP ou método multi-critério |
| **Penalidade MTC domina** | Crítica | P_MTC até −60 vs recompensa máx +100; na prática o objetivo é "não matar MTC" |
| **SLA não captura p99 de latência** | Crítica | sla_satisfaction=100% quando p99=1500ms — paradoxo |
| **URLLC tratado como termo flexível** | Alta | Deveria ser restrição rígida (D99 ≤ D_max), não penalidade suave |
| **Sem normalização dimensional** | Alta | Throughput (Mbps), delay (ms), PDR (%) em escalas diferentes |
| **Favorece eMBB implicitamente** | Alta | eMBB tem mais volume → mean_throughput dominado por eMBB |
| **Maximizar utilização é questionável** | Média | Premia esgotar espectro, não entregar QoS |

### Caso patológico confirmado: sa_eval_0021

```
cenário: congestion, seed: 2, método: sa, eval: 0021
pesos: [0.8162, 0.1799, 0.0039]
score: 85.28 (MAIOR score de SA em congestion/seed2)
eMBB: thr=158.9, PDR=86.7%, SLA=100%
MTC:  thr=1.5, PDR=2.4%, SLA=100%, delay=4479ms
URLLC: thr=5.1, PDR=80%, delay=3.4ms
```

**A função objetivo recompensa uma configuração que mata MTC** (PDR=2.4%) porque `sla_satisfaction=100%` mesmo com PDR<10%. Isso prova que a métrica `sla_satisfaction_pct` **não mede o que deveria**.

---

## 6. Resultados por Slice

### 6.1 eMBB
- **Throughput**: varia de 17–71 Mbps (normal) a 30–159 Mbps (congestion)
- **PDR sistematicamente baixo**: 16–100%; em congestion, tipicamente 20–40%
- **Buffer congestionado**: 2.0M–2.8M bytes em modo slice; 0 em modo puro
- **SLA**: bimodal — 100% na seed 2, 34–70% nas seeds 1/3
- **Interpretação**: eMBB sofre de bufferbloat severo sob slice-aware; puros não acumulam buffer

### 6.2 URLLC
- **delay_ms_mean**: 3.4–51.4 ms (violando 10ms em vários casos)
- **delay_ms_p99**: **21–1663 ms** — violação massiva do SLA de 10ms
- **61% das células têm p99 > 10ms**
- **PDR**: 43–100%; bimodal por seed
- **PLR**: universalmente 0.0 (inconsistente com PDR<100)
- **SLA**: paradoxalmente 100% mesmo com p99>1000ms
- **Starvation**: nenhum caso (throughput sempre >0.6 Mbps)

### 6.3 MTC
- **Throughput**: 0.0–45.7 Mbps; starvation em `congestion/pure_bcqi/seed1` (thr=0)
- **PDR**: 0–83%; patológico em `sa_eval_0021` (PDR=2.4%)
- **Delay**: até 4479ms (patológico); tipicamente 4–200ms em congestion
- **SLA**: 100% universalmente (mesmo com PDR<10% — paradoxo)
- **Starvation**: 1 caso confirmado (pure_bcqi congestion seed1)

---

## 7. Resultados por Cenário

### Low Traffic
- **Bimodal extremo**: seed2 atinge ~99 (SLA saturado); seeds 1/3 ficam em 43–60
- **Todos os métodos colapsam**: throughput idêntico entre os 14 modos (~39.5 Mbps)
- **Não há separação entre schedulers** — a otimização é irrelevante aqui
- **Veredito**: resultado negativo honesto — em baixa carga, o scheduler não importa

### Normal
- **Variabilidade alta**: CV 23–26% entre seeds
- **Seed2 domina**: ~98 (saturado); seeds 1/3 em 50–73
- **Diferença entre métodos**: 1.67 pts (vs ruído de 18.6 pts) — ruído domina
- **Veredito**: sem separação estatística

### Congestion
- **Mais estável**: CV 7–13% (contraintuitivo: mais tráfego → mais determinismo)
- **Único cenário com separação potencial**: GA 80.0 vs SA 77.8 (spread 2.22)
- **Mas ainda dentro do ruído**: spread/ruído = 0.25 (< 1.0)
- **URLLC p99 sempre > 10ms**: nenhuma meta-heurística cumpre SLA de latência
- **Veredito**: melhor cenário para análise, mas ainda sem significância

---

## 8. Comparação entre Algoritmos

| Cenário | GA | PSO | SA | Híbrida | Spread | Ruído (CV%) | Spread/Ruído |
|---|---:|---:|---:|---:|---:|---:|---:|
| low_traffic | 67.5 | **73.0** | 68.1 | 67.6 | 5.5 | 32.0% | 0.17 |
| normal | 73.8 | **75.5** | 74.8 | 74.2 | 1.7 | 25.0% | 0.07 |
| congestion | **80.0** | 78.9 | 77.8 | 79.9 | 2.2 | 9.0% | 0.25 |

**PSO aparece numericamente como "melhor" em low_traffic e normal; GA em congestion. Mas em TODOS os casos o spread é < 1.0× o ruído — não há diferença estatisticamente detectável.**

### Orçamento computacional
| Método | Evals | Igual? |
|---|---:|---|
| GA | 72 | ✓ |
| PSO | 72 | ✓ |
| SA | 72 | ✓ (equalizado nesta sessão) |
| Híbrida | 84 | +17% (refinamento local SA intrínseco) |

A híbrida tem 17% mais avaliações — pequena inequidade, documentável.

---

## 9. Comparação com Baselines

### Score F(w) — média 3 seeds

| Cenário | RR | PF | BCQI | Melhor Meta | Ganho Meta vs Melhor Base |
|---|---:|---:|---:|---:|---|
| low_traffic | **73.6** | 55.3 | 53.2 | 73.0 (PSO) | −0.8% (perde) |
| normal | 76.3 | 66.2 | **77.1** | 75.5 (PSO) | −2.1% (perde) |
| congestion | 62.7 | 66.2 | 49.1 | **80.0** (GA) | +20.9% (ganha) |

**As meta-heurísticas só ganham em congestion (+21%), e mesmo ali perdem na seed 3 (−10pts). Em low_traffic e normal, RR e BCQI puros superam.**

### Métricas físicas (congestion, média 3 seeds)

| Método | Thr_total | URLLC_p99 | MTC_PDR | Jain | minSLA |
|---|---:|---:|---:|---:|---:|
| pure_rr | 94.4 | 740ms | 65% | 0.72 | 77% |
| pure_pf | 94.5 | 537ms | 61% | 0.72 | 84% |
| pure_bcqi | **130.3** | 6.8ms | 25% | 0.46 | 67% |
| slice_weighted_pf | 76.6 | 161ms | 50% | 0.72 | 80% |
| GA (best) | 99.8 | 107ms | 46% | 0.71 | 100% |
| PSO (best) | 90.4 | 92ms | 39% | 0.64 | 100% |

**Insight crítico**: pure_bcqi tem o melhor URLLC p99 (6.8ms, único que cumpre <10ms!) mas mata MTC (PDR=25%). As meta-heurísticas têm minSLA=100% mas URLLC p99=92–107ms (violam). **Nenhum método cumpre todos os SLAs simultaneamente.**

---

## 10. Análise de Convergência

| Cenário | Método | Evals até 95% do ótimo (média 3 seeds) |
|---|---|---:|
| low_traffic | SA | **8.7** (mais rápido) |
| low_traffic | Híbrida | 28.3 (mais lento) |
| normal | Híbrida | **1.3** (instantâneo) |
| congestion | Híbrida | **7.3** |
| congestion | SA | 9.0 |

**Conclusão**: a convergência é rápida (~10 evals para 95% do ótimo). O orçamento de 300 evals é desperdiçado — a partir de ~30–50 evals não há ganho incremental. Reduzir para 50 evals não mudaria conclusões.

---

## 11. Análise Estatística

### Unidade experimental correta: melhor score por seed (não evals individuais)

**Com n=3 seeds, a análise estatística formal é IMPOSSÍVEL**:
- Wilcoxon pareado exige n≥5
- Friedman exige n≥5
- Não se pode calcular IC 95% significativo
- Não se pode rejeitar hipótese nula

### Declaração obrigatória
> "Com apenas 3 seeds independentes, a potência estatística é muito baixa para sustentar conclusões comparativas. A variabilidade inter-seed (CV 9–32%) excede a diferença inter-método (1.7–5.5 pts) por um fator 4–15. Recomenda-se um mínimo de 20 seeds para análise estatística formal."

---

## 12. Análise Multiobjetivo e Pareto

### Fronteiras de Pareto identificadas (throughput vs Jain)

- **Low Traffic**: degenerada — todos os métodos colapsam (~39.5 Mbps, Jain ~0.375). Não há trade-off.
- **Normal**: fronteira RR→PSO→BCQI (3 pontos não-dominados)
- **Congestion**: fronteira PF→GA→SA→BCQI (4 pontos não-dominados) — única fronteira significativa

### Problema: nenhuma meta-heurística é Pareto-ótima em throughput-vs-p99_URLLC
Em congestion, pure_bcqi domina todas as meta-heurísticas no eixo p99 (6.8ms vs 92–175ms). As meta-heurísticas só "vencem" na métrica composta F(w), não em métricas físicas de SLA.

---

## 13. Auditoria dos Pesos

| Cenário | Seed | Método | eMBB | URLLC | MTC | Soma | Coerente? |
|---|---|---|---:|---:|---:|---:|---|
| congestion | 1 | hybrid | 0.463 | 0.155 | 0.382 | 1.00 | ✓ eMBB dominante |
| congestion | 2 | **sa** | **0.816** | 0.180 | **0.004** | 1.00 | ⚠️ MTC quase zero |
| congestion | 3 | ga | 0.496 | 0.286 | 0.218 | 1.00 | ✓ balanceado |
| low_traffic | 3 | pso | **0.055** | 0.454 | 0.491 | 1.00 | ⚠️ eMBB quase zero |

**Variabilidade dos pesos entre seeds**: CV de 10–104%. Os pesos ótimos são **instáveis** entre seeds do mesmo cenário (exceto eMBB em congestion, CV 4–10%).

---

## 14. Auditoria dos Gráficos

### fig04_tradeoff_pareto.png
- **Eixo Y declarado como "Jain fairness index"** ✓ (corrigido nesta sessão)
- **Fronteira de Pareto**: calculada sobre valores reais (sem jitter) ✓
- **Low Traffic degenerado**: declarado com anotação ✓
- **Problema**: jitter visual infla separação entre meta-heurísticas em low_traffic por ~20× (corrigido para 4% da faixa)
- **Correspondência com dados brutos**: confirmada ✓

### Demais figuras (fig01–03, 05–07)
- Correspondem aos dados brutos ✓
- Boxplots (fig02) misturam evals internas (não resultados finais por seed) — deveria usar apenas o melhor por seed
- Convergência (fig01) usa melhor-acumulado médio — adequado

---

## 15. Adequação para Controle Online / O-RAN

| Método | Controle online? | Otimização periódica? | Análise offline? |
|---|---|---|---|
| GA | ✗ (requer 72 sims) | ✓ (offline, pesos estáticos) | ✓ |
| PSO | ✗ | ✓ | ✓ |
| SA | ✗ | ✓ | ✓ |
| Híbrida | ✗ (84 sims) | ✓ | ✓ |

**Nenhuma meta-heurística é viável para Near-RT RIC** (requer decisão em <10ms; cada simulação dura 47–160s). Adequadas apenas para Non-RT RIC (otimização offline periódica de pesos estáticos).

---

## 16. Inconsistências Críticas

1. **SLA=100% com PDR<10%** (sa_eval_0021 e outros) — a métrica sla_satisfaction_pct não mede entrega de pacotes.
2. **delay_ms_mean > delay_ms_p95** em 20/89 células — anomalia de amostragem (percentis por janela, média por pacote).
3. **plr_pct=0 universalmente** enquanto PDR varia de 2.4–100% — PLR+PDR deveria somar ~100.
4. **Função objetivo recompensa starvation de MTC** (sa_eval_0021: score 85.28 com MTC PDR=2.4%).
5. **URLLC tratado como penalidade suave, não restrição rígida** — permite violação de 10ms sem consequência severa.

---

## 17. Inconsistências Moderadas

1. **Híbrida tem 84 evals vs 72 dos demais** (+17%) — inequidade pequena mas não declarada.
2. **sim_time=5s** não captura regime permanente/transições.
3. **SLA MTC mínimo = 1 Mbps** (trivialmente baixo, sempre satisfeito).
4. **Sem mobilidade** (UEs estacionários) — irrealista para eMBB.
5. **Seed determina o regime** (seed2 = saturado ~99; seeds 1/3 = congestionado ~50) — o gerador de carga é hipersensível à seed.

---

## 18. Melhorias Recomendadas

### Prioridade CRÍTICA
1. **Corrigir sla_satisfaction_pct**: deve refletir PDR e latência p99, não apenas um limiar arbitrário.
2. **Reformular função objetivo**: URLLC como restrição rígida (D99 ≤ 10ms), não penalidade suave.
3. **Reconciliar PLR=0 com PDR<100**: a definição de plr_pct está errada ou incompleta.

### Prioridade ALTA
4. **Aumentar seeds para ≥10** (idealmente 30) para análise estatística.
5. **Reportar métricas físicas por slice**, não apenas score composto.
6. **Adicionar Jain baseado em satisfação de SLA** (não throughput bruto).
7. **Documentar que throughput das meta-heurísticas usa eval real**, não proxy.

### Prioridade MÉDIA
8. **Corrigir delay percentis** no C++ (amostragem por pacote, não por janela).
9. **Aumentar sim_time para ≥30s** para regime permanente.
10. **Calibrar SLA MTC** (>1 Mbps, significativo).

---

## 19. Experimentos Adicionais Necessários

1. CDF/CCDF da latência URLLC por método
2. Taxa de violação de deadline (pacotes após 10ms / total gerado)
3. Confiabilidade dentro do prazo (R_URLLC)
4. Throughput vs p99_URLLC (Pareto em métricas físicas)
5. Resultados finais por seed (não agregados em média)
6. Distribuição dos pesos por seed (boxplot/violin)
7. Utilização de RBGs por slice ao longo do tempo
8. Atraso de fila temporal (timeseries)

---

## 20. Classificação Final

| Critério | Nota (0–5) | Justificativa |
|---|---:|---|
| Qualidade da função objetivo | **1** | Ad-hoc; recompensa starvation; SLA inconsistente |
| Avaliação de eMBB | **3** | Throughput e buffer medidos, mas PDR sistematicamente baixo |
| Avaliação de URLLC | **1** | p99 viola 10ms em 61%; sem deadline violation rate; sem confiabilidade |
| Avaliação de mMTC | **2** | PDR medido mas starvation não protegido; SLA=100% com PDR=2.4% |
| Validade estatística | **1** | n=3; CV domina sinal; impossível declarar significância |
| Justiça da comparação | **3** | Orçamento equalizado (72/72/72/84); mas híbrida +17% |
| Qualidade dos baselines | **4** | RR/PF/BCQI + AQPS + Meta-Risk + heurísticas adaptativas |
| Robustez entre seeds | **1** | CV 9–32%; seed define regime mais que método |
| Eficiência computacional | **3** | Converge em ~10 evals; mas orçamento 300 é desperdiçado |
| Reprodutibilidade | **4** | Código versionado; seeds fixas; checkpoint robusto |
| Adequação para O-RAN | **2** | Apenas Non-RT RIC offline; sem integração xApp/rIC |
| Evidência de cumprimento de SLA | **1** | Nenhum método cumpre todos os SLAs simultaneamente |

**Média geral: 2.2/5 — insuficiente para publicação sem correções maiores.**

---

## 21. Conclusão Objetiva

### A otimização é eficiente?
**Não é possível concluir com os dados apresentados.** O ruído entre seeds (CV 9–32%) domina a diferença entre métodos (1.7–5.5 pts) por um fator 4–15. A função objetivo recompensa configurações que causam starvation.

### Qual algoritmo é o melhor em cada cenário?
Numericamente: PSO em low_traffic/normal, GA em congestion. Mas **nenhuma diferença é estatisticamente significativa** com n=3.

### O ganho é estatisticamente confiável?
**Não.** Com n=3, qualquer teste estatístico é inválido. A variabilidade inter-seed excede a diferença inter-método.

### O URLLC cumpre latência e confiabilidade?
**Não.** 61% das células violam p99 > 10ms (até 1663ms). Nenhuma meta-heurística mantém p99 ≤ 10ms em congestion. Confiabilidade dentro do prazo não foi calculada.

### O eMBB obtém ganho sem prejudicar os demais slices?
**Não.** O melhor eMBB (pure_bcqi: 130 Mbps) causa starvation de MTC (PDR=25%). As meta-heurísticas melhoram minSLA mas violam URLLC p99.

### O mMTC sofre starvation?
**Sim.** 1 caso confirmado (pure_bcqi congestion seed1: thr=0) e 1 caso patológico (sa_eval_0021: PDR=2.4%, delay=4479ms) que recebeu o SCORE MAIS ALTO.

### A função objetivo é adequada?
**Não.** Recompensa starvation, não captura p99 de latência, SLA inconsistente com PDR. Precisa de reformulação completa (restrições rígidas de SLA).

### A comparação entre algoritmos é justa?
**Parcialmente.** Orçamento equalizado (72 evals), mas híbrida tem +17%. Seeds pareadas. Função objetivo comum. Mas n=3 impede conclusões.

### Os resultados são suficientes para publicação científica?
**Não no estado atual.** Requer: (1) corrigir função objetivo, (2) ≥10 seeds, (3) métricas URLLC corretas (deadline violation, confiabilidade), (4) reconciliar PLR/PDR/SLA.

### A solução está pronta para uso online?
**Não.** Adequada apenas para otimização offline (Non-RT RIC). Cada avaliação dura 47–160s — inviável para Near-RT RIC (<10ms).

---

## SAÍDAS ADICIONAIS

### A. Tabela Consolidada (congestion, média 3 seeds)

| Método | Score | Thr_total | URLLC_p99 | Violação p99>10ms | Confiabilidade | Jain | minSLA | Situação SLA |
|---|---:|---:|---:|---|---|---:|---:|---|
| pure_rr | 62.7 | 94.4 | 740ms | SIM | não calculada | 0.72 | 77% | Viola URLLC |
| pure_pf | 66.2 | 94.5 | 537ms | SIM | não calculada | 0.72 | 84% | Viola URLLC |
| pure_bcqi | 49.1 | 130.3 | 6.8ms | NÃO | não calculada | 0.46 | 67% | Starvation MTC |
| GA | 80.0 | 99.8 | 107ms | SIM | não calculada | 0.71 | 100% | Viola URLLC |
| PSO | 78.9 | 90.4 | 92ms | SIM | não calculada | 0.64 | 100% | Viola URLLC |
| SA | 77.8 | 117.8 | 132ms | SIM | não calculada | 0.67 | 100% | Viola URLLC + MTC patológico |
| Híbrida | 79.9 | 98.0 | 110ms | SIM | não calculada | 0.67 | 100% | Viola URLLC |

### B. Lista Priorizada de Correções

**CRÍTICA**:
1. Corrigir sla_satisfaction_pct para refletir PDR + latência p99
2. Reformular F(w) com URLLC como restrição rígida (D99 ≤ 10ms)
3. Reconciliar PLR=0 com PDR<100

**ALTA**:
4. Aumentar para ≥10 seeds
5. Reportar métricas físicas por slice (não só score)
6. Adicionar Jain de satisfação de SLA

**MÉDIA**:
7. Corrigir percentis de delay no C++
8. Aumentar sim_time
9. Calibrar SLA MTC

**OPCIONAL**:
10. Adicionar eficiência espectral
11. Integrar RSLAQ DDQN como baseline DRL

### C. Gráficos Necessários

1. CDF/CCDF da latência URLLC
2. P99 por algoritmo (barra)
3. Violações de deadline (% pacotes > 10ms)
4. Confiabilidade dentro do prazo
5. Throughput vs p99_URLLC (Pareto físico)
6. Convergência com orçamento comum
7. Resultados finais por seed (dispersão)
8. Distribuição dos pesos por seed (violin)
9. Utilização de RBGs por slice temporal
10. Atraso de fila ao longo do tempo

### D. Proposta de Função Objetivo Corrigida

```
maximizar:
    U(w) = w1·S_eMBB + w2·S_URLLC + w3·S_MTC

sujeito a:
    D99_URLLC ≤ 10 ms                    (restrição rígida)
    PDR_URLLC ≥ 99.999%                  (confiabilidade URLLC)
    throughput_eMBB ≥ T_min_eMBB         (SLA eMBB)
    PDR_MTC ≥ P_min_MTC                  (evitar starvation)
    Σ w_i = 1, w_i ∈ [w_min, w_max]     (simplexo com limites)

onde:
    S_i = min(desempenho_i / requisito_i, 1)   (satisfação normalizada [0,1])
    S_URLLC = min(S_delay, S_PDR, S_conf)      (mínimo das métricas URLLC)
```

### E. Checklist de Reprodutibilidade

- [x] Seeds: 1, 2, 3 (META_RANDOM_SEED=2026)
- [x] Configuração: SIM_TIME=5, APP_START=0.4, PERIOD_MS=10, TX_POWER=43
- [x] Versões: ns-3 5G-LENA, Python 3.12
- [x] Comandos: `bash examples/resume_campaign.sh`
- [x] Arquivos de entrada: run_all_scenarios.sh, run_rslaq_metaheuristics.py
- [x] Arquivos de saída: candidate.json, summary.csv, best_candidate.json
- [x] Critérios de parada: 12 iterações × 6 população
- [x] Hardware: Linux 6.17, 15 GB RAM, 19 cores
- [x] Tempo de execução: ~47–160s por eval; ~4–5h por cenário
