# D3_execution_declaration_v2.md — declaração de conformidade da D3 (pós-D2-v2, conciliação 13/11)

**Data:** 13/11/2026 · **Base:** D2v2_report.md + conciliação do orientador (4 itens) · **Vedados:** aprendizado quantílico, SLAs, R19, espaço de ações comum. PILOT/MAIN bloqueados.

## 1. Conciliação do estado S3_lam6_s2102_t1420 (item 1) — resolvida

**Não houve inconsistência de pipeline: a diferença de classificação é o mecanismo da Opção B (online-frozen × offline completo).**

| | D1-v9, julgamento ONLINE-frozen (seleção/estratificação) | D2, julgamento OFFLINE (adjudicação) |
|---|---|---|
| evidência R19 | eMBB COMPLETE · **URLLC NO_ACTIVE_DEMAND** · mMTC COMPLETE | 3×COMPLETE (pool completo) |
| margens | eMBB +0,240 · URLLC **None (vacuous)** · mMTC +0,053 | eMBB 4,34 Mbps · URLLC p95 **10,15 ms** · mMTC PDR 1,00 |
| julgamento | sla_ok=[T,T,T] ⇒ **joint_frozen_ok=True** ⇒ estrato S3 (proj ∧ sem violação) | p95 10,15 > 10,0 ⇒ **STRICT_FAIL** (violação mínima, 1,5%) |

Na época da decisão, o pool URLLC da janela 1420 ainda não tinha entregas flushadas (censura lag=1 do callback — Emenda 3 ADENDO): o controlador **não viu** a violação e o estrator a registrou como "sem violação". Offline (pool completo), a violação existe e é mínima. **Desvio sistêmico da estratificação online-frozen, medido: 2/6 estados S3 e 0/3 S4 são offline-violados** (9 STRICT_FAIL totais: S1:4, S2:3, S3:2). Reportado como característica do desenho (a seleção usa o que o controlador viu; a avaliação usa o que aconteceu), não corrigido retroativamente (vedação).

**KPIs por ramo (t1420):** a_B0 (22,30,1): eMBB 4,34 · p95 10,15 · PDR 1,00 ⇒ STRICT_FAIL. Recuperadoras: alt2 (26,26,1): 5,32/8,10/1,00; alt10 (31,2,5): 6,48/8,10; alt11 (32,4,0): 6,54/8,10; alt12 equal (18,17,18): 3,73/8,10 — todas STRICT_PASS. As alts que mantêm a_u≥28 mantêm p95 10,15 FAIL; reduzir a_u para ≤26 derruba o p95 para 8,10 e transfere PRBs ao eMBB — recuperação por realocação URLLC→eMBB consistente com o mecanismo físico.

## 2. Estado t2064 re-validado com simTime integral (item 2) — paridade restaurada

13 ramos re-executados com simTime=300 (`d2_v2_t2064_full/`, dirs separados; os demais 377 replays NÃO foram repetidos):
- **A/A bit-exact: TRUE** (0 drift; `d2v2_t2064full_branches.csv`).
- STRICT: **UNKNOWN em todos os 13 ramos** (mMTC sem demanda na janela — o estado nunca contou para STRICT).
- **e∧u: a_B0 EU_FAIL (eMBB 2,93; p95 11,1) e 1 alt EU_PASS: alt3 (22,29,2): eMBB 4,65 · p95 8,10** — a recuperação e∧u do estado fica CONFIRMADA COM PARIDADE (a rodada com simTime truncado mostrava 2 alts EU_PASS; com paridade, 1). Registrada como evidência e∧u com A/A válida; STRICT-UNKNOWN por prevalência.

## 3. Controle `equal` (item 3) — registrado

2/4 recuperações STRICT foram obtidas com a alocação igualitária (18,17,18) [S2_lam6_s2101_t2824, S3_lam6_s2102_t1420]. **Registro como controle importante:** em 2 dos 4 estados, o B0 está longe do equilíbrio de alocação e uma redistribuição estática simples recupera — a seleção conjunta precisará demonstrar vantagem ALÉM do equal. Na D3: SQR-HET (B0) permanece o baseline principal; `equal` entra como **segundo baseline de referência na análise complementar** (o seletor é avaliado contra ambos; a contribuição candidata permanece Joint Action Selection).

## 4. Confirmações de preparação da D3 (item 4)

1. **Paridade A/A dos 4 estados recuperáveis STRICT:** t2886 (λ5s2103), t1608 (λ6s2102), t2824 (λ6s2101), t1420 (λ6s2102) — `aa_ok=True`, `prefix_ok=True`, `quotas_ok=True` em todos os 4 (tabela `D2v2_states.csv`). O único estado sem A/A (t2064) foi re-validado (§2) e é e∧u-only.
2. **Nenhum resultado da D2 seleciona/ajusta o controlador:** os modelos quantílicos, scaler, prior e a regra lexicográfica do JointSelector permanecem os congelados desde a D0-B (hashes verificados na D3); a D2 serve apenas como evidência de existência da lacuna — nunca como sinal de treino/ajuste.
3. **Independência dos estados:** conjunto de teste da D3 exclui os 30 estados da D2 **e a vizinhança ±5 janelas** em cada (run, janela); verificação de sobreposição de trajetórias/dados de treino executada na preparação da D3 (cuidado 2 vigente: excluir janelas não garante independência estatística entre runs — os 9 runs têm trajetórias independentes por seed, e a D3 avalia os MESMOS 9 runs com rótulos offline; a independência exigida é por estado, mantida pela exclusão de vizinhança).
4. **Protocolo congelado:** STRICT_ALL3 primário (cobertura/UNKNOWN reportados), e∧u complementar, mMTC mantido; rótulos offline completos; o seletor não usa o rótulo do replay na decisão; julgamento frozen × STRICT × UNKNOWN reportados separadamente.
5. **Métricas da D3:** por estado do conjunto de teste: label do B0 vs label do seletor; recuperações STRICT/e∧u por estrato, λ, seed, com/sem projeção; **controle equal** lado a lado; não-extrapolação de proporções; NOT_RECOVERED ≠ impossibilidade.

## 5. Parecer técnico (conciliação concluída): **GO para a D3**

A conciliação dos 4 itens fechou sem inconsistência remanescente: (1) t1420 explicado pelo mecanismo da Opção B (violação offline invisível online — 1,5% de p95); (2) t2064 re-validado com paridade bit-exact (e∧u confirmado, STRICT-UNKNOWN por prevalência); (3) controle equal registrado como análise complementar obrigatória da D3; (4) paridade A/A confirmada nos 4 estados STRICT-recuperáveis, sem vazamento D2→controlador, estados independentes por exclusão de vizinhança. A hipótese permanece: investigar se a seleção conjunta aproveita oportunidades admissíveis que o SQR-HET deixa de explorar — agora com o controle de que a vantagem deve superar também uma redistribuição estática simples.
