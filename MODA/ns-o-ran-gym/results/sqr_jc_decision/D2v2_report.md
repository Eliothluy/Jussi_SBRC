# D2v2_report.md — replays causais D2-v2 (população V9, Emenda 4): oportunidades confirmadas e avaliação da lacuna

**Data:** 13/11/2026 · **Execução:** 390/390 replays concluídos (3 shards 19:43–08:41; GO do orientador) · **Fontes:** `D2v2_results.csv` (por ramo), `D2v2_states.csv` (por estado), adjudicação offline com `window_kpis`/`_sla_oks` congelados · **Vedados intactos:** aprendizado, SLAs, R19, espaço de ações. D3/PILOT/MAIN bloqueados.

## 1. Execução e integridade

- **390/390 replays** (30 estados × 13 ramos: a_B0 + 12 alternativas). Ordem do CSV, idempotência `.done`, sem ordenamento pelo oracle (ajuste 5).
- **Incidente pré-execução** (documentado em `INCIDENTE_d2v2_base_seq_holdwindows.md`): a base seq não cobria janelas-relógio além da última decisão logada (runs com holds) — corrigido (quota executada por janela-relógio + backward-fill; clamp defensivo). 0 falhas pós-correção; ~78 replays pré-correção re-executados; os 26 concluídos antes foram mantidos (furos sem slot não têm efeito físico no KPM).
- **Paridade A/A (validada ANTES da interpretação — item 2 do ofício):** **29/30 estados bit-exact** (quotas vs run B0 + prefixo entre ramos + quotas do controller). **1 estado (S1_lam5_s2101_t2064) com desvio determinístico** a partir da janela 1809 (533/6.679 linhas KPM até t·100; colunas buffer/queue/delaySum; ΔRX eMBB +0,25%; delays URLLC mediana −17 ms; TX idênticos). **Causa isolada por re-execução: borda de fim de sim** — com simTime=300 o A/A é bit-exact (0 drift); com simTime truncado (208 s) o desvio é reproduzível byte-a-byte. O desvio é COMUM aos 13 ramos do estado (prefixo entre ramos bit-exact ⇒ comparação interna válida); o rótulo do estado (NOT_RECOVERED) não muda; registrado como inconsistência de borda sem alterar métricas (item 6).
- Prefixo KPM bit-idêntico entre ramos: **30/30 estados**.

## 2. Resultado principal (STRICT_ALL3 primário)

| label | n | significado |
|---|---|---|
| **RECOVERABLE_STRICT** | **4** | a_B0 viola STRICT e ≥1 alternativa admissível recupera |
| B0_ALREADY_PASS | 3 | a_B0 já passa STRICT (λ6s2103: t1961/t1379/t2004; 6/9/10 alts passam também) |
| NOT_RECOVERED_IN_TESTED_SET | 23 | nenhuma das 12 alts testadas recupera — **não** implica impossibilidade (item 5) |

**Universo de violação STRICT observável:** 9 estados com a_B0 STRICT_FAIL (6 λ6, 3 λ5) — **4/9 (44%) são recuperáveis** no conjunto testado. Os 23 NOT_RECOVERED = 5 violados não recuperados + 18 UNKNOWN (sem 3×demanda simultânea — prevalência conhecida da D1).

**STRICT_PASS por ramo:** nos 4 estados recuperáveis, alts passantes: 1, 1, 3, 4 de 12 — **densidade baixa**: a recuperação exige alocações específicas, não qualquer redistribuição.

### Recuperações (a_B0 → alts STRICT_PASS, mecanismo)

| estado | a_B0 | alts passantes (ação, mecanismo) |
|---|---|---|
| S1_lam5_s2103_t2886 | 37-11-5 | alt10: 34-6-11 [rnd] |
| S1_lam6_s2102_t1608 | 15-37-1 | alt9: 33-16-4 [rnd] |
| S2_lam6_s2101_t2824 | 16-3-0 | alt5: 18-3-30; alt8: 18-13-2; alt12: 18-17-18 [equal] |
| S3_lam6_s2102_t1420 | 22-30-1 | alt2: 26-26-1; alt10: 31-2-5; alt11: 32-4-0; alt12: equal |

**Padrão físico:** em todos, o slice violado ganha PRBs às custas do slice sobreprovisionado (URLLC→eMBB em λ6s2102; URLLC→mMTC/eq em λ6s2101; eMBB→mMTC em λ5s2103) — a ação do B0 deixa margem num slice que a janela não usa e o slice violado carece. A alternativa `equal` (18-17-18) recupera 2 dos 4 estados — o B0 está LONGE do equilíbrio nessas janelas, não próximo dele.

### Com/sem projeção (S1+S3 com, S2+S4 sem)

| | RECOVERABLE | B0_ALREADY | NOT_RECOVERED |
|---|---|---|---|
| com projeção (18) | **3** | 1 | 14 |
| sem projeção (12) | **1** | 2 | 9 |

### Por seed

| seed | RECOVERABLE | B0_ALREADY | NOT_RECOVERED |
|---|---|---|---|
| 2101 | 1 | 0 | 10 |
| 2102 | 2 | 0 | 8 |
| 2103 | 1 | 3 | 5 |

## 3. Complementar eMBB∧URLLC e UNKNOWN (pré-registrados)

- **e∧u:** 5 estados com a_B0 EU_FAIL têm ≥1 alt EU_PASS — os 4 RECOVERABLE_STRICT + S1_lam5_s2101_t2064 (o estado com borda de fim-de-sim; STRICT UNKNOWN por falta de demanda mMTC na janela). n∧u maior, conclusão consistente.
- **UNKNOWN:** URLLC sem demanda em 208/390 ramos (53%); eMBB 13/390; mMTC 91/390. A cobertura STRICT da amostra é limitada pela prevalência (12/30 estados com 3×demanda: S1:4/S2:4/S3:3/S4:1) — os 18 UNKNOWN são reportados, nunca contados como sucesso nem como falha.
- Violados a_B0 por vslice: e 6, none 2, e+m 1.

## 4. Avaliação fundamentada: a lacuna que motivaria Joint Selection existe?

**Sim — existe, é replicada nas 3 seeds, mas é estreita e seletiva.**

1. **Existência:** 4/9 estados com violação STRICT observável têm ≥1 alocação admissível (mesmo espaço Σ≤53, mesmos SLAs) que recupera a janela — o SQR-HET independente não a escolheu. Em e∧u o achado é 5 estados. Isso é a evidência direta de que alocações admissíveis melhores existem no espaço comum e são deixadas de aproveitar.
2. **Seletividade:** densidade 1–4/12; em 2 dos 4 estados só 1 alternativa específica recupera. Redistribuições arbitrárias não recuperam — capturar a oportunidade exige informação sobre a margem de cada slice (exatamente o que um seletor conjunto com previsões quantílicas por slice tem e um controlador por-slice não usa).
3. **Onde:** 100% das recuperações em λ5/λ6 (carga alta); 3/4 em estados com projeção (escassez força trade-offs inter-slice) — coerente com a motivação do Joint Action Selection. Em λ2 nenhuma recuperação (janelas violadas não tinham demanda URLLC simultânea; oportunidades não se materializam em carga baixa).
4. **Magnitude honesta:** a amostra cobre 30/15.124 decisões B0 c/ outcome (0,2%) e 12/27.720 ações por estado — as 4 recuperações são **cota inferior**; as proporções NÃO são extrapoláveis para a população (ajuste 6). O B0_ALREADY_PASS (3, todos λ6s2103) mostra também que parte das janelas STRICT-viáveis já é atendida.
5. **Implicação para a D3:** o sinal é suficiente para justificar a pergunta D3 (o seletor conjunto lexicográfico, congelado, captura essas alocações?) e o desenho de avaliação pareada por estado — mas o n de recuperações confirmadas (4) e a cobertura UNKNOWN motivam reportar STRICT e e∧u lado a lado na D3, conforme pré-registrado.

## 5. Registro de bordas e denominadores (item 6)

- 1/30 estados com desvio determinístico de fim-de-sim (detalhe no §1); recomendação registrada: para estados com win > ~2.400, considerar simTime integral em futuras campanhas (custo ~2×; aqui o rótulo não muda, nada foi re-executado).
- A/A formal: `aa_ok` por estado em `D2v2_states.csv` (29 True, 1 False documentado).
- Denominadores: todos os labels reportam a base (30 estados amostrados; 13 ramos; universo populacional citado onde aplicável); UNKNOWN reportado por slice; nenhuma métrica alterada silenciosamente.
- Artefatos preservados: `d2_v2_replays/` (390 runs), logs de shard (`run_d2_*_v9.nohup`), incidente (`INCIDENTE_d2v2_base_seq_holdwindows.md`), amostra/hash (`D2_sampling_FINAL_V9.csv`, manifest), adjudicador (`d2_analysis_v9.py`).

**Recomendação: prosseguir para a D3** (offline, pareada, modelos/estados congelados equivalentes, exclusão dos 30 estados ±5 janelas), avaliando STRICT primário + e∧u complementar. D3/PILOT/MAIN permanecem bloqueados até a avaliação deste relatório.

---

# ADENDO 13/11 — conciliação metodológica (ofício pré-D3)

## C1. Estado S3_lam6_s2102_t1420 — por que "all-OK frozen" na seleção e RECOVERABLE_STRICT na avaliação

**Mecanismo da Opção B, sem inconsistência de pipeline.** Seleção/estratificação usaram o julgamento ONLINE-frozen do runtime: na época da decisão, o pool URLLC da janela 1420 ainda não tinha entregas flushadas (censura lag=1) ⇒ `sla_evidence=[COMPLETE, NO_ACTIVE_DEMAND, COMPLETE]`, margens eMBB +0,240 / URLLC **None (vacuous)** / mMTC +0,053 ⇒ `joint_frozen_ok=True` ⇒ estrato S3. A adjudicação D2 usa o pool OFFLINE completo: **p95 10,15 ms > 10,0** — violação mínima (1,5%) ⇒ STRICT_FAIL com 3×COMPLETE. Desvio sistêmico da estratificação online-frozen, medido: **2/6 estados S3 e 0/3 S4 são offline-violados** (9 STRICT_FAIL: S1:4, S2:3, S3:2, S4:0). Característica do desenho (seleção usa o que o controlador viu; avaliação usa o que aconteceu), documentada — não corrigida retroativamente (vedação).

KPIs por ramo: a_B0 (22,30,1) 4,34/10,15/1,00 FAIL; recuperadoras — alt2 (26,26,1) 5,32/**8,10**; alt10 (31,2,5) 6,48/8,10; alt11 (32,4,0) 6,54/8,10; alt12 equal 3,73/8,10 — todas PASS. Alts com a_u≥28 mantêm p95 10,15; reduzir a_u≤26 derruba o p95 e transfere PRBs ao eMBB: recuperação por realocação URLLC→eMBB consistente com o mecanismo físico.

## C2. Estado t2064 re-validado com simTime=300 (item 2 do ofício)

13 ramos re-executados com simTime integral (`d2_v2_t2064_full/`; demais 377 replays não repetidos): **A/A bit-exact TRUE** (0 drift). STRICT: UNKNOWN em 13/13 (mMTC sem demanda — o estado nunca contou para STRICT). **e∧u: a_B0 EU_FAIL (eMBB 2,93; p95 11,1) e 1 alt EU_PASS: alt3 (22,29,2): 4,65/8,10** — recuperação e∧u CONFIRMADA COM PARIDADE (a rodada truncada mostrava 2 alts; com paridade, 1). Artefato: `d2v2_t2064full_branches.csv`. O resultado e∧u do estado passa a ser evidência causal com A/A válida; STRICT-UNKNOWN por prevalência.

## C3. Controle `equal` (item 3)

2/4 recuperações STRICT via (18,17,18) [t2824, t1420]. **Controle importante registrado:** em 2 estados o B0 está longe do equilíbrio e uma redistribuição estática simples recupera. Na D3: SQR-HET baseline principal; `equal` como **segundo baseline de referência** na análise complementar — a seleção conjunta deverá demonstrar vantagem além do equal.

## C4. Paridade A/A dos 4 estados STRICT-recuperáveis (item 4)

`aa_ok=True, prefix_ok=True, quotas_ok=True` em t2886 (λ5s2103), t1608 (λ6s2102), t2824 (λ6s2101), t1420 (λ6s2102). Resultados da D2 NÃO selecionam/ajustam o controlador (modelos/scaler/prior/regra lexicográfica congelados desde D0-B; hashes verificados na D3); conjunto de teste D3 exclui os 30 estados ±5 janelas; STRICT primário + e∧u complementar; seletor não usa rótulo do replay. Declaração completa: [`D3_execution_declaration_v2.md`](../../../../docs/sqr/jc_decision/D3_execution_declaration_v2.md).

**Parecer pós-conciliação: GO para a D3** (declaração v2 §§4–5). D3/PILOT/MAIN permanecem bloqueados até a avaliação do orientador.
