# DELAYED_FEEDBACK_investigacao.md — verificação técnica: updates URLLC com feedback tardio

**Data:** 12/11/2026 (noite) · **Solicitante:** orientador (após errata D1-v8) · **Escopo:** verificação curta, SEM novas campanhas — 1 teste isolado de 90 s com flag experimental; canônica D1-v8 intacta; D2-v2 permanece pausada.

---

## 1. Delayed feedback — o mecanismo atual e por que 0 updates

O `sqr_run.py` pergunta o outcome da janela governada W **exatamente uma vez** (na iteração de W+1). Com o lag=1 do ns-3 (KPM escrito antes do flush de delays, no mesmo callback), `stats(W)=None` naquela única pergunta ⇒ R19 descarta ⇒ 0 updates URLLC. **Não existe fila de pendentes** — a observação válida que se torna disponível 1–3 janelas depois nunca é utilizada. Isso é uma limitação do RUNTIME, não do contrato: é observabilidade parcial *imposta pelo desenho da pergunta*, não impossibilidade de usar observações posteriores válidas.

## 2. Viabilidade do rótulo online (sem dados futuros, sem assumir p95 completo por entrega parcial)

Chave: **`tx_kpm(W)` é informação online** — fecha com o KPM de W, disponível na decisão de W+1 (o runtime já o absorve via `HetRuntime.absorb`). E `delivered(W)` cresce no stream de delays (append-only, incremental). Logo o runtime pode testar ONLINE:

> **pool de W fechado ⇔ delivered(W) == tx_kpm(W)**

- Fechado ⇒ o p95/miss computados do pool são **completos** (idênticos ao offline) — não um p95 parcial.
- Nunca fecha (perda real RLC-UM) ⇒ permanece OPEN ⇒ descartado como vacuous por R19 (sem-RX nunca é miss nem sucesso implícito). **Nenhum timeout ad hoc** é preciso para validade; MAX_AGE=30 (3 s) é só limite de fila.
- **Não se assume** que "algumas recepções tornam o p95 conhecido" — o oposto: só atualiza quando o pool fecha por identidade observável.

**Quantificação nos 9 runs canônicos D1-v8** (fonte única, sem re-simular): das 11.335 janelas com demanda URLLC, **6.777 (59,8%) fecham o pool**, com atraso de fechamento mediano 1–3 janelas (p95 ≤ 17):

| run | janelas c/ demanda | pool fecha | % | k med | k p95 |
|---|---|---|---|---|---|
| λ2s2102 | 314 | 310 | 98,7% | 1 | 1 |
| λ2s2103 | 1.164 | 963 | 82,7% | 1 | 3 |
| λ5s2101 | 532 | 225 | 42,3% | 3 | 17 |
| λ5s2102 | 1.169 | 558 | 47,7% | 1 | 19 |
| λ5s2103 | 2.722 | 1.756 | 64,5% | 1 | 11 |
| λ6s2101 | 928 | 454 | 48,9% | 2 | 8 |
| λ6s2102 | 1.595 | 792 | 49,7% | 1 | 17 |
| λ6s2103 | 2.911 | 1.719 | 59,1% | 1 | 15 |
| **TOTAL** | **11.335** | **6.777** | **59,8%** | | |

## 3. Teste isolado (flag `--urllc_delayed_update` — canônica intacta)

**Desenho (2 estágios, função de aprendizado INALTERADA — `observe_outcome` per-slice com R19 untouched):**
- **Estágio 1 (idêntico à canônica):** update em W+1; eMBB/mMTC completos do cache; URLLC tipicamente None ⇒ R19 skip — exatamente como hoje.
- **Estágio 2 (novo, flag-only):** fila de pendentes {W, state, action}; a cada iteração, W com pool fechado recebe uma SEGUNDA `observe_outcome` com outcome **só-URLLC** (eMBB/mMTC None ⇒ R19 os exclui) ⇒ atualiza **somente** o modelo URLLC, com o MESMO mecanismo quantílico, 1× por janela.

**Execução:** 1 run λ5s2102, 90 s, warmup 20 s, seed 2102 (`d1_delayed_feedback_test/df_lam5_s2102`; desvio de duração declarado — teste, não campanha).

**Resultados da auditoria (script reproduzível, `urllc_delayed_updates.jsonl` + `decisions.jsonl` + stream):**

| critério exigido | resultado |
|---|---|
| updates URLLC efetivos | **237** (canônica: 0); 1 OPEN residual (cauda) |
| sem duplicatas | 0 dups de W (guard + auditoria) |
| associação estado-ação-outcome correta | 237/237 W atualizados têm decisão correspondente (`outcome_window_index==W`); margem URLLC do estágio 1 era None em 53/53 decisões B0 |
| ausência de vazamento temporal | 0/237 updates com entrega usada além de `observed_at_window` (última entrega de W recomputada do stream: `floor(last_rx·10) ≤ observed_at_window` em todos) |
| rótulo = gabarito offline | **237/237 p95 do update == p95 offline completo** da janela (pool fechado ⇒ idêntico por construção) |
| aprendizado quantílico preservado | `sqr_het.py`/`online_update.py` intocados; eMBB/mMTC atualizam no estágio 1 como na canônica |
| idade de fechamento | mediana 2 janelas |

Testes unitários: 7/7 (novo teste de ciclo de vida OPEN→CLOSED/NO_DEMAND).

## 4. Conciliação residual da errata

**4a. As "5.269 linhas restantes do universo 26.955" NÃO existem no v8.** O 26.955 era o universo das versões **quarentenadas** (v3/v4/v6/cold: 3.000 janelas-relógio/run × 9, com holds logados como decisões). O universo canônico v8, reconstruído: **27.000 janelas-relógio** (300 s × 9) = **21.713 janelas com KPM** + **5.287 janelas de célula vazia** (sem UE ativo: sem linha de KPM, sem decisão — o v8 corretamente NÃO loga, ao contrário do v6 que logava hold e re-observava = bug 3); 21.713 janelas KPM = **21.686 decisões** (6.553 bootstrap + 15.133 B0; 27 tail sem decisão) — os números da errata §1 estavam certos, apenas o rótulo "26.955 linhas" era herança indevida do relatório antigo. As 5.269 = diferença entre o universo quarentenado e o v8 (5.287 vazias − 18 de arredondamento de versão). **Correção registrada no ADENDO da errata.**

**4b. Identidade TX/RX λ6s2103 com a borda de 92 explícita** (recomputada do CSV):
`TX_kpm 376.427 = RX_delays 243.300 − 92 (borda) + 132.605 (perdidos no meio) + 614 (em voo na cauda, janelas 2990–2996, entrega não ocorreu antes do fim da sim)` — fecha EXATAMENTE. Os **92** são pacotes com txSec ∈ [299,9; 300,0) **após o último callback de KPM** (que fecha a janela 2998 no stamp 299900): são RX **sem** denominador TX no KPM (janela 2999 existe só no stream) — não são perdas, nem sobreposição de mecanismo; as janelas 2997/2998 têm RX completo (100/100).

## 5. Parecer técnico (item 5 do ofício)

**DELAYED FEEDBACK É VIÁVEL** com a instrumentação atual — demonstrado por teste isolado e quantificação. A alteração metodológica necessária é uma **EMENDA ao protocolo** (não mudança de código adicional):

1. **Emenda 4 (proposta): update URLLC por fechamento observável do pool.** O runtime mantém pendentes (estado, ação) de janelas URLLC cuja margem não estava disponível; atualiza o modelo quantílico URLLC — inalterado — na primeira iteração em que `delivered(W)==tx_kpm(W)` (ambos online). Janelas com perda real permanecem vacuous (R19). SLAs, espaço de ações, função de aprendizado e prioridades: **intocados**.
2. **Impacto sobre D1:** a canônica v8 permanece válida como está (observador parcial documentado); a emenda caracteriza-se como evolução do observador, não invalidação. Se o orientador optar por adotá-la como canônica, D1 deve ser re-executada (9 runs, ~3 h) com a flag — as trajetórias de telemetria física são determinísticas por seed, mas as decisões B0 divergem a partir do primeiro update URLLC.
3. **Impacto sobre D2:** nenhum se D2-v2 rodar com a condição "URLLC cego" declarada (estágio 1 = canônica). Se a emenda for adotada ANTES da D2-v2, os replays são idênticos (replays usam checkpoints congelados e não aprendem online — freeze), apenas a declaração de limitação muda. **Recomendação: D2-v2 pode prosseguir de qualquer forma.**
4. **Impacto sobre D3:** B1 (JointSelector) não usa updates URLLC para decidir (usa ranking lexicográfico sobre previsões congeladas); a emenda afetaria apenas o B0 futuro — para a comparação pareada D3 os checkpoints permanecem os mesmos.
5. **Validade científica:** a emenda REMOVE a limitação "URLLC online cego" sem introduzir vazamento — 59,8% das janelas com demanda tornam-se atualizáveis com rótulo idêntico ao offline; as 40,2% com perda seguem R19 (vacuous), o que é o tratamento correto e não um viés (não conta não-entregues nem como miss nem como sucesso).

**Decisão do orientador requerida:** (a) adotar a Emenda 4 como canônica (⇒ re-executar D1 com a flag, ~3 h, antes da D2-v2) — observador mais forte e limitação removida; ou (b) manter Opção B como está (⇒ D2-v2 prossegue com a limitação declarada e a emenda fica documentada como trabalho futuro). Em ambos os casos **não há bloqueio metodológico** para a D2-v2; a diferença é o relatório do observador.

**Vedados preservados:** aprendizado quantílico, SLAs, espaço de ações comum, nenhum novo componente no SQR-JC, D3/PILOT/MAIN bloqueados.
