# Rodada 2 — Etapa 2.2: Sanity em baixa carga (relatório com gate STOP)

**Data:** 20/10/2026 · **Protocolo:** MD §11 (`docs/artigo/rodada_2_fronteira_viabilidade_sqr.md`) · **Mandato:** orquestrador 20/10 (SLAs congelados R36–R38; chegadas explícitas por slice; artefatos em diretório próprio; STOP ante violação sistemática).

**Setup:** 3 seeds × 300 s, `--sliceLambdas=1,1,1 --sliceApps=greedy,urllc,mmtc`, rádio congelado (300 m, 30 dBm, 10 MHz/C=53, UMa, omni 15 dBi, TDD 3DL/7, μ=0, PF, RLC UM), equal-split, network-only. KPM + `sqr_packet_delays.csv` preservados em `lowload/hetero_s{1,2,3}/`.

**Gate pré-declarado (sistemático):** eMBB >5% das UE-janelas c/ thr<7 Mbps · URLLC >5% pacotes c/ delay>10 ms · mMTC PDR agregado <95%.

## Resultado (por run)

| run | eMBB viol% | eMBB thr med | URLLC miss% | URLLC p95 | URLLC med | mMTC PDR | util D2* | SNR p5 |
|---|---|---|---|---|---|---|---|---|
| s1 | 66,9 | 3,9 Mbps | 2,5% | 8,1 ms | 5,1 ms | 1,000 | 53,9%* | 20,8 dB |
| s2 | 43,9 | 7,2 Mbps | 15,8% | 14,1 ms | 6,1 ms | 1,000 | 51,6%* | 14,9 dB |
| s3 | 96,9 | 2,4 Mbps | 3,5% | 9,1 ms | 5,1 ms | 1,000 | 59,8%* | 20,8 dB |

*util D2 distorcida pelo bug de telemetria (abaixo).

## **VEREDICTO: STOP confirmado (§11.3)** — eMBB e (s2) URLLC acima do limiar. Diagnóstico por sintoma:

### 1. eMBB: comportamento = cota/N — NÃO é bug
Série por UE: sozinho, o UE eMBB entrega 7,1–7,7 Mbps (cota equal-split ≈ 1/3 da célula ≈ 8 Mbps); em clusters Poisson de 2–5 UEs simultâneos (λ=1/min, sessão 60 s), cada um fica com 8/N = 1,2–4 Mbps. Fila RLC cheia (2–3,7 MB — greedy de fato), cota da slice sempre consumida (sched≈budget na slice 0, correto). **O SLA de 7 Mbps por UE excede o que 1/3 da célula de 10 MHz sustenta para N≥1 com folga zero** (mediana 7,2 na melhor seed). Contexto citável: DORA opera com demanda agregada >106 PRBs (§VI) — banda ~2× a nossa (53 PRBs); 7 Mbps na célula deles ↔ ~3,5 Mbps na nossa.

**Decisão pendente do orquestrador:** (A) escalonar o SLA à banda → **3,5 Mbps** `[ARTIGO DORA Eq.(5)]` + escalonamento linear `[ASSUMIDO]`; (B) manter 7 Mbps e reduzir λ_embb no sanity (mas aí a fronteira eMBB domina todo o sweep 2.3); (C) manter 7 e redefinir o gate por N(t) (não recomendado — §11 exige atingibilidade em baixa carga).

### 2. URLLC: mediana cumpre (5–6 ms < 10 ms); cauda tensiona — comportamento desejado, com ressalva
488k pacotes/run; mediana 5,1–6,1 ms; miss 2,5–15,8% (p95 8–14 ms). O miss concentra-se em UEs/períodos específicos (ex. UE 8 na s2: med 9 ms). Com o bug de telemetria corrigido, re-avaliar se a cauda muda (não deveria — delays vêm do packet_delays, instrumento independente). **R37 (10 ms) parece bem calibrado**: cumprível em baixa carga na mediana, tensionável na cauda.

### 3. BUG de telemetria encontrado (correção em curso): `prb_scheduled_rbg` das slices CBR
A slice URLLC reporta `sched == budget == 952` RBGs/janela (cota cheia) enquanto o `ue_detail.csv` mostra ~169 TB/s de **1 RBG** (agregando ~8 pacotes de ~60 B cada — comportamento correto do MAC) e o KPM confirma 1628 pps entregues. O contador reporta ~56× o real (= cota×dl_slots) para slices CBR; a slice greedy está correta. Consequências: `util D2` inflada (33% real eMBB + ~19% fantasma = 51–60%) e — crítico — o oracle 2.4 distorceria F_t. **Dinâmica (cotas/delays/PDR) intacta — só o contador mente.**

**Causa raiz (encontrada pelo implementador 20/10, em validação):** o porte rslaq do scheduler omitiu a porta de suficiência de demanda do upstream 5G-LENA (`m_dlTbSize >= max(buffer,10)`, `contrib/nr/model/nr-mac-scheduler-ofdma.cc:113-121`) em `SqrMacScheduler::AssignDLRBG` — sem ela o MAC emite TBs mínimos a cada slot com buffer quase vazio e o acumulador de janela soma cota em vez de alocação. Fix aplicado (contrib/nr e scratch/rslaq intocados; 194 testes passando). Evidência independente de 300 s: sched_1/budget_1 = 0,122 (correto), slice 0 = 0,963 da cota, mmtc ≈ 0, PDR URLLC 31678/31686 = 99,97%. Gate do workflow emendado ($12/$15 — as colunas $10/$13 originais eram as colunas erradas da slice 2, erro do orquestrador; e λ=6,6,6 para garantir UEs URLLC no run de gate) — run `dwfrun-c16a4922`.

### 4. mMTC: PDR agregado = 1,000 em todas as runs (R38 ✓)
4–9 sessões/run, PDR de sessão = 1,0. Sem perdas em baixa carga, como esperado (fila grande, tráfego esparso).

## Próximos passos
1. ~~Conclusão do fix de telemetria~~ ✅ (ver §3v2 abaixo).
2. Decisão do orquestrador sobre o SLA eMBB (A/B/C acima).
3. Re-executar 2.2 com o SLA decidido → se OK, sweep 2.3.

---

## v2 (20/10, tarde) — correção aplicada e 2.2 re-executada com binário corrigido

**Diagnóstico final (corrige o §3 acima):** o contador e o dreno (`sqr-mac-scheduler.cc:734`, `sqr-sim.cc:1578-1610`) eram FIÉIS (≤3% de divergência vs `slice_alloc.csv`). O defeito estava no **alocador**: o laço de `SqrMacScheduler::AssignDLRBG` usava `demandPassed = (bufQueueSize > 0 || m_dlTbSize > 0)` com snapshot que nunca zera no laço ⇒ **qualquer slice com buffer > 0 recebia a cota inteira por slot** (TB cheia de padding). O upstream 5G-LENA tem a porta de suficiência (`AdvanceToNextUeToSchedule` pula UEs com `m_dlTbSize >= max(bufQueueSize,10)`, `nr-mac-scheduler-ofdma.cc:113-121`) que o porte rslaq omitiu. A leitura anterior do `ue_detail` ("~17 RBGs/janela = agregação") estava enviesada pela amostragem de log (~1,9% dos slots; cada linha = 1 RBG do TB crescente): 50.609 linhas / 3.011 chamadas = 16,8 RBG/chamada ≈ 918/janela — os TBs eram de padding mesmo. **Fix:** porta de suficiência restaurada (`scratch/sqr/sqr-mac-scheduler.cc`, laço ~547). Gate independente PASS (scheduled_1 < 25% do budget com UEs URLLC garantidos); suíte 194 PASS; `contrib/nr` e `scratch/rslaq` intocados.

**Escopo do impacto:** a correção altera o COMPORTAMENTO do MAC para slices CBR (elimina padding), não só telemetria. As rodadas homogêneas (1b, 2.0 — todas greedy, buffer sempre grande) **não são afetadas** (a porta nunca disparava); os runs heterogêneos da 1ª execução foram preservados em `lowload_v1_allocbug/` e a 2.2 re-executada em `lowload/`.

**Resultado do re-run (mesmas seeds/params, binário corrigido):** util D2 = 24,9–36,6% (era 51–60% — fantasma ~19% eliminado); KPIs de SLA materialmente idênticos (eMBB viol 48,2/…/97,3%, thr med 4,5 Mbps; URLLC med 5,1–6,1 ms, p95 8–14 ms, miss médio 8,5% — máximo 17,6% na s2; mMTC PDR 1,0). **STOP mantém-se** — eMBB por desenho (cota/N vs SLA 7 Mbps; decisão A/B/C pendente do orquestrador), URLLC cumprindo na mediana com cauda genuína na seed com mais sobreposição de UEs. R37 (10 ms) segue bem calibrado.

---

## v3 (20/10, noite) — SLA eMBB 3,5 Mbps (R36 emendada; mandato do orquestrador) e o item 5 acionado

**Gate re-executado com SLA operacional 3,5 Mbps** (mesmos runs determinísticos do binário corrigido — o SLA é parâmetro de análise): s1 38,2% · **s2 0,17% (passa quase perfeito)** · s3 75,6%. URLLC miss 2,0/17,6/5,7% (med 5,1–6,1 ms ✓). mMTC 1,0 ✓.

**Leitura (item 5 do mandato, confirmado quantitativamente):** a violação eMBB remanescente é dos **clusters Poisson de N=3–5 UEs** sob equal-split (8/N = 1,6–2,7 Mbps < 3,5) **com a célula a 29–37% de utilização** — demanda de pico (5×3,5 + URLLC + mMTC ≈ 19 Mbps) cabe com folga na célula (~24 Mbps DL). **Forte evidência de evitabilidade** (a demanda cabe na capacidade visível e a política não realoca; a classificação formal F_t≠∅ vs F_t=∅ fica reservada ao oracle da Etapa 2.4); o SLA é viável e o rádio está são — o gargalo é a política estática 1/3·1/3·1/3. Evidência empírica consistente com a tese do artigo já no sanity.

**Ação (item 5):** controlador **proporcional à demanda** (`prop`) no loop IPC existente (sem C++ novo; pisos p/ URLLC/mMTC [ASSUMIDO] R39) — workflow `dwfrun-e3f23b8e`. O sanity 2.2 fecha com o `prop` (atingibilidade com realocação); o sweep 2.3 mantém **equal-split como baseline estático** (consistência com 1b/2.0; o oracle 2.4 quantificará o evitável que o prop realiza).

---

## v4 (21/10) — sanity final: prop v3, λ=0,5/slice, 7 seeds — eMBB/mMTC aprovados; URLLC na cauda

**Calibração do prop** (workflows `dwfrun-e3f23b8e`→`dwfrun-f52e7e49`, emenda R39 v1.7-cal2): cap eMBB = 5,25 Mbps/UE (margem 1,5×), pisos {eMBB 5%, URLLC 25%, mMTC 5%}, 0,55 Mbps/RBG [ASSUMIDO]; 207/352 testes. Em baixa carga as ações são dominadas pelo pool saturado (quota eMBB ≈ 37 PRB), logo v1≈v2≈v3 — os gates intermediários com números idênticos são genuínos. Medição chave (run suplementar): 37 PRB entregam ~14,2 Mbps ⇒ **eficiência real ≈ 0,38 Mbps/PRB** (a constante 0,55 é conservadora; célula ~20 Mbps).

**Recalibragem do cenário do sanity (decisão do orquestrador, registrada):** λ=1/slice produz clusters Poisson de 4–5 UEs eMBB cuja demanda (14–17,5 Mbps) + reserva URLLC encosta na capacidade real (~20 Mbps) — isso é **região da fronteira** (matéria do sweep 2.3), não sanity. O §11 quer baixa *demanda*: λ=0,5/slice (E[N]=0,5, clusters ≥4 improvabilísticos).

**Resultado (7 seeds × 300 s, IPC, prop v3, runs em `lowload_prop_v2/`):**
- **eMBB: APROVADO sistematicamente** — violação 0,15–0,58% em 7/7 (suavizado 1 s: 0,0%); thr med 7,8–16 Mbps ≫ 3,5.
- **mMTC: APROVADO** — PDR 1,0 em todas as seeds com sessões (seed 4 sem chegadas mMTC: P(0)=e^-2,5≈8% — gate corrigido para NaN, era falso positivo com 0/0).
- **URLLC: 4/7 seeds passam (<5%); 3/7 com cauda 11–16%** (s2 10,95, s4 15,79, s7 13,20). Mediana SEMPRE 5,1–6,1 ms (folga 40% no deadline); p95 8–14 ms; p99 ≤17; max 25.

**Diagnóstico da cauda (s2 decomposta):** miss concentrado NO TEMPO (0,6–5,6% em períodos calmos; 15–39% nos períodos com 2 UEs URLLC e/ou atividade eMBB intensa nos mesmos slots) e espalhado entre UEs (todos com mediana saudável). Latência base do caminho (BSR/agregação de TB/agendamento TDD) ≈ 6 ms ⇒ p90 ≈ 10,1: a distribuição inteira fica junto do deadline e qualquer acoplamento com atividade empurra a cauda. **Investigação §11.3 completa: SLA ✓ (mediana), KPI ✓ (instrumento por pacote, independente), tráfego ✓, física ✓ (SNR 20–24 dB), recurso ✓ (URLLC usa fração mínima da cota — não é contenção de RBG), scheduler = mecanismo** (atendimento por slot/agregação, PF intra-slice congelado). A única alavanca restante é mudar o mecanismo de agendamento — i.e., mudar o cenário congelado no meio da caracterização.

**Posição do orquestrador (decisão pendente do usuário):** aprovar a 2.2 com ressalva documentada — a cauda URLLC é propriedade do scheduler congelado, mediana cumpre o deadline com folga, e a evolução dessa cauda com a carga é exatamente o que o sweep 2.3 e o oracle 2.4 devem medir (é sinal, não ruído). Alternativas (mudar scheduler/deadline) violam o congelamento do cenário ou o SLA ratificado. O sweep 2.3 permanece pronto (runner + análise) para lançamento imediato após o go.

---

## v5 (21/10) — ENCERRAMENTO: sanity qualificado + decomposição da cauda + sweep 2.3 em 2 braços

**Decisão do orquestrador (21/10):** Etapa 2.2 **encerrada como *sanity qualificado*** — eMBB e mMTC aprovados sistematicamente; URLLC com cauda dependente de estado/scheduler e mediana saudável. **Scheduler e deadline intocados.** Correção terminológica aplicada: as leituras anteriores de "F_t≠∅" são rebaixadas a **forte evidência de evitabilidade** — a classificação formal (F_t≠∅ vs F_t=∅) fica reservada ao oracle da Etapa 2.4.

### Decomposição da cauda URLLC por N_urllc(t) × atividade concorrente (dados existentes, sem nova campanha — `analyze_urllc_tail.py`, CSVs `round2_urllc_tail_decomp{,_agg}.csv`)

Miss% agregado por pacote (braço prop, 7 seeds / braço equal, 3 seeds):

| N_urllc \ N_embb | 0 | 1 | 2+ |
|---|---|---|---|
| **1** | 1,2% (n=711k) / 0,8% | 14,5% (n=346k) / 10,6% | 2,7% (n=104k) / 3,2% |
| **2** | 4,3% (n=203k) / 4,0% | 26,8% (n=138k) / 11,5% | 5,8% (n=47k) / 4,2% |
| **3+** | — / 5,0% | — / 34,6% | — |

1. **Monótona em N_urllc** (1→2→3+: miss cresce em toda coluna) — contenção intra-cota com mais UEs URLLC.
2. **NÃO-monótona em N_embb**: o estado com **1 UE eMBB ativo** é o pior (10–27%), pior que 0 e pior que 2+ UEs — nos DOIS braços (não é o `prop`). Corte por saúde do eMBB (quartis de throughput) no estrato 1×1: gradiente parcial no equal (Q1 20,6% → Q3/Q4 ~5%) e misto no prop — não fecha como confunder único. **Observação aberta** (hipóteses: fase TDD/BSR interagindo com a carga do slot; UEs eMBB em MCS ruim com HARQ retx ocupando o processamento do slot; confunder temporal dos estados). A densidade do sweep 2.3 deve separar essas hipóteses — fica como item de leitura do sweep.
3. Consequência prática: o miss de uma seed resume a fração do tempo nos estados (1×1)/(2×1) — variância entre seeds explicada pela sorte das sobreposições.

### Sweep 2.3 (iniciado 21/10, mandato: equal-split baseline estático + prop witness adaptativo)

Grade: Perfil A (λ_total ∈ {2,4,5,6,7} ÷ 3 por slice) × seeds 1–3 × 300 s × rádio congelado, **30 runs** em 2 braços (`hetero_sweep/` network-only equal; `hetero_sweep_prop/` IPC com `prop`). Infraestrutura, SLAs (R36–R39) e binário congelados. Análise: `analyze_hetero_sweep.py` (dois braços, CSV §32 por janela, utilizações D1/D2 separadas).
