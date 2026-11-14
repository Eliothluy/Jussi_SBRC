# PROTOCOL_FROZEN — EMENDA 3 (Opção B): estado online parcial × avaliação offline íntegra

**Data:** 12/11/2026 · **Autoridade:** decisão do orientador ("seguir com a Opção B, preservando 100 ms e separando rigorosamente estado online × avaliação offline") · **Vedações reafirmadas:** função de aprendizado, SLAs, espaço de ações e critérios de segurança INALTERADOS (qualquer mudança exige nova decisão metodológica). D3/PILOT/MAIN bloqueados.

## 1. Auditorias técnicas exigidas (respondidas com dados dos runs quarentenados v4/v5)

**Q1 — "a perda do deadline de 10 ms só pode ser identificada após a entrega?"**
**NÃO — com uma condição de instrumentação.** O `sqr_packet_delays.csv` registra linhas **apenas no Rx** (`adjudication`/`sqr-sim.cc:457-506`: ByteTag de TX no `SendOutgoing` do remoteHost, medida no `SqrUdpServerRxCb`). Para um pacote **entregue**, o miss é determinado por `delaySec·1000 > 10` — e o instante-de-expiração (`txSec+10 ms`) é **anterior** à entrega em **67,8%** dos pacotes (65.955/97.255 no λ5s2102; 685/959 janelas têm miss determinável antes da última entrega). Porém **pacotes nunca entregues não geram linha alguma** (medido: 30–35% do tráfego URLLC sem linha — perda RLC-UM real + cauda do run): a expiração **absoluta** desses exige um trace de TX puro, que **não existe** no contrato atual (o txSec só persiste junto do Rx). **Decisão documentada:** não inventar o trace de TX (seria alteração de instrumentação com nova campanha); a auditoria de denominadores passa a reportar explicitamente `tx_sem_entrega` por janela (do KPM `dTxPackets` × linhas de delays) e a avaliação primária usa o **denominador de entregas** com o número de não-entregues como métrica separada — NUNCA contando não-entregues como miss implícito nem como satisfação.

**Q2 — observação do miss vs valor exato de p95:**
Duas grandezas distintas, disponibilidades distintas:
- **miss por pacote**: determinável no instante `tx+10 ms` (para entregues) — observável online com atraso ≤1 janela para a fração entregue cedo;
- **p95 exato da janela**: exige o POOL completo dos pacotes entregues de W — só está fechado ~300–700 ms depois (p90 354–708 ms). **Convenção da Opção B:** o *runtime* não calcula p95 de W; o *offline* sim, do arquivo completo, sem censura.

**Q3 — como o SQR-HET trata pendentes; nº de updates efetivos; viés só-entregues:**
- Tratamento de pendentes: `observe_outcome` aplica **R19** — margem `None` ⇒ **exclusão do update** (nunca miss implícito, nunca satisfação por vacuidade). É o comportamento congelado e permanece.
- Updates efetivos (λ5s2102): **0 updates URLLC** no runtime censurado (v4) vs **959 janelas** no gabarito offline ⇒ **100% de perda de updates URLLC** por censura de janela. Os updates eMBB/mMTC não são afetados (KPM é drenado por janela).
- Viés só-entregues: o runtime (e qualquer avaliação por `delays.csv`) vê **apenas entregues** — a perda RLC-UM (35,1% do tx em λ5s2102) não aparece como miss nem como rótulo. Sob a Opção B isso passa a ser **documentado e quantificado** (métrica separada) em vez de silencioso. Não é corrigido no modelo (vedação) — é declarado como condição do observador.

## 2. Regras da Opção B (congeladas para D1-v6/D2-v2/D3)

1. **Runtime**: usa exclusivamente informação disponível no instante da decisão (estado do KPM fechado + stream de delays até então; R19 para o que não está lá). Sem cálculo de p95 da janela corrente. O que o controlador "via" fica registrado em `decisions.jsonl` (campos de estado/predições já existentes).
2. **Avaliação/rotulação**: **TODA** métrica de SLA (julgamento frozen, STRICT ALL3, miss, PDR) é computada **offline do arquivo completo** por `window_kpis`/convenções congeladas — 0% de censura (gabarito 959/959 validado). Denominadores auditados por janela: `tx, entregues, tx_sem_entrega, n_evaluable` por slice.
3. **Emparelhamento (§23):** o outcome offline da janela W é atribuído à ação que governou W (decisão no passo anterior) — a mesma regra do runtime, sem defasagem de leitura.
4. **Censura explícita**: janela com pacotes cuja entrega não ocorreu dentro do run (fim de sim) = PARTIAL por R19 no julgamento strict, reportada.
5. O aprendizado do B0 continua **com os outcomes que ele de fato observou** (R19) — nenhum rótulo offline retroalimenta o modelo (cuidado 1 da D3 mantido).

## 3. Plano de execução (pré-registrado)

1. **D1-v6** (9 runs; flags/bootstrap-PROP congelados; runtime como está — estado parcial): produz `decisions.jsonl` (o que foi visto) + telemetria completa (o que aconteceu). Análise offline: denominadores auditados, estratos, P_projection, §7.2. ~3 h.
2. **D2-v2**: re-amostragem (seed 20261108) sobre estratos offline íntegros; 390 replays ×3; adjudicação offline completa. ~12 h.
3. **D3**: decisões pareadas com rótulos offline íntegros; replays confirmatórios; gate exploratório. Conforme declaração de conformidade (4 cuidados).
4. Relatório final da cadeia D1→D3 com a tabela de denominadores por run.


## ADENDO 12/11 — mecanismo EXATO da observabilidade online URLLC (resposta ao ponto 1, provado por instrumentação)

Instrumentação direta (shim de log no DelayStream/HetRuntime, probe 25 s
λ5s2102, 231 decisões): na decisão da janela W, o stream contém delays
**até W−1** — `wins_last = W−1` em todas as 231 chamadas. Cadeia causal no
ns-3: o mesmo callback (`KpmAndControlCallback`) escreve o KPM **antes** e
flusha os delays **depois** (`FlushPacketDelays` no fim, sqr-sim.cc:1805) ⇒
na época da decisão W o arquivo contém entregas até W−1, e os pacotes de W
só entram no callback W+1 (flush após o KPM que dispara a decisão W+1).
**Consequência (conforme Opção B — estado parcial POR CONSTRUÇÃO):**
1. O controlador **nunca observa online o miss nem o p95 da janela
   corrente**; observa, com 1 passo de atraso, os pacotes de W−1 entregues
   até o callback de W. A afirmação "identifica online a expiração em
   tx+10 ms" é FALSA e não consta.
2. Margens URLLC online ≈ nunca atualizadas (updates efetivos ≈0 pós-
   bootstrap em todos os 9 runs da v8 — tabela no relatório D1-v8): a
   resposta Q3 do orientador ("se a ausência persistir, diagnóstico e
   impacto") ⇒ **diagnóstico**: é consequência LEGÍTIMIMA do observador
   parcial (Opção B), não instrumentação remanescente — o stream funciona
   (prova: polls n>0, wins crescentes; offline 959/959).
3. Impacto na validade B0×B1: PARIDADE total (mesmo checkpoint, mesmo
   fluxo de observação, mesmo mecanismo); o déficit é simétrico. As
   escolhas URLLC serão guiadas por prior/estado (n_u, fila) e não por
   previsão calibrada — declarado como condição do método.
4. Nenhum rótulo offline entra no modelo (checkpoints hasheados na D3).
