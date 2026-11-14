# INCIDENTE 11/11 — CONTAMINAÇÃO DO STREAM DE DELAYS NO D1 (SIM_BUG §5.9)

**Classe:** SIM_BUG (estado do controlador com dados de outro cenário — violação do INV-06 e do §14) · **Detecção:** durante o D3 (reconstrução offline dos estados produziu 100% UNKNOWN URLLC em λ2/λ5 — contradição com os denominadores D1) · **Status:** causa raiz corrigida; D1 em re-execução (v4); D2/D3 quarentenados e a re-derivar.

## Causa raiz (provada)

1. No `sqr_run.py`, o `HetRuntime` era construído **antes do `env.reset()`**, quando `env.sim_path` ainda não existe ⇒ `HetRuntime("")`.
2. Com path vazio, o `DelayStream` resolvia por `glob("**/sqr_packet_delays.csv")` **relativo ao CWD** (o GYM) ⇒ **609 arquivos** casados; `sorted()[0]` = `results/sqr_characterization_round2/hetero_sweep/A_lam2_s1/.../sqr_packet_delays.csv` (Rodada 2, braço equal, seed 1).
3. **Prova bit-a-bit:** as margens URLLC dos decisions.jsonl de TODOS os 9 runs D1-v3 coincidem com o outcome de A_lam2_s1 por janela: 2.566/2.566 (λ2s2101) e 2.704/2.704 (λ6s2103). O run λ2s2101 NÃO teve tráfego URLLC nenhum (0 pacotes sliceId=1 no delays local — chegada Poisson sem UE URLLC em 300 s), mas o runtime "via" 4.570 janelas URLLC do stream alheio.
4. Efeito: estados URLLC (features 7/21/22) e rótulos URLLC dos 9 runs D1 (v3 **e** cold) eram de outro cenário ⇒ decisões do B0 em malha aberta tomadas sob informação errada.

## Alcance (o que está/quarentenado vs. o que permanece válido)

| artefato | status |
|---|---|
| D1-v3 canônica + D1-cold (runs, decisions.jsonl, análise §7.2, denominadores, estratos) | **QUARENTENADOS** (`d1_runs_v3_CONTAMINADO_quarentena/`, `d1_runs_cold_CONTAMINADO_quarentena/`, CSVs/figuras em `quarentena_contaminacao_1111/`) |
| D2 (amostragem 30 estados, 390 replays, D2_report/veredito) | **QUARENTENADA** (`quarentena_contaminacao_1111/d2_*`) — os outcomes adjudicados por replay eram LIMPOS (`window_kpis` offline dos arquivos de cada replay), mas a AMOSTRA foi estratificada com violações conjuntas contaminadas ⇒ viés de seleção; re-derivar do universo D1-v4 |
| D3 (JointSelector, decisões d3_test) | **DESCARTADAS** (z_i contaminado) |
| D0-A / D0-B / G4-A / G4-B / Rodada 2 | **LIMPOS** (não usam o DelaysStream de runtime; adjudicações offline por arquivo) |
| Correção do gate 3-way etc. | não afetado |

## Correção da causa raiz

`HetRuntime` construído **LAZY dentro de `run_episode`, após `env.reset()`** (quando `env.sim_path` existe) — o delay stream passa a ser o DESTE run. Testes 19 passed. Sinal imediato de sanidade no D1-v4: λ2s2101 com **URLLC None em 1222/1222 decisões** — o verdadeiro estado vacuous (R19), antes mascarado pelo stream alheio.

## Plano de re-execução (pré-registrado; sem mudança de protocolo)

1. **D1-v4** (9 runs, seeds 2101–2103, flags congeladas, bootstrap-PROP §3.4) — em execução (~3 h). Estratos/estatísticas re-derivados.
2. **D2-v2**: re-gerar a amostragem (MESMA seed 20261108 sobre o universo limpo — a emenda e as regras não mudam) + 390 replays paralelizados ×3 (~12 h) + re-adjudicação. O D2_report v1 fica como registro do incidente.
3. **D3**: reiniciar decisões pareadas (estados limpos) + replays confirmatórios conforme a declaração de conformidade (4 cuidados do orientador inalterados).
4. Relatório ao orientador com este documento como peça central — nenhuma conclusão D1/D2/D3 anteriores deve ser usada.


## ADENDO 11/11 (tarde) — BUG Nº 2: censura tardia do outcome URLLC (D1-v4)

D1-v4 (stream local correto — margens 684/684 batem com o delays DO run) ainda
marcou só 1/959 janelas URLLC como COMPLETE. Causa: pacotes da janela W são
ENTREGUES/flushados até ~1 janela após o fim de W (medido: 630/959 janelas com
última entrega >50 ms após o fim); o runtime consultava `stats(W)` com o poll
feito ANTES dessas entregas chegarem ao arquivo. Correção: `poll()` extra no
início de `outcome()` e `evidence()` (a decisão de W+1 já vê os entregados
tardios de W). Validação offline: stream cobre 959/959 janelas do gabarito.
D1-v4 ⇒ `d1_runs_v4_CENSURA_quarentena/`; re-execução como **D1-v5**.

Nota positiva do v4: a estrutura FÍSICA está correta — o KPM tem os 3 slices
simultâneos em 792–1.709 janelas por run (interseção direta do arquivo), ou
seja, o cenário suporta ALL3_COMPLETE; era só a leitura runtime que censurava.


## ADENDO 12/11 — BUG Nº 3: hold repete outcome e colapsa o modelo URLLC (D1-v6)

Mecanismo esclarecido (λ5s2102 v6): entre t=25,3 s e t=93,8 s a célula ficou
VAZIA (UEs de 60 s partiram; chegada Poisson média 36 s/slice) ⇒ sem linhas
KPM ⇒ o builder do env mantém `last_window` ANTIGA (252) e o fluxo het do
sqr_run re-observava o MESMO outcome 684× (m_u=+0,19) e re-atualizava o
modelo URLLC com o MESMO (z,a,m) ⇒ previsão URLLC colapsou (−0,15→−3,06;
observado também na evolução por terços). kpm_missing=0 porque o arquivo
IPC existia (com header) — o caso correto é HOLD puro.
**FIX**: guard anti-repetição no sqr_run (timestamp não avançou ⇒ repete a
última ação sem absorver/observar/logar). 32 testes passed.
D1-v6 ⇒ `d1_runs_v6_HOLDREPEAT_quarentena/`; re-execução como **D1-v7**
(runs canôicos da Opção B; holds agora são passos reais de hold sem
distorção de aprendizado). Nota: holds afetam TODOS os runs (2151 holds no
λ2s2102 — 72% das decisões!) — o impacto quantificado entra no relatório.


## ADENDO 12/11 (noite) — BUG Nº 4 (meu, no guard do v7): spin sem bloqueio ⇒ OOM 10,3 GB

O guard anti-repetição do v7 fazia `continue` ANTES do `env.step()`: sem o
bloqueio do semáforo IPC, o loop girou em velocidade Python (~30k it/s) e o
`decisions_log` (drenado só no fim do episódio, sqr_run.py:889) cresceu sem
limite ⇒ python com anon-rss **10,3 GB** morto pelo OOM-killer às 20:28
(journalctl; runner e sessão ZCode no mesmo cgroup — o reinício da sessão).
Classificação: bug de wiring introduzido por mim no fix do bug 3 (não do
protocolo). Lições: (i) NUNCA `continue` em volta do env.step — o passo é
quem respeita o contrato de semáforos; (ii) listas acumuladas por iteração
precisam de dreno por passo.

**FIX v8 (estrutural, não cosmético):**
1. Hold: repete a última ação e SEGUE o fluxo normal até env.step (sem
   absorb/observe/log duplicado — bug 3 corrigido sem quebrar o contrato).
2. Pendente observado contra a PRÓPRIA janela governada
   (`outcome_for_window(w)`) com cache por época de eMBB/mMTC e stream
   URLLC acumulativo — pendentes que atravessam gaps ficam com o outcome
   COMPLETO da sua janela (Opção B sem censura para esses casos).
3. `evidence(w)` por índice de janela (evidência por época).
Validação: 45 testes passed; D1-v8 com 0 hold-repetidos; RSS do python
estável (180→191 MB em 5 min — crescimento normal do modelo); v7
quarentenado (`run_d1.nohup` do v7 perdido no OOM; runs v7 não existem).
