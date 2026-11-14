# PROTOCOL_FROZEN — EMENDA 2 (D2 final): 6 ajustes do orientador + critério A/A

**Data:** 10/11/2026 · **Autoridade:** decisão do orientador 10/11 ("ajustes antes da execução") · **Estado:** CONGELADA antes de qualquer replay D2 (amostragem e hashes registrados: `D2_sampling_manifest.json`). Emenda 1 permanece onde não conflita. **Vedações reafirmadas:** aprendizado quantílico, SLAs, espaço de ações comum e critérios do gate INALTERADOS; D3/PILOT/MAIN bloqueados.

## Ajuste 1 — D1-v3 (bootstrap-PROP) = campanha CANÔNICA

Declaração formal: os 9 runs `d1_runs/het_lam{2,5,6}_s{2101,2102,2103}` (09–10/11) são a base exclusiva do universo D2. Relatório de integridade: `results/sqr_jc_decision/D1v3_integrity_per_run.csv` — 9/9 uniformes (2.995 decisões = 1.000 PROP + 1.995 B0; 2.994 outcomes; KPM até 299,9 s; ns3 exit 0; .done) + SHA-256 por decisions.jsonl (também no manifest da amostragem). Cold-start: EXPLORATÓRIO (`d1_runs_coldstart_exploratory/`), excluído de D2/D3.

## Ajuste 2 — STRICT como referência PRIMÁRIA de recuperação

Rótulo primário de cada (estado, ação): **STRICT_PASS** = 3× evidência COMPLETE + todos os SLAs ok (ALL3_COMPLETE). Reportados separadamente: **FROZEN_OK** (julgamento de janela R36–R38, vacuous=OK — sem valor primário) e **UNKNOWN** (qualquer NO_ACTIVE_DEMAND/INVALID — jamais PASS). Métrica primária de D2: `RecoverabilityFoundRate_STRICT`; frozen e UNKNOWN sempre ao lado. Nenhuma recuperação é "confirmada" sob frozen-only sem qualificação.

## Ajuste 3 — 12 alternativas distintas, admissíveis, reproduzíveis + substituição determinística

Gerador `d2_sampling_generate.py` (no manifest): candidatos em ordem fixa; **dedupe exato** contra a_B0 e entre si; duplicata ⇒ substituição pela **cadeia de sobra ordenada** (paired δ∈{1,32} em P1/P2, nbh ±2, depois 12 draws RNG extras) — tudo determinístico dado (run, win). Resultado: 30 estados × 12, **0 estados com duplicatas**, 60 substituições registradas linha-a-linha (colunas `mechanism`, `substituted`).

## Ajuste 4 — movimentos de redistribuição especificados + mecanismo por alternativa

Cada linha carrega `mechanism`: `paired:(i,j):δ:hi|lo` (transferência de δ RBGs do slice de menor score de demanda para o de maior — score_i = n_ues_i + [backlog_i>0]; pares ordenados P1..P3 por assimetria |score_i−score_j| decrescente; δ∈{2,4,8,16} fixos, P1/P2/P3 designados; δ=16 na direção inversa por desenho), `nbh:Pi:±1` (vizinhança de a_B0), `rnd:k:sha1(run|win)` (sorteio uniforme do simplex Σ≤53), `ref:equal` (18,17,18), `sub-*` (substituições, com o mecanismo substituído em `replaced`). Nenhuma alternativa origina-se do oracle.

## Ajuste 5 — independência D2/D3; oracle só ordena (e aqui: nem ordena)

A amostra (30×13) está fixada e hasheada ANTES dos replays; **nenhum resultado de replay altera o conjunto**. A fila de execução usa a ordem do CSV (ordenamento pelo oracle NÃO utilizado — elimina qualquer risco). D3 usará sorteio futuro (seed 20261109) sobre janelas B0 EXCLUINDO os 30 estados D2 e vizinhança ±5 janelas; nenhum dado D2 alimenta escolha de D3 além do que o PROTOCOL_FROZEN §5 já fixou.

## Ajuste 6 — relatório por estrato/λ/seed SEM extrapolação

Todos os resultados D2 reportados como contagens/taxas DENTRO dos estratos amostrados com denominadores explícitos (fração amostrada por célula: ex. S1 λ5 4/242=1,7%); **proibido** extrapolar proporções da amostra para a população; unidade de inferência = seed/run.

## Critério A/A dos replays D2 (pré-registrado, com evidência das âncoras)

Âncoras executadas (λ5s2101 t2000, ação própria): **(i) replay×replay bit-idêntico** (2 execuções independentes, 6.427 linhas KPM idênticas — determinismo da maquinaria); **(ii) quotas executadas vs B0: 0 divergências** (2.926 janelas DL); **(iii) KPM do B0 vs replays: drift de 195/6.427 linhas (3,0%), ts 181–200 s, colunas queue/delay, magnitude ≤0,1% — idiossincrasia do run B0 (hipótese do simTime refutada com replay 300 s: mesmo drift)**. Critério por estado D2:
1. **Prefixo bit-idêntico ENTRE os ramos do replay** (a_B0 + 12 alternativas) até t·100 ms — verificado por estado (a comparação causal é entre ramos, todos bit-idênticos por construção);
2. a_B0 vs run B0 original: quotas executadas idênticas (obrigatório) + relatório do drift KPM (linhas, colunas, magnitude) — declarado, não bloqueante;
3. divergência além disso ⇒ estado marcado INVALID (excluído, reportado).
