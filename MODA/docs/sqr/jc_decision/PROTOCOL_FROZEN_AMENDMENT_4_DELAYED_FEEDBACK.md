# PROTOCOL_FROZEN — EMENDA 4 (Delayed Feedback URLLC): update por fechamento observável do pool

**Data:** 12/11/2026 (noite) · **Autoridade:** decisão do orientador ("adoção condicionada da Emenda 4, preservando o pipeline e a contribuição de Joint Action Selection") · **Base técnica:** `DELAYED_FEEDBACK_investigacao.md` (teste isolado 237/237) · **Vedados reafirmados:** função de regressão quantílica, SLAs, regras R19 e espaço de ações comum INALTERADOS — a emenda modifica **apenas o agendamento da entrega de observações válidas ao modelo**. D2/D3/PILOT/MAIN bloqueados até a avaliação dos entregáveis.

## 1. Regra congelada (dois estágios)

1. **Estágio 1 (idêntico à Opção B/Emenda 3):** na decisão de W+1, o pending (estado, ação) da decisão que governou W é observado; eMBB/mMTC completos do cache do KPM; URLLC do stream se já disponível (tipicamente None ⇒ R19 exclui).
2. **Estágio 2 (novo):** janelas W com demanda URLLC (`tx_kpm(W)>0`) e margem URLLC ainda ausente entram numa **fila de pendentes** {W, state, action}. A cada iteração, o runtime testa ONLINE o fechamento do pool:
   > **pool fechado ⇔ delivered(W) == tx_kpm(W)** (ambos disponíveis no instante: tx do KPM fechado; delivered do stream append-only).
   - Fechado ⇒ `observe_outcome` com outcome **só-URLLC** (eMBB/mMTC None ⇒ R19 os exclui) ⇒ **um único update do modelo URLLC por janela**, com p95/miss completos (idênticos ao offline — provado 237/237).
   - Sem demanda (`tx=0`) ⇒ **NO_ACTIVE_DEMAND** (categoria separada, R19 vacuous — nunca contabilizada como janela válida).
   - Pool que nunca fecha (perda real) ⇒ **INCOMPLETE/UNKNOWN**; expira em MAX_AGE=30 janelas (3 s, só limite de fila; sem timeout de validade) e é descartado como vacuous — **sem-RX nunca é miss nem sucesso implícito**.
3. **Sem informação futura:** o update usa somente linhas de entregas já flushadas até a iteração corrente (poll incremental); associação com estado/ação HISTÓRICOS do pending; guard anti-duplicação por W.

## 2. Testes de integridade (executados sobre os 9 runs D1-v8, 12/11)

- **Coortes TX/RX alinhadas por janela:** `delivered` conta linhas do stream por janela de chegada `floor(txSec·10)`; `tx_kpm` soma `dTxPackets` slice-1 da mesma janela (`ts//100−1`, convenção congelada). Identidade de fechamento exige igualdade exata.
- **Duplicações:** stream append-only com flush por pacote; contagem por janela é livre de duplicação (ns-3: `rx_without_tag=0`); teste isolado: 0 dups de W em 237 updates; guard `_urllc_updated_wins`.
- **Bordas:** janelas com RX sem KPM (ex. λ6s2103 janela 2999, 92 pkts após o último callback) ⇒ **NO_KPM** ⇒ nunca atualizam (sem denominador). Janelas de cauda do run ⇒ expiram como vacuous.
- **Fração atualizável dentro de MAX_AGE=30:** k de fechamento p95 ≤ 19 em todos os runs ⇒ dos 6.777 pools que fecham, ~99,7% fecham dentro de MAX_AGE (6.756). Por run: λ2 98,7/82,5% · λ5 42,3/47,5/64,3% · λ6 48,9/49,5/58,9% (resto = perda real, INCOMPLETE).

## 3. Viés de seleção (quantificado, declarado — não corrigido, vedado inventar rótulo)

Comparação janelas ATUALIZÁVEIS (pool fecha; n=6.756) vs EXCLUÍDAS (nunca fecha; n=4.579) nos runs v8:

| métrica (mediana) | atualizável | excluída |
|---|---|---|
| buffer URLLC (B) | 69 | **19.235** |
| resourceShare (%) | 24,5 | **0,0** |
| miss offline (%) | 1,5 | **100** |
| p95 offline (ms) | 8,1 | 236,0 |

**Interpretação registrada:** o update URLLC ocorre preferencialmente no regime não-congestionado; janelas com perda (congestão/share 0) não alimentam o modelo — viés de censoring físico-real (não administrativo). Consequências: (i) o modelo URLLC aprenderá margens do regime saudável e continuará dependendo do prior no regime de perda; (ii) para a comparação B0×B1 o viés é **simétrico** (mesmo mecanismo, mesmos checkpoints); (iii) as categorias NO_ACTIVE_DEMAND e INCOMPLETE/UNKNOWN são reportadas separadamente em TODAS as análises. Este viés é a contrapartida honesta da regra "sem rótulo inventado".

## 4. Impacto sobre a cadeia (conforme ofício)

- **D1:** nova execução canônica **D1-v9** (9 runs, mesmas flags congeladas + `--urllc_delayed_update`, bootstrap-PROP intacto); **D1-v8 preservada** como versão anterior (diretório `d1_runs/` intocado; v9 em `d1_runs_v9_delayed/`). Novos indicadores: projeção §7.2, SLA por λ×seed, updates por slice, **latência das atualizações** (age), desempenho.
- **D2:** população e amostragem reconstruídas **da D1-v9** (mesmos critérios, estratos, seed 20261108 e orçamento 390 replays aprovados); estados da D1-v8 **não reutilizados**; replays parciais antigos preservados em `d2v2_pre_audit_pause/` (histórico, não misturados).
- **D3:** offline com modelos/estados congelados equivalentes; em futuras comparações online, B0 e B1 usarão **exatamente o mesmo mecanismo de observação/aprendizado** (incluindo a Emenda 4), diferindo somente a seleção de ações.

## 5. Critérios de aceitação da D1-v9 (pré-registrados)

1. 9/9 runs completos (mesmos gates de completude do run_d1: KPM ≥290.000, ≥600 decisões, exit 0).
2. Updates URLLC > 0 em todos os runs com demanda; 0 duplicatas de W; ages dentro de MAX_AGE.
3. eMBB/mMTC com contagem de updates inalterada em relação ao padrão do v8 (estágio 1 intacto).
4. Indicadores reconciliados por universo explícito; viés de seleção reportado com a tabela do §3.
