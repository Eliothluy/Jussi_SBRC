# Contrato de Interface ns-3 ↔ Python — Cenário SQR (Fase 0, CONGELADO)

> **Versão:** 1.2 (consolidação Fase 1) — 13/10/2026 · **Responsáveis:** Agente B (redator) + Agente A (validador C++ — aceite da Fase 0, `PLANO_IMPLEMENTACAO_SQR_NORI.md` §3-Fase 0).
> **Base:** `AGENTS.md` §4 (formatos vigentes, L97–107) — este documento **congela** o contrato vigente e suas **extensões** para o cenário SQR. Nada aqui modifica `scratch/rslaq` nem o core do ns-o-ran-gym; o cenário novo (`scratch/sqr/sqr-sim.cc`) e o scheduler novo (`scratch/sqr/`, ver §3.3) clonam/estendem o mecanismo.
> **Mudanças pós-congelamento** exigem aprovação do orquestrador e bump de versão deste arquivo (AGENTS §7.1).
> **v1.1:** correção crítica da semântica de execução da cota (§3.3) — o caminho vigente renormaliza e tornaria a economia de RBs do artigo inobservável; tratamento de missing do SNR congelado (§4); referências cruzadas ao register R24–R27.
> **v1.2 (13/10/2026, emendas RATIFICADAS pelo orquestrador — register v1.3):** §3.2 — C = **C EFETIVO alinhado a RBG lido da telemetria** (`total_prb`; 53 @ 10 MHz μ=0, 104 @ 20 MHz — R01); §4 — `windowThroughputMbps` = delta ÷ **tempo ATIVO na janela** (não período cheio); alinhamento de janela do log de ações (`prb_requested/prb_executed` descrevem a janela drenada — implementação `sqr-sim.cc`, dreno antes da espera IPC) e degradação §5 com contadores exportados no fim do run (slice sem linha ⇒ 33,33 %/slice com warning; CSV ausente ⇒ mantém cota anterior).

---

## 1. Visão geral do loop (com divergências do caminho vigente marcadas)

```
ns-3 (C++, sqr-sim.cc — clone do mecanismo de rslaq-sim.cc :: KpmAndControlCallback, L1158–1383)
  1. ao fim de cada JANELA (100 ms simulados): escreve KPMs em sqr-kpms.txt
     (header de referência em rslaq-sim.cc:1227 — formato na §4 abaixo)
  2. sem_post(semMetricsReady)                      (rslaq-sim.cc:~1282)
  3. sem_timedwait(semControl, 50 ms)               (rslaq-sim.cc:~1296)
       → timeout: warning "no action from agent", MANTÉM config anterior
  4. lê rslaq_actions_for_ns3.csv                   (referência: rslaq-sim.cc:1300)
  5. chama o scheduler SQR com a COTA ABSOLUTA (§3.3) — NÃO renormaliza
     ⚠️ DIVERGÊNCIA CONGELADA: o caminho vigente do rslaq divide cada percentual
        por totalDed (soma dos percentuais) e renormaliza para soma 1
        (rslaq-sim.cc:1343–1364); o RslaqMacScheduler::SetSliceConfiguration
        renormaliza internamente de novo (rslaq-mac-scheduler.cc:272–283) e o
        particionamento redistribui TODOS os totalRbgs (rslaq-mac-scheduler.cc:
        1119–1143; assert budgetTotal == totalRbgs em ~L1150). O sqr-sim NÃO
        herda esse comportamento — ver §3.3 (register R27).

Python (ns-o-ran-gym; RslaqEnv como padrão de loop, consumidor novo src/nsoran/sqr/)
  1. bloqueia em metricsReadySemaphore.acquire(timeout=10 s)
  2. parse do arquivo de KPMs (wrapper novo em src/nsoran/sqr/; core intocado)
  3. agente SQR decide a = (a_1..a_L) e projeta (§3.2, forma de Π em R24)
  4. action_controller.create_control_action escreve o CSV de ações (§3)
  5. controlSemaphore.release()
```

Semáforos POSIX nomeados `/sem_metrics_*` / `/sem_control_<simUuid>` — mecanismo e nomes herdados do RslaqEnv/action_controller, sem alteração.

## 2. Janela de decisão = 100 ms (100 TTIs μ=0)

- **Janela** = período de decisão do SQR = **100 ms de tempo simulado** = 100 TTIs de 1 ms (numerologia μ=0, register R03) = "100 slots" do artigo [ARTIGO, Seção V: "modelo atualizado a cada 100 slots e cada atualização leva menos de 100 ms"].
- KPMs são **acumulados por janela** (deltas de bytes/pacotes desde a janela anterior) e emitidos **uma vez por janela**, ao fim dela — em contraste com o ciclo de 10 ms do cenário `rslaq-sim` atual (parametrização `periodMs` no `sqr-sim`).
- A ação escrita na janela t **vale para toda a janela t** (mantida constante, como no artigo: "cada passo representa um período de decisão durante o qual uma alocação de recursos é mantida constante" — [ARTIGO, Seção III-A]).
- **Casualidade (sem vazamento — GUIA §3.6):** a observação disponível no início de t contém apenas KPIs de janelas ≤ t−1; y_t (KPI realizado) só é usado no update do regressor **após** a janela t, e com a ação **final** executada (z_t, a_final, y_t — [ARTIGO, Algoritmo 1, T:L177]; register R22).

## 3. Ação (formato INALTERADO — semântica de execução congelada em §3.3)

### 3.1 Formato

**Arquivo:** `rslaq_actions_for_ns3.csv` — **uma linha por slice**, L = 3 linhas:

```
timestamp,sliceId,dedicatedPRB,minPRB,maxPRB[,algorithm]
```

- Exemplo vigente (AGENTS §4): `500,1,40.0,40.0,100.0,2`
- `dedicatedPRB`, `minPRB`, `maxPRB` são **percentuais 0–100** da capacidade da célula.
- `algorithm` (coluna opcional): 0 = RR (custo 1,0), 1 = PF (2,0), 2 = BCQI (2,0). **No cenário SQR é fixo** (register R17: PF padrão, RR alternativo) e **idêntico para todos os métodos comparados**; não faz parte do espaço de ação do SQR.
- `sliceId` ∈ {0, 1, 2} — 0-based, mesma convenção do KPM vigente (`rslaq-sim.cc:1234,1268` escreve o índice do slice diretamente).

### 3.2 Conversão SQR → CSV (congelada)

1. O agente SQR decide inteiros **a_i ∈ {0,…,C}**, com **Σ_i a_i ≤ C** garantido **no Python**, ANTES de escrever o CSV (projeção proporcional inteira Π — forma exata congelada no **register R24**: piso + maiores restos fracionários, desempate por índice de slice).
2. Conversão por linha de slice: **`dedicatedPRB_i = 100 · a_i / C`** (float, 0–100).
   - **C é o C EFETIVO alinhado a RBG, lido da telemetria** (emenda v1.2, RATIFICADA 13/10/2026 — register R01): `total_prb` do `sqr_prb_windows.csv` (= `totalRbg × PRBs/RBG`); **53 @ 10 MHz μ=0** (medido no smoke de 12/10/2026) e **104 @ 20 MHz** na variante. O Python NÃO deve hardcodar C: lê `total_prb` da telemetria. O CSV continua agnóstico a C.

### 3.3 Semântica de execução da cota no MAC (NOVO em v1.1 — apontamento crítico do revisor; register R27)

- **Cota absoluta**: `dedicatedPRB_i` representa `a_i` PRBs **alocados** à slice i [semântica do artigo, Seção III-B, T:L54]; **Σa_i < C ⇒ os RBGs restantes ficam OCIOSOS (não servidos)**. Esta é a semântica que torna observável o efeito de `γ·a/C` (Fig. 4, economia de RBs).
- **Proibido herdar a renormalização vigente**: os caminhos atuais (i) dividem os percentuais por `totalDed` e renormalizam para soma 1 (`rslaq-sim.cc:1343–1364`), (ii) `SetSliceConfiguration` renormaliza internamente (`rslaq-mac-scheduler.cc:272–283`) e (iii) o particionamento redistribui todos os `totalRbgs` por maiores restos com assert `budgetTotal == totalRbgs` (`rslaq-mac-scheduler.cc:1119–1143, ~1150`). Com qualquer um deles, `Σ dedicatedPRB < 100` é silenciosamente re-expandido para partição completa e, com tráfego greedy (R04), `prb_scheduled ≈ C` sempre.
- **Mecanismo congelado**: scheduler **novo em `scratch/sqr/`** (subclasse da linhagem `RslaqMacScheduler`/`NrMacSchedulerOfdma`), recebendo a cota absoluta do `sqr-sim` **sem renormalização**; RBGs não cobertos por nenhum orçamento de slice não são agendados. Regra de ouro preservada: **nenhum arquivo em `scratch/rslaq` é alterado** (AGENTS §5.2).
- **Critérios de validação no smoke (go/no-go da Fase 1)**:
  1. `prb_scheduled ≈ a_i` por slice (cota absoluta, não a renormalizada);
  2. teste determinístico: ação com Σa = 0,5·C ⇒ Σ prb_scheduled ≤ 0,5·C + L (L = nº de slices; folga de arredondamento);
  3. nenhum dado servido em RBG ocioso;
  4. build com `scratch/rslaq` intocado.
- **Fallback (decisão reservada ao orquestrador)**: se a subclasse não for viável na Fase 1, escolher entre coerção **Σa = C** (documentando a perda da semântica do artigo para a Fig. 4 e do objetivo do CMDP) ou outra partição; **nenhuma das duas pode ser adotada unilateralmente pelo implementador**.

### 3.4 Pisos/tetos por slice (min/max)

Congelado para o SQR: **`minPRB_i = maxPRB_i = dedicatedPRB_i`** (cota exata, sem pisos/tetos adicionais). ⚠️ Semântica exata de min/max dentro do scheduler a ser **validada pelo Agente A na Fase 1**; a leitura "cota exata" deve ser confirmada no smoke junto com os critérios de §3.3.

### 3.5 Logs de auditoria (por passo/janela)

`prb_requested` (pré-projeção), `prb_executed` (pós-projeção/conversão), `prb_scheduled` (RBGs reais do MAC), **`prb_idle`** (RBGs ociosos = C − Σ prb_scheduled) — PLANO Fase 1; essenciais para os critérios de §3.3.

## 4. KPM (formato estendido — 3 colunas novas por UE)

**Arquivo:** `sqr-kpms.txt` (nome novo no cenário SQR para não colidir com runs rslaq; formato idêntico ao contrato) — **uma linha por UE por janela**:

```
timestamp,ueImsi,sliceId,dTxBytes,dRxBytes,plr,resourceSharePct,dLostPackets,throughputMbps,bufferBytes,windowThroughputMbps,snrDb,queueBytes
```

(colunas 1–10 = header vigente, `rslaq-sim.cc:1227`; colunas 11–13 = **novas, congeladas aqui**)

| Col. | Nome | Unidade | Definição congelada |
|---|---|---|---|
| 11 | `windowThroughputMbps` | Mbps | Throughput do UE **na janela**: `dRxBytes_da_janela × 8 / tempo_ativo_na_janela / 10^6` (**emenda v1.2, RATIFICADA 13/10/2026** — o denominador é o tempo ATIVO do UE na janela, não o período cheio: janela parcial = desde a entrada (`trafficStart`) ou até a saída do UE; janela cheia = período de 100 ms; implementado no `sqr-sim`). A coluna 10 (`throughputMbps`) mantém a semântica legada (delta ÷ período cheio) por compatibilidade de ferramentas |
| 12 | `snrDb` | dB | Último SNR wideband (do PHY/gNB) do UE na janela; se indisponível no fim da janela, o mais recente conhecido. **Sentinela de fio: `-999`** quando não há valor algum. **Tratamento do missing congelado (register R09):** o Python NUNCA alimenta o estado com o sentinela — imputação determinística: último `snrDb` válido do UE; se nenhum, mediana da célula na janela anterior; se nenhum, **15 dB** (centro da escala). O valor imputado entra em `snr_norm = clip(snr/30, 0, 1)` e cada imputação é logada (taxa por janela). (Correção v1.1: clip(−999/30,0,1)=0 seria indistinguível de SNR real de 0 dB.) |
| 13 | `queueBytes` | bytes | Backlog total do UE (buffers RLC UM + PDCP/SDU pendentes) amostrado ao **fim** da janela |

- `timestamp` = tempo simulado em ms no **fim** da janela (inteiro, mesma convenção vigente: `nowMs`, `rslaq-sim.cc:1268`).
- `dTxBytes/dRxBytes/dLostPackets` = deltas da janela (semântica vigente, delta-based — `rslaq-sim.cc:1239-1244`).
- **Derivações ficam NO Python (nada disso vai no arquivo):** `lowest-user-throughput` (mín de `windowThroughputMbps` por slice), `p5` (5º percentil por slice → y_t, R18), `nº de UEs por slice` (contagem de linhas), violação per-UE (R19), agregados globais (PRBs das outras slices, total de UEs → s̄).
- UEs que entram/saem no meio da janela aparecem na janela seguinte com deltas desde a entrada; janelas sem UE em uma slice ⇒ **nenhuma linha para ela** ⇒ período excluído do denominador de QoS dessa slice (R19).

## 5. Falhas, timeouts e degradação (herdados, congelados)

| Evento | Comportamento congelado |
|---|---|
| Python não responde em 50 ms (`sem_timedwait` timeout) | C++ loga warning e **mantém a configuração anterior** do scheduler (rslaq-sim.cc:~1296) |
| CSV de ação ausente/malformado | Defaults do caminho vigente (33,33 %/slice, PF — rslaq-sim.cc:1305-1312) + warning; **o Python nunca deve contar com isso** — ele sempre escreve o CSV após projeção |
| KPM ausente no `acquire(timeout=10 s)` do Python | Erro contável; política de fallback determinística e logada (GUIA §9.5) |
| Parse do KPM | Feito por wrapper **novo** em `src/nsoran/sqr/` sobre o padrão existente (`rslaq_kpis.parse_kpm_file`) — core intocado, AGENTS §5.2 |

## 6. Responsabilidades e paralelismo

- **Agente A (C++):** `scratch/sqr/sqr-sim.cc` — janela de 100 ms, churn Poisson de UEs, KPMs com as colunas 11–13, **passagem de cota absoluta sem renormalização**, log `prb_requested/prb_executed/prb_scheduled/prb_idle`, implementação/validação do scheduler de cota absoluta em `scratch/sqr/` (§3.3), validação da semântica min/max (§3.4).
- **Agente B (Python):** consumidor/produtor do contrato via **mock** (JSON/CSV sintético com o mesmo schema) enquanto a Fase 1 avança — paralelismo total pós-congelamento (AGENTS §7.2); conversão a_i→%, projeção Π (R24), derivadas de KPM (p5, mín, contagens), imputação de SNR (§4).
- **Gatilho de aceite da Fase 0** (PLANO §3-Fase 0): contrato revisado por A e B; carregador de config do agente **falha explicitamente** se campo essencial (C, banda, paths de arquivo, período, SLAs) estiver nulo.

## 7. Parâmetros externos ao arquivo (referência cruzada)

C (R01), banda (R02), numerologia/duplex (R03), período 100 ms (§2), SLAs por UE 0,5/1,0/1,5 Mbps `[ARTIGO, Tabela I]`, δ=1% `[ARTIGO, Seção III-C]`, semântica da cota no MAC (R27), forma de Π (R24), tratamento de missing de SNR (R09) — valores em `assumptions_register.md`; fontes publicadas em `source_traceability.md`.

---
*Histórico: v1.0 (11/10/2026, Agente B) — congelamento na Fase 0; formatos-base copiados verbatim de AGENTS.md §4 (L97–103); extensões conforme AGENTS.md §4 (L105–107) e PLANO §3-Fase 0 item 3. **v1.1 (11/10/2026, Agente B)** — apontamentos do revisor: (CRÍTICO) nova §3.3 com semântica de cota ABSOLUTA e scheduler novo em `scratch/sqr/`, com as renormalizações do caminho vigente documentadas por arquivo:linha e critérios de validação + fallback ao orquestrador (register R27); (BAIXO-MÉDIO) sentinela −999 + imputação determinística de SNR congelados em §4 (register R09); (MÉDIO) referências a R24 (Π), R25 (γ=0), R26 (agregação), R22 (atualização real com a_final). **v1.2 (13/10/2026, Agente A — consolidação da Fase 1, emendas RATIFICADAS pelo orquestrador):** §3.2 C efetivo por telemetria (R01: 53/104); §4 `windowThroughputMbps` por tempo ativo; nota de implementação sobre alinhamento de janela (dreno ANTES da espera IPC no `KpmAndControlCallback`) e degradação §5 contável (33,33 %/slice p/ linha ausente; mantém cota p/ CSV ausente) com contadores `IPC_DEGRADATION` no log de fim de run.*
