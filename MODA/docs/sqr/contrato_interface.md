# Contrato de Interface ns-3 ↔ Python — Cenário SQR (Fase 0, CONGELADO)

> **Versão:** 1.4 (Rodada 2, Etapa 2.1) — 20/10/2026 · **Responsáveis:** Agente B (redator) + Agente A (validador C++ — aceite da Fase 0, `PLANO_IMPLEMENTACAO_SQR_NORI.md` §3-Fase 0).
> **Base:** `AGENTS.md` §4 (formatos vigentes, L97–107) — este documento **congela** o contrato vigente e suas **extensões** para o cenário SQR. Nada aqui modifica `scratch/rslaq` nem o core do ns-o-ran-gym; o cenário novo (`scratch/sqr/sqr-sim.cc`) e o scheduler novo (`scratch/sqr/`, ver §3.3) clonam/estendem o mecanismo.
> **Mudanças pós-congelamento** exigem aprovação do orquestrador e bump de versão deste arquivo (AGENTS §7.1).
> **v1.1:** correção crítica da semântica de execução da cota (§3.3) — o caminho vigente renormaliza e tornaria a economia de RBs do artigo inobservável; tratamento de missing do SNR congelado (§4); referências cruzadas ao register R24–R27.
> **v1.2 (13/10/2026, emendas RATIFICADAS pelo orquestrador — register v1.3):** §3.2 — C = **C EFETIVO alinhado a RBG lido da telemetria** (`total_prb`; 53 @ 10 MHz μ=0, 104 @ 20 MHz — R01); §4 — `windowThroughputMbps` = delta ÷ **tempo ATIVO na janela** (não período cheio); alinhamento de janela do log de ações (`prb_requested/prb_executed` descrevem a janela drenada — implementação `sqr-sim.cc`, dreno antes da espera IPC) e degradação §5 com contadores exportados no fim do run (slice sem linha ⇒ 33,33 %/slice com warning; CSV ausente ⇒ mantém cota anterior).
> **v1.3 (20/10/2026, Rodada 2 — Etapa 2.1, Agente A; ADITIVA, não toca nas 13 colunas congeladas):** §4.1 — colunas 14–17 (`dTxPackets,dRxPackets,dDelaySumSec,cqiWideband`) no FIM de cada linha de `sqr-kpms.txt`/`sqr_kpm_history.csv` + CSV novo `sqr_packet_delays.csv` (latência por pacote); sentinela do CQI = −1 (register R33); escolhas de instrumentação no register R32.
> **v1.4 (20/10/2026, Rodada 2 — Etapa 2.1 parte 2, Agente A; ADITIVA, NENHUM header de arquivo muda):** §4.2 — slices heterogêneos: flags `--sliceLambdas` (3 processos de Poisson INDEPENDENTES, default λ/3 por slice — register R34) e `--sliceApps` (perfis `greedy|urllc|mmtc`, default greedy×3; parâmetros URLLC/mMTC no register R35); a deadline do URLLC vai como **METADADO no header do run** (escolha registrada em §4.2), não como coluna nova.

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

### 4.1 Contrato v1.3 — colunas 14–17 e `sqr_packet_delays.csv` (Rodada 2, Etapa 2.1)

**Mudança ADITIVA (register R32/R33):** as 13 colunas da §4 permanecem idênticas e na MESMA ordem; 4 colunas novas SOMENTE NO FIM de cada linha, em **ambos** `sqr-kpms.txt` e `sqr_kpm_history.csv`:

```
timestamp,ueImsi,sliceId,dTxBytes,dRxBytes,plr,resourceSharePct,dLostPackets,throughputMbps,bufferBytes,windowThroughputMbps,snrDb,queueBytes,dTxPackets,dRxPackets,dDelaySumSec,cqiWideband
```

| Col. | Nome | Unidade | Definição congelada |
|---|---|---|---|
| 14 | `dTxPackets` | pacotes | Delta da janela do contador FlowMonitor `txPackets` por UE — mesmo padrão de delta/janela parcial das colunas 4–6 (`emitRow`, `scratch/sqr/sqr-sim.cc`) |
| 15 | `dRxPackets` | pacotes | Delta da janela do contador `rxPackets` por UE (idem col. 14) |
| 16 | `dDelaySumSec` | s | Delta da janela do `delaySum` **acumulado** do FlowMonitor por UE (6 casas decimais). Latência média por UE-janela = `1000·dDelaySumSec/dRxPackets` — derivada no Python |
| 17 | `cqiWideband` | índice 0–15 | CQI wideband **CRU** (inteiro) que embasa a inversão da col. 12 (`snrDb`), amostrado no MESMO snapshot do scheduler (`GetUeDlCqiWideband`); **sentinela de fio −1 = sem CQI** (register R33). A col. 12 e sua sentinela −999 permanecem INALTERADAS |

- **Derivações por UE-janela (Python, `nsoran.sqr.sqr_kpis.UeKpmRow`):** `latencyMeanMs = 1000·dDelaySumSec/dRxPackets` e `pdrWindow = dRxPackets/dTxPackets` — expostas como `None` quando o denominador não é > 0. O parser aceita arquivos de 13 **ou** 17 colunas (`SQR_KPM_COLUMNS_V13` opcional por nome; arquivo de 13 colunas ⇒ defaults 0.0/−1, sem quebra de API).
- **`sqr_packet_delays.csv` (NOVO)** — `results_sqr_network_only/seed=X_run=Y/sqr_packet_delays.csv`; header `ueImsi,sliceId,txSec,delaySec` (txSec/delaySec com 6 casas decimais); **uma linha por pacote ENTREGUE** (Rx no app do UE). Mecanismo (`scratch/sqr/sqr-sim.cc`, register R32): ByteTag `SqrTxTimestampTag` com o instante de TX, adicionada no trace `Ipv4L3Protocol::SendOutgoing` do remoteHost — dispara APÓS a cópia do `UdpSocketImpl` (`udp-socket-impl.cc:602`), logo o objeto traceado é o que viaja, e ANTES do `packet->Copy()` de `SendRealOut` (`ipv4-l3-protocol.cc:784`); callback de Rx no trace `Rx` do `UdpServer` de cada UE mede `delaySec = rxSec − txSec` (latência E2E app-a-app: backhaul P2P + encapsulamento GTP-U + fila/fase rádio). **ByteTag (não PacketTag)** porque a reassemblagem RLC-UM reconstrói o PDCP PDU via `Packet::AddAtEnd` (`contrib/nr/model/nr-rlc-um.cc:406`), que não funda packet tags — com segmentação RLC (UEs de borda) a tag se perderia; byte tags são fundidas com ajuste de offset. O buffer é drenado junto com o KPM periódico e no fim do run (cauda da última janela); auditoria no fim do run: `PACKET_DELAYS delivered_with_tag / rx_without_tag` (esperado 0 sem tag).
- **Chave `ueImsi` do CSV de delays** = o MESMO identificador da col. 2 do KPM (ueId sequencial herdado do rslaq; = IMSI real apenas quando coincidem) — o join `sqr_packet_delays × sqr_kpm_history` é direto por `ueImsi`.
- **Caminho E2 (Fases 5.3/5.4):** a indication E2SM-KPM (`SqrE2KpmUeRecord`) NÃO foi estendida nesta etapa — `xapp_sqr.write_kpm_window` segue materializando 13 colunas; as colunas 14–17 são OPCIONAIS no parser. Pendência para a Rodada 2 caso latency/PDR sejam exigidos no caminho E2 (exigiria estender `SqrE2KpmUeRecord` + re-rodar a paridade IPC×E2).
- **p95/p99 e deadline-miss** (protocolo Rodada 2 §6–7) derivam de `sqr_packet_delays.csv` no Python; nada disso é calculado no C++.

### 4.2 Contrato v1.4 — slices heterogêneos: chegadas e perfis de tráfego por slice (Rodada 2, Etapa 2.1 parte 2)

**Mudança ADITIVA (register R34/R35):** nenhuma coluna de `sqr-kpms.txt`, `sqr_kpm_history.csv`, `sqr_prb_windows.csv`, `sqr_summary.csv` ou `sqr_packet_delays.csv` muda; TODOS os perfis de tráfego alimentam as 13+4 colunas do KPM (§4/§4.1) e o CSV de delays por pacote **sem distinção**. Retrocompatibilidade congelada e verificada: com `--lambdaUesPerMin` sozinho (flags novas ausentes), os 5 headers (`sqr-kpms.txt`, `sqr_kpm_history.csv`, `sqr_prb_windows.csv`, `sqr_summary.csv`, `sqr_packet_delays.csv`) são **byte-idênticos** aos da v1.3 (diff executado pelo Agente A em 20/10/2026: binário pré-edição × pós-edição, run de 20 s, seed=21, raio 300 m — 5/5 idênticos).

**Flags novas (`scratch/sqr/sqr-sim.cc`):**

| Flag | Default | Semântica congelada |
|---|---|---|
| `--sliceLambdas="a,b,c"` | vazio/ausente | Taxa de chegada (UEs/min) de **3 processos de Poisson INDEPENDENTES** (inter-chegada Exp por slice, mesmos alinhamento TDD de 7 ms — R29 — e corte em `simTime − drain` do pré-2.1); **a chegada pertence à slice do próprio processo** (fim do sorteio uniforme global). Valor `0` desabilita a slice; soma 0 ou nº de valores ≠ 3 ⇒ `NS_FATAL`. **Default vazio ⇒ λ/3 por slice** (register R34): a superposição de 3 Poisson(λ/3) independentes é um Poisson(λ) total com marca de slice uniforme — **estatisticamente idêntico** ao comportamento pré-2.1 (processo único + sorteio uniforme; R13 leitura "taxa global"). NOTA: a retrocompatibilidade congelada é de CLI, semântica estatística e FORMATOS; **trajetórias para um mesmo seed NÃO são bit-idênticas** ao pré-2.1 (a estrutura de consumo dos streams RNG mudou: 3 cadeias de inter-chegada substituem o processo único + o sorteio `UniformRandomVariable` por chegada) |
| `--sliceApps="p0,p1,p2"` | `greedy,greedy,greedy` | Perfil de tráfego por slice (case-insensitive; valor inválido ou ≠ 3 valores ⇒ `NS_FATAL`): `greedy` = UDP full-buffer R04 (taxa `--ueOfferedMbps`, pacote `--pktSize` — comportamento pré-2.1); `urllc` = pacotes pequenos **periódicos**; `mmtc` = pacotes pequenos **esparsos**. Todos via `OnOffHelper` CBR (`SetConstantRate`) com `pktSize` do perfil — pipeline FlowMonitor/ByteTag/KPM inalterado |
| `--urllcPktBytes` | 32 | Payload UDP por pacote URLLC (B); headers IP/UDP/GTP somam por baixo [ASSUMIDO R35] |
| `--urllcIntervalMs` | 1,0 | Período entre pacotes URLLC (ms) ⇒ 32 B @ 1 ms = 256 kbps/UE [ASSUMIDO R35] |
| `--urllcDeadlineMs` | 10,0 | **Deadline de ANÁLISE** (ms): o escalonador NÃO é deadline-aware; o valor não altera a dinâmica — vai como metadado do run (abaixo). deadline-miss ratio = fração de amostras do `sqr_packet_delays.csv` da slice URLLC com `delaySec > deadline/1000`, computado no Python [ASSUMIDO R35] |
| `--mmtcPktBytes` | 200 | Payload UDP por pacote mMTC (B) [ASSUMIDO R35] |
| `--mmtcIntervalMs` | 1000,0 | Período entre pacotes mMTC (ms) ⇒ 200 B @ 1 s = 1,6 kbps/UE [ASSUMIDO R35] |

- **Taxa oferecida dos perfis não-greedy DERIVA dos parâmetros** (`pktBytes×8/intervalo`): `--ueOfferedMbps`/`--ueOfferedMult` aplicam-se SOMENTE a slices `greedy` (fallback legado mult×SLA inclusive). A coluna `offeredMbps` do `sqr_summary.csv` reporta a taxa efetiva do perfil; as colunas `slaMbps`/`slaMetMeanThr` mantêm a semântica de throughput congelada — **para slices não-greedy essa comparação NÃO é o KPI** (ver próximo item).
- **SLAs heterogêneos (latência/PDR) ficam na análise Python** (protocolo Rodada 2 §9–10: normalizações `S_i`/`D_i`): o C++ garante que os DADOS por slice existam (latência por pacote no `sqr_packet_delays.csv`; PDR por `dRxPackets/dTxPackets` do §4.1); `--slaMbps` (Mbps por slice) continua válido para eMBB.
- **Escolha registrada — deadline do URLLC como METADADO no header do run** (e não como coluna extra no `sqr_packet_delays.csv`): preserva os 5 headers byte-idênticos (critério de retrocompatibilidade desta etapa). Onde o metadado aparece: (a) banner de abertura do run — linhas `UE churn` (lambdas por slice), `Slices` (apps por slice) e `Traffic cfg` (parâmetros urllc/mmtc + deadline); (b) linha final do relatório de fim de run, machine-parseable do log:
  `[SqrSim] SLA_PROFILES slice_lambdas_per_min=6.000,6.000,6.000 apps=greedy,urllc,mmtc urllc_pkt_bytes=32 urllc_interval_ms=1.000 urllc_deadline_ms=10.000 mmtc_pkt_bytes=200 mmtc_interval_ms=1000.000 (contract v1.4 §4.2)`
- **Log de chegada estendido** (stdout, não é contrato de arquivo): `UE_ARRIVAL ... profile=<greedy|urllc|mmtc> ... pkt=<B> [urllc_deadline_ms=<ms>]`.
- **Caminho E2 (Fases 5.3/5.4):** sem mudança nesta etapa — a indication E2SM-KPM (`SqrE2KpmUeRecord`) continua 13 campos (pendência já registrada na §4.1); os perfis não introduzem campos novos na E2.

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
*Histórico: v1.0 (11/10/2026, Agente B) — congelamento na Fase 0; formatos-base copiados verbatim de AGENTS.md §4 (L97–103); extensões conforme AGENTS.md §4 (L105–107) e PLANO §3-Fase 0 item 3. **v1.1 (11/10/2026, Agente B)** — apontamentos do revisor: (CRÍTICO) nova §3.3 com semântica de cota ABSOLUTA e scheduler novo em `scratch/sqr/`, com as renormalizações do caminho vigente documentadas por arquivo:linha e critérios de validação + fallback ao orquestrador (register R27); (BAIXO-MÉDIO) sentinela −999 + imputação determinística de SNR congelados em §4 (register R09); (MÉDIO) referências a R24 (Π), R25 (γ=0), R26 (agregação), R22 (atualização real com a_final). **v1.2 (13/10/2026, Agente A — consolidação da Fase 1, emendas RATIFICADAS pelo orquestrador):** §3.2 C efetivo por telemetria (R01: 53/104); §4 `windowThroughputMbps` por tempo ativo; nota de implementação sobre alinhamento de janela (dreno ANTES da espera IPC no `KpmAndControlCallback`) e degradação §5 contável (33,33 %/slice p/ linha ausente; mantém cota p/ CSV ausente) com contadores `IPC_DEGRADATION` no log de fim de run.* **v1.3 (20/10/2026, Agente A — Rodada 2, Etapa 2.1, instrumentação de latência/PDR/CQI):** nova §4.1 — colunas 14–17 **ADITIVAS** ao fim da linha em `sqr-kpms.txt`/`sqr_kpm_history.csv` (`dTxPackets,dRxPackets,dDelaySumSec,cqiWideband`; sentinela do CQI = −1, register R33; snrDb/−999 inalterados) + CSV novo `sqr_packet_delays.csv` por pacote entregue (ByteTag de TX no `SendOutgoing` do remoteHost + Rx no `UdpServer`, dreno junto ao KPM; register R32); parser `sqr_kpis` aceita 13 ou 17 colunas e expõe `latencyMeanMs`/`pdrWindow`; indication E2 (13 campos) NÃO estendida nesta etapa — pendência registrada.* **v1.4 (20/10/2026, Agente A — Rodada 2, Etapa 2.1 parte 2, slices heterogêneos):** nova §4.2 — `--sliceLambdas` (3 processos de Poisson INDEPENDENTES por slice; default vazio ⇒ λ/3 por slice, estatisticamente idêntico ao pré-2.1 — register R34; trajetórias de mesmo seed NÃO são bit-idênticas, mudança documentada de consumo de streams RNG) e `--sliceApps` (perfis `greedy|urllc|mmtc`, default greedy×3; parâmetros `--urllcPktBytes/--urllcIntervalMs/--urllcDeadlineMs/--mmtcPktBytes/--mmtcIntervalMs` — register R35); deadline do URLLC como **METADADO no header do run** (banner + linha `SLA_PROFILES`), NÃO como coluna extra — critério de retro: com `--lambdaUesPerMin` sozinho, os 5 headers byte-idênticos à v1.3 (diff pré×pós edição executado); nenhum header de arquivo muda.*
