# Relatório da Etapa 2.1 — Instrumentação dos slices heterogêneos (Rodada 2)

> **Data**: 2026-10-20 · **Escopo**: `rodada_2_fronteira_viabilidade_sqr.md` §5 ("os três slices deverão possuir KPIs diferentes") — instrumentação de latência/PDR/CQI (contrato **v1.3**) + slices heterogêneos de chegada e perfis de tráfego (contrato **v1.4**).
> **Status**: ✅ **CONCLUÍDA** — 2 estágios de implementação (KPM v1.3, slices v1.4) com gates OK + tabela de SLAs da literatura fechada (gate §8 do protocolo).
> **Rastreabilidade**: contrato `MODA/docs/sqr/contrato_interface.md` §4.1–§4.2 (v1.3/v1.4) e `MODA/docs/sqr/assumptions_register.md` v1.5, linhas **R32–R35**.

---

## 1. O que foi implementado

### 1.1 Colunas KPM 14–17 (contrato v1.3 §4.1; register R33)

Quatro colunas **ADITIVAS ao FIM** de cada linha de `sqr-kpms.txt` e `sqr_kpm_history.csv` — as 13 colunas congeladas permanecem idênticas e na mesma ordem:

| Col. | Nome | Unidade | Definição |
|---|---|---|---|
| 14 | `dTxPackets` | pacotes | Delta da janela do FlowMonitor `txPackets` por UE (mesma regra de janela parcial das col. 4–6) |
| 15 | `dRxPackets` | pacotes | Delta da janela do `rxPackets` por UE |
| 16 | `dDelaySumSec` | s | Delta do `delaySum` **acumulado** por UE (6 casas decimais); latência média por UE-janela = `1000·dDelaySumSec/dRxPackets`, derivada **no Python** |
| 17 | `cqiWideband` | índice 0–15 | CQI wideband **CRU** amostrado no MESMO snapshot do scheduler que embasa a inversão do `snrDb` (`GetUeDlCqiWideband`, `scratch/sqr/rslaq-mac-scheduler.h:199` / `rslaq-mac-scheduler.cc:1625` + snapshot no `sqr-mac-scheduler.cc`); **sentinela de fio −1 = sem CQI válido** (R33). A col. 12 (`snrDb`) e sua sentinela −999 permanecem inalteradas |

Parser (`MODA/ns-o-ran-gym/src/nsoran/sqr/sqr_kpis.py`): colunas opcionais por nome (`SQR_KPM_COLUMNS_V13`, `sqr_kpis.py:87`); aceita arquivos de 13 **ou** 17 colunas (13 colunas ⇒ defaults 0,0/−1, sem quebra de API); expõe `latencyMeanMs` e `pdrWindow = dRxPackets/dTxPackets` como `None` quando o denominador não é > 0 (`sqr_kpis.py:148`).

### 1.2 `sqr_packet_delays.csv` — latência por pacote (contrato v1.3 §4.1; register R32)

CSV novo (`<outputDir>/sqr_packet_delays.csv`), header `ueImsi,sliceId,txSec,delaySec` (6 casas decimais), **uma linha por pacote ENTREGUE**:

- **Mecanismo** (`scratch/sqr/sqr-sim.cc`): ByteTag `SqrTxTimestampTag` (8 B, ns; `sqr-sim.cc:358`) com o instante de TX, adicionada no trace `Ipv4L3Protocol::SendOutgoing` do remoteHost + callback de Rx no trace `Rx` do `UdpServer` de cada UE; `delaySec = rxSec − txSec`. Buffer drenado junto com o KPM periódico e no fim do run; auditoria de fim de run `PACKET_DELAYS delivered_with_tag / rx_without_tag` (`sqr-sim.cc:1915`).
- **ByteTag (não PacketTag)** porque a reassemblagem RLC-UM reconstrói o PDCP PDU via `Packet::AddAtEnd` (`contrib/nr/model/nr-rlc-um.cc:406`), que não funda packet tags — a tag sobrevive à segmentação RLC comum em UEs de borda.
- **Semântica**: `txSec` = saída do socket IP do remoteHost ⇒ latência **E2E app-a-app** (backhaul P2P + GTP-U + rádio), **não** latência somente-rádio (se exigida, precisa de novo ponto de tagueamento no gNB — R32).
- Chave `ueImsi` = mesmo identificador da col. 2 do KPM ⇒ join direto com `sqr_kpm_history.csv`. p95/p99 e deadline-miss derivam deste CSV **no Python**; nada é calculado no C++.

### 1.3 Flags `--sliceLambdas` / `--sliceApps` e defaults (contrato v1.4 §4.2; register R34/R35)

Implementado em `scratch/sqr/sqr-sim.cc` sem tocar em `contrib/nr`, `scratch/rslaq` ou `scratch/our_paper`:

| Flag | Default | Semântica |
|---|---|---|
| `--sliceLambdas="a,b,c"` | vazio/ausente ⇒ **λ/3 por slice** | 3 processos de Poisson **INDEPENDENTES** (inter-chegada Exp por slice, mesmo alinhamento TDD de 7 ms — R29 — e corte em `simTime − drain`); a chegada pertence à slice do próprio processo. `0` desabilita a slice; soma 0 ou nº de valores ≠ 3 ⇒ `NS_FATAL` (`sqr-sim.cc:607–619`). Default λ/3: superposição de 3 Poisson(λ/3) = Poisson(λ) total com marca de slice uniforme — **estatisticamente idêntico** ao pré-2.1 (leitura R13 "taxa global") |
| `--sliceApps="p0,p1,p2"` | `greedy,greedy,greedy` | Perfis por slice (`sqr-sim.cc:182–196`): `greedy` = UDP full-buffer R04 inalterado; `urllc` = **32 B a cada 1 ms** (256 kbps/UE), deadline de ANÁLISE **10 ms**; `mmtc` = **200 B a cada 1000 ms** (1,6 kbps/UE). Parâmetros ajustáveis: `--urllcPktBytes/--urllcIntervalMs/--urllcDeadlineMs/--mmtcPktBytes/--mmtcIntervalMs` (todos `[ASSUMIDO]`, R35). Valor inválido ou ≠ 3 valores ⇒ `NS_FATAL` (`sqr-sim.cc:681`) |

**Escolha de desenho (deadline do URLLC como METADADO, não coluna extra)**: preserva os 5 headers de arquivo byte-idênticos (critério de retrocompatibilidade da etapa). Onde aparece: banner de abertura (`UE churn`/`Slices`/`Traffic cfg`) e linha final machine-parseable `SLA_PROFILES` (`sqr-sim.cc:1923`). O escalonador **não** é deadline-aware — o valor não altera a dinâmica; deadline-miss = fração de amostras URLLC com `delaySec > deadline/1000`, computado no Python.

**⚠️ Decisão de interpretação pendente de ratificação (R34)**: o ask pedia "replicar `--lambdaUesPerMin` nos 3 slices" com retrocompatibilidade estrita; a leitura literal (λ em **cada** slice) triplicaria a carga (E[N] 2→6 pela nota de Little) e violaria a retrocompatibilidade. Implementou-se **default ausente ⇒ λ/3 por slice** (estatisticamente idêntico ao pré-2.1); valor **explícito** usa os literais por slice. Registrado em R34/contrato §4.2 para ciência do orquestrador. Corolário documentado: **trajetórias de mesmo seed NÃO são bit-idênticas** ao pré-2.1 (consumo de streams RNG mudou); a retro entregue é de CLI + semântica estatística + formatos.

### 1.4 Arquivos tocados (verificado por `git status --porcelain` nesta sessão)

- `MODA/ns-3-dev/scratch/sqr/`: `sqr-sim.cc`, `rslaq-mac-scheduler.{h,cc}`, `sqr-mac-scheduler.cc`
- `MODA/ns-o-ran-gym/`: `src/nsoran/sqr/sqr_kpis.py`, `tests/test_sqr_kpms.py` (+4 testes novos)
- `MODA/docs/sqr/`: `contrato_interface.md` (v1.3 §4.1 + v1.4 §4.2), `assumptions_register.md` (v1.4→v1.5, R32–R35)
- **Nada** em `contrib/nr`, `scratch/rslaq`, `scratch/our_paper` ou `src/` do ns-3; **nenhum commit realizado** (regra AGENTS §5 — commit só sob pedido).

---

## 2. Evidências dos gates

### 2.1 Gate KPM (v1.3) — **OK**

Evidência da etapa (executada pelo estágio KPM em 20/10, registrada em R32/R33):

- Build via `bash MODA/ns-3-dev/scratch/sqr/ns3_gate.sh build` — OK.
- Run network-only 20 s (`seed=21`, raio 300 m): header de **17 colunas** em `sqr-kpms.txt` e `sqr_kpm_history.csv`; **6029 pacotes com tag / `rx_without_tag=0`**; linhas de exemplo `dTxPk=566 dRxPk=28 dDelaySumSec=1.014508 cqiWideband=15` com `snrDb=24.04` (inversão consistente com CQI 15).
- Re-executado por mim nesta consolidação (verificação do artefato vivo): `head -1 MODA/ns-3-dev/sqr-kpms.txt | awk -F',' '{print NF}'` → **17 colunas**, header idêntico ao do contrato v1.3 §4.1.

### 2.2 Gate slices heterogêneos (v1.4) — **OK**

Evidência da etapa (executada pelo estágio slices em 20/10, registrada em R34/R35):

- Build via `ns3_gate.sh build` — OK.
- Run heterogêneo 30 s (`seed=22 --sliceLambdas=6,6,6 --sliceApps=greedy,urllc,mmtc`, 300 m): 6 chegadas, cada uma com a slice/perfil do próprio processo (`UE_ARRIVAL ... profile=...`); cadências no `sqr_packet_delays.csv`: URLLC 1 pacote/ms, mMTC 1 pacote/s; delay URLLC ≈ 5,1 ms < deadline 10 ms; KPM com linhas das 3 slices (480/246/213); `rx_without_tag=0`.
- Run retro 20 s (`--lambdaUesPerMin=18` sozinho): **5/5 headers byte-idênticos** ao binário pré-edição (`sqr-kpms.txt`, `sqr_kpm_history.csv`, `sqr_prb_windows.csv`, `sqr_summary.csv`, `sqr_packet_delays.csv`; diff executado pré×pós).
- `NS_FATAL` verificado para 2 valores em `--sliceLambdas` e perfil inválido em `--sliceApps`.

### 2.3 Gate testes Python — **OK (re-executado nesta sessão)**

Comandos executados por mim (dentro de `MODA/ns-o-ran-gym`, venv `.venv` 3.13):

- `bash tests/run_sqr_tests.sh` → **`194 passed, 1 warning in 6.15s`** (pytest 9.1.1, Python 3.13.12; inclui os 4 testes v1.3 de `test_sqr_kpms.py`: `test_v13_columns_derived_latency_and_pdr`, `test_v13_zero_denominators_yield_none`, `test_v13_cqi_sentinel_and_unparsable_pass_through_raw`, `test_legacy_13col_file_still_parses_with_v13_defaults` — `test_sqr_kpms.py:252,277,293,308`).
- `bash tests/run_full_tests.sh` → **`339 passed, 1 skipped, 1 warning in 8.44s`**.

**Nota de honestidade**: os runs de simulação ns-3 citados em 2.1/2.2 foram executados pelos estágios de implementação (evidência fornecida pela etapa e registrada em R32–R35) e **não foram re-executados** nesta consolidação; os checks que eu rodei aqui são os dois comandos pytest, a contagem de colunas do header vivo e o `git status`. Os CSVs `sqr_packet_delays.csv` das validações não estão retidos em disco: o `g_outputDir` default é o diretório corrente do run (`sqr-sim.cc:302`) e os dirs `results_sqr_network_only/seed={1,2,3}_run=1` são de 18/10 (pré-etapa, sem o CSV novo).

---

## 3. Tabela de SLAs da literatura (gate §8) — resumo

Fonte completa: `docs/artigo/sla_table_round2.md` (fechada 2026-10-20, 5 fontes **verificadas por leitura direta do documento** naquela sessão; 12 queries/tentativas registradas em `docs/literature/tools/search-log.jsonl`). Regra aplicada: alvo sem fonte aberta verificada **não entra como alvo** — entra como "não reportado".

### 3.1 Alvos verificados (âncoras citáveis)

| Slice | KPI | Alvo | Fonte |
|---|---|---|---|
| URLLC | latência | **≤ 500 ms** (Tabela I, Eq. 2) | REAL, IEEE ICC-W 2025 (arXiv:2502.00715v1) |
| URLLC | latência | **≤ 400 ms** (`L_tar`, Eq. 4) | DORA, IEEE MECOM 2025 (arXiv:2509.07242v1) |
| eMBB | throughput por UE | **T_tar = 7 Mbps** (Eq. 5) | DORA, §III-B |
| eMBB | bitrate (alvo de recompensa) | 2×10⁵–4×10⁵ (unidade não explicitada; bps = claim do registro) | REAL, Tabela I |
| mMTC | atendimento | **≥ 5 PRBs/dispositivo** (critério do autor, não normativo) | DORA, §IV-C/Eq. 6 |
| URLLC | latência user plane | **0,5 ms UL/DL** (§7.5) | 3GPP TR 38.913 v18.0.0 |
| URLLC | confiabilidade | **99,999 %** (1−10⁻⁵, 32 B, 1 ms) (§7.9) | 3GPP TR 38.913 v18.0.0 |
| eMBB | latência user plane | **4 ms UL/DL** (§7.5) | 3GPP TR 38.913 v18.0.0 |
| eMBB | user experienced rate | 50/25 (macro), 300/50 (denso), 1000/500 Mbit/s (indoor) — **valores-alvo, não requisitos** (Tabela 7.1-1, Nota 5) | 3GPP TS 22.261 v19.15.0 |
| URLLC industrial | disponibilidade + latência E2E | 99,999–99,99999 %; E2E < intervalo (500 µs/1/2 ms), payloads 20–50 B (Tabela 5.2-1) | 3GPP TS 22.104 v19.2.0 |
| URLLC (ITS) | E2E 30 ms / disp. 99,9999 % / conf. 99,999 % (Tabela 7.2.3.2-1) | idem | 3GPP TS 22.261 v19.15.0 |
| mMTC | densidade | 10⁶ dispositivos/km² (§7.17) | 3GPP TR 38.913 v18.0.0 |

### 3.2 Linhas NÃO fechadas (não usar como alvo citável)

- **ITU-R M.2410-0**: inacessível por automação (404 em 4 esquemas de URL; landing "not available"; diretório 403) — os valores canônicos "ITU" (1 ms URLLC / 4 ms eMBB / 99,999 % / 10⁶ km⁻² / 100 Mbit/s) **não entram** até o PDF ser aberto. Cobertura: TR 38.913 v18 (verificado) carrega os mesmos KPIs no 3GPP.
- **CVaR-PPO (IEEE OJ-COMS 2026, doi:10.1109/OJCOMS.2026.3699475)**: full-text bloqueado por bot-detection IEEE (3 tentativas); abstract completo (Semantic Scholar, CC-BY) sem nenhum alvo numérico → "não reportado".

### 3.3 Lacunas e implicações para o artigo novo

1. Nenhum dos 3 artigos (REAL/DORA/CVaR-PPO) **ancora seus alvos em normas** — 400/500 ms são parâmetros de testbed, ordens de grandeza mais frouxas que os 0,5–1 ms normativos URLLC. Declarar a divergência (cenário de borda/RLF) é contribuição de honestidade metodológica.
2. **PDR/reliability para mMTC**: sem alvo numérico citável em fonte aberta — qualquer % será `[ASSUMIDO]` (DORA usa critério de PRBs; REAL usa razão bytes recebidos/esperados).
3. **Proposta de adoção (Seção C da tabela) pendente de go do orquestrador**: URLLC latência 400 **ou** 500 ms (a escolha muda o shape do sweep — não fechar sem go); eMBB 7 Mbps `[ARTIGO]`; mMTC ≥5 PRBs `[ARTIGO]`; PDR/reliability URLLC tipo 99,999 % é **inviável** no nosso cenário (2 km/30 dBm, RLF de borda, decisão 100 ms) — qualquer número adotado = `[ASSUMIDO]` com justificativa.

---

## 4. Pendências e próximas etapas

### 4.1 Pendências registradas nesta etapa

| # | Pendência | Origem |
|---|---|---|
| 1 | **Indicação E2 não estendida**: `SqrE2KpmUeRecord` segue com 13 campos e `xapp_sqr.write_kpm_window` materializa 13 colunas — as colunas 14–17 existem só no caminho C++ (IPC). Estender exigiria tocar a paridade IPC×E2 da Fase 5.4 | Contrato v1.3 §4.1; R33 |
| 2 | **Ratificação do default λ/3** (interpretação do ask "replicar λ nos 3 slices") | R34 — decisão do orquestrador |
| 3 | **Escolha 400 vs 500 ms** (SLA URLLC de referência) e demais adoções da Seção C da tabela de SLA | `sla_table_round2.md` §C |
| 4 | **Latência E2E app-a-app vs somente-rádio**: se a análise exigir latência só-rádio, será preciso novo ponto de tagueamento no gNB | R32 |
| 5 | **Retransmissão do `m_unsentPacket`** do OnOff mantém o 1º instante da tag — sob carga normal não ocorre (gating por `GetTxAvailable`); ocorrendo, superestimaria levemente o delay | R32 |
| 6 | **Commit da etapa** — nada commitado (aguarda orquestrador) | `git status` verificado |
| 7 | Artefatos `sqr_packet_delays.csv` das validações não retidos em `results_sqr_network_only/` (gravados no CWD do run; dirs existentes são pré-etapa) — preservar saídas da 2.2 em diretório versionável | verificado nesta sessão (`g_outputDir="."`, `sqr-sim.cc:302`) |

### 4.2 Próxima etapa — **2.2: Sanity check em baixa carga** (protocolo §11)

**Objetivo** (protocolo §11.1): antes de procurar congestionamento, demonstrar que os **três SLAs são atingíveis em baixa demanda** — evita confundir erro de implementação / SLA impossível / problema de rádio com falta de recursos.

**Condição esperada** (protocolo §11.2): `SLA_eMBB = SLA_URLLC = SLA_mMTC = OK`, com RLF ≈ 0, SINR operacional, fila baixa, alta folga de recursos, PDR elevado e latência dentro do esperado.

**Gate** (protocolo §11.3): se **qualquer slice violar de forma sistemática em baixa carga ⇒ STOP** — não prosseguir para a caracterização (Etapa 2.3, sweep heterogêneo); investigar SLA, KPI, tráfego, scheduler, camada física, cálculo da métrica e janela de agregação.

**Insumos que a Etapa 2.1 já provê para a 2.2**: latência por pacote (`sqr_packet_delays.csv` → p95/p99/deadline-miss URLLC), PDR por UE-janela (`dRxPackets/dTxPackets`), CQI cru de diagnóstico, perfis de tráfego por slice (`--sliceApps=greedy,urllc,mmtc`) e metadado `SLA_PROFILES` para auditoria do run.

**Aviso operacional herdado da validação desta etapa**: runs de 20 s com `--lambdaUesPerMin=2` (seeds 21 e 22) produziram **0 chegadas** (~50% de probabilidade de Poisson(λ·T/60)=Poisson(0,67) dar 0) — a 2.2, em baixa carga, precisa de **durações maiores** (ou λ um pouco acima) para gerar amostra com significância; alternativamente usar o mecanismo novo `--sliceLambdas` com valores explícitos por slice.

Depois da 2.2: **Etapa 2.3 — sweep heterogêneo** (protocolo §12), com os alvos da Seção C da tabela de SLA congelados pelo orquestrador.

---

*Relatório gerado pela consolidação da Etapa 2.1 (agente Redator-Etapa-2-1). Checks re-executados nesta sessão: `bash tests/run_sqr_tests.sh` (194 passed), `bash tests/run_full_tests.sh` (339 passed/1 skipped), contagem de colunas do `sqr-kpms.txt` vivo (17), `git status --porcelain`. Demais evidências de simulação citadas dos estágios de implementação e registradas em R32–R35 do assumptions register.*
