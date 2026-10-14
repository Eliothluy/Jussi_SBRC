# Fase 5.3 — Adaptador de controle RIC Control → SqrMacScheduler (caminho E2 do sqr-sim)

- **Data**: 2026-10-14. **Autor**: Agente A (ns-3/C++).
- **Escopo**: `--controlPath={ipc,e2}` no `scratch/sqr/sqr-sim.cc`; adaptador E2 + loopback RIC em `scratch/sqr/sqr-e2-adapter.{h,cc}`; fase `e2` no `smoke_test.sh` (critérios R27 i–iv idênticos aos do IPC); gate `scratch/sqr/run_e2_smoke.sh`.
- **Base**: PLANO §3-Fase 5.3 ("RIC Control (RAN Control SM) → SetSliceConfiguration... KPM indications alimentadas pelos mesmos KPIs do contrato da Fase 0 (paridade semântica com o caminho IPC)"); MODA-PORT.md §1 (modos FULL/DEGRADED) e §3 (stubs: "caminho de controle local preparado para o adaptador da Fase 5.3 sem depender de E2").

## 1. O que foi entregue

| Item | Arquivo | Papel |
|---|---|---|
| Adaptador E2 (lado gNB) | `scratch/sqr/sqr-e2-adapter.{h,cc}` (`ns3::SqrE2Adapter`) | Cria a superfície E2 do contrib/nori (E2Interface + E2Termination + function descriptions KPM=200/RC=300/CCC=4, mesma fiação do `E2TermHelper::InstallE2Term`, `E2-term-helper.cc:72-200`); envia a E2SM-KPM indication por janela (`SendKpmIndication`) e aplica o RIC Control recebido (`OnRicControlMessage` → callback de ação do sqr-sim) |
| Loopback RIC (DEGRADED) | `scratch/sqr/sqr-e2-adapter.{h,cc}` (`ns3::SqrE2LoopbackTerm`) | "E2Term simulado no próprio cenário": fecha o circuito em-processo com as MESMAS mensagens estruturadas da API nori; controlador dummy equal-split equivalente ao `sqr_dummy_controller.py` (fases Σa=0/0,5C/C, 100 janelas/fase) |
| Caminho de controle | `scratch/sqr/sqr-sim.cc` | `--controlPath` (default `ipc`), `--e2Phases`, `--e2WindowsPerPhase`, `--e2RicAddress/Port`; `ApplyQuotaAction()` = aplicação ÚNICA da cota R27 compartilhada por IPC e E2; ramo E2 no `KpmAndControlCallback` SEM semáforos/CSV de ação; telemetria `E2_CONTROL` no relatório final |
| Smoke | `scratch/sqr/smoke_test.sh` (fase `e2`, incluída no `all`), `scratch/sqr/sqr_smoke_validate.py` (`--phased` = alias de `--ipc`), `scratch/sqr/run_e2_smoke.sh` (build check + fase e2 + "E2 SMOKE OK") | GATE oficial cobre os dois caminhos |

## 2. Desenho do caminho E2

```
                    ┌─────────────────────────── sqr-sim (por janela de 100 ms) ───────────────────────────┐
KPIs do contrato §4 │ 1. coleta (FlowMonitor + scheduler) → sqr_kpm_history.csv / sqr_prb_windows.csv       │
(mesmos do IPC)     │ 2. dreno de telemetria prb_* (alinhamento §3.5 — a ação governa a janela SEGUINTE)   │
                    │ 3. SqrE2Adapter::SendKpmIndication(SqrE2KpmIndicationMessage)                         │
                    └──────────────────────────────────────────┬───────────────────────────────────────────┘
                                                               │ mensagem estruturada (FULL: E2AP/E2SM-KPM+SCTP;
                                                               │ DEGRADED: chamada em-processo)
                                                    SqrE2LoopbackTerm::OnE2smKpmIndication
                                                    (dummy equal-split: fase por nº de janelas)
                                                               │ SqrE2RicControlMessage{RAN_SLICING,
                                                               │  std::vector<ns3::SlicePRBQuota>}
                                                               ▼
                    SqrE2Adapter::OnRicControlMessage → pct = dedicatePRBRatio (clamp 0..100;
                    slice sem linha ⇒ §5: 33,33 %/slice, contável) → callback do sqr-sim →
                    ApplyQuotaAction(pct, algos, nowMs, "e2")  ← MESMA função do caminho IPC
                    (ComputeRbgQuotas → SetSliceQuotaRbg, sem renormalização, R27; Σa=0 aplicado)
```

- **Caminho de código idêntico FULL/DEGRADED**: o adaptador só usa a API do contrib/nori que é igual nos dois modos (`E2Interface`, `E2Termination`, function descriptions, `ns3::SlicePRBQuota` de `model/nr-rl-mac-scheduler-ofdma.h:14`). No FULL, o mesmo `SendKpmIndication` passa a codificar E2AP/E2SM-KPM e enviar por SCTP, e o mesmo `OnRicControlMessage` é alimentado pelo `E2Interface::ControlMessageReceivedCallback` após o decode `RicControlMessage` → `std::vector<SlicePRBQuota>` (conversão sst→sliceId espelhando `model/E2-interface.cc:501-539`). Nenhum ponto do caminho por janela do sqr-sim muda — só o endpoint de entrega, injetado em `Install()`.
- **Mensagens estruturadas**: `SqrE2KpmIndicationMessage`/`SqrE2KpmUeRecord` (13 colunas do contrato §4 por UE + agregados de célula/slice) e `SqrE2RicControlMessage` (espelha os campos que o `ns3::RicControlMessage` decoda: requestType `RAN_SLICING`, style/action/decision, `std::vector<SlicePRBQuota>`). Em DEGRADED as structs são a própria mensagem; em FULL alimentam o encode/decode ASN.1 do nori.
- **Semântica de janela preservada**: no modo e2 o dreno de telemetria continua ANTES da aplicação da ação (ordem IPC) — a 1ª janela tem a sentinela `prb_requested_pct=-1` e a ação da borda t governa (t, t+period]. Validado pelo critério de alinhamento do `sqr_smoke_validate.py`.
- **Paridade semântica dos KPMs**: a indication carrega exatamente os valores emitidos nas linhas do `sqr_kpm_history.csv` da mesma janela (contrato §4, incl. sentinelas R09).
- **Algoritmo intra-slice**: o RIC Control de RAN slicing não carrega algoritmo (fora do espaço de ação SQR, PLANO Fase 1) — no modo e2 ele permanece no valor de `--intraAlgo` (default PF). No IPC o CSV pode trocá-lo por ação (col. 5), inalterado.

## 3. Divergências e decisões registradas

1. **[ASSUMIDO Fase 5.3] Razões de PRB inteiras no loopback**: o tipo do RIC Control (`ns3::SlicePRBQuota`, `long`, ASN.1 INTEGER do E2SM-RC) não expressa frações — o dummy IPC envia 16.6667/16.6667/16.6666 (Σ=50), o loopback envia **17/17/16** (Σ=50) e **34/33/33** (Σ=100; maiores restos, desempate por índice de slice). Σa — o que os critérios R27 testam — é preservado exatamente; as cotas RBG resultantes em C=53 são idênticas às do dummy IPC (9/9/9 e 18/18/17 após `ComputeRbgQuotas`, verificado no smoke e2: util 1.0000 por slice na fase C). Um RIC real com E2SM-RC estaria sujeito à mesma restrição inteira.
2. **[ASSUMIDO Fase 5.3] SST = índice da slice lógica (0..2)**: mesmo mapeamento da conversão do FULL (`macQ.sliceId = q.sst`, `E2-interface.cc:526`); resourceType `PRB_DL` (a cota restringe só DL, escopo do artigo).
3. **Fix pontual no contrib/nori (2 includes)**: o destrutor inline de `E2Interface` (`stub/E2-interface.h:54-56` e o análogo `model/E2-interface.h:46-48`) destrói `Ptr<NrGnbRrc> m_rrc` com tipo incompleto em TUs que incluem o header standalone (o `.cc` do nori compilava por incluir `nr-gnb-rrc.h` antes). Adicionado `#include "ns3/nr-gnb-rrc.h"` aos DOIS headers — mudança de includes apenas, sem comportamento; necessária para o `sqr-e2-adapter.h` compilar. Sem patches defensivos em `contrib/nr` (regra de ouro preservada).
4. **Loopback não usa o transporte do stub**: `E2Termination::SendE2Message`/`BuildAndSendReportMessage` do stub descartam a mensagem por desenho (MODA-PORT §3); a entrega em-processo é o endpoint injetado (`SetControlDeliverCallback`), mantendo o stub nori intocado em comportamento. O `Start()` do E2Termination stub é exercitado pela fiação canônica (log `[DEGRADED]` do stub permanece disponível; o loopback loga o próprio E2-SETUP simulado).

## 4. Verificação (honesto)

| Check | Comando | Resultado |
|---|---|---|
| Build completo (DEGRADED) | `bash MODA/ns-3-dev/scratch/sqr/ns3_gate.sh build` | **exit 0** (nori + sqr-sim relinkados) |
| Gate E2 completo | `bash MODA/ns-3-dev/scratch/sqr/run_e2_smoke.sh 30 1` | **"E2 SMOKE OK"** — build check + fase e2 com: sem semáforos IPC e sem `rslaq_actions_for_ns3.csv`; E2Interface+E2Termination instalados; **299 indications / 299 RIC Controls** em 30 s; 3 fases (0/50/100, ≥90 janelas cada); R27 i (util 1.0000/slice na fase C), ii (ociosa min=0.491 mediana=0.512 na fase B), iii (nada acima da cota; fase A com Σsched=0 em 100 janelas), iv (ACTION/PRB_WINDOW com requested/executed/idle); alinhamento com sentinela -1; KPM contrato §4 13 colunas, sentinela SNR -999, janela parcial |
| Regressão IPC | `bash scratch/sqr/smoke_test.sh ipc 30 1` | **"SMOKE OK"** (299 respostas do dummy; mesmas fases/critérios — refactor `ApplyQuotaAction` não alterou o IPC) |
| Regressão network | `bash scratch/sqr/smoke_test.sh network 120 1` | **"SMOKE OK"** (baseline network-only idêntico ao pré-5.3) |
| R27-iv (git) | `git status --porcelain -- ns-3-dev/scratch/rslaq` (via smoke) | intocado em todos os runs |
|FULL (com e2sim) | — | **NÃO verificado, por definição**: o e2sim não existe na máquina (MODA-PORT §4); o adaptador compila contra a API compartilhada com o stub (garantia de mesma assinatura), e os pontos de integração FULL estão documentados no código (`sqr-e2-adapter.h`) |
| Soak E2 (600 s × seeds) | — | não executado nesta fase (fora do pedido; o soak oficial continua network-only/IPC default). O `all` do smoke oficial agora inclui a fase e2 de 30 s |

## 5. Como rodar

```bash
# gate E2 (build check + fase e2 do smoke + "E2 SMOKE OK"):
bash MODA/ns-3-dev/scratch/sqr/run_e2_smoke.sh [simTime=30] [seed=1]

# fase e2 isolada / gate oficial completo (agora com e2):
bash MODA/ns-3-dev/scratch/sqr/smoke_test.sh e2 [simTime] [seed]
bash MODA/ns-3-dev/scratch/sqr/smoke_test.sh all

# run manual E2 (loopback in-process):
bash MODA/ns-3-dev/scratch/sqr/ns3_gate.sh run sqr-sim -- --simTime=30 --controlPath=e2 \
    --e2Phases=0,50,100 --e2WindowsPerPhase=100 --outputDir=/tmp/e2run
# IPC (fallback permanente, inalterado): omitir --controlPath (default ipc) + --simId
```
