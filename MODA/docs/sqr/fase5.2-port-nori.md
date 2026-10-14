# Fase 5.2 — Port do módulo NORI para contrib/nori (registro)

- **Data**: 2026-10-14 · **Agente**: A (ns-3/C++ sênior)
- **Origem**: clone de [lasseufpa/nori](https://github.com/lasseufpa/nori) em `MODA/third_party/nori`, HEAD `9ea0db9` (inclui o núcleo das 12 classes — `E2-interface`, `E2-report`, `kpm-indication`, `kpm-function-description`, `ric-control-message`, `ric-control-function-description`, `oran-interface`/E2Termination, `asn1c-types`, `nr-rl-mac-scheduler-ofdma`, `nr-mac-scheduler-ue-info-rl`, `function-description` — mais as classes E2SM-CCC).
- **Destino**: `MODA/ns-3-dev/contrib/nori/` — ns-3.46-dev + 5G-LENA v5.2-dev.
- **Documento técnico completo**: `MODA/ns-3-dev/contrib/nori/docs/MODA-PORT.md` (tabela de adaptações A1–A12, arquitetura FULL/DEGRADED, status de verificação).

## Decisões e achados principais

1. **e2sim não disponível → modo DEGRADED (conforme pedido).** Não existe `MODA/third_party/e2sim` (Fase 5.1 pendente). Achado colateral: existe um e2sim instalado em `/usr/local` (`/usr/local/include/e2sim` + `libe2sim.a`, 8,1 MB, abr/2026), porém é uma **geração antiga incompatível** com o NORI HEAD — faltam headers ASN.1 que o NORI usa: `E2SM-KPM-IndicationMessage-Format3.h`, `UEID.h`, `UEID-GNB.h`, `UEMeasurementReport*`, `MeasurementDataItem.h`, `E2SM-RC-ControlMessage-Format1-Item.h`, `RANFunctionDefinition-Control*.h`, `ControlAction-RANParameter-Item.h`, `RANParameter-STRUCTURE-Item.h`, `RANParameter-ValueType-Choice-*.h`, `GUAMI.h`, `AMF-UE-NGAP-ID.h` (verificado por varredura dos `#include` de `model/`+`helper/` contra `/usr/local/include/e2sim/`, 287 headers). O CMake portado detecta isso e **recusa o candidato** com mensagem de STATUS, caindo em DEGRADED.
2. **Arquitetura do build**: variável `NORI_E2SIM_DIR` (cache CMake) + `NORI_E2SIM_ENABLED` (ON default). FULL quando um e2sim compatível for encontrado (marcadores exigidos: `E2SM-KPM-IndicationMessage-Format3.h` + `E2nodeComponentConfigAddition_ItemIEs.h`; `NORI_E2SIM_DIR` explícito e incompatível → erro duro de configure). DEGRADED compila stubs de `oran-interface`/`E2-interface`/3 function-descriptions com **a mesma API pública** (headers de nomes idênticos instalados em `build/include/ns3/`), de modo que `E2-term-helper.cc` e o exemplo compilam **sem alteração** nos dois modos.
3. **Stub de transporte E2**: `E2Termination::Start()` loga `NS_LOG_WARN "[DEGRADED] E2 transport unavailable (e2sim not linked). E2 connection attempt for gNB 1 towards RIC 127.0.0.1:36422 (local bind port 38471, PLMN ...) was NOT performed."` — sem socket/thread. `E2Interface` stub mantém `NoriE2Report` **real** (traces PHY/MAC) e `ApplySlicingControl()` **real** (repassa `SlicePRBQuota` ao `NrRLMacSchedulerOfdma::SetSlicingParameters`) — caminho de controle local já utilizável pelo adaptador da Fase 5.3 sem E2.
4. **Principal adaptação de API NR v5.x→v5.2-dev**: `NrMacSchedulerUeInfo::m_dlRBG/m_dlSym` viraram **vetores por símbolo** (multi-stream MIMO; `contrib/nr/model/nr-mac-scheduler-ue-info.h:334-336`) — o `AssignDLRBG` do scheduler RL foi reescrito no idioma de `NrMacSchedulerOfdma::AllocateCurrentResourceToUe` (`contrib/nr/model/nr-mac-scheduler-ofdma.cc:158`), com `GetDlBitmask()` e livro-razão de índices RBG (`std::set`) que preserva a contabilidade de pool do NORI (fases dedicated/min/max) sem dupla alocação. Demais adaptações (A4–A12 no MODA-PORT.md): `GetCellIds()`→`GetCellId(ccIdx)`, `NrEpsBearerTag`→`NrQosFlowTag` (RELEASE_NOTES.md:438), `ConfigureFactories` 3 args, constantes `iana::*`, trace paths `Nr*`, exemplo com `--simTime` (default 2 s) e `--ipE2TermRic` default 127.0.0.1.
5. **Landmine pré-existente resolvido (fora do escopo, efeito colateral necessário)**: o scanner de `scratch/` do ns-3 (regex `main[(| (]`, `scratch/CMakeLists.txt:15-20`) acionava erro de configure por menções a "main()" **em comentários** de `scratch/sqr/sqr-mac-scheduler.cc` (adicionado na Fase 1 e nunca reconfigurado desde então). Correção: reformulação de 3 comentários (zero mudança de comportamento) — `scratch/sqr` é território do Agente A.

## Verificação executada (comandos e resultados)

| Check | Comando | Resultado |
|---|---|---|
| Configure | `bash scratch/sqr/ns3_gate.sh configure --enable-examples` | `NORI: DEGRADED mode (e2sim not found)...` + `Build files have been written` |
| **Build inteiro** (gate do pedido) | `bash scratch/sqr/ns3_gate.sh build` | verde: 11/11 targets, `libns3.46-nori-default.so` + `ns3.46-nori-sample-default` + todos os executáveis pré-existentes (sqr-sim, rslaq-sim, GreenRan-slice, test-runner); re-run final: `ninja: no work to do.`, **exit 0** |
| **Run sem RIC** (gate do pedido) | `bash scratch/sqr/ns3_gate.sh run nori-sample -- --simTime=2` | **exit 0**; log: `[DEGRADED] E2 transport unavailable... E2 connection attempt for gNB 1 towards RIC 127.0.0.1:36422 ... was NOT performed`; UDP DL fluindo (PDCP/RLC RxPDU a cada 2 ms, 2072 B, delay ~2,24 ms); `NoriSample finished cleanly at t=2 s (E2 target 127.0.0.1:36422; PDCP trace seen: yes, RLC trace seen: yes)` |
| Timeout curto | idem com `--simTime=1` | 1,17 s wall time, exit 0 (realtime, sem hang) |

## Não verificado / pendências honestas

- **Modo FULL (e2sim real)**: os fontes e2sim-dependentes em `model/`+`helper/` foram portados com as adaptações identificadas por leitura (A1–A4, A11–A12 do MODA-PORT.md), mas **não compilam nesta máquina por definição** (não há e2sim compatível). O build FULL será exercitado na Fase 5.1/5.3 quando o fork `lasseufpa/e2sim` for construído (GCC 11 + ASN.1 modificado).
- Os demais exemplos upstream (`nori-rc-slicing-demo`, `nori-ccc-slicing-demo`, `nori-benchmark-scenario`, `nori-mimo-demo`) não foram portados (fora do escopo desta fase); permanecem em `MODA/third_party/nori`.
- `contrib/nr` intocado (git status limpo); nenhum patch novo em `contrib/nr`. Nota: durante o port um `cp` com cwd errado criou brevemente `contrib/nr/model/nr-rl-mac-scheduler-ofdma.h` (arquivo novo, sem sobrescrever nada) — **removido na mesma sessão**, confirmado por `git status` vazio em `contrib/nr`.

## Caminhos entregues

```
MODA/ns-3-dev/contrib/nori/
├── CMakeLists.txt                     (detecção FULL/DEGRADED + marcadores de compatibilidade)
├── README.md                          (port notes + quick start)
├── docs/MODA-PORT.md                  (adaptações A1–A12 detalhadas)
├── model/                             (12 pares upstream; nr-rl-mac-scheduler-ofdma.cc portado; E2-interface.cc com fix GetCellId; function-description.cc sem asn1c-types)
├── helper/                            (E2-term-helper, nori-slicing-helper, indication helpers — upstream)
├── stub/                              (oran-interface, E2-interface, 3 function-descriptions — DEGRADED)
└── examples/nori-sample.cc            (port v5.2 + --simTime/--ipE2TermRic)
```
