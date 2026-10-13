# Patches defensivos 5G-LENA (contrib/nr)

O módulo `contrib/nr` não é rastreado pelo git do projeto. Estas cópias contêm
os patches defensivos aplicados sobre o 5G-LENA para que condições extremas
(perdas pesadas em RLC-UM, churn de reestabelecimento RLF em runs longos) não
abortem a simulação (SIGABRT/NS_ASSERT).

Instalação:
    cp nr-pdcp.cc                   ../../ns-3-dev/contrib/nr/model/
    cp nr-rlc-um.cc                 ../../ns-3-dev/contrib/nr/model/
    cp nr-net-device.cc             ../../ns-3-dev/contrib/nr/model/
    cp nr-qos-rule-classifier.cc    ../../ns-3-dev/contrib/nr/model/
    cd ../../ns-3-dev && bash scratch/sqr/ns3_gate.sh build

Patches 1–3 (defesa em profundidade, descartam PDU/SDU malformada com NS_LOG_WARN):
1. nr-pdcp.cc      — NrPdcp::DoReceivePdu: valida bit D/C e tamanho mínimo
                     antes de RemoveHeader (descarta PDU corrompida).
2. nr-rlc-um.cc    — NrRlcUm::ReassembleAndDeliver (estado WAITING_SI_SF sem
                     perda sinalizada): descarta PDU com Framing Info inesperado
                     em vez de NS_ASSERT_MSG fatal.
3. nr-net-device.cc— NrNetDevice::Receive: exige tamanho mínimo (20 B IPv4 /
                     40 B IPv6) antes de PeekHeader; SDU curta cai no branch
                     de drop já existente.

Patch 4 (12/10/2026 — autorizado pelo orquestrador na rodada de correções da
Fase 1 SQR; a cópia neste diretório é o ARQUIVO PRÉ-PATCH, original intocado):
4. nr-qos-rule-classifier.cc — NrQosRuleClassifier::Add: dedup de regra QoS
                     IDÊNTICA (mesmo QFI + precedência + lista de filtros)
                     com NS_LOG_WARN antes do insert; o NS_ASSERT de
                     m_qosRuleMap.size() <= 64 PERMANECE para crescimento
                     legítimo (QFIs novos).
                     Motivo: NrEpcMmeApplication::DoInitialUeMessage
                     (nr-epc-mme-application.cc:119-162) reenvia TODA a lista
                     `flowsToBeActivated` acumulada a cada InitialUeMessage
                     (nunca a limpa); cada reestabelecimento RRC pós-RLF de um
                     UE de borda recria os fluxos e o PGW re-insere as regras
                     no classificador por UE. No soak do cenário SQR
                     (600 s simulados, seed 1) o multimap passou de 64 entradas
                     e o sanity assert abortou a simulação em t≈562,5 s
                     (SIGABRT, exit 250). Evidência:
                     ns-3-dev/scratch/sqr/smoke_out/soak_20261012_105012_s1/
                     sqr-sim.log — "NS_ASSERT failed, cond=m_qosRuleMap.size()
                     <= 64", stack: NrQosRuleClassifier::Add ←
                     NrEpcPgwApplication::NrUeInfo::AddFlow ←
                     NrEpcPgwApplication::DoRecvCreateSessionRequest. Em 600 s,
                     um único UE acumulou 26 trocas de RNTI (RLF churn). A
                     duplicata é semanticamente inerte: Classify retorna o
                     primeiro match, e regras idênticas não alteram roteamento.

O "4º patch" histórico (renormalização de pesos no RslaqMacScheduler) está no
arquivo versionado ns-3-dev/scratch/rslaq/rslaq-mac-scheduler.cc — não é um
patch de contrib/nr.
