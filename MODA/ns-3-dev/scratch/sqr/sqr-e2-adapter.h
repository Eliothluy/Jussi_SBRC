/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * SqrE2Adapter — Adaptador de controle RIC Control → SqrMacScheduler
 * (Fase 5.3, PLANO §3) + SqrE2LoopbackTerm — E2Term/Near-RT RIC simulado
 * em-processo (modo DEGRADED do contrib/nori, MODA-PORT.md §1/§3).
 *
 * CAMINHO DE CONTROLE --controlPath=e2 do sqr-sim:
 *   - as COTAS chegam como RIC Control (estilo RAN slicing do E2SM-RC /
 *     equivalente CCC do contrib/nori) e são aplicadas ao SqrMacScheduler
 *     pela MESMA aplicação de cota absoluta do caminho IPC (R27, sem
 *     renormalização) — SEM semáforos POSIX nem rslaq_actions_for_ns3.csv;
 *   - os KPMs saem como E2SM-KPM indications pela E2-interface: o adaptador
 *     monta a indication estruturada com os MESMOS KPIs do contrato §4
 *     (paridade semântica com o caminho IPC — PLANO §3-Fase 5.3).
 *
 * FULL vs DEGRADED (caminho de código do adaptador IDÊNTICO):
 *   - O adaptador fala apenas a API pública do contrib/nori que é IGUAL nos
 *     dois modos (os stubs DEGRADED existem exatamente para isso —
 *     MODA-PORT.md §1: "mesma API pública"): E2Interface, E2Termination,
 *     KpmFunctionDescription/RicControlFunctionDescription/
 *     CccFunctionDescription e o tipo estruturado ns3::SlicePRBQuota
 *     (model/nr-rl-mac-scheduler-ofdma.h — o mesmo tipo que o FULL
 *     E2Interface::ApplySlicingControl consome após decodificar o
 *     RicControlMessage ASN.1, E2-interface.cc:501-539).
 *   - DEGRADED (build atual, e2sim ausente): o transporte E2/SCTP não existe;
 *     o SqrE2LoopbackTerm (E2Term simulado, abaixo) fecha o circuito
 *     em-processo com as MESMAS mensagens estruturadas: OnE2smKpmIndication()
 *     recebe a indication e devolve um SqrE2RicControlMessage (RAN slicing)
 *     para SqrE2Adapter::OnRicControlMessage().
 *   - FULL (quando o e2sim for linkado, Fase 5.1): o mesmo SendKpmIndication
 *     passa a codificar E2AP/E2SM-KPM e enviar por SCTP (SendE2Message), e o
 *     mesmo OnRicControlMessage passa a ser alimentado pelo
 *     E2Interface::ControlMessageReceivedCallback (decodificação
 *     RicControlMessage → std::vector<SlicePRBQuota>). Nenhum ponto do
 *     caminho por janela muda no sqr-sim; só o endpoint de entrega (injetado
 *     em Install()) difere.
 *
 * O controlador do loopback (SqrE2LoopbackTerm) é o DUMMY EQUAL-SPLIT
 * equivalente ao sqr_dummy_controller.py do smoke IPC: ações fixas em fases
 * determinísticas Σa = 0 / 0,5·C / C (% de PRB, divididas igualmente entre as
 * 3 slices), uma resposta por indication recebida — valida o caminho E2
 * ponta a ponta (KPM indication → RIC → RIC Control → cota aplicada).
 *
 * Fase 5.4 — endpoint xApp (--e2Endpoint=xapp): o MESMO objeto ganha o modo
 * XAPP_UDP, em que ele deixa de decidir e passa a ser a PONTE de transporte
 * local do modo DEGRADED para o xApp SQR (processo Python,
 * examples/xapp_sqr.py): serializa a E2SM-KPM indication em JSON e envia por
 * UDP/localhost ao xApp; espera (budget wall do contrato §5, default 50 ms —
 * o mesmo orçamento do sem_timedwait do IPC) a resposta RIC Control em JSON,
 * decodifica no MESMO SqrE2RicControlMessage e entrega via
 * SetControlDeliverCallback. Timeout/malformado = degradação contável do
 * contrato §5 (mantém a cota anterior), idêntico ao timeout do IPC.
 * Transporte LOCAL documentado: UDP/loopback substitui a chamada em-processo
 * do modo loopback; em FULL o mesmo papel é do e2sim/SCTP — a lógica do xApp
 * (indication → decisão → RIC Control) não muda, só o endpoint de entrega.
 */

#pragma once

#include "sqr-mac-scheduler.h"

#include "ns3/E2-interface.h"
#include "ns3/oran-interface.h"

#include "ns3/nr-rl-mac-scheduler-ofdma.h"
#include "ns3/object.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace ns3
{

// ---------------------------------------------------------------------------
// Mensagens estruturadas (equivalentes API nori)
// ---------------------------------------------------------------------------

/**
 * @brief Registro de UE de uma E2SM-KPM indication (nível UE).
 *
 * Colunas = contrato §4 do cenário SQR (paridade semântica com o KPM do
 * caminho IPC em sqr-kpms.txt/sqr_kpm_history.csv). Em FULL, estes campos
 * alimentam o KpmIndicationMessage (E2SM-KPM IndicationMessage, nó+UE);
 * em DEGRADED, a struct é a própria mensagem trocada com o loopback.
 */
struct SqrE2KpmUeRecord
{
    uint64_t imsi = 0;              //!< IMSI do UE (UEID da E2SM-KPM)
    uint16_t rnti = 0;              //!< C-RNTI corrente
    uint32_t sliceId = 0;           //!< índice da slice lógica (0..2; SST no loopback)
    uint64_t dTxBytes = 0;          //!< contrato §4 col. 4
    uint64_t dRxBytes = 0;          //!< contrato §4 col. 5
    double plrPct = 0.0;            //!< contrato §4 col. 6 (%)
    double resourceSharePct = 0.0;  //!< contrato §4 col. 7 (% da cota RBG da slice)
    uint32_t dLostPackets = 0;      //!< contrato §4 col. 8
    double throughputMbps = 0.0;    //!< contrato §4 col. 9 (legado, delta/período)
    uint32_t bufferBytes = 0;       //!< contrato §4 col. 10
    double windowThroughputMbps = 0.0; //!< contrato §4 col. 11 (tempo ATIVO)
    double snrDb = -999.0;          //!< contrato §4 col. 12 (sentinela -999 = sem CQI)
    uint32_t queueBytes = 0;        //!< contrato §4 col. 13
};

/**
 * @brief Mensagem E2SM-KPM indication (estruturada; nível nó + registros UE).
 */
struct SqrE2KpmIndicationMessage
{
    uint64_t timestampMs = 0;       //!< fim da janela de medição (ms simulados)
    uint16_t cellId = 0;            //!< cell ID do gNB
    std::string plmnId = "00101";   //!< PLMN (mesmo do E2TermHelper)
    std::string gnbId;              //!< gNB ID (string do cellId, como no helper)
    uint32_t periodMs = 100;        //!< período de decisão [ARTIGO: 100 TTIs]
    uint16_t totalRbg = 0;          //!< RBGs do BWP (C efetivo, R01)
    std::array<uint32_t, 3> quotaRbg{};   //!< cota RBG corrente por slice
    std::array<uint32_t, 3> nUes{};       //!< UEs com KPM na janela, por slice
    std::vector<SqrE2KpmUeRecord> ueRecords;
};

/**
 * @brief Mensagem RIC Control (estruturada; estilo RAN slicing do E2SM-RC).
 *
 * Espelha os campos que o ns3::RicControlMessage (contrib/nori, FULL)
 * decoda do E2AP_PDU: ranFunctionId, request type RAN_SLICING,
 * ricStyleType/actionId/decision e as cotas de PRB por slice. As cotas usam
 * o tipo estruturado ns3::SlicePRBQuota do contrib/nori (o mesmo tipo que
 * E2Interface::ApplySlicingControl/ApplyCccSlicingControl consomem nos dois
 * modos de build).
 */
struct SqrE2RicControlMessage
{
    /// Request types (espelha RicControlMessage::ControlMessageRequestIdType,
    /// model/ric-control-message.h:61-66).
    enum RequestType : long
    {
        TS = 1001,
        QOS = 1002,
        RAN_SLICING = 1003,
    };

    long ranFunctionId = 300;        //!< RAN Function ID do E2SM-RC (E2TermHelper)
    long requestType = RAN_SLICING;  //!< só RAN_SLICING é aplicado
    long ricStyleType = 1;           //!< estilo de controle (RAN slicing)
    long ricControlActionId = 1;     //!< action ID
    long ricControlDecision = 0;     //!< 0 = accept, 1 = reject (convenção nori)
    std::vector<SlicePRBQuota> prbQuotas; //!< cota por slice (dedicate/min/max %)
};

// ---------------------------------------------------------------------------
// SqrE2LoopbackTerm — E2Term + Near-RT RIC simulados (modo DEGRADED)
// ---------------------------------------------------------------------------

/**
 * @brief Loopback em-processo do caminho E2 ("E2Term simulado no próprio
 *        cenário") com controlador dummy equal-split.
 *
 * Papel do RIC no modo DEGRADED: recebe a E2SM-KPM indication estruturada e
 * devolve UM RIC Control por indication, com ação fixa por fase
 * determinística — equivalente ao sqr_dummy_controller.py do smoke IPC
 * (mesmas fases Σa = 0 / 0,5·C / C e mesma contagem de janelas). DIVERGÊNCIA
 * única, imposta pela API nori: SlicePRBQuota carrega razões de PRB INTEIRAS
 * (long; ASN.1 INTEGER do E2SM-RC) — o equal-split inteiro usa maiores
 * restos com Σa preservado:
 *
 *   fase Σa=0   → dedicatePRB = (0, 0, 0) %
 *   fase Σa=0,5C → dedicatePRB = (17, 17, 16) %   [dummy IPC: 16.6667×2+16.6666]
 *   fase Σa=C   → dedicatePRB = (34, 33, 33) %    [dummy IPC: 33.3333×2+33.3334]
 *
 * (min = max = dedicate, como no CSV do dummy; SST = índice da slice; as
 * cotas RBG resultantes em C=53 são idênticas às do dummy IPC.)
 * Em FULL este objeto NÃO é usado: o endpoint real é o e2sim (SCTP) e o
 * controlador é o xApp no Near-RT RIC.
 */
class SqrE2LoopbackTerm : public Object
{
  public:
    /**
     * @brief Modo do endpoint RIC (Fase 5.4).
     *
     * LOOPBACK_DUMMY: controlador dummy equal-split em-processo (Fase 5.3,
     * default — smoke/gate). XAPP_UDP: ponte de transporte local (UDP JSON,
     * localhost) para o xApp SQR em Python — o "RIC" degradado é o processo
     * xApp; em FULL o transporte é o e2sim/SCTP e este objeto não é usado.
     */
    enum class EndpointMode
    {
        LOOPBACK_DUMMY,
        XAPP_UDP,
    };

    static TypeId GetTypeId();
    SqrE2LoopbackTerm();
    ~SqrE2LoopbackTerm() override;

    /**
     * @brief Configura o loopback (chamado uma vez, antes do Start).
     * @param ricAddress endereço do RIC (registrado; DEGRADED não conecta)
     * @param ricPort porta do RIC
     * @param phases somas de % dedicadas por fase (subconjunto de {0,50,100})
     * @param windowsPerPhase janelas (indications) por fase
     * @param logPath CSV de evidência (timestamp_ms,phase,pct_0,pct_1,pct_2)
     */
    void Configure(const std::string& ricAddress,
                   uint16_t ricPort,
                   const std::vector<long>& phases,
                   uint32_t windowsPerPhase,
                   const std::string& logPath);

    /**
     * @brief Configura o modo PONTE xApp (Fase 5.4, --e2Endpoint=xapp).
     *
     * Cria o socket UDP de envio (transporte local documentado do modo
     * DEGRADED; o xApp faz bind ANTES do sqr-sim arrancar). Cada
     * indication vira um datagrama JSON com `seq` monotônico; a resposta
     * RIC Control (JSON, mesmo `seq`) é decodificada e entregue ao
     * adaptador. @param timeoutMs budget wall por tentativa (contrato §5,
     * default 50 ms — igual ao sem_timedwait do IPC); @param retries
     * tentativas extras com o MESMO seq (o xApp deduplica por seq e
     * reenvia a decisão cacheada — retry é seguro e não re-decide).
     * @param logPath CSV de evidência (seq,timestamp_ms,kpm_ues,rtt_wall_ms,
     *        applied,timeout,reply_decision_ms)
     * @returns false se o socket não puder ser criado (o chamador aborta).
     */
    bool ConfigureXappUdp(const std::string& ricAddress,
                          uint16_t ricPort,
                          int timeoutMs,
                          int retries,
                          const std::string& logPath);

    /**
     * @brief Registra a entrega do RIC Control ao lado gNB (o adaptador).
     *
     * Em FULL, este papel é do transporte e2sim → E2Interface::
     * ControlMessageReceivedCallback. Aqui a entrega é uma chamada
     * em-processo síncrona com a MESMA mensagem estruturada.
     */
    void SetControlDeliverCallback(
        std::function<void(const SqrE2RicControlMessage&)> cb);

    /** @brief Loga o E2-SETUP simulado (equivalente ao Start do E2Term). */
    void Start();

    /**
     * @brief Entrada do lado RIC: consome uma E2SM-KPM indication e responde
     *        com o RIC Control da fase corrente (equal-split dummy).
     */
    void OnE2smKpmIndication(const SqrE2KpmIndicationMessage& ind);

    /** @brief Indications E2SM-KPM recebidas (janelas respondidas). */
    uint64_t GetIndicationsReceived() const;
    /** @brief RIC Controls emitidos (1 por indication). */
    uint64_t GetControlsSent() const;
    /** @brief Fases configuradas (somas de %). */
    const std::vector<long>& GetPhases() const;
    /** @brief Modo corrente do endpoint. */
    EndpointMode GetMode() const;
    /** @brief Janelas SEM resposta do xApp dentro do budget (degradação §5). */
    uint64_t GetXappTimeouts() const;

  private:
    /** @brief Ramo do dummy equal-split em-processo (Fase 5.3, inalterado). */
    void HandleIndicationLoopback(const SqrE2KpmIndicationMessage& ind);
    /** @brief Ramo da ponte xApp UDP (Fase 5.4). */
    void HandleIndicationXappUdp(const SqrE2KpmIndicationMessage& ind);

    std::string m_ricAddress{"127.0.0.1"};
    uint16_t m_ricPort{36422};
    std::vector<long> m_phases{0, 50, 100};
    uint32_t m_windowsPerPhase{100};
    uint64_t m_indicationsReceived{0};
    uint64_t m_controlsSent{0};
    std::ofstream m_log;
    std::function<void(const SqrE2RicControlMessage&)> m_deliver;

    // --- Fase 5.4: estado da ponte xApp (modo XAPP_UDP) ---
    EndpointMode m_mode{EndpointMode::LOOPBACK_DUMMY};
    int m_xappTimeoutMs{50};   //!< budget wall por tentativa (contrato §5)
    int m_xappRetries{1};      //!< tentativas extras com o mesmo seq
    uint64_t m_xappSeq{0};     //!< seq monotônico das indications enviadas
    uint64_t m_xappTimeouts{0};//!< janelas sem resposta dentro do budget
    int m_sockFd{-1};          //!< socket UDP de envio (transporte local)
    void* m_xappAddr{nullptr}; //!< sockaddr_in do xApp (void* p/ não poluir o header)
};

// ---------------------------------------------------------------------------
// SqrE2Adapter — lado gNB do caminho de controle E2
// ---------------------------------------------------------------------------

/**
 * @brief Adaptador Fase 5.3: RIC Control → SqrMacScheduler e KPMs do
 *        contrato §4 → E2SM-KPM indications.
 *
 * Sequência por janela (idêntica nos modos FULL e DEGRADED; a ordem preserva
 * o alinhamento de janela do contrato §3.5 — dreno de telemetria ANTES da
 * ação, que governa a janela seguinte):
 *   1. sqr-sim coleta os KPIs do contrato §4 e chama SendKpmIndication();
 *   2. o endpoint entrega a indication ao RIC (DEGRADED: loopback em-processo;
 *      FULL: codificação E2AP/E2SM-KPM + SCTP via e2sim);
 *   3. o RIC responde com RIC Control → OnRicControlMessage();
 *   4. OnRicControlMessage extrai as % por slice das SlicePRBQuota e chama o
 *      callback de ação do sqr-sim — a MESMA ApplyQuotaAction do caminho IPC
 *      (ComputeRbgQuotas → SetSliceQuotaRbg, R27, inclusive Σa=0).
 *
 * AInstall() cria a superfície E2 do contrib/nori (E2Interface +
 * E2Termination + registro das function descriptions KPM/RC/CCC com os
 * mesmos RAN function IDs do E2TermHelper: 200/300/4) — no DEGRADED os
 * stubs logam o aviso de transporte ausente; no FULL os mesmos objetos
 * conectam ao Near-RT RIC.
 */
class SqrE2Adapter : public Object
{
  public:
    static TypeId GetTypeId();
    SqrE2Adapter();
    ~SqrE2Adapter() override;

    /// Callback de ação do sqr-sim (aplicação da cota absoluta, R27).
    /// pct = % de PRB por slice já extraídas do RIC Control;
    /// sliceMissing = true se a mensagem não trouxe linha para ≥1 slice
    /// (degradação do contrato §5: default 33,33 %/slice, contável).
    using ActionCallback =
        std::function<void(const std::array<double, 3>& pct, bool sliceMissing)>;

    /**
     * @brief Instala o adaptador no gNB (superfície E2 nori + endpoint RIC).
     * @param gnbDev dispositivo do gNB (NrGnbNetDevice)
     * @param scheduler scheduler SQR (para telemetria de cota; a aplicação
     *        da ação continua no callback do sqr-sim)
     * @param ricEndpoint entrega das indications ao RIC (DEGRADED: loopback)
     * @param actionCb callback de aplicação da ação (sqr-sim)
     */
    void Install(Ptr<NetDevice> gnbDev,
                 Ptr<SqrMacScheduler> scheduler,
                 Ptr<SqrE2LoopbackTerm> ricEndpoint,
                 ActionCallback actionCb);

    /**
     * @brief Envia a E2SM-KPM indication da janela (lado KPM, saindo pela
     *        E2-interface). Um chamada por janela de decisão.
     */
    void SendKpmIndication(const SqrE2KpmIndicationMessage& ind);

    /**
     * @brief Entrada do RIC Control (lado controle): valida o estilo
     *        RAN slicing, extrai as % por slice (dedicatePRBRatio) e aplica
     *        via callback. Aceita Σa = 0 (R27 — nunca manter config anterior).
     * @returns true se uma ação foi entregue ao callback.
     */
    bool OnRicControlMessage(const SqrE2RicControlMessage& ctrl);

    /** Telemetria para o relatório final e para o smoke. */
    uint64_t GetIndicationsSent() const;
    uint64_t GetControlsReceived() const;
    uint64_t GetControlsApplied() const;
    uint64_t GetControlsIgnored() const;
    uint64_t GetSliceLineMissingCount() const;

  private:
    Ptr<E2Interface> m_e2Interface;    //!< superfície E2 do contrib/nori
    Ptr<E2Termination> m_e2Term;       //!< E2 termination do contrib/nori
    Ptr<SqrMacScheduler> m_scheduler;  //!< scheduler SQR (telemetria)
    Ptr<SqrE2LoopbackTerm> m_ricEndpoint; //!< endpoint RIC (DEGRADED: loopback)
    ActionCallback m_actionCb;
    uint64_t m_indicationsSent{0};
    uint64_t m_controlsReceived{0};
    uint64_t m_controlsApplied{0};
    uint64_t m_controlsIgnored{0};
    uint64_t m_sliceLineMissing{0};
};

} // namespace ns3
