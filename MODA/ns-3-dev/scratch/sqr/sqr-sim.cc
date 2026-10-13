/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * SQR Simulation — ns-3 / 5G-LENA, cenário fiel ao artigo SQR (CCNC 2026)
 *
 * "Safe RAN Slicing in O-RAN: Minimizing SLA Violations via Model-Based
 *  Reinforcement Learning" (Tuerxun & Nakao, Univ. de Tóquio,
 *  DOI 10.1109/CCNC65079.2026.11366589). Adaptado de scratch/rslaq/rslaq-sim.cc
 *  (original intocado); o mecanismo IPC (semáforos POSIX + CSV) é mantido
 *  verbatim. Diferenças vs rslaq-sim.cc:
 *
 *  1. Rádio fiel à Tabela I do artigo:
 *     - 1 gNB macro, portadora 2 GHz, raio de célula 2 km;
 *     - TxPower 30 dBm, ganho de antena 15 dBi, noise figure 9 dB;
 *     - pathloss urban macro 3GPP ("UMa" do 5G-LENA, TS 38.901);
 *     - numerologia mu=0 (SCS 15 kHz, slot de 1 ms — interpretação de
 *       "TTI = 1 ms"; C=52 PRBs em 10 MHz, aceita 106/20 MHz via --bandwidth);
 *     - FADING: o artigo usa o perfil "Pedestrian A" (TS 36.104). O 5G-LENA
 *       não implementa os perfis ITU/36.104; o canal "ThreeGpp"+"UMa" gera o
 *       small-scale fading do modelo 3GPP 38.901 UMa (clusters TDL-like),
 *       que é a aproximação disponível mais próxima (perfil atraso curto no
 *       caso LOS, típico de pedestres). Divergência registrada como [ASSUMIDO]
 *       no registro de premissas.
 *  2. UEs DINÂMICOS: chegadas Poisson lambda=2 UEs/min (inter-chegada
 *     exponencial via Simulator::Schedule), sessão exponencial média 60 s,
 *     slice atribuído uniformemente entre 3 slices eMBB na chegada,
 *     mobilidade pedestre ~1 m/s (random walk 2D com limites no disco da
 *     célula). Partida = fim do tráfego + remoção do mapeamento do scheduler
 *     (a liberação RRC por UE não é exposta publicamente na 5G-LENA v5.2 —
 *     NrGnbRrc::RemoveUe é privado, nr-gnb-rrc.h:1572).
 *  3. SLA POR UE 0,5/1,0/1,5 Mbps (3 slices eMBB, Tabela I), verificada no
 *     Python (y_t = p5 do throughput por UE). Tráfego DL UDP GREEDY
 *     FULL-BUFFER (register R04, ratificado 12/10/2026): taxa oferecida por
 *     UE --ueOfferedMbps com default 100 Mbps — acima do pico absoluto da
 *     célula (~24 Mbps @10 MHz / ~48 Mbps @20 MHz, μ=0, duty DL 3/7), o que
 *     mantém a fila RLC sempre não vazia para qualquer UE/cota/SINR.
 *     Fallback legado: --ueOfferedMbps=0 ⇒ ueOfferedMult × SLA.
 *     Pacote 1500 B (register R04).
 *  4. Período de decisão 100 ms (--periodMs, default 100; rslaq usa 10).
 *  5. KPM estendido (contrato §4): colunas novas
 *     windowThroughputMbps (throughput do UE na janela — UEs que entram/saem
 *     no meio da janela aparecem na janela seguinte com deltas desde a
 *     ENTRADA e janela throughput dividida pelo tempo ATIVO, não pelo período
 *     cheio), snrDb (SINR DL estimado por inversão do CQI wideband nas
 *     tabelas do NrAmc; sentinela de fio -999 = sem CQI, contrato §4/R09) e
 *     queueBytes (fila DL RLC/MAC); colunas originais preservadas.
 *     Arquivo do contrato: sqr-kpms.txt (não colide com runs rslaq).
 *  6. Telemetria de PRB por janela (contrato §3.5), ALINHADA AO MESMO PASSO:
 *     cada linha do sqr_prb_windows.csv drena a janela que acabou de terminar
 *     e traz prb_requested (ação vigente nela, %) + prb_executed (COTA efetiva
 *     em RBGs pós conversão determinística %→RBG) + prb_scheduled (RBGs reais
 *     do MAC) + prb_budget (cota × slots DL) + prb_idle (totalRbg×dlSlots −
 *     Σ prb_scheduled). Degradação do contrato §5 com contadores no relatório
 *     final: CSV ausente/ilegível ⇒ mantém cota anterior; slice sem linha no
 *     CSV ⇒ default 33,33 %/slice (nunca cota 0 silenciosa).
 *  7. SCHEDULER SqrMacScheduler (R27, RATIFICADO 12/10/2026): COTA ABSOLUTA
 *     de RBGs por slice. A ação do agente NÃO é renormalizada: o sqr-sim
 *     converte % de PRB → cota inteira de RBGs (piso + maiores restos,
 *     desempate por índice de slice, Σ ≤ totalRbg garantido — ver
 *     SqrMacScheduler::ComputeRbgQuotas) e o scheduler serve cada slice ATÉ
 *     sua cota; RBGs excedentes ficam OCIOSOS (sem redistribuição); Σa=0 ⇒
 *     nada servido (com log — nunca manter a configuração anterior).
 *
 * Topologia: remoteHost -- P2P 100 Gbps -- PGW/EPC -- gNB NR -- UEs NR (dinâmicos)
 * Modos de execução:
 * - network-only (default, sem --simId): KPMs + telemetria de PRB gerados sem
 *   agente Python (padrão baseline do run_all_scenarios.sh).
 * - IPC com Python (--simId=<uuid>): idêntico ao rslaq — semáforos
 *   /sem_metrics_<uuid> e /sem_control_<uuid> + rslaq_actions_for_ns3.csv.
 */

#include "sqr-mac-scheduler.h"

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <semaphore.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("SqrSim");

// ---------------------------------------------------------------------------
// Tipos e constantes
// ---------------------------------------------------------------------------

// Três slices lógicos, todos eMBB (artigo: 3 slices eMBB com SLA por UE).
// O RslaqMacScheduler exige exatamente 3 slices (SetSliceUeMapping).
enum class SliceType : uint8_t
{
    EMBB0 = 0,
    EMBB1 = 1,
    EMBB2 = 2
};

static const uint32_t NUM_SLICES = 3;
static const uint16_t BASE_DL_PORT = 12000;

// [ARTIGO Tabela I]: SLA por UE por slice (Mbps).
static std::array<double, NUM_SLICES> g_slaMbps = {0.5, 1.0, 1.5};

// ---------------------------------------------------------------------------
// Constantes IPC (verbatim de rslaq-sim.cc)
// ---------------------------------------------------------------------------

static std::string g_simUuid;           // UUID passado via --simId
static sem_t* g_semMetricsReady = nullptr;
static sem_t* g_semControl = nullptr;
static Ptr<SqrMacScheduler> g_schedulerPtr = nullptr;
static Ptr<FlowMonitor> g_monitorPtr = nullptr;
static Ptr<Ipv4FlowClassifier> g_classifierPtr = nullptr;
static FlowMonitorHelper* g_flowmonHelper = nullptr;
static std::map<uint16_t, uint16_t> g_portToUeId;
static std::map<uint16_t, SliceType> g_ueToSlice;
static std::map<uint16_t, uint16_t> g_ueIdToRnti;
static uint32_t g_indicationPeriodMs = 100; // SQR: default 100 ms (artigo: 100 TTIs)
static bool g_ipcEnabled = false;

// ---------------------------------------------------------------------------
// Estado dos UEs dinâmicos
// ---------------------------------------------------------------------------

struct SqrUe
{
    uint16_t ueId = 0;
    SliceType slice = SliceType::EMBB0;
    uint16_t port = 0;
    uint64_t imsi = 0;
    Ptr<Node> node;
    Ptr<NrUeNetDevice> dev;
    Ptr<Application> serverApp;
    Ptr<Application> clientApp;
    double arrivalSec = 0.0;
    double sessionSec = 0.0;      //!< duração de sessão sorteada (Exp(60 s))
    double departureSec = 0.0;    //!< partida efetiva (limitada por simTime)
    double trafficStartSec = 0.0; //!< início do tráfego (arrival + grace) — base dos deltas KPM
    double offeredMbps = 0.0;     //!< taxa ofertada efetiva deste UE (R04)
    bool attached = false;        //!< RNTI válido e registrado no scheduler
    bool active = true;           //!< dentro da sessão
    // Janela parcial (contrato §4): o UE só entra no KPM na PRIMEIRA borda de
    // janela após o início do tráfego ("aparece na janela seguinte com deltas
    // desde a entrada") e sai após a primeira borda ≥ partida.
    bool kpmFinished = false;  //!< linha final (pós-partida) já emitida
};

static std::map<uint16_t, SqrUe> g_ues;
static std::vector<std::vector<uint32_t>> g_sliceRntis(NUM_SLICES); //!< RNTIs por slice
static uint16_t g_nextUeId = 0;
static uint32_t g_arrivalCount = 0;
static uint32_t g_departureCount = 0;
//! Guarda: garante no máximo 1 evento AttachCheck em voo (BUG FIX: o loop era
//! agendado uma única vez no main (t=5 ms); com o primeiro Poisson de chegadas
//! após 5 ms, o AttachCheck encerrava sem reagendar e o RNTI pós-RACH nunca era
//! registrado no mapeamento RNTI→slice — scheduler sem UEs, prb_scheduled=0).
static bool g_attachCheckScheduled = false;

// Configuração dinâmica (CLI)
static double g_lambdaUesPerMin = 2.0; //!< [ARTIGO] chegadas Poisson 2 UEs/min
static double g_meanSessionSec = 60.0; //!< [ARTIGO] sessão exponencial média 60 s
static double g_ueOfferedMbps = 100.0; //!< [ASSUMIDO R04] UDP greedy full-buffer por UE
static double g_ueOfferedMult = 4.0;   //!< fallback legado (usado se --ueOfferedMbps=0)
static double g_simTimeSec = 120.0;
static double g_serverDrainSec = 0.2;    //!< drenagem pós-partida do UdpServer
static double g_attachGraceSec = 0.5;    //!< [ASSUMIDO] margem p/ RRC antes do tráfego
static uint32_t g_pktSize = 1500;        //!< [ASSUMIDO] pacote UDP DL (bytes)
static double g_cellRadiusM = 2000.0;    //!< [ARTIGO Tabela I] raio de célula 2 km
static double g_ueHeightM = 1.5;         //!< [ASSUMIDO] altura de UE pedestre (m)
static double g_slotPeriodSec = 0.001;   //!< duração do slot (1 ms p/ mu=0; setado no main)
static uint32_t g_patternSlots = 7;      //!< nº de slots do padrão TDD aplicado (setado no main)
static std::string g_outputDir = ".";
// Última ação APLICADA ao scheduler (telemetria prb_requested/prb_executed).
// Alinhamento de janela (contrato §3.5): a ação aplicada na borda de janela t
// governa a janela (t, t+period]; o dreno de prb_scheduled/budget/idle da
// janela (t, t+period] acontece na borda t+period e leva ESTES valores — os
// 4 campos do sqr_prb_windows.csv descrevem o MESMO passo. Sentinela -1/0 =
// nenhuma ação do agente ainda (ex.: modo network-only). Os valores PERSISTEM
// entre drenos: timeout/CSV ausente mantêm a config anterior (contrato §5),
// e a janela drenada de fato correu sob ela.
static std::array<double, NUM_SLICES> g_lastRequestedPct = {{-1.0, -1.0, -1.0}};
static std::array<int64_t, NUM_SLICES> g_lastExecutedRbg = {{0, 0, 0}};
// Contadores de degradação (contrato §5), exportados no relatório de fim de
// run (WriteFinalReport) para auditoria contável:
static uint64_t g_actionFileMissingCount = 0; //!< CSV ausente/ilegível ⇒ mantém cota anterior
static uint64_t g_sliceLineMissingCount = 0;  //!< slice sem linha no CSV ⇒ default 33,33 %/slice
static Ptr<NrHelper> g_nrHelper;
static Ptr<NrPointToPointEpcHelper> g_epcHelper;
static NetDeviceContainer g_gnbNetDev;
static Ptr<Node> g_remoteHostNode;
static BandwidthPartInfoPtrVector g_allBwps;
static std::string g_appliedTddPattern;

// Previous KPM values for delta calculation (IPC mode; verbatim rslaq)
struct KpmPrevValues
{
    uint64_t txBytes = 0;
    uint64_t rxBytes = 0;
    uint32_t txPackets = 0;
    uint32_t rxPackets = 0;
    uint32_t lostPackets = 0;
};
static std::map<uint16_t, KpmPrevValues> g_kpmPrevValues;

// ---------------------------------------------------------------------------
// Helpers (copiados de rslaq-sim.cc quando idênticos)
// ---------------------------------------------------------------------------

static std::string
ToUpper(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

static std::string
ToLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static std::string
SliceName(SliceType s)
{
    switch (s)
    {
    case SliceType::EMBB0:
        return "eMBB0";
    case SliceType::EMBB1:
        return "eMBB1";
    case SliceType::EMBB2:
        return "eMBB2";
    default:
        return "?";
    }
}

/**
 * @brief Parseia a lista de SLAs por slice (Mbps), ex. "0.5,1.0,1.5".
 */
static std::array<double, NUM_SLICES>
ParseSla(const std::string& str)
{
    std::array<double, NUM_SLICES> out = {0.0, 0.0, 0.0};
    std::stringstream ss(str);
    std::string token;
    uint32_t i = 0;
    while (std::getline(ss, token, ',') && i < NUM_SLICES)
    {
        out[i++] = std::stod(token);
    }
    if (i != NUM_SLICES)
    {
        NS_FATAL_ERROR("--slaMbps must have exactly " << NUM_SLICES << " values (got " << i << ")");
    }
    for (double v : out)
    {
        if (v <= 0.0)
        {
            NS_FATAL_ERROR("--slaMbps values must be positive");
        }
    }
    return out;
}

static std::vector<double>
ParseWeights(const std::string& str)
{
    std::vector<double> w;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        w.push_back(std::stod(token));
    }
    return w;
}

static RslaqMacScheduler::IntraSliceAlgorithm
ParseIntraAlgo(const std::string& algo)
{
    std::string a = ToUpper(algo);
    if (a == "RR")
    {
        return RslaqMacScheduler::IntraSliceAlgorithm::RR;
    }
    if (a == "PF")
    {
        return RslaqMacScheduler::IntraSliceAlgorithm::PF;
    }
    if (a == "BCQI" || a == "MAXCQI" || a == "MR")
    {
        return RslaqMacScheduler::IntraSliceAlgorithm::BCQI;
    }
    NS_FATAL_ERROR("Invalid intraAlgo: " << algo << " (valid: RR|PF|BCQI)");
}

static std::string
IntraAlgoName(RslaqMacScheduler::IntraSliceAlgorithm algo)
{
    switch (algo)
    {
    case RslaqMacScheduler::IntraSliceAlgorithm::RR:
        return "RR";
    case RslaqMacScheduler::IntraSliceAlgorithm::PF:
        return "PF";
    case RslaqMacScheduler::IntraSliceAlgorithm::BCQI:
        return "BCQI";
    }
    return "UNKNOWN";
}

static RslaqMacScheduler::IntraSliceAlgorithm
IntraAlgoFromActionId(int schedulerId)
{
    switch (schedulerId)
    {
    case 0:
        return RslaqMacScheduler::IntraSliceAlgorithm::RR;
    case 1:
        return RslaqMacScheduler::IntraSliceAlgorithm::PF;
    case 2:
        return RslaqMacScheduler::IntraSliceAlgorithm::BCQI;
    default:
        NS_LOG_WARN("[SqrSim] Unknown action scheduler id " << schedulerId << "; falling back to PF");
        return RslaqMacScheduler::IntraSliceAlgorithm::PF;
    }
}

/**
 * @brief Normaliza o padrão TDD textual (copiado verbatim de rslaq-sim.cc).
 */
static std::string
NormalizeTddPatternForNr(const std::string& requested, bool* applied)
{
    if (requested == "dl_only")
    {
        *applied = true;
        return "DL|DL|DL|DL|DL|DL|DL|DL|DL|DL|";
    }
    if (requested == "D|D|8D|4GB|4U|U|U" || requested == "article")
    {
        *applied = true;
        return "DL|DL|DL|S|UL|UL|UL|";
    }

    std::stringstream ss(requested);
    std::string token;
    std::ostringstream out;
    bool ok = true;
    uint32_t count = 0;
    while (std::getline(ss, token, '|'))
    {
        if (token.empty())
        {
            continue;
        }
        std::string t = ToUpper(token);
        if (t == "D")
        {
            t = "DL";
        }
        else if (t == "U")
        {
            t = "UL";
        }
        else if (t == "GB")
        {
            t = "S";
        }

        if (t != "DL" && t != "UL" && t != "S" && t != "F")
        {
            ok = false;
            break;
        }
        out << t << "|";
        count++;
    }

    *applied = ok && count > 0;
    return *applied ? out.str() : "DL|DL|DL|DL|DL|DL|DL|DL|DL|DL|";
}

/**
 * @brief Envia o mapeamento RNTI→slice atual ao scheduler.
 *
 * Chamado a cada attach/detach/troca de RNTI de UE dinâmico. Nota: o
 * SetSliceUeMapping reinicia os contadores SliceAllocationStats e apenas
 * REDIMENSIONA (sem zerar) os contadores de janela prb_scheduled/prb_budget
 * (rslaq-mac-scheduler.cc:343-356) — a janela corrente NÃO é mais descartada;
 * o único reset legítimo é ResetWindowCounters(), na fronteira de janela pelo
 * dreno do KpmAndControlCallback. O comentário anterior ("reinicia ... a
 * janela prb_scheduled") descrevia o comportamento antigo e ficou obsoleto
 * quando o cenário SQR passou a reenviar o mapeamento a cada 5 ms (churn).
 */
static void
PushSliceMappingToScheduler()
{
    if (!g_schedulerPtr)
    {
        return;
    }
    g_schedulerPtr->SetSliceUeMapping(NUM_SLICES, g_sliceRntis);
}

// ---------------------------------------------------------------------------
// Ciclo de vida dos UEs dinâmicos
// ---------------------------------------------------------------------------

static void ScheduleNextArrival();
static void UeArrival();
static void UeDeparture(uint16_t ueId);
static void AttachCheck();

/**
 * @brief Agenda a próxima chegada: processo de Poisson (inter-chegada Exp).
 *
 * [ARTIGO] lambda = 2 UEs/min global. ChPs: o processo é homogêneo do t=0.
 */
static void
ScheduleNextArrival()
{
    if (g_lambdaUesPerMin <= 0.0)
    {
        return;
    }
    // Inter-chegada exponencial (Poisson homogêneo), média 60/lambda segundos
    Ptr<ExponentialRandomVariable> interArrival = CreateObject<ExponentialRandomVariable>();
    interArrival->SetAttribute("Mean", DoubleValue(60.0 / g_lambdaUesPerMin));
    double dt = interArrival->GetValue();
    if (!(dt > 0.0))
    {
        dt = 0.001;
    }

    // ALINHAMENTO À FASE 0 DO PADRÃO TDD (bug do smoke 11/10/2026, 2a camada):
    // InstallUeDevice inicia o loop de slots do UE PHY com contador SfnSf(0,0,0)
    // no instante da chegada (nr-helper.cc:392 ScheduleStartEventLoop + StartEventLoop).
    // O gNB conta slots desde t=0; um UE criado MID-RUN portanto fica com a FASE
    // DO PADRÃO TDD defasada (ex.: chegada em 41.900 s => gNB no slot 42408 mod 7
    // = 2 (DL), UE no slot 508 mod 7 = 4 (UL)). O gNB então concede DL DATA num
    // slot que para o UE é UL: o executor de varTti do UE já despachou o próprio
    // UL CTRL (sym 13) e o DL DATA (sym 1) é executado "no passado" → o clamp de
    // EndVarTti (nr-ue-phy.cc:1346) escorrega o grid do UE em +928 us PERMANENTEMENTE
    // → UL CTRLs seguintes colidem com recepção DL → NS_FATAL "Cannot TX while RX"
    // (nr-spectrum-phy.cc:1018). Alinhar a chegada a um múltiplo do PERÍODO DO
    // PADRÃO (7 slots = 7 ms com o padrão DL-heavy) faz UE e gNB concordarem da
    // fase do padrão em todos os slots — emula a sincronização de SFN do acesso
    // inicial real (o UE lê o SFN no MIB antes de transmitir). [ASSUMIDO]
    const int64_t alignNs = static_cast<int64_t>(g_slotPeriodSec * 1e9) *
                            static_cast<int64_t>(g_patternSlots);
    const int64_t nowNs = Simulator::Now().GetNanoSeconds();
    const int64_t targetNs = nowNs + static_cast<int64_t>(dt * 1e9);
    int64_t alignedNs = ((targetNs + alignNs - 1) / alignNs) * alignNs;
    if (alignedNs <= nowNs)
    {
        alignedNs = nowNs + alignNs;
    }

    // Chegadas apenas dentro da janela ativa (deixa a sessão ter ao menos o drain)
    const double nextSec = alignedNs / 1e9;
    if (nextSec <= g_simTimeSec - g_serverDrainSec)
    {
        Simulator::Schedule(NanoSeconds(alignedNs - nowNs), &UeArrival);
    }
}

/**
 * @brief Chegada de um UE: cria nó, dispositivo NR, apps e agenda a partida.
 *
 * A partida (sessão) também é sorteada aqui — Exp(60 s) [ARTIGO] — e limitada
 * por simTime. Como Application::SetStartTime/SetStopTime são RELATIVOS ao
 * tempo de instalação (src/network/model/application.cc:88-91 tratam o Time
 * como delay; ver application.h:64), todos os tempos abaixo são delays a
 * partir do instante de chegada.
 */
static void
UeArrival()
{
    g_arrivalCount++;
    g_nextUeId += 1;
    const uint16_t ueId = g_nextUeId;
    const double nowSec = Simulator::Now().GetSeconds();

    // Slice uniforme entre os 3 eMBB [ARTIGO]. IMPORTANTE: RandomVariableStream
    // NÃO pode ser alocado na stack — o RngStream só é criado pela construção
    // de atributos (SetStream via Object::Construct); stack → m_rng nulo →
    // SIGSEGV em RandU01.
    Ptr<UniformRandomVariable> uni = CreateObject<UniformRandomVariable>();
    const uint32_t sliceIdx = static_cast<uint32_t>(uni->GetInteger(0, NUM_SLICES - 1));
    const SliceType slice = static_cast<SliceType>(sliceIdx);

    // Sessão exponencial média 60 s [ARTIGO]
    Ptr<ExponentialRandomVariable> sessionRv = CreateObject<ExponentialRandomVariable>();
    sessionRv->SetAttribute("Mean", DoubleValue(g_meanSessionSec));
    double sessionSec = sessionRv->GetValue();
    if (!(sessionSec > 0.0))
    {
        sessionSec = 0.1;
    }
    const double departureSec = std::min(nowSec + sessionSec, g_simTimeSec);

    // --- Nó + pilha IP ---
    NodeContainer ueContainer;
    ueContainer.Create(1);
    Ptr<Node> ueNode = ueContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(ueNode);

    // --- Mobilidade pedestre ~1 m/s (random walk 2D limitado ao disco) ---
    // [ASSUMIDO] velocidade U(0.5, 1.5) m/s (média 1 m/s); direção/pausas do
    // modelo RandomWalk2d. O artigo publica apenas "trajetórias aleatórias".
    const double cellRadius = g_cellRadiusM;
    MobilityHelper ueMob;
    ueMob.SetPositionAllocator("ns3::UniformDiscPositionAllocator",
                               "rho",
                               DoubleValue(cellRadius),
                               "X",
                               DoubleValue(0.0),
                               "Y",
                               DoubleValue(0.0),
                               "Z",
                               DoubleValue(g_ueHeightM));
    ueMob.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                           "Mode",
                           StringValue("Time"),
                           "Time",
                           StringValue("2s"),
                           "Speed",
                           StringValue("ns3::UniformRandomVariable[Min=0.5|Max=1.5]"),
                           "Bounds",
                           RectangleValue(Rectangle(-cellRadius, cellRadius, -cellRadius, cellRadius)));
    ueMob.Install(ueNode);

    // --- Dispositivo NR (numerologia herdada do gNB: nr-helper.cc:1093) ---
    NetDeviceContainer ueDevContainer = g_nrHelper->InstallUeDevice(ueContainer, g_allBwps);
    Ptr<NrUeNetDevice> ueDev = DynamicCast<NrUeNetDevice>(ueDevContainer.Get(0));
    g_nrHelper->GetUePhy(ueDev, 0)->SetPattern(g_appliedTddPattern);

    Ipv4InterfaceContainer ueIpIfaces = g_epcHelper->AssignUeIpv4Address(ueDevContainer);
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> ueStatic =
        ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
    ueStatic->SetDefaultRoute(g_epcHelper->GetUeDefaultGatewayAddress(), 1);
    // Idêntico ao rslaq: UE nunca repassa pacotes (evita crash do classificador
    // QoS sob corrupção RLC-UM; ver comentário em rslaq-sim.cc).
    ueNode->GetObject<Ipv4>()->SetAttribute("IpForward", BooleanValue(false));

    // --- Attach (procedimento RACH event-driven) ---
    g_nrHelper->AttachToGnb(ueDevContainer.Get(0), g_gnbNetDev.Get(0));

    // --- FlowMonitor no novo nó (mesmo monitor compartilhado; probe por nó) ---
    g_monitorPtr = g_flowmonHelper->Install(ueNode);

    // --- Aplicações DL ---
    const uint16_t port = static_cast<uint16_t>(BASE_DL_PORT + ueId);
    // R04 (greedy full-buffer): taxa absoluta por UE (default 100 Mbps > 2× o
    // pico da célula ⇒ fila sempre não vazia); 0 = fallback legado mult × SLA.
    const double offeredMbps =
        (g_ueOfferedMbps > 0.0) ? g_ueOfferedMbps : g_ueOfferedMult * g_slaMbps[sliceIdx];
    const double offeredBps = offeredMbps * 1e6;

    UdpServerHelper serverHelper(port);
    ApplicationContainer serverApps = serverHelper.Install(ueNode);
    Ptr<Application> serverApp = serverApps.Get(0);
    // Delays relativos à chegada (semântica SetStartTime/SetStopTime)
    serverApp->SetStartTime(Seconds(0.0));
    serverApp->SetStopTime(Seconds(std::min(departureSec + g_serverDrainSec, g_simTimeSec + g_serverDrainSec) - nowSec));

    OnOffHelper onOff("ns3::UdpSocketFactory", InetSocketAddress(ueIpIfaces.GetAddress(0), port));
    onOff.SetConstantRate(DataRate(offeredBps), g_pktSize);
    ApplicationContainer clientApps = onOff.Install(g_remoteHostNode);
    Ptr<Application> clientApp = clientApps.Get(0);
    // Tráfego só após a margem de conexão RRC [ASSUMIDO] e até a partida
    clientApp->SetStartTime(Seconds(g_attachGraceSec));
    clientApp->SetStopTime(Seconds(std::max(departureSec - nowSec, g_attachGraceSec + 0.1)));

    // --- Registro de estado ---
    SqrUe ue;
    ue.ueId = ueId;
    ue.slice = slice;
    ue.port = port;
    ue.node = ueNode;
    ue.dev = ueDev;
    ue.serverApp = serverApp;
    ue.clientApp = clientApp;
    ue.arrivalSec = nowSec;
    ue.sessionSec = sessionSec;
    ue.departureSec = departureSec;
    ue.trafficStartSec = nowSec + g_attachGraceSec; // = SetStartTime do OnOff
    ue.offeredMbps = offeredMbps;
    ue.active = true;
    ue.attached = false;
    ue.imsi = ueDev ? ueDev->GetImsi() : 0;
    g_ues[ueId] = ue;

    g_portToUeId[port] = ueId;
    g_ueToSlice[ueId] = slice;
    // O registro do RNTI no scheduler é feito pelo loop contínuo AttachCheck()
    // (também cobre trocas de RNTI por RLF/reestabelecimento).

    std::cout << std::fixed << std::setprecision(3)
              << "[SqrSim] UE_ARRIVAL ueId=" << ueId
              << " slice=" << sliceIdx
              << " t=" << nowSec << "s"
              << " session=" << sessionSec << "s"
              << " departure=" << departureSec << "s"
              << " port=" << port
              << " offered=" << offeredMbps << " Mbps"
              << " sla=" << g_slaMbps[sliceIdx] << " Mbps"
              << " activeUes=" << g_arrivalCount - g_departureCount << "\n";

    // --- Partida da sessão ---
    Simulator::Schedule(Seconds(departureSec - nowSec), &UeDeparture, ueId);

    // --- Próxima chegada (Poisson) ---
    ScheduleNextArrival();
}

/**
 * @brief Partida de UE: fim da sessão.
 *
 * Os apps já param nos seus stop times (client = partida; server = partida +
 * drain). Aqui removemos o UE do mapeamento do scheduler para que nunca mais
 * receba recursos. A liberação RRC por UE não é exposta publicamente na
 * 5G-LENA v5.2 (NrGnbRrc::RemoveUe é privado, nr-gnb-rrc.h:1572); como o
 * scheduler só aloca para UEs com demanda (BSR > 0), um UE sem tráfego não
 * consome PRBs — partida modelada como fim de tráfego + remoção de mapeamento
 * [ASSUMIDO].
 */
static void
UeDeparture(uint16_t ueId)
{
    auto it = g_ues.find(ueId);
    if (it == g_ues.end() || !it->second.active)
    {
        return;
    }
    SqrUe& ue = it->second;
    ue.active = false;
    g_departureCount++;

    // Remove do mapeamento RNTI→slice
    uint16_t rnti = 0;
    if (g_ueIdToRnti.count(ueId))
    {
        rnti = g_ueIdToRnti[ueId];
        g_ueIdToRnti.erase(ueId);
    }
    if (rnti != 0)
    {
        const uint32_t sliceIdx = static_cast<uint32_t>(ue.slice);
        auto& vec = g_sliceRntis[sliceIdx];
        vec.erase(std::remove(vec.begin(), vec.end(), static_cast<uint32_t>(rnti)), vec.end());
        PushSliceMappingToScheduler();
    }

    std::cout << std::fixed << std::setprecision(3)
              << "[SqrSim] UE_DEPARTURE ueId=" << ueId
              << " slice=" << static_cast<uint32_t>(ue.slice)
              << " t=" << Simulator::Now().GetSeconds() << "s"
              << " sessionDrawn=" << ue.sessionSec << "s"
              << " activeUes=" << g_arrivalCount - g_departureCount << "\n";
}

/**
 * @brief Sincroniza o mapeamento RNTI→slice de todos os UEs ativos (a cada 5 ms).
 *
 * Dois eventos cobertos:
 *  1. primeiro attach: o RRC só recebe RNTI após o RACH (T-C-RNTI no RAR);
 *  2. TROCA DE RNTI em plena sessão: UEs de borda podem sofrer RLF
 *     (out-of-sync → reestabelecimento RRC) e voltar com um C-RNTI NOVO —
 *     sem esta atualização o mapeamento fica obsoleto e o UE vira buraco
 *     negro de dados (observado no smoke: UE com RNTI 2 → RLF em 50.75 s →
 *     RNTI 4 não registrado; 9433 linhas "not_in_any_slice").
 * Loop contínuo com exatamente um evento em voo (g_attachCheckScheduled).
 */
static void
AttachCheck()
{
    g_attachCheckScheduled = false; // este evento está sendo consumido

    bool mappingChanged = false;

    for (auto& kv : g_ues)
    {
        SqrUe& ue = kv.second;
        if (!ue.active || !ue.dev || !ue.dev->GetRrc())
        {
            continue;
        }
        const uint16_t rnti = ue.dev->GetRrc()->GetRnti();
        if (rnti == 0 || rnti == UINT16_MAX)
        {
            continue; // ainda em RACH
        }

        const uint32_t sliceIdx = static_cast<uint32_t>(ue.slice);
        auto itR = g_ueIdToRnti.find(ue.ueId);
        if (itR == g_ueIdToRnti.end())
        {
            // Primeiro attach pós-RACH
            ue.attached = true;
            ue.imsi = ue.dev->GetImsi();
            g_ueIdToRnti[ue.ueId] = rnti;
            g_sliceRntis[sliceIdx].push_back(rnti);
            mappingChanged = true;
            std::cout << std::fixed << std::setprecision(3)
                      << "[SqrSim] UE_ATTACHED ueId=" << ue.ueId
                      << " rnti=" << rnti
                      << " slice=" << sliceIdx
                      << " t=" << Simulator::Now().GetSeconds() << "s\n";
        }
        else if (itR->second != rnti)
        {
            // Reestabelecimento RRC (RLF): o C-RNTI mudou — atualiza o mapeamento
            const uint16_t oldRnti = itR->second;
            auto& vec = g_sliceRntis[sliceIdx];
            vec.erase(std::remove(vec.begin(), vec.end(), oldRnti), vec.end());
            vec.push_back(rnti);
            g_ueIdToRnti[ue.ueId] = rnti;
            mappingChanged = true;
            std::cout << std::fixed << std::setprecision(3)
                      << "[SqrSim] UE_RNTI_CHANGED ueId=" << ue.ueId
                      << " oldRnti=" << oldRnti
                      << " newRnti=" << rnti
                      << " slice=" << sliceIdx
                      << " t=" << Simulator::Now().GetSeconds() << "s\n";
        }
    }

    if (mappingChanged)
    {
        PushSliceMappingToScheduler();
    }

    // Loop contínuo: churn de UEs e RLF podem ocorrer a qualquer momento; o
    // custo de re-verificar a cada 5 ms é desprezível (poucos UEs, leitura de
    // um campo do RRC por UE).
    g_attachCheckScheduled = true;
    Simulator::Schedule(MilliSeconds(5), &AttachCheck);
}

// ---------------------------------------------------------------------------
// Callback IPC: KPMs estendidos + ação do agente (adaptado de rslaq-sim.cc;
// mecanismo de semáforos/CSV preservado verbatim)
// ---------------------------------------------------------------------------

static void
KpmAndControlCallback()
{
    // SQR (divergência vs rslaq): o callback roda TAMBÉM em modo network-only
    // para exportar os KPMs estendidos e a telemetria prb_* (contrato SQR);
    // as partes de IPC (semáforo/ação) permanecem gated por g_ipcEnabled.
    uint64_t nowMs = static_cast<uint64_t>(Simulator::Now().GetMilliSeconds());

    // Coletar métricas do FlowMonitor (verbatim rslaq)
    if (g_monitorPtr && g_classifierPtr)
    {
        g_monitorPtr->CheckForLostPackets();
        FlowMonitor::FlowStatsContainer stats = g_monitorPtr->GetFlowStats();

        struct UeKpm
        {
            uint64_t txBytes = 0;
            uint64_t rxBytes = 0;
            uint32_t txPackets = 0;
            uint32_t rxPackets = 0;
            uint32_t lostPackets = 0;
            double delaySumSec = 0.0;
            uint32_t count = 0;
        };

        std::map<uint16_t, UeKpm> ueKpms; // key = ueId

        for (const auto& kv : stats)
        {
            FlowId flowId = kv.first;
            const FlowMonitor::FlowStats& st = kv.second;
            Ipv4FlowClassifier::FiveTuple tuple = g_classifierPtr->FindFlow(flowId);

            if (tuple.protocol != 17)
                continue;

            auto itPort = g_portToUeId.find(tuple.destinationPort);
            if (itPort == g_portToUeId.end())
                continue;

            uint16_t ueId = itPort->second;
            UeKpm& k = ueKpms[ueId];
            k.txBytes += st.txBytes;
            k.rxBytes += st.rxBytes;
            k.txPackets += st.txPackets;
            k.rxPackets += st.rxPackets;
            k.lostPackets += st.lostPackets;
            k.delaySumSec += st.delaySum.GetSeconds();
            k.count++;
        }

        // Escrever sqr-kpms.txt — KPM ESTENDIDO do contrato SQR §4 (nome novo:
        // não colide com o rslaq-kpms.txt dos runs rslaq, rastreado no git).
        // Colunas originais preservadas + windowThroughputMbps, snrDb, queueBytes.
        //
        // JANELA PARCIAL (contrato §4): UEs que entram no meio da janela
        // aparecem na janela SEGUINTE com deltas desde a ENTRADA (1ª borda de
        // janela após o início do tráfego); UEs que saem no meio geram uma
        // última linha com delta até a partida. windowThroughputMbps divide o
        // delta pelo tempo ATIVO do UE na janela (NÃO pelo período cheio — a
        // divisão pelo período cheio subestimava até ~20× a taxa e distorcia
        // o p5). A coluna legada throughputMbps mantém a semântica antiga
        // (delta / período cheio) por compatibilidade (contrato §4 col. 10).
        static std::ofstream kpmHistory;
        if (!kpmHistory.is_open())
        {
            std::filesystem::create_directories(g_outputDir);
            kpmHistory.open(g_outputDir + "/sqr_kpm_history.csv",
                            std::ios::out | std::ios::trunc);
            kpmHistory << "timestamp,ueImsi,sliceId,dTxBytes,dRxBytes,plr,resourceSharePct,"
                          "dLostPackets,throughputMbps,bufferBytes,windowThroughputMbps,snrDb,"
                          "queueBytes\n";
        }
        {
            std::ofstream kpmFile("sqr-kpms.txt", std::ios::out | std::ios::trunc);
            kpmFile << "timestamp,ueImsi,sliceId,dTxBytes,dRxBytes,plr,resourceSharePct,"
                       "dLostPackets,throughputMbps,bufferBytes,windowThroughputMbps,snrDb,"
                       "queueBytes\n";

            const double periodSec = static_cast<double>(g_indicationPeriodMs) / 1000.0;
            const double tEndSec = static_cast<double>(nowMs) / 1000.0;
            const double tWinStartSec = tEndSec - periodSec;

            // Cota corrente (R27) para o resourceSharePct do KPM: fração de RBG
            // da slice no BWP — não mais o peso renormalizado da base.
            const std::vector<uint32_t> quotaNow =
                g_schedulerPtr ? g_schedulerPtr->GetSliceQuotaRbg() : std::vector<uint32_t>{};
            const uint16_t totalRbgNow = g_schedulerPtr ? g_schedulerPtr->GetTotalRbgCount() : 0;

            auto emitRow = [&](std::ofstream& out,
                               uint64_t ts, uint16_t ueId, uint32_t sliceIdx,
                               uint64_t dTx, uint64_t dRx, double plr, double rsh,
                               uint32_t effLost, double thrLegacy, uint32_t buf,
                               double windowThr, double snr, uint32_t queue) {
                out << ts << "," << ueId << "," << sliceIdx << ","
                    << dTx << "," << dRx << ","
                    << std::fixed << std::setprecision(2) << plr << ","
                    << std::setprecision(2) << rsh << ","
                    << effLost << ","
                    << std::setprecision(4) << thrLegacy << ","
                    << buf << ","
                    << std::setprecision(4) << windowThr << ","
                    << std::setprecision(2) << snr << ","
                    << queue << "\n";
            };

            for (auto& kv : g_ues)
            {
                SqrUe& ue = kv.second;
                if (ue.kpmFinished)
                {
                    continue; // linha final pós-partida já emitida
                }

                // --- Elegibilidade de janela parcial (contrato §4) ---
                // 1) aparece apenas na 1ª borda de janela APÓS o início do
                //    tráfego ("janela seguinte", delta desde a entrada);
                if (!(tEndSec > ue.trafficStartSec + 1e-9))
                {
                    continue;
                }
                // 2) exige sobreposição ativa com a janela; sem ela, sem linha
                //    (e o UE encerra a emissão se a última janela ativa já
                //    foi coberta).
                if (!(ue.departureSec > tWinStartSec + 1e-9))
                {
                    ue.kpmFinished = true;
                    continue;
                }

                const uint16_t ueId = ue.ueId;
                const uint32_t sliceIdx = static_cast<uint32_t>(ue.slice);

                static const UeKpm zero{};
                auto itK = ueKpms.find(ueId);
                const UeKpm& k = (itK != ueKpms.end()) ? itK->second : zero;

                // Get previous values and calculate deltas (verbatim rslaq;
                // prev só é atualizado quando uma linha é emitida — assim o
                // 1º delta cobre tudo desde a entrada do UE).
                KpmPrevValues& prev = g_kpmPrevValues[ueId];
                uint64_t dTxBytes = (k.txBytes >= prev.txBytes) ? (k.txBytes - prev.txBytes) : 0;
                uint64_t dRxBytes = (k.rxBytes >= prev.rxBytes) ? (k.rxBytes - prev.rxBytes) : 0;
                uint32_t dTxPackets = (k.txPackets >= prev.txPackets) ? (k.txPackets - prev.txPackets) : 0;
                uint32_t dRxPackets = (k.rxPackets >= prev.rxPackets) ? (k.rxPackets - prev.rxPackets) : 0;

                prev.txBytes = k.txBytes;
                prev.rxBytes = k.rxBytes;
                prev.txPackets = k.txPackets;
                prev.rxPackets = k.rxPackets;
                prev.lostPackets = k.lostPackets;

                uint32_t effectiveLost = (dTxPackets >= dRxPackets) ? (dTxPackets - dRxPackets) : 0;

                double plr = (dTxPackets > 0)
                                 ? (static_cast<double>(effectiveLost) / dTxPackets * 100.0)
                                 : 0.0;
                double rsh = (g_schedulerPtr && totalRbgNow > 0 && sliceIdx < quotaNow.size())
                                 ? (100.0 * static_cast<double>(quotaNow[sliceIdx]) /
                                    static_cast<double>(totalRbgNow))
                                 : 0.0;
                // Coluna legada (contrato §4): delta / período cheio.
                double thrMbps = (periodSec > 0)
                                     ? (static_cast<double>(dRxBytes) * 8.0 / periodSec / 1e6)
                                     : 0.0;

                uint32_t bufferBytes = (g_schedulerPtr && g_ueIdToRnti.count(ueId))
                                           ? g_schedulerPtr->GetUeDlBufferSize(g_ueIdToRnti[ueId])
                                           : 0;

                // --- Colunas novas do contrato SQR ---
                // windowThroughputMbps: delta de bytes da janela ÷ tempo ATIVO
                // do UE na janela (janela parcial: desde a entrada / até a
                // saída; janela cheia: o período).
                const double activeStartSec = std::max(ue.trafficStartSec, tWinStartSec);
                const double activeEndSec = std::min(ue.departureSec, tEndSec);
                const double activeSec = std::max(activeEndSec - activeStartSec, 1e-6);
                double windowThrMbps =
                    static_cast<double>(dRxBytes) * 8.0 / activeSec / 1e6;
                // snrDb: SINR DL estimado (inversão do CQI wideband nas tabelas
                // do NrAmc; ver comentário "SQR" no scheduler). Sentinela do
                // contrato §4/R09: -999.0 = UE ainda sem CQI válido (não confun-
                // dir com medição real — clip(−999/30,0,1)=0 era ambíguo).
                double snrDb = -999.0;
                if (g_schedulerPtr && g_ueIdToRnti.count(ueId))
                {
                    snrDb = g_schedulerPtr->GetUeDlSinrDb(g_ueIdToRnti[ueId]);
                }
                // queueBytes: fila DL do UE no MAC/RLC (mesma fonte do bufferBytes)
                uint32_t queueBytes = bufferBytes;

                emitRow(kpmFile, nowMs, ueId, sliceIdx, dTxBytes, dRxBytes, plr, rsh,
                        effectiveLost, thrMbps, bufferBytes, windowThrMbps, snrDb, queueBytes);
                emitRow(kpmHistory, nowMs, ueId, sliceIdx, dTxBytes, dRxBytes, plr, rsh,
                        effectiveLost, thrMbps, bufferBytes, windowThrMbps, snrDb, queueBytes);

                if (ue.departureSec <= tEndSec + 1e-9)
                {
                    ue.kpmFinished = true; // última janela ativa emitida
                }
            }
            kpmFile.flush();
            kpmHistory.flush();
        }
    }

    // Sinalizar ao Python: métricas prontas (verbatim rslaq)
    if (g_ipcEnabled && g_semMetricsReady)
    {
        sem_post(g_semMetricsReady);
    }

    // --- Telemetria SQR por janela: prb_scheduled / prb_idle (contrato §3.5) ---
    // ALINHAMENTO DE JANELA: este dreno roda ANTES da espera pela ação do
    // agente (que fica logo abaixo). Assim a linha do sqr_prb_windows.csv
    // escrita na borda t descreve a janela (t−period, t] com os 4 campos do
    // MESMO passo: prb_scheduled/prb_budget/prb_idle drenam os contadores
    // acumulados NESSA janela e prb_requested/prb_executed (COTA efetiva em
    // RBGs) trazem a ação VIGENTE nela — aplicada na borda ANTERIOR (t−period)
    // e ainda armazenada em g_lastRequestedPct/g_lastExecutedRbg. A ação
    // recebida abaixo, na borda t, governa a janela (t, t+period] e só entra
    // no CSV do próximo dreno. Sentinela -1/0 = janela sem ação do agente
    // (ex.: modo network-only ou 1ª janela). prb_idle = RBGs OCIOSOS da
    // janela (contrato §3.5, agregação da definição por slot C − Σ
    // prb_scheduled): totalRbg × dlSlots − Σ prb_scheduled, com dlSlots = nº
    // de slots DL com chamada de scheduling (m_windowDlSlots; slots sem
    // nenhum UE ativo não têm chamada e ficam fora da base). Os valores de
    // g_last* PERSISTEM entre drenos: timeout/CSV ausente mantêm a config
    // anterior (contrato §5) e a janela drenada de fato correu sob ela.
    if (g_schedulerPtr)
    {
        static std::ofstream prbFile;
        if (!prbFile.is_open())
        {
            std::filesystem::create_directories(g_outputDir);
            prbFile.open(g_outputDir + "/sqr_prb_windows.csv",
                         std::ios::out | std::ios::trunc);
            prbFile << "timestamp_ms,total_rbg,total_prb,dl_slots,"
                       "prb_requested_pct_0,prb_requested_pct_1,prb_requested_pct_2,"
                       "prb_executed_rbg_0,prb_executed_rbg_1,prb_executed_rbg_2,"
                       "prb_scheduled_rbg_0,prb_scheduled_rbg_1,prb_scheduled_rbg_2,"
                       "prb_budget_rbg_0,prb_budget_rbg_1,prb_budget_rbg_2,"
                       "prb_idle_rbg\n";
        }

        const std::vector<uint64_t> sched = g_schedulerPtr->GetWindowScheduledRbg();
        const std::vector<uint64_t> budget = g_schedulerPtr->GetWindowBudgetRbg();
        const uint16_t totalRbg = g_schedulerPtr->GetTotalRbgCount();
        const uint64_t totalPrb = g_schedulerPtr->GetTotalPrbCount();
        const uint64_t dlSlots = g_schedulerPtr->GetWindowDlSlots();

        uint64_t schedSum = 0;
        for (uint32_t s = 0; s < NUM_SLICES; s++)
        {
            schedSum += (s < sched.size() ? sched[s] : 0);
        }
        const uint64_t totalRbgSlots = static_cast<uint64_t>(totalRbg) * dlSlots;
        const uint64_t prbIdle =
            (totalRbgSlots > schedSum) ? (totalRbgSlots - schedSum) : 0;

        prbFile << nowMs << "," << totalRbg << "," << totalPrb << "," << dlSlots;
        for (uint32_t s = 0; s < NUM_SLICES; s++)
        {
            prbFile << "," << std::fixed << std::setprecision(2) << g_lastRequestedPct[s];
        }
        for (uint32_t s = 0; s < NUM_SLICES; s++)
        {
            prbFile << "," << g_lastExecutedRbg[s];
        }
        for (uint32_t s = 0; s < NUM_SLICES; s++)
        {
            prbFile << "," << (s < sched.size() ? sched[s] : 0);
        }
        for (uint32_t s = 0; s < NUM_SLICES; s++)
        {
            prbFile << "," << (s < budget.size() ? budget[s] : 0);
        }
        prbFile << "," << prbIdle << "\n";
        prbFile.flush();

        std::cout << std::fixed << std::setprecision(0)
                  << "[SqrSim] PRB_WINDOW t=" << nowMs << "ms"
                  << " prb_scheduled_rbg=[" << (sched.size() > 0 ? sched[0] : 0) << ", "
                  << (sched.size() > 1 ? sched[1] : 0) << ", "
                  << (sched.size() > 2 ? sched[2] : 0) << "]"
                  << " prb_budget_rbg=[" << (budget.size() > 0 ? budget[0] : 0) << ", "
                  << (budget.size() > 1 ? budget[1] : 0) << ", "
                  << (budget.size() > 2 ? budget[2] : 0) << "]"
                  << " prb_executed_rbg=[" << g_lastExecutedRbg[0] << ", "
                  << g_lastExecutedRbg[1] << ", " << g_lastExecutedRbg[2] << "]"
                  << " dl_slots=" << dlSlots
                  << " prb_idle_rbg=" << prbIdle
                  << " totalRbg=" << totalRbg << "\n";

        // Único reset legítimo dos contadores de janela (fronteira de janela).
        // g_lastRequestedPct/g_lastExecutedRbg NÃO são zerados: eles descrevem
        // a ação vigente e devem sobreviver até a próxima ação ser aplicada.
        g_schedulerPtr->ResetWindowCounters();
    }

    // Esperar ação do agente (timeout de 50ms; verbatim rslaq)
    if (g_ipcEnabled && g_semControl)
    {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 50 * 1000 * 1000; // 50ms
        if (ts.tv_nsec >= 1000000000)
        {
            ts.tv_sec += 1;
            ts.tv_nsec -= 1000000000;
        }
        int ret = sem_timedwait(g_semControl, &ts);
        if (ret == 0)
        {
            // Ler ação do agente (formato do contrato §3: timestamp,sliceId,
            // dedicatedPRB,minPRB,maxPRB[,algorithm]; percentuais 0–100).
            std::ifstream actionFile("rslaq_actions_for_ns3.csv");
            if (actionFile.is_open())
            {
                std::string line;
                std::vector<double> dedicatedPrb(NUM_SLICES, -1.0);
                std::vector<RslaqMacScheduler::IntraSliceAlgorithm> algos(
                    NUM_SLICES,
                    RslaqMacScheduler::IntraSliceAlgorithm::PF);
                while (std::getline(actionFile, line))
                {
                    std::stringstream ss(line);
                    std::string token;
                    std::vector<std::string> cols;
                    while (std::getline(ss, token, ','))
                    {
                        cols.push_back(token);
                    }
                    if (cols.size() >= 4)
                    {
                        try
                        {
                            uint32_t sliceId = static_cast<uint32_t>(std::stoi(cols[1]));
                            double ded = std::stod(cols[2]);
                            if (sliceId < NUM_SLICES)
                            {
                                dedicatedPrb[sliceId] = ded;
                                if (cols.size() >= 6)
                                {
                                    algos[sliceId] = IntraAlgoFromActionId(std::stoi(cols[5]));
                                }
                            }
                        }
                        catch (const std::exception&)
                        {
                            continue;
                        }
                    }
                }
                actionFile.close();

                // ==========================================================
                // SQR-R27 (RATIFICADO 12/10/2026): COTA ABSOLUTA — NADA de
                // renormalização. O caminho antigo dividia por totalDed e
                // renormalizava para soma 1 (o scheduler re-expansibia Σa < C
                // para partição completa, anulando γ·a/C e a Fig. 4). Agora:
                // (1) % de PRB → cota inteira de RBGs com a regra determinís-
                //     tica SqrMacScheduler::ComputeRbgQuotas (piso + maiores
                //     restos, desempate por índice de slice; Σ quotas ≤
                //     totalRbg garantido; SEM renormalização);
                // (2) a cota é aplicada SEMPRE — inclusive Σa = 0 (nada
                //     servido, com log; NUNCA manter a configuração anterior
                //     silenciosamente, como o código antigo fazia ao descartar
                //     totalDed ≤ 0). Timeout do semáforo continua mantendo a
                //     config anterior (contrato §5 — degradação congelada).
                // ==========================================================
                // DEGRADAÇÃO (contrato §5, RATIFICADO 13/10/2026): slice SEM
                // linha no CSV ⇒ default do caminho vigente 33,33 %/slice
                // (rslaq-sim.cc:1306) com warning contável — NUNCA cota 0
                // silenciosa (0 explícito na linha da slice é ação legítima
                // e continua sendo aplicado como tal, R27).
                std::array<double, NUM_SLICES> pct = {};
                std::vector<uint32_t> missingSlices;
                for (uint32_t s = 0; s < NUM_SLICES; s++)
                {
                    if (dedicatedPrb[s] < 0.0)
                    {
                        pct[s] = 33.33;
                        missingSlices.push_back(s);
                    }
                    else
                    {
                        pct[s] = dedicatedPrb[s];
                    }
                }
                if (!missingSlices.empty())
                {
                    g_sliceLineMissingCount++;
                    std::ostringstream missList;
                    for (uint32_t s : missingSlices)
                    {
                        missList << (missingSlices.front() == s ? "" : ",") << s;
                    }
                    NS_LOG_WARN("[SqrSim] action CSV without line(s) for slice(s) ["
                                << missList.str() << "] — applying default 33.33%"
                                " (degradation #" << g_sliceLineMissingCount << ", contract §5)");
                    if (g_sliceLineMissingCount == 1)
                    {
                        std::cout << "[SqrSim] WARNING: action CSV without line(s) for"
                                  << " slice(s) [" << missList.str()
                                  << "] — default 33.33%/slice applied (contract §5,"
                                  << " countable; see final report)\n";
                    }
                }

                const uint16_t totalRbg = g_schedulerPtr->GetTotalRbgCount();
                const std::vector<double> pctVec(pct.begin(), pct.end());
                const std::vector<uint32_t> quotaRbg =
                    SqrMacScheduler::ComputeRbgQuotas(pctVec, totalRbg, NUM_SLICES);
                g_schedulerPtr->SetSliceQuotaRbg(quotaRbg, algos);

                // --- Telemetria SQR: prb_requested / prb_executed ---
                // prb_requested = ação aplicada (% por slice; linha ausente ⇒
                // default 33,33 já substituído acima); prb_executed = COTA
                // EFETIVA em RBGs entregue ao scheduler (pós-conversão
                // determinística %→RBG). Ficam em g_last* e são consumidos
                // pelo dreno da janela que esta ação governa (próxima borda —
                // ver comentário do dreno acima).
                double pctSum = 0.0;
                for (uint32_t s = 0; s < NUM_SLICES; s++)
                {
                    g_lastRequestedPct[s] = pct[s];
                    g_lastExecutedRbg[s] =
                        (s < quotaRbg.size()) ? static_cast<int64_t>(quotaRbg[s]) : 0;
                    pctSum += pct[s];
                }

                std::cout << std::fixed << std::setprecision(2)
                          << "[SqrSim] ACTION t=" << nowMs << "ms"
                          << " prb_requested_pct=[" << pct[0] << ", "
                          << pct[1] << ", " << pct[2] << "]"
                          << " prb_executed_rbg=[" << g_lastExecutedRbg[0] << ", "
                          << g_lastExecutedRbg[1] << ", " << g_lastExecutedRbg[2] << "]"
                          << " totalRbg=" << totalRbg
                          << " pctSum=" << pctSum
                          << (pctSum <= 0.0 ? " zero_quota_nothing_served" : "")
                          << "\n";

                NS_LOG_INFO("[SqrSim] IPC absolute quota applied at t=" << nowMs
                            << "ms: quota_rbg=[" << g_lastExecutedRbg[0] << ", "
                            << g_lastExecutedRbg[1] << ", " << g_lastExecutedRbg[2] << "]"
                            << " algos=[" << IntraAlgoName(algos[0]) << ", "
                            << IntraAlgoName(algos[1]) << ", "
                            << IntraAlgoName(algos[2]) << "]");
            }
            else
            {
                // DEGRADAÇÃO (contrato §5): semáforo liberado mas CSV ausente/
                // ilegível ⇒ mantém a cota anterior COM warning contável.
                g_actionFileMissingCount++;
                NS_LOG_WARN("[SqrSim] control semaphore released but rslaq_actions_for_ns3.csv"
                            " is missing/unreadable — KEEPING previous quota configuration"
                            " (degradation #" << g_actionFileMissingCount << ", contract §5)");
                if (g_actionFileMissingCount == 1)
                {
                    std::cout << "[SqrSim] WARNING: rslaq_actions_for_ns3.csv missing/"
                              << "unreadable after control semaphore — KEEPING previous"
                              << " quota (contract §5, countable; see final report)\n";
                }
            }
        }
        else
        {
            NS_LOG_WARN("[SqrSim] sem_timedwait timeout — no action from agent");
        }
    }

    // Agendar próximo callback
    Simulator::Schedule(MilliSeconds(g_indicationPeriodMs), &KpmAndControlCallback);
}

// ---------------------------------------------------------------------------
// Relatório final
// ---------------------------------------------------------------------------

static void
WriteFinalReport(const Ptr<Ipv4FlowClassifier>& classifier)
{
    if (!g_monitorPtr)
    {
        return;
    }
    g_monitorPtr->CheckForLostPackets();
    FlowMonitor::FlowStatsContainer stats = g_monitorPtr->GetFlowStats();

    struct UeTotals
    {
        uint64_t txBytes = 0;
        uint64_t rxBytes = 0;
        uint32_t txPackets = 0;
        uint32_t rxPackets = 0;
        uint32_t lostPackets = 0;
    };
    std::map<uint16_t, UeTotals> totals;
    for (const auto& kv : stats)
    {
        const FlowMonitor::FlowStats& st = kv.second;
        Ipv4FlowClassifier::FiveTuple tuple = classifier->FindFlow(kv.first);
        if (tuple.protocol != 17)
            continue;
        auto itPort = g_portToUeId.find(tuple.destinationPort);
        if (itPort == g_portToUeId.end())
            continue;
        UeTotals& t = totals[itPort->second];
        t.txBytes += st.txBytes;
        t.rxBytes += st.rxBytes;
        t.txPackets += st.txPackets;
        t.rxPackets += st.rxPackets;
        t.lostPackets += st.lostPackets;
    }

    std::filesystem::create_directories(g_outputDir);
    const std::string csvPath = g_outputDir + "/sqr_summary.csv";
    std::ofstream csv(csvPath, std::ios::out | std::ios::trunc);
    csv << "ueId,slice,slaMbps,offeredMbps,arrivalSec,departureSec,attached,"
           "txPackets,rxPackets,lostPackets,meanThrMbps,slaMetMeanThr\n";

    std::cout << "\n========================================\n"
              << "  SQR RESULTS (per UE)\n"
              << "========================================\n"
              << std::fixed;

    for (const auto& kv : g_ues)
    {
        const SqrUe& ue = kv.second;
        const uint32_t sliceIdx = static_cast<uint32_t>(ue.slice);
        const double slaMbps = g_slaMbps[sliceIdx];
        const double activeSec = std::max(
            (ue.active ? g_simTimeSec : ue.departureSec) - (ue.arrivalSec + g_attachGraceSec),
            0.1);
        const UeTotals& t = totals[ue.ueId];
        // R04: taxa ofertada efetiva do UE (greedy full-buffer --ueOfferedMbps
        // ou fallback legado mult × SLA).
        const double offeredMbps = (ue.offeredMbps > 0.0)
                                       ? ue.offeredMbps
                                       : g_ueOfferedMult * slaMbps;
        const double meanThr = t.rxBytes * 8.0 / activeSec / 1e6;

        std::cout << std::setprecision(3)
                  << "UE " << ue.ueId << " [" << SliceName(ue.slice) << "]"
                  << " arrival=" << ue.arrivalSec << "s"
                  << " departure=" << (ue.active ? -1.0 : ue.departureSec) << "s"
                  << " sla=" << slaMbps << " Mbps"
                  << " meanThr=" << meanThr << " Mbps"
                  << " pdr=" << (t.txPackets > 0
                                     ? static_cast<double>(t.rxPackets) / t.txPackets
                                     : 0.0)
                  << "\n";

        csv << ue.ueId << "," << sliceIdx << "," << slaMbps << "," << offeredMbps << ","
            << ue.arrivalSec << "," << (ue.active ? -1.0 : ue.departureSec) << ","
            << (ue.attached ? 1 : 0) << ","
            << t.txPackets << "," << t.rxPackets << "," << t.lostPackets << ","
            << std::setprecision(4) << meanThr << ","
            << (meanThr >= slaMbps ? 1 : 0) << "\n";
    }

    std::cout << "========================================\n"
              << "Arrivals=" << g_arrivalCount
              << " Departures=" << g_departureCount
              << " StillActive=" << (g_arrivalCount - g_departureCount)
              << "\nSummary CSV: " << csvPath << "\n"
              // Degradações do contrato §5 (contáveis; 0 = nenhuma ocorrência):
              << "[SqrSim] IPC_DEGRADATION action_file_missing_keep_prev_quota="
              << g_actionFileMissingCount
              << " action_slice_line_missing_default_33pct=" << g_sliceLineMissingCount
              << " (contract §5)\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int
main(int argc, char* argv[])
{
    /*
     * Fluxo principal (cenário SQR):
     * 1. Ler CLI (Tabela I parametrizada).
     * 2. Criar topologia EPC + 1 gNB + remoteHost (UEs são DINÂMICOS).
     * 3. Configurar canal NR UMa, antenas, numerologia mu=0, TDD e RLC UM.
     * 4. Agendar Poisson de chegadas, verificador de attach e callback IPC/KPM.
     * 5. Rodar e emitir relatório final por UE.
     */

    std::string outputDirBase = ".";
    uint32_t seed = 1;
    uint32_t run = 1;
    std::string simId = "";
    double centralFrequency = 2.0e9;    //!< [ARTIGO Tabela I] 2 GHz
    double bandwidth = 10.0e6;          //!< [ASSUMIDO] 10 MHz → C=52 PRBs (mu=0)
    uint16_t numerology = 0;            //!< [ASSUMIDO] mu=0, slot de 1 ms
    double txPowerDbm = 30.0;           //!< [ARTIGO Tabela I] 30 dBm
    double gnbGainDbi = 15.0;           //!< [ARTIGO Tabela I] antena 15 dBi
    double ueNoiseFigureDb = 9.0;       //!< [ARTIGO Tabela I] noise figure 9 dB
    double gnbNoiseFigureDb = 9.0;      //!< mesmo valor (DL é o receptor relevante)
    double gnbHeightM = 25.0;           //!< [ASSUMIDO] altura macro urbana típica
    double gnbTiltDeg = 0.0;
    std::string slaStr = "0.5,1.0,1.5"; //!< [ARTIGO Tabela I] SLA por UE
    std::string weightsStr = "0.333333,0.333333,0.333334"; //!< inicial equal-split
    std::string intraAlgo = "PF";
    std::string tddPattern = "D|D|8D|4GB|4U|U|U"; //!< herdado do rslaq (DL-heavy)
    std::string rlcMode = "um";
    bool logAllMacSlots = false;
    uint32_t macLoggingPeriodMs = 97;

    CommandLine cmd(__FILE__);
    cmd.AddValue("simTime", "Total sim time (s)", g_simTimeSec);
    cmd.AddValue("seed", "RNG seed", seed);
    cmd.AddValue("run", "RNG run", run);
    cmd.AddValue("outputDir", "Output directory base", outputDirBase);
    cmd.AddValue("periodMs", "Decision/KPM period in ms (artigo: 100)", g_indicationPeriodMs);
    cmd.AddValue("centralFrequency", "NR carrier center frequency (Hz) [ARTIGO: 2e9]", centralFrequency);
    cmd.AddValue("bandwidth", "NR channel bandwidth (Hz): 10e6→52 PRBs, 20e6→106 PRBs (mu=0)", bandwidth);
    cmd.AddValue("numerology", "NR numerology mu (0 → slot 1 ms)", numerology);
    cmd.AddValue("txPower", "gNB TX power (dBm) [ARTIGO: 30]", txPowerDbm);
    cmd.AddValue("gnbGainDbi", "gNB antenna max gain (dBi) [ARTIGO: 15]", gnbGainDbi);
    cmd.AddValue("ueNoiseFigure", "UE noise figure (dB) [ARTIGO: 9]", ueNoiseFigureDb);
    cmd.AddValue("gnbNoiseFigure", "gNB noise figure (dB)", gnbNoiseFigureDb);
    cmd.AddValue("cellRadius", "Cell radius for UE placement/walk (m) [ARTIGO: 2000]", g_cellRadiusM);
    cmd.AddValue("gnbHeight", "gNB antenna height (m)", gnbHeightM);
    cmd.AddValue("ueHeight", "UE antenna height (m)", g_ueHeightM);
    cmd.AddValue("gnbTiltDeg", "gNB panel downtilt (deg)", gnbTiltDeg);
    cmd.AddValue("lambdaUesPerMin", "Poisson arrival rate (UEs/min) [ARTIGO: 2]", g_lambdaUesPerMin);
    cmd.AddValue("meanSessionSec", "Mean session duration (s, exponential) [ARTIGO: 60]", g_meanSessionSec);
    cmd.AddValue("slaMbps", "Per-slice per-UE SLA in Mbps [ARTIGO: 0.5,1.0,1.5]", slaStr);
    cmd.AddValue("ueOfferedMbps", "Per-UE offered DL UDP rate in Mbps — greedy full-buffer "
                                  "(default 100 > 2x cell peak; 0 = legacy ueOfferedMult x SLA)",
                 g_ueOfferedMbps);
    cmd.AddValue("ueOfferedMult", "Legacy per-UE offered load = mult x SLA (used only if ueOfferedMbps=0)",
                 g_ueOfferedMult);
    cmd.AddValue("pktSize", "UDP packet size (bytes)", g_pktSize);
    cmd.AddValue("tddPattern", "TDD slot pattern string", tddPattern);
    cmd.AddValue("rlcMode", "RLC mode for data radio bearers: am or um", rlcMode);
    cmd.AddValue("intraAlgo", "Intra-slice scheduler: RR|PF|BCQI", intraAlgo);
    cmd.AddValue("weights", "Initial slice weights (comma list summing 1)", weightsStr);
    cmd.AddValue("LogAllMacSlots", "Log every MAC scheduling call (large files)", logAllMacSlots);
    cmd.AddValue("MacLoggingPeriodMs", "MAC logging period in ms", macLoggingPeriodMs);
    cmd.AddValue("simId", "Simulation UUID for IPC semaphores (empty = network-only)", simId);
    cmd.Parse(argc, argv);

    // --- Validações ---
    if (g_simTimeSec <= 1.0)
    {
        NS_FATAL_ERROR("simTime must be > 1 s");
    }
    if (centralFrequency <= 0.0 || bandwidth <= 0.0)
    {
        NS_FATAL_ERROR("centralFrequency and bandwidth must be positive");
    }
    if (g_cellRadiusM <= 0.0 || gnbHeightM <= 0.0 || g_ueHeightM <= 0.0)
    {
        NS_FATAL_ERROR("cellRadius/gnbHeight/ueHeight must be positive");
    }
    if (g_lambdaUesPerMin <= 0.0 || g_meanSessionSec <= 0.0)
    {
        NS_FATAL_ERROR("lambdaUesPerMin and meanSessionSec must be positive");
    }
    if (g_indicationPeriodMs == 0)
    {
        NS_FATAL_ERROR("periodMs must be positive");
    }
    if (g_ueOfferedMbps < 0.0)
    {
        NS_FATAL_ERROR("ueOfferedMbps must be non-negative (0 = legacy mult mode)");
    }
    if (g_ueOfferedMbps == 0.0 && g_ueOfferedMult <= 0.0)
    {
        NS_FATAL_ERROR("offered load must be positive: set ueOfferedMbps > 0 or ueOfferedMult > 0");
    }
    if (rlcMode != "am" && rlcMode != "um")
    {
        NS_FATAL_ERROR("rlcMode must be either 'am' or 'um'");
    }
    rlcMode = ToLower(rlcMode);

    g_slaMbps = ParseSla(slaStr);
    g_outputDir = (std::filesystem::path(outputDirBase) / "results_sqr_network_only" /
                   ("seed=" + std::to_string(seed) + "_run=" + std::to_string(run)))
                      .string();
    std::filesystem::create_directories(g_outputDir);

    std::vector<double> sliceWeights = ParseWeights(weightsStr);
    if (sliceWeights.size() != NUM_SLICES)
    {
        NS_FATAL_ERROR("weights must have exactly " << NUM_SLICES << " elements");
    }
    double wsum = 0.0;
    for (double w : sliceWeights)
    {
        if (w < 0.0)
        {
            NS_FATAL_ERROR("weights must be non-negative");
        }
        wsum += w;
    }
    if (std::abs(wsum - 1.0) > 1e-3)
    {
        NS_FATAL_ERROR("weights must sum to 1.0 (got " << wsum << ")");
    }

    RslaqMacScheduler::IntraSliceAlgorithm parsedIntraAlgo = ParseIntraAlgo(intraAlgo);
    std::vector<RslaqMacScheduler::IntraSliceAlgorithm> sliceAlgos(NUM_SLICES, parsedIntraAlgo);

    bool tddApplied = false;
    g_appliedTddPattern = NormalizeTddPatternForNr(tddPattern, &tddApplied);
    if (!tddApplied)
    {
        std::cout << "WARNING: TDD pattern not applied. Running simplified DL-only simulation.\n";
    }

    const double scsKhz = 15.0 * std::pow(2.0, numerology);
    // Duração do slot NR: 1 ms / 2^mu (mu=0 → 1 ms; mu=1 → 0,5 ms). Usada para
    // alinhar as chegadas de UE dinâmicos à fase do padrão TDD do gNB
    // (ver ScheduleNextArrival).
    g_slotPeriodSec = 0.001 / std::pow(2.0, numerology);
    {
        // Nº de slots do padrão TDD aplicado (tokens não vazios separados por '|')
        uint32_t count = 0;
        std::stringstream ss(g_appliedTddPattern);
        std::string token;
        while (std::getline(ss, token, '|'))
        {
            if (!token.empty())
            {
                count++;
            }
        }
        if (count > 0)
        {
            g_patternSlots = count;
        }
    }

    std::cout << "========================================\n"
              << "  SQR Simulation (cenario fiel ao artigo, IPC opcional)\n"
              << "========================================\n"
              << "Frequency    : " << centralFrequency / 1e9 << " GHz [ARTIGO Tabela I]\n"
              << "Bandwidth    : " << bandwidth / 1e6 << " MHz (C="
              << (bandwidth / 1e6 >= 15.0 ? 106 : 52) << " PRBs aprox., mu=" << numerology << ")\n"
              << "Numerology   : " << numerology << " (SCS " << scsKhz << " kHz, slot 1 ms)\n"
              << "TxPower      : " << txPowerDbm << " dBm [ARTIGO]\n"
              << "Antenna gain : " << gnbGainDbi << " dBi [ARTIGO]\n"
              << "UE noise fig : " << ueNoiseFigureDb << " dB [ARTIGO]\n"
              << "Pathloss     : 3GPP UMa (urban macro) [ARTIGO]\n"
              << "Cell radius  : " << g_cellRadiusM << " m [ARTIGO]\n"
              << "UE churn     : Poisson " << g_lambdaUesPerMin << " UEs/min, "
              << "session Exp(" << g_meanSessionSec << " s) [ARTIGO]\n"
              << "Mobility     : RandomWalk2d ~1 m/s [ASSUMIDO]\n"
              << "Slices       : 3x eMBB, SLA/UE = " << slaStr << " Mbps [ARTIGO]\n"
              << "Offered/UE   : "
              << (g_ueOfferedMbps > 0.0
                      ? (std::ostringstream{} << g_ueOfferedMbps << " Mbps UDP greedy full-buffer"
                                              << " [ASSUMIDO R04]").str()
                      : (std::ostringstream{} << g_ueOfferedMult << "x SLA (legacy) [ASSUMIDO]").str())
              << "\n"
              << "Scheduler    : SqrMacScheduler (cota absoluta de RBGs, R27)\n"
              << "Decision pd  : " << g_indicationPeriodMs << " ms [ARTIGO: 100 TTIs]\n"
              << "RLC mode     : " << ToUpper(rlcMode) << "\n"
              << "TDD pattern  : " << g_appliedTddPattern << (tddApplied ? "" : " (fallback DL-only)")
              << "\n"
              << "IntraAlgo    : " << IntraAlgoName(parsedIntraAlgo) << "\n"
              << "Seed/Run     : " << seed << "/" << run << "\n"
              << "IPC enabled  : " << (simId.empty() ? "NO (network-only)" : "YES") << "\n"
              << "OutputDir    : " << g_outputDir << "\n\n";

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(run);

    // --- IPC: semáforos (verbatim de rslaq-sim.cc) ---
    if (!simId.empty())
    {
        g_simUuid = simId;
        g_ipcEnabled = true;

        std::string semMetricsName = "/sem_metrics_" + simId;
        std::string semControlName = "/sem_control_" + simId;

        g_semMetricsReady = sem_open(semMetricsName.c_str(), O_CREAT, 0660, 0);
        g_semControl = sem_open(semControlName.c_str(), O_CREAT, 0660, 0);

        if (g_semMetricsReady == SEM_FAILED || g_semControl == SEM_FAILED)
        {
            std::cerr << "WARNING: Failed to create POSIX semaphores. Running standalone.\n";
            g_ipcEnabled = false;
            if (g_semMetricsReady != SEM_FAILED)
                sem_close(g_semMetricsReady);
            if (g_semControl != SEM_FAILED)
                sem_close(g_semControl);
            g_semMetricsReady = nullptr;
            g_semControl = nullptr;
        }
        else
        {
            std::cout << "[SqrSim] IPC semaphores created: "
                      << semMetricsName << " and " << semControlName << "\n";
        }
    }

    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(10485760));
    Config::SetDefault("ns3::NrGnbRrc::QosFlowToRlcMapping",
                       EnumValue(rlcMode == "am" ? NrGnbRrc::RLC_AM_ALWAYS
                                                  : NrGnbRrc::RLC_UM_ALWAYS));
    Config::SetDefault("ns3::ThreeGppChannelModel::UpdatePeriod", TimeValue(MilliSeconds(0)));

    // --- Nós fixos: 1 gNB + remoteHost (UEs criados em runtime) ---
    NodeContainer gNbNodes;
    gNbNodes.Create(1);
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    g_remoteHostNode = remoteHostContainer.Get(0);

    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    {
        Ptr<ListPositionAllocator> pos = CreateObject<ListPositionAllocator>();
        pos->Add(Vector(0.0, 0.0, gnbHeightM));
        mobility.SetPositionAllocator(pos);
        mobility.Install(gNbNodes);
    }

    // --- Helpers NR/EPC ---
    g_epcHelper = CreateObject<NrPointToPointEpcHelper>();
    g_nrHelper = CreateObject<NrHelper>();
    g_nrHelper->SetEpcHelper(g_epcHelper);
    // SQR-R27: scheduler de COTA ABSOLUTA (subclasse de RslaqMacScheduler).
    g_nrHelper->SetSchedulerTypeId(SqrMacScheduler::GetTypeId());

    // UE: antena isotrópica 1x1 (sem beamforming no lado UE)
    g_nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(1));
    g_nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(1));
    g_nrHelper->SetUeAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<IsotropicAntennaModel>()));

    // gNB: painel de elemento único com ganho máximo 15 dBi [ARTIGO].
    // [ASSUMIDO] cobertura OMNIDIRECIONAL no plano horizontal (beamwidth 360°)
    // com feixe vertical largo 65° e tilt 0: a Tabela I publica apenas
    // "antena 15 dBi", sem padrão de radiação; um painel setorial 65° deprimiria
    // em até 30 dB os UEs fora do setor e inviabilizaria o raio de célula de
    // 2 km do próprio artigo. Sobre o disco de 2 km (UE a 1,5 m, gNB a 25 m,
    // elevação ~0,7°) o ganho efetivo fica ≈ 15 dBi em qualquer direção.
    Ptr<CosineAntennaModel> gnbPanelElement = CreateObject<CosineAntennaModel>();
    gnbPanelElement->SetAttribute("HorizontalBeamwidth", DoubleValue(360.0));
    gnbPanelElement->SetAttribute("VerticalBeamwidth", DoubleValue(65.0));
    gnbPanelElement->SetAttribute("MaxGain", DoubleValue(gnbGainDbi));
    g_nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(1));
    g_nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(1));
    g_nrHelper->SetGnbAntennaAttribute("IsDualPolarized", BooleanValue(false));
    g_nrHelper->SetGnbAntennaAttribute("NumHorizontalPorts", UintegerValue(1));
    g_nrHelper->SetGnbAntennaAttribute("NumVerticalPorts", UintegerValue(1));
    g_nrHelper->SetGnbAntennaAttribute("PolSlantAngle", DoubleValue(0.0));
    g_nrHelper->SetGnbAntennaAttribute("DowntiltAngle",
                                       DoubleValue(gnbTiltDeg * std::acos(-1.0) / 180.0));
    g_nrHelper->SetGnbAntennaAttribute("BearingAngle", DoubleValue(0.0));
    g_nrHelper->SetGnbAntennaAttribute("AntennaElement", PointerValue(gnbPanelElement));

    // Noise figure 9 dB [ARTIGO Tabela I] — aplicado a TODOS os UEs instalados
    // (inclusive os dinâmicos, pois os atributos ficam no helper).
    g_nrHelper->SetUePhyAttribute("NoiseFigure", DoubleValue(ueNoiseFigureDb));
    g_nrHelper->SetGnbPhyAttribute("NoiseFigure", DoubleValue(gnbNoiseFigureDb));

    // --- Spectrum: 1 CC / 1 BWP ---
    // Pathloss urban macro 3GPP: cenário "UMa" do NrChannelHelper (TS 38.901 UMa
    // — LOS/NLOS + shadowing + fading small-scale 3GPP). Aproximação ao perfil
    // "Pedestrian A" (TS 36.104) comentada no cabeçalho deste arquivo.
    CcBwpCreator ccBwpCreator;
    CcBwpCreator::SimpleOperationBandConf bandConf(centralFrequency, bandwidth, 1);
    bandConf.m_numBwp = 1;
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);

    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories("UMa", "Default", "ThreeGpp");
    channelHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
    channelHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(true));
    channelHelper->AssignChannelsToBands({band});

    g_allBwps = CcBwpCreator::GetAllBwps({band});

    // --- Internet (remoteHost) ---
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    // --- gNB device ---
    g_gnbNetDev = g_nrHelper->InstallGnbDevice(gNbNodes, g_allBwps);
    g_nrHelper->GetGnbPhy(g_gnbNetDev.Get(0), 0)->SetAttribute("Numerology", UintegerValue(numerology));
    g_nrHelper->GetGnbPhy(g_gnbNetDev.Get(0), 0)->SetAttribute("TxPower", DoubleValue(txPowerDbm));
    g_nrHelper->GetGnbPhy(g_gnbNetDev.Get(0), 0)->SetPattern(g_appliedTddPattern);

    const int64_t randomStream = seed;
    g_nrHelper->AssignStreams(g_gnbNetDev, randomStream);

    // --- EPC backhaul (verbatim rslaq) ---
    // Link P2P de alta capacidade para que o gargalo relevante seja a RAN.
    Ptr<Node> pgw = g_epcHelper->GetPgwNode();
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gbps")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(2500));
    p2ph.SetChannelAttribute("Delay", TimeValue(Seconds(0.001)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHostContainer.Get(0));

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    ipv4h.Assign(internetDevices);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteStatic =
        ipv4RoutingHelper.GetStaticRouting(remoteHostContainer.Get(0)->GetObject<Ipv4>());
    remoteStatic->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    // --- FlowMonitor: monitor compartilhado + probe no remoteHost ---
    static std::unique_ptr<FlowMonitorHelper> flowmonHelperOwner;
    flowmonHelperOwner = std::make_unique<FlowMonitorHelper>();
    g_flowmonHelper = flowmonHelperOwner.get();
    g_monitorPtr = g_flowmonHelper->Install(remoteHostContainer);
    g_classifierPtr = DynamicCast<Ipv4FlowClassifier>(g_flowmonHelper->GetClassifier());

    // --- Scheduler SQR de cota absoluta (R27, em scratch/sqr) ---
    Ptr<NrMacScheduler> schedBase = NrHelper::GetScheduler(g_gnbNetDev.Get(0), 0);
    Ptr<SqrMacScheduler> scheduler = DynamicCast<SqrMacScheduler>(schedBase);
    NS_ASSERT_MSG(scheduler != nullptr, "Failed to cast to SqrMacScheduler");
    g_schedulerPtr = scheduler;
    scheduler->SetScenarioName("sqr");
    scheduler->SetOutputDir(g_outputDir);
    scheduler->SetAttribute("LogAllMacSlots", BooleanValue(logAllMacSlots));
    scheduler->SetAttribute("MacLoggingPeriodMs", UintegerValue(macLoggingPeriodMs));
    scheduler->SetAttribute("EnableDetailedMacLogging", BooleanValue(true));

    // Mapeamento inicial (vazio) + configuração inicial do SQR/estático.
    // O OVERRIDE de SetSliceConfiguration converte as frações em COTA RBG
    // (piso + maiores restos, Σ ≤ totalRbg; frações somando 1 ⇒ partição
    // completa inicial, sem alteração de comportamento no baseline).
    scheduler->SetSliceUeMapping(NUM_SLICES, g_sliceRntis);
    scheduler->SetSliceWeightPolicy(RslaqMacScheduler::SliceWeightPolicy::STATIC);
    scheduler->SetSliceConfiguration(sliceWeights, sliceAlgos);

    std::cout << "[SqrSim] Initial absolute quota (RBG): [";
    {
        const std::vector<uint32_t> q0 = scheduler->GetSliceQuotaRbg();
        for (uint32_t s = 0; s < q0.size(); ++s)
        {
            std::cout << (s ? ", " : "") << q0[s];
        }
    }
    std::cout << "] totalRbg=" << scheduler->GetTotalRbgCount()
              << " totalPrb=" << scheduler->GetTotalPrbCount()
              << (scheduler->GetTotalPrbCount() == 0
                      ? " (deferred — quota materialized on first DL slot)"
                      : "")
              << "\n";

    // --- Agendamentos ---
    // 1) verificador de attach (acorda quando houver chegada pendente;
    //    UeArrival re-arma o loop via g_attachCheckScheduled)
    g_attachCheckScheduled = true;
    Simulator::Schedule(MilliSeconds(5), &AttachCheck);
    // 2) callback KPM/controle (a cada periodMs; exporta KPMs mesmo sem agente)
    Simulator::Schedule(MilliSeconds(g_indicationPeriodMs), &KpmAndControlCallback);
    // 3) processo de Poisson de chegadas (a partir de t=0)
    Simulator::Schedule(Seconds(0.0), &ScheduleNextArrival);

    std::cout << "[SqrSim] Callbacks scheduled (attach check, KPM period="
              << g_indicationPeriodMs << " ms, Poisson lambda="
              << g_lambdaUesPerMin << " UEs/min)\n";

    // --- Run ---
    Simulator::Stop(Seconds(g_simTimeSec));
    Simulator::Run();

    // --- Relatório final ---
    WriteFinalReport(g_classifierPtr);

    // --- Cleanup IPC (verbatim rslaq) ---
    if (g_ipcEnabled)
    {
        if (g_semMetricsReady)
        {
            sem_close(g_semMetricsReady);
            std::string name = "/sem_metrics_" + g_simUuid;
            sem_unlink(name.c_str());
        }
        if (g_semControl)
        {
            sem_close(g_semControl);
            std::string name = "/sem_control_" + g_simUuid;
            sem_unlink(name.c_str());
        }
        std::cout << "[SqrSim] IPC semaphores cleaned up.\n";
    }

    Simulator::Destroy();
    return 0;
}
