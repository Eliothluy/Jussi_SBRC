/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * SqrE2Adapter / SqrE2LoopbackTerm — implementação (Fase 5.3).
 * Ver sqr-e2-adapter.h para o desenho (API, modos FULL/DEGRADED, mensagens).
 */

#include "sqr-e2-adapter.h"

#include "ns3/abort.h"
#include "ns3/log.h"
#include "ns3/nr-gnb-net-device.h"
#include "ns3/pointer.h"
#include "ns3/simulator.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("SqrE2Adapter");

// ---------------------------------------------------------------------------
// Constantes do dummy equal-split (equivalente ao sqr_dummy_controller.py —
// fases Σa = 0 / 0,5·C / C). DIVERGÊNCIA ÚNICA vs o dummy IPC, imposta pela
// API: o tipo estruturado do RIC Control (ns3::SlicePRBQuota do contrib/nori,
// igual nos modos FULL/DEGRADED) carrega razões de PRB INTEIRAS (long,
// ASN.1 INTEGER do E2SM-RC) — frações como 16.6667 % não são expressíveis.
// O loopback reparte igual com MAIORES RESTOS inteiros, preservando a soma
// da fase EXATAMENTE (o que os critérios R27 testam é Σa):
//   50  → (17, 17, 16)   Σ = 50
//   100 → (34, 33, 33)   Σ = 100
// (o dummy IPC usa frações com soma exata 16.6667/16.6667/16.6666 etc.; as
// cotas RBG resultantes após ComputeRbgQuotas são idênticas — 9/9/9 e
// 18/18/17 em C=53.)
// ---------------------------------------------------------------------------
namespace
{
constexpr uint32_t SQR_E2_NUM_SLICES = 3;

std::array<double, SQR_E2_NUM_SLICES>
PhasePcts(long phaseSum)
{
    switch (phaseSum)
    {
    case 0:
        return {0.0, 0.0, 0.0};
    case 50:
        return {17.0, 17.0, 16.0}; // Σ = 50 (maiores restos de 50/3)
    case 100:
        return {34.0, 33.0, 33.0}; // Σ = 100 (maiores restos de 100/3)
    default:
        return {0.0, 0.0, 0.0};
    }
}

/** Codificação PLMN idêntica a E2TermHelper::InstallE2Term (branch 5 chars). */
std::string
EncodePlmn(const std::string& plmnId)
{
    if (plmnId.length() == 5)
    {
        const uint8_t mcc1 = plmnId[0] - '0';
        const uint8_t mcc2 = plmnId[1] - '0';
        const uint8_t mcc3 = plmnId[2] - '0';
        const uint8_t mnc1 = plmnId[3] - '0';
        const uint8_t mnc2 = plmnId[4] - '0';
        std::string out;
        out.resize(3);
        out[0] = static_cast<char>((mcc2 << 4) | mcc1);
        out[1] = static_cast<char>((0xF << 4) | mcc3);
        out[2] = static_cast<char>((mnc2 << 4) | mnc1);
        return out;
    }
    if (plmnId.length() == 3)
    {
        return plmnId; // caso especial de teste do helper
    }
    NS_FATAL_ERROR("SqrE2Adapter: invalid PLMN length " << plmnId.length());
    return "";
}

// -------------------------------------------------------------------------
// Fase 5.4: helpers da ponte xApp (transporte local UDP + JSON mínimo).
// O JSON é escrito/parseado à mão de propósito: schema FIXO e controlado nas
// duas pontas (este arquivo e examples/xapp_sqr.py), sem dependência externa
// no ns-3. Doubles com %.17g = round-trip exato IEEE754 (o xApp reformata os
// KPIs na precisão do fio do contrato §4 — %.2f/%.4f do emitRow — antes de
// parsear, garantindo paridade bit a bit com o caminho IPC).
// -------------------------------------------------------------------------

/// Wall-clock CLOCK_MONOTONIC em ms (latência da ponte; nunca tempo simulado).
double
WallMonotonicMs()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) * 1e3 + static_cast<double>(ts.tv_nsec) / 1e6;
}

/// Double JSON com round-trip exato (%.17g; inf/nan nunca ocorrem nos KPIs).
std::string
JsonNum(double v)
{
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return std::string(buf);
}

/// Posição do VALOR de `"key":` no JSON, tolerante a espaços após ':'
/// (std::string::npos se ausente — json.dumps pode emitir `"key": value`).
size_t
JsonValuePos(const std::string& s, const std::string& key)
{
    const size_t k = s.find("\"" + key + "\"");
    if (k == std::string::npos)
    {
        return std::string::npos;
    }
    size_t p = s.find(':', k + key.size() + 2);
    if (p == std::string::npos)
    {
        return std::string::npos;
    }
    ++p;
    while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p])))
    {
        ++p;
    }
    return p;
}

/// long de `"key": <int>` (defaultValue se a chave não existir).
long
JsonGetLong(const std::string& s, const std::string& key, long defaultValue)
{
    const size_t p = JsonValuePos(s, key);
    if (p == std::string::npos)
    {
        return defaultValue;
    }
    return std::strtol(s.c_str() + p, nullptr, 10);
}

/// double de `"key": <num>` (defaultValue se a chave não existir).
double
JsonGetDouble(const std::string& s, const std::string& key, double defaultValue)
{
    const size_t p = JsonValuePos(s, key);
    if (p == std::string::npos)
    {
        return defaultValue;
    }
    return std::strtod(s.c_str() + p, nullptr);
}
} // anonymous namespace

// ---------------------------------------------------------------------------
// SqrE2LoopbackTerm
// ---------------------------------------------------------------------------

NS_OBJECT_ENSURE_REGISTERED(SqrE2LoopbackTerm);

TypeId
SqrE2LoopbackTerm::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::SqrE2LoopbackTerm")
            .SetParent<Object>()
            .SetGroupName("Sqr")
            .AddConstructor<SqrE2LoopbackTerm>();
    return tid;
}

SqrE2LoopbackTerm::SqrE2LoopbackTerm() = default;

SqrE2LoopbackTerm::~SqrE2LoopbackTerm()
{
    if (m_log.is_open())
    {
        m_log.flush();
        m_log.close();
    }
    if (m_sockFd >= 0)
    {
        ::close(m_sockFd);
        m_sockFd = -1;
    }
    delete static_cast<struct sockaddr_in*>(m_xappAddr);
    m_xappAddr = nullptr;
}

bool
SqrE2LoopbackTerm::ConfigureXappUdp(const std::string& ricAddress,
                                    uint16_t ricPort,
                                    int timeoutMs,
                                    int retries,
                                    const std::string& logPath)
{
    m_ricAddress = ricAddress;
    m_ricPort = ricPort;
    m_xappTimeoutMs = std::max(timeoutMs, 1);
    m_xappRetries = std::max(retries, 0);
    m_mode = EndpointMode::XAPP_UDP;

    auto* addr = new struct sockaddr_in();
    std::memset(addr, 0, sizeof(*addr));
    addr->sin_family = AF_INET;
    addr->sin_port = htons(ricPort);
    if (::inet_pton(AF_INET, ricAddress.c_str(), &addr->sin_addr) != 1)
    {
        // Endereço não-IP (ex.: hostname) — resolução best-effort p/ localhost.
        const struct hostent* he = ::gethostbyname(ricAddress.c_str());
        NS_ABORT_MSG_IF(!he, "SqrE2XappBridge: cannot resolve xApp address " << ricAddress);
        std::memcpy(&addr->sin_addr, he->h_addr_list[0], sizeof(struct in_addr));
    }
    m_xappAddr = addr;

    m_sockFd = ::socket(AF_INET, SOCK_DGRAM, 0);
    NS_ABORT_MSG_IF(m_sockFd < 0, "SqrE2XappBridge: socket() failed: " << std::strerror(errno));

    m_log.open(logPath, std::ios::out | std::ios::trunc);
    if (m_log.is_open())
    {
        m_log << "seq,timestamp_ms,kpm_ues,rtt_wall_ms,applied,timeout,"
                 "reply_decision_ms\n";
    }
    else
    {
        NS_LOG_WARN("[SqrE2XappBridge] cannot open bridge log " << logPath);
    }
    return m_sockFd >= 0;
}

void
SqrE2LoopbackTerm::Configure(const std::string& ricAddress,
                             uint16_t ricPort,
                             const std::vector<long>& phases,
                             uint32_t windowsPerPhase,
                             const std::string& logPath)
{
    m_ricAddress = ricAddress;
    m_ricPort = ricPort;
    m_phases = phases;
    m_windowsPerPhase = std::max<uint32_t>(windowsPerPhase, 1);

    m_log.open(logPath, std::ios::out | std::ios::trunc);
    if (m_log.is_open())
    {
        m_log << "timestamp_ms,phase,pct_0,pct_1,pct_2\n";
    }
    else
    {
        NS_LOG_WARN("[SqrE2Loopback] cannot open loopback log " << logPath);
    }
}

void
SqrE2LoopbackTerm::SetControlDeliverCallback(
    std::function<void(const SqrE2RicControlMessage&)> cb)
{
    m_deliver = std::move(cb);
}

void
SqrE2LoopbackTerm::Start()
{
    // Equivalente ao E2Termination::Start do modo FULL (E2-SETUP em direção
    // ao RIC). Em DEGRADED o transporte real não existe: o log registra a
    // tentativa e o circuito fecha em-processo (loopback) ou via ponte UDP
    // local para o xApp (Fase 5.4, --e2Endpoint=xapp).
    if (m_mode == EndpointMode::XAPP_UDP)
    {
        std::cout << "[SqrE2XappBridge] E2-SETUP (simulated, UDP bridge): gNB"
                  << " E2term -> xApp SQR " << m_ricAddress << ":" << m_ricPort
                  << " (DEGRADED local transport: UDP/localhost JSON;"
                  << " budget " << m_xappTimeoutMs << " ms/tentativa, retries "
                  << m_xappRetries << "; FULL mode replaces this bridge with"
                  << " e2sim/SCTP — xApp logic unchanged)\n";
        return;
    }
    std::cout << "[SqrE2Loopback] E2-SETUP (simulated, in-process loopback): gNB"
              << " E2term -> RIC " << m_ricAddress << ":" << m_ricPort
              << " (DEGRADED: no e2sim/SCTP — contrib/nori stub; FULL mode"
              << " replaces this loopback with the real transport, same"
              << " structured messages)\n";
    std::cout << "[SqrE2Loopback] dummy equal-split controller ready:"
              << " phases=";
    for (size_t i = 0; i < m_phases.size(); ++i)
    {
        std::cout << (i ? "," : "") << m_phases[i];
    }
    std::cout << "% windowsPerPhase=" << m_windowsPerPhase << "\n";
}

// ---------------------------------------------------------------------------
// Fase 5.4: ramo xApp do Start + ponte UDP (implementação em HandleIndication-
// XappUdp). O E2-SETUP simulado é o mesmo evento do modo loopback; só o
// endpoint impresso muda (xApp UDP local em vez de loopback em-processo).
// ---------------------------------------------------------------------------

void
SqrE2LoopbackTerm::OnE2smKpmIndication(const SqrE2KpmIndicationMessage& ind)
{
    // Dispatcher do endpoint (Fase 5.4): mesmo ponto de entrada da Fase 5.3;
    // só o destino da indication muda (dummy em-processo vs ponte xApp UDP).
    if (m_mode == EndpointMode::XAPP_UDP)
    {
        HandleIndicationXappUdp(ind);
        return;
    }
    HandleIndicationLoopback(ind);
}

void
SqrE2LoopbackTerm::HandleIndicationLoopback(const SqrE2KpmIndicationMessage& ind)
{
    // Fase corrente por número de janelas respondidas — MESMA regra do
    // sqr_dummy_controller.py (phase_idx = min(n // wpp, len(phases)-1)).
    const size_t phaseIdx = std::min(m_indicationsReceived / m_windowsPerPhase,
                                     m_phases.size() - 1);
    const long phaseSum = m_phases[phaseIdx];
    const std::array<double, SQR_E2_NUM_SLICES> pcts = PhasePcts(phaseSum);

    // RIC Control estruturado (RAN slicing): dedicate = min = max = pct,
    // SST = índice da slice (mesma conversão de ApplySlicingControl do FULL,
    // E2-interface.cc:521-527: macQ.sliceId = q.sst).
    SqrE2RicControlMessage ctrl;
    ctrl.ranFunctionId = 300; // E2SM-RC (mesmo ID registrado no E2TermHelper)
    ctrl.requestType = SqrE2RicControlMessage::RAN_SLICING;
    ctrl.ricStyleType = 1;
    ctrl.ricControlActionId = 1;
    ctrl.ricControlDecision = 0; // accept
    ctrl.prbQuotas.reserve(SQR_E2_NUM_SLICES);
    for (uint32_t s = 0; s < SQR_E2_NUM_SLICES; ++s)
    {
        SlicePRBQuota q;
        q.sliceId = s; // SST = índice da slice lógica [ASSUMIDO Fase 5.3]
        q.dedicatePRBRatio = static_cast<long>(pcts[s]);
        q.minPRBRatio = static_cast<long>(pcts[s]);
        q.maxPRBRatio = static_cast<long>(pcts[s]);
        q.resourceType = "PRB_DL"; // cota restringe só DL (escopo do artigo)
        ctrl.prbQuotas.push_back(q);
    }

    ++m_indicationsReceived;

    // Entrega em-processo (DEGRADED). Em FULL a resposta chega pelo
    // e2sim: SCTP → E2Interface::ControlMessageReceivedCallback →
    // RicControlMessage(pdu) → mesmas SlicePRBQuota.
    if (m_deliver)
    {
        ++m_controlsSent;
        m_deliver(ctrl);
    }

    // Evidência: CSV com as MESMAS colunas do dummy_controller_log.csv do
    // caminho IPC + linha stdout equivalente ao "[dummy] ... action written".
    if (m_log.is_open())
    {
        m_log << ind.timestampMs << "," << phaseSum << "," << std::fixed
              << std::setprecision(4) << pcts[0] << "," << pcts[1] << ","
              << pcts[2] << "\n";
        m_log.flush();
    }
    std::cout << std::fixed << std::setprecision(1)
              << "[SqrE2Loopback] window=" << (m_indicationsReceived - 1)
              << " ts=" << ind.timestampMs << " phase=" << phaseSum << "%"
              << " kpm_ues=" << ind.ueRecords.size()
              << " -> RIC control sent (RAN slicing, Σdedicate=" << phaseSum
              << "%)\n";
}

void
SqrE2LoopbackTerm::HandleIndicationXappUdp(const SqrE2KpmIndicationMessage& ind)
{
    ++m_indicationsReceived;
    const uint64_t seq = ++m_xappSeq;

    // --- 1. Serializa a E2SM-KPM indication (JSON, schema do xapp_sqr.py) ---
    std::ostringstream js;
    js << "{\"type\":\"kpm_indication\",\"seq\":" << seq
       << ",\"timestamp_ms\":" << ind.timestampMs
       << ",\"period_ms\":" << ind.periodMs
       << ",\"cell_id\":" << ind.cellId
       << ",\"plmn_id\":\"" << ind.plmnId << "\""
       << ",\"gnb_id\":\"" << ind.gnbId << "\""
       << ",\"total_rbg\":" << ind.totalRbg
       << ",\"quota_rbg\":[" << ind.quotaRbg[0] << "," << ind.quotaRbg[1] << ","
       << ind.quotaRbg[2] << "]"
       << ",\"n_ues\":[" << ind.nUes[0] << "," << ind.nUes[1] << ","
       << ind.nUes[2] << "]"
       << ",\"ue_records\":[";
    for (size_t i = 0; i < ind.ueRecords.size(); ++i)
    {
        const SqrE2KpmUeRecord& r = ind.ueRecords[i];
        js << (i ? "," : "") << "[" << r.imsi << "," << r.rnti << ","
           << r.sliceId << "," << r.dTxBytes << "," << r.dRxBytes << ","
           << JsonNum(r.plrPct) << "," << JsonNum(r.resourceSharePct) << ","
           << r.dLostPackets << "," << JsonNum(r.throughputMbps) << ","
           << r.bufferBytes << "," << JsonNum(r.windowThroughputMbps) << ","
           << JsonNum(r.snrDb) << "," << r.queueBytes << "]";
    }
    const double tSend = WallMonotonicMs();
    js << "],\"sent_wall_ms\":" << JsonNum(tSend) << "}";
    const std::string payload = js.str();

    // --- 2. Envia e espera a resposta RIC Control (budget wall contrato §5) ---
    // Timeout = degradação contável do contrato §5 (mantém a cota anterior),
    // MESMA semântica do sem_timedwait(50 ms) do caminho IPC. Retries reenviam
    // o MESMO seq — o xApp deduplica por seq e reenvia a decisão cacheada.
    auto* dest = static_cast<struct sockaddr_in*>(m_xappAddr);
    bool got = false;
    std::string reply;
    double tReply = tSend;
    for (int attempt = 0; attempt <= m_xappRetries && !got; ++attempt)
    {
        struct timeval tv;
        tv.tv_sec = m_xappTimeoutMs / 1000;
        tv.tv_usec = (m_xappTimeoutMs % 1000) * 1000;
        ::setsockopt(m_sockFd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        const ssize_t n = ::sendto(m_sockFd, payload.data(),
                                   static_cast<ssize_t>(payload.size()), 0,
                                   reinterpret_cast<const struct sockaddr*>(dest),
                                   sizeof(*dest));
        if (n < 0)
        {
            NS_LOG_WARN("[SqrE2XappBridge] sendto failed: " << std::strerror(errno));
            break;
        }
        // Drena respostas até bater o seq corrente (descarta respostas tardias
        // de janelas anteriores que perderam o prazo).
        while (true)
        {
            char buf[65536];
            const ssize_t r = ::recvfrom(m_sockFd, buf, sizeof(buf) - 1, 0, nullptr, nullptr);
            if (r < 0)
            {
                break; // EAGAIN: estourou o budget desta tentativa
            }
            buf[r] = '\0';
            std::string cand(buf);
            if (JsonGetLong(cand, "seq", -1) == static_cast<long>(seq))
            {
                got = true;
                tReply = WallMonotonicMs();
                reply = std::move(cand);
                break;
            }
            NS_LOG_WARN("[SqrE2XappBridge] stale reply (seq="
                        << JsonGetLong(cand, "seq", -1) << " != " << seq << ") discarded");
        }
    }

    const double rttMs = tReply - tSend;
    if (!got)
    {
        ++m_xappTimeouts;
        NS_LOG_WARN("[SqrE2XappBridge] no RIC Control from xApp within "
                    << m_xappTimeoutMs << " ms (seq=" << seq
                    << ") — keeping previous quota (contract §5, countable)");
        if (m_log.is_open())
        {
            m_log << seq << "," << ind.timestampMs << "," << ind.ueRecords.size()
                  << "," << std::fixed << std::setprecision(3) << rttMs
                  << ",0,1,0\n";
            m_log.flush();
        }
        return;
    }

    // --- 3. Decodifica o RIC Control no MESMO mensaje estruturado da Fase 5.3
    //        (requestType RAN_SLICING; SlicePRBQuota com razões INTEIRAS de %,
    //        restrição do tipo do contrib/nori — ASN.1 INTEGER do E2SM-RC; o
    //        xApp escolhe inteiros que reproduzem as cotas RBG alvo) ---
    SqrE2RicControlMessage ctrl;
    ctrl.ranFunctionId = JsonGetLong(reply, "ran_function_id", 300);
    ctrl.requestType = JsonGetLong(reply, "request_type",
                                   SqrE2RicControlMessage::RAN_SLICING);
    ctrl.ricStyleType = JsonGetLong(reply, "ric_style_type", 1);
    ctrl.ricControlActionId = JsonGetLong(reply, "action_id", 1);
    ctrl.ricControlDecision = JsonGetLong(reply, "decision", 0);

    const size_t qa = JsonValuePos(reply, "prb_quotas");
    const size_t qaArr = (qa == std::string::npos) ? std::string::npos
                                                   : reply.find('[', qa);
    if (qaArr == std::string::npos)
    {
        ++m_xappTimeouts;
        NS_LOG_WARN("[SqrE2XappBridge] reply without prb_quotas (seq=" << seq
                    << ") — keeping previous quota (contract §5, countable)");
        if (m_log.is_open())
        {
            m_log << seq << "," << ind.timestampMs << "," << ind.ueRecords.size()
                  << "," << std::fixed << std::setprecision(3) << rttMs
                  << ",0,1,0\n";
            m_log.flush();
        }
        return;
    }
    const size_t qb = reply.find(']', qaArr);
    const std::string arr = reply.substr(qaArr, (qb == std::string::npos)
                                                    ? std::string::npos
                                                    : qb - qaArr + 1);
    size_t pos = 0;
    while (true)
    {
        const size_t ob = arr.find('{', pos);
        if (ob == std::string::npos)
        {
            break;
        }
        const size_t cb = arr.find('}', ob);
        if (cb == std::string::npos)
        {
            break;
        }
        const std::string obj = arr.substr(ob, cb - ob + 1);
        SlicePRBQuota q;
        q.sliceId = JsonGetLong(obj, "slice_id", 0); // SST = índice da slice
        q.dedicatePRBRatio = JsonGetLong(obj, "dedicate_prb_ratio", 0);
        q.minPRBRatio = JsonGetLong(obj, "min_prb_ratio", q.dedicatePRBRatio);
        q.maxPRBRatio = JsonGetLong(obj, "max_prb_ratio", q.dedicatePRBRatio);
        q.resourceType = "PRB_DL";
        ctrl.prbQuotas.push_back(q);
        pos = cb + 1;
    }

    ++m_controlsSent;
    if (m_deliver)
    {
        m_deliver(ctrl);
    }

    const double replyDecisionMs = JsonGetDouble(reply, "decision_ms", 0.0);
    if (m_log.is_open())
    {
        m_log << seq << "," << ind.timestampMs << "," << ind.ueRecords.size()
              << "," << std::fixed << std::setprecision(3) << rttMs
              << ",1,0," << std::setprecision(3) << replyDecisionMs << "\n";
        m_log.flush();
    }
    std::cout << std::fixed << std::setprecision(1)
              << "[SqrE2XappBridge] window=" << (m_indicationsReceived - 1)
              << " ts=" << ind.timestampMs << " kpm_ues=" << ind.ueRecords.size()
              << " rtt=" << rttMs << "ms"
              << " xapp_decision=" << replyDecisionMs << "ms"
              << " -> RIC control applied\n";
}

uint64_t
SqrE2LoopbackTerm::GetIndicationsReceived() const
{
    return m_indicationsReceived;
}

uint64_t
SqrE2LoopbackTerm::GetControlsSent() const
{
    return m_controlsSent;
}

const std::vector<long>&
SqrE2LoopbackTerm::GetPhases() const
{
    return m_phases;
}

SqrE2LoopbackTerm::EndpointMode
SqrE2LoopbackTerm::GetMode() const
{
    return m_mode;
}

uint64_t
SqrE2LoopbackTerm::GetXappTimeouts() const
{
    return m_xappTimeouts;
}

// ---------------------------------------------------------------------------
// SqrE2Adapter
// ---------------------------------------------------------------------------

NS_OBJECT_ENSURE_REGISTERED(SqrE2Adapter);

TypeId
SqrE2Adapter::GetTypeId()
{
    static TypeId tid = TypeId("ns3::SqrE2Adapter")
                            .SetParent<Object>()
                            .SetGroupName("Sqr")
                            .AddConstructor<SqrE2Adapter>();
    return tid;
}

SqrE2Adapter::SqrE2Adapter() = default;

SqrE2Adapter::~SqrE2Adapter() = default;

void
SqrE2Adapter::Install(Ptr<NetDevice> gnbDev,
                      Ptr<SqrMacScheduler> scheduler,
                      Ptr<SqrE2LoopbackTerm> ricEndpoint,
                      ActionCallback actionCb)
{
    NS_LOG_FUNCTION(this << gnbDev);
    m_scheduler = scheduler;
    m_ricEndpoint = ricEndpoint;
    m_actionCb = std::move(actionCb);

    auto gnbNode = DynamicCast<NrGnbNetDevice>(gnbDev);
    NS_ABORT_MSG_IF(!gnbNode, "SqrE2Adapter::Install: net device is not a NrGnbNetDevice");

    // --- Superfície E2 do contrib/nori (MESMA API nos modos FULL/DEGRADED;
    // fiação canônica replicada do E2TermHelper::InstallE2Term, sem os
    // trace sinks de PDCP/RLC/PHY — os KPMs do SQR vêm dos MESMOS KPIs do
    // contrato §4 do caminho IPC, PLANO §3-Fase 5.3). No DEGRADED, os
    // métodos de transporte do stub logam o aviso; no FULL, os MESMOS
    // objetos conectam ao Near-RT RIC via e2sim.
    const std::string plmnId = "00101"; // MCC=001, MNC=01 (E2TermHelper)
    const uint16_t cellId = gnbNode->GetCellId();
    const uint16_t localPort = 38470 + cellId; // default E2LocalPort + cellId
    m_e2Interface = CreateObject<E2Interface>(gnbDev);
    m_e2Term = CreateObject<E2Termination>("127.0.0.1", // A10 (MODA-PORT.md)
                                           36422,
                                           localPort,
                                           std::to_string(cellId),
                                           EncodePlmn(plmnId));
    gnbDev->AggregateObject(m_e2Term);
    m_e2Interface->SetAttribute("E2Term", PointerValue(m_e2Term));

    // Registro das function descriptions de serviço (KPM=200, RC=300, CCC=4 —
    // mesmos IDs do E2TermHelper::InstallE2Term). Os callbacks são os pontos
    // de entrada que o RIC aciona; em FULL chegam pelo e2sim, em DEGRADED os
    // stubs logam e o loopback assume a entrega das mensagens estruturadas.
    Ptr<KpmFunctionDescription> kpmFd = Create<KpmFunctionDescription>();
    m_e2Term->RegisterKpmCallbackToE2Sm(
        200, kpmFd, [this](E2AP_PDU_t* pdu) {
            m_e2Interface->FunctionServiceSubscriptionCallback(pdu);
        });
    Ptr<RicControlFunctionDescription> rcFd = Create<RicControlFunctionDescription>();
    m_e2Term->RegisterSmCallbackToE2Sm(
        300, rcFd, [this](E2AP_PDU_t* pdu) {
            m_e2Interface->ControlMessageReceivedCallback(pdu);
        });
    Ptr<CccFunctionDescription> cccFd = Create<CccFunctionDescription>();
    m_e2Term->RegisterCccCallbackToE2Sm(
        4, cccFd, [this](E2AP_PDU_t* pdu) {
            m_e2Interface->CccSubscriptionCallback(pdu);
        });

    // Endpoint RIC (DEGRADED: loopback em-processo com as mesmas mensagens
    // estruturadas; em FULL o endpoint é o transporte e2sim e este objeto
    // não é usado — lá a chegada é assíncrona, via e2sim → thread →
    // Simulator::ScheduleWithContext como em E2-interface.cc:477; aqui a
    // entrega é síncrona, na mesma thread de simulação). O E2-SETUP
    // simulado roda no mesmo instante em que o helper agendaria o Start
    // real (MicroSeconds(0)).
    NS_ABORT_MSG_IF(!m_ricEndpoint, "SqrE2Adapter::Install: null RIC endpoint");
    m_ricEndpoint->SetControlDeliverCallback(
        [this](const SqrE2RicControlMessage& ctrl) {
            this->OnRicControlMessage(ctrl);
        });
    Simulator::Schedule(MicroSeconds(0), &SqrE2LoopbackTerm::Start, m_ricEndpoint);

    std::cout << "[SqrE2Adapter] installed on gNB cellId=" << cellId
              << " (E2Interface+E2Termination of contrib/nori; RIC 127.0.0.1:36422,"
              << " RAN functions KPM=200 RC=300 CCC=4)\n";
}

void
SqrE2Adapter::SendKpmIndication(const SqrE2KpmIndicationMessage& ind)
{
    ++m_indicationsSent;
    // Ponto único de saída dos KPMs (E2SM-KPM indication). Em DEGRADED o
    // endpoint é o loopback em-processo; em FULL, aqui entraria a codificação
    // E2AP/E2SM-KPM (KpmIndicationHeader/Message do nori) + SendE2Message —
    // o caminho por janela do sqr-sim NÃO muda.
    m_ricEndpoint->OnE2smKpmIndication(ind);
}

bool
SqrE2Adapter::OnRicControlMessage(const SqrE2RicControlMessage& ctrl)
{
    ++m_controlsReceived;

    if (ctrl.requestType != SqrE2RicControlMessage::RAN_SLICING)
    {
        ++m_controlsIgnored;
        NS_LOG_WARN("[SqrE2Adapter] RIC Control with requestType=" << ctrl.requestType
                                                                   << " ignored (only RAN_SLICING is applied)");
        return false;
    }
    if (ctrl.ricControlDecision != 0)
    {
        ++m_controlsIgnored;
        NS_LOG_WARN("[SqrE2Adapter] RIC Control rejected by decision="
                    << ctrl.ricControlDecision << " — not applied");
        return false;
    }

    // Extração das % por slice (dedicatePRBRatio; SST = índice da slice).
    // Slice SEM linha na mensagem ⇒ degradação do contrato §5 (default
    // 33,33 %/slice, contável) — mesma regra do caminho IPC.
    std::array<double, 3> pct = {0.0, 0.0, 0.0};
    std::array<bool, 3> seen = {false, false, false};
    for (const auto& q : ctrl.prbQuotas)
    {
        if (q.sliceId >= SQR_E2_NUM_SLICES)
        {
            NS_LOG_WARN("[SqrE2Adapter] RIC Control quota for unknown sliceId="
                        << q.sliceId << " — ignored");
            continue;
        }
        pct[q.sliceId] = std::clamp(static_cast<double>(q.dedicatePRBRatio), 0.0, 100.0);
        seen[q.sliceId] = true;
    }
    const bool sliceMissing = !seen[0] || !seen[1] || !seen[2];
    if (sliceMissing)
    {
        ++m_sliceLineMissing;
        for (uint32_t s = 0; s < SQR_E2_NUM_SLICES; ++s)
        {
            if (!seen[s])
            {
                pct[s] = 33.33; // default do contrato §5 (contável)
            }
        }
    }

    if (!m_actionCb)
    {
        ++m_controlsIgnored;
        NS_LOG_ERROR("[SqrE2Adapter] no action callback installed — RIC Control dropped");
        return false;
    }
    ++m_controlsApplied;
    m_actionCb(pct, sliceMissing);
    return true;
}

uint64_t
SqrE2Adapter::GetIndicationsSent() const
{
    return m_indicationsSent;
}

uint64_t
SqrE2Adapter::GetControlsReceived() const
{
    return m_controlsReceived;
}

uint64_t
SqrE2Adapter::GetControlsApplied() const
{
    return m_controlsApplied;
}

uint64_t
SqrE2Adapter::GetControlsIgnored() const
{
    return m_controlsIgnored;
}

uint64_t
SqrE2Adapter::GetSliceLineMissingCount() const
{
    return m_sliceLineMissing;
}

} // namespace ns3
