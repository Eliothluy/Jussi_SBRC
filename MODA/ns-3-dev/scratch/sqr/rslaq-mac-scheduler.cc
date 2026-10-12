/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * RSLAQ Meta-Scheduler implementation for 5G-LENA (ns-3)
 *
 * Slice-aware OFDMA MAC scheduler with dynamic intra-slice algorithm
 * selection. Partitions PRBs across slices by configurable weights
 * and applies per-slice RR, PF, or BCQI ordering.
 *
 * CÓPIA INTEIRA de scratch/rslaq/rslaq-mac-scheduler.cc (original intocado),
 * usada pelo cenário SQR. Diferenças vs o original (marcadas com "SQR:"):
 * 1. Contadores de janela por slice de RBGs agendados/orçados
 *    (prb_scheduled/prb_budget do contrato SQR) — somente leitura via
 *    GetWindowScheduledRbg()/GetWindowBudgetRbg()/ResetWindowCounters().
 * 2. Snapshot do SINR DL estimado por RNTI (inversão do CQI wideband nas
 *    tabelas do NrAmc), exposto por GetUeDlSinrDb() para a coluna snrDb do KPM.
 * 3. GetTotalRbgCount()/GetTotalPrbCount() para a conversão percentual→RBG/PRB
 *    do controle SQR.
 * Nenhuma mudança na lógica de alocação/agendamento.
 */

#include "rslaq-mac-scheduler.h"

#include "ns3/nr-amc.h"
#include "ns3/nr-mac-scheduler-ue-info-pf.h"
#include "ns3/nr-mac-scheduler-ue-info-mr.h"
#include "ns3/nr-fh-control.h"

#include "ns3/log.h"
#include "ns3/simulator.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("RslaqMacScheduler");
NS_OBJECT_ENSURE_REGISTERED(RslaqMacScheduler);

namespace
{

std::vector<double>
NormalizeWeights(const std::vector<double>& raw,
                 const std::vector<bool>& active,
                 double minActiveShare)
{
    std::vector<double> clean(raw.size(), 0.0);
    uint32_t activeCount = 0;
    for (size_t i = 0; i < raw.size(); ++i)
    {
        bool isActive = active.empty() || active[i];
        if (isActive)
        {
            clean[i] = std::max(raw[i], 0.0);
            activeCount++;
        }
    }

    if (activeCount == 0)
    {
        return std::vector<double>(raw.size(), raw.empty() ? 0.0 : 1.0 / raw.size());
    }

    double sum = std::accumulate(clean.begin(), clean.end(), 0.0);
    if (sum <= 0.0)
    {
        for (size_t i = 0; i < clean.size(); ++i)
        {
            clean[i] = (!active.empty() && !active[i]) ? 0.0 : 1.0 / activeCount;
        }
        return clean;
    }

    for (double& value : clean)
    {
        value /= sum;
    }

    if (minActiveShare > 0.0)
    {
        double floorShare = std::min(minActiveShare, 0.95 / static_cast<double>(activeCount));
        double remaining = 1.0 - floorShare * static_cast<double>(activeCount);
        for (size_t i = 0; i < clean.size(); ++i)
        {
            if (!active.empty() && !active[i])
            {
                clean[i] = 0.0;
            }
            else
            {
                clean[i] = floorShare + remaining * clean[i];
            }
        }
    }

    double normalizedSum = std::accumulate(clean.begin(), clean.end(), 0.0);
    if (normalizedSum > 0.0)
    {
        for (double& value : clean)
        {
            value /= normalizedSum;
        }
    }
    return clean;
}

uint64_t
Mix64(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

double
Clamp01(double value)
{
    return std::max(0.0, std::min(1.0, value));
}

double
AqpsAlpha(uint32_t slice)
{
    if (slice == 1) // URLLC
    {
        return 0.8;
    }
    if (slice == 0) // eMBB
    {
        return 0.5;
    }
    return 0.2; // MTC as BE/mMTC class
}

double
AqpsPriority(uint32_t slice)
{
    if (slice == 1) // URLLC
    {
        return 0.545;
    }
    if (slice == 0) // eMBB
    {
        return 0.273;
    }
    return 0.182; // MTC as BE/mMTC class
}

uint32_t
AqpsPriorityOrder(uint32_t index)
{
    static const uint32_t order[3] = {1, 0, 2}; // URLLC -> eMBB -> MTC/BE
    return order[index % 3];
}

} // namespace

TypeId
RslaqMacScheduler::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::RslaqMacScheduler")
            .SetParent<NrMacSchedulerOfdmaRR>()
            .AddConstructor<RslaqMacScheduler>()
            .AddAttribute("TimeWindow",
                          "PF averaging time window (in slots)",
                          DoubleValue(99.0),
                          MakeDoubleAccessor(&RslaqMacScheduler::m_timeWindow),
                          MakeDoubleChecker<double>(1.0))
            .AddAttribute("FairnessIndex",
                          "PF alpha parameter (0=full fairness, 1=throughput maximization)",
                          DoubleValue(0.0),
                          MakeDoubleAccessor(&RslaqMacScheduler::m_alpha),
                          MakeDoubleChecker<double>(0.0, 1.0))
            .AddAttribute("FallbackUnmappedUe",
                          "If true, unmapped UEs are forced into slice 0 (debug only)",
                          BooleanValue(false),
                          MakeBooleanAccessor(&RslaqMacScheduler::m_fallbackUnmappedToSlice0),
                          MakeBooleanChecker())
            .AddAttribute("LogAllMacSlots",
                          "If true, log every MAC scheduling call (use with caution, produces large files)",
                          BooleanValue(false),
                          MakeBooleanAccessor(&RslaqMacScheduler::m_logAllMacSlots),
                          MakeBooleanChecker())
            .AddAttribute("MacLoggingPeriodMs",
                          "Period of MAC logging in milliseconds when LogAllMacSlots is false",
                          UintegerValue(100),
                          MakeUintegerAccessor(&RslaqMacScheduler::m_macLoggingPeriodMs),
                          MakeUintegerChecker<uint32_t>(1))
            .AddAttribute("EnableDetailedMacLogging",
                          "If true, enable detailed MAC-level logging with call IDs and slot info",
                          BooleanValue(false),
                          MakeBooleanAccessor(&RslaqMacScheduler::m_enableDetailedMacLogging),
                          MakeBooleanChecker());
    return tid;
}

RslaqMacScheduler::RslaqMacScheduler()
    : NrMacSchedulerOfdmaRR()
{
    NS_LOG_FUNCTION(this);
}

RslaqMacScheduler::~RslaqMacScheduler()
{
}

bool
RslaqMacScheduler::ShouldLogSlot(uint64_t timeMs) const
{
    if (m_logAllMacSlots)
    {
        return true;
    }
    if (m_enableDetailedMacLogging)
    {
        if (timeMs - m_lastLoggedMs >= m_macLoggingPeriodMs)
        {
            m_lastLoggedMs = timeMs;
            return true;
        }
        return false;
    }
    return (timeMs <= 200) || ((timeMs + 37) % 97 < 2);
}

uint32_t
RslaqMacScheduler::GetCurrentSlot() const
{
    return m_currentSlot;
}

uint32_t
RslaqMacScheduler::GetBwpId() const
{
    return m_bwpId;
}

uint64_t
RslaqMacScheduler::GetNextCallId() const
{
    return ++m_dlSchedCallSeq;
}

void
RslaqMacScheduler::SetCurrentSlot(uint32_t slot)
{
    m_currentSlot = slot;
}

void
RslaqMacScheduler::SetBwpId(uint32_t bwpId)
{
    m_bwpId = bwpId;
}

void
RslaqMacScheduler::SetSliceConfiguration(const std::vector<double>& prbWeights,
                                          const std::vector<IntraSliceAlgorithm>& algorithms)
{
    /*
     * Runtime control point used by both static baselines and the Python agent.
     * prbWeights is the slice share vector [eMBB, URLLC, MTC]; algorithms stores
     * the intra-slice policy applied after each slice receives its RBG budget.
     */
    NS_LOG_FUNCTION(this);
    NS_ASSERT_MSG(prbWeights.size() == algorithms.size(), "Weights and algorithms must have same size");
    NS_ASSERT_MSG(prbWeights.size() == m_numSlices, "Must match configured number of slices");

    double sum = std::accumulate(prbWeights.begin(), prbWeights.end(), 0.0);
    for (double weight : prbWeights)
    {
        NS_ASSERT_MSG(weight >= 0.0, "PRB weights must be non-negative, got " << weight);
    }
    NS_ASSERT_MSG(sum > 0.0, "PRB weights must have positive sum, got " << sum);

    // Renormalize to absorb rounding drift from external callers (e.g. weights
    // serialized with 6 decimal places can sum to 0.999999 or 1.000001, which
    // tripped the original strict 1e-6 assert and aborted the simulation).
    // The scheduler always works with sum=1.0 internally after this point.
    std::vector<double> normalizedWeights;
    normalizedWeights.reserve(prbWeights.size());
    for (double w : prbWeights)
    {
        normalizedWeights.push_back(w / sum);
    }

    m_prbWeights = normalizedWeights;
    m_lastDecisionWeights = normalizedWeights;
    m_intraAlgorithms = algorithms;

    std::ostringstream oss;
    oss << "[RslaqScheduler] Updated config:";
    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        oss << " Slice" << s << "(w=" << m_prbWeights[s];
        switch (m_intraAlgorithms[s])
        {
        case IntraSliceAlgorithm::RR:
            oss << ",RR";
            break;
        case IntraSliceAlgorithm::PF:
            oss << ",PF";
            break;
        case IntraSliceAlgorithm::BCQI:
            oss << ",BCQI";
            break;
        }
        oss << ")";
    }
    NS_LOG_INFO(oss.str());
}

void
RslaqMacScheduler::SetSliceUeMapping(uint32_t numSlices,
                                      const std::vector<std::vector<uint32_t>>& sliceUeRnti)
{
    /*
     * Called once after attach, when real RNTIs are known. The simulator maps
     * UE IDs to slices, but the MAC scheduler receives active users by RNTI.
     */
    NS_LOG_FUNCTION(this);
    m_numSlices = numSlices;
    m_sliceUeRnti = sliceUeRnti;
    NS_ASSERT_MSG(m_numSlices == 3, "RSLAQ experiments require exactly 3 slices");

    m_prbWeights.resize(m_numSlices, 1.0 / m_numSlices);
    m_lastDecisionWeights = m_prbWeights;
    m_intraAlgorithms.resize(m_numSlices, IntraSliceAlgorithm::RR);
    m_sliceAllocationStats.assign(m_numSlices, SliceAllocationStats());

    // SQR: (re)dimensiona contadores de janela; o mapeamento é reenviado a cada
    // chegada/partida de UE dinâmico no cenário SQR, o que reinicia a janela —
    // comportamento aceitável, pois o sqr-sim drena os contadores a cada período.
    m_windowScheduledRbg.assign(m_numSlices, 0);
    m_windowBudgetRbg.assign(m_numSlices, 0);

    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        std::ostringstream oss;
        oss << "[RslaqScheduler] Slice " << s << " RNTIs:";
        for (uint32_t rnti : m_sliceUeRnti[s])
        {
            oss << " " << rnti;
        }
        NS_LOG_INFO(oss.str());
    }
}

uint32_t
RslaqMacScheduler::GetNumSlices() const
{
    return m_numSlices;
}

double
RslaqMacScheduler::GetPrbWeight(uint32_t sliceIdx) const
{
    NS_ASSERT(sliceIdx < m_numSlices);
    if (m_lastDecisionWeights.size() == m_numSlices)
    {
        return m_lastDecisionWeights[sliceIdx];
    }
    return m_prbWeights[sliceIdx];
}

RslaqMacScheduler::IntraSliceAlgorithm
RslaqMacScheduler::GetIntraAlgorithm(uint32_t sliceIdx) const
{
    NS_ASSERT(sliceIdx < m_numSlices);
    return m_intraAlgorithms[sliceIdx];
}

void
RslaqMacScheduler::SetSliceWeightPolicy(SliceWeightPolicy policy)
{
    m_sliceWeightPolicy = policy;
}

RslaqMacScheduler::SliceWeightPolicy
RslaqMacScheduler::GetSliceWeightPolicy() const
{
    return m_sliceWeightPolicy;
}

int32_t
RslaqMacScheduler::GetSliceIndexForRnti(uint16_t rnti) const
{
    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        for (uint32_t r : m_sliceUeRnti[s])
        {
            if (r == rnti)
            {
                return static_cast<int32_t>(s);
            }
        }
    }
    return -1;
}

void
RslaqMacScheduler::DumpSliceConfiguration() const
{
    std::ostringstream oss;
    oss << "[RslaqScheduler] DumpSliceConfiguration: numSlices=" << m_numSlices;
    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        oss << " | Slice" << s << " w=" << m_prbWeights[s] << " alg=";
        switch (m_intraAlgorithms[s])
        {
        case IntraSliceAlgorithm::RR:
            oss << "RR";
            break;
        case IntraSliceAlgorithm::PF:
            oss << "PF";
            break;
        case IntraSliceAlgorithm::BCQI:
            oss << "BCQI";
            break;
        }
        oss << " rntis=[";
        for (size_t i = 0; i < m_sliceUeRnti[s].size(); ++i)
        {
            if (i)
                oss << ",";
            oss << m_sliceUeRnti[s][i];
        }
        oss << "]";
    }
    NS_LOG_INFO(oss.str());
}

void
RslaqMacScheduler::SetScenarioName(const std::string& name)
{
    m_scenarioName = name;
}

void
RslaqMacScheduler::SetOutputDir(const std::string& dir)
{
    m_outputDir = dir;
}

void
RslaqMacScheduler::OpenCsvFiles() const
{
    std::string prefix = m_outputDir;
    if (!prefix.empty() && prefix.back() != '/')
    {
        prefix += '/';
    }
    if (!m_sliceAllocCsv.is_open())
    {
        std::string path = prefix + "slice_alloc.csv";
        m_sliceAllocCsv.open(path, std::ios::out | std::ios::trunc);
        m_sliceAllocCsv << "timestamp_ms,slice,configured_weight,budget_rbg,allocated_rbg,"
                           "total_allocated_rbg,rsh_real_pct,budget_utilization_pct,"
                           "unused_budget_pct,allocation_fidelity_pct,active_ues,effective_weight,"
                           "beam_id,has_demand,reason,redistributed_idle_rbg,rntis\n";
        m_sliceAllocCsv.flush();
    }
    if (!m_unmappedRntiCsv.is_open())
    {
        std::string path = prefix + "unmapped_rntis.csv";
        m_unmappedRntiCsv.open(path, std::ios::out | std::ios::trunc);
        // Header with scenario field
        m_unmappedRntiCsv << "callId,timeMs,scenario,rnti,reason\n";
        m_unmappedRntiCsv.flush();
    }
    if (!m_ueAllocCsv.is_open())
    {
        std::string path = prefix + "ue_detail.csv";
        m_ueAllocCsv.open(path, std::ios::out | std::ios::trunc);
        // Header with new fields: callId, slot, bwpId, beamId, tbSizeBytes, reason
        m_ueAllocCsv << "callId,timeMs,scenario,slot,bwpId,beamId,sliceId,rnti,bufQueueSize,"
                           "dlTbSizeBefore,hasDemand,demandPassed,rbgAllocated,uniqueRbgs,"
                           "tbSizeBytes,mcs,rank,numRbPerRbg,reason\n";
        m_ueAllocCsv.flush();
    }
    if (!m_harqTrackingCsv.is_open())
    {
        std::string path = prefix + "harq_tracking.csv";
        m_harqTrackingCsv.open(path, std::ios::out | std::ios::trunc);
        m_harqTrackingCsv << "callId,timeMs,scenario,sliceId,rnti,activeHarqCount,"
                             "harqCapacity,canInsert,dlBufferSize,beamId\n";
        m_harqTrackingCsv.flush();
    }
    if (!m_activeDlDiagCsv.is_open())
    {
        std::string path = prefix + "active_dl_diag.csv";
        m_activeDlDiagCsv.open(path, std::ios::out | std::ios::trunc);
        m_activeDlDiagCsv << "callId,timeMs,scenario,sliceId,rnti,inActiveDl,dlBufferSize,"
                             "canInsertHarq,harqActiveCount,diagReason\n";
        m_activeDlDiagCsv.flush();
    }
}

std::shared_ptr<NrMacSchedulerUeInfo>
RslaqMacScheduler::CreateUeRepresentation(
    const NrMacCschedSapProvider::CschedUeConfigReqParameters& params) const
{
    NS_LOG_FUNCTION(this);
    return std::make_shared<NrMacSchedulerUeInfoPF>(
        static_cast<float>(m_alpha),
        params.m_rnti,
        params.m_beamId,
        std::bind(&RslaqMacScheduler::GetNumRbPerRbg, this));
}

void
RslaqMacScheduler::SortUeVectorByAlgorithm(std::vector<UePtrAndBufferReq>& ueVector,
                                            IntraSliceAlgorithm algo) const
{
    switch (algo)
    {
    case IntraSliceAlgorithm::RR:
        std::stable_sort(ueVector.begin(), ueVector.end(),
                         NrMacSchedulerUeInfoRR::CompareUeWeightsDl);
        break;
    case IntraSliceAlgorithm::PF:
        std::stable_sort(ueVector.begin(), ueVector.end(),
                         NrMacSchedulerUeInfoPF::CompareUeWeightsDl);
        break;
    case IntraSliceAlgorithm::BCQI:
        std::stable_sort(ueVector.begin(), ueVector.end(),
                         NrMacSchedulerUeInfoMR::CompareUeWeightsDl);
        break;
    }
}

std::vector<double>
RslaqMacScheduler::ComputeDecisionWeights(
    const std::vector<std::vector<UePtrAndBufferReq>>& sliceUeVec,
    const std::vector<bool>& sliceHasDemand,
    uint64_t timeMs) const
{
    if (m_sliceWeightPolicy == SliceWeightPolicy::STATIC || m_numSlices == 0)
    {
        return m_prbWeights;
    }

    std::vector<double> bufferSum(m_numSlices, 0.0);
    std::vector<double> maxBuffer(m_numSlices, 0.0);
    std::vector<uint32_t> activeUes(m_numSlices, 0);

    for (uint32_t s = 0; s < m_numSlices; ++s)
    {
        if (!sliceHasDemand[s])
        {
            continue;
        }
        for (const auto& ue : sliceUeVec[s])
        {
            double buf = static_cast<double>(ue.second);
            double tb = static_cast<double>(ue.first->m_dlTbSize);
            if (buf > 0.0 || tb > 0.0)
            {
                activeUes[s] += 1;
            }
            bufferSum[s] += std::max(buf, 0.0) + std::max(tb, 0.0);
            maxBuffer[s] = std::max(maxBuffer[s], std::max(buf, 0.0));
        }
    }

    std::vector<double> raw(m_numSlices, 0.0);
    switch (m_sliceWeightPolicy)
    {
    case SliceWeightPolicy::DEMAND_GREEDY:
        for (uint32_t s = 0; s < m_numSlices; ++s)
        {
            if (sliceHasDemand[s])
            {
                raw[s] = bufferSum[s] + 1500.0 * static_cast<double>(activeUes[s]) + 1.0;
            }
        }
        return NormalizeWeights(raw, sliceHasDemand, 0.02);

    case SliceWeightPolicy::SLA_GREEDY:
        for (uint32_t s = 0; s < m_numSlices; ++s)
        {
            if (!sliceHasDemand[s])
            {
                continue;
            }

            double base = bufferSum[s] + 1500.0 * static_cast<double>(activeUes[s]) + 1.0;
            double urgency = 1.0;
            if (s == 0)
            {
                urgency += std::min(bufferSum[s] / 1000000.0, 4.0);
            }
            else if (s == 1)
            {
                urgency += 1.0 + 2.0 * std::min(maxBuffer[s] / 10000.0, 8.0);
            }
            else if (s == 2)
            {
                urgency += std::min(bufferSum[s] / 250000.0, 4.0);
            }
            raw[s] = base * urgency;
        }
        return NormalizeWeights(raw, sliceHasDemand, 0.03);

    case SliceWeightPolicy::LEAST_WASTE:
        for (uint32_t s = 0; s < m_numSlices; ++s)
        {
            if (sliceHasDemand[s])
            {
                raw[s] = bufferSum[s] + 256.0 * static_cast<double>(activeUes[s]) + 1.0;
            }
        }
        return NormalizeWeights(raw, sliceHasDemand, 0.005);

    case SliceWeightPolicy::RANDOM_VINE:
        for (uint32_t s = 0; s < m_numSlices; ++s)
        {
            if (sliceHasDemand[s])
            {
                uint64_t mixed = Mix64((timeMs + 1ULL) ^ (static_cast<uint64_t>(s + 1) << 32) ^
                                       (static_cast<uint64_t>(m_currentSlot + 17) << 8));
                raw[s] = 1.0 + static_cast<double>(mixed % 1000ULL);
            }
        }
        return NormalizeWeights(raw, sliceHasDemand, 0.01);

    case SliceWeightPolicy::META_RISK_ELASTIC:
    {
        std::vector<double> riskRaw(m_numSlices, 0.0);
        std::vector<double> priorRaw(m_numSlices, 0.0);

        double maxPressure = 0.0;
        for (uint32_t s = 0; s < m_numSlices; ++s)
        {
            if (!sliceHasDemand[s])
            {
                continue;
            }

            double base = bufferSum[s] + 1500.0 * static_cast<double>(activeUes[s]) + 1.0;
            double urgency = 1.0;
            double pressure = 0.0;
            if (s == 0)
            {
                urgency += std::min(bufferSum[s] / 1000000.0, 4.0);
                pressure = Clamp01(bufferSum[s] / 2000000.0);
            }
            else if (s == 1)
            {
                urgency += 1.5 + 2.0 * std::min(maxBuffer[s] / 10000.0, 8.0) +
                           std::min(bufferSum[s] / 100000.0, 3.0);
                pressure = Clamp01(std::max(maxBuffer[s] / 10000.0, bufferSum[s] / 100000.0));
            }
            else if (s == 2)
            {
                urgency += std::min(bufferSum[s] / 250000.0, 4.0) +
                           0.5 * std::min(static_cast<double>(activeUes[s]) / 20.0, 2.0);
                pressure = Clamp01(bufferSum[s] / 250000.0);
            }

            riskRaw[s] = base * urgency;
            priorRaw[s] = (s < m_prbWeights.size() ? m_prbWeights[s] : 1.0 / m_numSlices);
            maxPressure = std::max(maxPressure, pressure);
        }

        std::vector<double> riskWeights = NormalizeWeights(riskRaw, sliceHasDemand, 0.02);
        std::vector<double> priorWeights = NormalizeWeights(priorRaw, sliceHasDemand, 0.02);
        std::vector<double> blended(m_numSlices, 0.0);

        // Stable queues keep the offline prior; pressure shifts weight to online risk control.
        double priorShare = 0.75 - 0.55 * maxPressure;
        priorShare = std::max(0.20, std::min(0.75, priorShare));
        for (uint32_t s = 0; s < m_numSlices; ++s)
        {
            if (sliceHasDemand[s])
            {
                blended[s] = priorShare * priorWeights[s] + (1.0 - priorShare) * riskWeights[s];
            }
        }
        return NormalizeWeights(blended, sliceHasDemand, 0.02);
    }

    case SliceWeightPolicy::AQPS:
        return NormalizeWeights(m_prbWeights, sliceHasDemand, 0.0);

    case SliceWeightPolicy::STATIC:
        break;
    }

    return m_prbWeights;
}

std::vector<uint32_t>
RslaqMacScheduler::ComputeAqpsSliceBudgets(
    const std::vector<std::vector<UePtrAndBufferReq>>& sliceUeVec,
    const std::vector<bool>& sliceHasDemand,
    uint32_t totalRbgs) const
{
    std::vector<uint32_t> budgets(m_numSlices, 0);
    if (m_numSlices == 0 || totalRbgs == 0)
    {
        return budgets;
    }

    std::vector<uint32_t> activeUes(m_numSlices, 0);
    std::vector<uint32_t> requiredRbgs(m_numSlices, 0);
    std::vector<double> urgency(m_numSlices, 0.0);

    for (uint32_t s = 0; s < m_numSlices; ++s)
    {
        if (!sliceHasDemand[s])
        {
            continue;
        }

        for (const auto& ue : sliceUeVec[s])
        {
            double bufferBytes = std::max(0.0, static_cast<double>(ue.second));
            double tbBytes = std::max(0.0, static_cast<double>(ue.first->m_dlTbSize));
            if (bufferBytes <= 0.0 && tbBytes <= 0.0)
            {
                continue;
            }

            activeUes[s] += 1;
            double mcs = static_cast<double>(ue.first->GetDlMcs());
            double spectralFactor = 1.0 + std::min(mcs / 28.0, 1.0);
            double packetBytes = (s == 1) ? 50.0 : ((s == 2) ? 100.0 : 1500.0);
            double bytesPerRbg = std::max(packetBytes, packetBytes * GetNumRbPerRbg() * spectralFactor);
            double demandBytes = std::max(packetBytes, bufferBytes + tbBytes);

            uint32_t ueRequired = static_cast<uint32_t>(
                std::ceil(demandBytes / std::max(bytesPerRbg, 1.0)));
            ueRequired = std::max<uint32_t>(1, ueRequired);
            requiredRbgs[s] += ueRequired;

            double qosTerm = 0.0;
            if (s == 1)
            {
                qosTerm = 50000.0 * Clamp01(bufferBytes / 10000.0) +
                          5000.0 * static_cast<double>(ueRequired);
            }
            else if (s == 0)
            {
                qosTerm = 20000.0 * Clamp01(bufferBytes / 1000000.0) +
                          5000.0 / spectralFactor;
            }
            else
            {
                qosTerm = 10000.0 * Clamp01(bufferBytes / 250000.0) +
                          1000.0 * static_cast<double>(ueRequired);
            }
            urgency[s] += std::max(0.0, (demandBytes + qosTerm) * spectralFactor * AqpsPriority(s));
        }
    }

    uint32_t totalActive = std::accumulate(activeUes.begin(), activeUes.end(), 0u);
    if (totalActive == 0)
    {
        return budgets;
    }

    auto allocateMin = [&](uint32_t slice, uint32_t requested) {
        uint32_t allocated = std::min(requested, totalRbgs - std::accumulate(budgets.begin(), budgets.end(), 0u));
        budgets[slice] += allocated;
    };

    uint32_t urllcMin = std::max(activeUes[1], requiredRbgs[1]);
    allocateMin(1, urllcMin);
    allocateMin(0, activeUes[0]);
    allocateMin(2, activeUes[2]);

    uint32_t allocatedAfterMin = std::accumulate(budgets.begin(), budgets.end(), 0u);
    uint32_t remaining = (allocatedAfterMin < totalRbgs) ? totalRbgs - allocatedAfterMin : 0;

    double weightedUrgencySum = 0.0;
    for (uint32_t s = 0; s < m_numSlices; ++s)
    {
        if (activeUes[s] > 0)
        {
            weightedUrgencySum += urgency[s];
        }
    }

    uint32_t weightedAllocated = 0;
    if (remaining > 0 && weightedUrgencySum > 0.0)
    {
        std::vector<double> combinedRaw(m_numSlices, 0.0);
        double combinedSum = 0.0;
        for (uint32_t s = 0; s < m_numSlices; ++s)
        {
            if (activeUes[s] == 0)
            {
                continue;
            }
            double urgencyShare = urgency[s] / weightedUrgencySum;
            double activeShare = static_cast<double>(activeUes[s]) / static_cast<double>(totalActive);
            combinedRaw[s] = AqpsAlpha(s) * urgencyShare + (1.0 - AqpsAlpha(s)) * activeShare;
            combinedSum += combinedRaw[s];
        }
        if (combinedSum > 0.0)
        {
            for (uint32_t s = 0; s < m_numSlices; ++s)
            {
                if (activeUes[s] == 0)
                {
                    continue;
                }
                double combinedShare = combinedRaw[s] / combinedSum;
                uint32_t add = static_cast<uint32_t>(std::floor(static_cast<double>(remaining) * combinedShare));
                budgets[s] += add;
                weightedAllocated += add;
            }
        }
    }

    uint32_t left = (weightedAllocated < remaining) ? remaining - weightedAllocated : 0;
    while (left > 0)
    {
        bool assignedInRound = false;
        for (uint32_t i = 0; i < m_numSlices && left > 0; ++i)
        {
            uint32_t s = AqpsPriorityOrder(i);
            if (s < m_numSlices && activeUes[s] > 0)
            {
                budgets[s] += 1;
                left -= 1;
                assignedInRound = true;
            }
        }
        if (!assignedInRound)
        {
            break;
        }
    }

    uint32_t budgetSum = std::accumulate(budgets.begin(), budgets.end(), 0u);
    NS_ASSERT_MSG(budgetSum <= totalRbgs, "AQPS budget exceeded total RBGs");
    return budgets;
}

std::set<uint16_t>
RslaqMacScheduler::CollectActiveRntis(const ActiveUeMap& activeDl) const
{
    std::set<uint16_t> rntis;
    for (const auto& beam : activeDl)
    {
        for (const auto& ue : beam.second)
        {
            rntis.insert(ue.first->m_rnti);
        }
    }
    return rntis;
}

uint32_t
RslaqMacScheduler::GetUeDlBufferSize(uint16_t rnti) const
{
    auto it = m_lastDlBufferSize.find(rnti);
    return (it != m_lastDlBufferSize.end()) ? it->second : 0;
}

std::vector<RslaqMacScheduler::SliceAllocationStats>
RslaqMacScheduler::GetSliceAllocationStats() const
{
    return m_sliceAllocationStats;
}

void
RslaqMacScheduler::LogActiveDlDiagnostic(uint64_t callId, uint64_t timeMs,
                                          const ActiveUeMap& activeDl) const
{
    if (!m_activeDlDiagCsv.is_open())
    {
        return;
    }

    std::set<uint16_t> activeRntis = CollectActiveRntis(activeDl);

    std::map<uint16_t, uint32_t> activeRntiBufSize;
    for (const auto& beam : activeDl)
    {
        for (const auto& ue : beam.second)
        {
            activeRntiBufSize[ue.first->m_rnti] = ue.second;
        }
    }

    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        for (uint32_t rnti : m_sliceUeRnti[s])
        {
            uint16_t r = static_cast<uint16_t>(rnti);
            bool inActiveDl = activeRntis.count(r) > 0;
            uint32_t bufSize = inActiveDl ? activeRntiBufSize[r] : 0;

            std::string reason;
            if (inActiveDl)
            {
                reason = "in_active_dl";
            }
            else
            {
                reason = "not_in_active_dl";
            }

            m_activeDlDiagCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                              << s << "," << r << "," << (inActiveDl ? 1 : 0) << ","
                              << bufSize << "," << 0 << ","
                              << 0 << "," << reason << "\n";
        }
    }
    m_activeDlDiagCsv.flush();
}

void
RslaqMacScheduler::LogHarqState(uint64_t callId, uint64_t timeMs) const
{
    // Corrige o defeito do harq_tracking.csv vazio (só cabeçalho). Esta subclasse
    // não expõe as contagens de processos HARQ (activeHarqCount/harqCapacity/
    // canInsert vivem na maquinaria HARQ do NrMacSchedulerNs3, inacessível aqui —
    // o próprio LogActiveDlDiagnostic escreve 0 nesses campos). Registramos o que
    // é de fato conhecido — dlBufferSize por RNTI (m_lastDlBufferSize) — e
    // marcamos os campos de HARQ como NA, sem fabricar valores.
    if (!m_harqTrackingCsv.is_open())
    {
        return;
    }
    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        for (uint32_t rnti : m_sliceUeRnti[s])
        {
            uint16_t r = static_cast<uint16_t>(rnti);
            uint32_t bufSize = GetUeDlBufferSize(r);
            m_harqTrackingCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                              << s << "," << r << ","
                              << "NA" << "," << "NA" << "," << "NA" << ","
                              << bufSize << "," << "NA" << "\n";
        }
    }
    m_harqTrackingCsv.flush();
}

NrMacSchedulerNs3::BeamSymbolMap
RslaqMacScheduler::AssignDLRBG(uint32_t symAvail, const ActiveUeMap& activeDl) const
{
    /*
     * Core slice-aware allocation loop:
     * 1. Group active DL UEs by their configured slice RNTI mapping.
     * 2. Detect which slices currently have downlink demand.
     * 3. Redistribute the available RBG budget only among active-demand slices.
     * 4. Sort UEs inside each slice by RR/PF/BCQI and allocate RBGs round-robin.
     * 5. Export per-slice and per-UE diagnostics for later reward/analysis.
     */
    NS_LOG_FUNCTION(this);
    NS_LOG_DEBUG("AssignDLRBG: #beams=" << activeDl.size() << " symAvail=" << symAvail);

    if (m_firstRun)
    {
        m_firstRun = false;
        OpenCsvFiles();
        DumpSliceConfiguration();
    }

    GetFirst GetBeamId;
    GetSecond GetUeVector;
    BeamSymbolMap symPerBeam = GetSymPerBeam(symAvail, activeDl);

    uint64_t timeMs = Simulator::Now().GetMilliSeconds();
    uint64_t callId = GetNextCallId();
    bool logThisSlot = ShouldLogSlot(timeMs);

    if (logThisSlot)
    {
        LogActiveDlDiagnostic(callId, timeMs, activeDl);
        LogHarqState(callId, timeMs);
    }

    // Helper to serialize BeamId to string
    auto beamIdToStr = [](const BeamId& bid) -> std::string {
        std::ostringstream oss;
        oss << bid.GetSector() << "_" << bid.GetElevation();
        return oss.str();
    };

    for (const auto& el : activeDl)
    {
        BeamId beamId = GetBeamId(el);
        uint32_t beamSym = symPerBeam.at(beamId);
        const std::vector<bool> dlNotchedMask = GetDlNotchedRbgMask();
        std::string beamIdStr = beamIdToStr(beamId);

        std::vector<uint32_t> availableRbgIds;
        if (!dlNotchedMask.empty())
        {
            for (uint32_t i = 0; i < dlNotchedMask.size(); ++i)
            {
                if (dlNotchedMask[i])
                {
                    availableRbgIds.push_back(i);
                }
            }
        }
        else
        {
            uint32_t bwRbg = GetBandwidthInRbg();
            availableRbgIds.reserve(bwRbg);
            for (uint32_t i = 0; i < bwRbg; ++i)
            {
                availableRbgIds.push_back(i);
            }
        }

        uint32_t totalRbgs = static_cast<uint32_t>(availableRbgIds.size());
        NS_ASSERT(totalRbgs > 0);

        std::vector<bool> rbgRealUsed(GetBandwidthInRbg(), false);
        if (!dlNotchedMask.empty())
        {
            for (uint32_t i = 0; i < dlNotchedMask.size(); ++i)
            {
                if (!dlNotchedMask[i])
                {
                    rbgRealUsed[i] = true;
                }
            }
        }

        FTResources assigned(0, 0);

        std::vector<std::vector<UePtrAndBufferReq>> sliceUeVec(m_numSlices);
        std::vector<uint16_t> unmappedRntis;
        for (const auto& ue : GetUeVector(el))
        {
            uint16_t rnti = ue.first->m_rnti;

            // SQR: snapshot do SINR DL por RNTI. Melhor aproximação disponível
            // sem sonda PHY dedicada: inverte o CQI wideband reportado
            // (m_dlCqi.m_wbCqi) nas próprias tabelas do AMC do 5G-LENA
            // (SE = NrAmc::GetSpectralEfficiencyForCqi(cqi);
            //  SINR = NrAmc::GetSinrFromSpectralEfficiency(SE), mapeamento
            //  TS 38.214). É o SINR que produziu o CQI reportado — não é uma
            //  medição instantânea por slot. CQI 0 (sem relatório válido)
            //  vira o sentinel -100.0 dB.
            {
                uint8_t wbCqi = ue.first->m_dlCqi.m_wbCqi;
                double sinrDb = -100.0;
                if (wbCqi > 0 && GetDlAmc())
                {
                    double se = GetDlAmc()->GetSpectralEfficiencyForCqi(wbCqi);
                    double sinrLin = GetDlAmc()->GetSinrFromSpectralEfficiency(se);
                    sinrDb = 10.0 * std::log10(std::max(sinrLin, 1e-12));
                }
                m_lastDlSinrDb[rnti] = sinrDb;
            }

            int32_t sIdx = GetSliceIndexForRnti(rnti);
            if (sIdx >= 0)
            {
                sliceUeVec[static_cast<uint32_t>(sIdx)].emplace_back(ue);
            }
            else
            {
                NS_LOG_WARN("[RslaqScheduler] RNTI " << rnti << " not found in any slice!");
                unmappedRntis.push_back(rnti);
                if (m_unmappedRntiCsv.is_open())
                {
                    m_unmappedRntiCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                                     << rnti << ",not_in_any_slice\n";
                }
                if (m_fallbackUnmappedToSlice0 && m_numSlices > 0)
                {
                    sliceUeVec[0].emplace_back(ue);
                }
            }
        }

        std::vector<bool> sliceHasDemand(m_numSlices, false);
        for (uint32_t s = 0; s < m_numSlices; s++)
        {
            if (sliceUeVec[s].empty())
            {
                continue;
            }
            bool demand = false;
            for (const auto& ue : sliceUeVec[s])
            {
                uint32_t bufQueueSize = ue.second;
                // Condição mais flexível: considera UE ativo se houver qualquer buffer
                // OU se estiver recebendo dados (detecção de atividade de tráfego)
                if (bufQueueSize > 0 || ue.first->m_dlTbSize > 0)
                {
                    demand = true;
                    break;
                }
            }
            if (demand)
            {
                sliceHasDemand[s] = true;
            }
        }

        std::vector<double> decisionWeights(m_numSlices, 0.0);
        std::vector<double> effectiveWeight(m_numSlices, 0.0);
        std::vector<uint32_t> sliceRbgBudget(m_numSlices, 0);
        double activeWeightSum = 0.0;

        if (m_sliceWeightPolicy == SliceWeightPolicy::AQPS)
        {
            sliceRbgBudget = ComputeAqpsSliceBudgets(sliceUeVec, sliceHasDemand, totalRbgs);
            uint32_t budgetSum = std::accumulate(sliceRbgBudget.begin(), sliceRbgBudget.end(), 0u);
            if (budgetSum > 0)
            {
                activeWeightSum = 1.0;
                for (uint32_t s = 0; s < m_numSlices; ++s)
                {
                    decisionWeights[s] = static_cast<double>(sliceRbgBudget[s]) /
                                         static_cast<double>(budgetSum);
                    effectiveWeight[s] = decisionWeights[s];
                }
            }
        }
        else
        {
            decisionWeights = ComputeDecisionWeights(sliceUeVec, sliceHasDemand, timeMs);
            for (uint32_t s = 0; s < m_numSlices; s++)
            {
                if (sliceHasDemand[s])
                {
                    activeWeightSum += decisionWeights[s];
                }
            }

            if (activeWeightSum > 0.0)
            {
                struct BudgetCandidate
                {
                    uint32_t slice = 0;
                    double remainder = 0.0;
                };
                std::vector<BudgetCandidate> candidates;
                uint32_t budgetSum = 0;
                for (uint32_t s = 0; s < m_numSlices; s++)
                {
                    if (!sliceHasDemand[s])
                    {
                        continue;
                    }
                    effectiveWeight[s] = decisionWeights[s] / activeWeightSum;
                    double rawBudget = static_cast<double>(totalRbgs) * effectiveWeight[s];
                    double floored = std::floor(rawBudget);
                    sliceRbgBudget[s] = static_cast<uint32_t>(floored);
                    budgetSum += sliceRbgBudget[s];
                    candidates.push_back({s, rawBudget - floored});
                }

                NS_ASSERT_MSG(budgetSum <= totalRbgs, "Floor budgets exceeded total RBGs");
                uint32_t remaining = totalRbgs - budgetSum;
                uint32_t rotation = (static_cast<uint32_t>(timeMs) + m_currentSlot) % m_numSlices;
                std::stable_sort(candidates.begin(),
                                 candidates.end(),
                                 [rotation, this](const BudgetCandidate& a, const BudgetCandidate& b) {
                                     if (std::abs(a.remainder - b.remainder) > 1e-12)
                                     {
                                         return a.remainder > b.remainder;
                                     }
                                     uint32_t ar = (a.slice + m_numSlices - rotation) % m_numSlices;
                                     uint32_t br = (b.slice + m_numSlices - rotation) % m_numSlices;
                                     return ar < br;
                                 });

                for (uint32_t i = 0; i < remaining && i < candidates.size(); ++i)
                {
                    sliceRbgBudget[candidates[i].slice] += 1;
                }
            }
        }
        m_lastDecisionWeights = decisionWeights;

        uint32_t budgetTotal = std::accumulate(sliceRbgBudget.begin(), sliceRbgBudget.end(), 0u);
        NS_ASSERT_MSG(activeWeightSum <= 0.0 || budgetTotal == totalRbgs,
                      "sum(sliceBudgetRbg)=" << budgetTotal << " totalRbgs=" << totalRbgs);

        for (uint32_t s = 0; s < m_numSlices; s++)
        {
            for (auto& ue : sliceUeVec[s])
            {
                BeforeDlSched(ue, FTResources(totalRbgs, beamSym));
            }
        }

        std::vector<uint32_t> sliceAllocatedVec(m_numSlices, 0);
        std::vector<std::string> sliceReasonVec(m_numSlices);
        std::vector<std::string> sliceRntiListVec(m_numSlices);

        // --- Slice-by-slice allocation with diagnostic logging ---
        for (uint32_t s = 0; s < m_numSlices; s++)
        {
            std::string reason;
            uint32_t sliceAllocated = 0;

            if (sliceUeVec[s].empty())
            {
                reason = "no_active_ues";
            }
            else if (!sliceHasDemand[s])
            {
                reason = "no_demand";
            }
            else if (sliceRbgBudget[s] == 0)
            {
                reason = "zero_budget";
            }
            else
            {
                SortUeVectorByAlgorithm(sliceUeVec[s], m_intraAlgorithms[s]);

                uint32_t iteration = 0;
                const uint32_t maxIterations = sliceRbgBudget[s] * sliceUeVec[s].size() + sliceUeVec[s].size();

                while (sliceAllocated < sliceRbgBudget[s] && iteration < maxIterations)
                {
                    iteration++;
                    bool anyUeNeeded = false;

                    for (auto& ue : sliceUeVec[s])
                    {
                        if (sliceAllocated >= sliceRbgBudget[s])
                        {
                            break;
                        }

                        uint32_t bufQueueSize = ue.second;
                        uint32_t prevTbSize = ue.first->m_dlTbSize;
                        // Condição mais flexível para detecção de demanda
                        bool demandPassed = (bufQueueSize > 0 || prevTbSize > 0);

                        if (logThisSlot && m_ueAllocCsv.is_open())
                        {
                            uint32_t curUnique = 0;
                            if (!ue.first->m_dlRBG.empty())
                            {
                                std::set<uint16_t> uniq(ue.first->m_dlRBG.begin(),
                                                        ue.first->m_dlRBG.end());
                                curUnique = static_cast<uint32_t>(uniq.size());
                            }
                            m_ueAllocCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                                         << m_currentSlot << "," << m_bwpId << "," << beamIdStr << ","
                                         << s << "," << ue.first->m_rnti << ","
                                         << bufQueueSize << "," << prevTbSize << ","
                                         << 0 << "," << (demandPassed ? 1 : 0) << ","
                                         << 0 << "," << curUnique << "," << 0 << ","
                                         << static_cast<uint32_t>(ue.first->GetDlMcs()) << ","
                                         << static_cast<uint32_t>(ue.first->m_dlRank) << ","
                                         << GetNumRbPerRbg() << ",pre_allocation\n";
                        }

                        if (!demandPassed)
                        {
                            continue;
                        }
                        anyUeNeeded = true;

                        uint32_t logicalIdx = 0;
                        for (; logicalIdx < totalRbgs; ++logicalIdx)
                        {
                            uint32_t realId = availableRbgIds[logicalIdx];
                            if (!rbgRealUsed[realId])
                            {
                                break;
                            }
                        }
                        if (logicalIdx >= totalRbgs)
                        {
                            break;
                        }

                        uint32_t actualRbgId = availableRbgIds[logicalIdx];
                        rbgRealUsed[actualRbgId] = true;

                        auto& dlRbg = ue.first->m_dlRBG;
                        auto existingRbgs = dlRbg.size();
                        dlRbg.resize(dlRbg.size() + beamSym);
                        std::fill(dlRbg.begin() + existingRbgs, dlRbg.end(), actualRbgId);

                        auto& dlSym = ue.first->m_dlSym;
                        auto existingSyms = dlSym.size();
                        dlSym.resize(dlSym.size() + beamSym);
                        std::iota(dlSym.begin() + existingSyms, dlSym.end(), 0);

                        assigned.m_rbg += 1;
                        assigned.m_sym = beamSym;

                        sliceAllocated++;

                        AssignedDlResources(ue, FTResources(1, beamSym), assigned);

                        for (auto& otherUe : sliceUeVec[s])
                        {
                            if (ue.first->m_rnti != otherUe.first->m_rnti)
                            {
                                NotAssignedDlResources(otherUe, FTResources(1, beamSym), assigned);
                            }
                        }

                        if (logThisSlot && m_ueAllocCsv.is_open())
                        {
                            std::set<uint16_t> uniq(ue.first->m_dlRBG.begin(),
                                                    ue.first->m_dlRBG.end());
                            uint32_t tbSize = ue.first->m_dlTbSize;
                            m_ueAllocCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                                         << m_currentSlot << "," << m_bwpId << "," << beamIdStr << ","
                                         << s << "," << ue.first->m_rnti << ","
                                         << bufQueueSize << "," << ue.first->m_dlTbSize << ","
                                         << 1 << "," << 1 << "," << 1 << ","
                                         << static_cast<uint32_t>(uniq.size()) << "," << tbSize << ","
                                         << static_cast<uint32_t>(ue.first->GetDlMcs()) << ","
                                         << static_cast<uint32_t>(ue.first->m_dlRank) << ","
                                         << GetNumRbPerRbg() << ",allocated\n";
                        }
                    }

                    if (!anyUeNeeded)
                    {
                        break;
                    }
                }

                if (sliceAllocated == 0)
                {
                    reason = "all_ue_demand_satisfied";
                }
                else if (sliceAllocated < sliceRbgBudget[s])
                {
                    reason = "partial_alloc";
                }
                else
                {
                    reason = "full_alloc";
                }
            }

            sliceAllocatedVec[s] = sliceAllocated;
            sliceReasonVec[s] = reason;
            std::ostringstream rntiList;
            for (size_t i = 0; i < sliceUeVec[s].size(); ++i)
            {
                if (i)
                    rntiList << ";";
                rntiList << sliceUeVec[s][i].first->m_rnti;
            }
            sliceRntiListVec[s] = rntiList.str();

            // Log per-UE demand details for no_demand slices
            if (logThisSlot && reason == "no_demand" && m_ueAllocCsv.is_open())
            {
                for (const auto& ue : sliceUeVec[s])
                {
                    uint32_t bufQueueSize = ue.second;
                    uint32_t prevTbSize = ue.first->m_dlTbSize;
                    m_ueAllocCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                                 << m_currentSlot << "," << m_bwpId << "," << beamIdStr << ","
                                 << s << "," << ue.first->m_rnti << ","
                                 << bufQueueSize << "," << prevTbSize << ","
                                 << 0 << "," << 0 << "," << 0 << "," << 0 << ",0,"
                                 << static_cast<uint32_t>(ue.first->GetDlMcs()) << ","
                                 << static_cast<uint32_t>(ue.first->m_dlRank) << ","
                                 << GetNumRbPerRbg() << ",no_demand\n";
                }
            }
        }

        uint32_t totalAllocatedRbg = std::accumulate(sliceAllocatedVec.begin(),
                                                     sliceAllocatedVec.end(),
                                                     0u);

        for (uint32_t s = 0; s < m_numSlices; ++s)
        {
            NS_ASSERT_MSG(sliceAllocatedVec[s] <= sliceRbgBudget[s],
                          "allocatedRbg=" << sliceAllocatedVec[s]
                                          << " exceeds budgetRbg=" << sliceRbgBudget[s]
                                          << " for slice " << s);

            double rshRealPct = (totalAllocatedRbg > 0)
                                    ? 100.0 * static_cast<double>(sliceAllocatedVec[s]) /
                                          static_cast<double>(totalAllocatedRbg)
                                    : 0.0;
            double budgetUtilizationPct = (sliceRbgBudget[s] > 0)
                                              ? 100.0 * static_cast<double>(sliceAllocatedVec[s]) /
                                                    static_cast<double>(sliceRbgBudget[s])
                                              : 0.0;
            budgetUtilizationPct = std::clamp(budgetUtilizationPct, 0.0, 100.0);
            double unusedBudgetPct = (sliceRbgBudget[s] > 0) ? 100.0 - budgetUtilizationPct : 0.0;
            double targetSharePct = effectiveWeight[s] * 100.0;
            double allocationFidelityPct = std::clamp(100.0 - std::abs(rshRealPct - targetSharePct),
                                                      0.0,
                                                      100.0);

            if (m_sliceAllocationStats.size() == m_numSlices)
            {
                m_sliceAllocationStats[s].samples += 1;
                if (sliceRbgBudget[s] > 0)
                {
                    m_sliceAllocationStats[s].samplesWithBudget += 1;
                    m_sliceAllocationStats[s].budgetUtilizationPctSum += budgetUtilizationPct;
                    m_sliceAllocationStats[s].unusedBudgetPctSum += unusedBudgetPct;
                }
                m_sliceAllocationStats[s].budgetRbgTotal += sliceRbgBudget[s];
                m_sliceAllocationStats[s].allocatedRbgTotal += sliceAllocatedVec[s];
                m_sliceAllocationStats[s].rshRealPctSum += rshRealPct;
                m_sliceAllocationStats[s].allocationFidelityPctSum += allocationFidelityPct;
            }

            // SQR: acumula os contadores de janela por slice — prb_scheduled =
            // RBGs efetivamente agendados (sliceAllocatedVec, já somado por
            // beam/slot); prb_budget = orçamento (cota) atribuído ao slice.
            if (m_windowScheduledRbg.size() == m_numSlices)
            {
                m_windowScheduledRbg[s] += sliceAllocatedVec[s];
                m_windowBudgetRbg[s] += sliceRbgBudget[s];
            }

            if (logThisSlot && m_sliceAllocCsv.is_open())
            {
                uint32_t redistributedIdleRbg =
                    (!sliceHasDemand[s] && sliceRbgBudget[s] == 0 && decisionWeights[s] > 0.0)
                        ? static_cast<uint32_t>(std::round(totalRbgs * decisionWeights[s]))
                        : 0;

                m_sliceAllocCsv << timeMs << "," << s << ","
                                << std::fixed << std::setprecision(4) << decisionWeights[s] << ","
                                << sliceRbgBudget[s] << ","
                                << sliceAllocatedVec[s] << ","
                                << totalAllocatedRbg << ","
                                << std::setprecision(4) << rshRealPct << ","
                                << std::setprecision(4) << budgetUtilizationPct << ","
                                << std::setprecision(4) << unusedBudgetPct << ","
                                << std::setprecision(4) << allocationFidelityPct << ","
                                << sliceUeVec[s].size() << ","
                                << std::setprecision(4) << effectiveWeight[s] << ","
                                << beamIdStr << ","
                                << (sliceHasDemand[s] ? 1 : 0) << ","
                                << sliceReasonVec[s] << ","
                                << redistributedIdleRbg << ","
                                << "\"" << sliceRntiListVec[s] << "\"\n";
            }
        }
    }

    if (m_sliceAllocCsv.is_open())
    {
        m_sliceAllocCsv.flush();
    }
    if (m_unmappedRntiCsv.is_open())
    {
        m_unmappedRntiCsv.flush();
    }
    if (m_ueAllocCsv.is_open())
    {
        m_ueAllocCsv.flush();
    }
    if (m_harqTrackingCsv.is_open())
    {
        m_harqTrackingCsv.flush();
    }
    if (m_activeDlDiagCsv.is_open())
    {
        m_activeDlDiagCsv.flush();
    }

    return symPerBeam;
}

void
RslaqMacScheduler::AssignedDlResources(const UePtrAndBufferReq& ue,
                                        const FTResources& assigned,
                                        const FTResources& totAssigned) const
{
    NS_LOG_FUNCTION(this);
    GetFirst GetUe;
    auto uePtr = dynamic_cast<NrMacSchedulerUeInfoPF*>(GetUe(ue).get());
    NS_ASSERT(uePtr != nullptr);
    uePtr->UpdateDlPFMetric(totAssigned, m_timeWindow);
}

void
RslaqMacScheduler::AssignedUlResources(const UePtrAndBufferReq& ue,
                                        const FTResources& assigned,
                                        const FTResources& totAssigned) const
{
    NS_LOG_FUNCTION(this);
    GetFirst GetUe;
    auto uePtr = dynamic_cast<NrMacSchedulerUeInfoPF*>(GetUe(ue).get());
    NS_ASSERT(uePtr != nullptr);
    uePtr->UpdateUlPFMetric(totAssigned, m_timeWindow);
}

void
RslaqMacScheduler::NotAssignedDlResources(const UePtrAndBufferReq& ue,
                                           const FTResources& notAssigned,
                                           const FTResources& totalAssigned) const
{
    NS_LOG_FUNCTION(this);
    GetFirst GetUe;
    auto uePtr = dynamic_cast<NrMacSchedulerUeInfoPF*>(GetUe(ue).get());
    NS_ASSERT(uePtr != nullptr);
    uePtr->UpdateDlPFMetric(totalAssigned, m_timeWindow);
}

void
RslaqMacScheduler::NotAssignedUlResources(const UePtrAndBufferReq& ue,
                                           const FTResources& notAssigned,
                                           const FTResources& totalAssigned) const
{
    NS_LOG_FUNCTION(this);
    GetFirst GetUe;
    auto uePtr = dynamic_cast<NrMacSchedulerUeInfoPF*>(GetUe(ue).get());
    NS_ASSERT(uePtr != nullptr);
    uePtr->UpdateUlPFMetric(totalAssigned, m_timeWindow);
}

void
RslaqMacScheduler::BeforeDlSched(const UePtrAndBufferReq& ue,
                                   const FTResources& assignableInIteration) const
{
    NS_LOG_FUNCTION(this);
    GetFirst GetUe;
    auto ueInfo = GetUe(ue);
    if (ueInfo->m_dlMcs == 0)
    {
        ueInfo->m_dlMcs = 15;
    }
    auto uePtr = dynamic_cast<NrMacSchedulerUeInfoPF*>(ueInfo.get());
    NS_ASSERT(uePtr != nullptr);
    uePtr->CalculatePotentialTPutDl(assignableInIteration);

    // Track buffer size for external KPM export
    m_lastDlBufferSize[ueInfo->m_rnti] = ue.second;
}

void
RslaqMacScheduler::BeforeUlSched(const UePtrAndBufferReq& ue,
                                  const FTResources& assignableInIteration) const
{
    NS_LOG_FUNCTION(this);
    GetFirst GetUe;
    auto uePtr = dynamic_cast<NrMacSchedulerUeInfoPF*>(GetUe(ue).get());
    NS_ASSERT(uePtr != nullptr);
    uePtr->CalculatePotentialTPutUl(assignableInIteration);
}

// ------------------- SQR: API de leitura (somente leitura) -------------------

std::vector<uint64_t>
RslaqMacScheduler::GetWindowScheduledRbg() const
{
    return m_windowScheduledRbg;
}

std::vector<uint64_t>
RslaqMacScheduler::GetWindowBudgetRbg() const
{
    return m_windowBudgetRbg;
}

void
RslaqMacScheduler::ResetWindowCounters() const
{
    std::fill(m_windowScheduledRbg.begin(), m_windowScheduledRbg.end(), 0);
    std::fill(m_windowBudgetRbg.begin(), m_windowBudgetRbg.end(), 0);
}

uint16_t
RslaqMacScheduler::GetTotalRbgCount() const
{
    return GetBandwidthInRbg();
}

uint64_t
RslaqMacScheduler::GetTotalPrbCount() const
{
    return static_cast<uint64_t>(GetBandwidthInRbg()) * GetNumRbPerRbg();
}

double
RslaqMacScheduler::GetUeDlSinrDb(uint16_t rnti) const
{
    auto it = m_lastDlSinrDb.find(rnti);
    return (it != m_lastDlSinrDb.end()) ? it->second : -100.0;
}

} // namespace ns3
