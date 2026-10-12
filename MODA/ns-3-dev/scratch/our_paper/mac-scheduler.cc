/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * GreenRAN Slice-Aware Meta-Scheduler for 5G-LENA (ns-3)
 *
 * OFDMA MAC scheduler that partitions RBGs across N slices by
 * configurable SlicePolicy weights and applies per-slice intra-slice
 * algorithm (RR, PF, or BCQI).
 *
 * DL slicing: AssignDLRBG — slice-aware RBG allocation in downlink.
 * UL slicing: AssignULRBG — controlled by EnableUlSliceScheduling.
 *   When disabled (default), falls back to NrMacSchedulerOfdma::AssignULRBG.
 *
 * UE-group slicing limitation:
 *   RNTI → slice mapping implements UE-group slicing only.
 *   Not per-bearer, per-QoS-flow, or per-LCG slicing.
 */

#include "mac-scheduler.h"

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

NS_LOG_COMPONENT_DEFINE("MacScheduler");
NS_OBJECT_ENSURE_REGISTERED(MacScheduler);

TypeId
MacScheduler::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::MacScheduler")
            .SetParent<NrMacSchedulerOfdmaRR>()
            .AddConstructor<MacScheduler>()
            .AddAttribute("TimeWindow",
                          "PF averaging time window (in slots)",
                          DoubleValue(99.0),
                          MakeDoubleAccessor(&MacScheduler::m_timeWindow),
                          MakeDoubleChecker<double>(1.0))
            .AddAttribute("FairnessIndex",
                          "PF alpha parameter (0=full fairness, 1=throughput maximization)",
                          DoubleValue(0.0),
                          MakeDoubleAccessor(&MacScheduler::m_alpha),
                          MakeDoubleChecker<double>(0.0, 1.0))
            .AddAttribute("FallbackUnmappedUe",
                          "If true, unmapped UEs are forced into slice 0 (debug only)",
                          BooleanValue(false),
                          MakeBooleanAccessor(&MacScheduler::m_fallbackUnmappedToSlice0),
                          MakeBooleanChecker())
            .AddAttribute("EnableUlSliceScheduling",
                          "If true, apply slice-aware scheduling to UL. "
                          "If false, fall back to base OFDMA RR UL scheduling.",
                          BooleanValue(false),
                          MakeBooleanAccessor(&MacScheduler::m_enableUlSliceScheduling),
                          MakeBooleanChecker())
            .AddAttribute("LogAllMacSlots",
                          "If true, log every MAC scheduling call (use with caution, produces large files)",
                          BooleanValue(false),
                          MakeBooleanAccessor(&MacScheduler::m_logAllMacSlots),
                          MakeBooleanChecker())
            .AddAttribute("MacLoggingPeriodMs",
                          "Period of MAC logging in milliseconds when LogAllMacSlots is false",
                          UintegerValue(100),
                          MakeUintegerAccessor(&MacScheduler::m_macLoggingPeriodMs),
                          MakeUintegerChecker<uint32_t>(1))
            .AddAttribute("EnableDetailedMacLogging",
                          "If true, enable detailed MAC-level logging with call IDs and slot info",
                          BooleanValue(true),
                          MakeBooleanAccessor(&MacScheduler::m_enableDetailedMacLogging),
                          MakeBooleanChecker())
            .AddAttribute("MaxSliceRatio",
                          "Maximum ratio of total RBGs any single slice can consume after borrowing. "
                          "1.0 = strict weight-based caps, 2.0 = generous borrowing.",
                          DoubleValue(1.5),
                          MakeDoubleAccessor(&MacScheduler::m_maxSliceRatio),
                          MakeDoubleChecker<double>(1.0, 5.0));
    return tid;
}

MacScheduler::MacScheduler()
    : NrMacSchedulerOfdmaRR()
{
    NS_LOG_FUNCTION(this);
}

MacScheduler::~MacScheduler()
{
}

bool
MacScheduler::ShouldLogSlot(uint64_t timeMs) const
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
MacScheduler::GetCurrentSlot() const
{
    return m_currentSlot;
}

uint32_t
MacScheduler::GetBwpId() const
{
    return m_bwpId;
}

uint64_t
MacScheduler::GetNextCallId() const
{
    return ++m_dlSchedCallSeq;
}

uint64_t
MacScheduler::GetNextUlCallId() const
{
    return ++m_ulSchedCallSeq;
}

void
MacScheduler::SetCurrentSlot(uint32_t slot)
{
    m_currentSlot = slot;
}

void
MacScheduler::SetBwpId(uint32_t bwpId)
{
    m_bwpId = bwpId;
}

void
MacScheduler::SetSliceConfiguration(const std::vector<double>& prbWeights,
                                     const std::vector<IntraSliceAlgorithm>& algorithms)
{
    NS_LOG_FUNCTION(this);
    std::vector<SlicePolicy> policies;
    for (uint32_t s = 0; s < prbWeights.size(); s++)
    {
        SlicePolicy p;
        p.sliceId = s;
        p.weight = prbWeights[s];
        p.allowBorrowing = true;
        policies.push_back(p);
    }
    SetSlicePolicies(policies, algorithms);
}

void
MacScheduler::SetSlicePolicies(const std::vector<SlicePolicy>& policies,
                                const std::vector<IntraSliceAlgorithm>& algorithms)
{
    NS_LOG_FUNCTION(this);
    NS_ASSERT_MSG(policies.size() == algorithms.size(), "Policies and algorithms must have same size");
    NS_ASSERT_MSG(policies.size() == m_numSlices, "Must match configured number of slices");

    m_slicePolicies = policies;
    m_prbWeights.resize(m_numSlices);
    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        m_prbWeights[s] = policies[s].weight;
    }
    m_intraAlgorithms = algorithms;

    NS_ASSERT_MSG(ValidateSliceConfig(), "Invalid slice configuration in SetSlicePolicies");

    std::ostringstream oss;
    oss << "[MacScheduler] Updated policies:";
    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        oss << " Slice" << s << "(w=" << std::fixed << std::setprecision(4) << m_prbWeights[s];
        oss << ",borrow=" << (m_slicePolicies[s].allowBorrowing ? "Y" : "N");
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

bool
MacScheduler::ValidateSliceConfig() const
{
    if (m_prbWeights.size() != m_numSlices)
    {
        NS_LOG_ERROR("ValidateSliceConfig: weights size " << m_prbWeights.size()
                      << " != numSlices " << m_numSlices);
        return false;
    }
    if (m_intraAlgorithms.size() != m_numSlices)
    {
        NS_LOG_ERROR("ValidateSliceConfig: algorithms size " << m_intraAlgorithms.size()
                      << " != numSlices " << m_numSlices);
        return false;
    }
    if (!m_slicePolicies.empty() && m_slicePolicies.size() != m_numSlices)
    {
        NS_LOG_ERROR("ValidateSliceConfig: policies size " << m_slicePolicies.size()
                      << " != numSlices " << m_numSlices);
        return false;
    }

    double weightSum = 0.0;
    std::set<uint32_t> seenIds;
    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        double w = m_prbWeights[s];
        if (w < 0.0 || std::isnan(w) || std::isinf(w))
        {
            NS_LOG_ERROR("ValidateSliceConfig: slice " << s << " has invalid weight " << w);
            return false;
        }
        weightSum += w;

        if (!m_slicePolicies.empty())
        {
            const auto& p = m_slicePolicies[s];
            if (seenIds.count(p.sliceId))
            {
                NS_LOG_ERROR("ValidateSliceConfig: duplicate sliceId " << p.sliceId);
                return false;
            }
            seenIds.insert(p.sliceId);
            if (p.minShare < 0.0 || std::isnan(p.minShare))
            {
                NS_LOG_ERROR("ValidateSliceConfig: slice " << s << " has invalid minShare " << p.minShare);
                return false;
            }
            if (p.maxShare > 1.0 || std::isnan(p.maxShare))
            {
                NS_LOG_ERROR("ValidateSliceConfig: slice " << s << " has invalid maxShare " << p.maxShare);
                return false;
            }
            if (p.minShare > p.maxShare)
            {
                NS_LOG_ERROR("ValidateSliceConfig: slice " << s << " minShare " << p.minShare
                              << " > maxShare " << p.maxShare);
                return false;
            }
        }
    }
    if (weightSum <= 0.0 && m_numSlices > 0)
    {
        NS_LOG_ERROR("ValidateSliceConfig: total weight sum " << weightSum << " <= 0");
        return false;
    }
    return true;
}

void
MacScheduler::SetSliceUeMapping(uint32_t numSlices,
                                 const std::vector<std::vector<uint32_t>>& sliceUeRnti)
{
    NS_LOG_FUNCTION(this);
    m_numSlices = numSlices;
    m_sliceUeRnti = sliceUeRnti;

    m_prbWeights.resize(m_numSlices, 1.0 / m_numSlices);
    m_intraAlgorithms.resize(m_numSlices, IntraSliceAlgorithm::RR);
    m_cumulSliceAllocRbg.resize(m_numSlices, 0);
    m_cumulSliceAllocCalls.resize(m_numSlices, 0);
    m_cumulSliceAllocUlRbg.resize(m_numSlices, 0);
    m_cumulSliceAllocUlCalls.resize(m_numSlices, 0);

    m_slicePolicies.clear();
    m_slicePolicies.resize(m_numSlices);
    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        m_slicePolicies[s] = {s, m_prbWeights[s], 0.0, 1.0, 0, true};
    }

    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        std::ostringstream oss;
        oss << "[MacScheduler] Slice " << s << " RNTIs:";
        for (uint32_t rnti : m_sliceUeRnti[s])
        {
            oss << " " << rnti;
        }
        NS_LOG_INFO(oss.str());
    }
}

uint32_t
MacScheduler::GetNumSlices() const
{
    return m_numSlices;
}

double
MacScheduler::GetPrbWeight(uint32_t sliceIdx) const
{
    NS_ASSERT(sliceIdx < m_numSlices);
    return m_prbWeights[sliceIdx];
}

MacScheduler::IntraSliceAlgorithm
MacScheduler::GetIntraAlgorithm(uint32_t sliceIdx) const
{
    NS_ASSERT(sliceIdx < m_numSlices);
    return m_intraAlgorithms[sliceIdx];
}

int32_t
MacScheduler::GetSliceIndexForRnti(uint16_t rnti) const
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
MacScheduler::DumpSliceConfiguration() const
{
    std::ostringstream oss;
    oss << "[MacScheduler] DumpSliceConfiguration: numSlices=" << m_numSlices;
    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        oss << " | Slice" << s << " w=" << std::fixed << std::setprecision(4) << m_prbWeights[s];
        oss << " alg=";
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
        if (!m_slicePolicies.empty())
        {
            oss << " borrow=" << (m_slicePolicies[s].allowBorrowing ? "Y" : "N");
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
MacScheduler::SetScenarioName(const std::string& name)
{
    m_scenarioName = name;
}

void
MacScheduler::SetOutputDir(const std::string& dir)
{
    m_outputDir = dir;
}

void
MacScheduler::OpenCsvFiles() const
{
    std::string prefix = m_outputDir;
    if (!prefix.empty() && prefix.back() != '/')
    {
        prefix += '/';
    }
    if (!m_scenarioName.empty())
    {
        prefix += m_scenarioName + "_";
    }

    if (!m_sliceAllocCsv.is_open())
    {
        std::string path = prefix + "slice_alloc.csv";
        m_sliceAllocCsv.open(path, std::ios::out | std::ios::trunc);
        m_sliceAllocCsv << "callId,timeMs,scenario,slot,bwpId,beamId,sliceId,configuredWeight,"
                            "effectiveWeight,activeUes,hasActiveUe,hasDemand,totalRbg,budgetRbg,"
                            "allocatedRbg,borrowedIn,borrowedOut,reserved,sliceCap,wastedRbg,"
                            "cumulAllocRbg,cumulCalls,reason,rntis\n";
        m_sliceAllocCsv.flush();
    }
    if (!m_unmappedRntiCsv.is_open())
    {
        std::string path = prefix + "unmapped_rntis.csv";
        m_unmappedRntiCsv.open(path, std::ios::out | std::ios::trunc);
        m_unmappedRntiCsv << "callId,timeMs,scenario,rnti,reason\n";
        m_unmappedRntiCsv.flush();
    }
    if (!m_ueAllocCsv.is_open())
    {
        std::string path = prefix + "ue_detail.csv";
        m_ueAllocCsv.open(path, std::ios::out | std::ios::trunc);
        m_ueAllocCsv << "callId,timeMs,scenario,slot,bwpId,beamId,sliceId,rnti,bufQueueSize,"
                          "dlTbSizeBefore,hasDemand,demandPassed,rbgAllocated,uniqueRbgs,"
                          "tbSizeBytes,mcs,rank,numRbPerRbg,reason\n";
        m_ueAllocCsv.flush();
    }
    if (!m_activeDlDiagCsv.is_open())
    {
        std::string path = prefix + "active_dl_diag.csv";
        m_activeDlDiagCsv.open(path, std::ios::out | std::ios::trunc);
        m_activeDlDiagCsv << "callId,timeMs,scenario,sliceId,rnti,inActiveDl,dlBufferSize,"
                              "canInsertHarq,harqActiveCount,diagReason\n";
        m_activeDlDiagCsv.flush();
    }

    if (m_enableUlSliceScheduling)
    {
        if (!m_sliceAllocUlCsv.is_open())
        {
            std::string path = prefix + "slice_alloc_ul.csv";
            m_sliceAllocUlCsv.open(path, std::ios::out | std::ios::trunc);
            m_sliceAllocUlCsv << "callId,timeMs,scenario,slot,bwpId,beamId,sliceId,configuredWeight,"
                                  "effectiveWeight,activeUes,hasActiveUe,hasDemand,totalRbg,budgetRbg,"
                                  "allocatedRbg,borrowedIn,borrowedOut,reserved,sliceCap,wastedRbg,"
                                  "cumulAllocRbg,cumulCalls,reason,rntis\n";
            m_sliceAllocUlCsv.flush();
        }
        if (!m_ueAllocUlCsv.is_open())
        {
            std::string path = prefix + "ue_detail_ul.csv";
            m_ueAllocUlCsv.open(path, std::ios::out | std::ios::trunc);
            m_ueAllocUlCsv << "callId,timeMs,scenario,slot,bwpId,beamId,sliceId,rnti,bufQueueSize,"
                                "ulTbSizeBefore,hasDemand,demandPassed,rbgAllocated,uniqueRbgs,"
                                "tbSizeBytes,mcs,rank,numRbPerRbg,reason\n";
            m_ueAllocUlCsv.flush();
        }
        if (!m_activeUlDiagCsv.is_open())
        {
            std::string path = prefix + "active_ul_diag.csv";
            m_activeUlDiagCsv.open(path, std::ios::out | std::ios::trunc);
            m_activeUlDiagCsv << "callId,timeMs,scenario,sliceId,rnti,inActiveUl,ulBufferSize,"
                                  "diagReason\n";
            m_activeUlDiagCsv.flush();
        }
    }
}

std::shared_ptr<NrMacSchedulerUeInfo>
MacScheduler::CreateUeRepresentation(
    const NrMacCschedSapProvider::CschedUeConfigReqParameters& params) const
{
    NS_LOG_FUNCTION(this);
    return std::make_shared<NrMacSchedulerUeInfoPF>(
        static_cast<float>(m_alpha),
        params.m_rnti,
        params.m_beamId,
        std::bind(&MacScheduler::GetNumRbPerRbg, this));
}

void
MacScheduler::SortUeVectorByAlgorithm(std::vector<UePtrAndBufferReq>& ueVector,
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

void
MacScheduler::SortUeVectorByAlgorithmUl(std::vector<UePtrAndBufferReq>& ueVector,
                                          IntraSliceAlgorithm algo) const
{
    switch (algo)
    {
    case IntraSliceAlgorithm::RR:
        std::stable_sort(ueVector.begin(), ueVector.end(),
                         NrMacSchedulerUeInfoRR::CompareUeWeightsUl);
        break;
    case IntraSliceAlgorithm::PF:
        std::stable_sort(ueVector.begin(), ueVector.end(),
                         NrMacSchedulerUeInfoPF::CompareUeWeightsUl);
        break;
    case IntraSliceAlgorithm::BCQI:
        std::stable_sort(ueVector.begin(), ueVector.end(),
                         NrMacSchedulerUeInfoMR::CompareUeWeightsUl);
        break;
    }
}

MacScheduler::UeVec2D
MacScheduler::GroupActiveUesBySlice(const UeVec& ueVector,
                                      std::vector<uint16_t>& unmappedRntis) const
{
    UeVec2D sliceUeVec(m_numSlices);
    unmappedRntis.clear();

    for (const auto& ue : ueVector)
    {
        uint16_t rnti = ue.first->m_rnti;
        int32_t sIdx = GetSliceIndexForRnti(rnti);
        if (sIdx >= 0)
        {
            sliceUeVec[static_cast<uint32_t>(sIdx)].emplace_back(ue);
        }
        else
        {
            NS_LOG_WARN("[MacScheduler] RNTI " << rnti << " not found in any slice (UNMAPPED_RNTI)");
            unmappedRntis.push_back(rnti);
            if (m_fallbackUnmappedToSlice0 && m_numSlices > 0)
            {
                sliceUeVec[0].emplace_back(ue);
            }
        }
    }
    return sliceUeVec;
}

MacScheduler::SliceBudgetResult
MacScheduler::ComputeSliceBudgets(
    uint32_t totalRbgs,
    const UeVec2D& sliceUeVec,
    const std::vector<bool>& sliceHasDemand,
    const std::vector<bool>& sliceConfigured,
    bool applyCap) const
{
    SliceBudgetResult result;
    result.effectiveWeight.resize(m_numSlices, 0.0);
    result.budgetRbg.resize(m_numSlices, 0);
    result.borrowedInRbg.resize(m_numSlices, 0);
    result.borrowedOutRbg.resize(m_numSlices, 0);
    result.reservedRbg.resize(m_numSlices, 0);
    result.sliceCapRbg.resize(m_numSlices, 0);
    result.sliceHasDemand = sliceHasDemand;
    result.sliceHasActiveUe.resize(m_numSlices, false);
    result.sliceConfigured = sliceConfigured;

    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        result.sliceHasActiveUe[s] = !sliceUeVec[s].empty();
    }

    // Step 1: Compute effective weights for slices that participate
    double activeWeightSum = 0.0;
    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        bool participates = sliceHasDemand[s] || result.sliceHasActiveUe[s] || sliceConfigured[s];
        if (participates)
        {
            result.effectiveWeight[s] = m_prbWeights[s];
            activeWeightSum += m_prbWeights[s];
        }
        else
        {
            result.effectiveWeight[s] = 0.0;
        }
    }

    if (activeWeightSum <= 0.0)
    {
        return result;
    }

    // Normalize effective weights to sum to 1.0
    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        result.effectiveWeight[s] /= activeWeightSum;
    }

    // Step 2: Compute base budget using floor, distribute remainder by largest fractional part
    std::vector<double> fractional(m_numSlices, 0.0);
    uint32_t guaranteedRbgTotal = 0;

    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        if (result.effectiveWeight[s] <= 0.0)
        {
            result.budgetRbg[s] = 0;
            fractional[s] = 0.0;
            continue;
        }
        double raw = static_cast<double>(totalRbgs) * result.effectiveWeight[s];
        result.budgetRbg[s] = static_cast<uint32_t>(raw);
        fractional[s] = raw - result.budgetRbg[s];
        guaranteedRbgTotal += result.budgetRbg[s];
    }

    uint32_t remainder = totalRbgs - guaranteedRbgTotal;
    if (remainder > 0 && remainder < m_numSlices)
    {
        std::vector<std::pair<double, uint32_t>> fracOrder;
        for (uint32_t s = 0; s < m_numSlices; s++)
        {
            if (result.effectiveWeight[s] > 0.0)
            {
                fracOrder.push_back({fractional[s], s});
            }
        }
        std::sort(fracOrder.begin(), fracOrder.end(), std::greater<std::pair<double, uint32_t>>());
        for (uint32_t i = 0; i < remainder && i < fracOrder.size(); i++)
        {
            result.budgetRbg[fracOrder[i].second]++;
        }
    }

    // Step 3: Borrowing — redistribute budget from slices without demand
    uint32_t totalBorrowed = 0;
    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        if (!sliceHasDemand[s] && result.budgetRbg[s] > 0)
        {
            bool canBorrow = m_slicePolicies.empty() || m_slicePolicies[s].allowBorrowing;
            if (canBorrow)
            {
                result.borrowedOutRbg[s] = result.budgetRbg[s];
                totalBorrowed += result.budgetRbg[s];
                result.budgetRbg[s] = 0;
            }
            else
            {
                result.reservedRbg[s] = result.budgetRbg[s];
                result.budgetRbg[s] = 0;
            }
        }
    }

    // Step 4: Redistribute borrowed budget to slices with demand, capped per slice
    if (totalBorrowed > 0)
    {
        std::vector<uint32_t> demandSlices;
        double demandWeightSum = 0.0;
        for (uint32_t s = 0; s < m_numSlices; s++)
        {
            if (sliceHasDemand[s])
            {
                demandSlices.push_back(s);
                demandWeightSum += m_prbWeights[s];
            }
        }

        if (demandWeightSum > 0 && !demandSlices.empty())
        {
            if (applyCap)
            {
                // Capped redistribution: no slice exceeds m_maxSliceRatio × weight-based share
                std::vector<uint32_t> sliceCap(m_numSlices, 0);
                for (uint32_t s = 0; s < m_numSlices; s++)
                {
                    if (m_prbWeights[s] > 0.0)
                    {
                        sliceCap[s] = static_cast<uint32_t>(
                            std::ceil(m_prbWeights[s] * static_cast<double>(totalRbgs) * m_maxSliceRatio));
                    }
                }

                uint32_t redistributedSoFar = 0;
                uint32_t cappedAway = 0;
                std::vector<double> demandFrac(m_numSlices, 0.0);

                for (size_t i = 0; i < demandSlices.size(); i++)
                {
                    uint32_t s = demandSlices[i];
                    double share = m_prbWeights[s] / demandWeightSum;
                    double rawExtra = totalBorrowed * share;
                    uint32_t extra = static_cast<uint32_t>(rawExtra);

                    uint32_t maxAllowed = (sliceCap[s] > result.budgetRbg[s])
                                              ? (sliceCap[s] - result.budgetRbg[s])
                                              : 0;
                    if (extra > maxAllowed)
                    {
                        cappedAway += extra - maxAllowed;
                        extra = maxAllowed;
                    }

                    result.borrowedInRbg[s] = extra;
                    result.budgetRbg[s] += extra;
                    demandFrac[s] = rawExtra - static_cast<uint32_t>(rawExtra);
                    redistributedSoFar += extra;
                }

                // Redistribute capped-away RBGs to slices that still have room
                uint32_t toRedistribute = cappedAway;
                for (uint32_t round = 0; round < m_numSlices && toRedistribute > 0; round++)
                {
                    for (size_t i = 0; i < demandSlices.size() && toRedistribute > 0; i++)
                    {
                        uint32_t s = demandSlices[i];
                        uint32_t maxAllowed = (sliceCap[s] > result.budgetRbg[s])
                                                  ? (sliceCap[s] - result.budgetRbg[s])
                                                  : 0;
                        if (maxAllowed > 0)
                        {
                            result.budgetRbg[s]++;
                            result.borrowedInRbg[s]++;
                            toRedistribute--;
                            redistributedSoFar++;
                        }
                    }
                }

                uint32_t remainingBorrowed = totalBorrowed - redistributedSoFar;
                if (remainingBorrowed > 0 && remainingBorrowed <= m_numSlices)
                {
                    std::vector<std::pair<double, uint32_t>> demandFracOrder;
                    for (uint32_t s : demandSlices)
                    {
                        uint32_t maxAllowed = (sliceCap[s] > result.budgetRbg[s])
                                                  ? (sliceCap[s] - result.budgetRbg[s])
                                                  : 0;
                        if (maxAllowed > 0)
                        {
                            demandFracOrder.push_back({demandFrac[s], s});
                        }
                    }
                    std::sort(demandFracOrder.begin(), demandFracOrder.end(),
                              std::greater<std::pair<double, uint32_t>>());
                    for (uint32_t i = 0; i < remainingBorrowed && i < demandFracOrder.size(); i++)
                    {
                        uint32_t s = demandFracOrder[i].second;
                        result.budgetRbg[s]++;
                        result.borrowedInRbg[s]++;
                    }
                }
            }
            else
            {
                // Uncapped redistribution (DL direction): give all borrowed to demanding slices
                std::vector<double> demandFrac(m_numSlices, 0.0);
                uint32_t redistributedSoFar = 0;

                for (size_t i = 0; i < demandSlices.size(); i++)
                {
                    uint32_t s = demandSlices[i];
                    double share = m_prbWeights[s] / demandWeightSum;
                    double rawExtra = totalBorrowed * share;
                    uint32_t extra = static_cast<uint32_t>(rawExtra);
                    result.borrowedInRbg[s] = extra;
                    result.budgetRbg[s] += extra;
                    demandFrac[s] = rawExtra - extra;
                    redistributedSoFar += extra;
                }

                uint32_t remainingBorrowed = totalBorrowed - redistributedSoFar;
                if (remainingBorrowed > 0 && remainingBorrowed < m_numSlices)
                {
                    std::vector<std::pair<double, uint32_t>> demandFracOrder;
                    for (uint32_t s : demandSlices)
                    {
                        demandFracOrder.push_back({demandFrac[s], s});
                    }
                    std::sort(demandFracOrder.begin(), demandFracOrder.end(),
                              std::greater<std::pair<double, uint32_t>>());
                    for (uint32_t i = 0; i < remainingBorrowed && i < demandFracOrder.size(); i++)
                    {
                        uint32_t s = demandFracOrder[i].second;
                        result.budgetRbg[s]++;
                        result.borrowedInRbg[s]++;
                    }
                }
            }
        }
    }

    // Compute per-slice caps and wasted RBGs
    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        if (m_prbWeights[s] > 0.0)
        {
            result.sliceCapRbg[s] = static_cast<uint32_t>(
                std::ceil(m_prbWeights[s] * static_cast<double>(totalRbgs) * m_maxSliceRatio));
        }
    }

    uint32_t totalBudget = 0;
    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        totalBudget += result.budgetRbg[s];
    }
    NS_ASSERT_MSG(totalBudget <= totalRbgs,
                   "Total budget " << totalBudget << " exceeds total RBGs " << totalRbgs);

    result.wastedRbg = totalRbgs - totalBudget - result.reservedRbg[0]; // approximate
    uint32_t totalAllocAndReserve = totalBudget;
    for (uint32_t s = 0; s < m_numSlices; s++)
    {
        totalAllocAndReserve += result.reservedRbg[s];
    }
    result.wastedRbg = (totalAllocAndReserve <= totalRbgs) ? (totalRbgs - totalAllocAndReserve) : 0;

    return result;
}

std::set<uint16_t>
MacScheduler::CollectActiveRntis(const ActiveUeMap& activeMap) const
{
    std::set<uint16_t> rntis;
    for (const auto& beam : activeMap)
    {
        for (const auto& ue : beam.second)
        {
            rntis.insert(ue.first->m_rnti);
        }
    }
    return rntis;
}

uint32_t
MacScheduler::GetUeDlBufferSize(uint16_t rnti) const
{
    auto it = m_lastDlBufferSize.find(rnti);
    return (it != m_lastDlBufferSize.end()) ? it->second : 0;
}

uint32_t
MacScheduler::GetUeUlBufferSize(uint16_t rnti) const
{
    auto it = m_lastUlBufferSize.find(rnti);
    return (it != m_lastUlBufferSize.end()) ? it->second : 0;
}

void
MacScheduler::LogActiveDlDiagnostic(uint64_t callId, uint64_t timeMs,
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

            std::string reason = inActiveDl ? "in_active_dl" : "not_in_active_dl";

            m_activeDlDiagCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                              << s << "," << r << "," << (inActiveDl ? 1 : 0) << ","
                              << bufSize << "," << 0 << ","
                              << 0 << "," << reason << "\n";
        }
    }
    m_activeDlDiagCsv.flush();
}

void
MacScheduler::LogActiveUlDiagnostic(uint64_t callId, uint64_t timeMs,
                                      const ActiveUeMap& activeUl) const
{
    if (!m_activeUlDiagCsv.is_open() || !m_enableUlSliceScheduling)
    {
        return;
    }

    std::set<uint16_t> activeRntis = CollectActiveRntis(activeUl);

    std::map<uint16_t, uint32_t> activeRntiBufSize;
    for (const auto& beam : activeUl)
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
            bool inActiveUl = activeRntis.count(r) > 0;
            uint32_t bufSize = inActiveUl ? activeRntiBufSize[r] : 0;

            std::string reason = inActiveUl ? "in_active_ul" : "not_in_active_ul";

            m_activeUlDiagCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                              << s << "," << r << "," << (inActiveUl ? 1 : 0) << ","
                              << bufSize << "," << 0 << ","
                              << 0 << "," << reason << "\n";
        }
    }
    m_activeUlDiagCsv.flush();
}

NrMacSchedulerNs3::BeamSymbolMap
MacScheduler::AssignDLRBG(uint32_t symAvail, const ActiveUeMap& activeDl) const
{
    NS_LOG_FUNCTION(this);
    NS_LOG_DEBUG("AssignDLRBG: #beams=" << activeDl.size() << " symAvail=" << symAvail);

    if (m_firstRun)
    {
        m_firstRun = false;
        OpenCsvFiles();
        DumpSliceConfiguration();
        NS_ASSERT_MSG(ValidateSliceConfig(), "Invalid slice configuration at first DL scheduling call");
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
    }

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

        // Use GroupActiveUesBySlice helper
        std::vector<uint16_t> unmappedRntis;
        auto sliceUeVec = GroupActiveUesBySlice(GetUeVector(el), unmappedRntis);

        // Log unmapped RNTIs
        for (uint16_t rnti : unmappedRntis)
        {
            NS_LOG_WARN("[MacScheduler] UNMAPPED_RNTI: RNTI " << rnti << " not in any slice");
            if (m_unmappedRntiCsv.is_open())
            {
                m_unmappedRntiCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                                  << rnti << ",UNMAPPED_RNTI\n";
            }
        }

        // Detect demand per slice (DL: bufQueueSize > 0 || m_dlTbSize > 0)
        std::vector<bool> sliceHasDemand(m_numSlices, false);
        std::vector<bool> sliceConfigured(m_numSlices, false);
        for (uint32_t s = 0; s < m_numSlices; s++)
        {
            sliceConfigured[s] = !m_sliceUeRnti[s].empty();
            for (const auto& ue : sliceUeVec[s])
            {
                uint32_t bufQueueSize = ue.second;
                if (bufQueueSize > 0 || ue.first->m_dlTbSize > 0)
                {
                    sliceHasDemand[s] = true;
                    break;
                }
            }
        }

        // Use ComputeSliceBudgets helper
        SliceBudgetResult budget = ComputeSliceBudgets(totalRbgs, sliceUeVec, sliceHasDemand, sliceConfigured, false);

        // Call BeforeDlSched for all UEs across all slices
        for (uint32_t s = 0; s < m_numSlices; s++)
        {
            for (auto& ue : sliceUeVec[s])
            {
                BeforeDlSched(ue, FTResources(totalRbgs, beamSym));
            }
        }

        // Slice-by-slice allocation
        for (uint32_t s = 0; s < m_numSlices; s++)
        {
            std::string reason;
            uint32_t sliceAllocated = 0;
            bool sliceHasActiveUe = !sliceUeVec[s].empty();

            if (sliceUeVec[s].empty() && sliceConfigured[s])
            {
                reason = "NO_ACTIVE_UE";
            }
            else if (sliceUeVec[s].empty())
            {
                reason = "UNCONFIGURED_SLICE";
            }
            else if (!sliceHasDemand[s])
            {
                reason = "ACTIVE_UE_NO_DEMAND";
            }
            else if (budget.budgetRbg[s] == 0)
            {
                reason = "ZERO_BUDGET";
            }
            else
            {
                SortUeVectorByAlgorithm(sliceUeVec[s], m_intraAlgorithms[s]);

                uint32_t iteration = 0;
                const uint32_t maxIterations = budget.budgetRbg[s] * sliceUeVec[s].size() + sliceUeVec[s].size();

                while (sliceAllocated < budget.budgetRbg[s] && iteration < maxIterations)
                {
                    iteration++;
                    bool anyUeNeeded = false;

                    for (auto& ue : sliceUeVec[s])
                    {
                        if (sliceAllocated >= budget.budgetRbg[s])
                        {
                            break;
                        }

                        uint32_t bufQueueSize = ue.second;
                        uint32_t prevTbSize = ue.first->m_dlTbSize;
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
                                         << (sliceHasDemand[s] ? 1 : 0) << "," << (demandPassed ? 1 : 0) << ","
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
                    reason = "ALL_UE_DEMAND_SATISFIED";
                }
                else if (sliceAllocated < budget.budgetRbg[s])
                {
                    reason = "PARTIAL_ALLOC";
                }
                else
                {
                    reason = "FULL_ALLOC";
                }
            }

            m_cumulSliceAllocRbg[s] += sliceAllocated;
            m_cumulSliceAllocCalls[s]++;

            if (logThisSlot && m_sliceAllocCsv.is_open())
            {
                std::ostringstream rntiList;
                for (size_t i = 0; i < sliceUeVec[s].size(); ++i)
                {
                    if (i)
                        rntiList << ";";
                    rntiList << sliceUeVec[s][i].first->m_rnti;
                }
                m_sliceAllocCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                                << m_currentSlot << "," << m_bwpId << "," << beamIdStr << ","
                                << s << ","
                                << std::fixed << std::setprecision(4) << m_prbWeights[s] << ","
                                << std::setprecision(4) << budget.effectiveWeight[s] << ","
                                << sliceUeVec[s].size() << ","
                                << (sliceHasActiveUe ? 1 : 0) << ","
                                << (sliceHasDemand[s] ? 1 : 0) << ","
                                << totalRbgs << ","
                                << budget.budgetRbg[s] << ","
                                << sliceAllocated << ","
                                << budget.borrowedInRbg[s] << ","
                                << budget.borrowedOutRbg[s] << ","
                                << budget.reservedRbg[s] << ","
                                << budget.sliceCapRbg[s] << ","
                                << budget.wastedRbg << ","
                                << m_cumulSliceAllocRbg[s] << ","
                                << m_cumulSliceAllocCalls[s] << ","
                                << reason << ","
                                << "\"" << rntiList.str() << "\"\n";
            }

            if (logThisSlot && (reason == "ACTIVE_UE_NO_DEMAND" || reason == "NO_ACTIVE_UE") && m_ueAllocCsv.is_open())
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
                                 << GetNumRbPerRbg() << "," << reason << "\n";
                }
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
    if (m_activeDlDiagCsv.is_open())
    {
        m_activeDlDiagCsv.flush();
    }

    return symPerBeam;
}

NrMacSchedulerNs3::BeamSymbolMap
MacScheduler::AssignULRBG(uint32_t symAvail, const ActiveUeMap& activeUl) const
{
    if (!m_enableUlSliceScheduling)
    {
        // Fallback: delegate to base class OFDMA scheduler.
        // UL slicing is not enabled; the base OFDMA RR scheduler
        // allocates RBGs without slice discrimination.
        // See NrMacSchedulerOfdma::AssignULRBG for original semantics.
        return NrMacSchedulerOfdmaRR::AssignULRBG(symAvail, activeUl);
    }

    NS_LOG_FUNCTION(this);
    NS_LOG_DEBUG("AssignULRBG (slice-aware): #beams=" << activeUl.size() << " symAvail=" << symAvail);

    // Open UL CSV files on first UL call (may be after DL first run)
    if (m_firstUlRun)
    {
        m_firstUlRun = false;
        // UL CSV files are opened inside OpenCsvFiles if EnableUlSliceScheduling is true.
        // But OpenCsvFiles is called by AssignDLRBG on first run. If UL runs before DL,
        // open them explicitly here. Since m_enableUlSliceScheduling is true here,
        // ensure the files are open.
        if (!m_sliceAllocUlCsv.is_open())
        {
            // Force open UL files by calling OpenCsvFiles if not opened yet.
            // This also triggers DL file opening if not done.
            if (m_firstRun)
            {
                m_firstRun = false;
                OpenCsvFiles();
                DumpSliceConfiguration();
            }
            else
            {
                // DL already ran but UL files not opened - open them now
                std::string prefix = m_outputDir;
                if (!prefix.empty() && prefix.back() != '/')
                {
                    prefix += '/';
                }
                if (!m_scenarioName.empty())
                {
                    prefix += m_scenarioName + "_";
                }
                if (!m_sliceAllocUlCsv.is_open())
                {
                    m_sliceAllocUlCsv.open(prefix + "slice_alloc_ul.csv", std::ios::out | std::ios::trunc);
                    m_sliceAllocUlCsv << "callId,timeMs,scenario,slot,bwpId,beamId,sliceId,configuredWeight,"
                                          "effectiveWeight,activeUes,hasActiveUe,hasDemand,totalRbg,budgetRbg,"
                                          "allocatedRbg,borrowedIn,borrowedOut,reserved,sliceCap,wastedRbg,"
                                          "cumulAllocRbg,cumulCalls,reason,rntis\n";
                    m_sliceAllocUlCsv.flush();
                }
                if (!m_ueAllocUlCsv.is_open())
                {
                    m_ueAllocUlCsv.open(prefix + "ue_detail_ul.csv", std::ios::out | std::ios::trunc);
                    m_ueAllocUlCsv << "callId,timeMs,scenario,slot,bwpId,beamId,sliceId,rnti,bufQueueSize,"
                                        "ulTbSizeBefore,hasDemand,demandPassed,rbgAllocated,uniqueRbgs,"
                                        "tbSizeBytes,mcs,rank,numRbPerRbg,reason\n";
                    m_ueAllocUlCsv.flush();
                }
                if (!m_activeUlDiagCsv.is_open())
                {
                    m_activeUlDiagCsv.open(prefix + "active_ul_diag.csv", std::ios::out | std::ios::trunc);
                    m_activeUlDiagCsv << "callId,timeMs,scenario,sliceId,rnti,inActiveUl,ulBufferSize,"
                                          "diagReason\n";
                    m_activeUlDiagCsv.flush();
                }
            }
        }
        NS_ASSERT_MSG(ValidateSliceConfig(), "Invalid slice configuration at first UL scheduling call");
    }

    GetFirst GetBeamId;
    GetSecond GetUeVector;
    BeamSymbolMap symPerBeam = GetSymPerBeam(symAvail, activeUl);

    uint64_t timeMs = Simulator::Now().GetMilliSeconds();
    uint64_t callId = GetNextUlCallId();

    bool logThisSlot = false;
    if (m_logAllMacSlots)
    {
        logThisSlot = true;
    }
    else if (m_enableDetailedMacLogging)
    {
        if (timeMs - m_lastUlLoggedMs >= m_macLoggingPeriodMs)
        {
            m_lastUlLoggedMs = timeMs;
            logThisSlot = true;
        }
    }
    else
    {
        logThisSlot = (timeMs <= 200) || ((timeMs + 37) % 97 < 2);
    }

    if (logThisSlot && m_enableUlSliceScheduling)
    {
        LogActiveUlDiagnostic(callId, timeMs, activeUl);
    }

    auto beamIdToStr = [](const BeamId& bid) -> std::string {
        std::ostringstream oss;
        oss << bid.GetSector() << "_" << bid.GetElevation();
        return oss.str();
    };

    for (const auto& el : activeUl)
    {
        BeamId beamId = GetBeamId(el);
        uint32_t beamSym = symPerBeam.at(beamId);

        // UL uses GetUlBitmask() — this is a critical difference from DL which uses GetDlNotchedRbgMask()
        const std::vector<bool> ulBitmask = GetUlBitmask();
        std::string beamIdStr = beamIdToStr(beamId);

        std::vector<uint32_t> availableRbgIds;
        if (!ulBitmask.empty())
        {
            for (uint32_t i = 0; i < ulBitmask.size(); ++i)
            {
                if (ulBitmask[i])
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

        // Track which RBGs are used — similar to DL rbgRealUsed approach
        std::vector<bool> rbgRealUsed(GetBandwidthInRbg(), false);
        if (!ulBitmask.empty())
        {
            for (uint32_t i = 0; i < ulBitmask.size(); ++i)
            {
                if (!ulBitmask[i])
                {
                    rbgRealUsed[i] = true;
                }
            }
        }

        FTResources assigned(0, 0);

        // Use GroupActiveUesBySlice helper
        std::vector<uint16_t> unmappedRntis;
        auto sliceUeVec = GroupActiveUesBySlice(GetUeVector(el), unmappedRntis);

        // Log unmapped RNTIs
        for (uint16_t rnti : unmappedRntis)
        {
            NS_LOG_WARN("[MacScheduler] UL UNMAPPED_RNTI: RNTI " << rnti << " not in any slice");
            if (m_unmappedRntiCsv.is_open())
            {
                m_unmappedRntiCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                                  << rnti << ",UL_UNMAPPED_RNTI\n";
            }
        }

        // Detect demand per slice (UL: bufQueueSize > 0 || m_ulTbSize > 0)
        std::vector<bool> sliceHasDemand(m_numSlices, false);
        std::vector<bool> sliceConfigured(m_numSlices, false);
        for (uint32_t s = 0; s < m_numSlices; s++)
        {
            sliceConfigured[s] = !m_sliceUeRnti[s].empty();
            for (const auto& ue : sliceUeVec[s])
            {
                uint32_t bufQueueSize = ue.second;
                if (bufQueueSize > 0 || ue.first->m_ulTbSize > 0)
                {
                    sliceHasDemand[s] = true;
                    break;
                }
            }
        }

        // Use ComputeSliceBudgets helper
        SliceBudgetResult budget = ComputeSliceBudgets(totalRbgs, sliceUeVec, sliceHasDemand, sliceConfigured, true);

        // Call BeforeUlSched for all UEs across all slices
        // CRITICAL: UL uses FTResources(beamSym * beamSym, beamSym), not (beamSym, beamSym) like DL
        for (uint32_t s = 0; s < m_numSlices; s++)
        {
            for (auto& ue : sliceUeVec[s])
            {
                BeforeUlSched(ue, FTResources(beamSym * beamSym, beamSym));
            }
        }

        // Slice-by-slice UL allocation
        for (uint32_t s = 0; s < m_numSlices; s++)
        {
            std::string reason;
            uint32_t sliceAllocated = 0;
            bool sliceHasActiveUe = !sliceUeVec[s].empty();

            if (sliceUeVec[s].empty() && sliceConfigured[s])
            {
                reason = "NO_ACTIVE_UE";
            }
            else if (sliceUeVec[s].empty())
            {
                reason = "UNCONFIGURED_SLICE";
            }
            else if (!sliceHasDemand[s])
            {
                reason = "ACTIVE_UE_NO_DEMAND";
            }
            else if (budget.budgetRbg[s] == 0)
            {
                reason = "ZERO_BUDGET";
            }
            else
            {
                // UL uses UL comparators, not DL
                SortUeVectorByAlgorithmUl(sliceUeVec[s], m_intraAlgorithms[s]);

                uint32_t iteration = 0;
                const uint32_t maxIterations = budget.budgetRbg[s] * sliceUeVec[s].size() + sliceUeVec[s].size();

                while (sliceAllocated < budget.budgetRbg[s] && iteration < maxIterations)
                {
                    iteration++;
                    bool anyUeNeeded = false;

                    for (auto& ue : sliceUeVec[s])
                    {
                        if (sliceAllocated >= budget.budgetRbg[s])
                        {
                            break;
                        }

                        uint32_t bufQueueSize = ue.second;
                        uint32_t prevTbSize = ue.first->m_ulTbSize;
                        // UL demand detection: consistent with base class flexibility
                        bool demandPassed = (bufQueueSize > 0 || prevTbSize > 0);

                        if (logThisSlot && m_ueAllocUlCsv.is_open())
                        {
                            uint32_t curUnique = 0;
                            if (!ue.first->m_ulRBG.empty())
                            {
                                std::set<uint16_t> uniq(ue.first->m_ulRBG.begin(),
                                                        ue.first->m_ulRBG.end());
                                curUnique = static_cast<uint32_t>(uniq.size());
                            }
                            m_ueAllocUlCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                                            << m_currentSlot << "," << m_bwpId << "," << beamIdStr << ","
                                            << s << "," << ue.first->m_rnti << ","
                                            << bufQueueSize << "," << prevTbSize << ","
                                            << (sliceHasDemand[s] ? 1 : 0) << "," << (demandPassed ? 1 : 0) << ","
                                            << 0 << "," << curUnique << "," << 0 << ","
                                            << static_cast<uint32_t>(ue.first->m_ulMcs) << ","
                                            << static_cast<uint32_t>(ue.first->m_ulRank) << ","
                                            << GetNumRbPerRbg() << ",pre_allocation\n";
                        }

                        if (!demandPassed)
                        {
                            continue;
                        }
                        anyUeNeeded = true;

                        // Find next available RBG (same approach as our DL)
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

                        // UL: fill m_ulRBG and m_ulSym — same pattern as our DL
                        auto& ulRbg = ue.first->m_ulRBG;
                        auto existingRbgs = ulRbg.size();
                        ulRbg.resize(ulRbg.size() + beamSym);
                        std::fill(ulRbg.begin() + existingRbgs, ulRbg.end(), actualRbgId);

                        auto& ulSym = ue.first->m_ulSym;
                        auto existingSyms = ulSym.size();
                        ulSym.resize(ulSym.size() + beamSym);
                        std::iota(ulSym.begin() + existingSyms, ulSym.end(), 0);

                        assigned.m_rbg += 1;
                        assigned.m_sym = beamSym;

                        sliceAllocated++;

                        // UL calls AssignedUlResources / NotAssignedUlResources
                        AssignedUlResources(ue, FTResources(1, beamSym), assigned);

                        for (auto& otherUe : sliceUeVec[s])
                        {
                            if (ue.first->m_rnti != otherUe.first->m_rnti)
                            {
                                NotAssignedUlResources(otherUe, FTResources(1, beamSym), assigned);
                            }
                        }

                        if (logThisSlot && m_ueAllocUlCsv.is_open())
                        {
                            std::set<uint16_t> uniq(ue.first->m_ulRBG.begin(),
                                                    ue.first->m_ulRBG.end());
                            uint32_t tbSize = ue.first->m_ulTbSize;
                            m_ueAllocUlCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                                            << m_currentSlot << "," << m_bwpId << "," << beamIdStr << ","
                                            << s << "," << ue.first->m_rnti << ","
                                            << bufQueueSize << "," << ue.first->m_ulTbSize << ","
                                            << 1 << "," << 1 << "," << 1 << ","
                                            << static_cast<uint32_t>(uniq.size()) << "," << tbSize << ","
                                            << static_cast<uint32_t>(ue.first->m_ulMcs) << ","
                                            << static_cast<uint32_t>(ue.first->m_ulRank) << ","
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
                    reason = "ALL_UE_DEMAND_SATISFIED";
                }
                else if (sliceAllocated < budget.budgetRbg[s])
                {
                    reason = "PARTIAL_ALLOC";
                }
                else
                {
                    reason = "FULL_ALLOC";
                }
            }

            m_cumulSliceAllocUlRbg[s] += sliceAllocated;
            m_cumulSliceAllocUlCalls[s]++;

            if (logThisSlot && m_sliceAllocUlCsv.is_open())
            {
                std::ostringstream rntiList;
                for (size_t i = 0; i < sliceUeVec[s].size(); ++i)
                {
                    if (i)
                        rntiList << ";";
                    rntiList << sliceUeVec[s][i].first->m_rnti;
                }
                m_sliceAllocUlCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                                    << m_currentSlot << "," << m_bwpId << "," << beamIdStr << ","
                                    << s << ","
                                    << std::fixed << std::setprecision(4) << m_prbWeights[s] << ","
                                    << std::setprecision(4) << budget.effectiveWeight[s] << ","
                                    << sliceUeVec[s].size() << ","
                                    << (sliceHasActiveUe ? 1 : 0) << ","
                                    << (sliceHasDemand[s] ? 1 : 0) << ","
                                    << totalRbgs << ","
                                    << budget.budgetRbg[s] << ","
                                    << sliceAllocated << ","
                                    << budget.borrowedInRbg[s] << ","
                                    << budget.borrowedOutRbg[s] << ","
                                    << budget.reservedRbg[s] << ","
                                    << budget.sliceCapRbg[s] << ","
                                    << budget.wastedRbg << ","
                                    << m_cumulSliceAllocUlRbg[s] << ","
                                    << m_cumulSliceAllocUlCalls[s] << ","
                                    << reason << ","
                                    << "\"" << rntiList.str() << "\"\n";
            }

            if (logThisSlot && (reason == "ACTIVE_UE_NO_DEMAND" || reason == "NO_ACTIVE_UE") && m_ueAllocUlCsv.is_open())
            {
                for (const auto& ue : sliceUeVec[s])
                {
                    uint32_t bufQueueSize = ue.second;
                    uint32_t prevTbSize = ue.first->m_ulTbSize;
                    m_ueAllocUlCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                                    << m_currentSlot << "," << m_bwpId << "," << beamIdStr << ","
                                    << s << "," << ue.first->m_rnti << ","
                                    << bufQueueSize << "," << prevTbSize << ","
                                    << 0 << "," << 0 << "," << 0 << "," << 0 << ",0,"
                                    << static_cast<uint32_t>(ue.first->m_ulMcs) << ","
                                    << static_cast<uint32_t>(ue.first->m_ulRank) << ","
                                    << GetNumRbPerRbg() << "," << reason << "\n";
                }
            }
        }
    }

    if (m_sliceAllocUlCsv.is_open())
    {
        m_sliceAllocUlCsv.flush();
    }
    if (m_ueAllocUlCsv.is_open())
    {
        m_ueAllocUlCsv.flush();
    }
    if (m_activeUlDiagCsv.is_open())
    {
        m_activeUlDiagCsv.flush();
    }

    return symPerBeam;
}

void
MacScheduler::AssignedDlResources(const UePtrAndBufferReq& ue,
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
MacScheduler::AssignedUlResources(const UePtrAndBufferReq& ue,
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
MacScheduler::NotAssignedDlResources(const UePtrAndBufferReq& ue,
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
MacScheduler::NotAssignedUlResources(const UePtrAndBufferReq& ue,
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
MacScheduler::BeforeDlSched(const UePtrAndBufferReq& ue,
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

    m_lastDlBufferSize[ueInfo->m_rnti] = ue.second;
}

void
MacScheduler::BeforeUlSched(const UePtrAndBufferReq& ue,
                              const FTResources& assignableInIteration) const
{
    NS_LOG_FUNCTION(this);
    GetFirst GetUe;
    auto ueInfo = GetUe(ue);
    auto uePtr = dynamic_cast<NrMacSchedulerUeInfoPF*>(ueInfo.get());
    NS_ASSERT(uePtr != nullptr);
    uePtr->CalculatePotentialTPutUl(assignableInIteration);

    m_lastUlBufferSize[ueInfo->m_rnti] = ue.second;
}

} // namespace ns3