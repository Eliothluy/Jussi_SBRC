/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * GreenRAN Slice-Aware Meta-Scheduler for 5G-LENA (ns-3)
 *
 * OFDMA MAC scheduler that partitions RBGs across N slices by
 * configurable SlicePolicy weights and applies per-slice intra-slice
 * algorithm (RR, PF, or BCQI).
 *
 * ARCHITECTURE NOTE — UE-group slicing limitation:
 *   The current RNTI → slice mapping implements UE-group slicing.
 *   It does NOT represent per-bearer, per-QoS-flow, per-LCG, or
 *   per-5QI slicing.  A single UE belongs to exactly one slice.
 *   This limitation will be addressed in a future iteration.
 *
 * UL SLICING NOTE:
 *   AssignULRBG is overridden but protected by EnableUlSliceScheduling.
 *   When disabled (default), the base class OFDMA RR scheduler handles
 *   UL without slice discrimination.  When enabled, slice-aware UL RBG
 *   allocation follows the semantics of NrMacSchedulerOfdma::AssignULRBG
 *   — NOT derived from the DL path.
 */

#pragma once

#include "ns3/nr-mac-scheduler-ofdma-rr.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace ns3
{

struct SlicePolicy
{
    uint32_t sliceId{0};
    double weight{0.0};
    double minShare{0.0};
    double maxShare{1.0};
    uint8_t priority{0};
    bool allowBorrowing{true};
};

class MacScheduler : public NrMacSchedulerOfdmaRR
{
  public:
    enum class IntraSliceAlgorithm : uint8_t
    {
        RR = 0,
        PF = 1,
        BCQI = 2
    };

    static TypeId GetTypeId();

    MacScheduler();
    ~MacScheduler() override;

    void SetSliceConfiguration(const std::vector<double>& prbWeights,
                               const std::vector<IntraSliceAlgorithm>& algorithms);

    void SetSlicePolicies(const std::vector<SlicePolicy>& policies,
                          const std::vector<IntraSliceAlgorithm>& algorithms);

    void SetSliceUeMapping(uint32_t numSlices,
                           const std::vector<std::vector<uint32_t>>& sliceUeRnti);

    uint32_t GetNumSlices() const;
    double GetPrbWeight(uint32_t sliceIdx) const;
    IntraSliceAlgorithm GetIntraAlgorithm(uint32_t sliceIdx) const;

    int32_t GetSliceIndexForRnti(uint16_t rnti) const;

    bool ValidateSliceConfig() const;

    void DumpSliceConfiguration() const;
    void SetScenarioName(const std::string& name);
    void SetOutputDir(const std::string& dir);

    uint32_t GetUeDlBufferSize(uint16_t rnti) const;
    uint32_t GetUeUlBufferSize(uint16_t rnti) const;

    void SetCurrentSlot(uint32_t slot);
    void SetBwpId(uint32_t bwpId);

  protected:
    std::shared_ptr<NrMacSchedulerUeInfo> CreateUeRepresentation(
        const NrMacCschedSapProvider::CschedUeConfigReqParameters& params) const override;

    BeamSymbolMap AssignDLRBG(uint32_t symAvail, const ActiveUeMap& activeDl) const override;

    BeamSymbolMap AssignULRBG(uint32_t symAvail, const ActiveUeMap& activeUl) const override;

    void AssignedDlResources(const UePtrAndBufferReq& ue,
                             const FTResources& assigned,
                             const FTResources& totAssigned) const override;

    void AssignedUlResources(const UePtrAndBufferReq& ue,
                             const FTResources& assigned,
                             const FTResources& totAssigned) const override;

    void NotAssignedDlResources(const UePtrAndBufferReq& ue,
                                const FTResources& notAssigned,
                                const FTResources& totalAssigned) const override;

    void NotAssignedUlResources(const UePtrAndBufferReq& ue,
                                const FTResources& notAssigned,
                                const FTResources& totalAssigned) const override;

    void BeforeDlSched(const UePtrAndBufferReq& ue,
                       const FTResources& assignableInIteration) const override;

    void BeforeUlSched(const UePtrAndBufferReq& ue,
                       const FTResources& assignableInIteration) const override;

  private:
    using UeVec = std::vector<UePtrAndBufferReq>;
    using UeVec2D = std::vector<std::vector<UePtrAndBufferReq>>;

    struct SliceBudgetResult
    {
        std::vector<double> effectiveWeight;
        std::vector<uint32_t> budgetRbg;
        std::vector<uint32_t> borrowedInRbg;
        std::vector<uint32_t> borrowedOutRbg;
        std::vector<uint32_t> reservedRbg;
        std::vector<uint32_t> sliceCapRbg;
        uint32_t wastedRbg{0};
        std::vector<bool> sliceHasDemand;
        std::vector<bool> sliceHasActiveUe;
        std::vector<bool> sliceConfigured;
    };

    void SortUeVectorByAlgorithm(UeVec& ueVector,
                                  IntraSliceAlgorithm algo) const;

    void SortUeVectorByAlgorithmUl(UeVec& ueVector,
                                    IntraSliceAlgorithm algo) const;

    UeVec2D GroupActiveUesBySlice(const UeVec& ueVector,
                                   std::vector<uint16_t>& unmappedRntis) const;

    SliceBudgetResult ComputeSliceBudgets(
        uint32_t totalRbgs,
        const UeVec2D& sliceUeVec,
        const std::vector<bool>& sliceHasDemand,
        const std::vector<bool>& sliceConfigured,
        bool applyCap) const;

    void OpenCsvFiles() const;

    bool ShouldLogSlot(uint64_t timeMs) const;
    uint32_t GetCurrentSlot() const;
    uint32_t GetBwpId() const;
    uint64_t GetNextCallId() const;
    uint64_t GetNextUlCallId() const;

    std::set<uint16_t> CollectActiveRntis(const ActiveUeMap& activeMap) const;

    void LogActiveDlDiagnostic(uint64_t callId, uint64_t timeMs, const ActiveUeMap& activeDl) const;
    void LogActiveUlDiagnostic(uint64_t callId, uint64_t timeMs, const ActiveUeMap& activeUl) const;

    uint32_t m_numSlices{0};
    std::vector<double> m_prbWeights;
    std::vector<IntraSliceAlgorithm> m_intraAlgorithms;
    std::vector<std::vector<uint32_t>> m_sliceUeRnti;
    std::vector<SlicePolicy> m_slicePolicies;

    double m_timeWindow{99.0};
    double m_alpha{0.0};

    bool m_fallbackUnmappedToSlice0{false};
    mutable bool m_firstRun{true};
    mutable bool m_firstUlRun{true};

    std::string m_scenarioName;
    std::string m_outputDir;

    bool m_enableUlSliceScheduling{false};

    mutable std::ofstream m_sliceAllocCsv;
    mutable std::ofstream m_unmappedRntiCsv;
    mutable std::ofstream m_ueAllocCsv;
    mutable std::ofstream m_activeDlDiagCsv;

    mutable std::ofstream m_sliceAllocUlCsv;
    mutable std::ofstream m_ueAllocUlCsv;
    mutable std::ofstream m_activeUlDiagCsv;

    mutable uint64_t m_dlSchedCallSeq{0};
    mutable uint64_t m_ulSchedCallSeq{0};
    mutable uint32_t m_currentSlot{0};
    mutable uint32_t m_bwpId{0};

    bool m_logAllMacSlots{false};
    uint32_t m_macLoggingPeriodMs{100};
    bool m_enableDetailedMacLogging{false};
    double m_maxSliceRatio{1.5};
    mutable uint64_t m_lastLoggedMs{0};
    mutable uint64_t m_lastUlLoggedMs{0};

    mutable std::map<uint16_t, uint32_t> m_lastDlBufferSize;
    mutable std::map<uint16_t, uint32_t> m_lastUlBufferSize;

    mutable std::vector<uint64_t> m_cumulSliceAllocRbg;
    mutable std::vector<uint64_t> m_cumulSliceAllocCalls;
    mutable std::vector<uint64_t> m_cumulSliceAllocUlRbg;
    mutable std::vector<uint64_t> m_cumulSliceAllocUlCalls;
};

} // namespace ns3