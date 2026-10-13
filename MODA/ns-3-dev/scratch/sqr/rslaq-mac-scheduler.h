/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * RSLAQ Meta-Scheduler for 5G-LENA (ns-3)
 *
 * Slice-aware OFDMA MAC scheduler with dynamic intra-slice algorithm
 * selection. Partitions PRBs (RBGs) across slices by configurable
 * weights and applies per-slice Round Robin, Proportional Fair, or
 * Best CQI (Maximum Rate) ordering.
 *
 * CÓPIA INTEIRA de scratch/rslaq/rslaq-mac-scheduler.h (original intocado),
 * usada pelo cenário SQR (scratch/sqr/sqr-sim.cc). Únicas diferenças vs o
 * original (marcadas com "SQR:" no corpo): API de leitura adicional para o
 * cenário SQR — contadores de janela prb_scheduled/prb_budget por slice,
 * estimativa de SINR DL por RNTI (inversão do CQI wideband nas tabelas do
 * NrAmc) e contagem de RBGs/PRBs do BWP. Nenhuma mudança na lógica de
 * alocação/agendamento.
 *
 * Diferenças adicionais exigidas pela Fase 1/R27 (12/10/2026, Agente A;
 * ainda restritas a esta CÓPIA — scratch/rslaq permanece intocado):
 *  1. SetSliceConfiguration declarada VIRTUAL (SqrMacScheduler sobrepõe com
 *     semântica de cota absoluta — contrato §3.3, register R27);
 *  2. seção private dos helpers/diagnósticos tornada PROTECTED para a
 *     subclasse SqrMacScheduler reutilizar a maquinaria de log/alocação;
 *  3. contador de janela m_windowDlSlots (slots DL com chamada de scheduling
 *     desde o último reset) + acessor GetWindowDlSlots();
 *  4. sentinela de SNR -999 dB (contrato §4 / register R09) no lugar de
 *     -100 dB, no snapshot e no default de GetUeDlSinrDb.
 */

#pragma once

#include "ns3/nr-mac-scheduler-ofdma-rr.h"

#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace ns3
{

/**
 * @ingroup scheduler
 * @brief RSLAQ Meta-Scheduler: slice-aware OFDMA with dynamic intra-slice policy
 *
 * The scheduler partitions the available RBGs among N configured slices
 * according to a vector of proportional weights that sum to 1.0.
 * Within each slice, UEs are sorted by a selectable intra-slice algorithm:
 *
 *   RR   - Round Robin (fewest assigned RBGs first)
 *   PF   - Proportional Fair (potentialTput / avgTput)
 *   BCQI - Best CQI / Maximum Rate (highest MCS first)
 *
 * Configuration is updated at run-time through SetSliceConfiguration(),
 * typically called every 10 ms (one 5G frame) by the DRL agent callback.
 */
class RslaqMacScheduler : public NrMacSchedulerOfdmaRR
{
  public:
    /**
     * @brief Intra-slice scheduling algorithm identifier
     */
    enum class IntraSliceAlgorithm : uint8_t
    {
        RR = 0,   //!< Round Robin
        PF = 1,   //!< Proportional Fair
        BCQI = 2  //!< Best CQI (Maximum Rate)
    };

    /**
     * @brief Slice PRB/RBG weight policy.
     *
     * STATIC preserves the configured weights. The remaining policies compute
     * the slice weights from active MAC demand at scheduling time. META_RISK_ELASTIC
     * uses configured weights as an offline/metaheuristic prior and corrects them
     * online when buffer pressure indicates slice SLA risk. AQPS computes integer
     * RBG budgets with the QoS-aware three-phase allocation from the adaptive
     * priority scheduling literature.
     */
    enum class SliceWeightPolicy : uint8_t
    {
        STATIC = 0,
        DEMAND_GREEDY = 1,
        SLA_GREEDY = 2,
        LEAST_WASTE = 3,
        RANDOM_VINE = 4,
        META_RISK_ELASTIC = 5,
        AQPS = 6
    };

    static TypeId GetTypeId();

    RslaqMacScheduler();
    ~RslaqMacScheduler() override;

    /**
     * @brief Configure slice PRB weights and intra-slice algorithms
     *
     * @param prbWeights     Vector of N weights (must sum to 1.0)
     * @param algorithms     Vector of N intra-slice algorithms
     *
     * Can be called at any time to update the scheduler policy (e.g., every 10 ms).
     *
     * SQR (R27): virtual para permitir a sobrecarga com semântica de COTA
     * ABSOLUTA em SqrMacScheduler (contrato v1.1 §3.3).
     */
    virtual void SetSliceConfiguration(const std::vector<double>& prbWeights,
                                       const std::vector<IntraSliceAlgorithm>& algorithms);

    /**
     * @brief Map UE RNTIs to slices
     *
     * @param numSlices      Number of slices
     * @param sliceUeRnti    Vector of N vectors; sliceUeRnti[s] lists RNTIs of slice s
     */
    void SetSliceUeMapping(uint32_t numSlices,
                           const std::vector<std::vector<uint32_t>>& sliceUeRnti);

    uint32_t GetNumSlices() const;
    double GetPrbWeight(uint32_t sliceIdx) const;
    IntraSliceAlgorithm GetIntraAlgorithm(uint32_t sliceIdx) const;
    void SetSliceWeightPolicy(SliceWeightPolicy policy);
    SliceWeightPolicy GetSliceWeightPolicy() const;

    /**
     * @brief Get the slice index for a given RNTI
     * @param rnti  UE RNTI
     * @return slice index or -1 if not found
     */
    int32_t GetSliceIndexForRnti(uint16_t rnti) const;

    void DumpSliceConfiguration() const;
    void SetScenarioName(const std::string& name);
    void SetOutputDir(const std::string& dir);

    /**
     * @brief Check buffer status of a UE in m_ueMap
     * @param rnti UE RNTI
     * @return total DL buffer size in bytes, 0 if not found
     */
    uint32_t GetUeDlBufferSize(uint16_t rnti) const;

    // ------------------- SQR: API de leitura (somente leitura) -------------------

    /**
     * @brief RBGs efetivamente agendados por slice na janela corrente
     *
     * Acumulado desde o último ResetWindowCounters() (ou desde o último
     * SetSliceUeMapping). Consumido pelo KPM do sqr-sim a cada período de
     * decisão (prb_scheduled do contrato SQR).
     * @return vetor com um valor por slice
     */
    std::vector<uint64_t> GetWindowScheduledRbg() const;

    /**
     * @brief Orçamento de RBGs (cota) atribuído por slice na janela corrente
     * @return vetor com um valor por slice
     */
    std::vector<uint64_t> GetWindowBudgetRbg() const;

    /**
     * @brief Zera os contadores de janela (chamado pelo sqr-sim a cada período)
     */
    void ResetWindowCounters() const;

    /**
     * @brief Total de RBGs do BWP corrente (conversão percentual → RBG)
     */
    uint16_t GetTotalRbgCount() const;

    /**
     * @brief Total de PRBs do BWP corrente (RBGs × PRBs por RBG) — o "C" do SQR
     */
    uint64_t GetTotalPrbCount() const;

    /**
     * @brief Última estimativa de SINR DL (dB) do UE, derivada do CQI wideband
     *
     * Aproximação documentada: inverte o CQI reportado nas tabelas do NrAmc
     * (SE = GetSpectralEfficiencyForCqi(cqi); SINR = GetSinrFromSpectralEfficiency(SE)).
     * @param rnti UE RNTI
     * @return SINR estimado em dB; -999.0 (sentinela do contrato SQR §4/R09)
     *         se o UE ainda não reportou CQI
     */
    double GetUeDlSinrDb(uint16_t rnti) const;

    /**
     * @brief Slots DL com chamada de scheduling desde o último reset de janela
     *
     * SQR: um contador por janela de decisão (incrementado a cada
     * AssignDLRBG — 1 beam no cenário SQR ⇒ 1 chamada = 1 slot DL com pelo
     * menos 1 UE ativo). Base para prb_idle = totalRbg × dlSlots − Σ
     * prb_scheduled no CSV de janelas.
     */
    uint64_t GetWindowDlSlots() const;

    struct SliceAllocationStats
    {
        /** Aggregate MAC allocation counters exported to summary.csv. */
        uint64_t samples = 0;
        uint64_t samplesWithBudget = 0;
        uint64_t budgetRbgTotal = 0;
        uint64_t allocatedRbgTotal = 0;
        double rshRealPctSum = 0.0;
        double budgetUtilizationPctSum = 0.0;
        double unusedBudgetPctSum = 0.0;
        double allocationFidelityPctSum = 0.0;
    };

    std::vector<SliceAllocationStats> GetSliceAllocationStats() const;

    /**
     * @brief Set current slot number for logging
     * @param slot Current slot number
     */
    void SetCurrentSlot(uint32_t slot);

    /**
     * @brief Set current BWP ID for logging
     * @param bwpId Current BWP ID
     */
    void SetBwpId(uint32_t bwpId);

  protected:
    std::shared_ptr<NrMacSchedulerUeInfo> CreateUeRepresentation(
        const NrMacCschedSapProvider::CschedUeConfigReqParameters& params) const override;

    BeamSymbolMap AssignDLRBG(uint32_t symAvail, const ActiveUeMap& activeDl) const override;

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

  protected:
    // SQR (R27/Fase 1): seção tornada protected para a subclasse
    // SqrMacScheduler (scratch/sqr/sqr-mac-scheduler.{h,cc}) reutilizar a
    // maquinaria de ordenação, logging e contadores de janela. No original
    // (scratch/rslaq) esta seção é private — nada da lógica mudou.
    /**
     * @brief Sort UEs within a slice by the selected intra-slice algorithm
     */
    void SortUeVectorByAlgorithm(std::vector<UePtrAndBufferReq>& ueVector,
                                   IntraSliceAlgorithm algo) const;

    /**
     * @brief Compute per-slice PRB weights for the selected heuristic policy.
     */
    std::vector<double> ComputeDecisionWeights(
        const std::vector<std::vector<UePtrAndBufferReq>>& sliceUeVec,
        const std::vector<bool>& sliceHasDemand,
        uint64_t timeMs) const;

    /**
     * @brief Compute AQPS integer RBG budgets using minimum guarantee,
     * weighted urgency distribution, and priority round-robin leftovers.
     */
    std::vector<uint32_t> ComputeAqpsSliceBudgets(
        const std::vector<std::vector<UePtrAndBufferReq>>& sliceUeVec,
        const std::vector<bool>& sliceHasDemand,
        uint32_t totalRbgs) const;

    void OpenCsvFiles() const;

    /**
     * @brief Determine if current slot should be logged
     * @param timeMs Current time in milliseconds
     * @return true if this slot should be logged
     */
    bool ShouldLogSlot(uint64_t timeMs) const;

    /**
     * @brief Get current slot number for logging
     * @return current slot number
     */
    uint32_t GetCurrentSlot() const;

    /**
     * @brief Get current BWP ID for logging
     * @return current BWP ID
     */
    uint32_t GetBwpId() const;

    /**
     * @brief Increment and get next DL scheduling call ID
     * @return next call ID
     */
    uint64_t GetNextCallId() const;

    uint32_t m_numSlices{0};
    std::vector<double> m_prbWeights;                     // per-slice PRB proportion
    mutable std::vector<double> m_lastDecisionWeights;     // last static/heuristic decision
    std::vector<IntraSliceAlgorithm> m_intraAlgorithms;    // per-slice algorithm
    std::vector<std::vector<uint32_t>> m_sliceUeRnti;      // RNTIs per slice; filled after attach
    SliceWeightPolicy m_sliceWeightPolicy{SliceWeightPolicy::STATIC};

    double m_timeWindow{99.0}; //!< PF averaging time window
    double m_alpha{0.0};       //!< PF fairness index (0 = full fairness)

    bool m_fallbackUnmappedToSlice0{false};
    mutable bool m_firstRun{true};

    std::string m_scenarioName;
    std::string m_outputDir;

    mutable std::ofstream m_sliceAllocCsv;
    mutable std::ofstream m_unmappedRntiCsv;
    mutable std::ofstream m_ueAllocCsv;
    mutable std::ofstream m_harqTrackingCsv;
    mutable std::ofstream m_activeDlDiagCsv;

    // Instrumentation: call counter and tracking for MAC-level diagnostic CSVs.
    mutable uint64_t m_dlSchedCallSeq{0};
    mutable uint32_t m_currentSlot{0};
    mutable uint32_t m_bwpId{0};

    // Logging control attributes
    bool m_logAllMacSlots{false};
    uint32_t m_macLoggingPeriodMs{100};
    bool m_enableDetailedMacLogging{false};
    mutable uint64_t m_lastLoggedMs{0};

    /**
     * @brief Last known DL buffer size per RNTI (updated during scheduling)
     */
    mutable std::map<uint16_t, uint32_t> m_lastDlBufferSize;

    // ------------------- SQR: estado adicional (somente contadores) -------------------
    /** RBGs agendados por slice na janela corrente (reset a cada período SQR). */
    mutable std::vector<uint64_t> m_windowScheduledRbg;
    /** Orçamento de RBGs por slice na janela corrente (reset a cada período SQR). */
    mutable std::vector<uint64_t> m_windowBudgetRbg;
    /**
     * SQR (Fase 1/R27): slots DL com chamada de scheduling na janela corrente
     * (1 chamada de AssignDLRBG = 1 slot DL com ≥1 UE ativo; cenário SQR tem
     * 1 beam). Reset junto com os contadores acima.
     */
    mutable uint64_t m_windowDlSlots{0};
    /** Último SINR DL estimado por RNTI (dB), a partir do CQI wideband. */
    mutable std::map<uint16_t, double> m_lastDlSinrDb;

    mutable std::vector<SliceAllocationStats> m_sliceAllocationStats;

    /**
     * @brief Collect RNTIs present in the activeDl map
     * @param activeDl Active UE map from base scheduler
     * @return Set of RNTIs found in activeDl
     */
    std::set<uint16_t> CollectActiveRntis(const ActiveUeMap& activeDl) const;

    /**
     * @brief Log activeDl diagnostic: which slice RNTIs are present/missing
     * @param callId Scheduling call ID
     * @param timeMs Current time in ms
     * @param activeDl Active UE map
     */
    void LogActiveDlDiagnostic(uint64_t callId, uint64_t timeMs, const ActiveUeMap& activeDl) const;

    /**
     * @brief Log HARQ state for all slice UEs
     * @param callId Scheduling call ID
     * @param timeMs Current time in ms
     */
    void LogHarqState(uint64_t callId, uint64_t timeMs) const;
};

} // namespace ns3
