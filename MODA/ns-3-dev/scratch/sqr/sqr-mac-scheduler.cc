/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * SqrMacScheduler — implementação (cota absoluta de RBGs por slice).
 *
 * Ver sqr-mac-scheduler.h para a motivação (register R27 / contrato v1.1
 * §3.3) e a semântica congelada. Este arquivo NÃO altera nenhum caminho de
 * scratch/rslaq nem de contrib/nr: reutiliza por herança a maquinaria da
 * cópia scratch/sqr/rslaq-mac-scheduler (helpers de log, ordenação
 * RR/PF/BCQI, contadores de janela) e substitui apenas o loop de partição
 * de RBGs (AssignDLRBG).
 */

#include "sqr-mac-scheduler.h"

#include "ns3/log.h"
#include "ns3/simulator.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("SqrMacScheduler");
NS_OBJECT_ENSURE_REGISTERED(SqrMacScheduler);

TypeId
SqrMacScheduler::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::SqrMacScheduler")
            .SetParent<RslaqMacScheduler>()
            .AddConstructor<SqrMacScheduler>()
            .SetGroupName("Nr");
    return tid;
}

SqrMacScheduler::SqrMacScheduler()
    : RslaqMacScheduler()
{
    NS_LOG_FUNCTION(this);
}

SqrMacScheduler::~SqrMacScheduler()
{
}

// ---------------------------------------------------------------------------
// Conversão determinística % PRB → cota inteira de RBGs (R27)
// ---------------------------------------------------------------------------

std::vector<uint32_t>
SqrMacScheduler::ComputeRbgQuotas(const std::vector<double>& pctPrb,
                                  uint32_t totalRbg,
                                  uint32_t numSlices)
{
    /*
     * Regra determinística documentada (contrato v1.1 §3.3 + register R27):
     *   raw_i  = pct_i/100 × totalRbg          (SEM renormalização)
     *   alvo   = round(Σ raw), limitado a totalRbg
     *   quota_i = floor(raw_i); excedente (alvo − Σ floor) distribuído 1 RBG
     *             por slice pelos maiores restos fracionários, desempate pelo
     *             MENOR índice de slice.
     * Garantias: 0 ≤ Σ quota ≤ totalRbg; Σ quota ≥ floor(Σ raw) (folga de
     * arredondamento < L RBGs — critério (ii) do smoke: Σ prb_scheduled ≤
     * 0,5·C + L com Σa = 0,5·C). Σ pct > 100 (violação de Σa ≤ C pelo agente)
     * é coerçado pelo teto do alvo — sem redistribuição além do teto.
     */
    const uint32_t n = std::max(numSlices, static_cast<uint32_t>(pctPrb.size()));
    std::vector<uint32_t> quotas(n, 0);
    if (totalRbg == 0 || n == 0)
    {
        return quotas;
    }

    std::vector<double> raw(n, 0.0);
    double rawSum = 0.0;
    for (uint32_t i = 0; i < n; ++i)
    {
        const double pct = (i < pctPrb.size()) ? std::clamp(pctPrb[i], 0.0, 100.0) : 0.0;
        raw[i] = pct / 100.0 * static_cast<double>(totalRbg);
        rawSum += raw[i];
    }

    uint64_t target = static_cast<uint64_t>(std::llround(rawSum));
    if (target > totalRbg)
    {
        NS_LOG_WARN("[SqrScheduler] ComputeRbgQuotas: sum(pct)=" << rawSum / totalRbg * 100.0
                                                                 << "% > 100% — coercing target to " << totalRbg << " RBGs");
        target = totalRbg;
    }

    std::vector<double> frac(n, 0.0);
    uint64_t base = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        const uint64_t f = static_cast<uint64_t>(std::floor(raw[i]));
        quotas[i] = static_cast<uint32_t>(f);
        frac[i] = raw[i] - static_cast<double>(f);
        base += f;
    }

    int64_t extra = static_cast<int64_t>(target) - static_cast<int64_t>(base);
    NS_ASSERT_MSG(extra >= 0, "round(sum) < sum(floor) — impossível: extra=" << extra);
    while (extra > 0)
    {
        // maior resto fracionário; desempate → menor índice de slice
        int64_t best = -1;
        for (uint32_t i = 0; i < n; ++i)
        {
            if (frac[i] >= 0.0 && (best < 0 || frac[i] > frac[best]))
            {
                best = i;
            }
        }
        if (best < 0)
        {
            break; // sem restos disponíveis (defensivo; inalcançável)
        }
        quotas[best] += 1;
        frac[best] = -1.0; // consome o resto da slice
        extra -= 1;
    }
    return quotas;
}

// ---------------------------------------------------------------------------
// Configuração (cota absoluta)
// ---------------------------------------------------------------------------

void
SqrMacScheduler::ApplyQuotas(const std::vector<uint32_t>& quotaRbg,
                             const std::vector<IntraSliceAlgorithm>& algorithms,
                             const char* source)
{
    const uint32_t totalRbg = GetBandwidthInRbg();
    NS_ASSERT_MSG(quotaRbg.size() == algorithms.size(), "quotas and algorithms must have same size");
    if (m_numSlices > 0)
    {
        NS_ASSERT_MSG(quotaRbg.size() == m_numSlices, "Must match configured number of slices");
    }

    std::vector<uint32_t> clamped(quotaRbg);
    uint64_t sum = 0;
    for (uint32_t q : clamped)
    {
        sum += q;
    }
    if (totalRbg > 0 && sum > totalRbg)
    {
        // Só ocorre se o agente violar Σa ≤ C (contrato §3.2/projeção Π).
        // Coerção slice a slice ao teto, SEM redistribuição (documentada).
        for (uint32_t& q : clamped)
        {
            q = std::min(q, totalRbg);
        }
        NS_LOG_WARN("[SqrScheduler] sum(quotas)=" << sum << " > totalRbg=" << totalRbg
                                                  << " — coerced per-slice (agent violated Σa ≤ C)");
    }

    m_sliceQuotaRbg = clamped;
    m_intraAlgorithms = algorithms;

    // Pesos "de decisão" = fração da cota no BWP (telemetria rsh/KPM e CSVs).
    m_lastDecisionWeights.assign(clamped.size(), 0.0);
    for (uint32_t s = 0; s < clamped.size(); ++s)
    {
        m_lastDecisionWeights[s] =
            (totalRbg > 0) ? static_cast<double>(clamped[s]) / static_cast<double>(totalRbg) : 0.0;
    }

    std::ostringstream oss;
    oss << "[SqrScheduler] Absolute RBG quota applied (" << source << "):";
    for (uint32_t s = 0; s < clamped.size(); ++s)
    {
        oss << " Slice" << s << "(q=" << clamped[s] << " RBG";
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
    if (sum == 0)
    {
        // Σa = 0 ⇒ NADA servido até a próxima configuração (R27) — visível.
        oss << " | ZERO QUOTA: no slice will be served";
        NS_LOG_WARN(oss.str());
    }
    else
    {
        NS_LOG_INFO(oss.str());
    }
}

void
SqrMacScheduler::SetSliceQuotaRbg(const std::vector<uint32_t>& quotaRbg,
                                  const std::vector<IntraSliceAlgorithm>& algorithms)
{
    NS_LOG_FUNCTION(this);
    // A ação do agente SUPERSEDE a configuração estática inicial: descarta
    // qualquer cota pendente de materialização (o main() configura a partição
    // estática antes do Run, mas a 1ª slot DL pode ocorrer DEPOIS da primeira
    // ação — materializá-la aí sobrescreveria a cota já aplicada pelo agente).
    m_pendingQuotaPct.clear();
    ApplyQuotas(quotaRbg, algorithms, "agent action");
}

void
SqrMacScheduler::SetSliceConfiguration(const std::vector<double>& prbWeights,
                                       const std::vector<IntraSliceAlgorithm>& algorithms)
{
    /*
     * OVERRIDE da base (que normaliza para soma 1 — proibido por R27).
     * Interpreta as frações como % de PRB (pct = 100·w, Σ ≤ 100, Σ = 0
     * permitido) e aplica a MESMA regra determinística de cota do agente.
     * Usada apenas para a configuração estática inicial do sqr-sim.
     */
    NS_LOG_FUNCTION(this);
    NS_ASSERT_MSG(prbWeights.size() == algorithms.size(), "Weights and algorithms must have same size");
    for (double w : prbWeights)
    {
        NS_ASSERT_MSG(w >= 0.0, "PRB weight fractions must be non-negative, got " << w);
        NS_ASSERT_MSG(w <= 1.0 + 1e-6, "PRB weight fraction must be <= 1.0, got " << w);
    }

    std::vector<double> pct(prbWeights.size());
    for (uint32_t s = 0; s < prbWeights.size(); ++s)
    {
        pct[s] = 100.0 * prbWeights[s];
    }

    // Estado da base mantido coerente para DumpSliceConfiguration/GetPrbWeight
    // (sem renormalização — os pesos são armazenados como recebidos).
    m_prbWeights = prbWeights;

    const uint32_t totalRbg = GetBandwidthInRbg();
    if (totalRbg == 0)
    {
        // Chamada do main() ANTES do Simulator::Run: o SAP user do MAC ainda
        // não existe e o total de RBGs do BWP é desconhecido. Guarda o pct e
        // materializa a cota na 1ª chamada de AssignDLRBG.
        m_pendingQuotaPct = pct;
        m_intraAlgorithms = algorithms;
        NS_LOG_INFO("[SqrScheduler] static config stored as pct (totalRbg unknown "
                    "before Run); quota will be materialized on first DL slot");
        return;
    }
    const std::vector<uint32_t> quotas =
        ComputeRbgQuotas(pct, totalRbg, static_cast<uint32_t>(pct.size()));
    ApplyQuotas(quotas, algorithms, "static config");
}

std::vector<uint32_t>
SqrMacScheduler::GetSliceQuotaRbg() const
{
    return m_sliceQuotaRbg;
}

// ---------------------------------------------------------------------------
// Alocação OFDMA por cota absoluta
// ---------------------------------------------------------------------------

NrMacSchedulerNs3::BeamSymbolMap
SqrMacScheduler::AssignDLRBG(uint32_t symAvail, const ActiveUeMap& activeDl) const
{
    /*
     * Override do loop de partição de RslaqMacScheduler::AssignDLRBG com a
     * semântica de COTA ABSOLUTA (R27): orçamento por slice = cota corrente
     * (NÃO recalculada por demanda, NÃO renormalizada); RBGs excedentes
     * ficam ociosos; a cota nunca é ultrapassada. Estrutura de logging
     * (slice_alloc.csv, ue_detail.csv, etc.) e de alocação intra-slice
     * (RR/PF/BCQI round-robin) reutilizada da base via herança.
     */
    NS_LOG_FUNCTION(this << symAvail);

    // SQR-R27: materializa a cota da configuração estática inicial na 1ª
    // chamada (no main() o total de RBGs do BWP ainda era desconhecido).
    // Escreve apenas estado mutable (AssignDLRBG é const; os algoritmos
    // intra-slice já foram guardados pelo caminho pendente de
    // SetSliceConfiguration).
    if (!m_pendingQuotaPct.empty() && m_numSlices > 0)
    {
        const uint32_t totalRbgNow = GetBandwidthInRbg();
        std::vector<uint32_t> quotas = ComputeRbgQuotas(
            m_pendingQuotaPct, totalRbgNow, static_cast<uint32_t>(m_pendingQuotaPct.size()));
        if (totalRbgNow > 0)
        {
            uint64_t sum = 0;
            for (uint32_t q : quotas)
            {
                sum += q;
            }
            if (sum > totalRbgNow)
            {
                for (uint32_t& q : quotas)
                {
                    q = std::min(q, totalRbgNow);
                }
            }
        }
        m_sliceQuotaRbg = quotas;
        m_lastDecisionWeights.assign(quotas.size(), 0.0);
        for (uint32_t s = 0; s < quotas.size(); ++s)
        {
            m_lastDecisionWeights[s] =
                (totalRbgNow > 0)
                    ? static_cast<double>(quotas[s]) / static_cast<double>(totalRbgNow)
                    : 0.0;
        }
        std::ostringstream oss;
        oss << "[SqrScheduler] Absolute RBG quota materialized (static config,"
               " deferred to first DL slot):";
        for (uint32_t s = 0; s < quotas.size(); ++s)
        {
            oss << " Slice" << s << "(q=" << quotas[s] << ")";
        }
        NS_LOG_INFO(oss.str());
        m_pendingQuotaPct.clear();
    }

    if (m_sqrFirstRun)
    {
        m_sqrFirstRun = false;
        OpenCsvFiles();
        DumpSliceConfiguration();
    }

    GetFirst GetBeamId;
    GetSecond GetUeVector;
    BeamSymbolMap symPerBeam = GetSymPerBeam(symAvail, activeDl);

    const uint64_t timeMs = Simulator::Now().GetMilliSeconds();
    const uint64_t callId = GetNextCallId();
    const bool logThisSlot = ShouldLogSlot(timeMs);

    if (logThisSlot)
    {
        LogActiveDlDiagnostic(callId, timeMs, activeDl);
        LogHarqState(callId, timeMs);
    }

    // SQR-R27: contabiliza o slot DL na janela corrente (1 beam ⇒ 1 chamada
    // = 1 slot DL com ≥1 UE ativo; base de prb_idle no CSV de janelas).
    m_windowDlSlots += 1;

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
        const std::string beamIdStr = beamIdToStr(beamId);

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
            const uint32_t bwRbg = GetBandwidthInRbg();
            availableRbgIds.reserve(bwRbg);
            for (uint32_t i = 0; i < bwRbg; ++i)
            {
                availableRbgIds.push_back(i);
            }
        }

        const uint32_t totalRbgs = static_cast<uint32_t>(availableRbgIds.size());
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

        // --- Agrupa UEs ativos por slice (idêntico à base) ---
        std::vector<std::vector<UePtrAndBufferReq>> sliceUeVec(m_numSlices);
        for (const auto& ue : GetUeVector(el))
        {
            const uint16_t rnti = ue.first->m_rnti;

            // SQR: snapshot do SINR DL por RNTI (inversão do CQI wideband nas
            // tabelas do NrAmc). Sentinela -999 dB sem CQI válido (contrato
            // §4 / register R09).
            {
                const uint8_t wbCqi = ue.first->m_dlCqi.m_wbCqi;
                double sinrDb = -999.0;
                if (wbCqi > 0 && GetDlAmc())
                {
                    const double se = GetDlAmc()->GetSpectralEfficiencyForCqi(wbCqi);
                    const double sinrLin = GetDlAmc()->GetSinrFromSpectralEfficiency(se);
                    sinrDb = 10.0 * std::log10(std::max(sinrLin, 1e-12));
                }
                m_lastDlSinrDb[rnti] = sinrDb;
            }

            const int32_t sIdx = GetSliceIndexForRnti(rnti);
            if (sIdx >= 0)
            {
                sliceUeVec[static_cast<uint32_t>(sIdx)].emplace_back(ue);
            }
            else
            {
                NS_LOG_WARN("[SqrScheduler] RNTI " << rnti << " not found in any slice!");
                if (m_unmappedRntiCsv.is_open())
                {
                    m_unmappedRntiCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                                      << rnti << ",not_in_any_slice\n";
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
            for (const auto& ue : sliceUeVec[s])
            {
                // Detecção de demanda (idêntica à base): buffer pendente OU
                // TB em voo conta como tráfego ativo.
                if (ue.second > 0 || ue.first->m_dlTbSize > 0)
                {
                    sliceHasDemand[s] = true;
                    break;
                }
            }
        }

        // --- SQR-R27: orçamento = COTA ABSOLUTA (sem recalcular/renormalizar) ---
        std::vector<uint32_t> sliceRbgBudget(m_numSlices, 0);
        std::vector<double> decisionWeights(m_numSlices, 0.0);
        std::vector<double> effectiveWeight(m_numSlices, 0.0);
        for (uint32_t s = 0; s < m_numSlices; s++)
        {
            const uint32_t quota =
                (s < m_sliceQuotaRbg.size()) ? m_sliceQuotaRbg[s] : 0;
            sliceRbgBudget[s] = std::min(quota, totalRbgs);
            if (sliceRbgBudget[s] < quota)
            {
                NS_LOG_WARN("[SqrScheduler] quota " << quota << " of slice " << s
                                                    << " clamped to totalRbgs=" << totalRbgs);
            }
            decisionWeights[s] = static_cast<double>(sliceRbgBudget[s]) / totalRbgs;
            effectiveWeight[s] = decisionWeights[s];
        }
        m_lastDecisionWeights = decisionWeights;

        const uint32_t budgetTotal =
            std::accumulate(sliceRbgBudget.begin(), sliceRbgBudget.end(), 0u);
        // DIVERGÊNCIA DELIBERADA vs base (R27): a base asserta
        // budgetTotal == totalRbgs (partição completa); aqui a SOMA DAS COTAS
        // deve ser ≤ total (RBGs excedentes ficam ociosos por construção).
        NS_ASSERT_MSG(budgetTotal <= totalRbgs,
                      "sum(quotaRbg)=" << budgetTotal << " exceeds totalRbgs=" << totalRbgs);

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

        // --- Alocação slice a slice, até a cota (idêntica à base) ---
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
                // Cota 0 (inclui Σa=0): a slice NÃO é servida e seus RBGs
                // NÃO são redistribuídos — ficam ociosos (R27).
                reason = "zero_quota";
            }
            else
            {
                SortUeVectorByAlgorithm(sliceUeVec[s], m_intraAlgorithms[s]);

                uint32_t iteration = 0;
                const uint32_t maxIterations =
                    sliceRbgBudget[s] * sliceUeVec[s].size() + sliceUeVec[s].size();

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

                        const uint32_t bufQueueSize = ue.second;
                        const bool demandPassed = (bufQueueSize > 0 || ue.first->m_dlTbSize > 0);

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
                                         << bufQueueSize << "," << ue.first->m_dlTbSize << ","
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
                            if (!rbgRealUsed[availableRbgIds[logicalIdx]])
                            {
                                break;
                            }
                        }
                        if (logicalIdx >= totalRbgs)
                        {
                            break; // sem RBG livre; excedentes permanecem ociosos
                        }

                        const uint32_t actualRbgId = availableRbgIds[logicalIdx];
                        rbgRealUsed[actualRbgId] = true;

                        auto& dlRbg = ue.first->m_dlRBG;
                        const auto existingRbgs = dlRbg.size();
                        dlRbg.resize(dlRbg.size() + beamSym);
                        std::fill(dlRbg.begin() + existingRbgs, dlRbg.end(), actualRbgId);

                        auto& dlSym = ue.first->m_dlSym;
                        const auto existingSyms = dlSym.size();
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
                            const uint32_t tbSize = ue.first->m_dlTbSize;
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
                {
                    rntiList << ";";
                }
                rntiList << sliceUeVec[s][i].first->m_rnti;
            }
            sliceRntiListVec[s] = rntiList.str();

            if (logThisSlot && reason == "no_demand" && m_ueAllocCsv.is_open())
            {
                for (const auto& ue : sliceUeVec[s])
                {
                    m_ueAllocCsv << callId << "," << timeMs << "," << m_scenarioName << ","
                                 << m_currentSlot << "," << m_bwpId << "," << beamIdStr << ","
                                 << s << "," << ue.first->m_rnti << ","
                                 << ue.second << "," << ue.first->m_dlTbSize << ","
                                 << 0 << "," << 0 << "," << 0 << "," << 0 << ",0,"
                                 << static_cast<uint32_t>(ue.first->GetDlMcs()) << ","
                                 << static_cast<uint32_t>(ue.first->m_dlRank) << ","
                                 << GetNumRbPerRbg() << ",no_demand\n";
                }
            }
        }

        const uint32_t totalAllocatedRbg =
            std::accumulate(sliceAllocatedVec.begin(), sliceAllocatedVec.end(), 0u);

        // --- Estatísticas, contadores de janela e CSV por slice ---
        for (uint32_t s = 0; s < m_numSlices; ++s)
        {
            // SQR-R27: a cota NUNCA é ultrapassada — validação (iii) do smoke
            // ("nenhum dado servido em RBG ocioso") é garantida aqui em cada
            // slot, pois os únicos RBGs tocados são os de cotas por slice.
            NS_ASSERT_MSG(sliceAllocatedVec[s] <= sliceRbgBudget[s],
                          "allocatedRbg=" << sliceAllocatedVec[s]
                                          << " exceeds quotaRbg=" << sliceRbgBudget[s]
                                          << " for slice " << s);

            const double rshRealPct =
                (totalAllocatedRbg > 0)
                    ? 100.0 * static_cast<double>(sliceAllocatedVec[s]) /
                          static_cast<double>(totalAllocatedRbg)
                    : 0.0;
            const double budgetUtilizationPct =
                std::clamp((sliceRbgBudget[s] > 0)
                               ? 100.0 * static_cast<double>(sliceAllocatedVec[s]) /
                                     static_cast<double>(sliceRbgBudget[s])
                               : 0.0,
                           0.0,
                           100.0);
            const double unusedBudgetPct =
                (sliceRbgBudget[s] > 0) ? 100.0 - budgetUtilizationPct : 0.0;
            const double targetSharePct = effectiveWeight[s] * 100.0;
            const double allocationFidelityPct =
                std::clamp(100.0 - std::abs(rshRealPct - targetSharePct), 0.0, 100.0);

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

            // SQR-R27: acumula prb_scheduled (RBGs reais) e prb_budget (cota
            // por slot) para o CSV de janelas do sqr-sim.
            if (m_windowScheduledRbg.size() == m_numSlices)
            {
                m_windowScheduledRbg[s] += sliceAllocatedVec[s];
                m_windowBudgetRbg[s] += sliceRbgBudget[s];
            }

            if (logThisSlot && m_sliceAllocCsv.is_open())
            {
                // redistributed_idle_rbg: campo herdado do esquema da base; sob
                // cota absoluta NADA é redistribuído — grava 0 (os RBGs ociosos
                // reais aparecem em sqr_prb_windows.csv:prb_idle_rbg).
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
                                << 0 << ","
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

} // namespace ns3
