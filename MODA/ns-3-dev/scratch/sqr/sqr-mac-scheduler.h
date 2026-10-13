/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * SqrMacScheduler — MAC scheduler de COTA ABSOLUTA de RBGs por slice
 * (cenário SQR, Fase 1 — contrato_interface.md §3.3, register R27).
 *
 * Subclasse de RslaqMacScheduler (a cópia em scratch/sqr; o original em
 * scratch/rslaq permanece intocado — AGENTS §5.2).
 *
 * MOTIVAÇÃO (R27, revisão 11/10/2026): o caminho vigente renormaliza a ação —
 * rslaq-sim divide os percentuais por totalDed (soma) e o
 * RslaqMacScheduler::SetSliceConfiguration renormaliza internamente para
 * soma 1; o particionamento redistribui TODOS os totalRbgs por maiores
 * restos (assert budgetTotal == totalRbgs). Com qualquer um deles,
 * Σ dedicatedPRB < 100 é silenciosamente re-expandido para partição completa
 * e, com tráfego greedy (R04), prb_scheduled ≈ C sempre — o custo γ·a/C e a
 * Fig. 4 (economia de RBs) ficam inobserváveis.
 *
 * SEMÂNTICA CONGELADA (cota absoluta):
 *  - A configuração por slice é interpretada como número INTEIRO de RBGs
 *    (cota) por slot DL: cada slice é servida ATÉ sua cota;
 *  - RBGs excedentes ficam OCIOSOS (nunca redistribuídos para outra slice);
 *  - Σ quotas = 0 ⇒ NADA é servido (com NS_LOG_WARN — sem manter/configurar
 *    partição anterior);
 *  - a cota nunca é ultrapassada (assert sliceAllocated <= quota por slot).
 *
 * A conversão % de PRB → cota inteira de RBGs é feita pelo sqr-sim com a
 * regra determinística de ComputeRbgQuotas() (piso + maiores restos
 * fracionários, alvo round(Σ raw), desempate pelo índice de slice — SEM
 * renormalização; Σ quotas ≤ totalRbg garantido). A configuração estática
 * inicial pode usar SetSliceConfiguration (frações somando ≤ 1), que aplica
 * a mesma regra internamente.
 *
 * A cota restringe somente DL (escopo do artigo: SLAs de throughput DL);
 * o agendamento UL herda o RR do NrMacSchedulerOfdmaRR.
 */

#pragma once

#include "rslaq-mac-scheduler.h"

#include <vector>

namespace ns3
{

class SqrMacScheduler : public RslaqMacScheduler
{
  public:
    static TypeId GetTypeId();

    SqrMacScheduler();
    ~SqrMacScheduler() override;

    /**
     * @brief Conversão determinística % de PRB → cota inteira de RBGs (R27).
     *
     * Regra congelada (sem renormalização): raw_i = pct_i/100 × totalRbg;
     * alvo = round(Σ raw) limitado a totalRbg; quota_i = floor(raw_i) e o
     * excedente (alvo − Σ floor) distribuído 1 RBG por slice pelos MAIORES
     * restos fracionários, desempate pelo MENOR índice de slice. Garante
     * 0 ≤ Σ quotas ≤ totalRbg e Σ quotas ≥ floor(Σ raw) (folga de arredonda-
     * mento ≤ L RBGs — critério (ii) do smoke).
     *
     * @param pctPrb   percentual de PRB por slice (0–100; valores fora são
     *                 clampados); Σ pct ≤ 100 é garantido pelo agente (proje-
     * ção Π no Python, R24) — Σ > 100 é coerçado ao teto com warn
     * @param totalRbg total de RBGs do BWP
     * @param numSlices número de slices (tamanho do vetor de saída)
     */
    static std::vector<uint32_t> ComputeRbgQuotas(const std::vector<double>& pctPrb,
                                                  uint32_t totalRbg,
                                                  uint32_t numSlices);

    /**
     * @brief Aplica a COTA ABSOLUTA (RBGs inteiros por slice) e os algoritmos
     * intra-slice. Sem renormalização. Aceita Σ = 0 (nada servido, com warn).
     *
     * Chamada pelo sqr-sim a cada ação do agente (após ComputeRbgQuotas).
     * Valores por slice são clampados a [0, totalRbg] (warn se clampar);
     * Σ quotas > totalRbg só ocorre se o agente violar Σa ≤ C (contrato §3.2)
     * e é coerçado slice a slice, sem redistribuição.
     */
    void SetSliceQuotaRbg(const std::vector<uint32_t>& quotaRbg,
                          const std::vector<IntraSliceAlgorithm>& algorithms);

    /**
     * @brief Configuração estática inicial (frações somando ≤ 1, inclusive 0).
     *
     * Sobrepõe a versão da base (que normaliza para soma 1 — proibido em R27).
     * Converte frações → % → cota RBG via ComputeRbgQuotas no totalRbg atual.
     */
    void SetSliceConfiguration(const std::vector<double>& prbWeights,
                               const std::vector<IntraSliceAlgorithm>& algorithms) override;

    /** Cota RBG corrente por slice (telemetria prb_executed do sqr-sim). */
    std::vector<uint32_t> GetSliceQuotaRbg() const;

  protected:
    /**
     * @brief Alocação OFDMA por cota absoluta (override do loop da base).
     *
     * Diferenças vs RslaqMacScheduler::AssignDLRBG:
     *  - orçamento por slice = cota absoluta corrente (m_sliceQuotaRbg),
     *    clampada ao total de RBGs do beam; NÃO recalculada por demanda;
     *  - RBGs não cobertos pelas cotas ficam ociosos (sem redistribuição);
     *  - assert de coerência trocado: Σ budgets ≤ totalRbgs (a base exigia
     *    igualdade — a igualdade é justamente a renormalização proibida);
     *  - contadores de janela acumulam prb_scheduled (real) e prb_budget
     *    (cota × slots), e m_windowDlSlots conta os slots DL chamados.
     */
    BeamSymbolMap AssignDLRBG(uint32_t symAvail, const ActiveUeMap& activeDl) const override;

  private:
    /** Aplica cotas clampadas + algoritmos + pesos "de decisão" p/ telemetria. */
    void ApplyQuotas(const std::vector<uint32_t>& quotaRbg,
                     const std::vector<IntraSliceAlgorithm>& algorithms,
                     const char* source);

    /** Cota absoluta corrente (RBGs inteiros por slot DL, por slice). */
    mutable std::vector<uint32_t> m_sliceQuotaRbg{};
    /**
     * % de PRB pendente de conversão (config estática inicial): no main(),
     * GetBandwidthInRbg() ainda é 0 (o SAP user do MAC não foi instalado);
     * a cota é materializada na 1ª chamada de AssignDLRBG, quando o total de
     * RBGs do BWP já é conhecido. Vazio = sem pendência.
     */
    mutable std::vector<double> m_pendingQuotaPct{};
    mutable bool m_sqrFirstRun{true};
};

} // namespace ns3
