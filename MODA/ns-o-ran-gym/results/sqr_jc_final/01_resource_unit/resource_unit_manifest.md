# resource_unit_manifest.md — HARD GATE §21/§22 (unidade de recurso e semântica da ação)

**Etapa 2.5 SQR-JC — protocolo FINAL (2026-11-03). Status: RESOLVIDO POR LEITURA DE CÓDIGO.**
Data da auditoria: 2026-11-03 · Agente B (pre-flight) · Evidência: código C++ congelado + telemetria congelada + código Python congelado.

---

## 1. O que C=53 representa (§21, primeira pergunta)

**Resposta: C = 53 é SIMULTANEAMENTE o número de RBGs da malha DL e o número de PRBs, porque cada RBG cobre exatamente 1 PRB na configuração congelada do cenário.**

Cadeia de prova (todos os pontos lidos no código-fonte congelado):

1. **Contagem de PRBs do BWP** (`NrPhy::DoUpdateRbNum`, `contrib/nr/model/nr-phy.cc:571-584`):
   `m_rbNum = floor(channelBandwidth × (1 − rbOverhead) / (SCS × 12))`
   = floor(10 MHz × (1 − 0,04) / 180 kHz) = floor(53,33) = **53 PRBs**.
   O overhead de 4% vem do atributo `RbOverhead` do `NrHelper` (default 0,04 — `contrib/nr/helper/nr-helper.cc:154-157`), repassado à PHY em `nr-helper.cc:543/661`. O cenário usa bandwidth = 10 MHz, numerology μ=0 (`scratch/sqr/sqr-sim.cc:1983-1984`).
2. **Tamanho do RBG**: o atributo `NumRbPerRbg` do `NrHelper` tem default **1** (`nr-helper.cc:158-161`, `UintegerValue(1)`), aplicado ao MAC via `mac->SetNumRbPerRbg(m_numRbPerRbg)` (`nr-helper.cc:823`). Como o valor já é ≠ 0, `NrGnbMac::DoConfigureMac` NÃO o recalcula pela tabela 38.214 (`contrib/nr/model/nr-gnb-mac.cc:1361-1368` — o ramo `SetNumRbPerRbg(nr::NumRbsPerRbg(bw_in_rb))` só roda com valor 0).
3. **RBGs totais**: `bw_in_rbg = bw_in_rb / numRbsPerRbg` = 53 / 1 = **53 RBGs** (`nr-gnb-mac.cc:1370-1375`), propagado ao scheduler via `DoCschedCellConfigReq` (`nr-gnb-mac.cc:1388-1393` → `nr-mac-scheduler-ns3.cc:563-564`) e exposto por `GetBandwidthInRbg()` (`nr-mac-scheduler-ns3.cc:2457-2460`).
4. **Telemetria congelada confirma byte-a-byte**: `MODA/ns-3-dev/results_sqr_network_only/seed=*/sqr_prb_windows.csv` tem `total_rbg=53, total_prb=53` em todas as linhas. `GetTotalPrbCount() = GetBandwidthInRbg() × GetNumRbPerRbg()` = 53×1 (`scratch/sqr/rslaq-mac-scheduler.cc:1605-1615`).
5. **Consequência para a conversão Python→C++**: o Python envia `dedicatedPRB = 100·a_i/53` (contrato §3.2, `sqr_env.py:243-246`); o C++ converte de volta `raw_i = pct_i/100 × totalRbg` = `a_i` exato (sem resto de arredondamento) — a cota inteira aplicada no MAC é **igual unidade a unidade à ação inteira do agente** (`SqrMacScheduler::ComputeRbgQuotas`, `sqr-mac-scheduler.cc:56-129`).

⚠️ **Divergência registrada (não corrigida)**: o docstring de `sqr_env.py:99-100` (register R01 narrativa) diz "53 = 13 RBGs × 4 PRBs + 1" — mecanismo REFUTADO pelo código (daria 52, e a tabela 38.214 daria RBG de 4). O número 53 está correto; o MECANISMO correto é `NumRbPerRbg=1` + overhead 4%. O scheduler de fato agenda RBGs de 1 PRB (`GetNumRbPerRbg()` logado em `sqr-mac-scheduler.cc:591/654`). Nada de código foi alterado; a narrativa do register deve ser emendada pelo orquestrador quando ratificar este manifest.

## 2. "Allocation unit" = quantos PRBs físicos? (§21)

**1 allocation unit = 1 RBG = 1 PRB por slot DL.** A cota restringe por SLOT DL (o budget `sliceRbgBudget[s]` é consumido dentro de `AssignDLRBG`, que é chamado 1× por slot DL com UEs ativos; o acumulador de janela soma `prb_budget = cota × slots DL` — `sqr-mac-scheduler.cc:754-760`, `sqr-sim.cc:1589`). Por janela de decisão de 100 ms (10 slots DL máx no TDD congelado), a cota × slots é o teto de RBG-slots, e a ação unitária é 1 PRB por slot.

## 3. O scheduler pode deixar budget sem uso? (§21)

**SIM — provado em código.** Com cota absoluta (register R27), a base exigia `budgetTotal == totalRbgs` (partição completa); o scheduler SQR trocou o assert para `budgetTotal <= totalRbgs` e os RBGs não cobertos pelas cotas **ficam ociosos** (`sqr-mac-scheduler.h:104-109`, `.cc:492-498`; razão `zero_quota`/`no_demand`/`all_ue_demand_satisfied` em `.cc:518-675`). A porta de suficiência de demanda do upstream 5G-LENA (`m_dlTbSize ≥ max(buffer,10)` ⇒ UE não recebe mais RBG no slot, `.cc:553-573`) permite `prb_scheduled < prb_budget` mesmo com demanda registrada.

## 4. Budget não utilizado pode ser emprestado? (§21)

**NÃO — nunca redistribuído.** "RBGs excedentes ficam OCIOSOS (nunca redistribuídos para outra slice)" (`sqr-mac-scheduler.h:19-24`; telemetria `prb_idle_rbg` em `sqr-sim.cc:1555-1591`). Estatístico: `redistributed_idle_rbg` é gravado como 0 fixo sob cota absoluta (`sqr-mac-scheduler.cc:762-768`).

## 5. A ação representa cap, quota, reserva ou uso real? (§21)

**CAP (= cota absoluta por slot DL).** Cada slice é servida ATÉ sua cota (`sliceAllocated <= quota` com assert por slot, `sqr-mac-scheduler.cc:716-719`); o uso real (`prb_scheduled`) pode ser menor (demanda insuficiente). A ação NÃO é o uso real e NÃO é partição obrigatória do espectro.

---

## 6. ACTION_SEMANTICS (§22) — decisão

```text
ACTION_SEMANTICS = ALLOCATED_USAGE
action_semantics: ALLOCATED_USAGE
resource_unit: 1 RBG = 1 PRB per DL slot
C: 53
```

**Justificativa baseada em código (nunca por conveniência computacional):**

- A leitura **BUDGET_PARTITION (Σa = C, 1.485 combinações)** é REFUTADA: com R27, Σ quotas < totalRbg é legítimo e os RBGs restantes ficam ociosos por construção — a partição completa é exatamente a renormalização proibida (motivação do R27, `sqr-mac-scheduler.h:9-16`).
- A leitura **ALLOCATED_USAGE (Σa ≤ C, 27.720 combinações)** é a que corresponde ao executor: ações com Σ < C são admissíveis e deixam idle real (prova nas seções 3–4).
- O espaço implementado (`sqr_jc/joint_action_space.py`): `{(a_e, a_u, a_m) ∈ Z³≥0 : a_i ≤ C, Σa ≤ C}`, |A| = (C+1)(C+2)(C+3)/6 = **27.720** para C=53, enumeração lexicográfica determinística (a_e, a_u, a_m ascendentes) usada por todos os desempates (§50/§51/§52 chave 5).
- Invariantes §35: nenhuma ação negativa; nenhuma a_i > C; Σa ≤ C; nenhuma violação de piso ativo (`validate_action`/`filter_feasible_mask`).
- Consistência com o oracle da Rodada 2: `oracle/simplex.py:15-17` já enumerava 27.720 pontos com Σ ≤ C=53 — a semântica auditable da Rodada 2 é a mesma.

---

## 7. Tabela-resposta §21

| Pergunta §21 | Resposta | Evidência |
|---|---|---|
| C=53 representa PRB? RBG? quantum? outro? | RBGs da malha DL = 53 PRBs (RBG=1 PRB no congelamento) | nr-helper.cc:158-161; nr-phy.cc:584; nr-gnb-mac.cc:1374; telemetria total_rbg=total_prb=53 |
| 1 allocation unit = quantos PRBs? | 1 RBG = 1 PRB por slot DL | GetNumRbPerRbg()=1; ComputeRbgQuotas |
| Scheduler pode deixar budget sem uso? | SIM | sqr-mac-scheduler.cc:492-498 (assert ≤) |
| Budget ocioso é emprestado? | NUNCA (fica idle) | sqr-mac-scheduler.h:19-24; prb_idle_rbg |
| Ação = cap, quota, reserva ou uso real? | CAP (cota absoluta por slot DL, uso real ≤ cota) | sqr-mac-scheduler.cc:716-719 |

## 8. Compatibilidade com o §43 (pisos fallback)

A auditoria CONFIRMA mesma unidade e mesma semântica da Rodada 2 (cota absoluta de RBGs, 1 RBG=1 PRB): os fallbacks históricos URLLC=13 / mMTC=3 são compatíveis e entram como pisos conservativos do SQR-JC (`conditional_floors.py`), com URLLC=5 / mMTC=1 como âncoras diagnósticas apenas (§44).

**Regra de escrita do artigo (§21):** proibido escrever "53 PRBs" ou "53 RBGs" sem a qualificação acima; a formulação correta é "C = 53 allocation units (1 unit = 1 RBG = 1 PRB per DL slot under the frozen NumRbPerRbg=1 grid)".
