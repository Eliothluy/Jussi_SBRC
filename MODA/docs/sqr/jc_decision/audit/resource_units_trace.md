# resource_units_trace.md — D0-A §5.5 (A06)

**Resposta: C=53 = 53 RBGs = 53 PRBs (1 RBG = 1 PRB por slot DL).** Prova por leitura de
código (congelada em `results/sqr_jc_final/01_resource_unit/resource_unit_manifest.md`,
03/11) + spot-checks desta auditoria (08/11):

1. PRBs do BWP: `NrPhy::DoUpdateRbNum` = floor(10 MHz × (1−0,04) / 180 kHz) = 53
   (`nr-phy.cc:571-584`; RbOverhead=0,04; μ=0).
2. RBG: `NumRbPerRbg=1` default (`nr-helper.cc:158-161`) ⇒ `bw_in_rbg=53`
   (`nr-gnb-mac.cc:1370-1375`) — o ramo da tabela 38.214 NÃO roda (valor ≠ 0).
3. Telemetria: `total_rbg=53, total_prb=53` em todas as linhas
   (spot-check: `hetero_sweep/A_lam6_s3/.../sqr_prb_windows.csv`).
4. Cadeia Python→IPC→C++→MAC: ação inteira a_i → `dedicatedPRB=100·a_i/53`
   (`sqr_env.py:243-246`) → C++ `raw_i = pct/100 × totalRbg = a_i` exato
   (`SqrMacScheduler::ComputeRbgQuotas`) — cota inteira aplicada = ação do agente,
   unidade a unidade. Ação = CAP por slot DL (uso real ≤ cota; excedente OCIOSO,
   nunca redistribuído; porta de suficiência upstream).
5. Semântica congelada: `ALLOCATED_USAGE`, Σa ≤ C, 27.720 ações
   (`joint_action_space.py`; idêntica ao oracle da Rodada 2, `simplex.py:15-17`).
6. Divergência registrada: narrativa R01 "53 = 13 RBGs × 4 PRBs + 1" REFUTADA pelo
   código — emenda do assumptions_register pendente (D0-B).

Invariantes INV-01/02/04/05: satisfeitos pela cadeia acima; `a_effectively_applied`
(prb_executed_rbg_i na telemetria) é coluna separada de `a_sent_to_mac` (budget).
