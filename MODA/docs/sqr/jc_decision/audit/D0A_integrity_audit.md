# D0A_integrity_audit.md — GATE DE INTEGRIDADE DAS EVIDÊNCIAS (PASS com resultados revisados)

**Protocolo:** `docs/artigo/SQR_JC_Protocolo_Implementacao_Auditoria_v2.md` §5 (D0-A) · **Data:** 08/11/2026 · **Checkout:** `b9f3c0e` (main, dirty — artefatos de out/nov não commitados, inventariados com SHA-256) · **Auditor:** orquestrador (GLM-5.3), funções congeladas do adjudicador II-c reutilizadas sem alteração.

**Entregáveis:** `D0A_input_inventory.csv` (52 itens, **0 MISSING**) · `D0A_replay_5x3.csv` (15 células recomputadas de traces brutos) · `r19_by_seed_lambda.csv` · `D0A_CLAIM_LEDGER.csv` · `resource_units_trace.md` · `urllc_timing_audit.md` · `claims_interpretation_review.md` · `D0A_GATE.json` · figuras revisadas em `docs/artigo/figuras_tese/audit_v2/`.

---

## 1. GATE REPLAY_5x3 — a matriz recomputada dos traces brutos

Todas as 15 células (5 estados × 3 ações) recomputadas com `window_kpis`/`_sla_oks` **congeladas** (adjudication.py — as mesmas funções que adjudicaram o G2/II-c; valores A2 batem bit-a-bit com `g2_deltas.csv`). Ação executada confere com a planejada **15/15** (`sqr_prb_windows.csv`, override único). **Prefixos A/A (offline, sem novas simulações): KPM bit-idêntico 10/10 e delays bit-idênticos 10/10** (regra congelada II-b: entrega < t·100 ms, pacotes entregues dentro de t excluídos por deslocarem ±1 slot legitimamente).

| Ação | frozen (R36–R38, vacuous=OK) | strict ALL3_COMPLETE | erro anterior |
|---|---|---|---|
| A0 (18,17,18) original | **0/5** ✓ igual ao reportado | 0/5 + 1 UNKNOWN | — |
| A1 (37,13,3) in-suporte | **3/5** (t298 vacuous, t1812 ✓, t1955 ✓) | **2/5** + 1 UNKNOWN | **"0/5" estava ERRADO** |
| A2 (48,5,0) fora-de-piso | **5/5** ✓ igual ao reportado | **4/5** + 1 UNKNOWN (t298) | "5/5" vale só no frozen |

### 1.1 A02 — causa raiz do "0/5" errado (METRIC_BUG, §5.9)

`oracle_recalibrated_v2/support_extension/stage1_validate.py` julgou o E1 com **três métricas divergentes das congeladas**: (i) eMBB por `thr_min` (lowest-user) em vez da **mediana** R36 — daí o intervalo "thr 0,0–3,42" citado no relatório de encerramento; (ii) URLLC por `miss ≤ 5%` em vez de `p95 ≤ 10 ms` R37; (iii) mMTC por PDR de janela 100 ms em vez do rolling 1 s R38. Além disso marcou `replay_valid=False` nas 5 linhas por **falso-negativo do check de banner** (o runner v2 não imprime o formato procurado; arquivos presentes, `a_exec=[37,13,3]` exata) — e `confusion_matrix_3way_v2` **usou as linhas sem filtrar por replay_valid**. Os replays em si são íntegros (prefixos bit-idênticos); a falha é de métrica e de processo, não de execução.

### 1.2 A01/G0A-4 — o mMTC sob `a_m=0` (atenção crítica §5.3)

Coorte estrita da janela (100 ms): **tx=0 em TODAS as 15 células** (mMTC = 1 pkt/s/UE; uma janela de 100 ms contém tx só ~1 em cada 10 janelas). Consequências: (i) o PDR rolling = 1,0 dos 4 resgates A2 com demanda é **carregado por pacotes pré-t** (1–3 pacotes do 1 s anterior, transmitidos sob a ação antiga); (ii) sob evidência estrita in-window, o mMTC é `NO_ACTIVE_DEMAND`/UNKNOWN nas 5 células A2 ⇒ **strict joint = 0/5 PASS** (4 UNKNOWN por mMTC-sem-evidência-in-window + 1 UNKNOWN duplo em t298); (iii) a evidência positiva disponível sobre `a_m=0` aponta o contrário: o único tx in-window sob a_m=0 (t1423, fora dos 5) foi **perdido (PDR 100 ms = 0,0)**, e o D6 (II-d) mediu `a_m_min = 1`. **Leitura honesta do "5/5"**: 5/5 no eMBB e 4/5 no URLLC (coorte in-window); o mMTC não é testável in-window nesses 5 estados — a alegação de recuperação conjunta depende da semântica rolling congelada (legítima como regra pré-registrada, mas deve ser divulgada como portada por histórico).

## 2. R19 (A04/A05) — reconciliação exata

`24.482 total · 12.075 completas · 12.407 não-completas (50,68%)` — **exato** a partir do CSV fonte, por filtro `obs_partial`. Por λ (não-completas): **λ2 85,6% · λ5 48,2% · λ6 25,1%** — o parecer leu corretamente (86/48/25); o rótulo "~100%" que constava na legenda da galeria de figuras estava **errado** (corrigido). Ausência por slice nas não-completas (sobrepostas): eMBB 6.600 · URLLC 5.132 · mMTC 5.779. Por seed: s1201 100,0% · s1202 64,5% · s1203 96,7% · s1204 67,8% · s1205 61,2% · s1206 14,0% · s1207 18,0% · s1208 37,8% · s1209 19,3%. G4-A histórico reconciliado: 8.592/1.041 (12,1% completas) ✓. Unidade de inferência estatística continua sendo seed/run (autocorrelação de janelas).

## 3. C=53 (A06) — unidade rastreada até o MAC

`resource_unit_manifest.md` (03/11, resolvido por leitura de código) **confirma**: C = 53 = **53 RBGs = 53 PRBs** — `m_rbNum = floor(10 MHz × 0,96 / 180 kHz) = 53`; `NumRbPerRbg = 1` (default do NrHelper, não recalculado por 38.214); `bw_in_rbg = 53`; telemetria `total_rbg=53, total_prb=53` (spot-check feito nesta auditoria em `A_lam6_s3/sqr_prb_windows.csv`). Cadeia Python→IPC→C++: `dedicatedPRB = 100·a_i/53` → `raw_i = pct/100 × totalRbg = a_i` exato (`ComputeRbgQuotas`). Semântica `ALLOCATED_USAGE` (Σ ≤ C, 27.720 ações; ocioso nunca redistribuído). **Divergência narrativa R01** ("13 RBGs × 4 + 1") refutada e documentada no manifest — pendente de emenda no register.

## 4. URLLC (A07/G0A-7) — coorte causal vs rolling

A curva a_u (fig3) foi recomputada dos 6 replays G4 com o parser congelado de **janela de chegada** (= coorte causal pós-override): **bit-idêntica** ao reportado (miss 5,0–5,5% plano em a_u 13→35; p95 9,15–10,03 ms). O rolling 1 s aparece SOMENTE como agregação de observação dos rótulos G4-B (nunca como prazo). Colapso a_u=0: REPORTED (G3 II-c: D_urllc_obs ≈ 8,8–9,2; t1423 PDR100=0 corrobora por contraste) — não recomputado nesta auditoria. **Atribuição "TDD/scheduler"**: a evidência suporta apenas *insensibilidade a a_u no interior* (13→35) e colapso em 0; a causa física fina (slot TDD vs decisão do PF) fica `MECHANISM_NOT_IDENTIFIED` (sem traces sub-slot) — revisado o texto das figuras/relatórios onde dizia "scheduler/TDD-dominada" sem qualificação.

## 5. Revisões de interpretação (A08/A10/A12)

- **F/M/I**: permanecem CANDIDATE (gate v2 STOP). Nota: o check R1/C3 usou o "0/5" do E1 como justificativa — **o número muda para 3/5 frozen**; a classificação INFEASIBLE do oracle nesses estados permanece o fato do gate, mas a narrativa "recuperação impossível in-suporte" cai.
- **RLF≈0/rádio saudável**: REPORTED multi-campanha (r290s; rlf_count=0 por replay nas 15 células).
- **Figuras**: fig2 **revisada por mudança de dado** (linha A1: ✓V ✓ ✓ ✗ ✗ + revelação do mMTC) em `audit_v2/` com originais preservados; fig4/6/7 regeradas com `bbox_inches='tight'` (clipping de rodapé era cosmético). Nenhum denominator/SLA alterado para "fazer bater".

## 6. GATE D0-A

Ver `D0A_GATE.json`. **D0A_PASS com resultados científicos revisados** (§5.11: a integridade dos dados/replays prevalece; os labels foram corrigidos honestamente). Propagação das revisões (figuras, relatórios, PLANO, hipóteses) é trabalho de D0-B. **Nada de D1/D2/D3 até D0-B PASS.**
