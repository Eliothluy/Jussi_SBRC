# preflight_checklist.md — Checklist §96 para autorizar PILOT

**Etapa 2.5 SQR-JC — protocolo FINAL · preenchido em 2026-11-03 pelo Agente B (PRE-FLIGHT).**
Regra do §96: "Se qualquer item falhar: NÃO RODAR PILOT". Itens marcados **AGUARDANDO EXECUÇÃO DO ORQUESTRADOR** dependem de runs ns-3 que este agente NÃO executa (despacho: "o orquestrador roda o G3 e o benchmark externamente").

## Gates G0–G6

| Gate | Item §96 | Status | Evidência |
|---|---|---|---|
| **G0** | SQR audit concluído | ✅ **PASS** | `00_audit/sqr_original_audit.md` — 17 campos do §3 lidos do código (`src/nsoran/sqr/*.py`), conformidade com artigo verificada, divergências registradas |
| **G1** | Resource unit definido | ✅ **PASS** | `../01_resource_unit/resource_unit_manifest.md` — C=53 provado por código C++ + telemetria (1 RBG = 1 PRB/slot DL); ACTION_SEMANTICS = ALLOCATED_USAGE (27.720 combos); teste `test_resource_unit_manifest_complete` + `test_action_semantics` PASS |
| **G1** | Action semantics definido | ✅ **PASS** | idem; `ACTION_SEMANTICS="ALLOCATED_USAGE"` congelado em `joint_action_space.py` |
| **G2** | SLA manifest congelado | ✅ **PASS** | `src/nsoran/sqr_jc/sla_manifest.yaml` — extraído do código congelado da Rodada 2 (oracle/dataset.py:71-73, adjudication._sla_oks); **divergência registrada**: despacho pedia miss≤5%, o Gate II-d usou p95≤10 ms (adjudication.py:404-412) — primária = p95 conforme §7 ("conforme o código auditado"); miss% = secundária report-only |
| **G4** | Feature scaler congelado | ⏳ **AGUARDANDO EXECUÇÃO DO ORQUESTRADOR** | mecanismo pronto e VALIDADO offline em run real: `examples/sqr_jc_calibrate.py` produziu `feature_scaler_manifest.yaml` (2965 janelas, /tmp/caltest_out); congelamento real após o CAL de 9 runs (`03_calibration/run_cal.sh`) |
| **G4** | Kernel congelado | ✅ **PASS (com ressalva §14)** | Matérn ν=1.5, ls=1.0 (auditoria G0) herdados; única mudança autorizada = grid §14 pós-CAL se a calibração falhar |
| **G4** | q25/q75 calibrados | ⏳ **AGUARDANDO EXECUÇÃO DO ORQUESTRADOR** | mecanismo §16 implementado (grid percentis 50–95, block bootstrap 5 s × 1000, erro ≤0.10, suporte ≥70%); validação indicativa em λ=2 única deu **STOP** (cobertura q25 ≈ 0.93 vs alvo 0.25 — cabe esperar: distribuição degenerada sem overload; o CAL real com λ∈{2,5,6} decide) |
| **G4** | sigma_support congelado | ⏳ **AGUARDANDO EXECUÇÃO DO ORQUESTRADOR** | idem (`sigma_support_manifest.yaml` gerado pelo `run_cal.sh`) |
| **G3** | rádio saudável | ⏳ **AGUARDANDO EXECUÇÃO DO ORQUESTRADOR** | `02_radio_sanity/run_radio_sanity.sh` pronto (λ2, seeds 1–3, 300 s, prop, flags congeladas, LC_ALL=C, .done, timeout 1800 s) + analisador `examples/sqr_jc_radio_sanity.py` (RLF/SNR/CQI/PHY → PASS/STOP §24) |
| — | decision period validado | ✅ **PASS** | 100 ms congelado (contrato §2); alinhamento ação k → janela k+1 → stamp (k+2)·100 ms implementado e testado (`test_decision_window_alignment`; convenções frozen de adjudication.py:343 e sqr-sim.cc:1545-1553) |
| **G5** | action search passou runtime gate | ⏳ **AGUARDANDO EXECUÇÃO DO ORQUESTRADOR** (com achado indicativo) | harness §37 validado em run real: busca exaustiva com ~1.079 SV/slice → **p95 ≈ 12,2 s ≫ 80 ms → FAIL esperado**; caminho coarse-to-fine (§38) implementado (step 4 ≈ 300 ms — também acima; parâmetros CAL-tunáveis, ex. step 8) — o gate real decide após o CAL; MARGINAL_UNRESOLVED (§39) cobre o runtime até lá |
| **G6** | SQR-HET validado | ✅ **PASS** (unit) | `sqr_het.py` (§27): reusa o `SqrSliceController` auditado sem modificação, margem comum Λ=0, projeção Π + R12 originais; SEM joint/floors/q25/q75 (testes `test_sqr_het_no_joint_machinery`, `test_sqr_het_slice_wise_plus_projection`); validação em loop fechado = pós-PILOT (G7) |
| — | logs validados | ✅ **PASS** | `logging.py` com as 46 colunas exatas do §56 + outcome CSV + kernel log §57; teste `test_logging_exact_columns` PASS |
| — | unit tests PASS | ✅ **PASS** | suíte `tests/run_sqr_tests.sh` COMPLETA: **339 passed** (inclui 64 testes SQR-JC novos: §75 completo + §76 qualitativos) — executada nesta sessão em 55,69 s |

## Checklist §96 literal

```text
[x] SQR audit concluído              (G0 — 00_audit/sqr_original_audit.md)
[x] Resource unit definido           (G1 — 01_resource_unit/resource_unit_manifest.md)
[x] Action semantics definido        (G1 — ALLOCATED_USAGE, Σa ≤ C)
[x] SLA manifest congelado           (G2 — src/nsoran/sqr_jc/sla_manifest.yaml)
[ ] Feature scaler congelado         (AGUARDANDO EXECUÇÃO DO ORQUESTRADOR — run_cal.sh)
[x] Kernel congelado                 (ν=1.5/ls=1.0 audited; grid §14 autorizado pós-CAL)
[ ] q25/q75 calibrados               (AGUARDANDO EXECUÇÃO DO ORQUESTRADOR — §16 no CAL)
[ ] sigma_support congelado          (AGUARDANDO EXECUÇÃO DO ORQUESTRADOR — §16 no CAL)
[ ] rádio saudável                   (AGUARDANDO EXECUÇÃO DO ORQUESTRADOR — run_radio_sanity.sh)
[x] decision period validado         (100 ms + alinhamento §23 testado)
[ ] action search passou runtime gate (AGUARDANDO EXECUÇÃO DO ORQUESTRADOR — indicativo: exaustivo FAIL, §38 necessário)
[x] SQR-HET validado                 (unit-level; loop fechado pós-PILOT)
[x] logs validados                   (§56/§57 exatos, testados)
[x] unit tests PASS                  (339 passed — suíte completa)
```

## Veredicto

**6 de 14 itens PASS com evidência executada nesta sessão; 5 itens AGUARDAM EXECUÇÃO DO ORQUESTRADOR** (G3 radio sanity, CAL/scalers/q25-q75/sigma_support, G5 benchmark — todos com scripts prontos e pipeline validado offline); **3 itens PASS com ressalvas registradas** (kernel com grid §14 pendente de necessidade, SQR-HET unit-level, SLA manifest com divergência miss/p95 documentada).

**NÃO INICIAR PILOT** até os 5 itens pendentes fecharem (§96: qualquer item falho ⇒ sem PILOT). Próximo passo do orquestrador:
1. `LC_ALL=C bash -c 'results/sqr_jc_final/02_radio_sanity/run_radio_sanity.sh'` → G3;
2. `LC_ALL=C bash -c 'results/sqr_jc_final/03_calibration/run_cal.sh'` → scalers + σ_support + q25/q75 + benchmark G5 (§37 exaustivo; se FAIL, validar §38 coarse-to-fine);
3. re-avaliar este checklist; só então PILOT (seeds 2001–2003, λ∈{2,5,6}).
