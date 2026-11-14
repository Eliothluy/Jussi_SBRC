# G4B_validation_report.md — Emenda G4-B: veredito **STOP FINAL (marginal)**

**Data:** 04/11/2026 · **Protocolo:** `docs/artigo/ETAPA_2_5_G4B_SQR_JC.md` (§§1–21) · **Artefatos:** `results/sqr_jc_final/03_calibration/g4b/` (g4b_verdict.json, g4b_validation_windows.csv, g4b_validate.log, cal_extra_runs/, g4b_val_runs/) · **Pipeline:** `g4b_validate.py` (réplica A crosscheckada bit-a-bit).

## 1. Referência ao FAIL do G4-A (item 1)

G4-A (04/11) = STOP: FFR 21,4% (9/42; 100% cauda URLLC λ5, 7/9 rasas miss 5,1–5,2%) e FIR 36,9% (58/157; 100% λ5/λ6, q75-eMBB super-fecha na alta carga) em `docs/artigo/G4A_validation_report.md`. A G4-B autorizou exatamente duas correções: (A) URLLC rolling 1 s; (B) CAL_EXTRA_FIT λ5/λ6.

## 2. Configuração congelada (item 2) — nada além do autorizado mudou

Kernel Matérn **ls=2.0**, **thr=90** (percentil; valor recalculado no σ do fit: A=0,4289922671, B=0,3879764351), q25/q75=0,25/0,75, eta/tau, feature set (23 dims), normalização (**scaler congelado do CAL** — p1/p99 só de contagens/filas/ofertas; as dims URLLC têm escala natural /10 e [0,1], portanto o manifesto permanece exatamente compatível), SLAs eMBB/mMTC, cenário rádio, scheduler, TDD, período 100 ms. Runs gerados com as MESMAS flags do CAL congelado (PROP bootstrap §29, 300 s, slas 3,5, c_prbs 52, warmup 100 s, cellRadius 300).

## 3. Definição exata do rolling 1 s da URLLC (item 3)

p95 primário, miss secundário e p99 computados sobre o **pool de pacotes URLLC com janela de chegada em [w−9, w]** (10 janelas de decisão = 1000 ms), atualizado a cada decisão de 100 ms. A fórmula primária da margem permanece `m_u = (10 − p95_1s)/10` (sla_margin.py intocado); muda apenas a agregação. **Verificação empírica:** no pool, `p95_1s > 10 ms ⟺ miss_1s > 5%` em 100% das 1.949 janelas-1s ativas do run de referência (XOR = 0) — as duas leituras da emenda são a mesma condição. Deadline por pacote = 10 ms e miss threshold = 5% **inalterados** (não é "deadline de 1 s"). Features de estado 7/21/22 (p95/miss/p99) usam o mesmo pool — coerência estado-rótulo ("URLLC risk observed over previous 1 s").

**Warmup §5:** janela com <1 s de histórico de stream (w < w_start+9) ⇒ KPIs URLLC None ⇒ **excluída de calibration, FFR, FIR e gate** (~9 janelas/run).

**Efeito mecanístico observado (importante):** a violação URLLC é *clusterizada* — o pooling 1 s **aumenta** a densidade de janelas violadas no λ6 (25,9%→40,5% no seed de referência s1003) porque cada burst contamina 10 janelas do pool; no λ5 suaviza pouco (13,9%→16,4%). O rolling reduz a variância do rótulo mas o torna mais denso em violações — o detector B declara INFEASIBLE com muito mais frequência (9.862 vs 6.739) e o MARGINAL desaba (1.491 vs 4.617).

## 4. Atualização do sla_manifest (item 4)

`src/nsoran/sqr_jc/sla_manifest.yaml` → **v1.1-g4b**: bloco `g4b` (§19: packet_deadline_ms 10,0; miss_threshold 0,05; aggregation_window_ms 1000; decision_period_ms 100; gate max_ffr/max_fir 0,10; status PENDING→**STOP_FINAL**) + adendo documentando a mudança de agregação sem alteração de deadline/threshold/fórmula.

## 5. Novas seeds CAL de λ=5/6 (item 5) — CAL_EXTRA_FIT

λ5 × {1101, 1102, 1103} · λ6 × {1104, 1105, 1106} — 6 runs novos, disjuntos de CAL {1001–1003}, PILOT {2001–2003}, MAIN {3001–3010} e da validação. Usados SOMENTE no fit (§7). Fit do detector B = CAL original válido (9 runs, incluindo s1003 — agora elegível porque a validação da G4-B usa seeds próprias, §8) + 6 extras = **15 runs**.

**Incidentes operacionais (registro honesto):** (i) kill manual meu do ns-3 filho do s1102 às 11:57 produziu saída rc=0 com KPM truncado (237/300 s) — `.done` falso-positivo; run quarentenado (`_quarentena/extra_lam5_s1102_TRUNCADO_1157`) e refeito; adicionado assert de completude ao pipeline (lição: nunca matar o filho ns-3 — só o timeout do wrapper). (ii) o seed 1102 é patologicamente lento (crawl determinístico a partir de ~200 s de sim; 3 timeouts de 30 min) — completado com orçamento de 90 min (TIMEOUT_S=5400). (iii) val_lam2_s1201 tem último KPM em 277,9 s porque todos os UEs saíram antes do fim — 2.995 decisões (299,5 s) com exit 0 ⇒ simulação COMPLETA (critério de completude corrigido para decisões+exit).

## 6. Seeds exclusivas de validação (item 6) — G4B_VALIDATION

λ2 × {1201, 1202, 1203} · λ5 × {1204, 1205, 1206} · λ6 × {1207, 1208, 1209} — 9 runs novos, **nunca usados em fit, threshold ou qualquer escolha**; PILOT/MAIN proibidos (§9) e não usados. Ambos os detectores (A e B) avaliados nas MESMAS 9 seeds — comparação antes/depois limpa.

## 7. Coverage q25/q75 na validação (item 7)

| Detector | slice | q25 cov | err | q75 cov | err |
|---|---|---|---|---|---|
| A (100 ms) | eMBB | 0,157 | 0,093 | 0,606 | 0,144 |
| A | URLLC | 0,210 | 0,040 | 0,734 | 0,016 |
| A | mMTC | 0,160 | 0,090 | 0,676 | 0,074 |
| **B (1 s)** | eMBB | 0,116 | 0,134 | 0,529 | 0,221 |
| **B** | URLLC | 0,172 | 0,078 | 0,696 | 0,054 |
| **B** | mMTC | 0,170 | 0,080 | 0,660 | 0,090 |

(n = 12.066/12.022 janelas confiantes com margem observada.) Subcobertura generalizada do q75 (bandas otimistas demais ⇒ declara INFEASIBLE cedo) e do q25 — o mesmo viés residual do learner já documentado no G4-A §8–9; §15 explicita que cobertura perfeita não é requisito. No eMBB a cobertura piorou no B (0,61→0,53) mesmo com mais dados de alta carga — o rótulo URLLC mais denso desloca a mistura de estados das janelas confiantes.

## 8. FFR (item 8) — **11,1% > 10% ✗**

Detector B, outcomes completos (R19: 12.407 janelas vacuous/parciais excluídas de 24.482; 12.075 completas), classes confidentes (OOS = 9): **FFR = 79/713 = 11,08%**. CI 95% binomial ≈ [8,8%, 13,3%]. Por λ: λ2 6,7% (12/179) · λ5 12,9% (48/371) · λ6 11,7% (19/163).

## 9. FIR (item 9) — **9,5% ≤ 10% ✓**

**FIR = 938/9.862 = 9,52%**. Por λ: λ2 7,4% · **λ5 1,3%** (o núcleo do G4-A foi eliminado pelo CAL_EXTRA + rolling) · λ6 14,9%.

## 10. Tabela FEASIBLE/MARGINAL/INFEASIBLE (item 10)

| Detector | Predição | Observado | n | Rate |
|---|---|---|---:|---|
| **B (G4-B)** | FEASIBLE | all-SLA-OK | 634 | 88,9% |
| **B** | FEASIBLE | violado | **79** | **FFR = 11,1%** ✗ |
| **B** | INFEASIBLE | violado | 8.924 | 90,5% |
| **B** | INFEASIBLE | all-OK | **938** | **FIR = 9,5%** ✓ |
| **B** | MARGINAL | any | 1.491 | all-OK 59,1% |
| A (réplica G4-A, mesmas seeds) | FEASIBLE | violado | 116/666 | FFR = 17,4% |
| A | INFEASIBLE | all-OK | 857/6.739 | FIR = 12,7% |
| A | MARGINAL | any | 4.617 | all-OK 52,3% |

**Ordenação esperada §12 ✓ (B):** P(all-OK|F) 0,889 > P(all-OK|M) 0,591 > P(all-OK|I) 0,095 — o detector ordena corretamente e é muito mais decidido (MARGINAL 1.491 vs 4.617).

## 11. Diagnóstico separado URLLC / eMBB (item 11)

**URLLC (§13, λ5/6):** falsas-FEASIBLE do B: 67 (62 causadas por URLLC) vs 84 (80) no A — melhora de ~20%, não resolve. Distribuição miss_1s das 67: **metade rasa (34 com miss 5–6%)**, caudas 7,5–45,5%; p95_1s tipicamente 10,0–11,1 ms. É o mesmo mecanismo do G4-A: **cauda scheduler/TDD imprevisível por estado-ação** — o q25 prevê OK e o pool-1s escapa por poucos décimos de ms. A agregação temporal reduziu a variância (FFR caiu 21,4→11,1) mas não elimina a cauda; confirmado que **não é erro do deadline** (10 ms intacto).
Comparação janela 100 ms vs rolling 1 s (mesmas seeds, mesmas classes): FFR λ5 22,7%→12,9% · λ6 12,5%→11,7% · λ2 18,4%→6,7%.

**eMBB (§14, λ5/6):** falsas-INFEASIBLE: 799 (A) → 897 (B) em valor absoluto, mas a classe INFEASIBLE cresceu 6.556→9.305 (rótulo URLLC mais denso) e a **taxa caiu 12,2%→9,6%** (G4-A histórico: 36,9% — queda de 74%). Todas as 938 falsas-I do B têm margem eMBB observada ≥0 (média +2,44; mín +0,048) — o INFEASIBLE conjunta continua declarado pelo q75-eMBB super-fechando (subcobertura q75 0,53), agora compensado pela maior base. Suporte do fit B: 2.000 SV/slice (memória saturada — os runs EXTRA de alta carga dominam a cauda do buffer).

## 12. Comparação G4-A vs G4-B (item 12)

| Métrica | G4-A (s1003) | réplica A (seeds novas) | **G4-B (seeds novas)** | Δ G4-A→G4-B |
|---|---|---|---|---|
| FFR | 21,4% | 17,4% | **11,1%** | −48% |
| FIR | 36,9% | 12,7% | **9,5%** | −74% |
| Ordenação F>M>I | ✓ | ✓ | **✓** | — |
| FIR λ5 | (dominante) | 9,1% | **1,3%** | núcleo eliminado |
| MARGINAL (decisão) | 842 | 4.617 | **1.491** | detector mais decidido |

As duas alterações autorizadas produziram exatamente os efeitos mecânicos pretendidos (suavização da cauda URLLC; cobertura de alta carga para o eMBB). O FFR residual concentra-se na cauda rasa (miss 5–6%) da URLLC em λ5/λ6.

## 13. Decisão PASS/STOP (item 13)

**G4-B = STOP FINAL.** §12 exige FFR ≤ 0,10 **E** FIR ≤ 0,10: FFR = 0,1108 > 0,10 (FIR ✓ 0,0952). A falha é **marginal** (+1,1 pp; CI [8,8%, 13,3%] contém 10%), não estrutural como no G4-A — mas o gate é pontual e pré-registrado. Conforme §17: **não criar G4-C**; não testar novas features/thresholds/eta/quantis/kernels; decisão ao orientador.

## 14. G5/PILOT (item 14) — **BLOQUEADOS**

G5 (coarse-to-fine §38) **não liberado** (só após G4-B PASS, §16). PILOT **permanece bloqueado** (§0/§16). Congelamento definitivo previsto no §16 NÃO se aplica (é condicional ao PASS).

---

## Diagnóstico final para o orientador (§17)

1. **O que a G4-B demonstrou:** (i) o FIR do G4-A era de fato carência de dados de alta carga — CAL_EXTRA λ5/λ6 + rolling derrubaram o falsos-INFEASIBLE de λ5 para 1,3% e o agregado para 9,5% (gate ✓); (ii) a agregação rolling 1 s reduz substancialmente a variância da cauda URLLC (FFR 21,4→11,1%) sem tocar no deadline de 10 ms; (iii) o detector 3-way com as mudanças é **operacionalmente mais decidido** (MARGINAL 842→1.491 nas comparações mesmo-base) e preserva a ordenação das classes.
2. **O que nenhuma das duas alterações resolve:** a **cauda rasa scheduler/TDD da URLLC** (miss 5–6% vs alvo 5%) — 72 das 79 falsas-FEASIBLE; metade rasa. É o mesmo fenômeno do G4-A e do G4 causal da Rodada 2 (a_u 13→35 não muda a cauda): **resíduo previsível-impossível com a meta-representação estado-ação atual**. O gate ≤10% exige essencialmente previsão perfeita dessa cauda rasa em λ5/λ6.
3. **Estado das opções (todas exigem decisão do orientador; nada é executado automaticamente):** (a) STOP da Etapa 2.5 e artigo com a caracterização da Rodada 2 + o aprendizado negativo agora mais forte (G4-A→G4-B mostra o piso do método: FFR ~11% estruturalmente ligado à cauda rasa; FIR resolvido); (b) nova emenda de protocolo (exclusivamente decisão do orientador — ex. gate com CI, SLA URLLC em percentil de sessão, ou mudança de meta-representação — cada uma reabre congelamentos); (c) qualquer outra direção que o orientador indique. O FIR ≤10% e a ordenação preservada são resultados positivos honestos para o artigo independentemente da decisão.
