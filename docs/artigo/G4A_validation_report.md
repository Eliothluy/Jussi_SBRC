# G4A_validation_report.md — Emenda G4-A: veredito **STOP**

**Data:** 04/11/2026 · **Protocolo:** `docs/artigo/etapa_2_5_emenda_g4a_sqr_jc.md` (§22) · **Config:** ls=2.0 FROZEN, thr=90 FROZEN, split fit={1001,1002}/validation={1003} · **Artefatos:** `results/sqr_jc_final/03_calibration/g4a/` (g4a_verdict.json, g4a_validation_windows.csv, g4a_partial_count.json, logs).

## 1–3. Bug original, correção e suíte

Bug pinball-por-cabeça corrigido (q25/q75 comparavam y com `f_safe_before`; §82: BUG FIX, não retuning; detalhes em `03_calibration/G4_diagnostico.md` §1). Suíte: **342 passed** (com o fix). Commit: pendente de push (regra do projeto).

## 4–7. Cobertura pré/pós-fix e sweep

Pré-fix: q25 ≈ 0,97 (degenerado). Pós-fix + sweep {0,5; 1,0; 2,0}: **ls=2.0** melhor — erro máximo 0,063 ≤ 0,10 nominal. thr=90 congelado. Detalhes no G4_diagnostico.md §2–5.

## 8–9. Block-bootstrap

G4 original = **FAIL formal mantido** (alvo τ fora do CI bootstrap; viés residual reconhecido, não negado). G4-A acionado como substituto operacional.

## 10–12. Validação held-out e tabela FFR/FIR

Split por seed (§12): fit 6 runs (45.339 sigmas) · validation 3 runs (8.592 janelas; **7.551 excluídas por KPI vacuous** — regra R19 aplicada ao outcome: janelas sem as 3 margens observadas não são evidência; contagem reportada em g4a_partial_count.json). **1.041 janelas com outcome completo.**

| Predicted | Observed | n | Rate |
|---|---|---:|---|
| FEASIBLE | all-SLA-OK | 33 | — |
| FEASIBLE | violated | **9** | **FFR = 21,4%** ✗ |
| INFEASIBLE | violated | 99 | — |
| INFEASIBLE | all-SLA-OK | **58** | **FIR = 36,9%** ✗ |
| MARGINAL | any | 842 | (all-OK 57,4%) |

**Veredito §11: STOP** (FFR > 0,10 E FIR > 0,10).

## 13–14. Contagens e suporte

Confidentes: 199/1.041 (19%) — suficiente (Caso D não se aplica). OOS no fit-threshold: 0 na validação (o threshold do fit filtra bem o suporte).

## Perfil dos falsos (a análise que importa)

**FFR — 100% culpa da cauda URLLC (λ5):** as 9 violações são TODAS da margem URLLC (eMBB/mMTC ok em 9/9), e 7 das 9 são **rasas** (−0,003 a −0,015 normalizado = miss 5,1–5,2% vs alvo ≤5%); 2 moderadas (−0,10/−0,31/−0,41). Padrão idêntico ao da Rodada 2: **cauda scheduler/TDD-dominada** imprevisível por estado — o q25 da URLLC prevê OK e a cauda escapa na janela seguinte. Não é erro de calibração do eMBB.

**FIR — 100% λ5/λ6, eMBB super-fechando na alta carga:** os 58 falsos-INFEASIBLE têm margens eMBB confortáveis (média +0,84; mínima +0,013) — o classificador declara INFEASIBLE por q75-eMBB < 0 quando o outcome é all-OK. É o **mesmo viés residual da calibração** (q75 cobre 0,72 vs 0,75), agora com consequência operacional: na alta carga o modelo subestima o throughput atingível e declara inviável cedo demais.

**Ordenação preservada ✓:** all-OK decresce FEASIBLE (78,6%) > MARGINAL (57,4%) > INFEASIBLE (36,9%) — o detector ordena corretamente; a calibração das classes confidentes é que não atinge a precisão ≤10%.

## 15–16. Decisão

**G4-A = STOP** (Caso B+C simultâneos). Conforme §18 da emenda: proibido novo tuning; **STOP metodológico — nova versão de protocolo é decisão do orientador**. PILOT permanece BLOQUEADO.

## Diagnóstico estrutural para o orientador (o que uma G4-B precisaria resolver)

1. **URLLC (FFR):** o SLA miss≤5% por janela de 100 ms com tráfego 32B@1ms é **dominado pela cauda scheduler/TDD** (Rodada 2, G4 causal: a_u 13→35 não muda). Prever miss por janela por estado-ação é quase-impossível com o learner atual — as falsas-FEASIBLE são inerentes a essa meta-representação. Opções: (a) SLA de URLLC por janela deslizante de 1 s (suaviza a cauda; muda o sla_manifest — decisão metodológica); (b) margem de segurança ad-hoc no q25 da URLLC (viola §18 — exigiria emenda).
2. **eMBB (FIR):** o viés do q75 na alta carga (~3pp de cobertura) gera FIR de 37% — as bandas na alta carga são largas (C7 do II-d já mostrava). Refit com mais amostras de alta carga no fit (CAL desbalanceado: λ2 tem 4.046 janelas None e poucas margens) poderia estreitar — mas exigiria re-CAL com seeds de alta carga extras (permitido pelo §12 "é permitido rerodar CAL", porém §18 proíbe "novas features": rerodá-lo com os mesmos settings não é tuning).

## 16. Próximos passos

- PILOT bloqueado (§24 da emenda). Sem tuning adicional (§18).
- Decisão ao orientador: (i) emenda G4-B com SLA URLLC deslizante 1 s + re-CAL; (ii) re-CAL com seeds extras de λ5/λ6 para o fit (mesma config); (iii) STOP da Etapa 2.5 e redação do artigo com a caracterização da Rodada 2 + o aprendizado negativo honesto do G4-A.
