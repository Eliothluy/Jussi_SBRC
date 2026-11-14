# D1_projection_diagnostic.md — v2 (CANÔNICA: bootstrap-PROP §3.4)

**Protocolo:** v2 §7 / PROTOCOL_FROZEN §3.4 (decisão do orientador 09/11) · **Data:** 09–10/11/2026 · **Runs:** 9/9 B0 (SQR-HET) λ{2,5,6} × seeds {2101,2102,2103}, **bootstrap-PROP compartilhado 0–100 s** (controlador prop R39 decide as primeiras 1.000 janelas; o learner SQR-HET observa e atualiza sobre esses outcomes; B0 assume em t=100 s) — smoke de validação: 200 janelas PROP exatas + transição limpa. Flags congeladas idênticas. Cold-start anterior preservado como **exploratório** (§final).

## Esclarecimentos pedidos (orientador, 09/11)

1. **26.955 vs 26.946:** 26.955 = 9 runs × 2.995 decisões; a última decisão de cada run governa uma janela ALÉM do fim do sim ⇒ 9 linhas `end_of_run` (missing_reason, sem outcome) ⇒ 26.946 com outcome. Na v2 (bootstrap): 9.000 decisões PROP + **17.955 decisões do próprio B0** (17.946 com outcome, 99,9%).
2. **Denominadores dos SLAs por λ** (janelas B0; observadas = margem mensurável; excluídas = vacuous/R19 — NUNCA contadas como satisfeitas nas taxas):

| λ | B0 janelas | eMBB obs/excl | eMBB viol | URLLC obs/excl | URLLC viol | mMTC obs/excl | mMTC viol | all3-compl |
|---|---|---|---|---|---|---|---|---|
| 2 | 5.982 | 2.435/3.547 | 73,0% | 4.570/1.412 | 2,4% | 2.008/3.974 | 4,3% | 1.415 |
| 5 | 5.982 | 5.091/891 | 77,1% | 5.859/123 | 3,5% | 4.745/1.237 | 11,0% | 3.974 |
| 6 | 5.982 | 5.433/549 | 78,6% | 5.859/123 | 3,5% | 5.156/826 | 3,1% | 4.576 |

3. **Resultados por λ e seed:** tabela abaixo (P_projection, violação conjunta frozen, strict).

## §7.2 — estatísticas (decisões do B0 pós-bootstrap; n=17.946 com outcome)

| λ | seed | P_proj | viol_joint frozen | strict PASS |
|---|---|---|---|---|
| 2 | 2101 | 4,9% | 63,4% | 12,7% |
| 2 | 2102 | 2,4% | 3,6% | 0,0% |
| 2 | 2103 | 3,6% | 27,5% | 0,0% |
| 5 | 2101 | 5,7% | 80,4% | 15,0% |
| 5 | 2102 | 8,9% | 68,1% | 6,6% |
| 5 | 2103 | 10,4% | 54,0% | 19,9% |
| 6 | 2101 | 6,5% | 81,2% | 13,6% |
| 6 | 2102 | 8,0% | 77,1% | 8,0% |
| 6 | 2103 | 6,3% | 59,8% | 26,7% |

**1–2.** P_projection = **6,29%** (1.129 eventos; Σ propostas/C p95 1,06; corte mediano 9 RBGs). **3.** **43,0%** das decisões com projeção degradam min_i f_τ (Q2; sem atribuição causal). **4–5.** Violação por slice (denominadores acima): eMBB 73–79% · URLLC 2,4–3,5% · mMTC 3–11%. Matriz projeção×outcome (frozen): sem 42,3% all-OK (7.107/16.817) vs **com 50,1%** (566/1.129) — com bootstrap, as janelas projetadas têm outcome MELHOR que a base (inversão vs cold-start 39,8/55,3 — o learner aquecido projeta em estados diferentes; associação, não causal). Strict: com projeção 231 PASS/441 FAIL/457 UNKNOWN; sem 1.813/7.480/7.524. **6.** Estratificação por demanda em `projection_strata_demanda.csv`. **7.** sent = applied **17.946/17.946** (INV-03) · 0 janelas sem KPM.

**Estratos para D2 (populações B0):** S1 proj∧viol **563** (λ2/5/6: 74/242/247; por seed: todas ≥5) · S2 ¬proj∧viol **9.710** (dominado e-only 8.548; e+m 538; e+u 323) · S3 proj∧ok **566** (143/257/166) · S4 ¬proj∧ok **7.107** (3.954/1.687/1.466). all3-completas: S1 441 · S2 7.480 · S3 231 · S4 1.813.

## GATE D1 (v2, canônica): **PASS**

Logs íntegros (100% outcomes pós-bootstrap; INV-03 limpo; bootstrap verificado janela-a-janela) ✓ · Estados abundantes nos 4 estratos com ≥3 seeds e 3 λ ✓ · Q2 ativo (43% das projeções degradam a margem mínima prevista) ✓ · Denominadores declarados ✓.

## Cold-start (exploratório, preservado)

`d1_runs_coldstart_exploratory/` (9 runs, parser-v2, sem bootstrap): P_proj 3,87% · 41,3% degradação · eMBB 84,6% viol · com-proj 39,8% vs sem 55,3% all-OK. Diferenças principais do canônico: learner frio propõe menos (menos projeções), viola eMBB mais, e a associação projeção×outcome inverte. **Nenhum número do cold-start entra em D2/D3.**

## Proposta de amostragem D2 (para avaliação do gate — emenda 1 já congelada)

Ver `D2_sampling_proposal.md`. Resumo: **30 estados** (S1×12: 4/λ · S2×9: 3/λ sub-estratificados e-only×2+e+m×1 · S3×6: 2/λ · S4×3: 1/λ), seleção programática seed **20261108**, sem sobreposição >50%, seeds 2101–2103 rodiziadas por estrato×λ; **13 replays/estado** (a_B0 paridade/A-A + 12 alternativas de cobertura fixa: 4 redistribuições ±Δ{2,4,8,16} por assimetria de demanda + 3 vizinhanças de a_B0 + 4 aleatórias do simplex por SHA1(run‖win) + equal); modelos congelados do pré-screening (hashes na emenda) SEM vazimento (treino Rodada-2/CAL, disjunto de D1/D2). **~390 replays ≈ 20–35 h.**
