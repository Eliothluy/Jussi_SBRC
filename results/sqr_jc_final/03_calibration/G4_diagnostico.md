# G4 — Diagnóstico e sweep §14 (orquestrador, 03–04/11)

## 1. STOP original: BUG de implementação (corrigido, §82)
As cabeças q25/q75 comparavam y com **f_safe_before** (previsão da cabeça safe, τ≈0,01–0,05 — sub-prevê por design) em vez das próprias previsões ⇒ atualização positiva crônica ⇒ quantis degenerados a quase-máximos (cobertura q25 ≈ 0,97 vs alvo 0,25). Fix: pinball por cabeça (`f_q25_before`/`f_q75_before`), comentário no código, suíte 342 passed. CAL reprocessado OFFLINE (mesmos dados — §82: bug de implementação, não recalibração).

## 2. Pós-fix (ls=1.0 auditado): cobertura na vizinhança do alvo
| slice | q25 (alvo 0,25) | q75 (alvo 0,75) |
|---|---|---|
| eMBB | 0,18 | 0,70 |
| URLLC | 0,23 | **0,75** |
| mMTC | **0,25** | **0,75** |

Erros absolutos TODOS ≤ 0,10 (critério nominal §16 ✓). FAIL restante: alvo fora do CI bootstrap (block CIs muito estreitos quando a cobertura é ~constante por janela — as bandas [0,16–0,19] não contêm 0,25). Interpretação honesta: o eMBB tem cobertura sistematicamente 0,17–0,18 no q25 — um viés pequeno (~0,07) mas consistente, não ruído de amostra.

## 3. Sweep §14 (grid autorizado ls × {0.5, 1.0, 2.0})
- ls=0.5: eMBB piora levemente (q25 0,16; q75 0,62–0,65). URLLC/mMTC idem 1.0.
- ls=1.0: melhor até agora (tabela acima).
- ls=2.0: em execução (fila após benchmark de ls=0.5).

## 4. Decisão sobre o G4 (a formalizar após ls=2.0)
O protocolo §16 manda STOP se "nenhum threshold satisfizer". Opções:
(a) Se ls=2.0 melhorar o eMBB para dentro do CI: escolher ls+thr, FREEZE, G4=PASS;
(b) Senão: registrar que o critério de CI do §16 é indevidamente estrito para cobertura ~determinística (variância inter-janela ≈ 0 ⇒ CI degenerate) e LEVAR AO ORIENTADOR a decisão entre: relaxar para o critério de erro absoluto ≤0,10 (já satisfeito) ou revisar features (§16 manda "revisar features/kernel" — ex.: incluir features de curva/prev-KPI que capturem a assimetria da margem eMBB).
Nada de ajustar thresholds pós-hoc: decisão registrada ANTES de ver ls=2.0.

## 5. RESULTADO do sweep (04/11, ~00:30) — ls=2.0 é o melhor; FAIL só pelo CI estreito

| config | eMBB q25/q75 | URLLC | mMTC | erro máx q25/q75 |
|---|---|---|---|---|
| ls=0.5 | 0,16/0,62 | 0,23/0,73 | 0,25/0,75 | 0,087/0,127 |
| ls=1.0 | 0,18/0,68 | 0,23/0,74 | 0,25/0,75 | 0,074/0,070 |
| **ls=2.0** | **0,19/0,72** | **0,23/0,75** | **0,24/0,74** | **0,062/0,039** |

ls=2.0 thr≥70: **erro absoluto máximo ≤ 0,063 em TODAS as cabeças/slices** (critério nominal |cov−τ|≤0,10 satisfeito com folga de 37%). FAIL restante exclusivamente por "alvo fora do CI bootstrap" — os CIs têm meia-largura ~0,01–0,014 (ex.: eMBB q25 [0,174; 0,201] não contém 0,25). Isso é um viés pequeno (~0,05–0,06) mas REAL do learner online nesses dados (não é problema de amostra: 23.720 janelas). Benchmark exaustivo ls=2.0: p95 = 14,7 s (G5 ⇒ coarse-to-fine, como nos outros).

## 6. DECISÃO REGISTRADA (regra pré-fixada no §4 aplicada ao item (b))

Conforme registrado ANTES de ver ls=2.0: o FAIL do G4 não é de threshold nem de lengthscale — é um **viés de calibração do pinball online** (~0,06 no q25 do eMBB; clássico de gradientes com η fixo e dados autocorrelacionados). As opções do §16 ("revisar features/kernel") foram esgotadas no grid autorizado (§14: só {0,5,1,2}, uma etapa). Alterar mais (η, τ, features) violaria §80/§81 sem ser bug.

**Estado G4: STOP formal mantido — LEVADO AO ORIENTADOR** com duas opções tecnicamente fundamentadas:
1. **Prosseguir com ls=2.0/thr=90 FREEZE** aceitando o critério de erro absoluto (≤0,10 §16 nominal; medido ≤0,063), documentando o CI-bootstrap como indevidamente estrito para cobertura ~determinística por janela (variância inter-bloco ≈ 0 ⇒ CI degenerado). O viés de −0,06 no q25 é CONSERVADOR para a função dos quantis (superestima a incerteza ⇒ mais MARGINAL, menos falsa confiança — direção segura para o SQR-JC);
2. **STOP para revisão de features** (ex.: features de curva/prev-KPI do eMBB) — custa re-implementação + re-CAL, dentro do espírito §16.
