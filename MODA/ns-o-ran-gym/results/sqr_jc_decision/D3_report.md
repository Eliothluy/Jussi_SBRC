# D3_report.md — seleção conjunta pareada offline (B0 × B1 × equal): gate exploratório → STOP-JOINT

**Data:** 13/11/2026 · **Base:** D3_execution_declaration.md (4 cuidados) + D3_execution_declaration_v2.md + ofício pré-D3 · **Estados:** runs existentes da D1-v9, seed 2101 como teste (split por RUN, cuidado 2); exclusão dos 30 estados D2 ±5 janelas (121 janelas marcadas) · **PILOT/MAIN bloqueados.**

## 1. Conformidade e hashes (registrados em `d3_frozen_hashes.json`)

- **Checkpoints (cuidado 1 — congelados):** `het_lam2_s2101` `8a7a419aaf19bdc0` · `het_lam5_s2101` `b71461e47aa33c14` · `het_lam6_s2101` `976df0a3a59f72f4` — hash antes/depois idêntico (assert no `d3_decide.py`); zero retreino/ajuste; nenhum dado da D2 entra nos modelos (seletor consome apenas estado + modelos).
- **Código congelado:** `joint_selector.py` `3d719d7bcbbf3fd6` · `joint_state.py` `bac94bb07ebdfbef` · `sla_margin.py` `ebd9621d71f55412` · `replay_controller.py` `5c0c85f607920135` · `sqr_run.py` `a2df8b28dedcbe4f` · `d3_decide.py` `280a2e8c3b5a122b` (v2).
- **Espaço de ações (cuidado 3):** A_common = simplex Σ≤53 idêntico para os 3 braços.
- **Cegueira a rótulos (cuidado 4):** o seletor consome apenas (z_i, modelos, A_common); ranking lexicográfico congelado §5.4 (max critical → max efficient → min Σa → lex).
- **Independência (declarada):** a D3 usa estados distintos DENTRO de runs já existentes (mesma infraestrutura, seeds 2101); NÃO se afirma independência estatística entre todas as janelas — janelas vizinhas são autocorrelacionadas por construção; o split por RUN isola o conjunto de teste do desenvolvimento.

## 2. Passo 2 — decisões pareadas offline (8.488 janelas; seed 2101)

`d3_paired_offline.csv`: 2.628 + 2.924 + 2.936 pares (121 excluídos D2±5). **Resultado: o seletor congelado degenera** — b1≠a_B0 em 8.488/8.488; **b1=(0,0,0) em 83%** (λ2: 99,8% · λ5: 99,7% · λ6: 51,3%); o restante concentra-se no outro extremo (ex.: (0,53,0) — tudo para URLLC). Σa(B1) mediana 0 vs Σa(B0) 39–48.

## 3. Diagnóstico técnico (por que degenera)

O `f_τ` do regressor SQR é **K(x, memória) @ alphas** — um acumulador de correções pinball (sem baseline). Com kernel de banda larga (length_scale=1 em todas as 7 dims sobre z∈[0,1]), cada falha crítica insere α=−η·(τ−1)·P=**−0,198** com influência GLOBAL: ~37 falhas críticas eMBB (λ6) × −0,198 ≈ **offset −7,3** — reproduzido empiricamente: f(z_j, a_j) − y_j = −7,3 **nos próprios pontos de memória** (erro de calibração, não de estado). Consequências: (i) f quase constante em a (variação ≤1,8) e sistematicamente negativo — o modelo não mede o efeito causal de a na margem (correlação espúria demanda↔alocação dominante); (ii) o desempate congelado min Σa resolve os empates para a ação nula. **O seletor é fiel ao protocolo e ao modelo; o mecanismo congelado degenera pela estrutura do regressor online SQR.** (Correção do rótulo semântico do registro online: `NO_ACTIVE_DEMAND` → `PENDING_OBSERVATION` quando há demanda real ainda não flushada — ofício item 1; nenhum julgamento alterado.)

## 4. Passo 3 — replays confirmatórios (amostra mínima, 30 replays)

Redução do orçamento congelado (200–400 → 30) **justificada pelo gate estrutural do passo 2** — amostra pré-registrada (seed 20261113): 6 estados B0-FAIL, 6 controles B0-PASS (todos λ6s2101, único run com cobertura 3×COMPLETE nos runs de teste), 3 estados de seletor não-degenerado (sa>0); ramos B1 + equal por estado. **A/A: 30/30 prefixos bit-exact** (paridade válida).

| par (B0 → B1 → equal) | contagem |
|---|---|
| PASS → **FAIL (b1)** → PASS (5/6 equal) | 6 controles: **Harm 6/6 do B1** (colapso: eMBB 0–1,1 Mbps; p95 URLLC 99–105 ms) · equal recupera 5/6 |
| FAIL → FAIL (b1) → FAIL (equal) | 6 estados violados: **Rescue 0/6** por ambos |
| UNKNOWN → UNKNOWN → UNKNOWN | 3 estados ND (b1=(0,53,0)) |

**Gate exploratório §5.7: (a) Δ_allOK>0 FALHA** (0 resgates, 6 prejuízos; NetRescue=−6) · **(b) Harm<Rescue FALHA** (6>0) · **(c) sem degradação sistemática FALHA** (fome total nas ações (0,0,0)) · (d) não aplicável. → **STOP-JOINT.**

## 5. Avaliação científica

1. **A lacuna (hipótese) permanece viva:** a D2 mostrou 4/9 estados violados com recuperação admissível — alocações melhores EXISTEM no espaço comum. O que falha é o MECANISMO: o ranking lexicográfico sobre os f_τ dos modelos SQR-HET originais não identifica essas alocações — degenera antes.
2. **O controle equal foi decisivo (ofício item 3):** equal recupera 5/6 controles e não resgata os FAIL — na comparação pareada, equal ≫ B1; a seleção conjunta congelada não oferece vantagem sequer sobre a redistribuição estática.
3. **Nenhuma melhora → STOP-JOINT sem ajuste da política** (vedação respeitada: zero alteração em modelos, ranking, desempate, SLAs, R19, espaço — a política congelada foi avaliada e parou no gate; os diagnósticos acima são para decisão do orientador, não mudanças implementadas).
4. **Caminhos possíveis (trabalho futuro — exigem nova decisão metodológica, não executados):** (i) ranking por **deltas** f_τ(a)−f_τ(a_ref) com a_ref da demanda projetada (comparação causal, imune ao offset global); (ii) modelos com f(a) causal (ex.: condicionamento por demanda ou treinamento com ações aleatórias de sondagem); (iii) seletor híbrido (projeção de demanda + verificação conjunta de margens). Todos alteram a política congelada e exigem novo ciclo D0-B→D3.
5. **Limitações declaradas:** estados dentro de runs existentes (sem independência estatística entre todas as janelas); STRICT-avaliável nos runs de teste concentrado em λ6s2101 (prevalência de 3-slices); amostra confirmatória mínima (30) por decisão de gate — o sinal era estrutural e consistente no passo 2 (8.488 janelas).

**Veredito: STOP-JOINT** — a contribuição candidata (Joint Action Selection sobre os modelos quantílicos do SQR-HET, ranking lexicográfico congelado) **não supera o gate exploratório**; a cadeia D1→D2→D3 está fechada com: (i) pipeline íntegro e auditado, (ii) evidência de existência de oportunidades (D2), (iii) diagnóstico causal da falha do mecanismo (D3). PILOT e MAIN permanecem bloqueados.
