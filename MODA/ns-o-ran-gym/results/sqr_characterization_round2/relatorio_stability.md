# Rodada 2 — Etapa 2.0: Estabilidade temporal (relatório)

**Data:** 19/10/2026 · **Protocolo:** MD do orientador §4 (`docs/artigo/rodada_2_fronteira_viabilidade_sqr.md`) · **Execução:** λ∈{2,5,7}×seeds 1–3×600 s, mesmo rádio congelado da Rodada 1b (300 m, 30 dBm, 10 MHz/C=53, UMa, omni 15 dBi, TDD 3DL/7, μ=0, PF, RLC UM, greedy 100 Mbps/UE, sessão Exp 60 s, equal-split, network-only). Warm-up 0–180 s tratado na análise. 9/9 runs íntegros (5999 janelas, RLF=0, SNR p5 17–24 dB).

## Resultado (médias por λ; violação por UE-janela c/ demanda)

| λ | 180–390s | 390–600s | 180–600s | E[N] | util D2 | thr med/UE | SNR p5 |
|---|---|---|---|---|---|---|---|
| 2 | 0,0% | 0,4% | 0,1% | 1,6 | 28–36% | 6,7–7,5 Mbps | 17–24 dB |
| 5 | 18,2% | 22,8% | 20,6% | 4,3 | 71–79% | 2,4–2,7 | ~21 dB |
| 7 | 34,3% | 45,1% | 40,3% | 6,5–6,8 | 92% | 1,5–1,9 | ~21 dB |

Por seed, violação 390–600s: λ5 = {39,1 · 17,7 · 11,7}% · λ7 = {49,7 · 37,1 · 48,6}%.

## Veredicto do critério §4.4

1. **Ordenação dos regimes: MANTIDA** — λ2 leve/normal, λ5 marginal, λ7 overload candidato. ✓
2. **Estabilidade pós-warm-up: λ2 ✓; λ5/λ7 ✗** — a violação não converge para média estável em 600 s (deltas entre metades: λ5 = +20/−6/−0 pp; λ7 = −5/+11/+27 pp) e a variância entre seeds é grande (λ5: 12–39%).

## Causa identificada (não é warm-up, não é rádio)

- SNR p5 e E[N] estáveis entre metades; RLF=0 → o crescimento não é drift de canal nem de enchimento.
- Série por minuto (λ5_s1): violação ~0% enquanto N=5–9; no minuto 4 um **burst de chegadas** leva N a 15 → thr agregado cai à metade, violação dispara (16→28→55%) e **a recuperação é lenta** (backlog de fila RLC ~3,7 MB/UE drena por minutos). Entre 180–600 s há poucos bursts por run → a média fica dependente da sorte da seed.

**Interpretação:** na banda λ5–λ7 o cenário é intrinsecamente **episódico**: a fronteira de viabilidade é atravessada por flutuação de chegada (Poisson) e o retorno é amortecido pelo backlog. Isto (i) explica a não-estacionariedade, (ii) é exatamente o regime de "overload transitório" que o MD §25 define como região principal do novo artigo — agora com evidência de que ele emerge **espontaneamente** do churn, e (iii) valida o desenho da Etapa 2.5 (episódio controlado L_N→L_O→L_N).

## Recomendações (aplicação do §4.4 — "revisar método de amostragem", não a simulação)

1. **Não alongar T_total** — a não-estacionariedade não é warm-up; mais tempo não reduz a variância episódica (custo ×2, benefício ~0).
2. **Estatística por episódio/janela, não por run**: reportar distribuição entre seeds + segmentar janelas por N(t) efetivo (ex.: N≤5 / 6–9 / ≥10) — a caracterização 2.3/2.4 deve usar N(t) como variável de estado, não só λ nominal.
3. **Violação suavizada** (média móvel de 1 s sobre UE-janela) ao lado da de 100 ms — separa flutuação de agendamento de déficit sustentado (o thr por janela de 100 ms oscila fortemente sob PF).
4. **Mais seeds na transição** (5+) se a granularidade por run for inevitável em alguma análise.
5. Para 2.5: medir **tempo de entrada em overload e tempo de recuperação** por episódio (o backlog é o amortecedor — candidato a discussão de anti-windup no SQR-JC, MD §30 Caso B).

## Artefatos

`results/sqr_characterization_round2/`: `stability/lam{2,5,7}_s{1,2,3}/` (+`.done`), `round2_stability.csv` (por λ×seed×segmento), `round2_stability_by_seg.csv`, `round2_stability_summary.csv`, `analyze_stability.py`, `run_stability.sh` (retomável). Custo: ~2,5 h de parede (λ7_s1 mais lento: 2h23, churn de logging `unmapped_rntis` ~8 M linhas/run — inofensivo, mas candidato a gate de verbosidade se incomodar).
