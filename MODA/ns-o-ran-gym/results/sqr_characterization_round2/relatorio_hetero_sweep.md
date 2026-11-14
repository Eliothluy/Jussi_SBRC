# Rodada 2 — Etapa 2.3: Sweep heterogêneo Perfil A, dois braços (relatório)

**Data:** 21–22/10/2026 · **Protocolo:** MD §12–16 · **Mandato:** equal-split (baseline estático) × prop (witness adaptativo), rádio/SLAs (R36–R39)/binário congelados. 30/30 runs (2 braços × λ_total ∈ {2,4,5,6,7} ÷ 3/slice × seeds 1–3 × 300 s, greedy/urllc/mmtc, 300 m). Figuras: `figures_hetero_sweep.png`; CSVs: `round2_hetero_sweep{,_by_lambda}.csv`.

## Resultado (média das 3 seeds)

| λ | viol eMBB eq→prop | thr/UE med eq→prop | URLLC miss eq→prop | util D2 eq→prop | mMTC PDR |
|---|---|---|---|---|---|
| 2 | 23,2 → **5,0** | 5,9 → 12,1 | 9,5 → 6,1 | 23 → 43% | 1,0 |
| 4 | 46,8 → **18,5** | 3,8 → 8,2 | 10,7 → 12,0 | 33 → 62% | 1,0 |
| 5 | 57,4 → **25,4** | 3,0 → 5,7 | 9,1 → 22,0 | 36 → 69% | 1,0 |
| 6 | 70,1 → **36,2** | 2,1 → 4,1 | 14,6 → 19,7 | 39 → 71% | 1,0 |
| 7 | 85,0 → **45,2** | 1,2 → 3,4 | 19,2 → 23,4 | 42 → 74% | 1,0 |

SNR p5 = 20–24 dB em todos os pontos (rádio são); RLF não observado; mMTC irrepreensível (PDR 1,0 até λ7 — 1,6 kbps/UE é ordens abaixo da capacidade).

## Leituras

1. **Forte evidência de evitabilidade, agora em curva contínua**: o braço equal viola o SLA eMBB em massa (23→85%) **com a célula ociosa** (util D2 22–42%); o `prop` recupera 18–40 pp de violação em TODOS os pontos usando exatamente essa capacidade desperdiçada (D2 43–74%; thr/UE 2–3× maior). A classificação formal (F_t≠∅ vs F_t=∅ por janela) segue reservada ao oracle 2.4.
2. **Trade-off emergente que motiva o oracle**: nenhum braço domina. O `prop` reduz eMBB drasticamente mas **piora a cauda URLLC sob pressão eMBB** (miss 9→22% em λ5; nas seeds com eMBB pesado: 12→34%) — o piso URLLC de 25% (R39) é espremido quando a demanda eMBB satura o pool. O equal protege a cauda URLLC (miss ~9–11% até λ5) ao preço de violação eMBB massiva. **Existe alocação que salve ambos nos λ4–7? Essa é exatamente a pergunta do oracle 2.4.**
3. **Variabilidade entre seeds dominante no equal** (λ4: 0,7% na seed 2 vs 87,4% na seed 3 — sorte dos clusters Poisson): consolida a diretriz da Etapa 2.0 (estatística por janela/N(t), não por run). O `prop` comprime também essa variância (máx. 60% vs 98% no equal).
4. **A curva do `prop` marca a transição**: viol eMBB < ~5% só em λ2; 18–25% em λ4–5; 36–45% em λ6–7 com thr med ainda ≥ 3,4 Mbps. Combinando com o URLLC: λ4–5 é a região onde ambos os KPIs estão "quase" ao alcance (região da fronteira conjunta); λ6–7 é stress.

## Encaminhamento (Etapa 2.4 — oracle)

1. **Oracle por janela** (modelo offline + validação executável, desenho já registrado em `analise_rodada2_orientador.md` §2.2): para cada janela dos runs existentes (e amostra de janelas de λ4–7), enumerar/projetar F_t sobre o simplex de cotas (C=53) com os SLAs R36–R38 e produzir P_inf(λ), AVR, |F_t|, D_min — a tabela do gate §26.
2. **Entrada de dados pronta**: KPM 17 colunas + packet_delays por run/janela/janela × 30 runs; N(t) por slice por janela já extraível.
3. Pendências menores: λ6_s3-equal recuperado pós-timeout (CSV íntegro, shutdown excedeu — nota no `.done`); lock/reaper funcionaram; mistério script×inline permanece registrado no dossiê (fora do caminho crítico).

## Artefatos
`hetero_sweep/` (equal) e `hetero_sweep_prop/` (prop) com 15 `.done` cada; `round2_hetero_sweep.csv` (por arm×λ×seed), `..._by_lambda.csv`, `figures_hetero_sweep.png` (4 painéis A–D), `analyze_hetero_sweep.py`, logs `loop.log`/`loop2.log`.
