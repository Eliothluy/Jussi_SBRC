# D1v6_report.md — D1 canônica (Opção B): relatório v6-offline + v8 runtime

**Protocolo:** v2 §7 + Emenda 3 (Opção B, ratificada 12/11) · **Datas:** 11–12/11/2026 · **Canônica:** **D1-v8** (runtime com os 3 fixes; 9/9 runs às 22:26) — **v6 complementa a auditoria de observabilidade** (foi executado ANTES do fix do bug 3, com holds re-observando; seus outcomes online são inválidos, mas a telemetria física é íntegra e foi usada para o perfil de atraso). Quarentenas: v1 (parser), v3+cold (contaminação), v4/v5 (censura), v6 (hold-repetição), v7 (spin/OOM, sem runs).

## 1. Os 4 pontos do orientador — respostas com dados

**Q1 Observabilidade temporal (3 instantes, separados por instrumentação):**
- **Instante da violação:** `txSec + 10 ms` — existe FISICAMENTE mas não é observado por ninguém na hora: o `txSec` só é registrado no arquivo quando o pacote é **entregue** (única origem de linha = Rx, `sqr-sim.cc:457-480` — confirmado em código: não há trace de TX puro).
- **Instante de observação online:** quando o runtime pergunta (decisão de W+1) e o pacote já foi **entregue e flushado** — medido: 67,8% dos pacotes são entregues DEPOIS do instante de expiração ⇒ o controlador **não identifica online a expiração em tx+10 ms** (afirmação proibida e nunca feita).
- **Avaliação offline:** arquivo completo pós-run — completo **para os atrasos observados dos pacotes recebidos** (não representa o tráfego integral: 30–35% do tx sem RX não entra no denominador de miss).

**Q2 Denominadores auditados (por run, offline completo):**

| run | tx | entregues | semRX | pendentes-fim | perdidos-meio | miss% (dos entregues) | p95 (ms) |
|---|---|---|---|---|---|---|---|
| λ2s2101 | 0 | 0 | 0 | 0 | 0 | — | — |
| λ2s2102 | 31.084 | 30.333 | 751 | 0 | 751 | 8,0% | 159,1 |
| λ2s2103 | 116.100 | 115.270 | 830 | 200 | 630 | 6,1% | 12,1 |
| λ5s2101 | 79.102 | 27.234 | 51.868 | 0 | 51.868 | 77,6% | 1.801,1 |
| λ5s2102 | 149.884 | 97.255 | 52.629 | 400 | 52.229 | 67,8% | 637,1 |
| λ5s2103 | 329.443 | 222.645 | 106.798 | 200 | 106.598 | 31,8% | 1.153,1 |
| λ6s2101 | 163.794 | 57.568 | 106.226 | 600 | 105.626 | 74,5% | 1.979,1 |
| λ6s2102 | 246.248 | 239.438 | 6.810 | 0 | 7.086* | 9,2% | 54,1 |
| λ6s2103 | 376.427 | 244.985 | 131.442 | 0 | 131.534* | 33,8% | 1.326,1 |

(*λ6: período sem RX inclui janelas de fim; categorização pend/feim por janela em `d1_offline_labels.csv`.) Regras: **sem-RX nunca é sucesso nem miss implícito** — é métrica separada; miss e p95 computados SOMENTE sobre entregues; janelas sem demanda = NO_ACTIVE_DEMAND (R19).

## 2. D1-v8 — resultados por λ e seed (runtime; 15.124 decisões B0 com outcome de 17.955)

| λ | seed | decisões | P_proj | viol_joint (online-frozen) |
|---|---|---|---|---|
| 2 | 2101/02/03 | 1.800/564/843 | 10,4/11,2/7,9% | 2,9/2,0/13,0% |
| 5 | 2101/02/03 | 1.994/1.994/1.947 | 11,9/10,5/13,2% | 13,2/38,4/11,5% |
| 6 | 2101/02/03 | 1.994/1.994/1.994 | 12,5/13,5/16,0% | 36,9/50,7/28,1% |

- **Projeção §7.2:** P_projection = **12,3%** (Σ/C mediana 0,57, p95 1,28; ||Δ||₁ mediano 11) — maior que v3/v6 porque os holds agora são reais (o B0 decide menos vezes em célula vazia, e as decisões restantes concentram demanda).
- **Degradação de margens previstas pela projeção: 37,4%** das decisões com projeção (Q2 mantém).
- Matriz projeção×outcome frozen: sem 3.161 FAIL / 12.824 OK; com 561/1.400 (associação, não causal).

## 3. Rótulos OFFLINE íntegros (Opção B — `d1_offline_labels.csv`)

| run | B0 | STRICT_PASS | STRICT_FAIL | UNKNOWN | miss médio | p95 med |
|---|---|---|---|---|---|---|
| λ2s2101 | 1.800 | 0 | 0 | 1.800 | — (sem URLLC) | — |
| λ2s2102 | 564 | 0 | 0 | 564 | 25,2% | 8,1 |
| λ2s2103 | 843 | 0 | 0 | 843 | 57,7% | 99,1 |
| λ5s2101 | 1.994 | 0 | 0 | 1.994 | 74,7% | 216,9 |
| λ5s2102 | 1.994 | 37 | 307 | 1.650 | 70,7% | 163,1 |
| λ5s2103 | 1.947 | 39 | 665 | 1.243 | 52,9% | 26,1 |
| λ6s2101 | 1.994 | 65 | 310 | 1.619 | 70,0% | 160,1 |
| λ6s2102 | 1.994 | 29 | 571 | 1.394 | 67,3% | 198,1 |
| λ6s2103 | 1.994 | 68 | 663 | 1.263 | 55,1% | 27,1 |

**Achado de prevalência (importante):** a demanda **simultânea dos 3 slices** é rara por construção do perfil — λ6s2103: 1.942 janelas com URLLC, 1.384 com eMBB∧URLLC, mas só **96 com os 3** (mMTC 1 pkt/s) ⇒ 88 decisões B0 com outcome em janela 3-demanda. **Consequência metodológica:** o STRICT ALL3 tem n pequeno por natureza neste cenário; as métricas primárias do D2/D3 serão reportadas com **(i) STRICT_ALL3 (n pequeno, quando existe)** e **(ii) eMBB+URLLC co-avaliadas (n maior)** — pré-registrado antes do D2-v2.

## 4. Q3 — aprendizado SQR-HET (updates efetivos, evolução, viés)

- **Updates URLLC online (runtime) por run:** λ2s2101 0 · λ2s2102 692→0 pós-bootstrap · λ5s2102 0 · λ6s2102 0 · λ6s2103 0 — **a censura de 1 janela mata praticamente todos os updates URLLC pós-bootstrap** (só λ2s2102 manteve 1.333, com entregas majoritariamente intra-janela).
- **Diagnóstico e impacto:** com a Opção B, o modelo URLLC do B0 opera com pouquíssima informação online — as previsões URLLC médias degradam por run (ex.: λ6s2102 −0,09→−6,29) e a margem de segurança URLLC fica dominada pelo prior. **Impacto na comparação B0×B1:** válido e justamente pareado — ambos partem do MESMO checkpoint e observam o MESMO fluxo (qualquer déficit de informação penaliza os dois igualmente); o que muda é SÓ a seleção. Risco documentado: nenhum dos braços "aprende" URLLC — as escolhas URLLC serão guiadas por prior/estado atual (n_e, fila), não por previsão calibrada.
- **B0×B1 mesmo mecanismo** (cuidado 3): garantido por construção (mesma classe, mesmos checkpoints, mesmo fluxo de observação — o B1 só muda a regra de seleção final).
- **Nada retroalimentado** (cuidado 1): hashes dos checkpoints antes/depois verificados na D3.

## 5. Estado e próximo passo

- D1-v8 canônica ✓ com relatório (este) e artefatos (`d1_offline_labels.csv`, `projection_*.csv/png`).
- **D2-v2 liberada sob a Opção B**: re-amostragem (seed 20261108) sobre os estratos offline íntegros; métricas primárias pré-registradas no ponto 3; 390 replays ×3 (~12 h); adjudicação offline completa com denominadores auditados.
- D3 após D2-v2 (declaração de conformidade vigente). D4/PILOT e D5/MAIN bloqueados.
