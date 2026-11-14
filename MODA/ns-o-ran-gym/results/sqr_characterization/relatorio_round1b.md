# Rodada 1b — Refinamento da transição (λ∈{5,6,7}), análise fina e auditoria de utilização

**Data:** 19/10/2026 · **Mandato:** go do orientador (14/10) para (1) refinar λ∈{5,6,7} com rádio congelado; (2) percentis p5/p10/p25 por UE e violação por slice/janela; (3) auditar a utilização de 38% antes de interpretar saturação; (4) só depois Rodada 2.

**Protocolo:** idêntico à Rodada 1 (nada além de λ muda): raio 300 m, 30 dBm, 10 MHz (C=53 RBG), UMa, omni 15 dBi, TDD DL|DL|DL|S|UL|UL|UL, μ=0, PF intra-slice, RLC UM, greedy 100 Mbps/UE, sessão Exp(60 s), SLA/UE 0,5/1,0/1,5, equal-split, network-only, 300 s, 3 seeds. 9 runs novos (todos íntegros: timestamps monotônicos, 2999 janelas); 24 runs no total.

---

## 1. Auditoria da utilização — o "38%" era artefato de denominador

**Métrica antiga** (`analyze_round1.py`): `mean(Σ_i prb_scheduled_rbg_i) / total_rbg` — soma RBGs da janela inteira e divide por 53 (um único slot). Dimensão errada: divide "RBGs por janela" por "RBGs por slot".

**Semântica correta** (`sqr-sim.cc` §3.5/contrato): `prb_scheduled` acumula por **janela** sobre os slots DL; `dl_slots` = nº de slots DL **com ≥1 UE ativo** na janela (`sqr-mac-scheduler.cc:356`); `prb_idle = totalRbg×dlSlots − Σscheduled`; `prb_budget = cota×dlSlots`; `prb_executed` = cota por slot (RBG).

**Denominadores auditados:**
- **D2 (adotado)** = `53 × dl_slots` — capacidade DL nos slots com demanda. É o que o CSV suporta sem ambiguidade.
- **D1** = `53 × 57` (teto 4/7 slots DL+S por janela de 100 ms) — capacidade potencial; subestima a utilização em λ baixo (slots sem demanda contam como ociosos).

**Correção (média 3 seeds, D2):** λ0,5→21% · λ1→26% · λ2→41% · λ4→57% · λ5→68% · λ6→76% · λ7→84% · λ8→**88%**.

**Leitura corrigida:** a utilização cresce monotônica e suavemente com λ e chega a ~88% em λ=8. O achado anterior "saturação de throughput/UE antes do espectro" **era artefato** — não existe mistério. Ressalva registrada: dl_slots conta o slot S como DL (4/7), mas o S tem menos símbolos DL úteis; a utilização em REs físicos é um pouco maior que D2 reporta. Ressalva 2: dl_slots ≥1 UE ativo ⇒ D2 é "uso condicionado à demanda"; em λ≥6 praticamente todos os slots têm UEs, então D2 ≈ ocupação real da célula.

## 2. Joelho da degradação (24 runs, violação UE-janela c/ demanda)

| λ | violação % (min–máx seeds) | util D2 | thr/UE p5 / p10 / p25 / med (Mbps) | UEs que violam ≥1× | viol t1/t2/t3 (terços do run) |
|---|---|---|---|---|---|
| 0,5 | 0,0 | 21% | 5,4 / 5,7 / 6,5 / 7,3 | 0% | 0/0/0 |
| 1 | 2,3 | 26% | 2,9 / 3,0 / 4,6 / 7,1 | 24% | 0 / 2 / 3 |
| 2 | 2,0 | 41% | 3,2 / 3,5 / 5,2 / 6,7 | 23% | 2 / 3 / 0 |
| 4 | 5,6 (0,4–8,2) | 57% | 1,9 / 2,5 / 3,1 / 4,4 | 33% | 8 / 1 / 5 |
| **5** | **14,0 (6,8–23,5)** | **68%** | 1,4 / 1,6 / 2,0 / 2,9 | 66% | 11 / 9 / 17 |
| 6 | 18,3 (10,2–23,3) | 76% | 0,8 / 1,2 / 2,0 / 3,0 | 60% | 11 / 12 / 19 |
| 7 | 24,1 (16,6–30,1) | 84% | 0,6 / 0,8 / 1,7 / 2,4 | 79% | 14 / 19 / 28 |
| 8 | 35,8 (33,4–37,2) | 88% | 0,04 / 0,4 / 1,1 / 1,8 | 84% | 16 / 32 / 46 |

**Onde está o joelho:** a violação quase triplica de λ4→λ5 (5,6→14,0) e a mediana de thr/UE cai 34% (4,4→2,9). Porém os intervalos das seeds λ4 (0,4–8,2) e λ5 (6,8–23,5) se sobrepõem — a transição é **gradual, começando em λ≈4–5**, sem degrau discreto. Separação inequívoca a partir de λ6; λ8 é compacto (33–37%) porque o sistema está saturado. O p5 colapsa primeiro (1,9→1,4→0,8→0,6→0,04): **a cauda é o indicador precoce**, a mediana só cai em λ7–8.

## 3. Quem viola, e em quais janelas

- **Por slice:** a slice que mais viola **muda de seed para seed** (λ8: s1=64% na seed1, s2=51% na seed2, s1=50% na seed3) — é variância estocástica do churn Poisson (o slice com mais UEs naquele run viola mais), não propriedade do SLA. Agregado por λ deve ser lido como "média sobre sorteios", não ranking de slice.
- **Por fase da sessão:** violação já aparece na 1ª segundo (λ8: 13–15% em 0–1s) e cresce com o tempo de sessão (λ8 >10s: 40–45%). Transiente de entrada curto + convivência com mais co-UEs. O `windowThroughput` já normaliza pelo tempo ativo (contrato §4), não é artefato de diluição.
- **Por terço do run:** em λ≥5 a violação cresce ao longo do run (λ8: 16→32→46%) — 300 s ainda é transiente de acumulação em overload (E[N] não estaciona).
- **Rádio dos violadores:** SNR mediana dos violadores = **24,0 dB = SNR dos não-violadores** em λ≥5 (λ8 idem). **A violação não é de rádio** — confirma degradação por demanda (Q5 do orientador, agora com evidência por UE-janela).
- **Cota como gargalo:** fração de janelas usando <80% da cota cai de 100% (λ0,5) para 25% (λ8); em λ6+ o log mostra `scheduled == budget, prb_idle=0` — o scheduler consome a cota inteira e ainda tem fila.

## 4. Recomendações para a Rodada 2 (congelamento proposto)

- **L_N = λ2** (normal): violação 2%, p5 = 3,2 Mbps (≈2× SLA máx), 41% D2.
- **Joelho = λ4–5** (marginal): 5,6–14%, ~35% do espectro DL ainda ocioso ⇒ violação majoritariamente **evitável** — região onde um alocador inteligente (SQR) tem o que ganhar.
- **L_O = λ7** (não λ8): 24% de violação com p5=0,64 e 84% D2 — severo mas não degenerado. λ8 é overload severo demais (p5≈0, mediana 1,8 < SLA máx; quase tudo inevitável) — serve como limite superior, não como ponto operacional.
- A pergunta da Rodada 3 (oracle) fica: **em λ4–5 a violação é evitável (F_t≠∅); em λ7–8 é inevitável (F_t=∅)?** A auditoria de utilização (57–68% vs 84–88% D2) sugere que sim.

## 5. Artefatos

- Runs: `results/sqr_characterization/sweep/lam{5,6,7}_s{1,2,3}/` (+`.done`), logs `lam*.log`.
- Análise: `analyze_round1b.py` (gate de integridade por janelas), `characterization_round1b.csv` (24 runs), `..._by_lambda.csv`, `..._viol_by_phase.csv`, `figures_round1b.png` (4 painéis: joelho, percentis, fase da sessão, por slice).
- Runner retomável: `run_sweep_refine.sh` (JOBS=9, pula `.done`).

**Incidente operacional (registro):** durante a troca JOBS=3→9, dois runners coexistiram e 3 runs lam7 foram corrompidos (dois processos nos mesmos CSVs); detectido pelos PIDs/cmdline, corridos os duplicados, lam7 refeitos do zero (íntegros, validados). Lição: `setsid` retorna PID do wrapper — matar o bash real (PPID aponta o filho), e nunca dois runners no mesmo diretório sweep.
