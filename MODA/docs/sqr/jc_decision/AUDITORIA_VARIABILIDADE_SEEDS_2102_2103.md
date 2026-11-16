# Auditoria Final da Variabilidade entre *Seeds* (`2102` vs. `2103`) e Interpretação do `STRICT_ALL3_R19`

**Data:** 2026-11-15 · **Escopo:** $\lambda \in \{5, 6\}$ (`het_lam{5,6}_s2102` vs. `het_lam{5,6}_s2103` sob `B0-rest = A1+A2a+A_scaler`).  
**Fonte de dados:** telemetria bruta `stdout`, `sqr_summary.csv`, `sqr_kpm_history.csv`, `sqr_prb_windows.csv` e `decisions.jsonl` extraída por [`audit_seed_variability_2102_2103.py`](file:///home/eliothluy/.gemini/antigravity/brain/4622041f-48e4-41e3-8c43-cf5834bb15d6/scratch/audit_seed_variability_2102_2103.py). Nenhum experimento histórico foi alterado.

---

## 1. Verificação de Equivalência da Configuração de Tráfego (`sqr-sim.cc`)

A comparação linha a linha dos *banners* de configuração emitidos em `stdout` pelos 4 runs confirma **equivalência exata** em $\lambda=5$ e em $\lambda=6$ (únicas diferenças: `Seed/Run` e `OutputDir`):

- Mesmo gerador RNG (`RngSeedManager::SetSeed(seed)`, `SetRun(1)`), mesmos 3 processos de Poisson independentes desde `t=0` (`1,667/1,667/1,667` em $\lambda=5$; `2,0/2,0/2,0` em $\lambda=6$), sessão `Exp(60 s)` limitada a `simTime=300 s`, mobilidade `RandomWalk2d ~1 m/s` em raio `300 m`, canal 3GPP UMa, `2 GHz`, `30 dBm`, `15 dBi`, scheduler `SqrMacScheduler` (`totalRbg=totalPrb=53`, `period=100 ms`), e perfis idênticos (`eMBB=100 Mbps`, `URLLC=32 B / 1 ms = 0,256 Mbps UDP`, `mMTC=200 B / 1000 ms = 0,0016 Mbps`).
- **Conclusão sobre configuração:** **zero divergência de configuração ou bug de geração**. A diferença observada é **característica legítima das realizações estocásticas** (`Exp(60 s)` e chegadas de Poisson).

---

## 2. Quantitativo por $\lambda$ e Slice (`sqr_summary.csv` + `sqr_kpm_history.csv`)

> *Nota metodológica sobre `sqr_summary.csv`:* `sqr-sim.cc` grava `departureSec = -1` para sessões ainda ativas no encerramento (`t = 300 s`). Abaixo, a duração ativa dentro da simulação usa $\min(\text{departureSec}_{\text{efetivo}}, 300) - \text{arrivalSec}$ com $\text{departureSec}_{\text{efetivo}} = 300$ quando `-1`. O tráfego KPM inclui cabeçalhos IP/UDP/GTP (`URLLC ~0,48 Mbps`).

### 2.1 Sessões, Chegadas, Durações e Tráfego (0–300 s)

| Caso | Slice | UEs (warm / post / atv@300) | Chegadas `min..max` (s) | Duração média (soma UE·s) | Oferecido UDP (MB) | `txPackets` | `rxPackets` | KPM TX / RX (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`λ5 s2102`** | eMBB (0) | 9 (1 / 8 / 2) | `93,20..298,90` | 39,19 s (`352,7 s`) | 4408,75 | 2.902.048 | 149.176 | 4351,62 / 215,67 |
| **`λ5 s2102`** | URLLC (1) | 6 (3 / 3 / 2) | `0,924..270,30` | 25,53 s (`153,2 s`) | 4,90 | 150.084 | 150.055 | 7,14 / 7,14 |
| **`λ5 s2102`** | mMTC (2) | 5 (0 / 5 / 1) | `103,70..235,50` | 104,54 s (`522,7 s`) | 0,10 | 519 | 519 | 0,12 / 0,12 |
| **`λ5 s2103`** | eMBB (0) | 9 (2 / 7 / 3) | `40,74..290,00` | 37,79 s (`340,1 s`) | 4251,00 | 2.796.327 | 161.265 | 3437,70 / 148,52 |
| **`λ5 s2103`** | URLLC (1) | 6 (2 / 4 / 1) | `2,121..282,20` | **55,43 s (`332,6 s`)** | **10,64** | **329.543** | **329.524** | **13,84 / 13,83** |
| **`λ5 s2103`** | mMTC (2) | 4 (1 / 3 / 0) | `7,245..212,50` | 55,32 s (`221,3 s`) | 0,04 | 217 | 217 | 0,04 / 0,04 |
| **`λ6 s2102`** | eMBB (0) | 10 (1 / 9 / 3) | `77,69..283,20` | 44,54 s (`445,4 s`) | 5567,88 | 3.670.630 | 169.223 | 5395,64 / 226,62 |
| **`λ6 s2102`** | URLLC (1) | 7 (3 / 4 / 3) | `0,770..290,40` | 35,74 s (`250,2 s`) | 8,01 | 246.548 | 246.504 | 12,91 / 12,91 |
| **`λ6 s2102`** | mMTC (2) | 5 (1 / 4 / 0) | `86,41..196,30` | 108,74 s (`543,7 s`) | 0,11 | 540 | 540 | 0,12 / 0,12 |
| **`λ6 s2103`** | eMBB (0) | 10 (2 / 8 / 3) | `33,96..284,10` | 46,83 s (`468,3 s`) | 5853,25 | 3.858.712 | 174.727 | 4974,66 / 146,68 |
| **`λ6 s2103`** | URLLC (1) | 6 (3 / 3 / 1) | `1,764..235,20` | **63,26 s (`379,6 s`)** | **12,15** | **376.527** | **376.502** | **16,33 / 16,33** |
| **`λ6 s2103`** | mMTC (2) | 6 (1 / 5 / 1) | `6,034..280,80` | 41,08 s (`246,5 s`) | 0,05 | 240 | 240 | 0,05 / 0,05 |

**Mecanismo exato da discrepância:** O total de UEs sorteados foi quase idêntico (`20` vs `19` em $\lambda=5$; `22` vs `22` em $\lambda=6$, com `6` vs `6` UEs URLLC em $\lambda=5$ e `7` vs `6` em $\lambda=6$). Porém, na `seed=2103` os sorteios `Exp(60 s)` geraram duas sessões URLLC muito longas (`UE 1: 86,16 s` no *warmup* e `UE 7: 166,5 s` de `128,1 s` a `294,6 s`), mais do que dobrando os **UE·segundos URLLC** (`153,2 s -> 332,6 s` em $\lambda=5$), enquanto na `seed=2102` $\lambda=5$ houve dois longos vazios sem nenhum UE URLLC (`100,2–146,6 s` e `192,2–254,7 s`).

### 2.2 Distribuição Temporal de UEs Ativos, Simultaneidade, Filas, Utilização DL e Excesso sobre $C=53$ (Pós-*warmup* `100–300 s`)

| Métrica (`100–300 s`) | `λ5 s2102` | `λ5 s2103` | `λ6 s2102` | `λ6 s2103` |
|---|---:|---:|---:|---:|
| **UEs ativos por janela (média / max):** eMBB / URLLC / mMTC | 1,71 (4) / 0,60 (2) / 2,60 (5) | 1,35 (4) / **1,15 (2)** / 0,92 (2) | 2,12 (5) / 1,08 (3) / 2,64 (5) | 1,96 (4) / **1,36 (2)** / 1,04 (2) |
| **Distribuição UEs URLLC `[0 / 1 / 2 / >=3]` (%)** | `55,0 / 30,4 / 14,6 / 0,0` | **`6,8 / 71,0 / 22,2 / 0,0`** | `33,9 / 29,1 / 32,5 / 4,5` | **`0,1 / 63,6 / 36,3 / 0,0`** |
| **Distribuição UEs eMBB `[0 / 1 / 2 / >=3]` (%)** | `23,8 / 15,1 / 30,4 / 30,6` | `20,9 / 51,9 / 5,8 / 21,4` | `15,8 / 5,0 / 39,9 / 39,2` | `11,8 / 38,2 / 6,0 / 44,0` |
| **Fração janelas `[0 / 1 / 2 / =3]` slices ativas (grade 2000 jan.)** | `0,05 / 20,40 / 39,95 / 39,60` | `2,40 / 13,65 / 10,15 /` **`73,80`** | `0,05 / 11,35 / 28,75 / 59,85` | `0,05 / 4,10 / 14,25 /` **`81,60`** |
| **Idem em janelas `het` fechadas `[1 / 2 / =3]` (%)** | `20,46 / 40,07 / 39,47` | `14,02 / 10,17 /` **`75,81`** | `11,38 / 28,59 / 60,03` | `4,11 / 14,29 /` **`81,59`** |
| **Fila média quando ativa (eMBB / URLLC / mMTC)** | 4,40 MB / 162 B / 525 B | 3,51 MB / 152 B / 229 B | 4,99 MB / 188 B / 599 B | 4,46 MB / 167 B / 246 B |
| **Fila p95 quando ativa (eMBB / URLLC / mMTC)** | 8,64 MB / 522 B / 948 B | 10,03 MB / 453 B / 474 B | 9,56 MB / 522 B / 1185 B | 9,73 MB / 522 B / 474 B |
| **Utilização DL MAC (`win_mean` / `slot_weighted`)** | 46,75% / 45,06% | 41,19% / 40,26% | 48,01% / 47,08% | 41,04% / 40,51% |
| **Propostas `> 53` (`prop>53` / `raw>53`)** | 42,78% / 41,68% | **54,13% / 52,85%** | 62,29% / 61,63% | **83,55% / 83,00%** |
| **Soma média propostas / excesso `sum(prop)-53` (geral \| `>0`)** | 45,95 PRBs (`6,19 \| 14,46`) | **59,89** (`13,71 \| 25,33`) | 59,28 PRBs (`14,21 \| 22,81`) | **72,32** (`20,78 \| 24,87`) |

*(Na grade integral `0–300 s` = 3000 janelas, as frações `[0 / 1 / 2 / 3]` slices ativas foram `23,30 / 22,67 / 27,63 / 26,40%` em `λ5 s2102` vs. `2,47 / 14,70 / 32,93 / 49,90%` em `λ5 s2103`; e `18,07 / 19,17 / 22,87 / 39,90%` em `λ6 s2102` vs. `0,77 / 6,37 / 35,90 / 56,97%` em `λ6 s2103`.)*

---

## 3. Interpretação Rigorosa de `STRICT_ALL3_R19` (PASS / FAIL / UNKNOWN Explícitos)

| Caso | $N_{\text{het,closed}}$ | `PASS` | `FAIL` | `UNKNOWN` (`unk%`) | Avaliável `known` (`cov%`) | **`PASS / known`** | `PASS / N_het_closed` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`λ5 s2102`** | 1994 | **563** | 222 | **1209** (`60,63%`) | **785** (`39,37%`) | **`71,72%`** | `28,23%` |
| **`λ5 s2103`** | 1947 | **657** *(+94)* | 820 | **470** (`24,14%`) | **1477** (`75,86%`) | **`44,48%`** | `33,74%` |
| **`λ6 s2102`** | 1994 | 662 | 535 | 797 (`39,97%`) | 1197 (`60,03%`) | `55,30%` | `33,20%` |
| **`λ6 s2103`** | 1994 | 465 | 1152 | 377 (`18,91%`) | 1617 (`81,09%`) | `28,76%` | `23,32%` |

- Em $\lambda=5$, os `PASS` absolutos **subiram de `563` para `657` (`+94` janelas)** e a taxa incondicional `PASS / N` subiu de `28,23%` para `33,74%`. Contudo, o desaparecimento das janelas URLLC vazias reduziu `UNKNOWN` de `60,63%` para `24,14%`, quase **dobrando o denominador condicional `known`** (`785 -> 1477`) sob intensa contenção simultânea (`75,81%` em `=3` ativas) — fazendo a taxa condicional cair de `71,72%` para `44,48%`.
- Portanto, **`UNKNOWN` nunca deve ser contado como `PASS` ou `FAIL`**, e percentuais condicionais entre *seeds* não devem ser interpretados sem reportar explicitamente a cobertura `known / N`. Em $\lambda=6$ (`s2103`), além do aumento de cobertura (`60,03% -> 81,09%`), houve queda real de `PASS` (`662 -> 465`) pela saturação de competição trilateral (`81,59%` em `=3` ativas e `83,55%` de projeção).
