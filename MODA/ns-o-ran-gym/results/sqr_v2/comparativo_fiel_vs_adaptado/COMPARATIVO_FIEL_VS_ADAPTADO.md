# Comparativo Experimental: Implementação Fiel (Artigo CCNC 2026) vs. Adaptada (`v2` / `SQR-CER`)

**Data:** 2026-11-19  
**Branch de execução:** `exp/sqr-v2-infra` (`25/25` runs ao vivo com `returncode = 0` e `ns3_exitcode = 0`)  
**Artefatos de dados:**
- Runs fiéis individuais (`24` runs): [`comparativo_fiel_runs.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/comparativo_fiel_vs_adaptado/comparativo_fiel_runs.csv)
- Agregado fiel por cenário × método: [`comparativo_fiel_summary.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/comparativo_fiel_vs_adaptado/comparativo_fiel_summary.csv)
- Validação ao vivo `v2` (`seed=3001`) + paridade `0.0`: [`comparativo_v2_live_validation.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/comparativo_fiel_vs_adaptado/comparativo_v2_live_validation.json)
- Campanha completa `v2` (`105` episódios `MAIN` + `OOS` e sweep multi-raio): [`fase3_method_summary.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/fase3_method_summary.csv), [`fase3_rescue_harm_summary.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/fase3_rescue_harm_summary.csv) e [`fase3_radius_sweep_summary.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/fase3_radius_sweep_summary.csv)

---

## 1. Configuração Experimental dos 3 Cenários

| Parâmetro | Cenário 1 — Fiel `v1` Literal | Cenário 2 — Fiel `v2` Viável | Cenário 3 — Adaptado `v2` Heterogêneo (`Fase B`) |
|---|---|---|---|
| **Simulador / Runner** | [`sqr-sim.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr/sqr-sim.cc) / [`sqr_run.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/examples/sqr_run.py) (`v1` congelado) | [`sqr-v2-sim.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc) / [`sqr_v2_run.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/examples/sqr_v2_run.py) (sem `--het`) | [`sqr-v2-sim.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc) / [`sqr_v2_run.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/examples/sqr_v2_run.py) (`--het`) |
| **Raio da célula** | `2000 m` (Tabela I do artigo) | `300 m` (regime físico viável) | `300 m` (regime físico viável) |
| **Aplicações / Slices** | `3× eMBB` (`greedy,greedy,greedy`) | `3× eMBB` (`greedy,greedy,greedy`) | Heterogêneo: `eMBB` (`greedy`) + `URLLC` + `mMTC` |
| **SLAs por slice** | `0,5 / 1,0 / 1,5 Mbps` (`y_t = p5` throughput) | `0,5 / 1,0 / 1,5 Mbps` (`y_t = p5` throughput) | eMBB `≥ 3,5 Mbps`, URLLC `p95 ≤ 10 ms` (`miss ≤ 1%`), mMTC `PDR ≥ 0,95` |
| **Chegadas Poisson (`λ`)** | `2 UEs/min` total (`0,67 / 0,67 / 0,67` por slice) | `2 UEs/min` total (`0,67 / 0,67 / 0,67` por slice) | `3,3,3` (`9 UEs/min` totais = `3` por slice) |
| **Padrão TDD (`μ=0`, `10 MHz`, `C=53 PRBs`)** | Legado `7 ms` (`D\|D\|8D\|4GB\|4U\|U\|U`, máx. `58` slots DL+S / `100 ms`) | Normativo `5 ms` (`DL\|DL\|DL\|S\|UL\|`, máx. `80` slots DL+S / `100 ms`) | Normativo `5 ms` (`DL\|DL\|DL\|S\|UL\|`, máx. `80` slots DL+S / `100 ms`) |
| **IPC & Saneamento (`BUG-01/06/07/08`)** | IPC `v1` (legado) | Handshake `v2` (`strict_rt`, `50 ms`), bugs corrigidos | Handshake `v2` (`strict_rt`, `50 ms`), bugs corrigidos |
| **Duração / Seeds** | `90 s` (`25 s` warmup, `645` passos pós-warmup), `seeds = {101, 102, 103}` | `90 s` (`25 s` warmup, `648` passos pós-warmup), `seeds = {101, 102, 103}` | `90 s` (`25 s` warmup, `648` janelas pós-warmup); validação ao vivo `seed=3001` + `MAIN` (`10` seeds) + `OOS` (`5` seeds) |

> **Nota sobre a escolha de `λ = 2/min` no Cenário 2:** Em execução exploratória preliminar com `3× greedy` (`100 Mbps/UE`) sob `sliceLambdas = 3,3,3` (`9 UEs/min`), o volume agregado de pacotes UDP excedeu `timeout = 600 s` na `seed=103`. Usar `λ = 2/min` no Cenário 2 isola com precisão o efeito combinado de `2000 m → 300 m` e do saneamento `v1 → v2` mantendo os demais parâmetros do artigo idênticos ao Cenário 1. Para o regime de estresse homogêneo `3× eMBB` sob `3,3,3` (`9 UEs/min`), ver a referência cruzada na Seção 3.1 (dados de [`fase3_radius_sweep_summary.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/fase3_radius_sweep_summary.csv)).

---

## 2. Cenário 1 — Fiel `v1` Literal (`2000 m`, `3× eMBB`, `λ=2/min`, `TDD 7 ms`)

Fonte: [`comparativo_fiel_summary.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/comparativo_fiel_vs_adaptado/comparativo_fiel_summary.csv) e [`comparativo_fiel_runs.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/comparativo_fiel_vs_adaptado/comparativo_fiel_runs.csv) (`12` runs, todos `ns3_exitcode = 0`).

| Método | Violação UE-time (%) | Violação Slice-time (%) | Violação Conjunta (≥1 slice, %) | `thr_p5` médio (Mbps) | PRB Util (`exec/C`, %) | Slots DL ativos / jan. | Cons. Err (`rb_win`) | `decision_ms` p50 / p95 |
|---|---|---|---|---|---|---|---|---|
| `sqr_g1` | `63.87 ± 52.75` | `62.71 ± 52.33` | `67.47 ± 54.86` | `1.29 ± 1.95` | `51.51 ± 22.50` | `47.21` (máx. `58`) | `0.0` | `4.95 / 10.94 ms` |
| `sqr_g0` | `64.45 ± 51.74` | `63.29 ± 51.32` | `68.47 ± 53.12` | `1.18 ± 1.75` | `50.89 ± 23.14` | `47.18` (máx. `58`) | `0.0` | `5.03 / 10.32 ms` |
| `equal` | `65.13 ± 56.33` | `65.24 ± 56.42` | `65.28 ± 56.44` | `2.48 ± 4.14` | `100.00 ± 0.00` | `46.42` (máx. `58`) | `0.0` | `0.037 / 0.045 ms` |
| `kbrl` | `69.54 ± 43.14` | `68.18 ± 42.66` | `69.51 ± 36.68` | `0.78 ± 1.01` | `100.00 ± 0.00` | `46.56` (máx. `58`) | `0.0` | `1.11 / 2.57 ms` |

**Detalhe por seed (por que o desvio-padrão a `2000 m` é `43–56 pp`):**
- **`seed=101`** (UEs na borda da célula): todos os 4 métodos colapsam — `UE_viol`: `96.48%` (`sqr_g0/g1`), `98.36%` (`equal`), `98.98%` (`kbrl`), `thr_p5 = 0.07–0.13 Mbps`.
- **`seed=103`** (UEs distantes): `UE_viol`: `92.11%` (`sqr_g0/g1`), `96.95%` (`equal`), `89.61%` (`kbrl`), `thr_p5 = 0.10–0.31 Mbps`.
- **`seed=102`** (poucos UEs próximos à gNB, `slice0` sem tráfego pós-warmup): `UE_viol` cai para `0.09%` (`equal`), `3.02%` (`sqr_g1`), `4.75%` (`sqr_g0`), `20.02%` (`kbrl`).
- *Observação de telemetria:* sob `λ = 2/min` e antena isotrópica `1×1 SISO`, não ocorreram janelas multi-beam (`rb_win_multibeam_budget_gt_cap_windows = 0` e `rb_win_conservation_abs_err = 0.0` nos `12` runs `v1`). Além disso, como `m_windowDlSlots` em [`sqr-mac-scheduler.cc:477`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-mac-scheduler.cc#L477) conta apenas slots com `≥ 1` UE ativo, janelas vazias ou de churn reduzem a média observada para `~46,4–47,2` slots/janela (teto `58` do TDD `7 ms`).

---

## 3. Cenário 2 — Fiel `v2` Viável (`300 m`, `3× eMBB`, `λ=2/min`, `TDD 5 ms`)

Fonte: [`comparativo_fiel_summary.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/comparativo_fiel_vs_adaptado/comparativo_fiel_summary.csv) e [`comparativo_fiel_runs.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/comparativo_fiel_vs_adaptado/comparativo_fiel_runs.csv) (`12` runs, todos `ns3_exitcode = 0`, `timeout_hold_steps = 0`, `window_gap_steps = 0`, `rb_win_conservation_abs_err = 0.0`).

| Método | Violação UE-time (%) | Violação Slice-time (%) | Violação Conjunta (≥1 slice, %) | `thr_p5` médio (Mbps) | PRB Util (`exec/C`, %) | Slots DL ativos / jan. | Modo Confiante (%) | `decision_ms` p50 / p95 |
|---|---|---|---|---|---|---|---|---|
| `equal` | **`0.03 ± 0.05`** | **`0.03 ± 0.05`** | **`0.05 ± 0.09`** | **`9.13 ± 1.36`** | `100.00 ± 0.00` (`53,0/53`) | `75.48` (máx. `80`) | — | `0.039 / 0.058 ms` |
| `kbrl` | `6.90 ± 5.38` | `7.65 ± 5.93` | `11.11 ± 8.69` | `4.59 ± 3.64` | `86.45 ± 18.64` (`45,8/53`) | `74.96` (máx. `80`) | — | `1.08 / 2.56 ms` |
| `sqr_g1` | `16.32 ± 14.37` | `11.83 ± 10.12` | `19.91 ± 16.26` | `4.07 ± 0.73` | **`35.46 ± 2.79`** (`18,8/53`) | `77.25` (máx. `80`) | `0.70%` | `1.24 / 2.11 ms` |
| `sqr_g0` | `19.43 ± 13.26` | `14.92 ± 9.01` | `24.43 ± 14.19` | `3.57 ± 0.77` | **`32.74 ± 2.71`** (`17,3/53`) | `77.17` (máx. `80`) | `7.97%` | `1.21 / 2.00 ms` |

**Leitura técnica dos resultados em `300 m` (`λ = 2/min`, `90 s`):**
1. **Remoção do gargalo físico:** Ao passar de `2000 m` para `300 m`, a violação cai drasticamente em todos os métodos (`equal`: `65.13% → 0.03%`; `kbrl`: `69.54% → 6.90%`; `sqr_g1`: `63.87% → 16.32%`; `sqr_g0`: `64.45% → 19.43%`).
2. **Por que `equal` atinge `0.03%` enquanto `sqr_g1` fica em `16.32%` em episódio curto:** Sob `λ = 2/min` a `300 m`, a partição estática `18 / 17 / 18` PRBs (`100%` dos PRBs) entrega `~8,8–9,6 Mbps` por slice — **6× a 18× acima** dos SLAs de `0,5 / 1,0 / 1,5 Mbps`, tornando o problema trivial por sobreprovisionamento cego. Já o SQR online partindo do zero em apenas `648` passos de avaliação (com média de ~`0,67` chegada/min/slice) permanece `>92%–99%` do tempo em modos incertos (`modes_confident_group_pct = 0.70%` em `sqr_g1`, `7.97%` em `sqr_g0`), operando no limite mínimo de banda (`17,3–18,8 PRBs` = **`~65%–67%` de economia de PRBs** frente ao `equal`) e sofrendo violações transitórias nas chegadas de novos UEs.
3. **Ganho computacional e normativo `v1 → v2`:** A latência `p95` de decisão do SQR cai de **`10.32–10.94 ms`** (`v1`) para **`2.00–2.11 ms`** (`v2`), com conservação exata (`0.0`) e alinhamento `TDD 5 ms` (`80` slots/janela).

### 3.1 Referência cruzada — Fiel `v2` (`sqr_g1`) sob carga alta `3,3,3` (`fase3_radius_sweep_summary.csv`)
Para comparar diretamente com a carga `sliceLambdas = 3,3,3` (`9 UEs/min`) do Cenário 3 (dados de [`fase3_radius_sweep_summary.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/fase3_radius_sweep_summary.csv), `seeds = {101,102,103}`, `60 s`, perfil `embb_article_05_10_15`):
- A `300 m`: violação slice-time `44.59%`, `thr_p5 = 1.873 Mbps`, PRB util `39.95%`, `cons_err = 0.0`.
- A `2000 m`: violação sobe para `78.64%`, `thr_p5 = 0.771 Mbps`, PRB util `59.27%`, `cons_err = 0.0`.

---

## 4. Cenário 3 — Adaptado `v2` (`300 m`, `eMBB/URLLC/mMTC`, `sliceLambdas=3,3,3`, `TDD 5 ms`)

### 4.1 Validação ao vivo (`sqr_cer`, `seed=3001`) vs. campanha salva (`fase3_per_seed_summary.csv`)

Fonte: [`comparativo_v2_live_validation.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/comparativo_fiel_vs_adaptado/comparativo_v2_live_validation.json).

| Métrica (`MAIN`, `sqr_cer`, `seed=3001`) | Execução ao vivo (`19/11`) | Salvo na Fase 3 (`18/11`) | Diferença absoluta |
|---|---|---|---|
| `ns3_exitcode` / `steps` / `applied_steps` | `0` / `898` / `898` | `0` / `898` / `898` | `0.0` |
| `timeout_hold_steps` / `window_gap_steps` / `cons_err` | `0` / `0` / `0.0` | `0` / `0` / `0.0` | `0.0` |
| Violação Conjunta (`sla_viol_all3_pct`) | `50.00%` | `50.00%` | `0.0` |
| Violações por slice (`eMBB` / `URLLC` / `mMTC`) | `31.79%` / `40.59%` / `0.46%` | `31.79%` / `40.59%` / `0.46%` | `0.0` |
| `embb_thr_mean_mbps` | `6.041175 Mbps` | `6.041175 Mbps` | `0.0` |
| `urllc_p95_ms_mean` / `p95` (`miss %`) | `11.615 ms` / `16.029 ms` (`10.91%`) | `11.615 ms` / `16.029 ms` (`10.91%`) | `0.0` |
| `mmtc_pdr_mean` / `prb_util_pct` | `1.0015` / `100.00%` | `1.0015` / `100.00%` | `0.0` |
| `cer_interventions` | `379` | `379` | `0.0` |
| Latência IPC (`agent_ms` / `cer_ms` p50 & p95) | `1.510 / 2.913 ms` · `0.182 / 0.288 ms` | — | — |

Reprodutibilidade **bit-a-bit (`abs_diff = 0.0` em todas as 17 chaves)** verificada após a limpeza do workspace.

### 4.2 Resumo completo da campanha `Fase B` (`105` episódios)

Fontes: [`fase3_method_summary.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/fase3_method_summary.csv) e [`fase3_rescue_harm_summary.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/fase3_rescue_harm_summary.csv) (`rb_win_conservation_abs_err = 0.0`, `timeout_hold_steps = 0`, `window_gap_steps = 0` em `105/105` runs).

| Split | Método | Violação Conjunta (%) | eMBB (%) | URLLC (%) | mMTC (%) | eMBB Thr (Mbps) | URLLC `p95` Mean (`p95`) | mMTC PDR | PRB Util (%) | Ganho Pareado vs `sqr_g1` (Win Ratio) | `agent_ms` p50 / p95 (`cer_ms` p95) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **MAIN** (`10` seeds) | **`sqr_cer`** | **`45.14 ± 24.68`** | `28.94` | `21.99` | **`0.71`** | `7.02` | `11.60` (`32.82`) ms | `1.014` | `98.81%` | **`+11.26 pp` (`0.7718`)** | `1.353 / 2.321` (`0.359`) ms |
| **MAIN** (`10` seeds) | `sqr_g0` | `53.29 ± 17.96` | `32.56` | `29.31` | `4.10` | `6.31` | `12.37` (`22.60`) ms | `0.991` | `71.66%` | `+3.01 pp` (`0.5588`) | `1.579 / 2.876` ms |
| **MAIN** (`10` seeds) | `sqr_g1` | `56.27 ± 21.24` | `36.40` | `31.82` | `3.77` | `6.49` | `19.15` (`44.98`) ms | `0.992` | `70.60%` | base (`0.00 pp`) | `1.680 / 2.971` ms |
| **MAIN** (`10` seeds) | `priority_sqr` | `56.99 ± 22.24` | `51.34` | **`7.16`** | `28.44` | `3.96` | **`7.63`** (`11.45`) ms | `0.679` | `92.40%` | `-0.73 pp` (`0.4831`) | `1.384 / 2.517` (`0.026`) ms |
| **MAIN** (`10` seeds) | `equal` | `60.79 ± 27.26` | `48.52` | `23.84` | `0.43` | `5.08` | `8.91` (`13.21`) ms | `0.999` | `100.00%` | `-4.58 pp` (`0.4297`) | `0.048 / 0.060` ms |
| **MAIN** (`10` seeds) | `kbrl` | `63.19 ± 26.05` | `50.63` | `26.73` | `2.95` | `3.58` | `9.98` (`14.08`) ms | `0.997` | `98.46%` | `-7.01 pp` (`0.4242`) | `1.974 / 3.115` ms |
| **MAIN** (`10` seeds) | `prop` | `82.70 ± 20.17` | `18.21` | `22.93` | `74.35` | `9.62` | `9.67` (`14.47`) ms | `0.000` | `98.46%` | `-26.75 pp` (`0.1375`) | `0.155 / 0.205` ms |
| **OOS** (`5` seeds) | `equal` | **`42.31 ± 34.25`** | `35.46` | `33.77` | **`0.06`** | `5.35` | `10.01` (`13.89`) ms | `1.000` | `100.00%` | `+4.63 pp` (`0.6198`) | `0.047 / 0.060` ms |
| **OOS** (`5` seeds) | `sqr_g1` | `46.94 ± 28.55` | `23.36` | `38.98` | `3.02` | `5.94` | `17.50` (`54.77`) ms | `0.987` | `68.42%` | base (`0.00 pp`) | `1.499 / 2.882` ms |
| **OOS** (`5` seeds) | **`sqr_cer`** | `49.04 ± 28.20` | `26.79` | `38.36` | `0.56` | `6.76` | `20.10` (`55.98`) ms | `0.998` | `100.00%` | `-2.10 pp` (`0.5417`) | `1.287 / 2.245` (`0.301`) ms |
| **OOS** (`5` seeds) | `sqr_g0` | `50.68 ± 30.08` | `28.80` | `43.58` | `2.62` | `7.38` | `15.72` (`33.65`) ms | `0.987` | `74.82%` | `-3.74 pp` (`0.3256`) | `1.524 / 2.860` ms |
| **OOS** (`5` seeds) | `priority_sqr` | `59.63 ± 26.25` | `51.33` | **`30.09`** | `19.81` | `3.22` | **`9.69`** (`14.47`) ms | `0.738` | `81.27%` | `-12.69 pp` (`0.1659`) | `1.142 / 2.593` (`0.026`) ms |
| **OOS** (`5` seeds) | `kbrl` | `62.62 ± 26.30` | `54.26` | `36.79` | `2.59` | `3.37` | `13.55` (`25.45`) ms | `0.995` | `98.39%` | `-15.68 pp` (`0.2829`) | `1.692 / 2.941` ms |
| **OOS** (`5` seeds) | `prop` | `96.70 ± 3.24` | `7.35` | `47.28` | `80.49` | `10.60` | `12.46` (`17.50`) ms | `0.000` | `92.83%` | `-49.75 pp` (`0.0185`) | `0.145 / 0.199` ms |

---

## 5. Síntese Comparativa Lado a Lado e Principais Achados

| Dimensão | Cenário 1 — Fiel `v1` (`2000 m`, `3×eMBB`, `λ=2/min`) | Cenário 2 — Fiel `v2` (`300 m`, `3×eMBB`, `λ=2/min`) | Cenário 3 — Adaptado `v2` (`300 m`, Het `3,3,3`, `MAIN` `10` seeds) |
|---|---|---|---|
| **Melhor controlador (violação conjunta)** | `equal` `65.28%` / `sqr_g1` `67.47%` (todos inviáveis) | `equal` `0.05%` (`100%` PRBs) vs `sqr_g1` `19.91%` (`35.5%` PRBs) | **`sqr_cer` `45.14%`** vs `sqr_g1` `56.27%` vs `equal` `60.79%` vs `kbrl` `63.19%` |
| **Desempenho `SQR γ=1` vs `KBRL`** | `UE_viol`: `63.87%` vs `69.54%` (economia PRB: `51.5%` vs `100%`) | `UE_viol`: `16.32%` vs `6.90%` (economia PRB: `35.5%` vs `86.5%`) | Conjunta `MAIN`: `56.27%` (`sqr_g1`) / **`45.14%` (`sqr_cer`)** vs `63.19%` (`kbrl`) |
| **Eficiência espectral (`PRB Util`)** | `sqr`: `50.9–51.5%` · `kbrl/equal`: `100%` | `sqr`: `32.7–35.5%` · `kbrl`: `86.5%` · `equal`: `100%` | `sqr_g1`: `70.6%` · **`sqr_cer`: `98.8%`** (realoca sobra para reduzir SLA sem desperdiçar) |
| **Latência de controle (`p95`)** | `SQR`: `10.32–10.94 ms` | `SQR`: `2.00–2.11 ms` (**5× mais rápido**) | `sqr_cer`: `2.32 ms` total (`cer`: **`0.359 ms`**) |
| **Confiabilidade física & IPC** | TDD `7 ms` legado (`máx. 58` slots/jan.) | TDD `5 ms` (`máx. 80` slots/jan.), `cons_err = 0.0`, `0` holds/gaps | TDD `5 ms` (`80` slots), `cons_err = 0.0`, `0` holds/gaps, conforme 3GPP |

### Principais conclusões para o artigo
1. **Inviabilidade do cenário literal do artigo (`2000 m / 30 dBm / 1×1 SISO`):** Mesmo sem bugs e em episódios de `90 s`, todas as políticas sofrem `63.9%–69.5%` de violação média em `UE-time` por déficit de link budget nas bordas (`seeds 101` e `103` entre `89.6%` e `99.0%`).
2. **Quando o canal é viável (`300 m`), a heterogeneidade de slices define onde o aprendizado seguro se justifica:**
   - Em tráfego leve e homogêneo (`3× eMBB`, `λ=2/min`, SLAs baixos de `0,5–1,5 Mbps`), a partição estática `equal` resolve o problema por força bruta (`0.03%` de violação consumindo `100%` da célula), enquanto `SQR` economiza `~65%` dos PRBs (`35.5%` de uso), pagando custo de exploração online (`16.32%` em `648` passos).
   - Sob **contenção heterogênea real (`eMBB + URLLC + mMTC`, `λ=3,3,3`)**, `equal` falha (`60.79%` no `MAIN`), `prop` zera o `mMTC` (`PDR = 0.0`) e `priority_sqr` protege o `URLLC` (`7.16%`) à custa de `eMBB` (`51.34%`) e `mMTC` (`28.44%`). O **`SQR-CER`** passa a liderar no `MAIN` (**`45.14%`**, ganho pareado de **`+11.26 pp`** e win ratio **`0.7718`** sobre `sqr_g1`), mantendo `mMTC PDR = 1.014` (`0.71%` violação) e `eMBB = 7.02 Mbps`, com ressalva transparente de generalização no `OOS` (`49.04%` geral, vitória em `3/5` seeds e `win_ratio = 0.5417`, com perda concentrada na `seed=3101`).
3. **Saneamento técnico `v1 → v2`:** A vetorização do kernel Matérn e a migração para `TDD 5 ms` reduzem a latência `p95` do SQR em **~5×** (`~10.9 ms → ~2.0–2.3 ms`, bem abaixo do timeout de `50 ms` do RIC), com `rb_win_conservation_abs_err = 0.0` e `0` perdas de sincronismo em todas as execuções.
