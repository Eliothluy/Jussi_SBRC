# GATE CONTEXTUAL — E2: Análise Discriminativa Pré-Decisão (Gate G1)

**Base de dados causal:** [`GATE_CONTEXTUAL_ESTADOS.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/GATE_CONTEXTUAL_ESTADOS.csv) (`25` estados, `152` colunas, `0` violações temporais)  
**Script reprodutível:** [`gate_contextual_analyze.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/gate_contextual_analyze.py) · **Resultados JSON:** [`GATE_CONTEXTUAL_ANALYSIS.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/GATE_CONTEXTUAL_ANALYSIS.json)  
**Regra metodológica aplicada:** análise descritiva **sem ajuste/calibração de limiares** sobre os `25` estados e sem usar colunas do Bloco B (`off_*`) como variáveis preditoras.

---

## 1. Resumo Executivo e Veredito G1

**Parecer G1:** **`FRÁGIL` -> `NO-GO para heurística contextual simples ajustada sobre estes 25 estados`** *(nota de escopo: este `NO-GO` incide sobre a heurística contextual imediata nesta amostra restrita, **sem refutar** a viabilidade científica geral da linha de identificação + alocação inter-slice em formulação e amostra adequadas)*.

Embora existam **dois padrões físicos legítimos pré-decisão** no subconjunto eMBB (colapsos abruptos de alocação do `B0-rest` em eMBB/URLLC e penalização de `prop` sob competição `mMTC` com $n_m \ge 4$ e `b0_f_exec_e < 0`), **não há sinal discriminativo replicável para o gatilho crítico URLLC** (`p95 > 10 ms`), o único `HARM` de `equal` em `s2103` (`w1621`: `8,1 -> 10,1 ms`) é indistinguível antes da decisão de seus vizinhos `BENIGN`/`RESCUE`, o universo está **100% confundido por `seed`** (`2102` vs `2103`), existem apenas **`2` eventos `HARM` por opção** e `1/6` resgate de `equal` (`lam6_s2103_w1254`) advém do efeito colateral de **reserva estática fixa (`18` PRBs com `n_e = 0`) absorvendo uma chegada eMBB intra-janela**, não replicável por políticas que zeram slices ociosas (`active_mask`).

---

## 2. Proveniência Temporal das Variáveis (`feature_time_provenance`)

| Grupo de colunas em `GATE_CONTEXTUAL_ESTADOS.csv` | Bloco | Instante de observação (`observed_at`) | Defasagem até `decision_at = gw * 100 ms` | Disponibilidade em `ALL3_COMPLETE (N=18)` |
|---|---|---|---|---|
| `state_id, run_tag, lam, seed, step_index, state_window_index, governed_window_index, decision_at_ms, universe, stratum_id, aa_parity_b0_eq_prop, clean_cut_step` | Chave / Metadado | Fixado no instante `gw * 100 ms` (`sw = gw - 1`) | `0 ms` (chaves/auditoria pré-registrada) | `18/18` (`aa_parity = True`) |
| `n_{e,u,m}, n_ues_total, queue_{e,u,m}, offered_{e,u,m}_mbps, snr_{e,u,m}_db, n_active_causal, active_mask, live_floor_mask, tx_pkts_*_wprev, prev_a_*, prev_prb_sched_*, prev_prb_idle_est, prev_kpi_e_mbps, prev_kpi_m_pdr, urllc_pool_open_at_dec, mmtc_rolling_active_at_dec, viol_e_prev, viol_m_prev` | **Bloco A** (KPM fechado de `sw`) | Final da janela `sw = gw - 1` (`<= gw * 100 ms`) | `0` janelas no instante da decisão (reflete a janela recém-encerrada `sw`) | `18/18` (`prev_kpi_e_mbps = NaN` apenas em `w1254` onde `n_e = 0`) |
| `prev_kpi_u_p95_imm, viol_u_prev_imm` | **Bloco A** (KPM imediato URLLC) | Tentativa ao fim de `sw` | `0` janelas quando fecha; `OPEN` caso contrário | **`0/18` fechados** (`18/18` `NaN` — pool aberto em `17/18`, sem tráfego em `w2552`) |
| `hist10_embb_viol_cnt, hist10_embb_obs_cnt, hist10_mmtc_viol_cnt, hist10_proj_cnt, hist10_urllc_imm_viol_cnt` | **Bloco A** (`decisions.jsonl` histórico) | Janelas com `outcome_window_index <= sw` (`<= gw * 100 ms`) | `>= 1` janela fechada | `18/18` |
| `hist10_urllc_closed_delayed_cnt, hist10_urllc_closed_delayed_viol_cnt, last_closed_urllc_win, last_closed_urllc_obs_win, last_closed_urllc_lag_from_dec, last_closed_urllc_p95_ms` | **Bloco A** (Emenda 4 `urllc_delayed_updates.jsonl`) | Janelas URLLC fechadas com `observed_at_window <= sw` | `lag = 2` janelas (`200 ms`) em `17/18`; `lag = 631` em `w2552` | `18/18` com último valor histórico; **sujeito a censura de perdas** |
| `b0_a_raw_*, b0_a_prop_*, b0_a_exec_*, b0_sum_exec, b0_idle_prbs, b0_proj_triggered, b0_proj_delta_l1, b0_sum_proposals_over_C, b0_prop_excess_gt53, b0_mode_*, b0_tau_*, b0_f_exec_*, b0_sig_exec_*` | **Bloco A** (`B0-rest` pré-envio) | Calculado online antes de enviar ação para `gw` (`<= gw * 100 ms`) | `0 ms` (sinal interno do controlador nominal) | `18/18` |
| `eq_*, eq_act_*, prop_*, delta_eq_minus_b0_*, l1_b0_eq, delta_prop_minus_b0_*, l1_b0_prop, b0_f_{eq,prop}_*, delta_f_{eq,prop}_minus_b0_*` | **Bloco A** (braços candidatos + avaliação no modelo SQR pré-decisão) | Determinístico a partir de `state` e `per_slice_model` em `gw * 100 ms` | `0 ms` (sem executar replay) | `18/18` |
| `off_{b0,eq,prop}_*` e `off_label_{eq,prop}_*` | **Bloco B** (**OFFLINE PROIBIDO COMO FEATURE**) | Pós-replay de janela `gw` em horizonte `+3 s` (`> decision_at_ms`) | Pós-decisão (`>= +100 ms` a `+3000 ms`) | Exclusivo para rótulos e auditoria offline |

---

## 3. Quadro Pareado de Rótulos por Opção, Universo, `seed` e $\lambda$

Definições estritas sobre pares avaliáveis em relação ao `B0-rest`: `RESCUE` (`B0=FAIL, k=PASS`), `HARM` (`B0=PASS, k=FAIL`), `BENIGN` (`B0=PASS, k=PASS`), `NO_RESCUE` (`B0=FAIL, k=FAIL`), `UNKNOWN` (qualquer braço `UNKNOWN`).

### 3.1 Síntese por Universo e Métrica (`equal` vs `prop`)

| Universo / Corte | Opção $k$ | $N_{\text{total}}$ | $N_{\text{eval}}$ | `B0` PASS | $k$ PASS | `RESCUE` | `HARM` | `BENIGN` | `NO_RESCUE` | `UNKNOWN` | `NET_RESCUE` | Trocas úteis a identificar vs abster |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`ALL3_COMPLETE` (`STRICT_ALL3`)** | **`equal`** *(cego `18-17-18`)* | **18** | **18** | **8** | **12** | **6** | **2** | **6** | **4** | **0** | **+4** | Identificar 6 resgates (5 sob `active_mask` + `w1254` via reserva estática) e abster nos 2 `HARM` (e idealmente nos 6 `BENIGN` + 4 `NO_RESCUE`) |
| **`ALL3_COMPLETE` (`STRICT_ALL3`)** | **`prop`** *(ativo)* | **18** | **18** | **8** | **11** | **5** | **2** | **6** | **5** | **0** | **+3** | Identificar 5 resgates e abster nos 2 `HARM` |
| `ALL25` (`STRICT_ALL3` completo) | `equal` | 25 | 18 | 8 | 12 | 6 | 2 | 6 | 4 | **7** | +4 | 7 `UNKNOWN` (`4` em $\lambda=2$, `2` em $\lambda=5$, `1` em $\lambda=6$) |
| `ALL25` (`STRICT_ALL3` completo) | `prop` | 25 | 18 | 8 | 11 | 5 | 2 | 6 | 5 | **7** | +3 | Idem acima (`7` `UNKNOWN`) |
| `LT3_DEMAND` (`STRICT_DEMAND`) | `equal` | 5 | 5 | 2 | 3 | **1** | 0 | 2 | 2 | 0 | +1 | O único `RESCUE` (`lam2_s2102_w2685`) nasce de `B0=(0,0,0)` com $n_m=1$ (e `eq` aloca 35 PRBs inativos) |
| `LT3_DEMAND` (`STRICT_DEMAND`) | `prop` | 5 | 5 | 2 | 3 | **1** | 0 | 2 | 2 | 0 | +1 | Mesmo resgate (`lam2_s2102_w2685`) com `prop=(0,0,53)`, sem desperdiçar PRBs |
| Complementar `eMBB∧URLLC` (`ALL25`) | `equal` | 25 | 19 | 9 | 13 | 6 | 2 | **7** | 4 | **6** | +4 | `+1 BENIGN` em `lam2_s2102_w2442` (`e+u` avaliáveis sem mMTC) |
| Complementar `eMBB∧URLLC` (`ALL25`) | `prop` | 25 | 19 | 9 | 12 | 5 | 2 | **7** | 5 | **6** | +3 | Idem acima (`+1 BENIGN`) |

*\*Nota crítica sobre `equal` em `lam6_s2103_w1254` (distinção entre alocação causalmente válida e uso de informação futura):* no instante da decisão (`decision_at_ms = 125400 ms`, fim de `sw = 1253`), `n_e = 0, queue_e = 0, offered_e = 0` (`active_mask = 011`). Durante a janela futura `gw = 1254`, chegou um novo fluxo eMBB (`step_metrics.csv step=1254, ts=125500, n_ues=1`). É essencial **não confundir** o efeito de uma política estática com **vazamento de informação futura**:
- **`equal = (18, 17, 18)` é 100% causal no tempo** (`observed_at <= decision_at`): ele não utiliza nenhuma informação do futuro sobre a chegada em `gw = 1254`; simplesmente mantém uma **reserva fixa cega à demanda (`18` PRBs)** em toda janela, absorvendo por acaso a chegada eMBB intra-janela (`6,8626 Mbps`, `RESCUE`) ao preço de alocar PRBs em fatias ociosas em `9/25` estados (`17–36` PRBs).
- **`B0 = (0, 40, 13)`, `prop = (0, 35, 18)` e `eq_act = (0, 27, 26)` também são alocações causalmente válidas** sob a regra de zeragem de fatia sem demanda observada (`active_mask`), porém permanecem vulneráveis a chegadas estocásticas surgidas após `125400 ms` (`0,0 Mbps`, `FAIL`).
- **Política que utilizaria informações futuras (inexistente e proibida)** seria apenas aquela que alocasse `a_e > 0` *seletivamente* em `w1254` antecipando a chegada futura em `gw = 1254`. Assim, a lição de `w1254` (e de seu análogo `w2552` com `n_u = 0`) não é vazamento em `equal`, mas o trade-off de projeto entre **zeragem estrita por `active_mask`** (`equal_active` cai para `5 RESCUE / 2 HARM`) e **manutenção de piso protetor mínimo em fatias ociosas**.

### 3.2 Quebra por `seed` e por $\lambda$ em `ALL3_COMPLETE` (`N = 18`)

| Eixo | Estrato | $N$ | `equal` (`R / H / B / NR`) | `prop` (`R / H / B / NR`) | Observação estrutural |
|---|---|---:|---|---|---|
| **Seed** | `2102` | 9 | `1 / 1 / 3 / 4` (`net = 0`) | `1 / 2 / 2 / 4` (`net = -1`) | `equal` não gera ganho líquido em `2102`; `prop` tem saldo negativo |
| **Seed** | `2103` | 9 | `5 / 1 / 3 / 0` (`net = +4`) | `4 / 0 / 4 / 1` (`net = +4`) | **100% do ganho líquido concentrado em `seed = 2103`** (`4` estados URLLC + `1` por reserva fixa `w1254` em `equal`) |
| **Carga** | $\lambda = 5$ | 9 | `3 / 2 / 2 / 2` (`net = +1`) | `3 / 1 / 3 / 2` (`net = +2`) | Todos os `2` `HARM` de `equal` ocorrem em $\lambda = 5$ (`w1890`, `w1621`) |
| **Carga** | $\lambda = 6$ | 9 | `3 / 0 / 4 / 2` (`net = +3`) | `2 / 1 / 3 / 3` (`net = +1`) | Único `HARM` de $\lambda = 6$ é `prop` em `w2229` (`s2102`) |

### 3.3 Degradação de KPI em Estados `PASS/BENIGN` (Separada de `HARM`, §3.1)
Nos `6` estados `BENIGN` de `ALL3_COMPLETE`, intervir não viola o SLA (`STRICT_ALL3 = PASS`), mas penaliza a margem eMBB por reduzir a alocação nominal de `B0`:
* **`equal` (`BENIGN`, `n = 6`):** $\Delta\text{eMBB}_{\text{thr}}$ mediana **`-0,856 Mbps`** (`-4,645` a `+3,545 Mbps`); $\Delta\text{URLLC}_{p95}$ mediana **`0,000 ms`** (`0,000` a `0,000 ms`).
* **`prop` (`BENIGN`, `n = 6`):** $\Delta\text{eMBB}_{\text{thr}}$ mediana **`-4,890 Mbps`** (`-6,846` a `+0,489 Mbps`); $\Delta\text{URLLC}_{p95}$ mediana **`0,000 ms`** (`0,000` a `+0,933 ms` em `w1588`). Nos `5` `RESCUE` de `prop`, o ganho em URLLC continua acompanhado de redução de $\text{eMBB}$ (mediana `-5,134 Mbps`, mantendo-se acima de `3,5 Mbps`).

---

## 4. Tabela Completa dos 18 Estados `ALL3_COMPLETE` (Bloco A × Bloco B)

| `state_id` | $\lambda$/seed | $n_{(e,u,m)}$ | `prev_a` | `a_B0_exec` (`proj ΔL1`) | `prop` | `prev_e` (Mbps) | `last_u_p95` (`lag`, `h10_u`) | `b0_f_e` | `B0` | `eq` | `prop` | $\text{eMBB}_{B0 \to eq \to prop}$ | $\text{URLLC}_{B0 \to eq \to prop}$ |
|---|---|---:|---|---|---|---|---|---:|---|---|---|---|---|
| `lam5_s2102_w1654` | 5/2102 | `(1,1,3)` | `(10,24,8)` | `(10,25,7)` (0) | `(11,10,32)` | `4,40` | `8,10` (2, 0) | `+0,006` | `PASS` | `BENIGN` | `BENIGN` | `4,28 -> 7,82 -> 4,77` | `8,1 -> 8,1 -> 8,1` |
| **`lam5_s2102_w1890`** | 5/2102 | **`(3,1,4)`** | `(37,11,5)` | `(32,17,4)` (19) | `(20,7,26)` | `5,01` | `8,10` (2, 1) | **`-0,049`** | `PASS` | **`HARM [e]`** | **`HARM [e]`** | **`4,52 -> 2,57 -> 2,81`** | `8,1 -> 8,1 -> 8,1` |
| **`lam5_s2102_w2667`** | 5/2102 | `(2,1,3)` | `(23,27,3)` | **`(7,33,4)`** (0) | `(18,9,26)` | `4,83` | `8,10` (2, 2) | `-0,623` | `FAIL` | **`RESCUE [e]`** | **`RESCUE [e]`** | **`1,53 -> 3,91 -> 3,91`** | `8,1 -> 8,1 -> 8,1` |
| `lam5_s2102_w2578` | 5/2102 | `(3,1,3)` | `(30,19,4)` | `(27,20,6)` (28) | `(23,7,23)` | `4,03` | `10,15` (2, 6) | `-0,520` | `FAIL` | `NO_RESCUE` | `NO_RESCUE` | `3,30 -> 2,44 -> 2,93` | `8,15 -> 8,1 -> 8,15` |
| `lam5_s2102_w2552` | 5/2102 | `(3,0,3)` | `(45,0,8)` | `(45,0,8)` (5) | `(27,0,26)` | `5,75` | `9,10` (631, 0) | `-0,339` | `FAIL` | `NO_RESCUE` | `NO_RESCUE` | `5,62 -> 2,32 -> 3,42` | `64,05 -> 9,10 -> 64,05` |
| `lam6_s2102_w1406` | 6/2102 | `(1,1,3)` | `(14,37,2)` | `(20,23,1)` (0) | `(11,10,32)` | `6,36` | `8,10` (2, 1) | `+0,010` | `PASS` | `BENIGN` | `BENIGN` | `8,56 -> 7,70 -> 4,65` | `8,1 -> 8,1 -> 8,1` |
| **`lam6_s2102_w2229`** | 6/2102 | **`(2,1,4)`** | `(20,17,16)` | `(21,11,21)` (22) | `(15,8,30)` | `4,40` | `9,08` (2, 0) | **`-0,512`** | `PASS` | `BENIGN` | **`HARM [e]`** | **`4,46 -> 3,85 -> 3,18`** | `8,1 -> 8,1 -> 8,1` |
| `lam6_s2102_w1396` | 6/2102 | `(1,1,3)` | `(10,30,5)` | `(10,5,4)` (0) | `(11,10,32)` | `4,28` | `8,10` (2, 2) | `+0,011` | `FAIL` | `NO_RESCUE` | `NO_RESCUE` | `3,55 -> 6,48 -> 3,91` | `13,10 -> 10,15 -> 13,10` |
| `lam6_s2102_w2153` | 6/2102 | `(4,1,5)` | `(21,18,14)` | `(23,11,19)` (41) | `(21,5,27)` | `1,96` | `9,10` (2, 5) | `-0,590` | `FAIL` | `NO_RESCUE` | `NO_RESCUE` | `2,08 -> 1,65 -> 1,96` | `10,15 -> 10,15 -> 10,15` |
| `lam5_s2103_w1588` | 5/2103 | `(1,2,1)` | `(28,13,4)` | `(28,13,7)` (0) | `(13,27,13)` | `11,25` | `8,10` (2, 1) | `+1,419` | `PASS` | `BENIGN` | `BENIGN` | `12,10 -> 7,82 -> 5,62` | `8,1 -> 8,1 -> 9,03` |
| **`lam5_s2103_w1621`** | 5/2103 | **`(1,2,1)`** | `(27,13,9)` | `(27,13,13)` (1) | `(13,27,13)` | `11,37` | **`8,10` (2, 0)** | `+1,504` | `PASS` | **`HARM [u]`** | `BENIGN` | `11,98 -> 7,95 -> 6,11` | **`8,1 -> 10,1 -> 8,1`** |
| **`lam5_s2103_w1546`** | 5/2103 | `(1,2,1)` | `(27,13,2)` | `(27,13,2)` (0) | `(13,27,13)` | `11,74` | `10,03` (2, 3) | `+1,193` | `FAIL` | **`RESCUE [u]`** | **`RESCUE [u]`** | `10,88 -> 7,82 -> 5,38` | **`10,1 -> 8,1 -> 8,1`** |
| **`lam5_s2103_w1560`** | 5/2103 | `(1,2,1)` | `(27,12,14)` | `(27,12,14)` (3) | `(13,27,13)` | `11,98` | `12,10` (2, 1) | `+1,318` | `FAIL` | **`RESCUE [u]`** | **`RESCUE [u]`** | `10,76 -> 7,58 -> 5,62` | **`12,1 -> 8,1 -> 8,1`** |
| `lam6_s2103_w1378` | 6/2103 | `(1,2,1)` | `(21,32,0)` | `(29,9,0)` (0) | `(13,27,13)` | `9,29` | `8,10` (2, 1) | `+1,169` | `PASS` | `BENIGN` | `BENIGN` | `12,47 -> 7,82 -> 5,62` | `8,1 -> 8,1 -> 8,1` |
| `lam6_s2103_w1361` | 6/2103 | `(1,2,1)` | `(17,32,4)` | `(20,28,5)` (22) | `(13,27,13)` | `7,33` | `8,10` (2, 2) | `+0,850` | `PASS` | `BENIGN` | `BENIGN` | `8,56 -> 7,70 -> 5,50` | `8,1 -> 8,1 -> 8,1` |
| **`lam6_s2103_w1283`** | 6/2103 | `(1,2,1)` | `(19,33,1)` | **`(28,2,2)`** (0) | `(13,27,13)` | `8,43` | `8,10` (2, 0) | `+1,097` | `FAIL` | **`RESCUE [u]`** | **`RESCUE [u]`** | `12,10 -> 7,82 -> 5,62` | **`27,1 -> 8,1 -> 8,1`** |
| **`lam6_s2103_w1397`** | 6/2103 | `(1,2,1)` | `(18,26,9)` | `(19,24,10)` (26) | `(13,27,13)` | `7,70` | `8,10` (2, 0) | `+0,904` | `FAIL` | **`RESCUE [u]`** | **`RESCUE [u]`** | `7,70 -> 7,58 -> 5,62` | **`14,15 -> 8,1 -> 8,1`** |
| **`lam6_s2103_w1254`** | 6/2103 | **`(0,2,1)`** | `(0,6,16)` | `(0,40,13)` (1) | `(0,35,18)` | `NaN` | `8,10` (2, 0) | `+0,013` | `FAIL` | **`RESCUE [e]*`** | `NO_RESCUE` | `0,00 -> 6,86 -> 0,00` | `8,1 -> 8,1 -> 8,1` |

*\*Nota:* em `w1254`, `RESCUE [e]` de `equal` decorre da **reserva fixa causal `(18,17,18)` sem zeragem por `active_mask`** (`action_e = 18` com `n_e = 0` no instante `125400 ms`), que protege uma chegada eMBB ocorrida durante a janela `1254` (sem uso de informação futura, ao custo de alocar PRBs em slice ociosa no momento da decisão).

---

## 5. Varredura Descritiva de Sinais Pré-Decisão (§3.2 — Respostas às 6 Questões)

### 5.1 O Efeito Dominante: Confundimento Completo por Seed (`2102` vs `2103`)
Antes de interpretar qualquer AUROC univariado agregado em `ALL3_COMPLETE`, é obrigatório explicitar que os `9` estados de `seed=2102` e os `9` estados de `seed=2103` ocupam **regimes físicos disjuntos**:

| Feature pré-decisão | `seed = 2102` (`n = 9`: `1R/1H` eq; `1R/2H` prop) | `seed = 2103` (`n = 9`: `5R/1H` eq; `4R/0H` prop) |
|---|---|---|
| `n_u` | `0..1` (mediana `1`) | **`2..2`** (mediana `2`, `9/9`) |
| `n_m` | `3..5` (mediana `3`) | **`1..1`** (mediana `1`, `9/9`) |
| `offered_u_mbps` | `0,000..0,499` (mediana `0,475`) | **`0,941..0,979`** (mediana `0,960`) |
| `b0_f_exec_e` | `-0,623..+0,011` (mediana `-0,339`) | **`+0,013..+1,504`** (mediana `+1,169`) |

Consequentemente, qualquer feature correlacionada à seed (`n_u`, `n_m`, `offered_u_mbps`, `b0_f_exec_e`, `queue_e`, `delta_f_prop_minus_b0_u`) aparenta alto AUROC agregado (`0,75–0,90`) apenas porque **`5/6` resgates de `equal` (`4/5` em `prop`) ocorreram em `seed=2103`**. O teste real precisa resistir **dentro da mesma seed**.

### 5.2 Análise Dentro da Mesma Seed: Onde Há Sinal Real vs Onde Falta Sinal

1. **Sinal real #1 — Risco de dano eMBB por corte excessivo (`seed = 2102`):**
   * Nos `2` estados `HARM` de `prop` (`w1890`: `b0=(32,17,4) -> prop=(20,7,26)` e `w2229`: `b0=(21,11,21) -> prop=(15,8,30)`), e também no `HARM` de `equal` (`w1890`: `-> (18,17,18)`), **há competição pesada** ($n_m = 4$, projeção ativa `b0_proj_delta_l1 = 19` e `22`, `queue_e = 8,39 MB` e `4,89 MB`, **`b0_f_exec_e < 0`** em `-0,049` e `-0,512`), e a intervenção **corta fortemente eMBB** (`Δa_e = -12 a -14` em `w1890`, `-6` em `w2229`).
   * Limitação desse sinal: **dentro de `2102`, ele NÃO separa `HARM` (`w1890`) do único `RESCUE` (`w2667`)** por si só: em `w2667`, `b0_f_exec_e = -0,623` era ainda pior, mas `B0` havia colapsado `a_e` de `23 -> 7` PRBs (dando `33` para `1` UE URLLC) — logo `equal`/`prop` **aumentaram** `a_e` (`+11`), em vez de cortá-lo.

2. **Sinal real #2 — Colapsos abruptos de `B0` observáveis em Bloco A (`seed 2102` + `2103` + `LT3`):**
   * Em **`w2667`** (`s2102`, `n=(2,1,3)`): `B0` reduziu `a_e: 23 -> 7` (`3,5 PRB/UE`), derrubando eMBB para `1,53 Mbps` (`RESCUE` por `eq` e `prop`).
   * Em **`w1283`** (`s2103`, `n=(1,2,1)`): `B0` em modo `softmax`/`knn` reduziu `a_u: 33 -> 2` PRBs (`2` PRBs para `2` UEs URLLC!), elevando `p95` para `27,1 ms` (`RESCUE` por `eq` e `prop`).
   * Em **`w2685`** (`LT3_DEMAND_COMPLETE`, `s2102`, `n=(0,0,1)`): `B0` propôs `(0,0,0)` deixando `mMTC` ativo com `0` PRBs (`PDR 0,0 -> 1,0`, `RESCUE` por `eq` e `prop`).
   * **Esses `3` colapsos são observáveis causalmente antes da ação**, pois constam em `(n_s, a_B0_exec)`.

3. **AUSÊNCIA DE SINAL CRÍTICO — O enigma URLLC dentro de `seed = 2103` (`w1621` vs `w1588` vs `w1546` vs `w1560` vs `w1397`):**
   * Observe os quatro estados de `lam5_s2103` em sequência quase idêntica no **Bloco A**:
     * `w1588` (`BENIGN`): `n=(1,2,1)`, `q=(2,78 MB, 138 B)`, `a_B0=(28,13,7)`, `eq=(18,17,18)` -> `B0 u_p95=8,1`, `eq u_p95=8,1`.
     * **`w1621` (`HARM` em `equal`)**: `n=(1,2,1)`, `q=(2,58 MB, 138 B)`, `a_B0=(27,13,13)`, `last_u_p95=8,1 (lag=2, h10_u=0)`, `eq=(18,17,18)` -> mesmo **aumentando `a_u` de `13 -> 17`**, `equal` subiu `u_p95` de **`8,1 -> 10,1 ms` (`HARM [u]`)**, enquanto `prop=(13,27,13)` deu `8,1 ms`!
     * **`w1546` (`RESCUE` em `eq/prop`)**: `n=(1,2,1)`, `q=(3,72 MB, 138 B)`, `a_B0=(27,13,2)` -> com os **mesmos `13` PRBs em URLLC** que `w1621` e `w1588`, `B0` deu `10,1 ms` (`FAIL`) e `eq=(18,17,18)` deu `8,1 ms` (`RESCUE`)!
     * Além disso, em **`w1397`** (`lam6_s2103`), **`B0` deu `24` PRBs a URLLC e violou (`14,15 ms`)**, enquanto `equal` deu apenas **`17` PRBs a URLLC e passou (`8,10 ms`, `RESCUE`)**!
   * **Interpretação física:** em janelas de `100 ms` com `1–2` UEs URLLC enviando datagramas esparsos (`69 B/pacote`) perto do limiar rígido de `10 ms`, o `p95` sofre alta variância por **fase exata de chegada futura dentro dos slots TDD DL/UL** e **escalonamento intra-slot do MAC** — fatores **não observáveis no instante `decision_at_ms`** (e com `prev_kpi_u_p95_imm = NaN` em `18/18` estados).

### 5.3 Respostas Explícitas às 6 Questões do §3.2

1. **A feature existia antes da decisão? Com qual atraso?** Todas as `105` features do Bloco A existem com `observed_at <= decision_at_ms` (`0 ms` para estado/`a_B0`/modelo; `>= 2` janelas = `200 ms` para `last_closed_urllc_p95_ms`; **`18/18` `NaN`** para `prev_kpi_u_p95_imm`).
2. **Diferencia `RESCUE` de `HARM` e de `NO_RESCUE/BENIGN`?** Apenas parcialmente e no recorte eMBB/colapso (`2–3` estados claros em `25`). Para `RESCUE vs HARM` em URLLC (`s2103`), `w1621` (`HARM`) tem estado Bloco A virtualmente indistinguível de `w1588` (`BENIGN`) e `w1546` (`RESCUE`). Os `4` estados `NO_RESCUE` (`w2578, w2552, w1396, w2153`) misturam sobrecarga severa (`prev_kpi_e = 1,96..5,75 Mbps`, `h10_u <= 6`) que nenhum braço resolve com `C = 53`.
3. **Replicável entre seeds/runs ou específico de um caso?** **Não replicável entre seeds**: `5/6` resgates de `eq` (`4/5` em `prop`) e `1/2` `HARM` de `eq` dependem de `s2103`, enquanto `0/2` `HARM` de `prop` se repetem em `s2103`.
4. **Ajuda a tomar decisão `ABSTER-SE / MANTER B0`?** Só sustenta uma **abstenção altamente conservadora**: evitar intervenções que reduzam eMBB quando `b0_f_exec_e` já está negativo/marginal ou quando `B0` não apresenta colapso de PRBs (`a_exec_s <= 2`), deixando sem explicação os `RESCUE` de `w1546`, `w1560`, `w1397` (onde `B0` deu `12, 13, 24` PRBs a URLLC).
5. **Preço de falsos positivos (FP -> `HARM`) vs falsos negativos (FN -> oportunidade perdida)?**
   * **FP (intervir indevidamente):** em `ALL3_COMPLETE`, trocar nos `8` `B0=PASS` causa **`2/8` (`25%`) de `HARM`** tanto em `eq` quanto em `prop`, além de degradação mediana de `-0,86` (`eq`) a `-4,89 Mbps` (`prop`) em `BENIGN`. Em cenário seguro, **cada `HARM` anula 1 `RESCUE` no `NET_RESCUE`**.
   * **FN (abster demais):** abster em estados com `a_B0` não-colapsado perde `3/5` resgates causais de `equal` (`w1546, w1560, w1397`) e `3/5` de `prop`, mantendo `B0` próximo de `44–56%` em vez de `67–72%`.
6. **Relevância nos casos de demanda parcial (`LT3_DEMAND_COMPLETE`, `N=5`)?** Em `LT3_DEMAND`, `prop` e `equal` geram `1 RESCUE / 0 HARM` (`w2685`, `B0=(0,0,0)` com $n_m=1$), porém **`equal` desperdiça `17–36` PRBs em `5/5` estados** por ignorar `active_mask`, tornando `prop_active` ou preenchimento sobre slices ativas estritamente superior nos `LT3`. Em `ALL25_EU`, há apenas `+1 BENIGN` (`w2442`), sem alteração no balanço `RESCUE/HARM`.

---

## 6. Teste Conceitual de Regras Candidatas (Sem Ajuste de Parâmetros)

Avaliação **puramente diagnóstica**, sem qualquer otimização de limiares sobre os `25` estados, de predicados estruturais *a priori* com decisão `{MANTER_B0, TENTAR_equal}` em `ALL3_COMPLETE` (`B0 = 8/18`):

| Regra estrutural *a priori* (sem ajuste) | Acionamentos | `PASS` total | `RESCUE` | `HARM` | `NET` | `seed 2102` (`PASS`, `R/H`) | `seed 2103` (`PASS`, `R/H`) | Leitura científica |
|---|---:|---:|---:|---:|---:|---|---|---|
| `R1`: intervir (`eq_act`) só se colapso estrito `∃s: n_s>0 ∧ a_B0_s = 0` | `0/18` (`1/5` em `LT3`) | `8/18` | `0` | `0` | `0` | `4/9` (`0/0`) | `4/9` (`0/0`) | Muito fraca em `ALL3_COMPLETE`; só evita `w2685` em `LT3` |
| `R2`: intervir (`eq`) se `∃s: n_s>0 ∧ a_B0_s <= 2` (inanição severa) | `4/18` (`w2552` fora pois `n_u=0`) | `10/18` | `2` (`w1283`, ...) | `0` | `+2` | `4/9` (`0/0`) | `6/9` (`2/0`) | Só `w1283` em `s2103` (+ `w1406, w1378` que dão `BENIGN`, mais `w1546` se limiar mMTC); **não explica `w2667, w1546, w1560, w1397`** |
| `R3` (candidata conceitual híbrida): `TENTAR_prop` somente se `(∃s ativa com a_B0_s <= 2)` OU `(sem risco eMBB: b0_f_e >= 0 ∧ n_m < 4 ∧ prop_e >= 13)` | (não calibrada — ilustra dependência da seed) | — | — | — | — | — | — | O segundo termo `b0_f_e >= 0 ∧ n_m < 4` **apenas reproduz a separação `seed 2103` vs `seed 2102`**! |

*(Nota sobre `R2` acima: dos `4` acionados por `a_B0_s <= 2` com `n_s > 0` em `ALL3_COMPLETE`, estão `w1406` (`a_m=1`), `w1546` (`a_m=2`), `w1378` (`a_m=0`), `w1283` (`a_u=2, a_m=2`) — capturando `w1546` (`RESCUE`), `w1283` (`RESCUE`), `w1406` (`BENIGN`), `w1378` (`BENIGN`), com `0 HARM` porque `n_m` tinha `a_m<=2`, o que já revela que acionar por `a_m<=2` seria **acidentalmente correlacionado** aos resgates URLLC de `s2103`, sem mecanismo causal URLLC!)*

Além disso, qualquer regra de separação por `b0_f_exec_e >= 0` ou `n_m < 4` **colapsaria para discriminar `seed 2103` de `seed 2102`**, não podendo ser promovida a política generalizável sem teste em novas seeds.

---

## 7. Comparação com os `12` Estados `ALL3` da `D2-v2` (`equal` vs `a_B0`)

Para conferir se o sinal de `equal` se repete fora da amostra `B0-rest`: nos `30` estados da [`D2v2_results.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/D2v2_results.csv) (sobre `A0` = `D1-v9`, seeds `2101–2103`), **`12` estados** têm `COMPLETE|COMPLETE|COMPLETE` simultâneo para `alt0 (a_B0)` e `alt12 (ref:equal)`:
* Resultado em `D2-v2` (`N = 12`): **`2 RESCUE` (`S2_lam6_s2101_t2824` com `a_u=3->17` e `S3_lam6_s2102_t1420` com `u_p95=10,15->8,10`), `0 HARM`, `3 BENIGN`, `7 NO_RESCUE`** (`B0 3/12` -> `eq 5/12`). Sob `STRICT_DEMAND` (`30` estados): `11 RESCUE, 1 HARM, 7 BENIGN, 11 NO_RESCUE`.
* Isso confirma em uma base distinta (`A0`) que: (i) **`equal` cego recupera principalmente colapsos ou subalocações pontuais (`a_u=3`) e falhas moderadas**, mas (ii) **continua falhando na maioria dos estados congestionados (`7/12 NO_RESCUE`)**, mantendo pequena a fração de contraste limpo.

---

## 8. Conclusão Formal G1

**Classificação G1:** **`FRÁGIL`** -> **`NO-GO` específico para implementar heurística contextual simples sobre estes 25 estados** *(separado da viabilidade geral da linha de identificação + alocação inter-slice, que permanece aberta em formulação adequada e amostra independente)*.

1. **O que existe de sinal legítimo:** subalocação severa de `B0` (`a_u=2`, `a_e=7`, `a=(0,0,0)`), vulnerabilidade de `prop` quando corta `eMBB` com `b0_f_exec_e < 0` e competição `n_m >= 4`, além do contraste estrutural em `w1254`/`w2552` entre zeragem estrita (`active_mask`) e piso fixo causal (`equal=(18,17,18)`).
2. **Por que o conjunto é `FRÁGIL` para uma heurística imediata:** `N=18` em apenas `4` runs/`2` seeds totalmente confundidas; só `2` eventos `HARM` por braço; ausência de preditor Bloco A para falhas/danos URLLC próximos ao corte de `10 ms` (`w1621` vs `w1546`/`w1588`, `w1397`); e `1/6` dos resgates históricos de `equal` (`w1254`) dependente de manter reserva fixa em fatia com `n_e=0` na decisão (mecanismo não compartilhado por `equal_active`).
3. **Consequência (§3.3 / §4 / §6):** ativar a regra de parada para heurísticas manuais (`STOP RULE`), distinguir claramente o `NO-GO` da heurística sobre 25 estados da viabilidade da contribuição geral, e analisar as duas formulações conceituais alternativas sob o Gate G2 atualizado ([`GATE_CONTEXTUAL_03_ORIGINALIDADE_E_ESCOLHA.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/GATE_CONTEXTUAL_03_ORIGINALIDADE_E_ESCOLHA.md)).
