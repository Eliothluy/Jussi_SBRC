# Relatório de Diagnóstico Exploratório Offline (`v1` Preservada / Linha `v2`)

**Escopo de integridade e causalidade:** este documento consolida exclusivamente evidências observacionais obtidas offline sobre artefatos históricos preservados da linha `v1` (`Gate 1 / B0-rest`, `D1-v9`, `D2-v2` e traços contínuos `b0_traced_gate1`), **sem executar simulações `ns-3`, sem tocar seeds confirmatórias (`MAIN {3001..3010}`, `OOS >= 3101`, `G4B_VALIDATION {1201..1209}`), sem abrir nenhum arquivo `sqr_checkpoint.pkl` de fim de episódio (`t = 300 s`, `checkpoints_pkl_opened = 0`), sem implementar novo controlador e sem modificar código fora de `MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/`**.

- Script gerador: [`run_diag_exploratorio_v1.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/run_diag_exploratorio_v1.py)
- Registro JSON consolidado: [`diag_exploratorio_v1_resumo.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/diag_exploratorio_v1_resumo.json)
- Manifesto de integridade pré/pós: [`integridade_v1_verificacao.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/integridade_v1_verificacao.json)

---

## 0. Separação Estrutural dos Conjuntos Analisados

Para evitar contaminação metodológica entre configurações distintas da `v1`:

| Conjunto | Papel | Configuração do Modelo | Base Causal Usada | Tamanho |
|---|---|---|---|---|
| **`Gate 1` (`B0-rest`)** | **Primário** | `A1 + A2a + A_scaler` (`virtual_alpha = 0.0`, `empty_slice_guard = True`, `fix_scaler_index = True`, Emenda 4 ativa, `C = 53`) | Snapshots exatos pré-decisão de `causal_trace.jsonl.gz` (todos pré-lag) | `25` estados, `127` replays únicos (`6` braços lógicos; `aa_prefix_valid = True` em `25/25`) |
| **`D2-v2` (`PRE_DESYNC`)** | **Secundário** | Original `A0` (`D1-v9`: `virtual_alpha = 0.05`, `empty_slice_guard = False`, `fix_scaler_index = False`) | Reconstrução causal *forward* passo a passo desde `t = 0` (`0.0` erro vs `decisions.jsonl`), avaliada em **`A0` exato** e no contrafactual **`Shadow B0-rest`** | `23/30` estados limpos pré-desync (`13` ramos/estado; `S1_lam5_s2101_t2064` via `d2v2_t2064full_branches.csv`) |
| **`D2-v2` (`POST_DESYNC`)** | Excluído das análises causais | `A0` após início de dessincronização IPC (`BUG-03`) | Marcado explicitamente como `POST_DESYNC` | `7/30` estados (incluindo `2` dos `4` `RECOVERABLE_STRICT`) |
| **`b0_traced_gate1`** | **Primário (T5)** | `B0-rest` contínuo pós-warmup (`t > 100 s`) | `causal_trace.jsonl.gz` + `decisions.jsonl` + `urllc_delayed_updates.jsonl` pré-lag vs desfecho offline completo | `6` runs (`lam ∈ {2,5,6} × seeds {2102,2103}`), `25.028` linhas `(step, W_preq, slice)` |

---

## 1. Tarefa 1 — Parâmetros Congelados, Auditoria das 152 Colunas e Telemetria Sub-100 ms

Tabelas geradas: [`t1_b0_rest_manifest_params.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/t1_b0_rest_manifest_params.csv), [`t1_bloco_a_152_colunas_audit.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/t1_bloco_a_152_colunas_audit.csv), [`t1_urllc_queue_causal_extra.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/t1_urllc_queue_causal_extra.csv) e [`t1_cpp_sub100ms_telemetry_audit.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/t1_cpp_sub100ms_telemetry_audit.csv).

### 1.1 Contratos de SLA, Emenda 4 e Hiperparâmetros do `B0-rest`
- **SLAs e margens normalizadas (`sla_manifest.yaml` `1.1-g4b`, limite de satisfação $m_i \ge 0.0$):**
  - **eMBB:** limiar $3,5\text{ Mbps}$, $m_e = (T_e - 3,5)/3,5$ (mediana entre UE-janelas com `dTxBytes > 0`).
  - **URLLC:** limiar $10,0\text{ ms}$ no $p_{95}$ da coorte de chegada `floor(txSec * 10)`, $m_u = (10,0 - p_{95})/10,0$; métrica secundária `miss <= 5,0%` marcada `REPORT_ONLY` (nunca critério de gate). Nota histórica no manifesto: bloco `g4b.status = STOP_FINAL`.
  - **mMTC:** limiar $\text{PDR} = 0,95$ em janela deslizante de $1\text{ s}$ ($10$ janelas), $m_m = (\text{PDR} - 0,95)/0,95$.
- **Regra de fechamento URLLC (Emenda 4):** *pool* de chegada fecha apenas quando `delivered(W) == tx_kpm(W)` com `tx_kpm(W) > 0` (`CLOSED`), emitindo $1$ atualização ao regressor somente nesse instante; janelas sem envio recebem `NO_ACTIVE_DEMAND`; *pools* não fechados expiram em `MAX_AGE = 30` janelas ($3,0\text{ s}$) como `expired_vacuous_R19` (censura à direita sem atualização no modelo).
- **Configuração `B0-rest` (`config_sha256 = a2b40861...`):**
  - Kernel Matérn $\nu = 1,5$ com **`length_scale = 1.0` isotrópico em $\mathbb{R}^7$** sobre $x = [z_i(6),\, a_i / C]$;
  - Memória $B = 2000$ vetores de suporte por slice; poda `R10` remove `argmin_j |alpha_j| * k(x_new, x_j)` (desempate pelo menor índice / mais antigo);
  - $\eta = 0,1$, penalização de falha crítica $P_{\text{penalty}} = 2,0$, $\gamma = 1,0$;
  - Quantil adaptativo $\tau_0 = 0,05$, faixa $[0,01,\, 0,05]$, passo $\Delta\tau = 0,005$ em bloco $W = 100$ ($\delta = 0,01$, recupera após $2$ blocos limpos); margem de incerteza $M_0 = 10,0$, limites $[2,0,\, 50,0]$; exploração $\varepsilon = 0,1$, $T = 1,0$, `knn_k = 10`;
  - *Warmup:* $100,0\text{ s}$ ($1000$ janelas) sob `PropDemandController`; capacidade CLI $52 \to$ MAC `C = 53` RBGs/slot DL;
  - Escalonador de estado (`feature_scaler_manifest.yaml`, `n_cal = 23.720`): `n_e/u/m` `[0,4]`, `[0,3]`, `[0,2]`; `queue_e/u/m` `[0, 9.844.456,46]`, `[0, 709]`, `[0, 1422]`; `offered_e/u/m` `[0, 16,5024]`, `[0, 1,4256]`, `[0, 0,0182]`; correção `A_scaler` corrige indexação `'eum'`; `R12` desativado (`virtual_alpha = 0.0`); `A2a` zera slice vazia pré-decisão; `mean_field = False`.

### 1.2 Auditoria das 152 Colunas de `GATE_CONTEXTUAL_ESTADOS.csv`
A auditoria completa classifica as `152` colunas em:
- **`META_E_INTEGRIDADE` (`12` colunas):** chaves de estado, estrato e `aa_parity_b0_eq_prop` (`0` NaNs).
- **`BLOCO_A_PRE_DECISAO` (`99` colunas):** demanda/fila/SNR (`13`), ações e predições `B0` pré-decisão (`38`), histórico causal anterior (`13`), integridade (`1`) e observabilidade/backlog URLLC (`34`). Ponto crítico: **`prev_kpi_u_p95_imm` é 100% `NaN` (`25/25` linhas)** devido ao atraso de fechamento da Emenda 4 (`prev_kpi_e_mbps` tem `5` NaNs por janelas anteriores sem demanda eMBB; `prev_kpi_m_pdr` tem `3` NaNs).
- **`BLOCO_B_POS_REPLAY_VEDADO` (`41` colunas):** desfechos *post-replay* estritamente vedados para qualquer decisão causal.
- **Lacunas no Bloco A reconstruídas causally em `t1_urllc_queue_causal_extra.csv`:** a CSV original informa apenas o booleano `urllc_pool_open_at_dec` e o *lag* da última janela fechada, mas omite quantos *pools* estavam simultaneamente abertos ou expirados no instante da decisão. A reconstrução causal sobre os `25` estados do `Gate 1` mostrou `urllc_pool_open_at_dec = True` em `21/25` estados (com **`2` *pools* abertos** no momento da decisão) e `False` em `4/25`, *pools* acumulados expirados por `R19` entre **`2` e `4`**, e `last_closed_lag` com mediana de `2` janelas e máximo de **`631` janelas** (`63,1 s` sem fechamento em `lam5_s2102_w2552`).

### 1.3 Auditoria C++/IPC de Telemetria Sub-100 ms (`MODA/ns-3-dev/scratch/sqr/`)
| Sinal Sub-100 ms | Estado no Código Atual | Evidência Exata |
|---|---|---|
| **RLC Head-of-Line (HoL) delay** | **AUSENTE** | Nenhum campo HoL nem consulta ao timestamp do pacote *head* em `NrRlcUm`; `sqr-sim.cc:1460-1495` lê apenas `GetUeDlBufferSize` (bytes totais). |
| **Derivada intra-janela de fila RLC** | **AUSENTE NO IPC** | `harq_tracking.csv` e `ue_detail.csv` gravam `dlBufferSize` por slot localmente em disco, mas o KPM IPC transmite apenas `queueBytes` na borda de $100\text{ ms}$. |
| **Contadores HARQ (NACKs / processos ativos)** | **`NA` / INACESSÍVEL NA SUBCLASSE** | `LogHarqState` (`rslaq-mac-scheduler.cc:938-962`) escreve literalmente `'NA,NA,NA'` porque `activeHarqCount`/`harqCapacity`/`canInsert` são privados em `NrMacSchedulerNs3`; nada vai ao IPC. |
| **Atraso por pacote (`sqr_packet_delays.csv`)** | **PÓS-ENTREGA DL APENAS** | `WritePacketDelay` (`sqr-sim.cc:478-506`) é acionado somente na recepção RX; pacotes presos em fila ou perdidos antes da entrega não aparecem pré-decisão. |

---

## 2. Tarefa 2 — Auditoria `BUG-03` em `D1-v9` e `D2-v2` (Reconstrução Causal Sem `.pkl`)

Tabelas geradas: [`t2_d1v9_runs_ipc_lag_audit.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/t2_d1v9_runs_ipc_lag_audit.csv) e [`t2_d2v2_30_states_desync_status.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/t2_d2v2_30_states_desync_status.csv).

### 2.1 Auditoria dos 9 Runs `d1_runs_v9_delayed/`
Verificando diretamente o descasamento `prb_executed_rbg != decision["action"]` a partir de `outcome_window >= 1` (`decisions.jsonl` × `sqr_prb_windows.csv`):

| Run | $\lambda$ | Seed | Decisões | Descasamentos (`ow >= 1`) | `onset_outcome_window` | Eventos `> 50 ms` | `max_decision_time_ms` | Status |
|---|---|---|---|---|---|---|---|---|
| `het_lam2_s2101` | 2 | 2101 | 2627 | **0** | — | 0 | 26,8744 | **LIMPO** |
| `het_lam2_s2102` | 2 | 2102 | 848 | **0** | — | 0 | 5,7004 | **LIMPO** |
| `het_lam2_s2103` | 2 | 2103 | 1703 | **0** | — | 0 | 16,1769 | **LIMPO** |
| `het_lam5_s2101` | 5 | 2101 | 2923 | 191 | **2734** | 0 | 33,6549 | Desync `ow>=2734` |
| `het_lam5_s2102` | 5 | 2102 | 2298 | 113 | **2844** | 0 | 36,7680 | Desync `ow>=2844` |
| `het_lam5_s2103` | 5 | 2103 | 2923 | 584 | **2313** | 2 (`step 2737`) | 51,6062 | Desync `ow>=2313` |
| `het_lam6_s2101` | 6 | 2101 | 2935 | 298 | **2654** | 0 | 34,9342 | Desync `ow>=2654` |
| `het_lam6_s2102` | 6 | 2102 | 2455 | 455 | **2500** | 0 | 35,4972 | Desync `ow>=2500` |
| `het_lam6_s2103` | 6 | 2103 | 2974 | 849 | **1902** | 0 | 43,2449 | Desync `ow>=1902` |

**Achado estrutural relevante:** em `5/6` runs dessincronizados (`het_lam5_s2101/2102` e `het_lam6_s2101/2102/2103`), o descasamento permanente iniciou **antes mesmo de qualquer passo ultrapassar `50 ms` no cronômetro Python** (máximos entre `33,65 ms` e `43,24 ms`), confirmando deslocamento de fase / *timeout* de borda no contrato IPC `v1` não-versionado (`BUG-03`).

### 2.2 Classificação dos 30 Estados `D2-v2` e Validação da Reconstrução Causal
- **`PRE_DESYNC`: `23/30` estados** (`t_win < onset_outcome_window` ou run limpo).
- **`POST_DESYNC`: `7/30` estados** (`S1_lam5_s2103_t2886`, `S1_lam6_s2103_t2404`, `S1_lam6_s2103_t2909`, `S2_lam5_s2103_t2743`, `S2_lam6_s2101_t2824`, `S2_lam6_s2103_t1961`, `S4_lam6_s2103_t2004`).
- **Dos `4` estados `RECOVERABLE_STRICT`:**
  - `S3_lam6_s2102_t1420` (`t=1420 < 2500`) $\to$ **`PRE_DESYNC`** (válido para diagnóstico causal).
  - `S1_lam6_s2102_t1608` (`t=1608 < 2500`) $\to$ **`PRE_DESYNC`** (válido para diagnóstico causal).
  - `S2_lam6_s2101_t2824` (`t=2824 >= 2654`) $\to$ **`POST_DESYNC`** (descartado da análise causal).
  - `S1_lam5_s2103_t2886` (`t=2886 >= 2313`) $\to$ **`POST_DESYNC`** (descartado da análise causal).
- **Reconstrução causal *forward* zero-checkpoint:** nos `23/23` estados `PRE_DESYNC`, a reprodução sequencial de `A0` a partir do zero casou bit-a-bit com as predições logadas online antes e depois da projeção (**erro máximo absoluto = `0.000e+00`** em ambos).

---

## 3. Tarefa 3 — Curvas Quantílicas ($a \in [0..53]$), Monotonicidade, Achatamento e Rearranjo

Tabelas geradas: [`t3_curvas_quantilicas_por_estado_slice.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/t3_curvas_quantilicas_por_estado_slice.csv) (`213` curvas = `(25 + 23 + 23) × 3`) e [`t3_resumo_monotonicidade_achatamento_por_slice.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/t3_resumo_monotonicidade_achatamento_por_slice.csv).

| Conjunto | Slice | $\bar{f}_{\min}$ | $\bar{f}_{\max}$ | $\text{med}(\Delta f_{\text{tot}})$ | Inversão Geral | Inversão $\sigma \le 0,15$ | Inversão $0,15 < \sigma \le 0,55$ | Inversão $\sigma > 0,55$ | $\text{Sort } L_\infty$ ($\sigma \le 0,15$) | $\text{PAVA } L_\infty$ ($\sigma \le 0,15$) | $\|\Delta_2 f\| < \gamma\frac{2}{53}$ ($\sigma \le 0,15$) | $\|\Delta_8 f\| < \gamma\frac{8}{53}$ ($\sigma \le 0,15$) | $\text{p50 } \Delta_8 f$ ($\sigma \le 0,15$) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **`Gate 1 B0-rest`** | **eMBB** | -0,0043 | 0,4528 | 0,3973 | 7,55% | **6,60%** (1060) | 12,55% (239) | 0,00% (26) | 0,0191 | 0,0135 | 78,59% | 78,31% | 0,0653 |
| **`Gate 1 B0-rest`** | **URLLC** | -0,4279 | **-0,2452** | 0,1503 | 13,06% | **14,72%** (1162) | 1,23% (163) | — (0) | 0,0127 | 0,0092 | **100,00%** | **100,00%** | 0,0251 |
| **`Gate 1 B0-rest`** | **mMTC** | -0,0619 | 0,0982 | 0,1513 | 19,02% | **11,46%** (1073) | **85,83%** (120) | 19,70% (132) | 0,0024 | 0,0016 | **100,00%** | **100,00%** | 0,0249 |
| **`D2-v2 A0 exato`** | eMBB | -1,6536 | -1,2072 | 0,4554 | 45,61% | **44,30%** (1158) | 69,49% (59) | 100% (2) | 0,2566 | 0,1860 | 84,07% | 85,66% | 0,0146 |
| **`D2-v2 A0 exato`** | URLLC | -2,7410 | -2,1999 | 0,4794 | 49,14% | **48,91%** (1143) | 52,63% (76) | — (0) | 0,4622 | 0,3345 | 63,69% | 67,85% | 0,0022 |
| **`D2-v2 A0 exato`** | mMTC | -1,0994 | -0,6045 | 0,2552 | 69,07% | **76,65%** (1088) | 10,26% (78) | 0,00% (53) | 0,5114 | 0,3249 | 80,21% | 78,80% | -0,0515 |
| **`D2-v2 Shadow B0-rest`** | eMBB | -0,3667 | 0,3462 | 0,5578 | 8,78% | **5,79%** (1105) | 41,00% (100) | 14,29% (14) | 0,0133 | 0,0094 | 75,28% | 74,11% | 0,0935 |
| **`D2-v2 Shadow B0-rest`** | URLLC | -0,3485 | -0,2155 | 0,0479 | 9,60% | **14,23%** (548) | 13,00% (300) | 0,00% (371) | 0,0095 | 0,0065 | 95,87% | 95,71% | 0,0097 |
| **`D2-v2 Shadow B0-rest`** | mMTC | -0,0842 | -0,0303 | 0,0309 | 19,77% | **28,52%** (817) | 6,67% (90) | 0,64% (312) | 0,0086 | 0,0050 | **100,00%** | **100,00%** | 0,0061 |

**Interpretação física e algorítmica:**
1. **Efeito de desativar `R12` + `A_scaler`:** em `D2-v2`, a mera remoção de `virtual_alpha = 0,05` + correção do escalonador derruba as inversões no suporte ($\sigma \le 0,15$) de **`44,3%–76,7%`** (`A0`) para **`5,8%–28,5%`** (`Shadow B0-rest`), alinhando-se aos **`6,6%–14,7%`** do `Gate 1` e removendo o viés artificial negativo ($\bar{f}_{\max}$ de `-1,21..-2,20` para `+0,35..-0,22`).
2. **Achatamento espacial causado por `length_scale = 1.0`:** mesmo sem `R12`, o kernel Matérn isotrópico $\ell = 1,0$ em $[0,1]^7$ achata a inclinação $\Delta f / \Delta a$: em **URLLC** e **mMTC**, **100%** dos passos $\Delta \in \{2,4,8\}$ no suporte têm $|\Delta f| < \gamma \cdot \Delta/53$ no `Gate 1` (mediana de variação em $+8$ PRBs é apenas $+0,025$ vs custo de seleção $+0,151$). Em **eMBB**, isso ainda ocorre em `78,3%–78,6%` dos passos no suporte.
3. **Custo de rearranjo monotônico (`sort` vs `PAVA`):** sobre curvas `B0-rest`, o rearranjo `PAVA` é pouco intrusivo ($L_\infty \le 0,0135$ em eMBB, $\le 0,0092$ em URLLC, $\le 0,0050$ em mMTC no suporte; $L_1 \le 0,0024$) e sempre menos distorcivo que `np.sort` (`sort` apenas permuta pontos e pode inverter picos/vales locais quando a curva sofre desordem estrutural, como em `A0`).
4. **Saturação negativa da URLLC:** em `Gate 1 B0-rest`, apenas **`3/25` (`12,0%`)** dos estados atingem $f_{\max} \ge 0$ em toda a faixa $a \in [0..53]$ para URLLC (`1/23 = 4,3%` em `A0`; `9/23 = 39,1%` em `Shadow B0-rest`), contrastando com `19/25` (`76,0%`) em eMBB e `24/25` (`96,0%`) em mMTC.

---

## 4. Tarefa 4 — Confronto contra Desfechos Reais Separado em Regimes A e B

Tabelas geradas: [`t4_regime_a_ociosidade_estados.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/t4_regime_a_ociosidade_estados.csv), [`t4_regime_b_pares_replays_por_estado_slice.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/t4_regime_b_pares_replays_por_estado_slice.csv), [`t4_regime_b_resumo_sign_spearman.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/t4_regime_b_resumo_sign_spearman.csv), [`t4_regime_b_falso_doador_pares_brutos.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/t4_regime_b_falso_doador_pares_brutos.csv) e [`t4_regime_b_falso_doador_seguro.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/t4_regime_b_falso_doador_seguro.csv).

### 4.1 Regime A — Ociosidade, Alocação em Slice Vazia e Chegadas Intra-Janela

| Conjunto | Estados | PRBs Não-Alocados (`>0`) | PRBs em Slice Vazia (`>0`) | Chegadas Intra-Janela em Slice Vazia | Ordenação Ganho Total $\Delta f_i$ vs $\Delta y_i$ (`raw` / `sort` / `pava`) | Ordenação Inclinação Marginal $\frac{\Delta f_i}{\Delta a_i}$ (`raw` / `sort` / `pava`) |
|---|---|---|---|---|---|---|
| **`Gate 1` (`B0-rest`)** | 25 | 13 (`52%`) | **0 (`0%`)** | eMBB: `1` (`w1254`), URLLC: `1` (`w2552`), mMTC: `0` | `3/7` = `42,9%` / `42,9%` / `42,9%` | **`6/7` = `85,7%`** / `85,7%` / `85,7%` |
| **`D2-v2 A0 exato`** | 23 | 8 (`35%`) | **15 (`65%`)** | `0 / 0 / 0` | `16/20` (`80%`) / `11/20` (`55%`) / `17/20` (`85%`) | `15/20` (`75%`) / `7/20` (`35%`) / `16/20` (`80%`) |
| **`D2-v2 Shadow B0-rest`** | 23 | 8 (`35%`) | 15 (`65%`) | `0 / 0 / 0` | **`17/20` (`85%`)** / `17/20` (`85%`) / `17/20` (`85%`) | **`16/20` (`80%`)** / `16/20` (`80%`) / `16/20` (`80%`) |

**Achados do Regime A:**
1. **`A2a` (`empty_slice_guard`) zerou o desperdício trivial em slices vazias:** `0/25` no `Gate 1` contra `15/23` no `D2-v2 A0`. As chegadas intra-janela em slices inicialmente vazias ocorreram em apenas `2/25` estados do `Gate 1` (`lam6_s2103_w1254` em eMBB e `lam5_s2102_w2552` em URLLC; nenhuma em `D2-v2`).
2. **Sinal útil de alocação marginal quando há folga ociosa:** após deduplicar braços idênticos em ação (`7` pares no `Gate 1`, em `lam5_s2102_w1654` e `lam6_s2102_w1396`), a inclinação marginal $\Delta f_i / \Delta a_i$ ordena corretamente qual slice ativa obtém maior ganho marginal real por PRB em **`6/7` (`85,7%`)** dos pares no `Gate 1` e **`16/20` (`80,0%`)** no `D2-v2 Shadow` (enquanto `np.sort` sobre a curva corrompida do `A0` cai para `35,0%`).

### 4.2 Regime B — Transferência Inter-Slice, Empates (`dy = 0`) e Segurança da Doadora

#### (a) Ordenação Direcional ($\Delta f$ vs $\Delta y$ em pares de replay $a_2 > a_1$)

| Conjunto | Slice | Pares (`dy != 0`) | Empates `dy=0` | `Sign Acc` estrita (`bruto` / `sort` / `PAVA`) | Spearman `pooled` (`bruto` / `sort` / `PAVA`, `dy!=0`) | Média Spearman **intra-estado** (`n_est`, `bruto` / `sort` / `PAVA`) |
|---|---|---|---|---|---|---|
| **`Gate 1 B0-rest`** | **eMBB** | 209 (209) | **0,0%** | **100,0%** / 100,0% / 100,0% | **+0,8055** / +0,8055 / +0,8055 | **+0,9816** (`20`) / +0,9816 / +0,9816 |
| **`Gate 1 B0-rest`** | **URLLC** | 186 (57) | **69,4%** | **59,6%** / 66,7% / 50,9% | +0,0747 (`p=0,58`) / +0,0582 / +0,0867 | **+0,1947** (`7`) / +0,2130 / +0,1873 |
| **`Gate 1 B0-rest`** | **mMTC** | 227 (34) | **85,0%** | **94,1%** / 94,1% / 76,5% | -0,6918 / -0,6918 / -0,6935 | -0,4140 (`1`) / -0,4140 / -0,4140 |
| `D2-v2 A0 exato` | eMBB | 1596 (1511) | 5,3% | 39,9% / 98,4% / 33,0% | +0,1506 / +0,6028 / +0,3848 | -0,1635 (`21`) / +0,9351 / +0,8770 |
| `D2-v2 A0 exato` | URLLC | 510 (260) | 49,0% | 47,7% / 62,3% / 20,0% | -0,2411 / -0,0695 / -0,0505 | -0,0840 (`4`) / +0,0148 / -0,1206 |
| `D2-v2 Shadow B0-rest` | **eMBB** | 1596 (1511) | **5,3%** | **96,7%** (`98,3%` sup.) / 98,4% / 93,5% | **+0,6964** (`+0,7757` sup.) / +0,6991 / +0,6970 | **+0,9092** (`21`) / +0,9673 / +0,9662 |
| `D2-v2 Shadow B0-rest` | **URLLC** | 510 (260) | **49,0%** | **49,2%** / 62,3% / 50,8% | -0,2690 (`p=1.1e-5`) / -0,0245 / -0,1677 | **-0,0854** (`4`) / +0,0461 / +0,0022 |
| `D2-v2 Shadow B0-rest` | **mMTC** | 1134 (46) | **95,9%** | **100,0%** / 100,0% / 100,0% | +0,4782 / +0,4932 / +0,4932 | — (`0`) |

- **Em eMBB (`Gate 1` e `D2-v2 Shadow`), as curvas quantílicas ordenam o ganho com alta fidelidade:** `100%` de acerto direcional no `Gate 1` e **`+0,9816` de Spearman médio intra-estado** (`+0,9092` bruto / `+0,9662` PAVA no `Shadow`).
- **Em URLLC e mMTC, a resposta real ao número de PRBs é degenerada por platôs/degraus:** `69,4%` de empates `dy = 0` em URLLC (`85,0%` em mMTC) no `Gate 1`; dentro dos poucos estados URLLC com `dy != 0`, a correlação média intra-estado fica em `+0,1947` (`Gate 1`, `7` estados) e `-0,0854` (`D2-v2 Shadow`, `4` estados).

#### (b) Segurança da Doadora (`False Safe Donor Rate` vs `Safe Donation Recall`)

| Conjunto | Slice | Pares Base Segura (`true_safe / true_harm`) | Regra Quantil Bruto $f_\tau \ge 0$ (`Auth` / `FDR` / `Recall`) | Regra $f_\tau - \sigma \ge 0$ (`Auth` / `FDR` / `Recall`) | Regra $f_\tau \ge 0 \land \sigma \le 0,15$ (`Auth` / `FDR` / `Recall`) | **Envelope `prop_active`** (`Auth` / `FDR` / `Recall`) | **Envelope Físico Fila+SNR** (`Auth` / `FDR` / `Recall`) |
|---|---|---|---|---|---|---|---|
| **`Gate 1 B0-rest`** | **eMBB** | 168 (`72 / 96`) | 94 / **55,3%** / 58,3% | 88 / **59,1%** / 50,0% | 94 / **55,3%** / 58,3% | 70 / **27,1%** / **70,8%** | 4 / 100,0% / 0,0% |
| **`Gate 1 B0-rest`** | **URLLC** | 161 (`130 / 31`) | 1 / 0,0% / **0,77%** | 0 / — / **0,0%** | 1 / 0,0% / **0,77%** | 94 / **11,7%** / **63,8%** | 161 / **19,3%** / **100,0%** |
| **`Gate 1 B0-rest`** | **mMTC** | 226 (`198 / 28`) | 73 / 19,2% / 29,8% | 72 / 19,4% / 29,3% | 73 / 19,2% / 29,8% | 11 / 0,0% / 5,6% | 126 / **2,4%** / **62,1%** |
| `D2-v2 Shadow` | eMBB | 908 (`422 / 486`) | 486 / **35,0%** / 74,9% | 446 / **29,8%** / 74,2% | 453 / **30,9%** / 74,2% | 153 / **3,9%** / 34,8% | 0 / — / 0,0% |
| `D2-v2 Shadow` | URLLC | 323 (`259 / 64`) | 13 / **53,8%** / **2,3%** | 13 / **53,8%** / **2,3%** | 13 / **53,8%** / **2,3%** | 136 / **5,1%** / **49,8%** | 286 / **9,4%** / **100,0%** |
| `D2-v2 Shadow` | mMTC | 1134 (`1100 / 34`) | 326 / 0,0% / 29,6% | 326 / 0,0% / 29,6% | 326 / 0,0% / 29,6% | 118 / 0,0% / 10,7% | 844 / **0,0%** / **76,7%** |

**Diagnóstico crítico da segurança da doadora:**
1. **Paralisia e falso-seguro na URLLC via quantil:** no `Gate 1 B0-rest`, como `f_tau` URLLC permanece negativo em `88%` dos estados, usar $f_\tau \ge 0$ autoriza **apenas `1` de `130` doações seguras (`recall = 0,77%`)**, bloqueando transferências legítimas de platô; no `D2-v2 Shadow`, as poucas `13` autorizações erram `7` vezes (`FDR = 53,8%`). Exemplo direto: em `S1_lam6_s2102_t1608` (`PRE_DESYNC`), URLLC segura `8,10 ms` de `a_u = 37` até `a_u = 8` (só colapsando para `98,15 ms` em `a_u = 0`), e doar até `a_u = 16` (`alt9: 33-16-4`) salva a eMBB (`2,08 -> 4,65 Mbps`, `STRICT_PASS`); o quantil `B0-rest` prevê `-0,177` (veta), enquanto `prop_active_floor = 7` e `phys_snr_floor = 1` autorizam `16`.
2. **Platôs com acoplamento inter-slice / quantização em `t1420` (`S3_lam6_s2102_t1420`, `PRE_DESYNC`):** nos replays de `t1420`, `alt0 (22-30-1)`, `alt7 (23-30-0)`, `alt5 (22-29-2)` e `alt1 (22-28-3)` falham por pouco (`10,15 ms`), ao passo que `alt3 (14-30-9)` com **exatos mesmos `a_u = 30`**, `alt2 (26-26-1)`, `alt12 (18-17-18)`, `alt10 (31-2-5)` e `alt11 (32-4-0)` passam com **`8,10 ms`**. Isso mostra que na fronteira de $10\text{ ms}$, pequenas alterações no escalonamento OFDMA conjunto ou no alinhamento de slots/HARQ deslocam a amostra discreta de atraso entre `8,10 ms` e `10,15 ms` independentemente de variações suaves na cota `a_u`.
3. **Efeito complementar dos envelopes causais:** na **URLLC**, o envelope simples `prop_active` corta o `FDR` para **`11,7%`** (`Gate 1`) / **`5,1%`** (`D2-v2`) com recall de **`63,8%`** / **`49,8%`**, enquanto em **eMBB** supera os quantis puros em precisão (`FDR 27,1%` vs `55,3%` no `Gate 1`), porque o quantil `B0-rest` sofre **sobrepredição agressiva em eMBB** (`55,3%` de falso-seguro quando autoriza doação). Por outro lado, o envelope físico simplificado de Shannon+fila superestima capacidade em eMBB congestionado (`recall 0%`), mostrando necessidade de casamento específico por slice.

---

## 5. Tarefa 5 — Índice de Confiabilidade Prequencial Causal vs. $\sigma_{\text{kernel}}$ (`b0_traced_gate1`)

Tabelas geradas: [`t5_prequencial_vs_sigma_por_slice_run.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/t5_prequencial_vs_sigma_por_slice_run.csv) e [`t5_prequencial_resumo_auroc_spearman.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/t5_prequencial_resumo_auroc_spearman.csv).
*(Abaixo: $W = 20$ no escopo `pre_traced_lag` e entre parênteses `pre_clean_cut_strict`; resultados de $W = 50$ seguem o mesmo padrão e constam integralmente na CSV).*

| Slice | Amostras (`cl / dem`) | Violações | Preditor | Spearman $\|y - f\|$ | **AUROC Violação SLA** (`closed`) | **AUROC Viol/Unclosed** |
|---|---|---|---|---|---|---|
| **eMBB** | 3979 (3137) | 468 (457) | `sigma_kernel` | -0,0845 (-0,0661) | **0,7595** (0,7455) | **0,7595** (0,7455) |
| **eMBB** | 3979 (3137) | 468 (457) | `pinball_rolling` | **+0,3293** (+0,2541) | **0,2394** (0,2677) | 0,2394 (0,2677) |
| **eMBB** | 3979 (3137) | 468 (457) | `overpred_rate_rolling` | -0,1928 (-0,1359) | **0,6339** (0,6221) | 0,6339 (0,6221) |
| **eMBB** | 3979 (3137) | 468 (457) | `viol_hist_rate_rolling` | -0,2146 (-0,1810) | **0,8737** (0,8594) | **0,8737** (0,8594) |
| **eMBB** | 3979 (3137) | 468 (457) | `U_preq_assimetrico` | -0,2595 (-0,2064) | **0,8696** (0,8544) | **0,8696** (0,8544) |
| **URLLC** | 3370 / 3378 (2530 / 2536) | 781 / 789 (492 / 498) | `sigma_kernel` | -0,0080 (-0,0131) | **0,4668** (0,4716) | **0,4667** (0,4714) |
| **URLLC** | 3370 / 3378 | 781 / 789 | `pinball_rolling` | -0,0257 (-0,1285) | 0,5326 (0,5363) | 0,5320 (0,5354) |
| **URLLC** | 3370 / 3378 | 781 / 789 | `overpred_rate_rolling` | -0,0989 (-0,1419) | 0,5210 (0,5183) | 0,5206 (0,5182) |
| **URLLC** | 3370 / 3378 | 781 / 789 | `viol_hist_rate_rolling` | **+0,2956** (+0,2129) [`+0,3600` $W=50$] | **0,7144** (0,7103) | **0,7125** (0,7091) |
| **URLLC** | 3370 / 3378 | 781 / 789 | `U_preq_assimetrico` | **+0,2681** (+0,1767) [`+0,3303` $W=50$] | **0,7113** (0,7069) | **0,7094** (0,7059) |
| **URLLC** | 3370 / 3378 | 781 / 789 | `urllc_censorship_frac` | -0,0669 (-0,0863) | 0,4920 (0,4937) | 0,4918 (0,4934) |
| **mMTC** | 5157 (4315) | 80 (68) | `sigma_kernel` | +0,0885 (+0,1148) | 0,4316 (0,4094) | 0,4316 (0,4094) |
| **mMTC** | 5157 (4315) | 80 (68) | `pinball_rolling` | **+0,2975** (+0,2839) | 0,3693 (0,3809) | 0,3693 (0,3809) |
| **mMTC** | 5157 (4315) | 80 (68) | `viol_hist_rate_rolling` | +0,2652 (+0,2558) | 0,3561 (0,3486) | 0,3561 (0,3486) |

**Conclusões empíricas da Tarefa 5:**
1. **`sigma_kernel` funciona em eMBB (`AUROC = 0,760`) mas é cego em URLLC (`AUROC = 0,467`, pior que aleatório):** na URLLC, os vetores de entrada $z_u$ concentram-se em uma região densa de baixa incerteza geométrica ($\bar{\sigma} = 0,0079$ em violações vs $0,0092$ em janelas seguras no recorte limpo `W=20`), apesar da taxa de violação real ser **`19,4%`**. Logo, $\sigma_{\text{kernel}} \le 0,15$ **não garante segurança preditiva na URLLC**.
2. **Por que a `pinball` simétrica inverte o risco em eMBB (`AUROC = 0,239`) enquanto `U_preq_assimetrico` / `viol_hist_rate_rolling` alcançam `0,870–0,874` em eMBB e `0,711–0,714` em URLLC:** em episódios eMBB folgados, a margem real sobe até $p_{95} = +2,248$ ($11,4\text{ Mbps}$) enquanto $f_\tau$ ($\tau=0,05$, mediana `-0,066`) gera grandes desvios positivos benignos (`y >> f`), elevando `pinball_rolling` médio em janelas seguras (`0,0340`) acima das violações (`0,0191`). Ao substituir o desvio bidirecional pelo **risco unilateral/histórico causal** (`viol_hist_rate_rolling` e `U_preq_assimetrico`), o índice supera `sigma_kernel` tanto em **eMBB** (`AUROC 0,870 vs 0,760`) quanto em **URLLC** (`AUROC 0,711 vs 0,467`, com ganho de `+24,5` pontos percentuais de AUROC).
3. **Limitação na mMTC:** violações mMTC em `b0_traced_gate1` são esparsas (`1,55%–1,58%`, `95%` das janelas com $\text{PDR} = 1,0 \implies y = +0,0526$), e todos os preditores baseados em janela de `20–50` passos ficam em `AUROC <= 0,45` por reversão rápida após picos pontuais.

---

## 6. Síntese Crítica — Sinal já Disponível na `v1` vs. Dependências de `v2` (Etapa 4 / C++)

| Aspecto / Premissa | O que já funciona nos dados limpos da `v1` | O que falha hoje e do que depende em `v2` |
|---|---|---|
| **1. Controle de slices vazias e remoção de viés (`A2a`, `R12=0`, `A_scaler`)** | `A2a` zera alocação em slice vazia (`0/25` no `Gate 1`); `R12=0 + A_scaler` derruba inversões em eMBB de `44,3%` para `5,8%–6,6%`. | Revalidação física na **Etapa 3/4** com IPC determinístico (`BUG-03` corrigido) e `B0-rest` íntegro. |
| **2. Ordenação de sensibilidade em eMBB** | Sinal muito forte em `Gate 1` (`100%` acerto de sinal, Spearman intra-estado `+0,9816`) e `D2-v2 Shadow` (`96,7%`, intra `+0,9092`); PAVA pouco intrusivo (`L_inf = 0,0135`). | Calibração de escala `l_a` (**Etapa 4**) para corrigir sobrepredição nas doações (`FDR = 55,3%` hoje quando $f_\tau \ge 0$) e reduzir achatamento (`78,3%`). |
| **3. Absorção de capacidade ociosa (Regime A)** | Inclinação marginal $\Delta f / \Delta a$ ordena `85,7%` (`6/7`) no `Gate 1` e `80,0%` (`16/20`) no `D2-v2 Shadow` onde há múltiplos receptores ativos. | Baixa frequência de pares multi-slice não-nulos na amostra offline (`2/25` estados com ociosidade ativa no `Gate 1`); precisa confirmação em **Etapa 5 (`E5-Idle`)**. |
| **4. Identificação de doadora segura em URLLC** | Envelope simples `prop_active` detecta platôs seguros (`recall 63,8%`, `FDR 11,7%` `Gate 1`; `49,8%`, `FDR 5,1%` `D2-v2`). | Quantis `B0-rest` sofrem **paralisia** (`recall 0,77%` `Gate 1`) + achatamento total (`100% < γ·Δ/53`) + `69,4%` empates `dy=0` + viés de censura da Emenda 4 + ausência de **HoL/derivada RLC/HARQ no IPC** (exige **Etapa 4** e eventual telemetria C++ sub-100 ms). |
| **5. Detecção de confiabilidade pré-decisão** | Índice prequencial unilateral (`viol_hist_rate_rolling` / `U_preq_assimetrico`) já alcança `AUROC 0,870–0,874` (eMBB) e **`0,711–0,714` (URLLC)** sobre dados estritamente causais. | `sigma_kernel` falha na URLLC (`AUROC 0,467`) e `pinball` simétrica inverte em eMBB (`AUROC 0,239`). Em mMTC (`1,6%` violações), memória de `20–50` passos reverte tarde (`AUROC < 0,45`). |
