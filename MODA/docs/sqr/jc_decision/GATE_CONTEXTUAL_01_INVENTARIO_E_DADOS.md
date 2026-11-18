# GATE CONTEXTUAL — E1: Inventário, Reconciliação de Populações e Integridade Causal (Gate G0)

**Data-base do Ofício:** 16/11/2026 · **Conclusão da auditoria G0:** 17/11/2026 · **Horizonte de decisão:** 19/11/2026  
**Documento autorizador:** [`SQR_JC_OFICIO_AGENTE_GATE_CONTEXTUAL_2026-11-16.md`](file:///home/eliothluy/Downloads/Telegram%20Desktop/SQR_JC_OFICIO_AGENTE_GATE_CONTEXTUAL_2026-11-16.md) (`sha256: 55aff720098ff796ce19cd39ac65b8e109e5fde3410f5d1b9b5e9dfe04158b1b`, `24.279` bytes)  
**Git HEAD preservado (sem alterações):** `3afe72cc80092039aca5cd22065db4812e9ee3cc` (`3afe72c`)

---

## 1. Resumo Executivo e Veredito G0

**Parecer G0:** **`GO-DADOS` (com limitação severa de amostra efetiva: `N = 18` em `ALL3_COMPLETE`, provenientes de apenas `2` seeds de desenvolvimento `2102` e `2103` em $\lambda \in \{5, 6\}$).**

Os artefatos do Gate 1 da A5, da referência congelada `B0-rest` e dos logs causais (`causal_trace.jsonl.gz`, `decisions.jsonl`, `urllc_delayed_updates.jsonl` e `step_metrics.csv`) permitem construir, **sem novas simulações** e **sem vazamento temporal** (`observed_at <= decision_at`), a tabela única de estados [`GATE_CONTEXTUAL_ESTADOS.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/GATE_CONTEXTUAL_ESTADOS.csv) (`25` estados × `3` braços avaliados `B0-rest` / `equal` / `prop`, `152` colunas). Todas as verificações estruturais passaram (`0` inconsistências nos `17` hashes congelados de `B0-rest`, `25/25` paridades `A/A` preservadas, `0` violações temporais no Bloco A).

Entretanto, duas ressalvas obrigatórias restringem o uso desses dados:
1. O arquivo `/home/eliothluy/Downloads/Telegram Desktop/SQR_DIRECIONAMENTO_CIENTIFICO_E_REVISAO_LITERATURA_2026-11-16.md` citado no §0.3 item 8 do Ofício **não existe no sistema de arquivos** (busca recursiva em `/home/eliothluy/Downloads` e `/home/eliothluy/Documentos` retornou `0` ocorrências). A lacuna de mapeamento bibliográfico prévio é suprida apenas parcialmente pela tabela resumida no §4 do próprio Ofício e por verificação web direta em G2 ([`GATE_CONTEXTUAL_03_ORIGINALIDADE_E_ESCOLHA.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/GATE_CONTEXTUAL_03_ORIGINALIDADE_E_ESCOLHA.md)).
2. A amostra de `25` estados é **estratificada pós-congelamento exploratório em 2 seeds** (`2102` e `2103`), com `N = 18` no desfecho primário `STRICT_ALL3` (`0` estados em $\lambda = 2$), insuficiente para calibrar limiares ou declarar generalização.

---

## 2. Inventário Completo dos Artefatos (§0.3)

| Item (§0.3) | Artefato | Caminho real | Tamanho (bytes) | SHA-256 | Status | Escopo / Uso no Gate |
|---|---|---|---:|---|---|---|
| 1 | Relatório exploratório Gate 1 | [`A5_GATE1_RELATORIO_EXPLORATORIO.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/A5_GATE1_RELATORIO_EXPLORATORIO.md) | 20.069 | `18193a5ac2c0d4cdb373d881bf4fb83ddce456947dc8022b3db56a7c9734fe37` | OK | Síntese canônica de `127` replays / `25` estados e paridade `B0-rest` |
| 1b | Diagnóstico retrospectivo Gate 1 | [`A5_GATE1_DIAGNOSTICO_RETROSPECTIVO.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/A5_GATE1_DIAGNOSTICO_RETROSPECTIVO.md) | 26.262 | `139196c624b445f72450f482d22853e43ca9b5f0792718d7bd5ac0be8b49c0a9` | OK | Separação URLLC max-min vs $-\beta\sigma$ e teto oracular `O_3` (`new_replays=0`) |
| 2 | Tabela métricas por braço | [`A5_GATE1_METRICS.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_METRICS.csv) | 47.825 | `895a94fccfcda9bc086b3416f30376665458ed867cd34f280fe96e2ce78092dc` | OK | `150` linhas (`25` estados × `6` braços): fonte canônica do Bloco B |
| 3 | Decisões e trace JSON | [`A5_GATE1_DECISIONS.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_DECISIONS.json) | 180.553 | `03a32e97351321f5e2e6d1b3cad49f22297545af051cc7bf86fea985511df4f1` | OK | `n_states=25`, `n_unique_replays=127`, `n_logical_arm_evals=150` |
| 3b | Diagnósticos agregados | [`A5_GATE1_DIAGNOSTICS.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_DIAGNOSTICS.json) | 58.448 | `e9f8a70422772751e2d1b9f4bd3ea843fa330a7416c8d5aab593d42f6c5e9b31` | OK | Universo A/B/C e tabelas de latência `ns-o-ran-gym` (`p50~36,2–36,4 ms`) |
| 4 | Plano estratificado CSV | [`A5_GATE1_SAMPLING_PLAN.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_SAMPLING_PLAN.csv) | 9.941 | `771b9a2c69e10bbbbdcd4c505684552c3523cb1edaa01c30c308b145f8b4650c` | OK | Chave exata (`state_id`, `b0_traced_dir`, `step_index`, `governed_window_index`) |
| 4b | Manifesto de amostragem | [`A5_GATE1_SAMPLING_MANIFEST.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_SAMPLING_MANIFEST.json) | 11.157 | `a468aa360543d44976fd06c5bda1152d927f0efc2bcb6e1a43caeaac0e85b585` | OK | Estratos pré-registrados e `clean_cut_steps_by_run` |
| 4c | Integridade dos replays | [`A5_GATE1_REPLAY_INTEGRITY.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_REPLAY_INTEGRITY.json) | 84.984 | `79ab53f14dc9affbfccafd409a54ae45dfdfaf73e200a0b5ba93edd8765bf9d9` | OK | `127/127` válidos, prefixos `KPM`/`PRB`/`delay` exatos |
| 4d | Relatório paridade B0 trace | [`A5_GATE1_B0_PARITY_REPORT.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_B0_PARITY_REPORT.json) | 14.873 | `abbc476099fd599ab5260f79247dc4e4ab54714c61b2fdb1f5b0cc690b4f1168` | OK | `all_ok=true`, paridade entre B0 sem trace e B0 com `--causal_trace_path` |
| 5 | Manifesto B0-rest congelado | [`B0_FROZEN_MANIFEST.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/B0_FROZEN_MANIFEST.json) | 13.092 | `aeff9bc27b5fe1d1532ea419b70cbcd296696c39c9a05d4a578055b3d9181b34` | OK | Commit base `060d0581f950d041fb5c4f6482259528089cf019`, `17/17` hashes ok |
| 5b | Documento prontidão B0 | [`B0_PRONTIDAO_E_CONGELAMENTO.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/B0_PRONTIDAO_E_CONGELAMENTO.md) | 11.641 | `fcb9b19cf0c6102767abc7f9903a47fff489d27637434cfd0a0fd94444236c55` | OK | Variabilidade `2102` vs `2103` no B0 congelado |
| 6 | Traces causais + código A5 | [`a5_joint_selector.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_jc/a5_joint_selector.py) e [`A5_ESPECIFICACAO_CONCEITUAL.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/A5_ESPECIFICACAO_CONCEITUAL.md) | 32.126 / 16.728 | `af93a9e4d3f5e8b6714de94099f132b7fd6f00458c3de9e935f1cfa698f0eb27` / `01f9548ccbd06cdb469c469a95d3ceac8502616b4f22229b0b076291b8cfb08b` | OK | Definições de `causal_active_slices`, `equal_active_split`, `prop_active_split` e logs brutos por run (`causal_trace.jsonl.gz`, `decisions.jsonl`, `urllc_delayed_updates.jsonl`, `step_metrics.csv`) |
| 7 | `D2-v2` relatório | [`D2v2_report.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/D2v2_report.md) | 10.720 | `ee168224867c390f0542eadd101b0975263abf0bd6e40f90f5759be45ef94534` | OK | Contexto histórico (`A0` = `D1-v9`, `30` estados × `13` ramos = `390` replays) |
| 7b | `D2-v2` estados CSV | [`D2v2_states.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/D2v2_states.csv) | 2.407 | `a6a02151dda110139f58938eb62665a44d517c64a3f2b566bfb37f5a8186f8f3` | OK | `30` estados (`seeds 2101, 2102, 2103`, $\lambda \in \{2,5,6\}$) |
| 7c | `D2-v2` resultados CSV | [`D2v2_results.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/D2v2_results.csv) | 55.767 | `07110c6efcb90f4b5c2aea9f27edeb13e8d1567853320c11476bfbdfec7b202a` | OK | `390` linhas; `alt0=a_B0`, `alt1..11` heurísticas, `alt12=ref:equal` |
| 7d | Retificação D3-v1 | [`D3_RETIFICACAO_E_PLANO_ABLACAO.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/D3_RETIFICACAO_E_PLANO_ABLACAO.md) | 36.311 | `b633266341da6afc4eb8037aac5b379d9c03ea59ab16cf316fc413716c354cd8` | OK | Razões da anulação do `STOP-JOINT` antigo (`D3-v1`) |
| 7e | Relatório D1-v9 | [`D1v9_report.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/D1v9_report.md) | 10.287 | `e8da541d40566edadd34c698347a632c791e807cc64c0ae6a119a7b461024af9` | OK | Auditoria URLLC delayed updates (`11.249/11.335` fechados; `0,8%` censura) |
| 7f | Emenda 4 | [`PROTOCOL_FROZEN_AMENDMENT_4_DELAYED_FEEDBACK.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/PROTOCOL_FROZEN_AMENDMENT_4_DELAYED_FEEDBACK.md) | 5.536 | `a122e998a3b16d17df92abee576f324cc4cc974e045060c0d0713350b8aaf21e` | OK | Estágio 2 URLLC (`MAX_AGE=30`, viés de seleção `69 vs 19.235 B`) |
| 8 | Direcionamento e revisão 16/11 | `/home/eliothluy/Downloads/Telegram Desktop/SQR_DIRECIONAMENTO_CIENTIFICO_E_REVISAO_LITERATURA_2026-11-16.md` | — | — | **AUSENTE** | Não encontrado em Downloads/Documentos; cobertura parcial pela §4 do Ofício |

**Artefatos derivados desta auditoria (somente leitura sobre bases congeladas):**
- Script reprodutível: [`gate_contextual_analyze.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/gate_contextual_analyze.py) (`31.481` B, `sha256: e6d10fe2370f7159ae28df800c387cacd053cb4eebca1c5deb39cc49089a1870`)
- Tabela única de estados: [`GATE_CONTEXTUAL_ESTADOS.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/GATE_CONTEXTUAL_ESTADOS.csv) (`21.931` B, `sha256: f8b3ffed9adc30f2f2f2636b87b5cb270ee803dda5b8508904ea696f286066e3`)
- Saída JSON estruturada: [`GATE_CONTEXTUAL_ANALYSIS.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/GATE_CONTEXTUAL_ANALYSIS.json) (`87.902` B, `sha256: 8b9ca0ffc606ce6f96fe53a5d409d3eefa0836956c28456ed9c8158b17277bc3`)

### 2.1 Verificação de Integridade do Manifesto `B0-rest`
Conferência direta de [`B0_FROZEN_MANIFEST.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/B0_FROZEN_MANIFEST.json) (`config_sha256: a2b40861ace7ba87ce2d321ccc62f42f22e6b3798c4432171875ae55d7040d7c`): todos os **17 arquivos** listados em `git_provenance.code_sha256` (incluindo [`sqr-sim.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr/sqr-sim.cc), [`sqr-mac-scheduler.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr/sqr-mac-scheduler.cc), [`het_runtime.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_jc/het_runtime.py), [`sqr_het.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_jc/sqr_het.py), [`feature_scaler.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_jc/feature_scaler.py), [`sla_manifest.yaml`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_jc/sla_manifest.yaml) e [`freeze_contract.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_jc/freeze_contract.py)) permanecem **100% íntegros (`0 mismatches`)**.

---

## 3. Reconciliação das Populações e Regra de Junção

### 3.1 População do Gate 1 (`25` estados, `127` replays únicos)
Os `25` estados foram selecionados antes da execução de qualquer seletor A5 a partir de `5` runs `B0-rest` com `--causal_trace_path`:

| `b0_traced_tag` | $\lambda$ | Seed | `ALL3_COMPLETE` | `LT3_DEMAND_COMPLETE` | `UNKNOWN_INTEGRITY` | Total |
|---|---:|---:|---:|---:|---:|---:|
| `het_lam2_s2102` | 2 | 2102 | 0 | 3 | 1 | 4 |
| `het_lam5_s2102` | 5 | 2102 | 5 | 0 | 0 | 5 |
| `het_lam5_s2103` | 5 | 2103 | 4 | 1 | 1 | 6 |
| `het_lam6_s2102` | 6 | 2102 | 4 | 1 | 0 | 5 |
| `het_lam6_s2103` | 6 | 2103 | 5 | 0 | 0 | 5 |
| **Total** | — | — | **18** | **5** | **2** | **25** |

* **Presença de braços e paridade A/A:** `B0-rest` (`b0`), `equal` (`eq`) e `prop` (`prop`) estão presentes e avaliados em **`25/25` estados (`100%`)**, com `aa_prefix_valid = True` nos três braços em **`25/25` estados (`0` casos sem paridade)**. Todos os `127/127` replays únicos em `A5_GATE1_REPLAY_INTEGRITY.json` têm `valid = True`, `ns3_rc0 = True`, `total_rbg_all_53 = True` (`C = 53`), `kpm_missing_total = 0` e prefixos `kpm`/`prb`/`delay` exatos.
* **Reconciliação dos 3 universos:**
  1. **`ALL3_COMPLETE` (`N = 18`, 72%):** todas as 3 slices têm evidência `COMPLETE` em `B0-rest`, `eq` e `prop` (`STRICT_ALL3_R19` 100% avaliável, `0` `UNKNOWN`). Atenção à **estrutura estatística**: apenas `4` runs ($\lambda \in \{5,6\} \times \text{seed} \in \{2102, 2103\}$), sendo zero estados de $\lambda = 2$.
  2. **`LT3_DEMAND_COMPLETE` (`N = 5`, 20%):** `lam2_s2102_w2685`, `lam2_s2102_w2442`, `lam2_s2102_w2731` (em $\lambda=2$), `lam5_s2103_w1241` ($\lambda=5$) e `lam6_s2102_w2071` ($\lambda=6$). Por haver apenas 1 ou 2 slices com demanda causal no instante da decisão (ou na janela), `STRICT_ALL3_R19 = UNKNOWN` em todos os `5` casos (`STRICT_DEMAND_R19` é avaliável em `5/5`).
  3. **`UNKNOWN_INTEGRITY` (`N = 2`, 8%):** `lam2_s2102_w2448` (`tx_u = 86, rx_u = 81`) e `lam5_s2103_w1351` (`tx_u = 178, rx_u = 176`), nos quais o pool URLLC permaneceu `urllc_pool_tag = "OPEN"` ao término do horizonte `+3 s` de replay (`STRICT_ALL3_R19 = UNKNOWN` e `STRICT_DEMAND_R19 = UNKNOWN`).

### 3.2 Diferença Estrutural entre Gate 1 (`N = 25`) e `D2-v2` (`N = 30`)
Para evitar comparação indevida entre amostras heterogêneas:
* **`D2-v2`** testou `30` estados (seeds `2101, 2102, 2103`; $\lambda \in \{2,5,6\}$) sobre o baseline anterior `A0` (`D1-v9`), com `a_B0` + `11` heurísticas (`paired`, `nbh`, `rnd`) + `alt12 = ref:equal (18,17,18)` (`390` replays; `29/30` paridade `A/A` inicial + `S1_lam5_s2101_t2064` validado integralmente com `simTime=300` bit-exact). **`D2-v2` não avaliou `prop_active`**, e apenas **`12/30` estados** tiveram `COMPLETE|COMPLETE|COMPLETE` simultâneo entre `alt0` (`a_B0`) e `alt12` (`equal`).
* **Gate 1 (`N = 25`)** partiu do **`B0-rest` congelado** (`A1+A2a+A_scaler`), com `C = 53`, seeds `2102` e `2103` (deixando `2101` fora do Gate 1), incluindo `B0`, `eq = (18,17,18)` e `prop_active` (`prop`), atingindo **`18/25` estados em `ALL3_COMPLETE`**.
* **Regra de execução de `equal` no Gate 1 (distinção causal crítica):** ao inspecionar [`A5_GATE1_METRICS.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_METRICS.csv), verifica-se que o braço `arm="eq"` executou **`(18, 17, 18)` estático sem máscara em todos os `25/25` estados**, alocando PRBs a slices sem demanda pré-decisão em `9/25` estados (`inactive_prbs_allocated = 17` em `w2552`, `18` em `w1254` e `17–36` em `7/7` de `Universo B/C`). Trata-se de uma política **temporalmente causal** (reserva fixa que não usa nenhuma informação futura), distinta tanto de uma alocação causalmente válida com zeragem por demanda observada (`active_mask`, adotada por `prop` e `equal_active`) quanto de qualquer política hipotética com vazamento futuro. Em contrapartida, `prop` sempre respeitou `active_mask` (`0` PRBs alocados a slices ociosas).

### 3.3 Regra de Junção da Tabela Única (`GATE_CONTEXTUAL_ESTADOS.csv`)
Cada linha de [`GATE_CONTEXTUAL_ESTADOS.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/GATE_CONTEXTUAL_ESTADOS.csv) é identificada por `state_id` (`25/25` únicos) e une:
1. [`A5_GATE1_SAMPLING_PLAN.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_SAMPLING_PLAN.csv) (`run_tag = b0_traced_tag`, `lam`, `seed`, `step_index`, `state_window_index`, `governed_window_index`, `sim_time_ms`), validando `state_window_index == governed_window_index - 1` e `sim_time_ms == governed_window_index * 100`.
2. Logs causais do respectivo diretório `b0_traced_dir`: `causal_trace.jsonl.gz` (por `governed_window_index`), `decisions.jsonl` (por `state_window_index` para entradas pré-ação e `outcome_window_index <= state_window_index` para histórico), `urllc_delayed_updates.jsonl` (por `observed_at_window <= state_window_index`) e `step_metrics.csv` (por `step == step_index - 1`).
3. [`A5_GATE1_METRICS.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_METRICS.csv) indexada por `(state_id, arm)` com `arm in {"b0", "eq", "prop"}` exclusivamente para o **Bloco B** (desfechos offline pós-replay).

---

## 4. Auditoria Temporal e Armadilhas de Vazamento Evitadas

No pipeline [`sqr_run.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/examples/sqr_run.py) + [`het_runtime.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_jc/het_runtime.py), a decisão que rege a janela `governed_window_index = gw` ocorre no instante `decision_at_ms = gw * 100 ms`, imediatamente ao término da janela anterior `state_window_index = sw = gw - 1`.

| Fonte bruta | Campos admitidos no **Bloco A** (causais) | Fronteira temporal imposta | Campos **EXCLUÍDOS** por vazamento temporal |
|---|---|---|---|
| `causal_trace.jsonl.gz` | `state` (`n_s, queue_s, offered_s, snr_s, prev_a_s, kpi_*_raw, viol_*, extras`), `per_slice_model` (`f_candidates`, `sigma_candidates`, `tau`), `modes_per_slice`, `a_raw_per_slice`, `a_prop_per_slice`, `a_exec_per_slice` | Registro da iteração `gw` gerado antes de enviar `a_exec` ao `ns-3` (`<= gw * 100 ms`) | Nenhum neste bloco (todos os campos são pré-ação); `action_candidates` pós-decisão não existem fora do `per_slice_model` |
| `decisions.jsonl` | Linha `state_window_index == sw`: `projection_triggered`, `projection_delta_l1`, `sum_proposals_over_C`; linhas anteriores: apenas com `outcome_window_index <= sw` | `outcome_window_index <= sw` (`outcome_kpm_stamp_ms <= gw * 100 ms`) | **Na linha `state_window_index == sw`:** `sla_evidence`, `sla_outcome_margins`, `sla_ok`, `joint_frozen_ok` e `outcome_window_index == gw` — pertencem a **`(gw + 1) * 100 ms`** (pós-ação!) |
| `urllc_delayed_updates.jsonl` | Entradas com `updated == True` e `p95_ms` presente | `observed_at_window <= sw` (Emenda 4: atraso típico `2` janelas; `MAX_AGE = 30`) | Entradas com `observed_at_window > sw` (fechamento só após `decision_at_ms`) |
| `step_metrics.csv` | `prb_scheduled`, `prb_idle_est` da linha `step == step_index - 1` | `timestamp_ms <= decision_at_ms` | Linha **`step == step_index`** (`timestamp_ms == (gw + 1) * 100 ms` — **pós-ação!**) |
| `A5_GATE1_METRICS.csv` | **Zero** (não entra no Bloco A) | Pós-replay (`+3 s`) | Todas as colunas ficam restritas ao **Bloco B** (`off_*`) |

**Resultado da auditoria causal automática em [`gate_contextual_analyze.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/gate_contextual_analyze.py):** **`0` violações temporais** (`temporal_violations = []`).

**Lacuna física de observabilidade confirmada (sem imputação):**
* Em **`18/18` estados** de `ALL3_COMPLETE` (e **`25/25`** no total), `prev_kpi_u_p95_imm = NaN`: o `p95` da janela imediatamente anterior `sw` nunca está fechado no instante de decisão de `gw` (`urllc_pool_open_at_dec = True` em `17/18` de `ALL3_COMPLETE`; `False` apenas em `w2552` com `n_u = 0`). O último dado URLLC fechado (`last_closed_urllc_p95_ms`) chega com defasagem mínima de **`2` janelas (`200 ms`)** em `17/18` estados (e `631` janelas em `w2552`).
* Conforme [`PROTOCOL_FROZEN_AMENDMENT_4_DELAYED_FEEDBACK.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/PROTOCOL_FROZEN_AMENDMENT_4_DELAYED_FEEDBACK.md), janelas URLLC com perda pesada de pacotes expiram sem fechar o pool e não geram atualização (`buffer 69 vs 19.235 B`, `resourceShare 24,5% vs 0%`, `delay médio 8,1 vs 236 ms`), criando censura de eventos extremos no sinal histórico online.

---

## 5. Lacunas Remanescentes e Conclusão G0

1. **Ausência documental:** `SQR_DIRECIONAMENTO_CIENTIFICO_E_REVISAO_LITERATURA_2026-11-16.md` não está no workspace/Downloads.
2. **Amostra estatística restrita:** apenas `2` seeds (`2102`, `2103`), nenhuma observação `ALL3_COMPLETE` em $\lambda = 2$, e somente `4` runs independentes.
3. **Diferença `equal` cego vs `equal` ativo:** no Gate 1, `equal` alocou `18-17-18` sem mascarar demanda inativa (`inactive_prbs_allocated > 0` em `2/18` de `ALL3_COMPLETE` e `7/7` dos demais universos), exigindo separação explícita em G1 (§3.2/§5.1).

**Decisão G0:** **`GO-DADOS`**. A tabela [`GATE_CONTEXTUAL_ESTADOS.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/GATE_CONTEXTUAL_ESTADOS.csv) é íntegra, auditável e livre de vazamento temporal, habilitando a análise discriminativa **estritamente exploratória** do Gate G1 ([`GATE_CONTEXTUAL_02_DISCRIMINACAO.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/GATE_CONTEXTUAL_02_DISCRIMINACAO.md)).
